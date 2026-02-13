"""Agent API views.

Exposes the multi-agent workbench functionality via REST API.
"""

import os
import sys
import logging

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

logger = logging.getLogger(__name__)


# =============================================================================
# Problem Integration Helpers
# =============================================================================

def add_finding_to_problem(user, problem_id: str, summary: str,
                           evidence_type: str = "research", source: str = "Agent",
                           supports: list = None, weakens: list = None) -> dict | None:
    """
    Add a finding from an agent to a problem as evidence.

    Returns the evidence dict if successful, None otherwise.
    """
    if not problem_id:
        return None

    try:
        from .models import Problem
        from .problem_views import write_context_file

        problem = Problem.objects.get(id=problem_id, user=user)

        evidence = problem.add_evidence(
            summary=summary,
            evidence_type=evidence_type,
            source=source,
            supports=supports or [],
            weakens=weakens or [],
        )

        # Update probable causes
        problem.update_understanding()
        write_context_file(problem)

        logger.info(f"Added evidence to problem {problem_id}: {summary[:50]}...")
        return evidence

    except Exception as e:
        logger.warning(f"Could not add finding to problem {problem_id}: {e}")
        return None


def get_problem_context_for_agent(user, problem_id: str) -> str:
    """
    Get problem context formatted for an agent prompt.

    Returns empty string if problem not found.
    """
    if not problem_id:
        return ""

    try:
        from .models import Problem

        problem = Problem.objects.get(id=problem_id, user=user)

        context_parts = [
            f"## Problem Context: {problem.title}",
            "",
            f"**Effect being investigated:** {problem.effect_description}",
        ]

        if problem.effect_magnitude:
            context_parts.append(f"**Magnitude:** {problem.effect_magnitude}")

        if problem.domain:
            context_parts.append(f"**Domain:** {problem.domain}")

        hypotheses = problem.get_hypotheses()
        if hypotheses:
            context_parts.append("")
            context_parts.append("**Current hypotheses:**")
            for h in hypotheses[:5]:  # Top 5
                context_parts.append(f"- {h.get('cause', '')} ({h.get('probability', 0.5)*100:.0f}% likely)")

        if problem.key_uncertainties:
            context_parts.append("")
            context_parts.append("**Key uncertainties:**")
            for u in problem.key_uncertainties[:3]:
                context_parts.append(f"- {u}")

        context_parts.append("")
        context_parts.append("---")
        context_parts.append("Please focus your analysis on this problem context.")
        context_parts.append("")

        return "\n".join(context_parts)

    except Exception as e:
        logger.warning(f"Could not get problem context for {problem_id}: {e}")
        return ""


# Agent paths are configured centrally in settings.py:
# - KJERNE_PATH (root core/) for shared agent libs
# - KJERNE_PATH/services/svend/agents/agents/ for agent modules
# - KJERNE_PATH/services/ for scrub, forge, etc.


# Use centralized LLM management
from .llm_manager import get_shared_llm, get_coder_llm


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def researcher_agent(request):
    """Run the research agent.

    Optionally accepts problem_id to add findings as evidence.
    """
    query = request.data.get("query", "")
    focus = request.data.get("focus", "general")
    depth = request.data.get("depth", "standard")
    problem_id = request.data.get("problem_id")

    if not query:
        return Response({"error": "Query is required"}, status=status.HTTP_400_BAD_REQUEST)

    # Add problem context to query if available
    problem_context = get_problem_context_for_agent(request.user, problem_id)
    if problem_context:
        query = f"{problem_context}\n\n**Research Question:** {query}"

    try:
        from researcher.agent import ResearchAgent, ResearchQuery

        llm = get_shared_llm()
        agent = ResearchAgent(llm=llm)
        result = agent.run(ResearchQuery(question=query, focus=focus, depth=depth))

        summary = result.summary if hasattr(result, "summary") else str(result)

        response = {
            "summary": summary,
            "sources": [
                {"title": s.title, "url": s.url}
                for s in result.sources
            ] if hasattr(result, "sources") else [],
            "markdown": result.to_markdown() if hasattr(result, "to_markdown") else None,
        }

        if llm is None:
            response["note"] = "Running without LLM - synthesis may be limited."

        # Add finding to problem if problem_id provided
        if problem_id and summary:
            evidence = add_finding_to_problem(
                user=request.user,
                problem_id=problem_id,
                summary=f"Research finding: {summary[:500]}",
                evidence_type="research",
                source="Researcher Agent",
            )
            if evidence:
                response["evidence_added"] = True
                response["evidence_id"] = evidence.get("id")

        # Track usage
        request.user.increment_queries()

        return Response(response)
    except ImportError as e:
        logger.error(f"Import error in researcher agent: {e}")
        return Response({
            "error": "Research agent not available",
            "summary": f"Mock research result for: {query}",
            "sources": [],
        })
    except Exception as e:
        logger.exception("Researcher agent error")
        return Response({"error": str(e)[:200]}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def coder_agent(request):
    """Run the coder agent."""
    prompt = request.data.get("prompt", "")
    language = request.data.get("language", "python")

    if not prompt:
        return Response({"error": "Prompt is required"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        from coder.agent import CodingAgent, CodingTask

        llm = get_coder_llm()
        agent = CodingAgent(llm=llm)
        result = agent.run(CodingTask(description=prompt, language=language))

        response = {
            "code": result.code,
            "qa_report": result.qa_report() if hasattr(result, "qa_report") else None,
        }

        if llm is None:
            response["note"] = "Running without LLM - using pattern matching fallback."

        # Track usage
        request.user.increment_queries()

        return Response(response)
    except ImportError as e:
        logger.error(f"Import error in coder agent: {e}")
        # Return a simple mock response
        mock_code = f"# Generated code for: {prompt}\n# Language: {language}\n\ndef main():\n    pass\n"
        request.user.increment_queries()
        return Response({
            "code": mock_code,
            "note": "Coder agent not available, returning mock code.",
        })
    except Exception as e:
        logger.exception("Coder agent error")
        return Response({"error": str(e)[:200]}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def writer_agent(request):
    """Run the writer agent with Editor integration."""
    topic = request.data.get("topic", "")
    template = request.data.get("template", "general")
    run_editor = request.data.get("run_editor", True)
    original_prompt = request.data.get("prompt", "")

    if not topic:
        return Response({"error": "Topic is required"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        from writer.agent import WriterAgent, DocumentRequest, DocumentType

        llm = get_shared_llm()
        agent = WriterAgent(llm=llm)
        doc_type = getattr(DocumentType, template.upper(), DocumentType.EXECUTIVE_SUMMARY)
        result = agent.write(
            DocumentRequest(topic=topic, doc_type=doc_type),
            run_editor=run_editor,
            original_prompt=original_prompt or f"Write a {template} about {topic}",
        )

        response = {
            "content": result.content if hasattr(result, "content") else str(result),
            "quality_report": result.quality_report() if hasattr(result, "quality_report") else None,
            "quality_passed": getattr(result, "quality_passed", True),
            "quality_issues": getattr(result, "quality_issues", []),
        }

        # Include editor results if available
        if hasattr(result, "editor_result") and result.editor_result:
            response["editor"] = {
                "original_grade": result.editor_result.original_grade,
                "improved_grade": result.editor_result.improved_grade,
                "citation_confidence": result.editor_result.citation_confidence,
                "prompt_alignment": result.editor_result.prompt_alignment,
                "edits_made": result.editor_result.edits_made,
                "repeated_stats": len([r for r in result.editor_result.repetitions if r.issue_type == "statistic"]),
                "gaps": len(result.editor_result.gaps),
                "citation_concerns": len(result.editor_result.citation_concerns),
                "editorial_report": result.editor_result.editorial_report,
            }
            response["cleaned_content"] = result.editor_result.cleaned_document

        if llm is None:
            response["note"] = "Running without LLM - output is template-based."

        # Track usage
        request.user.increment_queries()

        return Response(response)
    except ImportError as e:
        logger.error(f"Import error in writer agent: {e}")
        request.user.increment_queries()
        return Response({
            "content": f"# {topic}\n\nThis is a placeholder document about {topic}.",
            "note": "Writer agent not available.",
        })
    except Exception as e:
        logger.exception("Writer agent error")
        return Response({"error": str(e)[:200]}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def editor_agent(request):
    """Run the Editor agent on a document."""
    document = request.data.get("document", "")
    title = request.data.get("title", "Document")
    rubric_type = request.data.get("rubric_type", "auto")
    prompt = request.data.get("prompt", "")

    if not document:
        return Response({"error": "Document is required"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        from reviewer.editor import Editor

        editor = Editor()
        result = editor.edit(
            document=document,
            title=title,
            rubric_type=rubric_type,
            prompt=prompt,
        )

        response = {
            "original_grade": result.original_grade,
            "improved_grade": result.improved_grade,
            "citation_confidence": result.citation_confidence,
            "prompt_alignment": result.prompt_alignment,
            "edits_made": result.edits_made,
            "words_removed": result.words_removed,
            "cleaned_document": result.cleaned_document,
            "editorial_report": result.editorial_report,
            "issues": {
                "grammar_fixes": len(result.grammar_fixes),
                "repetitions": len(result.repetitions),
                "repeated_statistics": len([r for r in result.repetitions if r.issue_type == "statistic"]),
                "citation_concerns": len(result.citation_concerns),
                "gaps": len(result.gaps),
                "drift_issues": len(result.drift_issues),
            },
            "details": {
                "citation_concerns": [
                    {"text": c.citation_text[:100], "issues": c.issues, "confidence": c.confidence}
                    for c in result.citation_concerns[:5]
                ],
                "repeated_stats": [
                    {"text": r.text, "count": r.count, "suggestion": r.suggestion}
                    for r in result.repetitions if r.issue_type == "statistic"
                ][:5],
                "gaps": [
                    {"topic": g.topic, "issue": g.issue, "suggestion": g.suggestion}
                    for g in result.gaps
                ],
                "drift_issues": [
                    {"expected": d.expected, "severity": d.severity, "suggestion": d.suggestion}
                    for d in result.drift_issues
                ],
            }
        }

        # Track usage
        request.user.increment_queries()

        return Response(response)
    except ImportError as e:
        logger.error(f"Import error in editor agent: {e}")
        request.user.increment_queries()
        return Response({
            "original_grade": "B",
            "improved_grade": "B",
            "citation_confidence": 0.8,
            "prompt_alignment": 0.9,
            "edits_made": 0,
            "cleaned_document": document,
            "editorial_report": "Editor agent not available - returning document as-is.",
            "note": "Editor agent not available.",
        })
    except Exception as e:
        logger.exception("Editor agent error")
        return Response({"error": str(e)[:200]}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def experimenter_agent(request):
    """Run the experimenter agent."""
    goal = request.data.get("goal", "")
    exp_type = request.data.get("type", "power")

    if not goal:
        return Response({"error": "Goal is required"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        from experimenter import ExperimenterAgent, ExperimentRequest

        llm = get_shared_llm()
        agent = ExperimenterAgent(llm=llm, seed=42)

        # Build request based on experiment type
        if exp_type == "power":
            exp_req = ExperimentRequest(
                goal=goal,
                request_type="power",
                test_type="ttest_ind",
                effect_size=0.5,
            )
        elif exp_type == "factorial":
            exp_req = ExperimentRequest(
                goal=goal,
                request_type="design",
                design_type="full_factorial",
                factors=[
                    {"name": "Factor A", "levels": ["-", "+"]},
                    {"name": "Factor B", "levels": ["-", "+"]},
                    {"name": "Factor C", "levels": ["-", "+"]},
                ],
            )
        elif exp_type == "response_surface":
            exp_req = ExperimentRequest(
                goal=goal,
                request_type="design",
                design_type="central_composite",
                factors=[
                    {"name": "Factor A", "levels": [-1, 0, 1]},
                    {"name": "Factor B", "levels": [-1, 0, 1]},
                ],
            )
        else:
            exp_req = ExperimentRequest(
                goal=goal,
                request_type="power",
                test_type="ttest_ind",
                effect_size=0.5,
            )

        result = agent.design_experiment(exp_req)

        response = {
            "summary": result.to_markdown() if hasattr(result, "to_markdown") else str(result),
            "experiment_type": exp_type,
        }

        if hasattr(result, "power_result") and result.power_result:
            response["sample_size"] = result.power_result.sample_size
            response["power"] = result.power_result.power
            response["effect_size"] = result.power_result.effect_size

        if hasattr(result, "design") and result.design:
            response["design"] = result.design.to_dict() if hasattr(result.design, "to_dict") else str(result.design)

        if llm is None:
            response["note"] = "Running without LLM - using statistical defaults."

        # Track usage
        request.user.increment_queries()

        return Response(response)
    except ImportError as e:
        logger.error(f"Import error in experimenter agent: {e}")
        request.user.increment_queries()
        return Response({
            "summary": f"Mock experiment design for: {goal}",
            "experiment_type": exp_type,
            "sample_size": 100,
            "power": 0.8,
            "note": "Experimenter agent not available.",
        })
    except Exception as e:
        logger.exception("Experimenter agent error")
        return Response({"error": str(e)[:200]}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def eda_agent(request):
    """Run automated EDA on uploaded data.

    Optionally accepts problem_id to add findings as evidence.
    """
    import pandas as pd

    if "file" not in request.FILES:
        return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)

    file = request.FILES["file"]
    name = request.data.get("name", "dataset")
    generate_charts = request.data.get("charts", "true").lower() == "true"
    problem_id = request.data.get("problem_id")

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return Response({"error": f"Failed to read CSV: {e}"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        from analyst import quick_eda

        report = quick_eda(df, name=name, generate_charts=generate_charts)

        # Calculate data quality score
        missing_penalty = min(report.total_missing_pct, 0.3)
        duplicate_penalty = min(report.duplicate_pct, 0.1)
        data_quality_score = 1.0 - missing_penalty - duplicate_penalty

        response = {
            "name": report.dataset_name,
            "shape": [report.n_rows, report.n_cols],
            "columns": [c.name for c in report.columns],
            "data_quality_score": data_quality_score,
            "summary": report.summary(),
            "profiles": [
                {
                    "name": c.name,
                    "dtype": c.dtype,
                    "missing_count": c.missing,
                    "missing_pct": c.missing_pct,
                    "unique_count": c.unique,
                    "mean": c.mean,
                    "std": c.std,
                    "min": c.min,
                    "max": c.max,
                    "top_values": c.top_values[:5] if c.top_values else None,
                    "is_numeric": c.is_numeric,
                    "has_outliers": c.has_outliers,
                    "outlier_count": c.outlier_count,
                }
                for c in report.columns
            ],
            "missing_report": {
                "total_missing": report.total_missing,
                "total_missing_pct": report.total_missing_pct,
                "columns_with_missing": len([c for c in report.columns if c.missing > 0]),
                "by_column": {c.name: c.missing for c in report.columns if c.missing > 0},
            } if report.total_missing > 0 else None,
            "outliers": [
                {"column": c.name, "count": c.outlier_count, "pct": c.outlier_count / c.count if c.count else 0}
                for c in report.columns if c.has_outliers
            ],
            "correlations": [
                {"col1": c[0], "col2": c[1], "value": c[2], "type": "pearson"}
                for c in (report.high_correlations or [])[:10]
            ],
            "recommendations": [],
        }

        # Add recommendations
        if report.total_missing_pct > 0.05:
            response["recommendations"].append(f"Address missing values ({report.total_missing_pct:.1%} of data)")
        if report.duplicate_pct > 0.01:
            response["recommendations"].append(f"Review duplicate rows ({report.duplicate_pct:.1%})")
        outlier_cols = [c for c in report.columns if c.has_outliers]
        if outlier_cols:
            response["recommendations"].append(f"Review outliers in {len(outlier_cols)} column(s)")
        if report.high_correlations:
            response["recommendations"].append(f"Consider multicollinearity ({len(report.high_correlations)} high correlations)")

        if report.charts:
            response["charts"] = report.charts

        # Add EDA findings to problem if problem_id provided
        if problem_id:
            summary_parts = [f"EDA on {name}: {report.n_rows} rows, {report.n_cols} columns."]
            if response["recommendations"]:
                summary_parts.append("Key findings: " + "; ".join(response["recommendations"][:3]))
            evidence = add_finding_to_problem(
                user=request.user,
                problem_id=problem_id,
                summary=" ".join(summary_parts),
                evidence_type="data_analysis",
                source="EDA Agent",
            )
            if evidence:
                response["evidence_added"] = True
                response["evidence_id"] = evidence.get("id")

        # Track usage
        request.user.increment_queries()

        return Response(response)
    except ImportError as e:
        logger.error(f"Import error in EDA agent: {e}")
        request.user.increment_queries()
        # Return basic pandas info
        return Response({
            "name": name,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "data_quality_score": 1.0 - df.isnull().sum().sum() / df.size,
            "summary": f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.",
            "note": "Full EDA agent not available - basic info only.",
        })
    except Exception as e:
        logger.exception("EDA agent error")
        return Response({"error": str(e)[:200]}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
