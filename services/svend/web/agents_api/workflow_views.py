"""Workflow API views."""

import json
import uuid
from datetime import datetime

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated, require_auth
from .models import Workflow


@csrf_exempt
@require_http_methods(["GET", "POST"])
@require_auth
def workflows_list(request):
    """List or create workflows."""
    if request.method == "GET":
        workflows = Workflow.objects.filter(user=request.user).order_by("-created_at")
        return JsonResponse({
            "workflows": [
                {
                    "id": str(wf.id),
                    "name": wf.name,
                    "steps": wf.steps,
                    "created_at": wf.created_at.isoformat(),
                    "last_run": wf.last_run.isoformat() if wf.last_run else None,
                }
                for wf in workflows
            ]
        })

    # POST - create new workflow
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    name = data.get("name", "").strip()
    steps = data.get("steps", [])

    if not name:
        return JsonResponse({"error": "Name is required"}, status=400)

    workflow = Workflow.objects.create(
        user=request.user,
        name=name,
        steps=json.dumps(steps) if isinstance(steps, list) else steps,
    )

    return JsonResponse({"id": str(workflow.id), "success": True})


@csrf_exempt
@require_http_methods(["GET", "PUT", "DELETE"])
@require_auth
def workflow_detail(request, workflow_id):
    """Get, update, or delete a workflow."""
    try:
        workflow = Workflow.objects.get(id=workflow_id, user=request.user)
    except Workflow.DoesNotExist:
        return JsonResponse({"error": "Workflow not found"}, status=404)

    if request.method == "GET":
        return JsonResponse({
            "id": str(workflow.id),
            "name": workflow.name,
            "steps": workflow.steps,
            "created_at": workflow.created_at.isoformat(),
            "last_run": workflow.last_run.isoformat() if workflow.last_run else None,
        })

    if request.method == "DELETE":
        workflow.delete()
        return JsonResponse({"success": True})

    # PUT - update
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if "name" in data:
        workflow.name = data["name"]
    if "steps" in data:
        workflow.steps = json.dumps(data["steps"]) if isinstance(data["steps"], list) else data["steps"]

    workflow.save()
    return JsonResponse({"success": True})


@csrf_exempt
@require_http_methods(["POST"])
@gated
def workflow_run(request, workflow_id):
    """Run a workflow."""
    try:
        workflow = Workflow.objects.get(id=workflow_id, user=request.user)
    except Workflow.DoesNotExist:
        return JsonResponse({"error": "Workflow not found"}, status=404)

    # Parse steps
    steps = json.loads(workflow.steps) if isinstance(workflow.steps, str) else workflow.steps

    # Update last run
    workflow.last_run = datetime.now()
    workflow.save()

    # Execute workflow steps
    results = []
    context = {}  # Pass data between steps

    for i, step in enumerate(steps):
        step_type = step.get("type")
        step_name = step.get("name", f"Step {i + 1}")

        try:
            if step_type == "decision_guide":
                result = _run_decision_guide_step(step, context)
            elif step_type == "researcher":
                result = _run_researcher_step(step, context)
            elif step_type == "scrub":
                result = _run_scrub_step(step, context)
            elif step_type == "analyst":
                result = _run_analyst_step(step, context)
            elif step_type == "coder":
                result = _run_coder_step(step, context)
            elif step_type == "writer":
                result = _run_writer_step(step, context)
            elif step_type == "editor":
                result = _run_editor_step(step, context)
            elif step_type == "eda":
                result = _run_eda_step(step, context)
            else:
                result = {"status": "skipped", "reason": f"Unknown step type: {step_type}"}

            results.append({
                "step": step_name,
                "type": step_type,
                "status": "completed",
                "result": result,
            })

            # Store result for next step
            context[step_name] = result

        except Exception as e:
            results.append({
                "step": step_name,
                "type": step_type,
                "status": "error",
                "error": str(e),
            })

    return JsonResponse({
        "success": True,
        "result": {
            "workflow": workflow.name,
            "steps_executed": len(results),
            "results": results,
        }
    })


def _run_researcher_step(step, context):
    """Run researcher agent step."""
    from .views import get_shared_llm

    query = step.get("query", "")
    if not query:
        return {"status": "skipped", "reason": "No query provided"}

    try:
        from researcher.agent import ResearchAgent, ResearchQuery

        llm = get_shared_llm()
        agent = ResearchAgent(llm=llm)
        result = agent.run(ResearchQuery(question=query))

        return {
            "query": query,  # Include original query for downstream steps
            "summary": result.summary if hasattr(result, "summary") else str(result),
            "sources": [{"title": s.title, "url": s.url} for s in result.sources] if hasattr(result, "sources") else [],
        }
    except Exception as e:
        return {"error": str(e)}


def _run_coder_step(step, context):
    """Run coder agent step."""
    from .views import get_coder_llm

    prompt = step.get("prompt", "")
    if not prompt:
        return {"status": "skipped", "reason": "No prompt provided"}

    try:
        from coder.agent import CodingAgent, CodingTask

        llm = get_coder_llm()
        agent = CodingAgent(llm=llm)
        result = agent.run(CodingTask(description=prompt))

        return {"code": result.code}
    except Exception as e:
        return {"error": str(e)}


def _run_writer_step(step, context):
    """Run writer agent step."""
    from .views import get_shared_llm

    template = step.get("template", "general")
    topic = step.get("topic", "") or step.get("name", "")

    # Get research context from previous steps
    research_content = ""
    research_topic = ""
    for key, value in context.items():
        if isinstance(value, dict):
            # Get the original research query as the topic
            if "query" in value and not topic:
                research_topic = value["query"]
            if "summary" in value:
                research_content += f"\n\n## Research Summary:\n{value['summary']}"
            if "sources" in value and value["sources"]:
                research_content += "\n\n## Key Sources:\n"
                for src in value["sources"][:5]:
                    if isinstance(src, dict):
                        research_content += f"- {src.get('title', 'Unknown')}\n"

    # Use research topic if no explicit topic provided
    if not topic and research_topic:
        topic = research_topic

    if not topic:
        topic = "Document"

    # Build topic with context - be explicit about using research
    if research_content:
        full_topic = f"""Write a comprehensive document about: {topic}

IMPORTANT: Base your writing on this research. Do NOT write generic content.
{research_content}

Write about the specific findings from this research, not about writing in general."""
    else:
        full_topic = topic

    try:
        from writer.agent import WriterAgent, DocumentRequest, DocumentType

        llm = get_shared_llm()
        agent = WriterAgent(llm=llm)
        doc_type = getattr(DocumentType, template.upper(), DocumentType.EXECUTIVE_SUMMARY)
        result = agent.write(
            DocumentRequest(topic=full_topic, doc_type=doc_type),
            original_prompt=full_topic,
        )

        return {"content": result.content if hasattr(result, "content") else str(result)}
    except Exception as e:
        return {"error": str(e)}


def _run_editor_step(step, context):
    """Run editor agent step."""
    # Get document from previous step if available
    document = ""
    for key, value in context.items():
        if isinstance(value, dict) and "content" in value:
            document = value["content"]
            break

    if not document:
        return {"status": "skipped", "reason": "No document to edit"}

    try:
        from reviewer.editor import Editor

        editor = Editor()
        result = editor.edit(document=document)

        return {
            "original_grade": result.original_grade,
            "improved_grade": result.improved_grade,
            "edits_made": result.edits_made,
            "cleaned_document": result.cleaned_document,
        }
    except Exception as e:
        return {"error": str(e)}


def _run_eda_step(step, context):
    """Run EDA step (requires file upload, returns mock for workflow)."""
    return {
        "status": "skipped",
        "reason": "EDA requires file upload - use the EDA agent directly",
    }


def _run_decision_guide_step(step, context):
    """Run decision guide step - helps frame decisions and detect biases."""
    situation = step.get("situation", "")
    if not situation:
        return {"status": "skipped", "reason": "No situation provided"}

    try:
        from guide.decision import DecisionGuide, quick_bias_check

        # Quick bias check on the situation
        bias_check = quick_bias_check(situation)

        # Create a brief for downstream agents
        guide = DecisionGuide()
        guide.state.answers = {
            "situation": situation,
            "decision": step.get("decision", situation),
            "decision_type": "Exploring/understanding a topic",
            "confidence": "5",
            "have_data": step.get("have_data", "Unknown"),
            "data_quality": step.get("data_quality", "Unknown"),
        }

        # Add detected biases
        from guide.decision import detect_biases
        guide.bias_warnings = detect_biases(situation)

        brief = guide.get_brief()

        return {
            "situation": situation,
            "bias_warnings": bias_check["warnings"],
            "recommended_agent": brief.recommended_agent,
            "recommended_action": brief.recommended_action,
            "reasoning": brief.routing_reasoning,
            "brief": brief.to_dict(),
            "prompt_context": brief.to_prompt_context(),
        }
    except Exception as e:
        return {"error": str(e)}


def _run_scrub_step(step, context):
    """Run Scrub step - clean and validate data."""
    # Get data from previous step
    data = None
    for key, value in context.items():
        if isinstance(value, dict) and "data" in value:
            data = value["data"]
            break

    if data is None:
        return {
            "status": "skipped",
            "reason": "No data from previous step. Scrub requires data input (from DSW, file upload, etc.)",
        }

    try:
        from dsw.interfaces import ScrubAdapter, ScrubRequest
        import pandas as pd

        # Convert to DataFrame if needed
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data

        adapter = ScrubAdapter()
        result = adapter.run(ScrubRequest(data=df))

        return {
            "original_shape": result.original_shape,
            "cleaned_shape": result.cleaned_shape,
            "outliers_flagged": result.outliers_flagged,
            "missing_filled": result.missing_filled,
            "warnings": result.warnings,
            "data": result.data.to_dict(orient="records") if hasattr(result.data, "to_dict") else result.data,
        }
    except Exception as e:
        return {"error": str(e)}


def _run_analyst_step(step, context):
    """Run Analyst step - train ML model."""
    target = step.get("target", "")
    if not target:
        return {"status": "skipped", "reason": "No target column specified"}

    # Get data from previous step
    data = None
    for key, value in context.items():
        if isinstance(value, dict) and "data" in value:
            data = value["data"]
            break

    if data is None:
        return {
            "status": "skipped",
            "reason": "No data from previous step. Analyst requires data input.",
        }

    try:
        from dsw.interfaces import AnalystAdapter, AnalystRequest
        import pandas as pd

        # Convert to DataFrame if needed
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data

        adapter = AnalystAdapter()
        result = adapter.run(AnalystRequest(data=df, target=target))

        return {
            "model_type": result.model_type,
            "task_type": result.task_type,
            "metrics": result.metrics,
            "feature_importance": result.feature_importance[:10] if result.feature_importance else [],
            "report": result.report_markdown,
        }
    except Exception as e:
        return {"error": str(e)}
