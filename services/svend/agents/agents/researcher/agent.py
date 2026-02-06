"""
Research Agent

Thorough multi-source research with:
- Query decomposition into sub-questions
- Diverse source searching
- Synthesis and structured output
- Citation tracking
- Confidence assessment
"""

import json
import re
from dataclasses import dataclass, field
from typing import Literal

import sys
sys.path.insert(0, '/home/eric/Desktop/agents')

from core.intent import IntentTracker, Action, AlignmentStatus
from core.sources import SourceManager, Source, SourceType, ResearchFindings
from core.reasoning import CodeReasoner  # Reuse Bayesian reasoning
from core.search import MultiSearch, SearchResult as APISearchResult, DataRetriever
from researcher.validator import ResearchValidator, ValidationResult, quick_validate


@dataclass
class ResearchQuery:
    """A research query with parameters."""
    question: str
    depth: Literal["quick", "standard", "thorough"] = "standard"
    focus: Literal["scientific", "market", "general"] = "general"
    max_sources: int = 10


# Use SearchResult from core.search (imported as APISearchResult)
SearchResult = APISearchResult


class ResearchAgent:
    """
    Research agent for thorough, multi-source investigation.

    Flow:
    1. Decompose query into sub-questions
    2. Generate diverse search queries
    3. Search multiple source types
    4. Synthesize findings
    5. Generate structured documentation
    """

    SYSTEM_PROMPT = """You are a thorough research assistant. Your job is to find accurate, well-sourced information.

Rules:
1. Use diverse sources - academic, industry, news, official
2. Cross-reference claims across sources
3. Note confidence levels and conflicts
4. Cite all sources properly
5. Identify gaps in available information

Be thorough but objective. Prefer primary sources over secondary."""

    DECOMPOSITION_PROMPT = """Break down this research question into 3-5 specific sub-questions that would help answer it comprehensively.

Research question: {query}

Focus area: {focus}

Output a JSON array of sub-questions:
["sub-question 1", "sub-question 2", ...]"""

    SYNTHESIS_PROMPT = """Synthesize these research findings into a coherent summary.

Original question: {query}

Findings from sources:
{findings}

Evidence quality notes:
{evidence_quality}

Write a structured summary following scientific epistemology:
1. Key findings (2-3 sentences) - use hedged language ("suggests", "indicates", "may")
2. Supporting evidence with source citations [S1], [S2], etc. - note study types where relevant
3. Include current statistics from data sources (FRED, World Bank) when available - cite as [FRED], [World Bank], etc.
4. Conflicts or uncertainties - acknowledge when studies disagree
5. Evidence strength assessment - prioritize meta-analyses > RCTs > cohort studies > case reports
6. Confidence assessment (high/medium/low) based on evidence quality

Output as JSON:
{{
    "summary": "...",
    "key_findings": ["...", "..."],
    "evidence": [{{"claim": "...", "sources": ["S1", "S2"], "confidence": "high", "evidence_level": "RCT"}}],
    "current_data": [{{"indicator": "...", "value": "...", "source": "FRED"}}],
    "conflicts": ["..."],
    "gaps": ["..."],
    "limitations": ["..."],
    "overall_confidence": "medium"
}}"""

    def __init__(self, llm=None, searcher=None,
                 brave_api_key: str = None, use_real_search: bool = True,
                 validate: bool = True, strict_validation: bool = False):
        """
        Args:
            llm: Language model for synthesis
            searcher: Custom searcher function
            brave_api_key: API key for Brave search
            use_real_search: Whether to use real APIs (vs mock)
            validate: Whether to validate research output (default True)
            strict_validation: Use stricter validation thresholds
        """
        self.llm = llm
        self.source_manager = SourceManager()
        self.intent_tracker = IntentTracker(llm=llm)
        self.validate = validate
        self.validator = ResearchValidator(strict_mode=strict_validation) if validate else None

        # Set up search
        if searcher:
            self.searcher = searcher
        elif use_real_search:
            self.multi_search = MultiSearch(brave_api_key=brave_api_key)
            self.searcher = None  # Will use multi_search
        else:
            self.searcher = None
            self.multi_search = None

        # Set up data retrieval (FRED, World Bank, Wikipedia)
        self.data_retriever = DataRetriever() if use_real_search else None

        # Depth settings
        self.depth_settings = {
            "quick": {"sub_questions": 2, "sources_per_q": 3, "iterations": 1},
            "standard": {"sub_questions": 4, "sources_per_q": 5, "iterations": 2},
            "thorough": {"sub_questions": 6, "sources_per_q": 8, "iterations": 3},
        }

    def _llm_generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate text from LLM."""
        if self.llm is None:
            return ""
        if hasattr(self.llm, 'generate'):
            return self.llm.generate(prompt, max_tokens=max_tokens)
        elif hasattr(self.llm, 'complete'):
            return self.llm.complete(prompt, max_tokens=max_tokens)
        return ""

    def run(self, query: ResearchQuery) -> ResearchFindings:
        """Execute a research query."""
        # Set intent
        self.intent_tracker.set_intent(
            raw_input=query.question,
            parsed_goal=f"Research: {query.question}",
            constraints=[
                "Must cite all sources",
                "Must use diverse source types",
                "Must note confidence levels",
            ],
        )

        settings = self.depth_settings[query.depth]

        # 1. Decompose into sub-questions
        sub_questions = self._decompose_query(query, settings["sub_questions"])

        # 2. Generate search queries for each sub-question
        search_queries = self._generate_search_queries(query, sub_questions)

        # 3. Search and collect sources
        all_findings = []
        evidence_quality_notes = []  # Track evidence quality for synthesis

        for sq in search_queries:
            results = self._search(sq, settings["sources_per_q"], query.focus)
            for result in results:
                source = self.source_manager.add_source(
                    title=result.title,
                    url=result.url,
                    content=result.snippet,
                    source_type=result.source_type,
                    snippet=result.snippet,
                    author=result.author,
                    date=result.date,
                    publication=result.publication,
                )

                # Track evidence quality metadata
                study_type = getattr(result, 'study_type', None)
                peer_reviewed = getattr(result, 'peer_reviewed', True)
                evidence_strength = getattr(result, 'evidence_strength', 0.5)

                finding_entry = {
                    "query": sq,
                    "source": source,
                    "finding": result.snippet,
                    "study_type": study_type.value if study_type else "unknown",
                    "peer_reviewed": peer_reviewed,
                    "evidence_strength": evidence_strength,
                    "citations": result.citations,
                }
                all_findings.append(finding_entry)

                # Build evidence quality note
                if study_type and study_type.value != "unknown":
                    quality_note = f"[{source.id}] {study_type.value}"
                    if not peer_reviewed:
                        quality_note += " (preprint)"
                    if result.citations > 100:
                        quality_note += f" ({result.citations} citations)"
                    evidence_quality_notes.append(quality_note)

        # 4. Check source diversity
        diversity = self.source_manager.get_diversity_score()
        if diversity < 0.5:
            missing = self.source_manager.suggest_missing_types()
            # Try to fill gaps
            for source_type in missing[:2]:
                gap_results = self._search_specific_type(
                    query.question, source_type, 2
                )
                for result in gap_results:
                    source = self.source_manager.add_source(
                        title=result.title,
                        url=result.url,
                        content=result.snippet,
                        source_type=result.source_type,
                    )
                    all_findings.append({
                        "query": query.question,
                        "source": source,
                        "finding": result.snippet,
                    })

        # 4.5. Fetch real data (FRED, World Bank, Wikipedia) if relevant
        current_data_text = ""
        if self.data_retriever:
            data_points = self.data_retriever.get_data(query.question)
            if data_points:
                current_data_text = self.data_retriever.format_for_research(data_points)
                # Add data sources to evidence quality notes
                for dp in data_points:
                    evidence_quality_notes.append(f"[{dp.source}] {dp.indicator}: {dp.value} ({dp.date})")

        # 5. Synthesize findings with evidence quality context
        synthesis = self._synthesize(query, all_findings, evidence_quality_notes, current_data_text)

        # 6. Validate research output
        validation_result = None
        validation_warnings = []

        if self.validate and self.validator:
            validation_result = self.validator.validate(
                synthesis=synthesis,
                findings=all_findings,
                sources=list(self.source_manager.sources.values()),
            )

            # Add validation warnings
            validation_warnings = validation_result.warnings + [
                f"CRITICAL: {issue}" for issue in validation_result.critical_issues
            ]

            # Adjust confidence based on validation
            if not validation_result.is_valid:
                # Research quality is poor - note it prominently
                synthesis["summary"] = (
                    f"**Note: Research validation found quality issues "
                    f"({validation_result.overall_quality:.0%} quality score)**\n\n"
                    + synthesis.get("summary", "")
                )

        # 7. Build structured output
        sources_list = list(self.source_manager.sources.values())
        findings = ResearchFindings(
            query=query.question,
            summary=synthesis.get("summary", ""),
            sections=self._build_sections(synthesis, all_findings),
            sources=sources_list,
            confidence=self._calculate_confidence(synthesis, validation_result),
            gaps=synthesis.get("gaps", []) + (validation_result.failed_gates if validation_result else []),
        )

        # Add validation metadata if available
        if validation_result:
            findings.metadata = {
                "validation": {
                    "is_valid": validation_result.is_valid,
                    "quality_score": validation_result.overall_quality,
                    "claims_supported": validation_result.claims_supported,
                    "claims_unsupported": validation_result.claims_unsupported,
                    "consistency_score": validation_result.source_consistency.consistency_score,
                    "confidence_level": validation_result.confidence_justification.level,
                    "confidence_justification": validation_result.confidence_justification.recommendation,
                    "passed_gates": validation_result.passed_gates,
                    "failed_gates": validation_result.failed_gates,
                    "warnings": validation_warnings,
                }
            }

        # Record action
        alignment_score = 1.0
        if validation_result and not validation_result.is_valid:
            alignment_score = validation_result.overall_quality

        action = Action(
            id="research_complete",
            description=f"Completed research on: {query.question}",
            action_type="research",
            content=findings.summary,
            alignment_score=alignment_score,
            reasoning=f"Found {len(findings.sources)} sources across {len(set(s.source_type for s in sources_list))} types. "
                      f"Validation: {'PASSED' if (validation_result and validation_result.is_valid) else 'WARNING' if validation_result else 'SKIPPED'}",
        )
        self.intent_tracker.record_action(action)

        return findings

    def _decompose_query(self, query: ResearchQuery, num_questions: int) -> list[str]:
        """Break query into sub-questions."""
        if self.llm is None:
            return self._mock_decompose(query, num_questions)

        prompt = self.DECOMPOSITION_PROMPT.format(
            query=query.question,
            focus=query.focus,
        )

        response = self._llm_generate(prompt, max_tokens=500)

        # Parse JSON array
        try:
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                questions = json.loads(match.group())
                return questions[:num_questions]
        except (json.JSONDecodeError, AttributeError):
            pass

        return self._mock_decompose(query, num_questions)

    def _mock_decompose(self, query: ResearchQuery, num: int) -> list[str]:
        """Generate mock sub-questions."""
        base = query.question.lower()

        if query.focus == "scientific":
            templates = [
                f"What is the current scientific consensus on {base}?",
                f"What are the key research findings about {base}?",
                f"What methodologies are used to study {base}?",
                f"What are the limitations of current research on {base}?",
                f"What are future research directions for {base}?",
                f"Who are the leading researchers in {base}?",
            ]
        elif query.focus == "market":
            templates = [
                f"What is the current market size for {base}?",
                f"Who are the key players in {base}?",
                f"What are the growth trends in {base}?",
                f"What are the challenges facing {base}?",
                f"What is the competitive landscape for {base}?",
                f"What are the regulatory considerations for {base}?",
            ]
        else:
            templates = [
                f"What is {base}?",
                f"What are the key aspects of {base}?",
                f"What is the history of {base}?",
                f"What are current developments in {base}?",
                f"What are expert opinions on {base}?",
                f"What are the implications of {base}?",
            ]

        return templates[:num]

    def _generate_search_queries(self, query: ResearchQuery,
                                  sub_questions: list[str]) -> list[str]:
        """Generate diverse search queries."""
        queries = []

        # Base query variations
        base = query.question
        queries.append(base)
        queries.append(f"{base} research")
        queries.append(f"{base} statistics data")

        # Focus-specific queries
        if query.focus == "scientific":
            queries.append(f"{base} peer reviewed")
            queries.append(f"{base} meta analysis")
            queries.append(f"{base} clinical study")
        elif query.focus == "market":
            queries.append(f"{base} market report")
            queries.append(f"{base} industry analysis")
            queries.append(f"{base} market forecast")

        # Sub-questions as queries
        queries.extend(sub_questions)

        return queries[:10]  # Cap

    def _search(self, query: str, max_results: int,
                focus: str) -> list[SearchResult]:
        """Search for sources."""
        if self.searcher:
            return self.searcher(query, max_results)

        if self.multi_search:
            try:
                results = self.multi_search.search(query, limit=max_results, focus=focus)
                return results  # Return whatever we got, even if empty
            except Exception as e:
                print(f"Search error: {e}")
                return []  # Return empty, don't mock

        # Only use mock if explicitly configured without real search
        if not self.multi_search:
            return self._mock_search(query, max_results, focus)

        return []

    def _search_specific_type(self, query: str, source_type: SourceType,
                               max_results: int) -> list[SearchResult]:
        """Search for a specific type of source."""
        if self.multi_search:
            try:
                if source_type == SourceType.ACADEMIC:
                    # Use dedicated academic search
                    return self.multi_search.search_academic(query, limit=max_results)
                elif source_type == SourceType.INDUSTRY:
                    # Search with industry keywords
                    query = f"{query} market report industry analysis"
                    return self.multi_search.search_web(query, limit=max_results)
                else:
                    return self.multi_search.search_web(query, limit=max_results)
            except Exception as e:
                print(f"Specific search error: {e}")

        # Fallback: Add type-specific keywords for mock
        if source_type == SourceType.ACADEMIC:
            query = f"{query} research study"
        elif source_type == SourceType.OFFICIAL:
            query = f"{query} government policy"
        elif source_type == SourceType.INDUSTRY:
            query = f"{query} market report industry analysis"

        return self._mock_search(query, max_results, "general")

    def _mock_search(self, query: str, max_results: int,
                     focus: str) -> list[SearchResult]:
        """Generate mock search results for testing."""
        results = []

        # Simulate diverse source types
        mock_sources = [
            # Academic
            SearchResult(
                title=f"Research on {query[:50]} - Nature",
                url="https://nature.com/articles/example",
                snippet=f"A comprehensive study examining {query}. Our findings suggest significant implications for the field.",
                source_type=SourceType.ACADEMIC,
            ),
            SearchResult(
                title=f"Meta-analysis: {query[:40]} - PubMed",
                url="https://pubmed.ncbi.nlm.nih.gov/12345",
                snippet=f"This meta-analysis reviews 47 studies on {query}. Results indicate moderate effect sizes.",
                source_type=SourceType.ACADEMIC,
            ),
            # Official
            SearchResult(
                title=f"Government Report on {query[:40]}",
                url="https://example.gov/reports/topic",
                snippet=f"Official government findings on {query}. Policy recommendations included.",
                source_type=SourceType.OFFICIAL,
            ),
            # Industry
            SearchResult(
                title=f"Market Analysis: {query[:40]} - Statista",
                url="https://statista.com/study/example",
                snippet=f"Market size for {query} estimated at $X billion. Growth projected at Y% CAGR.",
                source_type=SourceType.INDUSTRY,
            ),
            SearchResult(
                title=f"{query[:40]} Industry Report - McKinsey",
                url="https://mckinsey.com/industries/example",
                snippet=f"Industry leaders are investing heavily in {query}. Key trends identified.",
                source_type=SourceType.INDUSTRY,
            ),
            # News
            SearchResult(
                title=f"Latest developments in {query[:40]} - Reuters",
                url="https://reuters.com/article/example",
                snippet=f"Recent news coverage of {query}. Experts weigh in on implications.",
                source_type=SourceType.NEWS,
            ),
            # Reference
            SearchResult(
                title=f"{query[:50]} - Wikipedia",
                url="https://en.wikipedia.org/wiki/Example",
                snippet=f"Overview of {query}. Historical context and key concepts explained.",
                source_type=SourceType.REFERENCE,
            ),
        ]

        # Select based on focus
        if focus == "scientific":
            priority = [SourceType.ACADEMIC, SourceType.OFFICIAL, SourceType.REFERENCE]
        elif focus == "market":
            priority = [SourceType.INDUSTRY, SourceType.NEWS, SourceType.OFFICIAL]
        else:
            priority = [SourceType.REFERENCE, SourceType.NEWS, SourceType.ACADEMIC]

        # Sort by priority
        sorted_sources = sorted(
            mock_sources,
            key=lambda s: priority.index(s.source_type) if s.source_type in priority else 99
        )

        return sorted_sources[:max_results]

    def _synthesize(self, query: ResearchQuery,
                    findings: list[dict],
                    evidence_quality_notes: list[str] = None,
                    current_data: str = "") -> dict:
        """Synthesize findings into structured output with evidence quality awareness."""
        if self.llm is None:
            return self._mock_synthesize(query, findings)

        # Format findings for LLM with evidence metadata
        findings_text = []
        for f in findings:
            source = f["source"]
            study_info = ""
            if f.get("study_type") and f["study_type"] != "unknown":
                study_info = f" [{f['study_type']}"
                if not f.get("peer_reviewed", True):
                    study_info += ", preprint"
                study_info += "]"
            findings_text.append(
                f"[{source.id}] {source.title}{study_info}\n{f['finding']}\n"
            )

        # Add current data if available
        if current_data:
            findings_text.append(f"\n{current_data}\n")

        # Format evidence quality notes
        quality_text = "\n".join(evidence_quality_notes) if evidence_quality_notes else "No study type metadata available."

        prompt = self.SYNTHESIS_PROMPT.format(
            query=query.question,
            findings="\n".join(findings_text),
            evidence_quality=quality_text,
        )

        response = self._llm_generate(prompt, max_tokens=1500)

        # Parse JSON response
        try:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, AttributeError):
            pass

        return self._mock_synthesize(query, findings)

    def _mock_synthesize(self, query: ResearchQuery,
                         findings: list[dict]) -> dict:
        """Generate mock synthesis."""
        sources = [f["source"] for f in findings]
        source_ids = [s.id for s in sources]

        return {
            "summary": f"Research on '{query.question}' reveals multiple perspectives. "
                      f"Based on {len(sources)} sources, key themes emerge around methodology, "
                      f"current state, and future directions.",
            "key_findings": [
                f"Finding 1 supported by multiple sources [{source_ids[0] if source_ids else 'S1'}]",
                f"Finding 2 shows consensus across academic literature [{source_ids[1] if len(source_ids) > 1 else 'S2'}]",
                "Finding 3 indicates ongoing debate in the field",
            ],
            "evidence": [
                {"claim": "Primary claim", "sources": source_ids[:2], "confidence": "high"},
                {"claim": "Secondary claim", "sources": source_ids[2:4] if len(source_ids) > 2 else source_ids, "confidence": "medium"},
            ],
            "conflicts": ["Some sources disagree on specific details"],
            "gaps": [
                "Long-term studies are limited",
                "Regional data is incomplete",
            ],
            "overall_confidence": "medium",
        }

    def _build_sections(self, synthesis: dict,
                        findings: list[dict]) -> list[dict]:
        """Build document sections from synthesis."""
        sections = []

        # Key findings section
        if synthesis.get("key_findings"):
            sections.append({
                "title": "Key Findings",
                "content": "\n".join(f"- {f}" for f in synthesis["key_findings"]),
            })

        # Evidence section
        if synthesis.get("evidence"):
            evidence_lines = []
            for e in synthesis["evidence"]:
                sources_str = ", ".join(f"[{s}]" for s in e.get("sources", []))
                confidence = e.get("confidence", "unknown")
                evidence_lines.append(
                    f"- **{e['claim']}** (Confidence: {confidence}) {sources_str}"
                )
            sections.append({
                "title": "Supporting Evidence",
                "content": "\n".join(evidence_lines),
            })

        # Conflicts section
        if synthesis.get("conflicts"):
            sections.append({
                "title": "Conflicting Information",
                "content": "\n".join(f"- {c}" for c in synthesis["conflicts"]),
            })

        return sections

    def _calculate_confidence(self, synthesis: dict,
                               validation: ValidationResult = None) -> float:
        """Calculate overall confidence score."""
        conf_map = {"high": 0.85, "medium": 0.6, "low": 0.3}
        overall = synthesis.get("overall_confidence", "medium")

        base_conf = conf_map.get(overall, 0.5)

        # Adjust by source quality
        source_conf = self.source_manager.get_credibility_score()
        diversity = self.source_manager.get_diversity_score()

        # Base calculation
        confidence = (base_conf * 0.4) + (source_conf * 0.3) + (diversity * 0.2)

        # Integrate validation results if available
        if validation:
            # Validation quality weighs 30%
            confidence = (confidence * 0.7) + (validation.overall_quality * 0.3)

            # Penalize for failed gates
            if validation.failed_gates:
                confidence *= max(0.5, 1.0 - (len(validation.failed_gates) * 0.1))

            # Penalize for critical issues
            if validation.critical_issues:
                confidence *= max(0.3, 1.0 - (len(validation.critical_issues) * 0.15))

        return min(1.0, max(0.0, confidence))
