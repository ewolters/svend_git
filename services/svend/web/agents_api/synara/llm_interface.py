"""
Synara LLM Interface: Bridge Between Formal Engine and Language Models

LLMs provide the intelligence layer:
- Research: find prior art, base rates, similar cases
- Generate: propose hypotheses, suggest causes
- Document: write reports, summaries
- Validate: parse causal graphs for logical errors

Synara provides the formal engine:
- Bayesian updates
- Probability propagation
- Expansion signals

This module bridges them.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from datetime import datetime

from .kernel import (
    HypothesisRegion,
    Evidence,
    CausalGraph,
    CausalLink,
    ExpansionSignal,
)
from .synara import Synara, UpdateResult


@dataclass
class LogicalIssue:
    """A logical issue detected in the causal graph."""
    issue_type: str  # "circular", "unsupported_leap", "missing_premise", "contradiction"
    severity: str    # "error", "warning", "suggestion"
    description: str
    involved_hypotheses: list[str]
    suggested_fix: Optional[str] = None


@dataclass
class GraphAnalysis:
    """LLM analysis of the causal graph."""
    issues: list[LogicalIssue]
    strengths: list[str]
    gaps: list[str]
    suggested_hypotheses: list[dict]
    suggested_evidence: list[str]
    summary: str


class SynaraLLMInterface:
    """
    Interface for LLM interaction with Synara.

    Provides structured prompts and parsing for:
    1. Generating hypotheses from expansion signals
    2. Validating causal graph logic
    3. Interpreting evidence
    4. Documenting findings
    """

    def __init__(self, synara: Synara):
        self.synara = synara

    # =========================================================================
    # Prompt Generation
    # =========================================================================

    def generate_hypothesis_prompt(
        self,
        expansion_signal: ExpansionSignal,
    ) -> str:
        """
        Generate a prompt for the LLM to propose hypotheses.

        Used when expansion signal indicates incomplete causal surface.
        """
        existing_hypotheses = "\n".join([
            f"- {h.description} (P={h.posterior:.2f})"
            for h in self.synara.get_all_hypotheses()
        ])

        return f"""An evidence observation does not fit any existing hypothesis well.

EVIDENCE:
- Event: {expansion_signal.event}
- Context: {expansion_signal.context}

EXISTING HYPOTHESES (all have low likelihood for this evidence):
{existing_hypotheses}

LIKELIHOODS:
{expansion_signal.likelihoods}

The causal surface appears incomplete. Either:
1. A new cause exists that we haven't considered (missing disjunct)
2. An existing hypothesis needs additional premises (missing conjunct)

Please propose:
1. 1-3 new hypothesis regions that could explain this evidence
2. For each, specify:
   - Description (what cause does this represent?)
   - Domain conditions (when does this apply?)
   - Behavior class (what behavior does it produce?)
   - Latent causes (what underlying factors?)
   - Initial probability estimate

Format as JSON:
{{
  "hypotheses": [
    {{
      "description": "...",
      "domain_conditions": {{}},
      "behavior_class": "...",
      "latent_causes": [],
      "prior": 0.3
    }}
  ],
  "reasoning": "..."
}}"""

    def generate_validation_prompt(self) -> str:
        """
        Generate a prompt for the LLM to validate the causal graph.

        Checks for logical errors, unsupported leaps, circular reasoning.
        """
        # Build graph description
        hypotheses_desc = []
        for h in self.synara.get_all_hypotheses():
            hypotheses_desc.append(
                f"H[{h.id}]: {h.description}\n"
                f"  Domain: {h.domain_conditions}\n"
                f"  Behavior: {h.behavior_class}\n"
                f"  Latent causes: {h.latent_causes}\n"
                f"  P={h.posterior:.3f}"
            )

        links_desc = []
        for link in self.synara.graph.links:
            links_desc.append(
                f"{link.from_id} → {link.to_id}: {link.mechanism} (strength={link.strength})"
            )

        evidence_desc = []
        for e in self.synara.graph.evidence[-10:]:  # Last 10
            evidence_desc.append(f"E[{e.id}]: {e.event} | {e.context}")

        return f"""Please analyze this causal graph for logical issues.

HYPOTHESES:
{chr(10).join(hypotheses_desc)}

CAUSAL LINKS:
{chr(10).join(links_desc) if links_desc else "None"}

RECENT EVIDENCE:
{chr(10).join(evidence_desc) if evidence_desc else "None"}

Check for:
1. CIRCULAR REASONING: A → B → C → A
2. UNSUPPORTED LEAPS: Links without clear mechanism
3. MISSING PREMISES: Hypotheses that need additional conditions
4. CONTRADICTIONS: Evidence that supports and weakens same hypothesis
5. PROBABILITY ISSUES: Posteriors that don't make sense given evidence

Format response as JSON:
{{
  "issues": [
    {{
      "issue_type": "circular|unsupported_leap|missing_premise|contradiction|probability",
      "severity": "error|warning|suggestion",
      "description": "...",
      "involved_hypotheses": ["h_id1", "h_id2"],
      "suggested_fix": "..."
    }}
  ],
  "strengths": ["...", "..."],
  "gaps": ["...", "..."],
  "suggested_hypotheses": [
    {{
      "description": "...",
      "connects_to": ["h_id"]
    }}
  ],
  "suggested_evidence": ["What evidence would distinguish hypotheses?"],
  "summary": "Overall assessment..."
}}"""

    def generate_evidence_interpretation_prompt(
        self,
        evidence: Evidence,
        update_result: UpdateResult,
    ) -> str:
        """
        Generate a prompt for the LLM to interpret an evidence update.

        Explains what the evidence means for the hypothesis space.
        """
        top_hypotheses = self.synara.get_competing_hypotheses(threshold=0.15)
        top_desc = "\n".join([
            f"- {h.description}: P={h.posterior:.3f}"
            for h in top_hypotheses[:5]
        ])

        return f"""New evidence was added. Please interpret its implications.

EVIDENCE:
- Event: {evidence.event}
- Context: {evidence.context}
- Source: {evidence.source}

BELIEF CHANGES:
- Most supported hypothesis: {update_result.most_supported}
- Most weakened hypothesis: {update_result.most_weakened}
- Likelihoods: {update_result.likelihoods}

CURRENT TOP HYPOTHESES:
{top_desc}

{'EXPANSION SIGNAL: ' + update_result.expansion_signal.message if update_result.expansion_signal else ''}

Please provide:
1. A plain-language interpretation of what this evidence means
2. Which hypotheses are now more/less likely and why
3. What additional evidence would help distinguish remaining hypotheses
4. Any concerns about the evidence quality or interpretation

Keep response concise (2-3 paragraphs)."""

    def generate_documentation_prompt(
        self,
        format_type: str = "summary",
    ) -> str:
        """
        Generate a prompt for the LLM to document findings.

        format_type: "summary", "a3", "8d", "technical"
        """
        top_hypothesis = self.synara.get_most_likely_cause()
        competing = self.synara.get_competing_hypotheses(threshold=0.1)
        pending_expansions = self.synara.get_pending_expansions()

        format_instructions = {
            "summary": "Write a brief executive summary (3-5 paragraphs)",
            "a3": "Format as an A3 report with: Background, Current State, Root Cause, Countermeasures",
            "8d": "Format as an 8D report: D1-D8 sections",
            "technical": "Write a detailed technical analysis with methodology and limitations",
        }

        return f"""Document the current state of this causal analysis.

MOST LIKELY CAUSE:
{top_hypothesis.description if top_hypothesis else 'None identified'}
Probability: {top_hypothesis.posterior:.1%} if top_hypothesis else 'N/A'

COMPETING HYPOTHESES:
{chr(10).join([f'- {h.description} ({h.posterior:.1%})' for h in competing[:5]])}

EVIDENCE CONSIDERED:
{len(self.synara.graph.evidence)} pieces of evidence

UNRESOLVED ISSUES:
{len(pending_expansions)} expansion signals (incomplete causal surface)

FORMAT: {format_instructions.get(format_type, format_instructions['summary'])}

Include:
1. What we know with confidence
2. What remains uncertain
3. What evidence would resolve uncertainty
4. Recommended next steps"""

    # =========================================================================
    # Response Parsing
    # =========================================================================

    def parse_hypothesis_response(
        self,
        response: dict,
    ) -> list[HypothesisRegion]:
        """Parse LLM response into HypothesisRegion objects."""
        hypotheses = []

        for h_data in response.get("hypotheses", []):
            h = self.synara.create_hypothesis(
                description=h_data.get("description", "LLM-generated hypothesis"),
                domain_conditions=h_data.get("domain_conditions", {}),
                behavior_class=h_data.get("behavior_class", ""),
                latent_causes=h_data.get("latent_causes", []),
                prior=h_data.get("prior", 0.3),
                source="llm",
            )
            hypotheses.append(self.synara.get_hypothesis(h.id))

        return hypotheses

    def parse_validation_response(
        self,
        response: dict,
    ) -> GraphAnalysis:
        """Parse LLM validation response into structured analysis."""
        issues = []
        for issue_data in response.get("issues", []):
            issues.append(LogicalIssue(
                issue_type=issue_data.get("issue_type", "unknown"),
                severity=issue_data.get("severity", "warning"),
                description=issue_data.get("description", ""),
                involved_hypotheses=issue_data.get("involved_hypotheses", []),
                suggested_fix=issue_data.get("suggested_fix"),
            ))

        return GraphAnalysis(
            issues=issues,
            strengths=response.get("strengths", []),
            gaps=response.get("gaps", []),
            suggested_hypotheses=response.get("suggested_hypotheses", []),
            suggested_evidence=response.get("suggested_evidence", []),
            summary=response.get("summary", ""),
        )

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def get_state_summary(self) -> dict:
        """Get a summary of current Synara state for LLM context."""
        top = self.synara.get_most_likely_cause()
        competing = self.synara.get_competing_hypotheses()
        pending = self.synara.get_pending_expansions()

        return {
            "hypothesis_count": len(self.synara.graph.hypotheses),
            "evidence_count": len(self.synara.graph.evidence),
            "link_count": len(self.synara.graph.links),
            "top_hypothesis": {
                "id": top.id,
                "description": top.description,
                "posterior": top.posterior,
            } if top else None,
            "competing_count": len(competing),
            "pending_expansions": len(pending),
            "last_update": self.synara.update_history[-1].to_dict() if self.synara.update_history else None,
        }

    def format_for_context(self, max_hypotheses: int = 10) -> str:
        """Format Synara state as context for LLM prompts."""
        lines = ["Current belief state:"]

        for h in self.synara.get_all_hypotheses()[:max_hypotheses]:
            lines.append(f"- {h.description}: P={h.posterior:.2f}")

        if self.synara.expansion_signals:
            pending = self.synara.get_pending_expansions()
            if pending:
                lines.append(f"\n{len(pending)} unresolved expansion signals (incomplete causal surface)")

        return "\n".join(lines)
