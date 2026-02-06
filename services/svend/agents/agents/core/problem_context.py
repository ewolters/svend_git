"""
Problem Context Integration for Agents

Allows agents to:
1. Read problem context from shared files
2. Write evidence back to problems via API
3. Track their contributions to problem-solving

This is the bridge between the Problem Sessions and individual agents.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import requests


# Default context directory
CONTEXT_DIR = Path("/home/eric/kjerne/services/svend/shared_context/problems")
API_BASE = "http://localhost:8000/api/problems"


@dataclass
class ProblemContext:
    """Problem context loaded from shared file."""
    problem_id: str
    user_id: int
    title: str
    status: str

    # Methodology
    methodology: str
    dmaic_phase: str
    phase_guidance: dict

    # Effect
    effect_description: str
    effect_magnitude: str
    effect_first_observed: str

    # Context
    domain: str
    available_data: str
    can_experiment: bool
    stakeholders: list
    constraints: list

    # Current state
    hypotheses: list
    evidence: list
    probable_causes: list
    key_uncertainties: list
    bias_warnings: list

    # API endpoints for writing back
    add_evidence_url: str
    add_hypothesis_url: str

    @classmethod
    def from_dict(cls, data: dict) -> "ProblemContext":
        """Create from context file dict."""
        methodology = data.get("methodology", {})
        effect = data.get("effect", {})
        context = data.get("context", {})
        understanding = data.get("understanding", {})
        agent_context = data.get("agent_context", {})
        endpoints = agent_context.get("endpoints", {})

        return cls(
            problem_id=data.get("problem_id", ""),
            user_id=data.get("user_id", 0),
            title=data.get("title", ""),
            status=data.get("status", ""),
            methodology=methodology.get("type", "none"),
            dmaic_phase=methodology.get("dmaic_phase", ""),
            phase_guidance=methodology.get("guidance", {}),
            effect_description=effect.get("description", ""),
            effect_magnitude=effect.get("magnitude", ""),
            effect_first_observed=effect.get("first_observed", ""),
            domain=context.get("domain", ""),
            available_data=context.get("available_data", ""),
            can_experiment=context.get("can_experiment", True),
            stakeholders=context.get("stakeholders", []),
            constraints=context.get("constraints", []),
            hypotheses=data.get("hypotheses", []),
            evidence=data.get("evidence", []),
            probable_causes=understanding.get("probable_causes", []),
            key_uncertainties=understanding.get("key_uncertainties", []),
            bias_warnings=data.get("bias_warnings", []),
            add_evidence_url=endpoints.get("add_evidence", ""),
            add_hypothesis_url=endpoints.get("add_hypothesis", ""),
        )

    def to_prompt_context(self) -> str:
        """Format as context for an LLM prompt."""
        lines = [
            f"## Problem: {self.title}",
            "",
            f"**Status:** {self.status}",
            f"**Domain:** {self.domain or 'Not specified'}",
        ]

        if self.methodology != "none":
            lines.append(f"**Methodology:** {self.methodology.upper()}")
            if self.dmaic_phase:
                lines.append(f"**Current Phase:** {self.dmaic_phase.title()}")

        lines.extend([
            "",
            "### The Effect (what we're investigating)",
            self.effect_description,
        ])

        if self.effect_magnitude:
            lines.append(f"**Magnitude:** {self.effect_magnitude}")

        if self.available_data:
            lines.extend([
                "",
                "### Available Data",
                self.available_data,
            ])

        if self.hypotheses:
            lines.extend([
                "",
                "### Current Hypotheses",
            ])
            for h in sorted(self.hypotheses, key=lambda x: x.get("probability", 0), reverse=True):
                prob = h.get("probability", 0.5)
                lines.append(f"- **{h.get('cause', '')}** ({prob*100:.0f}% likely)")
                if h.get("mechanism"):
                    lines.append(f"  Mechanism: {h['mechanism']}")

        if self.probable_causes:
            lines.extend([
                "",
                "### Most Probable Causes",
            ])
            for c in self.probable_causes:
                lines.append(f"- {c.get('cause', '')} ({c.get('probability', 0)*100:.0f}%)")

        if self.key_uncertainties:
            lines.extend([
                "",
                "### Key Uncertainties",
            ])
            for u in self.key_uncertainties:
                lines.append(f"- {u}")

        if self.bias_warnings:
            lines.extend([
                "",
                "### Cognitive Bias Alerts",
            ])
            for b in self.bias_warnings:
                lines.append(f"- **{b.get('type', '')}**: {b.get('description', '')}")

        if self.phase_guidance:
            lines.extend([
                "",
                f"### Phase Guidance ({self.dmaic_phase.title() if self.dmaic_phase else 'Current'})",
                f"**Focus:** {self.phase_guidance.get('focus', '')}",
                "",
                "**Activities:**",
            ])
            for activity in self.phase_guidance.get("activities", []):
                lines.append(f"- {activity}")

        return "\n".join(lines)


class ProblemContextReader:
    """
    Read problem context from shared files.

    Usage:
        reader = ProblemContextReader(user_id=1)

        # Get specific problem
        context = reader.get_problem("problem-uuid")

        # List all problems for user
        problems = reader.list_problems()
    """

    def __init__(self, user_id: int, context_dir: Path = None):
        self.user_id = user_id
        self.context_dir = context_dir or CONTEXT_DIR
        self.user_dir = self.context_dir / str(user_id)

    def get_problem(self, problem_id: str) -> Optional[ProblemContext]:
        """Load a problem's context."""
        context_file = self.user_dir / f"{problem_id}.json"

        if not context_file.exists():
            return None

        with open(context_file) as f:
            data = json.load(f)

        return ProblemContext.from_dict(data)

    def list_problems(self) -> list[dict]:
        """List all problems for this user."""
        if not self.user_dir.exists():
            return []

        problems = []
        for f in self.user_dir.glob("*.json"):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                    problems.append({
                        "id": data.get("problem_id"),
                        "title": data.get("title"),
                        "status": data.get("status"),
                        "methodology": data.get("methodology", {}).get("type", "none"),
                        "dmaic_phase": data.get("methodology", {}).get("dmaic_phase", ""),
                    })
            except (json.JSONDecodeError, KeyError):
                continue

        return problems

    def get_active_problems(self) -> list[ProblemContext]:
        """Get all active problems."""
        problems = []
        for p in self.list_problems():
            if p["status"] == "active":
                ctx = self.get_problem(p["id"])
                if ctx:
                    problems.append(ctx)
        return problems


class ProblemContextWriter:
    """
    Write findings back to problems via API.

    Usage:
        writer = ProblemContextWriter(session_cookie="...")

        # Add evidence to a problem
        writer.add_evidence(
            problem_id="uuid",
            summary="Found correlation between X and Y",
            evidence_type="data_analysis",
            source="Analyst Agent",
            supports=["hypothesis-id-1"],
        )

        # Add a hypothesis
        writer.add_hypothesis(
            problem_id="uuid",
            cause="Increased temperature",
            mechanism="Higher temps cause faster degradation",
            probability=0.6,
        )
    """

    def __init__(self, session_cookie: str = None, api_base: str = None):
        self.session_cookie = session_cookie
        self.api_base = api_base or API_BASE
        self.session = requests.Session()

        if session_cookie:
            self.session.cookies.set("sessionid", session_cookie)

    def add_evidence(
        self,
        problem_id: str,
        summary: str,
        evidence_type: str = "observation",
        source: str = "",
        supports: list = None,
        weakens: list = None,
    ) -> dict:
        """Add evidence to a problem."""
        url = f"{self.api_base}/{problem_id}/evidence/"

        data = {
            "summary": summary,
            "type": evidence_type,
            "source": source,
            "supports": supports or [],
            "weakens": weakens or [],
        }

        response = self.session.post(url, json=data)
        return response.json()

    def add_hypothesis(
        self,
        problem_id: str,
        cause: str,
        mechanism: str = "",
        probability: float = 0.5,
    ) -> dict:
        """Add a hypothesis to a problem."""
        url = f"{self.api_base}/{problem_id}/hypotheses/"

        data = {
            "cause": cause,
            "mechanism": mechanism,
            "probability": probability,
        }

        response = self.session.post(url, json=data)
        return response.json()

    def advance_phase(self, problem_id: str, notes: str = "") -> dict:
        """Advance the problem to the next DMAIC phase."""
        url = f"{self.api_base}/{problem_id}/phase/advance/"

        data = {"notes": notes}

        response = self.session.post(url, json=data)
        return response.json()


# =============================================================================
# Convenience Functions
# =============================================================================

def load_problem_context(problem_id: str, user_id: int) -> Optional[ProblemContext]:
    """Quick function to load a problem context."""
    reader = ProblemContextReader(user_id)
    return reader.get_problem(problem_id)


def get_problem_prompt(problem_id: str, user_id: int) -> str:
    """Get problem context formatted for LLM prompt."""
    context = load_problem_context(problem_id, user_id)
    if context:
        return context.to_prompt_context()
    return ""


def list_user_problems(user_id: int) -> list[dict]:
    """List all problems for a user."""
    reader = ProblemContextReader(user_id)
    return reader.list_problems()
