"""
Intent Tracking System

Stores the user's original intent and checks alignment throughout execution.
Key anti-drift mechanism - every action is validated against the original goal.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class AlignmentStatus(Enum):
    """How well current state aligns with original intent."""
    ALIGNED = "aligned"           # On track
    DRIFTING = "drifting"         # Starting to wander
    OFF_TRACK = "off_track"       # Lost the plot
    COMPLETED = "completed"       # Goal achieved


@dataclass
class Intent:
    """User's original intent/goal."""

    id: str
    raw_input: str                          # Exactly what user typed
    parsed_goal: str                        # Clarified goal
    constraints: list[str] = field(default_factory=list)  # Things to avoid
    success_criteria: list[str] = field(default_factory=list)  # How we know we're done
    created_at: datetime = field(default_factory=datetime.now)

    # Tracking
    actions_taken: list[dict] = field(default_factory=list)
    current_alignment: AlignmentStatus = AlignmentStatus.ALIGNED
    alignment_score: float = 1.0  # 0.0 to 1.0


@dataclass
class Action:
    """An action taken by the agent."""

    id: str
    description: str
    action_type: str  # "code_generation", "file_write", "research", etc.
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)
    alignment_score: float = 1.0
    reasoning: str = ""  # Why this action was taken


class IntentTracker:
    """
    Tracks intent and detects drift.

    Core anti-drift mechanism:
    1. Parse and store original intent
    2. Before each action, check alignment
    3. After each action, update alignment score
    4. Alert or course-correct if drifting
    """

    def __init__(self, llm=None):
        self.llm = llm
        self.current_intent: Intent | None = None
        self.action_history: list[Action] = []

    def set_intent(self, raw_input: str, parsed_goal: str = None,
                   constraints: list[str] = None,
                   success_criteria: list[str] = None) -> Intent:
        """Set the user's intent for this session."""
        import uuid

        self.current_intent = Intent(
            id=str(uuid.uuid4())[:8],
            raw_input=raw_input,
            parsed_goal=parsed_goal or raw_input,
            constraints=constraints or [],
            success_criteria=success_criteria or [],
        )
        return self.current_intent

    def check_alignment(self, proposed_action: str) -> tuple[AlignmentStatus, float, str]:
        """
        Check if a proposed action aligns with the intent.

        Returns:
            (status, score, reasoning)
        """
        if not self.current_intent:
            return AlignmentStatus.ALIGNED, 1.0, "No intent set"

        # Check for constraint violations first
        for constraint in self.current_intent.constraints:
            if constraint.lower() in proposed_action.lower():
                return AlignmentStatus.OFF_TRACK, 0.0, f"Violates constraint: {constraint}"

        # Use LLM for alignment check if available
        if self.llm:
            return self._llm_alignment_check(proposed_action)

        # Fallback to keyword-based check
        return self._keyword_alignment_check(proposed_action)

    def _llm_alignment_check(self, proposed_action: str) -> tuple[AlignmentStatus, float, str]:
        """Use LLM to check alignment - more accurate than keywords."""
        prompt = f"""You are checking if an action aligns with the user's original intent.

ORIGINAL INTENT: {self.current_intent.parsed_goal}

CONSTRAINTS:
{chr(10).join(f'- {c}' for c in self.current_intent.constraints) or 'None'}

PROPOSED ACTION/OUTPUT:
{proposed_action[:1000]}

Rate the alignment from 0 to 100, where:
- 90-100: Perfectly aligned, exactly what was asked
- 70-89: Well aligned, minor additions acceptable
- 50-69: Partially aligned, some drift from intent
- 30-49: Significant drift, added unrequested features
- 0-29: Off track, not what was asked

Respond with ONLY a JSON object:
{{"score": <number>, "reasoning": "<brief explanation>"}}"""

        try:
            if hasattr(self.llm, 'complete'):
                response = self.llm.complete(prompt, max_tokens=150)
            else:
                response = self.llm.generate(prompt, max_tokens=150)

            # Parse response
            import json
            import re
            match = re.search(r'\{[^}]+\}', response)
            if match:
                data = json.loads(match.group())
                score = float(data.get('score', 50)) / 100
                reasoning = data.get('reasoning', 'LLM alignment check')

                if score > 0.7:
                    status = AlignmentStatus.ALIGNED
                elif score > 0.4:
                    status = AlignmentStatus.DRIFTING
                else:
                    status = AlignmentStatus.OFF_TRACK

                return status, score, reasoning
        except Exception as e:
            pass  # Fall back to keyword check

        return self._keyword_alignment_check(proposed_action)

    def _keyword_alignment_check(self, proposed_action: str) -> tuple[AlignmentStatus, float, str]:
        """Fallback keyword-based alignment check."""
        intent_words = set(self.current_intent.parsed_goal.lower().split())
        action_words = set(proposed_action.lower().split())

        # Simple overlap score
        overlap = len(intent_words & action_words)
        total = len(intent_words)
        score = overlap / total if total > 0 else 0.5

        # Adjust based on action history (drift accumulates)
        recent_scores = [a.alignment_score for a in self.action_history[-3:]]
        if recent_scores:
            trend = sum(recent_scores) / len(recent_scores)
            score = (score + trend) / 2

        if score > 0.7:
            status = AlignmentStatus.ALIGNED
            reasoning = "Action aligns with intent"
        elif score > 0.4:
            status = AlignmentStatus.DRIFTING
            reasoning = "Action partially aligns - review recommended"
        else:
            status = AlignmentStatus.OFF_TRACK
            reasoning = "Action does not align with original intent"

        return status, score, reasoning

    def record_action(self, action: Action) -> None:
        """Record an action and update alignment tracking."""
        self.action_history.append(action)

        if self.current_intent:
            self.current_intent.actions_taken.append({
                "id": action.id,
                "type": action.action_type,
                "description": action.description,
                "alignment": action.alignment_score,
            })

            # Update overall alignment
            recent_scores = [a.alignment_score for a in self.action_history[-5:]]
            self.current_intent.alignment_score = sum(recent_scores) / len(recent_scores)

            if self.current_intent.alignment_score > 0.7:
                self.current_intent.current_alignment = AlignmentStatus.ALIGNED
            elif self.current_intent.alignment_score > 0.4:
                self.current_intent.current_alignment = AlignmentStatus.DRIFTING
            else:
                self.current_intent.current_alignment = AlignmentStatus.OFF_TRACK

    def is_drifting(self) -> bool:
        """Quick check if we're drifting from intent."""
        if not self.current_intent:
            return False
        return self.current_intent.current_alignment in [
            AlignmentStatus.DRIFTING,
            AlignmentStatus.OFF_TRACK
        ]

    def get_course_correction(self) -> str:
        """Get suggestion to get back on track."""
        if not self.current_intent:
            return "No intent set"

        return f"""
Original goal: {self.current_intent.parsed_goal}

Recent actions seem to be drifting. Consider:
1. Re-reading the original request
2. Checking if current approach still serves the goal
3. Removing any additions not explicitly requested

Constraints to remember:
{chr(10).join(f'- {c}' for c in self.current_intent.constraints) or '- None specified'}
"""

    def summary(self) -> str:
        """Get current tracking summary."""
        if not self.current_intent:
            return "No intent set"

        return f"""
Intent: {self.current_intent.parsed_goal}
Status: {self.current_intent.current_alignment.value}
Alignment: {self.current_intent.alignment_score:.0%}
Actions taken: {len(self.action_history)}
"""
