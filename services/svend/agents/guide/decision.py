"""
Decision Guide - The Front Door to Decision Science

Helps users:
1. Frame their problem clearly
2. Detect cognitive biases in their thinking
3. Decide if data science is even the right approach
4. Route to appropriate tools or structured thinking

This is the "thinking partner" that makes good decisions more accessible.
"""

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional

from .agent import GuideAgent, Section, Question, QuestionType, InterviewResult


# =============================================================================
# Cognitive Bias Detection
# =============================================================================

@dataclass
class BiasWarning:
    """A detected cognitive bias."""
    bias_type: str
    description: str
    evidence: str  # What triggered the detection
    suggestion: str  # How to counter it
    severity: Literal["low", "medium", "high"] = "medium"


# Patterns that suggest cognitive biases
BIAS_PATTERNS = {
    "confirmation": {
        "patterns": [
            r"\b(prove|confirm|show that|validate)\b.*\b(right|correct|true)\b",
            r"\bi('m| am) (sure|certain|convinced)\b",
            r"\bwe know that\b",
            r"\bobviously\b",
            r"\beveryone knows\b",
        ],
        "description": "Confirmation bias - seeking evidence that supports existing beliefs",
        "suggestion": "Try asking: 'What evidence would change my mind?' Look for disconfirming data.",
    },
    "anchoring": {
        "patterns": [
            r"\b(around|about|approximately)\s+\d+",
            r"\bi('d| would) expect\s+(around|about)?\s*\d+",
            r"\bstarting (point|assumption)\b",
            r"\bbased on (my|our) (initial|first)\b",
        ],
        "description": "Anchoring bias - over-relying on first piece of information",
        "suggestion": "Generate your estimate independently before looking at any numbers. Consider multiple reference points.",
    },
    "availability": {
        "patterns": [
            r"\brecently\b.*\b(happened|saw|heard|read)\b",
            r"\bin the news\b",
            r"\beveryone('s| is) (talking about|worried about)\b",
            r"\bi (just )?(saw|heard|read)\b",
        ],
        "description": "Availability heuristic - overweighting recent or memorable events",
        "suggestion": "Look at base rates and historical data, not just recent examples.",
    },
    "survivorship": {
        "patterns": [
            r"\bsuccessful (companies|people|cases)\b",
            r"\bbest practices\b",
            r"\bwhat (winners|leaders|top performers) do\b",
            r"\blearn from (the best|success)\b",
        ],
        "description": "Survivorship bias - only looking at successful cases",
        "suggestion": "Also study failures. What did unsuccessful cases have in common with successful ones?",
    },
    "sunk_cost": {
        "patterns": [
            r"\balready (invested|spent|put in)\b",
            r"\bcan't (stop|quit|give up) now\b",
            r"\bcome (this|so) far\b",
            r"\btoo much (time|money|effort) to\b",
        ],
        "description": "Sunk cost fallacy - letting past investment affect future decisions",
        "suggestion": "Ask: 'If I were starting fresh today, would I make this choice?' Past costs are irrelevant to future value.",
    },
    "overconfidence": {
        "patterns": [
            r"\bdefinitely\b",
            r"\bguaranteed\b",
            r"\bno (way|chance)\b",
            r"\b100\s*%\s*(sure|certain|confident)\b",
            r"\bimpossible\b",
            r"\bcertain(ly)?\b",
        ],
        "description": "Overconfidence - being too certain about uncertain things",
        "suggestion": "Assign actual probabilities. Consider: what's the range of possible outcomes?",
    },
    "base_rate_neglect": {
        "patterns": [
            r"\bthis (case|situation|time) is (different|special|unique)\b",
            r"\bwe're not like (other|those|them)\b",
            r"\bdoesn't apply to (us|this)\b",
        ],
        "description": "Base rate neglect - ignoring how common something is in general",
        "suggestion": "Start with the base rate: How often does this happen in general? Then adjust based on specifics.",
    },
    "narrative": {
        "patterns": [
            r"\bthe (story|narrative) is\b",
            r"\bit makes sense (that|because)\b",
            r"\bthe reason (is|was)\b(?!.*\bdata\b)",
        ],
        "description": "Narrative fallacy - creating stories to explain random events",
        "suggestion": "Be wary of neat explanations. Reality is often messier. Look for data, not just stories.",
    },
}


def detect_biases(text: str) -> list[BiasWarning]:
    """Detect cognitive biases in user's text."""
    warnings = []
    text_lower = text.lower()

    for bias_type, config in BIAS_PATTERNS.items():
        for pattern in config["patterns"]:
            match = re.search(pattern, text_lower)
            if match:
                warnings.append(BiasWarning(
                    bias_type=bias_type,
                    description=config["description"],
                    evidence=match.group(0),
                    suggestion=config["suggestion"],
                    severity="medium",
                ))
                break  # One warning per bias type

    return warnings


# =============================================================================
# Decision Framing
# =============================================================================

@dataclass
class DecisionFrame:
    """A framed decision with key elements identified."""
    core_question: str
    decision_type: Literal["choice", "estimation", "prediction", "diagnosis", "optimization"]
    stakeholders: list[str]
    constraints: list[str]
    assumptions: list[str]
    success_criteria: list[str]
    reversibility: Literal["reversible", "partially_reversible", "irreversible"]
    time_horizon: str
    data_needs: Literal["no_data", "existing_data", "need_to_collect", "need_to_generate"]
    recommended_approach: str
    bias_warnings: list[BiasWarning] = field(default_factory=list)


@dataclass
class DecisionBrief:
    """
    Structured output for handoff to other agents.

    This is the standardized format that DSW, Researcher, Analyst, etc. can consume.
    """
    # Core decision info
    question: str
    context: str
    decision_type: str  # choice, estimation, prediction, diagnosis, optimization

    # Risk assessment
    reversibility: str  # reversible, partially_reversible, irreversible
    cost_of_wrong_high: str
    cost_of_wrong_low: str
    time_pressure: str

    # Current state
    initial_hypothesis: str
    confidence_level: int  # 1-10
    assumptions: list[str]

    # Data context
    available_data: str
    data_quality: str  # high, medium, low, none
    data_gaps: list[str]

    # Bias alerts
    detected_biases: list[dict]  # [{type, description, suggestion}]

    # Routing
    recommended_agent: str  # dsw, researcher, analyst, none
    recommended_action: str
    routing_reasoning: str

    # Stakeholders
    stakeholders: list[str]

    def to_json(self) -> str:
        """Export as JSON string."""
        return json.dumps(asdict(self), indent=2)

    def to_dict(self) -> dict:
        """Export as dictionary."""
        return asdict(self)

    @classmethod
    def from_json(cls, json_str: str) -> "DecisionBrief":
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls(**data)

    def to_prompt_context(self) -> str:
        """Format as context for an LLM prompt."""
        lines = [
            "## Decision Context",
            f"**Question:** {self.question}",
            f"**Type:** {self.decision_type}",
            f"**Context:** {self.context}",
            "",
            "## Risk Profile",
            f"- Reversibility: {self.reversibility}",
            f"- If wrong (overestimate): {self.cost_of_wrong_high}",
            f"- If wrong (underestimate): {self.cost_of_wrong_low}",
            f"- Time pressure: {self.time_pressure}",
            "",
            "## Current Thinking",
            f"- Initial hypothesis: {self.initial_hypothesis}",
            f"- Confidence: {self.confidence_level}/10",
            f"- Key assumptions: {', '.join(self.assumptions) if self.assumptions else 'None stated'}",
            "",
            "## Data Situation",
            f"- Available: {self.available_data}",
            f"- Quality: {self.data_quality}",
            f"- Gaps: {', '.join(self.data_gaps) if self.data_gaps else 'None identified'}",
        ]

        if self.detected_biases:
            lines.extend([
                "",
                "## ⚠️ Bias Alerts",
            ])
            for bias in self.detected_biases:
                lines.append(f"- **{bias['type']}**: {bias['description']}")

        return "\n".join(lines)


# =============================================================================
# Decision Guide Agent
# =============================================================================

class DecisionGuide(GuideAgent):
    """
    A guided interview that helps frame decisions and detect biases.

    This is the "front door" to the decision science workbench.
    Instead of jumping straight to data science, it helps users:
    1. Clarify what they're actually deciding
    2. Notice their own biases
    3. Choose the right approach (which may not be data science)
    """

    def __init__(self, llm=None):
        super().__init__(llm=llm)
        self.bias_warnings = []
        self._setup_sections()

    def _setup_sections(self):
        """Define the interview sections."""
        self.sections = [
            Section(
                id="framing",
                title="Understanding Your Decision",
                description="Let's make sure we understand what you're trying to figure out.",
                questions=[
                    Question(
                        id="situation",
                        text="What situation or challenge are you facing?",
                        question_type=QuestionType.TEXT,
                        help_text="Describe the context in your own words.",
                    ),
                    Question(
                        id="decision",
                        text="What specific decision do you need to make?",
                        question_type=QuestionType.TEXT,
                        help_text="Try to phrase it as a question you need to answer.",
                    ),
                    Question(
                        id="decision_type",
                        text="What type of decision is this?",
                        question_type=QuestionType.CHOICE,
                        options=[
                            "Choosing between options (A vs B vs C)",
                            "Estimating a number or quantity",
                            "Predicting what will happen",
                            "Diagnosing why something happened",
                            "Optimizing a process or outcome",
                            "Exploring/understanding a topic",
                        ],
                    ),
                    Question(
                        id="change_mind",
                        text="What information or evidence would change your mind?",
                        question_type=QuestionType.TEXT,
                        help_text="This helps identify what data actually matters.",
                    ),
                ],
            ),
            Section(
                id="stakes",
                title="Stakes and Constraints",
                description="Understanding what's at risk and what limits you have.",
                questions=[
                    Question(
                        id="wrong_high",
                        text="What happens if you're wrong on the high side?",
                        question_type=QuestionType.TEXT,
                        help_text="e.g., overestimating demand, being too optimistic",
                    ),
                    Question(
                        id="wrong_low",
                        text="What happens if you're wrong on the low side?",
                        question_type=QuestionType.TEXT,
                        help_text="e.g., underestimating risk, being too cautious",
                    ),
                    Question(
                        id="reversibility",
                        text="How reversible is this decision?",
                        question_type=QuestionType.CHOICE,
                        options=[
                            "Easily reversible - can change course anytime",
                            "Somewhat reversible - can undo but with effort/cost",
                            "Mostly irreversible - hard to go back",
                            "Completely irreversible - one-way door",
                        ],
                    ),
                    Question(
                        id="time_horizon",
                        text="When do you need to decide, and how long will the effects last?",
                        question_type=QuestionType.TEXT,
                        help_text="e.g., 'Decide by Friday, effects for 1 year'",
                    ),
                    Question(
                        id="stakeholders",
                        text="Who else is affected by this decision?",
                        question_type=QuestionType.TEXT,
                        help_text="List people, teams, customers, etc.",
                        required=False,
                    ),
                ],
            ),
            Section(
                id="current_thinking",
                title="Your Current Thinking",
                description="Understanding where you're starting from.",
                questions=[
                    Question(
                        id="initial_lean",
                        text="What's your gut feeling or initial lean?",
                        question_type=QuestionType.TEXT,
                        help_text="It's okay to have one - let's make it explicit.",
                    ),
                    Question(
                        id="confidence",
                        text="How confident are you in that initial view?",
                        question_type=QuestionType.SCALE,
                        scale_min=1,
                        scale_max=10,
                        help_text="1 = pure guess, 10 = highly confident",
                    ),
                    Question(
                        id="assumptions",
                        text="What assumptions are you making?",
                        question_type=QuestionType.TEXT,
                        help_text="What has to be true for your initial view to be correct?",
                    ),
                    Question(
                        id="tried",
                        text="What have you already tried or considered?",
                        question_type=QuestionType.TEXT,
                        required=False,
                    ),
                ],
            ),
            Section(
                id="data",
                title="Data and Evidence",
                description="Understanding what information is available.",
                questions=[
                    Question(
                        id="have_data",
                        text="What relevant data or evidence do you already have?",
                        question_type=QuestionType.TEXT,
                        help_text="Could be numbers, past experience, research, expert opinions...",
                    ),
                    Question(
                        id="data_quality",
                        text="How would you rate the quality of that data?",
                        question_type=QuestionType.CHOICE,
                        options=[
                            "High - reliable, recent, directly relevant",
                            "Medium - somewhat relevant but has gaps",
                            "Low - limited, outdated, or tangential",
                            "None - I don't really have data",
                        ],
                    ),
                    Question(
                        id="missing_data",
                        text="What data would be ideal but you don't have?",
                        question_type=QuestionType.TEXT,
                        required=False,
                    ),
                ],
            ),
        ]

    def submit_answer(self, answer):
        """Override to check for biases in text responses."""
        question = self.get_current_question()

        # Detect biases in text answers
        if question and question.question_type == QuestionType.TEXT and answer:
            biases = detect_biases(str(answer))
            self.bias_warnings.extend(biases)

        return super().submit_answer(answer)

    def synthesize(self) -> InterviewResult:
        """Synthesize into a decision frame with recommendations."""
        base_result = super().synthesize()

        # Analyze and create decision frame
        frame = self._create_decision_frame()

        # Add decision-specific synthesis
        base_result.synthesized_output = {
            "Decision Frame": {
                "Core Question": frame.core_question,
                "Decision Type": frame.decision_type,
                "Time Horizon": frame.time_horizon,
                "Reversibility": frame.reversibility,
            },
            "Key Assumptions": frame.assumptions,
            "What Would Change Your Mind": self.state.answers.get("change_mind", "Not specified"),
            "Asymmetric Risks": {
                "If wrong high": self.state.answers.get("wrong_high", ""),
                "If wrong low": self.state.answers.get("wrong_low", ""),
            },
            "Bias Warnings": [
                f"⚠️ {w.description}\n   Triggered by: '{w.evidence}'\n   Suggestion: {w.suggestion}"
                for w in self.bias_warnings
            ] if self.bias_warnings else ["No obvious biases detected (but stay vigilant!)"],
            "Recommended Approach": frame.recommended_approach,
            "Data Assessment": {
                "Available Data": self.state.answers.get("have_data", "None specified"),
                "Data Quality": self.state.answers.get("data_quality", "Unknown"),
                "Data Gaps": self.state.answers.get("missing_data", "None identified"),
                "Data Needs": frame.data_needs,
            },
        }

        return base_result

    def _create_decision_frame(self) -> DecisionFrame:
        """Create a structured decision frame from answers."""
        answers = self.state.answers

        # Determine decision type
        decision_type_map = {
            "Choosing between options": "choice",
            "Estimating a number": "estimation",
            "Predicting what will happen": "prediction",
            "Diagnosing why": "diagnosis",
            "Optimizing": "optimization",
            "Exploring": "diagnosis",
        }
        raw_type = answers.get("decision_type", "")
        decision_type = "choice"
        for key, val in decision_type_map.items():
            if key.lower() in raw_type.lower():
                decision_type = val
                break

        # Determine reversibility
        reversibility_map = {
            "Easily": "reversible",
            "Somewhat": "partially_reversible",
            "Mostly": "irreversible",
            "Completely": "irreversible",
        }
        raw_rev = answers.get("reversibility", "")
        reversibility = "partially_reversible"
        for key, val in reversibility_map.items():
            if key.lower() in raw_rev.lower():
                reversibility = val
                break

        # Determine data needs
        data_quality = answers.get("data_quality", "").lower()
        if "none" in data_quality or "don't" in data_quality:
            data_needs = "need_to_generate"
        elif "low" in data_quality:
            data_needs = "need_to_collect"
        elif "medium" in data_quality:
            data_needs = "existing_data"
        else:
            data_needs = "existing_data"

        # Recommend approach based on analysis
        recommended = self._recommend_approach(decision_type, reversibility, data_needs, answers)

        # Parse assumptions (split on common delimiters)
        assumptions_raw = answers.get("assumptions", "")
        assumptions = [a.strip() for a in re.split(r'[,;\n]', assumptions_raw) if a.strip()]

        # Parse stakeholders
        stakeholders_raw = answers.get("stakeholders", "")
        stakeholders = [s.strip() for s in re.split(r'[,;\n]', stakeholders_raw) if s.strip()]

        return DecisionFrame(
            core_question=answers.get("decision", ""),
            decision_type=decision_type,
            stakeholders=stakeholders,
            constraints=[],
            assumptions=assumptions if assumptions else ["None explicitly stated"],
            success_criteria=[],
            reversibility=reversibility,
            time_horizon=answers.get("time_horizon", "Not specified"),
            data_needs=data_needs,
            recommended_approach=recommended,
            bias_warnings=self.bias_warnings,
        )

    def _recommend_approach(self, decision_type: str, reversibility: str,
                           data_needs: str, answers: dict) -> str:
        """Recommend an approach based on the decision characteristics."""
        recommendations = []

        confidence = int(answers.get("confidence", 5))
        has_data = "high" in answers.get("data_quality", "").lower()

        # High confidence + irreversible = slow down
        if confidence >= 8 and reversibility == "irreversible":
            recommendations.append(
                "⚠️ You're very confident about an irreversible decision. "
                "Consider a 'pre-mortem': Imagine it's a year from now and this failed. What went wrong?"
            )

        # Low confidence + reversible = just try it
        if confidence <= 4 and reversibility == "reversible":
            recommendations.append(
                "This is reversible and you're uncertain. Consider just trying it and learning from the results "
                "rather than over-analyzing."
            )

        # Data-driven recommendations
        if data_needs == "need_to_generate" and decision_type in ["prediction", "estimation"]:
            recommendations.append(
                "→ Use DSW (Data Science Workbench) to generate synthetic data and build a model. "
                "This can help explore scenarios even without real data."
            )
        elif data_needs == "existing_data" and has_data:
            recommendations.append(
                "→ You have decent data. Consider using the Analyst to explore patterns, "
                "or DSW to build a predictive model."
            )
        elif data_needs in ["need_to_collect", "need_to_generate"]:
            recommendations.append(
                "→ Use the Researcher to find existing studies, statistics, and expert knowledge. "
                "This might be faster than collecting your own data."
            )

        # Decision-type specific
        if decision_type == "choice" and len(recommendations) == 0:
            recommendations.append(
                "For choices between options, try listing pros/cons for each, then ask: "
                "'Which downside am I most willing to live with?'"
            )

        if decision_type == "diagnosis":
            recommendations.append(
                "For diagnosis, the Researcher can help find similar cases and root cause analyses. "
                "Be careful of narrative fallacy - correlation isn't causation."
            )

        if not recommendations:
            recommendations.append(
                "Based on your inputs, I'd recommend starting with the Researcher to gather background, "
                "then deciding if you need data analysis."
            )

        return "\n\n".join(recommendations)

    def get_bias_summary(self) -> str:
        """Get a summary of detected biases."""
        if not self.bias_warnings:
            return "No cognitive biases detected in your responses."

        lines = ["## Potential Cognitive Biases Detected", ""]
        for w in self.bias_warnings:
            lines.append(f"### {w.bias_type.replace('_', ' ').title()}")
            lines.append(f"**What it is:** {w.description}")
            lines.append(f"**Triggered by:** \"{w.evidence}\"")
            lines.append(f"**Suggestion:** {w.suggestion}")
            lines.append("")

        return "\n".join(lines)

    def get_brief(self) -> DecisionBrief:
        """
        Get structured output for handoff to other agents.

        This is the primary output format for routing decisions.
        """
        answers = self.state.answers
        frame = self._create_decision_frame()

        # Determine recommended agent
        agent, action, reasoning = self._determine_routing(frame, answers)

        # Parse data gaps
        data_gaps_raw = answers.get("missing_data", "")
        data_gaps = [g.strip() for g in re.split(r'[,;\n]', data_gaps_raw) if g.strip()]

        return DecisionBrief(
            question=answers.get("decision", ""),
            context=answers.get("situation", ""),
            decision_type=frame.decision_type,
            reversibility=frame.reversibility,
            cost_of_wrong_high=answers.get("wrong_high", "Not specified"),
            cost_of_wrong_low=answers.get("wrong_low", "Not specified"),
            time_pressure=answers.get("time_horizon", "Not specified"),
            initial_hypothesis=answers.get("initial_lean", ""),
            confidence_level=int(answers.get("confidence", 5)),
            assumptions=frame.assumptions,
            available_data=answers.get("have_data", "None"),
            data_quality=answers.get("data_quality", "Unknown").split(" - ")[0].lower(),
            data_gaps=data_gaps if data_gaps else [],
            detected_biases=[
                {
                    "type": w.bias_type,
                    "description": w.description,
                    "evidence": w.evidence,
                    "suggestion": w.suggestion,
                }
                for w in self.bias_warnings
            ],
            recommended_agent=agent,
            recommended_action=action,
            routing_reasoning=reasoning,
            stakeholders=frame.stakeholders,
        )

    def _determine_routing(self, frame: DecisionFrame, answers: dict) -> tuple[str, str, str]:
        """Determine which agent/approach to route to."""
        confidence = int(answers.get("confidence", 5))
        data_quality = answers.get("data_quality", "").lower()
        decision_type = frame.decision_type

        # High confidence + irreversible = slow down, don't route yet
        if confidence >= 8 and frame.reversibility == "irreversible":
            return (
                "none",
                "pre_mortem",
                "High confidence on irreversible decision. Recommend pre-mortem exercise before proceeding."
            )

        # No data + need predictions = DSW for synthetic data
        if "none" in data_quality and decision_type in ["prediction", "estimation"]:
            return (
                "dsw",
                "generate_and_model",
                "No existing data but need predictions. DSW can generate synthetic scenarios."
            )

        # Have decent data + quantitative question = Analyst or DSW
        if "high" in data_quality or "medium" in data_quality:
            if decision_type in ["prediction", "optimization"]:
                return (
                    "dsw",
                    "build_model",
                    "Have data and need predictive capability. DSW can train a model."
                )
            else:
                return (
                    "analyst",
                    "analyze",
                    "Have data and need analysis. Analyst can explore patterns."
                )

        # Need background knowledge
        if decision_type == "diagnosis" or "low" in data_quality:
            return (
                "researcher",
                "research",
                "Need more information before analysis. Researcher can find relevant studies and data."
            )

        # Choice between options without data
        if decision_type == "choice":
            return (
                "none",
                "structured_thinking",
                "For choices, consider pros/cons analysis or decision matrix before adding data complexity."
            )

        # Default to researcher
        return (
            "researcher",
            "research",
            "Starting with research to gather background before determining next steps."
        )

    def to_json(self) -> str:
        """Export the decision brief as JSON."""
        return self.get_brief().to_json()

    def to_dict(self) -> dict:
        """Export the decision brief as a dictionary."""
        return self.get_brief().to_dict()


# =============================================================================
# Quick Bias Check (for use without full interview)
# =============================================================================

def quick_bias_check(text: str) -> dict:
    """
    Quick bias check on any text.

    Can be used to check queries before they go to other agents.
    """
    warnings = detect_biases(text)

    return {
        "has_warnings": len(warnings) > 0,
        "warnings": [
            {
                "type": w.bias_type,
                "description": w.description,
                "evidence": w.evidence,
                "suggestion": w.suggestion,
            }
            for w in warnings
        ],
        "summary": f"Found {len(warnings)} potential cognitive biases." if warnings else "No obvious biases detected.",
    }
