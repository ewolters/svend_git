"""
Guide Agent

Interview-style agent that asks structured questions and builds
output from user responses. Used for guided workflows like:
- Business plan creation
- Requirements gathering
- Onboarding questionnaires
- Assessment forms

The Guide inverts the typical agent pattern: instead of answering,
it asks questions and synthesizes responses.
"""

import json
from dataclasses import dataclass, field
from typing import Callable, Any
from enum import Enum

import sys
sys.path.insert(0, '/home/eric/Desktop/agents')

from core.intent import IntentTracker, Action


class QuestionType(Enum):
    """Types of questions."""
    TEXT = "text"           # Free text response
    CHOICE = "choice"       # Single choice from options
    MULTI = "multi"         # Multiple choice
    NUMBER = "number"       # Numeric input
    YESNO = "yesno"         # Yes/No
    SCALE = "scale"         # 1-10 rating


@dataclass
class Question:
    """A question in the interview flow."""
    id: str
    text: str
    question_type: QuestionType = QuestionType.TEXT
    options: list[str] = field(default_factory=list)  # For choice/multi
    required: bool = True
    help_text: str = ""  # Additional context
    validation: Callable[[str], bool] = None  # Custom validation
    follow_up: dict[str, str] = field(default_factory=dict)  # answer -> follow_up question id

    # For scale type
    scale_min: int = 1
    scale_max: int = 10


@dataclass
class Section:
    """A section of related questions."""
    id: str
    title: str
    description: str
    questions: list[Question]


@dataclass
class InterviewState:
    """Current state of an interview session."""
    current_section: int = 0
    current_question: int = 0
    answers: dict[str, Any] = field(default_factory=dict)
    completed: bool = False
    skipped: list[str] = field(default_factory=list)


@dataclass
class InterviewResult:
    """Result of a completed interview."""
    answers: dict[str, Any]
    sections_completed: int
    questions_answered: int
    questions_skipped: int
    synthesized_output: dict = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Export as markdown."""
        lines = ["# Interview Results", ""]

        for key, value in self.synthesized_output.items():
            lines.append(f"## {key}")
            lines.append("")
            if isinstance(value, list):
                for item in value:
                    lines.append(f"- {item}")
            elif isinstance(value, dict):
                for k, v in value.items():
                    lines.append(f"**{k}:** {v}")
            else:
                lines.append(str(value))
            lines.append("")

        return "\n".join(lines)


class GuideAgent:
    """
    Base class for interview-style guided agents.

    Subclasses define:
    - sections: List of Section objects with questions
    - synthesize(): How to turn answers into structured output

    The flow:
    1. get_next_question() returns the next question to ask
    2. submit_answer() records the answer and advances
    3. When done, synthesize() creates structured output
    """

    def __init__(self, llm=None):
        self.llm = llm
        self.sections: list[Section] = []
        self.state = InterviewState()
        self.intent_tracker = IntentTracker(llm=llm)

    def _llm_generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate text from LLM."""
        if self.llm is None:
            return ""
        if hasattr(self.llm, 'generate'):
            return self.llm.generate(prompt, max_tokens=max_tokens)
        elif hasattr(self.llm, 'complete'):
            return self.llm.complete(prompt, max_tokens=max_tokens)
        return ""

    def start(self) -> Question:
        """Start the interview and return first question."""
        self.state = InterviewState()
        self.intent_tracker.set_intent(
            raw_input="Guided interview session",
            parsed_goal="Complete structured questionnaire",
            constraints=["Answer all required questions"],
        )
        return self.get_current_question()

    def get_current_question(self) -> Question | None:
        """Get the current question."""
        if self.state.completed:
            return None

        if self.state.current_section >= len(self.sections):
            self.state.completed = True
            return None

        section = self.sections[self.state.current_section]
        if self.state.current_question >= len(section.questions):
            # Move to next section
            self.state.current_section += 1
            self.state.current_question = 0
            return self.get_current_question()

        return section.questions[self.state.current_question]

    def get_current_section(self) -> Section | None:
        """Get the current section."""
        if self.state.current_section >= len(self.sections):
            return None
        return self.sections[self.state.current_section]

    def submit_answer(self, answer: Any) -> tuple[bool, str, Question | None]:
        """
        Submit an answer to the current question.

        Returns:
            (success, message, next_question)
        """
        question = self.get_current_question()
        if question is None:
            return False, "No current question", None

        # Validate answer
        valid, message = self._validate_answer(question, answer)
        if not valid:
            return False, message, question

        # Store answer
        self.state.answers[question.id] = answer

        # Record action
        action = Action(
            id=f"answer_{question.id}",
            description=f"Answered: {question.id}",
            action_type="interview",
            content=str(answer)[:100],
            alignment_score=1.0,
        )
        self.intent_tracker.record_action(action)

        # Check for follow-up questions based on answer
        if question.follow_up and str(answer) in question.follow_up:
            follow_up_id = question.follow_up[str(answer)]
            # Insert follow-up question next in the flow
            self._insert_follow_up(follow_up_id)

        # Advance to next question
        self.state.current_question += 1
        next_q = self.get_current_question()

        return True, "Answer recorded", next_q

    def skip_question(self) -> tuple[bool, str, Question | None]:
        """Skip the current question if not required."""
        question = self.get_current_question()
        if question is None:
            return False, "No current question", None

        if question.required:
            return False, "This question is required", question

        self.state.skipped.append(question.id)
        self.state.current_question += 1
        next_q = self.get_current_question()

        return True, "Question skipped", next_q

    def _validate_answer(self, question: Question, answer: Any) -> tuple[bool, str]:
        """Validate an answer against question requirements."""
        # Empty check
        if question.required and (answer is None or answer == ""):
            return False, "This question requires an answer"

        # Type-specific validation
        if question.question_type == QuestionType.NUMBER:
            try:
                float(answer)
            except (ValueError, TypeError):
                return False, "Please enter a valid number"

        if question.question_type == QuestionType.CHOICE:
            if answer not in question.options:
                return False, f"Please choose from: {', '.join(question.options)}"

        if question.question_type == QuestionType.YESNO:
            if str(answer).lower() not in ['yes', 'no', 'y', 'n', 'true', 'false']:
                return False, "Please answer Yes or No"

        if question.question_type == QuestionType.SCALE:
            try:
                val = int(answer)
                if val < question.scale_min or val > question.scale_max:
                    return False, f"Please enter a number between {question.scale_min} and {question.scale_max}"
            except (ValueError, TypeError):
                return False, "Please enter a valid number"

        # Custom validation
        if question.validation:
            if not question.validation(answer):
                return False, "Invalid answer format"

        return True, "OK"

    def _insert_follow_up(self, follow_up_id: str) -> None:
        """Insert a follow-up question into the flow."""
        # Find the follow-up question in all sections
        for section in self.sections:
            for i, q in enumerate(section.questions):
                if q.id == follow_up_id:
                    # Insert at current position (will be asked next)
                    current_section = self.sections[self.state.current_section]
                    # Create a copy to avoid modifying the template
                    follow_up_q = Question(
                        id=f"{q.id}_followup",
                        text=q.text,
                        question_type=q.question_type,
                        options=q.options.copy() if q.options else [],
                        required=q.required,
                        help_text=q.help_text,
                        validation=q.validation,
                    )
                    current_section.questions.insert(
                        self.state.current_question,
                        follow_up_q
                    )
                    return

    def get_progress(self) -> dict:
        """Get current progress through the interview."""
        total_questions = sum(len(s.questions) for s in self.sections)
        answered = len(self.state.answers)
        skipped = len(self.state.skipped)

        return {
            "total_sections": len(self.sections),
            "current_section": self.state.current_section + 1,
            "total_questions": total_questions,
            "answered": answered,
            "skipped": skipped,
            "remaining": total_questions - answered - skipped,
            "percent_complete": (answered + skipped) / total_questions * 100 if total_questions > 0 else 0,
        }

    def is_complete(self) -> bool:
        """Check if interview is complete."""
        return self.state.completed

    def synthesize(self) -> InterviewResult:
        """
        Synthesize answers into structured output.

        Override in subclasses for custom synthesis logic.
        """
        progress = self.get_progress()

        return InterviewResult(
            answers=self.state.answers.copy(),
            sections_completed=self.state.current_section,
            questions_answered=progress["answered"],
            questions_skipped=progress["skipped"],
            synthesized_output=self._default_synthesis(),
        )

    def _default_synthesis(self) -> dict:
        """Default synthesis - group answers by section."""
        output = {}

        for section in self.sections:
            section_answers = {}
            for q in section.questions:
                if q.id in self.state.answers:
                    section_answers[q.text] = self.state.answers[q.id]
            if section_answers:
                output[section.title] = section_answers

        return output

    def format_question(self, question: Question) -> str:
        """Format a question for display."""
        lines = [question.text]

        if question.help_text:
            lines.append(f"  ({question.help_text})")

        if question.question_type == QuestionType.CHOICE:
            for i, opt in enumerate(question.options, 1):
                lines.append(f"  {i}. {opt}")

        if question.question_type == QuestionType.MULTI:
            lines.append("  (Select all that apply)")
            for i, opt in enumerate(question.options, 1):
                lines.append(f"  {i}. {opt}")

        if question.question_type == QuestionType.SCALE:
            lines.append(f"  (Rate from {question.scale_min} to {question.scale_max})")

        if question.question_type == QuestionType.YESNO:
            lines.append("  (Yes/No)")

        if not question.required:
            lines.append("  [Optional - press Enter to skip]")

        return "\n".join(lines)
