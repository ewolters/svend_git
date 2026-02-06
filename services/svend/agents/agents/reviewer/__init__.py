"""
Document Reviewer & Editor Agents

Includes:
- DocumentReviewer: General-purpose review with checklists
- RubricGrader: Grade documents against custom rubrics with citation verification
- Editor: Clean up documents and produce editorial reports

Usage:
    # Basic review
    from reviewer import DocumentReviewer
    reviewer = DocumentReviewer()
    result = reviewer.review(document, doc_type="technical")

    # Rubric grading with citation check
    from reviewer import RubricGrader, WHITEPAPER_RUBRIC, quick_grade
    grader = RubricGrader(WHITEPAPER_RUBRIC)
    result = grader.grade(document)
    print(result.summary())

    # Quick grade one-liner
    result = quick_grade(document, WHITEPAPER_RUBRIC)

    # Editor agent (cleanup + report)
    from reviewer import Editor, quick_edit
    editor = Editor()
    result = editor.edit(document)
    print(result.cleaned_document)
    print(result.editorial_report)
"""

from .agent import (
    DocumentReviewer,
    ReviewResult,
    Finding,
    Severity,
    ReviewDimension,
    ChecklistItem,
)

from .rubric import (
    RubricGrader,
    Rubric,
    Criterion,
    GradingResult,
    CitationCheck,
    quick_grade,
    # Built-in rubrics
    ESSAY_RUBRIC,
    RESEARCH_PAPER_RUBRIC,
    WHITEPAPER_RUBRIC,
)

from .editor import (
    Editor,
    EditorResult,
    EditSuggestion,
    RepetitionIssue,
    quick_edit,
)

__all__ = [
    # Document Reviewer
    "DocumentReviewer",
    "ReviewResult",
    "Finding",
    "Severity",
    "ReviewDimension",
    "ChecklistItem",
    # Rubric Grader
    "RubricGrader",
    "Rubric",
    "Criterion",
    "GradingResult",
    "CitationCheck",
    "quick_grade",
    "ESSAY_RUBRIC",
    "RESEARCH_PAPER_RUBRIC",
    "WHITEPAPER_RUBRIC",
    # Editor
    "Editor",
    "EditorResult",
    "EditSuggestion",
    "RepetitionIssue",
    "quick_edit",
]
