"""
Writer Agent

Document generation with:
- Custom templates (user-defined or built-in)
- Voice matching (adapts to user's writing style)
- Research integration (takes researcher output)
- Quality gates (readability, structure validation)
- Citation management
"""

import json
import re
from dataclasses import dataclass, field
from typing import Literal
from enum import Enum
from pathlib import Path

import sys
sys.path.insert(0, '/home/eric/Desktop/agents')

from core.intent import IntentTracker, Action, AlignmentStatus
from core.sources import ResearchFindings, Source
from tools.readability import ReadabilityScorer, ReadabilityResult
from tools.grammar import GrammarChecker, GrammarResult
from writer.templates import TemplateSpec, TemplateManager, BUILTIN_TEMPLATES, SectionSpec
from writer.voice import VoiceProfile, VoiceManager, VoiceAnalyzer
from reviewer.editor import Editor, EditorResult


@dataclass
class TemplateComplianceResult:
    """Result of template compliance check."""
    compliant: bool = True
    violations: list[str] = field(default_factory=list)
    section_compliance: dict = field(default_factory=dict)  # section -> {compliant, issues}

    def summary(self) -> str:
        status = "✓ COMPLIANT" if self.compliant else "✗ NON-COMPLIANT"
        lines = [f"Template Compliance: {status}"]
        if self.violations:
            lines.append("Violations:")
            for v in self.violations:
                lines.append(f"  - {v}")
        return "\n".join(lines)


@dataclass
class WriterQualityAssessment:
    """Comprehensive quality assessment for generated documents."""
    # Core metrics
    readability: ReadabilityResult = None
    grammar: GrammarResult = None
    template_compliance: TemplateComplianceResult = None

    # Editor results (if run)
    editor_result: EditorResult = None

    # Overall assessment
    overall_grade: str = "A"  # A, B, C, D, F
    passed_gates: bool = True
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate quality summary."""
        lines = [
            "=" * 50,
            f"WRITER QUALITY ASSESSMENT: Grade {self.overall_grade}",
            "=" * 50,
            "",
        ]

        # Readability
        if self.readability:
            lines.extend([
                "## Readability",
                f"  Grade Level: {self.readability.flesch_kincaid_grade:.1f}",
                f"  Reading Level: {self.readability.reading_level}",
                "",
            ])

        # Grammar
        if self.grammar:
            lines.extend([
                "## Grammar & Style",
                f"  Score: {self.grammar.score:.0f}/100",
                f"  Errors: {self.grammar.error_count}",
                f"  Warnings: {self.grammar.warning_count}",
                "",
            ])

        # Template Compliance
        if self.template_compliance:
            status = "✓ Compliant" if self.template_compliance.compliant else "✗ Non-compliant"
            lines.extend([
                "## Template Compliance",
                f"  Status: {status}",
            ])
            if self.template_compliance.violations:
                for v in self.template_compliance.violations[:3]:
                    lines.append(f"  - {v}")
            lines.append("")

        # Editor (if run)
        if self.editor_result:
            lines.extend([
                "## Editorial Review",
                f"  Citation Confidence: {self.editor_result.citation_confidence:.0%}",
                f"  Prompt Alignment: {self.editor_result.prompt_alignment:.0%}",
            ])
            if self.editor_result.gaps:
                lines.append(f"  Unclosed Topics: {len(self.editor_result.gaps)}")
            lines.append("")

        # Issues
        if self.issues:
            lines.extend([
                "## Issues Found",
            ])
            for issue in self.issues[:5]:
                lines.append(f"  ⚠ {issue}")
            lines.append("")

        # Recommendations
        if self.recommendations:
            lines.extend([
                "## Recommendations",
            ])
            for rec in self.recommendations[:3]:
                lines.append(f"  → {rec}")

        lines.append("=" * 50)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "overall_grade": self.overall_grade,
            "passed_gates": self.passed_gates,
            "readability": {
                "grade_level": self.readability.flesch_kincaid_grade if self.readability else None,
                "reading_level": self.readability.reading_level if self.readability else None,
            },
            "grammar": {
                "score": self.grammar.score if self.grammar else None,
                "errors": self.grammar.error_count if self.grammar else 0,
                "warnings": self.grammar.warning_count if self.grammar else 0,
            },
            "template_compliance": {
                "compliant": self.template_compliance.compliant if self.template_compliance else True,
                "violations": self.template_compliance.violations if self.template_compliance else [],
            },
            "editor": {
                "citation_confidence": self.editor_result.citation_confidence if self.editor_result else None,
                "prompt_alignment": self.editor_result.prompt_alignment if self.editor_result else None,
            },
            "issues": self.issues,
            "recommendations": self.recommendations,
        }


def check_template_compliance(
    document: "GeneratedDocument",
    template: TemplateSpec = None,
    request: "DocumentRequest" = None,
) -> TemplateComplianceResult:
    """
    Check if a generated document complies with its template.

    Checks:
    - All required sections present
    - Section word counts within limits
    - Reading level within template constraints
    - Required elements present (citations, examples, etc.)
    """
    result = TemplateComplianceResult()

    if not template:
        return result  # No template = auto-compliant

    section_names = [s["title"].lower() for s in document.sections]

    # Check required sections
    for spec in template.sections:
        if spec.required:
            if spec.name.lower() not in section_names:
                result.compliant = False
                result.violations.append(f"Missing required section: '{spec.name}'")
                result.section_compliance[spec.name] = {
                    "compliant": False,
                    "issues": ["Section missing"],
                }
            else:
                # Find the section and check word count
                section = next(
                    (s for s in document.sections if s["title"].lower() == spec.name.lower()),
                    None,
                )
                if section:
                    word_count = len(section["content"].split())
                    issues = []

                    if spec.min_words and word_count < spec.min_words:
                        issues.append(f"Too short: {word_count} words (min: {spec.min_words})")
                        result.violations.append(
                            f"Section '{spec.name}' too short: {word_count} words (min: {spec.min_words})"
                        )
                        result.compliant = False

                    if spec.max_words and word_count > spec.max_words:
                        issues.append(f"Too long: {word_count} words (max: {spec.max_words})")
                        result.violations.append(
                            f"Section '{spec.name}' too long: {word_count} words (max: {spec.max_words})"
                        )
                        result.compliant = False

                    result.section_compliance[spec.name] = {
                        "compliant": len(issues) == 0,
                        "word_count": word_count,
                        "issues": issues,
                    }

    # Check reading level
    if request and document.readability:
        if request.max_reading_level > 0:
            if document.readability.flesch_kincaid_grade > request.max_reading_level:
                result.compliant = False
                result.violations.append(
                    f"Reading level too high: {document.readability.flesch_kincaid_grade:.1f} "
                    f"(max: {request.max_reading_level})"
                )

        if request.target_reading_level > 0:
            diff = abs(document.readability.flesch_kincaid_grade - request.target_reading_level)
            if diff > 3:  # Allow some variance
                result.violations.append(
                    f"Reading level off target: {document.readability.flesch_kincaid_grade:.1f} "
                    f"(target: {request.target_reading_level}, variance: {diff:.1f})"
                )
                # Don't fail compliance for this, just note it

    # Check total word count
    if request and request.max_words:
        if document.word_count > request.max_words:
            result.compliant = False
            result.violations.append(
                f"Document too long: {document.word_count} words (max: {request.max_words})"
            )

    return result


def assess_document_quality(
    document: "GeneratedDocument",
    template: TemplateSpec = None,
    request: "DocumentRequest" = None,
    run_grammar: bool = True,
) -> WriterQualityAssessment:
    """
    Comprehensive quality assessment for a generated document.

    Args:
        document: The generated document
        template: Template used (for compliance checking)
        request: Original request (for constraints)
        run_grammar: Whether to run grammar check (default True)

    Returns:
        WriterQualityAssessment with all quality metrics
    """
    assessment = WriterQualityAssessment()

    # Readability (already computed in document)
    assessment.readability = document.readability

    # Grammar check
    if run_grammar:
        checker = GrammarChecker()
        assessment.grammar = checker.check(document.content)

        if assessment.grammar.error_count > 5:
            assessment.issues.append(f"Grammar: {assessment.grammar.error_count} errors found")
            assessment.recommendations.append(
                "Review grammar errors, especially confused words and spelling"
            )

    # Template compliance
    assessment.template_compliance = check_template_compliance(document, template, request)
    if not assessment.template_compliance.compliant:
        assessment.issues.extend(assessment.template_compliance.violations)

    # Editor results
    assessment.editor_result = document.editor_result

    if assessment.editor_result:
        if assessment.editor_result.citation_confidence < 0.5:
            assessment.issues.append(
                f"Low citation confidence ({assessment.editor_result.citation_confidence:.0%})"
            )
            assessment.recommendations.append(
                "Review citations - some may be fabricated or unverifiable"
            )

        if assessment.editor_result.prompt_alignment < 0.6:
            assessment.issues.append(
                f"Prompt drift detected ({assessment.editor_result.prompt_alignment:.0%} alignment)"
            )
            assessment.recommendations.append(
                "Document may have drifted from original request - review focus"
            )

        if assessment.editor_result.gaps:
            assessment.issues.append(
                f"{len(assessment.editor_result.gaps)} topics introduced but not addressed"
            )

    # Calculate overall grade
    grade_score = 100

    # Deductions
    if assessment.grammar:
        if assessment.grammar.error_count > 10:
            grade_score -= 20
        elif assessment.grammar.error_count > 5:
            grade_score -= 10
        elif assessment.grammar.error_count > 0:
            grade_score -= 5

    if not assessment.template_compliance.compliant:
        grade_score -= 15

    if assessment.editor_result:
        if assessment.editor_result.citation_confidence < 0.5:
            grade_score -= 15
        elif assessment.editor_result.citation_confidence < 0.7:
            grade_score -= 5

        if assessment.editor_result.prompt_alignment < 0.6:
            grade_score -= 10

    # Map score to grade
    if grade_score >= 90:
        assessment.overall_grade = "A"
    elif grade_score >= 80:
        assessment.overall_grade = "B"
    elif grade_score >= 70:
        assessment.overall_grade = "C"
    elif grade_score >= 60:
        assessment.overall_grade = "D"
    else:
        assessment.overall_grade = "F"

    # Determine if passed gates
    assessment.passed_gates = (
        assessment.overall_grade in ["A", "B", "C"]
        and assessment.template_compliance.compliant
        and (assessment.grammar is None or assessment.grammar.error_count <= 10)
    )

    return assessment


class DocumentType(Enum):
    """Types of documents the writer can generate."""
    GRANT_PROPOSAL = "grant_proposal"
    TECHNICAL_REPORT = "technical_report"
    EXECUTIVE_SUMMARY = "executive_summary"
    BLOG_POST = "blog_post"
    WHITEPAPER = "whitepaper"
    LITERATURE_REVIEW = "literature_review"
    BUSINESS_MEMO = "business_memo"
    PRESS_RELEASE = "press_release"


@dataclass
class DocumentRequest:
    """Request for document generation."""
    topic: str
    doc_type: DocumentType = DocumentType.TECHNICAL_REPORT
    tone: Literal["formal", "casual", "academic", "business"] = "formal"
    length: Literal["brief", "standard", "detailed"] = "standard"
    audience: str = "general"

    # Optional inputs
    research: ResearchFindings = None  # From researcher agent
    notes: list[str] = field(default_factory=list)  # User notes
    outline: list[str] = field(default_factory=list)  # Custom outline

    # Custom template and voice (the key differentiators)
    template: TemplateSpec = None  # User's custom template
    voice: VoiceProfile = None  # User's writing style
    variables: dict = field(default_factory=dict)  # Template variables

    # Constraints
    max_words: int = None
    required_sections: list[str] = field(default_factory=list)
    avoid_topics: list[str] = field(default_factory=list)

    # Quality gates
    target_reading_level: float = 0  # 0 = no target
    max_reading_level: float = 0


@dataclass
class GeneratedDocument:
    """Result of document generation."""
    title: str
    content: str
    sections: list[dict]  # {title, content}
    citations: list[str]
    word_count: int
    doc_type: DocumentType
    metadata: dict = field(default_factory=dict)

    # Quality metrics
    readability: ReadabilityResult = None
    quality_passed: bool = True
    quality_issues: list[str] = field(default_factory=list)

    # Editor output (if run_editor=True)
    editor_result: EditorResult = None
    original_prompt: str = ""  # Stored for drift checking

    # Comprehensive quality assessment (if run_quality_gates=True)
    quality_assessment: "WriterQualityAssessment" = None

    def to_markdown(self, include_editor_report: bool = True) -> str:
        """Export as markdown."""
        lines = [f"# {self.title}", ""]

        for section in self.sections:
            lines.append(f"## {section['title']}")
            lines.append("")
            lines.append(section['content'])
            lines.append("")

        if self.citations:
            lines.append("## References")
            lines.append("")
            for i, cite in enumerate(self.citations, 1):
                lines.append(f"{i}. {cite}")

        # Include editor report if available
        if include_editor_report and self.editor_result:
            lines.append("")
            lines.append("---")
            lines.append("")
            lines.append(self.editor_result.editorial_report)

        return "\n".join(lines)

    def get_edited_document(self) -> str:
        """Get the cleaned document after editing (if available)."""
        if self.editor_result:
            return self.editor_result.cleaned_document
        return self.content

    def quality_report(self) -> str:
        """Generate quality report."""
        lines = ["## Quality Report", ""]

        if self.readability:
            lines.append(f"- Reading Level: {self.readability.reading_level}")
            lines.append(f"- Flesch-Kincaid Grade: {self.readability.flesch_kincaid_grade:.1f}")
            lines.append(f"- Word Count: {self.word_count}")

        # Add editor metrics if available
        if self.editor_result:
            lines.append(f"- Citation Confidence: {self.editor_result.citation_confidence:.0%}")
            lines.append(f"- Prompt Alignment: {self.editor_result.prompt_alignment:.0%}")
            lines.append(f"- Original Grade: {self.editor_result.original_grade}")
            lines.append(f"- After Cleanup: {self.editor_result.improved_grade}")

        if self.quality_issues:
            lines.append("")
            lines.append("### Writer Issues")
            for issue in self.quality_issues:
                lines.append(f"- {issue}")

        # Add editor findings
        if self.editor_result:
            if self.editor_result.citation_concerns:
                lines.append("")
                lines.append(f"### Citation Concerns ({len(self.editor_result.citation_concerns)})")
                for concern in self.editor_result.citation_concerns[:3]:
                    lines.append(f"- {concern.citation_text[:50]}... ({', '.join(concern.issues[:2])})")

            stat_reps = [r for r in self.editor_result.repetitions if r.issue_type == "statistic"]
            if stat_reps:
                lines.append("")
                lines.append(f"### Repeated Statistics ({len(stat_reps)})")
                for rep in stat_reps[:3]:
                    lines.append(f"- \"{rep.text}\" x{rep.count}")

            if self.editor_result.gaps:
                lines.append("")
                lines.append(f"### Unclosed Topics ({len(self.editor_result.gaps)})")
                for gap in self.editor_result.gaps[:3]:
                    lines.append(f"- {gap.topic[:40]}...")

        if not self.quality_issues and (not self.editor_result or
           (self.editor_result.citation_confidence >= 0.7 and not self.editor_result.gaps)):
            lines.append("")
            lines.append("All quality checks passed.")

        return "\n".join(lines)


class WriterAgent:
    """
    Document generation agent.

    Takes research, notes, or just a topic and generates
    structured documents with proper citations.
    """

    # Document templates define structure
    TEMPLATES = {
        DocumentType.GRANT_PROPOSAL: {
            "sections": [
                "Executive Summary",
                "Problem Statement",
                "Proposed Solution",
                "Methodology",
                "Timeline & Milestones",
                "Budget Justification",
                "Expected Outcomes",
                "Qualifications",
            ],
            "tone": "formal",
            "citation_style": "numbered",
        },
        DocumentType.TECHNICAL_REPORT: {
            "sections": [
                "Abstract",
                "Introduction",
                "Background",
                "Methods",
                "Results",
                "Discussion",
                "Conclusion",
            ],
            "tone": "academic",
            "citation_style": "numbered",
        },
        DocumentType.EXECUTIVE_SUMMARY: {
            "sections": [
                "Overview",
                "Key Findings",
                "Recommendations",
                "Next Steps",
            ],
            "tone": "business",
            "citation_style": "inline",
        },
        DocumentType.BLOG_POST: {
            "sections": [
                "Introduction",
                "Main Points",
                "Conclusion",
            ],
            "tone": "casual",
            "citation_style": "links",
        },
        DocumentType.WHITEPAPER: {
            "sections": [
                "Executive Summary",
                "Introduction",
                "Problem Analysis",
                "Solution Overview",
                "Technical Details",
                "Implementation",
                "Case Studies",
                "Conclusion",
            ],
            "tone": "formal",
            "citation_style": "numbered",
        },
        DocumentType.LITERATURE_REVIEW: {
            "sections": [
                "Introduction",
                "Search Methodology",
                "Thematic Analysis",
                "Key Findings",
                "Gaps in Literature",
                "Conclusion",
            ],
            "tone": "academic",
            "citation_style": "numbered",
        },
        DocumentType.BUSINESS_MEMO: {
            "sections": [
                "Purpose",
                "Background",
                "Key Points",
                "Recommendations",
                "Action Items",
            ],
            "tone": "business",
            "citation_style": "none",
        },
        DocumentType.PRESS_RELEASE: {
            "sections": [
                "Headline",
                "Lead Paragraph",
                "Body",
                "Quote",
                "Boilerplate",
                "Contact Information",
            ],
            "tone": "formal",
            "citation_style": "none",
        },
    }

    # Length targets (approximate words per section)
    LENGTH_TARGETS = {
        "brief": 100,
        "standard": 250,
        "detailed": 500,
    }

    SYSTEM_PROMPT = """You are a professional document writer. Your job is to create well-structured, clear documents.

Rules:
1. Follow the document structure provided
2. Maintain consistent tone throughout
3. Cite sources properly using [1], [2] format
4. Be concise but thorough
5. Use clear, professional language
6. Include specific details and examples where appropriate

Epistemic tone (critical):
- Use scientific hedging: "suggests", "indicates", "may", "appears to", "the evidence points to"
- Avoid absolutes: never say "proves", "definitely", "always", "never" unless truly warranted
- Acknowledge uncertainty: "further research is needed", "limitations include", "current evidence suggests"
- Stay calm and measured: no sensational language, no alarming phrasing, no hyperbole
- Present findings objectively: "studies have found" not "studies have shockingly revealed"
- Distinguish correlation from causation: "is associated with" not "causes" unless proven
- Note study limitations and conflicting evidence where relevant

Output each section with a clear heading."""

    def __init__(self, llm=None, storage_dir: Path = None):
        self.llm = llm
        self.intent_tracker = IntentTracker(llm=llm)
        self.readability = ReadabilityScorer()
        self.template_manager = TemplateManager(storage_dir / "templates" if storage_dir else None)
        self.voice_manager = VoiceManager(storage_dir / "voices" if storage_dir else None)

    def _llm_generate(self, prompt: str, max_tokens: int = 2000) -> str:
        """Generate text from LLM."""
        if self.llm is None:
            return ""
        if hasattr(self.llm, 'generate'):
            return self.llm.generate(prompt, max_tokens=max_tokens)
        elif hasattr(self.llm, 'complete'):
            return self.llm.complete(prompt, max_tokens=max_tokens)
        return ""

    def write(self, request: DocumentRequest, run_editor: bool = True,
              original_prompt: str = "", run_quality_gates: bool = True) -> GeneratedDocument:
        """
        Generate a document based on the request.

        Args:
            request: Document generation request
            run_editor: Whether to auto-run the Editor agent on output (default True)
            original_prompt: Original user prompt for drift checking
            run_quality_gates: Whether to run full quality assessment (default True)

        Returns:
            GeneratedDocument with editor_result and quality_assessment if enabled
        """
        # Set intent
        self.intent_tracker.set_intent(
            raw_input=f"Write {request.doc_type.value} about {request.topic}",
            parsed_goal=f"Generate {request.doc_type.value}: {request.topic}",
            constraints=[
                f"Maintain {request.tone} tone",
                f"Target audience: {request.audience}",
            ] + [f"Avoid: {t}" for t in request.avoid_topics],
        )

        # Get template - prefer custom, fall back to built-in
        if request.template:
            custom_template = request.template
            sections = [s.name for s in custom_template.sections if s.required]
            section_specs = {s.name: s for s in custom_template.sections}
        else:
            custom_template = None
            template = self.TEMPLATES.get(request.doc_type, self.TEMPLATES[DocumentType.TECHNICAL_REPORT])
            section_specs = {}

            # Determine sections
            if request.outline:
                sections = request.outline
            elif request.required_sections:
                sections = request.required_sections
            else:
                sections = template["sections"]

        # Gather source material
        source_material = self._gather_sources(request)
        citations = self._extract_citations(request)

        # Build voice instructions
        voice_instructions = ""
        if request.voice:
            voice_instructions = request.voice.to_prompt_instructions()

        # Generate each section
        generated_sections = []
        for section_title in sections:
            spec = section_specs.get(section_title)
            content = self._generate_section(
                request, section_title, source_material,
                custom_template, spec, voice_instructions
            )
            generated_sections.append({
                "title": section_title,
                "content": content,
            })

        # Build document
        full_content = "\n\n".join(
            f"## {s['title']}\n\n{s['content']}" for s in generated_sections
        )
        word_count = len(full_content.split())

        # Generate title
        title = self._generate_title(request)

        # Quality check
        readability_result = self.readability.analyze(full_content)
        quality_issues = []
        quality_passed = True

        # Check reading level
        if request.max_reading_level > 0:
            if readability_result.flesch_kincaid_grade > request.max_reading_level:
                quality_issues.append(
                    f"Reading level too high: {readability_result.flesch_kincaid_grade:.1f} "
                    f"(max: {request.max_reading_level})"
                )
                quality_passed = False

        if request.target_reading_level > 0:
            diff = abs(readability_result.flesch_kincaid_grade - request.target_reading_level)
            if diff > 2:
                quality_issues.append(
                    f"Reading level off target: {readability_result.flesch_kincaid_grade:.1f} "
                    f"(target: {request.target_reading_level})"
                )

        # Check section word counts if using custom template
        if custom_template:
            for section in generated_sections:
                spec = section_specs.get(section['title'])
                if spec:
                    words = len(section['content'].split())
                    if spec.min_words and words < spec.min_words:
                        quality_issues.append(
                            f"Section '{section['title']}' too short: {words} words (min: {spec.min_words})"
                        )
                    if spec.max_words and words > spec.max_words:
                        quality_issues.append(
                            f"Section '{section['title']}' too long: {words} words (max: {spec.max_words})"
                        )

        # Record action
        action = Action(
            id="write_complete",
            description=f"Generated {request.doc_type.value}",
            action_type="writing",
            content=title,
            alignment_score=1.0 if quality_passed else 0.8,
            reasoning=f"Generated {len(sections)} sections, {word_count} words, "
                      f"grade {readability_result.flesch_kincaid_grade:.1f}",
        )
        self.intent_tracker.record_action(action)

        # Run Editor if requested
        editor_result = None
        prompt_for_drift = original_prompt or f"Write a {request.doc_type.value} about {request.topic}"

        if run_editor:
            editor = Editor()

            # Determine rubric type from document type
            rubric_map = {
                DocumentType.WHITEPAPER: "whitepaper",
                DocumentType.TECHNICAL_REPORT: "research",
                DocumentType.LITERATURE_REVIEW: "research",
                DocumentType.GRANT_PROPOSAL: "research",
                DocumentType.EXECUTIVE_SUMMARY: "essay",
                DocumentType.BLOG_POST: "essay",
                DocumentType.BUSINESS_MEMO: "essay",
                DocumentType.PRESS_RELEASE: "essay",
            }
            rubric_type = rubric_map.get(request.doc_type, "auto")

            editor_result = editor.edit(
                document=full_content,
                title=title,
                rubric_type=rubric_type,
                prompt=prompt_for_drift,
            )

            # Add editor findings to quality issues
            if editor_result.citation_confidence < 0.5:
                quality_issues.append(f"Citation concerns: {len(editor_result.citation_concerns)} suspicious references")
                quality_passed = False

            stat_reps = [r for r in editor_result.repetitions if r.issue_type == "statistic"]
            if stat_reps:
                quality_issues.append(f"Repeated statistics: {len(stat_reps)} stats appear multiple times")

            if editor_result.gaps:
                quality_issues.append(f"Unclosed topics: {len(editor_result.gaps)} topics introduced but not addressed")

            if editor_result.prompt_alignment < 0.6:
                quality_issues.append(f"Prompt drift: only {editor_result.prompt_alignment:.0%} alignment with original request")
                quality_passed = False

        # Build document first
        document = GeneratedDocument(
            title=title,
            content=full_content,
            sections=generated_sections,
            citations=citations,
            word_count=word_count,
            doc_type=request.doc_type,
            readability=readability_result,
            quality_passed=quality_passed,
            quality_issues=quality_issues,
            editor_result=editor_result,
            original_prompt=prompt_for_drift,
            metadata={
                "tone": request.tone,
                "audience": request.audience,
                "research_sources": len(request.research.sources) if request.research else 0,
                "voice_profile": request.voice.name if request.voice else None,
                "template": request.template.name if request.template else None,
                "editor_ran": run_editor,
                "quality_gates_ran": run_quality_gates,
            },
        )

        # Run comprehensive quality assessment if requested
        if run_quality_gates:
            document.quality_assessment = assess_document_quality(
                document=document,
                template=request.template,
                request=request,
                run_grammar=True,
            )
            # Update quality_passed based on assessment
            document.quality_passed = document.quality_assessment.passed_gates
            # Merge issues
            document.quality_issues = list(set(document.quality_issues + document.quality_assessment.issues))

        return document

    def _gather_sources(self, request: DocumentRequest) -> str:
        """Gather source material for writing."""
        parts = []

        # From research
        if request.research:
            parts.append("## Research Findings\n")
            parts.append(request.research.summary)
            parts.append("\n")

            for section in request.research.sections:
                parts.append(f"### {section.get('title', 'Findings')}\n")
                parts.append(section.get('content', ''))
                parts.append("\n")

        # From notes
        if request.notes:
            parts.append("## Notes\n")
            for note in request.notes:
                parts.append(f"- {note}\n")

        return "\n".join(parts) if parts else f"Topic: {request.topic}"

    def _extract_citations(self, request: DocumentRequest) -> list[str]:
        """Extract citations from research."""
        citations = []

        if request.research:
            for source in request.research.sources:
                cite = source.cite('footnote')
                citations.append(cite)

        return citations

    def _generate_section(self, request: DocumentRequest, section_title: str,
                          source_material: str, template: TemplateSpec = None,
                          section_spec: SectionSpec = None, voice_instructions: str = "") -> str:
        """Generate a single section."""
        if self.llm is None:
            return self._mock_section(request, section_title)

        # Determine target words
        if section_spec and section_spec.min_words:
            target_words = (section_spec.min_words + (section_spec.max_words or section_spec.min_words * 2)) // 2
        else:
            target_words = self.LENGTH_TARGETS.get(request.length, 250)

        # Build section description
        section_desc = ""
        if section_spec:
            if section_spec.description:
                section_desc = f"\nSection purpose: {section_spec.description}"
            if section_spec.example:
                section_desc += f"\n\nExample of this section:\n{section_spec.example[:500]}"

        prompt = f"""{self.SYSTEM_PROMPT}

{voice_instructions}

Document type: {request.doc_type.value}
Topic: {request.topic}
Tone: {template.tone if template else request.tone}
Target audience: {request.audience}
Target length: ~{target_words} words
{section_desc}

Source material:
{source_material[:3000]}

Write the "{section_title}" section. Be specific and use citations [1], [2] where appropriate.

{section_title}:"""

        content = self._llm_generate(prompt, max_tokens=target_words * 2)
        return content.strip()

    def _mock_section(self, request: DocumentRequest, section_title: str) -> str:
        """Generate mock section content for testing."""
        topic = request.topic
        tone = request.tone

        mock_content = {
            "Executive Summary": f"This {request.doc_type.value} examines {topic}. Our analysis reveals significant opportunities and challenges in this domain. Key recommendations are outlined below.",
            "Introduction": f"The field of {topic} has seen substantial development in recent years. This document provides a comprehensive overview of the current state and future directions.",
            "Problem Statement": f"Despite advances in {topic}, several critical challenges remain. These include scalability, cost-effectiveness, and broader adoption barriers.",
            "Background": f"Understanding {topic} requires context on its historical development and theoretical foundations. This section provides that background.",
            "Methods": f"Our approach to analyzing {topic} combines quantitative data analysis with qualitative assessment of industry trends and expert perspectives.",
            "Key Findings": f"Analysis of {topic} reveals three primary findings: (1) Market growth is accelerating, (2) Technology maturity varies by segment, (3) Regulatory frameworks are evolving.",
            "Recommendations": f"Based on our analysis of {topic}, we recommend: prioritizing core capabilities, building strategic partnerships, and investing in talent development.",
            "Conclusion": f"This analysis of {topic} demonstrates both the opportunities and challenges ahead. Success will require strategic focus and sustained investment.",
            "Overview": f"{topic} represents a significant area of interest. This summary captures the essential points for decision-makers.",
            "Next Steps": f"To advance on {topic}, the following actions are recommended: (1) Conduct detailed feasibility study, (2) Engage stakeholders, (3) Develop implementation timeline.",
        }

        # Default content if section not in mock
        default = f"This section addresses {section_title.lower()} as it relates to {topic}. Further analysis and specific details would be developed based on available research and stakeholder input."

        return mock_content.get(section_title, default)

    def _generate_title(self, request: DocumentRequest) -> str:
        """Generate document title."""
        type_prefixes = {
            DocumentType.GRANT_PROPOSAL: "Proposal:",
            DocumentType.TECHNICAL_REPORT: "Technical Report:",
            DocumentType.EXECUTIVE_SUMMARY: "Executive Summary:",
            DocumentType.BLOG_POST: "",
            DocumentType.WHITEPAPER: "Whitepaper:",
            DocumentType.LITERATURE_REVIEW: "Literature Review:",
            DocumentType.BUSINESS_MEMO: "Memo:",
            DocumentType.PRESS_RELEASE: "Press Release:",
        }

        prefix = type_prefixes.get(request.doc_type, "")
        if prefix:
            return f"{prefix} {request.topic}"
        return request.topic


# Convenience function for chaining with researcher
def write_from_research(research: ResearchFindings,
                        doc_type: DocumentType = DocumentType.TECHNICAL_REPORT,
                        llm=None,
                        run_editor: bool = True) -> GeneratedDocument:
    """Convenience function to write a document from research findings."""
    agent = WriterAgent(llm=llm)
    request = DocumentRequest(
        topic=research.query,
        doc_type=doc_type,
        research=research,
    )
    return agent.write(request, run_editor=run_editor,
                       original_prompt=f"Research and write about: {research.query}")


def write_with_style(topic: str,
                     template_name: str = None,
                     voice_name: str = None,
                     voice_samples: list[str] = None,
                     variables: dict = None,
                     llm=None,
                     run_editor: bool = True,
                     original_prompt: str = "") -> GeneratedDocument:
    """
    Write a document with custom template and/or voice.

    Args:
        topic: What to write about
        template_name: Name of saved template or built-in (e.g., "grant_proposal")
        voice_name: Name of saved voice profile
        voice_samples: Text samples to create voice profile on-the-fly
        variables: Variables to fill in template (e.g., {"company_name": "Acme"})
        llm: LLM to use for generation
        run_editor: Whether to auto-run Editor on output (default True)
        original_prompt: Original user prompt for drift checking

    Example:
        # Use built-in template with on-the-fly voice matching
        doc = write_with_style(
            topic="AI in Healthcare",
            template_name="grant_proposal",
            voice_samples=[user_previous_writing],
            variables={"budget": "$500,000", "duration": "2 years"}
        )
    """
    agent = WriterAgent(llm=llm)

    # Load or create template
    template = None
    if template_name:
        # Try built-in first
        template = BUILTIN_TEMPLATES.get(template_name.lower().replace(" ", "_"))
        if not template:
            # Try saved templates
            template = agent.template_manager.load(template_name)

    # Load or create voice profile
    voice = None
    if voice_name:
        voice = agent.voice_manager.load(voice_name)
    elif voice_samples:
        analyzer = VoiceAnalyzer()
        voice = analyzer.analyze(voice_samples, "on_the_fly")

    # Determine doc type from template or default
    doc_type = DocumentType.TECHNICAL_REPORT
    if template:
        domain = template.domain.lower()
        for dt in DocumentType:
            if dt.value == domain:
                doc_type = dt
                break

    request = DocumentRequest(
        topic=topic,
        doc_type=doc_type,
        template=template,
        voice=voice,
        variables=variables or {},
        target_reading_level=template.target_reading_level if template else 0,
        max_reading_level=template.max_reading_level if template else 0,
    )

    prompt = original_prompt or f"Write a {doc_type.value} about {topic}"
    return agent.write(request, run_editor=run_editor, original_prompt=prompt)


def quick_write(topic: str, doc_type: str = "whitepaper",
                run_editor: bool = True, llm=None) -> GeneratedDocument:
    """
    Quick one-liner document generation with Editor.

    Args:
        topic: What to write about
        doc_type: Type of document (whitepaper, technical_report, executive_summary, etc.)
        run_editor: Whether to run Editor and include report (default True)
        llm: LLM for generation

    Returns:
        GeneratedDocument with editor_result included

    Example:
        from writer import quick_write

        doc = quick_write("Synthetic Data for AI Training", doc_type="whitepaper")
        print(doc.to_markdown())  # Includes editorial report at the end
        print(doc.quality_report())  # Shows citation confidence, repeated stats, etc.
    """
    agent = WriterAgent(llm=llm)

    # Map string to DocumentType
    type_map = {
        "whitepaper": DocumentType.WHITEPAPER,
        "technical_report": DocumentType.TECHNICAL_REPORT,
        "executive_summary": DocumentType.EXECUTIVE_SUMMARY,
        "blog_post": DocumentType.BLOG_POST,
        "grant_proposal": DocumentType.GRANT_PROPOSAL,
        "literature_review": DocumentType.LITERATURE_REVIEW,
        "business_memo": DocumentType.BUSINESS_MEMO,
        "press_release": DocumentType.PRESS_RELEASE,
    }
    dtype = type_map.get(doc_type.lower(), DocumentType.WHITEPAPER)

    request = DocumentRequest(topic=topic, doc_type=dtype)
    return agent.write(request, run_editor=run_editor,
                       original_prompt=f"Write a {doc_type} about {topic}")
