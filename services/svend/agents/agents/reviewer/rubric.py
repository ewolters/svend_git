"""
Rubric-Based Document Grader

Grade essays and documents against a provided rubric with:
- Criterion-by-criterion scoring
- Grammar and spelling checks
- Logical consistency detection
- Citation/reference verification (checks for hallucinated sources)
- Detailed feedback per criterion

Usage:
    from reviewer import RubricGrader, Rubric, Criterion

    rubric = Rubric(
        name="Essay Rubric",
        criteria=[
            Criterion("thesis", "Clear thesis statement", max_points=20),
            Criterion("evidence", "Supporting evidence with citations", max_points=25),
            Criterion("organization", "Logical structure and flow", max_points=20),
            Criterion("grammar", "Grammar and mechanics", max_points=15),
            Criterion("citations", "Proper citation format", max_points=20),
        ]
    )

    grader = RubricGrader(rubric)
    result = grader.grade(document)

    print(result.summary())
    print(f"Total: {result.total_score}/{result.max_score}")
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


@dataclass
class Criterion:
    """A single rubric criterion."""
    id: str
    description: str
    max_points: int
    weight: float = 1.0  # Optional weight multiplier

    # Optional: scoring levels
    levels: Optional[dict] = None  # {points: description}


@dataclass
class Rubric:
    """A complete grading rubric."""
    name: str
    criteria: list[Criterion]
    total_points: int = 0

    def __post_init__(self):
        if self.total_points == 0:
            self.total_points = sum(c.max_points for c in self.criteria)


@dataclass
class CriterionScore:
    """Score for a single criterion."""
    criterion_id: str
    criterion_desc: str
    points_earned: float
    max_points: int
    percentage: float
    feedback: str
    evidence: list[str] = field(default_factory=list)  # Quotes from doc supporting score


@dataclass
class CitationCheck:
    """Result of checking a single citation."""
    citation_text: str
    appears_valid: bool
    confidence: float  # 0-1
    issues: list[str] = field(default_factory=list)
    location: str = ""


@dataclass
class GradingResult:
    """Complete grading result."""
    document_title: str
    rubric_name: str

    # Scores
    criterion_scores: list[CriterionScore]
    total_score: float
    max_score: int
    percentage: float
    letter_grade: str

    # Additional checks
    grammar_issues: list[dict] = field(default_factory=list)
    logic_issues: list[dict] = field(default_factory=list)
    citation_checks: list[CitationCheck] = field(default_factory=list)

    # Flags
    has_citation_concerns: bool = False
    citation_confidence: float = 1.0  # Overall confidence citations are real

    def summary(self) -> str:
        """Generate summary report."""
        lines = [
            f"# Grading Report: {self.document_title}",
            f"**Rubric:** {self.rubric_name}",
            "",
            f"## Overall Score: {self.total_score:.1f}/{self.max_score} ({self.percentage:.0%}) - {self.letter_grade}",
            "",
            "## Criterion Scores",
            "",
        ]

        for cs in self.criterion_scores:
            bar_fill = int(cs.percentage * 10)
            bar = "█" * bar_fill + "░" * (10 - bar_fill)
            lines.append(f"### {cs.criterion_desc}")
            lines.append(f"**Score:** {cs.points_earned:.1f}/{cs.max_points} [{bar}] {cs.percentage:.0%}")
            lines.append(f"**Feedback:** {cs.feedback}")
            if cs.evidence:
                lines.append("**Evidence:**")
                for ev in cs.evidence[:3]:
                    lines.append(f'> "{ev[:100]}..."')
            lines.append("")

        # Grammar issues
        if self.grammar_issues:
            lines.extend([
                "## Grammar & Mechanics Issues",
                f"Found {len(self.grammar_issues)} issues:",
                ""
            ])
            for issue in self.grammar_issues[:10]:
                lines.append(f"- **{issue['type']}:** {issue['text'][:50]}... → {issue.get('suggestion', 'Review')}")
            lines.append("")

        # Logic issues
        if self.logic_issues:
            lines.extend([
                "## Logic & Consistency Issues",
                ""
            ])
            for issue in self.logic_issues[:5]:
                lines.append(f"- **{issue['type']}:** {issue['description']}")
            lines.append("")

        # Citation concerns
        if self.citation_checks:
            valid = sum(1 for c in self.citation_checks if c.appears_valid)
            total = len(self.citation_checks)
            lines.extend([
                "## Citation Verification",
                f"**Checked:** {total} citations",
                f"**Appear Valid:** {valid}/{total}",
                f"**Confidence:** {self.citation_confidence:.0%}",
                ""
            ])

            suspicious = [c for c in self.citation_checks if not c.appears_valid]
            if suspicious:
                lines.append("### Potentially Problematic Citations:")
                for cit in suspicious[:5]:
                    lines.append(f"- `{cit.citation_text[:80]}...`")
                    for issue in cit.issues:
                        lines.append(f"  - ⚠️ {issue}")
                lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "document_title": self.document_title,
            "rubric_name": self.rubric_name,
            "total_score": self.total_score,
            "max_score": self.max_score,
            "percentage": self.percentage,
            "letter_grade": self.letter_grade,
            "criterion_scores": [
                {
                    "criterion": cs.criterion_id,
                    "points": cs.points_earned,
                    "max": cs.max_points,
                    "feedback": cs.feedback,
                }
                for cs in self.criterion_scores
            ],
            "grammar_issues_count": len(self.grammar_issues),
            "logic_issues_count": len(self.logic_issues),
            "citation_confidence": self.citation_confidence,
            "has_citation_concerns": self.has_citation_concerns,
        }


# Common rubric templates
ESSAY_RUBRIC = Rubric(
    name="Standard Essay Rubric",
    criteria=[
        Criterion("thesis", "Clear and arguable thesis statement", 20),
        Criterion("evidence", "Supporting evidence and examples", 25),
        Criterion("analysis", "Analysis and critical thinking", 20),
        Criterion("organization", "Organization and structure", 15),
        Criterion("grammar", "Grammar, spelling, and mechanics", 10),
        Criterion("citations", "Proper citations and references", 10),
    ]
)

RESEARCH_PAPER_RUBRIC = Rubric(
    name="Research Paper Rubric",
    criteria=[
        Criterion("thesis", "Clear research question/thesis", 15),
        Criterion("literature", "Literature review and context", 20),
        Criterion("methodology", "Clear methodology/approach", 15),
        Criterion("evidence", "Quality of evidence and data", 20),
        Criterion("analysis", "Analysis and interpretation", 15),
        Criterion("citations", "Citation quality and formatting", 15),
    ]
)

WHITEPAPER_RUBRIC = Rubric(
    name="Whitepaper Rubric",
    criteria=[
        Criterion("problem", "Clear problem statement", 15),
        Criterion("solution", "Well-defined solution", 20),
        Criterion("evidence", "Supporting data and research", 20),
        Criterion("feasibility", "Implementation feasibility", 15),
        Criterion("clarity", "Clear and professional writing", 15),
        Criterion("citations", "Credible sources and citations", 15),
    ]
)


class RubricGrader:
    """
    Grade documents against a rubric.

    Features:
    - Criterion-by-criterion scoring
    - Grammar/spelling checking
    - Logic/consistency checking
    - Citation verification (detects potentially fake citations)
    """

    # Patterns for detecting citations
    CITATION_PATTERNS = [
        # Academic style: [1], [2,3], [Author, Year]
        r'\[(\d+(?:,\s*\d+)*)\]',
        r'\[([A-Z][a-z]+(?:\s+(?:et\s+al\.?|&|and)\s+[A-Z][a-z]+)?,?\s*\d{4}[a-z]?)\]',
        # Parenthetical: (Author, Year), (Author et al., Year)
        r'\(([A-Z][a-z]+(?:\s+(?:et\s+al\.?|&|and)\s+[A-Z][a-z]+)?,?\s*\d{4}[a-z]?)\)',
        # Footnote markers
        r'\[(\d+)\]',
    ]

    # Red flags for fake citations
    CITATION_RED_FLAGS = [
        # Suspiciously round page numbers
        (r'pp?\.\s*\d+00\s*[-–]\s*\d+00', "Suspiciously round page numbers"),
        # Made-up sounding journal names
        (r'Journal of (?:Advanced|Modern|International|Global) [A-Z][a-z]+ (?:Research|Studies|Science)', "Generic journal name pattern"),
        # Excessive precision in old sources
        (r'\b(?:19[0-5]\d).*?vol\.\s*\d+,\s*no\.\s*\d+,\s*pp\.\s*\d+-\d+', "Excessive detail for old source"),
        # "Retrieved from" with no URL
        (r'[Rr]etrieved from(?!\s*http)', "Retrieved from with no URL"),
        # Broken URLs
        (r'https?://[^\s]+\.\.', "Malformed URL"),
    ]

    # Known fake/suspicious citation patterns
    KNOWN_FAKE_PATTERNS = [
        # Generic AI-generated author names
        r'\b(?:J\.|M\.|A\.|D\.)\s+(?:Smith|Johnson|Williams|Brown|Jones)\s+et\s+al\.',
        # Impossible volume/issue combinations
        r'vol\.\s*[1-9]\d{2,},',  # Volume > 100 is rare
        # Placeholder-like references
        r'\[?\d+\]?\s*[A-Z]\.\s*[A-Z][a-z]+,\s*"[^"]+,"\s*\d{4}\.',
    ]

    def __init__(self, rubric: Rubric = None, llm=None, verify_citations: bool = True):
        """
        Args:
            rubric: Grading rubric (default: ESSAY_RUBRIC)
            llm: Optional LLM for deeper analysis
            verify_citations: Whether to check citations for validity
        """
        self.rubric = rubric or ESSAY_RUBRIC
        self.llm = llm
        self.verify_citations = verify_citations

    def grade(self, document: str, title: str = "Untitled") -> GradingResult:
        """
        Grade a document against the rubric.

        Args:
            document: Document text
            title: Document title

        Returns:
            GradingResult with scores and feedback
        """
        criterion_scores = []

        # Score each criterion
        for criterion in self.rubric.criteria:
            score = self._score_criterion(document, criterion)
            criterion_scores.append(score)

        # Calculate totals
        total_score = sum(cs.points_earned for cs in criterion_scores)
        max_score = self.rubric.total_points
        percentage = total_score / max_score if max_score > 0 else 0
        letter_grade = self._calculate_letter_grade(percentage)

        # Additional checks
        grammar_issues = self._check_grammar(document)
        logic_issues = self._check_logic(document)

        # Citation verification
        citation_checks = []
        has_citation_concerns = False
        citation_confidence = 1.0

        if self.verify_citations:
            citation_checks = self._verify_citations(document)
            if citation_checks:
                suspicious = [c for c in citation_checks if not c.appears_valid]
                if suspicious:
                    has_citation_concerns = True
                    citation_confidence = 1.0 - (len(suspicious) / len(citation_checks))

        return GradingResult(
            document_title=title,
            rubric_name=self.rubric.name,
            criterion_scores=criterion_scores,
            total_score=total_score,
            max_score=max_score,
            percentage=percentage,
            letter_grade=letter_grade,
            grammar_issues=grammar_issues,
            logic_issues=logic_issues,
            citation_checks=citation_checks,
            has_citation_concerns=has_citation_concerns,
            citation_confidence=citation_confidence,
        )

    def _score_criterion(self, document: str, criterion: Criterion) -> CriterionScore:
        """Score a single criterion."""
        doc_lower = document.lower()
        crit_id = criterion.id.lower()

        # Default scoring based on criterion type
        base_score = 0.7  # Start at 70%
        feedback = ""
        evidence = []

        # Criterion-specific checks
        if "thesis" in crit_id:
            score, feedback, evidence = self._check_thesis(document)

        elif "evidence" in crit_id or "support" in crit_id:
            score, feedback, evidence = self._check_evidence(document)

        elif "organization" in crit_id or "structure" in crit_id:
            score, feedback, evidence = self._check_organization(document)

        elif "grammar" in crit_id or "mechanic" in crit_id:
            score, feedback, evidence = self._check_grammar_criterion(document)

        elif "citation" in crit_id or "reference" in crit_id:
            score, feedback, evidence = self._check_citations_criterion(document)

        elif "analysis" in crit_id or "critical" in crit_id:
            score, feedback, evidence = self._check_analysis(document)

        elif "clarity" in crit_id or "writing" in crit_id:
            score, feedback, evidence = self._check_clarity(document)

        elif "problem" in crit_id:
            score, feedback, evidence = self._check_problem_statement(document)

        elif "solution" in crit_id:
            score, feedback, evidence = self._check_solution(document)

        elif "literature" in crit_id:
            score, feedback, evidence = self._check_literature_review(document)

        elif "methodology" in crit_id or "method" in crit_id:
            score, feedback, evidence = self._check_methodology(document)

        elif "feasibility" in crit_id or "implementation" in crit_id:
            score, feedback, evidence = self._check_feasibility(document)

        else:
            # Generic scoring
            score = base_score
            feedback = "Criterion assessed based on general document quality."

        points_earned = round(score * criterion.max_points, 1)

        return CriterionScore(
            criterion_id=criterion.id,
            criterion_desc=criterion.description,
            points_earned=points_earned,
            max_points=criterion.max_points,
            percentage=score,
            feedback=feedback,
            evidence=evidence,
        )

    def _check_thesis(self, document: str) -> tuple[float, str, list]:
        """Check for clear thesis statement."""
        # Look for thesis indicators in first 20% of document
        intro_section = document[:len(document) // 5]

        thesis_indicators = [
            r'(?:this\s+(?:paper|essay|article|study)\s+(?:argues|examines|explores|demonstrates|shows))',
            r'(?:the\s+(?:purpose|goal|aim|objective)\s+(?:of\s+this|is\s+to))',
            r'(?:I\s+(?:argue|contend|propose|suggest)\s+that)',
            r'(?:this\s+(?:analysis|research|investigation))',
        ]

        found_indicators = []
        for pattern in thesis_indicators:
            matches = re.findall(pattern, intro_section, re.IGNORECASE)
            found_indicators.extend(matches)

        if len(found_indicators) >= 2:
            return 0.9, "Clear thesis statement present in introduction.", found_indicators[:2]
        elif len(found_indicators) == 1:
            return 0.75, "Thesis present but could be more explicit.", found_indicators
        else:
            return 0.5, "Thesis statement is unclear or missing.", []

    def _check_evidence(self, document: str) -> tuple[float, str, list]:
        """Check for supporting evidence."""
        evidence_patterns = [
            r'(?:according\s+to|research\s+(?:shows|indicates|suggests))',
            r'(?:for\s+example|for\s+instance|such\s+as)',
            r'(?:studies?\s+(?:show|demonstrate|found|indicate))',
            r'(?:data\s+(?:shows?|suggests?|indicates?))',
            r'\[\d+\]',  # Citation markers
            r'\([A-Z][a-z]+,?\s*\d{4}\)',  # Author-year citations
        ]

        evidence_count = 0
        examples = []

        for pattern in evidence_patterns:
            matches = re.findall(pattern, document, re.IGNORECASE)
            evidence_count += len(matches)
            examples.extend(matches[:2])

        if evidence_count >= 10:
            return 0.9, "Strong evidence with multiple citations and examples.", examples[:3]
        elif evidence_count >= 5:
            return 0.75, "Adequate evidence provided.", examples[:2]
        elif evidence_count >= 2:
            return 0.6, "Some evidence present but needs more support.", examples
        else:
            return 0.4, "Insufficient evidence to support claims.", []

    def _check_organization(self, document: str) -> tuple[float, str, list]:
        """Check document organization."""
        # Count headings
        headings = re.findall(r'^#{1,4}\s+.+$', document, re.MULTILINE)
        sections = document.split('\n\n')

        # Check for intro/conclusion markers
        has_intro = bool(re.search(r'(?:introduction|overview|background)', document[:500], re.IGNORECASE))
        has_conclusion = bool(re.search(r'(?:conclusion|summary|in\s+summary)', document[-1000:], re.IGNORECASE))

        score = 0.5
        feedback_parts = []

        if len(headings) >= 3:
            score += 0.2
            feedback_parts.append(f"{len(headings)} clear section headings")
        elif len(headings) >= 1:
            score += 0.1
            feedback_parts.append("Some headings present")
        else:
            feedback_parts.append("Missing section headings")

        if has_intro:
            score += 0.1
            feedback_parts.append("Has introduction")

        if has_conclusion:
            score += 0.1
            feedback_parts.append("Has conclusion")

        # Check paragraph length consistency
        para_lengths = [len(p.split()) for p in sections if p.strip()]
        if para_lengths:
            avg_len = sum(para_lengths) / len(para_lengths)
            if 50 <= avg_len <= 200:
                score += 0.1
                feedback_parts.append("Good paragraph length")

        return min(score, 1.0), "; ".join(feedback_parts), headings[:3]

    def _check_grammar_criterion(self, document: str) -> tuple[float, str, list]:
        """Check grammar and mechanics for rubric scoring."""
        issues = self._check_grammar(document)

        # Calculate score based on issue density
        words = len(document.split())
        issue_rate = len(issues) / (words / 100) if words > 0 else 0

        if issue_rate < 0.5:
            return 0.95, "Excellent grammar and mechanics.", []
        elif issue_rate < 1.5:
            return 0.85, "Good grammar with minor issues.", [i['text'] for i in issues[:2]]
        elif issue_rate < 3:
            return 0.7, "Several grammar issues present.", [i['text'] for i in issues[:3]]
        else:
            return 0.5, "Significant grammar problems.", [i['text'] for i in issues[:3]]

    def _check_citations_criterion(self, document: str) -> tuple[float, str, list]:
        """Check citation quality for rubric scoring."""
        # Find all citations
        citations = []
        for pattern in self.CITATION_PATTERNS:
            citations.extend(re.findall(pattern, document))

        if not citations:
            return 0.3, "No citations found.", []

        # Check for reference section
        has_refs = bool(re.search(r'(?:references?|bibliography|works?\s+cited)', document, re.IGNORECASE))

        # Verify citations
        checks = self._verify_citations(document)
        valid_pct = sum(1 for c in checks if c.appears_valid) / len(checks) if checks else 0

        score = 0.4
        feedback_parts = []

        if len(citations) >= 5:
            score += 0.2
            feedback_parts.append(f"{len(citations)} citations")
        elif len(citations) >= 2:
            score += 0.1
            feedback_parts.append(f"{len(citations)} citations (could use more)")

        if has_refs:
            score += 0.15
            feedback_parts.append("Has reference section")
        else:
            feedback_parts.append("Missing reference section")

        if valid_pct >= 0.8:
            score += 0.2
            feedback_parts.append("Citations appear credible")
        elif valid_pct >= 0.5:
            score += 0.1
            feedback_parts.append("Some citation concerns")
        else:
            feedback_parts.append("Multiple questionable citations")

        return min(score, 1.0), "; ".join(feedback_parts), citations[:3]

    def _check_analysis(self, document: str) -> tuple[float, str, list]:
        """Check for analytical depth."""
        analysis_markers = [
            r'(?:this\s+(?:suggests|implies|indicates|demonstrates))',
            r'(?:therefore|consequently|as\s+a\s+result|thus)',
            r'(?:the\s+(?:significance|importance|implication)\s+of)',
            r'(?:this\s+(?:finding|result|evidence)\s+(?:shows|reveals))',
            r'(?:(?:one|another)\s+(?:interpretation|explanation))',
            r'(?:however|on\s+the\s+other\s+hand|in\s+contrast)',
        ]

        analysis_count = 0
        examples = []

        for pattern in analysis_markers:
            matches = re.findall(pattern, document, re.IGNORECASE)
            analysis_count += len(matches)
            examples.extend(matches[:1])

        if analysis_count >= 8:
            return 0.9, "Strong analytical depth with clear interpretation.", examples[:3]
        elif analysis_count >= 4:
            return 0.75, "Good analysis present.", examples[:2]
        elif analysis_count >= 2:
            return 0.6, "Some analysis but needs more depth.", examples
        else:
            return 0.4, "Lacks analytical depth - mostly descriptive.", []

    def _check_clarity(self, document: str) -> tuple[float, str, list]:
        """Check writing clarity."""
        sentences = re.split(r'[.!?]+', document)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.5, "Unable to assess clarity.", []

        # Average sentence length
        avg_words = sum(len(s.split()) for s in sentences) / len(sentences)

        # Complex word ratio (words > 3 syllables, approximated by length > 10)
        words = document.split()
        complex_ratio = sum(1 for w in words if len(w) > 10) / len(words) if words else 0

        score = 0.7
        feedback_parts = []

        if avg_words <= 20:
            score += 0.15
            feedback_parts.append("Good sentence length")
        elif avg_words <= 25:
            score += 0.05
            feedback_parts.append("Acceptable sentence length")
        else:
            score -= 0.1
            feedback_parts.append("Sentences too long")

        if complex_ratio < 0.1:
            score += 0.1
            feedback_parts.append("Clear vocabulary")
        elif complex_ratio > 0.2:
            score -= 0.1
            feedback_parts.append("May be overly complex")

        return min(max(score, 0.3), 1.0), "; ".join(feedback_parts), []

    def _check_problem_statement(self, document: str) -> tuple[float, str, list]:
        """Check for clear problem statement."""
        problem_markers = [
            r'(?:the\s+(?:problem|issue|challenge)\s+(?:is|involves|concerns))',
            r'(?:(?:organizations?|companies|enterprises?)\s+(?:face|struggle|encounter))',
            r'(?:(?:this|the)\s+(?:gap|limitation|bottleneck))',
            r'(?:currently|traditionally),?\s+(?:there\s+is|we\s+see)',
        ]

        found = []
        for pattern in problem_markers:
            matches = re.findall(pattern, document[:2000], re.IGNORECASE)
            found.extend(matches)

        if len(found) >= 3:
            return 0.9, "Clear problem statement with context.", found[:2]
        elif len(found) >= 1:
            return 0.7, "Problem stated but could be clearer.", found
        else:
            return 0.4, "Problem statement unclear or missing.", []

    def _check_solution(self, document: str) -> tuple[float, str, list]:
        """Check for well-defined solution."""
        solution_markers = [
            r'(?:(?:the|our)\s+(?:solution|approach|method)\s+(?:is|involves|uses))',
            r'(?:we\s+(?:propose|present|introduce|describe))',
            r'(?:this\s+(?:approach|method|technique|system))',
            r'(?:(?:by|through)\s+(?:using|leveraging|implementing))',
        ]

        found = []
        for pattern in solution_markers:
            matches = re.findall(pattern, document, re.IGNORECASE)
            found.extend(matches)

        if len(found) >= 4:
            return 0.9, "Solution well-defined and explained.", found[:2]
        elif len(found) >= 2:
            return 0.7, "Solution present but could be more detailed.", found
        else:
            return 0.4, "Solution unclear or not well-defined.", []

    def _check_literature_review(self, document: str) -> tuple[float, str, list]:
        """Check for literature review quality."""
        lit_markers = [
            r'(?:previous\s+(?:research|studies|work))',
            r'(?:(?:prior|existing)\s+(?:literature|research))',
            r'(?:[A-Z][a-z]+\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+))',
            r'(?:according\s+to\s+[A-Z][a-z]+)',
        ]

        found = []
        for pattern in lit_markers:
            matches = re.findall(pattern, document, re.IGNORECASE)
            found.extend(matches)

        citations = re.findall(r'\[\d+\]|\([A-Z][a-z]+,?\s*\d{4}\)', document)

        if len(found) >= 5 and len(citations) >= 5:
            return 0.9, "Strong literature review with multiple sources.", found[:2]
        elif len(found) >= 2 and len(citations) >= 2:
            return 0.7, "Adequate literature coverage.", found[:2]
        else:
            return 0.4, "Literature review needs more sources.", []

    def _check_methodology(self, document: str) -> tuple[float, str, list]:
        """Check methodology section."""
        method_markers = [
            r'(?:(?:our|the)\s+(?:method|approach|methodology))',
            r'(?:we\s+(?:collected|gathered|analyzed|measured))',
            r'(?:(?:the|this)\s+(?:study|analysis|experiment))',
            r'(?:(?:data|sample)\s+(?:was|were)\s+(?:collected|gathered))',
        ]

        found = []
        for pattern in method_markers:
            matches = re.findall(pattern, document, re.IGNORECASE)
            found.extend(matches)

        if len(found) >= 3:
            return 0.85, "Clear methodology described.", found[:2]
        elif len(found) >= 1:
            return 0.65, "Some methodology present but needs more detail.", found
        else:
            return 0.4, "Methodology unclear or missing.", []

    def _check_feasibility(self, document: str) -> tuple[float, str, list]:
        """Check implementation feasibility discussion."""
        feasibility_markers = [
            r'(?:implement(?:ation|ing|ed))',
            r'(?:practical|feasib(?:le|ility))',
            r'(?:(?:can|could)\s+be\s+(?:used|applied|deployed))',
            r'(?:(?:cost|time|resource)\s+(?:effective|efficient))',
            r'(?:(?:real|production)\s*(?:-|\s)?(?:world|environment))',
        ]

        found = []
        for pattern in feasibility_markers:
            matches = re.findall(pattern, document, re.IGNORECASE)
            found.extend(matches)

        if len(found) >= 4:
            return 0.85, "Good discussion of implementation feasibility.", found[:2]
        elif len(found) >= 2:
            return 0.65, "Some feasibility discussion present.", found
        else:
            return 0.4, "Needs more discussion of practical implementation.", []

    def _check_grammar(self, document: str) -> list[dict]:
        """Check for grammar issues."""
        issues = []

        # Common grammar patterns
        checks = [
            (r'\s{2,}', "Extra spaces", "Remove extra spaces"),
            (r'[.!?]\s*[a-z]', "Missing capitalization", "Capitalize after sentence end"),
            (r'\bi\b', "Uncapitalized 'I'", "Capitalize 'I'"),
            (r'(?<!\.)\.{2}(?!\.)', "Double period", "Use single period or ellipsis"),
            (r'\b(their|there|they\'re)\b.*\b(their|there|they\'re)\b', "Possible their/there/they're confusion", "Verify correct usage"),
            (r'\b(its|it\'s)\b.*\b(its|it\'s)\b', "Possible its/it's confusion", "Verify correct usage"),
            (r'\b(\w+)\s+\1\b', "Repeated word", "Remove duplicate"),
        ]

        for pattern, issue_type, suggestion in checks:
            for match in re.finditer(pattern, document):
                issues.append({
                    "type": issue_type,
                    "text": match.group()[:50],
                    "position": match.start(),
                    "suggestion": suggestion,
                })

        return issues[:20]  # Limit to 20 issues

    def _check_logic(self, document: str) -> list[dict]:
        """Check for logical issues."""
        issues = []

        # Contradictory statements pattern (simplified)
        contradiction_pairs = [
            (r'always', r'never'),
            (r'all\s+\w+', r'no\s+\w+'),
            (r'increases?', r'decreases?'),
        ]

        sentences = re.split(r'[.!?]+', document)

        # Check for unsupported claims
        unsupported_markers = [
            r'(?:obviously|clearly|everyone\s+knows|it\s+is\s+(?:well\s+)?known)',
            r'(?:undoubtedly|without\s+(?:a\s+)?doubt|certainly)',
        ]

        for pattern in unsupported_markers:
            for match in re.finditer(pattern, document, re.IGNORECASE):
                issues.append({
                    "type": "Unsupported assertion",
                    "description": f"Claim may need evidence: '{match.group()}'",
                })

        # Check for circular reasoning indicators
        circular_patterns = [
            r'because\s+it\s+(?:is|does)',
            r'this\s+is\s+true\s+because',
        ]

        for pattern in circular_patterns:
            if re.search(pattern, document, re.IGNORECASE):
                issues.append({
                    "type": "Possible circular reasoning",
                    "description": "Argument may be circular - verify reasoning chain",
                })

        return issues[:10]

    def _verify_citations(self, document: str) -> list[CitationCheck]:
        """Verify citations for potential hallucination."""
        checks = []

        # Extract reference section if exists
        ref_section = ""
        ref_match = re.search(r'(?:references?|bibliography|works?\s+cited)\s*:?\s*\n(.*)', document, re.IGNORECASE | re.DOTALL)
        if ref_match:
            ref_section = ref_match.group(1)

        # Find all references in reference section
        ref_patterns = [
            # [1] Author. "Title." Journal, vol. X, pp. Y-Z, Year.
            r'\[\d+\][^\[\]]{10,300}(?:\d{4})',
            # Author (Year). Title. Source.
            r'[A-Z][a-z]+(?:,\s*[A-Z]\.)+\s*\(\d{4}\)[^.]+\.',
            # - Author et al. (Year). Title. Source.
            r'-\s*[A-Z][a-z]+(?:\s+et\s+al\.?)?\s*\(\d{4}\)[^.]+\.',
            # Author et al. (Year). Title.
            r'^[A-Z][a-z]+(?:\s+et\s+al\.?)?\s*\(\d{4}\)[^.]+\.',
            # Lines starting with - that contain a year
            r'-\s*[^\n]+\(\d{4}\)[^\n]*',
        ]

        references = []
        for pattern in ref_patterns:
            matches = re.findall(pattern, ref_section, re.MULTILINE)
            references.extend(matches)

        # Also check in-text citations
        in_text_patterns = [
            # (Author, Year) or (Author Year)
            r'\([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*\d{4}[a-z]?\)',
            # Author et al. (Year)
            r'[A-Z][a-z]+(?:\s+et\s+al\.?)\s*\(\d{4}\)',
            # Author (Year)
            r'[A-Z][a-z]+\s+\(\d{4}\)',
        ]

        for pattern in in_text_patterns:
            matches = re.findall(pattern, document)
            in_text = [m for m in matches if m not in references]
            references.extend(in_text[:5])  # Limit in-text additions

        # Deduplicate
        seen = set()
        unique_refs = []
        for ref in references:
            ref_clean = ref.strip()
            if ref_clean not in seen and len(ref_clean) > 10:
                seen.add(ref_clean)
                unique_refs.append(ref_clean)

        for citation in unique_refs[:15]:  # Limit to 15
            check = self._analyze_citation(citation)
            checks.append(check)

        return checks

    def _analyze_citation(self, citation: str) -> CitationCheck:
        """Analyze a single citation for validity."""
        issues = []
        confidence = 1.0

        # Check against red flags
        for pattern, issue in self.CITATION_RED_FLAGS:
            if re.search(pattern, citation):
                issues.append(issue)
                confidence -= 0.2

        # Check against known fake patterns
        for pattern in self.KNOWN_FAKE_PATTERNS:
            if re.search(pattern, citation):
                issues.append("Matches common AI-generated citation pattern")
                confidence -= 0.3

        # Check year
        from datetime import datetime
        current_year = datetime.now().year

        year_match = re.search(r'\b(19|20)\d{2}\b', citation)
        if year_match:
            year = int(year_match.group())
            if year > current_year:
                issues.append(f"Future publication year: {year} (current year: {current_year})")
                confidence -= 0.5
            elif year == current_year:
                issues.append(f"Very recent ({year}) - may be preprint or not yet published")
                confidence -= 0.1
            elif year < 1900:
                issues.append(f"Suspiciously old: {year}")
                confidence -= 0.2

        # Check for excessively specific details (common in hallucinations)
        if re.search(r'vol\.\s*\d+,\s*no\.\s*\d+,\s*pp\.\s*\d+-\d+', citation):
            # This level of detail is suspicious without verification
            if not re.search(r'doi|http|ISBN', citation, re.IGNORECASE):
                issues.append("Detailed citation without DOI/URL (harder to verify)")
                confidence -= 0.1

        # Check for generic author combinations
        generic_authors = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Davis', 'Miller']
        author_count = sum(1 for author in generic_authors if author in citation)
        if author_count >= 2:
            issues.append("Multiple generic author names")
            confidence -= 0.15

        confidence = max(0.0, confidence)

        # Check for critical issues that automatically make citation suspect
        critical_issues = ['Future publication', 'Matches common AI-generated']
        has_critical = any(any(crit in issue for crit in critical_issues) for issue in issues)

        appears_valid = confidence >= 0.6 and len(issues) <= 1 and not has_critical

        return CitationCheck(
            citation_text=citation[:150],
            appears_valid=appears_valid,
            confidence=confidence,
            issues=issues,
        )

    def _calculate_letter_grade(self, percentage: float) -> str:
        """Convert percentage to letter grade."""
        if percentage >= 0.97:
            return "A+"
        elif percentage >= 0.93:
            return "A"
        elif percentage >= 0.90:
            return "A-"
        elif percentage >= 0.87:
            return "B+"
        elif percentage >= 0.83:
            return "B"
        elif percentage >= 0.80:
            return "B-"
        elif percentage >= 0.77:
            return "C+"
        elif percentage >= 0.73:
            return "C"
        elif percentage >= 0.70:
            return "C-"
        elif percentage >= 0.67:
            return "D+"
        elif percentage >= 0.63:
            return "D"
        elif percentage >= 0.60:
            return "D-"
        else:
            return "F"


def quick_grade(document: str, rubric: Rubric = None, title: str = "Document") -> GradingResult:
    """
    Quick one-liner grading.

    Args:
        document: Document text
        rubric: Grading rubric (default: ESSAY_RUBRIC)
        title: Document title

    Returns:
        GradingResult

    Example:
        from reviewer import quick_grade, WHITEPAPER_RUBRIC

        result = quick_grade(my_document, WHITEPAPER_RUBRIC)
        print(result.summary())
    """
    grader = RubricGrader(rubric)
    return grader.grade(document, title)
