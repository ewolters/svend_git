"""
Document Reviewer Agent

General-purpose document review with:
- Multiple review dimensions (style, structure, clarity, completeness)
- Configurable checklists
- Severity-based findings
- Remediation suggestions
- Supports any document type

Can integrate compliance frameworks (ISO, SOC2) for specialized reviews.
"""

import re
from dataclasses import dataclass, field
from typing import Literal
from enum import Enum

import sys
sys.path.insert(0, '/home/eric/Desktop/agents')

from core.intent import IntentTracker, Action


class ReviewDimension(Enum):
    """Dimensions of document review."""
    STRUCTURE = "structure"       # Organization, sections, flow
    CLARITY = "clarity"           # Readability, jargon, ambiguity
    COMPLETENESS = "completeness" # Missing sections, gaps
    CONSISTENCY = "consistency"   # Style, terminology, formatting
    ACCURACY = "accuracy"         # Facts, claims, references
    GRAMMAR = "grammar"           # Spelling, punctuation, grammar
    COMPLIANCE = "compliance"     # Standards adherence


class Severity(Enum):
    """Finding severity levels."""
    CRITICAL = "critical"   # Must fix before publication
    MAJOR = "major"         # Should fix, impacts quality
    MINOR = "minor"         # Nice to fix, polish
    SUGGESTION = "suggestion"  # Optional improvement


@dataclass
class Finding:
    """A review finding."""
    id: str
    dimension: ReviewDimension
    severity: Severity
    title: str
    description: str
    location: str = ""  # Section/paragraph reference
    suggestion: str = ""  # How to fix
    context: str = ""  # Surrounding text

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "dimension": self.dimension.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "location": self.location,
            "suggestion": self.suggestion,
        }


@dataclass
class ChecklistItem:
    """A checklist item for review."""
    id: str
    dimension: ReviewDimension
    description: str
    check_func: callable = None  # Optional automated check
    severity_if_missing: Severity = Severity.MAJOR


@dataclass
class ReviewResult:
    """Result of document review."""
    document_title: str
    findings: list[Finding]
    checklist_results: dict[str, bool]  # checklist_id -> passed
    scores: dict[str, float]  # dimension -> 0-1 score
    overall_score: float
    summary: str

    def to_markdown(self) -> str:
        """Export as markdown report."""
        lines = [
            f"# Document Review: {self.document_title}",
            "",
            f"**Overall Score:** {self.overall_score:.0%}",
            "",
            "## Dimension Scores",
            "",
        ]

        for dim, score in sorted(self.scores.items()):
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            lines.append(f"- {dim}: [{bar}] {score:.0%}")

        lines.extend(["", "## Summary", "", self.summary, ""])

        # Findings by severity
        for severity in [Severity.CRITICAL, Severity.MAJOR, Severity.MINOR, Severity.SUGGESTION]:
            sev_findings = [f for f in self.findings if f.severity == severity]
            if sev_findings:
                lines.append(f"## {severity.value.title()} Findings ({len(sev_findings)})")
                lines.append("")
                for f in sev_findings:
                    lines.append(f"### {f.id}: {f.title}")
                    lines.append(f"**Dimension:** {f.dimension.value}")
                    if f.location:
                        lines.append(f"**Location:** {f.location}")
                    lines.append("")
                    lines.append(f.description)
                    if f.suggestion:
                        lines.append("")
                        lines.append(f"**Suggestion:** {f.suggestion}")
                    lines.append("")

        return "\n".join(lines)


class DocumentReviewer:
    """
    General-purpose document reviewer.

    Reviews documents across multiple dimensions and generates
    findings with remediation suggestions.
    """

    # Default checklists by document type
    CHECKLISTS = {
        "technical": [
            ChecklistItem("tech_01", ReviewDimension.STRUCTURE,
                         "Has clear introduction explaining purpose"),
            ChecklistItem("tech_02", ReviewDimension.STRUCTURE,
                         "Has table of contents for docs > 5 pages"),
            ChecklistItem("tech_03", ReviewDimension.COMPLETENESS,
                         "All acronyms are defined on first use"),
            ChecklistItem("tech_04", ReviewDimension.COMPLETENESS,
                         "Has prerequisites/requirements section"),
            ChecklistItem("tech_05", ReviewDimension.CLARITY,
                         "Code examples are syntax-highlighted"),
            ChecklistItem("tech_06", ReviewDimension.CONSISTENCY,
                         "Consistent terminology throughout"),
            ChecklistItem("tech_07", ReviewDimension.ACCURACY,
                         "Version numbers are current"),
            ChecklistItem("tech_08", ReviewDimension.COMPLETENESS,
                         "Has troubleshooting or FAQ section"),
        ],
        "business": [
            ChecklistItem("biz_01", ReviewDimension.STRUCTURE,
                         "Has executive summary"),
            ChecklistItem("biz_02", ReviewDimension.STRUCTURE,
                         "Clear sections with headings"),
            ChecklistItem("biz_03", ReviewDimension.CLARITY,
                         "Avoids jargon or explains technical terms"),
            ChecklistItem("biz_04", ReviewDimension.COMPLETENESS,
                         "Has clear call to action or next steps"),
            ChecklistItem("biz_05", ReviewDimension.ACCURACY,
                         "Data and statistics are sourced"),
            ChecklistItem("biz_06", ReviewDimension.CONSISTENCY,
                         "Consistent formatting (fonts, spacing)"),
        ],
        "academic": [
            ChecklistItem("acad_01", ReviewDimension.STRUCTURE,
                         "Has abstract"),
            ChecklistItem("acad_02", ReviewDimension.STRUCTURE,
                         "Has introduction, methods, results, discussion"),
            ChecklistItem("acad_03", ReviewDimension.ACCURACY,
                         "All claims have citations"),
            ChecklistItem("acad_04", ReviewDimension.COMPLETENESS,
                         "Has complete references section"),
            ChecklistItem("acad_05", ReviewDimension.CLARITY,
                         "Figures and tables are labeled and referenced"),
            ChecklistItem("acad_06", ReviewDimension.CONSISTENCY,
                         "Citation style is consistent"),
        ],
        "general": [
            ChecklistItem("gen_01", ReviewDimension.STRUCTURE,
                         "Has clear beginning, middle, end"),
            ChecklistItem("gen_02", ReviewDimension.CLARITY,
                         "Sentences are concise (< 25 words average)"),
            ChecklistItem("gen_03", ReviewDimension.GRAMMAR,
                         "No obvious spelling errors"),
            ChecklistItem("gen_04", ReviewDimension.CONSISTENCY,
                         "Consistent tense throughout"),
        ],
    }

    # Common issues to check
    ISSUE_PATTERNS = [
        # Clarity issues
        (r'\b(very|really|quite|somewhat|fairly)\b', ReviewDimension.CLARITY,
         Severity.MINOR, "Weak modifier", "Consider removing or using stronger language"),
        (r'\b(thing|stuff|it)\b(?!s\b)', ReviewDimension.CLARITY,
         Severity.MINOR, "Vague reference", "Be more specific about what 'it' or 'thing' refers to"),
        (r'(?i)\betc\.?\b', ReviewDimension.CLARITY,
         Severity.SUGGESTION, "Vague 'etc.'", "List specific items or use 'and more'"),

        # Passive voice (simple detection)
        (r'\b(is|are|was|were|been|being)\s+\w+ed\b', ReviewDimension.CLARITY,
         Severity.SUGGESTION, "Possible passive voice", "Consider active voice for clarity"),

        # Consistency issues
        (r'(?i)\b(e\.?g\.?|i\.?e\.?)\b', ReviewDimension.CONSISTENCY,
         Severity.MINOR, "Inconsistent abbreviation", "Use consistent format: 'e.g.,' or 'for example'"),

        # Structure issues
        (r'^#{1,6}\s*$', ReviewDimension.STRUCTURE,
         Severity.MAJOR, "Empty heading", "Remove or add content to heading"),

        # Grammar (basic)
        (r'\s{2,}', ReviewDimension.GRAMMAR,
         Severity.MINOR, "Multiple spaces", "Use single space between words"),
        (r'[.!?]\s*[a-z]', ReviewDimension.GRAMMAR,
         Severity.MINOR, "Missing capitalization", "Capitalize after sentence-ending punctuation"),
    ]

    def __init__(self, llm=None):
        self.llm = llm
        self.intent_tracker = IntentTracker(llm=llm)
        self._finding_counter = 0

    def _llm_generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate text from LLM."""
        if self.llm is None:
            return ""
        if hasattr(self.llm, 'generate'):
            return self.llm.generate(prompt, max_tokens=max_tokens)
        elif hasattr(self.llm, 'complete'):
            return self.llm.complete(prompt, max_tokens=max_tokens)
        return ""

    def _next_finding_id(self) -> str:
        """Generate next finding ID."""
        self._finding_counter += 1
        return f"F{self._finding_counter:03d}"

    def review(self, document: str, title: str = "Untitled",
               doc_type: Literal["technical", "business", "academic", "general"] = "general",
               custom_checklist: list[ChecklistItem] = None) -> ReviewResult:
        """
        Review a document.

        Args:
            document: Document text (plain text or markdown)
            title: Document title
            doc_type: Type of document for checklist selection
            custom_checklist: Additional checklist items

        Returns:
            ReviewResult with findings and scores
        """
        self._finding_counter = 0

        self.intent_tracker.set_intent(
            raw_input=f"Review document: {title}",
            parsed_goal=f"Thoroughly review {doc_type} document for quality issues",
            constraints=["Identify all significant issues", "Provide actionable suggestions"],
        )

        findings = []

        # 1. Pattern-based checks
        pattern_findings = self._check_patterns(document)
        findings.extend(pattern_findings)

        # 2. Structural analysis
        structure_findings = self._analyze_structure(document, doc_type)
        findings.extend(structure_findings)

        # 3. Readability analysis
        readability_findings = self._analyze_readability(document)
        findings.extend(readability_findings)

        # 4. Checklist review
        checklist = self.CHECKLISTS.get(doc_type, self.CHECKLISTS["general"])
        if custom_checklist:
            checklist = checklist + custom_checklist

        checklist_results = self._run_checklist(document, checklist, findings)

        # 5. LLM-based review (if available)
        if self.llm:
            llm_findings = self._llm_review(document, doc_type)
            findings.extend(llm_findings)

        # Calculate scores
        scores = self._calculate_scores(findings, checklist_results)
        overall_score = sum(scores.values()) / len(scores) if scores else 0.5

        # Generate summary
        summary = self._generate_summary(findings, scores, overall_score)

        # Record action
        action = Action(
            id="review_complete",
            description=f"Completed review of {title}",
            action_type="review",
            content=f"{len(findings)} findings",
            alignment_score=1.0,
            reasoning=f"Found {len([f for f in findings if f.severity == Severity.CRITICAL])} critical, "
                     f"{len([f for f in findings if f.severity == Severity.MAJOR])} major issues",
        )
        self.intent_tracker.record_action(action)

        return ReviewResult(
            document_title=title,
            findings=findings,
            checklist_results=checklist_results,
            scores=scores,
            overall_score=overall_score,
            summary=summary,
        )

    def _check_patterns(self, document: str) -> list[Finding]:
        """Check document against issue patterns."""
        findings = []

        for pattern, dimension, severity, title, suggestion in self.ISSUE_PATTERNS:
            matches = list(re.finditer(pattern, document, re.MULTILINE))

            # Limit to first 5 instances
            for match in matches[:5]:
                # Get context
                start = max(0, match.start() - 30)
                end = min(len(document), match.end() + 30)
                context = document[start:end].replace('\n', ' ')

                # Estimate location
                line_num = document[:match.start()].count('\n') + 1

                findings.append(Finding(
                    id=self._next_finding_id(),
                    dimension=dimension,
                    severity=severity,
                    title=title,
                    description=f"Found: '{match.group()}'",
                    location=f"Line ~{line_num}",
                    suggestion=suggestion,
                    context=f"...{context}...",
                ))

        return findings

    def _analyze_structure(self, document: str, doc_type: str) -> list[Finding]:
        """Analyze document structure."""
        findings = []

        lines = document.split('\n')
        headings = [l for l in lines if l.startswith('#')]

        # Check for headings
        if len(document) > 1000 and not headings:
            findings.append(Finding(
                id=self._next_finding_id(),
                dimension=ReviewDimension.STRUCTURE,
                severity=Severity.MAJOR,
                title="No headings in long document",
                description="Document is over 1000 characters but has no headings",
                suggestion="Add section headings to improve navigation",
            ))

        # Check heading hierarchy
        prev_level = 0
        for heading in headings:
            level = len(heading) - len(heading.lstrip('#'))
            if level > prev_level + 1:
                findings.append(Finding(
                    id=self._next_finding_id(),
                    dimension=ReviewDimension.STRUCTURE,
                    severity=Severity.MINOR,
                    title="Skipped heading level",
                    description=f"Jumped from H{prev_level} to H{level}",
                    location=heading[:50],
                    suggestion=f"Use H{prev_level + 1} instead",
                ))
            prev_level = level

        # Check for very short or very long sections
        sections = re.split(r'^#+\s', document, flags=re.MULTILINE)
        for i, section in enumerate(sections[1:], 1):  # Skip preamble
            words = len(section.split())
            if words < 20:
                findings.append(Finding(
                    id=self._next_finding_id(),
                    dimension=ReviewDimension.COMPLETENESS,
                    severity=Severity.SUGGESTION,
                    title="Very short section",
                    description=f"Section has only {words} words",
                    location=f"Section {i}",
                    suggestion="Consider expanding or merging with another section",
                ))
            elif words > 1000:
                findings.append(Finding(
                    id=self._next_finding_id(),
                    dimension=ReviewDimension.STRUCTURE,
                    severity=Severity.SUGGESTION,
                    title="Very long section",
                    description=f"Section has {words} words",
                    location=f"Section {i}",
                    suggestion="Consider breaking into subsections",
                ))

        return findings

    def _analyze_readability(self, document: str) -> list[Finding]:
        """Analyze readability metrics."""
        findings = []

        # Simple sentence analysis
        sentences = re.split(r'[.!?]+', document)
        sentences = [s.strip() for s in sentences if s.strip()]

        if sentences:
            # Average sentence length
            avg_words = sum(len(s.split()) for s in sentences) / len(sentences)

            if avg_words > 25:
                findings.append(Finding(
                    id=self._next_finding_id(),
                    dimension=ReviewDimension.CLARITY,
                    severity=Severity.MAJOR,
                    title="High average sentence length",
                    description=f"Average sentence is {avg_words:.1f} words (target: < 25)",
                    suggestion="Break long sentences into shorter ones",
                ))

            # Very long sentences
            long_sentences = [s for s in sentences if len(s.split()) > 40]
            if long_sentences:
                findings.append(Finding(
                    id=self._next_finding_id(),
                    dimension=ReviewDimension.CLARITY,
                    severity=Severity.MAJOR,
                    title=f"{len(long_sentences)} very long sentences",
                    description="Sentences over 40 words are difficult to read",
                    location=f"First instance: '{long_sentences[0][:50]}...'",
                    suggestion="Split into multiple sentences",
                ))

        # Paragraph analysis
        paragraphs = document.split('\n\n')
        long_paragraphs = [p for p in paragraphs if len(p.split()) > 200]
        if long_paragraphs:
            findings.append(Finding(
                id=self._next_finding_id(),
                dimension=ReviewDimension.CLARITY,
                severity=Severity.MINOR,
                title=f"{len(long_paragraphs)} very long paragraphs",
                description="Paragraphs over 200 words can be overwhelming",
                suggestion="Break into smaller paragraphs",
            ))

        return findings

    def _run_checklist(self, document: str, checklist: list[ChecklistItem],
                       findings: list[Finding]) -> dict[str, bool]:
        """Run checklist items and add findings for failures."""
        results = {}

        doc_lower = document.lower()

        for item in checklist:
            # Simple keyword-based checks
            passed = True

            # Structure checks
            if "table of contents" in item.description.lower():
                passed = "table of contents" in doc_lower or "## contents" in doc_lower

            elif "executive summary" in item.description.lower():
                passed = "executive summary" in doc_lower or "## summary" in doc_lower

            elif "introduction" in item.description.lower():
                passed = "introduction" in doc_lower or "## intro" in doc_lower

            elif "abstract" in item.description.lower():
                passed = "abstract" in doc_lower

            elif "references" in item.description.lower():
                passed = "reference" in doc_lower or "bibliography" in doc_lower

            elif "acronyms" in item.description.lower():
                # Check if there are undefined acronyms (simple heuristic)
                acronyms = re.findall(r'\b[A-Z]{2,}\b', document)
                # Very simplified - just check if there are some
                passed = len(acronyms) < 5 or "acronym" in doc_lower or any(
                    f"({a})" in document or f"{a} (" in document for a in acronyms[:3]
                )

            # Default: mark as passed if we can't check automatically
            results[item.id] = passed

            if not passed:
                findings.append(Finding(
                    id=self._next_finding_id(),
                    dimension=item.dimension,
                    severity=item.severity_if_missing,
                    title=f"Checklist: {item.description}",
                    description=f"Document does not appear to meet this requirement",
                    suggestion=f"Add or improve: {item.description}",
                ))

        return results

    def _llm_review(self, document: str, doc_type: str) -> list[Finding]:
        """Use LLM for deeper review."""
        findings = []

        prompt = f"""Review this {doc_type} document for quality issues.

Document (first 2000 chars):
{document[:2000]}

Identify 3-5 specific issues with:
1. What the issue is
2. Where it occurs (quote specific text)
3. How to fix it
4. Severity (critical/major/minor)

Output as JSON array:
[{{"issue": "...", "quote": "...", "fix": "...", "severity": "..."}}]"""

        response = self._llm_generate(prompt, max_tokens=800)

        # Parse response (simplified)
        import json
        try:
            # Find JSON in response
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                issues = json.loads(match.group())
                for issue in issues[:5]:
                    severity_map = {
                        "critical": Severity.CRITICAL,
                        "major": Severity.MAJOR,
                        "minor": Severity.MINOR,
                    }
                    findings.append(Finding(
                        id=self._next_finding_id(),
                        dimension=ReviewDimension.CLARITY,  # LLM findings default to clarity
                        severity=severity_map.get(issue.get("severity", "minor").lower(), Severity.MINOR),
                        title=issue.get("issue", "LLM Finding")[:100],
                        description=issue.get("issue", ""),
                        location=issue.get("quote", "")[:100],
                        suggestion=issue.get("fix", ""),
                    ))
        except (json.JSONDecodeError, AttributeError):
            pass  # Skip if parsing fails

        return findings

    def _calculate_scores(self, findings: list[Finding],
                          checklist_results: dict[str, bool]) -> dict[str, float]:
        """Calculate scores per dimension."""
        scores = {}

        # Base scores
        for dim in ReviewDimension:
            dim_findings = [f for f in findings if f.dimension == dim]

            # Start at 1.0, deduct based on severity
            score = 1.0
            for f in dim_findings:
                if f.severity == Severity.CRITICAL:
                    score -= 0.3
                elif f.severity == Severity.MAJOR:
                    score -= 0.15
                elif f.severity == Severity.MINOR:
                    score -= 0.05
                elif f.severity == Severity.SUGGESTION:
                    score -= 0.02

            scores[dim.value] = max(0.0, min(1.0, score))

        # Checklist impact
        if checklist_results:
            pass_rate = sum(checklist_results.values()) / len(checklist_results)
            # Blend checklist results into completeness score
            if "completeness" in scores:
                scores["completeness"] = (scores["completeness"] + pass_rate) / 2

        return scores

    def _generate_summary(self, findings: list[Finding], scores: dict[str, float],
                          overall_score: float) -> str:
        """Generate review summary."""
        critical = len([f for f in findings if f.severity == Severity.CRITICAL])
        major = len([f for f in findings if f.severity == Severity.MAJOR])
        minor = len([f for f in findings if f.severity == Severity.MINOR])

        if critical > 0:
            status = "Needs significant revision"
        elif major > 5:
            status = "Needs revision"
        elif major > 0:
            status = "Needs minor revisions"
        else:
            status = "Ready with minor polish"

        # Find weakest dimensions
        weak_dims = sorted(scores.items(), key=lambda x: x[1])[:2]

        summary = f"""**Status:** {status}

**Findings:** {critical} critical, {major} major, {minor} minor issues found.

**Weakest areas:** {', '.join(d[0] for d in weak_dims)}

**Recommendation:** {"Address critical issues immediately." if critical > 0 else "Review major findings and address before publication." if major > 0 else "Document is in good shape. Consider addressing minor issues."}"""

        return summary
