"""
Editor Agent

Cleans up documents and produces an editorial report:
- Fixes grammar and spelling
- Flags questionable citations
- Identifies repetition
- Produces cleaned document + editorial report

Usage:
    from reviewer import Editor

    editor = Editor()
    result = editor.edit(document, title="My Whitepaper")

    print(result.cleaned_document)
    print(result.editorial_report)
    result.save('edited_output/')
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Optional

from .rubric import RubricGrader, CitationCheck, WHITEPAPER_RUBRIC, ESSAY_RUBRIC, RESEARCH_PAPER_RUBRIC


# Forward reference for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass


@dataclass
class EditSuggestion:
    """A suggested edit."""
    original: str
    replacement: str
    reason: str
    location: int  # Character position
    applied: bool = False


@dataclass
class RepetitionIssue:
    """A repetition detected in the document."""
    text: str
    count: int
    locations: list[int]
    suggestion: str
    issue_type: str = "phrase"  # "phrase", "statistic", "claim"


@dataclass
class GapIssue:
    """An unclosed gap or unresolved topic."""
    topic: str
    introduced_at: str  # Section/location where introduced
    issue: str  # What's missing
    suggestion: str


@dataclass
class DriftIssue:
    """Drift from the original prompt."""
    expected: str
    actual: str
    severity: str  # "minor", "moderate", "major"
    suggestion: str


@dataclass
class EditorResult:
    """Result from the Editor agent."""
    original_title: str

    # Cleaned document
    cleaned_document: str

    # Stats
    edits_made: int
    edits_suggested: int
    words_removed: int

    # Issues found (required fields first)
    grammar_fixes: list[EditSuggestion]
    repetitions: list[RepetitionIssue]
    citation_concerns: list[CitationCheck]

    # Scores (required fields)
    original_grade: str
    improved_grade: str
    citation_confidence: float

    # Report (required)
    editorial_report: str

    # Fields with defaults must come last
    gaps: list[GapIssue] = field(default_factory=list)
    drift_issues: list[DriftIssue] = field(default_factory=list)
    prompt_alignment: float = 1.0  # 0-1 score for how well it matches prompt
    original_prompt: str = ""

    # Metadata
    edited_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def get_full_output(self) -> str:
        """Get cleaned document with editorial report appended."""
        return f"""{self.cleaned_document}

---

{self.editorial_report}
"""

    def save(self, directory: str) -> dict:
        """Save all outputs."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        saved = {}

        # Save cleaned document
        clean_path = directory / "cleaned_document.md"
        clean_path.write_text(self.cleaned_document)
        saved["cleaned"] = str(clean_path)

        # Save full output (document + report)
        full_path = directory / "full_output.md"
        full_path.write_text(self.get_full_output())
        saved["full"] = str(full_path)

        # Save just the report
        report_path = directory / "editorial_report.md"
        report_path.write_text(self.editorial_report)
        saved["report"] = str(report_path)

        return saved


class Editor:
    """
    Editor agent that cleans documents and produces editorial reports.

    Features:
    - Grammar and spelling fixes
    - Repetition detection (phrases, statistics, claims)
    - Citation verification (detects hallucinated sources)
    - Gap detection (unclosed topics)
    - Prompt drift checking
    - Rubric-based assessment
    - Editorial report generation
    """

    # Words/phrases that are often repeated unnecessarily
    FILLER_PHRASES = [
        r'\b(very|really|quite|rather|somewhat)\s+(very|really|quite|rather|somewhat)\b',
        r'\b(in\s+order)\s+to\b',  # "in order to" -> "to"
        r'\b(due\s+to\s+the\s+fact\s+that)\b',  # -> "because"
        r'\b(at\s+this\s+point\s+in\s+time)\b',  # -> "now"
        r'\b(in\s+the\s+event\s+that)\b',  # -> "if"
        r'\b(it\s+is\s+important\s+to\s+note\s+that)\b',  # often unnecessary
    ]

    # Simple grammar fixes
    GRAMMAR_FIXES = [
        (r'\s{2,}', ' ', "Multiple spaces"),
        (r'\.{2}(?!\.)', '.', "Double period"),
        (r'\bi\b', 'I', "Uncapitalized I"),
        (r'(\w)\s+,', r'\1,', "Space before comma"),
        (r'(\w)\s+\.', r'\1.', "Space before period"),
        (r',([^\s])', r', \1', "Missing space after comma"),
        (r'\.([A-Z])', r'. \1', "Missing space after period"),
    ]

    # Patterns for statistics/claims that shouldn't be repeated
    STATISTIC_PATTERNS = [
        r'\d+(?:\.\d+)?\s*%',  # Percentages
        r'\$\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:billion|million|thousand))?',  # Dollar amounts
        r'\d+(?:,\d{3})*(?:\.\d+)?\s*(?:billion|million|thousand)',  # Large numbers
        r'(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:in|out\s+of)\s+(?:ten|\d+)',  # "X out of Y"
    ]

    # Topic introduction patterns
    TOPIC_INTRO_PATTERNS = [
        (r'(?:this\s+(?:paper|article|study|report)\s+(?:will\s+)?(?:examine|explore|discuss|address|analyze))\s+([^.]+)', 'examine'),
        (r'(?:we\s+will\s+(?:examine|explore|discuss|address|analyze))\s+([^.]+)', 'examine'),
        (r'(?:the\s+following\s+(?:section|chapter)s?\s+(?:will\s+)?(?:cover|discuss|examine))\s+([^.]+)', 'section'),
        (r'(?:key\s+(?:topics?|areas?|issues?)\s+include)\s*:?\s*([^.]+)', 'topics'),
    ]

    def __init__(self, llm=None, auto_fix_grammar: bool = True):
        """
        Args:
            llm: Optional LLM for deeper analysis
            auto_fix_grammar: Whether to automatically apply grammar fixes
        """
        self.llm = llm
        self.auto_fix_grammar = auto_fix_grammar
        self.grader = RubricGrader(verify_citations=True)

    def edit(self, document: str, title: str = "Document",
             rubric_type: str = "auto", prompt: str = "") -> EditorResult:
        """
        Edit a document and produce an editorial report.

        Args:
            document: Document text
            title: Document title
            rubric_type: "essay", "research", "whitepaper", or "auto"
            prompt: Original prompt/request (for drift checking)

        Returns:
            EditorResult with cleaned document and report
        """
        # Select rubric
        if rubric_type == "auto":
            rubric = self._detect_rubric(document)
        elif rubric_type == "essay":
            rubric = ESSAY_RUBRIC
        elif rubric_type == "research":
            rubric = RESEARCH_PAPER_RUBRIC
        else:
            rubric = WHITEPAPER_RUBRIC

        self.grader.rubric = rubric

        # Grade original
        original_result = self.grader.grade(document, title)
        original_grade = original_result.letter_grade

        # Track changes
        cleaned = document
        grammar_fixes = []
        edits_made = 0
        words_original = len(document.split())

        # 1. Apply grammar fixes
        if self.auto_fix_grammar:
            cleaned, grammar_fixes, edits_made = self._apply_grammar_fixes(cleaned)

        # 2. Detect repetitions (including repeated stats/claims)
        repetitions = self._detect_repetitions(cleaned)
        stat_repetitions = self._detect_repeated_statistics(cleaned)
        repetitions.extend(stat_repetitions)

        # 3. Clean up repetitive phrases
        cleaned, phrase_removals = self._clean_repetitive_phrases(cleaned)
        edits_made += phrase_removals

        # 4. Verify citations
        citation_concerns = original_result.citation_checks
        suspicious_citations = [c for c in citation_concerns if not c.appears_valid]

        # 5. Detect gaps (unclosed topics)
        gaps = self._detect_gaps(cleaned)

        # 6. Check prompt drift (if prompt provided)
        drift_issues = []
        prompt_alignment = 1.0
        if prompt:
            drift_issues, prompt_alignment = self._check_prompt_drift(cleaned, prompt)

        # 7. Grade improved version
        improved_result = self.grader.grade(cleaned, title)
        improved_grade = improved_result.letter_grade

        # Calculate stats
        words_cleaned = len(cleaned.split())
        words_removed = words_original - words_cleaned

        # 8. Generate editorial report
        editorial_report = self._generate_report(
            title=title,
            original_grade=original_grade,
            improved_grade=improved_grade,
            edits_made=edits_made,
            words_removed=words_removed,
            grammar_fixes=grammar_fixes,
            repetitions=repetitions,
            citation_concerns=suspicious_citations,
            citation_confidence=original_result.citation_confidence,
            rubric_name=rubric.name,
            gaps=gaps,
            drift_issues=drift_issues,
            prompt_alignment=prompt_alignment,
            prompt=prompt,
        )

        return EditorResult(
            original_title=title,
            cleaned_document=cleaned,
            edits_made=edits_made,
            edits_suggested=len(grammar_fixes) + len(repetitions),
            words_removed=words_removed,
            grammar_fixes=grammar_fixes,
            repetitions=repetitions,
            citation_concerns=suspicious_citations,
            gaps=gaps,
            drift_issues=drift_issues,
            original_grade=original_grade,
            improved_grade=improved_grade,
            citation_confidence=original_result.citation_confidence,
            prompt_alignment=prompt_alignment,
            original_prompt=prompt,
            editorial_report=editorial_report,
        )

    def _detect_rubric(self, document: str) -> object:
        """Auto-detect appropriate rubric based on document content."""
        doc_lower = document.lower()

        # Whitepaper indicators
        whitepaper_markers = ['executive summary', 'problem statement', 'solution overview',
                             'implementation', 'roi', 'enterprise']
        whitepaper_score = sum(1 for m in whitepaper_markers if m in doc_lower)

        # Research paper indicators
        research_markers = ['abstract', 'methodology', 'literature review', 'hypothesis',
                           'findings', 'discussion', 'implications']
        research_score = sum(1 for m in research_markers if m in doc_lower)

        # Essay indicators
        essay_markers = ['thesis', 'argument', 'in conclusion', 'therefore', 'essay']
        essay_score = sum(1 for m in essay_markers if m in doc_lower)

        if whitepaper_score >= research_score and whitepaper_score >= essay_score:
            return WHITEPAPER_RUBRIC
        elif research_score >= essay_score:
            return RESEARCH_PAPER_RUBRIC
        else:
            return ESSAY_RUBRIC

    def _apply_grammar_fixes(self, text: str) -> tuple[str, list[EditSuggestion], int]:
        """Apply grammar fixes and track changes."""
        fixes = []
        edits = 0
        result = text

        for pattern, replacement, reason in self.GRAMMAR_FIXES:
            matches = list(re.finditer(pattern, result))

            for match in matches[:10]:  # Limit per rule
                original = match.group()
                new_text = re.sub(pattern, replacement, original)

                if original != new_text:
                    fixes.append(EditSuggestion(
                        original=original[:30],
                        replacement=new_text[:30],
                        reason=reason,
                        location=match.start(),
                        applied=True,
                    ))

            new_result = re.sub(pattern, replacement, result)
            if new_result != result:
                edits += len(matches)
                result = new_result

        return result, fixes, edits

    def _detect_repetitions(self, text: str) -> list[RepetitionIssue]:
        """Detect repeated phrases and words."""
        repetitions = []

        # Find repeated phrases (3+ words appearing 2+ times)
        words = text.split()
        phrase_counts = {}

        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3]).lower()
            # Clean punctuation for comparison
            clean_phrase = re.sub(r'[^\w\s]', '', phrase)
            if len(clean_phrase) > 10:  # Meaningful phrase
                if clean_phrase not in phrase_counts:
                    phrase_counts[clean_phrase] = []
                phrase_counts[clean_phrase].append(i)

        for phrase, locations in phrase_counts.items():
            if len(locations) >= 3:  # Repeated 3+ times
                repetitions.append(RepetitionIssue(
                    text=phrase,
                    count=len(locations),
                    locations=locations[:5],
                    suggestion=f"Phrase '{phrase}' appears {len(locations)} times - consider varying language",
                ))

        # Find repeated sentences
        sentences = re.split(r'[.!?]+', text)
        sentence_counts = {}

        for i, sent in enumerate(sentences):
            sent_clean = sent.strip().lower()
            if len(sent_clean) > 30:
                if sent_clean not in sentence_counts:
                    sentence_counts[sent_clean] = []
                sentence_counts[sent_clean].append(i)

        for sent, locations in sentence_counts.items():
            if len(locations) >= 2:
                repetitions.append(RepetitionIssue(
                    text=sent[:50] + "...",
                    count=len(locations),
                    locations=locations,
                    suggestion="Duplicate sentence - remove or rephrase",
                ))

        return repetitions[:10]  # Limit

    def _detect_repeated_statistics(self, text: str) -> list[RepetitionIssue]:
        """Detect repeated statistics, percentages, and claims."""
        repetitions = []

        # Find all statistics
        stats_found = {}

        for pattern in self.STATISTIC_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                stat = match.group().strip()
                # Normalize the stat for comparison
                stat_normalized = re.sub(r'\s+', ' ', stat.lower())

                if stat_normalized not in stats_found:
                    stats_found[stat_normalized] = {
                        'original': stat,
                        'locations': [],
                        'contexts': []
                    }

                stats_found[stat_normalized]['locations'].append(match.start())

                # Get surrounding context
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].replace('\n', ' ')
                stats_found[stat_normalized]['contexts'].append(context)

        # Flag statistics that appear more than once
        for stat_norm, data in stats_found.items():
            if len(data['locations']) >= 2:
                # Check if the contexts suggest it's the same type of claim
                # (not just same number appearing in different unrelated contexts)

                # Flag if:
                # 1. Same stat appears 3+ times (likely problematic regardless)
                # 2. Or appears 2+ times with similar surrounding words
                should_flag = False

                if len(data['locations']) >= 3:
                    # 3+ occurrences - always flag
                    should_flag = True
                else:
                    # 2 occurrences - check if contexts are similar
                    ctx1_words = set(data['contexts'][0].lower().split())
                    ctx2_words = set(data['contexts'][1].lower().split())
                    overlap = len(ctx1_words & ctx2_words)
                    # If more than 3 words overlap (besides common words), flag
                    common = {'the', 'a', 'an', 'is', 'are', 'of', 'to', 'in', 'and', 'that', 'this'}
                    meaningful_overlap = len((ctx1_words & ctx2_words) - common)
                    if meaningful_overlap >= 3:
                        should_flag = True

                if should_flag:
                    repetitions.append(RepetitionIssue(
                        text=data['original'],
                        count=len(data['locations']),
                        locations=data['locations'][:5],
                        suggestion=f"Statistic '{data['original']}' appears {len(data['locations'])} times - consider consolidating or varying presentation",
                        issue_type="statistic"
                    ))

        # Also detect repeated claims (sentences with similar structure)
        sentences = re.split(r'[.!?]+', text)
        claim_patterns = {}

        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if len(sent) < 20:
                continue

            # Extract the claim pattern (first 5 words + any numbers)
            words = sent.split()[:8]
            pattern = ' '.join(w if not w.isdigit() else '#' for w in words).lower()

            if pattern not in claim_patterns:
                claim_patterns[pattern] = []
            claim_patterns[pattern].append((i, sent))

        for pattern, occurrences in claim_patterns.items():
            if len(occurrences) >= 2:
                # Check if these are actually similar claims
                first_sent = occurrences[0][1]
                similar_count = sum(1 for _, s in occurrences if self._sentence_similarity(first_sent, s) > 0.6)

                if similar_count >= 2:
                    repetitions.append(RepetitionIssue(
                        text=occurrences[0][1][:60] + "...",
                        count=similar_count,
                        locations=[o[0] for o in occurrences[:5]],
                        suggestion=f"Similar claim appears {similar_count} times - consolidate into single authoritative statement",
                        issue_type="claim"
                    ))

        return repetitions[:5]

    def _sentence_similarity(self, s1: str, s2: str) -> float:
        """Simple word overlap similarity."""
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _detect_gaps(self, text: str) -> list[GapIssue]:
        """Detect unclosed topics - things promised but not delivered."""
        gaps = []

        # Find topics introduced
        topics_introduced = []

        for pattern, intro_type in self.TOPIC_INTRO_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                topic = match.group(1).strip()
                topics_introduced.append({
                    'topic': topic,
                    'type': intro_type,
                    'location': match.start(),
                    'context': match.group()
                })

        # Check if introduced topics are actually covered
        for topic_info in topics_introduced:
            topic = topic_info['topic']
            # Extract key terms from topic
            key_terms = [w.lower() for w in topic.split() if len(w) > 4]

            if not key_terms:
                continue

            # Check if these terms appear later in the document
            rest_of_doc = text[topic_info['location'] + 100:]  # Skip intro area

            terms_found = sum(1 for term in key_terms if term in rest_of_doc.lower())
            coverage = terms_found / len(key_terms) if key_terms else 0

            if coverage < 0.3:
                gaps.append(GapIssue(
                    topic=topic[:100],
                    introduced_at=topic_info['context'][:60] + "...",
                    issue=f"Topic mentioned but only {coverage:.0%} of key terms appear in subsequent content",
                    suggestion=f"Add section covering: {topic[:50]}..."
                ))

        # Check for common structural gaps
        structural_checks = [
            ('problem', 'solution', "Problem described but solution unclear"),
            ('objective', 'result', "Objectives stated but results not shown"),
            ('method', 'finding', "Methods described but findings missing"),
            ('hypothesis', 'conclusion', "Hypothesis stated but not concluded"),
            ('question', 'answer', "Questions raised but not answered"),
        ]

        text_lower = text.lower()

        for intro_term, resolution_term, issue in structural_checks:
            has_intro = intro_term in text_lower
            has_resolution = resolution_term in text_lower

            if has_intro and not has_resolution:
                gaps.append(GapIssue(
                    topic=intro_term.capitalize(),
                    introduced_at="Document structure",
                    issue=issue,
                    suggestion=f"Add content that addresses the {resolution_term}"
                ))

        return gaps[:5]

    def _check_prompt_drift(self, text: str, prompt: str) -> tuple[list[DriftIssue], float]:
        """Check how well the document addresses the original prompt."""
        drift_issues = []

        # Extract key terms from prompt
        prompt_lower = prompt.lower()
        prompt_words = set(w for w in re.findall(r'\b\w{4,}\b', prompt_lower))

        # Remove common words
        stop_words = {'this', 'that', 'with', 'from', 'have', 'will', 'about', 'what',
                     'your', 'they', 'their', 'would', 'could', 'should', 'which',
                     'there', 'where', 'when', 'into', 'more', 'some', 'than',
                     'write', 'create', 'make', 'help', 'please'}
        prompt_terms = prompt_words - stop_words

        if not prompt_terms:
            return [], 1.0

        # Check coverage of prompt terms in document
        text_lower = text.lower()
        terms_covered = sum(1 for term in prompt_terms if term in text_lower)
        coverage = terms_covered / len(prompt_terms)

        # Check for specific topics mentioned in prompt
        topic_patterns = [
            (r'(?:about|regarding|on|for)\s+([^,.:]+)', 'topic'),
            (r'(?:addressing|covering|including)\s+([^,.:]+)', 'scope'),
            (r'(?:such\s+as|like|including)\s+([^,.:]+)', 'examples'),
        ]

        expected_topics = []
        for pattern, topic_type in topic_patterns:
            for match in re.finditer(pattern, prompt_lower):
                expected_topics.append(match.group(1).strip())

        # Check if expected topics are covered
        for topic in expected_topics[:5]:
            topic_terms = [w for w in topic.split() if len(w) > 3]
            if topic_terms:
                topic_coverage = sum(1 for t in topic_terms if t in text_lower) / len(topic_terms)

                if topic_coverage < 0.5:
                    drift_issues.append(DriftIssue(
                        expected=topic,
                        actual=f"Only {topic_coverage:.0%} coverage",
                        severity="moderate" if topic_coverage > 0.2 else "major",
                        suggestion=f"Add more content addressing: {topic}"
                    ))

        # Check for prompt-specified requirements
        requirement_patterns = [
            (r'(?:must|should|need\s+to)\s+(?:include|have|contain)\s+([^,.:]+)', 'requirement'),
            (r'(?:make\s+sure|ensure|be\s+sure)\s+(?:to|that)?\s*([^,.:]+)', 'requirement'),
        ]

        for pattern, req_type in requirement_patterns:
            for match in re.finditer(pattern, prompt_lower):
                requirement = match.group(1).strip()
                req_terms = [w for w in requirement.split() if len(w) > 3]

                if req_terms:
                    req_coverage = sum(1 for t in req_terms if t in text_lower) / len(req_terms)

                    if req_coverage < 0.5:
                        drift_issues.append(DriftIssue(
                            expected=requirement,
                            actual="Not adequately addressed",
                            severity="major",
                            suggestion=f"Address requirement: {requirement}"
                        ))

        # Calculate overall alignment score
        base_score = coverage
        penalty = len([d for d in drift_issues if d.severity == "major"]) * 0.15
        penalty += len([d for d in drift_issues if d.severity == "moderate"]) * 0.05

        alignment = max(0.0, min(1.0, base_score - penalty))

        return drift_issues[:5], alignment

    def _clean_repetitive_phrases(self, text: str) -> tuple[str, int]:
        """Remove unnecessarily wordy phrases."""
        result = text
        removals = 0

        # Simple replacements for wordy phrases
        replacements = [
            (r'\bin\s+order\s+to\b', 'to'),
            (r'\bdue\s+to\s+the\s+fact\s+that\b', 'because'),
            (r'\bat\s+this\s+point\s+in\s+time\b', 'now'),
            (r'\bin\s+the\s+event\s+that\b', 'if'),
            (r'\bfor\s+the\s+purpose\s+of\b', 'to'),
            (r'\bwith\s+regard\s+to\b', 'regarding'),
            (r'\bin\s+spite\s+of\s+the\s+fact\s+that\b', 'although'),
            (r'\b(very|really)\s+(very|really)\b', r'\2'),  # Remove doubled intensifiers
        ]

        for pattern, replacement in replacements:
            matches = re.findall(pattern, result, re.IGNORECASE)
            if matches:
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
                removals += len(matches)

        return result, removals

    def _generate_report(self, title: str, original_grade: str, improved_grade: str,
                        edits_made: int, words_removed: int, grammar_fixes: list,
                        repetitions: list, citation_concerns: list,
                        citation_confidence: float, rubric_name: str,
                        gaps: list = None, drift_issues: list = None,
                        prompt_alignment: float = 1.0, prompt: str = "") -> str:
        """Generate the editorial report."""
        gaps = gaps or []
        drift_issues = drift_issues or []

        # Determine status
        major_issues = []
        if citation_confidence < 0.5:
            major_issues.append("citation concerns")
        if len([r for r in repetitions if r.issue_type == "statistic"]) > 0:
            major_issues.append("repeated statistics")
        if len(gaps) > 2:
            major_issues.append("unclosed topics")
        if prompt_alignment < 0.7:
            major_issues.append("prompt drift")

        if major_issues or original_grade in ['D', 'D+', 'D-', 'F']:
            status = f"REQUIRES ATTENTION - {', '.join(major_issues) if major_issues else 'Low grade'}"
            status_emoji = "🔴"
        elif original_grade in ['C', 'C+', 'C-'] or len(gaps) > 0:
            status = "ACCEPTABLE WITH REVISIONS"
            status_emoji = "🟡"
        else:
            status = "READY FOR REVIEW"
            status_emoji = "🟢"

        report = f"""# Editorial Report: {title}

**Status:** {status_emoji} {status}
**Rubric:** {rubric_name}
**Edited:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## Quality Assessment

| Metric | Value |
|--------|-------|
| Original Grade | {original_grade} |
| After Cleanup | {improved_grade} |
| Edits Applied | {edits_made} |
| Words Removed | {words_removed} |
| Citation Confidence | {citation_confidence:.0%} |
| Prompt Alignment | {prompt_alignment:.0%} |
| Unclosed Topics | {len(gaps)} |
| Repeated Stats | {len([r for r in repetitions if r.issue_type == "statistic"])} |

---

## Edits Applied

"""

        if grammar_fixes:
            report += f"### Grammar & Mechanics ({len(grammar_fixes)} fixes)\n\n"
            for fix in grammar_fixes[:5]:
                report += f"- **{fix.reason}:** `{fix.original}` → `{fix.replacement}`\n"
            if len(grammar_fixes) > 5:
                report += f"- ... and {len(grammar_fixes) - 5} more\n"
            report += "\n"
        else:
            report += "### Grammar & Mechanics\n\nNo significant issues found.\n\n"

        if repetitions:
            report += f"### Repetition Issues ({len(repetitions)} found)\n\n"
            for rep in repetitions[:5]:
                report += f"- **\"{rep.text}\"** appears {rep.count} times\n"
                report += f"  - {rep.suggestion}\n"
            report += "\n"
        else:
            report += "### Repetition\n\nNo excessive repetition detected.\n\n"

        report += "---\n\n## Citation Review\n\n"

        if citation_concerns:
            report += f"**⚠️ {len(citation_concerns)} potentially problematic citations found:**\n\n"
            for cit in citation_concerns[:5]:
                report += f"### Suspicious Citation\n"
                report += f"```\n{cit.citation_text[:100]}...\n```\n"
                report += "**Issues:**\n"
                for issue in cit.issues:
                    report += f"- {issue}\n"
                report += f"**Confidence:** {cit.confidence:.0%}\n\n"

            report += """**Recommendation:** Verify these citations against actual sources.
AI-generated content sometimes includes plausible-sounding but non-existent references.

"""
        else:
            report += "All citations appear properly formatted. Manual verification still recommended.\n\n"

        # Separate repeated statistics section
        stat_reps = [r for r in repetitions if r.issue_type == "statistic"]
        claim_reps = [r for r in repetitions if r.issue_type == "claim"]

        if stat_reps or claim_reps:
            report += "---\n\n## Repeated Statistics & Claims\n\n"
            report += "**⚠️ The following statistics/claims appear multiple times:**\n\n"

            if stat_reps:
                report += "### Repeated Statistics\n\n"
                for rep in stat_reps:
                    report += f"- **{rep.text}** appears {rep.count} times\n"
                    report += f"  - *Impact:* Reader may question credibility when same stat is repeated\n"
                    report += f"  - *Fix:* {rep.suggestion}\n\n"

            if claim_reps:
                report += "### Repeated Claims\n\n"
                for rep in claim_reps:
                    report += f"- **\"{rep.text}\"** appears {rep.count} times\n"
                    report += f"  - *Fix:* {rep.suggestion}\n\n"

        # Gap detection section
        if gaps:
            report += "---\n\n## Unclosed Topics (Gaps)\n\n"
            report += "**The following topics were introduced but not fully addressed:**\n\n"

            for gap in gaps:
                report += f"### {gap.topic[:50]}\n"
                report += f"- **Introduced at:** {gap.introduced_at}\n"
                report += f"- **Issue:** {gap.issue}\n"
                report += f"- **Suggestion:** {gap.suggestion}\n\n"

        # Prompt drift section
        if prompt:
            report += "---\n\n## Prompt Alignment\n\n"

            alignment_status = "🟢 Excellent" if prompt_alignment >= 0.8 else "🟡 Acceptable" if prompt_alignment >= 0.6 else "🔴 Poor"
            report += f"**Alignment Score:** {prompt_alignment:.0%} ({alignment_status})\n\n"

            if drift_issues:
                report += "**Drift Issues Found:**\n\n"
                for drift in drift_issues:
                    severity_icon = "🔴" if drift.severity == "major" else "🟡" if drift.severity == "moderate" else "⚪"
                    report += f"- {severity_icon} **Expected:** {drift.expected}\n"
                    report += f"  - **Actual:** {drift.actual}\n"
                    report += f"  - **Fix:** {drift.suggestion}\n\n"
            else:
                report += "No significant drift from original prompt detected.\n\n"

            # Show original prompt (truncated)
            report += f"**Original Prompt:** {prompt[:200]}{'...' if len(prompt) > 200 else ''}\n\n"

        report += """---

## Recommendations

"""

        recommendations = []

        if citation_confidence < 0.7:
            recommendations.append("🔴 **HIGH PRIORITY:** Verify all citations against actual sources before publication")

        if stat_reps:
            recommendations.append(f"🔴 **HIGH PRIORITY:** Consolidate {len(stat_reps)} repeated statistics - same data point should appear once")

        if gaps:
            recommendations.append(f"🟡 Address {len(gaps)} unclosed topic(s) - promises made but not delivered")

        if prompt_alignment < 0.7:
            recommendations.append(f"🟡 Document drifted from prompt (only {prompt_alignment:.0%} alignment) - refocus content")

        phrase_reps = [r for r in repetitions if r.issue_type == "phrase"]
        if phrase_reps:
            recommendations.append(f"🟡 Address {len(phrase_reps)} phrase repetition issues for better readability")

        if original_grade in ['C', 'C+', 'C-', 'D', 'D+', 'D-', 'F']:
            recommendations.append("🟡 Consider strengthening arguments with more evidence")

        if not recommendations:
            recommendations.append("🟢 Document is in good shape - minor polish recommended")

        for rec in recommendations:
            report += f"- {rec}\n"

        report += """

---

*Generated by SVEND Editor Agent*
"""

        return report


def quick_edit(document: str, title: str = "Document") -> EditorResult:
    """
    Quick one-liner editing.

    Args:
        document: Document text
        title: Document title

    Returns:
        EditorResult

    Example:
        from reviewer import quick_edit

        result = quick_edit(my_document, "My Whitepaper")
        print(result.cleaned_document)
        print(result.editorial_report)
    """
    editor = Editor()
    return editor.edit(document, title)
