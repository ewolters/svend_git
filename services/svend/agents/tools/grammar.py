"""
Grammar and Style Checker

Deterministic text quality analysis:
- Spelling errors
- Common grammar mistakes
- Style issues (passive voice, wordiness)
- Sentence structure analysis

No AI required - rule-based checking.
"""

import re
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class GrammarIssue:
    """A grammar or style issue found in text."""
    type: str  # spelling, grammar, style
    severity: str  # error, warning, suggestion
    message: str
    position: int  # Character position
    context: str  # Surrounding text
    suggestion: str = ""


@dataclass
class GrammarResult:
    """Result of grammar analysis."""
    text: str
    issues: list[GrammarIssue] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return len([i for i in self.issues if i.severity == "error"])

    @property
    def warning_count(self) -> int:
        return len([i for i in self.issues if i.severity == "warning"])

    @property
    def score(self) -> float:
        """Grammar score 0-100."""
        if not self.text.strip():
            return 100.0
        words = len(self.text.split())
        if words == 0:
            return 100.0
        # Deduct points for issues
        deductions = self.error_count * 5 + self.warning_count * 2
        return max(0, 100 - (deductions / words * 100))

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "errors": self.error_count,
            "warnings": self.warning_count,
            "issues": [
                {
                    "type": i.type,
                    "severity": i.severity,
                    "message": i.message,
                    "suggestion": i.suggestion,
                }
                for i in self.issues
            ],
            "stats": self.stats,
        }

    def summary(self) -> str:
        lines = [
            f"Grammar Score: {self.score:.0f}/100",
            f"Errors: {self.error_count}",
            f"Warnings: {self.warning_count}",
        ]
        if self.issues:
            lines.append("\nIssues:")
            for issue in self.issues[:10]:  # Top 10
                icon = "!" if issue.severity == "error" else "~"
                lines.append(f"  [{icon}] {issue.message}")
                if issue.suggestion:
                    lines.append(f"      Suggestion: {issue.suggestion}")
        return "\n".join(lines)


class GrammarChecker:
    """
    Rule-based grammar and style checker.

    Checks for:
    - Spelling errors (using pyspellchecker)
    - Common grammar mistakes
    - Style issues (passive voice, wordiness, etc.)
    - Sentence structure
    """

    # Common confused words
    CONFUSED_WORDS = {
        r"\btheir\s+(?:is|are|was|were)\b": ("their", "there", "their + verb should be 'there'"),
        r"\bthey're\s+(?:own|house|car)\b": ("they're", "their", "they're = they are, use 'their' for possession"),
        r"\bits\s+(?:a|the|very|quite)\b": ("its", "it's", "its = possessive, it's = it is"),
        r"\byour\s+(?:welcome|right|wrong)\b": ("your", "you're", "your = possessive, you're = you are"),
        r"\beffect\s+(?:the|a|on)\b": ("effect", "affect", "effect = noun, affect = verb"),
        r"\bthen\s+(?:I|you|we|they)\b": ("then", "than", "then = time, than = comparison"),
        r"\bwould\s+of\b": ("would of", "would have", "use 'would have' not 'would of'"),
        r"\bcould\s+of\b": ("could of", "could have", "use 'could have' not 'could of'"),
        r"\bshould\s+of\b": ("should of", "should have", "use 'should have' not 'should of'"),
    }

    # Wordy phrases
    WORDY_PHRASES = {
        "in order to": "to",
        "due to the fact that": "because",
        "at this point in time": "now",
        "in the event that": "if",
        "for the purpose of": "to",
        "in spite of the fact that": "although",
        "a large number of": "many",
        "in close proximity to": "near",
        "has the ability to": "can",
        "is able to": "can",
        "on a daily basis": "daily",
        "at the present time": "now",
        "in the near future": "soon",
        "make a decision": "decide",
        "take into consideration": "consider",
    }

    # Passive voice patterns
    PASSIVE_PATTERNS = [
        r"\b(?:is|are|was|were|been|being)\s+\w+ed\b",
        r"\b(?:is|are|was|were|been|being)\s+(?:made|done|taken|given|shown|found|seen)\b",
    ]

    def __init__(self, use_spellcheck: bool = True):
        self.use_spellcheck = use_spellcheck
        self._spell = None

    @property
    def spell(self):
        if self._spell is None and self.use_spellcheck:
            try:
                from spellchecker import SpellChecker
                self._spell = SpellChecker()
            except ImportError:
                self.use_spellcheck = False
        return self._spell

    def check(self, text: str) -> GrammarResult:
        """Run all grammar checks on text."""
        issues = []

        # Spelling check
        if self.use_spellcheck:
            issues.extend(self._check_spelling(text))

        # Grammar rules
        issues.extend(self._check_confused_words(text))
        issues.extend(self._check_double_words(text))

        # Style checks
        issues.extend(self._check_wordy_phrases(text))
        issues.extend(self._check_passive_voice(text))
        issues.extend(self._check_sentence_length(text))

        # Calculate stats
        stats = self._calculate_stats(text)

        return GrammarResult(
            text=text,
            issues=issues,
            stats=stats,
        )

    def _check_spelling(self, text: str) -> list[GrammarIssue]:
        """Check for spelling errors."""
        issues = []
        if not self.spell:
            return issues

        # Extract words (ignore code blocks, URLs, etc.)
        clean_text = re.sub(r'```[\s\S]*?```', '', text)  # Remove code blocks
        clean_text = re.sub(r'https?://\S+', '', clean_text)  # Remove URLs
        clean_text = re.sub(r'`[^`]+`', '', clean_text)  # Remove inline code

        words = re.findall(r'\b[a-zA-Z]+\b', clean_text)

        # Filter out likely proper nouns and technical terms
        words = [w for w in words if w[0].islower() and len(w) > 2]

        misspelled = self.spell.unknown(words)

        for word in misspelled:
            # Find position in original text
            match = re.search(rf'\b{re.escape(word)}\b', text, re.IGNORECASE)
            pos = match.start() if match else 0

            correction = self.spell.correction(word)
            issues.append(GrammarIssue(
                type="spelling",
                severity="error",
                message=f"Possible misspelling: '{word}'",
                position=pos,
                context=self._get_context(text, pos),
                suggestion=correction if correction != word else "",
            ))

        return issues

    def _check_confused_words(self, text: str) -> list[GrammarIssue]:
        """Check for commonly confused words."""
        issues = []

        for pattern, (wrong, right, explanation) in self.CONFUSED_WORDS.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                issues.append(GrammarIssue(
                    type="grammar",
                    severity="error",
                    message=f"Confused word: {explanation}",
                    position=match.start(),
                    context=self._get_context(text, match.start()),
                    suggestion=f"Did you mean '{right}'?",
                ))

        return issues

    def _check_double_words(self, text: str) -> list[GrammarIssue]:
        """Check for repeated words (the the, a a, etc.)."""
        issues = []

        pattern = r'\b(\w+)\s+\1\b'
        for match in re.finditer(pattern, text, re.IGNORECASE):
            word = match.group(1)
            # Skip intentional repetitions
            if word.lower() in ['that', 'had', 'very']:
                continue

            issues.append(GrammarIssue(
                type="grammar",
                severity="warning",
                message=f"Repeated word: '{word} {word}'",
                position=match.start(),
                context=self._get_context(text, match.start()),
                suggestion=f"Remove duplicate '{word}'",
            ))

        return issues

    def _check_wordy_phrases(self, text: str) -> list[GrammarIssue]:
        """Check for wordy phrases that can be simplified."""
        issues = []
        text_lower = text.lower()

        for phrase, replacement in self.WORDY_PHRASES.items():
            pattern = re.escape(phrase)
            for match in re.finditer(pattern, text_lower):
                issues.append(GrammarIssue(
                    type="style",
                    severity="suggestion",
                    message=f"Wordy phrase: '{phrase}'",
                    position=match.start(),
                    context=self._get_context(text, match.start()),
                    suggestion=f"Consider using '{replacement}' instead",
                ))

        return issues

    def _check_passive_voice(self, text: str) -> list[GrammarIssue]:
        """Check for passive voice usage."""
        issues = []

        for pattern in self.PASSIVE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                issues.append(GrammarIssue(
                    type="style",
                    severity="suggestion",
                    message="Possible passive voice",
                    position=match.start(),
                    context=self._get_context(text, match.start()),
                    suggestion="Consider using active voice for clarity",
                ))

        return issues

    def _check_sentence_length(self, text: str) -> list[GrammarIssue]:
        """Check for overly long sentences."""
        issues = []

        # Split into sentences
        sentences = re.split(r'[.!?]+', text)

        for sentence in sentences:
            words = sentence.split()
            if len(words) > 40:
                issues.append(GrammarIssue(
                    type="style",
                    severity="warning",
                    message=f"Long sentence ({len(words)} words)",
                    position=text.find(sentence[:50]),
                    context=sentence[:80] + "...",
                    suggestion="Consider breaking into shorter sentences",
                ))

        return issues

    def _get_context(self, text: str, position: int, window: int = 30) -> str:
        """Get surrounding context for an issue."""
        start = max(0, position - window)
        end = min(len(text), position + window)
        context = text[start:end]
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."
        return context

    def _calculate_stats(self, text: str) -> dict:
        """Calculate text statistics."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        words = text.split()
        word_lengths = [len(w) for w in words]

        return {
            "sentences": len(sentences),
            "words": len(words),
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
            "avg_word_length": sum(word_lengths) / len(word_lengths) if word_lengths else 0,
        }


def check_grammar(text: str) -> GrammarResult:
    """Quick helper to check grammar."""
    checker = GrammarChecker()
    return checker.check(text)


def main():
    """CLI for grammar checking."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python grammar.py <text or file>")
        sys.exit(1)

    arg = sys.argv[1]

    # Check if it's a file
    try:
        with open(arg) as f:
            text = f.read()
    except FileNotFoundError:
        text = arg

    result = check_grammar(text)
    print(result.summary())


if __name__ == "__main__":
    main()
