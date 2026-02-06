"""
Readability Scorer

Calculates readability metrics:
- Flesch Reading Ease
- Flesch-Kincaid Grade Level
- Gunning Fog Index
- SMOG Index
- Coleman-Liau Index
- Automated Readability Index (ARI)

No external dependencies - pure Python implementation.
"""

import re
from dataclasses import dataclass


@dataclass
class ReadabilityResult:
    """Results from readability analysis."""
    flesch_reading_ease: float      # 0-100, higher = easier
    flesch_kincaid_grade: float     # US grade level
    gunning_fog: float              # Years of education needed
    smog_index: float               # Years of education needed
    coleman_liau: float             # US grade level
    ari: float                      # US grade level

    # Raw stats
    word_count: int
    sentence_count: int
    syllable_count: int
    complex_word_count: int         # 3+ syllables
    char_count: int

    @property
    def average_grade_level(self) -> float:
        """Average of grade-level metrics."""
        grades = [
            self.flesch_kincaid_grade,
            self.gunning_fog,
            self.smog_index,
            self.coleman_liau,
            self.ari,
        ]
        return sum(grades) / len(grades)

    @property
    def reading_level(self) -> str:
        """Human-readable reading level."""
        fre = self.flesch_reading_ease
        if fre >= 90:
            return "Very Easy (5th grade)"
        elif fre >= 80:
            return "Easy (6th grade)"
        elif fre >= 70:
            return "Fairly Easy (7th grade)"
        elif fre >= 60:
            return "Standard (8th-9th grade)"
        elif fre >= 50:
            return "Fairly Difficult (10th-12th grade)"
        elif fre >= 30:
            return "Difficult (College)"
        else:
            return "Very Difficult (College graduate)"

    def to_dict(self) -> dict:
        return {
            "flesch_reading_ease": round(self.flesch_reading_ease, 1),
            "flesch_kincaid_grade": round(self.flesch_kincaid_grade, 1),
            "gunning_fog": round(self.gunning_fog, 1),
            "smog_index": round(self.smog_index, 1),
            "coleman_liau": round(self.coleman_liau, 1),
            "ari": round(self.ari, 1),
            "average_grade_level": round(self.average_grade_level, 1),
            "reading_level": self.reading_level,
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "syllable_count": self.syllable_count,
            "complex_word_count": self.complex_word_count,
        }


class ReadabilityScorer:
    """
    Calculate readability metrics for text.

    Usage:
        scorer = ReadabilityScorer()
        result = scorer.analyze("Your text here...")
        print(result.flesch_reading_ease)
        print(result.reading_level)
    """

    # Common syllable patterns
    SYLLABLE_SUBSYL = [
        r'cial', r'tia', r'cius', r'cious', r'giu', r'ion', r'iou',
        r'sia$', r'.ely$', r'sed$', r'ed$',
    ]
    SYLLABLE_ADDSYL = [
        r'ia', r'riet', r'dien', r'iu', r'io', r'ii',
        r'[aeiouym]bl$', r'[aeiou]{3}', r'^mc', r'ism$',
        r'([^aeiouy])l$', r'[^l]lien', r'^coa[dglx].',
        r'[^gq]ua[^auieo]', r'dnt$',
    ]

    def analyze(self, text: str) -> ReadabilityResult:
        """Analyze text and return readability metrics."""
        # Clean text
        text = self._clean_text(text)

        # Basic counts
        words = self._get_words(text)
        sentences = self._get_sentences(text)

        word_count = len(words)
        sentence_count = max(1, len(sentences))
        char_count = sum(len(w) for w in words)

        # Syllable counting
        syllable_count = sum(self._count_syllables(w) for w in words)
        complex_words = [w for w in words if self._count_syllables(w) >= 3]
        complex_word_count = len(complex_words)

        # Avoid division by zero
        if word_count == 0:
            return ReadabilityResult(
                flesch_reading_ease=0,
                flesch_kincaid_grade=0,
                gunning_fog=0,
                smog_index=0,
                coleman_liau=0,
                ari=0,
                word_count=0,
                sentence_count=0,
                syllable_count=0,
                complex_word_count=0,
                char_count=0,
            )

        # Calculate metrics
        avg_sentence_length = word_count / sentence_count
        avg_syllables_per_word = syllable_count / word_count
        percent_complex = (complex_word_count / word_count) * 100

        # Flesch Reading Ease
        # FRE = 206.835 - 1.015(words/sentences) - 84.6(syllables/words)
        fre = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        fre = max(0, min(100, fre))

        # Flesch-Kincaid Grade Level
        # FK = 0.39(words/sentences) + 11.8(syllables/words) - 15.59
        fk = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
        fk = max(0, fk)

        # Gunning Fog Index
        # GF = 0.4 * ((words/sentences) + 100(complex/words))
        gf = 0.4 * (avg_sentence_length + percent_complex)
        gf = max(0, gf)

        # SMOG Index (requires 30+ sentences for accuracy)
        # SMOG = 1.0430 * sqrt(complex * (30/sentences)) + 3.1291
        import math
        if sentence_count >= 30:
            smog = 1.0430 * math.sqrt(complex_word_count * (30 / sentence_count)) + 3.1291
        else:
            # Simplified for short texts
            smog = 1.0430 * math.sqrt(complex_word_count * (30 / max(sentence_count, 1))) + 3.1291
        smog = max(0, smog)

        # Coleman-Liau Index
        # CLI = 0.0588L - 0.296S - 15.8
        # L = avg letters per 100 words, S = avg sentences per 100 words
        L = (char_count / word_count) * 100
        S = (sentence_count / word_count) * 100
        cli = (0.0588 * L) - (0.296 * S) - 15.8
        cli = max(0, cli)

        # Automated Readability Index
        # ARI = 4.71(chars/words) + 0.5(words/sentences) - 21.43
        ari = 4.71 * (char_count / word_count) + 0.5 * avg_sentence_length - 21.43
        ari = max(0, ari)

        return ReadabilityResult(
            flesch_reading_ease=fre,
            flesch_kincaid_grade=fk,
            gunning_fog=gf,
            smog_index=smog,
            coleman_liau=cli,
            ari=ari,
            word_count=word_count,
            sentence_count=sentence_count,
            syllable_count=syllable_count,
            complex_word_count=complex_word_count,
            char_count=char_count,
        )

    def _clean_text(self, text: str) -> str:
        """Clean text for analysis."""
        # Remove markdown formatting
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # Code blocks
        text = re.sub(r'`[^`]+`', '', text)  # Inline code
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Links
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)  # Headers
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)  # Lists
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic

        return text

    def _get_words(self, text: str) -> list[str]:
        """Extract words from text."""
        # Split on whitespace and punctuation, keep only words
        words = re.findall(r"[a-zA-Z']+", text.lower())
        return [w for w in words if len(w) > 0]

    def _get_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Split on sentence-ending punctuation
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word."""
        word = word.lower().strip()
        if len(word) <= 2:
            return 1

        # Remove trailing e (silent e)
        if word.endswith('e'):
            word = word[:-1]

        # Count vowel groups
        count = len(re.findall(r'[aeiouy]+', word))

        # Adjust for special patterns
        for pattern in self.SYLLABLE_SUBSYL:
            if re.search(pattern, word):
                count -= 1
        for pattern in self.SYLLABLE_ADDSYL:
            if re.search(pattern, word):
                count += 1

        return max(1, count)


def analyze_readability(text: str) -> dict:
    """Convenience function for quick analysis."""
    scorer = ReadabilityScorer()
    result = scorer.analyze(text)
    return result.to_dict()


# CLI
def main():
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Analyze text readability")
    parser.add_argument("input", type=Path, help="Text file to analyze")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    text = args.input.read_text()
    scorer = ReadabilityScorer()
    result = scorer.analyze(text)

    if args.json:
        import json
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Readability Analysis: {args.input.name}")
        print("=" * 50)
        print(f"Reading Level: {result.reading_level}")
        print(f"Flesch Reading Ease: {result.flesch_reading_ease:.1f}")
        print(f"Flesch-Kincaid Grade: {result.flesch_kincaid_grade:.1f}")
        print(f"Gunning Fog Index: {result.gunning_fog:.1f}")
        print(f"SMOG Index: {result.smog_index:.1f}")
        print(f"Coleman-Liau Index: {result.coleman_liau:.1f}")
        print(f"Average Grade Level: {result.average_grade_level:.1f}")
        print("-" * 50)
        print(f"Words: {result.word_count}")
        print(f"Sentences: {result.sentence_count}")
        print(f"Syllables: {result.syllable_count}")
        print(f"Complex words (3+ syllables): {result.complex_word_count}")


if __name__ == "__main__":
    main()
