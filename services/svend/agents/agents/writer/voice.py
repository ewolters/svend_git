"""
Voice Matching System

Extract writing style patterns from user samples:
- Sentence structure (length, complexity)
- Vocabulary preferences
- Tone markers
- Common phrases/patterns

Creates a "voice profile" to guide LLM generation.
"""

import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter
from typing import Any

# Add parent to path for tools access
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.readability import ReadabilityScorer
from tools.stats import analyze_text


@dataclass
class VoiceProfile:
    """Extracted writing style profile."""
    name: str

    # Sentence patterns
    avg_sentence_length: float  # words
    sentence_length_variance: float  # how much it varies
    avg_word_length: float

    # Readability
    reading_level: float  # Flesch-Kincaid grade
    flesch_ease: float

    # Vocabulary
    vocabulary_richness: float  # unique/total words
    common_words: list[tuple[str, int]]  # frequent non-stopwords
    preferred_phrases: list[str]  # 2-3 word patterns

    # Tone markers
    formality_score: float  # 0=casual, 1=formal
    uses_contractions: bool
    uses_first_person: bool
    uses_passive_voice: float  # percentage

    # Structural patterns
    avg_paragraph_length: float  # sentences
    uses_lists: bool
    uses_headers: bool

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "sentence": {
                "avg_length": round(self.avg_sentence_length, 1),
                "variance": round(self.sentence_length_variance, 2),
                "avg_word_length": round(self.avg_word_length, 1),
            },
            "readability": {
                "grade_level": round(self.reading_level, 1),
                "flesch_ease": round(self.flesch_ease, 1),
            },
            "vocabulary": {
                "richness": round(self.vocabulary_richness, 3),
                "common_words": self.common_words[:20],
                "preferred_phrases": self.preferred_phrases[:10],
            },
            "tone": {
                "formality": round(self.formality_score, 2),
                "uses_contractions": self.uses_contractions,
                "uses_first_person": self.uses_first_person,
                "passive_voice_pct": round(self.uses_passive_voice, 2),
            },
            "structure": {
                "avg_paragraph_length": round(self.avg_paragraph_length, 1),
                "uses_lists": self.uses_lists,
                "uses_headers": self.uses_headers,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VoiceProfile":
        return cls(
            name=data["name"],
            avg_sentence_length=data["sentence"]["avg_length"],
            sentence_length_variance=data["sentence"]["variance"],
            avg_word_length=data["sentence"]["avg_word_length"],
            reading_level=data["readability"]["grade_level"],
            flesch_ease=data["readability"]["flesch_ease"],
            vocabulary_richness=data["vocabulary"]["richness"],
            common_words=data["vocabulary"]["common_words"],
            preferred_phrases=data["vocabulary"]["preferred_phrases"],
            formality_score=data["tone"]["formality"],
            uses_contractions=data["tone"]["uses_contractions"],
            uses_first_person=data["tone"]["uses_first_person"],
            uses_passive_voice=data["tone"]["passive_voice_pct"],
            avg_paragraph_length=data["structure"]["avg_paragraph_length"],
            uses_lists=data["structure"]["uses_lists"],
            uses_headers=data["structure"]["uses_headers"],
        )

    def to_prompt_instructions(self) -> str:
        """Convert profile to LLM instructions."""
        instructions = ["## Writing Style Guidelines", ""]

        # Sentence structure
        if self.avg_sentence_length < 15:
            instructions.append("- Use SHORT sentences (under 15 words average)")
        elif self.avg_sentence_length > 25:
            instructions.append("- Use LONGER, more complex sentences (25+ words)")
        else:
            instructions.append("- Use moderate sentence length (15-25 words)")

        if self.sentence_length_variance > 0.5:
            instructions.append("- VARY sentence length significantly for rhythm")
        else:
            instructions.append("- Keep sentence length consistent")

        # Readability
        if self.reading_level < 8:
            instructions.append("- Write at an EASY reading level (8th grade or below)")
        elif self.reading_level > 12:
            instructions.append("- Write at an ADVANCED reading level (college+)")
        else:
            instructions.append(f"- Target grade {self.reading_level:.0f} reading level")

        # Tone
        if self.formality_score > 0.7:
            instructions.append("- Use FORMAL tone (no contractions, professional language)")
        elif self.formality_score < 0.3:
            instructions.append("- Use CASUAL tone (contractions OK, conversational)")
        else:
            instructions.append("- Use professional but approachable tone")

        if self.uses_contractions:
            instructions.append("- Contractions are OK (don't, won't, etc.)")
        else:
            instructions.append("- Avoid contractions (use 'do not' instead of 'don't')")

        if self.uses_first_person:
            instructions.append("- First person is OK (I, we, our)")
        else:
            instructions.append("- Avoid first person (use 'the team' instead of 'we')")

        if self.uses_passive_voice > 0.3:
            instructions.append("- Passive voice is acceptable")
        else:
            instructions.append("- Prefer active voice")

        # Vocabulary
        if self.vocabulary_richness > 0.6:
            instructions.append("- Use VARIED vocabulary (avoid repetition)")
        else:
            instructions.append("- Repeat key terms for clarity")

        if self.common_words:
            words = [w for w, c in self.common_words[:5]]
            instructions.append(f"- Preferred terms: {', '.join(words)}")

        if self.preferred_phrases:
            instructions.append(f"- Common phrases to use: {', '.join(self.preferred_phrases[:3])}")

        # Structure
        if self.uses_lists:
            instructions.append("- Use bullet points and lists where appropriate")
        if self.uses_headers:
            instructions.append("- Use clear section headers")

        return "\n".join(instructions)


class VoiceAnalyzer:
    """Analyze text samples to extract voice profile."""

    # Formal language markers
    FORMAL_MARKERS = [
        r'\bfurthermore\b', r'\btherefore\b', r'\bhowever\b', r'\bmoreover\b',
        r'\bconsequently\b', r'\bnevertheless\b', r'\baccordingly\b',
        r'\bhence\b', r'\bthus\b', r'\bwhereas\b',
    ]

    # Casual markers
    CASUAL_MARKERS = [
        r"\bdon't\b", r"\bwon't\b", r"\bcan't\b", r"\bit's\b", r"\bthat's\b",
        r"\byou're\b", r"\bwe're\b", r"\bthey're\b", r"\bgonna\b", r"\bwanna\b",
        r"\bokay\b", r"\bso\b,", r"\banyway\b", r"\bbasically\b",
    ]

    # Passive voice patterns
    PASSIVE_PATTERNS = [
        r'\bwas\s+\w+ed\b', r'\bwere\s+\w+ed\b', r'\bis\s+\w+ed\b',
        r'\bare\s+\w+ed\b', r'\bbeen\s+\w+ed\b', r'\bbe\s+\w+ed\b',
    ]

    def __init__(self):
        self.readability = ReadabilityScorer()

    def analyze(self, samples: list[str], name: str = "custom") -> VoiceProfile:
        """Analyze multiple text samples to create voice profile."""
        if not samples:
            raise ValueError("Need at least one sample")

        combined = "\n\n".join(samples)

        # Get readability metrics
        read_result = self.readability.analyze(combined)

        # Sentence analysis
        sentences = self._get_sentences(combined)
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_sentence_length = sum(sentence_lengths) / max(len(sentence_lengths), 1)

        if sentence_lengths:
            mean = avg_sentence_length
            variance = sum((x - mean) ** 2 for x in sentence_lengths) / len(sentence_lengths)
            sentence_variance = (variance ** 0.5) / max(mean, 1)  # coefficient of variation
        else:
            sentence_variance = 0

        # Word analysis
        words = re.findall(r'\b[a-zA-Z]+\b', combined.lower())
        avg_word_length = sum(len(w) for w in words) / max(len(words), 1)
        unique_words = len(set(words))
        vocabulary_richness = unique_words / max(len(words), 1)

        # Common words (excluding stopwords)
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'this', 'that', 'these', 'those', 'it', 'its', 'as', 'if',
            'not', 'no', 'so', 'than', 'too', 'very', 'just', 'also', 'more',
        }
        filtered = [w for w in words if w not in stopwords and len(w) > 3]
        common_words = Counter(filtered).most_common(20)

        # Preferred phrases (2-3 word patterns)
        phrases = self._extract_phrases(combined)
        preferred_phrases = [p for p, c in Counter(phrases).most_common(10) if c > 1]

        # Formality analysis
        formal_count = sum(len(re.findall(p, combined, re.I)) for p in self.FORMAL_MARKERS)
        casual_count = sum(len(re.findall(p, combined, re.I)) for p in self.CASUAL_MARKERS)
        total_markers = formal_count + casual_count
        if total_markers > 0:
            formality_score = formal_count / total_markers
        else:
            formality_score = 0.5  # neutral

        # Specific checks
        uses_contractions = any(re.search(p, combined, re.I) for p in self.CASUAL_MARKERS[:8])
        uses_first_person = bool(re.search(r'\b(I|we|our|my)\b', combined, re.I))

        # Passive voice
        passive_count = sum(len(re.findall(p, combined, re.I)) for p in self.PASSIVE_PATTERNS)
        passive_ratio = passive_count / max(len(sentences), 1)

        # Paragraph analysis
        paragraphs = [p.strip() for p in combined.split('\n\n') if p.strip()]
        para_lengths = [len(self._get_sentences(p)) for p in paragraphs]
        avg_para_length = sum(para_lengths) / max(len(para_lengths), 1)

        # Structural elements
        uses_lists = bool(re.search(r'^\s*[-*•]\s', combined, re.M))
        uses_headers = bool(re.search(r'^#+\s|^[A-Z][^.!?]*:\s*$', combined, re.M))

        return VoiceProfile(
            name=name,
            avg_sentence_length=avg_sentence_length,
            sentence_length_variance=sentence_variance,
            avg_word_length=avg_word_length,
            reading_level=read_result.flesch_kincaid_grade,
            flesch_ease=read_result.flesch_reading_ease,
            vocabulary_richness=vocabulary_richness,
            common_words=common_words,
            preferred_phrases=preferred_phrases,
            formality_score=formality_score,
            uses_contractions=uses_contractions,
            uses_first_person=uses_first_person,
            uses_passive_voice=passive_ratio,
            avg_paragraph_length=avg_para_length,
            uses_lists=uses_lists,
            uses_headers=uses_headers,
        )

    def _get_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.split()) > 2]

    def _extract_phrases(self, text: str) -> list[str]:
        """Extract 2-3 word phrases."""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        phrases = []

        for i in range(len(words) - 1):
            # 2-word phrases
            phrases.append(f"{words[i]} {words[i+1]}")
            # 3-word phrases
            if i < len(words) - 2:
                phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")

        return phrases


class VoiceManager:
    """Manage user voice profiles."""

    def __init__(self, storage_dir: Path = None):
        self.storage_dir = storage_dir or Path.home() / ".svend" / "voices"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.analyzer = VoiceAnalyzer()
        self._cache: dict[str, VoiceProfile] = {}

    def create_from_samples(self, samples: list[str], name: str) -> VoiceProfile:
        """Create voice profile from text samples."""
        profile = self.analyzer.analyze(samples, name)
        self.save(profile)
        return profile

    def create_from_files(self, paths: list[Path], name: str) -> VoiceProfile:
        """Create voice profile from files."""
        samples = []
        for path in paths:
            if path.exists():
                samples.append(path.read_text())
        return self.create_from_samples(samples, name)

    def save(self, profile: VoiceProfile) -> Path:
        """Save voice profile."""
        filename = re.sub(r'[^\w\-]', '_', profile.name.lower()) + ".json"
        path = self.storage_dir / filename
        path.write_text(json.dumps(profile.to_dict(), indent=2))
        self._cache[profile.name] = profile
        return path

    def load(self, name: str) -> VoiceProfile | None:
        """Load voice profile by name."""
        if name in self._cache:
            return self._cache[name]

        filename = re.sub(r'[^\w\-]', '_', name.lower()) + ".json"
        path = self.storage_dir / filename
        if path.exists():
            data = json.loads(path.read_text())
            profile = VoiceProfile.from_dict(data)
            self._cache[name] = profile
            return profile
        return None

    def list_profiles(self) -> list[str]:
        """List all voice profiles."""
        profiles = []
        for path in self.storage_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                profiles.append(data.get("name", path.stem))
            except (json.JSONDecodeError, KeyError):
                continue
        return profiles

    def delete(self, name: str) -> bool:
        """Delete a voice profile."""
        filename = re.sub(r'[^\w\-]', '_', name.lower()) + ".json"
        path = self.storage_dir / filename
        if path.exists():
            path.unlink()
            self._cache.pop(name, None)
            return True
        return False
