"""
Source Management for Research Agent

Handles:
- Source tracking and citation
- Source diversity scoring
- Credibility assessment
- Bibliography generation
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import hashlib
import re


class SourceType(Enum):
    """Types of sources for research."""
    ACADEMIC = "academic"       # Papers, journals, preprints
    NEWS = "news"               # News articles, press releases
    OFFICIAL = "official"       # Government, org websites
    INDUSTRY = "industry"       # Market reports, company data
    REFERENCE = "reference"     # Wikipedia, encyclopedias
    SOCIAL = "social"           # Forums, discussions, blogs
    PRIMARY = "primary"         # Original data, interviews


@dataclass
class Source:
    """A research source with metadata."""
    id: str
    title: str
    url: str
    source_type: SourceType
    content: str = ""
    snippet: str = ""  # Key excerpt

    # Metadata
    author: str = ""
    date: str = ""
    publication: str = ""

    # Quality signals
    credibility_score: float = 0.5  # 0-1
    relevance_score: float = 0.5    # 0-1

    # Tracking
    retrieved_at: datetime = field(default_factory=datetime.now)
    used_in_sections: list[str] = field(default_factory=list)

    def cite(self, style: str = "inline") -> str:
        """Generate citation string."""
        if style == "inline":
            return f"[{self.id}]"
        elif style == "footnote":
            author = self.author or "Unknown"
            date = self.date or "n.d."
            return f"{author} ({date}). {self.title}. {self.url}"
        elif style == "markdown":
            return f"[{self.title}]({self.url})"
        return self.id


@dataclass
class ResearchFindings:
    """Structured research findings with sources."""
    query: str
    summary: str
    sections: list[dict] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    confidence: float = 0.5
    gaps: list[str] = field(default_factory=list)  # What we couldn't find
    metadata: dict = field(default_factory=dict)  # Validation results, etc.

    @property
    def validation(self) -> dict:
        """Get validation results if available."""
        return self.metadata.get("validation", {})

    @property
    def is_validated(self) -> bool:
        """Check if research was validated."""
        return "validation" in self.metadata

    @property
    def quality_score(self) -> float:
        """Get quality score from validation (or confidence as fallback)."""
        if self.is_validated:
            return self.validation.get("quality_score", self.confidence)
        return self.confidence

    def to_markdown(self) -> str:
        """Export findings as markdown document."""
        lines = [
            f"# Research: {self.query}",
            "",
            f"**Confidence:** {self.confidence:.0%}",
            f"**Sources:** {len(self.sources)}",
            "",
            "## Summary",
            "",
            self.summary,
            "",
        ]

        # Sections
        for section in self.sections:
            lines.append(f"## {section.get('title', 'Findings')}")
            lines.append("")
            lines.append(section.get('content', ''))
            lines.append("")

        # Gaps
        if self.gaps:
            lines.append("## Research Gaps")
            lines.append("")
            for gap in self.gaps:
                lines.append(f"- {gap}")
            lines.append("")

        # Bibliography
        lines.append("## Sources")
        lines.append("")
        for source in sorted(self.sources, key=lambda s: s.id):
            lines.append(f"- **[{source.id}]** {source.cite('footnote')}")

        return "\n".join(lines)


class SourceManager:
    """
    Manages sources for a research session.

    Ensures:
    - Source diversity (not all from one type)
    - Credibility tracking
    - Proper citation
    - Deduplication
    """

    # Credibility heuristics by domain
    HIGH_CREDIBILITY_DOMAINS = {
        '.gov', '.edu', 'nature.com', 'science.org', 'nih.gov',
        'who.int', 'worldbank.org', 'imf.org', 'arxiv.org',
        'pubmed.ncbi', 'ieee.org', 'acm.org', 'springer.com',
        'wiley.com', 'elsevier.com', 'reuters.com', 'apnews.com',
    }

    MEDIUM_CREDIBILITY_DOMAINS = {
        'wikipedia.org', 'britannica.com', 'statista.com',
        'bloomberg.com', 'wsj.com', 'nytimes.com', 'bbc.com',
        'forbes.com', 'hbr.org', 'mckinsey.com', 'gartner.com',
    }

    # Source type weights for diversity
    TYPE_WEIGHTS = {
        SourceType.ACADEMIC: 1.5,
        SourceType.OFFICIAL: 1.3,
        SourceType.INDUSTRY: 1.2,
        SourceType.NEWS: 1.0,
        SourceType.REFERENCE: 0.9,
        SourceType.SOCIAL: 0.6,
        SourceType.PRIMARY: 1.4,
    }

    def __init__(self):
        self.sources: dict[str, Source] = {}
        self._id_counter = 0

    def add_source(self, title: str, url: str, content: str = "",
                   source_type: SourceType = None, **metadata) -> Source:
        """Add a source and return it."""
        # Deduplicate by URL
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        if url_hash in self.sources:
            return self.sources[url_hash]

        # Auto-detect source type if not provided
        if source_type is None:
            source_type = self._detect_source_type(url, title)

        # Generate ID
        self._id_counter += 1
        source_id = f"S{self._id_counter}"

        # Calculate credibility
        credibility = self._calculate_credibility(url, source_type)

        source = Source(
            id=source_id,
            title=title,
            url=url,
            source_type=source_type,
            content=content,
            credibility_score=credibility,
            author=metadata.get('author', ''),
            date=metadata.get('date', ''),
            publication=metadata.get('publication', ''),
            snippet=metadata.get('snippet', content[:500] if content else ''),
        )

        self.sources[url_hash] = source
        return source

    def _detect_source_type(self, url: str, title: str) -> SourceType:
        """Auto-detect source type from URL and title."""
        url_lower = url.lower()
        title_lower = title.lower()

        # Academic indicators
        if any(d in url_lower for d in ['arxiv', 'pubmed', 'doi.org', 'scholar.google',
                                         'jstor', 'springer', 'wiley', 'elsevier',
                                         'ieee', 'acm.org', 'nature.com', 'science.org']):
            return SourceType.ACADEMIC

        # Official/government
        if '.gov' in url_lower or '.edu' in url_lower:
            return SourceType.OFFICIAL
        if any(d in url_lower for d in ['who.int', 'un.org', 'worldbank', 'imf.org']):
            return SourceType.OFFICIAL

        # Industry/market
        if any(d in url_lower for d in ['statista', 'gartner', 'mckinsey', 'deloitte',
                                         'pwc.com', 'accenture', 'forrester', 'idc.com']):
            return SourceType.INDUSTRY

        # News
        if any(d in url_lower for d in ['reuters', 'apnews', 'bbc.com', 'nytimes',
                                         'wsj.com', 'bloomberg', 'cnbc', 'techcrunch']):
            return SourceType.NEWS

        # Reference
        if any(d in url_lower for d in ['wikipedia', 'britannica', 'encyclopedia']):
            return SourceType.REFERENCE

        # Social/blogs
        if any(d in url_lower for d in ['reddit', 'medium.com', 'substack', 'twitter',
                                         'linkedin', 'quora', 'stackoverflow']):
            return SourceType.SOCIAL

        return SourceType.NEWS  # Default

    def _calculate_credibility(self, url: str, source_type: SourceType) -> float:
        """Calculate credibility score for a source."""
        base_score = 0.5

        # Domain-based scoring
        url_lower = url.lower()
        for domain in self.HIGH_CREDIBILITY_DOMAINS:
            if domain in url_lower:
                base_score = 0.85
                break
        else:
            for domain in self.MEDIUM_CREDIBILITY_DOMAINS:
                if domain in url_lower:
                    base_score = 0.7
                    break

        # Type-based adjustment
        type_weight = self.TYPE_WEIGHTS.get(source_type, 1.0)

        return min(1.0, base_score * type_weight)

    def get_diversity_score(self) -> float:
        """Calculate source diversity (0-1)."""
        if not self.sources:
            return 0.0

        # Count types
        type_counts = {}
        for source in self.sources.values():
            t = source.source_type
            type_counts[t] = type_counts.get(t, 0) + 1

        # Shannon diversity index (normalized)
        total = len(self.sources)
        diversity = 0.0
        for count in type_counts.values():
            if count > 0:
                p = count / total
                diversity -= p * (p if p == 0 else __import__('math').log2(p))

        # Normalize by max possible diversity
        max_diversity = __import__('math').log2(len(SourceType))
        return diversity / max_diversity if max_diversity > 0 else 0.0

    def get_credibility_score(self) -> float:
        """Calculate overall source credibility."""
        if not self.sources:
            return 0.0

        # Weighted average by relevance
        total_weight = 0.0
        weighted_sum = 0.0
        for source in self.sources.values():
            weight = source.relevance_score
            weighted_sum += source.credibility_score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def suggest_missing_types(self) -> list[SourceType]:
        """Suggest source types that would improve diversity."""
        present_types = {s.source_type for s in self.sources.values()}

        # Prioritize high-value missing types
        suggestions = []
        for t in [SourceType.ACADEMIC, SourceType.OFFICIAL, SourceType.INDUSTRY]:
            if t not in present_types:
                suggestions.append(t)

        return suggestions

    def get_bibliography(self) -> str:
        """Generate formatted bibliography."""
        lines = []
        for source in sorted(self.sources.values(), key=lambda s: s.id):
            lines.append(f"[{source.id}] {source.cite('footnote')}")
        return "\n".join(lines)

    def get_sources_by_type(self, source_type: SourceType) -> list[Source]:
        """Get all sources of a specific type."""
        return [s for s in self.sources.values() if s.source_type == source_type]
