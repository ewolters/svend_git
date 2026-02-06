"""
Multi-Source Search APIs

Integrates:
- Semantic Scholar (academic papers) - Free, no key
- arXiv (preprints) - Free, no key
- Brave Search (web) - Free tier with key
- DuckDuckGo (web fallback) - Unofficial, no key

All APIs use retry with exponential backoff to handle rate limits.
"""

import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from enum import Enum
from typing import Iterator
from xml.etree import ElementTree

from .sources import SourceType


def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 2.0):
    """
    Retry a function with exponential backoff.

    Args:
        func: Callable to retry
        max_retries: Maximum number of retries
        base_delay: Base delay in seconds (doubles each retry)

    Returns:
        Result of func, or raises last exception
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except urllib.error.HTTPError as e:
            last_exception = e
            if e.code == 429:  # Rate limited
                delay = base_delay * (2 ** attempt)
                print(f"Rate limited (429). Waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(delay)
            elif e.code >= 500:  # Server error
                delay = base_delay * (2 ** attempt)
                print(f"Server error ({e.code}). Waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(delay)
            else:
                raise  # Don't retry client errors (4xx except 429)
        except urllib.error.URLError as e:
            last_exception = e
            delay = base_delay * (2 ** attempt)
            print(f"Connection error. Waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}...")
            time.sleep(delay)

    raise last_exception


class StudyType(Enum):
    """Evidence hierarchy for study types."""
    META_ANALYSIS = "meta_analysis"          # Highest evidence
    SYSTEMATIC_REVIEW = "systematic_review"
    RCT = "rct"                              # Randomized controlled trial
    COHORT = "cohort"                        # Prospective/retrospective cohort
    CASE_CONTROL = "case_control"
    CROSS_SECTIONAL = "cross_sectional"
    CASE_REPORT = "case_report"
    EXPERT_OPINION = "expert_opinion"
    PREPRINT = "preprint"                    # Not peer-reviewed
    UNKNOWN = "unknown"


# Evidence strength weights for study types
EVIDENCE_WEIGHTS = {
    StudyType.META_ANALYSIS: 1.0,
    StudyType.SYSTEMATIC_REVIEW: 0.95,
    StudyType.RCT: 0.9,
    StudyType.COHORT: 0.75,
    StudyType.CASE_CONTROL: 0.65,
    StudyType.CROSS_SECTIONAL: 0.55,
    StudyType.CASE_REPORT: 0.4,
    StudyType.EXPERT_OPINION: 0.3,
    StudyType.PREPRINT: 0.5,  # Lower because not peer-reviewed
    StudyType.UNKNOWN: 0.5,
}


@dataclass
class SearchResult:
    """Unified search result across all APIs."""
    title: str
    url: str
    snippet: str
    source_type: SourceType
    author: str = ""
    date: str = ""
    publication: str = ""
    citations: int = 0  # For academic papers
    relevance_score: float = 0.0

    # Evidence quality metadata
    study_type: StudyType = StudyType.UNKNOWN
    peer_reviewed: bool = True
    sample_size: int = 0  # If available
    domain: str = ""  # medical, physics, psychology, engineering, etc.

    @property
    def evidence_strength(self) -> float:
        """Calculate evidence strength score (0-1)."""
        base = EVIDENCE_WEIGHTS.get(self.study_type, 0.5)
        # Boost for peer review
        if not self.peer_reviewed:
            base *= 0.7
        # Boost for citations (capped)
        citation_boost = min(0.2, self.citations / 5000) if self.citations else 0
        # Boost for sample size (if known)
        size_boost = 0
        if self.sample_size > 1000:
            size_boost = 0.1
        elif self.sample_size > 100:
            size_boost = 0.05
        return min(1.0, base + citation_boost + size_boost)


class SemanticScholarSearch:
    """
    Semantic Scholar API for academic papers.

    Free, no API key needed for basic access.
    Rate limit: 100 requests/second (but be conservative)
    Docs: https://api.semanticscholar.org/
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: str = None):
        self.api_key = api_key  # Optional, increases rate limits
        self.headers = {"User-Agent": "ResearchAgent/1.0"}
        if api_key:
            self.headers["x-api-key"] = api_key
        self.last_request = 0
        self.min_delay = 3.0  # Conservative delay to avoid 429s

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Search for academic papers."""
        results = []

        # Rate limiting
        elapsed = time.time() - self.last_request
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)

        # URL encode query
        encoded_query = urllib.parse.quote(query)
        url = (
            f"{self.BASE_URL}/paper/search"
            f"?query={encoded_query}"
            f"&limit={limit}"
            f"&fields=title,url,abstract,authors,year,citationCount,venue"
        )

        def do_request():
            request = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(request, timeout=15) as response:
                self.last_request = time.time()
                return json.loads(response.read().decode())

        try:
            data = retry_with_backoff(do_request, max_retries=3, base_delay=2.0)

            for paper in data.get("data", []):
                # Extract first author
                authors = paper.get("authors", [])
                author = authors[0].get("name", "") if authors else ""
                if len(authors) > 1:
                    author += " et al."

                # Build URL (use paperId if no url)
                paper_url = paper.get("url") or f"https://www.semanticscholar.org/paper/{paper.get('paperId', '')}"

                results.append(SearchResult(
                    title=paper.get("title", ""),
                    url=paper_url,
                    snippet=paper.get("abstract", "")[:500] if paper.get("abstract") else "",
                    source_type=SourceType.ACADEMIC,
                    author=author,
                    date=str(paper.get("year", "")),
                    publication=paper.get("venue", ""),
                    citations=paper.get("citationCount", 0),
                    relevance_score=min(1.0, paper.get("citationCount", 0) / 1000),
                ))
        except Exception as e:
            print(f"Semantic Scholar search failed after retries: {e}")

        return results


class ArxivSearch:
    """
    arXiv API for preprints.

    Free, no API key needed.
    Rate limit: Be reasonable (1 req/3 sec recommended)
    Docs: https://info.arxiv.org/help/api/
    """

    BASE_URL = "http://export.arxiv.org/api/query"

    # Categories to exclude (not relevant for most research queries)
    EXCLUDED_CATEGORIES = {
        'hep-ph', 'hep-th', 'hep-ex', 'hep-lat',  # High energy physics
        'gr-qc', 'nucl-th', 'nucl-ex',  # Nuclear/gravity
        'astro-ph', 'cond-mat',  # Astrophysics, condensed matter
        'quant-ph',  # Quantum physics
        'math-ph', 'nlin',  # Math physics, nonlinear
    }

    def __init__(self):
        self.last_request = 0
        self.min_delay = 3.0  # Seconds between requests

    def search(self, query: str, limit: int = 10, category: str = None) -> list[SearchResult]:
        """Search arXiv for preprints."""
        results = []

        # Rate limiting
        elapsed = time.time() - self.last_request
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)

        # Build query - search in title and abstract only (more precise than "all:")
        encoded_query = urllib.parse.quote(query)

        # Use ti: (title) and abs: (abstract) for more relevant results
        search_query = f"ti:{encoded_query}+OR+abs:{encoded_query}"

        # Add category filter if specified
        if category:
            search_query = f"({search_query})+AND+cat:{category}"

        url = (
            f"{self.BASE_URL}"
            f"?search_query={search_query}"
            f"&start=0"
            f"&max_results={limit * 2}"  # Get more, then filter
            f"&sortBy=relevance"
            f"&sortOrder=descending"
        )

        def do_request():
            request = urllib.request.Request(url, headers={"User-Agent": "ResearchAgent/1.0"})
            with urllib.request.urlopen(request, timeout=15) as response:
                self.last_request = time.time()
                return response.read()

        try:
            response_data = retry_with_backoff(do_request, max_retries=3, base_delay=3.0)

            # Parse XML response
            root = ElementTree.fromstring(response_data)

            # Namespace handling
            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom"
            }

            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns)
                summary = entry.find("atom:summary", ns)
                published = entry.find("atom:published", ns)

                # Get authors
                authors = entry.findall("atom:author/atom:name", ns)
                author = authors[0].text if authors else ""
                if len(authors) > 1:
                    author += " et al."

                # Get PDF link
                pdf_link = ""
                for link in entry.findall("atom:link", ns):
                    if link.get("title") == "pdf":
                        pdf_link = link.get("href", "")
                        break

                # Fallback to abstract page
                if not pdf_link:
                    id_elem = entry.find("atom:id", ns)
                    pdf_link = id_elem.text if id_elem is not None else ""

                # Extract date (YYYY-MM-DD from ISO format)
                date = ""
                if published is not None and published.text:
                    date = published.text[:10]

                # Get category and filter irrelevant physics papers
                category_elem = entry.find("arxiv:primary_category", ns)
                primary_category = category_elem.get("term", "") if category_elem is not None else ""
                venue = f"arXiv ({primary_category})" if primary_category else "arXiv"

                # Skip papers from excluded physics categories
                category_prefix = primary_category.split('.')[0] if primary_category else ""
                if category_prefix in self.EXCLUDED_CATEGORIES or primary_category in self.EXCLUDED_CATEGORIES:
                    continue

                # Detect domain from category
                domain = "physics"  # Default for arXiv
                if primary_category.startswith("cs."):
                    domain = "computer_science"
                elif primary_category.startswith("q-bio"):
                    domain = "biology"
                elif primary_category.startswith("stat"):
                    domain = "statistics"
                elif primary_category.startswith("econ"):
                    domain = "economics"
                elif primary_category.startswith("math"):
                    domain = "mathematics"

                results.append(SearchResult(
                    title=title.text.strip().replace("\n", " ") if title is not None and title.text else "",
                    url=pdf_link,
                    snippet=summary.text.strip().replace("\n", " ")[:500] if summary is not None and summary.text else "",
                    source_type=SourceType.ACADEMIC,
                    author=author,
                    date=date,
                    publication=venue,
                    citations=0,  # arXiv doesn't provide this
                    relevance_score=0.7,  # Default for preprints
                    study_type=StudyType.PREPRINT,
                    peer_reviewed=False,
                    domain=domain,
                ))

        except Exception as e:
            print(f"arXiv search failed after retries: {e}")

        return results


class PubMedSearch:
    """
    PubMed/NCBI E-utilities API for biomedical literature.

    Free, no API key needed (but key increases rate limits).
    Rate limit: 3 req/sec without key, 10 req/sec with key
    Docs: https://www.ncbi.nlm.nih.gov/books/NBK25500/
    """

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    # Study type detection patterns
    STUDY_PATTERNS = {
        StudyType.META_ANALYSIS: ['meta-analysis', 'meta analysis', 'pooled analysis'],
        StudyType.SYSTEMATIC_REVIEW: ['systematic review', 'cochrane review'],
        StudyType.RCT: ['randomized controlled trial', 'randomised controlled trial', 'rct', 'randomized trial'],
        StudyType.COHORT: ['cohort study', 'prospective study', 'longitudinal study', 'follow-up study'],
        StudyType.CASE_CONTROL: ['case-control', 'case control'],
        StudyType.CROSS_SECTIONAL: ['cross-sectional', 'cross sectional', 'prevalence study'],
        StudyType.CASE_REPORT: ['case report', 'case series'],
    }

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.last_request = 0
        self.min_delay = 0.34 if api_key else 0.5  # 3/sec without key, 10/sec with

    def _detect_study_type(self, title: str, abstract: str) -> StudyType:
        """Detect study type from title/abstract."""
        text = f"{title} {abstract}".lower()
        for study_type, patterns in self.STUDY_PATTERNS.items():
            if any(p in text for p in patterns):
                return study_type
        return StudyType.UNKNOWN

    def _extract_sample_size(self, abstract: str) -> int:
        """Try to extract sample size from abstract."""
        import re
        # Common patterns: "n=1234", "N = 1,234", "1234 patients", "1234 participants"
        patterns = [
            r'[nN]\s*=\s*([\d,]+)',
            r'([\d,]+)\s*(?:patients|participants|subjects|individuals|people)',
            r'sample\s*(?:size|of)\s*([\d,]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, abstract)
            if match:
                try:
                    return int(match.group(1).replace(',', ''))
                except ValueError:
                    pass
        return 0

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Search PubMed for biomedical papers."""
        results = []

        # Rate limiting
        elapsed = time.time() - self.last_request
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)

        # Step 1: Search for PMIDs
        encoded_query = urllib.parse.quote(query)
        search_url = (
            f"{self.BASE_URL}/esearch.fcgi"
            f"?db=pubmed"
            f"&term={encoded_query}"
            f"&retmax={limit}"
            f"&retmode=json"
            f"&sort=relevance"
        )
        if self.api_key:
            search_url += f"&api_key={self.api_key}"

        def do_search():
            request = urllib.request.Request(search_url, headers={"User-Agent": "ResearchAgent/1.0"})
            with urllib.request.urlopen(request, timeout=15) as response:
                self.last_request = time.time()
                return json.loads(response.read().decode())

        try:
            search_data = retry_with_backoff(do_search, max_retries=3, base_delay=2.0)
            pmids = search_data.get("esearchresult", {}).get("idlist", [])

            if not pmids:
                return results

            # Step 2: Fetch details for PMIDs
            pmid_str = ",".join(pmids)
            fetch_url = (
                f"{self.BASE_URL}/efetch.fcgi"
                f"?db=pubmed"
                f"&id={pmid_str}"
                f"&retmode=xml"
            )
            if self.api_key:
                fetch_url += f"&api_key={self.api_key}"

            def do_fetch():
                request = urllib.request.Request(fetch_url, headers={"User-Agent": "ResearchAgent/1.0"})
                with urllib.request.urlopen(request, timeout=20) as response:
                    self.last_request = time.time()
                    return response.read()

            fetch_data = retry_with_backoff(do_fetch, max_retries=2, base_delay=2.0)
            root = ElementTree.fromstring(fetch_data)

            for article in root.findall(".//PubmedArticle"):
                # Extract fields
                title_elem = article.find(".//ArticleTitle")
                title = title_elem.text if title_elem is not None and title_elem.text else ""

                abstract_elem = article.find(".//AbstractText")
                abstract = abstract_elem.text if abstract_elem is not None and abstract_elem.text else ""

                # Authors
                authors = article.findall(".//Author/LastName")
                author = authors[0].text if authors else ""
                if len(authors) > 1:
                    author += " et al."

                # PMID for URL
                pmid_elem = article.find(".//PMID")
                pmid = pmid_elem.text if pmid_elem is not None else ""
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

                # Date
                pub_date = article.find(".//PubDate")
                year = pub_date.find("Year").text if pub_date is not None and pub_date.find("Year") is not None else ""

                # Journal
                journal_elem = article.find(".//Journal/Title")
                journal = journal_elem.text if journal_elem is not None else "PubMed"

                # Detect study type and sample size
                study_type = self._detect_study_type(title, abstract)
                sample_size = self._extract_sample_size(abstract)

                results.append(SearchResult(
                    title=title,
                    url=url,
                    snippet=abstract[:500] if abstract else "",
                    source_type=SourceType.ACADEMIC,
                    author=author,
                    date=year,
                    publication=journal,
                    citations=0,  # Would need separate API call
                    relevance_score=0.85,  # PubMed is high quality
                    study_type=study_type,
                    peer_reviewed=True,
                    sample_size=sample_size,
                    domain="medical",
                ))

        except Exception as e:
            print(f"PubMed search failed: {e}")

        return results


class OpenAlexSearch:
    """
    OpenAlex API for broad academic coverage.

    Free, no API key needed. Covers all academic domains.
    Rate limit: 10 req/sec (polite pool), 100K/day
    Docs: https://docs.openalex.org/
    """

    BASE_URL = "https://api.openalex.org"

    # Domain detection from concepts/topics
    DOMAIN_KEYWORDS = {
        "medical": ["medicine", "health", "clinical", "disease", "patient", "therapy", "drug", "pharmaceutical"],
        "physics": ["physics", "quantum", "particle", "cosmology", "astrophysics", "mechanics"],
        "engineering": ["engineering", "robotics", "mechanical", "electrical", "civil", "materials"],
        "psychology": ["psychology", "cognitive", "behavioral", "mental", "psychiatric", "neuroscience"],
        "biology": ["biology", "genetics", "molecular", "cell", "organism", "ecology"],
        "chemistry": ["chemistry", "chemical", "molecular", "synthesis", "compound"],
        "computer_science": ["computer", "algorithm", "software", "machine learning", "artificial intelligence", "data"],
        "economics": ["economics", "market", "finance", "trade", "monetary"],
    }

    def __init__(self, email: str = None):
        self.email = email  # Polite pool access
        self.last_request = 0
        self.min_delay = 0.1  # 10 req/sec

    def _detect_domain(self, title: str, concepts: list) -> str:
        """Detect research domain from title and concepts."""
        text = title.lower() + " " + " ".join(c.lower() for c in concepts)
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return domain
        return "general"

    def _detect_study_type(self, work_type: str, title: str) -> StudyType:
        """Detect study type from OpenAlex work type and title."""
        title_lower = title.lower()

        # Check title for study type indicators
        if "meta-analysis" in title_lower or "meta analysis" in title_lower:
            return StudyType.META_ANALYSIS
        if "systematic review" in title_lower:
            return StudyType.SYSTEMATIC_REVIEW
        if "randomized" in title_lower or "randomised" in title_lower:
            return StudyType.RCT
        if "cohort" in title_lower:
            return StudyType.COHORT
        if "case-control" in title_lower:
            return StudyType.CASE_CONTROL

        # Map OpenAlex types
        type_map = {
            "article": StudyType.UNKNOWN,
            "review": StudyType.SYSTEMATIC_REVIEW,
            "preprint": StudyType.PREPRINT,
            "book-chapter": StudyType.EXPERT_OPINION,
            "editorial": StudyType.EXPERT_OPINION,
        }
        return type_map.get(work_type, StudyType.UNKNOWN)

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Search OpenAlex for academic papers across all domains."""
        results = []

        # Rate limiting
        elapsed = time.time() - self.last_request
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)

        encoded_query = urllib.parse.quote(query)
        url = (
            f"{self.BASE_URL}/works"
            f"?search={encoded_query}"
            f"&per_page={limit}"
            f"&sort=relevance_score:desc"
            f"&select=id,doi,title,authorships,publication_year,cited_by_count,type,primary_location,abstract_inverted_index,concepts"
        )

        headers = {"User-Agent": "ResearchAgent/1.0 (mailto:research@example.com)"}
        if self.email:
            headers["User-Agent"] = f"ResearchAgent/1.0 (mailto:{self.email})"

        def do_request():
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=15) as response:
                self.last_request = time.time()
                return json.loads(response.read().decode())

        try:
            data = retry_with_backoff(do_request, max_retries=3, base_delay=2.0)

            for work in data.get("results", []):
                title = work.get("title", "")
                if not title:
                    continue

                # Reconstruct abstract from inverted index
                abstract = ""
                abstract_idx = work.get("abstract_inverted_index")
                if abstract_idx:
                    # Convert inverted index to text
                    word_positions = []
                    for word, positions in abstract_idx.items():
                        for pos in positions:
                            word_positions.append((pos, word))
                    word_positions.sort()
                    abstract = " ".join(word for _, word in word_positions)[:500]

                # Authors
                authorships = work.get("authorships", [])
                if authorships and authorships[0].get("author"):
                    author = authorships[0]["author"].get("display_name", "")
                    if len(authorships) > 1:
                        author += " et al."
                else:
                    author = ""

                # URL (prefer DOI)
                doi = work.get("doi", "")
                paper_url = doi if doi else work.get("id", "")

                # Publication info
                year = str(work.get("publication_year", ""))
                primary_loc = work.get("primary_location", {}) or {}
                source = primary_loc.get("source", {}) or {}
                journal = source.get("display_name", "")

                # Citations
                citations = work.get("cited_by_count", 0)

                # Concepts for domain detection
                concepts = [c.get("display_name", "") for c in work.get("concepts", [])[:5]]
                domain = self._detect_domain(title, concepts)

                # Study type
                work_type = work.get("type", "")
                study_type = self._detect_study_type(work_type, title)

                # Peer review status
                peer_reviewed = work_type != "preprint"

                results.append(SearchResult(
                    title=title,
                    url=paper_url,
                    snippet=abstract,
                    source_type=SourceType.ACADEMIC,
                    author=author,
                    date=year,
                    publication=journal or "OpenAlex",
                    citations=citations,
                    relevance_score=0.8,
                    study_type=study_type,
                    peer_reviewed=peer_reviewed,
                    sample_size=0,  # Would need full text
                    domain=domain,
                ))

        except Exception as e:
            print(f"OpenAlex search failed: {e}")

        return results


class BraveSearch:
    """
    Brave Search API for web results.

    Free tier: 2000 queries/month (1 req/sec limit)
    Needs API key from: https://brave.com/search/api/
    """

    BASE_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        }
        self.last_request = 0
        self.min_delay = 1.5  # Free tier rate limit

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Search the web via Brave."""
        results = []

        if not self.api_key:
            return results

        # Rate limiting
        elapsed = time.time() - self.last_request
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)

        encoded_query = urllib.parse.quote(query)
        url = f"{self.BASE_URL}?q={encoded_query}&count={limit}"

        try:
            request = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(request, timeout=15) as response:
                self.last_request = time.time()
                data = json.loads(response.read().decode())

                for result in data.get("web", {}).get("results", []):
                    # Detect source type from URL
                    source_type = self._detect_source_type(result.get("url", ""))

                    results.append(SearchResult(
                        title=result.get("title", ""),
                        url=result.get("url", ""),
                        snippet=result.get("description", ""),
                        source_type=source_type,
                        date=result.get("age", ""),  # Brave returns relative age
                        relevance_score=0.6,
                    ))

        except Exception as e:
            print(f"Brave search error: {e}")

        return results

    def _detect_source_type(self, url: str) -> SourceType:
        """Detect source type from URL."""
        url_lower = url.lower()

        if any(d in url_lower for d in ['.gov', '.edu', 'who.int', 'un.org']):
            return SourceType.OFFICIAL
        if any(d in url_lower for d in ['arxiv', 'pubmed', 'nature.com', 'science.org', 'ieee']):
            return SourceType.ACADEMIC
        if any(d in url_lower for d in ['statista', 'mckinsey', 'gartner', 'forrester']):
            return SourceType.INDUSTRY
        if any(d in url_lower for d in ['wikipedia', 'britannica']):
            return SourceType.REFERENCE
        if any(d in url_lower for d in ['reddit', 'medium', 'stackoverflow', 'quora']):
            return SourceType.SOCIAL

        return SourceType.NEWS


class DuckDuckGoSearch:
    """
    DuckDuckGo search (unofficial).

    No API key needed, but unofficial so may break.
    Uses the instant answer API + HTML scraping fallback.
    """

    INSTANT_URL = "https://api.duckduckgo.com/"

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Search via DuckDuckGo."""
        results = []

        # Try instant answers first
        encoded_query = urllib.parse.quote(query)
        url = f"{self.INSTANT_URL}?q={encoded_query}&format=json&no_html=1"

        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ResearchAgent/1.0"})
            with urllib.request.urlopen(request, timeout=10) as response:
                data = json.loads(response.read().decode())

                # Abstract (main result)
                if data.get("Abstract"):
                    results.append(SearchResult(
                        title=data.get("Heading", query),
                        url=data.get("AbstractURL", ""),
                        snippet=data.get("Abstract", ""),
                        source_type=SourceType.REFERENCE,
                        publication=data.get("AbstractSource", ""),
                    ))

                # Related topics
                for topic in data.get("RelatedTopics", [])[:limit-1]:
                    if isinstance(topic, dict) and topic.get("FirstURL"):
                        results.append(SearchResult(
                            title=topic.get("Text", "")[:100],
                            url=topic.get("FirstURL", ""),
                            snippet=topic.get("Text", ""),
                            source_type=SourceType.REFERENCE,
                        ))

        except Exception as e:
            print(f"DuckDuckGo search error: {e}")

        return results[:limit]


class MultiSearch:
    """
    Unified search across multiple sources.

    Aggregates results from academic and web sources,
    deduplicates, and ranks by relevance and evidence quality.

    Sources:
    - Semantic Scholar: peer-reviewed papers (all domains)
    - PubMed: biomedical/health literature
    - OpenAlex: broad academic coverage (engineering, physics, psychology, etc.)
    - arXiv: preprints (physics, CS, math, etc.)
    - Brave/DDG: web sources
    """

    # Domain detection keywords
    MEDICAL_KEYWORDS = {
        'drug', 'patient', 'clinical', 'treatment', 'therapy', 'disease',
        'medical', 'health', 'symptom', 'diagnosis', 'pharmaceutical',
        'medication', 'hospital', 'cancer', 'diabetes', 'cardiac', 'statin',
        'rhabdomyolysis', 'adverse', 'trial', 'efficacy', 'dosage',
    }

    PHYSICS_KEYWORDS = {
        'quantum', 'particle', 'physics', 'relativity', 'cosmology',
        'gravitational', 'electromagnetic', 'photon', 'electron', 'neutron',
        'accelerator', 'collider', 'thermodynamic', 'entropy',
    }

    ENGINEERING_KEYWORDS = {
        'engineering', 'mechanical', 'electrical', 'civil', 'structural',
        'robotics', 'automation', 'manufacturing', 'design', 'materials',
        'system', 'circuit', 'signal', 'control', 'optimization',
    }

    PSYCHOLOGY_KEYWORDS = {
        'psychology', 'cognitive', 'behavioral', 'mental', 'psychiatric',
        'therapy', 'anxiety', 'depression', 'personality', 'perception',
        'memory', 'learning', 'emotion', 'motivation', 'neuroscience',
        'brain', 'cognition', 'psychotherapy',
    }

    CS_KEYWORDS = {
        'algorithm', 'software', 'programming', 'machine learning', 'neural',
        'artificial intelligence', 'data structure', 'database', 'network',
        'computing', 'compiler', 'distributed', 'cryptography',
    }

    def __init__(self, brave_api_key: str = None, semantic_scholar_key: str = None,
                 pubmed_api_key: str = None, openalex_email: str = None):
        import os

        # Check environment variables if keys not provided
        brave_key = brave_api_key or os.environ.get("BRAVE_API_KEY")
        ss_key = semantic_scholar_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        pubmed_key = pubmed_api_key or os.environ.get("PUBMED_API_KEY")
        oa_email = openalex_email or os.environ.get("OPENALEX_EMAIL")

        # Initialize all search sources
        self.semantic_scholar = SemanticScholarSearch(api_key=ss_key)
        self.pubmed = PubMedSearch(api_key=pubmed_key)
        self.openalex = OpenAlexSearch(email=oa_email)
        self.arxiv = ArxivSearch()
        self.brave = BraveSearch(api_key=brave_key) if brave_key else None
        self.ddg = DuckDuckGoSearch()

    def _detect_domain(self, query: str) -> str:
        """Detect the research domain from query to route to best sources."""
        query_lower = query.lower()

        # Count keyword matches for each domain
        scores = {
            "medical": sum(1 for kw in self.MEDICAL_KEYWORDS if kw in query_lower),
            "physics": sum(1 for kw in self.PHYSICS_KEYWORDS if kw in query_lower),
            "engineering": sum(1 for kw in self.ENGINEERING_KEYWORDS if kw in query_lower),
            "psychology": sum(1 for kw in self.PSYCHOLOGY_KEYWORDS if kw in query_lower),
            "computer_science": sum(1 for kw in self.CS_KEYWORDS if kw in query_lower),
        }

        # Return domain with highest score, or "general" if no matches
        if max(scores.values()) == 0:
            return "general"
        return max(scores, key=scores.get)

    def _extract_key_terms(self, query: str) -> set[str]:
        """Extract key terms from query for relevance filtering."""
        # Common stop words to ignore
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
            'research', 'study', 'analysis', 'data', 'information', 'about',
        }
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        return {w for w in words if w not in stop_words}

    def _is_relevant(self, result: SearchResult, key_terms: set[str], min_matches: int = 1) -> bool:
        """Check if a result is relevant to the query."""
        if not key_terms:
            return True

        # Combine title and snippet for checking
        text = f"{result.title} {result.snippet}".lower()

        # Count how many key terms appear
        matches = sum(1 for term in key_terms if term in text)

        return matches >= min_matches

    def search(self, query: str, limit: int = 10,
               focus: str = "general") -> list[SearchResult]:
        """
        Search across all sources with intelligent domain routing.

        Args:
            query: Search query
            limit: Max results per source
            focus: "scientific", "market", or "general"
        """
        all_results = []

        # Extract key terms for relevance filtering
        key_terms = self._extract_key_terms(query)

        # Detect domain for intelligent source routing
        domain = self._detect_domain(query)
        print(f"[MultiSearch] Detected domain: {domain} for query: {query[:50]}...")

        # Academic sources (always search for scientific, optionally for others)
        if focus in ["scientific", "general"]:
            # Domain-specific routing
            if domain == "medical":
                # Prioritize PubMed for medical queries
                print("[MultiSearch] Using PubMed (medical domain)")
                pubmed_results = self.pubmed.search(query, limit=limit)
                all_results.extend(pubmed_results)
                # Also get Semantic Scholar for broader coverage
                ss_results = self.semantic_scholar.search(query, limit=limit // 2)
                all_results.extend(ss_results)

            elif domain in ["physics", "computer_science"]:
                # Prioritize arXiv for physics/CS
                print(f"[MultiSearch] Using arXiv ({domain} domain)")
                arxiv_results = self.arxiv.search(query, limit=limit)
                all_results.extend(arxiv_results)
                # OpenAlex for peer-reviewed
                oa_results = self.openalex.search(query, limit=limit // 2)
                all_results.extend(oa_results)

            elif domain in ["engineering", "psychology"]:
                # Use OpenAlex (good coverage for these)
                print(f"[MultiSearch] Using OpenAlex ({domain} domain)")
                oa_results = self.openalex.search(query, limit=limit)
                all_results.extend(oa_results)
                # Supplement with Semantic Scholar
                ss_results = self.semantic_scholar.search(query, limit=limit // 2)
                all_results.extend(ss_results)

            else:
                # General academic search - use multiple sources
                print("[MultiSearch] Using broad academic search (general domain)")
                ss_results = self.semantic_scholar.search(query, limit=limit)
                all_results.extend(ss_results)
                oa_results = self.openalex.search(query, limit=limit // 2)
                all_results.extend(oa_results)
                arxiv_results = self.arxiv.search(query, limit=limit // 2)
                all_results.extend(arxiv_results)

        # Web sources
        if focus in ["market", "general"]:
            if self.brave:
                brave_results = self.brave.search(query, limit=limit)
                all_results.extend(brave_results)
            else:
                # Fallback to DuckDuckGo
                ddg_results = self.ddg.search(query, limit=limit // 2)
                all_results.extend(ddg_results)

        # Market-specific: add industry search terms
        if focus == "market":
            industry_query = f"{query} market analysis industry report"
            if self.brave:
                industry_results = self.brave.search(industry_query, limit=limit // 2)
                all_results.extend(industry_results)

        # Filter out irrelevant results
        # Require at least 1 key term for general, 2 for scientific focus
        min_matches = 2 if focus == "scientific" else 1
        relevant_results = [r for r in all_results if self._is_relevant(r, key_terms, min_matches)]

        # If too few relevant results, be more lenient
        if len(relevant_results) < limit // 2 and all_results:
            relevant_results = [r for r in all_results if self._is_relevant(r, key_terms, 1)]

        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for r in relevant_results:
            url_key = r.url.lower().rstrip('/')
            if url_key not in seen_urls:
                seen_urls.add(url_key)
                unique_results.append(r)

        # Sort by relevance, evidence quality, and source quality
        def score(r: SearchResult) -> float:
            type_weights = {
                SourceType.ACADEMIC: 1.5,
                SourceType.OFFICIAL: 1.3,
                SourceType.INDUSTRY: 1.2,
                SourceType.NEWS: 1.0,
                SourceType.REFERENCE: 0.9,
                SourceType.SOCIAL: 0.6,
            }
            base = type_weights.get(r.source_type, 1.0)
            citation_boost = min(0.5, r.citations / 2000) if r.citations else 0

            # Boost results that match more key terms
            text = f"{r.title} {r.snippet}".lower()
            term_matches = sum(1 for term in key_terms if term in text)
            term_boost = min(0.5, term_matches * 0.15)

            # Evidence strength boost (for academic sources)
            evidence_boost = r.evidence_strength * 0.3 if r.source_type == SourceType.ACADEMIC else 0

            # Peer review boost
            peer_review_boost = 0.2 if r.peer_reviewed else 0

            return base + r.relevance_score + citation_boost + term_boost + evidence_boost + peer_review_boost

        unique_results.sort(key=score, reverse=True)

        # Log evidence quality summary
        study_type_counts = {}
        for r in unique_results[:limit]:
            st = r.study_type.value if r.study_type else "unknown"
            study_type_counts[st] = study_type_counts.get(st, 0) + 1
        if study_type_counts:
            print(f"[MultiSearch] Evidence types in top results: {study_type_counts}")

        return unique_results[:limit * 2]  # Return more for diversity

    def search_academic(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Search only academic sources (all domains)."""
        results = []
        results.extend(self.semantic_scholar.search(query, limit=limit))
        results.extend(self.openalex.search(query, limit=limit // 2))
        results.extend(self.arxiv.search(query, limit=limit // 2))
        return results

    def search_medical(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Search medical/biomedical sources (PubMed prioritized)."""
        results = []
        results.extend(self.pubmed.search(query, limit=limit))
        results.extend(self.semantic_scholar.search(query, limit=limit // 2))
        return results

    def search_engineering(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Search engineering/technical sources."""
        results = []
        results.extend(self.openalex.search(query, limit=limit))
        results.extend(self.semantic_scholar.search(query, limit=limit // 2))
        return results

    def search_physics(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Search physics sources (arXiv prioritized)."""
        results = []
        results.extend(self.arxiv.search(query, limit=limit))
        results.extend(self.openalex.search(query, limit=limit // 2))
        return results

    def search_web(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Search only web sources."""
        if self.brave:
            return self.brave.search(query, limit=limit)
        return self.ddg.search(query, limit=limit)


# =============================================================================
# DATA APIs - Real statistics and facts (not just papers about them)
# =============================================================================

@dataclass
class DataPoint:
    """A single data point from a data API."""
    source: str  # FRED, WorldBank, Wikipedia
    indicator: str  # What this measures
    value: str  # The value (string to handle various formats)
    date: str = ""  # When this was measured
    unit: str = ""  # Unit of measurement
    country: str = ""  # For World Bank
    url: str = ""  # Source URL
    context: str = ""  # Additional context


class FREDDataAPI:
    """
    FRED (Federal Reserve Economic Data) API.

    Access real economic data - GDP, unemployment, inflation, interest rates, etc.
    Free, no API key required for basic access.
    Docs: https://fred.stlouisfed.org/docs/api/
    """

    BASE_URL = "https://api.stlouisfed.org/fred"

    # Common series IDs for quick lookup
    COMMON_SERIES = {
        "gdp": "GDP",
        "unemployment": "UNRATE",
        "inflation": "CPIAUCSL",
        "interest rate": "FEDFUNDS",
        "federal funds": "FEDFUNDS",
        "housing": "HOUST",
        "housing starts": "HOUST",
        "consumer price": "CPIAUCSL",
        "cpi": "CPIAUCSL",
        "pce": "PCE",
        "personal consumption": "PCE",
        "retail sales": "RSXFS",
        "industrial production": "INDPRO",
        "employment": "PAYEMS",
        "nonfarm payroll": "PAYEMS",
        "mortgage rate": "MORTGAGE30US",
        "30 year mortgage": "MORTGAGE30US",
        "treasury": "DGS10",
        "10 year treasury": "DGS10",
        "s&p 500": "SP500",
        "stock market": "SP500",
        "money supply": "M2SL",
        "m2": "M2SL",
    }

    def __init__(self, api_key: str = None):
        import os
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")
        self.last_request = 0
        self.min_delay = 0.5

    def _find_series_id(self, query: str) -> str:
        """Find FRED series ID from query."""
        query_lower = query.lower()
        for keyword, series_id in self.COMMON_SERIES.items():
            if keyword in query_lower:
                return series_id
        return None

    def search_series(self, query: str, limit: int = 5) -> list[dict]:
        """Search for FRED data series."""
        elapsed = time.time() - self.last_request
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)

        encoded_query = urllib.parse.quote(query)
        url = f"{self.BASE_URL}/series/search?search_text={encoded_query}&limit={limit}&file_type=json"
        if self.api_key:
            url += f"&api_key={self.api_key}"

        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ResearchAgent/1.0"})
            with urllib.request.urlopen(request, timeout=10) as response:
                self.last_request = time.time()
                data = json.loads(response.read().decode())
                return data.get("seriess", [])
        except Exception as e:
            print(f"FRED search error: {e}")
            return []

    def get_latest_value(self, series_id: str) -> DataPoint:
        """Get the most recent value for a series."""
        elapsed = time.time() - self.last_request
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)

        url = f"{self.BASE_URL}/series/observations?series_id={series_id}&sort_order=desc&limit=1&file_type=json"
        if self.api_key:
            url += f"&api_key={self.api_key}"

        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ResearchAgent/1.0"})
            with urllib.request.urlopen(request, timeout=10) as response:
                self.last_request = time.time()
                data = json.loads(response.read().decode())

                obs = data.get("observations", [])
                if obs:
                    latest = obs[0]
                    return DataPoint(
                        source="FRED",
                        indicator=series_id,
                        value=latest.get("value", ""),
                        date=latest.get("date", ""),
                        url=f"https://fred.stlouisfed.org/series/{series_id}",
                    )
        except Exception as e:
            print(f"FRED data error: {e}")

        return None

    def get_data(self, query: str) -> list[DataPoint]:
        """Get economic data matching query."""
        results = []

        # Try direct series lookup first
        series_id = self._find_series_id(query)
        if series_id:
            dp = self.get_latest_value(series_id)
            if dp:
                results.append(dp)

        # Also search for related series
        series_list = self.search_series(query, limit=3)
        for series in series_list:
            sid = series.get("id")
            if sid and sid != series_id:
                dp = self.get_latest_value(sid)
                if dp:
                    dp.indicator = f"{series.get('title', sid)} ({sid})"
                    dp.unit = series.get("units", "")
                    results.append(dp)
                    if len(results) >= 3:
                        break

        return results


class WorldBankDataAPI:
    """
    World Bank Open Data API.

    Access global development indicators by country.
    Free, no API key required.
    Docs: https://datahelpdesk.worldbank.org/knowledgebase/topics/125589
    """

    BASE_URL = "https://api.worldbank.org/v2"

    # Common indicators
    COMMON_INDICATORS = {
        "gdp": "NY.GDP.MKTP.CD",
        "gdp per capita": "NY.GDP.PCAP.CD",
        "population": "SP.POP.TOTL",
        "life expectancy": "SP.DYN.LE00.IN",
        "literacy": "SE.ADT.LITR.ZS",
        "poverty": "SI.POV.DDAY",
        "unemployment": "SL.UEM.TOTL.ZS",
        "inflation": "FP.CPI.TOTL.ZG",
        "co2 emissions": "EN.ATM.CO2E.PC",
        "internet users": "IT.NET.USER.ZS",
        "electricity": "EG.ELC.ACCS.ZS",
        "mortality": "SP.DYN.IMRT.IN",
        "infant mortality": "SP.DYN.IMRT.IN",
        "fertility": "SP.DYN.TFRT.IN",
        "education": "SE.XPD.TOTL.GD.ZS",
        "health expenditure": "SH.XPD.CHEX.GD.ZS",
        "trade": "NE.TRD.GNFS.ZS",
        "fdi": "BX.KLT.DINV.WD.GD.ZS",
        "gini": "SI.POV.GINI",
    }

    # Country code mappings
    COUNTRY_CODES = {
        "usa": "US", "united states": "US", "america": "US",
        "uk": "GB", "united kingdom": "GB", "britain": "GB", "england": "GB",
        "china": "CN", "germany": "DE", "france": "FR", "japan": "JP",
        "india": "IN", "brazil": "BR", "russia": "RU", "canada": "CA",
        "australia": "AU", "mexico": "MX", "spain": "ES", "italy": "IT",
        "south korea": "KR", "korea": "KR", "netherlands": "NL",
        "switzerland": "CH", "sweden": "SE", "norway": "NO",
        "world": "WLD", "global": "WLD",
    }

    def __init__(self):
        self.last_request = 0
        self.min_delay = 0.3

    def _find_indicator(self, query: str) -> str:
        """Find indicator code from query."""
        query_lower = query.lower()
        for keyword, code in self.COMMON_INDICATORS.items():
            if keyword in query_lower:
                return code
        return None

    def _find_country(self, query: str) -> str:
        """Find country code from query."""
        query_lower = query.lower()
        for name, code in self.COUNTRY_CODES.items():
            if name in query_lower:
                return code
        return "WLD"  # Default to world

    def get_indicator(self, indicator: str, country: str = "WLD", years: int = 1) -> list[DataPoint]:
        """Get indicator data."""
        elapsed = time.time() - self.last_request
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)

        url = f"{self.BASE_URL}/country/{country}/indicator/{indicator}?format=json&per_page={years}&mrv={years}"

        results = []
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ResearchAgent/1.0"})
            with urllib.request.urlopen(request, timeout=10) as response:
                self.last_request = time.time()
                data = json.loads(response.read().decode())

                if len(data) >= 2 and data[1]:
                    for obs in data[1]:
                        if obs.get("value") is not None:
                            results.append(DataPoint(
                                source="World Bank",
                                indicator=obs.get("indicator", {}).get("value", indicator),
                                value=str(obs.get("value")),
                                date=obs.get("date", ""),
                                country=obs.get("country", {}).get("value", country),
                                url=f"https://data.worldbank.org/indicator/{indicator}",
                            ))
        except Exception as e:
            print(f"World Bank error: {e}")

        return results

    def get_data(self, query: str) -> list[DataPoint]:
        """Get World Bank data matching query."""
        indicator = self._find_indicator(query)
        country = self._find_country(query)

        if indicator:
            return self.get_indicator(indicator, country)
        return []


class WikipediaAPI:
    """
    Wikipedia REST API.

    Quick factual lookups and definitions.
    Free, no API key required.
    """

    BASE_URL = "https://en.wikipedia.org/api/rest_v1"
    SEARCH_URL = "https://en.wikipedia.org/w/api.php"

    def __init__(self):
        self.last_request = 0
        self.min_delay = 0.2

    def get_summary(self, title: str) -> DataPoint:
        """Get article summary."""
        elapsed = time.time() - self.last_request
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)

        encoded_title = urllib.parse.quote(title.replace(" ", "_"))
        url = f"{self.BASE_URL}/page/summary/{encoded_title}"

        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ResearchAgent/1.0"})
            with urllib.request.urlopen(request, timeout=10) as response:
                self.last_request = time.time()
                data = json.loads(response.read().decode())

                return DataPoint(
                    source="Wikipedia",
                    indicator=data.get("title", title),
                    value=data.get("extract", "")[:500],
                    context=data.get("description", ""),
                    url=data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                )
        except Exception as e:
            print(f"Wikipedia error: {e}")

        return None

    def search(self, query: str, limit: int = 3) -> list[str]:
        """Search for article titles."""
        elapsed = time.time() - self.last_request
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)

        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
        }
        url = f"{self.SEARCH_URL}?{urllib.parse.urlencode(params)}"

        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ResearchAgent/1.0"})
            with urllib.request.urlopen(request, timeout=10) as response:
                self.last_request = time.time()
                data = json.loads(response.read().decode())

                results = data.get("query", {}).get("search", [])
                return [r.get("title") for r in results if r.get("title")]
        except Exception as e:
            print(f"Wikipedia search error: {e}")

        return []

    def get_data(self, query: str) -> list[DataPoint]:
        """Get Wikipedia data for query."""
        results = []

        # Search for relevant articles
        titles = self.search(query, limit=2)

        for title in titles:
            dp = self.get_summary(title)
            if dp:
                results.append(dp)

        return results


class DataRetriever:
    """
    Unified data retrieval across FRED, World Bank, Wikipedia.

    Gets actual statistics and facts, not just papers about them.
    """

    # Keywords that suggest we need actual data
    DATA_KEYWORDS = {
        "current", "latest", "recent", "today", "now", "2024", "2023", "2025",
        "rate", "percentage", "number", "amount", "level", "value",
        "how much", "how many", "what is the",
    }

    # Domain routing
    ECONOMIC_KEYWORDS = {
        "gdp", "unemployment", "inflation", "interest", "fed", "economy",
        "recession", "growth", "trade", "deficit", "debt", "monetary",
        "fiscal", "treasury", "bond", "stock", "market", "housing",
        "mortgage", "employment", "wage", "income", "price", "cpi",
    }

    GLOBAL_KEYWORDS = {
        "country", "countries", "nation", "global", "world", "international",
        "population", "poverty", "literacy", "life expectancy", "development",
        "emission", "climate", "mortality", "fertility", "education spending",
    }

    def __init__(self):
        self.fred = FREDDataAPI()
        self.worldbank = WorldBankDataAPI()
        self.wikipedia = WikipediaAPI()

    def needs_data(self, query: str) -> bool:
        """Check if query would benefit from actual data."""
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.DATA_KEYWORDS)

    def get_data(self, query: str) -> list[DataPoint]:
        """Get relevant data for query."""
        results = []
        query_lower = query.lower()

        # Route to appropriate data sources
        if any(kw in query_lower for kw in self.ECONOMIC_KEYWORDS):
            print(f"[DataRetriever] Fetching economic data for: {query[:50]}...")
            results.extend(self.fred.get_data(query))

        if any(kw in query_lower for kw in self.GLOBAL_KEYWORDS):
            print(f"[DataRetriever] Fetching global indicators for: {query[:50]}...")
            results.extend(self.worldbank.get_data(query))

        # Wikipedia for definitions/context (always useful)
        wiki_data = self.wikipedia.get_data(query)
        if wiki_data:
            results.extend(wiki_data[:1])  # Just top result

        return results

    def format_for_research(self, data_points: list[DataPoint]) -> str:
        """Format data points for inclusion in research synthesis."""
        if not data_points:
            return ""

        lines = ["**Current Data:**"]
        for dp in data_points:
            if dp.source == "Wikipedia":
                lines.append(f"- {dp.indicator}: {dp.value[:200]}... [{dp.source}]")
            else:
                date_str = f" ({dp.date})" if dp.date else ""
                unit_str = f" {dp.unit}" if dp.unit else ""
                country_str = f" - {dp.country}" if dp.country else ""
                lines.append(f"- {dp.indicator}{country_str}: {dp.value}{unit_str}{date_str} [{dp.source}]")

        return "\n".join(lines)
