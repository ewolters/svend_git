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

    def __init__(self):
        self.last_request = 0
        self.min_delay = 3.0  # Seconds between requests

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Search arXiv for preprints."""
        results = []

        # Rate limiting
        elapsed = time.time() - self.last_request
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)

        # Build query - arXiv uses specific syntax
        # Search in title, abstract, and all fields
        encoded_query = urllib.parse.quote(query)
        url = (
            f"{self.BASE_URL}"
            f"?search_query=all:{encoded_query}"
            f"&start=0"
            f"&max_results={limit}"
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

                # Get category
                category = entry.find("arxiv:primary_category", ns)
                venue = category.get("term", "arXiv") if category is not None else "arXiv"

                results.append(SearchResult(
                    title=title.text.strip().replace("\n", " ") if title is not None and title.text else "",
                    url=pdf_link,
                    snippet=summary.text.strip().replace("\n", " ")[:500] if summary is not None and summary.text else "",
                    source_type=SourceType.ACADEMIC,
                    author=author,
                    date=date,
                    publication=f"arXiv ({venue})",
                    citations=0,  # arXiv doesn't provide this
                    relevance_score=0.7,  # Default for preprints
                ))

        except Exception as e:
            print(f"arXiv search failed after retries: {e}")

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
    deduplicates, and ranks by relevance.
    """

    def __init__(self, brave_api_key: str = None, semantic_scholar_key: str = None):
        import os

        # Check environment variables if keys not provided
        brave_key = brave_api_key or os.environ.get("BRAVE_API_KEY")
        ss_key = semantic_scholar_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

        self.semantic_scholar = SemanticScholarSearch(api_key=ss_key)
        self.arxiv = ArxivSearch()
        self.brave = BraveSearch(api_key=brave_key) if brave_key else None
        self.ddg = DuckDuckGoSearch()

    def search(self, query: str, limit: int = 10,
               focus: str = "general") -> list[SearchResult]:
        """
        Search across all sources.

        Args:
            query: Search query
            limit: Max results per source
            focus: "scientific", "market", or "general"
        """
        all_results = []

        # Academic sources (always search for scientific, optionally for others)
        if focus in ["scientific", "general"]:
            # Semantic Scholar for peer-reviewed
            ss_results = self.semantic_scholar.search(query, limit=limit)
            all_results.extend(ss_results)

            # arXiv for preprints
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

        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for r in all_results:
            url_key = r.url.lower().rstrip('/')
            if url_key not in seen_urls:
                seen_urls.add(url_key)
                unique_results.append(r)

        # Sort by relevance and source quality
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
            return base + r.relevance_score + citation_boost

        unique_results.sort(key=score, reverse=True)

        return unique_results[:limit * 2]  # Return more for diversity

    def search_academic(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Search only academic sources."""
        results = []
        results.extend(self.semantic_scholar.search(query, limit=limit))
        results.extend(self.arxiv.search(query, limit=limit // 2))
        return results

    def search_web(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Search only web sources."""
        if self.brave:
            return self.brave.search(query, limit=limit)
        return self.ddg.search(query, limit=limit)
