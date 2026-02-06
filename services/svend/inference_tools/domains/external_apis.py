"""
External API Tools - Wolfram, PubChem, Web Search

Tools that connect to external services for verified data.
These require API keys or network access.
"""

from typing import Optional, Dict, Any, List
import json
import os

from .registry import Tool, ToolParameter, ToolResult, ToolStatus, ToolRegistry


# ==================== WOLFRAM ALPHA ====================

class WolframAlphaTool:
    """
    WolframAlpha API integration.

    Provides verified factual data, computations, and knowledge queries.
    Requires WOLFRAM_APP_ID environment variable.
    """

    def __init__(self):
        self._client = None
        self.app_id = os.environ.get("WOLFRAM_APP_ID")

    def query(
        self,
        query: str,
        format: str = "short",
    ) -> Dict[str, Any]:
        """
        Query WolframAlpha.

        Args:
            query: Natural language query
            format: "short" (just answer), "full" (detailed), "steps" (with steps)
        """
        if not self.app_id:
            return {
                "success": False,
                "error": "WOLFRAM_APP_ID not set. Get one at https://developer.wolframalpha.com/",
            }

        try:
            import requests

            # Use Short Answers API for quick results
            if format == "short":
                url = "https://api.wolframalpha.com/v1/result"
                params = {
                    "appid": self.app_id,
                    "i": query,
                }
                response = requests.get(url, params=params, timeout=10)

                if response.status_code == 200:
                    return {
                        "success": True,
                        "query": query,
                        "result": response.text,
                        "format": "short",
                    }
                else:
                    return {
                        "success": False,
                        "error": f"API error: {response.status_code}",
                        "message": response.text,
                    }

            else:
                # Use Full Results API
                url = "https://api.wolframalpha.com/v2/query"
                params = {
                    "appid": self.app_id,
                    "input": query,
                    "format": "plaintext",
                    "output": "json",
                }

                if format == "steps":
                    params["podstate"] = "Step-by-step solution"

                response = requests.get(url, params=params, timeout=15)
                data = response.json()

                if data.get("queryresult", {}).get("success"):
                    pods = data["queryresult"].get("pods", [])
                    results = []

                    for pod in pods:
                        pod_data = {
                            "title": pod.get("title"),
                            "content": [],
                        }
                        for subpod in pod.get("subpods", []):
                            if subpod.get("plaintext"):
                                pod_data["content"].append(subpod["plaintext"])

                        if pod_data["content"]:
                            results.append(pod_data)

                    return {
                        "success": True,
                        "query": query,
                        "results": results,
                        "format": format,
                    }
                else:
                    return {
                        "success": False,
                        "error": "Query failed",
                        "tips": data.get("queryresult", {}).get("tips"),
                    }

        except ImportError:
            return {"success": False, "error": "requests library required"}
        except Exception as e:
            return {"success": False, "error": str(e)}


def wolfram_tool(
    query: str,
    format: Optional[str] = None,
) -> ToolResult:
    """Tool function for WolframAlpha."""
    wolfram = WolframAlphaTool()
    result = wolfram.query(query, format or "short")

    if result.get("success"):
        if result.get("format") == "short":
            output = result["result"]
        else:
            # Format full results
            output_parts = []
            for pod in result.get("results", []):
                output_parts.append(f"**{pod['title']}**")
                output_parts.extend(pod["content"])
                output_parts.append("")
            output = "\n".join(output_parts)

        return ToolResult(
            status=ToolStatus.SUCCESS,
            output=output,
            metadata=result,
        )
    else:
        return ToolResult(
            status=ToolStatus.ERROR,
            output=None,
            error=result.get("error"),
        )


def create_wolfram_tool() -> Tool:
    """Create the WolframAlpha tool."""
    return Tool(
        name="wolfram",
        description="Query WolframAlpha for verified factual data, computations, and knowledge. Use for: distances, populations, historical dates, scientific data, unit conversions with context, mathematical computations.",
        parameters=[
            ToolParameter(
                name="query",
                description="Natural language query (e.g., 'distance from Earth to Mars', 'population of France')",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="format",
                description="Response format: 'short' (just answer), 'full' (detailed), 'steps' (with steps)",
                type="string",
                required=False,
                enum=["short", "full", "steps"],
            ),
        ],
        execute_fn=wolfram_tool,
        timeout_ms=20000,
    )


# ==================== PUBCHEM ====================

class PubChemTool:
    """
    PubChem database integration.

    Query chemical compounds for properties, structures, and safety data.
    No API key required.
    """

    BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    def search_compound(
        self,
        identifier: str,
        id_type: str = "name",
    ) -> Dict[str, Any]:
        """
        Search for a compound.

        Args:
            identifier: Compound name, formula, CID, or SMILES
            id_type: Type of identifier (name, formula, cid, smiles)
        """
        try:
            import requests

            # Map identifier type to PubChem namespace
            namespace_map = {
                "name": "name",
                "formula": "formula",
                "cid": "cid",
                "smiles": "smiles",
                "inchi": "inchi",
            }
            namespace = namespace_map.get(id_type, "name")

            # Get CID first
            url = f"{self.BASE_URL}/compound/{namespace}/{identifier}/cids/JSON"
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                return {"success": False, "error": f"Compound not found: {identifier}"}

            data = response.json()
            cid = data["IdentifierList"]["CID"][0]

            return {
                "success": True,
                "cid": cid,
                "identifier": identifier,
            }

        except ImportError:
            return {"success": False, "error": "requests library required"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_properties(
        self,
        identifier: str,
        properties: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get compound properties.

        Args:
            identifier: Compound name or CID
            properties: List of properties to fetch
        """
        try:
            import requests

            # Default properties
            if properties is None:
                properties = [
                    "MolecularFormula",
                    "MolecularWeight",
                    "CanonicalSMILES",
                    "IUPACName",
                    "XLogP",
                    "TPSA",
                    "HBondDonorCount",
                    "HBondAcceptorCount",
                ]

            props_str = ",".join(properties)

            # Try as name first, then as CID
            for namespace in ["name", "cid"]:
                url = f"{self.BASE_URL}/compound/{namespace}/{identifier}/property/{props_str}/JSON"
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    props = data["PropertyTable"]["Properties"][0]

                    return {
                        "success": True,
                        "compound": identifier,
                        "cid": props.get("CID"),
                        "properties": props,
                    }

            return {"success": False, "error": f"Compound not found: {identifier}"}

        except ImportError:
            return {"success": False, "error": "requests library required"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_synonyms(
        self,
        identifier: str,
    ) -> Dict[str, Any]:
        """Get compound synonyms/alternative names."""
        try:
            import requests

            url = f"{self.BASE_URL}/compound/name/{identifier}/synonyms/JSON"
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                return {"success": False, "error": f"Compound not found: {identifier}"}

            data = response.json()
            synonyms = data["InformationList"]["Information"][0]["Synonym"]

            return {
                "success": True,
                "compound": identifier,
                "synonyms": synonyms[:20],  # Limit to 20
                "total": len(synonyms),
            }

        except ImportError:
            return {"success": False, "error": "requests library required"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_safety(
        self,
        identifier: str,
    ) -> Dict[str, Any]:
        """Get GHS safety information."""
        try:
            import requests

            # First get CID
            search_result = self.search_compound(identifier)
            if not search_result.get("success"):
                return search_result

            cid = search_result["cid"]

            # Get GHS classification
            url = f"{self.BASE_URL}/compound/cid/{cid}/property/GHSClassification/JSON"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                # Parse GHS data
                return {
                    "success": True,
                    "compound": identifier,
                    "cid": cid,
                    "ghs_data": data,
                }

            # If no GHS data, return basic info
            return {
                "success": True,
                "compound": identifier,
                "cid": cid,
                "ghs_data": None,
                "note": "No GHS classification data available",
            }

        except ImportError:
            return {"success": False, "error": "requests library required"}
        except Exception as e:
            return {"success": False, "error": str(e)}


def pubchem_tool(
    operation: str,
    identifier: str,
    properties: Optional[str] = None,
) -> ToolResult:
    """Tool function for PubChem."""
    pubchem = PubChemTool()

    try:
        props_list = json.loads(properties) if properties else None
    except:
        props_list = None

    if operation == "properties":
        result = pubchem.get_properties(identifier, props_list)
    elif operation == "synonyms":
        result = pubchem.get_synonyms(identifier)
    elif operation == "safety":
        result = pubchem.get_safety(identifier)
    elif operation == "search":
        result = pubchem.search_compound(identifier)
    else:
        return ToolResult(
            status=ToolStatus.INVALID_INPUT,
            output=None,
            error=f"Unknown operation: {operation}",
        )

    if result.get("success"):
        return ToolResult(
            status=ToolStatus.SUCCESS,
            output=json.dumps(result, indent=2),
            metadata=result,
        )
    else:
        return ToolResult(
            status=ToolStatus.ERROR,
            output=None,
            error=result.get("error"),
        )


def create_pubchem_tool() -> Tool:
    """Create the PubChem tool."""
    return Tool(
        name="pubchem",
        description="Query PubChem database for chemical compound information. Get molecular properties, SMILES, synonyms, and safety data. No API key required.",
        parameters=[
            ToolParameter(
                name="operation",
                description="Operation: 'properties', 'synonyms', 'safety', 'search'",
                type="string",
                required=True,
                enum=["properties", "synonyms", "safety", "search"],
            ),
            ToolParameter(
                name="identifier",
                description="Compound identifier (name like 'aspirin', or CID like '2244')",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="properties",
                description="JSON array of specific properties to fetch (optional)",
                type="string",
                required=False,
            ),
        ],
        execute_fn=pubchem_tool,
        timeout_ms=15000,
    )


# ==================== WEB SEARCH ====================

class WebSearchTool:
    """
    Web search integration.

    Supports multiple backends (configurable).
    Default: DuckDuckGo (no API key required).
    """

    def search(
        self,
        query: str,
        max_results: int = 5,
        backend: str = "duckduckgo",
    ) -> Dict[str, Any]:
        """
        Search the web.

        Args:
            query: Search query
            max_results: Maximum number of results
            backend: Search backend to use
        """
        if backend == "duckduckgo":
            return self._search_ddg(query, max_results)
        else:
            return {"success": False, "error": f"Unknown backend: {backend}"}

    def _search_ddg(
        self,
        query: str,
        max_results: int,
    ) -> Dict[str, Any]:
        """Search using DuckDuckGo."""
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))

            formatted = []
            for r in results:
                formatted.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })

            return {
                "success": True,
                "query": query,
                "results": formatted,
                "count": len(formatted),
            }

        except ImportError:
            return {
                "success": False,
                "error": "duckduckgo-search library required. Install with: pip install duckduckgo-search",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


def web_search_tool(
    query: str,
    max_results: Optional[int] = None,
) -> ToolResult:
    """Tool function for web search."""
    search = WebSearchTool()
    result = search.search(query, max_results or 5)

    if result.get("success"):
        # Format results nicely
        output_parts = [f"Search results for: {query}\n"]
        for i, r in enumerate(result["results"], 1):
            output_parts.append(f"{i}. {r['title']}")
            output_parts.append(f"   {r['url']}")
            output_parts.append(f"   {r['snippet'][:200]}...")
            output_parts.append("")

        return ToolResult(
            status=ToolStatus.SUCCESS,
            output="\n".join(output_parts),
            metadata=result,
        )
    else:
        return ToolResult(
            status=ToolStatus.ERROR,
            output=None,
            error=result.get("error"),
        )


def create_web_search_tool() -> Tool:
    """Create the web search tool."""
    return Tool(
        name="web_search",
        description="Search the web for current information. Use for recent events, current data, or fact-checking beyond training knowledge cutoff.",
        parameters=[
            ToolParameter(
                name="query",
                description="Search query",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="max_results",
                description="Maximum number of results (default: 5)",
                type="number",
                required=False,
            ),
        ],
        execute_fn=web_search_tool,
        timeout_ms=15000,
    )


# ==================== FRED (Federal Reserve Economic Data) ====================

class FREDTool:
    """
    FRED API integration.

    Access Federal Reserve Economic Data - economic indicators, interest rates,
    GDP, unemployment, inflation, etc.
    No API key required for basic access.
    """

    BASE_URL = "https://api.stlouisfed.org/fred"

    def __init__(self):
        # FRED allows limited access without key, but key provides more
        self.api_key = os.environ.get("FRED_API_KEY", "")

    def search_series(
        self,
        query: str,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Search for economic data series."""
        try:
            import requests

            # Use the FRED series search
            url = f"{self.BASE_URL}/series/search"
            params = {
                "search_text": query,
                "limit": limit,
                "file_type": "json",
            }
            if self.api_key:
                params["api_key"] = self.api_key

            response = requests.get(url, params=params, timeout=10)

            if response.status_code != 200:
                return {"success": False, "error": f"FRED API error: {response.status_code}"}

            data = response.json()
            series_list = data.get("seriess", [])

            results = []
            for s in series_list[:limit]:
                results.append({
                    "id": s.get("id"),
                    "title": s.get("title"),
                    "frequency": s.get("frequency"),
                    "units": s.get("units"),
                    "observation_start": s.get("observation_start"),
                    "observation_end": s.get("observation_end"),
                })

            return {
                "success": True,
                "query": query,
                "series": results,
                "count": len(results),
            }

        except ImportError:
            return {"success": False, "error": "requests library required"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_series_data(
        self,
        series_id: str,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Get observations for a data series."""
        try:
            import requests

            url = f"{self.BASE_URL}/series/observations"
            params = {
                "series_id": series_id,
                "limit": limit,
                "sort_order": "desc",
                "file_type": "json",
            }
            if self.api_key:
                params["api_key"] = self.api_key

            response = requests.get(url, params=params, timeout=10)

            if response.status_code != 200:
                return {"success": False, "error": f"Series not found: {series_id}"}

            data = response.json()
            observations = data.get("observations", [])

            # Format observations
            results = []
            for obs in observations:
                results.append({
                    "date": obs.get("date"),
                    "value": obs.get("value"),
                })

            return {
                "success": True,
                "series_id": series_id,
                "observations": results,
                "count": len(results),
            }

        except ImportError:
            return {"success": False, "error": "requests library required"}
        except Exception as e:
            return {"success": False, "error": str(e)}


def fred_tool(
    operation: str,
    query: Optional[str] = None,
    series_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> ToolResult:
    """Tool function for FRED."""
    fred = FREDTool()

    if operation == "search":
        if not query:
            return ToolResult(
                status=ToolStatus.INVALID_INPUT,
                output=None,
                error="Query required for search operation",
            )
        result = fred.search_series(query, limit or 10)
    elif operation == "data":
        if not series_id:
            return ToolResult(
                status=ToolStatus.INVALID_INPUT,
                output=None,
                error="series_id required for data operation",
            )
        result = fred.get_series_data(series_id, limit or 100)
    else:
        return ToolResult(
            status=ToolStatus.INVALID_INPUT,
            output=None,
            error=f"Unknown operation: {operation}",
        )

    if result.get("success"):
        return ToolResult(
            status=ToolStatus.SUCCESS,
            output=json.dumps(result, indent=2),
            metadata=result,
        )
    else:
        return ToolResult(
            status=ToolStatus.ERROR,
            output=None,
            error=result.get("error"),
        )


def create_fred_tool() -> Tool:
    """Create the FRED tool."""
    return Tool(
        name="fred",
        description="Access Federal Reserve Economic Data (FRED). Search and retrieve economic indicators: GDP, unemployment, inflation, interest rates, housing data, and more. No API key required.",
        parameters=[
            ToolParameter(
                name="operation",
                description="Operation: 'search' (find series), 'data' (get observations)",
                type="string",
                required=True,
                enum=["search", "data"],
            ),
            ToolParameter(
                name="query",
                description="Search query (for 'search' operation)",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="series_id",
                description="FRED series ID like 'GDP', 'UNRATE', 'FEDFUNDS' (for 'data' operation)",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="limit",
                description="Maximum results to return",
                type="number",
                required=False,
            ),
        ],
        execute_fn=fred_tool,
        timeout_ms=15000,
    )


# ==================== ARXIV ====================

class ArxivTool:
    """
    arXiv API integration.

    Search and retrieve academic papers in physics, math, CS, and more.
    No API key required.
    """

    BASE_URL = "http://export.arxiv.org/api/query"

    def search(
        self,
        query: str,
        max_results: int = 10,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search arXiv papers.

        Args:
            query: Search query
            max_results: Maximum papers to return
            category: Optional category filter (cs.AI, cs.LG, stat.ML, etc.)
        """
        try:
            import requests
            import xml.etree.ElementTree as ET

            # Build search query
            search_query = query
            if category:
                search_query = f"cat:{category} AND all:{query}"

            params = {
                "search_query": f"all:{search_query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }

            response = requests.get(self.BASE_URL, params=params, timeout=15)

            if response.status_code != 200:
                return {"success": False, "error": f"arXiv API error: {response.status_code}"}

            # Parse XML response
            root = ET.fromstring(response.content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            papers = []
            for entry in root.findall("atom:entry", ns):
                paper = {
                    "id": entry.find("atom:id", ns).text.split("/abs/")[-1] if entry.find("atom:id", ns) is not None else "",
                    "title": entry.find("atom:title", ns).text.strip().replace("\n", " ") if entry.find("atom:title", ns) is not None else "",
                    "summary": entry.find("atom:summary", ns).text.strip()[:500] if entry.find("atom:summary", ns) is not None else "",
                    "published": entry.find("atom:published", ns).text[:10] if entry.find("atom:published", ns) is not None else "",
                    "authors": [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns) if a.find("atom:name", ns) is not None][:5],
                    "pdf_url": None,
                }

                # Get PDF link
                for link in entry.findall("atom:link", ns):
                    if link.get("title") == "pdf":
                        paper["pdf_url"] = link.get("href")
                        break

                papers.append(paper)

            return {
                "success": True,
                "query": query,
                "papers": papers,
                "count": len(papers),
            }

        except ImportError:
            return {"success": False, "error": "requests library required"}
        except Exception as e:
            return {"success": False, "error": str(e)}


def arxiv_tool(
    query: str,
    max_results: Optional[int] = None,
    category: Optional[str] = None,
) -> ToolResult:
    """Tool function for arXiv."""
    arxiv = ArxivTool()
    result = arxiv.search(query, max_results or 10, category)

    if result.get("success"):
        # Format results
        output_parts = [f"arXiv papers for: {query}\n"]
        for i, p in enumerate(result["papers"], 1):
            authors = ", ".join(p["authors"][:3])
            if len(p["authors"]) > 3:
                authors += " et al."
            output_parts.append(f"{i}. {p['title']}")
            output_parts.append(f"   Authors: {authors}")
            output_parts.append(f"   Published: {p['published']}")
            output_parts.append(f"   ID: {p['id']}")
            output_parts.append(f"   {p['summary'][:200]}...")
            output_parts.append("")

        return ToolResult(
            status=ToolStatus.SUCCESS,
            output="\n".join(output_parts),
            metadata=result,
        )
    else:
        return ToolResult(
            status=ToolStatus.ERROR,
            output=None,
            error=result.get("error"),
        )


def create_arxiv_tool() -> Tool:
    """Create the arXiv tool."""
    return Tool(
        name="arxiv",
        description="Search arXiv for academic papers in physics, mathematics, computer science, statistics, and more. Find research papers, preprints, and technical reports. No API key required.",
        parameters=[
            ToolParameter(
                name="query",
                description="Search query (e.g., 'transformer neural networks', 'reinforcement learning')",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="max_results",
                description="Maximum papers to return (default: 10)",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="category",
                description="arXiv category filter (cs.AI, cs.LG, stat.ML, math.ST, physics, etc.)",
                type="string",
                required=False,
            ),
        ],
        execute_fn=arxiv_tool,
        timeout_ms=20000,
    )


# ==================== WIKIPEDIA ====================

class WikipediaTool:
    """
    Wikipedia API integration.

    Search and retrieve Wikipedia articles for general knowledge.
    No API key required.
    """

    BASE_URL = "https://en.wikipedia.org/api/rest_v1"
    SEARCH_URL = "https://en.wikipedia.org/w/api.php"

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """Search Wikipedia articles."""
        try:
            import requests

            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": limit,
                "format": "json",
            }

            response = requests.get(self.SEARCH_URL, params=params, timeout=10)

            if response.status_code != 200:
                return {"success": False, "error": f"Wikipedia API error: {response.status_code}"}

            data = response.json()
            results = data.get("query", {}).get("search", [])

            articles = []
            for r in results:
                articles.append({
                    "title": r.get("title"),
                    "snippet": r.get("snippet", "").replace("<span class=\"searchmatch\">", "").replace("</span>", ""),
                    "pageid": r.get("pageid"),
                })

            return {
                "success": True,
                "query": query,
                "articles": articles,
                "count": len(articles),
            }

        except ImportError:
            return {"success": False, "error": "requests library required"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_summary(
        self,
        title: str,
    ) -> Dict[str, Any]:
        """Get article summary."""
        try:
            import requests
            from urllib.parse import quote

            url = f"{self.BASE_URL}/page/summary/{quote(title)}"
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                return {"success": False, "error": f"Article not found: {title}"}

            data = response.json()

            return {
                "success": True,
                "title": data.get("title"),
                "description": data.get("description"),
                "extract": data.get("extract"),
                "url": data.get("content_urls", {}).get("desktop", {}).get("page"),
            }

        except ImportError:
            return {"success": False, "error": "requests library required"}
        except Exception as e:
            return {"success": False, "error": str(e)}


def wikipedia_tool(
    operation: str,
    query: Optional[str] = None,
    title: Optional[str] = None,
) -> ToolResult:
    """Tool function for Wikipedia."""
    wiki = WikipediaTool()

    if operation == "search":
        if not query:
            return ToolResult(
                status=ToolStatus.INVALID_INPUT,
                output=None,
                error="Query required for search operation",
            )
        result = wiki.search(query)
    elif operation == "summary":
        if not title:
            return ToolResult(
                status=ToolStatus.INVALID_INPUT,
                output=None,
                error="Title required for summary operation",
            )
        result = wiki.get_summary(title)
    else:
        return ToolResult(
            status=ToolStatus.INVALID_INPUT,
            output=None,
            error=f"Unknown operation: {operation}",
        )

    if result.get("success"):
        if operation == "summary":
            output = f"**{result['title']}**\n\n{result.get('description', '')}\n\n{result['extract']}\n\nURL: {result.get('url', 'N/A')}"
        else:
            output_parts = [f"Wikipedia results for: {query}\n"]
            for i, a in enumerate(result["articles"], 1):
                output_parts.append(f"{i}. {a['title']}")
                output_parts.append(f"   {a['snippet']}")
                output_parts.append("")
            output = "\n".join(output_parts)

        return ToolResult(
            status=ToolStatus.SUCCESS,
            output=output,
            metadata=result,
        )
    else:
        return ToolResult(
            status=ToolStatus.ERROR,
            output=None,
            error=result.get("error"),
        )


def create_wikipedia_tool() -> Tool:
    """Create the Wikipedia tool."""
    return Tool(
        name="wikipedia",
        description="Search and retrieve Wikipedia articles for general knowledge, definitions, historical facts, and explanations. No API key required.",
        parameters=[
            ToolParameter(
                name="operation",
                description="Operation: 'search' (find articles), 'summary' (get article summary)",
                type="string",
                required=True,
                enum=["search", "summary"],
            ),
            ToolParameter(
                name="query",
                description="Search query (for 'search' operation)",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="title",
                description="Article title (for 'summary' operation)",
                type="string",
                required=False,
            ),
        ],
        execute_fn=wikipedia_tool,
        timeout_ms=15000,
    )


# ==================== WORLD BANK ====================

class WorldBankTool:
    """
    World Bank API integration.

    Access global development indicators by country.
    No API key required.
    """

    BASE_URL = "https://api.worldbank.org/v2"

    def get_indicator(
        self,
        indicator: str,
        country: str = "all",
        years: int = 10,
    ) -> Dict[str, Any]:
        """
        Get indicator data for a country.

        Common indicators:
        - NY.GDP.MKTP.CD: GDP (current US$)
        - SP.POP.TOTL: Population
        - SI.POV.DDAY: Poverty rate
        - SE.ADT.LITR.ZS: Literacy rate
        - SP.DYN.LE00.IN: Life expectancy
        """
        try:
            import requests

            url = f"{self.BASE_URL}/country/{country}/indicator/{indicator}"
            params = {
                "format": "json",
                "per_page": years,
                "mrv": years,  # Most recent values
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code != 200:
                return {"success": False, "error": f"World Bank API error: {response.status_code}"}

            data = response.json()

            if len(data) < 2 or not data[1]:
                return {"success": False, "error": f"No data found for indicator: {indicator}"}

            # Parse data
            observations = []
            for obs in data[1]:
                if obs.get("value") is not None:
                    observations.append({
                        "country": obs.get("country", {}).get("value"),
                        "year": obs.get("date"),
                        "value": obs.get("value"),
                    })

            indicator_info = data[1][0] if data[1] else {}

            return {
                "success": True,
                "indicator": indicator,
                "indicator_name": indicator_info.get("indicator", {}).get("value"),
                "country": country,
                "observations": observations,
                "count": len(observations),
            }

        except ImportError:
            return {"success": False, "error": "requests library required"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def search_indicators(
        self,
        query: str,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Search for available indicators."""
        try:
            import requests

            url = f"{self.BASE_URL}/indicator"
            params = {
                "format": "json",
                "per_page": 1000,  # Get many to filter locally
            }

            response = requests.get(url, params=params, timeout=15)

            if response.status_code != 200:
                return {"success": False, "error": f"World Bank API error: {response.status_code}"}

            data = response.json()

            if len(data) < 2:
                return {"success": False, "error": "No indicators found"}

            # Filter by query
            query_lower = query.lower()
            matches = []
            for ind in data[1]:
                name = ind.get("name", "").lower()
                if query_lower in name:
                    matches.append({
                        "id": ind.get("id"),
                        "name": ind.get("name"),
                        "source": ind.get("source", {}).get("value"),
                    })
                    if len(matches) >= limit:
                        break

            return {
                "success": True,
                "query": query,
                "indicators": matches,
                "count": len(matches),
            }

        except ImportError:
            return {"success": False, "error": "requests library required"}
        except Exception as e:
            return {"success": False, "error": str(e)}


def worldbank_tool(
    operation: str,
    indicator: Optional[str] = None,
    country: Optional[str] = None,
    query: Optional[str] = None,
) -> ToolResult:
    """Tool function for World Bank."""
    wb = WorldBankTool()

    if operation == "data":
        if not indicator:
            return ToolResult(
                status=ToolStatus.INVALID_INPUT,
                output=None,
                error="Indicator required for data operation",
            )
        result = wb.get_indicator(indicator, country or "all")
    elif operation == "search":
        if not query:
            return ToolResult(
                status=ToolStatus.INVALID_INPUT,
                output=None,
                error="Query required for search operation",
            )
        result = wb.search_indicators(query)
    else:
        return ToolResult(
            status=ToolStatus.INVALID_INPUT,
            output=None,
            error=f"Unknown operation: {operation}",
        )

    if result.get("success"):
        return ToolResult(
            status=ToolStatus.SUCCESS,
            output=json.dumps(result, indent=2),
            metadata=result,
        )
    else:
        return ToolResult(
            status=ToolStatus.ERROR,
            output=None,
            error=result.get("error"),
        )


def create_worldbank_tool() -> Tool:
    """Create the World Bank tool."""
    return Tool(
        name="worldbank",
        description="Access World Bank global development indicators. Get GDP, population, poverty rates, literacy, life expectancy, and more by country. No API key required.",
        parameters=[
            ToolParameter(
                name="operation",
                description="Operation: 'data' (get indicator values), 'search' (find indicators)",
                type="string",
                required=True,
                enum=["data", "search"],
            ),
            ToolParameter(
                name="indicator",
                description="World Bank indicator code (e.g., 'NY.GDP.MKTP.CD' for GDP, 'SP.POP.TOTL' for population)",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="country",
                description="Country code (e.g., 'US', 'GB', 'CN') or 'all' for all countries",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="query",
                description="Search query for finding indicators (for 'search' operation)",
                type="string",
                required=False,
            ),
        ],
        execute_fn=worldbank_tool,
        timeout_ms=20000,
    )


# ==================== REGISTRATION ====================

def register_external_tools(registry: ToolRegistry) -> None:
    """Register all external API tools."""
    registry.register(create_wolfram_tool())
    registry.register(create_pubchem_tool())
    registry.register(create_web_search_tool())
    registry.register(create_fred_tool())
    registry.register(create_arxiv_tool())
    registry.register(create_wikipedia_tool())
    registry.register(create_worldbank_tool())
