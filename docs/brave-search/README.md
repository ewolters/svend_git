# Brave Search API — Research Agent

**Purpose:** Web search capability for the researcher agent.
**Dashboard:** https://api.search.brave.com/app/dashboard

---

## How It Works

```
Researcher agent query → Brave Search API → Results ranked → Returned to user
```

The researcher agent uses Brave Search alongside arXiv and Semantic Scholar for multi-source research.

## Credentials

| Secret | Location | Purpose |
|--------|----------|---------|
| `BRAVE_API_KEY` | `services/svend/agents/.env` | API authentication |

## Integration Points

| File | Purpose |
|------|---------|
| `agents/agent_core/search.py` | Multi-source search (Brave + arXiv + Semantic Scholar) |
| `agents/researcher/` | Research agent that invokes search |

## Limits

| Plan | Requests/month | Cost |
|------|---------------|------|
| Free | 2,000 | $0 |
| Base | 20,000 | $5/mo |

Rate limiting: 1.5-second delay between requests (configured in `search.py`).

## Other Research APIs

These are free, no-auth APIs used alongside Brave:

| API | Endpoint | Purpose | Rate Limit |
|-----|----------|---------|------------|
| arXiv | `http://export.arxiv.org/api/query` | Academic preprints | 3s delay |
| Semantic Scholar | `https://api.semanticscholar.org/` | Paper metadata, citations | 1s delay |
