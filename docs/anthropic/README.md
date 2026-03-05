# Anthropic (Claude API) — LLM Service

**Purpose:** AI-powered features across the platform (chat, analysis, reasoning, guides).
**Dashboard:** https://console.anthropic.com
**Compliance:** SOC 2 Type II

---

## How It Works

```
User action → Django view → LLM Manager → Anthropic Messages API → Response → User
```

Claude powers conversational AI features, hypothesis generation, RCA critique, and the decision guide. Model tier is gated by subscription level.

## Credentials

| Secret | Location | Purpose |
|--------|----------|---------|
| `SVEND_ANTHROPIC_API_KEY` | `~/.svend_env` / `svend_config/config.py` | API authentication |

## Model Tiering

| Subscription | Model | Purpose |
|-------------|-------|---------|
| Free / Founder | claude-3-5-haiku-20241022 | Fast, cost-effective |
| Pro / Team | claude-sonnet-4-20250514 | Balanced quality/cost |
| Enterprise | claude-opus-4-20250514 | Maximum capability |

## Integration Points

| File | Purpose |
|------|---------|
| `agents_api/llm_manager.py` | Centralized LLM dispatch (model selection, token limits) |
| `inference/flywheel.py` | Confidence-based escalation (Haiku → Sonnet → Opus) |
| `agents_api/synara/llm_interface.py` | Synara belief engine LLM integration |
| `agents_api/guide_views.py` | AI decision guide (rate-limited) |
| `agents_api/rca_views.py` | Root cause analysis critique |
| `chat/` | Conversation system (multi-turn) |

## Flywheel Escalation

The flywheel system auto-escalates to a more capable model when confidence is low:

- **Confidence > 0.7:** Respond with current model
- **Confidence 0.4–0.7:** Escalate one tier
- **Confidence < 0.4:** Auto-escalate to highest available tier

## Rate Limits

- Per-user rate limiting built into DRF throttling
- Guide endpoint has separate stricter limits (prevents abuse on free tier)
- Max tokens: 4096 default, configurable per request

## Data Exposure

User queries and analysis context are sent to Anthropic's API. No PII beyond what the user types into prompts. Anthropic's data retention policy applies (no training on API data per their terms).

## Common Tasks

### Check API usage
Anthropic Console → Usage → filter by date

### Test API connection
```bash
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $SVEND_ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-3-5-haiku-20241022","max_tokens":10,"messages":[{"role":"user","content":"ping"}]}'
```

## Cost Management

- Haiku is ~10x cheaper than Opus — tiering by subscription keeps costs proportional to revenue
- Flywheel avoids unnecessary Opus calls for simple queries
- Monitor monthly spend in Anthropic Console → Billing
