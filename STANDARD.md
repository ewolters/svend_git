# Kjerne Lab Standards

**Version:** 2.0
**Last updated:** 2026-02-12
**Philosophy:** 5S + Traceability + Quality Gates

---

## 5S Methodology

The Kjerne lab follows 5S principles adapted for software development.

### 1. SEIRI (Sort)

**Rule:** Every file has a purpose. No dead code, no orphans.

| Keep | Remove |
|------|--------|
| Active source code | Commented-out code blocks |
| Tests with coverage | Tests that don't run |
| Documentation that's read | READMEs nobody updates |
| Config that's used | Legacy config files |

**Enforcement:** Manual review before commit. Check for unused imports, unreachable code, empty files.

### 2. SEITON (Set in Order)

**Rule:** Everything has a place. Find anything in 3 seconds.

```
~/kjerne/
├── core/                           # SHARED utilities
│   ├── llm.py                      # LLM loading (Qwen, Claude via Anthropic)
│   ├── quality.py                  # Quality framework
│   ├── bayesian.py                 # Unified BayesianUpdater
│   ├── reasoning.py                # Reasoning graph
│   ├── intent.py                   # Intent parsing
│   ├── executor.py                 # Tool execution
│   ├── context.py                  # Context management
│   ├── sources.py                  # Source tracking
│   ├── search.py                   # Search utilities
│   └── verifier.py                 # Verification logic
│
├── services/svend/                 # The Svend product
│   ├── agents/agents/              # Agent implementations
│   │   ├── experimenter/           # DOE agent (agent.py, doe.py, stats.py)
│   │   ├── guide/                  # Interview/decision guide
│   │   ├── writer/                 # Writing agent
│   │   ├── reviewer/               # Review agent
│   │   ├── researcher/             # Research agent
│   │   └── coder/                  # Code execution agent
│   │
│   ├── web/                        # Django web application
│   │   ├── manage.py
│   │   ├── svend/                  # Django project (settings.py, urls.py)
│   │   │
│   │   ├── accounts/               # Auth, billing, permissions
│   │   │   ├── models.py           # User, Subscription, InviteCode
│   │   │   ├── constants.py        # Tier, TIER_FEATURES, TIER_LIMITS
│   │   │   ├── permissions.py      # Access control decorators
│   │   │   └── billing.py          # Stripe webhook integration
│   │   │
│   │   ├── api/                    # Chat API + auth endpoints
│   │   │   └── views.py            # me(), login(), register(), update_profile()
│   │   │
│   │   ├── agents_api/             # Analysis & tool endpoints (15 modules)
│   │   │   ├── dsw_views.py        # Statistical engine (64+ analyses)
│   │   │   ├── spc_views.py        # SPC / control charts
│   │   │   ├── experimenter_views.py  # DOE
│   │   │   ├── synara_views.py     # Bayesian belief engine
│   │   │   ├── whiteboard_views.py # Collaborative whiteboard
│   │   │   ├── a3_views.py         # A3 problem-solving reports
│   │   │   ├── vsm_views.py        # Value stream mapping
│   │   │   ├── rca_views.py        # Root cause analysis
│   │   │   ├── forecast_views.py   # Time series forecasting
│   │   │   ├── learn_views.py      # Learning center (10 modules, 47 sections)
│   │   │   ├── guide_views.py      # AI guide (Enterprise)
│   │   │   ├── hoshin_views.py     # Hoshin Kanri CI (Enterprise)
│   │   │   ├── triage_views.py     # Data triage
│   │   │   ├── workflow_views.py   # Workflows
│   │   │   ├── problem_views.py    # Problem management
│   │   │   ├── synara/             # Synara engine (belief.py, dsl.py, kernel.py)
│   │   │   ├── models.py           # Problem, VSM, HoshinProject, ActionItem
│   │   │   └── learn_content.py    # Learning curriculum content
│   │   │
│   │   ├── core/                   # Target data models
│   │   │   └── models/
│   │   │       ├── project.py      # Project, Dataset, ExperimentDesign
│   │   │       └── hypothesis.py   # Hypothesis, Evidence, EvidenceLink
│   │   │
│   │   ├── chat/                   # Chat/conversation models
│   │   ├── workbench/              # Knowledge graph views
│   │   ├── inference/              # LLM pipeline (Qwen, cognition)
│   │   ├── forge/                  # Synthetic data (Forge)
│   │   ├── files/                  # File management
│   │   └── templates/              # Django templates (40+ HTML files)
│   │
│   └── site/                       # Landing page (svend.ai)
│
├── .kjerne/                        # Meta-tooling
│   ├── config.json                 # Lab version, service versions
│   ├── DEBT.md                     # Technical debt tracker
│   └── snapshots/                  # Point-in-time snapshots
│
├── CLAUDE.md                       # Project context for AI assistants
├── STANDARD.md                     # This file
├── log.md                          # Change log (all edits)
├── DEBT-001.md                     # Debt closure process
└── DSW_gaps.md                     # Competitive gap analysis vs Minitab/JMP
```

**Naming Conventions:**
- Directories: `lowercase_snake`
- Python files: `lowercase_snake.py`
- Classes: `PascalCase`
- Functions: `lowercase_snake`
- Constants: `UPPER_SNAKE`
- URL patterns: `kebab-case` (e.g., `/api/dsw/paired-t-test/`)
- Template files: `lowercase_snake.html`

### 3. SEISO (Shine)

**Rule:** Clean as you go. Technical debt is tracked, not ignored.

**Before committing:**
1. `python3 manage.py check` — Django system check
2. `python3 manage.py makemigrations --check --dry-run` — no pending migrations
3. Manual smoke test of affected pages
4. Log the change in `log.md`

**Debt tracking:**
- File: `~/kjerne/.kjerne/DEBT.md`
- Format: `[SERVICE] Description | Added: DATE | Priority: P1/P2/P3`
- Closure process: `DEBT-001.md`
- Review weekly

### 4. SEIKETSU (Standardize)

**Rule:** Same patterns everywhere. Copy-paste should work.

**API Endpoint Pattern (agents_api):**
```python
@csrf_exempt
@gated_paid         # or @rate_limited, @require_enterprise, etc.
def endpoint(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    data = json.loads(request.body)
    # ... process ...
    return JsonResponse({...})
```

**API Endpoint Pattern (api/):**
```python
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def endpoint(request):
    # ... process ...
    return Response({...})
```

**Error Handling:**
- All API endpoints return JSON
- Standard error shape: `{"error": "message"}`
- HTTP status codes: 400 (bad input), 401 (not authenticated), 403 (tier too low), 429 (rate limited), 500 (server error)

### 5. SHITSUKE (Sustain)

**Rule:** The system enforces itself where possible. Humans review the rest.

**Automated:**
- Git tracks all changes (this machine is production)
- `log.md` documents every change
- Feature gating enforced by decorators (can't bypass without code change)

**Manual review:**
- Weekly: Review `DEBT.md`, close or defer
- Monthly: Review service boundaries
- Quarterly: Review `core/` for bloat

---

## Subscription Tiers

Five-tier pricing model. Single source of truth: `accounts/constants.py`.

| Tier | Price | Queries/Day | Key Features |
|------|-------|-------------|--------------|
| **FREE** | $0 | 5 | Basic DSW only |
| **FOUNDER** | $19/mo | 50 | All PRO features + priority support (100 slots, locked rate) |
| **PRO** | $29/mo | 50 | Full tools, basic ML, Forge API |
| **TEAM** | $79/mo | 200 | + Collaboration |
| **ENTERPRISE** | $199/mo | 1000 | + Anthropic AI, + Hoshin Kanri CI |

**Feature flags** (8 total in `TIER_FEATURES`):
`basic_dsw`, `basic_ml`, `full_tools`, `ai_assistant`, `collaboration`, `forge_api`, `hoshin_kanri`, `priority_support`

**Helper functions:**
- `has_feature(tier, feature)` — check if tier has feature
- `is_paid_tier(tier)` — check if tier is paid
- `get_daily_limit(tier)` — get query limit
- `can_use_anthropic(tier)` — Enterprise only

---

## Feature Gating

### Backend (accounts/permissions.py)

Seven decorators for access control:

| Decorator | What it checks |
|-----------|---------------|
| `@require_auth` | Authenticated only |
| `@rate_limited` | Auth + daily query limit |
| `@gated_paid` | Auth + `full_tools` feature + rate limit |
| `@require_paid` | Auth + any paid tier |
| `@require_team` | Auth + TEAM+ tier |
| `@require_enterprise` | Auth + ENTERPRISE tier |
| `@require_feature(name)` | Auth + specific feature flag |
| `@require_ml` | Auth + `basic_ml` feature |

**Decorator stacking order:** `@csrf_exempt` → `@gated_paid` (or equivalent) → view function.

### Frontend (base_app.html)

- `window.svendUser` — global set by `checkAuth()` with tier, features, email_verified, etc.
- `svendUserReady` — CustomEvent dispatched when user data loaded. Pages listen for this to check access.
- **403 interceptor** — global fetch wrapper that catches "Upgrade required" responses and shows upgrade modal
- **Tool card gating** — dashboard cards with `data-feature="full_tools"` get `.locked` class + PRO badge for free users

**Page-level gate pattern:**
```javascript
window.addEventListener('svendUserReady', function(e) {
    if (!e.detail.features || !e.detail.features.full_tools) {
        showUpgradeModal();
    }
});
```

---

## Theme System

Six CSS themes defined in `base_app.html` using `[data-theme]` attribute on `<html>`:

| Theme | Type | Description |
|-------|------|-------------|
| (default) | Dark | Forest at Night — green accents on dark green |
| `light` | Light | Light — green accents on white |
| `nordic` | Light | Nordic Frost — cool blue-gray |
| `sandstone` | Light | Sandstone — warm cream/parchment |
| `midnight` | Dark | Midnight — deep blue |
| `contrast` | Dark | High Contrast — true black, OLED-friendly |

**WCAG AA compliance:** All themes pass 4.5:1 contrast for normal text, 3:1 for large text.

**Semantic CSS variables** (defined per theme):
- Colors: `--text-primary`, `--text-secondary`, `--text-dim`, `--bg-primary`, `--bg-secondary`, `--bg-tertiary`, `--card-bg`, `--border`
- Accents: `--accent-primary`, `--accent-blue`, `--accent-gold`, `--accent-orange`, `--accent-purple`
- Status: `--error`, `--warning`, `--info`, `--success`
- Semantic: `--error-dim`, `--error-border`, `--warning-dim`, `--warning-border`, `--accent-primary-dim`, `--accent-primary-border`

**Rule:** Never hardcode colors in templates. Always use CSS variables. Modal backgrounds use `var(--card-bg)`, not hex values.

---

## Template Pattern

All app pages extend `base_app.html`:

```html
{% extends "base_app.html" %}
{% block title %}Page Title - SVEND{% endblock %}
{% block content %}
    <!-- page content -->
{% endblock %}
{% block scripts %}
    <!-- page-specific JS -->
{% endblock %}
```

**base_app.html provides:**
- Navigation header with dropdowns
- Theme system (6 themes + JS switcher)
- Auth check (`checkAuth()` → `window.svendUser` → `svendUserReady` event)
- Verification banner (email not verified)
- Upgrade modal (triggered by 403)
- SvendTheme JS object (chart colors, CSS var getters)
- Chart.js, KaTeX, SmilesDrawer (loaded on demand)

**Standalone pages** (login, register, landing, etc.) define their own CSS variables and don't extend base_app.html.

---

## API Surface

| Route | Module | Purpose | Gate |
|-------|--------|---------|------|
| `/api/` | api/views.py | Chat, auth, user management | Mixed |
| `/api/dsw/` | dsw_views.py | Statistical analysis (64+ analyses) | @rate_limited |
| `/api/spc/` | spc_views.py | SPC / control charts | @rate_limited |
| `/api/experimenter/` | experimenter_views.py | DOE | @gated_paid |
| `/api/synara/` | synara_views.py | Bayesian belief engine | @gated_paid |
| `/api/whiteboard/` | whiteboard_views.py | Collaborative whiteboard | @gated_paid |
| `/api/a3/` | a3_views.py | A3 problem-solving | @gated_paid |
| `/api/vsm/` | vsm_views.py | Value stream mapping | @gated_paid |
| `/api/rca/` | rca_views.py | Root cause analysis | @gated_paid |
| `/api/forecast/` | forecast_views.py | Time series forecasting | @gated_paid |
| `/api/learn/` | learn_views.py | Learning center | @require_auth |
| `/api/guide/` | guide_views.py | AI guide | @require_enterprise |
| `/api/hoshin/` | hoshin_views.py | Hoshin Kanri CI | @require_enterprise |
| `/api/triage/` | triage_views.py | Data triage | @rate_limited |
| `/api/workflows/` | workflow_views.py | Workflows | @rate_limited |
| `/api/core/` | core/views.py | Projects, hypotheses, evidence | @require_auth |
| `/api/workbench/` | workbench/views.py | Knowledge graph | @require_auth |
| `/api/forge/` | forge/views.py | Synthetic data | @require_paid |
| `/api/files/` | files/views.py | File management | @require_auth |

---

## Data Model (Migration in Progress)

### Current State: Dual-Write (Phase 2)

Two systems coexist:
- **Legacy:** `agents_api.Problem` — stores hypotheses/evidence as JSON blobs in JSONFields
- **Target:** `core.Project` → `core.Hypothesis` → `core.Evidence` → `core.EvidenceLink` — proper FK relationships

Phase 2 (dual-write) is complete: all write paths create both Problem JSON and core.Project FKs. All read paths use core.Project with JSON fallback.

**Phase 3 (pending):** Drop JSON blob fields from Problem model entirely. Blocked until all workflows confirmed working on FK-only reads.

### Target Model Relationships

```
Project (uuid)
  ├── Hypothesis (uuid, FK→Project)
  │     ├── prior_probability, current_probability
  │     ├── probability_history (JSON)
  │     └── evidence_links → EvidenceLink
  │           ├── likelihood_ratio
  │           ├── direction (supports/opposes/neutral)
  │           └── FK→Evidence
  ├── Evidence (uuid, source_type, result_type, p_value, effect_size, ...)
  ├── Dataset (uuid, FK→Project)
  └── ExperimentDesign (uuid, FK→Project, FK→Hypothesis)
```

---

## User Profile

### Model Fields (accounts/models.py)

| Field | Type | Purpose |
|-------|------|---------|
| `tier` | CharField(choices) | Subscription tier (FREE→ENTERPRISE) |
| `display_name` | CharField | Display name |
| `bio` | TextField | User bio |
| `industry` | CharField(choices) | Manufacturing, healthcare, etc. |
| `role` | CharField(choices) | Engineer, manager, analyst, etc. |
| `experience_level` | CharField(choices) | Beginner, intermediate, advanced |
| `organization_size` | CharField(choices) | Small, medium, large, enterprise |
| `preferences` | JSONField | Flexible settings (show_reasoning, auto_scroll) |
| `current_theme` | CharField | Active theme name |
| `email_verified` | BooleanField | Email verification status |

### API

- `GET /api/auth/me/` — returns full user profile + features dict
- `PATCH /api/auth/profile/` — updates allowed profile fields with validation

---

## Production Environment

- **This machine is production.** Do not restructure, rename, or move files without explicit approval.
- **Database:** SQLite (on this machine)
- **Server:** Gunicorn behind Cloudflare Tunnel
- **Git is the safety net.** Always commit before making changes.
- **All changes logged** in `log.md` at the root.
- **Debt tracked** in `.kjerne/DEBT.md`, closed via `DEBT-001.md` process.

---

## Emergency Procedures

### Production Issue

1. Check gunicorn logs: `journalctl -u gunicorn` or process output
2. If Django import error: fix the import, reload gunicorn (`kill -HUP`)
3. If data issue: check SQLite directly
4. Document the incident in `log.md`
5. If severe: `git stash` or `git checkout` to revert

### Key Commands

```bash
# Activate venv
source ~/Desktop/svend_transfer/k/kjerne/.venv/bin/activate

# Reload gunicorn (graceful)
kill -HUP $(pgrep -f gunicorn | head -1)

# Django management
cd ~/kjerne/services/svend/web
python3 manage.py check
python3 manage.py makemigrations --check --dry-run
python3 manage.py migrate

# Verify no broken templates
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/login/
```

---

## Versioning

Lab and service versions tracked in `.kjerne/config.json`:

```json
{
  "version": "2.0.0",
  "services": {
    "svend": "2.0.0",
    "scrub": "1.0.0",
    "forge": "0.1.0"
  }
}
```

Bump rules:
- **MAJOR** — Breaking API changes, model migrations
- **MINOR** — New features, backward compatible
- **PATCH** — Bug fixes only
