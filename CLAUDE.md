# Kjerne — Project Context

## What is this?

Kjerne is the monorepo for **Svend** — a hypothesis-driven decision science platform. Two products:

- **DSW (Decision Science Workbench)** — statistical analysis, DOE, SPC, Bayesian reasoning
- **Whiteboard** — visual knowledge graph for causal reasoning

Target: $5/month subscription, launching May 2026. Competing against Minitab ($1,851/yr) and JMP ($1,320-$8,400/yr).

## Critical Context

- **This machine is the production server.** Do not restructure, rename, or move files without explicit approval.
- **Git is our safety net.** A prior version of Claude destroyed work. Always commit before making changes.
- All changes must be logged in `log.md` at the root.
- Technical debt is tracked in `.kjerne/DEBT.md`. Process for closing items is in `DEBT-001.md`.

## Architecture

```
~/kjerne/                          # Root — production server
├── CLAUDE.md                      # This file
├── STANDARD.md                    # 5S lab standards
├── log.md                         # Change log (all edits)
├── DEBT-001.md                    # Debt closure process
│
├── core/                          # Shared utilities
│   ├── llm.py                     # LLM loading (Qwen)
│   └── quality.py                 # Quality framework
│
├── services/svend/                # The Svend product
│   ├── CLAUDE.md                  # Reasoning model docs
│   ├── agents/agents/             # Agent implementations
│   │   ├── experimenter/          # DOE agent (agent.py, doe.py, stats.py)
│   │   ├── researcher/            # Research agent (DISABLED at router)
│   │   ├── coder/                 # Code execution agent (DISABLED at router)
│   │   ├── writer/                # Writing agent
│   │   ├── reviewer/              # Review agent
│   │   └── guide/                 # Interview/decision guide
│   │
│   ├── web/                       # Django web application
│   │   ├── manage.py
│   │   ├── svend_web/             # Django project settings
│   │   │   ├── settings.py
│   │   │   └── urls.py            # Root URL config
│   │   │
│   │   ├── core/                  # Core Django app (TARGET models)
│   │   │   └── models/
│   │   │       ├── project.py     # Project, Dataset, ExperimentDesign
│   │   │       └── hypothesis.py  # Hypothesis, Evidence, EvidenceLink
│   │   │
│   │   ├── agents_api/            # API views (main application logic)
│   │   │   ├── models.py          # Problem model (DEPRECATED — JSON blobs)
│   │   │   ├── dsw_views.py       # DSW statistical engine (8,439 lines, 64+ analyses)
│   │   │   ├── dsw_urls.py
│   │   │   ├── spc_views.py       # SPC endpoints
│   │   │   ├── spc.py             # SPC engine (control charts, capability)
│   │   │   ├── spc_urls.py
│   │   │   ├── experimenter_views.py  # DOE endpoints (1,758 lines)
│   │   │   ├── experimenter_urls.py
│   │   │   ├── problem_views.py   # Problem/hypothesis management
│   │   │   ├── problem_urls.py
│   │   │   ├── synara_views.py    # Synara belief engine API
│   │   │   ├── synara_urls.py
│   │   │   ├── synara/            # Synara engine (in-memory)
│   │   │   │   ├── synara.py      # Orchestrator
│   │   │   │   ├── belief.py      # Bayesian update math
│   │   │   │   ├── dsl.py         # Hypothesis DSL parser
│   │   │   │   ├── logic_engine.py
│   │   │   │   ├── kernel.py      # Data structures
│   │   │   │   └── llm_interface.py  # LLM prompts (STUBBED)
│   │   │   ├── forecast_views.py  # Time series forecasting
│   │   │   ├── learn_views.py     # Learning center
│   │   │   ├── views.py           # Agent dispatch + add_finding_to_problem()
│   │   │   └── urls.py            # Agent router (researcher/coder DISABLED)
│   │   │
│   │   ├── accounts/              # Auth, billing, permissions
│   │   │   └── permissions.py     # @gated, @require_auth decorators
│   │   │
│   │   └── templates/             # Django templates (HTML)
│   │
│   └── site/                      # Landing page (svend.ai)
│       └── index.html
│
├── .kjerne/                       # Meta-tooling
│   ├── DEBT.md                    # Technical debt tracker
│   ├── config.json                # Lab config
│   └── snapshots/                 # Point-in-time snapshots
│
└── .gitignore
```

## Data Model (Two Systems — Migration Pending)

### Current: `agents_api.Problem` (DEPRECATED)
- Stores hypotheses and evidence as **JSON blobs** in JSONFields
- `Problem.add_evidence()` appends to a JSON list
- Used by: problem_views.py, experimenter_views.py (partial), spc_views.py (partial)
- Helper: `add_finding_to_problem()` in views.py

### Target: `core.Project` → `core.Hypothesis` → `core.Evidence` → `core.EvidenceLink`
- Proper FK relationships with Bayesian probability tracking
- `Hypothesis.apply_evidence()` uses `core.bayesian.BayesianUpdater`
- `EvidenceLink` stores likelihood ratios per hypothesis
- Used by: core views (limited)
- **Migration is P1 debt**

### Model Relationships (Target)

```
Project (uuid)
  ├── Hypothesis (uuid, FK→Project)
  │     ├── prior_probability, current_probability
  │     ├── probability_history (JSON)
  │     └── evidence_links → EvidenceLink
  │           ├── likelihood_ratio
  │           ├── direction (supports/opposes/neutral)
  │           └── FK→Evidence
  ├── Evidence (uuid)
  │     ├── source_type (observation/analysis/experiment/...)
  │     ├── result_type (statistical/categorical/quantitative/qualitative)
  │     ├── p_value, effect_size, confidence_interval, sample_size
  │     └── confidence (0.0-1.0)
  ├── Dataset (uuid, FK→Project)
  └── ExperimentDesign (uuid, FK→Project, FK→Hypothesis)
```

## API Surface

| Route | Module | Integration with Problems |
|-------|--------|--------------------------|
| `/api/dsw/` | dsw_views.py | **NONE** — no problem_id support |
| `/api/spc/` | spc_views.py | **PARTIAL** — 3/7 endpoints |
| `/api/experimenter/` | experimenter_views.py | **PARTIAL** — 2/9 endpoints |
| `/api/problems/` | problem_views.py | Full |
| `/api/synara/` | synara_views.py | Full (in-memory only) |
| `/api/agents/` | views.py | EDA only; researcher/coder DISABLED |
| `/api/forecast/` | forecast_views.py | None |
| `/api/learn/` | learn_views.py | None |
| `/api/triage/` | triage_views.py | None |
| `/api/core/` | core/views.py | Full (new FK model) |

## Integration Pattern

Existing pattern for connecting analysis results to problems (from experimenter_views.py):

```python
problem_id = data.get("problem_id")
if problem_id:
    try:
        problem = Problem.objects.get(id=problem_id, user=request.user)
        problem.add_evidence(
            summary="Description of finding",
            evidence_type="data_analysis",  # or "experiment"
            source="Module Name",
        )
        problem.save()
    except Problem.DoesNotExist:
        pass  # Silently skip if problem not found
```

Helper in views.py:
```python
add_finding_to_problem(user, problem_id, summary, evidence_type, source, supports, weakens)
```

## Key Libraries

- **Django** — web framework, ORM
- **scipy, statsmodels** — statistical tests
- **sklearn** — ML models
- **plotly** — visualization (JSON format, client-side rendering)
- **Anthropic API** — Claude for Opus escalation
- **Qwen** — local LLM for code generation and analyst chat

## Serving

- **Gunicorn** behind **Cloudflare Tunnel**
- Static files served by Django (collectstatic)
- SQLite database (production — on this machine)

## Working Conventions

- Python 3.10+
- Type hints encouraged
- All views use `@csrf_exempt`, `@gated` (feature gate), `@require_auth` decorators
- JSON in, JSON out (all API endpoints return JsonResponse)
- Plotly charts returned as JSON traces (client renders)
- All changes logged in `log.md`
- Debt tracked in `.kjerne/DEBT.md`, closed via `DEBT-001.md` process
