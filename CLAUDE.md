# Kjerne — Project Context

## What is this?

Kjerne is the monorepo for **Svend** — a hypothesis-driven decision science platform. Core surfaces:

- **DSW (Decision Science Workbench)** — 64+ statistical analyses, DOE, SPC, Bayesian reasoning
- **Whiteboard** — collaborative visual knowledge graph for causal reasoning
- **Quality Tools** — RCA, FMEA, A3 reports, Hoshin Kanri, VSM
- **Forge** — synthetic data generation service
- **Triage** — data cleaning and validation
- **Learn** — courses, assessments, and certification tracking

Pricing: Free / Founder $19/mo / Pro $29/mo / Team $79/mo / Enterprise $199/mo. Live in production. Competing against Minitab ($1,851/yr) and JMP ($1,320-$8,400/yr).

## Critical Context

- **This machine is the production server.** Do not restructure, rename, or move files without explicit approval.
- **Git is our safety net.** A prior version of Claude destroyed work. Always commit before making changes.
- All changes must be logged in `log.md` at the root.
- Technical debt is tracked in `.kjerne/DEBT.md`. Process for closing items is in `DEBT-001.md`.
- Unified workbench direction is documented in `services/svend/reference_docs/ARCHITECTURE.md`.

## Architecture

```
~/kjerne/                              # Root — production server
├── CLAUDE.md                          # This file
├── STANDARD.md                        # 5S lab standards (v2.0)
├── DSW_gaps.md                        # Competitive gap analysis vs Minitab/JMP
├── log.md                             # Change log (all edits)
├── DEBT-001.md                        # Debt closure process
│
├── core/                              # Shared utilities
│   ├── llm.py                         # LLM loading (Qwen)
│   └── quality.py                     # Quality framework
│
├── services/svend/                    # The Svend product
│   ├── reference_docs/                # Architecture docs
│   │   └── ARCHITECTURE.md            # Unified workbench vision (Jan 2026)
│   │
│   ├── agents/agents/                 # Agent implementations
│   │   ├── experimenter/              # DOE agent (agent.py, doe.py, stats.py)
│   │   ├── researcher/                # Research agent (enabled)
│   │   ├── coder/                     # Code execution agent (DISABLED at router)
│   │   ├── writer/                    # Writing agent
│   │   ├── reviewer/                  # Review agent
│   │   └── guide/                     # Interview/decision guide
│   │
│   ├── web/                           # Django web application
│   │   ├── manage.py
│   │   ├── svend_web/                 # Django project settings
│   │   │   ├── settings.py
│   │   │   └── urls.py                # Root URL config
│   │   │
│   │   ├── core/                      # Core Django app (TARGET models)
│   │   │   └── models/
│   │   │       ├── project.py         # Project, Dataset, ExperimentDesign
│   │   │       ├── hypothesis.py      # Hypothesis, Evidence, EvidenceLink
│   │   │       ├── tenant.py          # Tenant, Membership, OrgInvitation
│   │   │       └── graph.py           # KnowledgeGraph, Entity, Relationship
│   │   │
│   │   ├── agents_api/                # API views (main application logic)
│   │   │   ├── models.py              # Problem (DEPRECATED), Board, A3Report, Report,
│   │   │   │                          #   FMEA, FMEARow, RCASession, ValueStreamMap,
│   │   │   │                          #   HoshinProject, ActionItem, Site, SavedModel,
│   │   │   │                          #   Workflow, DSWResult, TriageResult, etc.
│   │   │   ├── dsw_views.py           # DSW statistical engine (64+ analyses)
│   │   │   ├── spc_views.py           # SPC endpoints (control charts, capability, gage R&R)
│   │   │   ├── spc.py                 # SPC engine
│   │   │   ├── experimenter_views.py  # DOE endpoints
│   │   │   ├── problem_views.py       # Problem/hypothesis management (legacy)
│   │   │   ├── synara_views.py        # Synara belief engine API
│   │   │   ├── synara/                # Synara engine
│   │   │   │   ├── synara.py          # Orchestrator
│   │   │   │   ├── belief.py          # Bayesian update math
│   │   │   │   ├── dsl.py             # Hypothesis DSL parser
│   │   │   │   ├── logic_engine.py
│   │   │   │   ├── kernel.py          # Data structures
│   │   │   │   └── llm_interface.py   # LLM prompts (Claude)
│   │   │   ├── forecast_views.py      # Time series forecasting
│   │   │   ├── learn_views.py         # Learning center + assessments
│   │   │   ├── rca_views.py           # Root cause analysis
│   │   │   ├── a3_views.py            # A3 reports
│   │   │   ├── report_views.py        # Generic reports (CAPA, 8D)
│   │   │   ├── fmea_views.py          # FMEA
│   │   │   ├── hoshin_views.py        # Hoshin Kanri (enterprise)
│   │   │   ├── vsm_views.py           # Value stream mapping
│   │   │   ├── whiteboard_views.py    # Collaborative whiteboard
│   │   │   ├── triage_views.py        # Data cleaning/validation
│   │   │   ├── guide_views.py         # AI decision guide (rate-limited)
│   │   │   ├── views.py               # Agent dispatch + add_finding_to_problem()
│   │   │   └── urls.py                # Agent router (coder DISABLED)
│   │   │
│   │   ├── api/                       # Content, automation, feedback
│   │   │   ├── models.py              # BlogPost, OnboardingSurvey, EmailCampaign,
│   │   │   │                          #   Experiment, AutomationRule, Feedback, etc.
│   │   │   ├── views.py               # Auth, chat, feedback, internal dashboard,
│   │   │   │                          #   blog mgmt, A/B testing, automation, autopilot
│   │   │   └── urls.py
│   │   │
│   │   ├── chat/                      # LLM conversation system
│   │   │   └── models.py              # Conversation, Message, TraceLog, TrainingCandidate
│   │   │
│   │   ├── workbench/                 # Unified workbench (new platform)
│   │   │   ├── models.py              # Project, Hypothesis, Evidence, Workbench,
│   │   │   │                          #   Artifact, KnowledgeGraph, EpistemicLog
│   │   │   └── views.py               # Full CRUD + graph + epistemic log
│   │   │
│   │   ├── forge/                     # Synthetic data generation
│   │   │   ├── models.py              # APIKey, Job, SchemaTemplate
│   │   │   └── views.py               # generate, job status, download
│   │   │
│   │   ├── files/                     # File storage
│   │   │   └── models.py              # UserFile, UserQuota
│   │   │
│   │   ├── tempora/                   # Task scheduler (distributed)
│   │   │   └── models.py              # CognitiveTask, Schedule, DeadLetterEntry,
│   │   │                              #   CircuitBreakerState, ClusterMember
│   │   │
│   │   ├── accounts/                  # Auth, billing, permissions
│   │   │   ├── models.py              # User (custom), Subscription, InviteCode
│   │   │   └── permissions.py         # @gated, @require_auth, @gated_paid, etc.
│   │   │
│   │   └── templates/                 # 42 Django templates (HTML)
│   │
│   └── site/                          # Landing page (svend.ai)
│       └── index.html
│
├── .kjerne/                           # Meta-tooling
│   ├── DEBT.md                        # Technical debt tracker
│   ├── config.json                    # Lab config
│   └── snapshots/                     # Point-in-time snapshots
│
└── .gitignore
```

## Data Model (Two Systems — Dual-Write Phase 2)

### Legacy: `agents_api.Problem` (DEPRECATED)
- Stores hypotheses and evidence as **JSON blobs** in JSONFields
- `Problem.add_evidence()` appends to a JSON list
- `Problem.core_project` FK bridges to new system during migration
- Used by: problem_views.py, some older tool integrations
- Helper: `add_finding_to_problem()` in views.py

### Target: `core.Project` → `core.Hypothesis` → `core.Evidence` → `core.EvidenceLink`
- Proper FK relationships with Bayesian probability tracking
- `Hypothesis.apply_evidence()` uses `core.bayesian.BayesianUpdater`
- `EvidenceLink` stores likelihood ratios per hypothesis
- Used by: core views, workbench, and newer tool integrations
- **Phase 3 (drop JSON blobs) is P3 debt — not yet scheduled**

### Multi-Tenancy: `core.Tenant` → `core.Membership`
- Individual users: `Project.user` set, `Project.tenant` null
- Enterprise teams: share a `Tenant`, shared `KnowledgeGraph`
- Roles: owner / admin / member / viewer
- `OrgInvitation` for email-based team invites

### Model Relationships (Target)

```
Tenant (uuid, optional)
  └── Membership (user + role)

Project (uuid, FK→User or FK→Tenant)
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

KnowledgeGraph (uuid)
  ├── Entity (concept/variable/actor/event/data_source/finding)
  └── Relationship (causes/prevents/enables/correlates_with/is_part_of/...)
```

## API Surface

| Route | Module | Purpose |
|-------|--------|---------|
| `/api/dsw/` | dsw_views.py | Statistical analysis (64+ tests), models, code gen |
| `/api/spc/` | spc_views.py | Control charts, capability, gage R&R |
| `/api/experimenter/` | experimenter_views.py | DOE design, power analysis, optimization |
| `/api/forecast/` | forecast_views.py | Time series forecasting |
| `/api/problems/` | problem_views.py | Legacy problem/hypothesis management |
| `/api/synara/` | synara_views.py | Belief engine (hypotheses, evidence, causal links, DSL) |
| `/api/agents/` | views.py | Agent dispatch (researcher, writer, editor, experimenter, EDA) |
| `/api/guide/` | guide_views.py | AI decision guide (rate-limited) |
| `/api/core/` | core/views.py | Projects, hypotheses, evidence, datasets, designs, org, graph |
| `/api/workbench/` | workbench/views.py | Unified workbench: projects, artifacts, graph, epistemic log |
| `/api/a3/` | a3_views.py | A3 reports CRUD + auto-populate |
| `/api/reports/` | report_views.py | Generic reports (CAPA, 8D) |
| `/api/rca/` | rca_views.py | Root cause analysis sessions + AI critique |
| `/api/fmea/` | fmea_views.py | FMEA CRUD + RPN scoring + evidence linking |
| `/api/vsm/` | vsm_views.py | Value stream mapping + simulation |
| `/api/hoshin/` | hoshin_views.py | Hoshin Kanri CI (enterprise only) |
| `/api/whiteboard/` | whiteboard_views.py | Collaborative boards, voting, SVG export |
| `/api/learn/` | learn_views.py | Courses, progress tracking, assessments |
| `/api/triage/` | triage_views.py | Data cleaning, preview, download |
| `/api/forge/` | forge/views.py | Synthetic data generation jobs |
| `/api/files/` | files/views.py | File upload, download, sharing, quotas |
| `/api/auth/` | api/views.py | Register, login, logout, profile, verification |
| `/api/internal/` | api/views.py | Staff-only: analytics, email, blog, A/B tests, automation |
| `/api/feedback/` | api/views.py | In-app feedback submission |
| `/billing/` | accounts/views.py | Stripe checkout, portal, webhooks |

## Integration Pattern

Existing pattern for connecting analysis results to projects (from experimenter_views.py):

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

## Frontend

- **Vanilla JavaScript** — no SPA framework (React/Vue/Angular)
- **Django templates** — 42 HTML files, all app pages extend `base_app.html`
- **Inline CSS/JS** — styles and scripts embedded in templates, no build tool
- **CDN libraries** — Chart.js, KaTeX, Marked.js, SmilesDrawer
- **Theme system** — 6 themes via CSS variables (dark, light, nordic, sandstone, midnight, contrast)
- **Static files** — WhiteNoise with Brotli/gzip compression and content-hash versioning

## Key Libraries

- **Django** — web framework, ORM
- **scipy, statsmodels** — statistical tests
- **sklearn** — ML models
- **plotly** — visualization (JSON format, client-side rendering)
- **Anthropic API** — Claude for LLM features (guide, synara, autopilot)
- **Qwen** — local LLM for code generation and analyst chat
- **WhiteNoise** — static file serving with compression
- **Stripe** — billing and subscriptions

## Serving

- **Gunicorn** behind **Cloudflare Tunnel**
- Static files served by Django + WhiteNoise (collectstatic)
- PostgreSQL database (production — on this machine)

## Working Conventions

- Python 3.10+
- Type hints encouraged
- All views use `@csrf_exempt`, `@gated` (feature gate), `@require_auth` decorators
- JSON in, JSON out (all API endpoints return JsonResponse)
- Plotly charts returned as JSON traces (client renders)
- All changes logged in `log.md`
- Debt tracked in `.kjerne/DEBT.md`, closed via `DEBT-001.md` process
- Lab standards in `STANDARD.md` (5S methodology, v2.0)
