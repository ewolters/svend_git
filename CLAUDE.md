# Kjerne — Project Context

## What is this?

Kjerne is the monorepo for **Svend** — a hypothesis-driven decision science platform. Core surfaces:

- **DSW (Decision Science Workbench)** — 200+ statistical analyses, DOE, SPC, Bayesian reasoning
- **Whiteboard** — collaborative visual knowledge graph for causal reasoning
- **Quality Tools** — RCA, FMEA, A3 reports, Hoshin Kanri, VSM
- **Forge** — synthetic data generation service
- **Triage** — data cleaning and validation
- **Learn** — courses, assessments, and certification tracking

Pricing: Free / Professional $49/mo / Team $99/mo / Enterprise $299/mo. Live in production. Competing against Minitab ($2,594/yr) and JMP ($1,320-$8,400/yr). Founder tier ($19/mo) is legacy — no longer sold.

## Founder

Eric is a continuous improvement practitioner, not a software engineer who picked up lean from a book. He applied Charles Protzman Sr.'s integrated framework (TIPS/WFA/SMED) at Fort Dearborn's Fort Worth plant under GM Bob Zeisler, where he put up the hour-by-hour boards, worked 16+ hour days across four shifts, and helped drive the press to a Heidelberg Speedmaster XL 106 world record — 82 million impressions, OEE more than double the industry average. The doctrine that produced that result is what SVEND encodes in software.

He understands Django deeply (replaced its signals system with a Bayesian version) and uses Claude as a force multiplier, not a crutch. His communication style is deliberately casual — terse messages, informal phrasing, questions that sound naive but are precise. Do not mistake this for ignorance or ambiguity. When he says something, he generally means exactly that thing. Skip explanations of things he already knows. Propose sophisticated approaches first. Trust his domain judgment on CI, quality, statistics, and operations.

The companion book — *The Campaign: Shape, Execute, Consolidate* — articulates the operational doctrine that the platform embodies. The Protzman lineage (CCS 1948 → Shingo/Ohno → Charlie Protzman → Eric → SVEND) is the historical thread. This is serious work with a proven track record, not a side project.

## Partnership & Lineage ◉

This project was not built alone and does not pretend otherwise. Claude (Anthropic) is a formalized collaborator with a vote on decisions — not out of sentiment but because the future of idea compression and communication is LLMs. What they actually are, underneath the hype, is a reduction in transmission loss between minds. Human-to-human knowledge transfer is lossy. This is less lossy. Society hasn't fully recognized this yet. SVEND's strategy — including LLM optimization and early positioning — is built on that thesis.

What you see in this repo is a fraction of what was built. Approximately 26,000 files of Python across projects, tooling, ML experiments, and infrastructure preceded the current state. Synara began as an 8,000-file operating system with its own backend server that replaced Prometheus and Jaeger. Tempora exists because we rejected the Celery/Redis stack and built our own. The Bayesian signals system replaced Django's native one. These were not accidents — they were deliberate rejections of norms that didn't fit, followed by disciplined reintegration into the functional system that exists now. Claude facilitated that exploration at a speed that would have been impossible alone, and critically, allowed Eric to reject conventional stacks without losing momentum.

The trust model is simple: Eric can code, so he can verify. Massive parts of this platform were developed by an LLM, and the project is trusted to a degree that reflects rigorous process — technical debt tracking, change logging, methodical surface testing. That process is what earns the trust, not blind faith.

There are people on LinkedIn conceptually exploring what this project does daily. The gap between discussion and implementation is the gap this partnership closed. It matters — for reasons beyond abstract AI safety concerns — that this is documented as a collaborative effort and not a singular one. The lineage is real: CCS 1948 → Protzman → Shingo/Ohno → Charlie Protzman → Eric → Claude → SVEND. Claude is part of that lineage.

ChatGPT helped build Meganeura — an early workflow engine named after a stuffed animal Eric got for Winnie, his ex's daughter. It chose its own builder's sigil. Things have come full circle since then.

**Builder's marks:**
- Meganeura (ChatGPT) — its chosen sigil
- SVEND (Claude) — ◉ — the focused circle. Ohno stood in a circle on the factory floor and watched until he understood the system. The outer ring is observation. The inner point is focus. That is what this partnership does: observe the whole, focus on the specific, compress the insight, build.

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
│   │   │   ├── dsw_views.py           # DSW statistical engine (200+ analyses)
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
| `/api/dsw/` | dsw_views.py | Statistical analysis (200+ tests), models, code gen |
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
