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
- **All codebase changes MUST follow CHG-001.** No exceptions. See §Change Management below.
- All changes must be logged in `log.md` at the root.
- Technical debt is tracked in `.kjerne/DEBT.md`. Process for closing items is in `DEBT-001.md`.
- Unified workbench direction is documented in `services/svend/reference_docs/ARCHITECTURE.md`.

## WARNING: Change Request Enforcement (CHG-001 §7.1.1)

**DO NOT touch code without a ChangeRequest. This is not a guideline — it is enforced at three levels and there is no bypass.**

| Layer | Mechanism | What It Catches |
|-------|-----------|-----------------|
| **Model** | `ChangeRequest.clean()` | Rejects empty title (<10 chars), description (<20 chars), or author |
| **API** | `validate_for_transition(target_state)` | Blocks state transitions when required fields are missing |
| **Compliance** | `check_change_management()` — 14 checks, daily | Flags ALL gaps as FAIL or WARNING, reports field completeness |

**Before ANY code change:**
1. Create a ChangeRequest with `title` (>=10 chars), `description` (>=20 chars), `change_type`, `author`
2. Add `justification` and `affected_files` before submitting
3. Add `implementation_plan`, `testing_plan`, and `rollback_plan` (if applicable) before approval
4. Risk assessment REQUIRED for all code-touching types (feature, enhancement, bugfix, security, infrastructure, migration, debt)

**After implementation — BEFORE completing the CR (CHG-001 §5.4.2):**
1. Commit code → capture the SHA
2. Set `cr.commit_shas = [sha]` — this is the CR→git bridge
3. Set `cr.log_md_ref` to the log.md section reference
4. THEN transition CR to `completed`

**`validate_for_transition('completed')` will REJECT any code CR with empty `commit_shas`.** This is not a suggestion. The transition returns HTTP 400 and the CR stays in its current state until the field is populated.

Skipping this creates compliance failures that are permanently visible in the audit trail. There is no retroactive cleanup path that removes the gap — only honest markers acknowledging it.

## Architecture

```
~/kjerne/                              # Root — production server
├── CLAUDE.md                          # This file — READ THIS FIRST
├── STANDARD.md                        # 5S lab standards (v2.0)
├── log.md                             # Change log (all edits)
│
├── docs/                              # All documentation
│   ├── standards/                     # Kjerne Standards Library (25 standards)
│   │   ├── DOC-001.md                 # Documentation Structure
│   │   ├── XRF-001.md                 # Cross-Reference Syntax
│   │   ├── AUD-001.md                 # Audit Trail
│   │   ├── ERR-001.md                 # Error Handling
│   │   ├── LOG-001.md                 # Logging & Observability
│   │   ├── SEC-001.md                 # Security Architecture
│   │   ├── API-001.md                 # API Design
│   │   ├── DAT-001.md                 # Data Model
│   │   ├── CMP-001.md                 # Compliance Automation
│   │   ├── CHG-001.md                 # Change Management ← MANDATORY PROCESS
│   │   ├── SCH-001.md                 # Cognitive Scheduler
│   │   ├── OPS-001.md                 # Operations & Deployment
│   │   ├── BILL-001.md               # Billing & Subscription
│   │   ├── FE-001.md                  # Frontend Patterns
│   │   ├── TST-001.md                 # Testing Patterns
│   │   ├── LLM-001.md                # LLM Integration
│   │   ├── QMS-001.md                # Quality Management System
│   │   ├── QMS-002.md                # Resource Management
│   │   ├── MAP-001.md                # Architecture Map
│   │   ├── STY-001.md                # Code Style & Conventions
│   │   ├── DSW-001.md                # Decision Science Workbench
│   │   ├── ARCH-001.md               # Architecture & Structure
│   │   └── QUAL-001.md               # Output Quality Assurance
│   ├── compliance/                    # SOC 2 controls, policies, gap analysis
│   ├── planning/                      # Roadmaps, migration plans
│   └── reference/                     # Strategy docs, whitepapers
│
├── .kjerne/                           # Meta-tooling
│   ├── DEBT.md                        # Technical debt tracker
│   ├── DEBT-001.md                    # Debt closure process
│   ├── config.json                    # Lab config
│   ├── kjerne.py                      # Lab executable
│   └── snapshots/                     # Point-in-time snapshots
│
├── services/svend/                    # The Svend product
│   ├── reference_docs/                # Architecture docs
│   │   └── ARCHITECTURE.md            # Unified workbench vision
│   │
│   ├── agents/                        # DOE agent module (others moved to ~/agents_old/)
│   │   └── experimenter/              # Power analysis, DOE design, factorial/CCD/Latin square
│   │
│   ├── web/                           # Django web application
│   │   ├── manage.py
│   │   ├── ops/                       # Operations (scripts, systemd, configs)
│   │   ├── svend/                     # Django project settings
│   │   │   ├── settings.py
│   │   │   └── urls.py                # Root URL config
│   │   │
│   │   ├── syn/                       # Synara infrastructure layer
│   │   │   ├── core/                  # Base models (SynaraEntity, SynaraImmutableLog)
│   │   │   ├── log/                   # Logging (middleware, handlers, formatters)
│   │   │   ├── api/                   # API middleware (error envelope, idempotency)
│   │   │   ├── err/                   # Error hierarchy (SynaraError, retry, circuit breaker)
│   │   │   ├── sched/                 # Task scheduler (syn.sched — replaces Celery)
│   │   │   └── audit/                 # Audit & compliance subsystem
│   │   │       ├── models.py          # SysLogEntry, IntegrityViolation, DriftViolation,
│   │   │       │                      #   ComplianceCheck, ComplianceReport, CalibrationReport,
│   │   │       │                      #   ChangeRequest, ChangeLog, RiskAssessment, AgentVote
│   │   │       ├── compliance.py      # 28 automated compliance checks
│   │   │       ├── standards.py       # Standards parser (machine-readable hooks)
│   │   │       ├── utils.py           # generate_entry(), verify_chain_integrity()
│   │   │       ├── events.py          # Audit event catalog
│   │   │       ├── signals.py         # Django signal handlers
│   │   │       └── management/commands/run_compliance.py
│   │   │
│   │   ├── core/                      # Core Django app (TARGET models)
│   │   │   └── models/
│   │   │       ├── project.py         # Project, Dataset, ExperimentDesign
│   │   │       ├── hypothesis.py      # Hypothesis, Evidence, EvidenceLink
│   │   │       ├── tenant.py          # Tenant, Membership, OrgInvitation
│   │   │       └── graph.py           # KnowledgeGraph, Entity, Relationship
│   │   │
│   │   ├── agents_api/                # API views (main application logic)
│   │   │   ├── models.py              # Board, A3Report, Report, FMEA, FMEARow,
│   │   │   │                          #   RCASession, ValueStreamMap, HoshinProject,
│   │   │   │                          #   ActionItem, DSWResult, TriageResult, etc.
│   │   │   ├── dsw_views.py           # DSW statistical engine (200+ analyses)
│   │   │   ├── spc_views.py           # SPC endpoints (control charts, capability, gage R&R)
│   │   │   ├── spc.py                 # SPC engine
│   │   │   ├── experimenter_views.py  # DOE endpoints
│   │   │   ├── synara_views.py        # Synara belief engine API
│   │   │   ├── synara/                # Synara engine (orchestrator, belief, DSL, kernel)
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
│   │   │   └── views.py               # Agent dispatch
│   │   │
│   │   ├── api/                       # Content, automation, feedback, internal dashboard
│   │   │   ├── views.py               # Auth, chat, feedback, compliance (public)
│   │   │   ├── internal_views.py      # Staff dashboard, analytics, change management
│   │   │   └── urls.py
│   │   │
│   │   ├── chat/                      # LLM conversation system
│   │   ├── workbench/                 # Unified workbench (new platform)
│   │   ├── forge/                     # Synthetic data generation
│   │   ├── files/                     # File storage
│   │   ├── accounts/                  # Auth, billing, permissions
│   │   │   ├── models.py              # User (custom), Subscription, InviteCode
│   │   │   └── permissions.py         # @gated, @require_auth, @gated_paid, etc.
│   │   │
│   │   ├── svend_config/              # Environment configuration
│   │   ├── templates/                 # ~100 Django templates (HTML)
│   │   ├── static/                    # Source static files
│   │   └── ops/                       # Deployment scripts, systemd configs
│   │
│   └── site/                          # Landing page (svend.ai)
│       └── index.html
│
└── .gitignore
```

## Change Management (CHG-001) — MANDATORY

**Every codebase change MUST follow CHG-001. No exceptions. No shortcuts. This is not optional.**

Full standard: `docs/standards/CHG-001.md`
Compliance: SOC 2 CC8.1 (Change Management), CC3.4 (Risk Assessment), NIST SP 800-53 CM-3/CM-4

### The Rule

Before you write code, there must be a reason. That reason gets a `ChangeRequest`. The change gets logged at every step. When it's done, the chain of logs connects the reason to the result. If there's no chain, the change shouldn't have happened.

### Change Types

| Type | What | Risk Assessment | Approval |
|------|------|-----------------|----------|
| `feature` | New functionality | Multi-agent (4 roles) | Required |
| `enhancement` | Improve existing | Single agent | Required |
| `bugfix` | Fix defect | Single agent | Required |
| `hotfix` | Critical production fix | Expedited → retroactive 24h | Retroactive |
| `security` | Security patch | Security-focused | Required |
| `infrastructure` | Server, deploy, CI/CD | Operations-focused | Required |
| `migration` | Database schema | Multi-agent (4 roles) | Required |
| `documentation` | Docs, standards, README | None | None |
| `plan` | Architecture decisions | None — but logged | None |
| `debt` | Technical debt closure | Single agent | Required |

### Lifecycle

```
draft → submitted → risk_assessed → approved → in_progress → testing → completed
```

Every arrow creates an immutable `ChangeLog` entry. Every entry links to commits, issues, and related changes by UUID.

### Risk Assessment (Multi-Agent Voting)

For `feature`, `migration`, and `critical` risk changes, 4 agents vote from different perspectives:

- **security_analyst** — Auth, data exposure, SOC 2 impact (has veto power)
- **architect** — Architecture drift, coupling, scalability
- **operations** — Downtime risk, rollback plan, monitoring
- **quality** — Test coverage, regression risk, edge cases

Each scores 5 dimensions (Security, Availability, Integrity, Confidentiality, Privacy) on 1-5 scale. Overall score = max dimension. Score ≥4 requires staff review + mandatory rollback plan.

### Emergency Changes

Hotfixes can bypass normal approval. But:
- Retroactive risk assessment within **24 hours** (compliance check flags violation)
- Post-incident review within **48 hours**
- The `change_management` compliance check enforces this daily

### UUID Chain

Every finding, violation, change, and audit entry is UUID-linked bidirectionally:

```
ComplianceCheck.remediation_change_id  ←→  ChangeRequest.compliance_check_ids
DriftViolation.remediation_change_id   ←→  ChangeRequest.drift_violation_ids
SysLogEntry.correlation_id             ←→  ChangeRequest.correlation_id
ChangeRequest.parent_change_id         ←→  ChangeRequest (parent)
ChangeRequest.related_change_ids       ←→  [ChangeRequest UUIDs]
ChangeLog.details                      →   commit_sha, issue_url, log_md_ref
```

Start from any node. Traverse the entire chain.

### API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/internal/changes/` | GET | List changes (filter by type, status, risk) |
| `/api/internal/changes/create/` | POST | Create change request |
| `/api/internal/changes/<id>/` | GET | Full detail with logs, risk assessments, votes |
| `/api/internal/changes/<id>/transition/` | POST | State transition with log entry |

### Dashboard

`/internal/dashboard/` → Operations → Changes tab. Shows all change requests, filter by type/status/risk, click any row to see full detail panel with risk scores, agent votes, and log chain timeline.

### What Gets Flagged

The `change_management` compliance check (runs daily, SOC 2 CC8.1 + CC3.4) flags:
- Emergency changes without retroactive risk assessment (>24h)
- Changes stuck in `in_progress` >7 days without updates
- Completed changes missing log entries
- Features/migrations approved without risk assessment

## Planning System — MANDATORY

**This is the internal project management system. It replaces all ad-hoc planning.** Every feature, initiative, and task is tracked here with short IDs, dependencies, and status. Use it.

### Hierarchy

```
Initiative (INIT-xxx)  — strategic phase/theme
  └── Feature (FEAT-xxx)  — deliverable capability
        └── Task (TASK-xxx)  — implementation work item → ChangeRequest
```

Models: `api/models.py` (Initiative, Feature, PlanTask)
Management command: `syn/audit/management/commands/plan.py`
Dashboard: `/internal/dashboard/` → Product → Features tab

### Session Start Protocol

**At the start of every work session**, run this to get context:

```bash
python manage.py plan context          # Show active initiatives + actionable features
python manage.py plan show FEAT-xxx    # Deep dive on specific feature (if user provides one)
```

If the user says "work on FEAT-042" or gives any short ID, run `plan show` on it first.

### Commands

```bash
# Context
plan context                          # Active initiatives with ready/blocked features
plan progress                         # All initiatives with progress bars
plan show FEAT-042                    # Full context dump for a feature
plan show INIT-003                    # Initiative overview with all features

# Discovery
plan list --type feat --status in_progress  # List by type and status
plan search "CAPA"                    # Full-text search across everything
plan tree INIT-003                    # Visual hierarchy for an initiative
plan deps FEAT-042                    # Dependency graph (what blocks, what it blocks)
plan blocked                          # All blocked features

# Updates (Claude does this during/after work)
plan update FEAT-042 --status in_progress   # Update status
plan update TASK-117 --status completed     # Complete a task
plan activate INIT-003 INIT-006             # Set initiatives as active focus

# Notes
plan note FEAT-042 "SSE endpoint needs auth middleware review"     # Claude note
plan note FEAT-042 "Priority — need this before demo" --user      # User note ($ prefix)
```

### Notes Convention

- **Claude notes:** Timestamped, no prefix — technical context, decisions, blockers
- **User notes:** Prefixed with `$` — priorities, business context, instructions
- User notes displayed in gold on the dashboard to distinguish from Claude's

### Workflow

1. **User activates initiatives:** `plan activate INIT-003 INIT-006` (or tells Claude which to activate)
2. **Claude runs `plan context`** to see what's ready to work on
3. **Pick a feature** — prioritize by: user notes ($), priority, unblocked status
4. **Create tasks** under the feature if none exist
5. **Create ChangeRequest** per CHG-001 (link task → CR)
6. **Update status** as work progresses
7. **Mark completed** when done — triggers progress rollup to initiative

### Short IDs

- `INIT-001` through `INIT-007` — initiatives (phases)
- `FEAT-001` through `FEAT-060` — features (from master plan, more can be added)
- `TASK-001+` — tasks (created as features are worked)
- `legacy_id` field maps to original master plan IDs (E3-001, E4-002, etc.)
- All IDs are grepable: `grep -r "FEAT-042"` finds every reference

### Active Initiatives

Default view shows only features from **active** initiatives. To see everything, use the "All Initiatives" filter on the dashboard or `plan list --initiative __all__`.

## Standards Library

`docs/standards/` contains 25 machine-readable standards with assertion hooks that the compliance system parses and verifies automatically:

| Standard | Scope | Assertions |
|----------|-------|------------|
| DOC-001 | Documentation structure, hook vocabulary | 5 (Foundation) |
| XRF-001 | Cross-reference syntax | Foundation |
| AUD-001 | Hash-chained audit trail, immutability | 9 |
| ERR-001 | Error hierarchy, retry, circuit breaker | 11 |
| LOG-001 | Logging, correlation, structured output | 23 |
| SEC-001 | Auth, encryption, tenant isolation, CSP | 26 |
| API-001 | URL design, error envelope, idempotency | 25 |
| DAT-001 | Base models, UUID PKs, field patterns | 27 |
| CMP-001 | Compliance automation, standards parser | 14 |
| CHG-001 | Change management, risk assessment | 11 |
| SCH-001 | Cognitive scheduler, backpressure, temporal | 11 |
| OPS-001 | Deployment, backup, TLS, systemd, purge | 13 |
| BILL-001 | Billing tiers, Stripe, feature gating | 13 |
| FE-001 | Frontend patterns, themes, CSRF, CDN | 9 |
| TST-001 | Testing framework, fixtures, conventions | 8 |
| LLM-001 | LLM integration, rate limits, encryption | 11 |
| QMS-001 | FMEA, RCA, A3, VSM, Hoshin Kanri, X-Matrix | 20 |
| QMS-002 | Resource management | — |
| SLA-001 | Service level agreements, availability targets | — |
| MAP-001 | Architecture map, standards registry, drift detection | 5 |
| STY-001 | Code style, naming conventions, import ordering | 5 |
| DSW-001 | Decision Science Workbench architecture | — |
| ARCH-001 | Architecture & structure, layer boundaries | 6 |
| CACHE-001 | Caching patterns, HTTP cache control, CDN integrity | 6 |
| QUAL-001 | Output quality assurance, calibration, bounds checking | 12 |
| CAL-001 | Software calibration & verification, coverage ratchet, golden files | 12 |

Standards support `<!-- test: module.Class.method -->` hooks that link assertions to executable tests. The compliance system verifies test existence and can run them:

Run `python manage.py run_compliance --standards` to verify all assertions + test existence.
Run `python manage.py run_compliance --standards --run-tests` to execute linked tests.
Run `python manage.py run_compliance --all` to run all 28 infrastructure checks.

## Compliance Checks (28)

| Check | Category | Schedule | SOC 2 |
|-------|----------|----------|-------|
| audit_integrity | processing_integrity | Daily (critical) | CC7.2, CC7.3 |
| security_config | security | Daily (critical) | CC6.1, CC6.2 |
| access_logging | security | Daily (critical) | CC6.1, CC7.1 |
| standards_compliance | processing_integrity | Daily (critical) | CC4.1, CC9.1 |
| change_management | processing_integrity | Daily (critical) | CC8.1, CC3.4 |
| dependency_vuln | security | Mon/Fri | CC6.2 |
| ssl_tls | confidentiality | Mon/Fri | CC6.2 |
| encryption_status | confidentiality | Tuesday | CC6.1 |
| password_policy | security | Tuesday | CC6.1 |
| permission_coverage | security | Wednesday | CC6.1, CC6.2 |
| backup_freshness | availability | Wednesday | CC9.2 |
| data_retention | privacy | Thursday | CC7.2 |
| output_quality | processing_integrity | Wednesday | CC4.1, CC7.2 |
| calibration_coverage | processing_integrity | Wednesday | CC4.1, CC7.2 |
| complexity_governance | processing_integrity | Wednesday | CC4.1 |
| endpoint_coverage | processing_integrity | Wednesday | CC4.1, CC7.2 |

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
| `/api/internal/` | internal_views.py | Staff dashboard, analytics, compliance, change management |
| `/api/feedback/` | api/views.py | In-app feedback submission |
| `/billing/` | accounts/views.py | Stripe checkout, portal, webhooks |
| `/compliance/` | api/views.py | Public compliance page (trust signal) |
| `/internal/dashboard/` | internal_views.py | Staff-only operational dashboard |

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
- **Django templates** — ~100 HTML files, all app pages extend `base_app.html`
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
- **All changes follow CHG-001** — no code touches production without a ChangeRequest
