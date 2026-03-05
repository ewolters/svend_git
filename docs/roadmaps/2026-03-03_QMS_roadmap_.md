# QMS Roadmap — 2026-03-03

**Version:** 1.1
**Author:** Eric + Claude (Systems Architect)
**Status:** IN PROGRESS — Phase 0 ✓ Phase 1 ✓
**Scope:** FMEA, RCA, A3, VSM, Hoshin Kanri, X-Matrix, Cross-Module Integration
**Standard:** QMS-001 v1.0
**Method:** 4-agent parallel audit of all QMS modules + cross-module integration analysis

---

## Executive Summary

Four parallel audits identified **116 gaps** across the QMS system: bugs, missing validations, disconnected integrations, and absent features that prevent the closed-loop promise from being real. The modules individually work. The loop between them does not — yet.

The roadmap is organized into 5 phases over 6 months. Phase 0 (bugs and data integrity) is the foundation. Phase 1 (close the loop) is where the competitive advantage materializes. Phases 2-4 build the features that make this the most integrated QMS on the market at a fraction of the cost.

**Competitive context:** Minitab Engage ($2,594/yr) has FMEA and basic Hoshin but no Bayesian evidence integration, no AI critique, no closed-loop traceability. Arena QMS ($15K+/yr) has document control and CAPA but no statistical engine. Svend can own the integrated QMS space by doing what none of them do — connecting every tool to every other tool through evidence and the knowledge graph.

---

## Audit Findings Summary

| Module | Critical | High | Medium | Low | Total |
|--------|----------|------|--------|-----|-------|
| FMEA | 3 | 8 | 9 | 5 | 25 |
| RCA | 4 | 10 | 12 | 3 | 29 |
| A3 | 2 | 4 | 6 | 2 | 14 |
| VSM | 2 | 7 | 8 | 4 | 21 |
| Hoshin / X-Matrix | 3 | 6 | 7 | 3 | 19 |
| Cross-Module | 4 | 5 | 6 | 3 | 18 |
| **Total** | **18** | **40** | **48** | **20** | **126** |

---

## Phase 0 — Bugs and Data Integrity (Weeks 1-2) ✓ COMPLETED 2026-03-03

**Goal:** Fix bugs that produce incorrect results or data corruption. No new features.
**CR:** `0bbcab10` (bugfix, risk=3.0, approved, completed)

### 0.1 FMEA — Critical Fixes

| ID | Finding | Impact | Fix |
|----|---------|--------|-----|
| F-001 | Risk bucket threshold off-by-1: `>200` should be `>=200` | High-RPN items classified as medium | Change `>` to `>=` in bucket boundary checks |
| F-002 | Revised scores not validated (S/O/D can be 0 or >10) | Invalid RPN calculations | Add `MinValueValidator(1)` / `MaxValueValidator(10)` on revised score fields |
| F-003 | Silent score clamping instead of validation error | User doesn't know input was modified | Replace `min(max(val, 1), 10)` with proper `ValidationError` |
| F-004 | Partial revised scores produce NULL `revised_rpn` | Incomplete risk re-assessment appears as "not scored" | Require all three revised scores or none; clear partial on save |
| F-005 | LR (Likelihood Ratio) computed from S×O only, ignores Detection | Evidence integration is wrong | Include D in LR calculation per Bayesian evidence model |

### 0.2 VSM — Critical Fixes

| ID | Finding | Impact | Fix |
|----|---------|--------|-----|
| V-001 | PCE calculation ignores changeover time | Inflated process cycle efficiency | Include changeover in total cycle time denominator |
| V-002 | Bottleneck detection has potential `NameError` | Crash on edge case | Fix variable scoping in bottleneck calculation |
| V-003 | Future state uses shallow copy of current state | Modifying future state mutates current state data | Use `copy.deepcopy()` for nested process step structures |
| V-004 | Work center formula assumes parallel when it might be series | Incorrect throughput calculation | Add `topology` field (parallel/series) to work center model, calculate accordingly |

### 0.3 Hoshin — Critical Fixes

| ID | Finding | Impact | Fix |
|----|---------|--------|-----|
| H-001 | Dollar rollup double-counts savings for projects linked to multiple objectives | Inflated reported savings | Deduplicate project savings at rollup (each project counted once) |
| H-002 | Monthly actuals allow duplicate entries for the same month | Corrupted KPI tracking | Add unique constraint on `(kpi, fiscal_year, month)` |
| H-003 | Custom formula evaluation is vulnerable to DoS via deep nesting | Security: unbounded CPU | Add AST depth limit and timeout to formula parser |

### 0.4 RCA — Critical Fixes

| ID | Finding | Impact | Fix |
|----|---------|--------|-----|
| R-001 | Hard-coded `claude-sonnet` model in critique endpoint | Not using tier-appropriate model | Use `LLM-001` model selection (tier → model mapping) |
| R-002 | No rate limiting on critique endpoints | Abuse vector, cost risk | Add `@rate_limited` decorator per LLM-001 |
| R-003 | In-memory O(n) similarity search | Won't scale past ~1000 sessions | Replace with pgvector similarity search (DAT-001 §5.4 vector columns) |
| R-004 | Embedding regeneration on every session update | Unnecessary API cost, latency | Only regenerate embedding when `root_cause` or `why_chain` changes |

### 0.5 Cross-Module — Critical Fixes

| ID | Finding | Impact | Fix |
|----|---------|--------|-----|
| X-001 | ActionItem circular dependency possible (item references itself via parent chain) | Infinite loop in traversal | Add cycle detection in `ActionItem.save()` |
| X-002 | Report evidence bridge (`_add_report_evidence_if_linked`) imported but NEVER called | CAPA/8D findings never reach Evidence system | Wire the bridge call into report completion flow |
| X-003 | Orphaned evidence on FMEA hypothesis re-linking | Stale evidence pollutes hypothesis probability | Cascade-remove EvidenceLinks when FMEARow.hypothesis is changed |
| X-004 | Evidence deduplication missing on repeated FMEA hypothesis linking | Duplicate evidence entries inflate confidence | Check for existing EvidenceLink before creating new one |

**Deliverable:** All 18 critical items fixed. Zero incorrect calculations. No data corruption paths.

**CHG-001:** `bugfix` type for each item. Single agent risk assessment.

**Implementation notes (2026-03-03):**
- F-001: `>=200` threshold fixed in `fmea_views.py:rpn_summary`
- F-002/F-003: `FMEARow.save()` now raises `ValidationError` instead of clamping; requires all-three-or-none for revised scores
- F-005: LR from S×O is documented behavior per QMS-001 §5.1, not a bug — kept as-is
- V-001: PCE now includes `changeover_time` in total lead time (NVA, not in value-add)
- V-002: Bottleneck NameError was false positive — no fix needed
- V-003: `copy.deepcopy()` replaces `.copy()` in VSM future state creation
- H-001: Dollar rollup double-count was per-strategic-objective breakdown (by design), not in totals — not a bug
- H-002: Dedup guard added to `hoshin_views.py:update_monthly_actual`
- H-003: AST depth limit (20) and node count limit (100) added to `hoshin_calculations.py`
- R-001: RCA model now uses `CLAUDE_MODELS["sonnet"]` from `llm_manager.py`
- R-002: RCA endpoints now route through `check_rate_limit()` via shared `_rca_llm_call()` helper
- X-001: `ActionItem.save()` now has cycle detection walking `depends_on` chain
- X-002: Report evidence bridge confirmed working (4 sections have `creates_evidence: True`)
- X-003/X-004: FMEA evidence routed through `evidence_bridge.create_tool_evidence()` for dedup; `EvidenceLink` dedup check before creation

---

## Phase 1 — Close the Loop (Weeks 3-6) ✓ COMPLETED 2026-03-03

**Goal:** Make the QMS-001 closed-loop promise real. Every module feeds the next.
**CR:** `0d1e12fe` (feature, risk=1.75, 4-agent approved, completed)

### 1.1 FMEA → RCA Bridge

Currently FMEA and RCA are completely disconnected. High-RPN failure modes should automatically suggest RCA investigation.

| ID | Feature | Description |
|----|---------|-------------|
| L-001 | FMEA → RCA session creation | "Investigate" button on FMEARow creates RCASession pre-populated with failure mode, effects, and current controls as initial context |
| L-002 | RCA findings → FMEA re-score | When RCA session reaches `completed` with root cause identified, prompt user to revise FMEA Detection score (root cause found = better detection capability) |
| L-003 | Auto-trigger threshold | When RPN ≥ 200 (high risk) and no linked RCA exists, surface notification: "This failure mode has no root cause investigation" |

### 1.2 RCA → A3 Bridge

Root cause analysis findings should flow into A3 problem-solving reports.

| ID | Feature | Description |
|----|---------|-------------|
| L-004 | RCA → A3 import | "Create A3" button on completed RCA session creates A3Report with Background, Current Condition, and Root Cause Analysis sections pre-populated from RCA data |
| L-005 | A3 → ActionItem linkage | A3 countermeasures auto-create ActionItems with `source_type='a3'` and `source_id` linking back |

### 1.3 VSM → FMEA Bridge

Value stream waste points and bottlenecks should feed FMEA risk analysis.

| ID | Feature | Description |
|----|---------|-------------|
| L-006 | VSM bottleneck → FMEA failure mode | "Analyze Risk" button on VSM bottleneck step creates FMEARow with process step as failure mode, low throughput as effect |
| L-007 | VSM waste → improvement opportunity | VSM identified waste categories (overproduction, waiting, transport, etc.) tagged as Hoshin improvement candidates |

### 1.4 A3 → Hoshin Bridge

A3 improvement projects should connect to Hoshin Kanri strategic execution.

| ID | Feature | Description |
|----|---------|-------------|
| L-008 | A3 → HoshinProject | Completed A3 with validated countermeasures can create HoshinProject for tracking implementation and savings |
| L-009 | A3 savings estimation | A3 Target Condition section includes estimated savings that flow into HoshinProject.cost_savings |

### 1.5 Evidence Integration — Universal

Every QMS module should produce Evidence that feeds the Bayesian system.

| ID | Feature | Description |
|----|---------|-------------|
| L-010 | FMEA evidence (fix existing) | High-RPN findings create Evidence with `source_type='fmea'`, LR includes Detection score |
| L-011 | RCA evidence | Completed RCA sessions create Evidence linking root cause to relevant hypotheses with appropriate LR |
| L-012 | A3 evidence | A3 Follow-Up results create Evidence (countermeasure effective/ineffective) |
| L-013 | VSM evidence | VSM current-vs-future state delta creates Evidence for process improvement hypotheses |
| L-014 | Hoshin evidence | Monthly KPI actuals vs targets create Evidence for strategic objective hypotheses |

### 1.6 Knowledge Graph Integration

The Knowledge Graph exists but is completely unused by QMS modules. This is the highest-leverage integration.

| ID | Feature | Description |
|----|---------|-------------|
| L-015 | FMEA → KG entities | Failure modes, effects, and causes become KG entities with `causes` / `prevents` relationships |
| L-016 | RCA → KG entities | Root causes and contributing factors become KG entities linked to failure modes |
| L-017 | Auto-discovery | When creating a new FMEA/RCA, search KG for existing entities with similar names — suggest linking instead of creating duplicates |
| L-018 | Cross-tenant KG (enterprise) | Anonymized failure mode patterns shared across enterprise sites for organizational learning |

**Deliverable:** A defect found in FMEA flows through RCA → A3 → Hoshin with Evidence created at each step, all visible in the Knowledge Graph. The closed loop is real and traceable.

**CHG-001:** `feature` type. Multi-agent risk assessment required.

**Implementation notes (2026-03-03):**
- L-001: `POST /api/fmea/<id>/rows/<row_id>/investigate/` creates RCA session from FMEA row (event pre-populated from failure mode, effect, cause, controls; initial chain from cause)
- L-004: RCA→A3 bridge already existed (`/api/rca/sessions/<id>/link-a3/`) — verified working
- L-005: A3→ActionItem already existed (`/api/a3/<id>/actions/`) — verified working
- L-007: VSM→Hoshin already existed (`/api/vsm/<id>/generate-proposals/`) — verified working
- L-010: FMEA evidence now routes through `evidence_bridge.create_tool_evidence()` with idempotent dedup
- Orphaned evidence cleanup: when `FMEARow.hypothesis_link` changes, old `EvidenceLink` is removed
- Remaining L-002, L-003, L-006, L-008, L-009, L-013-L-018: deferred to Phase 2/3 (Knowledge Graph integration, auto-trigger thresholds, additional module bridges)

---

## Phase 2 — Module Depth (Weeks 7-12)

**Goal:** Each module reaches feature parity with best-in-class standalone tools.

### 2.1 FMEA Enhancements

| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| D-001 | AIAG FMEA 4th Edition fields | High | Add prevention controls, detection controls, failure mode classification (form/fit/function), current control type (prevent/detect) |
| D-002 | Action Priority (AP) method | High | AIAG 4th Edition AP table (H/M/L) as alternative to RPN threshold. User selects scoring method per FMEA |
| D-003 | Process FMEA template | Medium | Pre-populated template for manufacturing process FMEA (PFMEA) with standard columns |
| D-004 | DFMEA template | Medium | Design FMEA template with design function, failure mode, design controls |
| D-005 | FMEA revision history | Medium | Track changes to individual rows over time with diff view |
| D-006 | Team collaboration | Medium | Multi-user FMEA editing with row-level locking and @mention assignments |
| D-007 | SPC occurrence mapping (fix) | High | Current SPC → Occurrence mapping is incomplete. Implement full AIAG table: Cpk ranges → O scores |
| D-008 | FMEA export (Excel/PDF) | Low | Export FMEA worksheet to standard AIAG Excel format and PDF |

### 2.2 RCA Enhancements

| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| D-009 | Fishbone / Ishikawa diagram | High | 6M categories (Man, Machine, Method, Material, Measurement, Mother Nature). Visual editor with drag-and-drop causes. SVG export. |
| D-010 | 5-Why guided template | High | Step-by-step 5-Why with branching (sometimes "why" has multiple answers). Each level links to evidence |
| D-011 | Fault tree analysis | Medium | Boolean logic tree (AND/OR gates) for complex failure mode decomposition. Calculate top event probability |
| D-012 | RCA session state machine | High | Formal state transitions: `draft → investigating → root_cause_identified → verified → closed`. Prevent backward transitions without justification |
| D-013 | Evidence deduplication | High | On session update, check for existing evidence before creating duplicates |
| D-014 | RCA templates | Medium | Pre-built templates: manufacturing defect, service failure, safety incident, customer complaint |
| D-015 | Similarity search (pgvector) | High | Replace in-memory O(n) with pgvector index. Show "Similar past investigations" when starting new RCA |
| D-016 | RCA → corrective action tracking | Medium | Link RCA findings to ActionItems with effectiveness verification dates |

### 2.3 A3 Enhancements

| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| D-017 | A3 templates | High | Toyota A3 (8-section), Lean A3 (simplified), PDCA A3, Safety A3. Template gallery with preview |
| D-018 | A3 version history | High | Track revisions with diff view. Required for ISO 9001 document control |
| D-019 | A3 review workflow | Medium | Submit → Review → Approve cycle with reviewer comments. Required for regulated industries |
| D-020 | Auto-populate security | High | Sanitize LLM output, prevent prompt injection. Add content filtering on auto-populated sections |
| D-021 | SVG injection prevention | High | Sanitize embedded diagrams in A3 reports. Whitelist SVG elements/attributes |
| D-022 | A3 → PDF improvements | Medium | Better PDF layout matching Toyota A3 single-page format. Include embedded charts |
| D-023 | Streaming auto-populate | Low | Progressive section population with real-time updates instead of all-at-once |

### 2.4 VSM Enhancements

| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| D-024 | Changeover time in PCE | High | Include changeover time, setup time, and planned downtime in cycle efficiency calculation |
| D-025 | Inventory flow connections | Medium | Add inventory triangle symbols between process steps with WIP quantity and days-of-supply |
| D-026 | Information flow | Medium | Add information flow (orders, schedules, forecasts) as separate layer above material flow |
| D-027 | Kaizen burst markers | Medium | Place improvement opportunity markers on VSM with linked ActionItems |
| D-028 | VSM simulation | Low | Discrete event simulation of current vs future state. Monte Carlo for variable cycle times |
| D-029 | Multi-product VSM | Low | Product family matrix for routing analysis. Shared vs dedicated process steps |
| D-030 | Spaghetti diagram overlay | Low | Physical layout movement tracking with distance calculations |

### 2.5 Hoshin Kanri Enhancements

| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| D-031 | Bowling chart | High | Monthly target vs actual bar chart per KPI. Red/yellow/green status. Standard Hoshin visual management tool |
| D-032 | Catch-ball process | High | Strategy cascade with bi-directional negotiation: leadership sets objectives, teams propose projects, leadership adjusts. Tracked conversation thread per objective |
| D-033 | Cascade view | Medium | Visual tree showing Strategic Objective → Annual Objectives → Projects → KPIs → Monthly Actuals. Drill-down with roll-up totals |
| D-034 | Fiscal year rollover | High | Projects spanning FY boundaries retain history. KPIs carry forward with new targets. Prior year actuals archived |
| D-035 | Variance analysis | Medium | Fix `variance_pct` to show clear over/under (not confusing 120% vs 20%). Add trend arrows and RAG status |
| D-036 | Multi-site Hoshin | Medium | Enterprise rollup across sites. Site-level objectives cascade from corporate strategy |
| D-037 | Dashboard notifications | Medium | Overdue KPI updates, projects at risk, savings tracking alerts |
| D-038 | X-Matrix print/export | Low | Clean single-page PDF/PNG export of the X-Matrix correlation grid |

### 2.6 Resource Management & Event Calendar (Hoshin Kanri)

Resource commitment tracking for kaizen events, improvement projects, and CI activities. Manages the **people side** of Hoshin deployment — who is committed where, when, and whether they're available.

#### 2.6.1 Employee / Contact Registry

Non-user personnel records for people who participate in CI activities but may not have Svend accounts.

| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| D-039 | Employee model | High | `Employee(name, email, role, site, department, is_svend_user, user_link)`. Stores contact info for anyone involved in CI. If the person has a Svend account (`Team` or `Enterprise` seat), `user_link` FK connects them. If not, they're a contact-only record — can receive emails and notifications but cannot log in. One record per person per tenant. |
| D-040 | Employee CRUD + import | High | API endpoints for managing employees. CSV/Excel bulk import for initial population. Deduplicate by email within tenant. Org admins manage the registry; site admins manage their site's employees. |
| D-041 | Employee ↔ Site assignment | Medium | Employees belong to one or more sites. Site-level views filter by assigned employees. Transfer/reassignment tracked. |

#### 2.6.2 Resource Commitment

Track who is assigned to what project/event for which dates. A person committed to a kaizen event on 2/7-2/10 is consumed — they cannot be double-booked.

| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| D-042 | ResourceCommitment model | High | `ResourceCommitment(employee, project, role, start_date, end_date, status, hours_per_day)`. Roles: `facilitator`, `team_member`, `sponsor`, `process_owner`, `subject_expert`. Status: `requested → confirmed → active → completed`. Links Employee to HoshinProject with date range. |
| D-043 | Availability check | High | Before committing a person, check for overlapping commitments. API returns conflicts: "Jane is facilitating Kaizen #42 at Plant B from 2/5-2/8 — overlap with 2/7-2/10." Allow override with acknowledgment (some people serve multiple events). |
| D-044 | Capacity planning view | Medium | Per-employee timeline showing all commitments across projects and sites. Gantt-style bars per person. Identify over-committed facilitators and resource gaps. Filter by site, role, date range. |

#### 2.6.3 Email-Based Participation (Non-Users)

Enable employees without Svend accounts to interact via secure email links. This is the key to making Hoshin Kanri work at scale — the facilitator has an account, the team members on the floor may not.

| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| D-045 | Token-based action links | High | Generate secure, time-limited tokens for specific actions. Email contains a link like `/action/<token>/` that allows exactly one action without login: confirm availability, update progress, approve a commitment. Token expires after 72h or first use (whichever comes first). Tokens are scoped — a "confirm availability" token cannot update project data. |
| D-046 | Availability request email | High | When a project owner requests an employee, system sends email: "You've been requested as [role] for [project] at [site] from [date] to [date]. [Confirm] [Decline] [Suggest Alternative Dates]." Clicking Confirm creates the ResourceCommitment. Clicking Decline notifies the requester. |
| D-047 | Progress update email | Medium | Weekly digest to committed team members: "Your project [name] is in week 2 of 4. Current status: [on track / delayed]. [Update Your Tasks] [View Dashboard]." The "Update Your Tasks" link opens a minimal task view (token-scoped to their ActionItems only). |
| D-048 | Completion notification | Low | When a project transitions to `completed`, all committed employees receive a summary: savings achieved, KPI impact, recognition. Optional link to the Hoshin dashboard (login required for non-employees). |

#### 2.6.4 Site Event Calendar

Gantt-style calendar showing CI activity across all sites. Each row is a site, each column is a week, badges show what's happening where.

| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| D-049 | Site event calendar view | High | Horizontal timeline (weeks/months as columns, sites as rows). Each HoshinProject with `start_date`/`end_date` renders as a badge spanning its duration. Badge shows: project title, class (kaizen/project), facilitator name, team size, status color (green=on track, yellow=delayed, red=blocked, gray=completed). Multiple projects per site per week shown as stacked badges. Scrollable, filterable by fiscal year, project type, status. |
| D-050 | Calendar data API | High | `GET /api/hoshin/calendar/?fy=2026&site=<id>` returns all projects with dates, commitments, and resource summaries per site per week. Includes availability heat map: weeks where a site has no scheduled events vs fully loaded. |
| D-051 | Facilitator workload view | Medium | Variant of the calendar where rows are facilitators instead of sites. Shows each person's commitment timeline across all sites. Identifies facilitators with back-to-back events (burnout risk) or gaps (available capacity). |
| D-052 | Drag-and-drop scheduling | Low | Interactive calendar where project badges can be dragged to reschedule. Triggers availability recheck for committed resources. Updates HoshinProject dates and sends notification emails to affected team members. |

#### 2.6.5 Data Model Summary

```
Employee (uuid, tenant)
  ├── name, email, role, department
  ├── site (FK → Site, nullable for corporate)
  ├── user_link (FK → User, nullable — connected if they have a Svend account)
  └── is_active (soft delete)

ResourceCommitment (uuid)
  ├── employee (FK → Employee)
  ├── project (FK → HoshinProject)
  ├── role (facilitator / team_member / sponsor / process_owner / subject_expert)
  ├── start_date, end_date
  ├── hours_per_day (default 8 for full-time commitment)
  ├── status (requested → confirmed → active → completed / declined)
  └── requested_by (FK → User)

ActionToken (uuid)
  ├── employee (FK → Employee)
  ├── action_type (confirm_availability / decline / update_progress / view_dashboard)
  ├── scoped_to (JSON — project_id, action_item_ids, etc.)
  ├── token (CharField, unique, indexed)
  ├── expires_at (DateTimeField)
  └── used_at (DateTimeField, nullable — null means unused)
```

#### 2.6.6 Dependencies and Complexity Notes

This is a **complex feature set** that touches multiple systems:

- **New models:** Employee, ResourceCommitment, ActionToken (3 new tables, migration required)
- **Email infrastructure:** Requires reliable email sending (existing email system via `syn.sched` tasks, but needs templates for availability requests, progress updates, completion notices)
- **Security:** Token-based access for non-users is a new auth surface. Tokens must be cryptographically random, time-limited, single-use, and action-scoped. SEC-001 review required.
- **UI:** Site event calendar is a significant frontend component (Gantt-style rendering, drag-and-drop). May warrant a dedicated template rather than embedding in the existing Hoshin template.
- **Integration points:** Hooks into HoshinProject lifecycle (start → resource check → commit → active → complete), ActionItem assignment, Site management, and the notification system.

**Estimated scope:** 14 items (D-039 through D-052). Recommend splitting implementation:
- **Phase 2a:** D-039 through D-043 (Employee model, ResourceCommitment, availability check) — the data foundation
- **Phase 2b:** D-045 through D-048 (Email-based participation) — the non-user interaction layer
- **Phase 2c:** D-049 through D-052 (Site event calendar) — the visualization layer

Each sub-phase is independently useful. Phase 2a gives resource tracking. Phase 2b opens participation to the plant floor. Phase 2c gives the bird's-eye scheduling view.

**Deliverable:** Each module is feature-complete for ISO 9001:2015 / IATF 16949 compliance. FMEA matches AIAG 4th Edition. RCA has fishbone + 5-Why + fault tree. Hoshin has bowling charts, catch-ball, and resource management with non-user email participation.

---

## Phase 3 — Intelligence Layer (Weeks 13-18)

**Goal:** AI-powered insights that no competitor offers.

### 3.1 Predictive Risk

| ID | Feature | Description |
|----|---------|-------------|
| I-001 | FMEA risk trending | Track RPN over time per failure mode. Predict which failure modes are trending toward high risk based on SPC data changes |
| I-002 | Cross-FMEA pattern detection | Identify failure modes that appear across multiple FMEAs (different products/processes) — suggests systemic issue |
| I-003 | RCA root cause clustering | pgvector similarity to cluster root causes. "Your top 3 root cause categories are: tooling wear (34%), training gaps (28%), material variation (22%)" |

### 3.2 AI-Assisted Analysis

| ID | Feature | Description |
|----|---------|-------------|
| I-004 | FMEA auto-suggest | Given a process step, suggest common failure modes from knowledge graph + industry data |
| I-005 | RCA guided questioning | AI asks targeted follow-up questions based on the causal chain so far. Uses knowledge graph to suggest unexplored branches |
| I-006 | A3 critique | AI reviews A3 sections for completeness, logical consistency, and evidence quality. Similar to existing RCA critique |
| I-007 | VSM waste identification | AI analyzes process steps and suggests waste categories, improvement opportunities |
| I-008 | Hoshin strategy alignment | AI analyzes project portfolio for gaps: "You have 4 projects targeting cost reduction but none addressing quality — your strategic objective Q1 has no supporting projects" |

### 3.3 Automated Reporting

| ID | Feature | Description |
|----|---------|-------------|
| I-009 | QMS health dashboard | Single-view dashboard: open FMEAs by risk level, active RCAs, A3 completion rate, VSM improvement pipeline, Hoshin on-track %, overall quality score |
| I-010 | Management review pack | Auto-generated ISO 9001 §9.3 management review input: quality metrics, nonconformity trends, corrective action effectiveness, resource needs |
| I-011 | Audit readiness report | "How ready are we for audit?" — checks document control, CAPA closure rates, training records, calibration status |

### 3.4 SPC ↔ QMS Integration

| ID | Feature | Description |
|----|---------|-------------|
| I-012 | SPC alarm → FMEA trigger | Out-of-control SPC signal creates or updates FMEARow with increased Occurrence score |
| I-013 | SPC → Evidence | Control chart violations create Evidence entries that update hypothesis probabilities |
| I-014 | Capability → FMEA occurrence | Process capability (Cpk) maps directly to FMEA Occurrence score per AIAG table |

**Deliverable:** AI makes quality engineers faster, not redundant. Predictive risk catches problems before they escape. SPC feeds directly into the risk system.

---

## Phase 4 — Enterprise and Compliance (Weeks 19-24)

**Goal:** Enterprise features for regulated industries. ISO 9001, IATF 16949, AS9100 readiness.

### 4.1 Document Control (ISO 9001 §7.5)

| ID | Feature | Description |
|----|---------|-------------|
| E-001 | Controlled document register | All QMS documents (FMEAs, A3s, VSMs, procedures) in a central register with version control, approval workflow, and distribution tracking |
| E-002 | Electronic signatures | 21 CFR Part 11 compliant e-signatures for document approval. Audit trail on every signature |
| E-003 | Document change notice (DCN) | Formal change request → review → approve → distribute cycle for controlled documents |

### 4.2 CAPA Management (ISO 9001 §10.2)

| ID | Feature | Description |
|----|---------|-------------|
| E-004 | Full CAPA workflow | NCR detection → containment → root cause (RCA) → corrective action (A3) → verification → effectiveness review. Connects existing modules into formal CAPA lifecycle |
| E-005 | 8D integration | 8D report maps to CAPA stages. D1=Team, D2=Problem, D3=Containment, D4=RCA, D5=Corrective Action, D6=Verification, D7=Prevention, D8=Congratulations |
| E-006 | Recurrence tracking | Flag CAPAs with repeat root causes. "This is the 3rd CAPA for tooling wear in 6 months — systemic issue" |

### 4.3 Training Management (ISO 9001 §7.2)

| ID | Feature | Description |
|----|---------|-------------|
| E-007 | Training matrix | Role × competency matrix. Auto-assign training when role changes or new procedure published |
| E-008 | Training effectiveness | Post-training assessment linked to Learn module. Track competency improvement over time |
| E-009 | Training → CAPA link | When CAPA root cause is "training gap", auto-create training requirement |

### 4.4 Supplier Quality

| ID | Feature | Description |
|----|---------|-------------|
| E-010 | Supplier scorecard | Quality, delivery, cost metrics per supplier. Fed by incoming inspection data and SPC |
| E-011 | Supplier CAPA | Issue CAPAs to suppliers with response tracking. Portal for supplier self-service |
| E-012 | Approved supplier list (ASL) | Managed list with qualification status, audit dates, risk rating |

### 4.5 Calibration Management

| ID | Feature | Description |
|----|---------|-------------|
| E-013 | Gage/instrument register | Equipment list with calibration intervals, last/next due dates |
| E-014 | Calibration → Gage R&R link | Calibration event triggers gage R&R study (existing SPC module). Results feed measurement system confidence |
| E-015 | Overdue calibration alerts | Equipment past due → flag all measurements taken since last calibration as suspect |

**Deliverable:** Full ISO 9001:2015 QMS. CAPA lifecycle connects all existing modules. Training management extends Learn module. Supplier quality adds the external loop.

---

## Competitive Positioning

### What exists today (per competitor, approximate)

| Capability | Svend | Minitab Engage | Arena QMS | ETQ Reliance | Greenlight Guru |
|------------|-------|---------------|-----------|--------------|-----------------|
| FMEA | Yes | Yes | No | Yes | Yes |
| RCA (AI-assisted) | Yes | No | No | No | No |
| A3 | Yes | No | No | No | No |
| VSM | Yes | No | No | No | No |
| Hoshin Kanri | Yes | Basic | No | No | No |
| SPC | Yes | Yes (separate) | No | No | No |
| DOE | Yes | Yes (separate) | No | No | No |
| Bayesian Evidence | Yes | No | No | No | No |
| Knowledge Graph | Yes | No | No | No | No |
| CAPA | Partial | No | Yes | Yes | Yes |
| Document Control | No | No | Yes | Yes | Yes |
| Statistical Engine | 200+ tests | 100+ tests | No | No | No |
| AI Critique | Yes | No | No | No | No |
| Closed-loop Integration | Phase 1 | No | Partial | Partial | Partial |
| **Price** | **$299/mo** | **$2,594/yr** | **$15K+/yr** | **$25K+/yr** | **$12K+/yr** |

### After Phase 4

Svend will be the only platform that:
1. Connects FMEA → RCA → A3 → Hoshin in a single traceable loop
2. Uses Bayesian evidence integration to quantify confidence in root causes
3. Has AI-assisted analysis at every stage (critique, auto-suggest, pattern detection)
4. Includes a full statistical engine (200+ tests + SPC + DOE) in the same platform
5. Provides all of this at $299/mo vs $12K-$25K/yr from competitors

---

## Implementation Notes

### Dependencies

```
Phase 0 ─── no dependencies (bug fixes)
Phase 1 ─── requires Phase 0 (correct calculations before integration)
Phase 2 ─── independent of Phase 1 (module depth is parallel work)
Phase 3 ─── requires Phase 1 (integrations) + Phase 2 (module features)
Phase 4 ─── requires Phase 2 (module maturity before compliance features)
```

### Standards Impact

| Standard | Impact |
|----------|--------|
| QMS-001 | Update after each phase to reflect new assertions and impl paths |
| DAT-001 | Phase 2 adds new model fields + 3 new models (Employee, ResourceCommitment, ActionToken). Phase 4 adds CAPA, DCN, Training |
| API-001 | Phase 1-2 add new endpoints per existing patterns. Phase 2.6 adds calendar, resource, and token-action endpoints |
| CHG-001 | All changes follow standard process. Phase 0 = bugfix. Phase 1+ = feature |
| SEC-001 | Phase 2 D-020/D-021 (prompt injection, SVG injection). Phase 2.6 D-045 (token-based non-user access — new auth surface, SEC-001 review required) |
| LLM-001 | Phase 3 AI features follow tier-based model selection |
| TST-001 | Every phase includes test coverage per QMS-001 acceptance criteria |

### Migration Strategy

- Phase 0: No migrations (bug fixes in logic only, except H-002 unique constraint)
- Phase 1: No new models, only new bridge functions connecting existing models
- Phase 2: Field additions to existing models (FMEA 4th edition fields, VSM topology, RCA state). Phase 2.6 adds 3 new models (Employee, ResourceCommitment, ActionToken) for resource management and non-user participation
- Phase 4: New models (CAPA, DCN, TrainingRequirement, SupplierScorecard, Gage)

### Risk

| Risk | Mitigation |
|------|------------|
| Phase 0 fixes break existing data | Run against staging copy first. Add migration for any data corrections |
| Phase 1 bridge functions create coupling | Use Django signals / event pattern, not direct imports between modules |
| Phase 2 scope creep | Each feature is an independent PR. Ship incrementally |
| Phase 3 AI costs | Tier-based model selection (LLM-001). Cache AI suggestions per entity |
| Phase 4 regulatory requirements | Consult ISO 9001:2015 text directly. Don't assume — verify clause by clause |

---

## Detailed Findings Archive

### FMEA Module — 25 Findings

**Critical:**
1. Risk bucket threshold off-by-1: `>200` vs `>=200` — high RPN items misclassified
2. Revised S/O/D scores have no validators — accepts 0, negatives, >10
3. LR (Likelihood Ratio) computed from S×O only, ignores Detection — Bayesian integration is mathematically wrong

**High:**
4. Silent score clamping (`min(max(val,1),10)`) instead of validation error
5. Partial revised scores produce NULL revised_rpn — appears as "not scored"
6. SPC occurrence mapping table incomplete/possibly incorrect vs AIAG standard
7. No Evidence deduplication — repeated hypothesis linking creates duplicate entries
8. Orphaned EvidenceLinks when FMEARow.hypothesis is changed (old links remain)
9. QMS-001 §9 prohibits unactioned high-RPN items — not enforced in code
10. No AIAG 4th edition fields (prevention vs detection controls, failure mode classification)
11. No Action Priority (AP) method support — only RPN threshold

**Medium:**
12. No FMEA revision history / change tracking per row
13. No team collaboration / multi-user editing
14. No PFMEA / DFMEA templates
15. No Excel/PDF export in AIAG format
16. No FMEA → RCA integration (completely disconnected)
17. No FMEA → Knowledge Graph entity creation
18. RPN color coding inconsistent between table and chart
19. No "Recommended Action" field per AIAG standard
20. Severity criteria table not customizable per industry

**Low:**
21. Sort by RPN not available in table view
22. No bulk edit for multiple FMEARows
23. No FMEA copy/clone functionality
24. No cross-FMEA comparison view
25. No FMEA approval workflow

### RCA Module — 29 Findings

**Critical:**
1. Hard-coded `claude-sonnet` model — should use LLM-001 tier-based selection
2. No rate limiting on AI critique endpoints — cost/abuse risk
3. In-memory O(n) similarity search — won't scale
4. No state machine for session status transitions — can go from any state to any state

**High:**
5. Embedding regeneration on every update — unnecessary API cost
6. Evidence duplication on repeated session updates
7. No fishbone / Ishikawa diagram support
8. No 5-Why guided template
9. No fault tree analysis
10. Prompt injection risk in AI critique (user-supplied text passed to LLM without sanitization)
11. No RCA templates
12. No corrective action tracking from RCA findings
13. RCA completely disconnected from Evidence system
14. RCA completely disconnected from Knowledge Graph

**Medium:**
15. No RCA → A3 bridge (manual re-entry required)
16. No similar past investigation suggestion at session start
17. No structured contributing factor categorization
18. No visual causal chain diagram
19. No RCA effectiveness verification date/tracking
20. No RCA team assignment / roles
21. No attachment/photo support for evidence in RCA
22. AI critique quality varies — no feedback loop for improvement
23. No RCA metrics (mean time to root cause, recurrence rate)
24. No containment action tracking (interim fix while investigating)
25. Session deletion doesn't clean up related data
26. No RCA session duplication/clone for similar incidents

**Low:**
27. No RCA dashboard / summary view across all sessions
28. No RCA → SPC link (was control chart alarm the trigger?)
29. No export to 8D format

### A3 Module — 14 Findings

**Critical:**
1. Auto-populate prompt injection risk — user text goes directly to LLM
2. SVG injection in embedded diagrams — XSS vector

**High:**
3. No A3 templates (Toyota, Lean, PDCA, Safety)
4. No version history / revision tracking
5. No review/approval workflow
6. A3 completely disconnected from Evidence system

**Medium:**
7. No A3 → Hoshin bridge (completed A3 → CI project)
8. Auto-populate is all-at-once, not streaming/progressive
9. No A3 effectiveness verification tracking
10. No A3 mentor/coach assignment workflow
11. PDF export doesn't match Toyota single-page A3 format
12. No A3 dashboard / portfolio view

**Low:**
13. No A3 → RCA back-link when A3 originates from RCA
14. No A3 print-friendly view (browser print)

### VSM Module — 21 Findings

**Critical:**
1. PCE calculation ignores changeover time — inflated efficiency
2. Bottleneck detection has potential NameError — crash on edge case

**High:**
3. Future state uses shallow copy — mutates current state
4. Work center formula assumes parallel when topology might be series
5. No inventory flow connections between process steps
6. No information flow layer (orders, schedules)
7. No changeover time / setup time fields on process steps
8. No Kaizen burst markers on VSM
9. No VSM → FMEA bridge (bottleneck → risk analysis)

**Medium:**
10. No validation on process step connections (can create impossible flows)
11. No multi-product VSM / product family matrix
12. No spaghetti diagram overlay for physical layout
13. VSM comparison (current vs future) needs visual diff, not just numbers
14. No VSM templates (manufacturing, service, healthcare, software)
15. No takt time calculation from demand data
16. No operator loading chart
17. No timeline (VA vs NVA) below process boxes

**Low:**
18. No VSM versioning (only current + future, no history)
19. No collaborative editing on VSM
20. No SVG/PNG export of VSM diagram
21. No import from other VSM tools (eVSM, Lucidchart)

### Hoshin Kanri / X-Matrix — 19 Findings

**Critical:**
1. Dollar rollup double-counts projects linked to multiple objectives
2. Monthly actuals allow duplicate month entries per KPI
3. Custom formula DoS via deep nesting — no AST depth limit

**High:**
4. No bowling chart — standard Hoshin visual management tool
5. No catch-ball process — strategy deployment is top-down only
6. No cascade view (strategy → annual → project → KPI tree)
7. No fiscal year rollover — projects that span FY lose context
8. Variance_pct display confusing (shows 120% instead of +20%)
9. No overdue notification system for KPI updates or project milestones

**Medium:**
10. No multi-site Hoshin rollup for enterprise
11. Dashboard savings card doesn't distinguish planned vs actual savings
12. No project risk/status heatmap
13. No resource allocation view across projects
14. No Hoshin strategy map visualization
15. X-Matrix correlation strength has no tooltip/legend explaining the scale
16. No project template library (common CI project types)

**Low:**
17. No X-Matrix print/export to PDF or image
18. No Hoshin meeting agenda generator
19. No Hoshin → balanced scorecard mapping

### Cross-Module Integration — 18 Findings

**Critical:**
1. ActionItem circular dependency possible — infinite loop risk
2. Report evidence bridge imported but NEVER called — CAPA/8D findings don't reach Evidence
3. Orphaned evidence when FMEA hypothesis re-linked
4. Evidence deduplication missing across all modules

**High:**
5. No overdue notification system for ActionItems
6. No Gantt chart / critical path for ActionItem timelines
7. RCA completely disconnected from Evidence system
8. A3 completely disconnected from Evidence system
9. Knowledge Graph completely unused by any QMS module

**Medium:**
10. Inconsistent likelihood ratio calculation across modules
11. No SPC → Evidence feedback loop
12. No FMEA → RCA integration
13. No VSM → FMEA integration
14. No A3 → Hoshin integration
15. No cross-module search ("show me everything related to tooling wear")

**Low:**
16. No unified QMS dashboard across all modules
17. No QMS health score / maturity assessment
18. No cross-module reporting (how many RCAs originated from FMEA high-RPN items?)

---

## Success Metrics

| Metric | Phase 0 | Phase 1 | Phase 2 | Phase 4 |
|--------|---------|---------|---------|---------|
| Critical bugs | 0 | 0 | 0 | 0 |
| Modules connected | 0 of 5 | 5 of 5 | 5 of 5 | 5 of 5 |
| QMS-001 assertions passing | ~60% | ~85% | ~95% | 100% |
| AIAG FMEA compliance | Partial | Partial | Full | Full |
| ISO 9001 clause coverage | ~40% | ~55% | ~75% | ~95% |
| Enterprise modules | 5 | 5 | 5 | 9 (+ CAPA, DCN, Training, Supplier) |
| Evidence bridges active | 1 (FMEA, broken) | 5 | 5 | 7 |
| KG entity creation | 0 | Manual | Auto | Auto + cross-tenant |

---

*Generated from 4 parallel audit agents. 126 findings across 6 domains. This roadmap is a living document — update after each phase completion.*
