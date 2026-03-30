# SVEND Migration Plan — Object 271 Final Phase

**Date:** 2026-03-30
**Status:** PLANNING
**Goal:** Rebuild SVEND's UI on the rack/widget architecture, wire forge packages into the backend, consolidate compliance through forgegov, and ship a single cohesive system.

---

## Principles

1. **One thing at a time.** Each session has one job. No session touches another session's files.
2. **Backend stable before frontend.** API contracts freeze before templates get built.
3. **forgegov validates every step.** No session declares "done" without `forgegov run` passing.
4. **Real data or it doesn't count.** "It renders" is not done. "It works with Eric's account data" is done.
5. **No parallel QMS systems.** One backend, one frontend, one navigation. Period.

---

## Current State

### What Exists and Works
- **Forge packages (7):** forgespc, forgedoc, forgesiop, forgecal, forgegov, forgedoe, forgeviz
- **Forge governance:** forgegov CLI, 6-stage pipeline, JSON report bridge
- **Widget library:** `static/css/svend-widgets.css` — 19 component categories
- **Rack templates:** `base_workspace.html`, `base_dashboard.html`, `base_crud.html`
- **Command palette:** `Cmd+K` in `base_app.html`
- **Backend APIs:** All loop/, iso/, graph/ endpoints intact
- **Backend models:** All models intact (loop, graph, agents_api)
- **Tools:** A3, FMEA, RCA, Safety, Analysis Workbench, Graph Map, Whiteboard, Simulator — all working
- **Demo:** `/app/demo/rack/` — CRUD rack with simulated signals

### What's Broken / Missing
- **QMS UI:** Blank placeholder at `/app/qms/`
- **Nav dropdowns:** Quality/Hoshin/Safety dropdowns reference deleted pages
- **Inline CSS duplication:** `base_app.html` has `sv-*` inline AND `svend-widgets.css` external
- **`workbench_new.html`:** 11,790 line duplicate of `workbench.html`
- **`collectstatic` not run:** Widget CSS not served via WhiteNoise
- **Forge packages not wired into SVEND:** forgespc/forgeviz/etc installed but kjerne still uses its own code
- **SVEND compliance bridge:** `check_forge_ecosystem()` not yet wired into `syn/audit/compliance.py`

---

## Session Roles

| Role | Responsibility | Cannot Touch |
|------|---------------|-------------|
| **Backend** | API contracts, model changes, forge package wiring, view functions | Templates, CSS, JS |
| **Frontend** | Templates on racks, widget usage, JS interactions | Models, views, migrations |
| **Quality (forgegov)** | Validate between steps, contract enforcement, compliance bridge | Feature code |
| **Eric** | Arbitrate, set priorities, test with real usage | — |

---

## Phase 1: Cleanup (Before Any Rebuild)

### 1.1 Remove inline CSS duplication
**Owner:** Frontend
**Action:** Remove the `sv-*` instrument library block from `base_app.html` (lines 1117-1212). The external `svend-widgets.css` is now canonical.
**Verify:** Demo page still renders correctly.

### 1.2 Run collectstatic
**Owner:** Backend (requires Django)
**Action:** `python3 manage.py collectstatic --noinput`
**Verify:** Widget CSS served at `/static/css/svend-widgets.css`

### 1.3 Delete `workbench_new.html`
**Owner:** Frontend
**Action:** Delete the duplicate. Verify `workbench.html` is the canonical one.
**Verify:** No URL references `workbench_new.html`.

### 1.4 Fix nav dropdown dead links
**Owner:** Frontend
**Action:** Remove or redirect Quality/Hoshin/Safety dropdown entries that reference deleted pages. Point QMS-related links to `/app/qms/`.
**Verify:** No nav link 404s.

### 1.5 Wire forgegov compliance bridge
**Owner:** Backend
**Action:** Add `check_forge_ecosystem()` to `syn/audit/compliance.py`. Reads `~/.forge/reports/forgegov_latest.json`, checks freshness + passed, records ComplianceCheck.
**Verify:** `python3 manage.py run_compliance --check forge_ecosystem` passes.

### 1.6 Run forgegov full pipeline
**Owner:** Quality
**Action:** `cd ~/forgegov && forgegov run`
**Verify:** All stages pass. Report written to `~/.forge/reports/`.

**Gate:** All 1.1-1.6 complete before Phase 2 begins.

---

## Phase 2: API Contract Freeze

### 2.1 Document every API endpoint shape
**Owner:** Backend
**Action:** For each API used by the QMS UI, document the exact JSON response shape in a contract file. Cover:
- `/api/loop/signals/` — list + detail
- `/api/loop/commitments/` — list + detail + actions
- `/api/loop/claims/` — list + detail + lifecycle
- `/api/loop/coas/` — list + detail + ingest
- `/api/loop/pcs/` — list + detail
- `/api/loop/ffts/` — list + detail
- `/api/loop/fmis/` — list + rows + posteriors
- `/api/loop/policies/` — list + detail + registry
- `/api/loop/dashboard/` — aggregated data
- `/api/loop/readiness/` — CI readiness score
- `/api/iso/ncrs/` — list + detail + lifecycle
- `/api/iso/audits/` — list + detail + findings
- `/api/iso/training/` — list + detail
- `/api/iso/reviews/` — list + detail
- `/api/iso/documents/` — list + detail
- `/api/iso/complaints/` — list + detail
- `/api/iso/suppliers/` — list + detail + evaluation
- `/api/iso/dashboard/` — ISO dashboard aggregates
- `/api/graph/data/` — graph nodes + edges
- `/api/graph/search/` — unified search
- `/api/graph/health/` — knowledge health metrics
- `/api/graph/activity/<type>/<id>/` — activity feed
- `/api/graph/gates/<type>/<id>/` — workflow gates

**Format:** JSON schema or example response in `docs/api_contracts/`.
**Verify:** Each endpoint returns data matching the documented contract.

### 2.2 Wire forge packages into backend
**Owner:** Backend
**Action per package:**
1. Update view functions to import from forge package instead of kjerne code
2. Run forgegov to verify package contracts
3. Verify API response shape doesn't change
4. Delete replaced kjerne code (after verification)

**Sequence:**
- forgespc → `spc_views.py` (replace `agents_api/spc.py`)
- forgeviz → chart rendering (replace `agents_api/analysis/viz/`)
- forgedoc → document generation (replace WeasyPrint calls)

**Gate:** forgegov passes after each wiring. API contract shapes unchanged.

---

## Phase 3: QMS Frontend Rebuild

### 3.1 QMS entry point — decide the rack
**Owner:** Eric + Frontend
**Decision:** Is the QMS entry point a dashboard (KPI overview → drill into sections) or a workspace (sidebar nav → content area)?
**Options:**
- A) `base_dashboard.html` — KPI strip + panels, each panel links to a CRUD subpage
- B) `base_workspace.html` — sidebar with Loop stages, canvas shows current section
- C) Hybrid — workspace rack with dashboard as the default canvas

### 3.2 Build QMS page-by-page
**Owner:** Frontend
**Rule:** One page per commit. Each page:
1. Extends a rack template
2. Uses only `sv-*` widget classes
3. Calls documented API endpoints (from Phase 2 contracts)
4. Works with Eric's real account data
5. Has keyboard navigation (from rack base)

**Sequence (suggested — Eric adjusts):**
1. Signals (CRUD rack) — triage queue, proven pattern
2. NCR Tracker (CRUD rack) — same pattern, different data
3. Commitments (CRUD rack) — already proven, rebuild on sv-* widgets
4. Suppliers (workspace rack) — scorecard + claims + CoA panels
5. Investigations (workspace rack) — 3-pane with tools
6. FMIS (workspace rack) — risk landscape table
7. QMS Overview (dashboard rack) — KPI strip + summary panels
8. Remaining sections as needed

### 3.3 Wire navigation
**Owner:** Frontend
**Action:** Update `base_app.html` nav to point to live QMS pages as they ship. Remove dead dropdown entries. Add new entries only when the page is functional.

**Gate:** Eric tests each page with real data before the next one starts.

---

## Phase 4: Polish + Compliance

### 4.1 Remove old CSS from `base_app.html`
**Owner:** Frontend
**Action:** Once all migrated templates use `sv-*` classes, remove the old `.card`, `.btn`, `.kpi-strip` etc. from the inline style block in `base_app.html`.

### 4.2 Migrate remaining templates
**Owner:** Frontend (gradual, as templates get touched for other work)
**Action:** When any old template needs changes, migrate it to a rack + sv-* widgets at the same time.

### 4.3 Full compliance run
**Owner:** Quality
**Action:**
- `forgegov run` — all forge packages
- `python3 manage.py run_compliance --all` — all SVEND checks
- Verify `check_forge_ecosystem` reads passing forgegov report
- CHG-001 compliance on all commits

### 4.4 Standards update
**Owner:** Backend
**Action:** Update standards to reflect the new architecture:
- LOOP-001 — reference new QMS UI structure
- GRAPH-001 — reference forge package integration
- New standard: FORGE-001 — forge ecosystem governance (from forgegov QUALITY_AGENT.md)

---

## File Ownership Map

| Files | Owner | Others May |
|-------|-------|-----------|
| `static/css/svend-widgets.css` | Frontend | Read only |
| `templates/base_*.html` | Frontend | Read only |
| `templates/qms*.html` | Frontend | Read only |
| `templates/rack_demo.html` | Frontend | Read only |
| `svend/urls.py` (template routes) | Frontend | Read only |
| `svend/urls.py` (API includes) | Backend | — |
| `loop/models.py` | Backend | Read only |
| `loop/views.py` | Backend | Read only |
| `loop/urls.py` | Backend | Read only |
| `graph/*` | Backend | Read only |
| `agents_api/*` | Backend | Read only |
| `syn/audit/compliance.py` | Quality/Backend | — |
| `~/forge*/` | Per-package owner | Quality reviews |
| `~/.forge/reports/` | Quality (forgegov writes) | SVEND reads |
| `docs/api_contracts/` | Backend writes | Frontend reads |

---

## Communication Protocol

1. **Before starting work:** Check `forgegov run` + read this plan
2. **Before committing:** Verify file ownership — don't touch files you don't own
3. **After committing:** Note in `object_271/migration_log.md` what was done
4. **When blocked:** Write a spec in `object_271/` describing what you need from another session
5. **When "done":** Quality session validates before anyone declares done

---

## Definition of Done (Migration Complete)

1. `/app/qms/` is a functional QMS surface built on racks + sv-* widgets
2. Every QMS section works with real data (not simulated)
3. Zero duplicate QMS UIs (no iso.html, no loop_dashboard.html, no loop shell)
4. All forge packages wired into SVEND (no duplicate computation in kjerne)
5. `forgegov run` passes
6. `python3 manage.py run_compliance --all` passes
7. All nav links point to working pages
8. `svend-widgets.css` is the only source of component CSS
9. `workbench_new.html` deleted
10. Eric signs off
