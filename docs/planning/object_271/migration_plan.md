# SVEND Product Rebuild — Object 271

**Date:** 2026-03-30
**Status:** IN PROGRESS
**Scope:** Full product upgrade — MVP to production.

---

## What This Is

Every template in the product gets rebuilt on the new stack. QMS overhaul (OLR-001) motivated it. The base template has changed, the widget library replaces per-template CSS, ForgeViz replaces Plotly, forge packages extract computation, and shared JS libraries replace per-template duplication.

The existing app stays live. New surfaces are built at parallel routes, wired to the same backend APIs. When ready, the nav link switches and old template gets deleted.

## Principles

1. **Build parallel, cut over when ready.** No breaking the production app.
2. **Same backend.** No model changes, no API changes, no migrations for the UI rebuild.
3. **One stack.** sv-* widgets, rack templates, ForgeViz charts, forge packages, shared JS. No custom CSS except where structurally required.
4. **Real data or it doesn't count.** Every surface tested with Eric's account before cutover.
5. **forgegov validates.** No session declares "done" without `forgegov run` passing.
6. **Quality reports after every sprint.** QM reviews all template changes.

---

## The Stack

| Layer | Old | New |
|-------|-----|-----|
| Page layout | Per-template CSS, `!important` overrides | Rack templates (workspace, dashboard, CRUD) |
| Components | Per-template `.btn`, `.card`, `.kpi-strip` | `svend-widgets.css` — sv-* classes only |
| Charts | Plotly.js (3.5MB CDN) | ForgeViz (15KB, zero deps, pip-installed) |
| Computation | Inline in Django views + client-side JS | Forge packages (forgespc, forgedoe, forgestat, forgesiop, forgequeue, etc.) |
| API calls | Per-template fetch wrappers | `SvApi.get/post/put/del` (sv-api.js) |
| Modals | Per-template modal CSS + JS | `SvModal.open/close/create/confirm` (sv-modal.js) |
| Formatting | Per-template formatDate, timeAgo | `SvFormat.*` (sv-format.js) + `esc()`, `timeAgo()`, `svCsrf()` in base |
| Pan/zoom canvas | Per-template updateTransform | `SvCanvas.init` (sv-canvas.js) |
| Documents | Per-template PDF generation | ForgeDoc (7 QMS builders, pip-installed) |
| Belief engine | agents_api/synara/ (inline) | ForgeSia (pip-installed, wiring pending) |
| Compliance | 37 checks + per-package duplication | 31 checks + forgegov bridge |

---

## What's Done (Phase 0)

- [x] Widget library (`svend-widgets.css` — 20 component categories)
- [x] Rack templates (workspace, dashboard, CRUD)
- [x] Full-width header
- [x] Command palette (Cmd+K)
- [x] `container_class` block system (replaces `:has()` hacks)
- [x] Compliance consolidation (forgegov bridge, 37→31 checks)
- [x] Critical review + 11 fixes on rack/widget architecture
- [x] ForgeViz wired (pip-installed, collectstatic, JS served)
- [x] ForgeDoc wired (7 builders, Branding export fixed)
- [x] Shared JS libraries (sv-api, sv-modal, sv-format, sv-canvas)
- [x] Shared utils in base_app.html (esc, timeAgo, svCsrf)
- [x] JsBarcode + QRCode moved to global
- [x] 34 templates marked with migration tier/status
- [x] Migration dashboard at `/app/demo/` (staff-only)
- [x] graph_map.html migrated (workspace rack, zero custom CSS, XSS fixed)
- [x] QMS workbench at `/app/demo/qms/` (workspace rack, signals + overview wired)

---

## Template Inventory

Every pending template has a `{# MIGRATION: ... #}` comment on line 2.
`grep -r "MIGRATION:" templates/` shows the full inventory.

### Tier 1 — Rebuild First

| Surface | Lines | Rack | Status | Notes |
|---------|-------|------|--------|-------|
| **QMS Workbench** | new | workspace | IN PROGRESS | At /app/demo/qms/. OLR-001 architecture. |
| **Graph Map** | 232 | workspace | DONE | Migrated. Zero custom CSS. |
| **Main Dashboard** | 568 | dashboard | PENDING | ForgeViz KPIs. |
| **Safety** | 1,685 | workspace | PENDING | HIRARC, Frontier Cards. Standalone workbench. |

### Tier 2 — Core Tools (UX improvements during migration)

| Surface | Lines | Rack | Status | Design Change |
|---------|-------|------|--------|---------------|
| **VSM** | 3,703 | workspace | PENDING | **Cockpit UX** — calculator panel integrated below map. See VSM Workbench spec below. |
| **Analysis Workbench** | 11,464 | workspace | PENDING | **Standalone product** — its own surface, not inside QMS. Forge packages replace client-side computation. |
| **FMEA** | 1,706 | workspace | PENDING | Standalone + FMIS integration in QMS workbench. |
| **RCA** | 2,137 | workspace | PENDING | |
| **A3** | 1,491 | workspace | PENDING | |
| **Hoshin** | 4,397 | workspace | PENDING | X-Matrix. Complex. |
| **Whiteboard** | 5,191 | workspace | PENDING | SvCanvas for pan/zoom. |
| **ISO Documents** | 1,362 | workspace | PENDING | ForgeDoc integration. |
| **Learn** | 5,060 | workspace | PENDING | Courses, assessments. |
| **Projects** | 4,479 | workspace | PENDING | Hypothesis management. |
| **Settings** | 1,448 | crud | PENDING | |
| **Investigations** | 808 | workspace | PENDING | Also a QMS workbench section. |

### Tier 3 — Forge Decomposition

Computation moves to forge packages. Templates become thin shells.

| Surface | Lines | Forge Packages | Status |
|---------|-------|----------------|--------|
| **Simulator** | 9,222 | forgesiop.simulation, forgespc, forgeviz | PENDING |
| **Calculators** | 8,199 | forgesiop.production, forgequeue, forgestat | PENDING — merging into VSM cockpit |
| **Models (ML)** | 2,550 | forgeml (TBD) | PENDING |
| **Experimenter** | 251 | forgedoe | PENDING |
| **Forecast** | 418 | forgestat.timeseries | PENDING |

### Tier 4 — Migrate When Touched

ishikawa, ce_matrix, kanban_cards, triage, notebooks, forge, workflows, coder, harada, knowledge, workbench, front_page (12 templates)

### Tier 5 — Content & Marketing

20 content pages. No rack needed. Widget cleanup when redesigning marketing.

### Delete

- workbench_new.html (11,790 lines — confirmed duplicate)
- safety_coming_soon.html (1,993 lines — replaced)
- iso_9001_qms.html (1,159 lines — replaced by QMS workbench)

---

## VSM Workbench Spec

**Current problem:** VSM and calculators are separate pages. User draws VSM, navigates to ops workbench, imports VSM data, runs calculator, navigates back. Context switch kills the workflow.

**New design:** Single workspace rack template with integrated calculator cockpit.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Toolbar: [Current/Future] [Fit] [Layout] [Takt: ___s] [Save]      │
├───────────┬─────────────────────────────────────────────────────────┤
│           │  VSM MAP (top ~60% of canvas)                           │
│  Sidebar  │  ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐         │
│           │  │ Sta 1│───▶│ Sta 2│───▶│ Sta 3│───▶│ Sta 4│         │
│  Steps    │  │ CT:45│    │CT:120│    │ CT:30│    │ CT:55│         │
│  Inventory│  └──────┘    └──────┘    └──────┘    └──────┘         │
│  Kaizen   │         ▲ selected                                     │
│  Legend   ├─────────────────────────────────────────────────────────┤
│           │  COCKPIT (bottom ~40%, tabs)                            │
│  Props    │  [Station] [Line Balance] [OEE] [Queue] [SMED] [Sim]   │
│  (of sel) │                                                         │
│           │  Station 2 — CT: 120s  Changeover: 600s  Uptime: 92%   │
│           │  ┌─────────────────┐  ┌────────────────────────────┐   │
│           │  │ Cycle time dist │  │ Takt vs CT (all stations)  │   │
│           │  │ [ForgeViz hist] │  │ [ForgeViz yamazumi bar]    │   │
│           │  └─────────────────┘  └────────────────────────────┘   │
│           │  Bottleneck: YES (CT/Takt = 1.33)  Utilization: 92%    │
├───────────┴─────────────────────────────────────────────────────────┤
│  Status: 4 stations | Takt: 90s | Total CT: 250s | VA ratio: 12%   │
└─────────────────────────────────────────────────────────────────────┘
```

**How it works:**
- Click a station on the map → cockpit updates with that station's data
- Cockpit tabs run different calculators scoped to the selection: station detail, line balance (yamazumi for all stations), OEE breakdown, queue analysis, SMED scenario, live simulation
- Change a parameter in the cockpit (e.g., reduce changeover time via SMED) → future state map updates
- ForgeViz renders all charts. Forge packages (forgesiop.production, forgequeue) do the math server-side.
- `calc-vsm.js` field mappings still work — they just run inline instead of navigating away
- Synara hypothesis linking on kaizen bursts stays (VSM → Synara is already built)

**What this replaces:**
- `calculators.html` as a standalone page (for VSM-integrated calculators)
- `calc-vsm.js` import/export modal workflow
- The page hop between VSM and calculators

**What stays separate:**
- Calculators that aren't VSM-related (queue theory standalone, QFD, probit, Bayesian stats) stay in the operations workbench as their own surface
- The operations workbench becomes the home for non-VSM calculators, standalone simulations, and tools that don't need a process context

---

## Forge Package Dependencies

See `docs/planning/object_271/forge_ops_spec.md` for full spec.

### P0 — Blocks operations workbench + VSM cockpit migration

| Package | What |
|---------|------|
| **forgesiop.production** | takt, oee, kanban, epei, littles_law, bottleneck, cost |
| **forgequeue** (new) | mm1, mmc, mmck, erlang_c, priority, tandem, staffing |

### P1 — Full calculator coverage

| Package | What |
|---------|------|
| **forgesiop.lean** | smed, changeover_matrix, product_family_analysis |
| **forgestat.quality.desirability** | derringer_suich multi-response optimization |
| **forgestat.core.sampling** | sample_normal/exp/weibull/poisson, seeded_rng |

### P2 — Post-migration polish

| Package | What |
|---------|------|
| **forgesiop.simulation.engine** | Discrete event sim from VSM data |
| **ForgeViz chart returns** | to_chart() on all forge computation functions |
| **calc-vsm.js refactor** | Single loader + per-calculator field mappings |

---

## Graph Integration Boundaries

**Graph producers** (write causal evidence):
- Investigations → writeback
- SPC → shift detection → edge staleness
- DOE → effect sizes → edge posteriors
- FFTs → detection capability
- FMEA/FMIS ↔ Graph (bidirectional — failure modes ARE causal claims)

**Graph consumers** (read process knowledge):
- Graph Map (visualization)
- QMS workbench graph health section
- Knowledge health metrics

**NOT graph-related:**
- Calculators — operational decision tools, consume VSM data
- VSM — stores process steps, feeds calculators
- Hoshin — standalone strategy deployment
- Simulations — scenario analysis on VSM data

---

## Tech Debt — Clean Up During Migration

Debt items are resolved during template migration, not as separate work. When a view file is touched for a template rebuild, the debt in that file gets resolved in the same CR.

### Dead Code — Delete

| File | Lines | Reason |
|------|-------|--------|
| `workbench_new.html` | 11,790 | Confirmed duplicate of analysis_workbench.html |
| `safety_coming_soon.html` | 1,993 | Replaced by safety_app.html |
| `iso_9001_qms.html` | 1,159 | Replaced by QMS workbench |
| `core/models/graph.py` | ~200 | DEPRECATED. Replaced by `graph/` app (GRAPH-001). Marked in code. |
| `workbench/models.py` KnowledgeGraph | ~100 | Third KG implementation. Replaced by `graph/` app. |

### Duplicate Computation — Forge Packages Replace

Wire forge imports when the corresponding view is touched during migration.

| SVEND Code | Lines | Forge Replacement | Wire When |
|---|---|---|---|
| `agents_api/spc.py` | 1,888 | `forgespc` (identical function names) | spc_views.py touched |
| `agents_api/dsw/stats_*.py` (7 files) | 21,988 | `forgestat` | dsw_views.py touched |
| `agents_api/synara/` (6 files) | 3,243 | `forgesia` | synara_views.py touched |
| `agents_api/dsw/chart_render.py` + `chart_defaults.py` | 518 | `forgeviz` | Any DSW chart endpoint touched |
| `agents_api/analysis/chart_render.py` + `chart_defaults.py` | 518 | `forgeviz` | Duplicate of dsw/ — delete |

### dsw/ vs analysis/ Duplication — 9.6MB of duplicate code

Two directories with the same statistical engines in different file structures:
- `agents_api/dsw/` — flat files (`stats_parametric.py`, 4.8MB)
- `agents_api/analysis/` — subdirectories (`stats/parametric/`, 4.8MB)

Both exist because someone restructured without deleting the old layout. Need to:
1. Check which one `dsw_views.py` actually imports at runtime
2. Keep that one
3. Delete the other (~4.8MB removed)
4. Eventually both go away when forgestat replaces all inline stats

### Deprecated Models Still in Production Views

| Model | Views | Replaced By | Action |
|---|---|---|---|
| `CAPAReport` | `capa_views.py` | Investigation + ForgeDoc CAPA generator | Deprecate views. Remove when old templates gone. |
| `ActionItem` | `action_views.py`, `fmea_views.py` | Commitment (LOOP-001 §3) | Deprecate views. Remove when old templates gone. |

Do NOT delete these models or views yet — old templates still call them. Add deprecation docstrings. Remove when the templates that use them are migrated.

### Plotly in Python — 15 Files Generate Plotly JSON Server-Side

Switch to ForgeViz ChartSpec when the view is touched. Main files:

| File | Plotly Usage | Action |
|---|---|---|
| `spc_views.py` | 3 `"template": "plotly_dark"` refs | Switch to `forgeviz.render(spec, format='dict')` |
| `spc.py` | Gage R&R generates 4 Plotly charts | Switch to `forgeviz.charts.gage` |
| `dsw/chart_render.py` | Generic Plotly defaults | Delete — ForgeViz replaces |
| `dsw/chart_defaults.py` | 459 lines of Plotly config | Delete — ForgeViz themes handle this |
| `dsw/stats_exploratory.py` | Chart generation in analysis results | Switch to forgeviz |
| `loop/models.py:1711` | `help_text` references Plotly | Update help_text string |

### Stale Views to Deprecate

| View | Lines | Issue | Action |
|---|---|---|---|
| `experimenter_views.py:power_analysis` | ~50 | Docstring says DEPRECATED, still served | Add HTTP deprecation header |
| `synara_views.py` | 1,136 | Operates on old agents_api/synara/ | Wire to forgesia when `__init__` exports fixed |
| `capa_views.py` | ~200 | CAPAReport deprecated per LOOP-001 | Deprecate, remove with template |
| `action_views.py` | ~60 | ActionItem superseded by Commitment | Deprecate, remove with template |

---

## Session Roles

| Role | Owns | Cannot Touch |
|------|------|-------------|
| **Systems Engineer** | Rack templates, widget CSS, ForgeViz integration, template rebuilds, shared JS | Models, views, migrations |
| **PM (S1)** | Forge packages, forge ops functions, package wiring into views | Templates, CSS |
| **Quality (forgegov)** | Validate between phases, contract enforcement, quality reviews | Feature code |
| **Eric** | Priorities, real-data testing, product decisions, cutover calls | — |

---

## Communication Protocol

1. **Before starting:** Read this plan. Check `grep -r "MIGRATION:" templates/` for current status.
2. **ALL new views and templates MUST be tested under `/app/demo/` first.** No exceptions. The "Migration" nav link (staff-only) is the staging area. Only after Eric approves does the nav link swap to the new template.
3. **Before committing:** CR required. File ownership check. No custom CSS without justification.
4. **After committing:** Update migration dashboard template data + this plan.
5. **After each sprint:** Quality report to QM with: what changed, what was added to widget library, any defects found, any XSS vectors.
6. **When blocked on forge packages:** Write spec in `object_271/`, notify PM.

---

## Definition of Done

1. Every app surface uses rack templates + sv-* widgets
2. Every chart renders via ForgeViz (no Plotly anywhere — Python or JS)
3. Every computation routes through forge packages (no inline stats/SPC/synara in agents_api/)
4. QMS workbench implements OLR-001 architecture
5. VSM workbench has integrated calculator cockpit
6. Analysis workbench is standalone surface with forge-backed computation
7. Zero duplicate templates
8. `svend-widgets.css` is the only source of component CSS
9. Shared JS (SvApi, SvModal, SvFormat, SvCanvas) used everywhere — no per-template duplication
10. `forgegov run` passes
11. `python3 manage.py run_compliance --all` passes
12. All nav links point to working pages
13. Dead code removed (core/models/graph.py, workbench KG, duplicate templates)
14. dsw/ vs analysis/ duplication resolved (one directory, not two)
15. Deprecated views removed (capa_views, action_views) after their templates are gone
16. Eric signs off
13. Eric signs off
