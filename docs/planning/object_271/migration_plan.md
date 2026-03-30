# SVEND Product Rebuild — Object 271

**Date:** 2026-03-30
**Status:** PLANNING
**Scope:** Full product upgrade — MVP to production. Every surface, every template, forge decomposition, forgeviz integration, OLR-001 quality system.

---

## What This Is

Not a QMS migration. A full product rebuild. The QMS overhaul (OLR-001, GRAPH-001, CANON-001/002) motivated it, but the base template has changed, the widget library replaces per-template CSS, forgeviz replaces Plotly, and forge packages are extracting computation out of SVEND. Every template in the product gets rebuilt on the new stack.

The existing app stays live throughout. New surfaces are built at parallel routes, wired to the same backend APIs. When a surface is ready, the nav link switches. Old template gets deleted.

## Principles

1. **Build parallel, cut over when ready.** No breaking the production app.
2. **Same backend.** No model changes, no API changes, no migrations for the UI rebuild. New templates consume existing endpoints.
3. **One stack.** sv-* widgets, rack templates, forgeviz charts, forge package computation. No exceptions.
4. **Real data or it doesn't count.** Every surface tested with Eric's account before cutover.
5. **forgegov validates.** No session declares "done" without `forgegov run` passing.

---

## The Stack

| Layer | Old | New |
|-------|-----|-----|
| Page layout | Per-template CSS, `!important` overrides | Rack templates (workspace, dashboard, CRUD) |
| Components | Per-template `.btn`, `.card`, `.kpi-strip` | `svend-widgets.css` — sv-* classes only |
| Charts | Plotly.js (3.5MB CDN) | ForgeViz (15KB, zero deps) |
| Computation | Inline in Django views (spc.py, dsw_views.py) | Forge packages (forgespc, forgedoe, forgestat, etc.) |
| Compliance | 37 SVEND checks + per-package duplication | 31 SVEND checks + forgegov bridge |
| Container | `.container { max-width: 1200px }` + per-template hacks | `{% block container_class %}sv-full{% endblock %}` |
| Colors | Hardcoded hex per template | `--sv-*` CSS variables (theme-aware) |

---

## Template Inventory (115 files, 160K lines)

### Tier 1 — Rebuild First (new rack templates, highest user value)

These are the core product surfaces that users interact with daily.

| Surface | Current | Lines | Rack | Notes |
|---------|---------|-------|------|-------|
| **QMS Workbench** | qms.html (blank) | 98 | workspace | NEW. Signals, investigations, commitments, FMIS, PCs, FFTs, suppliers, audits, training, docs, graph health, config. Single workbench with section sidebar + forgeviz. OLR-001 architecture. |
| **Main Dashboard** | dashboard.html | 568 | dashboard | Rebuild with forgeviz KPI charts, sv-* widgets. Current version is the "AI-built bento box" flagged in UI audit. |
| **Graph Map** | graph_map.html | 404 | workspace | Process map + Cytoscape. Already lightweight — rebuild on workspace rack. |
| **Investigations** | investigations.html | 808 | workspace | CANON-002 UI. Rebuild inside QMS workbench as a section, not standalone page. |
| **Safety** | safety_app.html | 1,685 | workspace | HIRARC, Frontier Cards, BBSO. Standalone workbench (not inside QMS). |

### Tier 2 — Rebuild Next (existing tools, high usage)

These are proven tools that need rack/widget migration but not fundamental redesign.

| Surface | Current | Lines | Rack | Notes |
|---------|---------|-------|------|-------|
| **FMEA** | fmea.html | 1,706 | workspace | Matrix editor. Standalone tool AND FMIS view inside QMS workbench. |
| **RCA** | rca.html | 2,137 | workspace | Root cause analysis + AI critique. |
| **A3** | a3.html | 1,491 | workspace | A3 report editor. |
| **Hoshin** | hoshin.html | 4,397 | workspace | X-Matrix. Complex — large rebuild. |
| **VSM** | vsm.html | 3,703 | workspace | Value stream mapping canvas. |
| **Whiteboard** | whiteboard.html | 5,191 | workspace | Collaborative visual. Large rebuild. |
| **ISO Documents** | iso_doc.html | 1,362 | workspace | Controlled document editor. |
| **Learn** | learn.html | 5,060 | workspace | Courses, assessments, certification. |
| **Settings** | settings.html | 1,448 | crud | User preferences + account. |
| **Projects** | projects.html | 4,479 | workspace | Hypothesis management. |

### Tier 3 — Decompose (forge extraction changes the architecture)

These don't just get new templates — the computation moves to forge packages first, then the UI rebuilds as a thin shell.

| Surface | Current | Lines | Forge Package | Notes |
|---------|---------|-------|---------------|-------|
| **Analysis Workbench** | analysis_workbench.html | 11,464 | forgestat, forgespc, forgedoe | IDE/CLI piece may become standalone forge system. SVEND keeps thin analysis section in QMS workbench. |
| **Simulator** | simulator.html | 9,222 | forgespc, forgeviz | Plant sim with SPC viz. Rebuild on forgeviz after SPC extraction complete. |
| **Calculators** | calculators.html | 8,199 | forgestat, forgespc | Operations workbench. 15 calculator tools in tools/ subdirectory. Forge packages replace inline math. |
| **Models (ML)** | models.html | 2,550 | forgeml (TBD) | Train/predict/deploy. Blocked on forgeml extraction. |
| **Experimenter** | experimenter.html + 12 partials | 251 + partials | forgedoe | DOE wizard. Already partially extracted. |
| **Forecast** | forecast.html | 418 | forgestat | Time series. Small. |

### Tier 4 — Migrate When Touched (low urgency, update on contact)

| Surface | Lines | Notes |
|---------|-------|-------|
| Ishikawa | 862 | Fishbone diagram. Simple tool. |
| CE Matrix | 525 | Scoring matrix. Simple tool. |
| Kanban Cards | 970 | Card generator. |
| Triage | 701 | Data cleaning. |
| Notebooks | 1,139 | Trial tracking. |
| Forge | 631 | Synthetic data UI. |
| Workflows | 947 | Agent pipeline builder. |
| Coder | 757 | AI coding assistant. |
| Harada | 1,039 | Personal practice. |
| Knowledge | 1,165 | Graph visualizer (may merge into graph_map). |
| Workbench (agent) | 1,341 | Multi-agent orchestration. |

### Tier 5 — Content & Marketing (no rack needed, just widget cleanup)

20 content pages (playbooks, blog, comparisons, landing). These extend `base_guest.html` or `tool_base.html`, not `base_app.html`. Update to sv-* widgets when redesigning marketing. Not on the critical path.

### Delete

| File | Lines | Reason |
|------|-------|--------|
| workbench_new.html | 11,790 | Confirmed duplicate of analysis_workbench.html |
| safety_coming_soon.html | 1,993 | Replaced by safety_app.html |
| iso_9001_qms.html | 1,159 | Replaced by QMS workbench |

### Keep As-Is

| Category | Files | Notes |
|----------|-------|-------|
| Error pages | 4 | 400/403/404/500 — 8 lines each |
| Print templates | 5 | PDF output — don't need racks |
| Auth/onboarding | 9 | Login, register, verify — functional, not product surfaces |
| Guest/legal | 4 | Terms, privacy, guest layouts |
| Internal dashboard | 1 | Staff-only. 8,323 lines. Rebuild last or never. |

---

## ForgeViz Integration

**Current state:** forgeviz package at ~/forgeviz/ — 17 chart builders, SVG + Plotly renderers, 1,158-line JS renderer. NOT wired into SVEND.

**Integration steps:**
1. Copy `forgeviz.js` to `static/js/`, add `<script>` to `base_app.html`
2. Run `collectstatic`
3. Add chart endpoint: view that accepts chart params, calls forge package, returns ChartSpec JSON
4. QMS workbench renders charts via `ForgeViz.render(el, spec)`
5. As each surface rebuilds, replace Plotly calls with forgeviz

**Plotly removal is per-template.** Each template that uses Plotly gets its charts replaced during its rebuild. The CDN `<script>` tag stays in base_app.html until the last Plotly consumer is gone.

---

## Forge Package Wiring

Forge packages are already extracted and have their own repos. Wiring means: SVEND view imports forge package instead of inline code, calls it, returns the result. API response shape doesn't change.

| Package | Replaces | Status |
|---------|----------|--------|
| forgespc | agents_api/spc.py | v0.1.0, ready to wire |
| forgedoe | agents_api/experimenter | v0.1.0, partially wired |
| forgedoc | WeasyPrint calls in views | v0.1.0, endpoint exists |
| forgestat | dsw_views.py statistical tests | v0.1.0, 200+ analyses |
| forgesiop | S&OP calculations | v0.1.0 |
| forgeviz | Plotly chart generation | v0.1.0, JS renderer ready |
| forgecal | Calibration service | 40% complete |
| forgeml | ML pipeline | TBD |
| forgesia | Synara belief engine | TBD |
| forgebay | Bayesian analysis | TBD |

Wiring happens **during** template rebuilds, not before. When a surface gets rebuilt, its computation moves to the forge package at the same time.

---

## Build Sequence

### Phase 0: Foundation (DONE)

- [x] Widget library (svend-widgets.css)
- [x] Rack templates (workspace, dashboard, CRUD)
- [x] Command palette (Cmd+K)
- [x] Full-width header
- [x] Compliance consolidation (forgegov bridge)
- [x] Critical review + 11 fixes on rack/widget architecture

### Phase 1: QMS Workbench + ForgeViz

**Goal:** The QMS workbench at `/app/qms/` with real data, forgeviz charts, OLR-001 architecture.

1. Wire forgeviz.js into SVEND
2. Build `base_qms.html` (extends workspace rack — section sidebar, KPI strip, swappable canvas)
3. Build sections one at a time, each wired to existing loop/ and graph/ APIs:
   - Signals (triage queue — proven pattern from demo)
   - Investigations
   - Commitments
   - FMIS (Bayesian FMEA matrix)
   - Suppliers (claims, CoA, scorecards)
   - Process Confirmations + Forced Failure Tests
   - Audits
   - Training
   - Documents
   - Graph health + gap report
   - Configuration (presets, thresholds)
   - Overview dashboard (forgeviz KPIs, trend charts, readiness score)
4. Eric tests each section with real data
5. When ~80% of sections work: switch nav to QMS workbench, delete old iso_9001_qms.html

### Phase 2: Core Tools

Rebuild proven tools on racks with sv-* widgets. One at a time.

- Main dashboard (forgeviz KPIs)
- Graph map (workspace rack + Cytoscape)
- Safety workbench
- FMEA (standalone + FMIS integration)
- RCA, A3, Hoshin, VSM
- ISO document editor
- Learn (courses + assessments)

### Phase 3: Forge Decomposition

As forge packages mature, decompose the heavy surfaces:

- Analysis workbench → forgestat/forgespc/forgedoe computation + thin SVEND shell (or standalone forge IDE)
- Simulator → forgespc + forgeviz
- Calculators → forge packages + forgeviz
- ML models → forgeml
- Experimenter → forgedoe

### Phase 4: Cleanup

- Remove old CSS from base_app.html (the `.btn`, `.card`, `.kpi-strip` classes)
- Remove Plotly CDN tag
- Delete old templates
- Run full compliance
- Update standards to reflect shipped architecture

---

## Session Roles

| Role | Owns | Cannot Touch |
|------|------|-------------|
| **Systems Engineer** | Rack templates, widget CSS, forgeviz integration, template rebuilds | Models, views, migrations |
| **Backend** | Forge package wiring, API contracts, view function updates | Templates, CSS, JS |
| **Quality (forgegov)** | Validate between phases, contract enforcement | Feature code |
| **Eric** | Priorities, testing with real data, product decisions, cutover calls | — |

---

## Communication Protocol

1. **Before starting:** Read this plan. Run `forgegov run`.
2. **Before committing:** Verify file ownership. CHG-001 CR required.
3. **After committing:** Note in `object_271/migration_log.md`.
4. **When blocked:** Write spec in `object_271/` describing what you need.
5. **When "done":** Quality session validates.

---

## Definition of Done (Full Rebuild Complete)

1. Every app surface uses rack templates + sv-* widgets
2. Every chart renders via forgeviz (no Plotly)
3. Every computation routes through forge packages (no inline math in Django views)
4. QMS workbench implements OLR-001 architecture (Signals, not NCRs)
5. Zero duplicate templates (no workbench_new, no iso_9001_qms)
6. `svend-widgets.css` is the only source of component CSS
7. `forgegov run` passes
8. `python3 manage.py run_compliance --all` passes
9. All nav links point to working pages
10. Eric signs off
