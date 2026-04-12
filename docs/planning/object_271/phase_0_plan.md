# Phase 0 — Forge Wiring, Tools App, Dead Code, iso_views Split

**Status:** DRAFT — planning artifact under CR `5bf7354c-3de5-4624-b505-a94a5b6ce0ea`
**Date:** 2026-04-10
**Author:** Claude (Systems Engineer role per Object 271)
**Inputs:**
- `docs/planning/object_271/qms_architecture.md` (locked v0.4) — target state
- `docs/planning/object_271/extraction_gap_analysis.md` — gap analysis §6
- `docs/planning/object_271/agents_api_inventory.md` — current state

**Purpose:** Detailed execution plan for Phase 0 of the agents_api extraction. Phase 0 must land completely before any model extraction (Phase 1A+) begins. It reduces `agents_api/` from ~289,313 LOC to ~220,000 LOC by wiring forge packages, deleting dead code, moving cross-cutting infrastructure to new homes, and splitting the `iso_views.py` monolith.

**Phase 0 is 10 CRs**, all under parent CR `5bf7354c`. Each CR is independently deployable and rollback-safe. No model migrations in Phase 0 — only code moves, rewiring, and deletions.

---

## Table of Contents

1. [CR-0.1 — Move dsw/ live files into analysis/](#cr-01)
2. [CR-0.2 — Wire forgespc, delete agents_api/spc.py](#cr-02)
3. [CR-0.3 — Wire forgestat, delete legacy dsw/ stats files](#cr-03)
4. [CR-0.4 — Wire forgesia, delete agents_api/synara/](#cr-04)
5. [CR-0.5 — Wire forgeviz completion](#cr-05)
6. [CR-0.6 — Dead-code deletion](#cr-06)
7. [CR-0.7 — permissions.py → qms_core/ shim](#cr-07)
8. [CR-0.8 — iso_views.py split-in-place](#cr-08)
9. [CR-0.9 — Create tools/ Django app (formalized tool router)](#cr-09)
10. [CR-0.10 — Phase 0 verification + LOC census](#cr-010)
11. [Dependency graph](#dependency-graph)
12. [Risk summary](#risk-summary)

---

<a id="cr-01"></a>
## CR-0.1 — Move dsw/ live files into analysis/

**Change type:** `enhancement`
**Risk:** LOW
**Dependencies:** none (first CR)
**CR partners:** none (within agents_api/)
**Blast radius:** ~12 files

### What

Per architecture §9.A J.11 + inventory §K.2 + gap analysis §6.1. `analysis/` is canonical. Move the 6 live `dsw/` files into `analysis/`:

| Source | LOC | Target |
|--------|-----|--------|
| `dsw/common.py` | 3,084 | `analysis/common.py` |
| `dsw/endpoints_data.py` | 1,832 | `analysis/endpoints_data.py` |
| `dsw/endpoints_ml.py` | 1,702 | `analysis/endpoints_ml.py` |
| `dsw/standardize.py` | 552 | `analysis/standardize.py` |
| `dsw/chart_render.py` | 59 | `analysis/chart_render.py` (replace the 7-line wrapper) |
| `dsw/chart_defaults.py` | 459 | `analysis/chart_defaults.py` (replace the 11-line wrapper) |

Delete the DOA stubs in `analysis/chart_render.py` (7 LOC) and `analysis/chart_defaults.py` (11 LOC).

### Import updates

Every file that imports from `agents_api.dsw.*` must update:
- `spc_views.py` — `from agents_api.dsw.common import ...` → `from agents_api.analysis.common import ...`
- `autopilot_views.py` — same pattern
- `forecast_views.py` — same
- `pbs_engine.py` — same
- `ml_pipeline.py` — same
- `dsw_views.py` — same
- `report_views.py` — `from agents_api.dsw.chart_render import ...` → `from agents_api.analysis.chart_render import ...`
- `a3_views.py` — same
- `dsw/dispatch.py` — update to `analysis/dispatch.py` refs (dispatch.py is already canonical in analysis/)

### Safety-net tests (5)

1. Upload CSV via `dsw_views` → correct response shape (200, result dict with expected keys)
2. `chart_render.render_chart()` → returns valid ChartSpec dict with `data`, `layout` keys
3. `standardize.post_process()` → `evidence_grade` field present in output
4. `common.read_csv_safe()` → handles malformed CSV gracefully (returns error, not 500)
5. `endpoints_data.run_analysis()` for a parametric test → returns `p_value` key

### Verification

```bash
# Before: count live dsw/ files
find agents_api/dsw/ -name "*.py" -not -name "__*" | wc -l

# After: verify all moved, stubs deleted
python manage.py check --deploy  # Django system check
python -m pytest tests/test_dsw_views_behavioral.py -x  # Existing behavioral tests pass
```

### LOC removed from agents_api/dsw/

~7,688 LOC moved (not deleted — relocated to analysis/). ~18 LOC of stubs deleted. Net: cleaner import boundary, no LOC reduction yet.

---

<a id="cr-02"></a>
## CR-0.2 — Wire forgespc, delete agents_api/spc.py

**Change type:** `enhancement`
**Risk:** MED
**Dependencies:** CR-0.1 (dsw/common now in analysis/)
**CR partners:** none
**Blast radius:** ~5 files

### What

Per gap analysis §6.2. Wire `forgespc` package into `spc_views.py` replacing inline imports from `agents_api.spc` (1,889 LOC) and any `dsw/spc*` legacy code.

**Delete:**
- `agents_api/spc.py` (1,889 LOC)
- `agents_api/dsw/spc.py` + `agents_api/dsw/spc_pkg/*` (if exists)

**Wire:** Replace `from agents_api.spc import ...` calls in `spc_views.py` with `import forgespc` calls. The `forgespc` package API should be verified against `spc_views.py` usage before the CR starts.

### Pre-CR checklist

- [ ] Verify `forgespc` covers all SPC analysis types used by `spc_views.py`: x-bar-R, x-bar-S, IMR, c-chart, p-chart, u-chart, np-chart, capability study (Cp/Cpk/Pp/Ppk), gage R&R (crossed/nested)
- [ ] Verify `forgespc` is importable on the production server (`python -c "import forgespc"`)
- [ ] Map each `spc_views.py` function to its `forgespc` equivalent

### Safety-net tests (7)

Per gap analysis §6.2 — one per major SPC analysis type:
1. x-bar-R chart → 10-key result contract populates (UCL, LCL, CL, data points, OOC flags)
2. x-bar-S chart → same contract
3. IMR chart → same contract
4. c-chart → count-based control limits correct
5. p-chart → proportion-based limits correct
6. Capability study → Cp, Cpk, Pp, Ppk all present and numeric
7. Gage R&R → repeatability, reproducibility, %GRR present

### LOC removed

~1,889 LOC from `spc.py` + any `dsw/spc*` duplicates.

---

<a id="cr-03"></a>
## CR-0.3 — Wire forgestat, delete legacy dsw/ stats files

**Change type:** `debt`
**Risk:** MED
**Dependencies:** CR-0.1 (dsw/common in analysis/)
**CR partners:** none
**Blast radius:** ~20+ files (largest Phase 0 CR)

### What

Per gap analysis §6.3 + inventory §H.2. After CR-0.1 moves live dsw/ files, the legacy compute files remain. These are the original inline statistical implementations that `forgestat` (215 tests, 68 analysis functions, 94% SVEND parity) supersedes.

**Delete (estimated ~50,000-55,000 LOC):**

| File/Dir | Est. LOC | Superseded by |
|----------|----------|---------------|
| `dsw/stats_parametric.py` | ~4,000 | `forgestat.parametric` |
| `dsw/stats_nonparametric.py` | ~3,500 | `forgestat.nonparametric` |
| `dsw/stats_posthoc.py` | ~2,000 | `forgestat.posthoc` |
| `dsw/stats_regression.py` | ~3,000 | `forgestat.regression` |
| `dsw/stats_advanced.py` | ~3,500 | `forgestat.advanced` |
| `dsw/stats_exploratory.py` | ~2,500 | `forgestat.exploratory` |
| `dsw/stats_quality.py` | ~2,000 | `forgespc` (quality metrics) |
| `dsw/bayesian.py` + `dsw/bayesian/*` | ~5,000 | `forgestat.bayesian` |
| `dsw/ml.py` | ~3,000 | `forgestat.ml` |
| `dsw/viz.py` | ~2,000 | `forgeviz` |
| `dsw/siop.py` | ~1,500 | `forgesiop` |
| `dsw/simulation.py` | ~2,000 | `forgesiop.simulation` |
| `dsw/reliability.py` | ~2,000 | `forgestat.reliability` |
| `dsw/d_type.py` | ~500 | `forgestat.types` |
| `dsw/exploratory/*` | ~4,000 | `forgestat.exploratory` |
| Remaining dsw/ files | ~10,000 | Various forge packages |

**Wire:** Verify that the existing `analysis/forge_*.py` bridge handlers (153 handlers across 11 files, already done per memory) correctly import from forge packages. Any remaining direct `dsw/stats_*` imports in view files must switch to forge bridge calls.

### Pre-CR checklist

- [ ] Run `grep -r "from agents_api.dsw.stats" agents_api/ --include="*.py"` — every hit must have a forge equivalent
- [ ] Run `grep -r "from agents_api.dsw.bayesian" agents_api/ --include="*.py"` — same
- [ ] Verify `forgestat` is importable and at expected version
- [ ] Spot-check 5 analysis types end-to-end via API before deletion

### Safety-net tests (10)

Per gap analysis §6.3 — one per analysis family:
1. Parametric: t-test → p_value, effect_size, CI present
2. Nonparametric: Mann-Whitney → U statistic, p_value present
3. Regression: OLS → coefficients, R², residuals present
4. ANOVA: one-way → F statistic, p_value, group means present
5. Post-hoc: Tukey HSD → pairwise comparisons present
6. Advanced: mixed effects → fixed effects, random effects, ICC present
7. Exploratory: PCA → loadings, explained_variance present
8. Bayesian: posterior → credible interval, Bayes factor present
9. Reliability: Weibull → shape, scale, MTBF present
10. ML pipeline: random forest → feature importance, cross-val score present

### LOC removed

**~50,000-55,000 LOC** — the single largest LOC reduction in the entire extraction.

### Note

This is the largest Phase 0 CR by file count. Consider splitting into 2 CRs if review burden is too high:
- CR-0.3a: Delete `dsw/stats_*` family (parametric, nonparametric, posthoc, regression, advanced, exploratory, quality)
- CR-0.3b: Delete `dsw/bayesian/*`, `dsw/ml.py`, `dsw/viz.py`, `dsw/siop.py`, `dsw/simulation.py`, `dsw/reliability.py`, `dsw/exploratory/*`, remaining files

---

<a id="cr-04"></a>
## CR-0.4 — Wire forgesia, delete agents_api/synara/

**Change type:** `enhancement`
**Risk:** MED (2-app CR: agents_api + graph)
**Dependencies:** none (independent of CR-0.1/0.2/0.3)
**CR partners:** `graph/` (3 files update imports)
**Blast radius:** ~10 files across 2 apps

### What

Per gap analysis §6.4. Wire `forgesia` package into `synara_views.py` and `graph/` modules. Delete `agents_api/synara/` engine (6 files, ~3,243 LOC).

**Known blocker:** `forgesia/__init__.py` exports need fixing first (per migration plan Tech Debt). This must be resolved as part of this CR.

**Delete:**
- `agents_api/synara/` directory (6 files, ~3,243 LOC)

**Wire:**

| File | Current import | New import |
|------|---------------|------------|
| `synara_views.py` | `from agents_api.synara.belief import BeliefEngine` | `from forgesia import BeliefEngine` |
| `synara_views.py` | `from agents_api.synara.kernel import ...` | `from forgesia.kernel import ...` |
| `synara_views.py` | `from agents_api.synara.dsl import ...` | `from forgesia.dsl import ...` |
| `graph/synara_adapter.py:14-15` | `from agents_api.synara.belief import BeliefEngine` | `from forgesia import BeliefEngine` |
| `graph/synara_adapter.py` | `from agents_api.synara.kernel import ...` | `from forgesia.kernel import ...` |
| `graph/service.py:281` | `from agents_api.synara.kernel import Evidence as SynaraEvidence` | `from forgesia import Evidence as SynaraEvidence` |
| `graph/tests_synara.py:11` | `from agents_api.synara.kernel import Evidence` | `from forgesia import Evidence` |

### Pre-CR checklist

- [ ] Fix `forgesia/__init__.py` exports — verify `BeliefEngine`, `Evidence`, `Kernel` all importable from top-level
- [ ] Run existing `graph/tests_synara.py` to establish baseline
- [ ] Verify `forgesia` DSL parser handles all patterns used by `synara_views.py`

### Safety-net tests (5)

Per gap analysis §6.4:
1. `BeliefEngine.update_belief()` with evidence → correct posterior (within 0.01 of expected)
2. DSL parse + evaluate → expected kernel state (round-trip)
3. Synara API endpoint `/api/synara/beliefs/` → returns belief state JSON with expected schema
4. `graph/synara_adapter` → belief update propagates to graph entity correctly
5. `graph/service` → evidence creation via forgesia works end-to-end

### LOC removed

~3,243 LOC.

---

<a id="cr-05"></a>
## CR-0.5 — Wire forgeviz completion

**Change type:** `enhancement`
**Risk:** LOW
**Dependencies:** CR-0.1 (chart files now in analysis/)
**CR partners:** none
**Blast radius:** ~5 files

### What

Per gap analysis §6.5. After CR-0.1 moves chart files to `analysis/`, ensure any remaining inline Plotly chart construction goes through `forgeviz` helpers. If `analysis/chart_render.py` and `analysis/chart_defaults.py` still build Plotly traces inline, wire them through `forgeviz`.

**Wire:** Replace inline `plotly.graph_objects` calls with `forgeviz.charts.*` calls where the forgeviz API covers the chart type.

**Note:** This is a "best effort" wiring CR. Some chart types may not have forgeviz equivalents yet. Those stay as-is and get wired when forgeviz adds coverage. The goal is to maximize forgeviz usage, not achieve 100%.

### Safety-net tests (4)

1. Control chart render → ChartSpec dict with `data`, `layout`, expected trace count
2. Capability histogram → correct bin edges, spec limit lines present
3. Residuals plot → expected axes and trace structure
4. Generic bar/line chart → valid Plotly-compatible JSON

### LOC removed

Minimal — this is a wiring CR, not a deletion CR. Some inline Plotly code may be replaceable.

---

<a id="cr-06"></a>
## CR-0.6 — Dead-code deletion

**Change type:** `debt`
**Risk:** LOW-MED
**Dependencies:** none (independent of forge wiring)
**CR partners:** `fmea_views.py`, `rca_views.py`, `a3_views.py`, `hoshin_views.py` (ActionItem ref updates); `forge/tasks.py:152` (get_shared_llm move)
**Blast radius:** ~15 files

### What

Per gap analysis §6.6. Delete confirmed dead code and convert ActionItem references to `loop.Commitment`.

**Delete outright:**

| Item | LOC | Rationale |
|------|-----|-----------|
| `agents_api/views.py` (except `get_shared_llm`) | ~430 | 5 agent dispatchers (researcher, writer, editor, experimenter, EDA) — per inventory §H.1, likely unused |
| `agents_api/urls.py` agent dispatch routes | ~16 | Dead routes for agent dispatch |
| `Workflow` model + `workflow_views.py` + `workflow_urls.py` | ~450 | Zero non-self FKs (inventory §H.5) |
| `ActionItem` model + `action_views.py` + `action_urls.py` | ~90 | Superseded by `loop.Commitment` (architecture §9.A #4) |
| `AuditChecklist` model | ~40 | Superseded by `Checklist` (inventory §I.3 + H.6) — verify no live tenant data first |
| Sweep test files (TST-001 §10.6 violations) | ~3,000 | `test_endpoint_smoke.py` (712), `test_t1_deep.py` (1,942), `test_t2_views_smoke.py` (338), `test_*_coverage.py` family (8 files) |

**Extract and relocate:**
- `get_shared_llm` helper from `agents_api/views.py` → new `llm/helpers.py` or `chat/helpers.py`
- Update `forge/tasks.py:152` to import from new location

**Convert ActionItem → Commitment:**

5 view files reference ActionItem. Each must be updated to use `loop.Commitment`:
- `fmea_views.py` — action item creation from FMEA findings
- `rca_views.py` — action item creation from RCA
- `a3_views.py` — action item list/create
- `hoshin_views.py` — action item list/create
- `action_views.py` — deleted entirely

### Pre-CR checklist

- [ ] `grep -r "Workflow" agents_api/ --include="*.py"` — confirm no live usage beyond model/view/url
- [ ] `grep -r "ActionItem" agents_api/ --include="*.py"` — map all 5 view references
- [ ] `grep -r "AuditChecklist" agents_api/ --include="*.py"` — confirm superseded; check for live tenant data: `python manage.py shell -c "from agents_api.models import AuditChecklist; print(AuditChecklist.objects.count())"`
- [ ] `grep -r "get_shared_llm" . --include="*.py"` — map all importers
- [ ] Verify `loop.Commitment` model exists and has equivalent fields for action tracking

### Safety-net tests (5)

1. Create commitment from FMEA finding → Commitment created with correct source linkage
2. Create commitment from RCA → same
3. Create commitment from A3 → same
4. `forge/tasks.py` imports `get_shared_llm` from new location → no import error
5. `/api/agents/` route returns 404 (deleted) — NOT 500

### LOC removed

~4,000+ LOC deleted. `get_shared_llm` (~10 LOC) relocated.

### Note on RackSession

Per Eric's decision (2026-04-10): **ForgeRack is OUT OF SCOPE.** `RackSession` stays in `agents_api/models.py`. `rack_views.py` stays mounted. No action in this CR or any Phase 0 CR.

---

<a id="cr-07"></a>
## CR-0.7 — permissions.py → qms_core/ shim

**Change type:** `enhancement`
**Risk:** LOW (coordinates 3-4 apps)
**Dependencies:** none
**CR partners:** `loop/supplier_views.py`, `notifications/webhook_views.py`, `syn/audit/compliance.py`
**Blast radius:** ~8 files across 4 apps

### What

Per gap analysis §6.7. Create the `qms_core/` Django app (initially empty models, just permissions) and move `agents_api/permissions.py` there. This is a prerequisite for every Phase 1A+ extraction because all `qms_*` view files will import permissions from the new location.

### Steps

1. **Create `qms_core/` Django app:**
   ```
   services/svend/web/qms_core/
   ├── __init__.py
   ├── apps.py          # QmsCoreConfig
   ├── models.py         # empty for now — models arrive at Phase 4
   ├── permissions.py    # moved from agents_api/permissions.py
   └── migrations/
       └── 0001_initial.py  # empty
   ```

2. **Add `qms_core` to `INSTALLED_APPS`** in `svend/settings.py`.

3. **Move `agents_api/permissions.py` contents** to `qms_core/permissions.py`.

4. **Create `agents_api/permissions.py` shim** that re-exports from `qms_core.permissions`:
   ```python
   """Shim — re-exports from qms_core.permissions during extraction."""
   from qms_core.permissions import *  # noqa: F401,F403
   ```
   This allows all existing `from agents_api.permissions import ...` to keep working. The shim is deleted at Phase 3 cutover.

5. **Update external importers to import from `qms_core.permissions` directly:**
   - `loop/supplier_views.py` — `from agents_api.permissions import get_tenant` → `from qms_core.permissions import get_tenant`
   - `notifications/webhook_views.py` (×2) — same
   - `syn/audit/compliance.py` — if applicable

### Safety-net tests (3)

1. `qms_can_edit(request, site)` returns correct permission for site admin vs regular user
2. `qms_queryset(model, request)` filters to user's tenant
3. `get_tenant(request)` returns current user's tenant object

### LOC removed

0 — this is a relocation CR. agents_api/permissions.py becomes a thin shim.

---

<a id="cr-08"></a>
## CR-0.8 — iso_views.py split-in-place

**Change type:** `enhancement`
**Risk:** MED (4,874 LOC across 85 functions, but no model changes)
**Dependencies:** none (independent of forge wiring)
**CR partners:** none (within agents_api/)
**Blast radius:** ~15 files (1 giant file → 12+ small files + URL rewiring)

### What

Per architecture §9.A #15 + §7.6 + gap analysis §3.3 + §6.8. Split `iso_views.py` (4,874 LOC, 85 view functions spanning 11 ISO clause areas) into sub-modules within `agents_api/`. This is a **within-app file reorganization** — no model changes, no URL changes for users, no behavior changes.

After the split, each sub-module is a self-contained file that Phase 2A extractions can move individually to their target `qms_*` apps.

### Target file structure

```
agents_api/iso/
├── __init__.py              # Re-export all views for URL compatibility
├── ncr_views.py             # §10.2 NCR functions (ncr_*)
├── capa_views.py            # §10.2 CAPA portion (capa_*) — DELETE target at Phase 3
├── audit_views.py           # §9.2 Internal audit (audit_*, audit_finding_*, audit_clause_coverage, audit_apply_checklist, audit_checklist_*)
├── training_views.py        # §7.2 Training (training_*)
├── management_review_views.py  # §9.3 Management review (review_*, review_template_*, review_narrative)
├── document_views.py        # §7.5 Document control (document_*)
├── supplier_views.py        # §8.4 Supplier (supplier_*)
├── equipment_views.py       # §7.1.5 Calibration (equipment_*, gage R&R links)
├── complaint_views.py       # §9.1.2 Customer complaints (complaint_*)
├── risk_views.py            # §6.1 Risk register (risk_*)
├── afe_views.py             # AFE + approval chain (afe_*)
├── control_plan_views.py    # Control Plan (control_plan_*)
└── checklist_views.py       # Checklist execution (checklist_*)
```

### Steps

1. Create `agents_api/iso/` package with `__init__.py`
2. For each ISO clause area, extract the matching functions from `iso_views.py` into the target sub-module
3. Each sub-module gets the shared imports at the top (Django, models, permissions, etc.)
4. `agents_api/iso/__init__.py` re-exports all view functions so existing `from agents_api.iso_views import ...` patterns in URL files still resolve
5. Update `iso_urls.py` to import from `agents_api.iso.*` sub-modules instead of `agents_api.iso_views`
6. Delete original `agents_api/iso_views.py`

### Shared imports to replicate per sub-module

From scanning the gap analysis §3.3, all sub-modules will need:
- `from django.http import JsonResponse`
- `from django.views.decorators.csrf import csrf_exempt`
- `from accounts.permissions import require_auth, gated_paid` (or `gated`)
- `from qms_core.permissions import qms_can_edit, qms_queryset, qms_set_ownership` (after CR-0.7)

Model imports are per-sub-module based on the models they touch.

### Regression suite

**Use the existing `iso_tests.py` (351 tests) as the regression suite.** All 351 tests must pass after the split with zero behavior change. This is the validation gate — if any test fails, the split introduced a bug.

### Safety-net tests

No new tests needed — the existing 351-test suite IS the safety net. If it doesn't cover a function, that's a gap for the Phase 2A extraction to address, not this split.

### LOC removed

0 — this is a reorganization, not a deletion. Same 4,874 LOC, now in 13 files instead of 1.

### Risk mitigation

- The `__init__.py` re-export layer ensures backwards compatibility during the parallel period
- If the split introduces import ordering issues, each sub-module's imports are self-contained (no cross-sub-module imports)
- Django's URL resolution doesn't care about file location — only the callable reference matters

---

<a id="cr-09"></a>
## CR-0.9 — Create tools/ Django app (formalized tool router)

**Change type:** `enhancement`
**Risk:** MED (load-bearing cross-tool layer, 17 importing files across 4 apps)
**Dependencies:** CR-0.7 (qms_core/ must exist for permissions import pattern precedent; not a hard dependency but follows the same shim pattern)
**CR partners:** `loop/evaluator.py`, `svend/urls.py`, 7 view files (a3, rca, fmea, vsm, spc, hoshin, whiteboard)
**Blast radius:** ~22 files across 4 apps

### What

Per Eric's directive (2026-04-10): "Formalize a tool router system — something we can expand and modularize." Create a new `tools/` Django app that houses the tool router, event bus, base model, and registration system. This is not a simple relocation — it's an opportunity to make the system explicitly expandable.

### Current state (913 LOC across 5 files)

| File | LOC | What it does |
|------|-----|-------------|
| `tool_router.py` | 224 | Singleton URL pattern generator with permission gating |
| `tool_events.py` | 103 | Pub/sub event bus with wildcard patterns |
| `tool_event_handlers.py` | 183 | 8 cross-cutting handlers (evidence, logging) |
| `tool_registry.py` | 165 | Registration of 6 tools at `apps.ready()` |
| `base_tool_model.py` | 238 | Abstract base with UUID PK, ownership, status FSM |

### Dependency surface (17 files across 4 apps)

**Emitters (7 view files):** `a3_views.py`, `rca_views.py`, `fmea_views.py`, `vsm_views.py`, `spc_views.py`, `hoshin_views.py`, `whiteboard_views.py` — all `from agents_api.tool_events import tool_events`

**URL integration (1 file):** `svend/urls.py` — calls `_get_tool_router_urls()` which defers to `tool_registry.register_tools()` + `ToolRouter.get_urlpatterns()`

**Event subscribers (1 file):** `loop/evaluator.py` — `from agents_api.tool_events import tool_events; tool_events.subscribe(...)`

**Initialization (1 file):** `agents_api/apps.py` — `import tool_event_handlers` at `AppConfig.ready()`

**Tests (3 files):** `test_tool_router.py`, `test_tool_events.py`, `test_base_tool_model.py`

### Target file structure

```
services/svend/web/tools/
├── __init__.py
├── apps.py                  # ToolsConfig — initializes handlers + registry at ready()
├── models.py                # Re-export ToolModel for Django model discovery
├── base_tool_model.py       # Abstract base (from agents_api/base_tool_model.py)
├── router.py                # ToolRouter class (from agents_api/tool_router.py)
├── events.py                # ToolEventBus + tool_events singleton (from agents_api/tool_events.py)
├── handlers.py              # Cross-cutting event handlers (from agents_api/tool_event_handlers.py)
├── registry.py              # Tool registration (from agents_api/tool_registry.py)
├── migrations/
│   └── 0001_initial.py      # Empty (ToolModel is abstract)
└── tests/
    ├── test_router.py
    ├── test_events.py
    └── test_base_model.py
```

### Changes for expandability

While relocating, make these specific improvements to support Eric's "expand and modularize" directive:

1. **Self-registration pattern.** Each tool app registers itself in its own `apps.py` instead of all tools being listed in a central `tool_registry.py`. The `tools/` app provides a `register()` function; tool apps call it at their `AppConfig.ready()`. This means adding a new tool doesn't require editing `tools/registry.py`.

   ```python
   # In tools/registry.py — public API
   def register_tool(slug, **kwargs):
       """Called by each tool's AppConfig.ready() to self-register."""
       ToolRouter.register(slug=slug, **kwargs)
   ```

   ```python
   # In qms_investigation/apps.py (example — future Phase 2A)
   class QmsInvestigationConfig(AppConfig):
       def ready(self):
           from tools.registry import register_tool
           from . import views
           register_tool(slug="rca", model=..., list_view=views.list_sessions, ...)
   ```

2. **Transitional registry.** During the extraction, `tools/registry.py` retains the current 6-tool registration as a bridge. As each tool app extracts (Phase 1A/2A), its registration migrates from `tools/registry.py` to its own `apps.py`. By Phase 3 cutover, `tools/registry.py` should be empty (or contain only ForgeRack if it's still in agents_api).

3. **Event handler registration follows the same pattern.** Each domain app registers its own handlers at `ready()` instead of `tool_event_handlers.py` knowing about all domains. The cross-cutting handlers (wildcard `*.created`, `*.updated` for project timeline logging) stay in `tools/handlers.py`. Domain-specific handlers (A3 evidence, RCA evidence, FMEA evidence) move with their domain app at extraction time.

4. **`base_tool_model.py` Site FK update.** Currently `site = models.ForeignKey("agents_api.Site", ...)`. After this CR: `site = models.ForeignKey("agents_api.Site", ...)` (unchanged — Site doesn't move until Phase 4). At Phase 4, this becomes `models.ForeignKey("qms_core.Site", ...)`. Document this deferred update in the file.

### Steps

1. Create `tools/` Django app with structure above
2. Move 5 files from `agents_api/` → `tools/`, with renames:
   - `tool_router.py` → `tools/router.py`
   - `tool_events.py` → `tools/events.py`
   - `tool_event_handlers.py` → `tools/handlers.py`
   - `tool_registry.py` → `tools/registry.py`
   - `base_tool_model.py` → `tools/base_tool_model.py`
3. Create shims in `agents_api/` for backwards compatibility:
   ```python
   # agents_api/tool_events.py (shim)
   """Shim — re-exports from tools.events during extraction."""
   from tools.events import *  # noqa: F401,F403
   from tools.events import tool_events  # noqa: F401 — explicit re-export of singleton
   ```
   Same pattern for `tool_router.py`, `tool_registry.py`, `tool_event_handlers.py`, `base_tool_model.py`.
4. Update direct importers to use new paths where practical:
   - `svend/urls.py` — update `_get_tool_router_urls()` to import from `tools.registry`
   - `loop/evaluator.py` — `from tools.events import tool_events`
   - `agents_api/apps.py` — remove handler import (moved to `tools/apps.py`)
5. Add `tools` to `INSTALLED_APPS` in `svend/settings.py`
6. Move test files to `tools/tests/`

### Safety-net tests (5)

1. `ToolRouter.register()` + `get_urlpatterns()` → generates expected URL count for 6 tools
2. `tool_events.emit("fmea.row_updated", ...)` → handler fires (evidence bridge called)
3. `tool_events.emit("a3.created", ...)` → wildcard `*.created` handler fires (project timeline logged)
4. `ToolModel` subclass validates transitions correctly (draft → active allowed, complete → draft rejected)
5. `loop/evaluator.py` subscribes and receives events correctly from `tools.events`

### LOC removed

0 — relocation + shims. Shims removed at Phase 3 cutover.

### Note on expandability direction

The self-registration pattern is the key design change. Today, adding a tool requires editing `tool_registry.py` (centralized). After this CR, adding a tool only requires the tool's own `apps.py` to call `register_tool()` (decentralized). This scales to the 25-app topology without a central bottleneck. The event handler decentralization follows the same principle — each domain owns its own cross-cutting integration code.

---

<a id="cr-010"></a>
## CR-0.10 — Phase 0 verification + LOC census

**Change type:** `documentation`
**Risk:** LOW
**Dependencies:** CR-0.1 through CR-0.9 (all of Phase 0)
**CR partners:** none
**Blast radius:** 1 file (documentation)

### What

Final verification that Phase 0 achieved its goals. Run LOC census, verify no broken imports, run full test suite, update planning artifacts.

### Steps

1. **LOC census:**
   ```bash
   find agents_api/ -name "*.py" -not -path "*/__pycache__/*" | xargs wc -l
   # Compare to pre-Phase-0 baseline (289,313 LOC)
   ```

2. **Import verification:**
   ```bash
   python manage.py check --deploy
   python -c "from agents_api import models; print('models OK')"
   python -c "from tools.events import tool_events; print('tools OK')"
   python -c "from qms_core.permissions import qms_can_edit; print('qms_core OK')"
   ```

3. **Full test suite:**
   ```bash
   python -m pytest tests/ -x --tb=short
   ```

4. **Update planning artifacts:**
   - Mark Phase 0 complete in `extraction_gap_analysis.md`
   - Update LOC numbers in project memory
   - Report Phase 0 results to Eric

### Expected LOC reduction

| Phase 0 CR | Est. LOC removed |
|------------|-----------------|
| CR-0.1 | 0 (relocated) |
| CR-0.2 | ~1,889 |
| CR-0.3 | ~50,000-55,000 |
| CR-0.4 | ~3,243 |
| CR-0.5 | minimal |
| CR-0.6 | ~4,000 |
| CR-0.7 | 0 (relocated) |
| CR-0.8 | 0 (reorganized) |
| CR-0.9 | 0 (relocated) |
| **TOTAL** | **~59,000-64,000 LOC** |

Post-Phase-0 `agents_api/` should be ~225,000-230,000 LOC, with clean forge boundaries, a standalone `tools/` app, a `qms_core/` permissions foundation, and `iso_views.py` split into 13 independently-extractable sub-modules.

---

<a id="dependency-graph"></a>
## Dependency graph

```
CR-0.1 (dsw→analysis move)
  ├──→ CR-0.2 (forgespc)
  ├──→ CR-0.3 (forgestat — largest deletion)
  └──→ CR-0.5 (forgeviz)

CR-0.4 (forgesia)        ← independent of 0.1
CR-0.6 (dead code)       ← independent of 0.1
CR-0.7 (qms_core perms)  ← independent of 0.1
CR-0.8 (iso split)       ← independent of 0.1
CR-0.9 (tools/ app)      ← independent of 0.1

CR-0.10 (verification)   ← depends on ALL above
```

**Parallelizable groups:**
- **Group A:** CR-0.1 → CR-0.2/0.3/0.5 (serial chain — dsw move first, then forge wiring)
- **Group B:** CR-0.4 (forgesia — independent)
- **Group C:** CR-0.6 (dead code — independent)
- **Group D:** CR-0.7 + CR-0.9 (infrastructure — independent of forge wiring)
- **Group E:** CR-0.8 (iso split — independent)

Groups B, C, D, E can all run in parallel with Group A. The critical path is Group A (0.1 → 0.3), because CR-0.3 is the largest single CR.

---

<a id="risk-summary"></a>
## Risk summary

| CR | Risk | Change type | Key risk factor |
|----|------|-------------|-----------------|
| CR-0.1 | LOW | enhancement | Within-app file move; behavioral tests exist |
| CR-0.2 | MED | enhancement | forgespc API parity — must verify before deletion |
| CR-0.3 | MED | debt | ~50k LOC deletion — largest blast radius. Consider splitting. |
| CR-0.4 | MED | enhancement | 2-app CR (agents_api + graph); forgesia __init__ blocker |
| CR-0.5 | LOW | enhancement | Best-effort wiring, no deletions gated on it |
| CR-0.6 | LOW-MED | debt | ActionItem→Commitment conversion touches 5 view files |
| CR-0.7 | LOW | enhancement | Shim pattern is proven; 3-4 app import updates |
| CR-0.8 | MED | enhancement | 4,874 LOC split — high review burden, but 351-test regression suite validates |
| CR-0.9 | MED | enhancement | Load-bearing layer, 17 importing files. Shim pattern mitigates risk. |
| CR-0.10 | LOW | documentation | Verification only |

**No HIGH or CRITICAL risk CRs in Phase 0.** The highest risk items (forgespc parity, forgesia blocker, iso_views split) are all MED and have clear validation strategies (test suites, pre-CR checklists).

---

## Appendix — Phase 0 → Phase 1A handoff

After Phase 0 completes (all 10 CRs landed + verification), the codebase is ready for Phase 1A leaf extractions. The specific preconditions met:

1. **`analysis/` is canonical** — all live compute code lives there, not in `dsw/`
2. **Forge packages are wired** — no inline statistical compute remains in agents_api
3. **Dead code is gone** — Workflow, ActionItem, agent dispatchers, sweep tests all deleted
4. **`qms_core/` exists** — permissions available from the new location for all future qms_* apps
5. **`tools/` exists** — tool router, event bus, and base model in their permanent home with self-registration pattern
6. **`iso_views.py` is split** — 13 independently-extractable sub-modules ready for Phase 2A
7. **Test baseline is clean** — all existing tests pass post-Phase-0

Phase 1A begins with the 7 leaf extractions (triage, whiteboard, vsm, simulation, qms_measurement, qms_suppliers, learn) per `extraction_sequence.md`.
