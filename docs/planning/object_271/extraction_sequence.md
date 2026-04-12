# agents_api Extraction — Sequenced Execution Plan

**Status:** DRAFT — planning artifact under CR `5bf7354c-3de5-4624-b505-a94a5b6ce0ea`
**Date:** 2026-04-10
**Author:** Claude (Systems Engineer role per Object 271)
**Inputs:**
- `docs/planning/object_271/qms_architecture.md` (locked v0.4) — target state
- `docs/planning/object_271/extraction_gap_analysis.md` — current-to-target mapping
- `docs/planning/object_271/agents_api_inventory.md` — current state inventory
- `~/.claude/projects/-home-eric/memory/project_agents_api_extraction.md` — extraction state

**Purpose:** Chronological CR-by-CR execution plan for decomposing `agents_api/` (289,313 LOC, 68 models) into 25 properly-scoped Django apps. Every CR has an ID, title, phase, change type, risk tier, contents, dependencies, CR partners, test count, blast radius, and review gates. This is the execution backbone that the team works from.

**What this document is NOT:**
- Not a gap analysis (that is `extraction_gap_analysis.md`, already complete)
- Not architecture decisions (those are locked in `qms_architecture.md` v0.4)
- Not a test plan (that is `test_suite_rebuild.md`, pending)

---

## Section 1 — Dependency Graph

Text-based dependency graph showing which CRs block which. Read left-to-right as "must land before."

```
Phase 0 (foundation cleanup — sequential chain with some parallelism)
=========================================================================

CR-0.1 (dsw→analysis consolidation)
  ├──→ CR-0.2 (forgespc wiring)
  ├──→ CR-0.3 (forgestat wiring)
  └──→ CR-0.5 (forgeviz completion)

CR-0.2 (forgespc) ──→ CR-0.3 (forgestat)  [forgestat depends on spc cleanup]

CR-0.4 (forgesia wiring) ── independent of 0.1-0.3, depends only on forgesia __init__.py fix

CR-0.6 (dead-code deletion) ── independent of 0.1-0.5 (can parallelize)

CR-0.7 (permissions.py → qms_core/ shim)
  └──→ ALL Phase 1A/2A extractions (every qms_* view imports permissions)

CR-0.8 (iso_views.py split-in-place)
  └──→ ALL Phase 2A qms_* extractions that consume iso_views slices

CR-0.9 (tools/ app creation)
  └──→ Phase 2A extractions that import tool_router/tool_registry/tool_events


Phase 1A (leaf relocations — parallelizable after Phase 0 completes)
=========================================================================

CR-1A.1 (triage/)         ── depends on: CR-0.7
CR-1A.2 (whiteboard/)     ── depends on: CR-0.7
CR-1A.3 (learn/)          ── depends on: CR-0.7
CR-1A.4 (vsm/)            ── depends on: CR-0.7, CR-0.8
CR-1A.5 (simulation/)     ── depends on: CR-0.7, CR-1A.4 (source_vsm FK)
CR-1A.6 (qms_measurement/)── depends on: CR-0.7, CR-0.8
CR-1A.7 (qms_suppliers/)  ── depends on: CR-0.7, CR-0.8

Parallelism: CR-1A.1, CR-1A.2, CR-1A.3 can land in any order (no cross-deps).
             CR-1A.4 before CR-1A.5 (VSM before simulation).
             CR-1A.6 and CR-1A.7 depend on CR-0.8 (iso_views split).


Phase 1B (leaf rebuilds — sequential per app, parallel across apps)
=========================================================================

CR-1B.1 (workbench pull contract + DSWResult conversion)
  └──→ ALL Phase 2B pull-contract consumers (hoshin, a3, reports, sop)
  └──→ CR-1B.1a (notebook DSWResult refs conversion in core/)

CR-1B.2 (triage/ pull contract)     ── depends on: CR-1A.1
CR-1B.3 (whiteboard/ pull contract)  ── depends on: CR-1A.2
CR-1B.4 (learn/ sv-* rebuild)       ── depends on: CR-1A.3
CR-1B.5 (vsm/ cockpit rebuild)      ── depends on: CR-1A.4
CR-1B.6 (simulation/ rebuild)       ── depends on: CR-1A.5
CR-1B.7 (qms_measurement/ pull)     ── depends on: CR-1A.6
CR-1B.8 (qms_suppliers/ pull)       ── depends on: CR-1A.7

CR-1B.1 (workbench) is the CRITICAL PATH item in Phase 1B.


═══════════════════════════════════════════════════════════
  *** ERIC REVIEW GATE 1 — between Phase 1B and Phase 2A ***
═══════════════════════════════════════════════════════════


Phase 2A (medium-coupling relocations — ordering constrained)
=========================================================================

CR-2A.1 (qms_risk/)           ── depends on: CR-0.7, CR-0.8, CR-0.9
CR-2A.2 (qms_documents/)      ── depends on: CR-0.7, CR-0.8, CR-0.9
CR-2A.3 (qms_training/)       ── depends on: CR-0.7, CR-0.8, CR-2A.2 (TrainingReq→ControlledDoc FK)
CR-2A.4 (qms_nonconformance/) ── depends on: CR-0.7, CR-0.8
CR-2A.5 (qms_audit/)          ── depends on: CR-0.7, CR-0.8, CR-2A.4 (AuditFinding.ncr FK)
CR-2A.6 (qms_investigation/)  ── depends on: CR-0.7, CR-0.9
CR-2A.7 (qms_a3/)             ── depends on: CR-2A.6 (A3.rca_session FK)
CR-2A.8 (hoshin/)             ── depends on: CR-0.7, CR-0.8, CR-1A.4 (vsm FK)
CR-2A.9 (notebook_views→core/)── depends on: CR-1B.1 (DSWResult conversion done)

Ordering constraints:
  CR-2A.2 before CR-2A.3 (training imports ControlledDocument)
  CR-2A.4 before CR-2A.5 (audit finding FK to NCR)
  CR-2A.6 before CR-2A.7 (A3 FK to RCASession)


Phase 2B (medium-coupling rebuilds — parallel within constraints)
=========================================================================

CR-2B.1 (qms_risk/ pull contract)           ── depends on: CR-2A.1
CR-2B.2 (qms_documents/ pull + ForgeDoc)     ── depends on: CR-2A.2
CR-2B.3 (qms_training/ pull contract)        ── depends on: CR-2A.3
CR-2B.4 (qms_nonconformance/ pull contract)  ── depends on: CR-2A.4
CR-2B.5 (qms_audit/ pull + audit upgrade)    ── depends on: CR-2A.5, CR-2B.4 (NCR pull ready)
CR-2B.6 (qms_investigation/ pull transition) ── depends on: CR-2A.6, CR-1B.1 (workbench pull)
CR-2B.7 (qms_a3/ full rebuild)              ── depends on: CR-2A.7, CR-2B.6 (investigation pull ready)
CR-2B.8 (hoshin/ HoshinKPI pull conversion)  ── depends on: CR-2A.8, CR-1B.1 (workbench pull)
CR-2B.9 (reports/ new sink)                  ── depends on: CR-2B.7 (A3 pull ready)
CR-2B.10 (sop/ new sink)                     ── depends on: CR-2B.6 (investigation pull ready)
CR-2B.11 (CAPA delete+replace)               ── depends on: CR-2B.6, CR-2B.9 (replacements ready)


═══════════════════════════════════════════════════════════
  *** ERIC REVIEW GATE 2 — between Phase 2B and Phase 3 ***
═══════════════════════════════════════════════════════════


Phase 3 (cutover)
=========================================================================

CR-3.1 (single-night URL swap + legacy deletion)
  ── depends on: ALL Phase 1B and 2B CRs complete + Eric gate 2 sign-off


Phase 4 (Site final move)
=========================================================================

CR-4.1 (Site + qms_core/ atomic migration)
  ── depends on: CR-3.1 (cutover complete, all apps running at final URLs)
```

---

## Section 2 — Phase 0 CRs (Foundation Cleanup)

Phase 0 reduces `agents_api/` from 289,313 LOC to roughly 220,000 LOC by wiring forge packages and deleting dead code. Every Phase 1+ extraction benefits from this reduced surface. Phase 0 is a mix of `enhancement`, `debt`, and `migration` change types.

---

### CR-0.1 — Consolidate dsw/ live files into analysis/

| Field | Value |
|---|---|
| **Phase** | 0 |
| **Change type** | `debt` |
| **Risk tier** | LOW |
| **What moves** | `dsw/common.py` (3,084 LOC), `dsw/endpoints_data.py` (1,832 LOC), `dsw/endpoints_ml.py` (1,702 LOC), `dsw/standardize.py` (552 LOC), `dsw/chart_render.py` (59 LOC), `dsw/chart_defaults.py` (459 LOC) — all move into `analysis/`. Delete `analysis/chart_render.py` (7 LOC wrapper) and `analysis/chart_defaults.py` (11 LOC wrapper). Per gap analysis §6.1, architecture §9.A J.11, inventory §K.2. |
| **Dependencies** | None — first CR in the sequence. |
| **CR partners** | None (within-app reorganization). |
| **Safety-net tests** | ~5 tests: "upload CSV via dsw_views → correct response shape", "chart render → expected ChartSpec dict", "standardize post-process → evidence_grade present", "endpoints_data helper returns correct structure", "endpoints_ml helper returns model spec". Per gap analysis §6.1. |
| **Blast radius** | ~15 files touched. `spc_views.py`, `autopilot_views.py`, `forecast_views.py`, `pbs_engine.py`, `ml_pipeline.py`, `dsw_views.py`, `report_views.py`, `a3_views.py` — all update imports from `agents_api.dsw.*` to `agents_api.analysis.*`. |

---

### CR-0.2 — Wire forgespc; delete agents_api/spc.py + dsw/spc*

| Field | Value |
|---|---|
| **Phase** | 0 |
| **Change type** | `debt` |
| **Risk tier** | MED |
| **What moves** | Wire `forgespc` package into `spc_views.py` (1,547 LOC, 15 funcs). Replace inline imports from `agents_api.spc` (1,889 LOC) and `dsw/spc.py` + `dsw/spc_pkg/*` (~10,000 LOC) with forgespc calls. Delete `agents_api/spc.py`, `dsw/spc.py`, `dsw/spc_pkg/`. Per gap analysis §6.2, migration plan Tech Debt. |
| **Dependencies** | CR-0.1 (chart helpers consolidated into analysis/ first). |
| **CR partners** | None. |
| **Safety-net tests** | ~7 tests — one per major SPC analysis type: xbar-R, xbar-S, IMR, c-chart, p-chart, capability study (Cpk), gage R&R. Each asserts 10-key result contract populates correctly from forgespc. Per gap analysis §6.2. |
| **Blast radius** | ~5 files touched. `spc_views.py` (main), `dsw_views.py` (any spc dispatch), plus 3 deleted files/directories. |

---

### CR-0.3 — Wire forgestat; delete legacy dsw/stats* and related files

| Field | Value |
|---|---|
| **Phase** | 0 |
| **Change type** | `debt` |
| **Risk tier** | MED |
| **What moves** | Wire `forgestat` into `analysis/forge_*.py` bridges for remaining handlers. Delete ~50,000-55,000 LOC of legacy files: `dsw/stats_parametric.py`, `stats_nonparametric.py`, `stats_posthoc.py`, `stats_regression.py`, `stats_advanced.py`, `stats_exploratory.py`, `stats_quality.py`, `bayesian.py`, `bayesian/*`, `ml.py`, `viz.py`, `siop.py`, `simulation.py`, `reliability.py`, `d_type.py`, `exploratory/*`. Per gap analysis §6.3, inventory §H.2. |
| **Dependencies** | CR-0.2 (forgespc wiring complete — SPC stats now come from forgespc, clearing dsw/stats_quality overlap). |
| **CR partners** | None. |
| **Safety-net tests** | ~8-10 tests — one per analysis family: parametric (t-test), nonparametric (Mann-Whitney), regression (OLS), ANOVA, posthoc (Tukey), advanced (mixed effects), exploratory (PCA), Bayesian (posterior update), ML (model fit). Per gap analysis §6.3. |
| **Blast radius** | ~25 files deleted, ~5 view files updated. `dsw_views.py`, `autopilot_views.py`, `forecast_views.py`, any view still calling into `dsw/stats_*`. |

---

### CR-0.4 — Wire forgesia; delete agents_api/synara/

| Field | Value |
|---|---|
| **Phase** | 0 |
| **Change type** | `debt` |
| **Risk tier** | MED |
| **What moves** | Fix `forgesia/__init__.py` exports (known blocker per migration plan Tech Debt). Wire into `synara_views.py` (1,136 LOC, 32 funcs). Update `graph/synara_adapter.py`, `graph/service.py`, `graph/tests_synara.py` to import from `forgesia` instead of `agents_api.synara.*`. Delete `agents_api/synara/` (6 files, 3,243 LOC). Per gap analysis §6.4, §5.2. |
| **Dependencies** | None from CR-0.1-0.3 (independent chain). Depends on forgesia `__init__.py` fix being completed. |
| **CR partners** | `graph/` — 2-app coordinated CR. `graph/synara_adapter.py:14-15`, `graph/service.py:281`, `graph/tests_synara.py:11` update imports. Per gap analysis §5.2. |
| **Safety-net tests** | ~5 tests: "BeliefEngine.update_belief with evidence → correct posterior", "DSL parse + evaluate → expected kernel state", "Synara API endpoint returns correct belief state", "graph adapter produces valid belief network", "graph service evidence integration works". Per gap analysis §6.4. |
| **Blast radius** | ~10 files touched. `synara_views.py` + 3 graph/ files + 6 deleted synara/ files. |

---

### CR-0.5 — Wire forgeviz completion; delete legacy chart helpers

| Field | Value |
|---|---|
| **Phase** | 0 |
| **Change type** | `debt` |
| **Risk tier** | LOW |
| **What moves** | Ensure remaining legacy chart helpers in `analysis/` (post CR-0.1 move) go through `forgeviz`. Wire forgeviz into `report_views.py`, `spc_views.py`, `a3_views.py`, any view rendering charts with inline Plotly. Delete the now-superseded chart files once forgeviz covers their cases. Per gap analysis §6.5. |
| **Dependencies** | CR-0.1 (chart files consolidated in analysis/). |
| **CR partners** | None. |
| **Safety-net tests** | ~3-5 tests — one per chart category: control chart, capability histogram, residuals plot, bar chart, line chart. Each asserts ChartSpec dict shape matches forgeviz output. Per gap analysis §6.5. |
| **Blast radius** | ~5 files touched. `report_views.py`, `spc_views.py`, `a3_views.py`, plus chart helper deletions. |

---

### CR-0.6 — Confirmed dead-code deletion

| Field | Value |
|---|---|
| **Phase** | 0 |
| **Change type** | `debt` |
| **Risk tier** | LOW-MED |
| **What moves** | Per gap analysis §6.6 + §10.1: (1) `agents_api/views.py` — delete 5 agent dispatchers (researcher, writer, editor, experimenter, eda), extract `get_shared_llm` to new helper module; (2) `agents_api/urls.py` — delete agent dispatch routes; (3) `Workflow` model + `workflow_views.py` (433 LOC) + `workflow_urls.py` (11 LOC) — full delete; (4) `ActionItem` model + `action_views.py` (65 LOC) + `action_urls.py` (21 LOC) — full delete, update 5 view files to use `loop.Commitment`; (5) `AuditChecklist` model — verify no live tenants, delete; (6) Sweep test files violating TST-001 §10.6: `test_endpoint_smoke.py`, `test_t1_deep.py`, `test_t2_views_smoke.py`, `test_*_coverage.py` family (~6,000 LOC). Per gap analysis §2.19, §10.1. |
| **Dependencies** | None (can run in parallel with CR-0.1-0.5). |
| **CR partners** | `fmea_views.py`, `rca_views.py`, `a3_views.py`, `hoshin_views.py` (ActionItem reference updates to Commitment); `forge/tasks.py:152` (`get_shared_llm` move). Per gap analysis §6.6. |
| **Safety-net tests** | ~3 tests: "loop.Commitment workflow handles commitment creation from FMEA finding", "Commitment creation from RCA finding", "Commitment creation from A3 action". Per gap analysis §6.6. |
| **Blast radius** | ~20 files touched. 4 models deleted, ~10 files deleted, 5 view files updated (ActionItem→Commitment refs), 8+ sweep test files deleted. |

---

### CR-0.7 — Move permissions.py to qms_core/ shim

| Field | Value |
|---|---|
| **Phase** | 0 |
| **Change type** | `enhancement` |
| **Risk tier** | LOW |
| **What moves** | Create stub `qms_core/` Django app. Move `agents_api/permissions.py` (`qms_can_edit`, `qms_queryset`, `qms_set_ownership`, `get_tenant`, `is_site_admin`) to `qms_core/permissions.py`. Leave `agents_api/permissions.py` as a shim re-exporting from `qms_core.permissions`. Update cross-app imports: `loop/supplier_views.py`, `notifications/webhook_views.py` (x2), `syn/audit/compliance.py`. Per gap analysis §2.1, §5.1, §5.6, §6.7. |
| **Dependencies** | None (can land in parallel with forge wiring). |
| **CR partners** | `loop/`, `notifications/`, `syn/audit/` — 3-app coordinated import update. Per gap analysis §6.7. |
| **Safety-net tests** | ~3 tests: "qms_can_edit returns correct permission for site admin vs regular user", "qms_queryset filters to user's tenant correctly", "get_tenant returns current user's tenant". Per gap analysis §6.7. |
| **Blast radius** | ~8 files touched. New `qms_core/` app created (apps.py, __init__.py, permissions.py), shim in agents_api/, 4 cross-app import updates. |

---

### CR-0.8 — Split iso_views.py in-place (prerequisite)

| Field | Value |
|---|---|
| **Phase** | 0 |
| **Change type** | `enhancement` |
| **Risk tier** | MED |
| **What moves** | Split `iso_views.py` (4,874 LOC, 85 view functions) into sub-modules within `agents_api/iso/` WITHOUT moving them to target apps yet. Create: `agents_api/iso/ncr_views.py`, `agents_api/iso/audit_views.py`, `agents_api/iso/training_views.py`, `agents_api/iso/management_review_views.py`, `agents_api/iso/document_views.py`, `agents_api/iso/supplier_views.py`, `agents_api/iso/equipment_views.py`, `agents_api/iso/complaint_views.py`, `agents_api/iso/risk_views.py`, `agents_api/iso/afe_views.py`, `agents_api/iso/control_plan_views.py`, `agents_api/iso/checklist_views.py`. Update `iso_urls.py` to point at sub-modules. Per gap analysis §3.3, architecture §9.A #15, §7.6. |
| **Dependencies** | None (can land in parallel with forge wiring). |
| **CR partners** | None (within-app reorganization). |
| **Safety-net tests** | 0 new tests — use existing 351-test `iso_tests.py` as regression suite. All 351 tests must pass after the split. Per gap analysis §6.8. |
| **Blast radius** | ~15 files created/modified. 1 large file split into 12 sub-modules, `iso_urls.py` updated, `__init__.py` created. |

---

### CR-0.9 — Create tools/ Django app (modular tool router)

| Field | Value |
|---|---|
| **Phase** | 0 |
| **Change type** | `feature` |
| **Risk tier** | MED |
| **What moves** | Create new `tools/` Django app as a first-class modular tool router system. Move from `agents_api/`: `tool_router.py`, `tool_registry.py`, `tool_event_handlers.py`, `tool_events.py`, `base_tool_model.py`. Design as an expandable, modular architecture for tool registration, routing, and event handling. All downstream apps (`loop/`, `graph/`, `workbench/`, etc.) import from `tools/` instead of `agents_api/`. Leave shim in `agents_api/` during parallel period. Per gap analysis §12.2 item 3 (Eric decision 2026-04-10). |
| **Dependencies** | CR-0.7 (permissions shim should be in place first — tools/ may need permission checks). |
| **CR partners** | `loop/evaluator.py:309` (imports `tool_events`), any other file importing `agents_api.tool_router` or `agents_api.tool_registry`. Per gap analysis §5.1, §5.6. |
| **Safety-net tests** | ~5 tests: "tool registration adds tool to registry", "tool router dispatches to correct handler", "tool event fires and handler receives", "tool registry lists all registered tools", "base tool model validates correctly". |
| **Blast radius** | ~10 files touched. New `tools/` app created (apps.py, __init__.py, router.py, registry.py, events.py, handlers.py, models.py), shims in agents_api/, import updates in loop/ and other consumers. |

---

### Phase 0 Summary

| CR | Title | Risk | Change type | Est. tests | Deps |
|---|---|---|---|---|---|
| CR-0.1 | dsw→analysis consolidation | LOW | debt | 5 | — |
| CR-0.2 | forgespc wiring + spc.py deletion | MED | debt | 7 | CR-0.1 |
| CR-0.3 | forgestat wiring + legacy dsw/ deletion | MED | debt | 8-10 | CR-0.2 |
| CR-0.4 | forgesia wiring + synara/ deletion | MED | debt | 5 | — (independent) |
| CR-0.5 | forgeviz completion | LOW | debt | 3-5 | CR-0.1 |
| CR-0.6 | Dead-code deletion | LOW-MED | debt | 3 | — (independent) |
| CR-0.7 | permissions.py → qms_core/ shim | LOW | enhancement | 3 | — (independent) |
| CR-0.8 | iso_views.py split-in-place | MED | enhancement | 0 (351 regression) | — (independent) |
| CR-0.9 | tools/ app creation | MED | feature | 5 | CR-0.7 |
| **TOTAL** | | | | **~39-43 tests** | |

**Phase 0 net effect:** ~60,000-70,000 LOC deleted. 4 models deleted. `qms_core/` and `tools/` apps created as foundation. `iso_views.py` split into 12 addressable sub-modules. Forge package boundaries clean.

---

## Section 3 — Phase 1A CRs (Leaf Relocations)

Phase 1A relocates the 7 lowest-risk leaf apps to parallel `/app/demo/...` paths AS-IS. No rebuilds yet. Per universal cutover pattern (architecture §13): old code keeps running unchanged at old URLs, new code runs at demo paths.

All Phase 1A CRs depend on CR-0.7 (permissions shim). Those consuming iso_views slices also depend on CR-0.8 (iso split).

---

### CR-1A.1 — Relocate triage/ (source app)

| Field | Value |
|---|---|
| **Phase** | 1A |
| **Change type** | `migration` |
| **Risk tier** | LOW |
| **What moves** | `TriageResult` model (gap analysis §2.3) from `agents_api/models.py:95`. `triage_views.py` (512 LOC, 10 funcs, gap analysis §3.2). `triage_urls.py` → new `triage/urls.py` at `/app/demo/triage/`. |
| **Dependencies** | CR-0.7 (permissions shim). |
| **CR partners** | None — self-contained. Gap analysis §7.1: "CR partners: none (self-contained)." |
| **Safety-net tests** | ~5 tests: "user can upload CSV for triage", "triage run returns cleaned dataset", "triage list returns only current user's runs", "triage detail returns full result", "triage delete removes the run". Per gap analysis §6.2 pattern. |
| **Blast radius** | ~6 files. New `triage/` app (models.py, views.py, urls.py, apps.py, admin.py, __init__.py), demo URL mount in `svend/urls.py`. |

---

### CR-1A.2 — Relocate whiteboard/ (source app)

| Field | Value |
|---|---|
| **Phase** | 1A |
| **Change type** | `migration` |
| **Risk tier** | LOW |
| **What moves** | 4 models: `Board` (line 399), `BoardParticipant` (line 466), `BoardVote` (line 493), `BoardGuestInvite` (line 534). Gap analysis §2.4. `whiteboard_views.py` (1,057 LOC, 23 funcs, gap analysis §3.1). `whiteboard_urls.py` → new `whiteboard/urls.py` at `/app/demo/whiteboard/`. Existing tests: `whiteboard_tests.py` (1,181 LOC, 83 funcs, gap analysis §7.2). |
| **Dependencies** | CR-0.7 (permissions shim). |
| **CR partners** | `notebook_views.py`, `iso_doc_views.py`, `report_views.py`, `a3_views.py` — all use board embeds. Defer import update until cutover if lazy imports. Per gap analysis §7.2. |
| **Safety-net tests** | ~5 tests: "user can create a new board", "board list returns only user's boards", "board participant can be added", "board vote is recorded", "board SVG export generates valid output". Gap analysis §7.2 notes 83 existing behavioral tests as additional regression. |
| **Blast radius** | ~8 files. New `whiteboard/` app created, 4 models relocated, demo URL mount. |

---

### CR-1A.3 — Relocate learn/ (standalone product surface)

| Field | Value |
|---|---|
| **Phase** | 1A |
| **Change type** | `migration` |
| **Risk tier** | LOW |
| **What moves** | 3 models: `SectionProgress` (line 1806), `AssessmentAttempt` (line 1842), `LearnSession` (line 1871). Gap analysis §2.18. `learn_views.py` (2,450 LOC, 32 funcs) + `harada_views.py` (909 LOC, 14 funcs). `learn_content/` directory (~14,556 LOC). `learn_content.py` flat file (8,024 LOC) — resolve canonical via grep (gap analysis §11.1). `learn_urls.py` + `harada_urls.py` → `/app/demo/learn/`. |
| **Dependencies** | CR-0.7 (permissions shim). |
| **CR partners** | None significant. Gap analysis §7.7: "CR partners: none significant." |
| **Safety-net tests** | ~5 tests: "user can list available courses", "section progress tracks completion", "assessment attempt records score", "learn session creates sandbox project correctly", "harada task list returns user's tasks". |
| **Blast radius** | ~15 files. New `learn/` app created, 3 models, 2 view files, 1-2 content data directories, demo URL mounts. Large by LOC but low complexity. |

---

### CR-1A.4 — Relocate vsm/ (operations lean)

| Field | Value |
|---|---|
| **Phase** | 1A |
| **Change type** | `migration` |
| **Risk tier** | MED |
| **What moves** | `ValueStreamMap` model (line 1559). Gap analysis §2.16. `vsm_views.py` (433 LOC, 14 funcs, gap analysis §3.2). URL mount to be determined (inventory §E notes no existing `vsm_urls.py` — may be served via hoshin_urls or direct mounts). Demo path: `/app/demo/vsm/`. |
| **Dependencies** | CR-0.7 (permissions shim), CR-0.8 (iso_views split — VSM may be referenced from iso views). |
| **CR partners** | `HoshinProject.source_vsm` FK, `PlantSimulation.source_vsm` FK — lazy imports updated. `xmatrix_views.py`, `learn_views.py`, `qms_views.py` — import path updates deferred to cutover if lazy. Per gap analysis §7.3. |
| **Safety-net tests** | ~5 tests: "user can create value stream map", "VSM detail returns JSON fields correctly", "current/future state pairing via self-FK works", "VSM list filters to user", "VSM update persists changes". |
| **Blast radius** | ~8 files. New `vsm/` app, 1 model, 1 view file, demo URL mount, lazy FK updates in 2-3 consumer files. |

---

### CR-1A.5 — Relocate simulation/ (operations)

| Field | Value |
|---|---|
| **Phase** | 1A |
| **Change type** | `migration` |
| **Risk tier** | MED |
| **What moves** | `PlantSimulation` model (line 1923). Gap analysis §2.17. `plantsim_views.py` (391 LOC, 8 funcs, gap analysis §3.2). `plantsim_urls.py` → `/app/demo/simulation/`. |
| **Dependencies** | CR-0.7 (permissions shim), CR-1A.4 (VSM relocated first — `source_vsm` FK rewires correctly). |
| **CR partners** | `vsm/` (`source_vsm` FK string lookup updates to new app path). Per gap analysis §7.4. |
| **Safety-net tests** | ~5 tests: "user can create plant simulation", "simulation links to source VSM correctly", "DES layout persists", "simulation run produces results", "simulation list filters to user". |
| **Blast radius** | ~6 files. New `simulation/` app, 1 model, 1 view file, demo URL mount. **Special handling:** coordinate with Eric's parallel simulator upgrade work per memory. |

---

### CR-1A.6 — Relocate qms_measurement/ (source app)

| Field | Value |
|---|---|
| **Phase** | 1A |
| **Change type** | `migration` |
| **Risk tier** | MED |
| **What moves** | `MeasurementEquipment` model (line 5647). Gap analysis §2.14. `equipment_*` slice from split `iso_views.py` (CR-0.8 prerequisite). MSA dispatch references in `spc_views.py`. Demo path: `/app/demo/qms-measurement/`. |
| **Dependencies** | CR-0.7 (permissions shim), CR-0.8 (iso_views split — equipment slice must exist). |
| **CR partners** | **`graph/`** — direct FK `linked_process_node` to `graph.ProcessNode` (line 5724) atomically swaps from `agents_api.MeasurementEquipment` to `qms_measurement.MeasurementEquipment`. `graph/integrations.py:601`, `graph/tests_qms.py` (x3) update imports. `loop/evaluator.py:309` updates import. `spc_views.py` MSA dispatch references. Per gap analysis §2.14, §5.2, §7.5. |
| **Safety-net tests** | ~5 tests: "user can create measurement equipment record", "equipment detail returns Weibull reliability fields", "equipment list filters to site", "graph.ProcessNode FK resolves after migration", "MSA dispatch from spc_views reaches correct handler". |
| **Blast radius** | ~10 files. New `qms_measurement/` app, 1 model, iso slice view file, `graph/integrations.py`, `graph/tests_qms.py` (x3), `loop/evaluator.py`, `spc_views.py`. |

---

### CR-1A.7 — Relocate qms_suppliers/ (source app)

| Field | Value |
|---|---|
| **Phase** | 1A |
| **Change type** | `migration` |
| **Risk tier** | MED |
| **What moves** | 2 models: `SupplierRecord` (line 4789), `SupplierStatusChange` (line 4921). Gap analysis §2.12. `supplier_*` slice from split `iso_views.py` (CR-0.8 prerequisite). Demo path: `/app/demo/qms-suppliers/`. |
| **Dependencies** | CR-0.7 (permissions shim), CR-0.8 (iso_views split — supplier slice must exist). |
| **CR partners** | `loop/models.py:1850, 2105` — 2 loop FKs update in lockstep. `graph/views.py:392` — import update. `NonconformanceRecord.supplier` FK — remains pointing at agents_api during parallel period, rewires at qms_nonconformance extraction (CR-2A.4). Per gap analysis §2.12, §5.1, §7.6. |
| **Safety-net tests** | ~5 tests: "user can create supplier record", "supplier state machine transitions correctly", "supplier list filters to site", "loop supplier FK resolves after migration", "supplier status change audit trail records correctly". |
| **Blast radius** | ~8 files. New `qms_suppliers/` app, 2 models, iso slice view file, `loop/models.py` (2 import sites), `graph/views.py`. |

---

### Phase 1A Summary

| CR | App | Models | Risk | Deps | Key coupling |
|---|---|---|---|---|---|
| CR-1A.1 | `triage/` | 1 | LOW | CR-0.7 | None |
| CR-1A.2 | `whiteboard/` | 4 | LOW | CR-0.7 | Lazy board embeds in 4 views |
| CR-1A.3 | `learn/` | 3 | LOW | CR-0.7 | None |
| CR-1A.4 | `vsm/` | 1 | MED | CR-0.7, CR-0.8 | Hoshin/Simulation lazy FKs |
| CR-1A.5 | `simulation/` | 1 | MED | CR-0.7, CR-1A.4 | VSM FK |
| CR-1A.6 | `qms_measurement/` | 1 | MED | CR-0.7, CR-0.8 | graph/ hard FK |
| CR-1A.7 | `qms_suppliers/` | 2 | MED | CR-0.7, CR-0.8 | loop/ x2 |
| **TOTAL** | **7 apps** | **13 models** | | | |

**Parallelism:** CR-1A.1, CR-1A.2, CR-1A.3 can run in parallel (no cross-dependencies). CR-1A.4 must precede CR-1A.5. CR-1A.6 and CR-1A.7 are independent of each other but both depend on CR-0.8.

---

## Section 4 — Phase 1B CRs (Leaf Rebuilds)

Phase 1B rebuilds the relocated leaf apps against the new architecture: pull contract endpoints, sv-* widgets, ForgeViz charts, source/transition/sink roles. Still under `/app/demo/...` paths. Each 1B CR depends on its corresponding 1A CR.

**CR-1B.1 is the critical-path item** — workbench pull contract infrastructure must exist before any Phase 2B consumer can pull from it.

---

### CR-1B.1 — Workbench pull contract + ArtifactReference + DSWResult conversion

| Field | Value |
|---|---|
| **Phase** | 1B |
| **Change type** | `feature` |
| **Risk tier** | HIGH |
| **What moves** | Add `ArtifactReference` model to `workbench/models.py`. Implement full pull contract per architecture §2.3: container browse (`GET /api/workbench/workbenches/`), container detail + manifest, single artifact fetch, sub-artifact fetch with dotted key addressing, reference registration (`POST /references/`), reference list, container delete with friction (409 Conflict + tombstone). Convert `DSWResult` storage layer so new analysis runs save to `workbench.Workbench` + `workbench.Artifact` rows. Delete `workbench.KnowledgeGraph` per architecture §9.B.1. Handle `SavedModel` (line 121) relocation per gap analysis §2.2. Per gap analysis §2.2, architecture §2.3-§2.4, §5.1-§5.4. |
| **Dependencies** | Phase 0 complete (forge wiring done, analysis dispatch consolidated). |
| **CR partners** | `core/models/notebook.py` — 4 direct DSWResult references (lines 71, 82, 217, 228) convert to workbench pull API in sub-CR CR-1B.1a. Per gap analysis §5.4. |
| **Safety-net tests** | ~8 tests: "save analysis result creates Workbench + Artifact rows", "container browse lists user's workbenches", "artifact manifest contains addressable keys per §5.1", "sub-artifact fetch returns correct value for statistics.cpk", "reference registration creates ArtifactReference row", "delete workbench with references returns 409 with reference list", "force delete marks references with source_deleted_at", "tombstone renders for deleted source artifact". Per architecture §6.2 pattern. |
| **Blast radius** | ~15 files. `workbench/models.py` (new ArtifactReference model, KnowledgeGraph deletion), `workbench/views.py` (pull API endpoints), `workbench/urls.py` (new routes), `analysis/dispatch.py` (save to new storage), `dsw/standardize.py` (manifest emission per architecture §5.4). |

---

### CR-1B.1a — Convert notebook DSWResult references to pull API

| Field | Value |
|---|---|
| **Phase** | 1B |
| **Change type** | `enhancement` |
| **Risk tier** | MED |
| **What moves** | Rewrite 4 direct `DSWResult` references in `core/models/notebook.py` (lines 71, 82, 217, 228) to use workbench pull API. Per gap analysis §5.4, §12.2 item 8: "They MUST convert in Phase 1B because that's when DSWResult itself is being converted." |
| **Dependencies** | CR-1B.1 (workbench pull contract must exist). |
| **CR partners** | `core/` — notebook model is already in core, so this is a single-app enhancement. |
| **Safety-net tests** | ~4 tests: "notebook cell referencing analysis result renders via pull API", "notebook cell with deleted source renders tombstone", "notebook cell reference is registered as ArtifactReference", "notebook list still works after migration". |
| **Blast radius** | ~3 files. `core/models/notebook.py`, `core/views.py` (notebook endpoints), `core/tests.py`. |

---

### CR-1B.2 — Triage pull contract

| Field | Value |
|---|---|
| **Phase** | 1B |
| **Change type** | `feature` |
| **Risk tier** | LOW |
| **What moves** | Add `TriageResultReference` model to `triage/models.py`. Implement pull contract endpoints: container browse, artifact fetch, reference registration, delete friction. Per architecture §2.3, gap analysis §7.1. |
| **Dependencies** | CR-1A.1 (triage relocated). |
| **CR partners** | None. |
| **Safety-net tests** | ~5 tests: "triage container browse lists user's datasets", "triage artifact fetch returns cleaned data", "reference registration creates row", "delete with references returns 409", "force delete tombstones references". |
| **Blast radius** | ~5 files. `triage/models.py`, `triage/views.py`, `triage/urls.py`. |

---

### CR-1B.3 — Whiteboard pull contract

| Field | Value |
|---|---|
| **Phase** | 1B |
| **Change type** | `feature` |
| **Risk tier** | LOW |
| **What moves** | Add `BoardReference` model to `whiteboard/models.py`. Implement pull contract: boards as causal-claim sources for investigations and A3s. Per architecture §2.3, gap analysis §7.2. |
| **Dependencies** | CR-1A.2 (whiteboard relocated). |
| **CR partners** | None. |
| **Safety-net tests** | ~5 tests: "board container browse lists user's boards", "board artifact (causal diagram) fetchable", "reference registration works", "delete with references returns 409", "board export as SVG remains functional". |
| **Blast radius** | ~5 files. `whiteboard/models.py`, `whiteboard/views.py`, `whiteboard/urls.py`. |

---

### CR-1B.4 — Learn sv-* widget rebuild

| Field | Value |
|---|---|
| **Phase** | 1B |
| **Change type** | `enhancement` |
| **Risk tier** | LOW |
| **What moves** | Rebuild learn templates against sv-* flat widget vocabulary. Learn is NOT in the pull graph — no pull contract needed. Minor template/CSS refresh. Per gap analysis §7.7: "Standalone product surface — not in pull graph. Minor rebuild for sv-* widgets." |
| **Dependencies** | CR-1A.3 (learn relocated). |
| **CR partners** | None. |
| **Safety-net tests** | ~3 tests: "course listing page renders with sv-* classes", "assessment page renders correctly", "progress tracking still records completions". |
| **Blast radius** | ~5 files. Learn templates + static assets. |

---

### CR-1B.5 — VSM cockpit rebuild

| Field | Value |
|---|---|
| **Phase** | 1B |
| **Change type** | `feature` |
| **Risk tier** | MED |
| **What moves** | Full cockpit rebuild per migration plan "VSM Workbench Spec" — integrated calculator panel, forgesiop + forgequeue integration, ForgeViz charts. VSM is sink-like for analysis (pulls cycle-time data) and source-like for kaizen bursts (pulled into Hoshin). Per gap analysis §7.3, migration plan Tier 2. |
| **Dependencies** | CR-1A.4 (VSM relocated). |
| **CR partners** | None — rebuild is self-contained. Hoshin pulls from VSM after hoshin/ extracts in Phase 2A. |
| **Safety-net tests** | ~5 tests: "VSM cockpit renders with calculator panel", "takt time calculation returns correct result from forgesiop", "VSM chart renders via ForgeViz", "VSM map data saves with all 16 JSON fields", "current/future state pairing still works". |
| **Blast radius** | ~8 files. `vsm/views.py`, `vsm/templates/`, `vsm/static/`, calculator integration. |

---

### CR-1B.6 — Simulation rebuild

| Field | Value |
|---|---|
| **Phase** | 1B |
| **Change type** | `enhancement` |
| **Risk tier** | MED |
| **What moves** | Rebuild simulation with ForgeSiop discrete-event engine per migration plan P2. Per gap analysis §7.4. |
| **Dependencies** | CR-1A.5 (simulation relocated). |
| **CR partners** | None. |
| **Safety-net tests** | ~5 tests: "simulation creation works", "DES run produces expected layout output", "simulation result persists", "source_vsm FK resolves to vsm/ model", "simulation list filters to user". |
| **Blast radius** | ~5 files. `simulation/views.py`, simulation templates. |

---

### CR-1B.7 — Measurement pull contract

| Field | Value |
|---|---|
| **Phase** | 1B |
| **Change type** | `feature` |
| **Risk tier** | LOW |
| **What moves** | Add `MeasurementReference` model. Source role pull contract — calibration evidence pulled into audits. MSA study integration with workbench. Per gap analysis §7.5. |
| **Dependencies** | CR-1A.6 (qms_measurement relocated). |
| **CR partners** | None. |
| **Safety-net tests** | ~5 tests: "equipment container browse lists site's equipment", "calibration record fetchable as artifact", "reference registration works", "Weibull reliability fields accessible via pull", "delete with references returns 409". |
| **Blast radius** | ~5 files. `qms_measurement/models.py`, `qms_measurement/views.py`, `qms_measurement/urls.py`. |

---

### CR-1B.8 — Suppliers pull contract

| Field | Value |
|---|---|
| **Phase** | 1B |
| **Change type** | `feature` |
| **Risk tier** | LOW |
| **What moves** | Add `SupplierReference` model. Source role pull contract — supplier records pulled into supplier qualification audits. Per gap analysis §7.6. |
| **Dependencies** | CR-1A.7 (qms_suppliers relocated). |
| **CR partners** | None. |
| **Safety-net tests** | ~5 tests: "supplier container browse lists site's suppliers", "supplier detail includes status history", "reference registration works", "state machine transitions correctly via pull", "delete with references returns 409". |
| **Blast radius** | ~5 files. `qms_suppliers/models.py`, `qms_suppliers/views.py`, `qms_suppliers/urls.py`. |

---

### Phase 1B Summary

| CR | App/Focus | Risk | Change type | Est. tests | Critical path? |
|---|---|---|---|---|---|
| CR-1B.1 | Workbench pull contract + DSWResult conversion | HIGH | feature | 8 | **YES** |
| CR-1B.1a | Notebook DSWResult refs → pull API | MED | enhancement | 4 | Yes (blocks notebook) |
| CR-1B.2 | Triage pull contract | LOW | feature | 5 | No |
| CR-1B.3 | Whiteboard pull contract | LOW | feature | 5 | No |
| CR-1B.4 | Learn sv-* rebuild | LOW | enhancement | 3 | No |
| CR-1B.5 | VSM cockpit rebuild | MED | feature | 5 | No |
| CR-1B.6 | Simulation rebuild | MED | enhancement | 5 | No |
| CR-1B.7 | Measurement pull contract | LOW | feature | 5 | No |
| CR-1B.8 | Suppliers pull contract | LOW | feature | 5 | No |
| **TOTAL** | | | | **~45 tests** | |

---

## Section 5 — Eric Review Gate 1

### Gate 1 Checklist — between Phase 1B completion and Phase 2A start

**Timing:** After ALL Phase 1A relocations and Phase 1B rebuilds are complete and demo-testable.

| # | Check | How to verify | Pass criterion |
|---|---|---|---|
| 1 | All 7 leaf apps running at `/app/demo/...` paths | Walk each demo URL in browser | Pages load, data renders |
| 2 | Old production URLs still serve old code unchanged | Walk each production URL | No regressions |
| 3 | Workbench pull contract operational | POST a reference registration, GET an artifact by dotted key | Correct response shapes |
| 4 | DSWResult conversion working for new analysis runs | Run a capability study, verify it saves as `workbench.Artifact` | Artifact row exists with 10-key content |
| 5 | Notebook DSWResult refs converted | Open a notebook that references an analysis result | Result renders via pull API, not direct ORM query |
| 6 | Triage, whiteboard, measurement, suppliers pull contracts live | Test reference registration + delete friction for each | 409 on delete with refs; tombstone renders |
| 7 | VSM cockpit calculator functional | Run a takt time calculation on a VSM map | Correct result via forgesiop |
| 8 | `loop/` imports still resolve for all updated apps | Run `loop/` tests | All pass |
| 9 | `graph/` FK to measurement resolves | Run `graph/tests_qms.py` | All pass |
| 10 | No production regressions in existing test suite | Full test suite run | Pass rate >= pre-extraction baseline |
| 11 | Phase 0 forge wiring holds | Spot-check 3 analysis types (SPC, parametric, Bayesian) | Correct 10-key results from forge packages |
| 12 | ForgeRack unaffected | Load ForgeRack demo | No regressions |

**Decision:** Eric reviews all demo paths, tests with real data, and gives explicit sign-off before Phase 2A begins. If any item fails, fix under demo paths before proceeding.

---

## Section 6 — Phase 2A CRs (Medium-Coupling Relocations)

Phase 2A extracts the medium-coupling apps — the QMS domain modules with loop/ and graph/ dependencies. Every extraction here is at minimum a 2-app coordinated CR (extraction + loop/ import update). Per gap analysis §5.1, §7.5: "Every QMS extraction is a 2-app coordinated CR."

**Ordering constraints (from gap analysis §8 summary):**
1. CR-2A.2 (qms_documents) before CR-2A.3 (qms_training) — `TrainingRequirement.controlled_doc` FK
2. CR-2A.4 (qms_nonconformance) before CR-2A.5 (qms_audit) — `AuditFinding.ncr` FK
3. CR-2A.6 (qms_investigation) before CR-2A.7 (qms_a3) — `A3.rca_session` FK
4. CR-1B.1 (workbench pull) must exist before CR-2A.8 (hoshin) — HoshinKPI conversion prep

---

### CR-2A.1 — Relocate qms_risk/ (FMEA + Risk register)

| Field | Value |
|---|---|
| **Phase** | 2A |
| **Change type** | `migration` |
| **Risk tier** | HIGH |
| **What moves** | 3 models: `FMEA` (line 839), `FMEARow` (line 936), `Risk` (line 5213). Gap analysis §2.9. `fmea_views.py` (1,589 LOC, 27 funcs, gap analysis §3.1). `risk_*` slice from split `iso_views.py` (CR-0.8). `fmea_urls.py` + risk URL slice → `/app/demo/qms-risk/`. |
| **Dependencies** | CR-0.7 (permissions), CR-0.8 (iso_views split), CR-0.9 (tools/ if risk views import tool events). |
| **CR partners** | **`loop/`** — 2 FMEARow FK references: `loop/models.py:1204`, `loop/models.py:1447`. Import swap from `agents_api.FMEARow` to `qms_risk.FMEARow`. `safety/models.py:736` reference. `notebook_views.py`, `learn_views.py` lazy import updates. Per gap analysis §2.9, §5.1, §8.1. |
| **Safety-net tests** | ~5 tests: "user can create FMEA with rows", "FMEARow.hypothesis_link to core.Hypothesis resolves", "risk register entry creates correctly", "loop.FMISRow FK to FMEARow resolves after migration", "FMEA list filters to site". Per gap analysis §8.1. |
| **Blast radius** | ~12 files. New `qms_risk/` app, 3 models, `fmea_views.py`, iso risk slice, `loop/models.py` (2 sites), `safety/models.py`, `notebook_views.py`, `learn_views.py`. |

---

### CR-2A.2 — Relocate qms_documents/ (controlled documents + ISO authoring)

| Field | Value |
|---|---|
| **Phase** | 2A |
| **Change type** | `migration` |
| **Risk tier** | HIGH |
| **What moves** | 7 models: `ControlledDocument` (line 4534), `DocumentRevision` (line 4702), `DocumentStatusChange` (line 4745), `ISODocument` (line 6027), `ISOSection` (line 6128), `ControlPlan` (line 6488), `ControlPlanItem` (line 6587). Gap analysis §2.10. `document_*` slice from split `iso_views.py` + `iso_doc_views.py` (716 LOC, gap analysis §3.2) + `control_plan_*` slice. Demo path: `/app/demo/qms-documents/`. |
| **Dependencies** | CR-0.7, CR-0.8. |
| **CR partners** | **`loop/`** — 6 import sites: `loop/models.py:694, 884, 895, 971, 1117`, `loop/services.py:297`. **`graph/`** — M2M `ControlledDocument.linked_process_nodes` to `graph.ProcessNode` (line 4591); `graph/views.py:392`. `ControlPlanItem` has cross-app FKs to `loop.FMISRow`, `graph.ProcessNode`, `MeasurementEquipment`. Per gap analysis §2.10, §5.1, §8.2. **3-app coordinated CR.** |
| **Safety-net tests** | ~7 tests: "create controlled document with revision", "document status change workflow works", "ISODocument → ControlledDocument publish link works", "ISOSection embedded media renders", "loop/ ControlledDocument FK resolves", "graph.ProcessNode M2M resolves after migration", "ControlPlanItem cross-app FKs resolve (loop, graph, measurement)". Per gap analysis §8.2. |
| **Blast radius** | ~18 files. **Highest-effort single CR after Site.** New `qms_documents/` app, 7 models, 2 view files + iso slices, `loop/models.py` (5 sites), `loop/services.py`, `graph/views.py`, `graph/models.py` M2M reference. Django M2M relocation requires explicit through-model data migration per gap analysis §8.2. |

---

### CR-2A.3 — Relocate qms_training/ (TWI competency)

| Field | Value |
|---|---|
| **Phase** | 2A |
| **Change type** | `migration` |
| **Risk tier** | HIGH |
| **What moves** | 3 models: `TrainingRequirement` (line 4133), `TrainingRecord` (line 4214), `TrainingRecordChange` (line 4295). Gap analysis §2.11. `training_*` slice from split `iso_views.py`. Demo path: `/app/demo/qms-training/`. |
| **Dependencies** | CR-0.7, CR-0.8, CR-2A.2 (`TrainingRequirement` has FK to `ControlledDocument` — must resolve to `qms_documents.ControlledDocument`). |
| **CR partners** | **`loop/`** — 4 sites: `loop/models.py:879`, `loop/views.py:2386`, `loop/services.py:253`, `loop/readiness.py:171`. Per gap analysis §2.11, §5.1, §8.3. |
| **Safety-net tests** | ~5 tests: "create training requirement linked to controlled document", "training record tracks TWI competency levels", "training record change audit trail works", "loop readiness check uses training records correctly", "training list filters to site". Per gap analysis §8.3. |
| **Blast radius** | ~10 files. New `qms_training/` app, 3 models, iso training slice, `loop/models.py`, `loop/views.py`, `loop/services.py`, `loop/readiness.py`. |

---

### CR-2A.4 — Relocate qms_nonconformance/ (NCR + complaints)

| Field | Value |
|---|---|
| **Phase** | 2A |
| **Change type** | `migration` |
| **Risk tier** | HIGH |
| **What moves** | 3 models: `NonconformanceRecord` (line 3440), `NCRStatusChange` (line 3688), `CustomerComplaint` (line 5022). Gap analysis §2.13. `ncr_*` and `complaint_*` slices from split `iso_views.py`. Demo path: `/app/demo/qms-nonconformance/`. |
| **Dependencies** | CR-0.7, CR-0.8. |
| **CR partners** | **`loop/models.py:1855`** — NCR FK update. **`graph/views.py:392`** — import update. `AuditFinding.ncr` FK — stays pointing at agents_api until qms_audit extracts (CR-2A.5). `Report`/`CAPAReport.ncr` references — removed at Phase 2B/3 deletion. Per gap analysis §2.13, §5.1, §8.4. |
| **Safety-net tests** | ~5 tests: "create NCR with state machine transitions", "NCR TRANSITION_REQUIRES enforced", "customer complaint links to NCR", "loop NCR FK resolves after migration", "NCR list filters to site with linked_process_node_ids". Per gap analysis §8.4. |
| **Blast radius** | ~12 files. New `qms_nonconformance/` app, 3 models, 2 iso slices, `loop/models.py`, `graph/views.py`. |

---

### CR-2A.5 — Relocate qms_audit/ (internal audit + management review)

| Field | Value |
|---|---|
| **Phase** | 2A |
| **Change type** | `migration` |
| **Risk tier** | MED |
| **What moves** | 4 models: `InternalAudit` (line 3977), `AuditFinding` (line 4076), `ManagementReviewTemplate` (line 4339), `ManagementReview` (line 4461). Gap analysis §2.8. `audit_*` and `review_*` slices from split `iso_views.py`. Demo path: `/app/demo/qms-audit/`. |
| **Dependencies** | CR-0.7, CR-0.8, **CR-2A.4** (`AuditFinding.ncr` FK must point to `qms_nonconformance.NonconformanceRecord`). |
| **CR partners** | `AuditFinding.ncr` FK rewire to `qms_nonconformance.NonconformanceRecord`. `ManagementReview.data_snapshot` consumers. Per gap analysis §2.8, §8.5. |
| **Safety-net tests** | ~5 tests: "create internal audit with findings", "audit finding links to NCR in qms_nonconformance/", "management review template with DEFAULT_SECTIONS loads", "management review data_snapshot captures metrics", "audit list filters to site". Per gap analysis §8.5. |
| **Blast radius** | ~10 files. New `qms_audit/` app, 4 models, 2 iso slices. |

---

### CR-2A.6 — Relocate qms_investigation/ (RCA + Ishikawa + CEMatrix + bridges)

| Field | Value |
|---|---|
| **Phase** | 2A |
| **Change type** | `migration` |
| **Risk tier** | HIGH |
| **What moves** | 3 models: `RCASession` (line 1143), `IshikawaDiagram` (line 1342), `CEMatrix` (line 1445). Gap analysis §2.5. `rca_views.py` (1,057 LOC, 17 funcs), `ishikawa_views.py` (158 LOC), `ce_views.py` (155 LOC), `investigation_views.py` (432 LOC), `investigation_bridge.py` (638 LOC), `evidence_bridge.py`, `evidence_weights.py`. Demo path: `/app/demo/qms-investigation/`. |
| **Dependencies** | CR-0.7 (permissions), CR-0.9 (tools/ — investigation bridge may use tool events). |
| **CR partners** | `a3_views.py` (`A3.rca_session` FK stays pointing at agents_api until CR-2A.7). `notebook_views.py` (lazy reads). `qms_views.py`. `NonconformanceRecord.rca_session` FK — already in qms_nonconformance from CR-2A.4, update to point at `qms_investigation.RCASession`. `CAPAReport.rca_session` FK — left as-is until deletion. Per gap analysis §2.5, §8.6. |
| **Safety-net tests** | ~5 tests: "RCA session state machine transitions correctly", "Ishikawa diagram creation and retrieval", "CEMatrix creation and retrieval", "investigation bridge translates between modules correctly", "RCA session embedding field persists for similarity search". Per gap analysis §8.6. |
| **Blast radius** | ~15 files. New `qms_investigation/` app, 3 models, 5 view/bridge files, qms_nonconformance FK update, notebook lazy import. |

---

### CR-2A.7 — Relocate qms_a3/ (A3 reports)

| Field | Value |
|---|---|
| **Phase** | 2A |
| **Change type** | `migration` |
| **Risk tier** | HIGH |
| **What moves** | 1 model: `A3Report` (line 594). Gap analysis §2.6. `a3_views.py` (1,389 LOC, 23 funcs, gap analysis §3.1). Demo path: `/app/demo/qms-a3/`. Relocate AS-IS first; full rebuild in CR-2B.7. |
| **Dependencies** | **CR-2A.6** (`A3.rca_session` FK must point to `qms_investigation.RCASession`). |
| **CR partners** | `rca_views.py` (RCA→A3 linking path update). `notebook_views.py` (lazy import). `learn_views.py` (lazy import). Site FK stays pointing at agents_api until Phase 4. Per gap analysis §2.6, §8.7. |
| **Safety-net tests** | ~5 tests: "A3 report creation works", "A3 links to RCA session in qms_investigation/", "A3 imported_from provenance JSON persists", "A3 embedded_diagrams JSON persists", "A3 list filters to user/site". Per gap analysis §8.7. |
| **Blast radius** | ~8 files. New `qms_a3/` app, 1 model, 1 view file, cross-app FK updates. |

---

### CR-2A.8 — Relocate hoshin/ (Hoshin Kanri + AFE + X-matrix)

| Field | Value |
|---|---|
| **Phase** | 2A |
| **Change type** | `migration` |
| **Risk tier** | HIGH |
| **What moves** | 9 models: `HoshinProject` (line 2135), `ProjectTemplate` (line 2323), `ResourceCommitment` (line 2573), `StrategicObjective` (line 2746), `AnnualObjective` (line 2823), `HoshinKPI` (line 2913), `XMatrixCorrelation` (line 3344), `AFE` (line 5386), `AFEApprovalLevel` (line 5562). Gap analysis §2.15. `hoshin_views.py` (2,094 LOC, 36 funcs) + `xmatrix_views.py` (1,069 LOC, 13 funcs). `afe_*` slice from split `iso_views.py`. Demo path: `/app/demo/hoshin/`. |
| **Dependencies** | CR-0.7, CR-0.8 (afe slice), CR-1A.4 (VSM relocated — `HoshinProject.source_vsm` FK resolves to `vsm.ValueStreamMap`). |
| **CR partners** | `fmea_views.py` — remove direct AFE creation per `feedback_afe_policy.md`. `vsm/` — `source_vsm` FK. Site + Employee FKs stay pointing at agents_api until Phase 4. XMatrixCorrelation post-delete signal handlers must be preserved. Per gap analysis §2.15, §8.8. |
| **Safety-net tests** | ~7 tests: "HoshinProject creation with source VSM link", "X-matrix correlation CRUD works", "AFE creation only via Hoshin (not FMEA direct)", "AFEApprovalLevel chain resolves", "ResourceCommitment state machine works", "HoshinKPI effective_actual still works (pre-conversion)", "post-delete signal handlers fire correctly for strategic/annual objectives". Per gap analysis §8.8. |
| **Blast radius** | ~15 files. New `hoshin/` app, 9 models, 2 view files + iso afe slice, `fmea_views.py` AFE removal, signal handler preservation. |

---

### CR-2A.9 — Relocate notebook_views.py to core/

| Field | Value |
|---|---|
| **Phase** | 2A |
| **Change type** | `migration` |
| **Risk tier** | LOW |
| **What moves** | `notebook_views.py` (1,621 LOC, 26 funcs) from `agents_api/` to `core/views/notebook.py`. Notebook model already lives in `core/models/notebook.py`. DSWResult refs already converted in CR-1B.1a. Per gap analysis §3.1, §12.2 item 4. |
| **Dependencies** | CR-1B.1 (DSWResult conversion complete — notebook refs already use pull API). |
| **CR partners** | None — model is already in core, views follow. |
| **Safety-net tests** | ~5 tests: "notebook list returns user's notebooks", "notebook cell CRUD works", "notebook cell referencing workbench artifact renders", "notebook analysis embed works via pull API", "notebook URLs resolve at demo path". |
| **Blast radius** | ~5 files. `core/views/notebook.py` (new), `agents_api/notebook_views.py` (deleted), URL mount update. |

---

### Phase 2A Summary

| CR | App | Models | Risk | Key coupling |
|---|---|---|---|---|
| CR-2A.1 | `qms_risk/` | 3 | HIGH | loop x2, safety |
| CR-2A.2 | `qms_documents/` | 7 | HIGH | loop x6, graph M2M |
| CR-2A.3 | `qms_training/` | 3 | HIGH | loop x4 |
| CR-2A.4 | `qms_nonconformance/` | 3 | HIGH | loop, graph |
| CR-2A.5 | `qms_audit/` | 4 | MED | NCR FK ordering |
| CR-2A.6 | `qms_investigation/` | 3 | HIGH | multi-source FKs |
| CR-2A.7 | `qms_a3/` | 1 | HIGH | RCA FK |
| CR-2A.8 | `hoshin/` | 9 | HIGH | VSM FK, pull contract prep |
| CR-2A.9 | `notebook→core/` | 0 (views only) | LOW | DSWResult conversion done |
| **TOTAL** | **9 extractions** | **33 models** | | |

---

## Section 7 — Phase 2B CRs (Medium-Coupling Rebuilds)

Phase 2B rebuilds the extracted medium-coupling apps against the new architecture. This is where pull contracts land for transitions, the HoshinKPI canonical conversion happens, and the new `reports/` and `sop/` sink apps are built from scratch. Per architecture §3.3, §3.4.

---

### CR-2B.1 — qms_risk/ pull contract (source role)

| Field | Value |
|---|---|
| **Phase** | 2B |
| **Change type** | `feature` |
| **Risk tier** | MED |
| **What moves** | Add `FMEAReference`/`RiskReference` models. Source role pull contract — FMEAs and individual risks pullable by investigations. Per gap analysis §8.1. |
| **Dependencies** | CR-2A.1 (qms_risk relocated). |
| **CR partners** | None — consumers pull later. |
| **Safety-net tests** | ~5 tests: "FMEA container browse lists site's FMEAs", "FMEARow artifact fetchable by ID", "risk register entry fetchable", "reference registration works", "delete with references returns 409". |
| **Blast radius** | ~5 files. `qms_risk/models.py`, `qms_risk/views.py`, `qms_risk/urls.py`. |

---

### CR-2B.2 — qms_documents/ pull contract + ForgeDoc integration

| Field | Value |
|---|---|
| **Phase** | 2B |
| **Change type** | `feature` |
| **Risk tier** | MED |
| **What moves** | Source role pull contract — documents pullable by audits + training. ForgeDoc PDF integration for document rendering. Per gap analysis §8.2. |
| **Dependencies** | CR-2A.2 (qms_documents relocated). |
| **CR partners** | None. |
| **Safety-net tests** | ~5 tests: "document container browse lists site's docs", "document revision history pullable", "ISOSection content fetchable", "ForgeDoc PDF renders from controlled document", "delete with references returns 409". |
| **Blast radius** | ~6 files. qms_documents/ models, views, urls, ForgeDoc integration. |

---

### CR-2B.3 — qms_training/ pull contract

| Field | Value |
|---|---|
| **Phase** | 2B |
| **Change type** | `feature` |
| **Risk tier** | LOW |
| **What moves** | Source role pull contract — training records pullable by supplier audits + internal audits. TWI competency levels per TRN-001 §3. Per gap analysis §8.3. |
| **Dependencies** | CR-2A.3 (qms_training relocated). |
| **CR partners** | None. |
| **Safety-net tests** | ~5 tests: "training record container browse works", "training requirement linked to document pullable", "certification status property resolves", "reference registration works", "delete with references returns 409". |
| **Blast radius** | ~5 files. qms_training/ models, views, urls. |

---

### CR-2B.4 — qms_nonconformance/ pull contract (transition+source dual)

| Field | Value |
|---|---|
| **Phase** | 2B |
| **Change type** | `feature` |
| **Risk tier** | MED |
| **What moves** | Transition+source dual role pull contract. NCRs pullable by investigations; investigations push back findings via reference registration. Per gap analysis §8.4. |
| **Dependencies** | CR-2A.4 (qms_nonconformance relocated). |
| **CR partners** | None. |
| **Safety-net tests** | ~5 tests: "NCR container browse works", "NCR artifact fetchable (state, linked data)", "customer complaint fetchable via NCR container", "reference registration works for investigation consumers", "delete with references returns 409". |
| **Blast radius** | ~5 files. qms_nonconformance/ models, views, urls. |

---

### CR-2B.5 — qms_audit/ pull contract + audit upgrade alignment

| Field | Value |
|---|---|
| **Phase** | 2B |
| **Change type** | `feature` |
| **Risk tier** | MED |
| **What moves** | Transition pull contract — audits pull evidence from documents, training, supplier records. Auditor independence rules per `project_audit_upgrade.md`. FMEA-based risk-driven audit frequency. Per gap analysis §8.5, architecture §3.3. |
| **Dependencies** | CR-2A.5 (qms_audit relocated), CR-2B.4 (NCR pull ready for audit findings to reference). |
| **CR partners** | None. |
| **Safety-net tests** | ~5 tests: "audit container browse works", "audit finding pulls NCR from qms_nonconformance", "management review data_snapshot captures cross-module metrics", "auditor independence validation works", "delete with references returns 409". |
| **Blast radius** | ~6 files. qms_audit/ models, views, urls, auditor independence logic. |

---

### CR-2B.6 — qms_investigation/ pull contract (canonical transition)

| Field | Value |
|---|---|
| **Phase** | 2B |
| **Change type** | `feature` |
| **Risk tier** | HIGH |
| **What moves** | Canonical transition pull contract — pulls from workbench + triage + whiteboard sources; emits investigation containers with conclusions, evidence, timeline. Investigation bridge rewired to use pull API instead of direct imports. This is the central transition in the pull graph. Per gap analysis §8.6, architecture §3.3. |
| **Dependencies** | CR-2A.6 (qms_investigation relocated), CR-1B.1 (workbench pull contract must exist for pull-from-workbench). |
| **CR partners** | None — downstream A3 and reports pull from investigation after they rebuild. |
| **Safety-net tests** | ~7 tests: "investigation container browse works", "investigation pulls chart from workbench via pull API", "investigation pulls triaged dataset via pull API", "investigation pulls board diagram from whiteboard", "RCA conclusion pullable as artifact by downstream", "investigation evidence bridge uses pull contract not direct ORM", "investigation delete with A3/report references returns 409". |
| **Blast radius** | ~10 files. qms_investigation/ models, views, urls, bridge rewrites. |

---

### CR-2B.7 — qms_a3/ full rebuild (transition)

| Field | Value |
|---|---|
| **Phase** | 2B |
| **Change type** | `feature` |
| **Risk tier** | HIGH |
| **What moves** | **Full rebuild** per Eric directive (architecture §3.3). New architecture, updated models, new pull contract interface. A3 pulls from workbench + qms_investigation, emits A3 container pullable by reports and SOPs. New front-end with sv-* widgets. Per gap analysis §8.7. |
| **Dependencies** | CR-2A.7 (qms_a3 relocated), CR-2B.6 (investigation pull ready for A3 to consume). |
| **CR partners** | None. |
| **Safety-net tests** | ~7 tests: "A3 pulls problem statement from workbench analysis", "A3 pulls root cause from investigation", "A3 pulls evidence artifacts from analysis sources", "A3 container with sections emits pullable manifest", "A3 reference registration to both workbench and investigation", "A3 list at demo path works", "A3 delete with report references returns 409". |
| **Blast radius** | ~12 files. qms_a3/ full rewrite — models, views, urls, templates, static assets. |

---

### CR-2B.8 — hoshin/ HoshinKPI pull contract conversion (CANONICAL)

| Field | Value |
|---|---|
| **Phase** | 2B |
| **Change type** | `feature` |
| **Risk tier** | HIGH |
| **What moves** | **Canonical pull contract conversion** per architecture §4.5. Add `linked_artifact` FK (→ `workbench.Artifact`, `on_delete=SET_NULL`) and `linked_artifact_key` CharField to `HoshinKPI`. Replace direct `DSWResult.objects.filter(user=..., analysis_type=...).order_by('-created_at').first()` at `agents_api/models.py:3285` with `workbench_pull_api.fetch_artifact_value(...)`. Auto-register `ArtifactReference` on first access. Tombstone handling when source deleted. Per gap analysis §5.5, architecture §4.5. |
| **Dependencies** | CR-2A.8 (hoshin relocated), CR-1B.1 (workbench pull contract exists). |
| **CR partners** | `workbench/` — ArtifactReference rows created for hoshin consumers. |
| **Safety-net tests** | ~7 tests: "HoshinKPI.effective_actual returns value from workbench artifact", "HoshinKPI auto-registers ArtifactReference on first access", "HoshinKPI returns None with tombstone when source deleted", "HoshinKPI linked_artifact FK resolves", "HoshinKPI linked_artifact_key dotted addressing works", "workbench delete friction shows hoshin reference", "KPI dashboard renders tombstone for deleted source". Per architecture §4.5. |
| **Blast radius** | ~8 files. `hoshin/models.py` (new fields + rewrite), `hoshin/views.py` (dashboard rendering), migration file, workbench integration test. |

---

### CR-2B.9 — reports/ new sink (brand new build)

| Field | Value |
|---|---|
| **Phase** | 2B |
| **Change type** | `feature` |
| **Risk tier** | HIGH |
| **What moves** | Brand new `reports/` sink app. Pulls from A3, investigations, analysis sources, audit findings. Uses ForgeDoc for PDF generation. Pro-tier-only per architecture §3.4. Replaces legacy `Report` model + `report_views.py` (deleted at Phase 3). Per gap analysis §8.9. |
| **Dependencies** | CR-2B.7 (A3 pull contract ready — reports pull from A3 containers). |
| **CR partners** | Deletion of `agents_api.Report` model deferred to Phase 3 cutover. |
| **Safety-net tests** | ~7 tests: "create report pulling from A3 container", "report pulls chart from workbench transitively via A3 reference", "report renders PDF via ForgeDoc", "report template CRUD works", "report sections assemble correctly", "report list at demo path works", "pro-tier gating enforced". |
| **Blast radius** | ~10 files. New `reports/` app — models.py, views.py, urls.py, templates, ForgeDoc integration. |

---

### CR-2B.10 — sop/ new sink (brand new build)

| Field | Value |
|---|---|
| **Phase** | 2B |
| **Change type** | `feature` |
| **Risk tier** | HIGH |
| **What moves** | Brand new `sop/` sink app. Standard work / SOP editor. Pulls best-practice content from investigations, A3s, training records. The "standardize" step in LOOP-001's Investigate → Standardize → Verify. Per architecture §3.4, gap analysis §8.9. |
| **Dependencies** | CR-2B.6 (investigation pull ready). |
| **CR partners** | None. |
| **Safety-net tests** | ~7 tests: "create SOP document pulling from investigation", "SOP pulls training content", "SOP version control works", "SOP signoff flow works", "SOP section CRUD", "SOP list at demo path", "SOP renders correctly". |
| **Blast radius** | ~10 files. New `sop/` app — models.py, views.py, urls.py, templates. |

---

### CR-2B.11 — CAPA delete+replace

| Field | Value |
|---|---|
| **Phase** | 2B |
| **Change type** | `debt` |
| **Risk tier** | MED |
| **What moves** | Prepare deletion of `CAPAReport` (line 3723), `CAPAStatusChange` (line 3942), `capa_views.py` (711 LOC). Per architecture §9.A decision #3: CAPA is delete+replace. New CAPA functionality lives in `qms_investigation/` + ForgeDoc CAPA generator (built as part of CR-2B.6). Remove `NonconformanceRecord.capa_report` FK. Remove `AFE` CAPA references. Actual deletion of code deferred to Phase 3 cutover. Per gap analysis §2.7, §10.2. |
| **Dependencies** | CR-2B.6 (investigation pull contract — replacement functionality ready), CR-2B.9 (reports sink — if any CAPA-style reports needed). |
| **CR partners** | `qms_nonconformance/` (FK removal), `hoshin/` (AFE CAPA references). |
| **Safety-net tests** | ~3 tests: "investigation produces CAPA-equivalent output via ForgeDoc", "NCR no longer requires capa_report FK", "existing CAPA data migration plan verified". |
| **Blast radius** | ~8 files. `qms_nonconformance/models.py` FK removal, `hoshin/models.py` reference cleanup, migration files. |

---

### Phase 2B Summary

| CR | App/Focus | Risk | Change type | Est. tests |
|---|---|---|---|---|
| CR-2B.1 | qms_risk/ pull contract | MED | feature | 5 |
| CR-2B.2 | qms_documents/ pull + ForgeDoc | MED | feature | 5 |
| CR-2B.3 | qms_training/ pull contract | LOW | feature | 5 |
| CR-2B.4 | qms_nonconformance/ pull contract | MED | feature | 5 |
| CR-2B.5 | qms_audit/ pull + audit upgrade | MED | feature | 5 |
| CR-2B.6 | qms_investigation/ canonical transition | HIGH | feature | 7 |
| CR-2B.7 | qms_a3/ full rebuild | HIGH | feature | 7 |
| CR-2B.8 | hoshin/ HoshinKPI conversion (CANONICAL) | HIGH | feature | 7 |
| CR-2B.9 | reports/ new sink | HIGH | feature | 7 |
| CR-2B.10 | sop/ new sink | HIGH | feature | 7 |
| CR-2B.11 | CAPA delete+replace | MED | debt | 3 |
| **TOTAL** | | | | **~63 tests** |

---

## Section 8 — Eric Review Gate 2

### Gate 2 Checklist — between Phase 2B completion and Phase 3 cutover

**Timing:** After ALL Phase 2A relocations and Phase 2B rebuilds are complete and demo-testable. This is the final gate before the single-night cutover.

| # | Check | How to verify | Pass criterion |
|---|---|---|---|
| 1 | All 19 new apps running at `/app/demo/...` paths | Walk every demo URL | Pages load, data renders with real data |
| 2 | All old production URLs still serve old code unchanged | Walk every production URL | Zero regressions |
| 3 | Full pull chain operational: workbench → investigation → A3 → report | End-to-end test with real data | Chain completes, references registered at every link |
| 4 | HoshinKPI canonical conversion working | Load Hoshin dashboard with linked KPI | KPI value from workbench artifact, not DSWResult |
| 5 | Investigation pulls from workbench, triage, whiteboard | Create investigation pulling from all 3 sources | Artifacts render, references registered |
| 6 | A3 pulls from investigation + workbench | Create A3 pulling investigation conclusion + chart | Renders correctly |
| 7 | Reports sink pulls from A3 + generates PDF | Create report from A3, render PDF | PDF generates via ForgeDoc |
| 8 | SOP sink pulls from investigation + training | Create SOP pulling content | Renders correctly |
| 9 | CAPA replacement functional in investigation | Investigation produces CAPA-equivalent output | ForgeDoc CAPA generation works |
| 10 | Delete friction working across all sources/transitions | Delete a workbench session with downstream refs | 409 Conflict with reference list; tombstones on force-delete |
| 11 | Multi-tenancy isolation verified | User A cannot pull from user B's containers | 403 or empty results |
| 12 | `loop/` integration holds for all extracted apps | Full loop test suite | All pass |
| 13 | `graph/` integration holds | Full graph test suite | All pass |
| 14 | `safety/` FKs still resolve (pre-Phase 4) | safety tests | All pass |
| 15 | No production regressions | Full test suite | Pass rate >= pre-extraction baseline |
| 16 | `reports/` and `sop/` demo-ready | New sinks reviewed by Eric with real data | Approved for production |
| 17 | ForgeRack unaffected | Load ForgeRack | No regressions |

**Decision:** Eric reviews ALL demo paths with real data, confirms pull chains work end-to-end, and gives explicit sign-off. Phase 3 cutover does not proceed without this sign-off.

---

## Section 9 — Phase 3 Cutover CR

### CR-3.1 — Single-night URL swap + legacy code deletion

| Field | Value |
|---|---|
| **Phase** | 3 |
| **Change type** | `migration` |
| **Risk tier** | HIGH |
| **Title** | Production cutover: swap all demo paths to production URLs |
| **Dependencies** | ALL Phase 1B and 2B CRs complete. Eric gate 2 sign-off. |
| **CR partners** | Every extracted app. `svend/urls.py` (central routing). All legacy template files. |

**What happens in this CR (single commit, one night):**

1. **URL swap in `svend/urls.py`:** Every `/app/demo/<thing>/` route becomes the production `/app/<thing>/` route. Every `/api/demo/<thing>/` becomes `/api/<thing>/`. Old URL mounts either delete or become 301 redirects per gap analysis §4.1 disposition column.

2. **Legacy code deletion per gap analysis §10.2-§10.3:**
   - `agents_api/models.py Report` + `report_views.py` (802 LOC) + `report_urls.py`
   - `agents_api/models.py CAPAReport` + `CAPAStatusChange` + `capa_views.py` (711 LOC) + `capa_urls.py`
   - `agents_api/report_types.py`
   - Legacy templates: `workbench_new.html` (11,790 LOC), `safety_coming_soon.html` (1,993 LOC), `iso_9001_qms.html` (1,159 LOC)
   - `core/models/graph.py` (~200 LOC) — DEPRECATED per migration plan
   - Legacy URL mounts: `/api/agents/`, `/api/workflows/`, `/api/actions/`, `/api/capa/`

3. **DSWResult data migration:** Either convert existing `DSWResult` rows to `workbench.Artifact` rows, or mark them legacy-read-only. Per gap analysis §2.2.

4. **Nav link updates:** All internal navigation links updated to point at new production URLs.

5. **Legacy test path cleanup:** Delete dual-route test fixtures, remove demo-path test assertions.

6. **301 redirects:** Per gap analysis §4.1: `/api/dsw/` → `/api/workbench/`, `/api/analysis/` → `/api/workbench/analysis/`, `/api/forecast/` → `/api/workbench/forecast/`, `/api/experimenter/` → `/api/workbench/experimenter/`, `/api/spc/` → `/api/workbench/spc/`, `/api/synara/` → `/api/workbench/synara/`, `/api/guide/` → `/api/chat/guide/`, `/api/plantsim/` → `/api/simulation/`, `/api/harada/` → `/api/learn/harada/`, `/api/iso-docs/` → `/api/qms-documents/`.

**Safety-net tests:** ~10 tests: "every new production URL returns 200", "every 301 redirect resolves", "pull chain workbench→investigation→A3→report works at production URLs", "HoshinKPI at production URL returns correct value", "no 404s in navigation links", "DSWResult legacy rows accessible (if legacy-read-only) or converted (if migrated)".

**Blast radius:** ~50+ files. This is the largest single commit. URL routing, template deletions, nav link updates, test cleanup.

**Risk mitigation:**
- Full backup + WAL archive check before CR starts (WAL at `/home/postgres_wal_archive`)
- Rollback script: revert commit + restore old URL routing
- Performed during lowest-traffic window
- Monitoring active for 30 minutes post-cutover

---

## Section 10 — Phase 4 Site Move CR

### CR-4.1 — Site + qms_core/ atomic migration (CRITICAL)

| Field | Value |
|---|---|
| **Phase** | 4 |
| **Change type** | `migration` |
| **Risk tier** | **CRITICAL** |
| **Title** | Atomic migration of Site, Employee, and qms_core infrastructure models |
| **Dependencies** | CR-3.1 (cutover complete — all apps running at final production URLs). |
| **CR partners** | **EVERY extracted app** — 15+ apps update FK string lookups atomically. `loop/`, `safety/`, `graph/` update cross-app FKs. |

**What moves (gap analysis §9.1):**

9 models to `qms_core/`:
- `Site` (line 2020) — **24 incoming FKs** (19 internal + 5 cross-app). Gap analysis §2.1.
- `SiteAccess` (line 2072) — unique-together `(site, user)`
- `Employee` (line 2511) — `loop/` x2, `safety/` x2, `hoshin/` ResourceCommitment FK
- `ActionToken` (line 2669) — FK to Employee
- `QMSFieldChange` (line 4965) — polymorphic `(record_type, record_id)`
- `QMSAttachment` (line 6390) — `ENTITY_MODEL_MAP` with 8 entity types
- `Checklist` (line 5843) — FK to Site
- `ChecklistExecution` (line 5923) — FK to Checklist
- `ElectronicSignature` (line 6232) — `SynaraImmutableLog` subclass, **hash-chained** (21 CFR Part 11 compliance)

Plus view files:
- `qms_views.py` (216 LOC) — cross-app dashboard
- `token_views.py` (214 LOC) — email-token system
- Checklist views from iso_views split
- Drop the `agents_api/permissions.py` shim (all callers now import `qms_core.permissions` directly)

**FK update locations (all atomic per gap analysis §9.2):**

| App | FKs to update |
|---|---|
| `qms_a3/` | A3Report.site |
| `qms_risk/` | FMEA.site, Risk.site |
| `qms_investigation/` | RCASession.site |
| `hoshin/` | ProjectTemplate.site, HoshinProject.site, AnnualObjective.site, AFE.site, ResourceCommitment.employee |
| `qms_audit/` | InternalAudit.site |
| `qms_training/` | TrainingRequirement.site |
| `qms_documents/` | ControlledDocument.site, ControlPlan.site |
| `qms_nonconformance/` | NonconformanceRecord.site, CustomerComplaint.site |
| `qms_measurement/` | MeasurementEquipment.site |
| `loop/` | models.py lines 424, 962, 982, 1190 (Site + Employee) |
| `safety/` | models.py lines 69, 220, 260, 344, 354 (Site + Employee) |
| `graph/` | models.py line 520 (Site) |

**Safety-net tests:** ~10 tests: "Site CRUD at qms_core works", "every extracted app's Site FK resolves to qms_core.Site", "Employee FK resolves from loop/, safety/, hoshin/", "ElectronicSignature hash chain integrity post-migration", "SiteAccess unique-together still enforced", "QMSFieldChange polymorphic lookup works from new location", "QMSAttachment ENTITY_MODEL_MAP resolves all 8 types", "token action email link still works", "Site admin permission check works from qms_core.permissions", "all 24 incoming FK sites resolve".

**Blast radius:** ~30+ files across 15+ apps. Migration files across every extracted app.

**Risk mitigation (gap analysis §9.4):**
- Two-phase migration: `--state-operations` separated from `--database` operations (per architecture §9.A #12)
- Full PostgreSQL backup + WAL archive check at `/home/postgres_wal_archive` before CR starts
- Dry-run on DB clone if feasible
- Rollback script: revert all migrations, restore shims
- ElectronicSignature integrity verification: run hash-chain audit before AND after migration
- `ENTITY_MODEL_MAP` updated to reference new app paths for all 8 entity types
- Performed during lowest-traffic window with monitoring active

**Post Phase 4 cleanup:**
- Delete `agents_api/models.py` residual (should be empty after all extractions)
- Remove `agents_api` from `INSTALLED_APPS` in `svend/settings.py` (if fully empty)
- Delete any orphaned `*_tests.py` files referencing deleted models
- Delete residual `urls.py` / `apps.py` shims
- Handle remaining infrastructure helpers per gap analysis §10.5 (cache.py, embeddings, etc.) — each moves with the CR that touches it

---

## Section 11 — Risk Summary Table

All CRs sorted by risk tier descending. CRITICAL first, then HIGH, MED, LOW.

| CR | Title | Phase | Risk | Change type | Est. tests | Blast radius (files) |
|---|---|---|---|---|---|---|
| **CR-4.1** | Site + qms_core/ atomic migration | 4 | **CRITICAL** | migration | 10 | 30+ |
| **CR-3.1** | Single-night cutover | 3 | HIGH | migration | 10 | 50+ |
| **CR-1B.1** | Workbench pull contract + DSWResult conversion | 1B | HIGH | feature | 8 | 15 |
| **CR-2A.1** | qms_risk/ relocation | 2A | HIGH | migration | 5 | 12 |
| **CR-2A.2** | qms_documents/ relocation | 2A | HIGH | migration | 7 | 18 |
| **CR-2A.3** | qms_training/ relocation | 2A | HIGH | migration | 5 | 10 |
| **CR-2A.4** | qms_nonconformance/ relocation | 2A | HIGH | migration | 5 | 12 |
| **CR-2A.6** | qms_investigation/ relocation | 2A | HIGH | migration | 5 | 15 |
| **CR-2A.7** | qms_a3/ relocation | 2A | HIGH | migration | 5 | 8 |
| **CR-2A.8** | hoshin/ relocation | 2A | HIGH | migration | 7 | 15 |
| **CR-2B.6** | qms_investigation/ canonical transition | 2B | HIGH | feature | 7 | 10 |
| **CR-2B.7** | qms_a3/ full rebuild | 2B | HIGH | feature | 7 | 12 |
| **CR-2B.8** | hoshin/ HoshinKPI canonical conversion | 2B | HIGH | feature | 7 | 8 |
| **CR-2B.9** | reports/ new sink | 2B | HIGH | feature | 7 | 10 |
| **CR-2B.10** | sop/ new sink | 2B | HIGH | feature | 7 | 10 |
| **CR-0.2** | forgespc wiring | 0 | MED | debt | 7 | 5 |
| **CR-0.3** | forgestat wiring | 0 | MED | debt | 8-10 | 25 |
| **CR-0.4** | forgesia wiring | 0 | MED | debt | 5 | 10 |
| **CR-0.8** | iso_views.py split | 0 | MED | enhancement | 0 (351 regr.) | 15 |
| **CR-0.9** | tools/ app creation | 0 | MED | feature | 5 | 10 |
| **CR-1A.4** | vsm/ relocation | 1A | MED | migration | 5 | 8 |
| **CR-1A.5** | simulation/ relocation | 1A | MED | migration | 5 | 6 |
| **CR-1A.6** | qms_measurement/ relocation | 1A | MED | migration | 5 | 10 |
| **CR-1A.7** | qms_suppliers/ relocation | 1A | MED | migration | 5 | 8 |
| **CR-1B.1a** | Notebook DSWResult refs conversion | 1B | MED | enhancement | 4 | 3 |
| **CR-1B.5** | VSM cockpit rebuild | 1B | MED | feature | 5 | 8 |
| **CR-1B.6** | Simulation rebuild | 1B | MED | enhancement | 5 | 5 |
| **CR-2A.5** | qms_audit/ relocation | 2A | MED | migration | 5 | 10 |
| **CR-2B.1** | qms_risk/ pull contract | 2B | MED | feature | 5 | 5 |
| **CR-2B.2** | qms_documents/ pull + ForgeDoc | 2B | MED | feature | 5 | 6 |
| **CR-2B.4** | qms_nonconformance/ pull contract | 2B | MED | feature | 5 | 5 |
| **CR-2B.5** | qms_audit/ pull + audit upgrade | 2B | MED | feature | 5 | 6 |
| **CR-2B.11** | CAPA delete+replace | 2B | MED | debt | 3 | 8 |
| **CR-0.6** | Dead-code deletion | 0 | LOW-MED | debt | 3 | 20 |
| **CR-0.1** | dsw→analysis consolidation | 0 | LOW | debt | 5 | 15 |
| **CR-0.5** | forgeviz completion | 0 | LOW | debt | 3-5 | 5 |
| **CR-0.7** | permissions.py → qms_core/ shim | 0 | LOW | enhancement | 3 | 8 |
| **CR-1A.1** | triage/ relocation | 1A | LOW | migration | 5 | 6 |
| **CR-1A.2** | whiteboard/ relocation | 1A | LOW | migration | 5 | 8 |
| **CR-1A.3** | learn/ relocation | 1A | LOW | migration | 5 | 15 |
| **CR-1B.2** | Triage pull contract | 1B | LOW | feature | 5 | 5 |
| **CR-1B.3** | Whiteboard pull contract | 1B | LOW | feature | 5 | 5 |
| **CR-1B.4** | Learn sv-* rebuild | 1B | LOW | enhancement | 3 | 5 |
| **CR-1B.7** | Measurement pull contract | 1B | LOW | feature | 5 | 5 |
| **CR-1B.8** | Suppliers pull contract | 1B | LOW | feature | 5 | 5 |
| **CR-2A.9** | notebook_views → core/ | 2A | LOW | migration | 5 | 5 |
| **CR-2B.3** | qms_training/ pull contract | 2B | LOW | feature | 5 | 5 |

**Total CRs: 45** (9 Phase 0 + 7 Phase 1A + 10 Phase 1B + 9 Phase 2A + 11 Phase 2B + 1 Phase 3 + 1 Phase 4 — includes CR-1B.1a sub-CR, rounds to 48 with the sub-CR counted separately).

**Total estimated behavior tests: ~250** across all CRs.

---

## Section 12 — Critical Path Analysis

### 12.1 The critical path

The critical path is the longest chain of dependent CRs that determines the minimum elapsed time for the full extraction. CRs not on the critical path can parallelize without affecting the end date.

```
CRITICAL PATH (serial chain — each blocks the next):

CR-0.1 (dsw→analysis)
  → CR-0.2 (forgespc)
    → CR-0.3 (forgestat)
      → CR-1B.1 (workbench pull contract + DSWResult conversion)   ← BOTTLENECK
        → CR-1B.1a (notebook DSWResult refs)
          → [Eric Gate 1]
            → CR-2A.6 (qms_investigation relocation)
              → CR-2A.7 (qms_a3 relocation)
                → CR-2B.6 (qms_investigation pull contract)
                  → CR-2B.7 (qms_a3 full rebuild)
                    → CR-2B.9 (reports/ new sink)
                      → [Eric Gate 2]
                        → CR-3.1 (cutover)
                          → CR-4.1 (Site move)
```

**Critical path length: 13 CRs + 2 review gates.**

### 12.2 The bottleneck

**CR-1B.1 (workbench pull contract + DSWResult conversion)** is the single biggest bottleneck in the extraction. It is:
- On the critical path
- A prerequisite for every Phase 2B consumer that pulls from workbench
- The point where `DSWResult` stops existing as a writable model
- The hardest single CR in Phase 1 (HIGH risk)

If CR-1B.1 slips, every downstream Phase 2B rebuild slips with it.

### 12.3 What can parallelize

**During Phase 0:**
- CR-0.4 (forgesia) is independent of CR-0.1→0.3 chain — can run in parallel
- CR-0.6 (dead-code deletion) is independent — can run in parallel
- CR-0.7 (permissions shim) is independent — can run in parallel
- CR-0.8 (iso_views split) is independent — can run in parallel
- CR-0.9 (tools/) depends only on CR-0.7 — can start as soon as 0.7 lands

**During Phase 1A:**
- CR-1A.1 (triage), CR-1A.2 (whiteboard), CR-1A.3 (learn) — all independent, fully parallel
- CR-1A.6 (measurement), CR-1A.7 (suppliers) — independent of each other, parallel

**During Phase 1B:**
- CR-1B.2 through CR-1B.8 — all independent of each other, fully parallel
- Only CR-1B.1 and CR-1B.1a are sequential (critical path)

**During Phase 2A:**
- CR-2A.1 (risk), CR-2A.2 (documents), CR-2A.4 (nonconformance) — independent, parallel
- CR-2A.8 (hoshin) — independent of 2A.1-2A.7 (depends only on Phase 0 + 1A.4)
- CR-2A.9 (notebook→core) — independent of all other 2A CRs

**During Phase 2B:**
- CR-2B.1 through CR-2B.5 — all independent, parallel
- CR-2B.8 (hoshin HoshinKPI) — independent of investigation/A3 chain
- CR-2B.10 (sop/) — depends only on CR-2B.6 (investigation), parallel with CR-2B.7-2B.9

### 12.4 Estimated timeline

Assuming ~2 CRs per week throughput (creation + testing + review + landing), with parallelism where possible:

| Phase | Calendar weeks | CRs (serial) | CRs (parallel possible) |
|---|---|---|---|
| Phase 0 | ~3 weeks | 4 serial (0.1→0.2→0.3, then 0.9) | 5 parallel (0.4, 0.5, 0.6, 0.7, 0.8) |
| Phase 1A | ~2 weeks | 2 serial (1A.4→1A.5) | 5 parallel (1A.1-3, 1A.6-7) |
| Phase 1B | ~2 weeks | 2 serial (1B.1→1B.1a) | 7 parallel (1B.2-8) |
| Eric Gate 1 | ~1 week | — | — |
| Phase 2A | ~3 weeks | 3 serial chains (2A.2→3, 2A.4→5, 2A.6→7) | 3 parallel (2A.1, 2A.8, 2A.9) |
| Phase 2B | ~3 weeks | 3 serial (2B.6→2B.7→2B.9) | 8 parallel (2B.1-5, 2B.8, 2B.10, 2B.11) |
| Eric Gate 2 | ~1 week | — | — |
| Phase 3 | ~1 week | 1 (CR-3.1) | — |
| Phase 4 | ~1 week | 1 (CR-4.1) | — |
| **TOTAL** | **~17 weeks** | | |

This is a lower bound assuming no blockers, no rework, and maximum parallelism. Realistic estimate with discovery and rework: **20-24 weeks** (~5-6 months).

### 12.5 Risk concentration

| Phase | HIGH+ CRs | % of phase CRs |
|---|---|---|
| Phase 0 | 0 | 0% |
| Phase 1A | 0 | 0% |
| Phase 1B | 1 (CR-1B.1) | 10% |
| Phase 2A | 7 | 78% |
| Phase 2B | 5 | 45% |
| Phase 3 | 1 | 100% |
| Phase 4 | 1 (CRITICAL) | 100% |

Risk is concentrated in Phases 2A-4. Phase 0 and 1A are deliberately low-risk to build extraction muscle memory before the hard work begins. This is by design per gap analysis §7.5 item 4: "First-extractions (lowest risk leaves) should be models loop/ does NOT import, to build extraction muscle without the coordination overhead."

---

**End of extraction sequence. Next documents in the planning bundle:**
1. `test_suite_rebuild.md` — the TST-001 behavior test plan derived from the ~5 tests per CR specified here
2. `phase_0_plan.md` — the Phase 0 runbook expanding CR-0.1 through CR-0.9 with pre-CR checklists, detailed file lists, and the tools/ app self-registration design
