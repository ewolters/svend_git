# agents_api Inventory — Object 271 Extraction Planning

**Date:** 2026-04-09
**Status:** DRAFT — current-state inventory under CR `5bf7354c-3de5-4624-b505-a94a5b6ce0ea`
**Author:** Claude (delegated explore agent, Object 271)
**Companion:** `qms_architecture.md` (target state) · `migration_plan.md` (decisions to date)
**Scope:** Read-only inventory of `services/svend/web/agents_api/`. No source files modified.

> This document is the **canonical description of what exists today** in `agents_api/`. It is the input that the gap analysis (`extraction_gap_analysis.md`) and the sequenced extraction plan (`extraction_sequence.md`) will be built against. The target end state lives in `qms_architecture.md` — when the two diverge, the architecture is the goal and this inventory is the starting line.

---

## Section A — Top-level summary

### A.1 Lines of code

| Scope | Python LOC |
|---|---|
| Total `agents_api/**/*.py` (excluding migrations and `__pycache__`) | **289,313** |
| Top-level `*.py` files in `agents_api/` (excluding subpackages) | **94,271** |
| `agents_api/dsw/` subpackage | **73,749** |
| `agents_api/analysis/` subpackage | **78,544** |
| `agents_api/synara/` subpackage | **3,243** |
| `agents_api/learn_content/` subpackage | **~14,556** (helper data, see §D) |
| `agents_api/tests/` subdirectory (49 files) | **22,863** |
| Top-level `*_tests.py` + `tests.py` (19 files) | **22,482** |

`models.py` alone is **6,797 lines**.

### A.2 File counts

| Group | Count | Notes |
|---|---|---|
| Top-level `*.py` files | **116** | Includes 33 view modules, 24 url modules, models.py, plus helpers |
| Total `*.py` (excluding `__pycache__`/migrations) | **315** | |
| `*_views.py` and `views.py` modules at top level | **33** | One per QMS/lean/strategy domain plus dispatch |
| `*_urls.py` and `urls.py` modules at top level | **24** | (1,287 lines total) |
| Top-level test files | **19** | Plus 49 files inside `tests/` |
| `models.py` files inside `agents_api/` | **1** | All 68 models in a single file |
| Sub-package directories | **5** | `dsw/`, `analysis/`, `synara/`, `learn_content/`, `tests/`, `migrations/` |

### A.3 Django models

`agents_api/models.py` defines **68 Django model classes** (67 inherit from `models.Model`, one inherits from `SynaraImmutableLog` for the immutable e-signature chain). No additional `models.py` exists in any subpackage.

### A.4 Test surface

- **68 test files** total within `agents_api/` (19 top-level `*_tests.py` plus 49 under `tests/`).
- Top-level test files contribute roughly **22,482 LOC**; the `tests/` subdirectory adds **22,863 LOC**.
- Behavior-mode test files exist (e.g. `iso_tests.py` — 351 named test functions, real CRUD-and-workflow tests). Sweep-mode files also exist (e.g. `tests/test_endpoint_smoke.py` — TST-001 §10.6 violation). See §F.

### A.5 Executive summary (one sentence)

`agents_api` currently contains **at least 17 distinct domains crammed into one Django app**: agent dispatch, statistical analysis storage, statistical engines (twice — `dsw/` and `analysis/`), Bayesian belief engine (`synara/`), SPC computation, whiteboards, A3 reports, FMEA, RCA, NCR, CAPA, internal audits, training records, controlled documents, supplier QA, customer complaints, organizational risk register, AFEs, X-matrix, Hoshin projects, value stream maps, plant simulation, ISO authored documents, control plans, measurement/calibration equipment, electronic signatures, learn content/courses, Harada method, ForgeRack persistence, action items, employee/resource commitments, and an authoring layer for ISO documents. Every one of those is a candidate for its own app under the Object 271 source/transition/sink topology.

---

## Section B — Model inventory (the centerpiece)

The inventory below covers every model class in `agents_api/models.py`. Counts of fields exclude PK and timestamps; counts of relationships are limited to the model's own declarations.

> **Reading the table:** "FKs OUT" means relations declared on this model. "FKs IN (agents_api)" lists same-app reverse pointers. "FKs IN (cross-app)" identifies relations from `loop/`, `graph/`, `core/`, `safety/`, `notifications/`, `accounts/`, etc. Site is the most-pointed-at hub model and is documented in detail at the end of the section.

### B.1 Model index by line

| # | Line | Model | Domain tag | Risk | Target home (proposed) |
|---|---|---|---|---|---|
| 1 | 17 | `Workflow` | `agent_layer` | LOW | `automation/` or DELETE |
| 2 | 45 | `DSWResult` | `analysis_storage` | HIGH | `workbench/` (Artifact) |
| 3 | 95 | `TriageResult` | `analysis_storage` | MED | `triage/` (new app) |
| 4 | 121 | `SavedModel` | `analysis_storage` | MED | `workbench/` |
| 5 | 179 | `AgentLog` | `agent_layer` | LOW | `chat/` or DELETE |
| 6 | 212 | `CacheEntry` | `infrastructure` | LOW | `core/` or `syn/` |
| 7 | 259 | `LLMUsage` | `infrastructure` | LOW | `accounts/` (billing-adjacent) |
| 8 | 352 | `RateLimitOverride` | `infrastructure` | LOW | `accounts/` |
| 9 | 399 | `Board` | `whiteboard` | MED | `whiteboard/` (new app) |
| 10 | 466 | `BoardParticipant` | `whiteboard` | LOW | `whiteboard/` |
| 11 | 493 | `BoardVote` | `whiteboard` | LOW | `whiteboard/` |
| 12 | 534 | `BoardGuestInvite` | `whiteboard` | LOW | `whiteboard/` |
| 13 | 594 | `A3Report` | `qms_a3` | HIGH | `qms_a3/` (new) |
| 14 | 736 | `Report` | `qms_capa` | HIGH | `qms_capa/` or sink `reports/` |
| 15 | 839 | `FMEA` | `qms_risk` | HIGH | `qms_risk/` (new) |
| 16 | 936 | `FMEARow` | `qms_risk` | HIGH | `qms_risk/` |
| 17 | 1143 | `RCASession` | `qms_investigation` | HIGH | `qms_investigation/` |
| 18 | 1342 | `IshikawaDiagram` | `qms_investigation` | LOW | `qms_investigation/` |
| 19 | 1445 | `CEMatrix` | `qms_investigation` | LOW | `qms_investigation/` |
| 20 | 1559 | `ValueStreamMap` | `operations_lean` | HIGH | `vsm/` (new) |
| 21 | 1806 | `SectionProgress` | `operations_workflow` | LOW | `learn/` (new app) |
| 22 | 1842 | `AssessmentAttempt` | `operations_workflow` | LOW | `learn/` |
| 23 | 1871 | `LearnSession` | `operations_workflow` | LOW | `learn/` |
| 24 | 1923 | `PlantSimulation` | `operations_simulation` | MED | `simulation/` (Eric owns) |
| 25 | 2020 | `Site` | `infrastructure` | **CRITICAL** | `qms_core/` or LAST extraction |
| 26 | 2072 | `SiteAccess` | `infrastructure` | MED | `qms_core/` |
| 27 | 2135 | `HoshinProject` | `operations_strategy` | HIGH | `hoshin/` (new) |
| 28 | 2323 | `ProjectTemplate` | `operations_strategy` | MED | `hoshin/` |
| 29 | 2406 | `ActionItem` | `operations_workflow` | MED | DELETE (LOOP-001 supersedes with Commitment) |
| 30 | 2511 | `Employee` | `infrastructure` | HIGH | `qms_core/` (TWI-adjacent) |
| 31 | 2573 | `ResourceCommitment` | `operations_strategy` | MED | `hoshin/` |
| 32 | 2669 | `ActionToken` | `infrastructure` | LOW | `qms_core/` |
| 33 | 2746 | `StrategicObjective` | `operations_strategy` | MED | `hoshin/` |
| 34 | 2823 | `AnnualObjective` | `operations_strategy` | MED | `hoshin/` |
| 35 | 2913 | `HoshinKPI` | `operations_strategy` | MED | `hoshin/` |
| 36 | 3344 | `XMatrixCorrelation` | `operations_strategy` | LOW | `hoshin/` |
| 37 | 3440 | `NonconformanceRecord` | `qms_nonconformance` | HIGH | `qms_nonconformance/` (new) |
| 38 | 3688 | `NCRStatusChange` | `qms_nonconformance` | LOW | `qms_nonconformance/` |
| 39 | 3723 | `CAPAReport` | `qms_capa` | HIGH | `qms_capa/` (or DELETE per migration plan) |
| 40 | 3942 | `CAPAStatusChange` | `qms_capa` | LOW | `qms_capa/` |
| 41 | 3977 | `InternalAudit` | `qms_audit` | HIGH | `qms_audit/` (new) |
| 42 | 4076 | `AuditFinding` | `qms_audit` | MED | `qms_audit/` |
| 43 | 4133 | `TrainingRequirement` | `qms_training` | HIGH | `qms_training/` (new) |
| 44 | 4214 | `TrainingRecord` | `qms_training` | HIGH | `qms_training/` |
| 45 | 4295 | `TrainingRecordChange` | `qms_training` | LOW | `qms_training/` |
| 46 | 4339 | `ManagementReviewTemplate` | `qms_audit` | LOW | `qms_audit/` (or `qms_documents/`) |
| 47 | 4461 | `ManagementReview` | `qms_audit` | MED | `qms_audit/` |
| 48 | 4534 | `ControlledDocument` | `qms_documents` | HIGH | `qms_documents/` (new) |
| 49 | 4702 | `DocumentRevision` | `qms_documents` | LOW | `qms_documents/` |
| 50 | 4745 | `DocumentStatusChange` | `qms_documents` | LOW | `qms_documents/` |
| 51 | 4789 | `SupplierRecord` | `qms_suppliers` | HIGH | `qms_suppliers/` (new) |
| 52 | 4921 | `SupplierStatusChange` | `qms_suppliers` | LOW | `qms_suppliers/` |
| 53 | 4965 | `QMSFieldChange` | `qms_audit` | MED | `qms_core/` (cross-cutting audit log) |
| 54 | 5022 | `CustomerComplaint` | `qms_nonconformance` | HIGH | `qms_nonconformance/` |
| 55 | 5213 | `Risk` | `qms_risk` | HIGH | `qms_risk/` |
| 56 | 5386 | `AFE` | `operations_strategy` | MED | `hoshin/` (per AFE policy) |
| 57 | 5562 | `AFEApprovalLevel` | `operations_strategy` | LOW | `hoshin/` |
| 58 | 5647 | `MeasurementEquipment` | `qms_measurement` | HIGH | `qms_measurement/` (new) |
| 59 | 5800 | `AuditChecklist` | `qms_audit` | LOW | `qms_audit/` (or DELETE — superseded by `Checklist`) |
| 60 | 5843 | `Checklist` | `qms_audit` | MED | `qms_core/` (cross-cutting) |
| 61 | 5923 | `ChecklistExecution` | `qms_audit` | MED | `qms_core/` |
| 62 | 6027 | `ISODocument` | `qms_documents` | MED | `qms_documents/` (authoring) |
| 63 | 6128 | `ISOSection` | `qms_documents` | MED | `qms_documents/` |
| 64 | 6232 | `ElectronicSignature` | `infrastructure` | HIGH | `qms_core/` (immutable log, hash chain) |
| 65 | 6390 | `QMSAttachment` | `qms_documents` | MED | `qms_core/` (cross-cutting attachment table) |
| 66 | 6488 | `ControlPlan` | `qms_documents` | MED | `qms_measurement/` or `qms_documents/` |
| 67 | 6587 | `ControlPlanItem` | `qms_documents` | MED | same as parent — links to `loop.FMISRow` and `graph.ProcessNode` |
| 68 | 6720 | `RackSession` | `unclassified` | LOW | new `forgerack/` app or DELETE (demo only) |

### B.2 Per-model detail (highlights)

> Full Read-tool inspection covered every model's field block, FK declarations, status enums, and `to_dict` methods. The summaries below capture what matters for extraction: relationships, hub status, deprecation flags, cross-app touchpoints. Field counts are exact.

#### Workflow (line 17, `agent_layer`)
- 5 fields (`name`, `steps` JSON, `created_at`, `last_run`); FK out: `core.Tenant`, `User`. No incoming FKs detected anywhere in repo. Served via `workflow_views.py` (433 LOC, 11 functions) and `workflow_urls.py`. **Likely dead** — not referenced from any non-test, non-self code. Leftover from original "chain of agents" era.

#### DSWResult (line 45, `analysis_storage`) — HIGH
- 8 fields incl. `data` (encrypted JSON), `result_type`, `title`, `project` FK to `core.Project`. PK is `CharField(50)` not UUID — interaction with `core.Notebook` via stable ID strings.
- **FKs IN (cross-app):** `core/models/notebook.py` references `agents_api.DSWResult` four times (notebook artifacts pull DSW results as evidence). `HoshinKPI.calculator_result_type` queries DSWResult by string for KPI auto-pull (line 3285).
- Used by `dsw_views.py`, `notebook_views.py`, `learn_views.py`, `report_views.py`, `a3_views.py`. The Object 271 architecture replaces this with `workbench.Workbench` + `workbench.Artifact` per `qms_architecture.md` §3.2 — DSWResult migration is the largest single-model lift.

#### TriageResult (line 95, `analysis_storage`)
- Encrypted CSV + report fields, owned by user/tenant. Used only by `triage_views.py`. Clean extraction to a new `triage/` app.

#### SavedModel (line 121, `analysis_storage`)
- ML model registry. Self-FK `parent_model` for retraining versions. FK `core.Project`. Used by `dsw_views.py`, `autopilot_views.py`. Belongs in `workbench/` or a future `models/` app.

#### AgentLog (line 179) / CacheEntry (212) / LLMUsage (259) / RateLimitOverride (352)
- Pure infrastructure. AgentLog is the original agent dispatch operational log — likely stale. LLMUsage and RateLimitOverride drive billing tier rate limits. CacheEntry has a custom `db_table = "session_cache"` and is consumed by `cache.py` (302 LOC, dedicated module).

#### Board family (lines 399, 466, 493, 534) — MED
- Whiteboard collaborative model with versioning, voting, guest invites. `whiteboard_views.py` (1,057 LOC, 23 funcs) and `whiteboard_urls.py` (83 LOC) plus `whiteboard_tests.py` (1,181 LOC). Used by `notebook_views.py`, `iso_doc_views.py` (board image embeds), `report_views.py`, `a3_views.py` (embedded SVG diagrams). Self-contained domain — extracts cleanly to a `whiteboard/` app.

#### A3Report (line 594) — HIGH
- 14 data fields incl. `imported_from` JSON (provenance), `embedded_diagrams` JSON (whiteboard exports), `last_critique` JSON (LLM critique). FK out: `User × 2` (owner + created_by), `Site`, `core.Tenant`, `core.Project`, `core.Notebook`. Used by `a3_views.py` (1,389 LOC, 23 funcs), `rca_views.py` (1057 LOC), `learn_views.py`, `notebook_views.py`. Per `qms_architecture.md` §3.3, this app is being **largely rebuilt** as part of Object 271 — the extraction is also the rebuild opportunity.

#### Report (line 736)
- Generic CAPA / 8D / future report shell. Registry-driven (`report_types.py` defines section schemas). Used by `report_views.py` (802 LOC), referenced by `NonconformanceRecord.capa_report` (line 3548). Currently the only model that holds CAPA-style records before the new sink reports app comes online. Per Eric: report assemblers are being built brand-new in Object 271 as sinks — `Report` is a candidate for **delete + replace**.

#### FMEA / FMEARow (lines 839, 936) — HIGH
- FMEA core (Process / Design / System types, RPN or AP scoring). FMEARow has S/O/D scoring with `compute_action_priority` AIAG/VDA static method, optional `hypothesis_link` to `core.Hypothesis` (Bayesian bridge), and `spc_measurement` text field for closed-loop SPC integration. **FKs IN (cross-app):** `loop/models.py` (FMIS rows reference `agents_api.FMEARow` twice — lines 1204, 1447), `safety/models.py` (line 736). Within agents_api: `Risk.fmea`, `Risk.fmea_row`, `AFE.fmea`. Used by `fmea_views.py` (1,589 LOC, 27 funcs), `notebook_views.py`, `learn_views.py`, `loop/services.py`. Critical extraction with cross-app FK rewiring.

#### RCASession (line 1143) — HIGH
- Special-cause causal-chain investigation with state machine (`VALID_TRANSITIONS`, `TRANSITION_REQUIREMENTS`). FKs out: `User × 2`, `Site`, `Tenant`, `Project`, `A3Report`. `embedding` BinaryField for similarity search. **FKs IN (agents_api):** `NonconformanceRecord.rca_session`, `CAPAReport.rca_session`. Used by `rca_views.py` (1,057 LOC, 17 funcs), `a3_views.py`, `notebook_views.py`, `qms_views.py`, `learn_views.py`, `iso_views.py`. Per architecture, becomes the canonical transition app `qms_investigation/`.

#### IshikawaDiagram / CEMatrix (lines 1342, 1445)
- Common-cause analysis tools. Both leaf models with no incoming FKs. CEMatrix has a `compute_totals` method. Used by `ishikawa_views.py` (158 LOC, 6 funcs), `ce_views.py` (155 LOC, 6 funcs), `notebook_views.py`. Clean extraction to `qms_investigation/`.

#### ValueStreamMap (line 1559) — HIGH
- Largest single non-CAPA Lean model. 16 JSON fields (process_steps, inventory, customers, suppliers, kaizen_bursts, work_centers, metric_snapshots, etc.) plus calculated PCE/lead-time. Self-FK `paired_with` (current ↔ future state). **FKs IN (agents_api):** `HoshinProject.source_vsm`, `PlantSimulation.source_vsm`. Used by `vsm_views.py` (433 LOC, 14 funcs), `xmatrix_views.py`, `learn_views.py`, `qms_views.py`, `plantsim_views.py`. Per migration plan: VSM is being rebuilt as a cockpit UX.

#### Section/Assessment/LearnSession (lines 1806, 1842, 1871)
- Course progress tracking. `LearnSession.project` FK to `core.Project` for sandbox sessions. Tightly coupled to `learn_views.py` (2,450 LOC, 32 funcs) and `learn_content/` subpackage (14,556 LOC of content data). Extract as new `learn/` app.

#### PlantSimulation (line 1923)
- DES layout + sim run records. FK `source_vsm` for VSM-derived simulators. Used by `plantsim_views.py` (391 LOC). Per memory, Eric is upgrading simulators in parallel — extraction must respect this.

#### Site (line 2020) — **CRITICAL HUB**
- The architectural chokepoint. Owned by `core.Tenant`. 9 contact/text fields plus `is_active`.
- **Incoming FK count from `agents_api/models.py`:** **15 distinct models** declare `site = models.ForeignKey("agents_api.Site", ...)` or equivalent: `A3Report`, `AFE`, `CAPAReport`, `Checklist`, `ControlledDocument`, `ControlPlan`, `CustomerComplaint`, `FMEA`, `InternalAudit`, `MeasurementEquipment`, `NonconformanceRecord`, `ProjectTemplate`, `RCASession`, `Risk`, `TrainingRequirement`. Plus `Employee.site` and `SiteAccess.site` and `AnnualObjective.site` and `HoshinProject.site` — bringing the **internal incoming-FK count to 19**.
- **Cross-app incoming FKs** (string lookups via grep `"agents_api.Site"`): `graph/models.py:520`, `loop/models.py:962`, `loop/models.py:1190`, `safety/models.py:69, 220, 354`. **3 outside apps** point at it.
- `SiteAccess` (line 2072) is the per-user permission table for sites — `(site, user)` unique-together with role choices. Used by `iso_views.py` and `hoshin_views.py` to gate per-site visibility.
- **Why this matters:** Site is the multi-site enterprise scoping wrapper around Tenant. It can't be moved before its consumers; if it moves at all, it should be the very last extraction OR it should remain in a new `qms_core/` shared app to avoid forcing every QMS app to depend on the catch-all.

#### HoshinProject + ProjectTemplate (lines 2135, 2323) — HIGH
- HoshinProject is OneToOne with `core.Project` — wraps the core project with CI-specific fields (project_class, project_type, opportunity, hoshin_status, fiscal_year, savings target, monthly_actuals JSON, baseline_data JSON, kaizen_charter JSON). FKs: `Site`, `ValueStreamMap` (source_vsm). **FKs IN:** `ResourceCommitment.project`, `AFE.hoshin_project`, `HoshinKPI.derived_from`. Used by `hoshin_views.py` (2,094 LOC, 36 funcs), `vsm_views.py`, `qms_views.py`, `xmatrix_views.py`, `iso_views.py`. ProjectTemplate is the related reusable template.

#### ActionItem (line 2406)
- Per migration plan: **superseded by `loop.Commitment` (LOOP-001 §3)** but still in production views. Used by `action_views.py` (65 LOC, 2 funcs), `fmea_views.py`, `rca_views.py`, `a3_views.py`, `hoshin_views.py`. Has its own `action_urls.py`. Includes a self-FK `depends_on` and a `source_type/source_id` reverse pattern. **Action: deprecate, delete with old templates.**

#### Employee + ResourceCommitment + ActionToken (lines 2511, 2573, 2669)
- QMS-002 Resource Management. Employee is the non-user contact record (one per email per tenant). ResourceCommitment is FK to Employee + HoshinProject with role/status state machine. ActionToken is the email-token system for non-users to confirm/decline assignments.
- **Cross-app FKs IN:** `loop/models.py:424` references `agents_api.Employee`, `loop/models.py:982` again. `safety/models.py` (lines 260, 344) references `Employee` for safety event tracking.
- Employee belongs in a `qms_core/` shared app or `accounts/`-adjacent module.

#### StrategicObjective / AnnualObjective / HoshinKPI / XMatrixCorrelation (lines 2746, 2823, 2913, 3344) — Hoshin X-matrix
- HoshinKPI is the most complex of the four — has a `METRIC_CATALOG` class dict with 14+ metric definitions, an `effective_actual` property that pulls from `monthly_actuals`, `weighted_avg`, OR a fresh `DSWResult` query (line 3285) by `calculator_result_type`. This is the most cross-cutting computation in agents_api outside of dispatch — it bridges Hoshin to the analysis storage and back. The KPI dispatch tightly couples Hoshin to DSWResult; extracting Hoshin requires that the workbench artifact API is in place.
- XMatrixCorrelation has post-delete signal handlers that clean orphan correlations when StrategicObjective/AnnualObjective/HoshinProject/HoshinKPI are deleted.

#### NonconformanceRecord + NCRStatusChange (lines 3440, 3688) — HIGH
- NCR per ISO 9001 §10.2. State machine `TRANSITIONS` + `TRANSITION_REQUIRES`. FKs out: `User × 4` (owner, created_by, raised_by, assigned_to, approved_by), `Site`, `Tenant`, `Project`, `RCASession`, `Report` (capa_report), `SupplierRecord`. `linked_process_node_ids` JSONField (graph linkage opt-in). M2M `files.UserFile`. **FKs IN (cross-app):** `loop/models.py:1855`, `graph/views.py:392` (graph integration), plus `CustomerComplaint.ncr` and `AuditFinding.ncr` within agents_api. Used by `iso_views.py` heavily.

#### CAPAReport + CAPAStatusChange (lines 3723, 3942) — HIGH
- Standalone CAPA model (separate from generic Report). Has its own state machine `TRANSITIONS`/`TRANSITION_REQUIRES`. FK to `RCASession`. Used by `capa_views.py` (711 LOC, 12 funcs) and `iso_views.py`. Per migration plan: **flagged as deprecated, replaced by Investigation + ForgeDoc CAPA generator**. Decision in §J.

#### InternalAudit + AuditFinding + ManagementReviewTemplate + ManagementReview (lines 3977, 4076, 4339, 4461) — HIGH
- The full ISO 9001 §9.2 + §9.3 audit + management review surface. ManagementReviewTemplate has a `DEFAULT_SECTIONS` class constant of 10 ISO 9001:2015 §9.3.2 inputs. ManagementReview has `data_snapshot` JSON for auto-captured metrics at review time. Used heavily by `iso_views.py` (4,874 LOC) — these models are the meat of the iso_views monolith.

#### TrainingRequirement + TrainingRecord + TrainingRecordChange (lines 4133, 4214, 4295)
- TWI competency model (TRN-001 §3) — has a `certification_status` property. FK to `ControlledDocument` for SOP-driven training. M2M `files.UserFile` for training artifacts.
- **Cross-app FK IN:** `loop/models.py:879` and `loop/views.py:2386`, `loop/services.py:253`, `loop/readiness.py:171` all import `TrainingRecord` / `TrainingRequirement`. The loop app's commitment workflow uses training records as readiness signals.

#### ControlledDocument + DocumentRevision + DocumentStatusChange (lines 4534, 4702, 4745) — HIGH
- ISO 9001 §7.5 document control. Has its own state machine. **M2M `graph.ProcessNode`** (`linked_process_nodes`) — line 4591. M2M `files.UserFile`. FK `source_study` to `core.Project`.
- **Cross-app FKs IN:** `loop/models.py:694, 884, 971`, `loop/views.py:2361`, `loop/services.py:297`, `graph/views.py:392`. Heavily integrated into loop's standardize workflow.

#### SupplierRecord + SupplierStatusChange (lines 4789, 4921) — HIGH
- Supplier QA per ISO 9001 §8.4 with state machine and TRANSITION_REQUIRES.
- **Cross-app FK IN:** `loop/models.py:1850, 2105` (supplier accountability workflow), `graph/views.py:392`.

#### QMSFieldChange (line 4965)
- Cross-cutting field-level audit log keyed by `(record_type, record_id)`. Used as a generic change log by NCR / Audit / Document / Supplier. Belongs in a shared `qms_core/` app, not in any single domain.

#### CustomerComplaint (line 5022) — HIGH
- FK chain: `NCR`, `CAPAReport`, `Project`. Has `linked_process_node_ids` opt-in graph linkage. Used by `iso_views.py`, `loop/views.py:2404`.

#### Risk (line 5213) — HIGH
- ISO 9001 §6.1 organizational risk register. Computed `risk_score = likelihood × impact`. FKs to `FMEA`, `FMEARow` (auto-promoted from process FMEAs), `Project`. Per `feedback_afe_policy.md`: AFEs only flow through Hoshin projects, NOT directly from Risk register.

#### AFE + AFEApprovalLevel (lines 5386, 5562)
- Authorization for Expenditure with N-level approval chain. FKs to `HoshinProject`, `Risk`, `FMEA`, `Checklist`, `Project`. AFEApprovalLevel has FK to `ElectronicSignature` for CFR-compliant sign-off. Per memory: AFE's only legitimate flow is through Hoshin.

#### MeasurementEquipment (line 5647) — HIGH
- ISO 9001 §7.1.5 calibration register. **FK to `graph.ProcessNode`** (line 5724). Has Weibull reliability fields (mtbf_hours, weibull_shape, weibull_scale, failure_history JSON). Per OLR-001 §11.
- **Cross-app FKs IN:** `graph/tests_qms.py` (×3), `graph/integrations.py:601`, `loop/evaluator.py:309`. Graph causal-evidence integration depends on this model.

#### AuditChecklist (line 5800) and Checklist + ChecklistExecution (lines 5843, 5923)
- AuditChecklist is the older audit-specific checklist; Checklist is the generic Gawande-style prompt-response template (READ-DO / DO-CONFIRM) with typed responses (yes_no, pass_fail_na, text, numeric, select, file, signature). ChecklistExecution links any checklist to any entity via `(entity_type, entity_id)`. **AuditChecklist may be redundant** with the more general Checklist — likely DELETE candidate.

#### ISODocument + ISOSection (lines 6027, 6128) — MED
- Structured ISO document authoring (separate from ControlledDocument). OneToOne with ControlledDocument when published. ISOSection has typed `section_type` (heading/paragraph/definition/reference/image/table/checklist/signature_block) and `embedded_media` JSON for whiteboard exports. Used by `iso_doc_views.py` (716 LOC, 11 funcs) and `loop/views.py:1048`, `loop/services.py:150`.

#### ElectronicSignature (line 6232) — HIGH (compliance-critical)
- Inherits from `SynaraImmutableLog` — participates in the hash chain. 21 CFR Part 11 + ISO 9001:2015 §7.5.3 compliance. FK `signer` (PROTECT). Polymorphic `(document_type, document_id)`. Has custom `_compute_hash_chain` and `_compute_entry_hash` for tamper detection. Belongs in `qms_core/` (shared infrastructure).

#### QMSAttachment (line 6390)
- Generic file-attachment table for QMS records. `(entity_type, entity_id)` polymorphic. ENTITY_MODEL_MAP class constant maps the 8 supported entity types to model class names. Cross-cutting → `qms_core/`.

#### ControlPlan + ControlPlanItem (lines 6488, 6587) — MED
- ISO 9001 §8.5.1 / IATF 16949 §8.5.1.1 control plans. ControlPlanItem has FKs to `loop.FMISRow`, `graph.ProcessNode`, `MeasurementEquipment`. Tight coupling to loop and graph.

#### RackSession (line 6720)
- ForgeRack persisted state. Used by `rack_views.py` (848 LOC, 27 funcs). Demo-only per `project_forgerack_architecture.md`. Likely belongs in its own `forgerack/` app or stays as a sandbox/demo construct.

### B.3 Site model — full hub treatment

Because Site is the load-bearing wire across the agents_api domain models:

| Source of FK | File | Line | Relation |
|---|---|---|---|
| `A3Report.site` | models.py | 620 | SET_NULL |
| `FMEA.site` | models.py | 871 | SET_NULL |
| `RCASession.site` | models.py | 1184 | SET_NULL |
| `ProjectTemplate.site` | models.py | 2340 | SET_NULL |
| `HoshinProject.site` | models.py | 2181 | SET_NULL |
| `Employee.site` | models.py | 2532 | SET_NULL |
| `AnnualObjective.site` | models.py | 2849 | SET_NULL |
| `NonconformanceRecord.site` | models.py | 3472 | SET_NULL |
| `CAPAReport.site` | models.py | 3777 | SET_NULL |
| `InternalAudit.site` | models.py | 3995 | SET_NULL |
| `TrainingRequirement.site` | models.py | 4144 | SET_NULL |
| `ControlledDocument.site` | models.py | 4551 | SET_NULL |
| `CustomerComplaint.site` | models.py | 5068 | SET_NULL |
| `Risk.site` | models.py | 5249 | SET_NULL |
| `AFE.site` | models.py | 5410 | SET_NULL |
| `MeasurementEquipment.site` | models.py | 5680 | SET_NULL |
| `Checklist.site` | models.py | 5871 | SET_NULL |
| `ControlPlan.site` | models.py | 6514 | SET_NULL |
| `SiteAccess.site` | models.py | 2085 | CASCADE |

Plus cross-app FKs to `agents_api.Site`:
| App | File | Line | Purpose |
|---|---|---|---|
| `graph/` | `models.py:520` | Process graph site scoping |
| `loop/` | `models.py:962, 1190` | Loop / commitment site scoping |
| `safety/` | `models.py:69, 220, 354` | Safety event / hazard site scoping |

**Total incoming FKs to Site: 19 internal + 5 cross-app references = 24 callers.** This is the architectural chokepoint of the entire QMS surface. Recommendations in §I.

---

## Section C — View module inventory

The 33 top-level view files in `agents_api/`. LOC and function counts come from direct file inspection. "Models touched" lists what each file imports from `.models`.

### C.1 Top tier (highest LOC / most coupling)

| File | LOC | Funcs | Models touched | URL prefix served | Domain | Proposed home | Forge replacement |
|---|---|---|---|---|---|---|---|
| `iso_views.py` | **4,874** | **85** | NCR, NCRStatusChange, CAPAReport, Audit, AuditFinding, Training×3, ManagementReview×2, ControlledDocument×3, Supplier×2, QMSFieldChange, QMSAttachment, MeasurementEquipment, AuditChecklist, Checklist, ChecklistExecution, ControlPlan, ControlPlanItem, ElectronicSignature, FMEA, AFE, AFEApprovalLevel, CustomerComplaint, RCASession, Risk | `/api/iso/` | `qms_*` (multiple) | **SPLIT across qms_audit/qms_documents/qms_training/qms_suppliers/qms_nonconformance/qms_capa/qms_measurement** | none — pure view code |
| `learn_views.py` | 2,450 | 32 | SectionProgress, AssessmentAttempt, LearnSession, RCASession, FMEA, FMEARow, A3Report, ValueStreamMap | `/api/learn/` | learn | `learn/` app | none |
| `experimenter_views.py` | 2,286 | 22 | (none — direct DOE compute) | `/api/experimenter/` | analysis | `workbench/handlers/` | **forgedoe** |
| `hoshin_views.py` | 2,094 | 36 | ActionItem, Checklist, ChecklistExecution, Employee, HoshinProject, ProjectTemplate, ResourceCommitment, Site, SiteAccess, ValueStreamMap, AnnualObjective, XMatrixCorrelation | `/api/hoshin/` | strategy | `hoshin/` | none |
| `autopilot_views.py` | 1,918 | 14 | (uses dsw/common helpers, SavedModel) | `/api/dsw/autopilot/` | analysis | `workbench/` | **forgestat** for compute |
| `notebook_views.py` | 1,621 | 26 | FMEA, Board, CEMatrix, DSWResult, IshikawaDiagram, RCASession (lazy imports) | `/api/notebooks/` | core | `core/` (Notebook is in core) | none |
| `fmea_views.py` | 1,589 | 27 | FMEA, ActionItem, CAPAReport, FMEARow, RCASession, Risk | `/api/fmea/` | risk | `qms_risk/` | none |
| `spc_views.py` | 1,547 | 15 | FMEARow (lazy) | `/api/spc/` | analysis | `workbench/handlers/` | **forgespc** (ready) |
| `a3_views.py` | 1,389 | 23 | A3Report, ActionItem, Board, DSWResult, RCASession | `/api/a3/` | a3 | `qms_a3/` (rebuild) | none |
| `synara_views.py` | 1,136 | 32 | (operates on agents_api/synara/ engine) | `/api/synara/` | analysis (Bayesian) | `workbench/handlers/` or stays | **forgesia** (pending __init__ exports) |
| `xmatrix_views.py` | 1,069 | 13 | AnnualObjective, HoshinKPI, HoshinProject, Site, StrategicObjective, ValueStreamMap, XMatrixCorrelation | `/api/hoshin/xmatrix/` (mixed into hoshin_urls?) | strategy | `hoshin/` | none |
| `whiteboard_views.py` | 1,057 | 23 | Board, BoardGuestInvite, BoardParticipant, BoardVote | `/api/whiteboard/` | whiteboard | `whiteboard/` | none |
| `rca_views.py` | 1,057 | 17 | ActionItem, CAPAReport, RCASession, A3Report (lazy) | `/api/rca/` | investigation | `qms_investigation/` | none |
| `harada_views.py` | 909 | 14 | (uses harada_tasks model state via JSON) | `/api/harada/` | learn | new `harada/` or stays | none |
| `rack_views.py` | 848 | 27 | RackSession | `/api/rack/` | rack | `forgerack/` (or DELETE) | none |
| `report_views.py` | 802 | 13 | Board, DSWResult, RCASession, Report | `/api/reports/` | reports | new sink `reports/` (rebuild) | uses `dsw/chart_render` (legacy) |

### C.2 Middle tier

| File | LOC | Funcs | Models touched | Domain | Proposed home |
|---|---|---|---|---|---|
| `iso_doc_views.py` | 716 | 11 | ControlledDocument, DocumentRevision, ISODocument, ISOSection, Board (lazy) | `qms_documents` | `qms_documents/` |
| `capa_views.py` | 711 | 12 | CAPAReport, CAPAStatusChange, NonconformanceRecord, QMSFieldChange, RCASession | `qms_capa` | `qms_capa/` (or DELETE) |
| `triage_views.py` | 512 | 10 | AgentLog, TriageResult | `triage` | `triage/` (new) |
| `forecast_views.py` | 456 | 7 | (none — direct compute) | analysis | `workbench/handlers/` (uses `dsw/common`) |
| `investigation_views.py` | 432 | 11 | **None from agents_api** — uses `core.Investigation`, `core.InvestigationMembership`, `core.InvestigationToolLink`, plus `agents_api/investigation_bridge.py` | investigation | `qms_investigation/` (move bridge with it) |
| `workflow_views.py` | 433 | 11 | Workflow | agent | DELETE or `automation/` |
| `plantsim_views.py` | 391 | 8 | PlantSimulation, ValueStreamMap | simulation | `simulation/` (Eric owns) |
| `guide_views.py` | 369 | 4 | (LLM rate-limited guide; no model FKs) | chat-adjacent | `chat/` or stay |
| `dsw_views.py` | 314 | 14 | TriageResult (lazy) | analysis | `workbench/` (it's the wrapper around `dsw.dispatch` → `analysis.dispatch`) |
| `qms_views.py` | 216 | 1 | FMEA, A3Report, CAPAReport, FMEARow, HoshinProject, RCASession, ValueStreamMap | dashboard | `qms_core/` (cross-app dashboard view) |
| `token_views.py` | 214 | 7 | ActionToken (via Employee) | infrastructure | `qms_core/` |
| `ishikawa_views.py` | 158 | 6 | IshikawaDiagram | investigation | `qms_investigation/` |
| `ce_views.py` | 155 | 6 | CEMatrix | investigation | `qms_investigation/` |
| `action_views.py` | 65 | 2 | ActionItem | workflow | DELETE (LOOP-001) |
| `vsm_views.py` | 433 | 14 | HoshinProject, ValueStreamMap | lean | `vsm/` (new) |
| `views.py` | 441 | (DRF) | (none — directly imports `researcher.agent`, `writer.agent`, `editor.agent`, `experimenter.agent`, `eda.agent`) | `agent_layer` | DELETE — see §H |

### C.3 The 4,874-line `iso_views.py` monolith

`iso_views.py` is the largest single view file in agents_api and the biggest extraction risk. It owns 85 view functions covering **eleven distinct ISO 9001 clause areas** that should each become their own app:

- **§10.2 NCR + CAPA (clauses 8.7, 10.2)** — `ncr_*` views
- **§9.2 Internal audit (clause 9.2)** — `audit_*` views, `audit_finding_*`, `audit_clause_coverage`, `audit_apply_checklist`, `audit_checklist_*`
- **§7.2 Training (clause 7.2)** — `training_*` views
- **§9.3 Management review (clause 9.3)** — `review_*`, `review_template_*`, `review_narrative`
- **§7.5 Document control (clause 7.5)** — `document_*`
- **§8.4 Supplier management (clause 8.4)** — `supplier_*`
- **§7.1.5 Calibration (clause 7.1.5)** — `equipment_*`, gage R&R links
- **§9.1.2 Customer complaints** — `complaint_*`
- **§6.1 Risk register** — `risk_*`
- **AFE + Approval chain** — `afe_*`
- **Control Plan** — `control_plan_*`
- **Checklist execution** — `checklist_*`

The file also imports `evidence_bridge.create_tool_evidence` (a 638-line cross-tool module) and `permissions.qms_can_edit / qms_queryset / qms_set_ownership`. The permissions helpers are agents_api-internal but several of them are imported from `loop/supplier_views.py` via `from agents_api.permissions import get_tenant`. **Splitting this file is the highest single-CR effort in the extraction.** Recommendation: extract one ISO clause at a time, leaving the helpers behind in agents_api as a temporary `qms_core/` shim until everything has been moved.

---

## Section D — Sub-package inventory

`agents_api/` has 5 active subdirectories under it (excluding `migrations/`, `__pycache__/`).

### D.1 `agents_api/dsw/` — 73,749 LOC

| File | LOC | Notes |
|---|---|---|
| `dispatch.py` | **17** | **THIN SHIM** — re-exports from `agents_api.analysis.dispatch` (CR d9c36a0b, 2026-03-26) |
| `common.py` | 3,084 | Active — shared helpers (`sanitize_for_json`, `_narrative`, `_clean_for_ml`, `_auto_train`); imported from `spc_views.py`, `autopilot_views.py`, `forecast_views.py`, `pbs_engine.py`, `ml_pipeline.py` |
| `endpoints_data.py` | 1,832 | Active — `upload_data`, `execute_code`, `generate_code`, `analyst_assistant`, `transform_data`, `download_data`, `retrieve_data`, `triage_data`, `triage_scan` (called from `dsw_views.py`) |
| `endpoints_ml.py` | 1,702 | Active — `dsw_from_intent`, `dsw_from_data`, `dsw_download`, `list_models`, `save_model_from_cache`, etc. (imported by `dsw_urls.py`) |
| `chart_render.py` | 59 | Active — imported from `report_views.py`, `a3_views.py`, and **mirrored to `analysis/chart_render.py` via `from agents_api.dsw.chart_render import *`** |
| `chart_defaults.py` | 459 | Active — same mirror pattern from `analysis/chart_defaults.py` |
| `standardize.py` | 552 | Active — imported from `spc_views.py` |
| `bayesian.py` (single file) | 5,308 | LEGACY — superseded by `analysis/bayesian/` package |
| `bayesian/*.py` (subpackage) | ~5,400 | LEGACY duplicate of `analysis/bayesian/` |
| `stats_*.py` (parametric, nonparametric, posthoc, regression, advanced, exploratory, quality) | ~21,988 | LEGACY — migration plan flags as "delete when forgestat replaces" |
| `spc.py` (5,453) | LEGACY — duplicates top-level `agents_api/spc.py` AND duplicates `analysis/forge_spc.py` AND `forgespc` package |
| `spc_pkg/*.py` (`shewhart`, `multivariate`, `capability`, `helpers`, `conformal`, `advanced`) | ~4,500 | LEGACY |
| `ml.py` (4,624) + `viz.py` (2,931) + `siop.py` (2,482) + `simulation.py` (1,090) + `reliability.py` (1,521) + `d_type.py` (2,929) + `exploratory/*` | ~22k | LEGACY |
| `registry.py` | 576 | Inspection of imports needed to confirm dead — likely superseded by `analysis/registry.py` |

**Forge package overlap:** ≥95% for stats/spc/bayesian/ml/viz subtrees (forgestat + forgespc + forgesia + forgeviz cover all of it).

**Recommended action:** **SPLIT.** Keep `dispatch.py` (shim), `common.py`, `endpoints_data.py`, `endpoints_ml.py`, `chart_render.py`, `chart_defaults.py`, `standardize.py` for now (they have live callers). Delete or relocate the rest after forge wiring is complete.

### D.2 `agents_api/analysis/` — 78,544 LOC

| File / Subdir | LOC | Notes |
|---|---|---|
| `dispatch.py` | **583** | **CANONICAL** dispatcher — `analysis_urls.py` routes `/api/analysis/run/` to it; `dsw/dispatch.py` re-exports from here |
| `excel_export.py` | (live) | Used in `analysis_urls.py` for xlsx export endpoint |
| `chart_render.py` | 7 | **WRAPPER** — only `from agents_api.dsw.chart_render import *` |
| `chart_defaults.py` | 11 | **WRAPPER** — only `from agents_api.dsw.chart_defaults import *` |
| `common.py` | (active — used by `dispatch.py`) | |
| `standardize.py` | (active) | Mirror with own copy |
| `registry.py` | 577 | Active analysis-side registry |
| `forge_*.py` (forge_spc, forge_stats_quality/advanced/anova/regression/exploratory/msa, forge_bayesian, forge_misc, forge_ml) | ~7,800 | **NEW forge bridge functions** — wrap forge package calls with the SVEND result schema |
| `stats/`, `bayesian/`, `spc/`, `exploratory/`, `ml/`, `reliability/`, `msa/`, `simulation/`, `siop.py`, `pbs/`, `ishap/`, `anytime/`, `drift/`, `education/`, `d_type/`, `quality_econ/`, `viz/` | ~50k+ | Restructured equivalents of `dsw/` content; canonical |

**Critical correction to prior finding:** The earlier session reported `dsw/` and `analysis/` as 100% duplicates and that one could be deleted by checking what `dsw_views.py` imports. The reality is more nuanced:

1. **`analysis/` is the canonical tree.** `analysis/dispatch.py` is the active analysis runner. `dsw/dispatch.py` is a 17-line shim that re-exports from `analysis/dispatch`.
2. **`dsw_views.py` imports `from .dsw.dispatch`** — but that immediately redirects to `analysis.dispatch`. So the *runtime path* is dsw_views → dsw/dispatch (shim) → analysis/dispatch.
3. **`dsw/` still has live tenants** for `common.py`, `endpoints_data.py`, `endpoints_ml.py`, `chart_render.py`, `chart_defaults.py`, `standardize.py` — multiple top-level views import directly from these.
4. **`analysis/chart_render.py` and `chart_defaults.py` are wrappers** around `dsw/chart_*` — meaning the chart side is currently dsw-canonical, opposite of the dispatch side.
5. **Tests heavily use both sides:** `tests/test_stats_*.py` imports from `agents_api.analysis.*`; `tests/test_dsw_views_behavioral.py` imports from `agents_api.dsw_views`.

**Recommended action:** MIGRATE/DELETE in stages.
- (a) Wire forge packages into `analysis/forge_*.py` bridges for the remaining handlers.
- (b) Move `dsw/common.py`, `endpoints_data.py`, `endpoints_ml.py`, `standardize.py`, `chart_render.py`, `chart_defaults.py` into `analysis/` (or into `workbench/handlers/`).
- (c) Update view imports to use `analysis.*` exclusively.
- (d) Then delete `dsw/` entirely.

### D.3 `agents_api/synara/` — 3,243 LOC

| File | LOC |
|---|---|
| `__init__.py` | 99 |
| `belief.py` | 343 |
| `dsl.py` | 665 |
| `kernel.py` | 368 |
| `llm_interface.py` | 459 |
| `logic_engine.py` | 871 |
| `synara.py` | 438 |

**Forge package overlap:** ≥95% — `forgesia` is the standalone package replacement, pending `__init__` export fixes per migration plan §"Tech Debt".

**Cross-app callers:** `graph/synara_adapter.py` imports `from agents_api.synara.belief import BeliefEngine` and `from agents_api.synara.kernel import (...)`. `graph/service.py:281` imports `Evidence as SynaraEvidence`. `graph/tests_synara.py` imports `Evidence` from kernel. These are the wires that need to swap to `forgesia` before the synara/ subpackage can be deleted.

**Recommended action:** MIGRATE then DELETE — wire `forgesia`, update graph/ imports, delete `agents_api/synara/`.

### D.4 `agents_api/learn_content/` — ~14,556 LOC

| File | LOC | Notes |
|---|---|---|
| `_datasets.py` | 6,302 | Bundled training datasets |
| `_registry.py` | 181 | Course / module registry |
| `__init__.py` | 8 | |
| 13 content files (statistical_inference, data_fundamentals, foundations, advanced_statistics, machine_learning, experimental_design, dsw_mastery, pbs_mastery, advanced_methods, case_studies, capstone, critical_evaluation, causal_inference) | ~7,800 | Course content as Python data structures |

Plus `learn_content.py` (8,024 LOC) at the top level — a separate file that aggregates / drives content.

**No models.** Pure content data. Tightly coupled to `learn_views.py` and the SectionProgress / AssessmentAttempt / LearnSession models.

**Recommended action:** MIGRATE — move wholesale into a new `learn/` app.

### D.5 `agents_api/tests/` subdirectory

49 test files, ~22,863 LOC. Mix of behavior tests, unit tests, smoke / sweep tests, golden file tests. See §F for assessment.

---

## Section E — URL surface

`svend/urls.py` mounts the following agents_api URL files (verified by grep at lines 409-460):

| Mount path | Source URL file | LOC | Notes |
|---|---|---|---|
| `/api/agents/` | `agents_api/urls.py` | 16 | Original agent dispatch (researcher, writer, editor, experimenter, eda — coder commented out) — loaded but likely dead in production |
| `/api/workflows/` | `workflow_urls.py` | 11 | Workflow CRUD — likely dead |
| `/api/dsw/` | `dsw_urls.py` | 79 | DSW analysis + ML model save/load + autopilot |
| `/api/analysis/` | `analysis_urls.py` | 15 | New analysis dispatcher: `run/`, `export/xlsx/<id>/`, `export/xlsx/` |
| `/api/triage/` | `triage_urls.py` | 16 | Data cleaning |
| `/api/forecast/` | `forecast_urls.py` | 10 | Time series forecasting |
| `/api/experimenter/` | `experimenter_urls.py` | 31 | DOE design |
| `/api/spc/` | `spc_urls.py` | 31 | SPC charts, capability, gage R&R |
| `/api/synara/` | `synara_urls.py` | 128 | Belief engine API |
| `/api/whiteboard/` | `whiteboard_urls.py` | 83 | Boards, voting, guest invites |
| `/api/guide/` | `guide_urls.py` | 11 | AI decision guide (rate-limited) |
| `/api/reports/` | `report_urls.py` | 37 | CAPA, 8D, generic reports |
| `/api/plantsim/` | `plantsim_urls.py` | 36 | Plant simulator |
| `/api/learn/` | `learn_urls.py` | 47 | Learning module |
| `/api/fmea/` | `fmea_urls.py` | 80 | FMEA CRUD + Bayesian linking |
| `/api/hoshin/` | `hoshin_urls.py` | 172 | Hoshin Kanri (enterprise) — biggest URL file |
| `/api/qms/` | `qms_urls.py` | 18 | QMS cross-module dashboard |
| `/api/iso/` | `iso_urls.py` | 218 | ISO 9001 QMS endpoints (NCR/audit/training/document/supplier/etc.) — second biggest |
| `/api/capa/` | `capa_urls.py` | 16 | CAPA standalone |
| `/api/iso-docs/` | `iso_doc_urls.py` | 50 | ISO Document Creator |
| `/api/actions/` | `action_urls.py` | 21 | ActionItem update/delete |
| `/api/investigations/` | `investigation_urls.py` | 48 | Investigation CRUD |
| `/api/notebooks/` | `notebook_urls.py` | 63 | Notebook lifecycle (NB-001) |
| `/api/harada/` | `harada_urls.py` | 38 | Harada method |
| `action/<token>/` | `token_urls.py` | 12 | ActionToken (no auth) for non-users |

Plus direct path imports from `svend/urls.py` line 262 / 267 for `rack_views.rack_compute` and `rack_views.rack_export_runsheet`, and line 92 for `a3_views.remove_diagram`.

**Note: There is no `vsm_urls.py` and no `xmatrix_urls.py` and no `rca_urls.py`** — these are mounted via other URL files or via `qms_urls.py` (216 lines, 1 def, dispatching to many models). VSM CRUD endpoints are inside `hoshin_urls.py` or via direct view import. RCA endpoints come through `investigation_urls.py` or `iso_urls.py` (`ncr_launch_rca`). Confirmation needed during gap analysis.

---

## Section F — Test surface

### F.1 Counts

- **Top-level test files in `agents_api/`** (`*_tests.py` + `tests.py`): 19 files, 22,482 LOC
- **`agents_api/tests/` subdirectory:** 49 files, 22,863 LOC
- **Total:** 68 test files, ~45,345 LOC of test code

Test function counts (sample of top-level files):
- `iso_tests.py`: **351** test functions
- `hoshin_deep_tests.py`: 91
- `whiteboard_tests.py`: 83
- `tests.py`: 75
- `hoshin_tests.py`: 49
- `vsm_tests.py`: 47
- `integration_tests.py`: 43
- `esig_tests.py`: 32
- `qms_phase3_tests.py`: 29
- `audit_bugfix_tests.py`: 27

### F.2 TST-001 §10.6 compliance assessment (sampled)

**Behavioral (TST-001 compliant):**
- `iso_tests.py` — real CRUD-and-workflow tests with assertions on response status codes, JSON content, and side effects on the database. Tests transitions through state machines. **PASS.**
- `tests/test_dsw_views_behavioral.py` — unit tests on `_read_csv_safe` with real BytesIO inputs and encoding fallbacks. Real behavior. **PASS.**

**Sweep / smoke (TST-001 violations):**
- `tests/test_endpoint_smoke.py` (712 LOC) — explicitly tests "every endpoint responds correctly" with tier-gated 401/403 checks and "doesn't return 500" assertions. Per its own docstring, tests three properties: "Unauthenticated requests are rejected", "Authenticated requests don't return 500", "Invalid input returns 400, not 500". This is the canonical sweep pattern §10.6 prohibits. **FAIL.**
- `tests/test_t1_deep.py` (1,942 LOC), `tests/test_t2_views_smoke.py` (338 LOC), `tests/test_t3_*` series — naming pattern (T1/T2/T3) suggests auto-generated coverage padding, likely sweep-flavored. **LIKELY FAIL** without further inspection.

**Coverage tests (questionable):**
- `tests/test_bayesian_coverage.py`, `test_ml_coverage.py`, `test_pbs_coverage.py`, `test_reliability_coverage.py`, `test_remaining_coverage.py`, `test_spc_coverage.py`, `test_stats_coverage.py`, `test_viz_coverage.py`, `test_vis_compliance.py`, `test_bounds_exhaustive.py` — names suggest existence-test or coverage-padding patterns. **LIKELY FAIL** TST-001.

### F.3 Coverage estimate

**Unknown — needs `coverage run`.** A `coverage.json` file exists per CLAUDE.md memory but I did not run `coverage run` against this snapshot. Per the prior calibration session note, the `measure_coverage` command reads `coverage.json` for the calibration tab. Symbol coverage (compliance tab) is a separate metric.

### F.4 Implications for extraction safety

Given that:
1. ~half of agents_api test files are in the prohibited sweep pattern,
2. only `iso_tests.py` and a handful of behavioral files actually exercise real behavior,
3. the new architecture demands TST-001-compliant behavior tests,

**Each extraction CR needs ~5-7 new behavior tests written before the move**, per `qms_architecture.md` §6.2. The existing `iso_tests.py` is the gold standard pattern to follow — it can be split across the extracted apps as the reference implementation. Sweep-mode test files should NOT be migrated as-is; they should be rewritten or deleted.

---

## Section G — Cross-app dependency hot spots

### G.1 Cross-app FK targets pointing INTO agents_api models

| Source app | File | Line | Target | Notes |
|---|---|---|---|---|
| `loop/` | `models.py:424` | `Employee` | Commitment / loop staffing |
| `loop/` | `models.py:694` | `ControlledDocument` | Standardize-step doc references |
| `loop/` | `models.py:879` | `TrainingRecord` | Readiness / commitment tracking |
| `loop/` | `models.py:884` | `ControlledDocument` | Same-app reuse |
| `loop/` | `models.py:895` | `ISOSection` | Embedded SOP sections |
| `loop/` | `models.py:962` | `Site` | Site scoping |
| `loop/` | `models.py:971` | `ControlledDocument` | Document binding |
| `loop/` | `models.py:982` | `Employee` | Loop participant |
| `loop/` | `models.py:1117` | `ISOSection` | |
| `loop/` | `models.py:1190` | `Site` | |
| `loop/` | `models.py:1204` | `FMEARow` | FMIS rows reference FMEA failure modes |
| `loop/` | `models.py:1447` | `FMEARow` | |
| `loop/` | `models.py:1850` | `SupplierRecord` | |
| `loop/` | `models.py:1855` | `NonconformanceRecord` | |
| `loop/` | `models.py:2105` | `SupplierRecord` | |
| `safety/` | `models.py:69` | `Site` | |
| `safety/` | `models.py:220` | `Site` | |
| `safety/` | `models.py:260` | `Employee` | |
| `safety/` | `models.py:344` | `Employee` | |
| `safety/` | `models.py:354` | `Site` | |
| `core/models/notebook.py` | `:71, 82, 217, 228` | `DSWResult` | Notebook artifact references |
| `graph/models.py` | `:520` | `Site` | Process graph site scoping |
| `notifications/tokens.py` | `:4` | `ActionToken` (mention/coexistence) | "Separate from agents_api.ActionToken (QMS-002) by design" — not an FK, just docs |

### G.2 Cross-app code that imports from agents_api (non-FK)

Detected via `grep "from agents_api"`:

- `loop/views.py` — 9 imports of agents_api models (Employee, ISOSection, ControlledDocument, TrainingRecord/Requirement, CustomerComplaint) plus `tool_events`
- `loop/services.py` — 5 imports (ISODocument, FMEA/FMEARow, TrainingRequirement, ControlledDocument)
- `loop/readiness.py` — `TrainingRecord`
- `loop/evaluator.py` — `tool_events`, `MeasurementEquipment`
- `loop/supplier_views.py` — `from agents_api.permissions import get_tenant`
- `graph/views.py:392` — `ControlledDocument, NonconformanceRecord, SupplierRecord`
- `graph/synara_adapter.py:14-15` — `BeliefEngine` from `synara/belief`, multiple from `synara/kernel`
- `graph/service.py:281` — `Evidence as SynaraEvidence` from `synara/kernel`
- `graph/tests_synara.py:11` — `Evidence` from `synara/kernel`
- `graph/integrations.py:601` — `MeasurementEquipment`
- `graph/tests_qms.py` (3 imports) — `MeasurementEquipment`
- `notifications/webhook_views.py` — `from agents_api.permissions import get_tenant` (×2)
- `forge/tasks.py:152` — `from agents_api.views import get_shared_llm`
- `core/views.py`, `core/tests.py`, `core/management/commands/seed_nlp_demo.py` — assorted imports
- `accounts/permissions.py`, `accounts/privacy_tasks.py` — assorted imports
- `api/views.py`, `api/internal_views.py`, `api/landing_views.py`, `api/tasks.py` — internal dashboard / staff metrics
- `safety/views.py`, `safety/tests.py`, `safety/models.py` — Site / Employee references
- `syn/audit/compliance.py`, `syn/audit/management/commands/generate_calibration_cert.py` — agents_api inspection by compliance
- `syn/audit/tests/` — 11 tests reference agents_api directly
- `syn/sched/svend_tasks.py` — scheduled task definitions

### G.3 Loop ↔ agents_api coupling — hot spot

Per the prior session's report: ~10 imports. **Verified:** loop/ has **17 distinct import sites** across `views.py`, `models.py`, `services.py`, `readiness.py`, `evaluator.py`, `supplier_views.py`. Most imports are deferred (`from agents_api.models import ...` inside functions) — mitigates circular import risk but does NOT mitigate behavioral coupling. Any extraction that moves Employee, TrainingRecord, ControlledDocument, ISOSection, FMEARow, SupplierRecord, NonconformanceRecord, Site, or MeasurementEquipment **breaks loop/** unless coordinated.

### G.4 Graph ↔ agents_api coupling — hot spot

`graph/` has **9 import sites** including 3 model FKs (Site, MeasurementEquipment, plus indirectly via ControlledDocument's `linked_process_nodes` M2M back-reference) and 4 synara/ engine imports. Graph causal evidence depends on being able to query agents_api models for measurement, document, NCR, and supplier nodes. Per `qms_architecture.md` §3.2, the canonical KnowledgeGraph is in `core/`, not the graph/ app — but the graph/ app is the active integration layer that ties agents_api QMS data to the unified product knowledge graph.

### G.5 Notebook ↔ DSWResult coupling

`core/models/notebook.py` has 4 references to `agents_api.DSWResult`. Notebooks pull DSW results as evidence. The architecture replaces this with `workbench.Workbench` + `workbench.Artifact`, but the notebook is in core, so the migration path is: workbench.Artifact replaces DSWResult in production → notebook.py migrates its references → DSWResult deletable.

---

## Section H — Dead code candidates (high-confidence only)

### H.1 The original "agents" dispatch layer
- `agents_api/views.py` (441 LOC, 6 view functions): `researcher_agent`, `writer_agent`, `editor_agent`, `experimenter_agent`, `eda_agent`. Each tries `from researcher.agent import ResearchAgent` etc. — these import paths point to standalone packages that may or may not still exist outside the SVEND repo. Comment in code: `"# Custom LLMs (Qwen/DeepSeek) removed"` — strong signal of stale dispatch.
- `agents_api/urls.py` (16 LOC) routes `/api/agents/` to these functions. The `coder/` route is commented out (`# Disabled`).
- Currently still imported by `forge/tasks.py:152` (`from agents_api.views import get_shared_llm`) — but that's a helper function, not the agent dispatchers themselves. Need to verify `get_shared_llm` is still defined and used.
- **Dead-code conclusion:** the 5 agent dispatchers themselves are likely unused at the HTTP layer in production. `get_shared_llm` may be live infrastructure for forge → LLM integration. Recommendation: extract `get_shared_llm` to a helper module, delete the rest.

### H.2 `agents_api/dsw/` legacy stats files
Per migration plan §"Tech Debt" — `agents_api/dsw/stats_parametric.py`, `stats_nonparametric.py`, `stats_advanced.py`, `stats_quality.py`, `stats_regression.py`, `stats_posthoc.py`, `stats_exploratory.py`, `bayesian.py` (single file), `ml.py`, `viz.py`, `siop.py`, `simulation.py`, `reliability.py`, `d_type.py`, `spc.py`, `spc_pkg/*` are all candidates for deletion AFTER `forgestat`/`forgespc`/`forgesia`/`forgeviz` are wired into `analysis/forge_*.py` bridges. Total: ~50,000 LOC of legacy compute code.

### H.3 `analysis/chart_render.py` and `analysis/chart_defaults.py`
Per migration plan, both are 7-line and 11-line wrappers around `agents_api.dsw.chart_*`. Per migration plan §"Duplicate Computation": "Duplicate of dsw/ — delete". These are unconditionally deletable once their callers are switched to forgeviz.

### H.4 `agents_api/tests/test_*_coverage.py` family
Files named `*_coverage.py` (8 files) and `test_t1_deep.py`, `test_t2_views_smoke.py`, `test_endpoint_smoke.py` are likely TST-001 §10.6 violations. Should be reviewed and either rewritten as behavior tests or deleted as part of the test rebuild.

### H.5 `Workflow` model + `workflow_views.py` + `workflow_urls.py`
- Model has zero non-self FKs in the codebase (verified via grep).
- `workflow_views.py` imports only `Workflow` and uses it for trivial CRUD.
- Likely the original "user-defined chain of agents" feature, never adopted at scale.
- **Recommendation:** delete entirely after confirming no production traffic at `/api/workflows/`.

### H.6 `AuditChecklist` (line 5800) vs `Checklist` (line 5843)
- AuditChecklist is older, narrower (audit-only), and the more general `Checklist` model with `category` and `entity_type/entity_id` polymorphism appears to be its successor.
- Need to check if `AuditChecklist` still has live tenants.
- **Recommendation:** confirm and delete AuditChecklist.

### H.7 `RackSession` (line 6720)
- Used only by `rack_views.py` which serves `/api/rack/` and `/api/forgerack/` demo routes.
- ForgeRack is documented as "demo only" in `project_forgerack_architecture.md`.
- **Action:** keep for now (Eric is iterating on ForgeRack), but it should be in its own `forgerack/` app, not `agents_api/`.

---

## Section I — Domain clustering proposal

This synthesizes Sections B and C into a target app structure. Each cluster cites its source models and views.

### I.1 New apps to create

#### `qms_core/` — shared QMS infrastructure
- **Models:** `Site`, `SiteAccess`, `Employee`, `ActionToken`, `QMSFieldChange`, `QMSAttachment`, `Checklist`, `ChecklistExecution`, `ElectronicSignature`
- **Views:** part of `iso_views.py` (sites, team_members, qms_dashboard from `qms_views.py`), `token_views.py`, parts of checklist endpoints
- **URL prefix:** `/api/qms-core/` (or stay distributed)
- **Role:** Infrastructural — not in source/transition/sink topology
- **Effort:** L (Site is the chokepoint)
- **Sequencing:** First or LAST — see §J.1
- **Risk:** HIGH (24 incoming FKs)

#### `qms_risk/` — FMEA + Risk register
- **Models:** `FMEA`, `FMEARow`, `Risk`
- **Views:** `fmea_views.py` (1,589 LOC, 27 funcs), parts of `iso_views.py` (`risk_*`)
- **URL prefix:** `/api/fmea/`, `/api/iso/risks/` → `/api/qms-risk/`
- **Role:** Source (individual FMEAs and risks pulled into investigations), dual-purpose registry
- **Effort:** M
- **Dependencies:** `qms_core/Site`, `core.Hypothesis` (FMEARow has hypothesis_link), `loop.FMISRow` (cross-app FK from loop pointing in)
- **Risk:** HIGH

#### `qms_investigation/` — RCA, Ishikawa, CEMatrix, investigation bridge
- **Models:** `RCASession`, `IshikawaDiagram`, `CEMatrix`
- **Views:** `rca_views.py` (1,057 LOC), `ishikawa_views.py` (158 LOC), `ce_views.py` (155 LOC), `investigation_views.py` (432 LOC), `investigation_bridge.py` (638 LOC helper)
- **URL prefix:** `/api/investigations/`, `/api/rca/`, `/api/iso/rca/`
- **Role:** Transition (pulls from analysis/triage sources, emits investigations)
- **Effort:** L
- **Dependencies:** `qms_core/`, `core.Investigation` (canonical model lives in core)
- **Risk:** HIGH

#### `qms_a3/` — A3 reports (rebuild)
- **Models:** `A3Report`
- **Views:** `a3_views.py` (1,389 LOC, 23 funcs)
- **URL prefix:** `/api/a3/`
- **Role:** Transition (pulls from investigations + workbench, emits A3 containers)
- **Effort:** XL — Eric flagged for full rebuild
- **Dependencies:** `qms_investigation/`, `workbench/`
- **Risk:** HIGH

#### `qms_capa/` — CAPA workflow
- **Models:** `CAPAReport`, `CAPAStatusChange`, optionally `Report` (generic shell)
- **Views:** `capa_views.py` (711 LOC), parts of `iso_views.py` (`capa_*`), `report_views.py`
- **URL prefix:** `/api/capa/`, `/api/reports/`
- **Role:** Transition — but per migration plan flagged for deprecation. May be **delete + replace** rather than relocate.
- **Effort:** S (if delete) or M (if relocate)
- **Risk:** MED

#### `qms_audit/` — Internal audits + management reviews
- **Models:** `InternalAudit`, `AuditFinding`, `AuditChecklist` (or DELETE), `ManagementReview`, `ManagementReviewTemplate`
- **Views:** parts of `iso_views.py` (`audit_*`, `review_*`)
- **URL prefix:** `/api/iso/audits/`, `/api/iso/reviews/` → `/api/qms-audit/`
- **Role:** Transition (pulls from documents + training + risk, emits audit containers)
- **Effort:** L
- **Risk:** MED

#### `qms_training/` — TWI competency / training records
- **Models:** `TrainingRequirement`, `TrainingRecord`, `TrainingRecordChange`
- **Views:** parts of `iso_views.py` (`training_*`)
- **URL prefix:** `/api/iso/training/` → `/api/qms-training/`
- **Role:** Source (training records pulled into audit & supplier qualifications)
- **Effort:** M
- **Dependencies:** loop/ imports — coordinate
- **Risk:** MED

#### `qms_documents/` — Controlled docs + ISO authoring
- **Models:** `ControlledDocument`, `DocumentRevision`, `DocumentStatusChange`, `ISODocument`, `ISOSection`, `ControlPlan`, `ControlPlanItem`
- **Views:** parts of `iso_views.py` (`document_*`), `iso_doc_views.py` (716 LOC)
- **URL prefix:** `/api/iso/documents/`, `/api/iso-docs/` → `/api/qms-documents/`
- **Role:** Source (documents pulled into audits + training)
- **Effort:** L
- **Dependencies:** `graph.ProcessNode` M2M, loop/ imports
- **Risk:** HIGH (loop coupling)

#### `qms_suppliers/` — Supplier QA
- **Models:** `SupplierRecord`, `SupplierStatusChange`
- **Views:** parts of `iso_views.py` (`supplier_*`)
- **URL prefix:** `/api/iso/suppliers/` → `/api/qms-suppliers/`
- **Role:** Source for supplier qualifications
- **Effort:** S
- **Dependencies:** loop/ has 2 incoming FKs
- **Risk:** MED

#### `qms_nonconformance/` — NCR + Customer complaints
- **Models:** `NonconformanceRecord`, `NCRStatusChange`, `CustomerComplaint`
- **Views:** parts of `iso_views.py` (`ncr_*`, `complaint_*`)
- **URL prefix:** `/api/iso/ncrs/`, `/api/iso/complaints/` → `/api/qms-nonconformance/`
- **Role:** Transition / Source dual role
- **Effort:** M
- **Risk:** HIGH (graph + loop coupling)

#### `qms_measurement/` — Calibration + MSA
- **Models:** `MeasurementEquipment`
- **Views:** parts of `iso_views.py` (`equipment_*`), MSA dispatch in `spc_views.py`
- **URL prefix:** `/api/iso/equipment/` → `/api/qms-measurement/`
- **Role:** Source for calibration evidence
- **Effort:** S
- **Dependencies:** `graph.ProcessNode` FK, loop/ + graph/ imports
- **Risk:** MED

#### `hoshin/` — Hoshin Kanri + AFE + X-matrix + resource commitments
- **Models:** `HoshinProject`, `ProjectTemplate`, `StrategicObjective`, `AnnualObjective`, `HoshinKPI`, `XMatrixCorrelation`, `AFE`, `AFEApprovalLevel`, `ResourceCommitment`
- **Views:** `hoshin_views.py` (2,094 LOC, 36 funcs), `xmatrix_views.py` (1,069 LOC, 13 funcs)
- **URL prefix:** `/api/hoshin/`
- **Role:** Operations strategy — not in pull graph (infrastructural for project management)
- **Effort:** L
- **Dependencies:** `qms_core/Site`, `qms_core/Employee`, VSM (source_vsm FK)
- **Risk:** HIGH (HoshinKPI auto-pulls DSWResult — coupling to analysis storage)

#### `vsm/` — Value Stream Maps
- **Models:** `ValueStreamMap`
- **Views:** `vsm_views.py` (433 LOC, 14 funcs)
- **URL prefix:** TBD (currently no `vsm_urls.py`)
- **Role:** Sink for analysis, source for kaizen bursts
- **Effort:** M (rebuild as cockpit per migration plan)
- **Dependencies:** None heavy
- **Risk:** MED

#### `simulation/` — PlantSimulation
- **Models:** `PlantSimulation`
- **Views:** `plantsim_views.py` (391 LOC, 8 funcs)
- **URL prefix:** `/api/plantsim/`
- **Role:** Sink + scenario explorer
- **Effort:** M
- **Notes:** Eric owns this in parallel — extraction must respect his work

#### `whiteboard/` — Boards
- **Models:** `Board`, `BoardParticipant`, `BoardVote`, `BoardGuestInvite`
- **Views:** `whiteboard_views.py` (1,057 LOC, 23 funcs)
- **URL prefix:** `/api/whiteboard/`
- **Role:** Source (boards pulled as evidence)
- **Effort:** M
- **Risk:** LOW

#### `triage/` — Data cleaning
- **Models:** `TriageResult`
- **Views:** `triage_views.py` (512 LOC, 10 funcs)
- **URL prefix:** `/api/triage/`
- **Role:** Source (datasets pulled by every analysis)
- **Effort:** S
- **Risk:** LOW

#### `learn/` — Courses, assessments, learn content
- **Models:** `SectionProgress`, `AssessmentAttempt`, `LearnSession`
- **Views:** `learn_views.py` (2,450 LOC, 32 funcs), `harada_views.py` (909 LOC) optionally
- **Sub-content:** `learn_content/` (~14,556 LOC), `learn_content.py` (8,024 LOC)
- **URL prefix:** `/api/learn/`, `/api/harada/`
- **Role:** Standalone product surface
- **Effort:** L
- **Risk:** LOW

#### `reports/` — Sink for formatted reports (rebuild)
- **Models:** new — replaces `Report`
- **Views:** new
- **URL prefix:** `/api/reports/` (cutover)
- **Role:** Sink
- **Effort:** XL (brand new, ForgeDoc-driven)

#### `sop/` — Standard work editors (new)
- **Models:** new
- **URL prefix:** `/api/sop/`
- **Role:** Sink
- **Effort:** XL (brand new)

### I.2 Existing apps that absorb material

| Existing app | Absorbs |
|---|---|
| `core/` | (already has Notebook, Investigation, InvestigationMembership, Hypothesis, Project, Tenant — gains nothing from agents_api except the cross-cutting permissions helpers move; alternatively `notebook_views.py` could move to core/ since Notebook is already there) |
| `workbench/` | `DSWResult` migration → `Workbench` + `Artifact`; `SavedModel`; `dsw/endpoints_data.py`, `endpoints_ml.py`, `dsw_views.py`, `autopilot_views.py`, `experimenter_views.py`, `forecast_views.py`, `spc_views.py`, `synara_views.py` (all become workbench handlers) |
| `chat/` | `guide_views.py` if conversational; `AgentLog`, `LLMUsage`, `RateLimitOverride` if chat-adjacent |
| `accounts/` | `LLMUsage`, `RateLimitOverride` if billing-adjacent |
| `forge/` | (no model migration — forge/ is the in-product UI for forge service; pip-installed packages do the compute) |

### I.3 Apps to delete or absorb fully

- `agents_api/views.py` (5 agent dispatchers) — DELETE except `get_shared_llm`
- `agents_api/synara/` (3,243 LOC) — replace with `forgesia`, DELETE
- `agents_api/dsw/` legacy stats — DELETE after forge wiring
- `analysis/chart_render.py` + `analysis/chart_defaults.py` — DELETE
- `Workflow` model + views + urls — DELETE
- `ActionItem` model + views + urls — DELETE (LOOP-001 supersedes)
- `Report` model + views + urls — likely DELETE + REPLACE with new `reports/` sink
- `CAPAReport` — likely DELETE + REPLACE with `qms_investigation/` + ForgeDoc generator
- `AuditChecklist` — likely DELETE (Checklist supersedes)

### I.4 What's left in agents_api after extraction

After everything moves out:
- `tool_router.py`, `tool_registry.py`, `tool_event_handlers.py`, `tool_events.py`, `base_tool_model.py` — the cross-tool event/registry layer used by loop/, graph/, and many views. Likely lifts to `core/` or stays as a `tools/` shim app.
- `evidence_bridge.py`, `evidence_weights.py`, `investigation_bridge.py` — bridge code that translates between modules. May move with `qms_investigation/`.
- `permissions.py` — `qms_can_edit`, `qms_queryset`, `qms_set_ownership`, `get_tenant`, `is_site_admin` etc. Used cross-app (e.g. `loop/supplier_views.py`, `notifications/webhook_views.py`). Lifts to `qms_core/`.
- `cache.py`, `embeddings.py`, `gpu_manager.py`, `llm_manager.py`, `llm_service.py` — infrastructure. Lifts to `chat/` or `core/`.
- `pbs_engine.py` (4,070 LOC) — Pull-based scheduling engine. Lifts wherever PBS belongs (likely a new `pbs/` app or absorbed into `forgepbs` package wrapping).
- `quality_economics.py` (1,141 LOC) — COPQ math. Lifts to `forgesiop` or stays as a domain helper.
- `bayes_core.py`, `bayes_doe.py` — Bayesian helpers. Move with their consumers.
- `causal_discovery.py`, `interventional_shap.py`, `clustering.py`, `conformal.py`, `drift_detection.py`, `anytime_valid.py`, `ml_pipeline.py`, `msa_bayes.py` — analytical libraries. Move to `workbench/handlers/` or get replaced by forge packages.
- `report_types.py`, `iso_document_types.py` — registry constants. Move with the corresponding rebuild.
- `front_page_tasks.py`, `harada_tasks.py`, `commitment_tasks.py`, `commitment_notifications.py` — Tempora scheduled tasks. Move with their domains.

The honest end state of `agents_api/` post-extraction is **near-empty or deleted**, with the residual being just `tool_router` / `tool_event_handlers` / `permissions` if they don't have a better home, or fully renamed to `qms_core/` if Site et al. stay there.

---

## Section J — Open questions and judgment calls

### J.1 Where does `Site` live?
Site has 24 incoming FKs (19 internal, 5 cross-app). Three options:
- **Option A:** Site stays in agents_api as the LAST extraction — every other model moves first. Massive risk: agents_api remains as a half-empty shell for the entire extraction sequence.
- **Option B:** Site moves first to a new `qms_core/` app. Every consumer gets updated FK references in lockstep. Highest one-shot risk but cleanest end state.
- **Option C:** Site moves to `core/` (alongside Tenant). Treats it as foundational infrastructure rather than QMS-specific. Risk: makes core/ heavier; signals a permanent "Site is part of multi-tenancy" decision.

**Recommendation:** Option B (`qms_core/`), executed early in the sequence after the leaf models (Workflow, RackSession, ActionItem, AuditChecklist) are deleted to reduce the surface.

### J.2 Where does `Employee` live?
Same question as Site, smaller scale. `Employee` is referenced by loop/, safety/, and HoshinProject/ResourceCommitment. Likely belongs in `qms_core/` next to Site, or could move to `accounts/` adjacent to User.

### J.3 CAPA — relocate or delete + replace?
Migration plan flags `CAPAReport` as deprecated → replaced by Investigation + ForgeDoc CAPA generator. Two consumers — `capa_views.py` (711 LOC) and parts of `iso_views.py`. Decision: relocate to `qms_capa/` for transition, or delete and let CAPA functionality come entirely from `qms_investigation/` + ForgeDoc?

### J.4 ActionItem — relocate or delete?
LOOP-001 `Commitment` supersedes ActionItem. Used in 5 view files. The migration plan says "Deprecate views. Remove when old templates gone." Should the extraction CR delete ActionItem now, or keep it as a temporary shim?

### J.5 Where does `analysis/dispatch.py` live?
Three options per `qms_architecture.md` §3.7:
- (a) `workbench/handlers/dispatch.py`
- (b) new `analysis/dispatch.py` standalone app
- (c) keep in agents_api as the last thing remaining

### J.6 Triage as its own app or co-located with workbench?
Triage is foundational data infrastructure used by every analysis. Is `triage/` worth its own app, or does it co-locate with `workbench/`?

### J.7 Whiteboard as its own app or extension of workbench?
Boards are inquiry surfaces similar to Workbench but with different storage and UI. Eric's call.

### J.8 Naming `qms_investigation/`
Alternatives: `qms_rca/`, `qms_root_cause/`, `investigations/`. The architecture doc leans toward `qms_investigation`.

### J.9 A3 rebuild scope
Eric said A3 will be largely rebuilt. Does the extraction CR include the rebuild, or extract first (relocate to `qms_a3/`) and rebuild in a follow-up CR?

### J.10 Is Hoshin's HoshinKPI → DSWResult coupling tolerable post-extraction?
HoshinKPI.effective_actual queries `DSWResult.objects.filter(...)` directly (line 3285). After workbench replaces DSWResult with Artifact, Hoshin must use the workbench pull API to fetch the latest KPI value from a saved analysis. This is the worked example of why the pull contract exists, but it adds round-trip cost to KPI rollups.

### J.11 dsw/ vs analysis/ — which way does the rest of the migration flow?
Currently: `dsw/dispatch.py` → `analysis/dispatch.py`, but `analysis/chart_*.py` → `dsw/chart_*.py`. The two trees are interlinked. Suggest: pick analysis/ as canonical, move chart_render and chart_defaults from dsw/ into analysis/, then update analysis/chart_*.py to be the real implementation.

### J.12 Per-extraction CR vs batched
Each extraction is a `migration` change_type per CHG-001 → multi-agent risk assessment. Per-extraction CRs preserve rollback granularity but multiply assessment overhead. Recommendation: per-extraction.

### J.13 `report_views.py` vs new `reports/` sink — disposition?
The existing `Report` model with `report_types.py` registry and the views in `report_views.py` (802 LOC) are functional and serve `/api/reports/`. The new `reports/` sink app per Object 271 is being built fresh. Does the existing code stay in production until the new one is ready, or is the old code deleted with the existing CAPA/8D templates?

### J.14 `learn_content.py` (8,024 LOC) vs `learn_content/` directory
Both exist at top level. They appear to be a flat-file vs directory parallel. Need to confirm which is canonical.

### J.15 Who owns `pbs_engine.py` (4,070 LOC)?
Not sure if this corresponds to forgepbs or to something SVEND-specific. Largest single Python file in agents_api after stats files. Must be classified before extraction can finalize.

---

## Section K — Validation of the prior 7 findings

### K.1 "Site is a critical hub with 15+ incoming FKs"
**CONFIRMED — refined.** Actual count is **19 internal FKs** (15 within agents_api models excluding Site/SiteAccess themselves, plus AnnualObjective/HoshinProject/Employee/SiteAccess) **plus 5 cross-app FK references** from `loop/models.py` (×2), `safety/models.py` (×3), `graph/models.py` (×1) — total **24 callers**. Highest-coupled model in the codebase. See §B.3.

### K.2 "9.6MB of exact duplication: dsw/ and analysis/ are 100% duplicate"
**PARTIALLY REFUTED — needs nuance.**
- `dsw/dispatch.py` is a **17-line shim** that re-exports from `analysis/dispatch.py`. So `analysis/` is the canonical dispatcher, NOT `dsw/`.
- `analysis/chart_render.py` and `analysis/chart_defaults.py` are 7-line and 11-line **wrappers around dsw/chart_*** — opposite direction.
- `dsw/common.py` (3,084), `endpoints_data.py` (1,832), `endpoints_ml.py` (1,702), `standardize.py` (552) are **active** — multiple top-level views import from them.
- The bulk of dsw/ — `stats_*.py`, `bayesian.py`, `ml.py`, `viz.py`, `siop.py`, `simulation.py`, `reliability.py`, `d_type.py`, `spc.py`, `spc_pkg/*`, `exploratory/*` — IS legacy duplicate of analysis/ counterparts and IS deletable, but only after forge wiring is complete.
- True total deletable LOC is ~50,000, not the entire dsw/ tree.

**Refined statement:** Of dsw/'s 73,749 LOC, ~17,000 LOC across `common.py`, `endpoints_*`, `chart_*`, `standardize.py` is still live; the remaining ~55,000 LOC is legacy stats/SPC/ML to be replaced by forge packages.

### K.3 "Forge replacement is 95%+ ready"
**CONFIRMED.**
- `spc.py` (1,888 LOC at top level) → `forgespc` ready (per migration plan)
- `synara/` (3,243 LOC) → `forgesia` (pending `__init__` exports)
- `dsw/` legacy stats (~50k LOC) → `forgestat`
- `dsw/chart_*` + `analysis/chart_*` → `forgeviz`
- Verified migration plan §"Tech Debt → Duplicate Computation" matches.

### K.4 "loop/ imports 10+ agents_api models"
**CONFIRMED — refined to 17 distinct import sites.** Per §G.3. Models touched: `Employee`, `TrainingRecord`, `TrainingRequirement`, `ControlledDocument`, `ISOSection`, `ISODocument`, `Site`, `FMEARow`, `FMEA`, `SupplierRecord`, `NonconformanceRecord`, `CustomerComplaint`, `MeasurementEquipment` plus `tool_events`, `permissions.get_tenant`. Extraction MUST coordinate with loop/ in lockstep.

### K.5 "iso_views.py is 4,875 LOC monolithic with deep ControlledDocument / ISODocument / ForgeDoc / management review coupling"
**CONFIRMED.** Verified at **4,874 LOC** with **85 view functions** (defs and class-based combined). Imports 30+ models from `.models`. Owns 11 distinct ISO clause domains. See §C.3 for the split plan.

### K.6 "Graph integration imports MeasurementEquipment, ControlledDocument, NonconformanceRecord, SupplierRecord into causal maps"
**CONFIRMED.** Per §G.4: `graph/views.py:392` imports `ControlledDocument, NonconformanceRecord, SupplierRecord`, `graph/integrations.py:601` imports `MeasurementEquipment`, `graph/tests_qms.py` references `MeasurementEquipment` ×3. Plus `MeasurementEquipment.linked_process_node` is a direct FK to `graph.ProcessNode` and `ControlledDocument.linked_process_nodes` is an M2M to `graph.ProcessNode`. Graph extraction tests must run against any qms_documents/, qms_measurement/, qms_nonconformance/, qms_suppliers/ extraction.

### K.7 "Recommended phases: (1) delete dead code, (2) wire forge packages into views, (3) extract analysis app, (4) extract domain apps"
**CONFIRMED with refinement.**
1. Delete confirmed dead code: `Workflow`, `ActionItem`, original agent dispatchers, `analysis/chart_*` wrappers, `RackSession` (if demo confirmed dead), `AuditChecklist`, sweep test files.
2. Wire forge packages into `analysis/forge_*.py` bridges. Cut `dsw/` legacy stats files. Move active dsw/ helpers (`common`, `endpoints_*`, `standardize`, `chart_*`) into `analysis/`.
3. **Extract leaf models first** (low incoming FK count): `whiteboard`, `triage`, `vsm`, `simulation`, `learn`, `qms_measurement`, `qms_suppliers`. These have minimal cross-app coupling and unblock the medium-tier work.
4. **Extract medium-coupling models:** `qms_investigation`, `qms_a3`, `qms_documents`, `qms_training`, `qms_nonconformance`, `qms_audit`, `qms_risk`, `hoshin`. Each requires lockstep updates to `loop/`, `graph/`, and `safety/`.
5. **Extract `qms_core/`** — Site, Employee, ESig, QMSAttachment, QMSFieldChange, Checklist family, ActionToken, permissions. Requires updating EVERY extracted app's FK references in one CR.
6. **Build new sinks** — `reports/`, `sop/`.
7. **Final cleanup** — anything left in agents_api gets renamed or deleted.

The prior session's 4-phase plan compresses steps 3-5 of the refined version. Recommend the more granular sequencing because each extraction needs its own behavior tests and rollback path.

---

## Appendix A — Files I read in full (verification trail)

- `agents_api/models.py` lines 1–6797 (full read in 5 segments)
- `agents_api/views.py` (head)
- `agents_api/urls.py`, `analysis_urls.py`, `dsw_urls.py`, `iso_urls.py` (head)
- `agents_api/iso_views.py` (head + import block)
- `agents_api/qms_views.py`, `fmea_views.py`, `rca_views.py`, `a3_views.py`, `investigation_views.py`, `iso_doc_views.py`, `capa_views.py`, `xmatrix_views.py`, `hoshin_views.py` (import blocks)
- `agents_api/dsw/dispatch.py` (full)
- `agents_api/analysis/dispatch.py` (head)
- `agents_api/iso_tests.py` (head + sample), `tests/test_endpoint_smoke.py` (head), `tests/test_dsw_views_behavioral.py` (head)
- `docs/planning/object_271/qms_architecture.md` lines 1–475
- `docs/planning/object_271/migration_plan.md` lines 1–300

## Appendix B — Counts I generated

- `wc -l` totals for all `agents_api/**/*.py`, broken down by file and subpackage
- `grep -c '^class .*\(models.Model\)'` against `models.py` → 67; +1 SynaraImmutableLog → 68 total
- `grep` for `"agents_api.Site"` and `site = models.ForeignKey(\s*Site` → 19 internal incoming FKs
- `grep` for `from agents_api` and `agents_api\.[A-Z]` across the repo → cross-app coupling map
- `grep` for `def test_` against test files → function counts cited in §F
