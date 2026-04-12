# agents_api Extraction — Gap Analysis

**Status:** DRAFT — planning artifact under CR `5bf7354c-3de5-4624-b505-a94a5b6ce0ea`
**Date:** 2026-04-09
**Author:** Claude (Systems Engineer role per Object 271)
**Inputs:**
- `docs/planning/object_271/qms_architecture.md` (locked v0.4) — target state
- `docs/planning/object_271/agents_api_inventory.md` — current state
- `docs/planning/object_271/migration_plan.md` — Object 271 stack decisions
- `~/.claude/projects/-home-eric/memory/project_agents_api_extraction.md` — extraction state

**Purpose:** Canonical mapping from current-state `agents_api/` to the target v0.4 topology of 25 Django apps. Every model, every view file, every URL surface, every cross-app coupling site has an action, a target home, a phase, a risk tier, and a list of coordinated CR partners. This document is the shared dependency for `extraction_sequence.md` (the sequenced execution plan) and `test_suite_rebuild.md` (the TST-001 safety net).

This is a **mapping document**, not an execution order. Ordering lives in `extraction_sequence.md`. This document does not propose code changes, does not propose new architectural decisions, and is not a final standard. It is the gap that the sequenced plan will close.

---

## Section 1 — Executive summary

### 1.1 Headline counts

| Metric | Count | Source |
|---|---|---|
| Models in `agents_api/models.py` today | **68** (67 `models.Model` + 1 `SynaraImmutableLog`) | inventory §A.3 |
| Models that survive extraction (relocated) | **~58** | derived — see §2 |
| Models deleted outright (no replacement) | **~5** | `Workflow`, `ActionItem`, `AuditChecklist`, `RackSession` (tentative), `AgentLog` (tentative) |
| Models deleted and replaced by new implementations | **~3-5** | `Report` (→ new `reports/` sink), `CAPAReport` (→ `qms_investigation/` + ForgeDoc), possibly others resolved at extraction time |
| `DSWResult` — converted not deleted | **1** | survives as `workbench.Artifact` rows; the class itself disappears |
| Target app count (architecture §3.0) | **25** Django apps in 6 categories | architecture §3.0 |
| New apps created in this extraction | **19** | 3 sources (§3.2), 4 transitions (§3.3), 2 sinks (§3.4), 6 QMS-adjacent (§3.5), 4 operations (§3.6) |
| Existing foundation apps touched | **6** | `core`, `accounts`, `chat`, `forge`, `files`, `api` (architecture §3.1) |
| `iso_views.py` LOC (single monolith) | **4,874 LOC, 85 view funcs** | inventory §C.3 |
| Total `agents_api/**/*.py` LOC | **289,313** | inventory §A.1 |
| Site incoming FKs (chokepoint) | **24** (19 internal + 5 cross-app) | inventory §B.3, architecture §3.6.1 |
| `loop/` ↔ agents_api import sites | **17** | inventory §G.3, §K.4 |
| `iso_views.py` split targets | **7 apps** | inventory §C.3 |

### 1.2 Phase-to-CR distribution (rough)

This is a high-level allocation to motivate the sequence plan. Exact CR counts are set in `extraction_sequence.md`. Each Phase 1A/2A extraction is a `migration` change_type; each Phase 1B/2B rebuild is a `feature` change_type; Phase 0 is a mix of `enhancement` and `debt` change types; Phase 3 cutover is a `feature`/`migration` hybrid; Phase 4 Site move is a single `migration` CR with high review burden.

| Phase | Sub-phase | What | Est. CRs | Notes |
|---|---|---|---|---|
| **Phase 0** | (single) | Forge wiring + dead-code cleanup prerequisite | **6-8** | See §6 — split by forge package + delete batches |
| **Phase 1A** | Relocate (leaves) | 7 leaf app extractions to `/app/demo/...` AS-IS | **7** | See §7 |
| **Phase 1B** | Rebuild (leaves) | Same 7 apps rebuilt against pull contract / sv-* / ForgeViz | **7-10** | Some rebuilds may split per complexity |
| **(gate 1)** | | Eric review of leaf parallel build | — | |
| **Phase 2A** | Relocate (medium coupling) | 8-9 medium-coupling extractions | **8-9** | See §8 — each is a 2-app coordinated CR (extraction + `loop/` update) |
| **Phase 2B** | Rebuild (medium coupling) | Same 8-9 rebuilt | **8-12** | `qms_a3/` and `reports/`/`sop/` are brand-new rebuilds |
| **(gate 2)** | | Eric review of medium-coupling parallel build | — | |
| **Phase 3** | Cutover | Single-night URL swap for all demo routes | **1** | Large commit; deletions + URL swaps bundled |
| **Phase 4** | Site final | `qms_core/` consolidation with Site as atomic move | **1-2** | Site is the single highest-risk CR in the whole work |
| **TOTAL** | | Full extraction | **~40-50 CRs** | Plus `iso_views.py` split prerequisite (1) and `learn_content` / `pbs_engine` classification (0-2) |

Phase 1A and 1B run in series for each individual leaf (1A relocates, 1B rebuilds) per the universal cutover pattern (architecture §13). The relocations are parallel across leaves only if their coordinated-CR partners don't conflict.

### 1.3 Top three surprises (from inventory) that shape this gap analysis

1. **`iso_views.py` is not a single view file — it is 11 ISO clause areas crammed into 4,874 LOC across 85 view functions that must split across 7 target apps.** This is the single largest extraction risk and requires a prerequisite "split-in-place" CR before any extraction can proceed (architecture §7.6, inventory §C.3).

2. **`dsw/` and `analysis/` are NOT 100% duplicate — the direction of dispatch is `dsw → analysis`, while the direction of chart helpers is `analysis → dsw`.** Only ~17,000 LOC of `dsw/` is live; ~50-55,000 LOC is legacy deletable post-forge-wiring. The migration is finer-grained than "delete the duplicate" (architecture §7.3, inventory §K.2).

3. **`HoshinKPI.effective_actual` queries `DSWResult.objects.filter(...)` directly at `agents_api/models.py:3285` — this is the canonical cross-app direct-ORM coupling that the pull contract exists to replace.** Every other direct-ORM cross-app coupling site converts to the same pattern. The worked example is in architecture §4.5 (inventory §B.2 HoshinKPI, §J.10).

---

## Section 2 — Model-by-model gap (the centerpiece)

This is the comprehensive model map. Every one of the 68 models in `agents_api/models.py` has a row. Grouped by target app for readability. Each row lists: source location, target home, action, phase assignment, risk tier, coordinated-CR partners, direct ORM coupling flags, and deletion rationale where applicable.

**Legend:**
- **Action:** `RELOCATE` (move as-is), `DELETE` (no replacement), `DELETE+REPLACE` (new implementation in new app), `CONVERT` (replaced at storage/API boundary — `DSWResult → workbench.Artifact` is the only instance)
- **Phase:** `0` (forge wiring / dead-code cleanup), `1A` (leaf relocate), `1B` (leaf rebuild), `2A` (medium-coupling relocate), `2B` (medium-coupling rebuild), `3` (cutover URL swap), `4` (Site atomic move)
- **Risk:** LOW / MED / HIGH / CRITICAL (CRITICAL reserved for Site)
- **CR partners:** apps that must update in the same CR

### 2.1 Target: `qms_core/` (new — cross-cutting QMS infrastructure)

Architecture §3.5: "Site chokepoint home." Phase 4 for Site itself; the rest of the app's models are set up as Site moves.

| # | Source | Target | Action | Phase | Risk | CR partners | ORM coupling flags | Notes |
|---|---|---|---|---|---|---|---|---|
| 25 | `agents_api/models.py:2020 Site` | `qms_core/` | RELOCATE | **4** | **CRITICAL** | EVERY extracted app (19 internal + 5 cross-app FKs must atomically swap); `loop/`, `safety/`, `graph/` (FK string lookups) | 24 incoming FKs — see inventory §B.3 | The single highest-risk migration in the entire extraction. Moves LAST per architecture §9.B.2 after full parallel rebuild reviewable in `/app/demo/`. |
| 26 | `agents_api/models.py:2072 SiteAccess` | `qms_core/` | RELOCATE | **4** | HIGH | co-moves with Site | Unique-together `(site, user)`; CASCADE on Site | Per-user site permission. Used by `iso_views.py`, `hoshin_views.py`. Moves in the same CR as Site. |
| 30 | `agents_api/models.py:2511 Employee` | `qms_core/` | RELOCATE | **4** | HIGH | `loop/` (2 FK refs), `safety/` (2 FK refs), `hoshin/` (ResourceCommitment FK) | loop/models.py:424, 982; safety/models.py:260, 344 | Per architecture §9.A J.2: co-located with Site in qms_core. Moves with Site (Phase 4). |
| 32 | `agents_api/models.py:2669 ActionToken` | `qms_core/` | RELOCATE | **4** | LOW | `token_views.py` moves with it | FK to Employee | Email-token system for non-users. Moves in the Site CR or just before. |
| 53 | `agents_api/models.py:4965 QMSFieldChange` | `qms_core/` | RELOCATE | **4** | MED | NCR, Audit, Document, Supplier view files reference this | `(record_type, record_id)` polymorphic — see inventory §B.2 | Cross-cutting field-level audit log. Belongs in qms_core as shared infrastructure. |
| 65 | `agents_api/models.py:6390 QMSAttachment` | `qms_core/` | RELOCATE | **4** | MED | `iso_doc_views.py`, `capa_views.py`, etc. | `ENTITY_MODEL_MAP` class constant maps 8 entity types | Polymorphic file attachment. Cross-cutting → qms_core. |
| 60 | `agents_api/models.py:5843 Checklist` | `qms_core/` | RELOCATE | **4** | MED | `iso_views.py`, `hoshin_views.py` | FK to Site | Gawande-style checklist template. Cross-cutting (audit + hoshin + SOP). |
| 61 | `agents_api/models.py:5923 ChecklistExecution` | `qms_core/` | RELOCATE | **4** | MED | co-moves with Checklist | FK to Checklist; `(entity_type, entity_id)` polymorphic | Execution log. Moves with Checklist. |
| 64 | `agents_api/models.py:6232 ElectronicSignature` | `qms_core/` | RELOCATE | **4** | HIGH | `capa_views.py` (sign-off flows), AFE approval chain, management review | Hash-chained via `SynaraImmutableLog` inheritance — **TST-001 critical** | 21 CFR Part 11 + ISO 9001:2015 §7.5.3 compliance. Tamper-detection hash chain must not break during migration. Moves in same CR as Site. |

**Also housed in `qms_core/` but not models:**
- `agents_api/permissions.py` (`qms_can_edit`, `qms_queryset`, `qms_set_ownership`, `get_tenant`, `is_site_admin`) — per inventory §I.4. Used cross-app by `loop/supplier_views.py`, `notifications/webhook_views.py`. These helpers must be available BEFORE any qms_* extraction because every extracted view file will import them. **Phase 0 action: move `permissions.py` to `qms_core/` as a no-op shim first**, then extractions can import from the new location.

### 2.2 Target: `workbench/` (existing app — source role)

Per architecture §3.2: workbench is the source for analysis sessions. Needs `ArtifactReference` model + pull API endpoints added in Phase 1B rebuild. `DSWResult` is NOT relocated — it is CONVERTED at the storage layer to `workbench.Workbench` + `workbench.Artifact` rows.

| # | Source | Target | Action | Phase | Risk | CR partners | ORM coupling flags | Notes |
|---|---|---|---|---|---|---|---|---|
| 2 | `agents_api/models.py:45 DSWResult` | `workbench/` (as `Artifact` subtype) | **CONVERT** | **1B** (rebuild) | **HIGH** | `hoshin/` (HoshinKPI.effective_actual rewrite), `core/models/notebook.py` (4 refs), `dsw_views.py`, `a3_views.py`, `report_views.py`, `learn_views.py` | **CANONICAL ORM COUPLING SITE — `agents_api/models.py:3285` HoshinKPI queries DSWResult directly**; `core/models/notebook.py:71, 82, 217, 228` also direct-references | **The largest single-model lift.** Becomes the pull-contract worked example (architecture §4.5). HoshinKPI rewrite is the canonical conversion (new FK `linked_artifact` + `linked_artifact_key` + auto-register reference on first access). Notebook rewrite is the second-canonical conversion — 4 more direct refs to rewrite. |
| 4 | `agents_api/models.py:121 SavedModel` | `workbench/` | RELOCATE | **1B** (rebuild) | MED | `autopilot_views.py`, `dsw_views.py` | Self-FK `parent_model` | ML model registry. Co-located with workbench because saved ML models are analysis session artifacts. Phase 1B because rebuild absorbs it into new workbench handler pattern. |

**Notes:**
- `DSWResult` does not "move" — after the cutover, new analysis runs save to `workbench.Workbench` + `workbench.Artifact` rows via the new handler pipeline. Old `DSWResult` rows exist during the parallel period; at cutover, a data migration either converts them to `workbench.Artifact` rows or marks them legacy-read-only. The data-migration CR is part of Phase 3 cutover.
- The `workbench.Workbench` and `workbench.Artifact` models already exist per architecture §3.2. What's added: `ArtifactReference` model, pull API endpoints, delete-friction handling (architecture §2.3-§2.4).
- `workbench/models.py KnowledgeGraph` — **DELETE** per architecture §9.B.1 (Eric decision 2026-04-09). This is not a model extracted from agents_api; it's a workbench-owned deletion. Listed here for completeness and because it's part of the workbench rebuild CR.

### 2.3 Target: `triage/` (new — source role)

Architecture §3.2: triage as own source app. Inventory I.1: LOW risk, S effort.

| # | Source | Target | Action | Phase | Risk | CR partners | ORM coupling flags | Notes |
|---|---|---|---|---|---|---|---|---|
| 3 | `agents_api/models.py:95 TriageResult` | `triage/` | RELOCATE | **1A** | LOW | none (self-contained) | Encrypted CSV + report fields | Clean extraction. Inventory §B.2: "Used only by `triage_views.py`." |

### 2.4 Target: `whiteboard/` (new — source role)

Architecture §3.2: whiteboard as own source app. Inventory I.1: LOW risk.

| # | Source | Target | Action | Phase | Risk | CR partners | ORM coupling flags | Notes |
|---|---|---|---|---|---|---|---|---|
| 9 | `agents_api/models.py:399 Board` | `whiteboard/` | RELOCATE | **1A** | MED | `notebook_views.py`, `iso_doc_views.py`, `report_views.py`, `a3_views.py` (board image embeds / diagrams) | Self-contained family | Used by several transitions for embedded diagrams. |
| 10 | `agents_api/models.py:466 BoardParticipant` | `whiteboard/` | RELOCATE | **1A** | LOW | co-moves with Board | — | |
| 11 | `agents_api/models.py:493 BoardVote` | `whiteboard/` | RELOCATE | **1A** | LOW | co-moves with Board | — | |
| 12 | `agents_api/models.py:534 BoardGuestInvite` | `whiteboard/` | RELOCATE | **1A** | LOW | co-moves with Board | — | |

### 2.5 Target: `qms_investigation/` (new — transition role)

Architecture §3.3: transition app, central to the QMS pull graph. Inventory I.1: HIGH risk, L effort.

| # | Source | Target | Action | Phase | Risk | CR partners | ORM coupling flags | Notes |
|---|---|---|---|---|---|---|---|---|
| 17 | `agents_api/models.py:1143 RCASession` | `qms_investigation/` | RELOCATE | **2A** | HIGH | `a3_views.py` (A3.rca_session FK), `notebook_views.py`, `qms_views.py`, `learn_views.py`, `iso_views.py`, `NonconformanceRecord.rca_session` FK, `CAPAReport.rca_session` FK | State-machine transitions; `embedding` BinaryField for similarity search | State machine must not break during migration. Behavior tests must verify state transitions. |
| 18 | `agents_api/models.py:1342 IshikawaDiagram` | `qms_investigation/` | RELOCATE | **2A** | LOW | `notebook_views.py` (reads only) | No incoming FKs | Leaf model. Clean extraction. |
| 19 | `agents_api/models.py:1445 CEMatrix` | `qms_investigation/` | RELOCATE | **2A** | LOW | `notebook_views.py` (reads only) | No incoming FKs | Leaf model. Clean extraction. |

**Also housed in `qms_investigation/`:**
- `agents_api/investigation_bridge.py` (638 LOC) — helper module that translates between modules. Per inventory §I.4: "bridge code that translates between modules. May move with qms_investigation/." Moves in Phase 2A with RCASession.
- `agents_api/evidence_bridge.py` and `evidence_weights.py` — per inventory §I.4: same class.

### 2.6 Target: `qms_a3/` (new — transition role, rebuild)

Architecture §3.3: "A3 will be largely rebuilt as part of Object 271 — new architecture, updated models, front-end polish." Extraction is the rebuild opportunity. Per universal cutover pattern (architecture §13), still 2-step: extract first, rebuild second.

| # | Source | Target | Action | Phase | Risk | CR partners | ORM coupling flags | Notes |
|---|---|---|---|---|---|---|---|---|
| 13 | `agents_api/models.py:594 A3Report` | `qms_a3/` | RELOCATE then REBUILD | **2A** (relocate) + **2B** (rebuild) | HIGH | `rca_views.py` (RCA → A3 linking), `notebook_views.py`, `learn_views.py`, `Site` FK | `imported_from` JSON (provenance), `embedded_diagrams` JSON (whiteboard exports), `last_critique` JSON | 2A is vanilla relocate at `/app/demo/a3/`. 2B is the full rebuild — new pull-contract interface pulling from workbench + qms_investigation. |

### 2.7 Target: `qms_capa/` — DELETE+REPLACE per architecture §9.A

Architecture §9.A decision #3: "Delete + replace. CAPAReport is deprecated per migration plan; new CAPA functionality comes from `qms_investigation/` + ForgeDoc CAPA generator. **No transitional `qms_capa/` app needed.**" There is no `qms_capa/` app in the target topology despite the §3.3 label — CAPA functionality lives in `qms_investigation/` + ForgeDoc.

| # | Source | Target | Action | Phase | Risk | CR partners | ORM coupling flags | Notes |
|---|---|---|---|---|---|---|---|---|
| 14 | `agents_api/models.py:736 Report` | new `reports/` sink (rebuild) | **DELETE+REPLACE** | **2B** (new `reports/` build-out) + **3** (deletion at cutover) | MED | `report_views.py` deletion, `NonconformanceRecord.capa_report` FK rewiring | Generic CAPA/8D shell; registry-driven via `report_types.py` | Per architecture §9.A J.13. Old code stays in production until new `reports/` sink is ready; deleted at Phase 3 cutover. |
| 39 | `agents_api/models.py:3723 CAPAReport` | replaced by `qms_investigation/` + ForgeDoc CAPA generator | **DELETE+REPLACE** | **2B** + **3** | HIGH | `capa_views.py` deletion, `iso_views.py capa_*` deletion, `NonconformanceRecord.capa_report` FK removal, `AFE` references | FK to RCASession; state machine | Per architecture §9.A decision #3. Migration plan flags as deprecated. |
| 40 | `agents_api/models.py:3942 CAPAStatusChange` | deleted with CAPAReport | **DELETE** | **3** | LOW | co-deletes with CAPAReport | — | |

### 2.8 Target: `qms_audit/` (new — transition role)

Architecture §3.3: transition app, aligns with `project_audit_upgrade.md`. Inventory I.1: L effort, MED risk.

| # | Source | Target | Action | Phase | Risk | CR partners | ORM coupling flags | Notes |
|---|---|---|---|---|---|---|---|---|
| 41 | `agents_api/models.py:3977 InternalAudit` | `qms_audit/` | RELOCATE | **2A** | HIGH | `iso_views.py audit_*` split, `iso_views.py audit_clause_coverage`, `iso_views.py audit_apply_checklist`, `iso_views.py audit_checklist_*` | Site FK | Part of the `iso_views.py` monolith split. Prerequisite: `iso_views.py` split CR lands first. |
| 42 | `agents_api/models.py:4076 AuditFinding` | `qms_audit/` | RELOCATE | **2A** | MED | co-moves with InternalAudit; FK to NonconformanceRecord | `ncr` FK | |
| 46 | `agents_api/models.py:4339 ManagementReviewTemplate` | `qms_audit/` | RELOCATE | **2A** | LOW | `iso_views.py review_template_*` split | `DEFAULT_SECTIONS` class constant | Per inventory B.2. |
| 47 | `agents_api/models.py:4461 ManagementReview` | `qms_audit/` | RELOCATE | **2A** | MED | `iso_views.py review_*` split | `data_snapshot` JSON (captured metrics) | |
| 59 | `agents_api/models.py:5800 AuditChecklist` | `qms_audit/` or **DELETE** | DELETE (likely) | **0** | LOW | `iso_views.py audit_checklist_*` deletion | Superseded by `Checklist` | Per inventory I.3 + H.6: "confirm and delete AuditChecklist." Resolution via grep for live tenants — if none, delete in Phase 0 with other dead code. |

### 2.9 Target: `qms_risk/` (new — source+transition dual)

Architecture §3.5. Inventory I.1: 3 models, HIGH risk (cross-app coupling).

| # | Source | Target | Action | Phase | Risk | CR partners | ORM coupling flags | Notes |
|---|---|---|---|---|---|---|---|---|
| 15 | `agents_api/models.py:839 FMEA` | `qms_risk/` | RELOCATE | **2A** | HIGH | `loop/` (FMIS rows; models.py:1204, 1447 reference FMEARow which FKs to FMEA), `safety/models.py:736`, `notebook_views.py`, `learn_views.py`, `fmea_views.py` rewire, `iso_views.py risk_*` split | Site FK; `Risk.fmea` back-ref | **2-app CR minimum** (qms_risk/ extraction + loop/ import update). |
| 16 | `agents_api/models.py:936 FMEARow` | `qms_risk/` | RELOCATE | **2A** | HIGH | same as FMEA; **cross-app FK sources: `loop/models.py:1204, 1447`** | `hypothesis_link` FK to `core.Hypothesis` (Bayesian bridge); `spc_measurement` text field | Critical model for FMIS integration. Tests must verify `loop.FMISRow` still attaches after FK update. |
| 55 | `agents_api/models.py:5213 Risk` | `qms_risk/` | RELOCATE | **2A** | HIGH | `iso_views.py risk_*` split | FK to FMEA, FMEARow, Project | ISO 9001 §6.1 organizational risk register. Per `feedback_afe_policy.md`: AFEs cannot flow directly from Risk. |

### 2.10 Target: `qms_documents/` (new — source role)

Architecture §3.5. Inventory I.1: 7 models, HIGH risk (loop + graph coupling).

| # | Source | Target | Action | Phase | Risk | CR partners | ORM coupling flags | Notes |
|---|---|---|---|---|---|---|---|---|
| 48 | `agents_api/models.py:4534 ControlledDocument` | `qms_documents/` | RELOCATE | **2A** | **HIGH** | `loop/models.py:694, 884, 971`, `loop/views.py:2361`, `loop/services.py:297`, `graph/views.py:392`, `iso_views.py document_*` split | **M2M `graph.ProcessNode` (linked_process_nodes) — line 4591**; FK `source_study` to `core.Project`; M2M `files.UserFile` | Per architecture §7.5: "HIGH risk because of graph.ProcessNode M2M." Coordinates with graph/ extraction. 3-app coordinated CR. |
| 49 | `agents_api/models.py:4702 DocumentRevision` | `qms_documents/` | RELOCATE | **2A** | LOW | co-moves with ControlledDocument | — | |
| 50 | `agents_api/models.py:4745 DocumentStatusChange` | `qms_documents/` | RELOCATE | **2A** | LOW | co-moves | — | |
| 62 | `agents_api/models.py:6027 ISODocument` | `qms_documents/` | RELOCATE | **2A** | MED | `loop/views.py:1048`, `loop/services.py:150`, `iso_doc_views.py` | OneToOne to ControlledDocument when published | Authoring tier of the same domain. |
| 63 | `agents_api/models.py:6128 ISOSection` | `qms_documents/` | RELOCATE | **2A** | MED | `loop/models.py:895, 1117` | `section_type` enum; `embedded_media` JSON | Section-level authoring. Loop uses embedded SOP sections. |
| 66 | `agents_api/models.py:6488 ControlPlan` | `qms_documents/` or `qms_measurement/` | RELOCATE | **2A** | MED | co-moves with ControlPlanItem | Site FK | Inventory §B.2 flags either home. Per architecture §3.5 ControlPlan fits under qms_documents (authoring-adjacent). |
| 67 | `agents_api/models.py:6587 ControlPlanItem` | same as parent | RELOCATE | **2A** | MED | **FKs to `loop.FMISRow`, `graph.ProcessNode`, `MeasurementEquipment`** | Cross-app FKs in 3 directions | Tight coupling to loop + graph + measurement. Test matrix must cover all three. |

### 2.11 Target: `qms_training/` (new — source role)

Architecture §3.5. Inventory I.1: 3 models, loop/ coupling required.

| # | Source | Target | Action | Phase | Risk | CR partners | ORM coupling flags | Notes |
|---|---|---|---|---|---|---|---|---|
| 43 | `agents_api/models.py:4133 TrainingRequirement` | `qms_training/` | RELOCATE | **2A** | HIGH | `loop/services.py:253`, `loop/readiness.py:171`, `iso_views.py training_*` split | FK to `ControlledDocument` (SOP-driven training) | 2-app coordinated CR minimum. |
| 44 | `agents_api/models.py:4214 TrainingRecord` | `qms_training/` | RELOCATE | **2A** | HIGH | `loop/models.py:879`, `loop/views.py:2386`, `loop/services.py:253`, `loop/readiness.py:171` | M2M `files.UserFile`; `certification_status` property | Loop commitment workflow uses these as readiness signals. TWI competency per TRN-001 §3. |
| 45 | `agents_api/models.py:4295 TrainingRecordChange` | `qms_training/` | RELOCATE | **2A** | LOW | co-moves | — | |

### 2.12 Target: `qms_suppliers/` (new — source role)

Architecture §3.5. Inventory I.1: 2 models, S effort, MED risk (loop × 2).

| # | Source | Target | Action | Phase | Risk | CR partners | ORM coupling flags | Notes |
|---|---|---|---|---|---|---|---|---|
| 51 | `agents_api/models.py:4789 SupplierRecord` | `qms_suppliers/` | RELOCATE | **1A** (leaf) or **2A** | MED | `loop/models.py:1850, 2105`, `graph/views.py:392`, `NonconformanceRecord.supplier` FK, `iso_views.py supplier_*` split | State machine + TRANSITION_REQUIRES | S effort, 2 loop/ FKs. Small enough for Phase 1A if coordinated early. |
| 52 | `agents_api/models.py:4921 SupplierStatusChange` | `qms_suppliers/` | RELOCATE | same | LOW | co-moves | — | |

### 2.13 Target: `qms_nonconformance/` (new — transition+source dual)

Architecture §3.5 (added in v0.3). Inventory I.1: 3 models, HIGH risk.

| # | Source | Target | Action | Phase | Risk | CR partners | ORM coupling flags | Notes |
|---|---|---|---|---|---|---|---|---|
| 37 | `agents_api/models.py:3440 NonconformanceRecord` | `qms_nonconformance/` | RELOCATE | **2A** | **HIGH** | `loop/models.py:1855`, `graph/views.py:392`, `iso_views.py ncr_*` split, `AuditFinding.ncr` FK, `CustomerComplaint.ncr` FK, `Report`/`CAPAReport.ncr` references | `linked_process_node_ids` JSONField (graph linkage opt-in); FK to RCASession; FK to CAPAReport/Report (removed at Phase 2B) | State machine `TRANSITIONS` + `TRANSITION_REQUIRES`. **HIGH risk** — both loop and graph couple in. |
| 38 | `agents_api/models.py:3688 NCRStatusChange` | `qms_nonconformance/` | RELOCATE | **2A** | LOW | co-moves | — | |
| 54 | `agents_api/models.py:5022 CustomerComplaint` | `qms_nonconformance/` | RELOCATE | **2A** | HIGH | `loop/views.py:2404`, `iso_views.py complaint_*` split, FK to NCR, FK to CAPAReport | `linked_process_node_ids` opt-in graph linkage | Tightly coupled to NCR. Moves together. |

### 2.14 Target: `qms_measurement/` (new — source role)

Architecture §3.5. Inventory I.1: 1 model, MED risk (graph coupling).

| # | Source | Target | Action | Phase | Risk | CR partners | ORM coupling flags | Notes |
|---|---|---|---|---|---|---|---|---|
| 58 | `agents_api/models.py:5647 MeasurementEquipment` | `qms_measurement/` | RELOCATE | **1A** | MED | `graph/integrations.py:601`, `graph/tests_qms.py` (×3), `loop/evaluator.py:309`, `iso_views.py equipment_*` split, `spc_views.py` MSA dispatch | **Direct FK to `graph.ProcessNode` (line 5724)**; Weibull reliability fields (`mtbf_hours`, `weibull_shape`, `weibull_scale`, `failure_history` JSON) | 1 model → Phase 1A leaf candidate. Graph FK is a direct coupling that the extraction CR must atomically swap. |

### 2.15 Target: `hoshin/` (new — operations strategy)

Architecture §3.6. Inventory I.1: 9 models, HIGH risk.

| # | Source | Target | Action | Phase | Risk | CR partners | ORM coupling flags | Notes |
|---|---|---|---|---|---|---|---|---|
| 27 | `agents_api/models.py:2135 HoshinProject` | `hoshin/` | RELOCATE | **2A** | HIGH | `qms_views.py`, `vsm_views.py`, `xmatrix_views.py`, `iso_views.py` | OneToOne with `core.Project`; FK Site; FK ValueStreamMap (`source_vsm`) | Enterprise-tier strategy deployment. |
| 28 | `agents_api/models.py:2323 ProjectTemplate` | `hoshin/` | RELOCATE | **2A** | MED | `hoshin_views.py` | FK Site | Reusable template. |
| 31 | `agents_api/models.py:2573 ResourceCommitment` | `hoshin/` | RELOCATE | **2A** | MED | `hoshin_views.py`, FK to Employee + HoshinProject | Role/status state machine | Moves with hoshin. Employee FK updates at Site move (Phase 4). |
| 33 | `agents_api/models.py:2746 StrategicObjective` | `hoshin/` | RELOCATE | **2A** | MED | `XMatrixCorrelation` signal handlers | — | |
| 34 | `agents_api/models.py:2823 AnnualObjective` | `hoshin/` | RELOCATE | **2A** | MED | Site FK | — | |
| 35 | `agents_api/models.py:2913 HoshinKPI` | `hoshin/` | RELOCATE+REWRITE | **2A** (relocate) + **2B** (rewrite `effective_actual`) | **HIGH** | **CANONICAL PULL CONTRACT CONVERSION** — `workbench/` ArtifactReference integration | **`agents_api/models.py:3285 DSWResult.objects.filter(...)` direct ORM query** — architecture §4.5 | See §5 below for the full conversion pattern. This is the canonical worked example. |
| 36 | `agents_api/models.py:3344 XMatrixCorrelation` | `hoshin/` | RELOCATE | **2A** | LOW | post-delete signal handlers for StrategicObjective/AnnualObjective/HoshinProject/HoshinKPI | — | Signal handlers must be preserved during migration. |
| 56 | `agents_api/models.py:5386 AFE` | `hoshin/` | RELOCATE | **2A** | MED | `fmea_views.py` (removes direct AFE creation per `feedback_afe_policy.md`), `iso_views.py afe_*` split | FK to HoshinProject, Risk, FMEA, Checklist, Project | Per `feedback_afe_policy.md`: AFEs only via Hoshin. |
| 57 | `agents_api/models.py:5562 AFEApprovalLevel` | `hoshin/` | RELOCATE | **2A** | LOW | co-moves with AFE | FK to ElectronicSignature (CFR sign-off) — updates at Phase 4 Site move | N-level approval chain. |

### 2.16 Target: `vsm/` (new — operations lean)

Architecture §3.6. Inventory I.1: 1 model, 433 LOC view, MED risk (rebuild target).

| # | Source | Target | Action | Phase | Risk | CR partners | ORM coupling flags | Notes |
|---|---|---|---|---|---|---|---|---|
| 20 | `agents_api/models.py:1559 ValueStreamMap` | `vsm/` | RELOCATE+REBUILD | **1A** + **1B** | MED | `HoshinProject.source_vsm` FK, `PlantSimulation.source_vsm` FK, `xmatrix_views.py`, `learn_views.py`, `qms_views.py` | 16 JSON fields; self-FK `paired_with` (current/future state) | Per migration plan Tier 2: VSM rebuild is cockpit UX with integrated calculator panel. 1A relocates model; 1B is the cockpit rebuild. |

### 2.17 Target: `simulation/` (new — operations)

Architecture §3.6. Inventory I.1: 1 model, 391 LOC view, MED risk (Eric active in parallel).

| # | Source | Target | Action | Phase | Risk | CR partners | ORM coupling flags | Notes |
|---|---|---|---|---|---|---|---|---|
| 24 | `agents_api/models.py:1923 PlantSimulation` | `simulation/` | RELOCATE | **1A** | MED | `ValueStreamMap` FK (source_vsm) | DES layout + sim run records | **Eric was upgrading simulators in parallel and paused for this work** — extraction must respect his parallel work per memory. |

### 2.18 Target: `learn/` (new — standalone product surface)

Architecture §3.6. Inventory I.1: 3 models, 2,450 LOC view, LOW risk.

| # | Source | Target | Action | Phase | Risk | CR partners | ORM coupling flags | Notes |
|---|---|---|---|---|---|---|---|---|
| 21 | `agents_api/models.py:1806 SectionProgress` | `learn/` | RELOCATE | **1A** | LOW | `learn_views.py`, `learn_content/` subpackage | — | |
| 22 | `agents_api/models.py:1842 AssessmentAttempt` | `learn/` | RELOCATE | **1A** | LOW | co-moves | — | |
| 23 | `agents_api/models.py:1871 LearnSession` | `learn/` | RELOCATE | **1A** | LOW | co-moves | FK `core.Project` (sandbox sessions) | |

Also in-scope for learn/ extraction (non-models):
- `agents_api/learn_content/` directory (~14,556 LOC of content data) — moves wholesale
- `agents_api/learn_content.py` flat file (8,024 LOC) — **defer: grep to verify canonical** (see §11)
- `agents_api/learn_views.py` (2,450 LOC, 32 funcs) — moves with models
- `agents_api/harada_views.py` (909 LOC) — per inventory C.1, optional co-location in `learn/` or own `harada/` app

### 2.19 Models destined for DELETE (no replacement)

These models do not have a target home. They are deleted in Phase 0 or during a relocation CR's cleanup.

| # | Source | Action | Phase | Rationale | Source citation |
|---|---|---|---|---|---|
| 1 | `agents_api/models.py:17 Workflow` | **DELETE** | **0** | Inventory §H.5: "zero non-self FKs in the codebase", "likely the original user-defined chain-of-agents feature, never adopted at scale." Architecture §7.2 confirms deletion candidate. | inventory H.5, architecture §7.2 |
| 5 | `agents_api/models.py:179 AgentLog` | DELETE (likely) | **0** | Inventory §B.2: "original agent dispatch operational log — likely stale." Confirm via grep for live writes; if none, delete with original agent dispatchers. | inventory B.2 |
| 29 | `agents_api/models.py:2406 ActionItem` | **DELETE** | **0** (model deletion) + **2A/2B** (view deletions by domain) | Architecture §9.A decision #4: "Delete. LOOP-001's Commitment supersedes. The 5 view files referencing it (action_views, fmea_views, rca_views, a3_views, hoshin_views) get their references replaced with Commitment." | architecture §9.A #4, LOOP-001, inventory I.3 |
| 68 | `agents_api/models.py:6720 RackSession` | DELETE or new `forgerack/` app | **0** (if dead) | Inventory §H.7: "Keep for now (Eric is iterating on ForgeRack), but it should be in its own `forgerack/` app, not agents_api/." Per memory `project_rack_canvas_checkpoint.md`: ForgeRack is active work. Decision at Phase 0 based on Eric's ForgeRack status. | inventory H.7 |

### 2.20 Models with unclear target (defer to extraction-time classification)

| # | Source | Tentative action | Notes |
|---|---|---|---|
| 6 | `agents_api/models.py:212 CacheEntry` | RELOCATE to `syn/` or `core/` | Infrastructure. Inventory §B.2: `db_table = "session_cache"`, dedicated `cache.py` module. Not urgent; can stay in agents_api throughout extraction and move last with whatever remnant. |
| 7 | `agents_api/models.py:259 LLMUsage` | RELOCATE to `accounts/` (billing-adjacent) | Drives billing tier rate limits. |
| 8 | `agents_api/models.py:352 RateLimitOverride` | RELOCATE to `accounts/` | Same. |

Phase: **3-4** (move with agents_api remnant cleanup at or after cutover).

### 2.21 Summary table — models by target app

| Target app | Model count | Phase majority | Risk majority |
|---|---|---|---|
| `qms_core/` (Site + cross-cutting) | 9 | 4 | HIGH-CRITICAL |
| `workbench/` (existing) | 2 (1 CONVERT + 1 RELOCATE) | 1B | HIGH |
| `triage/` (new) | 1 | 1A | LOW |
| `whiteboard/` (new) | 4 | 1A | LOW-MED |
| `qms_investigation/` (new) | 3 | 2A | LOW-HIGH |
| `qms_a3/` (new) | 1 | 2A+2B | HIGH |
| DELETE+REPLACE (Report, CAPA) | 3 | 2B+3 | MED-HIGH |
| `qms_audit/` (new) | 4 (+1 delete) | 2A | LOW-HIGH |
| `qms_risk/` (new) | 3 | 2A | HIGH |
| `qms_documents/` (new) | 7 | 2A | LOW-HIGH |
| `qms_training/` (new) | 3 | 2A | LOW-HIGH |
| `qms_suppliers/` (new) | 2 | 1A-2A | LOW-MED |
| `qms_nonconformance/` (new) | 3 | 2A | LOW-HIGH |
| `qms_measurement/` (new) | 1 | 1A | MED |
| `hoshin/` (new) | 9 | 2A+2B | LOW-HIGH |
| `vsm/` (new) | 1 | 1A+1B | MED |
| `simulation/` (new) | 1 | 1A | MED |
| `learn/` (new) | 3 | 1A | LOW |
| DELETE (Workflow, ActionItem, etc.) | 4-5 | 0 | LOW |
| `accounts/` / `syn/` / `core/` absorption | 3 | 3-4 | LOW |

**Total: 67 `models.Model` + 1 `SynaraImmutableLog` = 68.** All accounted for.

---

## Section 3 — View-by-view gap

The 33 top-level view files in `agents_api/`. Per inventory §C, each view file is listed with source location, target home, action, phase, models touched (which §2 rows it imports), forge wiring status, and coordinated-CR partners.

### 3.1 Top-tier view files (highest LOC)

| File | LOC | Funcs | Target | Action | Phase | Models touched | Forge wiring | CR partners |
|---|---|---|---|---|---|---|---|---|
| `iso_views.py` | **4,874** | **85** | **SPLIT across 7 apps** (qms_audit/qms_documents/qms_training/qms_suppliers/qms_nonconformance/qms_measurement/qms_risk) | **SPLIT** (prerequisite CR) + RELOCATE pieces | **0** (split CR) + **2A** (each piece) | NCR, Audit, Training×3, Docs×3, Supplier×2, NCR, QMSFieldChange, QMSAttachment, MeasurementEquipment, Checklist×2, ControlPlan×2, ESig, FMEA, AFE, CAPAReport, CustomerComplaint, RCASession, Risk | None (pure view code) | Per architecture §9.A #15: split-first-as-prerequisite-CR. Architecture §7.6 details. Split CR is `enhancement` change_type (low risk, no model changes). See §3.3 below for the 11 ISO clause subdivision. |
| `learn_views.py` | 2,450 | 32 | `learn/` | RELOCATE | **1A** | SectionProgress, AssessmentAttempt, LearnSession, RCASession (lazy), FMEA, FMEARow, A3Report, ValueStreamMap | None | Moves with `learn/` models. Lazy imports of qms models become cross-app imports post-extraction. |
| `experimenter_views.py` | 2,286 | 22 | `workbench/handlers/` | RELOCATE | **1B** | (none — direct DOE compute) | **forgedoe** (to be wired) | Workbench absorbs experimenter as a handler. |
| `hoshin_views.py` | 2,094 | 36 | `hoshin/` | RELOCATE | **2A** | ActionItem (delete refs), Checklist, ChecklistExecution, Employee, HoshinProject, ProjectTemplate, ResourceCommitment, Site, SiteAccess, ValueStreamMap, AnnualObjective, XMatrixCorrelation | None | Moves with hoshin/ models. ActionItem references rewritten to Commitment in same CR. |
| `autopilot_views.py` | 1,918 | 14 | `workbench/handlers/` | RELOCATE | **1B** | SavedModel + dsw/common helpers | **forgestat** (compute) | Workbench absorbs. |
| `notebook_views.py` | 1,621 | 26 | `core/` (Notebook already lives there) | RELOCATE | **2A** | FMEA, Board, CEMatrix, **DSWResult**, IshikawaDiagram, RCASession (lazy) | None | Notebook model is already in core. Views should follow. **4 direct DSWResult references must convert to workbench pull API** (`core/models/notebook.py:71, 82, 217, 228`). |
| `fmea_views.py` | 1,589 | 27 | `qms_risk/` | RELOCATE | **2A** | FMEA, ActionItem (delete refs), CAPAReport (delete refs), FMEARow, RCASession, Risk | None | Moves with qms_risk/ models. |
| `spc_views.py` | 1,547 | 15 | `workbench/handlers/` | RELOCATE | **1B** | FMEARow (lazy) | **forgespc** (ready per migration plan) | **Phase 0 forge wiring is here first** — wire forgespc before spc_views.py moves. |
| `a3_views.py` | 1,389 | 23 | `qms_a3/` | RELOCATE+REBUILD | **2A** + **2B** | A3Report, ActionItem, Board, DSWResult, RCASession | None | 2A relocates; 2B is the full rebuild with new pull-contract architecture (architecture §3.3 per Eric). |
| `synara_views.py` | 1,136 | 32 | `workbench/handlers/` or stays | RELOCATE | **1B** | (operates on `agents_api/synara/` engine — will use `forgesia`) | **forgesia** (BLOCKED on `__init__.py` exports per migration plan Tech Debt) | **Phase 0 forge wiring blocker.** |
| `xmatrix_views.py` | 1,069 | 13 | `hoshin/` | RELOCATE | **2A** | AnnualObjective, HoshinKPI, HoshinProject, Site, StrategicObjective, ValueStreamMap, XMatrixCorrelation | None | Moves with hoshin/. |
| `whiteboard_views.py` | 1,057 | 23 | `whiteboard/` | RELOCATE | **1A** | Board family (×4) | None | Self-contained. Clean leaf extraction. |
| `rca_views.py` | 1,057 | 17 | `qms_investigation/` | RELOCATE | **2A** | ActionItem (delete refs), CAPAReport (delete refs), RCASession, A3Report (lazy) | None | |
| `harada_views.py` | 909 | 14 | `learn/` or own `harada/` | RELOCATE | **1A** | (Harada tasks via JSON state) | None | Inventory C.1 suggests co-location with learn. |
| `rack_views.py` | 848 | 27 | `forgerack/` new app OR DELETE | RELOCATE (if kept) | **0-1A** | RackSession | None | Per inventory §H.7: keep for Eric's ForgeRack work; move to own app. |
| `report_views.py` | 802 | 13 | DELETE+REPLACE by new `reports/` sink | **DELETE+REPLACE** | **2B** (new build) + **3** (deletion) | Board, DSWResult, RCASession, Report | Uses legacy `dsw/chart_render` | Replaced by new `reports/` sink. Old code runs in production until new one cuts over (architecture §9.A #J.13). |

### 3.2 Middle-tier view files

| File | LOC | Funcs | Target | Action | Phase | Models touched | Forge | CR partners |
|---|---|---|---|---|---|---|---|---|
| `iso_doc_views.py` | 716 | 11 | `qms_documents/` | RELOCATE | **2A** | ControlledDocument, DocumentRevision, ISODocument, ISOSection, Board (lazy) | None | Authoring views. Moves with qms_documents/ models. |
| `capa_views.py` | 711 | 12 | **DELETE** | DELETE | **3** | CAPAReport, CAPAStatusChange, NonconformanceRecord, QMSFieldChange, RCASession | None | Per architecture §9.A #3: CAPA is delete+replace. `capa_views.py` has no target home — deleted at cutover. |
| `triage_views.py` | 512 | 10 | `triage/` | RELOCATE | **1A** | AgentLog, TriageResult | None | Leaf. |
| `forecast_views.py` | 456 | 7 | `workbench/handlers/` | RELOCATE | **1B** | (direct compute; uses `dsw/common`) | **forgestat.timeseries** | Workbench handler. |
| `investigation_views.py` | 432 | 11 | `qms_investigation/` | RELOCATE | **2A** | **None from agents_api** — uses `core.Investigation`, `core.InvestigationMembership`, `core.InvestigationToolLink`, plus `investigation_bridge.py` | None | Bridge module moves with it. |
| `workflow_views.py` | 433 | 11 | **DELETE** | DELETE | **0** | Workflow | None | Dead code. |
| `plantsim_views.py` | 391 | 8 | `simulation/` | RELOCATE | **1A** | PlantSimulation, ValueStreamMap | None | Respect Eric's parallel sim work. |
| `vsm_views.py` | 433 | 14 | `vsm/` | RELOCATE+REBUILD | **1A** + **1B** | HoshinProject, ValueStreamMap | None | 1A vanilla; 1B cockpit rebuild per migration plan VSM Workbench Spec. |
| `guide_views.py` | 369 | 4 | `chat/` | RELOCATE | **1A** | (LLM rate-limited; no model FKs) | None | AI decision guide. Conversational → chat/. |
| `dsw_views.py` | 314 | 14 | `workbench/handlers/` | RELOCATE | **1B** | TriageResult (lazy) | **forgestat** (wrap for compute) | Thin wrapper around `dsw.dispatch` → `analysis.dispatch`. |
| `qms_views.py` | 216 | 1 | `qms_core/` | RELOCATE | **4** | FMEA, A3Report, CAPAReport, FMEARow, HoshinProject, RCASession, ValueStreamMap | None | Cross-app dashboard view. Lives in qms_core as shared dashboard. |
| `token_views.py` | 214 | 7 | `qms_core/` | RELOCATE | **4** | ActionToken (via Employee) | None | Moves with ActionToken + Employee (Phase 4). |
| `ishikawa_views.py` | 158 | 6 | `qms_investigation/` | RELOCATE | **2A** | IshikawaDiagram | None | Small. |
| `ce_views.py` | 155 | 6 | `qms_investigation/` | RELOCATE | **2A** | CEMatrix | None | Small. |
| `action_views.py` | 65 | 2 | **DELETE** | DELETE | **0** | ActionItem | None | Per architecture §9.A #4: delete. |
| `views.py` | 441 | (DRF dispatch) | **DELETE except `get_shared_llm`** | DELETE+partial extract | **0** | (imports researcher/coder/writer/editor/eda agents — likely dead) | None | Per inventory §H.1: the 5 agent dispatchers are likely unused. `get_shared_llm` helper extracted to a new module used by `forge/tasks.py:152`. |

### 3.3 The `iso_views.py` split — sub-map

Per architecture §9.A #15 + §7.6 + inventory §C.3: `iso_views.py` is split in a prerequisite CR (Phase 0) BEFORE any of its slices relocate. The split is within-app file reorganization — no model changes — so it's low-risk `enhancement` change type. After the split, each slice is a distinct file that relocates in Phase 2A to its target qms_* app.

| ISO clause area | Function prefix in `iso_views.py` | Target app | Phase |
|---|---|---|---|
| §10.2 NCR + CAPA (8.7, 10.2) | `ncr_*` | `qms_nonconformance/` | **2A** |
| §10.2 CAPA portion | `capa_*` | **DELETE** (CAPA is delete+replace) | **3** |
| §9.2 Internal audit | `audit_*`, `audit_finding_*`, `audit_clause_coverage`, `audit_apply_checklist`, `audit_checklist_*` | `qms_audit/` | **2A** |
| §7.2 Training | `training_*` | `qms_training/` | **2A** |
| §9.3 Management review | `review_*`, `review_template_*`, `review_narrative` | `qms_audit/` | **2A** |
| §7.5 Document control | `document_*` | `qms_documents/` | **2A** |
| §8.4 Supplier | `supplier_*` | `qms_suppliers/` | **1A-2A** |
| §7.1.5 Calibration | `equipment_*`, gage R&R links | `qms_measurement/` | **1A** |
| §9.1.2 Customer complaints | `complaint_*` | `qms_nonconformance/` | **2A** |
| §6.1 Risk register | `risk_*` | `qms_risk/` | **2A** |
| AFE + approval chain | `afe_*` | `hoshin/` | **2A** |
| Control Plan | `control_plan_*` | `qms_documents/` | **2A** |
| Checklist execution | `checklist_*` | `qms_core/` | **4** |

**Also in `iso_views.py`:**
- Imports `evidence_bridge.create_tool_evidence` (a 638-line cross-tool module) — moves with `qms_investigation/`
- Imports `permissions.qms_can_edit`, `qms_queryset`, `qms_set_ownership` — these move to `qms_core/permissions.py` in Phase 0 prerequisite

---

## Section 4 — URL surface gap

Per inventory §E: 24 URL files mounting at 24+ URL mount paths. Each URL pattern has a current route, a target route prefix, a Phase 1A/2A demo-path, a final cutover URL, and an old-URL disposition.

Universal cutover pattern (architecture §13): new routes live at `/app/demo/<thing>/` during the parallel-build period. At Phase 3 cutover, routes swap atomically and old routes either DELETE or 301 REDIRECT.

### 4.1 URL mount map

| Current URL | Source file | Target app | Phase 1A/2A demo path | Final cutover URL | Old URL disposition |
|---|---|---|---|---|---|
| `/api/agents/` | `agents_api/urls.py` | (mostly DELETE) | n/a | DELETE `/api/agents/` | DELETE at Phase 0 |
| `/api/workflows/` | `workflow_urls.py` | DELETE | n/a | n/a | DELETE at Phase 0 |
| `/api/dsw/` | `dsw_urls.py` | `workbench/handlers/` | `/app/demo/dsw/` | `/api/workbench/` + legacy `/api/dsw/` 301 redirect | 301 REDIRECT to new workbench routes (for API backcompat) |
| `/api/analysis/` | `analysis_urls.py` | `workbench/handlers/` | `/app/demo/analysis/` | `/api/workbench/analysis/` | 301 REDIRECT |
| `/api/triage/` | `triage_urls.py` | `triage/` | `/app/demo/triage/` | `/api/triage/` (same) | KEEP (new app mounts same URL) |
| `/api/forecast/` | `forecast_urls.py` | `workbench/handlers/` | `/app/demo/forecast/` | `/api/workbench/forecast/` | 301 REDIRECT |
| `/api/experimenter/` | `experimenter_urls.py` | `workbench/handlers/` | `/app/demo/experimenter/` | `/api/workbench/experimenter/` | 301 REDIRECT |
| `/api/spc/` | `spc_urls.py` | `workbench/handlers/` | `/app/demo/spc/` | `/api/workbench/spc/` | 301 REDIRECT |
| `/api/synara/` | `synara_urls.py` | `workbench/handlers/` | `/app/demo/synara/` | `/api/workbench/synara/` | 301 REDIRECT |
| `/api/whiteboard/` | `whiteboard_urls.py` | `whiteboard/` | `/app/demo/whiteboard/` | `/api/whiteboard/` | KEEP |
| `/api/guide/` | `guide_urls.py` | `chat/` | `/app/demo/guide/` | `/api/chat/guide/` | 301 REDIRECT |
| `/api/reports/` | `report_urls.py` | **new `reports/` sink** | `/app/demo/reports/` | `/api/reports/` | DELETE old code at cutover |
| `/api/plantsim/` | `plantsim_urls.py` | `simulation/` | `/app/demo/simulation/` | `/api/simulation/` | 301 REDIRECT |
| `/api/learn/` | `learn_urls.py` | `learn/` | `/app/demo/learn/` | `/api/learn/` | KEEP |
| `/api/fmea/` | `fmea_urls.py` | `qms_risk/` | `/app/demo/fmea/` | `/api/qms-risk/fmea/` (or keep `/api/fmea/`) | Possibly KEEP (backcompat) |
| `/api/hoshin/` | `hoshin_urls.py` | `hoshin/` | `/app/demo/hoshin/` | `/api/hoshin/` | KEEP |
| `/api/qms/` | `qms_urls.py` | `qms_core/` | `/app/demo/qms/` | `/api/qms/` | KEEP |
| `/api/iso/` | `iso_urls.py` | **SPLIT across 7 apps** | `/app/demo/iso-<clause>/` per clause | `/api/qms-<domain>/...` | 301 REDIRECT per clause path |
| `/api/capa/` | `capa_urls.py` | DELETE | n/a | n/a | DELETE at Phase 3 |
| `/api/iso-docs/` | `iso_doc_urls.py` | `qms_documents/` | `/app/demo/iso-docs/` | `/api/qms-documents/` | 301 REDIRECT |
| `/api/actions/` | `action_urls.py` | DELETE | n/a | n/a | DELETE at Phase 0 |
| `/api/investigations/` | `investigation_urls.py` | `qms_investigation/` | `/app/demo/investigations/` | `/api/investigations/` | KEEP |
| `/api/notebooks/` | `notebook_urls.py` | `core/` (Notebook is there) | `/app/demo/notebooks/` | `/api/notebooks/` | KEEP |
| `/api/harada/` | `harada_urls.py` | `learn/` (or own app) | `/app/demo/harada/` | `/api/learn/harada/` | 301 REDIRECT |
| `action/<token>/` | `token_urls.py` | `qms_core/` | (no demo path — email tokens must keep working) | `action/<token>/` | KEEP (stability-critical) |

**Special direct-path mounts** (per inventory §E):
- `rack_views.rack_compute` and `rack_views.rack_export_runsheet` at direct `svend/urls.py` lines 262/267 — move with `rack_views.py` to `forgerack/` or delete
- `a3_views.remove_diagram` at direct `svend/urls.py` line 92 — moves with `a3_views.py` to `qms_a3/`

**URLs NOT present today but needed after extraction:**
- `/api/workbench/workbenches/` — pull API container browse
- `/api/workbench/artifacts/<id>/` — pull API single artifact fetch
- `/api/workbench/artifacts/<id>/<dotted.key>/` — sub-artifact fetch (architecture §2.3)
- `/api/workbench/artifacts/<id>/references/` — reference registration
- Same pattern for `qms_investigation/`, `qms_a3/` (every source and transition)
- `/api/qms-risk/`, `/api/qms-documents/`, `/api/qms-training/`, `/api/qms-suppliers/`, `/api/qms-nonconformance/`, `/api/qms-measurement/`, `/api/qms-audit/` prefixes

---

## Section 5 — Cross-app dependency hot spots (resolved per site)

Per inventory §G. Each file that imports from `agents_api` needs a target replacement: import from new app, call pull API, or update reference. Listed in priority order.

### 5.1 `loop/` ↔ `agents_api/` — 17 import sites (highest coupling)

Per inventory §K.4 (refined from 10 in prior session). Every `loop/` import site must update in lockstep with the relevant extraction CR. Architecture §9.A #14: "Accepted as unavoidable. Every QMS extraction is a 2-app coordinated CR."

| File | Line | Model imported | Target replacement | Coordinated CR |
|---|---|---|---|---|
| `loop/models.py` | 424 | `Employee` | Import from `qms_core.Employee` | Phase 4 (Site move CR) |
| `loop/models.py` | 694 | `ControlledDocument` | Import from `qms_documents.ControlledDocument` | Phase 2A qms_documents CR |
| `loop/models.py` | 879 | `TrainingRecord` | Import from `qms_training.TrainingRecord` | Phase 2A qms_training CR |
| `loop/models.py` | 884 | `ControlledDocument` | Import from `qms_documents.ControlledDocument` | Phase 2A qms_documents CR |
| `loop/models.py` | 895 | `ISOSection` | Import from `qms_documents.ISOSection` | Phase 2A qms_documents CR |
| `loop/models.py` | 962 | `Site` | Import from `qms_core.Site` | Phase 4 (Site move CR) |
| `loop/models.py` | 971 | `ControlledDocument` | Import from `qms_documents.ControlledDocument` | Phase 2A qms_documents CR |
| `loop/models.py` | 982 | `Employee` | Import from `qms_core.Employee` | Phase 4 (Site move CR) |
| `loop/models.py` | 1117 | `ISOSection` | Import from `qms_documents.ISOSection` | Phase 2A qms_documents CR |
| `loop/models.py` | 1190 | `Site` | Import from `qms_core.Site` | Phase 4 (Site move CR) |
| `loop/models.py` | 1204 | `FMEARow` | Import from `qms_risk.FMEARow` | Phase 2A qms_risk CR |
| `loop/models.py` | 1447 | `FMEARow` | Import from `qms_risk.FMEARow` | Phase 2A qms_risk CR |
| `loop/models.py` | 1850 | `SupplierRecord` | Import from `qms_suppliers.SupplierRecord` | Phase 1A-2A qms_suppliers CR |
| `loop/models.py` | 1855 | `NonconformanceRecord` | Import from `qms_nonconformance.NonconformanceRecord` | Phase 2A qms_nonconformance CR |
| `loop/models.py` | 2105 | `SupplierRecord` | Import from `qms_suppliers.SupplierRecord` | Phase 1A-2A qms_suppliers CR |
| `loop/views.py` | multiple (9 imports) | Employee, ISOSection, ControlledDocument, TrainingRecord, TrainingRequirement, CustomerComplaint, tool_events | Per-model target apps | Coordinated with each extraction |
| `loop/services.py` | 150, 253, 297 | ISODocument, FMEA/FMEARow, TrainingRequirement, ControlledDocument | Per-model target apps | Coordinated with each extraction |
| `loop/readiness.py` | 171 | `TrainingRecord` | `qms_training.TrainingRecord` | Phase 2A qms_training CR |
| `loop/evaluator.py` | 309 | `tool_events`, `MeasurementEquipment` | `tools/` (or `core/`) + `qms_measurement.MeasurementEquipment` | Phase 1A qms_measurement + tools-helper move |
| `loop/supplier_views.py` | — | `from agents_api.permissions import get_tenant` | `from qms_core.permissions import get_tenant` | **Phase 0** (permissions.py moves first) |

### 5.2 `graph/` ↔ `agents_api/` — 9 import sites

Per inventory §G.4. Graph has direct FKs into agents_api (Site, MeasurementEquipment via `linked_process_node`, ControlledDocument via M2M `linked_process_nodes`) plus 4 synara/ engine imports.

| File | Line | What | Target replacement | Coordinated CR |
|---|---|---|---|---|
| `graph/views.py` | 392 | `ControlledDocument, NonconformanceRecord, SupplierRecord` | Per-model new apps | Phases 1A-2A (each model) |
| `graph/synara_adapter.py` | 14-15 | `BeliefEngine` from `synara/belief`, multiple from `synara/kernel` | `forgesia` imports | **Phase 0** (forgesia wiring) |
| `graph/service.py` | 281 | `Evidence as SynaraEvidence` from `synara/kernel` | `forgesia.Evidence` | **Phase 0** |
| `graph/tests_synara.py` | 11 | `Evidence` from `synara/kernel` | `forgesia.Evidence` | **Phase 0** |
| `graph/integrations.py` | 601 | `MeasurementEquipment` | `qms_measurement.MeasurementEquipment` | Phase 1A qms_measurement CR |
| `graph/tests_qms.py` | 3 imports | `MeasurementEquipment` | `qms_measurement.MeasurementEquipment` | Phase 1A qms_measurement CR |
| `graph/models.py` | 520 | FK to `agents_api.Site` (string lookup) | FK to `qms_core.Site` | Phase 4 (Site move CR) |

### 5.3 `safety/` ↔ `agents_api/` — 5 cross-app FKs

| File | Line | What | Target replacement | Coordinated CR |
|---|---|---|---|---|
| `safety/models.py` | 69 | FK to `agents_api.Site` | FK to `qms_core.Site` | Phase 4 |
| `safety/models.py` | 220 | FK to `agents_api.Site` | FK to `qms_core.Site` | Phase 4 |
| `safety/models.py` | 260 | FK to `agents_api.Employee` | FK to `qms_core.Employee` | Phase 4 |
| `safety/models.py` | 344 | FK to `agents_api.Employee` | FK to `qms_core.Employee` | Phase 4 |
| `safety/models.py` | 354 | FK to `agents_api.Site` | FK to `qms_core.Site` | Phase 4 |
| `safety/views.py`, `safety/tests.py` | assorted | Site + Employee reads | Same | Phase 4 |

### 5.4 `core/` ↔ `agents_api/` — DSWResult direct refs (CANONICAL PULL CONVERSION)

| File | Line | What | Target replacement | Coordinated CR |
|---|---|---|---|---|
| `core/models/notebook.py` | 71 | Reference to `agents_api.DSWResult` | Convert to workbench pull API | Phase 1B workbench rebuild |
| `core/models/notebook.py` | 82 | Reference to `agents_api.DSWResult` | Convert to workbench pull API | Phase 1B |
| `core/models/notebook.py` | 217 | Reference to `agents_api.DSWResult` | Convert to workbench pull API | Phase 1B |
| `core/models/notebook.py` | 228 | Reference to `agents_api.DSWResult` | Convert to workbench pull API | Phase 1B |
| `core/views.py`, `core/tests.py`, `core/management/commands/seed_nlp_demo.py` | assorted | Various | Per-extraction | Various |

### 5.5 `HoshinKPI ↔ DSWResult` — the canonical conversion (architecture §4.5)

| File | Line | What | Target replacement | Coordinated CR |
|---|---|---|---|---|
| `agents_api/models.py` | **3285** | `DSWResult.objects.filter(user=..., analysis_type=...).order_by('-created_at').first()` — direct ORM query | Replace with `workbench_pull_api.fetch_artifact_value(artifact_id, key, consumer_module='hoshin', ...)`. Add new FK `linked_artifact → workbench.Artifact` + `linked_artifact_key` CharField. Auto-registers `ArtifactReference` on first access. Tombstone on source delete. | Phase 2B hoshin/ rebuild |

This is THE canonical worked example. Every other direct-ORM-query cross-app coupling discovered during extraction follows the same conversion pattern:
1. Add new FK to `workbench.Artifact` (with `on_delete=SET_NULL` for tombstone)
2. Add `_key` CharField for the dotted address into the artifact
3. Replace direct query with pull API call
4. Pull API auto-registers the `ArtifactReference` on first access
5. Source-side delete UI shows reference warnings; tombstone rendered on consumer side after delete

### 5.6 Other cross-app touchpoints

| File | Imports | Target replacement | Coordinated CR |
|---|---|---|---|
| `notifications/webhook_views.py` (×2) | `from agents_api.permissions import get_tenant` | `qms_core.permissions.get_tenant` | **Phase 0** (permissions.py move) |
| `forge/tasks.py:152` | `from agents_api.views import get_shared_llm` | Extract `get_shared_llm` to new helper module (e.g. `llm/helpers.py`) | Phase 0 (alongside agent dispatcher deletion) |
| `accounts/permissions.py`, `accounts/privacy_tasks.py` | assorted | Per-item | Phase 0 or extraction-time |
| `api/views.py`, `api/internal_views.py`, `api/landing_views.py`, `api/tasks.py` | internal dashboard/staff metrics | Per-item | Phase 0 or staged |
| `syn/audit/compliance.py`, `syn/audit/management/commands/generate_calibration_cert.py` | agents_api inspection | Update agents_api paths to new apps | Phase 3 cutover + follow-up |
| `syn/audit/tests/` (11 files) | Various | Update imports | Follow-up test rebuild |
| `syn/sched/svend_tasks.py` | Tempora scheduled tasks: `front_page_tasks`, `harada_tasks`, `commitment_tasks`, `commitment_notifications` | Move tasks to their domain apps | Coordinated with each domain extraction |

---

## Section 6 — Phase 0 forge wiring detail

**Purpose:** Sketch for the separate `phase_0_forge_wiring.md` doc. Not writing that file here. Phase 0 must land BEFORE any model extraction begins.

Per architecture §7.4: forge packages are 95%+ ready; the wiring (replacing inline imports with forge package calls in the view modules) is the missing step. Phase 0 reduces `agents_api/` from 289,313 LOC to something significantly smaller before the model extractions even start.

### Phase 0.1 — `dsw/` live files move into `analysis/`

**What:** Per architecture §9.A J.11 + inventory §K.2. `analysis/` is canonical. Move:
- `dsw/common.py` (3,084 LOC)
- `dsw/endpoints_data.py` (1,832 LOC)
- `dsw/endpoints_ml.py` (1,702 LOC)
- `dsw/standardize.py` (552 LOC)
- `dsw/chart_render.py` (59 LOC — the actual implementation)
- `dsw/chart_defaults.py` (459 LOC — the actual implementation)

into `analysis/`. Delete the `analysis/chart_render.py` (7-line wrapper) and `analysis/chart_defaults.py` (11-line wrapper) DOA stubs per architecture §7.3.

**View files touched:** `spc_views.py`, `autopilot_views.py`, `forecast_views.py`, `pbs_engine.py`, `ml_pipeline.py`, `dsw_views.py`, `report_views.py`, `a3_views.py` (all import from `agents_api.dsw.*`).

**CR partners:** none (within-app reorganization).

**Safety-net behavior tests:** existing `tests/test_dsw_views_behavioral.py` verifies `_read_csv_safe`. Add ~5 tests covering the move: "upload CSV via dsw_views → correct response shape", "chart render → expected ChartSpec dict", "standardize post-process → evidence_grade present".

### Phase 0.2 — `forgespc` wiring + delete `agents_api/spc.py`

**What:** Wire `forgespc` package into `spc_views.py` replacing inline imports from `agents_api.spc` (1,889 LOC file) and any `dsw/spc*` legacy code. Delete `agents_api/spc.py` and `dsw/spc.py` + `dsw/spc_pkg/*`.

**View files touched:** `spc_views.py` (1,547 LOC, 15 funcs).

**CR partners:** none (if tests pass before/after).

**Safety-net behavior tests:** ~7 tests — one per major SPC analysis type (xbar-r, xbar-s, imr, c, p, capability study, gage R&R). Each asserts the 10-key result contract still populates correctly from forgespc.

### Phase 0.3 — `forgestat` wiring + delete legacy `dsw/` stats files

**What:** Per inventory §H.2 + architecture §7.3. After the dsw→analysis move in 0.1, the legacy compute files remain: `dsw/stats_parametric.py`, `stats_nonparametric.py`, `stats_posthoc.py`, `stats_regression.py`, `stats_advanced.py`, `stats_exploratory.py`, `stats_quality.py`, `bayesian.py`, `bayesian/*`, `ml.py`, `viz.py`, `siop.py`, `simulation.py`, `reliability.py`, `d_type.py`, `exploratory/*` — total ~50,000-55,000 LOC.

Wire `forgestat` into `analysis/forge_*.py` bridges for any remaining handlers (most of the wiring is already done per memory: "153 handlers across 11 forge files. They get relocated as part of extraction, not re-done."). Then delete the legacy files.

**View files touched:** `dsw_views.py`, `autopilot_views.py`, `forecast_views.py`, any view that still calls into `dsw/stats_*`.

**CR partners:** none.

**Safety-net behavior tests:** Per-analysis-family behavior tests — one for parametric (t-test), nonparametric (Mann-Whitney), regression (OLS), ANOVA, posthoc (Tukey), quality (Cpk handled in Phase 0.2), advanced (mixed effects), exploratory (PCA). ~8-10 tests.

### Phase 0.4 — `forgesia` wiring + delete `agents_api/synara/`

**What:** Per migration plan Tech Debt: "Wire to forgesia when `__init__` exports fixed." Fix the `forgesia/__init__.py` exports first (known blocker). Then wire into `synara_views.py`. Update `graph/synara_adapter.py`, `graph/service.py`, `graph/tests_synara.py` to import from `forgesia` instead of `agents_api.synara.*`. Delete `agents_api/synara/` (6 files, 3,243 LOC).

**View files touched:** `synara_views.py` (1,136 LOC, 32 funcs) + `graph/synara_adapter.py`, `graph/service.py`, `graph/tests_synara.py`.

**CR partners:** `graph/` — this is a 2-app CR.

**Safety-net behavior tests:** ~5 tests — "BeliefEngine.update_belief with evidence → correct posterior", "DSL parse + evaluate → expected kernel state", "Synara API endpoint returns correct belief state".

### Phase 0.5 — `forgeviz` completion wiring

**What:** Ensure any remaining legacy chart helpers post Phase 0.1 go through `forgeviz`. If `dsw/chart_render.py` + `dsw/chart_defaults.py` got moved to `analysis/` in Phase 0.1 but still use inline Plotly, wire them through `forgeviz` helpers. Delete the now-moved chart files once forgeviz covers their cases.

**View files touched:** `report_views.py` (uses `dsw/chart_render` legacy), `spc_views.py`, `a3_views.py`, any view rendering charts.

**CR partners:** none.

**Safety-net behavior tests:** ~3-5 tests — one per chart category (control chart, capability histogram, residuals, bar, line). Each asserts the ChartSpec dict shape matches forgeviz output.

### Phase 0.6 — Confirmed dead-code deletion

**What:** Per inventory §I.3:
- `agents_api/views.py` (441 LOC) — delete except `get_shared_llm`; extract that helper to new module
- `agents_api/urls.py` (16 LOC) — delete agent dispatch routes; comment out `/api/agents/` mount in `svend/urls.py`
- `Workflow` model (`agents_api/models.py:17`) + `workflow_views.py` (433 LOC) + `workflow_urls.py` (11 LOC) — full delete
- `ActionItem` model (`agents_api/models.py:2406`) + `action_views.py` (65 LOC) + `action_urls.py` (21 LOC) — full delete. Update 5 view files that reference ActionItem (fmea_views, rca_views, a3_views, hoshin_views, action_views) to use `loop.Commitment`.
- `AuditChecklist` (`agents_api/models.py:5800`) — verify no live tenants via grep; if none, delete (Checklist supersedes)
- `RackSession` (`agents_api/models.py:6720`) — pending Eric confirmation on ForgeRack status; if demo-only, delete; if still active, move to new `forgerack/` app
- Original agent dispatchers: `researcher_agent`, `writer_agent`, `editor_agent`, `experimenter_agent`, `eda_agent` — all 5 dispatch functions in `agents_api/views.py`
- Sweep test files violating TST-001 §10.6: `tests/test_endpoint_smoke.py` (712 LOC), `tests/test_t1_deep.py` (1,942 LOC), `tests/test_t2_views_smoke.py` (338 LOC), `tests/test_*_coverage.py` family (8 files)

**CR partners:** `fmea_views.py`, `rca_views.py`, `a3_views.py`, `hoshin_views.py` (ActionItem reference updates); `forge/tasks.py:152` (get_shared_llm move).

**Safety-net behavior tests:** ~3 tests asserting `loop.Commitment` workflow handles the use cases that ActionItem previously served (create commitment from FMEA finding, from RCA, from A3).

### Phase 0.7 — `permissions.py` move to `qms_core/` shim

**What:** Move `agents_api/permissions.py` to a new stub `qms_core/` app as `qms_core/permissions.py`. Update cross-app imports:
- `loop/supplier_views.py` (`from agents_api.permissions import get_tenant`)
- `notifications/webhook_views.py` (×2) (same)
- `syn/audit/compliance.py`
- Any other importer

`agents_api/permissions.py` becomes a shim that re-exports from `qms_core.permissions` for any remaining internal callers during the parallel period. This is a prerequisite for every subsequent extraction because all qms_* view files will import permissions.

**CR partners:** `loop/`, `notifications/`, `syn/audit/`.

**Safety-net behavior tests:** ~3 tests — "qms_can_edit returns correct permission for site admin vs user", "qms_queryset filters to user's tenant", "get_tenant returns current user's tenant".

### Phase 0.8 — `iso_views.py` split-in-place (prerequisite)

**What:** Per architecture §9.A #15 + §7.6. Split `iso_views.py` (4,874 LOC, 85 view functions) into sub-modules within `agents_api/` first, WITHOUT moving them to their target apps yet:
- `agents_api/iso/ncr_views.py`
- `agents_api/iso/audit_views.py`
- `agents_api/iso/training_views.py`
- `agents_api/iso/management_review_views.py`
- `agents_api/iso/document_views.py`
- `agents_api/iso/supplier_views.py`
- `agents_api/iso/equipment_views.py`
- `agents_api/iso/complaint_views.py`
- `agents_api/iso/risk_views.py`
- `agents_api/iso/afe_views.py`
- `agents_api/iso/control_plan_views.py`
- `agents_api/iso/checklist_views.py`

Update `iso_urls.py` to point at each sub-module. Then the Phase 2A extractions can move individual files to their target apps.

**CR type:** `enhancement` (no model changes, file reorganization).

**CR partners:** none.

**Safety-net behavior tests:** Use the existing 351-test `iso_tests.py` as the regression suite. All 351 tests must still pass after the split.

### Phase 0 summary

| Sub-phase | What | CRs | Risk |
|---|---|---|---|
| 0.1 | Move `dsw/` live files into `analysis/`; delete wrappers | 1 | LOW |
| 0.2 | Wire `forgespc`; delete `agents_api/spc.py` + `dsw/spc*` | 1 | MED |
| 0.3 | Wire `forgestat`; delete legacy `dsw/stats*`, `bayesian`, `ml`, `viz`, etc. | 1-2 | MED |
| 0.4 | Wire `forgesia`; delete `agents_api/synara/` | 1 | MED (2-app) |
| 0.5 | Wire `forgeviz` completion | 1 | LOW |
| 0.6 | Dead-code deletion (Workflow, ActionItem, agents, sweep tests) | 1-2 | LOW-MED |
| 0.7 | `permissions.py` → `qms_core/` shim | 1 | LOW (coordinates 3 apps) |
| 0.8 | `iso_views.py` split-in-place prerequisite | 1 | MED (uses iso_tests.py regression suite) |
| **TOTAL** | | **~8 CRs** | |

After Phase 0, `agents_api/` is significantly reduced (~60,000-70,000 LOC removed) and the forge boundaries are clean. The actual model/view extractions in Phase 1+ run against a much smaller surface.

---

## Section 7 — Phase 1 leaf extractions sketch

The 7 lowest-risk leaf apps that kick off Phase 1A. All LOW or MED risk, minimal loop/ coupling, small surface. Each is a 2-step sequence: 1A relocate, 1B rebuild.

Per universal cutover pattern (architecture §13), all run at parallel `/app/demo/<thing>/` paths until Phase 3 cutover.

### 7.1 `triage/`

- **Models (1):** TriageResult
- **Views:** `triage_views.py` (512 LOC, 10 funcs)
- **URLs:** `triage_urls.py` → `/app/demo/triage/`
- **CR partners:** none (self-contained)
- **Risk:** LOW
- **Effort:** S
- **Phase 1A:** Model + views + URLs move AS-IS to `triage/` Django app at `/app/demo/triage/`.
- **Phase 1B:** Add pull contract endpoints (`/api/triage/containers/`, `/api/triage/artifacts/<id>/`, reference registration), `TriageResultReference` model, delete-friction handling.
- **Source role:** Yes — triaged datasets are pulled by every analysis as evidence.

### 7.2 `whiteboard/`

- **Models (4):** Board, BoardParticipant, BoardVote, BoardGuestInvite
- **Views:** `whiteboard_views.py` (1,057 LOC, 23 funcs)
- **URLs:** `whiteboard_urls.py` → `/app/demo/whiteboard/`
- **Tests:** `whiteboard_tests.py` (1,181 LOC, 83 funcs) — behavioral per inventory F
- **CR partners:** `notebook_views.py`, `iso_doc_views.py`, `report_views.py`, `a3_views.py` (all have board embeds) — may defer update until cutover if they use lazy imports
- **Risk:** LOW
- **Effort:** M
- **Phase 1A:** Full family relocates.
- **Phase 1B:** Pull contract for boards (boards as causal-claim sources).
- **Source role:** Yes — boards pulled as evidence by investigations/A3.

### 7.3 `vsm/`

- **Models (1):** ValueStreamMap
- **Views:** `vsm_views.py` (433 LOC, 14 funcs)
- **URLs:** TBD (inventory §E notes "no `vsm_urls.py`") — may be served via `hoshin_urls.py` or direct imports
- **CR partners:** `HoshinProject.source_vsm` FK, `PlantSimulation.source_vsm` FK, `xmatrix_views.py` → must update lazy-imports
- **Risk:** MED (rebuild target per migration plan)
- **Effort:** M (relocate) + XL (cockpit rebuild)
- **Phase 1A:** Model + views relocate.
- **Phase 1B:** Full cockpit rebuild per migration plan "VSM Workbench Spec" — integrated calculator panel, forgesiop + forgequeue, ForgeViz charts.
- **Source/sink role:** Sink for analysis (cycle-time data pulled in), source for kaizen bursts (pulled into Hoshin).

### 7.4 `simulation/`

- **Models (1):** PlantSimulation
- **Views:** `plantsim_views.py` (391 LOC, 8 funcs)
- **URLs:** `plantsim_urls.py` → `/app/demo/simulation/`
- **CR partners:** `vsm/` (source_vsm FK must be rewired after VSM move)
- **Risk:** MED
- **Effort:** M
- **Special handling:** Eric was upgrading simulators in parallel; coordinate to not stomp on his work.
- **Phase 1A:** Model + views relocate. `source_vsm` FK rewires after `vsm/` moves in same wave.
- **Phase 1B:** Rebuild with ForgeSiop discrete-event engine (per migration plan P2).

### 7.5 `qms_measurement/`

- **Models (1):** MeasurementEquipment
- **Views:** `equipment_*` slice of split `iso_views.py`, MSA dispatch in `spc_views.py`
- **URLs:** `/api/iso/equipment/` → `/app/demo/qms-measurement/` → `/api/qms-measurement/`
- **CR partners:** **`graph/`** (direct FK + 3 test references + integrations), `loop/evaluator.py:309`, `spc_views.py`
- **Risk:** MED (graph coupling)
- **Effort:** S
- **Phase 1A:** Model + equipment_views slice relocate. Graph FK atomically swaps from `agents_api.MeasurementEquipment` to `qms_measurement.MeasurementEquipment`.
- **Phase 1B:** Source role pull contract — calibration records pulled into audits.
- **Source role:** Yes — calibration evidence pulled by audits.

### 7.6 `qms_suppliers/`

- **Models (2):** SupplierRecord, SupplierStatusChange
- **Views:** `supplier_*` slice of split `iso_views.py`
- **URLs:** `/api/iso/suppliers/` → `/app/demo/qms-suppliers/` → `/api/qms-suppliers/`
- **CR partners:** `loop/models.py:1850, 2105` (2 loop FKs), `graph/views.py:392`, `NonconformanceRecord.supplier` FK
- **Risk:** MED (loop × 2)
- **Effort:** S
- **Phase 1A:** Model + supplier_views slice relocate. Loop imports swap.
- **Phase 1B:** Source role pull contract — supplier records pulled into supplier qualification audits.
- **Source role:** Yes.

**Note:** `qms_suppliers/` is borderline — it could land late in Phase 1A or early in Phase 2A depending on loop/ coordination scheduling. Placing it in Phase 1A buys leaf-extraction muscle memory before the medium-coupling work.

### 7.7 `learn/`

- **Models (3):** SectionProgress, AssessmentAttempt, LearnSession
- **Views:** `learn_views.py` (2,450 LOC, 32 funcs), `harada_views.py` (909 LOC) optionally
- **URLs:** `learn_urls.py`, `harada_urls.py` → `/app/demo/learn/`
- **Sub-content:** `learn_content/` directory (~14,556 LOC), `learn_content.py` (8,024 LOC — see §11)
- **CR partners:** none significant
- **Risk:** LOW
- **Effort:** L (large LOC, many helper files)
- **Phase 1A:** All models + `learn_content*` content data move wholesale to `learn/` app.
- **Phase 1B:** Standalone product surface — not in pull graph. Minor rebuild for sv-* widgets.

### Phase 1 summary

| App | Models | Views LOC | Risk | CR partners |
|---|---|---|---|---|
| `triage/` | 1 | 512 | LOW | none |
| `whiteboard/` | 4 | 1,057 | LOW | 4 view files lazy update |
| `vsm/` | 1 | 433 | MED | hoshin, simulation (lazy FK) |
| `simulation/` | 1 | 391 | MED | vsm |
| `qms_measurement/` | 1 | (part of iso_views) | MED | graph (hard), loop, spc |
| `qms_suppliers/` | 2 | (part of iso_views) | MED | loop ×2, graph |
| `learn/` | 3 | 2,450 + 22,580 content | LOW | none |
| **TOTAL** | **13 models** | | | |

---

## Section 8 — Phase 2 medium-coupling extractions sketch

After Phase 1 leaves are demo-tested and signed off, the medium-coupling apps extract. Each is a 2-app (or 3-app) coordinated CR because `loop/` and `graph/` couple in.

### 8.1 `qms_risk/` — FMEA + Risk register

- **Models (3):** FMEA, FMEARow, Risk
- **Views:** `fmea_views.py` (1,589 LOC, 27 funcs), `risk_*` slice of split `iso_views.py`
- **URLs:** `/api/fmea/`, `/api/iso/risks/` → `/app/demo/qms-risk/` → `/api/qms-risk/`
- **CR partners:** **`loop/` (2 FMEARow FKs + FMEA imports)**, `safety/models.py:736`, `notebook_views.py`, `learn_views.py`
- **Risk:** HIGH
- **Effort:** M
- **Phase 2A:** Models + views relocate. Loop FKs atomically swap.
- **Phase 2B:** Source role pull contract — FMEAs and individual risks pulled into investigations.
- **Special handling:** `FMEARow.hypothesis_link` to `core.Hypothesis` stays (cross-app but benign — core is foundation).

### 8.2 `qms_documents/` — Controlled documents + ISO authoring

- **Models (7):** ControlledDocument, DocumentRevision, DocumentStatusChange, ISODocument, ISOSection, ControlPlan, ControlPlanItem
- **Views:** `document_*` slice of split `iso_views.py` + `iso_doc_views.py` (716 LOC)
- **URLs:** `/api/iso/documents/`, `/api/iso-docs/` → `/app/demo/qms-documents/` → `/api/qms-documents/`
- **CR partners:** **`loop/` (6 import sites!)**, **`graph/` (M2M `linked_process_nodes`)**, `training_views`, `audit_views`
- **Risk:** **HIGH** (highest coupling after Site)
- **Effort:** L
- **Phase 2A:** Models + views relocate. Loop imports swap (6 sites). Graph M2M reference string updates.
- **Phase 2B:** Source role pull contract — documents pulled into audits + training. ForgeDoc integration.
- **Special handling:** `ControlledDocument.linked_process_nodes` is an M2M through table. Django M2M relocation requires a data migration with explicit through-model handling. **Highest-effort single CR after Site.**

### 8.3 `qms_training/` — TWI competency

- **Models (3):** TrainingRequirement, TrainingRecord, TrainingRecordChange
- **Views:** `training_*` slice of split `iso_views.py`
- **URLs:** `/api/iso/training/` → `/app/demo/qms-training/` → `/api/qms-training/`
- **CR partners:** `loop/models.py:879`, `loop/views.py:2386`, `loop/services.py:253`, `loop/readiness.py:171` (4 loop sites)
- **Risk:** HIGH (loop × 4)
- **Effort:** M
- **Phase 2A:** Models + views relocate.
- **Phase 2B:** Source role pull contract — training records pulled into supplier audits + internal audits.

### 8.4 `qms_nonconformance/` — NCR + Complaints

- **Models (3):** NonconformanceRecord, NCRStatusChange, CustomerComplaint
- **Views:** `ncr_*`, `complaint_*` slices of split `iso_views.py`
- **URLs:** `/api/iso/ncrs/`, `/api/iso/complaints/` → `/app/demo/qms-nonconformance/` → `/api/qms-nonconformance/`
- **CR partners:** **`loop/models.py:1855`**, **`graph/views.py:392`**, `AuditFinding.ncr` FK, `CustomerComplaint.ncr` FK, `Report`/`CAPAReport.ncr` (removed at Phase 2B)
- **Risk:** HIGH (loop + graph)
- **Effort:** M
- **Phase 2A:** Models + views relocate.
- **Phase 2B:** Transition+source dual role pull contract. NCRs pulled into investigations; investigations push back findings into NCRs via reference registration.

### 8.5 `qms_audit/` — Internal audit + management review

- **Models (4, -1 delete):** InternalAudit, AuditFinding, ManagementReviewTemplate, ManagementReview (+ AuditChecklist DELETE)
- **Views:** `audit_*`, `review_*` slices of split `iso_views.py`
- **URLs:** `/api/iso/audits/`, `/api/iso/reviews/` → `/app/demo/qms-audit/` → `/api/qms-audit/`
- **CR partners:** `AuditFinding.ncr` FK (updates after NCR moves), `ManagementReview.data_snapshot` consumers
- **Risk:** MED
- **Effort:** L
- **Phase 2A:** Models + views relocate. Must land AFTER `qms_nonconformance/` so `AuditFinding.ncr` FK lands correctly.
- **Phase 2B:** Transition pull contract — audits pull evidence from documents, training, supplier records. Auditor independence rules per `project_audit_upgrade.md`.

### 8.6 `qms_investigation/` — RCA + Ishikawa + CEMatrix + Investigation bridge

- **Models (3):** RCASession, IshikawaDiagram, CEMatrix
- **Views:** `rca_views.py` (1,057 LOC), `ishikawa_views.py` (158 LOC), `ce_views.py` (155 LOC), `investigation_views.py` (432 LOC), `investigation_bridge.py` (638 LOC)
- **URLs:** `/api/rca/`, `/api/investigations/`, `/api/iso/rca/` → `/app/demo/qms-investigation/`
- **CR partners:** `a3_views.py` (A3.rca_session FK), `notebook_views.py`, `qms_views.py`, `iso_views.py`, `NonconformanceRecord.rca_session` FK, `CAPAReport.rca_session` FK (removed at deletion)
- **Risk:** HIGH (state machine preservation, multi-source FK references)
- **Effort:** L
- **Phase 2A:** Models + views + `investigation_bridge.py` + `evidence_bridge.py` + `evidence_weights.py` relocate together.
- **Phase 2B:** Transition pull contract — pulls from workbench + triage + whiteboard; emits investigation containers. Canonical transition app.

### 8.7 `qms_a3/` — A3 reports (full rebuild)

- **Models (1):** A3Report
- **Views:** `a3_views.py` (1,389 LOC, 23 funcs)
- **URLs:** `/api/a3/` → `/app/demo/qms-a3/` → `/api/qms-a3/`
- **CR partners:** `rca_views.py`, `notebook_views.py`, `learn_views.py`, Site FK (Phase 4), RCASession FK (updates with qms_investigation move)
- **Risk:** HIGH (Eric rebuild flag)
- **Effort:** XL (full rebuild)
- **Phase 2A:** Model + views relocate AS-IS to `qms_a3/` at `/app/demo/qms-a3/`.
- **Phase 2B:** **Full rebuild** per Eric (architecture §3.3). New architecture, updated models, new pull contract (pulls from workbench + qms_investigation, emits A3 container). New front-end.

### 8.8 `hoshin/` — Hoshin Kanri + AFE + X-matrix (with HoshinKPI conversion)

- **Models (9):** HoshinProject, ProjectTemplate, StrategicObjective, AnnualObjective, HoshinKPI, XMatrixCorrelation, AFE, AFEApprovalLevel, ResourceCommitment
- **Views:** `hoshin_views.py` (2,094 LOC, 36 funcs), `xmatrix_views.py` (1,069 LOC, 13 funcs)
- **URLs:** `/api/hoshin/` → `/app/demo/hoshin/`
- **CR partners:** `qms_core/` Site + Employee FKs (Phase 4), `vsm/` (source_vsm FK), `fmea_views.py` (AFE creation refs removed per AFE policy), `workbench/` (HoshinKPI pull contract conversion)
- **Risk:** HIGH (canonical pull contract conversion)
- **Effort:** L
- **Phase 2A:** Models + views relocate.
- **Phase 2B:** **Canonical pull contract rewrite for `HoshinKPI.effective_actual`** per architecture §4.5. Add `linked_artifact` FK + `linked_artifact_key` field. Replace direct `DSWResult.objects.filter` with pull API call. Auto-register references. Tombstone handling.

### 8.9 `reports/` + `sop/` — New sinks (brand new, not extractions)

- **Models:** new (no agents_api models to relocate)
- **Views:** new (replaces `report_views.py` at Phase 3 cutover)
- **URLs:** `/api/reports/`, `/api/sop/` → `/app/demo/reports/`, `/app/demo/sop/`
- **CR partners:** deletion of `Report`, `report_views.py`, `CAPAReport`, `capa_views.py` at Phase 3
- **Risk:** HIGH (brand new builds; replaces live production `/api/reports/`)
- **Effort:** XL
- **Phase 2B:** Build new sink apps. Pull contract consumers only — sinks don't emit containers. Use ForgeDoc for PDF rendering.

### Phase 2 summary

| App | Models | Risk | Key CR partners |
|---|---|---|---|
| `qms_risk/` | 3 | HIGH | loop ×2, safety |
| `qms_documents/` | 7 | HIGH | loop ×6, graph M2M |
| `qms_training/` | 3 | HIGH | loop ×4 |
| `qms_nonconformance/` | 3 | HIGH | loop, graph |
| `qms_audit/` | 4 (-1 delete) | MED | qms_nonconformance (FK ordering) |
| `qms_investigation/` | 3 | HIGH | a3, notebook, ncr, capa |
| `qms_a3/` | 1 | HIGH | rca, notebook, workbench |
| `hoshin/` | 9 | HIGH | **workbench pull contract conversion** |
| `reports/` + `sop/` | new | HIGH | deletion of Report + CAPA |
| **TOTAL** | **33 models + new** | | |

Phase 2A ordering constraints:
1. `qms_nonconformance/` before `qms_audit/` (AuditFinding.ncr FK)
2. `qms_investigation/` before `qms_a3/` (A3.rca_session FK)
3. `qms_documents/` should land before `qms_training/` (TrainingRequirement.controlled_doc FK) and `qms_audit/` (audits reference documents)
4. `workbench/` pull contract infrastructure (from Phase 1B) must exist before `hoshin/` HoshinKPI conversion

---

## Section 9 — `qms_core/` and Site final move (Phase 4)

Per architecture §9.B.2: Site stays in agents_api until **the full parallel rebuild is complete, tested, and reviewable in `/app/demo/...` paths.** Phase 4 is the final atomic move.

### 9.1 What moves in Phase 4

All models destined for `qms_core/`:
- `Site` (line 2020)
- `SiteAccess` (line 2072)
- `Employee` (line 2511)
- `ActionToken` (line 2669)
- `QMSFieldChange` (line 4965)
- `QMSAttachment` (line 6390)
- `Checklist` (line 5843)
- `ChecklistExecution` (line 5923)
- `ElectronicSignature` (line 6232 — SynaraImmutableLog subclass)

Plus view files:
- `qms_views.py` (216 LOC, 1 func — cross-app dashboard)
- `token_views.py` (214 LOC, 7 funcs)
- Slices of `iso_views.py` that landed in the split but didn't have a non-qms_core home (e.g. checklist_* views)

Plus `permissions.py` (already moved as a Phase 0 shim — the shim is dropped now and callers point directly at `qms_core.permissions`).

### 9.2 FK update locations (all atomic in Phase 4 CR)

Every model that has a `FK → agents_api.Site` must have its FK string-lookup rewritten to `FK → qms_core.Site`. These are in the new extracted apps from Phases 1-2:

| FK declaration location | Model | Current FK target | New FK target |
|---|---|---|---|
| `qms_a3/models.py` | A3Report.site | `agents_api.Site` | `qms_core.Site` |
| `qms_risk/models.py` | FMEA.site, Risk.site | `agents_api.Site` | `qms_core.Site` |
| `qms_investigation/models.py` | RCASession.site | `agents_api.Site` | `qms_core.Site` |
| `hoshin/models.py` | ProjectTemplate.site, HoshinProject.site, AnnualObjective.site, AFE.site | `agents_api.Site` | `qms_core.Site` |
| `qms_audit/models.py` | InternalAudit.site | `agents_api.Site` | `qms_core.Site` |
| `qms_training/models.py` | TrainingRequirement.site | `agents_api.Site` | `qms_core.Site` |
| `qms_documents/models.py` | ControlledDocument.site, ControlPlan.site | `agents_api.Site` | `qms_core.Site` |
| `qms_nonconformance/models.py` | NonconformanceRecord.site, CustomerComplaint.site | `agents_api.Site` | `qms_core.Site` |
| `qms_measurement/models.py` | MeasurementEquipment.site | `agents_api.Site` | `qms_core.Site` |
| `qms_core/models.py` (Checklist.site, SiteAccess.site, Employee.site) | internal — same-app references | — | |
| `loop/models.py:962, 1190` | Site string lookup | `agents_api.Site` | `qms_core.Site` |
| `safety/models.py:69, 220, 354` | Site string lookup | `agents_api.Site` | `qms_core.Site` |
| `graph/models.py:520` | Site string lookup | `agents_api.Site` | `qms_core.Site` |
| `loop/models.py:424, 982` | Employee string lookup | `agents_api.Employee` | `qms_core.Employee` |
| `safety/models.py:260, 344` | Employee string lookup | `agents_api.Employee` | `qms_core.Employee` |
| `hoshin/models.py` | ResourceCommitment.employee | `agents_api.Employee` | `qms_core.Employee` |

Plus `ElectronicSignature` FKs for CFR-compliant sign-off in `AFEApprovalLevel`, `CAPAReport` (if still exists — probably deleted earlier), management review flows.

### 9.3 Why this is the highest-risk single CR

- **Every extracted app modifies its FK targets in the same CR.** Django migration files across 15+ apps must coordinate. A single mistake breaks production.
- **Cross-app string lookups in `loop/`, `safety/`, `graph/`** must update in lockstep. Production writes depend on these.
- **Hash-chained ElectronicSignature** must not break. The hash chain's immutability is a compliance requirement — any migration that re-creates the table invalidates all prior signatures.
- **19 intra-app FKs + 5 cross-app = 24 callers** all update atomically.

### 9.4 Why it's last

Per architecture §9.B.2: Site stays in `agents_api/` throughout the parallel-build period so that dependents can reference `agents_api.Site` via string lookup while their new apps exist at demo paths. Only after Eric reviews the full demo build and signs off does Site move, and at that point every dependent's new app already exists with its FK declaration ready to update.

The Phase 4 CR:
1. Creates `qms_core/` Django app with models.py
2. Runs `--state-operations` migration to relocate Site, SiteAccess, Employee, etc. (table names unchanged for data preservation where possible)
3. Updates every FK string reference across all 15+ apps in the same commit
4. Deletes `agents_api/models.py` models (Site, Employee, etc.)
5. Deletes remaining `agents_api/` residue per §10

Risk mitigation:
- Two-phase migration (`--state-operations` separated from `--database`) per architecture §9.A #12
- Full backup + WAL archive check before CR starts (per memory: `/home/postgres_wal_archive` is the safe location)
- Dry-run on staging (if applicable) or on a clone of production DB
- Rollback script ready

---

## Section 10 — Deletions full list

Every file, class, or function that gets deleted (not relocated) with rationale and source citation. Organized by phase.

### 10.1 Phase 0 deletions

| Item | LOC | Rationale | Source |
|---|---|---|---|
| `agents_api/views.py` (5 agent dispatchers: researcher, writer, editor, experimenter, eda) | ~400 (of 441) | Inventory §H.1: "the 5 agent dispatchers themselves are likely unused at the HTTP layer." Custom LLMs removed per inline comment. | inventory H.1 |
| `agents_api/urls.py` (agent routes) | 16 | Routes delete with dispatchers. | inventory H.1 |
| `Workflow` model (`models.py:17`) + `workflow_views.py` + `workflow_urls.py` | ~450 | Inventory §H.5: zero non-self FKs; dead. | inventory H.5 |
| `ActionItem` model (`models.py:2406`) + `action_views.py` + `action_urls.py` | ~170 | LOOP-001 `Commitment` supersedes. Architecture §9.A #4. | architecture 9.A #4 |
| `AuditChecklist` model (`models.py:5800`) | ~43 | Superseded by generic `Checklist`. Inventory §H.6. | inventory H.6 |
| `analysis/chart_render.py` (7 LOC wrapper) | 7 | DOA wrapper around `dsw/chart_render`. Architecture §7.3. | architecture 7.3 |
| `analysis/chart_defaults.py` (11 LOC wrapper) | 11 | Same. | architecture 7.3 |
| `agents_api/synara/*.py` (after forgesia wiring) | 3,243 | Replaced by `forgesia` package. Inventory §H.2. | inventory H.2, migration plan |
| `agents_api/spc.py` (1,889 LOC) | 1,889 | Replaced by `forgespc`. Migration plan tech debt + inventory §H.2. | migration plan Tech Debt |
| Legacy `dsw/stats_*.py` (parametric, nonparametric, posthoc, regression, advanced, exploratory, quality) | ~21,988 | Replaced by `forgestat`. Inventory §H.2. | inventory H.2, migration plan |
| `dsw/bayesian.py` + `dsw/bayesian/*` | ~10,700 | Replaced by `forgestat.bayesian`/`forgesia`. | inventory H.2 |
| `dsw/ml.py` (4,624) | 4,624 | Replaced by `forgestat.ml`. | inventory H.2 |
| `dsw/viz.py` (2,931) | 2,931 | Replaced by `forgeviz`. | inventory H.2 |
| `dsw/siop.py` (2,482) | 2,482 | Replaced by `forgesiop`. | inventory H.2 |
| `dsw/simulation.py` (1,090) | 1,090 | Replaced by `forgesiop.simulation`. | inventory H.2 |
| `dsw/reliability.py` (1,521) | 1,521 | Replaced by `forgestat.reliability` or dedicated package. | inventory H.2 |
| `dsw/d_type.py` (2,929) | 2,929 | Legacy data-type helpers. | inventory H.2 |
| `dsw/spc.py` + `dsw/spc_pkg/*` (~10,000) | ~10,000 | Replaced by `forgespc`. | inventory H.2 |
| `dsw/exploratory/*` | ~3,000 | Replaced by `forgestat.exploratory`. | inventory H.2 |
| Sweep test files: `tests/test_endpoint_smoke.py` (712), `tests/test_t1_deep.py` (1,942), `tests/test_t2_views_smoke.py` (338), `tests/test_*_coverage.py` family (8 files) | ~6,000 | TST-001 §10.6 violation. Inventory §F.2, §H.4. | TST-001, inventory F.2, H.4 |
| `tests/test_bounds_exhaustive.py` | — | Likely sweep. | inventory F.2 |

**Phase 0 deletion total:** roughly **60,000-70,000 LOC** plus 4 models.

### 10.2 Phase 2B deletions (delete-with-replacement)

| Item | Rationale | Replacement |
|---|---|---|
| `agents_api/models.py Report` (line 736) + `report_views.py` (802 LOC) + `report_urls.py` | Architecture §9.A J.13: Delete + replace. | New `reports/` sink |
| `agents_api/models.py CAPAReport` (line 3723) + CAPAStatusChange + `capa_views.py` (711 LOC) + `capa_urls.py` + `iso/capa_*` view slice | Architecture §9.A #3: Delete + replace. Migration plan flags deprecated. | `qms_investigation/` + ForgeDoc CAPA generator |
| `agents_api/report_types.py` | Registry constant for deleted Report model. | New `reports/` registry |
| `workbench/models.py KnowledgeGraph` | Architecture §9.B.1 (Eric decision 2026-04-09): single canonical KG in `graph/`. | `graph/` app's canonical KG |

### 10.3 Phase 3 cutover deletions

| Item | Rationale |
|---|---|
| Legacy `/api/agents/`, `/api/workflows/`, `/api/actions/`, `/api/capa/`, `/api/reports/` (old) URL mounts in `svend/urls.py` | All replaced by new routes at demo-path locations. |
| Legacy template files per migration plan: `workbench_new.html` (11,790 LOC), `safety_coming_soon.html` (1,993), `iso_9001_qms.html` (1,159) | Architecture §7.1 + migration plan. |
| `core/models/graph.py` (~200 LOC) | DEPRECATED per migration plan; replaced by `graph/` app. |

### 10.4 Phase 4 deletions

| Item | Rationale |
|---|---|
| `agents_api/models.py` residual (everything already extracted) | At Phase 4 Site move, the file should be empty. Delete. |
| `agents_api/` app itself | If completely emptied, remove from `INSTALLED_APPS` in `svend/settings.py`. If `tool_router`, `tool_registry`, etc. remain per inventory §I.4, rename to `tools/` or absorb into `core/`. |
| Any orphaned `*_tests.py` files at the old `agents_api/` root that referenced deleted models |
| Any residual `urls.py` / `apps.py` shims |

### 10.5 Uncertain deletions (defer to extraction-time)

| Item | Decision criterion | Source |
|---|---|---|
| `agents_api/cache.py` (302 LOC) | If `CacheEntry` relocates to `syn/` or `core/`, delete. Otherwise stays as helper. | inventory I.4 |
| `agents_api/embeddings.py`, `gpu_manager.py`, `llm_manager.py`, `llm_service.py` | Per inventory §I.4: "Lifts to `chat/` or `core/`." Relocation not deletion. | inventory I.4 |
| `agents_api/pbs_engine.py` (4,070 LOC) | **Defer — see §11** | inventory J.15 |
| `agents_api/quality_economics.py` (1,141 LOC) | Lifts to `forgesiop` or stays as domain helper per inventory I.4. | inventory I.4 |
| `agents_api/bayes_core.py`, `bayes_doe.py` | Move with consumers per inventory I.4. | inventory I.4 |
| `agents_api/causal_discovery.py`, `interventional_shap.py`, `clustering.py`, `conformal.py`, `drift_detection.py`, `anytime_valid.py`, `ml_pipeline.py`, `msa_bayes.py` | Move to `workbench/handlers/` or replaced by forge. | inventory I.4 |
| `agents_api/iso_document_types.py` | Registry constants. Move with `qms_documents/`. | inventory I.4 |
| `agents_api/front_page_tasks.py`, `harada_tasks.py`, `commitment_tasks.py`, `commitment_notifications.py` | Tempora scheduled tasks. Move with domain apps. | inventory I.4 |
| `agents_api/learn_content.py` (8,024 LOC flat file) | **Defer — see §11** | inventory J.14 |

---

## Section 11 — Deferred items needing resolution at extraction time

Two items that architecture §9.A marked "defer":

### 11.1 `learn_content.py` flat file (8,024 LOC) vs `learn_content/` directory (~14,556 LOC)

**Question:** Both files exist at the top level of `agents_api/`. Which is canonical? Is one a legacy copy of the other?

**Decision criterion:** Grep for imports of each during the Phase 1A `learn/` extraction CR. Whichever is actually imported by `learn_views.py` and referenced at runtime is the canonical file.

**Grep commands:**
```bash
# Check what learn_views.py actually imports
grep -n "learn_content" services/svend/web/agents_api/learn_views.py

# Check all imports of the flat file
grep -rn "from agents_api.learn_content import" services/svend/web/
grep -rn "from agents_api import learn_content" services/svend/web/

# Check all imports of the directory
grep -rn "from agents_api.learn_content\." services/svend/web/
grep -rn "agents_api.learn_content\._" services/svend/web/
```

**Expected resolution:**
- If `learn_views.py` imports `from agents_api.learn_content import X` → likely the flat file is canonical (`learn_content.py`), and the directory has been partially split off. Delete directory (or merge in).
- If `learn_views.py` imports `from agents_api.learn_content._datasets import X` → directory is canonical; flat file is legacy. Delete flat file.
- If both are referenced: may be transitional state. Pick canonical, merge content, delete other.

**Source:** inventory §J.14, architecture §9.A J.14 ("defer — small grep at gap analysis time").

**When to resolve:** During Phase 1A `learn/` extraction CR. Not a planning blocker.

### 11.2 `pbs_engine.py` (4,070 LOC) vs `forgepbs` package coverage

**Question:** Is `pbs_engine.py` the same computation as `forgepbs` package, or is it SVEND-specific scheduling logic beyond what forgepbs covers?

**Decision criterion:** Read `forgepbs/__init__.py` and `forgepbs/engine.py` (or equivalent) to get the exported function surface. Cross-reference with `pbs_engine.py`'s public function names. If ~95% overlap → replace with forgepbs import and delete. If pbs_engine.py has significant unique logic → relocate to new `pbs/` app or keep as a domain helper.

**Grep commands:**
```bash
# What does pbs_engine.py export?
grep -n "^def \|^class " services/svend/web/agents_api/pbs_engine.py | head -100

# What does forgepbs export?
python3 -c "import forgepbs; print([x for x in dir(forgepbs) if not x.startswith('_')])"

# What imports pbs_engine.py?
grep -rn "from agents_api.pbs_engine" services/svend/web/
grep -rn "import pbs_engine" services/svend/web/
```

**Expected resolution:**
- If forgepbs covers the public surface → replace + delete. Assigns to Phase 0.5 forge wiring.
- If unique SVEND logic → new `pbs/` app or absorbs into `workbench/handlers/`. Assigns to Phase 1B.

**Source:** inventory §J.15, architecture §9.A J.15 ("defer — small grep + read pass").

**When to resolve:** During Phase 0 forge wiring, specifically after Phase 0.3 `forgestat` wiring (since PBS depends on stat primitives). Not a planning blocker for Phase 0.1-0.2.

### 11.3 `RackSession` / ForgeRack status

**Question:** Is `RackSession` live (Eric's ForgeRack work) or dead (demo-only per inventory §H.7)?

**Decision criterion:** Confirm with Eric at Phase 0 start. Per memory `project_rack_canvas_checkpoint.md`: ForgeRack has 3 open CRs, active canvas editor work. Likely NOT deletable. Likely moves to own `forgerack/` app in Phase 1A.

**When to resolve:** Before Phase 0.6 (dead-code deletion). Ask Eric.

---

## Section 12 — What this gap analysis is NOT (and flagged concerns)

### 12.1 Scope boundaries

This document is **not**:
- **The sequenced extraction plan.** That's `extraction_sequence.md`, which depends on this gap as an input. It will convert §2/§3/§6/§7/§8/§9 into a strict CR-by-CR ordering with dependencies, gates, and check-in cadence.
- **The test suite rebuild plan.** That's `test_suite_rebuild.md`, which depends on this gap to identify per-extraction behavior test requirements (architecture §6.2-§6.3 sketches the pattern: ~5 tests per extraction).
- **The Phase 0 forge wiring sub-plan.** Section 6 here is a sketch, not a runbook. The runbook is `phase_0_forge_wiring.md` per memory phase plan.
- **A code change list.** No code is proposed. Every row says "move X to Y"; the CR execution determines HOW.
- **A new architecture decision.** Architecture v0.4 is locked. Where inventory and architecture diverge, the architecture wins. Where this document surfaces a conflict, it is flagged below as "needs Eric attention" — not resolved by this document.
- **Final.** This is a draft for Eric's review. Any row can change with Eric's input.

### 12.2 Items flagged "needs Eric attention" — 7 of 9 resolved 2026-04-09

After review, items 4-9 plus item 2 were resolvable without Eric's input (rationale documented per item below). **Only items 1 and 3 remain for Eric's call.** Both involve active work or load-bearing infrastructure where Eric's domain knowledge is required.

1. ~~**`RackSession` / ForgeRack disposition at Phase 0.**~~ **RESOLVED 2026-04-10 (Eric decision): OUT OF SCOPE.** ForgeRack is independent of agents_api. RackSession, rack_views.py (848 LOC, 27 funcs), and all rack-related code stay in place throughout this extraction. ForgeRack's eventual home is a separate initiative when the rack designer work matures. No Phase assignment, no CR in this extraction.

2. ~~**`agents_api/cache.py` and related infrastructure helpers.**~~ **RESOLVED 2026-04-09 (process default).** ~20 helper files (cache, embeddings, gpu_manager, llm_manager, llm_service, bayes_core, bayes_doe, causal_discovery, interventional_shap, clustering, conformal, drift_detection, anytime_valid, ml_pipeline, msa_bayes, report_types, iso_document_types, front_page_tasks, harada_tasks, commitment_tasks, commitment_notifications, quality_economics) get resolved **per-CR at extraction time**, not now. Each helper moves with the view that imports it; if multiple views import it, it lifts to the most natural shared home (chat/, core/utils/, or stays as a residual in agents_api/). The default is "evaluate at the CR that touches the helper, not in this gap analysis." If Eric wants any specific helper called out here, flag in conversation — otherwise default applies. Reasoning: 20 separate decisions piled into a planning doc is exactly the kind of premature standardization `feedback_build_then_standardize.md` warns against.

3. ~~**`tool_router.py`, `tool_registry.py`, `tool_event_handlers.py`, `tool_events.py`, `base_tool_model.py`.**~~ **RESOLVED 2026-04-10 (Eric decision): new `tools/` Django app, Phase 0, designed as a formal modular tool router system.** Eric's directive: "formalize a tool router system — something we can expand and modularize." This is NOT a residual shim or a simple relocation. Phase 0 CR creates `tools/` as a first-class Django app with an expandable, modular architecture for tool registration, routing, and event handling. All downstream apps (loop/, graph/, workbench/, etc.) import from `tools/` instead of `agents_api/`. This moves early (Phase 0) because everything depends on it — moving it late would mean rewriting imports twice.

4. ~~**`notebook_views.py` target.**~~ **RESOLVED 2026-04-09: `core/`.** Notebook model already lives in `core/models/notebook.py` per the architecture §3.1. Django convention is to keep views with their models. The 4 `DSWResult` references convert to workbench pull-API calls in Phase 1B (concurrent with the workbench rebuild). The relocation of `notebook_views.py` from `agents_api/` to `core/views/notebook.py` happens in Phase 2A as a small clean CR with no model migration (model is already in core). No genuine ambiguity here — defaulting to the obvious answer.

5. ~~**`investigation_views.py` target.**~~ **RESOLVED 2026-04-09: `qms_investigation/`.** Architecture §3.3 explicitly scopes `qms_investigation/` to handle RCA + Ishikawa + CEMatrix + investigations. The fact that the underlying `Investigation`, `InvestigationMembership`, `InvestigationToolLink` models live in `core/` is fine — Django views routinely import models from other apps. `investigation_views.py` and `investigation_bridge.py` both move to `qms_investigation/`. The model FKs into `core.Investigation` are a normal cross-app dependency, not a problem.

6. ~~**`autopilot_views.py` placement in Phase 1B with `workbench/handlers/`.**~~ **RESOLVED 2026-04-09: confirmed placement.** Uses `dsw/common` helpers and `SavedModel`. After Phase 0 the live `dsw/common` files move into `analysis/`, and `SavedModel` moves to `workbench/`. So both dependencies of autopilot_views land in workbench-adjacent homes. Co-locating autopilot in `workbench/handlers/` is the natural Phase 1B placement. Confirmed.

7. ~~**`experimenter_views.py` forgedoe wiring during extraction vs Phase 0.**~~ **RESOLVED 2026-04-09: Phase 1B inline.** `forgedoe` is still actively developing per `project_forgedoe_gaps.md` (Latin Square, D/I-Optimal, Taguchi migration, Mixture, Split-Plot, EVOP all open). Wiring incomplete forge package in Phase 0 means re-wiring later as forgedoe ships. Better: Phase 1B inline wiring against the current forgedoe state, with subsequent CRs adding capability as forgedoe gaps close. Defaulting to Phase 1B per the agent's recommendation.

8. ~~**Notebook `DSWResult` refs convert when?**~~ **RESOLVED 2026-04-09: Phase 1B.** They MUST convert in Phase 1B because that's when `DSWResult` itself is being converted to `workbench.Artifact`. Phase 2A is too late — by then `DSWResult` doesn't exist as a queryable model and notebook would break if its references hadn't already been switched to the pull API. The conversion is part of the workbench rebuild CR's blast radius in Phase 1B, not a follow-up.

9. ~~**`iso_views.py` split risk classification.**~~ **RESOLVED 2026-04-09: enhancement change_type, MED risk.** The agent's split-the-difference proposal is correct. Within-app reorganization with no model changes is technically an `enhancement` per CHG-001. But 4,874 LOC across 85 functions split into 7 target sub-modules is genuine review burden — that's MED risk despite being within-app. Risk tier is about review complexity and rollback exposure, not about whether models move. Documented as `enhancement` + MED.

**All 9 items resolved.** Items 1 and 3 resolved by Eric on 2026-04-10. Items 2, 4-9 resolved by Claude on 2026-04-09. Phase 0 forge wiring sub-plan is now unblocked — the `tools/` app creation is a Phase 0 CR, and ForgeRack is out of scope entirely.

---

## Appendix A — Model count reconciliation

| Source | Count | Notes |
|---|---|---|
| `models.py` `models.Model` subclasses | 67 | `grep -c '^class .*\(models.Model\)'` per inventory §A.3 |
| `models.py` `SynaraImmutableLog` subclasses | 1 | ElectronicSignature |
| **Total models in `agents_api/models.py`** | **68** | Matches inventory §A.3 |
| Models accounted for in §2 | 68 | All 68 have an entry |
| Relocated (survived) | 58-60 | Depending on uncertain classifications |
| Deleted (no replacement) | 4-5 | Workflow, ActionItem, AuditChecklist, (AgentLog?), (RackSession?) |
| Deleted and replaced | 3-5 | Report, CAPAReport, CAPAStatusChange, workbench.KnowledgeGraph (not agents_api but in scope) |
| Converted at storage boundary | 1 | DSWResult → workbench.Artifact |

## Appendix B — Phase/CR summary

| Phase | Models | View files | CRs | Risk profile |
|---|---|---|---|---|
| **Phase 0** — Forge wiring + dead-code cleanup + iso_views split + permissions move | 4-5 deleted | ~15 touched | ~8 | LOW-MED |
| **Phase 1A** — Leaf relocations | 13 relocated | 7 files | 7 | LOW-MED |
| **Phase 1B** — Leaf rebuilds + workbench pull contract + DSWResult conversion | — | 7 files (workbench-centric) | 7-10 | MED (DSWResult conversion is HIGH) |
| **Phase 2A** — Medium-coupling relocations | 33 relocated | ~15 files | 8-9 | HIGH (loop/graph coordination) |
| **Phase 2B** — Medium-coupling rebuilds + HoshinKPI conversion + reports/sop sinks | — | Hoshin + reports + sop | 8-12 | HIGH |
| **Phase 3** — Cutover night: URL swap, legacy code delete | — | every demo path | 1 (big) | HIGH |
| **Phase 4** — Site + qms_core final move | 9 (Site family) | ~5 files | 1-2 | **CRITICAL** |
| **TOTAL** | **~58 relocated + 9 deleted + 1 converted = 68** | ~33 view files | **~40-50 CRs** | |

---

**End of gap analysis. Next documents in the planning bundle:**
1. `phase_0_forge_wiring.md` — the Phase 0 runbook sketched in §6
2. `extraction_sequence.md` — the CR-by-CR execution order derived from §2/§3/§7/§8
3. `test_suite_rebuild.md` — the TST-001 behavior test plan per extraction
