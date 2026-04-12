# SVEND Target Architecture — QMS, Workbench, and the Source/Transition/Sink Model

**Status:** DRAFT — planning artifact under CR `5bf7354c-3de5-4624-b505-a94a5b6ce0ea`
**Date:** 2026-04-09
**Author:** Claude (Systems Engineer role per Object 271)
**Companion docs:**
- `migration_plan.md` — Object 271 stack decisions and template inventory (already exists)
- `agents_api_inventory.md` — current-state inventory of agents_api (in progress, delegated)
- `extraction_gap_analysis.md` — current vs target mapping (pending)
- `extraction_sequence.md` — sequenced execution plan (pending)
- `test_suite_rebuild.md` — TST-001 compliant safety net (pending)

---

## 1. Why this document exists

SVEND's codebase has accreted into one Django app (`agents_api/`) that owns 68 models and most of the product's URL surface. It is the remnant of the original reasoning-LLM era of SVEND, with everything from FMEA to ISO document control bolted on as the product evolved. Eric — who put up hour-by-hour boards at Fort Dearborn under Charlie Protzman's TIPS framework — called this *"zero 5S"* on 2026-04-09. It is.

The Object 271 architecture (specifically the source/transition/sink model resolved on 2026-04-09) cannot land cleanly while sources, transitions, and sinks all live in the same swamp app. This document defines the **target end state**: what apps exist, what each owns, what role each module plays in the pull graph, and how integrations actually work between them. Every extraction CR that follows will be measured against this target.

This document is **the shape we're building toward**, not a description of what exists today. The `agents_api_inventory.md` is the description of today.

---

## 2. The source / transition / sink model — formalized

Object 271 organizes inter-module data flow as a directed pull graph. There is no push.

### 2.1 Module roles

**Source.** Produces persistent artifacts. Does not pull. Does not push. Exposes saved containers (sessions / workbenches / artifacts) with stable, addressable contents. Consumers initiate all integration. The source's job is to be addressable, manifested, and available — nothing more.

**Transition.** Pulls from sources and from other transitions. Adds its own analysis or structuring. Emits its own pullable container that downstream modules can pull from in turn. Transitions are how chains compose: `analysis → SPC signal → RCA → A3 → report` is a pull chain where every middle node is a transition.

**Sink.** Pulls from sources and transitions. Assembles a deliverable. Does not emit pullable containers — terminal node. Sinks are the formatted-document, standard-work-publication, training-package end of the pipeline.

### 2.2 Worked example

```
[Source]      Analysis Workbench         → produces a saved analysis session with addressable artifacts
                  ↓ (pulled by)
[Transition]  Root Cause Analysis        → pulls SPC signal chart + Cpk statistic, adds 5-whys / fishbone, emits an RCA session
                  ↓ (pulled by)
[Transition]  A3 Report                  → pulls problem statement, RCA conclusion, target condition, emits an A3 instance
                  ↓ (pulled by)
[Sink]        Formatted Customer Report  → pulls A3 + selected charts + signed-off conclusion, renders PDF
```

Each arrow is a pull initiated by the downstream module. The upstream module never knows it has consumers until they register a reference back.

### 2.3 The pull contract (every source and transition implements)

| Capability | Description |
|---|---|
| **Container browse** | `GET /api/<module>/containers/` — list user's saved containers (workbenches, RCA sessions, A3s). Filterable, sortable, multi-tenant scoped. |
| **Container detail + manifest** | `GET /api/<module>/containers/<id>/` — metadata + machine-readable artifact manifest (every pullable artifact with stable key, type, label, and brief preview). |
| **Single artifact fetch** | `GET /api/<module>/artifacts/<artifact_id>/` — full artifact content. Content type matches artifact type (ChartSpec JSON for charts, scalar for numbers, text for narrative, etc.). |
| **Sub-artifact fetch (addressed)** | `GET /api/<module>/artifacts/<artifact_id>/<dotted.key>/` — address into the artifact's structured content. E.g. `plots.main`, `statistics.cpk`, `narrative.verdict`. |
| **Reference registration** | `POST /api/<module>/artifacts/<artifact_id>/references/` — consumer modules MUST call this when they store a reference. Body: `{consumer_module, consumer_record_type, consumer_record_id, artifact_key}`. Returns the reference ID. |
| **Reference list** | `GET /api/<module>/containers/<id>/references/` — what consumers reference this container. Used by delete-friction UI. |
| **Container delete with friction** | `DELETE /api/<module>/containers/<id>/` — returns `409 Conflict` with reference list if any references exist. `?force=true` allows the delete; references get marked with `source_deleted_at` and consumers see tombstones. |

### 2.4 The reference registry

Every source and transition owns a `*Reference` table that records "what is pulling from me." Schema (fields common to all source-side reference tables):

- `id` (UUID, PK)
- `source_container_id` (FK to the source container, `on_delete=PROTECT`)
- `artifact_key` (string — e.g. `"plots.main"`, `"statistics.cpk"`, `"narrative.verdict"`)
- `consumer_module` (string — `"rca"`, `"a3"`, `"report"`, `"sop"`, etc.)
- `consumer_record_type` (string — Django model name in consumer app)
- `consumer_record_id` (UUID — consumer's row PK)
- `created_at` (timestamp)
- `created_by` (FK User)
- `tenant_id` (FK Tenant — multi-tenancy enforcement)

**`on_delete=PROTECT`** is what creates the friction. Delete views catch `IntegrityError`, query the references, and return 409 with a list. If the user confirms `?force=true`, the delete proceeds via a transaction that first soft-marks references (`source_deleted_at = now()`) and then deletes the container. Consumers seeing soft-marked references render tombstones.

**Tombstone widget.** Per Eric's request: an SVEND-styled icon in the shared widget library (`svend-widgets.css`) — small inline SVG, matching the rest of the icon vocabulary, used wherever a consumer renders a pulled artifact whose source was deleted. Same widget across RCA, A3, reports, SOPs.

### 2.5 What this model is NOT

- **Not a message bus.** No async events. Pulls are synchronous HTTP at integration time.
- **Not a copy-on-pull cache.** Pulled artifacts are live references; if the source container is updated (rare — sources should be append-only) consumers see the updated content. Snapshots are explicit (the consumer can choose to bake a copy if they want frozen evidence).
- **Not a permission system.** Permissions are enforced at the source's pull endpoint per existing tenant/user rules. No new permission model.
- **Not a forge package.** This lives in Django views and models, not in standalone Python packages. Forge packages handle computation; the pull contract handles storage and addressability.

---

## 3. App topology — target end state

Django apps in the target architecture, grouped by concern. **This is the target.** The current state is documented in `agents_api_inventory.md` and the gap is in `extraction_gap_analysis.md`.

### 3.0 Total count and summary table

**Target end state: 25 Django apps across 6 categories.** Per Eric 2026-04-09: keep small and distributed, no consolidation.

| Category | Count | Apps |
|---|---|---|
| §3.1 Foundation (existing) | **6** | `core`, `accounts`, `chat`, `forge`, `files`, `api` |
| §3.2 Sources (1 existing + 2 new) | **3** | `workbench`, `triage` *(new)*, `whiteboard` *(new)* |
| §3.3 Transitions (4 new) | **4** | `qms_investigation`, `qms_a3`, `qms_capa`, `qms_audit` |
| §3.4 Sinks (2 new) | **2** | `reports`, `sop` |
| §3.5 QMS-adjacent (6 new) | **6** | `qms_core`, `qms_risk`, `qms_documents`, `qms_training`, `qms_suppliers`, `qms_measurement`, `qms_nonconformance` |
| §3.6 Operations & standalone (4 new) | **4** | `vsm`, `hoshin`, `simulation`, `learn` |
| **TOTAL** | **25** | |

`agents_api/` itself is fully decomposed in the end state — see §3.7.

The inventory I.3 list of deletions removes models like `Workflow`, `ActionItem`, `RackSession`, the original agent dispatchers, etc., so there is no `lean_tools/` catch-all app in the target — those models don't survive. CAPA and Report are likely delete+replace per inventory I.3.

### 3.1 Foundation apps (already exist, refined for the target)

| App | Owns | Notes |
|---|---|---|
| **`core/`** | Tenant, Membership, Project, Hypothesis, Evidence, EvidenceLink, Dataset, ExperimentDesign, KnowledgeGraph (canonical) | Already in good shape per CLAUDE.md. The KnowledgeGraph here is the canonical product knowledge graph (GRAPH-001). DO NOT confuse with `workbench.KnowledgeGraph` (different domain — investigation-scoped). |
| **`accounts/`** | User (custom), Subscription, InviteCode, Membership-related billing | Already in good shape. |
| **`chat/`** | LLM conversation system, message threads | Already exists. Used by AI guide and analyst features. |
| **`forge/`** | Synthetic data generation jobs, forge service integration | Already exists. The standalone `forgestat`/`forgespc`/etc. Python packages are NOT this app — they're pip-installed into the project's site-packages. This `forge/` Django app is the in-product UI for synthetic data generation. |
| **`files/`** | File storage, sharing, quotas | Already exists. |
| **`api/`** | Internal staff dashboard, content/automation, public auth/feedback views | Already exists. Cross-cutting glue, not a domain. |

### 3.2 Source apps

Sources produce artifacts that other modules pull from. They implement the full pull contract from §2.3.

| App | Owns | Source role | Notes |
|---|---|---|---|
| **`workbench/`** | `Workbench`, `Artifact`, `ArtifactReference` (new), `EpistemicLog`, scoped `KnowledgeGraph` (investigation-internal, NOT the canonical one) | **Source** — analysis workbench saved sessions and their artifacts. Replaces `agents_api.DSWResult` and friends as the storage layer for analysis runs. | Already exists with `Workbench` and `Artifact` models that fit the domain. Needs `ArtifactReference` model + pull API endpoints + delete-friction handling. The new `templates/demo/analysis_workbench.html` will write to this app on save. |
| **`triage/`** *(new)* | `TriageDataset`, `TriageRun`, `TriageProfile`, dataset cleaning state | **Source** — uploaded datasets and their cleaning lineage. RCA, FMEA, etc. can pull triaged datasets as evidence sources. | Currently in `agents_api/triage_views.py` and `agents_api.TriageResult`. Extract to its own app — datasets are foundational infrastructure that many downstream modules consume. |
| **`whiteboard/`** *(new)* | `Board`, board collaborators, votes, board exports | **Source** — collaborative whiteboard sessions. Causal claims drawn on a board can be pulled as evidence by transitions. | Currently in `agents_api.Board` + `agents_api/whiteboard_views.py`. Extract. |

### 3.3 Transition apps

Transitions pull from sources (and other transitions), add their own structuring, and emit pullable containers themselves.

| App | Owns | Transition role | Notes |
|---|---|---|---|
| **`qms_investigation/`** *(new — naming TBD with Eric)* | `Investigation`, `RCASession`, `RootCause`, `FiveWhysNode`, `Fishbone*` (if modeled), investigation evidence links | **Transition** — root cause investigation. Pulls SPC signals, charts, statistics, datasets from sources. Emits an investigation container with conclusions, supporting evidence, and timeline. Downstream A3 and CAPA pull from these. | Currently `agents_api.RCASession` + `agents_api/rca_views.py`. Centerpiece transition. |
| **`qms_a3/`** *(new)* | `A3Report`, A3 sections (problem statement, current condition, target, root cause, countermeasures, follow-up), A3 references | **Transition** — A3 problem-solving form. Pulls problem statement from a workbench analysis or an investigation. Pulls root cause from an investigation. Pulls evidence artifacts from analysis sources. Emits an A3 container that reports and SOPs pull from. | Currently `agents_api.A3Report` + `agents_api/a3_views.py`. Per Eric 2026-04-09: A3 will be **largely rebuilt** as part of Object 271 — new architecture, updated models, front-end polish. The extraction is an opportunity to restructure, not just relocate. |
| **`qms_capa/`** *(new)* | `CAPAReport`, corrective actions, preventive actions, CAPA references | **Transition** — CAPA workflow. Pulls findings from investigations, audits, nonconformance records. Emits a CAPA container with action plans. | Currently `agents_api.CAPAReport` + `agents_api/capa_views.py`. Migration plan flags this as deprecated → replaced by Investigation + ForgeDoc CAPA generator. The extraction may be a **delete + replace** rather than a relocate. To be decided in gap analysis. |
| **`qms_audit/`** *(new)* | `InternalAudit`, audit findings, audit schedules, auditor independence rules | **Transition** — internal audit. Pulls evidence from controlled documents, training records, risk registry. Emits audit container with findings. Findings can be pulled by investigations or CAPA. | Per `project_audit_upgrade.md` memory: audit has its own upgrade plan (FMEA-based risk, checklist execution, clause coverage, auditor independence). The extraction must align with that plan. |

### 3.4 Sink apps

Sinks pull from sources and transitions to assemble deliverables. They do not emit pullable containers themselves.

| App | Owns | Sink role | Notes |
|---|---|---|---|
| **`reports/`** *(new — formatted reports)* | `Report`, report templates, report sections, generation jobs | **Sink** — formatted customer-facing reports. Pulls from A3, investigations, analysis sources, audit findings. Uses ForgeDoc for PDF generation. | Per Eric 2026-04-09: report assemblers are sinks being built **brand new** as part of Object 271. Currently `agents_api.Report` exists but is generic; the new sinks will be purpose-built. The pro-tier-only constraint: reports are paid feature. |
| **`sop/`** *(new — standard work editors)* | `SOPDocument`, SOP sections, SOP versions, SOP signoffs | **Sink** — standard work / standard operating procedure editor. Pulls best-practice content from investigations, A3s, training records. The "standardize" step in LOOP-001's Investigate → Standardize → Verify. | New app per Object 271. Eric: "standard work editors we'll make as part of object 271." |

### 3.5 QMS-adjacent apps that don't fit neatly into source/transition/sink

Some QMS modules are infrastructural rather than pull-graph nodes. They have data, but the pull pattern doesn't apply (or applies asymmetrically).

| App | Owns | Notes |
|---|---|---|
| **`qms_core/`** *(new)* | `Site`, `SiteAccess`, `Employee`, `ActionToken`, `QMSFieldChange`, `QMSAttachment`, `Checklist`, `ChecklistExecution`, `ElectronicSignature`, `permissions.py` helpers (`qms_can_edit`, `qms_queryset`, `get_tenant`, `is_site_admin`) | Cross-cutting QMS infrastructure. Site is the chokepoint (24 incoming FKs) — see §3.6.1. Every QMS extraction depends on `qms_core/` for permissions and shared resources. Inventory §I.4 confirmed this is what's logically left after agents_api decomposes. |
| **`qms_risk/`** *(new)* | `Risk`, `FMEA`, `FMEARow` (and any future hazard analysis models) | Inventory §I.1: 3 models, view = `fmea_views.py` (1,589 LOC, 27 funcs) + parts of `iso_views.py` `risk_*`. Source role for individual risks pulled into investigations; dual-purpose registry. FMEARow has `hypothesis_link` FK to `core.Hypothesis`. `loop.FMISRow` has cross-app FK pointing in. Per `feedback_afe_policy.md`: AFEs only via Hoshin, not from FMEA/risk/audits. |
| **`qms_documents/`** *(new)* | `ControlledDocument`, `DocumentRevision`, `DocumentStatusChange`, `ISODocument`, `ISOSection`, `ControlPlan`, `ControlPlanItem` | Inventory §I.1: 7 models, views split out of `iso_views.py` `document_*` + `iso_doc_views.py` (716 LOC). Source role. **Heavy `loop/` and `graph/` coupling** — extraction is HIGH risk because of `graph.ProcessNode` M2M. ForgeDoc integration per migration plan. |
| **`qms_training/`** *(new)* | `TrainingRequirement`, `TrainingRecord`, `TrainingRecordChange` | Inventory §I.1: 3 models, views from `iso_views.py` `training_*`. Source role for audit + supplier qualifications. Per TRN-001 standard: TWI 4-level competency model. Per `project_harada.md`: Harada method is related but separate (lives in `learn/`). loop/ coupling required. |
| **`qms_suppliers/`** *(new)* | `SupplierRecord`, `SupplierStatusChange` | Inventory §I.1: 2 models, views from `iso_views.py` `supplier_*`. Source role for supplier qualifications. loop/ has 2 incoming FKs — coordinate extraction. Note: `loop/supplier_views.py` exists untracked per earlier finding — that's a separate WIP file unrelated to this extraction. |
| **`qms_measurement/`** *(new)* | `MeasurementEquipment` | Inventory §I.1: 1 model. Views from `iso_views.py` `equipment_*` + MSA dispatch in `spc_views.py`. Source role for calibration evidence in audits. Coordinates with `qms_audit/` and `workbench/` (since MSA studies are workbench analyses). `MeasurementEquipment.linked_process_node` is a direct FK to `graph.ProcessNode`. |
| **`qms_nonconformance/`** *(new — added from inventory)* | `NonconformanceRecord`, `NCRStatusChange`, `CustomerComplaint` | Inventory §I.1: 3 models, views from `iso_views.py` `ncr_*` + `complaint_*`. Transition+source dual role. **HIGH risk** — both `loop/` and `graph/` couple in. Was conflated into `qms_risk/` in v0.2 of this doc; inventory split it out as its own concern because NCR has its own workflow distinct from risk register and FMEA. |

### 3.6 Operations & standalone apps

These are not strictly QMS but live in agents_api today and need proper homes.

| App | Owns | Notes |
|---|---|---|
| **`vsm/`** *(new)* | `ValueStreamMap` (and any future VSM step/inventory/kaizen sub-models) | Inventory §I.1: 1 model currently, views in `vsm_views.py` (433 LOC, 14 funcs). Per migration plan: VSM workbench is a separate Object 271 deliverable with cockpit UX (calculator panel integrated below the map). VSM is **sink-like** for analysis (pull cycle-time data into it) and **source-like** for kaizen bursts (pulled into Hoshin). |
| **`hoshin/`** *(new)* | `HoshinProject`, `ProjectTemplate`, `StrategicObjective`, `AnnualObjective`, `HoshinKPI`, `XMatrixCorrelation`, `AFE`, `AFEApprovalLevel`, `ResourceCommitment` | Inventory §I.1: 9 models, views = `hoshin_views.py` (2,094 LOC, 36 funcs) + `xmatrix_views.py` (1,069 LOC, 13 funcs). Operations strategy infrastructure. Per `feedback_afe_policy.md`: AFEs only through Hoshin. Enterprise-tier only per CLAUDE.md. **HoshinKPI couples directly to DSWResult** — see §4.5 for the worked pull-contract example. |
| **`simulation/`** *(new)* | `PlantSimulation` | In scope for this extraction work per Eric 2026-04-09. (He had been upgrading simulators in another session and paused that work to focus on this planning round.) Inventory §I.1: 1 model, `plantsim_views.py` is 391 LOC / 8 funcs. Sink + scenario explorer. Small extraction. |
| **`learn/`** *(new — added from inventory)* | `SectionProgress`, `AssessmentAttempt`, `LearnSession` | Inventory §I.1: 3 models, views = `learn_views.py` (2,450 LOC, 32 funcs) plus optionally `harada_views.py` (909 LOC). Sub-content: `learn_content/` directory (~14,556 LOC) and possibly `learn_content.py` flat file (8,024 LOC) — disambiguation needed (inventory §J.14). Standalone product surface — courses, assessments, certifications. Not in pull graph. |

### 3.6.1 The `Site` chokepoint — addressed explicitly

**Refined by full inventory 2026-04-09 (`agents_api_inventory.md` §B.3, §K.1):** the `Site` model has **24 incoming ForeignKeys** total — 19 internal to `agents_api` (FMEA, A3, RCA, Hoshin, NonconformanceRecord, AnnualObjective, HoshinProject, Employee, SiteAccess, etc.) plus **5 cross-app** (`loop/models.py` ×2, `safety/models.py` ×3, `graph/models.py` ×1). Highest-coupled model in the entire codebase.

Implications:

1. **Site cannot be extracted piecemeal.** Moving `Site` to a new app while leaving its 24 dependents in `agents_api/` (or worse, distributed across 8+ different new apps) requires every dependent's ForeignKey to update in lockstep. This is the highest-risk single migration in the extraction plan.

2. **Site's target home is genuinely ambiguous.** Options:
   - **(a)** Keep `Site` in `agents_api/` indefinitely as shared infrastructure. agents_api becomes a thin "shared QMS resources" app. Acceptable but dilutes the extraction win.
   - **(b)** New `qms_core/` app for cross-cutting QMS infrastructure (Site, possibly Department, Process, etc.). All QMS apps depend on it. Cleaner but requires extracting Site coordinated with everything that touches it.
   - **(c)** Extract `Site` LAST, after every dependent has moved. By the time Site is the last thing standing, all the FK updates have already happened in the dependents' extractions. Then Site moves to its final home (probably `qms_core/` or `core/`) in one atomic migration with no dependents to break.
   - **(d)** Move `Site` to existing `core/` app since it's organizationally analogous to `Tenant`. But `core/` is currently for product knowledge (Project, Hypothesis, Evidence) — adding org structure may dilute its concern.

   **Recommendation:** option **(c)** — Site stays in agents_api until last, then atomic-migrates to a new `qms_core/` app. This is the safest sequencing. **OPEN QUESTION — see §9.13.**

3. **No extraction in Phase 1 should touch Site's FK relationships** unless it's explicitly a Site coordination CR.

### 3.7 The agent dispatch remnant

After extraction, what's left in `agents_api/` is the original LLM agent dispatch layer (researcher, coder, writer, editor) and any genuinely-still-agent code. Likely outcomes:

1. **Most of the original "agents" are dead.** Inventory will confirm. Dead code gets deleted.
2. **The genuinely-active ones get renamed.** `agents_api/` → `llm_agents/` or similar. Honest naming for what it actually is.
3. **OR** the active agents get absorbed into `chat/` (if they're conversational) or a new `automation/` app.
4. **The `agents_api/analysis/` sub-package** is the special case. Forge migration is replacing its handlers with `forgestat`/`forgespc`/`forgeviz`/etc. — what's left is dispatch, which logically belongs to `workbench/` since that's the source app for analysis. Extract `agents_api/analysis/dispatch.py` and the live forge handlers to a new home, probably under `workbench/handlers/` or a new `analysis/` app.

The exact disposition of `agents_api/` post-extraction is determined by the inventory, not by this document.

---

## 4. Integration patterns — how chains actually work

### 4.1 Worked example: SPC signal → RCA → A3 → customer report

1. **User runs SPC capability study in workbench.** New analysis workbench template at `/app/analysis/` (post-Object-271 cutover) calls `POST /api/workbench/analysis/run/`. Server dispatches to `forgespc.capability_cpk(...)`, gets back the 10-key result, creates a `workbench.Workbench` (or adds to an existing one) with a `workbench.Artifact` of type `CAPABILITY_STUDY` whose `content_json` field holds the full 10-key result. Returns the artifact ID.

2. **User saves the session.** Save button POSTs to `POST /api/workbench/workbenches/<id>/save/` with title and description. Container is now persistent and listed in the user's workbench list.

3. **Days later, user opens RCA from a different surface.** Maybe SPC alarms triggered an investigation, maybe a customer complaint did. They land in the RCA app at `/app/rca/sessions/<new_id>/`.

4. **In the RCA evidence picker, user pulls from the workbench.** RCA app calls `GET /api/workbench/workbenches/?user=current` to list available workbenches. User selects the capability study workbench, RCA calls `GET /api/workbench/workbenches/<id>/` to get the artifact manifest. Picker UI shows the available artifacts: control chart, capability histogram, Cpk = 1.13, narrative verdict, etc. User selects the control chart and the Cpk number.

5. **RCA registers the references.** `POST /api/workbench/artifacts/<chart_artifact_id>/references/` with body `{consumer_module: "rca", consumer_record_type: "RCASession", consumer_record_id: "<new_rca_id>", artifact_key: "plots.main"}`. Same for the Cpk stat with `artifact_key: "statistics.cpk"`. Two `workbench.ArtifactReference` rows now exist, owned by the workbench artifact, pointing at the RCA session.

6. **RCA renders the pulled artifacts in its UI.** RCA's evidence pane fetches the live data via `GET /api/workbench/artifacts/<id>/plots.main` and `GET /api/workbench/artifacts/<id>/statistics.cpk` whenever it needs to display them. The chart is a live ChartSpec rendered by ForgeViz; the number is just a number with a label.

7. **User completes the RCA**, adds 5-whys analysis, identifies a root cause, marks the RCASession as `completed`. RCASession is now itself a pullable container (`qms_investigation` is a transition).

8. **User opens A3 builder.** A3 has its own container model. In the "Cause" section, the user pulls the RCA conclusion via `GET /api/qms_investigation/sessions/<id>/` → manifest → `GET /api/qms_investigation/artifacts/<root_cause_id>/`. A3 also pulls the original SPC chart by going through the RCA's manifest, which includes the references RCA registered upstream — the A3 can transitively pull from the original source. This is how chain composition works.

9. **A3 registers references both with RCA and with workbench.** Direct references for what A3 pulled from each. Both `qms_investigation.RCAReference` and `workbench.ArtifactReference` now have rows pointing at this A3.

10. **User opens Customer Report sink.** Selects an A3 to base the report on. Report pulls from A3 manifest, possibly reaches transitively to RCA and workbench artifacts. Uses ForgeDoc to render PDF. Registers references.

11. **User later wants to delete the original workbench session.** `DELETE /api/workbench/workbenches/<id>/`. Backend queries `ArtifactReference` for the workbench's artifacts, finds 4 references (RCA's 2, A3's 1, report's 1). Returns `409 Conflict` with the list. Frontend shows GitHub-style warning: "*Deleting this workbench will orphan 4 references in 3 modules.*" User can cancel or confirm with `?force=true`. On force, the delete transaction marks all 4 references with `source_deleted_at` and removes the workbench. Next time RCA or A3 or the report renders, they show tombstone widgets in place of the orphaned artifacts.

### 4.2 Why this works

- **No module knows about its consumers in advance.** Workbench has no idea RCA exists; RCA finds workbench through the URL conventions and the user's selection.
- **The chain composes by reference**, not by copy. Updates to a source flow through (rare since sources should be append-only), and deletes are visible through tombstones.
- **Cross-tenant isolation** is enforced at every endpoint by tenant filter on the queryset. Multi-tenancy is row-level enforcement, not app-level.
- **Each link is testable in isolation.** RCA's pull from workbench can be tested without an A3. A3's pull from RCA can be tested with a stub workbench. The pull contract is the test boundary.

### 4.3 Failure modes the model handles

- **Source deleted while consumer exists.** Tombstone pattern. User informed at delete time, downstream UI shows tombstones gracefully.
- **Source updated while consumer references it.** Live reference reflects the update. If frozen evidence is needed, consumer can opt to snapshot at pull time (snapshot is a separate consumer-side concern).
- **Permission denied at fetch time.** Source endpoint enforces tenant/user permission. If user A's RCA references user B's workbench artifact (cross-tenant), the fetch returns 403. Consumer renders an "access denied" placeholder, not a tombstone.
- **Consumer record deleted.** Reference row should be cleaned up by the consumer's delete handler. The pull contract documents this as a consumer obligation. We may add a periodic compliance check that finds orphaned references whose consumer record no longer exists.

### 4.5 Worked example: HoshinKPI rollups (canonical pull-contract case)

Inventory finding J.10: `HoshinKPI.effective_actual` currently queries `DSWResult.objects.filter(...)` directly at `agents_api/models.py:3285`. After extraction, `DSWResult` no longer exists — saved analyses live as `workbench.Artifact` rows. Hoshin must use the pull contract to fetch the latest KPI value. This is the **canonical worked example** of why the pull contract exists.

**Before extraction:**
```python
class HoshinKPI(models.Model):
    @property
    def effective_actual(self):
        # Direct ORM query into agents_api.DSWResult — tightly coupled
        result = DSWResult.objects.filter(
            user=self.owner,
            analysis_type=self.linked_analysis_type,
        ).order_by('-created_at').first()
        return result.statistics.get(self.linked_stat_key) if result else None
```

**After extraction (Hoshin lives in `hoshin/`, workbench source is in `workbench/`):**
```python
class HoshinKPI(models.Model):
    linked_artifact = models.ForeignKey(
        'workbench.Artifact',
        on_delete=models.SET_NULL,  # tombstone if source deleted
        null=True,
    )
    linked_artifact_key = models.CharField(max_length=200)  # e.g. "statistics.cpk"

    def effective_actual(self):
        if not self.linked_artifact:
            return None  # tombstone
        # Use the pull API — registers a reference on first access
        return workbench_pull_api.fetch_artifact_value(
            artifact_id=self.linked_artifact_id,
            key=self.linked_artifact_key,
            consumer_module='hoshin',
            consumer_record_type='HoshinKPI',
            consumer_record_id=self.id,
        )
```

**On first call,** the pull API auto-registers a `workbench.ArtifactReference` row pointing back at this `HoshinKPI`. Now the workbench knows hoshin is consuming this artifact. If the user deletes the workbench session, the friction warning lists "1 reference in hoshin (HoshinKPI for Q1 line 3 Cpk target)." User confirms, the reference is soft-marked, and the next `effective_actual()` call returns None — Hoshin's KPI dashboard renders a tombstone for this metric.

**Why this is the canonical example:**
- It's the strongest existing cross-app coupling that the extraction must convert from "direct ORM query" to "pull contract."
- The behavior round-trip cost is acceptable (KPIs are computed on demand, not in tight loops).
- It demonstrates auto-registration on first access — the consumer doesn't have to remember to call the registration endpoint separately.
- It shows the tombstone pattern from the consumer side (the dashboard rendering, not just the source-side warning).

This pattern repeats for every other `direct ORM query into agents_api` that the inventory will surface. The gap analysis will list them all.

### 4.4 Failure modes the model does NOT handle (out of scope)

- **Real-time push updates.** If the source updates and the consumer is currently displaying it, the consumer doesn't auto-refresh. Refresh on next render.
- **Conflict resolution on simultaneous edits.** Sources should be append-only for this reason. If a source needs to be edited (rename a workbench, change a description), only metadata is mutable, never artifact content.
- **Cross-database references.** All tables are in the same Postgres database. If we ever shard, this model needs to evolve.

---

## 5. The 10-key contract — break-up for addressability

The current handler contract (per `project_analysis_workbench_migration.md`) produces 10 output keys:

```
plots, statistics, summary, narrative, diagnostics, guide_observation,
assumptions, education, bayesian_shadow, evidence_grade
```

The handler contract **does not change**. Handlers still produce all 10 keys. The change is at the **storage and addressing layer**: when a result is saved as a `workbench.Artifact`, its `content_json` field stores the full 10-key blob, but the pull API exposes individual addressable paths into it.

### 5.1 Stable artifact keys

| Key path | Type | What it returns |
|---|---|---|
| `plots.<chart_id>` | ChartSpec dict | A single ForgeViz ChartSpec ready to render |
| `plots` | list of ChartSpec dicts | All charts as a list (rare — usually consumers want one) |
| `statistics.<stat_name>` | scalar (number/bool/string) | One named statistic value |
| `statistics` | dict | Full statistics dict |
| `summary` | text with `<<COLOR:>>` markup | Rich summary text |
| `narrative.verdict` | string | One-line conclusion |
| `narrative.body` | string | 2-3 sentence detail |
| `narrative.next_steps` | string | Action directive |
| `narrative.chart_guidance` | string | How to read the chart |
| `narrative` | dict | All four narrative fields as one dict |
| `diagnostics` | list | Full diagnostics list |
| `diagnostics.<index>` | dict | One diagnostic by position |
| `assumptions.<name>` | dict | One assumption check `{pass, p, test, detail}` |
| `assumptions` | dict | Full assumptions dict |
| `bayesian_shadow` | dict | Bayesian shadow if present, else null |
| `education` | dict | Education content if present, else null |
| `evidence_grade` | string | Grade (A/B/C/D/F or whatever standardize.py produces) |
| `guide_observation` | string | One-liner for guide UI |

### 5.2 Stable chart IDs (NEW handler contract requirement)

The current handler contract says `plots: list[ChartSpec]`. The new contract requires `plots: list[{"id": str, "spec": ChartSpec}]` so that pulls can address individual charts by name. Chart IDs:

- `"main"` — the primary chart (mandatory if plots is non-empty)
- `"residuals"` — residual plot (regression-style analyses)
- `"diagnostic_<n>"` — additional diagnostic charts numbered
- Handler can use any stable string ID; convention is lowercase snake_case

This is a small breaking change to the handler contract. Migration: every existing handler must be updated to emit IDs. This is part of the test-suite-rebuild work, not a separate effort.

### 5.3 What the manifest looks like

When a consumer calls `GET /api/workbench/workbenches/<id>/`, the response includes an artifact manifest:

```json
{
  "id": "wb-abc-123",
  "title": "Q1 line 3 capability study",
  "user": "eric.wolters@svend.ai",
  "tenant": null,
  "status": "active",
  "created_at": "2026-04-12T14:32:00Z",
  "updated_at": "2026-04-12T14:35:00Z",
  "artifacts": [
    {
      "id": "art-xyz-456",
      "type": "capability_study",
      "title": "Cpk study, line 3, characteristic A",
      "created_at": "2026-04-12T14:32:00Z",
      "manifest": {
        "plots.main": { "type": "chart", "label": "Capability histogram with spec limits" },
        "plots.diagnostic_1": { "type": "chart", "label": "Normal probability plot" },
        "statistics.cpk": { "type": "number", "label": "Cpk", "value": 1.13 },
        "statistics.cpk_lower": { "type": "number", "label": "Cpk lower 95%", "value": 1.05 },
        "statistics.cpk_upper": { "type": "number", "label": "Cpk upper 95%", "value": 1.21 },
        "statistics.ppm_defective": { "type": "number", "label": "Estimated PPM defective", "value": 870 },
        "narrative.verdict": { "type": "text_short", "label": "Conclusion" },
        "narrative.body": { "type": "text_long", "label": "Detail" },
        "narrative.next_steps": { "type": "text_short", "label": "Next steps" },
        "narrative.chart_guidance": { "type": "text_long", "label": "Chart reading guide" },
        "summary": { "type": "text_rich", "label": "Summary" },
        "assumptions.normality": { "type": "assumption", "label": "Normality (Anderson-Darling)", "pass": true },
        "education": { "type": "education_content", "label": "About process capability" }
      }
    }
  ]
}
```

Consumers iterate the manifest and present a picker UI. Users select what to pull. Each pull is a `POST /references/` followed by a `GET <key>` for the actual content.

### 5.4 What standardize.py's role becomes

`standardize.py` currently post-processes handler output before returning to the workbench template. In the target architecture, it ALSO becomes responsible for:

1. Asserting that every chart in `plots` has a stable `id` field
2. Emitting the artifact manifest from the result blob (probably as a method `result.to_manifest()`)
3. Validating that the result is "save-ready" — all addressable keys present, no None values where they shouldn't be
4. Computing `evidence_grade` (already does this)

The handler→standardize→storage→manifest pipeline becomes the target architecture's data path for sources.

---

## 6. Test suite philosophy for the extraction work

Per `feedback_build_then_standardize.md`: build the working thing first, write the standard from stable state. For tests, this translates to: **the tests are the safety net for the build, not the standard for the build.** They exist to keep the extraction from regressing, not to define the architecture.

### 6.1 TST-001 compliance

Per the standards library, TST-001 §10.6 prohibits existence/sweep tests — tests must exercise real behavior. Existing agents_api tests are likely TST-001 violations (the migration plan acknowledges weak coverage). Extraction tests must pass TST-001:

- ✅ "User can pull a chart from a saved workbench session" (behavior)
- ❌ "`workbench.Workbench` model exists" (existence)
- ❌ "All views in `workbench/views.py` return 200" (sweep)

### 6.2 The extraction test pattern

For each model or view being extracted, the work order is:

1. **Define the behavior contract.** What does a user / consumer module observe before and after the extraction? List 3-7 concrete behaviors. Examples for extracting `RCASession`:
   - "User can create a new RCA session and it persists in the database"
   - "RCA session list returns only the current user's sessions"
   - "Pulling a finding from RCA registers a reference back to the source"
   - "Deleting an RCA session with active references returns 409"
2. **Write the tests for that behavior contract.** Real tests using Django's test client or DRF test client. Touch real endpoints, real DB.
3. **Run the tests against the current code.** They should pass — the behavior already works in agents_api. If they don't pass, the behavior contract is wrong; fix the test, not the code.
4. **Do the extraction in a CR.** Move the model, the view, the URL, update imports. Run the tests. They should still pass.
5. **If the tests fail, the extraction broke something.** Don't fix forward — revert the CR, revise the test (or the extraction) and retry.

This is the safety net pattern. The tests are not comprehensive coverage; they're regression detection at the behavior level for the specific things that must keep working through the extraction.

### 6.3 Coverage targets

The extraction is **NOT** an opportunity to write a complete new test suite. That's a separate, larger effort that should not be coupled to the extraction work. What we need from this round:

- **Behavior tests for every extraction step.** ~5 tests per extraction. Maybe 100-200 tests across the whole extraction sequence.
- **Integration tests for cross-app pull chains.** A handful — enough to verify that workbench → RCA → A3 actually works end-to-end at least once.
- **Multi-tenancy regression tests.** One per source app — verify that user A cannot pull from user B's containers.

The `test_suite_rebuild.md` companion doc will detail this further.

### 6.4 What we explicitly do NOT do

- We do NOT write tests for the existing behavior of models we're going to delete (e.g. CAPAReport per migration plan deprecation). Just verify the deprecation is real and delete.
- We do NOT pause the extraction to write thorough unit tests for forge package internals. Those packages have their own test suites in their own repos.
- We do NOT add tests for compliance check logic (`syn/audit/compliance.py` and friends) as part of this work. Compliance checks have their own coverage story.

---

## 7. What gets deleted

Per the migration plan and the inventory (still pending), the following are candidates for deletion rather than relocation. Final list will be in `extraction_gap_analysis.md`.

### 7.1 Confirmed deletions (per migration_plan.md)

- `workbench_new.html` (11,790 lines — confirmed duplicate of `analysis_workbench.html`)
- `safety_coming_soon.html` (1,993 lines — replaced)
- `iso_9001_qms.html` (1,159 lines — replaced by QMS workbench)
- `core/models/graph.py` (~200 lines — DEPRECATED, replaced by `graph/` app per GRAPH-001)
- `workbench/models.py KnowledgeGraph` (~100 lines — third KG implementation, replaced by `graph/` app)

Wait — there's a tension here. The migration plan says `workbench/models.py KnowledgeGraph` is going away. But §3.2 of this document just listed `workbench.KnowledgeGraph` as scoped (investigation-internal). I need to reconcile this with Eric: is the workbench.KnowledgeGraph being deleted per the migration plan, OR being kept as a scoped non-canonical KG? **OPEN QUESTION — see §10.**

### 7.2 Likely deletions (pending inventory confirmation)

- The original "agents" dispatch code (researcher, coder, writer, editor) IF the inventory shows nothing references it
- `agents_api/synara/*.py` (~3,250 LOC) once `forgesia` is fully wired (per migration plan tech debt section + inventory finding 2026-04-09)
- `agents_api/dsw/*.py` files duplicated by `forgestat`
- `agents_api/spc.py` (1,889 LOC) duplicated by `forgespc`
- `agents_api/dsw/chart_render.py` and `chart_defaults.py` (518 lines) duplicated by ForgeViz
- `agents_api/analysis/chart_render.py` and `chart_defaults.py` (518 lines, **duplicate of dsw/**) — delete unconditionally per migration plan

### 7.3 The dsw/ vs analysis/ relationship — REFINED (inventory finding 2026-04-09 §K.2)

The full inventory **partially refuted** the initial "100% duplicate" finding. The relationship is more complex and the prior framing was wrong:

**Sizes (corrected):**
- `agents_api/dsw/` — **73,749 LOC** (not 56,679)
- `agents_api/analysis/` — **78,544 LOC** (not 17,552)

**Direction of dispatch is `dsw → analysis`, NOT the other way around:**
- `dsw/dispatch.py` is a **17-line shim** that re-exports from `analysis/dispatch.py`. So `analysis/` is the canonical runtime dispatcher.
- `analysis/chart_render.py` and `analysis/chart_defaults.py` are **7-line and 11-line wrappers around `dsw/chart_*`** — opposite direction. Charts flow `analysis → dsw`.
- The two trees are **interlinked**, not independent duplicates.

**What's actually live in dsw/** (~17,000 LOC, NOT deletable):
- `dsw/common.py` (3,084 LOC) — imported by multiple top-level views
- `dsw/endpoints_data.py` (1,832 LOC) — active endpoint helpers
- `dsw/endpoints_ml.py` (1,702 LOC) — active ML endpoints
- `dsw/standardize.py` (552 LOC) — the result-standardization pipeline (referenced extensively in `project_analysis_workbench_migration.md` memory as the post-processor that owns `evidence_grade`, bounds, auto-shadow, etc.)
- `dsw/chart_render.py` and `dsw/chart_defaults.py` — the actual chart implementations that `analysis/chart_*` wrap

**What's actually deletable in dsw/** (~50,000-55,000 LOC, deletable AFTER forge wiring):
- `dsw/stats_*.py` files (parametric, nonparametric, regression, anova, exploratory, msa, quality)
- `dsw/bayesian.py`
- `dsw/ml.py`
- `dsw/viz.py`
- `dsw/siop.py`, `dsw/simulation.py`, `dsw/reliability.py`, `dsw/d_type.py`
- `dsw/spc.py`, `dsw/spc_pkg/*`, `dsw/exploratory/*`
- These are the legacy duplicates of `analysis/forge_*.py` counterparts. Deletable once forge package wiring is complete (§7.4).

**Execution implication:**
1. There is **no single "delete the duplicate" CR.** The cleanup is finer-grained.
2. Phase 0 task: **move** `dsw/common.py`, `dsw/endpoints_data.py`, `dsw/endpoints_ml.py`, `dsw/standardize.py`, `dsw/chart_render.py`, `dsw/chart_defaults.py` **into `analysis/`** (or wherever the dispatch home becomes). Update imports across the views. The two trees become one.
3. After forge wiring (§7.4) replaces the legacy `dsw/stats_*.py` etc. with calls to `forgestat`, the now-empty `dsw/` legacy files are deleted in a separate cleanup CR.
4. The `analysis/chart_render.py` and `analysis/chart_defaults.py` wrappers are deleted in Phase 0 as confirmed dead-on-arrival now that `forgeviz` is the chart layer.

**See `agents_api_inventory.md` §K.2** for the file-by-file evidence.

### 7.4 Forge wiring is Phase 0, NOT post-extraction

Inventory finding: forge replacement is 95%+ ready for `spc.py` → `forgespc`, `synara/` → `forgesia`, `dsw/`+`analysis/` → `forgestat`. The packages exist. The wiring (replacing inline imports with forge package calls in the view modules) is the missing step.

**This changes the extraction sequence.** Phase 0 — *before any model extraction* — is:

1. Delete the not-imported half of dsw/ vs analysis/ (cleanup)
2. Wire `forgespc` into `spc_views.py`, then delete `agents_api/spc.py`
3. Wire `forgestat` into `dsw_views.py` and any remaining dsw/ code paths, reducing the kept directory to a thin dispatcher
4. Wire `forgesia` into `synara_views.py` (after forgesia `__init__.py` exports are fixed — known blocker), then delete `agents_api/synara/`
5. Wire `forgeviz` into anywhere still using `chart_render.py` / `chart_defaults.py`, then delete both copies

After Phase 0, `agents_api/` is significantly smaller, the forge package boundaries are clean, and the actual model/view extractions can proceed against a much smaller surface. **Phase 0 is itself multiple CRs** — one per forge package wiring, each with its own behavior tests as a safety net.

This is a meaningful plan change from §11 of the original draft. Updating the next-steps list accordingly.

### 7.5 The `loop/` ↔ `agents_api/` coupling — extraction blocker

Inventory finding (refined 2026-04-09 §K.4): `loop/` app has **17 distinct import sites** across `agents_api` (not 10). Models touched: `Employee`, `TrainingRecord`, `TrainingRequirement`, `ControlledDocument`, `ISOSection`, `ISODocument`, `Site`, `FMEARow`, `FMEA`, `SupplierRecord`, `NonconformanceRecord`, `CustomerComplaint`, `MeasurementEquipment`, plus `tool_events` and `permissions.get_tenant`. Loop is the QMS operating surface (signals, commitments, investigations) per OLR-001 and `project_loop_ui.md` memory. Its commitment workflow runs in production.

Implications:

1. **Every QMS extraction must coordinate with `loop/` imports.** Moving `TrainingRecord` to `qms_training/` requires `loop/` to update its imports in the same CR. There is no "extract first, fix imports later" option — the imports must be atomic with the extraction.
2. **The CR for any model loop/ depends on becomes a 2-app change.** Higher review burden, more files, more risk. But unavoidable.
3. **Tests must include `loop/` integration coverage** for the affected models. The behavior contract for "extracting TrainingRecord" must include "loop/ commitment workflow still attaches to training records" as a verified behavior.
4. **Sequencing implication**: extractions of loop-coupled models should land mid-sequence, not first or last. First-extractions (lowest risk leaves) should be models loop/ does NOT import, to build extraction muscle without the coordination overhead. Last-extractions (highest-risk hubs like Site) need full loop/ coordination already in place.

### 7.6 The `iso_views.py` monolith

Inventory finding: `agents_api/iso_views.py` is approximately 4,875 LOC of view code with deep coupling to `ControlledDocument`, `ISODocument`, ForgeDoc PDF generation, and management review workflows. It's the largest single view file in agents_api and one of the highest-risk extractions.

Implications:

1. **`iso_views.py` should NOT be extracted as a single CR.** It needs to be split first into logical sub-modules (document-list views, document-detail views, management-review views, audit-link views), each of which extracts separately to its target home (`qms_documents/`, `qms_audit/`).
2. **The split itself is a CR** — likely an `enhancement` change_type, low-risk because it's just file reorganization within the same app, no model changes. But it's a prerequisite for the extraction CRs that follow.
3. **Coverage gap is acute here.** A 4,875-line view file with thin tests is a regression risk regardless of whether we touch it. Even if we don't extract it, the test suite rebuild for ISO documents should be done early.

### 7.3 Models flagged as deprecated (relocate vs delete TBD)

- `agents_api.CAPAReport` — migration plan says replaced by Investigation + ForgeDoc CAPA generator. May be **delete + replace** rather than relocate.
- `agents_api.ActionItem` — migration plan says superseded by Commitment per LOOP-001 §3. May be **delete + replace**.

---

## 8. Forge package boundaries

Forge packages (`forgestat`, `forgespc`, `forgeviz`, `forgedoe`, `forgesia`, `forgequeue`, `forgepad`, `forgepbs`, `forgesiop`, `forgedoc`, `forgegov`, `forgecal`) are standalone Python packages that live OUTSIDE the SVEND repo and are pip-installed into the project's site-packages. They contain:

- All statistical computation
- All SPC computation
- All chart rendering (replacing Plotly)
- All DOE computation
- All Bayesian/Synara computation
- All queueing theory
- ForgePad command palette
- All PBS/PB-style computation
- All Lean operations math (takt, OEE, kanban, etc.)
- All ForgeDoc PDF rendering
- All forge governance / cross-package health
- All calibration / golden file machinery

After the agents_api extraction, the SVEND Django apps own:

- Models (data persistence)
- Views (HTTP endpoints)
- Templates (HTML rendering)
- URL routing
- Forge package wiring (calling forge functions from views)
- Frontend JavaScript (sv-* utilities, ForgeViz client renders, ForgePad command UI)

The Django side calls into forge packages for computation. The forge packages do not call back into Django.

---

## 9. Decisions and open questions

Most v0.2 open questions are now resolved. Decisions resolved by Eric directly, by inventory evidence, or by my own call (with rationale) are listed first. Only **3 questions actually need Eric's input** before the gap analysis can finalize — see §9.B at the end.

### 9.A Decisions resolved

| # | Question | Decision | Source |
|---|---|---|---|
| 1 | Flat `qms_*` prefix vs umbrella `qms/` app | **Flat with prefix.** | Eric 2026-04-09: "keep small and distributed." |
| 3 | CAPA relocate vs delete + replace | **Delete + replace.** CAPAReport is deprecated per migration plan; new CAPA functionality comes from `qms_investigation/` + ForgeDoc CAPA generator. No transitional `qms_capa/` app needed. Inventory I.3 confirms. | Inventory + migration plan |
| 4 | ActionItem relocate vs delete | **Delete.** LOOP-001's `Commitment` supersedes. Inventory I.3 confirms ActionItem is on the deletion list. The 5 view files referencing it (action_views, fmea_views, rca_views, a3_views, hoshin_views) get their references replaced with Commitment. | LOOP-001 + inventory I.3 |
| 5 | Home for `agents_api/analysis/dispatch.py` | **`workbench/handlers/`** — analysis is the source domain, dispatch belongs with the source app. Inventory K.2 confirmed `analysis/dispatch.py` is the canonical runtime path; this just relocates it. | Inventory K.2 |
| 6 | Triage as its own app | **YES, own app `triage/`.** Inventory I.1: 1 model, 512 LOC view, S effort, LOW risk. Used by every analysis as source. Worth its own boundary. | Inventory I.1 |
| 7 | Whiteboard as own app or extension of workbench | **YES, own app `whiteboard/`.** Inventory I.1: 4 models (Board family), 1,057 LOC view, 1,181 LOC of tests. Self-contained, LOW risk. Different storage and UI patterns than workbench. | Inventory I.1 |
| 8 | `qms_investigation/` naming | **`qms_investigation/`** — broader than RCA, matches LOOP-001's canonical terminology, encompasses RCA + Ishikawa + CEMatrix per inventory I.1. | Self + inventory I.1 |
| 10 | Test scope minimal vs comprehensive | **Minimal safety-net.** §6.3 ~5 behavior tests per extraction. Per `feedback_build_then_standardize.md`: build first, write standards from stable state. Full TST-001 rebuild is a separate later effort. | feedback_build_then_standardize |
| 11 | Per-extraction CR vs batched | **Per-extraction CRs.** Rollback granularity is critical for high-risk work. Yes the multi-agent risk assessment overhead is real, but it's the right cost. | Self |
| 12 | Production deploy pattern | **Two-phase migrations.** `--state-operations` separated from `--database` operations where possible. Add new, dual-write, switch reads, drop old. No atomic-rename with read locks against production. | Self + production = this machine |
| 14 | `loop/` coupling acceptance | **Accepted as unavoidable.** Every QMS extraction is a 2-app coordinated CR (extraction + loop/ import update). Review burden per extraction is real but the alternative is leaving the QMS swamp longer. Inventory K.4 refined the count to **17 import sites** (not 10) — each gets explicit treatment in the extraction plan. | Self + inventory K.4 |
| 15 | `iso_views.py` split timing | **Split first as prerequisite CR**, then extract pieces. Inventory C.3: 4,874 LOC across 85 view functions touching 7 target apps. A whole-cloth extraction is too risky. The split CR is `enhancement` change_type (low risk, no model changes) and lands before any of the qms_documents/qms_audit/qms_training/etc. extractions. | Inventory C.3 |
| J.2 (new) | Where does `Employee` live | **`qms_core/`** alongside Site. Same pattern: high incoming FKs, infrastructural, naturally co-located with Site as part of QMS organizational core. | Self + inventory J.2 |
| J.10 (new) | HoshinKPI ↔ DSWResult coupling tolerable post-extraction? | **Yes — and this is the canonical worked example for the pull contract.** When DSWResult is replaced by `workbench.Artifact`, HoshinKPI.effective_actual stops querying DSWResult directly and instead pulls from a saved workbench session via `GET /api/workbench/artifacts/<id>/statistics.<key>`. Adds round-trip cost to KPI rollups (acceptable — KPIs are computed on demand, not in tight loops). Documented in §4.5 below. | Self + inventory J.10 |
| J.11 (new) | dsw/ vs analysis/ bidirectional dependency direction | **`analysis/` is canonical.** Move `dsw/common.py`, `dsw/endpoints_data.py`, `dsw/endpoints_ml.py`, `dsw/standardize.py`, `dsw/chart_render.py`, `dsw/chart_defaults.py` into `analysis/`. Delete `analysis/chart_render.py` and `analysis/chart_defaults.py` wrappers (now-DOA). Per inventory K.2 + J.11. | Inventory K.2, J.11 |
| J.13 (new) | `Report` model + `report_views.py` (802 LOC) disposition | **Delete + replace** as part of new `reports/` sink. Existing code stays in production until the new `reports/` app is ready, then deleted in the cutover CR. Same pattern as the workbench template cutover. | Self + inventory J.13 |
| J.14 (new) | `learn_content.py` flat (8,024 LOC) vs `learn_content/` directory (~14,556 LOC) | **Verify which is canonical via grep before extracting `learn/`.** This is a small grep job at extraction time, not a planning blocker. Will resolve in gap analysis. | Inventory J.14 — defer |
| J.15 (new) | `pbs_engine.py` (4,070 LOC) ownership | **Verify against `forgepbs` package coverage.** If `forgepbs` already implements the same logic, this is a delete + replace. If `pbs_engine.py` is SVEND-specific scheduling beyond what `forgepbs` covers, it gets its own `pbs/` app or moves to wherever PBS belongs. Resolution: small grep + read pass at gap-analysis time. | Inventory J.15 — defer |

### 9.B Three questions — RESOLVED 2026-04-09

All three answered by Eric. Gap analysis is now unblocked.

| # | Question | Resolution |
|---|---|---|
| **§9.B.1** | `workbench.KnowledgeGraph` disposition | **DELETE.** Force pulling into the canonical `graph/` KG. Eric's reasoning: KG is "purely abstract right now while we experiment with it — it isn't visible as a graph yet." There's only one KG; workbench investigations link to canonical entities, no scoped KG of their own. The migration plan's deletion list is correct. |
| **§9.B.2** | Site chokepoint disposition | **Option (a) WITH gating.** Site stays in agents_api until the **full parallel rebuild is complete, tested, and reviewable in a `/app/demo/...` path.** Site is the LAST move in the extraction sequence, and it only happens AFTER Eric reviews the entire parallel build. This is option (a) plus an explicit review gate before Site cuts over. |
| **§9.B.3** | A3 rebuild scope (and the pattern for all rebuilds) | **Extract first, rebuild second — universally.** This is the pattern for **every extraction in this work**, not just A3. Eric: *"Same pattern across the board: extract first then rebuild. We're building the full parallel system then seamlessly switching the urls over in one night."* This generalizes into the architecture-wide cutover principle — see §13. |

### 9.C The universal cutover pattern (lifted from §9.B.3 and §9.B.2 answers)

Eric's resolution to §9.B.3 elevates a pattern that applies to every extraction:

> **Build the full parallel system. Test it. Demo it under `/app/demo/...` paths. Eric reviews. Then switch URLs in one night.**

This is the **operating model for the entire agents_api extraction work**, not just one CR's pattern. It's now codified as §13 of this document and as `feedback_parallel_build_cutover.md` in long-term memory.

---

## 10. What this document is NOT

- **Not the inventory.** That's `agents_api_inventory.md`, written in parallel by the delegated explore agent.
- **Not the gap analysis.** That comes after both this and the inventory exist.
- **Not the execution plan.** That's `extraction_sequence.md`, derived from the gap.
- **Not a standard.** Per `feedback_build_then_standardize.md`, standards come after the working thing exists. This is a planning document. When the extraction is done and the architecture is stable, we'll cut the relevant pieces of this doc into actual standards (`SOURCE-001`, `PULL-001`, etc.) under `docs/standards/`.
- **Not final.** This is a draft for Eric's review. Every section is open to revision based on his input on §9 and any other concerns.

---

## 11. Next steps in this planning phase

1. ⏳ **Inventory completes** (re-delegated agent with Write access, in progress) → `agents_api_inventory.md`
2. ⏳ **Eric reviews this architecture doc** and answers the §9 open questions (15 questions, blocking subset is 1/2/5/6/7/8/13)
3. ⏳ **Reconcile architecture with inventory** — every model in the inventory gets a target home from the architecture; inventory may surface things the architecture didn't anticipate, requiring revisions here
4. ⏳ **Gap analysis** → `extraction_gap_analysis.md`
5. ⏳ **Phase 0 forge wiring plan** (per §7.4) — separate sub-plan that runs BEFORE any model extraction. Each forge package wiring is its own CR with behavior tests.
6. ⏳ **Sequenced extraction plan** → `extraction_sequence.md` (depends on Phase 0 plan, Site disposition, and loop/ coordination acceptance)
7. ⏳ **Test suite rebuild plan** → `test_suite_rebuild.md`
8. ⏳ **Eric final sign-off** on the bundle
9. ⏳ **Plan CR `5bf7354c-3de5-4624-b505-a94a5b6ce0ea` marked completed**
10. ⏳ **Phase 0 execution CRs opened** — forge wiring + duplicate-cleanup, as children of this plan
11. ⏳ **Phase 1 extraction execution CRs opened** after Phase 0 completes — lowest-risk leaf models first, Site last

## 13. The universal cutover pattern (Eric 2026-04-09)

Lifted from the §9.B.3 resolution. **This applies to every extraction, every rebuild, every URL move in this work.** Not just A3, not just the analysis workbench template — universally.

### 13.1 The pattern

Every extraction is at minimum a **2-step sequence** (extract-relocate, then rebuild), and lives at parallel `/app/demo/...` paths the entire time until cutover:

```
Step 1 — EXTRACT (smaller CR, lower risk)
   Move model + view + URL to new app, AS-IS.
   New URL lives at /app/demo/<thing>/ in parallel with /app/<thing>/.
   Old code keeps running unchanged. Old URL still works.
   Tests verify behavior is identical at both URLs.

Step 2 — REBUILD (separate CR, possibly multiple CRs)
   Against the new architecture: pull contract, sv-* widgets,
   ForgeViz charts, shared JS, source/transition/sink role.
   Still at /app/demo/<thing>/. Still parallel.
   Tests now verify the new behavior at the demo URL.
   Old URL still serves the old code unchanged.

Eric review of the demo build.
   Walk-through, real-data testing, sign-off.

Step 3 — CUTOVER (single short CR, one night)
   svend/urls.py: swap routes — /app/<thing>/ now serves the new code.
   Old templates and views deleted in the same commit.
   Nav links updated.
   Old route either deleted entirely or 301-redirects to new.
```

### 13.2 What this means for the extraction sequence

- **Chokepoints (Site, Employee, etc.) move LAST.** Their dependents extract first under demo paths. Site stays in agents_api throughout the dependents' parallel-build period. Only after every dependent is rebuilt and reviewed does Site move, and it moves into a structure where every dependent already knows how to FK to the new location.

- **No incremental cutover.** No "switch user A to the new system while user B stays on old." No feature flags for partial rollout. The cutover is atomic at URL routing.

- **No production URL ever points at a broken parallel build.** If the demo path isn't working, fix it under the demo path. The production URL keeps serving the old, working code throughout.

- **Tests verify both routes during the parallel period.** A behavior contract for "user can list workbenches" runs against both `/app/workbench/` (legacy) and `/app/demo/workbench/` (new) and asserts the responses are equivalent. After cutover, only the new route remains and the test continues to assert the same behavior.

### 13.3 What this means for the test plan

- **Behavior tests must be route-agnostic** during the parallel period. Use a fixture or parameter that makes a single test exercise either route.
- **Cutover CRs include a small "delete the legacy test paths" step** so the test suite stays clean after cutover. Don't keep dead test routes around.

### 13.4 What this means for the extraction sequence numbering

Phase 0 (forge wiring + dead code cleanup) is unchanged.

Phase 1 was "extract leaf models first." In light of §13.1, **Phase 1 has two sub-phases**:

- **Phase 1A — Relocate.** Each leaf extraction does the model/view/URL move under a demo path. No rebuild yet. The full agents_api topology stays intact, just with parallel routes.
- **Phase 1B — Rebuild.** Once a leaf is relocated, the rebuild CR replaces its model/view with the new pull-contract-aware version. Still under the demo path.

After Phase 1A and 1B for all leaves: an Eric review pass. Demo testing. If approved, Phase 2 cuts over the leaves in one night.

Phase 2 was "extract medium-coupling models." Same sub-pattern: 2A relocate, 2B rebuild, then a coordinated cutover.

Phase 3 was "extract Site." With the §9.B.2 resolution, Phase 3 is the FINAL cutover after all dependents are running in parallel and reviewed. Site moves AFTER Eric signs off on the rebuilt parallel system.

The `extraction_sequence.md` document will detail this with specific CR-level granularity.

---

## 12. Revision history of this document

- **2026-04-09 v0.1** — Initial draft: §1-9 with 12 open questions, before inventory data was available.
- **2026-04-09 v0.2** — Folded in initial (partially wrong) inventory findings: added §3.6.1 Site chokepoint, expanded §7 with subsections 7.3 (dsw/analysis duplication actionable detail), 7.4 (forge wiring as Phase 0), 7.5 (loop/ coupling), 7.6 (iso_views.py monolith). Added §9.13/14/15 open questions. Updated §11 next steps to reflect Phase 0 / Phase 1 split.
- **2026-04-09 v0.4** — Eric resolved §9.B.1, §9.B.2, §9.B.3:
   - **§9.B.1**: workbench.KnowledgeGraph → DELETE. Single canonical KG in `graph/` app.
   - **§9.B.2**: Site stays until full parallel rebuild is reviewable in `/app/demo/...`. Last move in the sequence.
   - **§9.B.3**: extract-first-then-rebuild pattern is universal across all extractions, not just A3.
   - **§9.C** added — codifies §9.B answers.
   - **§13 added** — the universal cutover pattern: build full parallel under demo paths, review, switch URLs in one night. Phase 1/2/3 sub-divided into A (relocate) and B (rebuild) sub-phases. Saved as `feedback_parallel_build_cutover.md` in long-term memory.
   - All §9 questions are now resolved. Architecture doc is **locked** pending Eric's final sign-off after gap analysis lands.
- **2026-04-09 v0.3** — Reconciled with full inventory document (`agents_api_inventory.md`, 978 lines). Major changes:
   - Added §3.0 total count summary table — 25 apps across 6 categories
   - Added `qms_core/` app to §3.5 (Site/Employee/Checklist/permissions home)
   - Added `qms_nonconformance/` app to §3.5 (was conflated into qms_risk in v0.2)
   - Added `learn/` app to §3.6 (Sections/Assessments — standalone product surface)
   - Removed `lean_tools/` from §3.6 (Workflow + ActionItem are deletes per inventory I.3, not relocations)
   - Removed "Eric won't touch" framing from `simulation/` per Eric 2026-04-09: in scope
   - Refined Site chokepoint: 24 incoming FKs (15+ → 24), 19 internal + 5 cross-app
   - Refined loop/ coupling: 17 import sites (10 → 17)
   - **Major rewrite of §7.3 — dsw/ vs analysis/ is NOT 100% duplicate.** Bidirectional dependencies. ~17,000 LOC of dsw/ is live, ~50,000-55,000 LOC is legacy duplicate. Execution plan revised.
   - Added §4.5 worked example: HoshinKPI ↔ DSWResult coupling as the canonical pull contract example
   - **§9 fully restructured.** Was 15 open questions (all unresolved). Now: 17 questions resolved (with rationale), 3 questions remaining for Eric (workbench.KnowledgeGraph disposition, Site disposition, A3 rebuild scope).
- **next** — After Eric answers the 3 remaining §9.B questions, the doc is locked and gap analysis begins.
