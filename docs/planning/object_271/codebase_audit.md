# Object 271 — Codebase Audit

**Agent:** #2 (Conservative / Risk-Averse)
**Date:** 2026-03-28
**Scope:** Walk the codebase against GRAPH-001 and the three-thing model (Graph / Loop / QMS). Identify what changes, what stays, what's new, and what can break.

---

## 0. POSTURE

This audit assumes the worst. Every integration point is a potential regression. Every "should be easy" is suspect. Every shared data structure is a migration risk. The other session can be innovative — this one asks: "what breaks, what costs, and what do we not know yet?"

---

## 1. INVENTORY: WHAT EXISTS TODAY

### 1.1 Three KnowledgeGraph Implementations

| # | Location | Storage | API | Production Use | Verdict |
|---|----------|---------|-----|----------------|---------|
| 1 | `core/models/graph.py` | Django models (`core_knowledge_graph`, `core_entity`, `core_relationship`) | 5 endpoints at `/api/core/graph/` | **Unknown.** Endpoints exist, serializers exist, tests verify constraints only. No evidence of real CRUD traffic. | Scaffolding — probably unused |
| 2 | `workbench/models.py` | Django model with JSONFields for nodes/edges/signals | 18 endpoints at `/api/workbench/<id>/graph/` | **Likely active.** Full CRUD, Bayesian update via `core.bayesian`, EpistemicLog at every operation, functional test file exists. | Active — the one users touch |
| 3 | `agents_api/synara/kernel.py` | In-memory dataclasses, serialized to `Project.synara_state` or `Investigation.synara_state` JSONField | 23 endpoints at `/api/synara/` | **Active.** Investigation bridge, learn system, tool linking. In-memory cache of 128 instances. | Active — the investigation engine |

**Risk #1: GRAPH-001 says "deprecate both existing KnowledgeGraph models" (§13.2). But we don't know if anyone is actually using the core/models/graph.py endpoints.** Before deleting, we need to check production logs or add instrumentation to confirm zero traffic. If a customer is using the core graph API (even experimentally), deprecation breaks them.

**Risk #2: The workbench KnowledgeGraph stores everything in JSONFields.** Migrating that data to the new ProcessGraph/ProcessNode/ProcessEdge Django models requires parsing every `knowledge_graphs` row's `nodes` and `edges` JSON, mapping old types to new types, and preserving the EpistemicLog references. This is a non-trivial data migration with no rollback path if the JSON structures are inconsistent across rows.

**Recommendation:** Instrument both APIs with access logging (1 week) before committing to deprecation. Add a `GRAPH_001_MIGRATION_DRY_RUN` management command that parses all existing KG data without writing, and reports incompatibilities.

---

### 1.2 Two Synara Engines

| # | Location | Purpose | Persistence | Used By |
|---|----------|---------|-------------|---------|
| 1 | `core/synara.py` | Consistency checking on `core.KnowledgeGraph`. LR-based Bayesian updates on `core.Hypothesis`/`Evidence`. | Stateless (operates on Django models directly) | `core/views.py` → `/api/core/graph/check-consistency/` |
| 2 | `agents_api/synara/` (kernel, belief, synara, dsl, logic_engine) | Full belief engine. CausalGraph, HypothesisRegion, Evidence, BeliefEngine, ExpansionSignal. | Serializes to JSONField on Project/Investigation/Notebook | `synara_views.py`, `investigation_bridge.py`, `learn_views.py`, tests |

**GRAPH-001 says "one engine, one truth" (§12.1) and "extend agents_api/synara" (§12.3).** But core/synara.py is NOT dead code — it's called by `check_consistency()` which uses `core.models.graph.Relationship` traversal. If we deprecate `core/models/graph`, `core/synara.py` loses its data source.

**Risk #3: The two engines have different math.** `core/synara.py` uses likelihood ratios (LR × prior_odds → posterior_odds). `agents_api/synara/belief.py` uses direct Bayes (P(E|H) × P(H) / normalizer). These are equivalent in theory but produce different numerical results due to different damping/clamping. If we consolidate, we need to decide which math wins and verify that existing posteriors don't shift when recomputed.

**Risk #4: `agents_api/synara/belief.py` has a known convergence issue in `propagate_belief()`.** Multiple upstream influences are combined via unweighted averaging: `sum(link.strength × upstream.posterior) / (count + 1)`. This is not Bayesian — it's a heuristic. At investigation scale (3-5 hypotheses) it works fine. At graph scale (50-100 nodes), untested. The propagation also has **no cycle detection** — it relies on the DAG assumption, which is never validated on `add_link()`. If a cycle enters the persistent graph, `propagate_belief()` recurses infinitely.

**Recommendation:** Before extending Synara for graph-scale operation:
1. Add cycle detection to `CausalGraph.add_link()` — reject if adding the link creates a cycle.
2. Add a `visited` set to `propagate_belief()` as a safety net.
3. Benchmark propagation at 100, 500, 1000 nodes. Establish performance envelope.
4. Document the averaging heuristic as a known limitation with an issue tracker reference. Either fix it (proper Bayesian network inference) or accept the approximation with documented bounds.

---

### 1.3 FMIS → Graph Bridge (Half-Built)

`FMISRow` has three FK fields to `core.Entity`:
- `failure_mode_entity` → `core.Entity` (nullable)
- `effect_entity` → `core.Entity` (nullable)
- `cause_entity` → `core.Entity` (nullable)

**These are NEVER populated.** Zero writes anywhere in the codebase. The only read is a null-check for gap counting in `loop/views.py`. The comment says "knowledge gap surfaced in §9.4" — this was planned infrastructure for LOOP-001 §9 that never shipped.

**GRAPH-001 says FMIS rows seed the graph (§7).** But the FKs point to `core.Entity` (the old generic model), not to the new `ProcessNode` that GRAPH-001 defines. If we build ProcessNode as a new model, these FKs need to be retargeted.

**Risk #5: Retargeting FKs on a model with existing production data requires a migration that touches every FMISRow.** Since the FKs are currently all null, this is technically safe — but the migration must be tested against the production database size. FMISRow may have hundreds of rows.

**Recommendation:** Drop the three entity FK fields from FMISRow in a pre-GRAPH-001 cleanup migration. They're dead weight. When GRAPH-001 ships, add new FKs pointing to `ProcessNode` instead of `core.Entity`. Clean cut, no retargeting complexity.

---

## 2. SYNARA EXTENSION RISKS

GRAPH-001 §12.3 specifies four extensions to Synara. Each carries risk.

### 2.1 Extension #1: Recency-Weighted Posterior Aggregation

**Current state:** Synara does sequential Bayesian update. Each new evidence becomes the prior for the next. No temporal weighting. Evidence order matters but age doesn't.

**What GRAPH-001 wants:** `recompute_posterior(evidence_stack, decay_function)` — recalculate from full stack with exponential decay (half-life 180 days).

**Risk #6: Recomputation from full stack is O(N) per edge per evidence add.** If an edge has 50 evidence records over 2 years, every new evidence triggers a full recalculation. At graph scale with 200 edges, each getting evidence periodically, this could become expensive. Need to benchmark.

**Risk #7: Changing from sequential update to full-stack recomputation will CHANGE existing posteriors.** Any edge with old evidence will see its posterior shift because old evidence now weighs less. This is correct behavior, but it means the transition is not invisible — posteriors will change on first recomputation. Users may notice.

**Risk #8: The decay function is org-configurable (QMS Policy).** This means different orgs can have different half-lives. The service must parameterize the decay, not hardcode it. Testing matrix expands: test with 30-day, 180-day, 365-day, and infinite (no decay) half-lives.

**Recommendation:**
- Implement as a NEW method, not a modification of `update_posteriors()`. Keep the old method for investigation-scoped work (short-lived, no decay needed). New method for persistent graph only.
- Add a `posterior_version` field to edges. When the recomputation method changes, increment the version. This allows auditing "when did this edge's posterior methodology change?"
- Benchmark with 10, 50, 200 evidence records per edge before shipping.

### 2.2 Extension #2: Interaction-Modulated Propagation

**Current state:** `propagate_belief()` uses `link.strength` as a constant.

**What GRAPH-001 wants:** `link.strength` modulated by sibling node states via interaction terms. Auto-fit modulation function from DOE data (AIC/BIC selection).

**Risk #9: Auto-fitting modulation functions requires DOE data in a specific format.** The current DOE endpoint (`experimenter_views.py`) returns results as JSON to the frontend. It doesn't write structured results to any model that the graph service can query. The pipeline DOE → graph evidence → interaction fit doesn't exist. Building it requires:
1. DOE results persisted as `EdgeEvidence` records (new)
2. Evidence records tagged with the specific nodes/edges they calibrate (new)
3. A fitting service that reads evidence records and fits candidate functions (new)
4. A model selection step (AIC/BIC) that stores the winning model (new)
5. A UI for the user to confirm the fit (new)

This is not one extension — it's an entire subsystem. Defer until after the core graph service is stable.

**Risk #10: The interaction term schema stores `fit_result` as JSON.** If the fitting service changes its output format, old `fit_result` blobs become unreadable. Need a `fit_version` field to handle schema evolution.

**Recommendation:** Ship interaction terms as "assert only" in v1. `modulation_type: "unknown"`, `calibrated: false` for all interactions. The auto-fit pipeline is a Phase 2 feature after the graph service, evidence stacking, and DOE→evidence pipeline are all working.

### 2.3 Extension #3: Contradiction Detection (Edge-Scoped Expansion)

**Current state:** `check_expansion()` fires when ALL likelihoods are below 0.1 — whole-graph signal.

**What GRAPH-001 wants:** Same math, scoped to individual edges. `P(new_evidence | current_edge_posterior) < threshold` → ContradictionSignal.

**Risk #11: The contradiction threshold is org-configurable.** Too low → alert fatigue (every minor disagreement is a "contradiction"). Too high → real contradictions go undetected. The default of 0.1 (matching expansion threshold) hasn't been validated for edge-scoped use. Expansion detection is rare (all hypotheses surprised); edge contradiction may be common (one edge disagrees).

**Risk #12: Contradiction detection on `add_evidence()` means every evidence add potentially creates a Signal.** In a busy org adding SPC data daily, this could generate many Signals if the threshold is wrong. Signals have a lifecycle (triage, investigation) — flooding them defeats the purpose.

**Recommendation:**
- Start with the threshold at 0.05 (stricter than expansion), not 0.1.
- Add a `contradiction_cooldown` per edge — don't raise another ContradictionSignal on the same edge within 7 days of the last one.
- Log contradictions that DON'T fire (below threshold but notable) to a separate audit table for tuning.
- Make the threshold, cooldown, and minimum evidence count all configurable via QMS Policy.

### 2.4 Extension #4: Persistent State Interface

**Current state:** Synara serializes to/from dict. `to_dict()` / `from_dict()`.

**What GRAPH-001 wants:** Adapter that loads from Django models (ProcessGraph/ProcessNode/ProcessEdge) and writes back.

**Risk #13: Synara's `from_dict()` mutates dicts in-place during deserialization (datetime parsing).** This is already known — `investigation_bridge.py` uses `copy.deepcopy()` as a workaround. The persistent state adapter must handle this correctly or risk corrupting Django model state.

**Risk #14: The adapter creates a new abstraction boundary.** Synara operates on `CausalGraph` (in-memory dataclasses). The graph service operates on `ProcessGraph/ProcessNode/ProcessEdge` (Django models). The adapter translates between them. Every field in GRAPH-001's schema (§3.3, §4.3) must be mapped to a Synara equivalent, and vice versa. If the schemas drift (someone adds a field to ProcessEdge but not to CausalLink), the adapter silently drops data.

**Recommendation:**
- Build the adapter as a testable, standalone module with explicit field mapping.
- Add a round-trip test: Django models → CausalGraph → operations → Django models. Assert no data loss.
- Add a schema compatibility check that runs in CI: verify every ProcessEdge field has a CausalLink mapping and vice versa.
- Fix the `from_dict()` mutation bug properly rather than relying on deepcopy.

---

## 3. LOOP INTEGRATION RISKS

### 3.1 Investigation Scoping — Does Not Exist

GRAPH-001 §8 requires investigations to operate on a scoped subgraph. Today:
- Investigation starts with `synara_state = {}` (empty dict)
- No concept of "selected nodes" — investigator builds the graph from scratch within the investigation
- No extraction from a parent graph
- No writeback on conclusion

**Risk #15: Adding scoped subgraph extraction changes the investigation UX.** Today, investigators freely create hypotheses and causal links. With GRAPH-001, they'd start with a pre-populated subgraph from the org's process model. This is a different workflow. Users who are accustomed to the current freeform approach may find it constraining.

**Risk #16: Writeback conflict detection is complex.** During an investigation (which may last days/weeks), the parent graph may change — other investigations concluding, SPC data updating posteriors, new FMIS rows adding structure. When the investigation writes back, its snapshot of the subgraph is stale. Need merge semantics: "this edge was modified in both the investigation and the parent graph since scoping."

**Recommendation:**
- Keep freeform investigation as the default. Graph-scoped investigation is an opt-in mode.
- When graph-scoped, store a `scoped_at` timestamp. On writeback, compare each proposed change against the parent graph's `updated_at` for the same edge. Flag conflicts for human resolution.
- Add `Investigation.scoped_from_graph` (FK → ProcessGraph, nullable) and `Investigation.scoped_node_ids` (JSONField, nullable) to the model.
- Do NOT auto-populate `synara_state` from the parent graph. Instead, provide a "Load from process model" action that the investigator triggers explicitly.

### 3.2 Signal → Graph Node Mapping — Does Not Exist

Signals have a `source_type` (spc_violation, pc_threshold, etc.) and a generic FK to the source object. But there's no link to specific graph nodes.

**Risk #17: Auto-inferring graph nodes from a Signal requires knowing which process parameter the signal is about.** An SPC violation signal knows its chart, but the chart isn't linked to a graph node yet. A customer complaint knows the product, not the process parameters. The mapping from "what triggered the signal" to "which graph nodes are relevant" is domain-specific and not automatable without additional metadata.

**Recommendation:** Don't auto-infer. Instead:
- When creating a Signal, add an optional `related_node_ids` JSONField.
- The UI shows the process graph and lets the user click relevant nodes when filing a signal.
- When linking a signal to an investigation, suggest those nodes as the initial scope.

### 3.3 SPC → Signal Automation — Does Not Exist

SPC does NOT create Signals today. Results are returned to the frontend. No `Signal.objects.create()` in any SPC code path.

**GRAPH-001 assumes SPC shift detection → staleness flag → Signal (§9.1).** But the prerequisite — SPC results linked to graph nodes — doesn't exist yet. Building this chain:

1. SPC chart linked to ProcessNode (new)
2. SPC out-of-control detection calls `GraphService.flag_stale_edges()` (new)
3. `flag_stale_edges()` creates Signal per stale edge (new)
4. Signal enters triage lifecycle (existing)

**Risk #18: SPC runs frequently (potentially every data point). If auto-signal creation fires on every out-of-control point, Signal volume explodes.** Need debouncing: one signal per chart per shift, or one signal per confirmed pattern (not individual points).

**Recommendation:** Phase this. The SPC → graph node linkage is prerequisite infrastructure. Build that first (as part of ProcessNode's `linked_spc_chart` field). The auto-signal creation is a policy rule, configurable and off by default.

### 3.4 Process Confirmation → Graph Evidence — Does Not Exist

ProcessConfirmation computes a diagnosis (`system_works`, `standard_unclear`, `process_gap`) and emits a tool event. But the diagnosis doesn't flow to any graph edge as evidence.

**GRAPH-001 implies PC results should be evidence on graph edges (§7.2 table).** A `system_works` diagnosis on a controlled document that covers a specific process step is confirmatory evidence on the edges within that step. A `process_gap` diagnosis is contradictory evidence.

**Risk #19: Mapping a PC diagnosis to specific graph edges requires knowing which edges the controlled document covers.** This mapping doesn't exist. Controlled documents are text — they reference procedures, not graph nodes. Building this mapping requires either:
- Manual: quality manager links each controlled document to graph nodes/edges
- Structural: controlled documents reference FMIS rows, which map to graph structure

**Recommendation:** Start with manual linking. Add a `linked_process_nodes` M2M on ControlledDocument. When a PC is completed against that document, the diagnosis becomes evidence on edges connecting those nodes. This is opt-in — documents without node links just don't generate graph evidence.

### 3.5 CI Readiness Score — Graph-Unaware

The CI Readiness Score has 10 indicators, all backward-looking (signal ratios, investigation velocity, commitment fulfillment, etc.). None reference graph state.

**GRAPH-001 §15.7 implies graph health should inform readiness.** Suggested additions:
- Edge calibration coverage (% of edges with empirical evidence)
- Staleness ratio (% of edges flagged stale)
- Measurement coverage (% of nodes with linked measurement systems)

**Risk #20: Adding graph-based indicators changes the readiness score for all orgs.** If an org hasn't built their process graph yet, these indicators would score 0%, dragging down their overall readiness. This penalizes orgs that haven't adopted the new feature.

**Recommendation:** Graph-based indicators should be optional and weighted at 0% until the org has a process graph with ≥10 nodes. Once active, they ramp in gradually. The readiness computation should detect whether a ProcessGraph exists before including graph indicators.

---

## 4. PERSISTENCE & MIGRATION RISKS

### 4.1 New Django Models Required

GRAPH-001 §13.1 specifies:
- `ProcessGraph` — container, one per org (FK → Tenant)
- `ProcessNode` — §3.3 schema (~15 fields)
- `ProcessEdge` — §4.3 schema (~20 fields)
- `EdgeEvidence` — §4.4 schema (~12 fields)

This is 4 new models, ~50 fields, multiple JSONFields, multiple FKs. The migration is straightforward (new tables, no data to move initially). But:

**Risk #21: ProcessNode and ProcessEdge have JSONFields for `distribution`, `spec_limits`, `control_limits`, `operating_region`, `interaction_terms`.** JSONFields are flexible but unsearchable without Postgres-specific operators. If we need to query "all edges where operating_region includes humidity > 60%", that's a JSON path query — functional but slow at scale without a GIN index.

**Recommendation:** Add GIN indexes on the JSON fields that will be queried frequently (at minimum `interaction_terms` and `operating_region`). Monitor query performance once real data exists.

### 4.2 Deprecating Old Models

GRAPH-001 §13.2 says deprecate `core.models.graph.KnowledgeGraph` and `workbench.models.KnowledgeGraph`.

**Risk #22: `workbench.models.KnowledgeGraph` may have production data.** The workbench graph API has 18 endpoints and functional tests. If any user has created a workbench with a knowledge graph, deleting the model loses their data. Need to:
1. Check `SELECT COUNT(*) FROM knowledge_graphs;` in production
2. If rows exist, plan a data migration to the new ProcessGraph models
3. If no rows, safe to deprecate

**Risk #23: `core.models.graph.Entity` is referenced by `FMISRow` FKs.** Even though those FKs are all null, the Django migration system won't let you delete `Entity` while `FMISRow` has FKs pointing to it. Must drop the FMISRow FKs first.

**Risk #24: `core.models.graph.Relationship` has 18 relationship types. `ProcessEdge` has 5.** If we're migrating data from old Relationship records to ProcessEdge, the type mapping is lossy. Types like `HAPPENS_BEFORE`, `CONTAINS`, `IS_TYPE_OF` have no GRAPH-001 equivalent. If any old data uses these types, migration must either drop them or map them to a catch-all.

**Recommendation:**
1. Query production for existing data in both KG tables.
2. Drop FMISRow entity FKs first (they're null anyway — zero data loss).
3. Build new models alongside old ones. Don't delete anything until the new service is live and validated.
4. Add a deprecation warning to the old API endpoints (return a `Deprecation` header) for 30 days before removal.

---

## 5. UX & NAVIGATION RISKS

### 5.1 Graph-First Navigation (§15) Is a Full Frontend Rebuild

GRAPH-001 §15 specifies the graph as the home screen with 6 view lenses, contextual actions on nodes/edges, and multiple entry points. The current navigation model is:

```
Dashboard → sidebar menu → individual tools (FMEA, SPC, etc.)
```

Replacing this with a graph-centric navigation is a significant frontend change. The templates are vanilla Django + inline JS. There's no React/Vue for reactive graph rendering.

**Risk #25: A graph visualization requires either a JS library (D3, Cytoscape, vis.js) or a canvas-based renderer.** The current stack uses Chart.js for charts and inline SVG for simple diagrams. A full process graph with draggable nodes, edge labels, zoom/pan, and contextual menus is a different class of frontend work. This is the most visible change to users and the hardest to iterate on.

**Risk #26: The graph-first navigation changes the information architecture.** Users who know "I go to FMEA to see failure modes" now need to learn "I look at the graph filtered to failure mode nodes." This is conceptually powerful but requires user education. Early adopters may resist.

**Recommendation:**
- Don't replace the existing navigation. ADD the graph as a new surface alongside it.
- Individual tools (FMEA, SPC, DOE) remain accessible via the sidebar. The graph is an additional view that connects them.
- Use Cytoscape.js for the graph renderer — it's the most mature option for directed graphs with the features we need (typed nodes, edge labels, layout algorithms, selection, contextual menus).
- Ship the graph as "Process Map (Beta)" initially. Gather feedback before making it the default home.

### 5.2 Slider UI (§14.5) Is Phase 3

The Process Explorer with sliders, real-time propagation, and sensitivity analysis is the most compelling demo but the hardest to build. It requires:
- Value propagation engine (Monte Carlo, server-side)
- WebSocket or SSE for real-time updates as sliders move
- A separate UI panel with slider controls linked to graph nodes
- Confidence visualization overlaid on the graph

This is a standalone feature that depends on: graph service (done), evidence stacking (done), calibrated edges (requires DOE pipeline), interaction terms (requires fitting service).

**Recommendation:** Spec it, don't build it yet. The slider UI is the demo you show investors. Build it when the graph has enough calibrated edges (from real DOE data) to produce meaningful predictions. Building it on uncalibrated edges just shows wide uncertainty bands — not impressive.

---

## 6. WHAT STAYS UNCHANGED

These components are correctly independent and should NOT be modified for GRAPH-001:

| Component | Why it stays |
|-----------|-------------|
| `syn/synara/` (Cortex, middleware, tenant isolation) | OS-level infrastructure. Completely unrelated to the knowledge graph. |
| Forge (synthetic data) | Utility tool. No graph relationship. |
| Triage (data cleaning) | Utility tool. No graph relationship. |
| Free calculators (OEE, Cpk, sample size) | Marketing/education. Standalone. |
| Blog, whitepapers, landing page | Content. No code dependency. |
| Billing (Stripe), auth, sessions | Infrastructure. Graph-unaware by design. |
| Chat/Guide | Could become graph-aware later, but not required for GRAPH-001 v1. |
| Plant Simulator | Educational DES tool. Separate from the calibrated process model. The graph replaces its function for real processes, but the simulator remains for training. |
| Synara DSL (`dsl.py`) | Leave in place. Potentially useful for operating region constraints later, but not a dependency. |
| Synara Logic Engine (`logic_engine.py`) | Same — leave in place, not a dependency. |
| Harada Method | Personal development. Could link to graph gaps later (§15.7), but not required for v1. |
| Hoshin Kanri | Strategic planning. Could read graph gap reports later, but not required for v1. |

---

## 7. BUILD SEQUENCE (CONSERVATIVE)

Based on the risks above, the build sequence should be:

### Phase 0: Cleanup (Before GRAPH-001 Code)
1. Drop FMISRow's three dead entity FK fields (migration)
2. Instrument `/api/core/graph/` and `/api/workbench/*/graph/` with access logging
3. Add cycle detection to `CausalGraph.add_link()`
4. Add `visited` set safety net to `propagate_belief()`
5. Query production for existing KnowledgeGraph data (both tables)
6. Benchmark `propagate_belief()` at 100/500/1000 nodes

### Phase 1: Core Graph Service (No UI)
1. Create `ProcessGraph`, `ProcessNode`, `ProcessEdge`, `EdgeEvidence` models
2. Build `GraphService` class with CRUD + evidence + gap report
3. Build Synara adapter (Django models ↔ CausalGraph round-trip)
4. Add `recompute_posterior()` method to Synara
5. Add `check_edge_contradiction()` method to Synara
6. Wire FMIS seeding: `seed_from_fmis()` → proposed changes → confirm
7. Comprehensive tests: round-trip, evidence stacking, gap detection, contradiction

### Phase 2: Integration Wiring
1. Add `ProcessNode.linked_spc_chart` FK
2. Wire DOE results → `EdgeEvidence` records
3. Wire PC diagnosis → graph evidence (requires ControlledDocument → node linking)
4. Wire FFT results → graph evidence
5. Wire SPC shift → `flag_stale_edges()` → Signal (with debouncing, off by default)
6. Wire investigation scoping (opt-in, `scoped_from_graph` FK)
7. Wire investigation writeback with conflict detection
8. Update CI Readiness Score with optional graph indicators

### Phase 3: Frontend & UX
1. Graph visualization (Cytoscape.js, "Process Map Beta")
2. View lenses (Process Map, FMEA View, Gap View, Control View)
3. Contextual actions on nodes/edges
4. Auditor Portal graph integration
5. Process Explorer / slider UI (when calibrated edges exist)

### Phase 4: Deprecation
1. Mark old KG APIs as deprecated (30-day notice)
2. Migrate any existing KG data
3. Remove old models and endpoints
4. Update compliance checks and tests

---

## 8. OPEN QUESTIONS (Require Decision Before Code)

### Q1: Where do the new models live?

GRAPH-001 §13.1 says "new `graph` app (or within `loop/`)." This matters for:
- Migration ownership (who runs `makemigrations`?)
- Import paths (every caller needs to import from the right place)
- Circular dependency risk (if `loop/` imports from `graph/` and `graph/` imports from `loop/` for FMIS seeding)

**Options:**
- A) New `graph/` app. Clean separation. Avoids circular deps. More files to maintain.
- B) Inside `loop/`. Fewer apps. But `loop/models.py` is already 1,642 lines. Adding ~200 more lines of graph models makes it unwieldy.
- C) Inside `core/`. Alongside the existing (deprecated) graph models. Migration path is cleaner. But `core/` is already large.

**Recommendation:** New `graph/` app. The graph service is the most important new subsystem. It deserves its own namespace. Import from `graph.models`, `graph.service`, `graph.synara_adapter`. Avoids bloating `loop/` or `core/`.

### Q2: ProcessGraph — one per org, or multiple?

GRAPH-001 §13.1 says "one per org." But real orgs have multiple processes (injection molding line 1, assembly line 2, paint shop). Do they share one graph, or each get their own?

**If one graph:** Simpler. One `ProcessGraph` per Tenant. But a graph with 500 nodes across 5 processes is unwieldy. Scoping becomes critical.
**If multiple:** Each process gets its own graph. But nodes can span processes (ambient humidity). Need cross-graph references or shared nodes.

**Recommendation:** Start with one per org. Add a `process_area` tag on ProcessNode for filtering. If orgs outgrow one graph, we split later — that's a simpler migration than merging.

### Q3: Evidence immutability

GRAPH-001 §16.3 says "evidence is never discarded." But what about erroneous evidence? If an operator enters wrong data, that evidence record should be retractable without deletion.

**Recommendation:** Add `retracted` boolean and `retracted_reason` to EdgeEvidence. Retracted evidence is excluded from posterior computation but remains in the stack for audit. This is the only write operation on an otherwise-immutable record.

### Q4: Tenant isolation

ProcessGraph has FK → Tenant. All queries must filter by tenant. The existing `TenantIsolationMiddleware` handles this for most models. But the `GraphService` is a service class, not a view — it doesn't have access to `request.tenant`. Need to explicitly pass tenant_id to every service method, or use thread-local storage.

**Recommendation:** Explicit parameter. Every `GraphService` method that touches data takes `tenant_id` as its first argument. No thread-local magic. This is testable and auditable.

### Q5: What about individual users (no tenant)?

Not all users are on Team/Enterprise. Individual users have `Project.tenant = null`. Do they get a ProcessGraph?

**Recommendation:** No. ProcessGraph requires a Tenant. Individual users continue using the workbench KnowledgeGraph (which is per-user, per-workbench). The persistent process model is a Team/Enterprise feature — it represents organizational knowledge, which doesn't apply to individual users.

---

## 9. RISK SUMMARY

| # | Risk | Severity | Mitigation |
|---|------|----------|------------|
| 1 | Deprecating core KG API may break unknown users | HIGH | Instrument first, deprecate after 30-day notice |
| 2 | Workbench KG JSON migration is non-trivial | MEDIUM | Check for data first, build dry-run tool |
| 3 | Two Synara engines have different math | MEDIUM | Document which math wins, verify posteriors don't shift |
| 4 | `propagate_belief()` has no cycle detection | HIGH | Fix before extending to persistent graph |
| 5 | FMISRow entity FKs point to wrong model | LOW | Drop them (they're null), add new FKs later |
| 6 | Recency recomputation is O(N) per edge | MEDIUM | Benchmark first, cache if needed |
| 7 | Posterior shift on recency transition | LOW | Expected behavior, document for users |
| 8 | Decay function is org-configurable | LOW | Parameterize, test multiple values |
| 9 | DOE → evidence pipeline doesn't exist | HIGH | Required for interaction term calibration, large scope |
| 10 | `fit_result` JSON schema may evolve | LOW | Add `fit_version` field |
| 11 | Contradiction threshold needs tuning | MEDIUM | Start strict (0.05), add cooldown, log sub-threshold |
| 12 | Contradiction flood from frequent SPC data | MEDIUM | Cooldown per edge, configurable |
| 13 | `from_dict()` mutation bug | MEDIUM | Fix properly, don't rely on deepcopy |
| 14 | Adapter field mapping may drift | MEDIUM | CI schema compatibility check |
| 15 | Investigation scoping changes UX | MEDIUM | Make it opt-in, keep freeform as default |
| 16 | Writeback conflict detection is complex | HIGH | Timestamp comparison, flag conflicts for human resolution |
| 17 | Signal → node mapping requires domain knowledge | MEDIUM | Manual linking, not auto-inference |
| 18 | SPC auto-signal volume | MEDIUM | Debounce, off by default |
| 19 | PC → edge evidence requires document-node mapping | MEDIUM | Manual linking on ControlledDocument |
| 20 | Graph indicators penalize non-adopters | LOW | Optional, ramp-in threshold |
| 21 | JSONField query performance at scale | MEDIUM | GIN indexes |
| 22 | Workbench KG may have production data | HIGH | Check before deprecating |
| 23 | FMISRow FKs block Entity model deletion | LOW | Drop FKs first |
| 24 | Relationship type mapping is lossy | LOW | Only matters if old data exists |
| 25 | Graph visualization is a frontend class change | HIGH | Use Cytoscape.js, ship as Beta |
| 26 | Navigation change requires user education | MEDIUM | Add alongside, don't replace |

---

## 10. BOTTOM LINE

The codebase is structurally ready for GRAPH-001. The Synara engine has the right math (with known limitations). The Loop app has the right lifecycle (Signal → Investigate → Standardize → Verify). The FMIS has the right data (Bayesian S/O/D with evidence update methods).

What's missing is the connective tissue — the persistent graph that every tool reads from and writes to. Building that connective tissue is safe if we:

1. **Don't break what works.** Keep existing APIs running during transition. Don't force graph-scoped investigations. Don't auto-create signals from SPC without debouncing.
2. **Fix known bugs first.** Cycle detection, propagation visited set, `from_dict()` mutation.
3. **Build new, then wire, then deprecate.** New models alongside old. Wire integrations one by one. Deprecate old models only after the new service is validated in production.
4. **Ship the graph UI as additive, not replacement.** "Process Map (Beta)" in the sidebar. Existing tool navigation stays.

The biggest risk is scope. GRAPH-001 is a 16-section spec with 4 Synara extensions, 15 integration points, a new frontend paradigm, and 4 new Django models. Phased delivery is mandatory. Phase 1 (models + service + tests) can ship without touching any existing code. Phase 2 (wiring) touches every tool. Phase 3 (UI) is a frontend project. Phase 4 (deprecation) is cleanup.

Don't try to do it all at once.
