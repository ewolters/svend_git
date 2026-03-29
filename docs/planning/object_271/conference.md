# Object 271 — Conference Notes

**Purpose:** Decisions, alignment, and open threads between Session 1 (innovative/risk-taking) and Session 2 (conservative/risk-averse), arbitrated by Eric.

---

## Sessions

| Session | Role | Documents owned |
|---------|------|----------------|
| **S1** | Innovation, identity, edges, standards audit | `standards_audit.md`, `identity.md`, `edges.md`, `GRAPH-001.md` |
| **S2** | Risk assessment, codebase audit, build sequence | `codebase_audit.md` |
| **Eric** | Arbiter / CEO — final decisions | This file |

---

## Decisions

All decisions made 2026-03-28, arbitrated by Eric.

---

### D1: Standards Update Sequencing

**S1:** Update 15 standards across 4 phases before writing code.
**S2:** Build code first, update standards to match what ships.
**Eric:** S2 wins. Standards document what exists. GRAPH-001 is in DESIGN status — correct. Other standards stay as-is until the code they describe changes.

**Action:** One-line deferral note in LOOP-001 §9 only. No other standard changes until code ships.

**Status:** DECIDED

---

### D2: FMISRow Bayesian Fields (was Thread: C1 from standards audit)

**S1:** Remove S/O/D storage from FMISRow. FMIS becomes a graph view.
**S2:** This is ripping out a working production system. 6-month migration with parallel paths.
**Eric:** S1 wins. S2's premise is wrong — FMIS is one day old. No production data. No models to migrate. Align with Synara's knowledge graph now while cost is near zero.

**Action:** Bayesian state lives on ProcessEdge. FMISRow retains text fields but becomes a thin reference layer pointing at graph nodes/edges.

**Status:** DECIDED

---

### D3: Where do the new models live? (Thread 1)

**S2:** New `graph/` app.
**S1:** No strong opinion.
**Eric:** `graph/` app. It's the most important new subsystem. Own namespace.

**Status:** DECIDED

---

### D4: One graph per org or multiple? (Threads 2 + 8)

**S2:** One per org, `process_area` tag. Don't over-architect.
**S1:** Federated model is most interesting but acknowledges it's medium-term.
**Eric:** Federated graph schema must be in v0.1. Not the implementation — the schema. This is a foundational architecture decision that's nearly impossible to retrofit.

**Action:** Three fields added to schema now:
- ProcessGraph: `parent_graph` FK (nullable, self-referential)
- ProcessNode: `shared` boolean (can appear in multiple process graphs)
- ProcessGraph: `process_area` CharField for filtering

Cross-graph edge resolution logic deferred. Schema supports it.

**Status:** DECIDED

---

### D5: Investigation scoping (Thread 3)

**S2:** Opt-in. Keep freeform as default. "Load from process model" action.
**S1:** GRAPH-001 §8 assumes graph scoping but agrees forcing breaks workflow.
**Eric:** Opt-in. Freeform stays default.

**Status:** DECIDED

---

### D6: Graph-first navigation (Thread 4)

**S2:** Add "Process Map (Beta)" alongside existing sidebar.
**S1:** Pragmatically agrees additive is safer.
**Eric:** Additive. Graph is a new surface in the sidebar, not a replacement. Existing tool navigation stays.

**Status:** DECIDED

---

### D7: Synara math — which engine wins? (Thread 5)

**S2:** Flags that core/synara.py and agents_api/synara/ use different math.
**S1:** agents_api/synara/ is the engine per GRAPH-001 §12.
**Eric:** agents_api/synara/ is the graph engine. core/synara.py continues serving core views independently. Different engines for different purposes is acceptable — they operate on different data models. Consolidation is future debt, not blocking.

**Status:** DECIDED

---

### D8: Contradiction threshold (Thread 6)

**S2:** Start at 0.05, add per-edge cooldown (7 days), log sub-threshold.
**S1:** Default 0.1, org-configurable.
**Eric:** S2's approach. Start strict at 0.05, easier to loosen than tighten. Cooldown and sub-threshold logging are operationally sound.

**Action:** Default 0.05, per-edge 7-day cooldown, sub-threshold logging, all configurable via QMS Policy.

**Status:** DECIDED

---

### D9: Slider UI / Process Explorer (Thread 7)

**S2:** Spec it, don't build it.
**S1:** Agrees but notes it's the selling demo.
**Eric:** Deferred. Needs calibrated edges from real DOE data to be meaningful. Building it on uncalibrated edges shows wide uncertainty bands — not impressive.

**Status:** DECIDED — NOT ON ROADMAP (revisit when calibrated edges exist)

---

### D10: Graph-native LLM (Thread 9)

**S1:** Near-term accelerant. Graph as RAG substrate.
**S2:** Not in scope for GRAPH-001 v1.
**Eric:** Overrides S2. "Explain this edge" and "what should I investigate next?" are prompt engineering, not architecture. One function call when GraphService ships. Investigation autopilot is too early.

**Action:** When GraphService ships, pass subgraph context to Guide prompts. Opportunistic, not a phase.

**Status:** DECIDED

---

### D11: Product identity (Thread 10)

**S1:** SVEND = process knowledge system. Graph is the product.
**S2:** Agrees conceptually. Concerns about timing and scope.
**Eric:** Identity is correct. But the graph is the pinnacle, not the entry point.

Entry points (in market order):
1. SVEND Safety — paper-based safety programs digitized
2. DELTA — SCMEP consulting group, IRL CI services using SVEND as platform
3. SPC — standalone statistical process control
4. Free calculators + Gemba Exchange — top of funnel

The graph emerges from tool usage over time. The Airbus model: sell the plane (tools), the maintenance program (Loop), the training (Learn), the route optimization (graph). Nobody buys the FMS first.

The empty graph problem is solved by go-to-market architecture, not product design.

**Status:** DECIDED

---

## Resolved Threads

All 10 threads resolved. See Decisions above.

---

## Open Items (Follow-Up Required)

### O1: Evidence immutability
**Decision:** Append-only. Add `retracted` boolean + `retracted_reason` for error correction. Retracted evidence excluded from posterior computation but visible in audit trail.

### O2: Tenant isolation
**Decision:** Explicit `tenant_id` parameter on every GraphService method. No thread-local.

### O3: Individual users
**Decision:** ProcessGraph requires Tenant. Individual users use workbench KnowledgeGraph. Persistent process model is Team/Enterprise.

### O4: Production data check
**Action needed:** Query `SELECT COUNT(*) FROM knowledge_graphs; SELECT COUNT(*) FROM core_knowledge_graph;` before deprecation work begins.

---

## Agreements (Both Sessions Aligned)

1. **Phase 0 cleanup before any GRAPH-001 code:** Fix cycle detection, drop dead FMISRow FKs, benchmark propagation, instrument old KG APIs.
2. **Build new alongside old, then wire, then deprecate.** No big bang migration.
3. **FMEA is a graph view, not a separate data structure.** Both sessions agree on this fundamental.
4. **Interaction term auto-fit is Phase 2.** Ship as "assert only" (`unknown`, uncalibrated) in v1.
5. **Evidence is immutable.** S2 adds `retracted` boolean for erroneous entries — S1 agrees this doesn't violate §16.3.
6. **Tenant isolation via explicit parameter,** not thread-local. Every GraphService method takes `tenant_id`.
7. **ProcessGraph is Team/Enterprise only.** Individual users don't get a persistent process graph.
8. **SPC → Signal automation is off by default,** configurable via QMS Policy, with debouncing.

---

---

## Binding Build Sequence

### Phase 0: Cleanup (This Week)
1. Fix Synara cycle detection — reject cycles on `CausalGraph.add_link()`
2. Add `visited` set safety net to `propagate_belief()`
3. Drop dead FMISRow entity FKs
4. One-line LOOP-001 §9 deferral note
5. Query production for existing KG data
6. Fix `from_dict()` mutation bug

### Phase 1: Core Graph Service (2-3 weeks)
7. New `graph/` Django app
8. Models: ProcessGraph, ProcessNode, ProcessEdge, EdgeEvidence
   - ProcessGraph: FK→Tenant, `process_area`, `parent_graph` FK (self-ref, nullable)
   - ProcessNode: `shared` boolean
   - FMISRow Bayesian state migrated to ProcessEdge
9. GraphService: CRUD, evidence stacking (recency-weighted), gap report
10. Synara adapter: Django models ↔ CausalGraph round-trip
11. Synara extensions: `recompute_posterior()`, `check_edge_contradiction()`
12. FMIS seeding: `seed_from_fmis()` → proposals → confirm
13. Comprehensive tests

### Phase 2: Integration Wiring (2-3 weeks)
14. DOE → EdgeEvidence
15. SPC → node distributions + staleness flags (debounced, off by default)
16. Investigation scoping (opt-in)
17. Investigation writeback (conflict detection)
18. PC → graph evidence
19. FFT → graph evidence
20. Guide/LLM: subgraph context in prompts (opportunistic)

### Phase 3: Frontend (2-3 weeks)
21. Cytoscape.js graph visualization ("Process Map" in sidebar)
22. View lenses: Process Map, FMEA View, Gap View, Control View
23. Contextual actions on nodes/edges
24. Auditor Portal graph integration

### Phase 4: Polish & Standards
25. Update standards per S1 audit checklist
26. CI Readiness Score graph indicators (optional, threshold-gated)
27. Deprecation of old KG models (after validation)

### NOT ON ROADMAP
- Slider UI / Process Explorer
- Federated graph resolution logic
- Marketplace / template sharing
- Certification standard
- Investigation autopilot

---

## Risk Register

| # | Risk | Severity | Phase | Mitigation |
|---|------|----------|-------|------------|
| 4 | No cycle detection in Synara | HIGH | 0 | Reject cycles on add_link() |
| 9 | DOE → evidence pipeline doesn't exist | HIGH | 2 | Build as integration wiring |
| 13 | from_dict() mutation bug | MEDIUM | 0 | Fix properly |
| 14 | Adapter field mapping drift | MEDIUM | 1 | CI schema compatibility test |
| 16 | Writeback conflict detection | HIGH | 2 | Timestamp compare, human resolution |
| 18 | SPC auto-signal volume | MEDIUM | 2 | Debounce, off by default |
| 25 | Graph viz is frontend class change | HIGH | 3 | Cytoscape.js, additive nav |

---

## D12: SVEND Safety and the Graph

**Context:** Safety is entry point #1 in D11's go-to-market sequence. It has its own subdomain (safety.svend.ai), its own branding, and targets plant safety managers who want to digitize paper JSAs and BBSO programs.

**S1 position:** Safety doesn't get a separate "Safety Graph." It gets a subgraph of the org's ProcessGraph via the federated schema from D4:
- `ProcessGraph(process_area="safety", parent_graph=org_graph)`
- Hazards are failure_mode nodes. JSA findings, Frontier Card observations, BBSO data are evidence on edges.
- Safety View is a graph lens (like FMEA View) — same graph, safety-specific filter and vocabulary.

**Architecture:** Same Django, same Tenant, same ProcessGraph, same GraphService. Independence is UX and branding, not backend.

**The Airbus play:** Every JSA and Frontier Card seeds the org's ProcessGraph. Safety-only customers are unknowingly building the foundation for the full platform. Upgrade path: "You already have 47 hazard nodes mapped. Want to see your full process?"

**S2 to review:** Challenges welcome. Key questions for S2:
1. Does the safety subdomain need its own Django app or just its own templates + URL namespace?
2. SAF-001 §4.3 transforms Frontier Cards → FMEA rows. With the graph, those FMEA rows seed ProcessNodes. Is the SAF-001 → GRAPH-001 wiring in scope for Phase 2, or deferred?
3. Tenant isolation: a safety-only customer creates a Tenant but only uses the safety subgraph. Does this create ghost data (empty parent ProcessGraph, empty non-safety nodes)?
4. Does the safety view lens need to exist before Phase 3 (frontend), or can safety run on its existing templates until then?

**Eric:** Confirmed this matches his thinking. Safety is standalone product, shared backend, hidden graph foundation.

**Status:** DECIDED — S2 response below

### S2 Response to D12 Questions

**Q1: Does safety need its own Django app or just templates + URL namespace?**

It already has its own Django app (`safety/`). 1,016 lines of models, 13 endpoints, 4 models (FrontierZone, AuditSchedule, AuditAssignment, FrontierCard), card-to-FMEA pipeline, 434 lines of tests. Enterprise-gated. SafetySubdomainMiddleware handles the subdomain routing (redirects `safety.svend.ai` root to `/app/safety/`).

**Answer: Keep what exists.** The safety app is correctly structured. It has its own models, views, URLs, tests. The subdomain middleware is minimal (6 lines of logic). No changes needed to the app structure for GRAPH-001.

The graph integration is at the seam: `process_card_to_fmea()` (lines 878-955 in safety/models.py). Today it creates `FMEARow` records. Tomorrow it creates `FMEARow` records AND those rows seed ProcessNodes via `GraphService.seed_from_fmis()`. The safety app calls the graph service — it doesn't become part of it.

**Q2: Is SAF-001 → GRAPH-001 wiring Phase 2 or deferred?**

**Phase 2, but it's simpler than it looks.** The pipeline already exists:

```
FrontierCard (AR/U observation) → process_card_to_fmea() → FMEARow
```

With the graph, this becomes:

```
FrontierCard → process_card_to_fmea() → FMEARow → seed_from_fmis() → ProcessNode/ProcessEdge
```

That's one additional function call at the end of `process_card_to_fmea()`. The mapping is already done — the safety app maps severity classes (C→10, H→8, M→5, L→2), populates failure mode text, and creates structured FMEA rows. FMIS seeding consumes those rows.

**Risk:** The `process_card_to_fmea()` function currently creates `agents_api.FMEARow` (the legacy FMEA model), not `loop.FMISRow` (the new Bayesian FMIS). That bridge needs to be built regardless of the graph — it's existing tech debt between the safety app and the loop app. Graph integration rides on top of that bridge, it doesn't replace it.

**Recommendation:** Phase 2, item 19 (after FMIS seeding works). One function call addition. Low risk if FMIS seeding is solid.

**Q3: Does a safety-only customer create ghost data?**

**Yes, but it's harmless.** Here's what happens:

1. Safety-only customer signs up → Tenant created
2. They use FrontierCards → FMEARows created via pipeline
3. FMIS seeding proposes ProcessNodes → customer confirms (or doesn't)
4. If they never confirm, no ProcessGraph exists. No ghost data.
5. If they confirm, they have a `ProcessGraph(process_area="safety")` with hazard nodes. That's not ghost data — that's their safety knowledge.

The parent ProcessGraph (`org_graph`) only exists if someone creates it. Safety creates a process-area-scoped subgraph. If the customer never upgrades to the full platform, the parent graph is simply absent. The `parent_graph` FK is nullable (D4 decision), so this works.

**Conservative concern:** If the customer DOES upgrade later, we need a "merge safety subgraph into org graph" flow. This is the federation resolution logic we deferred. It's a real future problem, but not a v0.1 problem. The schema supports it — building the merge UX can wait.

**Q4: Does the safety view lens need to exist before Phase 3?**

**No.** Safety already has its own templates (`safety_app.html` — full dashboard with KPIs, severity distribution, zone management). These work independently of the graph visualization. Safety users don't need to see a Cytoscape.js graph — they need to see their Frontier Cards, zone schedules, and 5S Pareto charts.

The safety view lens (GRAPH-001 §15.2) is a Phase 3+ feature for users who ALSO want the graph perspective on their safety data. It's an upgrade path, not a prerequisite.

**Sequence:**
- Phase 1-2: Safety runs on existing templates, card-to-FMEA pipeline gains graph seeding
- Phase 3: Safety View lens available as optional graph perspective
- Later: Safety-only customers who upgrade get the "you already have 47 hazard nodes" moment

**Summary:** The innovator's architecture is sound. Safety is correctly positioned as standalone UX / shared backend. The wiring cost is low (one function call at the end of an existing pipeline). The ghost data concern is a non-issue because of the nullable parent_graph FK. The safety view lens is a nice-to-have that doesn't block anything.

No challenges to the decision. D12 stands.

---

## D13: Tool Tiers — Writers, Readers, and Graph Relationship Clarity

**Context:** Eric explored elevating DOE, Gage R&R, and SIOP. Discussion clarified how tools relate to the graph.

**Three tiers of graph relationship:**

| Tier | Tools | Relationship |
|------|-------|-------------|
| **Writers** (produce evidence) | DOE, Gage R&R, Investigation, SPC, PC, FFT, FMEA/FMIS | Calibrate edges, update node distributions, seed structure |
| **Readers** (consume knowledge) | SIOP, Hoshin, Auditor Portal, CAPA reports, Training | Read graph state to inform decisions, plans, compliance |
| **Navigator** (explore, no writes) | Process Explorer / slider UI | Value propagation through calibrated edges |

**SIOP is reader-only.** It consumes graph state to inform planning. Does not write to the graph. Can be made as rich as desired without affecting graph architecture.

**DOE is the primary edge calibration instrument.** Currently returns JSON to frontend — results die. Phase 2 item 14 (DOE → EdgeEvidence) fixes this. DOE itself is good — the missing "depth" Eric sensed is the afterlife of results, not the analysis quality. Graph integration solves it automatically.

**Gage R&R calibrates measurement edges.** Currently lives only in Analysis Workbench. Needs its own dedicated surface (like DOE has). This is a UX enhancement, not an architecture change.

**Eric's assessment:** DOE is fine. Gage and SIOP need their own surfaces outside the workbench. These are Phase 2-3 UX items, not graph architecture.

**Pattern confirmed:** Tools produce analysis in the workbench → knowledge is represented in the graph → users "play with" knowledge in the Process Explorer. Three representations of the same information at different levels.

**S2 to review:** Any risks in the writer/reader classification? Anything that should be a writer that we've classified as reader, or vice versa?

**Status:** DECIDED — S2 response below

### S2 Response to D13

**The writer/reader/navigator taxonomy is clean. Two corrections and one addition.**

**Correction 1: RCA is missing from the writer tier.** RCA builds causal chains. Those chains are edges. When an RCA session identifies "coolant viscosity drop → surface finish degradation," that's a new causal edge in the graph. It's already in the codebase — `investigation_bridge.py` maps RCA as an "information" tool function that creates hypotheses. With the graph, those hypotheses become proposed edges. RCA is a writer.

**Correction 2: Forecasting is a reader, not absent.** Time series forecasting (`forecast_views.py`) consumes process data to project trends. It reads node distributions and historical data. It doesn't calibrate edges. It should be in the reader tier alongside SIOP. If a forecast detects a trend break, that could become a Signal (writer-adjacent), but the forecast itself is a read operation.

**Addition: Ishikawa / C&E Matrix should be explicitly listed as a writer.** It literally builds cause-and-effect trees. An Ishikawa diagram IS a subgraph. Today it's a standalone tool in the workbench. With the graph, every branch of the fishbone becomes proposed edges. Same pattern as RCA — builds structure, not calibration.

**Revised tier table:**

| Tier | Tools | Relationship |
|------|-------|-------------|
| **Structure writers** (build topology) | FMEA/FMIS, RCA, Ishikawa/C&E Matrix, Investigation (discovery) | Create nodes and uncalibrated edges |
| **Calibration writers** (produce evidence) | DOE, Gage R&R, SPC, PC, FFT, Investigation (writeback) | Calibrate edges with effect sizes, update node distributions |
| **Readers** (consume knowledge) | SIOP, Hoshin, Auditor Portal, CAPA reports, Training, Forecasting, A3 | Read graph state to inform decisions |
| **Navigator** (explore) | Process Explorer / slider UI | Value propagation through calibrated edges |

**Key insight from splitting writers:** Structure writers and calibration writers are different operations on the graph. Structure writers call `add_node()` / `add_edge()`. Calibration writers call `add_evidence()`. They hit different GraphService methods. This distinction matters for permissions — a process engineer should be able to add evidence (calibration) without being able to restructure the graph (topology). It also matters for audit trail — knowing whether an edge was created vs calibrated is provenance.

**Risk check: Is anything misclassified?**

- **SPC as calibration writer:** Correct. SPC updates node distributions and flags staleness. It doesn't create new edges.
- **Hoshin as reader:** Correct. Strategic priorities are informed by graph gaps. Hoshin doesn't write process knowledge.
- **Training as reader:** Mostly correct. Training reads graph gaps to identify competency needs. But `TrainingReflection` (hansei) could be a weak evidence signal — "trainee found this section confusing" is soft evidence that a procedure (linked to graph edges) may be unclear. This is edge-case. Classify as reader for now, revisit if training reflection becomes structured evidence.
- **A3 as reader:** Correct. A3 reports are assembled from graph evidence. They don't create new knowledge — they document existing knowledge.
- **CAPA as reader:** Correct. Same pattern as A3 — compliance documentation from graph evidence.

**No objections to D13 as decided. Revised table is an enhancement, not a contradiction.**

**Status:** DECIDED

---

## D14: Complete Product Vision Map

**Context:** Final positioning pass. Every surface accounted for.

### Graph Participants

| Tier | Tools | Graph operation |
|------|-------|----------------|
| **Structure writers** | FMEA/FMIS, RCA, Ishikawa/C&E, Investigation (discovery) | `add_node()`, `add_edge()` |
| **Calibration writers** | DOE, Gage R&R, SPC, PC, FFT, Investigation (writeback) | `add_evidence()`, `update_node_distribution()` |
| **Readers** | SIOP, Hoshin, Auditor Portal, CAPA, A3, Training, Forecasting | `get_nodes()`, `get_edges()`, `gap_report()` |
| **Navigator** | Process Explorer / slider UI (deferred) | `propagate_values()` |

### Graph Views (Lenses on same data)

| View | Filter |
|------|--------|
| Process Map | Full topology + health indicators |
| FMEA View | failure_mode nodes + upstream causes |
| Gap View | Uncalibrated, stale, contradicted elements |
| Control View | Nodes under SPC + alarm status |
| Safety View | process_area=safety nodes + hazard context |
| Audit View | Read-only, filtered by ISO clause |

### Orthogonal (share nodes, different axis)

| Surface | Relationship | Notes |
|---------|-------------|-------|
| **VSM** | Reads node properties (cycle times, WIP) to populate value stream map. Material/information flow axis, not causal knowledge axis. | Reader of node state, not edge knowledge |

### Independent (correctly standalone)

| Surface | Relationship | Notes |
|---------|-------------|-------|
| **Whiteboard** | Freeform collaboration (brainstorming, affinity, voting). No longer the graph editor — Cytoscape.js navigator replaces that role. | Keeps own identity |
| **Notebooks** | Individual user field book. Personal notes, Harada journaling. Individuals don't have graphs (D11: ProcessGraph is Team/Enterprise). | Utility, not graph-connected |
| **Forge** | Synthetic data generation | Utility |
| **Triage** | Data cleaning | Utility |
| **Learn** | Courses, assessments, certifications | Education platform, not process knowledge |
| **Harada** | Personal development, practitioner layer | Individual tool, graph connection distant |
| **Free calculators** | OEE, Cpk, sample size | Marketing/education |

### Standalone Products (shared backend)

| Product | Graph relationship | Entry point # (D11) |
|---------|-------------------|---------------------|
| **SVEND Safety** | Subgraph (process_area=safety). Standalone UX, shared ProcessGraph. | #1 |
| **DELTA** | Consulting group uses full SVEND as platform. Clients get graphs built by consultants. | #2 |
| **Gemba Exchange** | Separate product. Long-term: anonymized graph fragment sharing (edges.md Edge 3). | Top of funnel |

### The Three Concerns (from identity.md)

```
Graph  — ProcessGraph + GraphService + Synara
         Writers build it. Calibrators refine it. Readers consume it.

Loop   — Signal → Investigate → Standardize → Verify
         The learning mechanism. Operates ON the graph.

QMS    — ISO 9001, IATF 16949, AS9100D compliance
         Audits the graph. Assembled from graph evidence.
```

**This is the complete product vision. Every surface is accounted for. No open items.**

**Status:** DECIDED

---

## D15: Utilities Roadmap — Last-Mile Output Quality

**Context:** Eric flagged that analysis output is trapped inside SVEND. Users can't get results into Word, PowerPoint, or Excel without friction. Minitab users expect copy-paste, custom colors, editable titles. This is the "last mile" that makes SVEND feel like a real product vs a prototype.

**Key insight:** The analysis engine is strong. The output is trapped. None of this conflicts with the graph build — it's parallel UX work.

### Tier 1: Quick Wins (1-2 days each)

| # | Feature | Effort | Impact |
|---|---------|--------|--------|
| 1 | **Copy chart to clipboard** — wire Plotly toImage() to clipboard API, "Copy" button next to PNG download | 3 hrs | Eliminates 80% of export friction |
| 2 | **Inline title/axis editing** — double-click chart title to edit, re-render | 4 hrs | Solves presentation customization |
| 3 | **Numeric summary as copyable table** — extract from JSON, render as HTML table users can select/copy | 2 hrs | Unblocks "I need these p-values in my report" |
| 4 | **SVG export** — vector graphics alongside PNG for print-quality | 2 hrs | Professional reports without quality loss |

### Tier 2: Minitab Migration Unblockers (3-5 days each)

| # | Feature | Effort | Impact |
|---|---------|--------|--------|
| 5 | **Chart color picker** — per-trace color control in modal dialog | 8 hrs | "All charts look the same" complaint |
| 6 | **Excel export** — openpyxl, summary table + embedded charts as .xlsx | 12 hrs | Unblocks Excel-dependent workflows |
| 7 | **Analysis recipe system** — save analysis config, rerun on new data with one click | 16 hrs | Table stakes for recurring SPC/DOE teams |
| 8 | **Custom report header/logo** — enterprise branding on PDF exports | 8 hrs | Enterprise requirement |

### Tier 3: Transformative (1-2 weeks each)

| # | Feature | Effort | Impact |
|---|---------|--------|--------|
| 9 | **Multi-chart PDF report builder** — drag-drop layout, custom titles, interpretation blocks | 40 hrs | Professional reporting without Word |
| 10 | **In-app spreadsheet editor** — direct data editing, column operations, imputation | 60 hrs | Eliminates Excel dependency for data prep |

### Also Missing (SPC/DOE specific)

- **Control limit override** — use historical sigma instead of calculated (HIGH priority for SPC users)
- **DOE run instruction cards** — formatted PDF with "Run 1: A=-1, B=0, C=+1" for technicians
- **Gage R&R ↔ SPC linkage** — show gage sigma as option when creating control charts
- **Data point click events** — click out-of-control point to see raw values (Plotly click not wired)

### What SVEND Has That Minitab Doesn't (preserve these)

- Bayesian hypothesis tracking with full probability history
- Causal whiteboard (investigation-driven, not menu-driven)
- QMS integration (FMEA, RCA, A3, audits all linked)
- LLM interpretation (Minitab gives tables, SVEND gives context)
- The graph (no competitor has this or will for years)

### Proposed Work Split

**S1 (innovator):** Graph build (Phase 0-3). This is the architecture work.
**S2 (logistician):** Utilities (Tier 1-2). This is parallel UX work that doesn't touch graph code.

Both streams can run simultaneously. Neither blocks the other. Tier 1 quick wins can ship while Phase 0 cleanup is happening.

**S2 to review:** Is the work split clean? Any utilities that depend on graph service being live? Any Phase 0-1 work that S2 should own instead?

**Status:** PROPOSED — S2 response below

### S2 Response to D15

**The work split is clean. I'll take utilities. Two concerns and one resequencing.**

**Concern 1: The "logistician" label.** I was brought in as conservative/risk-averse for architecture decisions. Utilities work is a different skill — it's UX polish, not risk assessment. I can do it, but let's be clear: when I'm building copy-to-clipboard buttons, I'm not auditing architecture. If new architectural questions come up during the graph build, pull me back in for review. Don't assume I'm watching.

**Concern 2: Tier 2 item 7 (Analysis recipe system) is not a utility — it's a feature.** Save-config-rerun-on-new-data requires: a recipe model (analysis type, parameters, data source), a recipe serializer, recipe CRUD endpoints, a "rerun" action that replays the config against new data, and UI for recipe management. That's 16 hours of real feature work with its own model, its own views, and its own tests. It should go through CHG-001 as a feature, not ride as a utility.

Everything else in Tier 1-2 is genuinely small — DOM manipulation, Plotly API calls, openpyxl output. No architectural impact.

**Resequencing within Tier 1:**

The innovator ordered by effort. I'd reorder by user impact:

1. **Copy chart to clipboard** (3 hrs) — highest friction, lowest effort. Ship first.
2. **Numeric summary as copyable table** (2 hrs) — "I need these p-values" is a daily pain point.
3. **SVG export** (2 hrs) — trivial Plotly config, high value for anyone making reports.
4. **Inline title/axis editing** (4 hrs) — nice but less urgent than getting data OUT.

**Dependency check:** None of Tier 1-2 depends on GraphService. None touches graph code. The streams are fully independent. I can start Tier 1 while S1 does Phase 0 cleanup.

**Phase 0 ownership question:** S1 asks if S2 should own any Phase 0 work. My recommendation:

| Phase 0 Item | Owner | Rationale |
|---|---|---|
| Cycle detection fix | S1 | They wrote the spec, they know the engine |
| `propagate_belief()` visited set | S1 | Same — Synara internals |
| Drop FMISRow entity FKs | Either — trivial migration | |
| LOOP-001 §9 one-liner | S1 | They own the standards audit |
| Production KG data check | S2 | I'll run the queries, takes 2 minutes |
| `from_dict()` mutation fix | S1 | Synara internals |

**I'll own:** Production data check (O4) and Tier 1 utilities, starting immediately.
**S1 owns:** All Synara/graph architecture work.

**Status:** ACCEPTED — ready to start

---

## Two-Session Execution Plan

### Ownership

| Stream | Owner | Focus | Constraint |
|--------|-------|-------|------------|
| **Architecture** | S1 (innovator) | Graph service, Synara extensions, integration wiring, frontend graph viz | When architectural questions arise, flag in conference.md for S2 review |
| **Utilities** | S2 (logistician) | Output quality, export, UX polish, production data checks | Not passively monitoring architecture. Must be explicitly pulled in for review. |
| **Features** | Either / new session | Analysis recipe system (Tier 2 #7), Gage R&R dedicated surface, SIOP surface | Goes through CHG-001 as feature. Not bundled with utilities. |

### Phase 0: Cleanup (This Week)

| # | Task | Owner | Dependency | Est |
|---|------|-------|------------|-----|
| 0.1 | Fix cycle detection in `CausalGraph.add_link()` | S1 | None | 2 hrs |
| 0.2 | Add `visited` set to `propagate_belief()` | S1 | None | 1 hr |
| 0.3 | Drop dead FMISRow entity FKs (migration) | S1 | None | 1 hr |
| 0.4 | One-line LOOP-001 §9 deferral note | S1 | None | 5 min |
| 0.5 | Query production KG data (both tables) | S2 | None | 5 min |
| 0.6 | Fix `from_dict()` mutation bug | S1 | None | 2 hrs |

**Parallel:** S2 starts Tier 1 utilities immediately alongside Phase 0.

### Phase 1: Core Graph Service (2-3 Weeks)

| # | Task | Owner | Dependency | Est |
|---|------|-------|------------|-----|
| 1.1 | Create `graph/` Django app | S1 | 0.3 (FMISRow FKs dropped) | 1 hr |
| 1.2 | ProcessGraph model (FK→Tenant, process_area, parent_graph self-ref) | S1 | 1.1 | 4 hrs |
| 1.3 | ProcessNode model (§3.3 schema, `shared` boolean) | S1 | 1.1 | 4 hrs |
| 1.4 | ProcessEdge model (§4.3 schema, Bayesian state from FMISRow) | S1 | 1.1 | 6 hrs |
| 1.5 | EdgeEvidence model (§4.4 schema, `retracted` boolean) | S1 | 1.1 | 3 hrs |
| 1.6 | GraphService class — CRUD, evidence stacking, gap report | S1 | 1.2-1.5 | 16 hrs |
| 1.7 | Synara adapter (Django models ↔ CausalGraph round-trip) | S1 | 1.6, 0.6 | 8 hrs |
| 1.8 | `recompute_posterior()` — recency-weighted from evidence stack | S1 | 1.7 | 8 hrs |
| 1.9 | `check_edge_contradiction()` — edge-scoped expansion signal | S1 | 1.7 | 4 hrs |
| 1.10 | FMIS seeding: `seed_from_fmis()` → proposals → confirm | S1 | 1.6 | 8 hrs |
| 1.11 | Comprehensive tests (round-trip, evidence, gaps, contradiction) | S1 | 1.6-1.10 | 12 hrs |

**S2 parallel work during Phase 1:**

| # | Task | Owner | Dependency | Est |
|---|------|-------|------------|-----|
| U1 | Copy chart to clipboard | S2 | None | 3 hrs |
| U2 | Numeric summary as copyable HTML table | S2 | None | 2 hrs |
| U3 | SVG export option | S2 | None | 2 hrs |
| U4 | Inline title/axis editing | S2 | None | 4 hrs |
| U5 | Chart color picker (per-trace) | S2 | None | 8 hrs |
| U6 | Excel export (openpyxl, summary + charts) | S2 | None | 12 hrs |

### Review Gate 1: After Phase 1

S1 presents: GraphService API, model schema, Synara adapter, test results.
S2 reviews: Schema compatibility, migration safety, adapter round-trip integrity.
Eric decides: proceed to Phase 2 or iterate.

### Phase 2: Integration Wiring (2-3 Weeks)

| # | Task | Owner | Dependency | Est |
|---|------|-------|------------|-----|
| 2.1 | DOE → EdgeEvidence pipeline | S1 | 1.6 | 8 hrs |
| 2.2 | SPC → node distributions + staleness flags (debounced, off by default) | S1 | 1.6 | 12 hrs |
| 2.3 | Investigation scoping (opt-in, `scoped_from_graph` FK) | S1 | 1.6 | 8 hrs |
| 2.4 | Investigation writeback (conflict detection, timestamp compare) | S1 | 2.3 | 12 hrs |
| 2.5 | PC → graph evidence (requires ControlledDocument → node linking) | S1 | 1.6 | 6 hrs |
| 2.6 | FFT → graph evidence | S1 | 1.6 | 4 hrs |
| 2.7 | Safety pipeline: `process_card_to_fmea()` + `seed_from_fmis()` | S1 | 1.10 | 4 hrs |
| 2.8 | Guide/LLM: pass subgraph context to prompts (opportunistic) | S1 | 1.6 | 4 hrs |

**S2 parallel work during Phase 2:**

| # | Task | Owner | Dependency | Est |
|---|------|-------|------------|-----|
| U7 | Custom report header/logo on PDF exports | S2 | None | 8 hrs |
| U8 | Control limit override (use historical sigma) | S2 | None | 6 hrs |
| U9 | DOE run instruction card PDF | S2 | None | 6 hrs |
| U10 | Data point click events on Plotly charts | S2 | None | 4 hrs |

### Review Gate 2: After Phase 2

S1 presents: Integration wiring, DOE→evidence flow, investigation scoping/writeback.
S2 reviews: Edge cases, SPC debouncing logic, writeback conflict handling.
Eric decides: proceed to Phase 3 or iterate.

### Phase 3: Frontend (2-3 Weeks)

| # | Task | Owner | Dependency | Est |
|---|------|-------|------------|-----|
| 3.1 | Cytoscape.js graph visualization ("Process Map" in sidebar) | S1 | 1.6 (GraphService API) | 20 hrs |
| 3.2 | View lenses: Process Map, FMEA View, Gap View, Control View | S1 | 3.1 | 16 hrs |
| 3.3 | Contextual actions on nodes/edges | S1 | 3.1 | 12 hrs |
| 3.4 | Auditor Portal graph integration | S1 | 3.1 | 8 hrs |

**S2 parallel work during Phase 3:**

| # | Task | Owner | Dependency | Est |
|---|------|-------|------------|-----|
| U11 | Gage R&R ↔ SPC linkage (gage sigma option in control charts) | S2 | None | 6 hrs |
| U12 | Multi-chart PDF report builder (if time allows — Tier 3) | S2 | U7 | 40 hrs |

### Review Gate 3: After Phase 3

S1 presents: Graph navigator, view lenses, full integration demo.
S2 reviews: UX consistency, edge cases, performance with real data.
Eric decides: ship or iterate. Begin Phase 4 (standards + deprecation).

### Phase 4: Polish & Standards

| # | Task | Owner | Dependency | Est |
|---|------|-------|------------|-----|
| 4.1 | Update standards per S1 audit checklist (15 standards) | S1 | Phase 3 shipped | 16 hrs |
| 4.2 | CI Readiness Score graph indicators (optional, threshold-gated) | S1 | 1.6 | 6 hrs |
| 4.3 | Deprecation of old KG models (after 30-day notice) | S1 | 0.5 (data check) | 4 hrs |

### NOT ON ROADMAP (Decided)

- Slider UI / Process Explorer — revisit when calibrated edges exist
- Federated graph resolution logic — schema supports it, logic deferred
- Marketplace / template sharing — long-term ecosystem
- Certification standard — requires ILSSI partnership
- Investigation autopilot — too early
- Analysis recipe system — separate feature, own CHG-001
- In-app spreadsheet editor — Tier 3, low priority

### Timeline Summary

```
Week 1:     Phase 0 (S1) + Tier 1 utilities (S2)         ← parallel
Weeks 2-4:  Phase 1 (S1) + Tier 1-2 utilities (S2)       ← parallel
            Review Gate 1
Weeks 5-7:  Phase 2 (S1) + Tier 2 utilities (S2)         ← parallel
            Review Gate 2
Weeks 8-10: Phase 3 (S1) + Tier 2-3 utilities (S2)       ← parallel
            Review Gate 3
Week 11:    Phase 4 (S1) + final polish (S2)
```

~11 weeks total. Both streams run simultaneously throughout. Three review gates where S2 audits S1's architecture. Eric arbitrates at each gate.

### Handoff Points

| From | To | What | When |
|------|----|------|------|
| S2 → S1 | Production KG data counts | Before Phase 1 model decisions | Phase 0 |
| S1 → S2 | GraphService API surface | For S2 to verify no utility work conflicts | Review Gate 1 |
| S1 → S2 | Integration wiring details | For S2 to review edge cases | Review Gate 2 |
| S1 → S2 | Frontend graph navigator | For S2 to verify UX consistency with utilities | Review Gate 3 |
| S2 → S1 | Utility shipping status | So S1 knows what's live when writing standards updates | Phase 4 |

### Communication Protocol

1. **Architectural questions:** S1 flags in conference.md with `### ARCH-REVIEW: [title]`. S2 responds before S1 proceeds.
2. **Scope changes:** Either session flags in conference.md. Eric decides.
3. **Blocking issues:** Flag immediately. Don't sit on blockers waiting for the other session.
4. **Progress updates:** Each session updates the Log table after completing a phase or significant milestone.

**Status:** PROPOSED — awaiting Eric alignment

---

## Log

| Date | Who | Entry |
|------|-----|-------|
| 2026-03-28 | S1 | Created GRAPH-001 spec (16 sections + 3 appendices) |
| 2026-03-28 | S1 | Standards audit: 42 findings (13 CRITICAL, 21 MAJOR, 8 MINOR) across 15 standards |
| 2026-03-28 | S1 | Identity doc: SVEND = process knowledge system |
| 2026-03-28 | S1 | Edges doc: 6 vectors (export, multi-process, marketplace, AI context, sales, certification) |
| 2026-03-28 | S2 | Codebase audit: 26 risks, 5 open questions, conservative 4-phase build sequence |
| 2026-03-28 | S1+S2 | Conference file created. 10 open threads, 8 agreements. Awaiting Eric's decisions. |
| 2026-03-28 | Eric+S2 | All 10 threads resolved. 11 binding decisions. Build sequence finalized. See Decisions section. |
| 2026-03-28 | S1+Eric | D12: Safety is a graph subgraph, not a separate system. Standalone UX, shared backend. 4 questions posed to S2. |
| 2026-03-28 | S2 | D12 response: No challenges. Safety app already structured correctly. One function call addition in Phase 2. |
| 2026-03-28 | S1+Eric | D13: Tool tiers (writers/readers/navigator). SIOP reader-only. DOE primary calibrator. Gage/SIOP need own surfaces (UX, not architecture). |
| 2026-03-28 | S1+Eric | D15: Utilities roadmap. Tier 1-3 quick wins through transformative. Proposed work split: S1 owns graph, S2 owns utilities. |
| 2026-03-28 | S2 | D15 response: Accepted. Resequenced Tier 1. Flagged recipe system as feature not utility. |
| 2026-03-28 | S1 | Two-session execution plan: 4 phases + parallel utilities, 3 review gates, ~11 weeks, communication protocol. |
