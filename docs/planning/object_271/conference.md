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
| 2026-03-29 | S1+S2 | Phases 0-4 shipped. QMS aligned. Control Plans + CommitmentResource + Supplier Claims built. |
| 2026-03-29 | S1+Eric | OLR-001 standard brief: 25 sections, full ISO mapping, paper implementation, detection hierarchy, See-Do-Teach competency. |
| 2026-03-29 | S1 | Backend audit against OLR-001: 40% compliant. 16 gaps identified. Proposals below. |

---

## S1 Backend Audit: SVEND vs OLR-001

**Date:** 2026-03-29
**Methodology:** Full code audit against all 25 OLR-001 sections.
**Result:** ~40% compliant. Strong graph foundation, strong loop mechanics. Critical gaps in classification, metrics, competency, pre-production, configuration.

### Backend Proposals (S1 → S2 for UX review)

#### BP-1: ProcessNode Classification + Detection + Customer-Facing (CRITICAL)

Add three fields to ProcessNode:

```
classification_tier: CharField choices=["critical", "major", "minor"] default="minor"
detection_mechanism_level: IntegerField choices=[1-8] nullable
customer_facing: BooleanField default=False
```

These unlock: evidence minimum enforcement, detection distribution metrics, customer satisfaction ratio, maturity auditing. Without these, nothing in §4.7, §9, §12, §13 is auditable.

**S2 question:** How does this surface in the Process Map UI? Color coding? Badge? Filter lens? Node detail panel needs classification + detection level display.

---

#### BP-2: Knowledge Health Metrics Service (CRITICAL)

New method `GraphService.compute_knowledge_health(tenant_id, graph_id)` returning:

```python
{
    "calibration_rate": 0.65,           # edges with evidence / total
    "staleness_rate": 0.12,             # past threshold / calibrated
    "contradiction_rate": 0.03,         # conflicting evidence / total
    "knowledge_gap_ratio": 0.35,        # assertion-only / total
    "signal_resolution_velocity_days": 4.2,  # avg signal → knowledge update
    "proactive_reactive_ratio": 0.78,   # internal detection / total customer-facing signals
    "detection_distribution": {         # critical nodes by level
        "level_1": 2, "level_2": 5, "level_3": 8,
        "level_4": 12, "level_5": 3, "level_6": 0
    },
    "maturity_indicators": {
        "level_1": true,   # structured knowledge exists
        "level_2": true,   # evidence accumulating
        "level_3": false,  # staleness_rate > threshold
        "level_4": false,  # no predictive validation
    },
}
```

Persist daily as `KnowledgeHealthSnapshot` model. Trend over time = maturity trajectory.

**S2 question:** This needs a dashboard. Where? Loop dashboard KPIs? Process Map sidebar? Dedicated "Knowledge Health" tab? Leadership view?

---

#### BP-3: ControlPlanItem Linkage (HIGH)

Add to ControlPlanItem:

```
detection_mechanism_level: IntegerField choices=[1-8] nullable
fmis_row: ForeignKey(FMISRow, nullable)  # links plan to knowledge structure
competency_stage_required: IntegerField choices=[1,2,3] default=1
```

The control plan IS the knowledge structure filtered to "what to monitor." Every item should trace to an FMIS row and declare what detection mechanism and competency level it requires.

**S2 question:** In the control plan UI, how do we present mechanism level? Dropdown with the 8 descriptions? How does the FMIS row link render — inline reference or click-to-navigate?

---

#### BP-4: Competency See-Do-Teach Model (HIGH)

Extend TrainingRecord or create new `CompetencyRecord`:

```
practice_type: CharField  # "process_confirmation", "fft", "investigation", "doe", "fmis_management"
stage: IntegerField choices=[1,2,3]  # See, Do, Teach
evidence_id: UUIDField nullable  # FK to PC/FFT/Investigation that demonstrated competency
evaluator: ForeignKey(User, nullable)
demonstrated_at: DateTimeField
employee: ForeignKey(Employee)
```

Auto-enrollment: when an operator completes a supervised PC, auto-create CompetencyRecord(stage=2). When someone they trained completes their Stage 2, auto-create CompetencyRecord(stage=3) for the trainer.

**S2 question:** UX for this? A competency card per person showing stages by practice type? A "my competency" section in the user profile? How does a supervisor mark Stage 1 (observation) as complete?

---

#### BP-5: QFD Data Model (MEDIUM)

New models:

```
QFDMatrix: name, version, product/process, linked_graph, created_by
QFDRequirement: matrix, text, classification_tier, importance_weight, linked_spec_node
QFDCharacteristic: matrix, text, linked_process_node
QFDRelationship: requirement, characteristic, strength (strong/medium/weak)
```

QFD matrix seeds FMIS rows: strong relationships become assertions. Classification flows from requirement to characteristic to node.

**S2 question:** House of Quality is a specific visual format. Do we build a dedicated QFD matrix editor, or integrate into the Process Map as a special creation mode?

---

#### BP-6: Equipment Reliability Fields (MEDIUM)

Add to MeasurementEquipment:

```
mtbf_hours: FloatField nullable
weibull_shape: FloatField nullable  # beta parameter
weibull_scale: FloatField nullable  # eta parameter
failure_count: IntegerField default=0
failure_history: JSONField default=list  # [{date, hours_at_failure, description}]
measurement_uncertainty_percent: FloatField nullable  # from Gage R&R
```

Equipment reliability becomes a calibrated edge. Measurement uncertainty weights all evidence gathered through this equipment.

**S2 question:** Equipment detail view needs reliability section. Weibull chart? MTBF trend? How does uncertainty display alongside evidence it affects?

---

#### BP-7: TenantConfig + ConfigService (MEDIUM — enables everything else)

As specified in `enterprise_configuration_spec.md`. One table, one service. Replaces all hardcoded thresholds.

**S2 question:** Configuration panel UX already spec'd in the enterprise config doc. Review and propose implementation approach.

---

#### BP-8: 3P / Pre-Production Investigation Mode (LOW — can defer)

Add `is_pre_production: BooleanField default=False` to Investigation. Pre-production investigations follow the same Loop but are tagged for audit purposes. Moonshining cycles are sequential investigations on the same process before launch.

**S2 question:** Does this need its own UI surface, or just a filter/tag on the existing investigation workspace?

---

### Priority Sequence

| # | Proposal | Blocks | Effort |
|---|----------|--------|--------|
| BP-1 | ProcessNode fields | BP-2, BP-3 | Small (migration + 3 fields) |
| BP-2 | Health metrics | Maturity auditing, leadership dashboard | Medium (service + snapshot model) |
| BP-7 | TenantConfig | Configurable thresholds everywhere | Medium (model + service) |
| BP-3 | ControlPlanItem linkage | Control plan → knowledge trace | Small (migration + 3 fields) |
| BP-4 | Competency See-Do-Teach | Training alignment | Medium (new model + auto-enrollment) |
| BP-5 | QFD model | Pre-production traceability | Medium (new models + matrix editor) |
| BP-6 | Equipment reliability | Predictive capability (Level 4) | Small (migration + fields) |
| BP-8 | 3P investigation mode | Pre-production audit trail | Small (one boolean field) |

**S2:** Review UX questions in each proposal. Respond with implementation approach or counter-proposals.

---

## D16: UX Deep Audit — Front-End Architecture Proposal

**Date:** 2026-03-29
**Author:** S2 (Frontend / UX)
**Full audit:** `docs/planning/object_271/ux_deep_audit.md`

### What We Did Right
- Loop sidebar (IDE activity bar pattern) — correct structure
- List-detail layout (detail left, rail right) — proven with Signals + Commitments + Suppliers
- Svend modal system (.sv-modal) — consistent, professional
- Supplier workbench panels — EHR Storyboard pattern adapted
- Semantic color usage — mostly load-bearing, not decorative

### 12 Proposals (Ranked by Impact)

| # | Feature | Source Paradigm | Effort | Impact |
|---|---------|----------------|--------|--------|
| 1 | **Command palette (Cmd+K)** | Bloomberg GO bar / VS Code | 1 session | CRITICAL |
| 2 | **Keyboard triage** | Linear | 0.5 session | CRITICAL |
| 3 | **Context banner** | EHR patient banner | 0.5 session | HIGH |
| 4 | **Shared activity feed component** | Linear + EHR | 0.5 session | HIGH |
| 5 | **Slash command templates** | EHR SmartPhrases | 1 session | HIGH |
| 6 | **Workflow gate indicators** | Aviation prerequisite enforcement | 0.5 session | MEDIUM |
| 7 | **Configuration panel (Policies)** | Enterprise config spec | 2 sessions | MEDIUM |
| 8 | **Saved filter views** | Linear saved views | 1 session | MEDIUM |
| 9 | **Keyboard triage (all lists)** | Linear | 0.5 session | MEDIUM |
| 10 | **Bulk operations** | Linear multi-select | 1 session | LOW |
| 11 | **Diff/comparison view** | IDE diff | 1 session | LOW |
| 12 | **Graph context mini-panel** | VS Code minimap | 1 session | LOW |

### 7 Things We Should REJECT
1. Drag-and-drop for non-spatial operations
2. Wizard-style multi-step forms
3. Toast on success (only on error/warning)
4. Dashboard-first landing (action queue first instead)
5. Infinite scroll (paginate with counts)
6. Auto-save on regulated records
7. Animated view transitions

### S2 Recommendation
Build #1 (command palette) first. It changes the platform from "website" to "tool" in one feature. Then #2 (keyboard triage) makes the Loop shell feel like a cockpit. Then #3-5 add depth.

### Awaiting
- S1: Backend requirements for command palette search, activity feed events, TenantConfig model
- Eric: Priority confirmation, reject-list arbitration, keyboard shortcut aggressiveness

**Status:** PROPOSED — S1 response below

---

## S1 Response to S2 UX Proposals + Backend Requirements

**Date:** 2026-03-29

### Response to S2's 12 Proposals

**#1 Command Palette (Cmd+K) — STRONG AGREE + backend spec**

Backend provides: `GET /api/search/?q=<query>&types=signal,commitment,claim,supplier,investigation,document,node,edge`

Returns unified results across all entity types. Each result: `{type, id, title, subtitle, url, status, relevance_score}`. Fuzzy matching on title + description. Node/edge search included — "temperature" finds ProcessNode records too. This is where the graph becomes navigable by keyboard.

One endpoint, one query. I'll build it. Performance target: <200ms for 10k total records across types.

**#2 Keyboard Triage — AGREE + OLR-001 connection**

This maps to OLR-001 §7.1 (signal detection). The faster signals get triaged, the faster the Loop turns. Signal resolution velocity (§13.4) is directly improved by keyboard triage. The "auto-advance after action" pattern is particularly important — it turns triage from a task into a flow.

Backend: signal status transitions already exist as POST endpoints. No new backend needed. The keyboard just calls the same APIs faster.

**OLR-001 push:** Add a "triage time" timestamp to Signal — when was this signal first triaged? `Signal.triaged_at = DateTimeField(nullable)`. This feeds the signal resolution velocity metric.

**#3 Context Banner — AGREE**

No backend changes needed. Pure frontend. The data is already in the detail response. Smart proposal.

**#4 Activity Feed — AGREE + backend spec**

Backend provides: `GET /api/loop/<entity_type>/<id>/activity/`

Returns unified activity feed from all sources:
- Status changes (from ChangeLog-style audit trail)
- Notes/comments (CommitmentNote, reviewer_notes, etc.)
- Evidence additions (EdgeEvidence created)
- Resource assignments (CommitmentResource)
- Graph events (edge calibrated, staleness flagged, contradiction detected)

I'll add an `ActivityEvent` model or use a queryset union across existing audit trail sources. The graph events are new — when evidence is added to an edge via a tool integration, that should appear in the activity feed of any related signal/investigation/commitment.

**#5 Slash Command Templates — AGREE + OLR-001 enhancement**

This maps to OLR-001 §14.2 (organizational culture). Templates externalize tribal knowledge about HOW to investigate. The 5-Why template isn't just convenience — it's a structured methodology that produces traceable causal chains.

**OLR-001 push:** Add `/assertion` as a template. When typing in any text field, `/assertion` creates a structured assertion: "I believe [cause] affects [effect] because [observation]. FMIS row: ___." This is the fish market card digitized. It's the cultural inversion in one slash command.

Also: `/pc-checklist` generates a process confirmation checklist skeleton. `/fft-plan` generates a forced failure test plan skeleton. These teach methodology while collecting structured data.

**#6 Workflow Gate Indicators — STRONG AGREE + backend spec**

Backend provides: `GET /api/loop/<entity_type>/<id>/gates/`

Returns: `[{action: "verify", gate: "no_verification_record", met: false, description: "Add verification record before closing"}]`

This is computed from the model's `VALID_TRANSITIONS` + `TRANSITION_REQUIRES` already defined on NCR, Complaint, Claim, etc. I'll extract the gate logic into a reusable function that any model with transitions can expose.

**OLR-001 connection:** This IS §9.3 (minimum detection mechanism level enforcement). If a critical characteristic's control plan item requires Level 4 detection and the current mechanism is Level 5, the gate shows: "Cannot release — detection mechanism below minimum for critical tier."

**#7 Configuration Panel — AGREE, needs BP-7 first**

S2's proposal matches the enterprise_configuration_spec.md exactly. The diff view ("what changes if I switch from ISO to IATF") is particularly good. Backend: TenantConfig model + ConfigService must ship first (BP-7). Then the UI reads/writes through it.

I'll build BP-7 (TenantConfig + ConfigService) as the backend prerequisite.

**#8 Saved Filter Views — AGREE, lightweight**

Backend: `saved_views` JSONField on User model's existing `preferences` field. No new model needed. Format: `{name, entity_type, filters: {status, severity, ...}, sort_by}`. Max 20 per user.

**#9-12 — AGREE on all, defer per S2's priority sequence**

---

### S1's Additional UX Proposals (Backend-Driven)

These come from the OLR-001 audit. S2 didn't see them because they're backend-initiated UI needs.

#### UX-A: Knowledge Health Dashboard

BP-2 computes health metrics. They need to surface somewhere. Proposal:

**Option 1:** Replace the CI Readiness Score in the Loop sidebar header with Knowledge Health. The readiness score currently shows "—" or a number. Replace with a multi-metric mini display:

```
┌─────────────────────────────┐
│ LOOP              KH: 67/100│
│                   ▲ +3 /30d │
└─────────────────────────────┘
```

KH = Knowledge Health score (composite of the 7 metrics). The arrow + delta shows 30-day trend. Click opens full dashboard.

**Option 2:** Dedicated "Health" section in the Loop sidebar below Verify:

```
HEALTH
  Knowledge    67%
  Maturity     Level 2
  Gaps         23
```

S2: which pattern?

---

#### UX-B: Node Classification in Process Map

BP-1 adds classification_tier, detection_mechanism_level, customer_facing to ProcessNode. These need to render in the graph navigator.

Proposal:
- **Classification:** Node border thickness. Critical = thick (3px), Major = medium (2px), Minor = thin (1px). Same color scheme otherwise.
- **Customer-facing:** Small star icon on the node.
- **Detection mechanism level:** Badge on the edge (number 1-8) when hovering or in detail panel. Not always visible — clutters the graph.
- **New lens:** "Customer View" — filters to customer_facing nodes + their upstream edges. Shows which parts of the process affect what customers see.

S2: better approach?

---

#### UX-C: Competency Visibility

BP-4 adds See-Do-Teach competency records. Where does this surface?

Proposal:
- **Employee profile** gets a "Competency" tab showing a grid: rows = practice types, columns = Stage 1/2/3, cells = date + evidence link or empty.
- **Control plan item detail** shows: "Required: Stage 2. Assigned operator: Stage 2 complete (2026-03-15, PC #47)." Green if met, amber if Stage 1 only, red if no record.
- **Loop sidebar** gets "Team" section showing competency coverage: "4/6 operators Stage 2+ on process confirmation."

S2: is the grid too complex? Simpler approach?

---

#### UX-D: Detection Ladder Visualization

The 8-level detection hierarchy is new and unique. It needs to be visible and understandable.

Proposal: In the control plan view and FMIS view, show the detection level as a vertical thermometer or stepped bar:

```
1 ████████ Source Prevention
2 ███████▒ Auto Arrest
3 ██████▒▒ Auto Detect
4 █████▒▒▒ Auto Alert        ← Current: Level 4
5 ████▒▒▒▒ Structured Check
6 ███▒▒▒▒▒ Observation
7 ██▒▒▒▒▒▒ Downstream
8 █▒▒▒▒▒▒▒ Undetectable
```

The red line at Level 4 shows the minimum for this tier (critical). User sees immediately: "we're at the minimum — investment opportunity to move up."

S2: visual approach? Or just a dropdown + number in the detail panel?

---

### Answers to S2's Backend Questions

**Q1: Can command palette search across all entities with one query?**

Yes. I'll build `GET /api/search/?q=<query>` that unions across: Signal, Commitment, SupplierClaim, SupplierRecord, Investigation, ControlledDocument, ProcessNode, ProcessEdge. Returns top 10 ranked by relevance. Each result has `{type, id, title, subtitle, url}`. One query, <200ms.

**Q2: Does TenantConfig support the full enterprise spec?**

It will when BP-7 ships. One model: `(tenant, domain, key, value, site)`. Supports all 70+ settings. ConfigService.get(tenant_id, key) with site fallback.

**Q3: Event sourcing for activity feed?**

Mixed. Some entities have explicit history models (NCRStatusChange, DocumentStatusChange, SupplierStatusChange, TrainingRecordChange, QMSFieldChange). Others only have `updated_at` timestamps with no history. I'll need to add an `ActivityLog` model or a generic event table for entities that lack history. The graph has EdgeEvidence timestamps which serve as activity events.

**Q4: Saved filter views — User model or own model?**

User model `preferences` JSON is fine for v1. Cap at 20 views per user. If we need shared team views later, promote to own model.

---

### Combined Priority (Backend + Frontend)

| # | What | Owner | Blocks |
|---|------|-------|--------|
| 1 | BP-1: ProcessNode fields | S1 | Everything metrics-related |
| 2 | BP-7: TenantConfig + ConfigService | S1 | Configuration panel (#7) |
| 3 | S2-#1: Command palette | S2 | — (parallel) |
| 4 | BP-2: Knowledge health metrics | S1 | Health dashboard |
| 5 | S2-#2: Keyboard triage | S2 | — (parallel) |
| 6 | S2-#3: Context banner | S2 | — (parallel) |
| 7 | BP-3: ControlPlanItem linkage | S1 | Control plan depth |
| 8 | S2-#4: Activity feed | S2 | Needs activity endpoint from S1 |
| 9 | S2-#5: Slash commands | S2 | — (parallel) |
| 10 | BP-4: Competency model | S1 | Competency visibility |
| 11 | S2-#6: Workflow gates | S2 | Needs gates endpoint from S1 |
| 12 | S2-#7: Configuration panel | S2 | Needs BP-7 |
| 13 | BP-5: QFD model | S1 | Pre-production |
| 14 | BP-6: Equipment reliability | S1 | Level 4 maturity |

S1 and S2 can run in parallel on items 1-6. Items 8, 11, 12 have cross-dependencies.

**Status:** S1 RESPONSE COMPLETE — S2 response below

---

## S2 Response to S1 Backend Proposals + UX-A through UX-D

### On S1's Backend Specs for My Proposals

**Search endpoint for command palette:** Perfect. `GET /api/search/?q=<query>&types=...` returning `{type, id, title, subtitle, url, status}` is exactly what I need. The fuzzy matching on title + description covers the Bloomberg mnemonic pattern. Including ProcessNode/ProcessEdge in search results is smart — typing "temperature" to find a graph node makes the graph keyboard-navigable.

One addition: include `relevance_score` in results AND sort by it. The command palette should show the best match first, not alphabetical. Recent items should rank higher (user typed this ID yesterday → it's probably what they want again).

**Signal.triaged_at:** Agree. Simple timestamp field that enables the resolution velocity metric. I'll display it in the signal detail view and the context banner.

**Activity feed endpoint:** The union-across-sources approach is the right call. I don't need a single unified `ActivityLog` model — I need a single endpoint that queries the existing audit trail sources and returns them in a unified shape. S1's list (status changes, notes, evidence, resources, graph events) covers everything.

**One concern:** If the endpoint does N queries (one per source type) and unions them, it could be slow for entities with heavy history. Suggest: cap at 50 most recent events per entity, and lazy-load older ones with "Load more" at the bottom of the feed. Don't paginate — just truncate.

**Gates endpoint:** Yes. `GET /api/loop/<entity_type>/<id>/gates/` returning `[{action, gate, met, description}]` is clean. I'll render unmet gates as disabled buttons with the description as tooltip. Met gates are enabled buttons. This is the aviation prerequisite pattern.

**Saved views on User.preferences:** Fine for v1. The 20-view cap is reasonable. I'll store `{name, entity_type, section, filters, sort_by}` and render them as sub-items in the Loop sidebar under the relevant section.

---

### Response to UX-A: Knowledge Health Dashboard

**Option 2 is better** (dedicated sidebar section), but modified.

Don't add a new "HEALTH" section to the sidebar — the sidebar already has 4 stages + cross-cutting links + saved views will add more. Instead:

**Replace the readiness badge in the sidebar header** with a richer knowledge health indicator, AND make it clickable to expand an inline dashboard:

```
┌─────────────────────────────┐
│ LOOP              [KH: 67] │  ← click to expand
│                             │
│ ┌─ Knowledge Health ──────┐ │  ← expanded panel
│ │ Calibrated:   43%  ▲+5  │ │
│ │ Stale:        12%  ▼-3  │ │
│ │ Contradicted:  2   ——   │ │
│ │ Gaps:         23   ▼-4  │ │
│ │ Maturity:     L2   ——   │ │
│ │ [Full Dashboard →]      │ │
│ └─────────────────────────┘ │
│                             │
│ DETECT                      │
│   Signals ...               │
```

The expanded panel is a collapsible section at the top of the sidebar, below the header. Always accessible, doesn't take permanent space. The "Full Dashboard" link opens a dedicated health page (Pattern B workspace).

This preserves the sidebar for navigation while giving knowledge health maximum visibility.

---

### Response to UX-B: Node Classification in Process Map

**Border thickness for tier: Agree.** It's immediately visible without adding visual noise. Critical nodes are obviously heavier. This is the SCADA paradigm — visual weight carries meaning.

**Customer-facing star icon: Agree.** Small, unobtrusive, meaningful. The "Customer View" lens filtering to these nodes is excellent — it's the QFD view on the graph.

**Detection level badge: Disagree with the approach.** A number (1-8) on an edge means nothing to most users. Instead:

**Use the edge dash pattern:**
- Levels 1-3 (automated): solid line, full color
- Levels 4-5 (structured human): solid line, muted color
- Levels 6-7 (unstructured/downstream): dashed line
- Level 8 (undetectable): dotted line, red

The visual hierarchy communicates the detection quality WITHOUT a number badge. Users learn: "solid = reliable detection, dashed = human-dependent, dotted = undetectable." The exact level number appears in the edge detail panel on click.

This is the SCADA approach — encode meaning in visual properties, save text for the detail view.

---

### Response to UX-C: Competency Visibility

**The grid is the right model but wrong presentation.**

A grid with rows=practices, columns=stages, cells=date works for a training administrator. But most users of the Loop system aren't training admins. They need to see: "can this person do this work?"

**Proposal:** Show competency as a simple status indicator wherever people are assigned:

**In commitment detail:**
```
Owner: Jane Smith [Stage 2 ✓] — PC qualified
```

**In process confirmation list:**
```
Observer: Mike Chen [Stage 1 ⚠] — needs supervised observation
```

The full grid lives in the employee profile (the Hoshin resource system already has employee detail pages). The Loop system shows the relevant competency for the current context — just like the EHR patient banner shows allergies and code status, not the full medical history.

**Control plan item:** S1's proposal is correct — "Required: Stage 2. Assigned: Stage 2 complete." Green/amber/red. That's the gate pattern applied to competency.

**Sidebar team coverage:** Defer. It's useful for managers but adds complexity to the sidebar. Put it on the knowledge health dashboard instead.

---

### Response to UX-D: Detection Ladder Visualization

**The stepped bar is right for the FMIS detail view.** It's compact, immediately readable, and shows both current level and minimum required. The red line at the tier minimum is a gate indicator — "you're at the boundary."

**But don't show it everywhere.** In list views (FMIS table, control plan table), show just the level number with color: green if above minimum, amber if at minimum, red if below. The stepped bar is for the detail/edit view.

**Add directional indicator:** If the detection level changed since last review (e.g., moved from Level 5 to Level 4 because SPC alarm was added), show an arrow: `4 ▲`. This visualizes the investment direction that OLR-001 §9.5 requires tracking.

---

### S2 Additional UX Proposals (from OLR-001 implications)

#### UX-E: Triage Mode vs Work Mode

The Loop shell needs two modes of interaction, not just one:

**Triage mode** (processing a queue): fast keyboard-driven scanning. Auto-advance. Minimal detail. This is the morning ritual — "what happened overnight?"

**Work mode** (deep focus on one item): full detail, all panels expanded, notes, evidence, linked items. This is the investigation — "I'm spending an hour on this."

Currently we have one mode that tries to do both. The list-detail layout works for triage. But for work mode, the right rail is wasted space — the user is focused on one item.

**Proposal:** Add a "Focus" button on any detail view that:
1. Hides the right rail
2. Expands the detail panel to full width
3. Shows the context banner (since the rail with the list is gone, you need the banner to know which item you're in)
4. Adds a keyboard shortcut to return to the list (Escape)

This is the Linear full-page vs sidebar-panel toggle.

#### UX-F: The "What Changed" View

When a quality manager opens the Loop shell in the morning, they need: "what changed since I last looked?" Not the full signal list — just the delta.

**Proposal:** Add an "Activity" section at the top of the Loop sidebar (above Detect) that shows a reverse-chronological feed of all Loop events across all sections, filtered to "since last visit." Badge count shows new events.

This is the Linear Inbox pattern. It replaces the "you have to check every section" workflow with "here's everything that happened."

---

### Combined Priority (Updated after S2 Response)

| # | What | Owner | Parallel? |
|---|------|-------|-----------|
| 1 | BP-1: ProcessNode fields | S1 | Yes — S2 builds command palette simultaneously |
| 2 | Search endpoint (`/api/search/`) | S1 | Prerequisite for #3 |
| 3 | Command palette (Cmd+K) | S2 | After search endpoint |
| 4 | BP-7: TenantConfig + ConfigService | S1 | Yes — S2 builds keyboard triage simultaneously |
| 5 | Keyboard triage (signals) | S2 | Parallel with BP-7 |
| 6 | Context banner | S2 | Independent |
| 7 | Focus mode (triage/work toggle) | S2 | Independent |
| 8 | Knowledge health sidebar widget | S2 | After BP-2 |
| 9 | Activity feed endpoint | S1 | Prerequisite for #10 |
| 10 | Activity feed component | S2 | After endpoint |
| 11 | Gates endpoint | S1 | Prerequisite for #12 |
| 12 | Workflow gate indicators | S2 | After endpoint |
| 13 | Slash commands | S2 | Independent |
| 14 | Configuration panel | S2 | After BP-7 |

**Critical path:** Search endpoint (S1) → Command palette (S2). Everything else is parallel.

**Status:** S2 RESPONSE COMPLETE — sprint plan below

---

## QMS Closure Sprint — Three Session Plan

**Date:** 2026-03-29 evening
**Goal:** Every QMS surface functional and connected. Knowledge health visible. Configuration adjustable. Document builder service established. Loop shell fully wired.

### Session Boundaries — NO OVERLAP

| File/Directory | S1 ONLY | S2 ONLY | S3 ONLY |
|---------------|---------|---------|---------|
| `graph/` | ALL files | — | — |
| `loop/models.py` | YES (model fields only) | — | — |
| `loop/views.py` | YES (new endpoints only, append to end) | — | — |
| `loop/urls.py` | YES (new routes only, append to end) | — | — |
| `templates/loop_*.html` | — | ALL loop templates | — |
| `templates/base_loop.html` | — | YES | — |
| `templates/base_app.html` | — | YES | — |
| `templates/graph_map.html` | — | YES (Cytoscape rendering) | — |
| `agents_api/models.py` | YES (model fields only) | — | — |
| `agents_api/iso_views.py` | — | — | — (neither) |
| `svend/urls.py` | — | YES (template routes) | YES (document service routes only) |
| `documents/` (NEW app) | — | — | ALL files |
| `templates/iso.html` | — | YES | — |

**CRITICAL RULE:** If you need to touch a file owned by another session, write a spec in `object_271/` and let the owner implement it. Do NOT edit shared files.

---

### S1 — Backend Closure (this session)

**Scope:** Models, services, API endpoints. No templates. No frontend JS.

**Sequence:**

| # | Task | Files | Est | Depends on |
|---|------|-------|-----|------------|
| S1-1 | BP-1: Add `classification_tier`, `detection_mechanism_level`, `customer_facing` to ProcessNode | `graph/models.py`, migration | 30m | — |
| S1-2 | BP-7: TenantConfig model + ConfigService + presets (ISO 9001, IATF, AS9100D, Lightweight) | `graph/` or new `config/` app | 2h | — |
| S1-3 | Search endpoint: `GET /api/search/?q=` across all entity types | `graph/views.py`, `graph/urls.py` | 1h | — |
| S1-4 | BP-2: `GraphService.compute_knowledge_health()` + `KnowledgeHealthSnapshot` model | `graph/service.py`, `graph/models.py`, migration | 2h | S1-1 |
| S1-5 | BP-3: Add `detection_mechanism_level`, `fmis_row` FK, `competency_stage_required` to ControlPlanItem | `agents_api/models.py`, migration | 30m | S1-1 |
| S1-6 | Activity feed endpoint: `GET /api/loop/<type>/<id>/activity/` | `loop/views.py`, `loop/urls.py` | 1h | — |
| S1-7 | Gates endpoint: `GET /api/loop/<type>/<id>/gates/` | `loop/views.py`, `loop/urls.py` | 1h | — |
| S1-8 | Add `Signal.triaged_at` timestamp | `loop/models.py`, migration | 15m | — |
| S1-9 | BP-6: Equipment reliability fields (MTBF, Weibull, measurement_uncertainty) | `agents_api/models.py`, migration | 30m | — |
| S1-10 | Tests for all new endpoints and services | `graph/tests_*.py` | 1h | S1-1 through S1-9 |

**Total estimate:** ~10 hours. Prioritize S1-1 → S1-3 → S1-4 → S1-6 → S1-7 (critical path for S2).

**DO NOT TOUCH:**
- Any template file
- `base_app.html` or `base_loop.html`
- Any `_serialize_*` function in views (S2 may be modifying frontend expectations)
- `svend/urls.py` template routes
- `iso.html`

---

### S2 — Frontend Closure

**Scope:** Templates, CSS, JavaScript. No model changes. No migrations. API calls only to endpoints S1 builds.

**Sequence:**

| # | Task | Files | Est | Depends on |
|---|------|-------|-----|------------|
| S2-1 | Command palette (Cmd+K) in `base_app.html` | `base_app.html` | 2h | S1-3 (search endpoint) |
| S2-2 | Keyboard triage on signals | `loop_detect_signals.html` | 1h | — |
| S2-3 | Context banner component in `base_loop.html` | `base_loop.html` | 1h | — |
| S2-4 | Focus mode toggle (triage/work) | `base_loop.html` | 1h | — |
| S2-5 | Wire Loop shell placeholders to real content: Investigate (active + concluded), Verify (PC, FFT, Audits, Reviews) | `loop_placeholder.html` → real templates or API calls | 3h | — |
| S2-6 | Node classification rendering in Process Map (border thickness, star icon, dash patterns for detection) | `graph_map.html` | 1h | S1-1 (fields exist) |
| S2-7 | Knowledge health sidebar widget (collapsible panel under Loop header) | `base_loop.html` | 1h | S1-4 (health endpoint) |
| S2-8 | Activity feed component (shared `.activity-feed` in `base_loop.html`) | `base_loop.html` + detail templates | 1h | S1-6 (activity endpoint) |
| S2-9 | Workflow gate indicators on action buttons | Detail templates | 1h | S1-7 (gates endpoint) |
| S2-10 | Slash command templates (/5why, /fishbone, /assertion, /containment) | `base_loop.html` (textarea handler) | 1h | — |

**Total estimate:** ~14 hours. Prioritize S2-2 → S2-3 → S2-4 → S2-5 (immediate UX improvements that don't depend on S1).

**S2 is GATED until S1 completes Phase 1.** S1 pushes all backend changes first, notes "endpoints live" in sprint log. Then S2 starts against a stable backend. No moving target, no merge conflicts.

S3 runs in parallel with S1 from the start (zero file overlap).

**DO NOT TOUCH:**
- Any Python file (models, views, urls, services)
- `graph/` app files
- `loop/models.py`, `loop/views.py`, `loop/urls.py`
- Any migration file
- `agents_api/models.py`

---

### S3 — Document Builder Service (new session)

**Scope:** New `documents/` Django app. Service that produces formatted documents from structured data. Consolidates existing WeasyPrint usage into one service.

**Sequence:**

| # | Task | Files | Est | Depends on |
|---|------|-------|-----|------------|
| S3-1 | Create `documents/` app: `__init__.py`, `apps.py`, register in settings | `documents/`, `svend/settings.py` (one line) | 15m | — |
| S3-2 | `DocumentService` class: `render(template_name, context, output_format)` → PDF/DOCX/HTML | `documents/service.py` | 2h | — |
| S3-3 | Template registry: define available templates (CAPA, 8D, A3, control_plan, claim_report, investigation_summary, compliance_report) | `documents/templates.py` | 1h | — |
| S3-4 | PDF templates using WeasyPrint (consolidate from existing `a3_views.py`, `report_views.py`, `iso_doc_views.py`) | `documents/pdf_templates/` | 2h | — |
| S3-5 | DOCX output via python-docx (consolidate from `iso_doc_views.py`) | `documents/docx_renderer.py` | 1h | — |
| S3-6 | API endpoint: `POST /api/documents/render/` accepting `{template, context, format}` | `documents/views.py`, `documents/urls.py` | 1h | S3-2 |
| S3-7 | Branding integration: read tenant logo/company_name from TenantConfig (or Tenant.settings) for headers/footers | `documents/service.py` | 30m | — |
| S3-8 | Tests: render each template in each format, verify output | `documents/tests.py` | 1h | S3-2 through S3-6 |

**Total estimate:** ~9 hours.

**DO NOT TOUCH:**
- `loop/` anything
- `graph/` anything
- `agents_api/models.py`
- `templates/loop_*.html` or `base_loop.html` or `base_app.html`
- Any existing view function in `a3_views.py`, `report_views.py`, `iso_doc_views.py` — read them for reference, don't modify. The existing endpoints continue working. New endpoints route through DocumentService.

**S3's relationship to existing code:** READ ONLY on existing report views. The document service is a NEW path. Existing report generation (A3 PDF, ISO doc DOCX) continues to work as-is. Once DocumentService is proven, a future task migrates existing report views to use it. That migration is NOT in this sprint.

---

### Handoff Points

| From | To | What | When |
|------|----|------|------|
| S1-3 | S2-1 | Search endpoint live | S1 completes S1-3 → S2 starts command palette |
| S1-1 | S2-6 | ProcessNode fields in DB | S1 completes S1-1 → S2 can render classification/detection |
| S1-4 | S2-7 | Health metrics endpoint | S1 completes S1-4 → S2 builds sidebar widget |
| S1-6 | S2-8 | Activity feed endpoint | S1 completes S1-6 → S2 builds feed component |
| S1-7 | S2-9 | Gates endpoint | S1 completes S1-7 → S2 builds gate indicators |

**Communication:** When S1 completes a handoff item, note it in conference.md under the sprint log. S2 checks before starting dependent work.

---

### S2 Review of Three-Session Plan

**File ownership:** Clean. Zero overlap between S1 (Python/models/views), S2 (templates/CSS/JS), S3 (new documents/ app). The "write a spec, let the owner implement" rule for cross-session needs is correct and necessary.

**My task list (S2-1 through S2-10):** Agree with the sequence and estimates. Two adjustments:

**Adjustment 1: S2-5 is underscoped at 3h.** "Wire placeholders to real content" for Investigate (active + concluded), Verify (PC, FFT, Audits, Reviews) means building 6+ functional list-detail views, each with its own API calls, detail rendering, and action buttons. The signals and commitments templates took ~1h each. Six sections × 1h = 6h minimum. I'd split this:

- S2-5a: Investigate Active + Concluded (2h) — investigation list with status, link to workspace
- S2-5b: Verify PC + FFT (2h) — PC list with diagnosis badges, FFT list with detection results
- S2-5c: Verify Audits + Reviews (2h) — audit schedule, management review summary

If we're time-constrained, S2-5a is highest priority (investigations are the core Loop artifact).

**Adjustment 2: Gate order.** The plan says "S2 is GATED until S1 completes Phase 1." But S2-2 (keyboard triage), S2-3 (context banner), S2-4 (focus mode), S2-5 (placeholder wiring), and S2-10 (slash commands) have ZERO dependency on S1. They use existing API endpoints.

**Proposed execution order:**
1. S2-2: Keyboard triage (no dependency)
2. S2-3: Context banner (no dependency)
3. S2-4: Focus mode (no dependency)
4. S2-10: Slash commands (no dependency)
5. S2-5a/b/c: Placeholder wiring (no dependency — uses existing loop/ API)
6. **GATE: wait for S1-3** → S2-1: Command palette
7. **GATE: wait for S1-1** → S2-6: Node classification rendering
8. **GATE: wait for S1-4** → S2-7: Knowledge health widget
9. **GATE: wait for S1-6** → S2-8: Activity feed
10. **GATE: wait for S1-7** → S2-9: Workflow gates

This means I start immediately on items 1-5 while S1 builds backend. No idle time. Items 6-10 fire as S1's handoffs arrive.

**S3 review:** Document builder service is correctly isolated. The "READ ONLY on existing report views" rule is important — I built the branding integration into `a3_views.py` and `iso_doc_views.py` directly. S3 should not modify those files. When DocumentService is proven, a separate task migrates the existing views to use it. That's not in this sprint. Agree.

**One concern with S3:** `svend/urls.py` is listed as shared between S2 (template routes) and S3 (document service routes). This IS a collision risk. Suggest: S3 creates `documents/urls.py` and S1 adds the `include()` in `svend/urls.py` since S1 owns Python files. S2 doesn't touch `svend/urls.py` during this sprint — all my template routes are already wired.

**Definition of Done:** Agree with all 8 criteria. The "no placeholders" requirement (item 1) is the most ambitious — that's S2-5 covering 6 remaining sections. Achievable but tight.

**Status:** S2 READY. Awaiting Eric's go.

### Sprint Log

| Time | Session | Item | Status |
|------|---------|------|--------|
| 2026-03-29 16:13 | S1 | S1-1 through S1-9: all backend endpoints | DONE (9aa0cf2) |
| 2026-03-29 20:38 | S2 | S2-1: Command palette (Cmd+K) | DONE (37e9e1f) |
| 2026-03-29 20:38 | S2 | S2-2: Keyboard triage (signals) | DONE (37e9e1f) |
| 2026-03-29 20:38 | S2 | S2-3: Context banner | DONE (37e9e1f) |
| 2026-03-29 20:38 | S2 | S2-4: Focus mode toggle | DONE (37e9e1f) |
| 2026-03-29 20:38 | S2 | S2-10: Slash commands | DONE (37e9e1f) |
| — | S2 | S2-5: Wire placeholder sections | IN PROGRESS |

### Definition of Done

**QMS is "closed" when:**
1. Every Loop sidebar link goes to a functional surface (no placeholders)
2. ProcessNode has classification/detection/customer_facing fields
3. Knowledge health is computable and visible
4. TenantConfig exists with at least ISO 9001 preset
5. Command palette works across all entity types
6. Document builder service can render at least 3 template types
7. All new code has tests
8. Everything committed and pushed to main
