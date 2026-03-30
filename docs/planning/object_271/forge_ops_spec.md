# Forge Operations Spec — Calculator Modularization + Cross-Tool Integration

**Date:** 2026-03-30
**From:** Systems Engineer (S2)
**To:** PM (S1)
**Status:** SPEC — ready for implementation

---

## Context

The operations workbench (calculators.html, 8,199 lines) has ~40 calculators across 8 categories, backed by 18 JS files. Most do computation client-side in `svend-math.js` (305 lines) and the `calc-*.js` modules. The migration moves computation server-side to forge packages, leaving JS as thin UI.

This spec covers: (1) forge additions needed, (2) cross-tool integration improvements, (3) optimization opportunities across QMS/Hoshin/VSM/calculators.

---

## Part 1: Forge Package Additions

### forgesiop — new `production` and `lean` submodules

These are the core manufacturing calculations that `svend-math.js` and `calc-production.js` currently do client-side. Every manufacturing user needs these. They should be in forgesiop because they're operational, not statistical.

```
forgesiop/
├── production/
│   ├── takt.py
│   │   - takt_time(available_time, demand) → float
│   │   - cycle_time_analysis(steps: list[dict]) → dict  # bottleneck, balance %, idle time
│   │   - line_balance(steps, takt, operators) → dict  # yamazumi data, reassignment proposals
│   │
│   ├── oee.py
│   │   - oee(planned_min, downtime_min, ideal_cycle_min, produced, defects) → dict
│   │     # Returns: availability, performance, quality, oee, losses breakdown
│   │   - oee_waterfall(oee_result) → ForgeViz ChartSpec  # direct chart output
│   │   - teep(oee, loading_pct) → float  # total effective equipment performance
│   │
│   ├── kanban.py
│   │   - kanban_quantity(daily_demand, lead_time_days, safety_factor, container_size) → dict
│   │   - epei(changeover_time, available_time, num_products) → float
│   │   - pitch(takt_time, pack_size) → float
│   │
│   ├── flow.py
│   │   - littles_law(solve_for, **params) → dict  # any 2 of WIP/throughput/lead_time → 3rd
│   │   - throughput(completed, elapsed_seconds) → float
│   │   - bottleneck_analysis(stations: list[dict]) → dict  # identify constraint, utilizations
│   │   - value_ratio(value_add_time, total_lead_time) → float
│   │
│   └── cost.py
│       - cost_of_quality(prevention, appraisal, internal_failure, external_failure) → dict
│       - cost_accounting(params) → dict  # unit cost breakdown
│       - payback_period(investment, annual_savings) → float
│
├── lean/
│   ├── smed.py
│   │   - classify_elements(elements: list[dict]) → dict  # internal/external split
│   │   - smed_reduction(current_elements, proposed_elements) → dict  # time savings, % reduction
│   │
│   ├── changeover.py
│   │   - changeover_matrix(products, times_matrix) → dict  # optimal sequence
│   │   - total_changeover_time(sequence, matrix) → float
│   │
│   └── family.py
│       - product_family_analysis(products, routings) → dict  # PFA groupings
│       - workload_family_analysis(products, workloads) → dict  # WFA groupings
```

### forgequeue — new package

Queueing theory is used across calculators, capacity planning, and staffing. Currently all in `calc-queue.js` client-side. This should be its own package because it's a complete mathematical domain.

```
forgequeue/
├── single.py
│   - mm1(arrival_rate, service_rate) → dict  # Lq, Wq, L, W, utilization, P0
│   - md1(arrival_rate, service_rate) → dict  # deterministic service
│   - mg1(arrival_rate, service_rate, cv_service) → dict  # general service
│
├── multi.py
│   - mmc(arrival_rate, service_rate, servers) → dict  # multi-server
│   - mmck(arrival_rate, service_rate, servers, capacity) → dict  # finite buffer
│   - erlang_b(traffic, servers) → float  # blocking probability
│   - erlang_c(traffic, servers) → float  # waiting probability
│
├── priority.py
│   - priority_queue(classes: list[dict]) → dict  # per-class metrics with preemption
│
├── network.py
│   - tandem(stages: list[dict]) → dict  # multi-stage sequential
│   - jackson_network(nodes: list[dict], routing_matrix) → dict  # open network
│
└── staffing.py
    - optimal_servers(arrival_rate, service_rate, target_wait, target_prob) → int
    - staffing_cost(servers, wage_rate, arrival_rate, service_rate, wait_cost) → dict
    - staffing_table(arrival_rate, service_rate, min_servers, max_servers) → list[dict]
```

### forgestat additions

```
forgestat/
├── quality/
│   └── desirability.py
│       - derringer_suich(responses: list[dict]) → dict
│         # Each response: {value, target, lower, upper, weight, importance}
│         # Returns: individual desirabilities, composite D, optimal point
│
└── core/
    └── sampling.py
        - sample_normal(mean, std, n, seed=None) → list[float]
        - sample_exponential(mean, n, seed=None) → list[float]
        - sample_weibull(shape, scale, n, seed=None) → list[float]
        - sample_uniform(low, high, n, seed=None) → list[float]
        - sample_poisson(lam, n, seed=None) → list[int]
        - seeded_rng(seed) → callable  # reproducible random for simulations
```

---

## Part 2: Cross-Tool Integration Improvements

### Current state (what exists)

```
VSM ──imports──→ Calculators (calc-vsm.js: 16 loaders, 10 exporters)
VSM ──hypotheses──→ Synara (kaizen bursts linked to Bayesian hypotheses)
Hoshin ──proposals──→ VSM (generate CI projects from VSM waste analysis)
Safety ──FMEA bridge──→ FMEA (Frontier Card findings create FMEA rows)
Safety ──AFEs──→ Hoshin (safety capital requests route through Hoshin projects)
RCA ──link──→ A3 (investigation results attach to A3 report)
```

### What's missing — the graph is the integration hub

Right now these are point-to-point integrations. Each tool talks to 1-2 other tools via custom bridge code. The graph (GRAPH-001) should be the universal connector. Every tool reads from and writes to the graph. The point-to-point bridges become graph reads.

**Current:** VSM → calc-vsm.js → Calculator (custom bridge per calculator)
**Proposed:** VSM writes process steps as graph nodes → Calculator reads graph nodes → results write back as edge evidence

**Current:** Safety finding → custom FMEA bridge → FMEA row
**Proposed:** Safety finding → Signal → Investigation → graph evidence → FMIS row (graph view)

**Current:** RCA → custom A3 link → A3 report
**Proposed:** RCA investigation → graph writeback → ForgeDoc A3 generated from investigation data (already built)

### Specific improvements

#### 1. VSM → Graph → Calculators (replaces calc-vsm.js)

VSM already stores process steps with cycle time, changeover time, uptime, WIP, scrap rate. These ARE graph nodes. When a VSM is saved, the process steps should seed/update ProcessNodes in the graph. Then any calculator that needs process data reads from the graph, not from a VSM-specific API.

**What PM builds:**
- `GraphService.seed_from_vsm(vsm_id)` — create ProcessNodes from VSM steps, ProcessEdges for flow sequence
- Each node gets: cycle_time, changeover_time, uptime, scrap_rate as node properties
- Calculator API endpoints accept `node_id` param — read process data from graph instead of requiring manual input
- `calc-vsm.js` becomes a thin wrapper that calls `GraphService.seed_from_vsm()` then redirects to calculator with `node_id`

**Impact:** Eliminates 16 per-calculator VSM import loaders. Any new calculator automatically has VSM data because it reads from the graph.

#### 2. Hoshin → Graph → VSM (replaces proposal generation bridge)

Hoshin currently calls `/api/vsm/{id}/generate-proposals/` to create CI project proposals from VSM waste analysis. This is a direct bridge. With the graph:

- VSM waste (NVA time, excessive WIP, long changeovers) are graph gap nodes
- Hoshin reads gap nodes from graph, proposes projects to close them
- Project completion updates graph edges with evidence

**What PM builds:**
- `GraphService.gap_report(filter=waste_categories)` — already exists, extend with VSM-specific waste types
- Hoshin proposal generation reads from gap report instead of calling VSM API directly
- When Hoshin project completes, `GraphService.add_evidence()` updates the relevant edges

#### 3. Calculator results → Graph evidence (new)

Right now calculator results vanish when you close the page. If you run an OEE calculation and get 72%, that number isn't stored anywhere useful. With the graph:

- OEE result → `GraphService.add_evidence(equipment_node, evidence_type='oee', value=0.72)`
- Capability study → `GraphService.add_evidence(ctq_node, evidence_type='capability', cpk=1.45)`
- Line balance → updates cycle_time distributions on process step nodes
- Queue analysis → updates service rate / arrival rate on graph edges

**What PM builds:**
- "Save to Graph" action on calculator results (optional, user-initiated)
- `GraphService.add_calculator_evidence(node_id, calculator_type, result_dict)` convenience method
- ForgeViz sparkline on graph nodes showing historical calculator results

#### 4. FMEA ↔ Graph (bidirectional, replaces FMIS seeding)

FMEA rows should BE graph relationships. `seed_from_fmis()` already exists. The reverse should too:

- New graph edge (causal relationship) → proposes FMIS row
- FMIS row updated (S/O/D posteriors from FFT) → updates graph edge posterior
- FMEA calculator results (RPN, criticality) → edge metadata

**Already mostly built.** Just needs the reverse direction: graph edge creation proposes FMIS rows.

#### 5. Safety → Loop → Graph (replaces FMEA bridge)

The Safety app currently bridges Frontier Card findings directly to FMEA rows. Under OLR-001:

- Frontier Card finding → Signal (source: gemba)
- Signal triage → Investigation
- Investigation concludes → graph writeback
- Graph evidence → FMIS row updates (S/O/D posteriors)

**The Safety → FMEA bridge becomes Signal → Loop → Graph → FMIS.** No custom bridge code. The Loop handles it.

**What PM builds:**
- Safety `create_signal_from_card(card_id)` — creates Loop Signal with frontier card data
- Remove direct FMEA row creation from safety app (it goes through the Loop now)

---

## Part 3: Optimization Opportunities

### A. Unified simulation engine

Three simulation JS files (`calc-sim-line.js`, `calc-sim-flow.js`, `calc-sim-quality.js`) share patterns: discrete event loop, Monte Carlo sampling, real-time animation, statistics collection. They should share a simulation kernel.

**Proposal:** `forgesiop.simulation.engine` — a discrete event simulation engine in Python that:
- Defines a process network from graph nodes/edges
- Runs Monte Carlo with configurable distributions per node
- Returns time-series results as ForgeViz ChartSpecs
- JS handles animation only (reads sim frames from API via SSE or polling)

This replaces `svend-sim-core.js` (249 lines) and the three sim files with a server-side engine that's testable, calibratable, and uses real process data from the graph.

```
forgesiop/simulation/
├── engine.py
│   - SimNetwork(nodes, edges, config) → simulator
│   - simulator.run(duration, replications) → SimResult
│   - SimResult.summary() → dict (throughput, WIP, lead_time, utilization per station)
│   - SimResult.timeseries() → list[dict] (per-tick state for animation)
│   - SimResult.to_charts() → list[ForgeViz ChartSpec]
│
├── models.py
│   - Station(cycle_time_dist, changeover_dist, uptime, batch_size)
│   - Buffer(capacity, initial_wip)
│   - Source(arrival_dist)
│   - Sink()
│
└── scenarios.py
    - from_vsm(vsm_data) → SimNetwork  # build sim from VSM process steps
    - from_graph(graph_id, node_filter) → SimNetwork  # build sim from process graph
    - compare(baseline: SimResult, proposed: SimResult) → ComparisonResult
```

### B. Hoshin X-Matrix → Graph → KPI tracking

Hoshin strategic objectives and KPIs are currently standalone. They should be graph nodes:

- Strategic objective = high-level goal node
- Annual objective = decomposed goal node (edge: contributes_to)
- KPI = measurement node (edge: measures)
- Improvement project = intervention node (edge: addresses)

This means the X-Matrix IS a graph view — filtered to goal/measurement/intervention nodes. The correlation matrix in the X-Matrix becomes graph edge strengths.

**What PM builds:**
- `GraphService.seed_from_hoshin(hoshin_project_id)` — strategic/annual objectives as nodes, KPIs as measurement nodes, correlations as edges
- X-Matrix rendering reads from graph instead of custom Hoshin API
- KPI actuals update graph node distributions, triggering staleness/contradiction signals on connected edges

### C. ForgeViz direct integration for all calculators

Every calculator currently renders through Plotly (3.5MB). The migration replaces with ForgeViz. But we can do better:

- `forgesiop.production.oee()` returns a ForgeViz `ChartSpec` directly (waterfall chart)
- `forgespc.charts.from_spc_result()` already does this for control charts
- Every forge computation function should have an optional `chart=True` param that returns computation + visualization in one call

**What PM builds:**
- Add `to_chart()` or `chart=True` to all new forgesiop production/lean functions
- Use `forgeviz.charts.generic` builders (bar, stacked_bar, gauge) inside forge packages
- Calculator JS becomes: call API → get result + chart spec → `ForgeViz.render(el, result.chart)`

### D. Unified parameter store

Calculators currently require manual input of takt time, demand, cycle times. VSM has this data. The graph has this data. SPC has real-time distributions. But the calculators don't know about any of it.

**Proposal:** When a calculator opens, it checks the graph for relevant node data and pre-fills:
- Takt time → from VSM or `process.takt` config setting
- Cycle times → from graph process step nodes
- Defect rates → from SPC Cpk/dpmo on quality characteristic nodes
- Demand → from forgesiop demand sensing
- Setup times → from graph changeover edges

**What PM builds:**
- `GraphService.get_calculator_context(calculator_type)` → dict of pre-fill values
- Calculator API returns `{defaults: {...}, from_graph: true}` when graph data is available
- UI shows "from graph" badge on pre-filled fields, user can override

---

## Part 4: What NOT to extract

These stay as client-side JS because they're primarily UI/visualization, not computation:

- **QFD / House of Quality** — interactive matrix with drag-and-drop weighting. The computation is trivial (weighted sums). The value is the UI.
- **Job sequencing drag-and-drop** — scheduling optimization could go server-side eventually, but the interactive Gantt chart is the product.
- **Simulation animations** — the sim engine moves server-side, but the real-time animation loop stays in JS (reads frames, renders SVG/canvas).
- **Whiteboard / VSM canvas** — pan/zoom/drag spatial layouts. Use `SvCanvas` shared library. Pure client-side.

---

## Implementation Priority

| Priority | Package | Functions | Blocks |
|----------|---------|-----------|--------|
| **P0** | forgesiop.production | takt, oee, kanban, epei, flow, cost | Operations workbench migration |
| **P0** | forgequeue | mm1, mmc, mmck, erlang_c, priority, tandem, staffing | Operations workbench migration |
| **P1** | forgesiop.lean | smed, changeover, family analysis | Operations workbench migration |
| **P1** | forgestat.quality.desirability | derringer_suich | Advanced calculator migration |
| **P1** | forgestat.core.sampling | sample_normal/exp/weibull/poisson, seeded_rng | All simulators |
| **P2** | forgesiop.simulation.engine | SimNetwork, run, from_vsm, from_graph | Simulator decomposition |
| **P2** | GraphService extensions | seed_from_vsm, get_calculator_context, add_calculator_evidence | Cross-tool integration |
| **P3** | GraphService.seed_from_hoshin | X-Matrix as graph view | Hoshin integration |
| **P3** | ForgeViz chart returns | to_chart() on all forge functions | Calculator UX improvement |

P0 is needed before operations workbench migration can complete. P1 is needed for full calculator coverage. P2-P3 are integration improvements that can happen after migration.
