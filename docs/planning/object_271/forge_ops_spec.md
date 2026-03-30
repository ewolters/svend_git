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

### What the graph does and doesn't connect

The graph stores calibrated causal relationships — "X causes Y" with Bayesian posteriors updated by evidence from DOEs, investigations, SPC, and forced failure tests. It's a knowledge capture system.

**Calculators are NOT graph producers.** They're operational decision tools — takt time, OEE, queue analysis, line balance. They consume process data and return results for a human to act on. A takt time calculation doesn't produce evidence about causal relationships.

**Tools that DO produce graph evidence:** Investigations, SPC (shift detection → staleness), DOE (effect sizes → edge posteriors), FFTs (detection capability). These write to the graph through the Loop.

**Calculators read from VSM, not from the graph.** The VSM stores process step data (cycle times, changeover times, uptime, WIP). The calculators consume it. This is a data-sharing relationship, not a knowledge relationship. Routing it through the graph would be architecturally wrong — the graph is for causal knowledge, not operational parameters.

### Improvements to existing bridges

#### 1. VSM → Calculators (simplify calc-vsm.js)

The 16 per-calculator VSM import loaders in `calc-vsm.js` share a pattern: fetch VSM, extract step data, map to calculator fields. This should be one shared loader with per-calculator field mappings, not 16 separate functions.

**What PM builds:**
- Refactor `calc-vsm.js`: one `loadVSMData(vsm_id)` function returns normalized step data
- Per-calculator mappings as config objects, not separate functions
- Same VSM API (`/api/vsm/`), same data flow — just less code

#### 2. Safety → Loop (replaces FMEA bridge)

The Safety app currently bridges Frontier Card findings directly to FMEA rows. Under OLR-001:

- Frontier Card finding → Signal (source: gemba)
- Signal triage → Investigation (if warranted)
- Investigation concludes → graph writeback (this IS evidence)
- Graph evidence → FMIS row updates (S/O/D posteriors)

The direct Safety → FMEA bridge becomes Safety → Signal → Loop. The Loop handles routing.

**What PM builds:**
- Safety `create_signal_from_card(card_id)` — creates Loop Signal with frontier card data
- Direct FMEA row creation stays as a shortcut option (not everything needs an investigation)

#### 3. FMEA ↔ Graph (bidirectional — already mostly built)

`seed_from_fmis()` exists. The reverse should too: new graph edge proposes FMIS row. FMIS S/O/D posteriors (from FFTs) update graph edge posteriors. This is genuine causal knowledge flow — FMEA failure modes ARE causal claims about the process.

#### 4. Hoshin → VSM proposals (keep direct bridge)

Hoshin's VSM proposal generation (`/api/vsm/{id}/generate-proposals/`) works. It reads waste data from a specific VSM and proposes CI projects. This is a direct analytical operation, not a graph relationship. Keep it as-is.

---

## Part 3: Optimization Opportunities

### A. Unified simulation engine

Three simulation JS files (`calc-sim-line.js`, `calc-sim-flow.js`, `calc-sim-quality.js`) share patterns: discrete event loop, Monte Carlo sampling, real-time animation, statistics collection. They should share a simulation kernel.

**Proposal:** `forgesiop.simulation.engine` — a discrete event simulation engine in Python that:
- Defines a process network from VSM data or manual station definitions
- Runs Monte Carlo with configurable distributions per station
- Returns time-series results as ForgeViz ChartSpecs
- JS handles animation only (reads sim frames from API via SSE or polling)

This replaces `svend-sim-core.js` (249 lines) and the three sim files with a server-side engine that's testable and uses real process data from VSM.

```
forgesiop/simulation/
├── engine.py
│   - SimNetwork(stations, buffers, config) → simulator
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
    - compare(baseline: SimResult, proposed: SimResult) → ComparisonResult
```

### B. ForgeViz direct integration for all calculators

Every calculator currently renders through Plotly (3.5MB). The migration replaces with ForgeViz. But we can do better:

- `forgesiop.production.oee()` returns a ForgeViz `ChartSpec` directly (waterfall chart)
- `forgespc.charts.from_spc_result()` already does this for control charts
- Every forge computation function should have an optional `chart=True` param that returns computation + visualization in one call

**What PM builds:**
- Add `to_chart()` or `chart=True` to all new forgesiop production/lean functions
- Use `forgeviz.charts.generic` builders (bar, stacked_bar, gauge) inside forge packages
- Calculator JS becomes: call API → get result + chart spec → `ForgeViz.render(el, result.chart)`

### C. VSM-aware parameter pre-fill

Calculators currently require manual input of takt time, demand, cycle times. VSM already has this data. The `calc-vsm.js` import system exists but requires the user to explicitly open the import modal and select a VSM.

**Proposal:** When a calculator opens and a VSM is active (user came from VSM or has a recent VSM), pre-fill fields automatically:
- Takt time → from active VSM's takt_time field
- Cycle times → from VSM process steps
- Changeover/setup times → from VSM process steps
- Demand → from VSM customer demand or ConfigService `process.demand`

**What PM builds:**
- `GraphService.get_calculator_context(calculator_type)` → dict of pre-fill values
- Calculator API accepts optional `vsm_id` param → returns `{defaults: {...}, from_vsm: true}`
- Refactored `calc-vsm.js` passes VSM context to calculator on navigation
- UI shows "from VSM" badge on pre-filled fields, user can override

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
| **P2** | forgesiop.simulation.engine | SimNetwork, run, from_vsm, compare | Simulator decomposition |
| **P2** | calc-vsm.js refactor | Single loader + per-calc field mappings | VSM import simplification |
| **P2** | ForgeViz chart returns | to_chart() on all forge functions | Calculator UX improvement |

P0 is needed before operations workbench migration can complete. P1 is needed for full calculator coverage. P2 is post-migration polish.
