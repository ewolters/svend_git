# Operations Calculator Flows & Surface Area

> Audit date: 2026-02-13
> Source: `services/svend/web/templates/calculators.html` (~869KB, ~17,750 lines)

---

## 1. Calculator Inventory (54 total: 49 active + 5 coming soon)

### Crewing (5)
| Calculator | Status | Publishes | Financial Output |
|-----------|--------|-----------|-----------------|
| Takt Time | Active | `takt`, `taktMin` | — |
| RTO (Staffing) | Active | `rtoStaff`, `lineEfficiency` | — |
| Yamazumi | Active | — | — |
| Line Simulator | Active | `simThroughput`, `simWIP`, `simEfficiency` | — |
| Cell Design Simulator | Coming Soon | — | — |

### Inventory (6)
| Calculator | Status | Publishes | Financial Output |
|-----------|--------|-----------|-----------------|
| Kanban Sizing | Active | `kanbanCards`, `kanbanInventory` | — |
| EPEI | Active | `epei` | — |
| Safety Stock | Active | `safetyStock`, `rop` | — |
| EOQ | Active | `eoq` | Annual order cost, holding cost, total cost ($) |
| Kanban Simulator | Active | — | — |
| Safety Stock Simulator | Coming Soon | — | — |

### Capacity (3)
| Calculator | Status | Publishes | Financial Output |
|-----------|--------|-----------|-----------------|
| OEE | Active | `oee`, `oeeAvailability` | — |
| Bottleneck | Active | `bottleneckCT`, `bottleneckThroughput` | — |
| TOC / DBR Simulator | Active | — | Throughput rate (indirect) |

### Flow (5)
| Calculator | Status | Publishes | Financial Output |
|-----------|--------|-----------|-----------------|
| Little's Law | Active | `littlesResult` | — |
| Pitch | Active | `pitch` | — |
| Product Flow (PFA) | Active | — | — |
| Workflow (WFA) | Active | — | — |
| Beer Game | Active | — | Total SC cost ($), holding vs backlog split |

### Scheduling (5)
| Calculator | Status | Publishes | Financial Output |
|-----------|--------|-----------|-----------------|
| Job Sequencer | Active | — | — |
| Sequence Optimizer | Active | — | — |
| Capacity Load | Active | — | Utilization %, overload days |
| Mixed-Model | Active | — | — |
| Due Date Risk Simulator | Active | — | — |

### Queuing Lab (8)
| Calculator | Status | Publishes | Financial Output |
|-----------|--------|-----------|-----------------|
| M/M/c Queue | Active | — | — |
| M/M/c/K Finite | Active | — | — |
| Priority Queue | Active | — | — |
| Staffing Optimizer | Active | — | Total cost/hr ($), optimal servers |
| Live Queue Simulator | Active | — | — |
| A/B Compare Simulator | Active | — | — |
| Multi-Stage (Tandem) | Active | — | — |
| Erlang C Staffing | Active | `erlang_agents`, `erlang_sl` | — |

### Quality (2)
| Calculator | Status | Publishes | Financial Output |
|-----------|--------|-----------|-----------------|
| RTY | Active | `rty` | — |
| DPMO / Sigma | Active | `sigma`, `dpmo` | — |

### Financial (2)
| Calculator | Status | Publishes | Financial Output |
|-----------|--------|-----------|-----------------|
| Inventory Turns | Active | `turns`, `daysOnHand` | Turns/yr, days on hand (from COGS $) |
| Cost of Quality | Active | `coqTotal`, `coqFailure` | Prevention/appraisal/failure ($) |

### Changeover (3)
| Calculator | Status | Publishes | Financial Output |
|-----------|--------|-----------|-----------------|
| SMED | Active | `changeoverInternal` | Annual value ($), hours saved, capacity gain % |
| Changeover Matrix | Active | — | — |
| SMED Simulator | Coming Soon | — | — |

### Risk & Quality (5)
| Calculator | Status | Publishes | Financial Output |
|-----------|--------|-----------|-----------------|
| FMEA / RPN | Active | — | — |
| Cp / Cpk | Active | `cpk`, `cp` | — |
| Sample Size | Active | — | — |
| FMEA Monte Carlo | Coming Soon | — | — |
| Risk Matrix | Active | — | — |

### Line Performance (4)
| Calculator | Status | Publishes | Financial Output |
|-----------|--------|-----------|-----------------|
| Line Efficiency | Active | `lineEffCalc`, `lineEffActualRate` | — |
| OLE | Active | `ole` | — |
| Cycle Time Study | Active | `cycleTimeTotal`, `cycleTimeVA` | — |
| MTBF / MTTR | Active | `mtbf`, `mttr`, `availability` | — |

### Analysis (3)
| Calculator | Status | Publishes | Financial Output |
|-----------|--------|-----------|-----------------|
| Before / After | Active | — | — |
| Heijunka | Active | — | — |
| Heijunka Simulator | Coming Soon | — | — |

### 3P (1)
| Calculator | Status | Publishes | Financial Output |
|-----------|--------|-----------|-----------------|
| QFD (House of Quality) | Active | — | — |

### Quality & DOE (3)
| Calculator | Status | Publishes | Financial Output |
|-----------|--------|-----------|-----------------|
| Desirability Optimizer | Active | — | — |
| SPC Rare Events | Active | — | — |
| Probit / Dose-Response | Active | — | — |

---

## 2. SvendOps Data Bus Architecture

### Publish-Pull Model (not pub-sub)

Calculators **publish** results to `SvendOps.values` (a global state object). Other calculators **pull** on demand via UI buttons. There are no automatic subscriptions — the user decides when to import.

**38 published keys** across all calculators. Each stores `{value, unit, source, timestamp}`.

### Pull Connections (cross-calculator data flow)

| From | To | Data | Mechanism |
|------|----|------|-----------|
| Takt Time | Line Simulator | `takt` → `ls-takt` | `SvendOps.pull()` |
| Line Simulator | Job Sequencer | Order data | `pullJobsFromLineSim()` |
| Job Sequencer | Sequence Optimizer | Job list | `pullFromSequencer()` |
| Job Sequencer | Capacity Load | Workload | `pullFromSequencerToCapacity()` |
| Job Sequencer | Due Date Sim | Order list | `pullFromSequencerToDDS()` |
| Heijunka | Mixed-Model | Product mix | `pullFromHeijunka()` |
| Changeover Matrix | Sequence Optimizer | Per-pair setup times | `pullChangeoverMatrix()` |
| Mixed-Model | Line Simulator | Leveled sequence | `pullFromMixedModel()` |
| Line Simulator | Kanban Simulator | Station layout | `pullLinesToKanban()` |
| Kanban Simulator | Line Simulator | Station layout | `pullKanbanToLine()` |
| Line Simulator | TOC/DBR | Stations (CT→capacity) | `pullLinesToTOC()` |
| Bottleneck | TOC/DBR | Stations (CT→capacity) | `pullBottleneckToTOC()` |

### Push Connections (reverse / complex object flow)

| From | To | Function |
|------|----|----------|
| Job Sequencer | Line Simulator | `pushSequenceToLineSim()` |
| Mixed-Model | Line Simulator | `pushMixedToLineSim()` |

### VSM Import (universal)

Available on ALL calculator pages via `openVSMImport()`. Imports `effectiveStations` structure (`{name, cycle_time, n_machines}`) from any saved VSM. Primary consumer is Line Simulator — after import, auto-switches to Line Sim tab.

**VSM import consumers** (calculators that can use station data):
- Line Simulator (stations as processing steps)
- Kanban Simulator (station-based pull system)
- TOC / DBR Simulator (station-based constraint flow)
- Beer Game (indirectly — supply chain tiers)

---

## 3. Data Flow Graph

```
                                    ┌──────────────┐
                                    │   VSM Maps    │
                                    └──────┬───────┘
                                           │ openVSMImport()
                                           ▼
┌───────────┐  takt   ┌────────────────────────┐  pullJobs   ┌──────────────┐
│ Takt Time ├────────►│   Line Simulator        ├────────────►│ Job Sequencer│
└───────────┘         │ (discrete-event, 500ms) │◄────────────┤              │
                      └─────────┬──────────────┘ pushSequence └──┬───┬───┬──┘
                                │                                │   │   │
                      simThroughput, simWIP, simEfficiency        │   │   │
                                │                                │   │   │
                                ▼                                │   │   │
                         SvendOps.values                         │   │   │
                                                                 │   │   │
                      ┌──────────────────────────────────────────┘   │   │
                      │ pullFromSequencer                            │   │
                      ▼                                              │   │
              ┌───────────────┐                                      │   │
              │  Sequence     │                                      │   │
              │  Optimizer    │                                      │   │
              │ (NN,2-Opt,    │                                      │   │
              │  EDD,SPT)     │                                      │   │
              └───────────────┘                                      │   │
                                                                     │   │
                      ┌──────────────────────────────────────────────┘   │
                      │ pullFromSequencerToCapacity                      │
                      ▼                                                  │
              ┌───────────────┐                                          │
              │ Capacity Load │                                          │
              │ (by day/shift)│                                          │
              └───────────────┘                                          │
                                                                         │
                      ┌──────────────────────────────────────────────────┘
                      │ pullFromSequencerToDDS
                      ▼
              ┌───────────────┐
              │ Due Date Risk │
              │ (Monte Carlo) │
              └───────────────┘

┌───────────┐  pullFromHeijunka  ┌─────────────┐  pushMixedToLineSim  ┌────────────────┐
│  Heijunka ├──────────────────►│ Mixed-Model  ├─────────────────────►│ Line Simulator │
└───────────┘                    └─────────────┘                      └────────────────┘

┌───────────┐         ┌─────────────┐
│    EOQ    ├────────►│   Kanban    │  (container size input)
└───────────┘         └─────────────┘

┌───────────┐         ┌─────────────┐
│   SMED    ├────────►│    EPEI     │  (changeover time input)
└───────────┘         └─────────────┘

┌───────────┐         ┌─────────────┐
│ MTBF/MTTR ├────────►│     OEE     │  (availability input)
└───────────┘         └─────────────┘

┌───────────┐         ┌─────────────┐
│ Bottleneck├────────►│ Little's Law│  (throughput input)
└───────────┘         └─────────────┘
```

### Islands (no outbound or inbound connections)
These calculators publish to SvendOps but have no direct pull/push connections:
- Safety Stock, RTY, DPMO/Sigma, Inventory Turns, CoQ, Cpk, Line Efficiency, OLE, Cycle Time Study, Erlang C, Pitch
- Beer Game, Kanban Simulator, TOC/DBR, Queue Simulator, A/B Compare
- QFD, FMEA, Risk Matrix, Sample Size, Probit, Desirability Optimizer, SPC Rare Events
- Before/After, PFA, WFA, Yamazumi, Changeover Matrix

---

## 4. Simulation Engines

### Discrete-Event Simulations (6)

| Engine | Tick Rate | State Object | What It Simulates |
|--------|-----------|-------------|-------------------|
| **Line Simulator** | `speed × 500ms` | `lineSimState` | Multi-station production line, breakdowns, blocking/starving, changeovers, order-driven or continuous |
| **Queue Simulator** | `100ms` | `simState` | M/M/c arrivals, multi-server service, queue formation, burst detection |
| **A/B Queue Compare** | `80ms` | `compareState` | Two queue scenarios side-by-side |
| **Kanban Simulator** | Variable | `kanbanState` | Push vs pull, kanban card circulation, supermarket inventory, stockouts |
| **Beer Game** | `100ms` | `beerState` | 4-tier supply chain, lead time delays, bullwhip effect, ordering policies |
| **TOC / DBR** | `100ms` | `tocState` | Multi-station constraint flow, uncontrolled vs DBR, buffer management |

All use `setInterval` loops with pause/resume and variable speed controls.

### Monte Carlo Simulations (7)

All use the shared `MonteCarlo.simulate()` engine with **2,000 runs**.

| Calculator | Function | Varies |
|-----------|----------|--------|
| Kanban | `runKanbanMonteCarlo()` | Demand, lead time |
| Safety Stock | `runSafetyMonteCarlo()` | Demand, lead time |
| EOQ | `runEOQMonteCarlo()` | Cost, demand |
| M/M/c Queue | `runQueueMonteCarlo()` | Arrival rate, service rate |
| M/M/c/K Finite | `runQFMonteCarlo()` | Arrival, service |
| Tandem Queue | `runTandemMonteCarlo()` | Stage rates |
| Cpk | `runCpkMonteCarlo()` | Spec, process variation |

### Statistical / Optimization (3)

| Engine | Type | Method |
|--------|------|--------|
| **Due Date Risk** | Monte Carlo probability | Process time variation → on-time probability |
| **Desirability Optimizer** | Grid search | Multi-response composite desirability across factor space |
| **SPC Rare Events** | Control chart simulation | G/T chart with shift injection, instant or animated |

---

## 5. Financial Capabilities

### Calculators with Direct Dollar Output

| Calculator | Financial Outputs | Type |
|-----------|-------------------|------|
| **Cost of Quality** | Prevention ($), Appraisal ($), Failure ($), Total CoQ ($) | Cost breakdown |
| **EOQ** | Annual order cost ($), holding cost ($), total cost ($) | Cost optimization |
| **Inventory Turns** | Days on hand, weeks on hand (from COGS $ / avg inventory $) | Financial ratio |
| **SMED** | Annual value ($), hours saved/yr, capacity gain % | Savings |
| **Staffing Optimizer** | Total cost/hr ($), server cost ($), wait cost ($) | Cost optimization |
| **Beer Game** | Total SC cost ($), holding ($0.50/unit) vs backlog ($1.00/unit) | Simulation |

### Financial Gaps

| Gap | Impact | Effort |
|-----|--------|--------|
| **No ROI / payback period calculator** | Can't answer "when does this pay for itself?" | Medium — needs investment input + savings stream |
| **No working capital impact** | Inventory changes don't show cash flow effect | Medium — needs inventory $ × carrying cost rate |
| **No throughput accounting ($)** | TOC/DBR shows units/hr but not $/hr | Low — multiply throughput × revenue per unit |
| **No changeover cost matrix** | Changeover Matrix tracks time but not $ per transition | Low — add hourly rate column |
| **No aggregate financial dashboard** | Individual calculators show $, but no total view | High — needs cross-calculator rollup |
| **SMED is the only calculator that values time savings** | Line Sim, Sequence Optimizer, etc. show time improvements but not $ | Medium — need hourly cost input per calculator |

---

## 6. Scheduling Capabilities

### Current State

| Feature | Status | Detail |
|---------|--------|--------|
| Manual job sequencing | Done | Drag-and-drop in Job Sequencer |
| Algorithmic optimization | Done | 4 algorithms: Nearest Neighbor, 2-Opt, EDD, SPT |
| Capacity load planning | Done | Load vs available hours by day |
| Due date risk analysis | Done | Monte Carlo on-time probability |
| Mixed-model leveling | Done | Heijunka → Mixed-Model → Line Sim pipeline |
| Multi-resource scheduling | Missing | Jobs assigned to single resource, no parallel machines |
| Finite capacity scheduling | Missing | Capacity Load shows overload but doesn't reschedule |
| Setup-dependent sequencing from Changeover Matrix | Missing | Matrix exists but doesn't feed Sequence Optimizer |
| Calendar / shift pattern | Missing | Fixed hours/day, no shift templates |

### Scheduling Flow (current)

```
Heijunka → Mixed-Model → Line Simulator
                              ↑
Job Sequencer → Sequence Optimizer
     │
     ├──→ Capacity Load
     └──→ Due Date Risk
```

### Scheduling Gaps

| Gap | Impact |
|-----|--------|
| **Changeover Matrix → Sequence Optimizer** not connected | Matrix has setup times per product pair, but optimizer uses flat setup time |
| **No multi-machine scheduling** | Can't model parallel lines or work centers |
| **Capacity Load is display-only** | Shows overload but can't auto-level or suggest rescheduling |
| **Due Date Risk results don't feed back** | Risk assessment is terminal — no "reschedule risky orders" action |

---

## 7. Multi-Material Kanban Simulation (Future — v2)

### Current Kanban Simulator Limitations
- **Single product only** — one material type flowing through stations
- **No scheduling integration** — push vs pull toggle but no connection to Job Sequencer
- **No financial layer** — tracks WIP and throughput but not $ inventory value
- **No lot sizing** — continuous flow, no batch/container concepts
- **No demand patterns** — constant demand rate only

### Proposed v2 Architecture

**Multi-SKU Kanban with Scheduling + Financial Layer**

```
VSM Future State
       │
       ▼
┌─────────────────────────┐
│  Multi-Material Kanban  │
│  Simulator v2           │
│                         │
│  Materials:             │
│  ├─ Type A (high vol)   │
│  ├─ Type B (med vol)    │
│  └─ Type C (low vol)    │
│                         │
│  Stations:              │
│  ├─ Shared resources    │
│  ├─ Changeover times    │  ◄── Changeover Matrix
│  └─ Material-specific   │
│     cycle times         │
│                         │
│  Pull signals:          │
│  ├─ Per-material cards  │
│  ├─ Supermarket levels  │
│  └─ Replenishment rules │
│                         │
│  Demand:                │
│  ├─ Per-material rates  │
│  ├─ Patterns (seasonal, │  ◄── Heijunka product mix
│  │  step, random)       │
│  └─ Priority classes    │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Financial Layer         │
│                         │
│  Per material:          │
│  ├─ Unit cost ($)       │
│  ├─ Holding cost rate   │
│  ├─ Stockout penalty    │
│  └─ Revenue per unit    │
│                         │
│  Outputs:               │
│  ├─ WIP inventory ($)   │
│  ├─ Carrying cost/yr    │
│  ├─ Stockout cost/yr    │
│  ├─ Throughput $/hr     │
│  ├─ Total SC cost       │
│  └─ Working capital req │
└─────────────────────────┘
```

**Integration points:**
- Import stations from VSM future state
- Import changeover times from Changeover Matrix
- Import product mix from Heijunka
- Import demand patterns from Beer Game demand profiles
- Export throughput data to Capacity Load
- Export financial results to a rollup dashboard

**Estimated scope:** ~400 lines JS (simulator) + ~100 lines (financial layer) + ~100 lines (UI/controls)

---

## 8. Critical Flow Gaps (Disconnected Connections)

### High-Priority (break user stories)

| Gap | From → To | Why It Matters |
|-----|-----------|---------------|
| **Mixed-Model → Line Sim is push-only** | Mixed-Model pushes sequence, but Line Sim can't pull | User must manually push; easy to forget |
| **Due Date Risk → nothing** | Risk results are terminal | Can't act on risk — need "reschedule" or "flag order" action |
| **Changeover Matrix → Sequence Optimizer** | Matrix has per-pair setup times, optimizer uses flat time | Optimizer ignores sequence-dependent setup — defeats the purpose of having both |
| **Before/After → nothing** | Comparison results don't link anywhere | Should feed evidence to Synara or project hypotheses |

### Medium-Priority (missed value)

| Gap | From → To | Why It Matters |
|-----|-----------|---------------|
| **No calculator → Synara evidence bridge** (except DSW, just added) | Operations results don't feed hypothesis tracking | Major disconnect between tools and scientific reasoning |
| **Beer Game → Kanban Simulator** | Beer Game demonstrates bullwhip, Kanban demonstrates pull solution, but they don't connect | Natural pedagogical flow is broken |
| **OEE → Line Simulator** | OEE calculates availability/performance/quality, Line Sim could use these as station parameters | User must manually transfer values |
| **FMEA RPN → Risk Matrix** | Both assess risk but independently | High-RPN failure modes should auto-populate risk matrix |
| **Cpk → Sample Size** | Process capability informs required sample size for next study | No connection |
| **Erlang C → Staffing Optimizer** | Both solve staffing problems with different approaches | Should cross-reference results |

### Low-Priority (nice to have)

| Gap | From → To |
|-----|-----------|
| RTY → CoQ (yield loss → failure cost estimate) |
| DPMO → Cpk (defect rate → capability index cross-check) |
| Cycle Time Study → Yamazumi (VA/NVA breakdown → balance chart) |
| MTBF/MTTR → Kanban Simulator (breakdown events) |
| Little's Law → Line Simulator (validate WIP = TH × CT) |

---

## 9. Coming Soon Items (5 placeholders)

| Calculator | Category | What It Would Do | Dependencies |
|-----------|----------|-----------------|-------------|
| **Cell Design Simulator** | Crewing | Simulate U-cell vs I-line vs parallel layouts with walking time | Takt Time, Yamazumi, RTO |
| **Safety Stock Simulator** | Inventory | Discrete-event demand/replenishment simulation (vs current formula) | Safety Stock MC, demand profiles |
| **SMED Simulator** | Changeover | Animate changeover process, track internal→external conversion | SMED analysis, Changeover Matrix |
| **FMEA Monte Carlo** | Risk & Quality | Simulate RPN distributions with uncertainty in S/O/D ratings | FMEA/RPN calculator |
| **Heijunka Simulator** | Analysis | Animate leveled vs unleveled production with WIP/delivery comparison | Heijunka, Mixed-Model |

---

## 10. Cross-Simulator Integration Opportunities

Currently all 6 discrete-event simulators run independently. Potential integrations:

| Integration | What It Enables |
|-------------|----------------|
| **Line Sim → Kanban Sim** | Production line feeds pull system — see end-to-end flow |
| **Kanban Sim → Beer Game** | Pull system feeds supply chain — see multi-tier pull behavior |
| **TOC/DBR → Line Sim** | Constraint identification informs line simulation parameters |
| **Queue Sim → Line Sim** | Queue theory validates/calibrates production line WIP behavior |
| **Line Sim + Kanban Sim + Beer Game** as "Supply Chain Simulator" | Full factory-to-customer discrete-event simulation |

### Unified Simulation Vision

```
Raw Material ──► [Line Sim: Production] ──► [Kanban Sim: Pull Control]
                        │                          │
                        │                          ▼
                  [TOC/DBR: Constraint]    [Beer Game: Supply Chain]
                                                   │
                                                   ▼
                                            Customer Demand
```

This would be a unique differentiator — no competitor offers end-to-end discrete-event simulation from factory floor through supply chain in a single tool.

---

## 11. VSM Import Surface

The VSM import (`openVSMImport()`) is available on all calculator pages but currently only auto-routes to Line Simulator. Other calculators that should consume VSM data:

| Calculator | What It Could Import | Current State |
|-----------|---------------------|---------------|
| Line Simulator | Station names, cycle times, n_machines | Working |
| Kanban Simulator | Station structure for pull simulation | Not connected — must manually enter |
| TOC/DBR | Station structure for constraint analysis | Not connected |
| Bottleneck | Station cycle times for constraint identification | Not connected |
| Takt Time | Total demand from VSM customer box | Not connected |
| OEE | Station-level OEE from VSM data boxes | Not connected |
| Capacity Load | Station workloads from VSM | Not connected |

---

## 12. Summary Counts

| Metric | Count |
|--------|-------|
| Total calculators | 54 (49 active + 5 coming soon) |
| Published SvendOps keys | 38 |
| Pull connections | 12 |
| Push connections | 2 |
| Discrete-event simulators | 6 |
| Monte Carlo simulators | 7 |
| Statistical/optimization engines | 3 |
| Calculators with direct $ output | 6 |
| Categories | 14 |
| High-priority flow gaps | 4 |
| Medium-priority flow gaps | 6 |
| Low-priority flow gaps | 5 |
| VSM import candidates (unconnected) | 6 |
