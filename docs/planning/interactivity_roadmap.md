# Operations Workbench — Interactivity Roadmap

**Created:** 2026-02-20
**Goal:** Education, simulation, interactivity — tools that build understanding, not just compute numbers
**Constraint:** Must not duplicate DSW, DOE, SPC, Forecasting, FMEA, RCA, VSM, or A3 modules

## Already Exists Elsewhere (DO NOT BUILD)

These were identified as gaps vs Minitab/JMP but **Svend already has them**:

| Tool | Where it lives |
|------|---------------|
| Gage R&R (crossed, nested, expanded, linearity, Type 1, attribute) | DSW `gage_rr` family (6 types) |
| Attribute Agreement Analysis | DSW `attribute_agreement` |
| Capability Sixpack | DSW `capability_sixpack` |
| Weibull / Life Data | DSW reliability (Weibull, lognormal, exponential, distribution ID) |
| Accelerated Life Testing (ALT) | DSW `accelerated_life` |
| Repairable Systems (NHPP) | DSW `repairable_systems` |
| Warranty Prediction | DSW `warranty` |
| Acceptance Sampling + OC Curves | DSW `acceptance_sampling`, `variable_acceptance_sampling`, `multiple_plan_comparison` |
| DOE Screening (fractional factorial) | DOE module (fractional, PB, DSD) |
| Full Factorial + Interaction Explorer | DOE `full_factorial` + `main_effects` + `interaction` + `analyze_results` |
| Response Surface / Optimization | DOE `ccd`, `box_behnken`, `contour_plot`, `optimize_response` |
| Taguchi Robust Design | DOE `taguchi` |
| Demand Forecasting (SMA, WMA, ES, Holt-Winters) | Forecast module |
| Time Series Decomposition | DSW `decomposition` |
| Multi-Vari Chart | DSW `multi_vari` |
| All hypothesis tests (t, ANOVA, chi-sq, proportion, nonparametric) | DSW stats (40+ tests) |
| Power & Sample Size (9 types) | DSW stats |
| Equivalence / TOST | DSW `equivalence` |
| Regression (linear, logistic, robust, nonlinear, PLS, GLM, stepwise) | DSW stats + ML |
| Monte Carlo Simulation | DSW `simulation` (20 vars, 100K iterations) |
| Cost of Quality | Already in calculators.html (`coq`) |
| FMEA / RPN | Already in calculators.html (`fmea`) + full FMEA module |
| Cp/Cpk | Already in calculators.html (`cpk`) + DSW + SPC module |
| Sample Size | Already in calculators.html (`samplesize`) + DSW (9 types) |

---

## Ship "Coming Soon" (already in nav, visible to users)

These are promised but not yet implemented. Ship first.

| ID | Tool | Category | What to build |
|----|------|----------|---------------|
| S1 | Cell Design Simulator | Crewing | Animate operator walking patterns in line, U-cell, and parallel layouts. Show travel distance, WIP, and throughput differences |
| S2 | Safety Stock Simulator | Inventory | Stochastic demand over time with visual inventory level, reorder point, stockout events. Drag safety stock slider to see effect |
| S3 | SMED Simulator | Changeover | Drag changeover elements between internal/external. Watch total changeover time shrink. Before/after timeline |
| S4 | FMEA Monte Carlo | Risk & Quality | Run failure cascades through system. See how compounding small risks create system-level failures |
| S5 | Heijunka Simulator | Analysis | Side-by-side batched vs leveled production. Show WIP accumulation and lead time difference in real time |

---

## Tier 1 — High Impact Additions

Unique to Operations Workbench. Not in DSW/DOE/SPC. Strong educational value.

### Reliability & Maintenance

| ID | Tool | Why it adds value |
|----|------|-------------------|
| T1 | **Bathtub Curve Explorer** | DSW does Weibull *analysis* on real data. This is the educational counterpart — drag beta/eta sliders and watch the hazard function shift between infant mortality (beta<1), useful life (beta=1), and wear-out (beta>1). Builds intuition for why Weibull matters. Pure Plotly, no backend needed |
| T2 | **PM Optimization (Age Replacement)** | Given failure distribution + planned/unplanned repair costs, find the optimal preventive replacement interval. Interactive cost-vs-interval chart. Extends existing MTBF/MTTR calculator into actionable decision-making |
| T3 | **Spare Parts Calculator** | Given demand rate (from MTBF) + target service level, compute optimal spare parts stock via Poisson distribution. Interactive chart: stock level vs stockout probability. Connects MTBF/MTTR → inventory decision |

### Planning & Supply Chain

| ID | Tool | Why it adds value |
|----|------|-------------------|
| T4 | **MRP Explosion Calculator** | Enter BOM tree + MPS + on-hand + lead times. Walk through netting, lot-sizing, offsetting level by level. Visual BOM tree with planned order releases. The #1 APICS/CPIM topic — no good interactive web version exists |
| T5 | **Aggregate Planning Simulator** | Compare Chase (hire/fire), Level (constant workforce), and Hybrid strategies over a multi-period horizon. Drag demand curve, watch cost bars update: hiring, layoff, overtime, inventory holding, stockout. Classic HBS simulation |
| T6 | **Newsvendor (Single-Period Inventory)** | For perishable/seasonal products — set overage cost, underage cost, demand distribution. See the critical ratio and optimal order quantity. Drag distribution parameters, watch the payoff chart update. Pairs with existing EOQ (multi-period) and Safety Stock |

### Project & Network

| ID | Tool | Why it adds value |
|----|------|-------------------|
| T7 | **PERT/CPM Network Builder** | Enter activities + durations (optimistic/likely/pessimistic) + dependencies. Draws AON network diagram. Highlights critical path, shows slack per activity. Click activity to see what happens if it's delayed. Nothing like this on the platform |
| T8 | **PERT Monte Carlo** | Extension of T7 — run 10K simulations using PERT distributions. Show histogram of project completion times. Demonstrates merge bias (why deterministic CPM underestimates). Pairs naturally with T7 |

### Industrial Engineering

| ID | Tool | Why it adds value |
|----|------|-------------------|
| T9 | **Learning Curve Calculator** | Enter first-unit time + learning rate. Plot unit time and cumulative time for Wright (cumulative avg) and Crawford (unit) models side by side. Drag learning rate slider — see the curve flatten or steepen. Every IE program teaches this, no good web version |
| T10 | **Assembly Line Balancing** | Enter task times + precedence diagram. Apply heuristics (Ranked Positional Weight, Largest Candidate). Visual station-task assignment with efficiency %, balance delay, idle time per station. Complements existing Yamazumi (which shows imbalance but doesn't solve it) |
| T11 | **NIOSH Lifting Equation** | Enter task parameters (weight, distances, frequency, coupling). Compute RWL and Lifting Index with visual breakdown of each multiplier's contribution. Red/yellow/green risk indication. Standard mfg ergonomics tool |

---

## Tier 2 — Good Additions, Moderate Priority

| ID | Tool | Category | Why it adds value |
|----|------|----------|-------------------|
| T12 | **Control Chart Pattern Trainer** | Quality/Education | DSW *runs* SPC on real data. This *teaches* pattern recognition — generates synthetic data with injected shifts, trends, cycles, stratification. User identifies the pattern and applies Western Electric / Nelson rules. Gamified with score tracking |
| T13 | **Facility Center of Gravity** | Facility Planning | Given demand points with coordinates and volumes, compute weighted center of gravity. Interactive scatter plot — drag points, watch optimal location move. Simple, visual, frequently taught |
| T14 | **Decision Tree / EMV Calculator** | Operations Research | Build decision tree with chance nodes, decision nodes, payoffs, probabilities. Rollback to compute EMV at each node. Useful for make-vs-buy, capacity expansion, project selection decisions |
| T15 | **LP Visualizer (2-Variable)** | Operations Research | Plot constraints as lines, shade feasible region, drag iso-profit line to find optimal corner point. Purely educational — builds geometric intuition for LP. Transitions users to DSW for real optimization |
| T16 | **Project Crashing / Time-Cost Tradeoff** | Project | Extension of T7 — given crash costs and crash times per activity, find minimum-cost way to shorten project. Interactive Pareto: cost vs duration. Pairs with PERT/CPM |

---

## Tier 3 — Nice to Have

| ID | Tool | Category | Notes |
|----|------|----------|-------|
| T17 | Transportation Problem Solver | OR | NW Corner, Least Cost, VAM + MODI stepping stone. Step-by-step viz |
| T18 | Assignment Problem (Hungarian) | OR | Step-by-step row/column reduction visualization |
| T19 | Rough-Cut Capacity Planning (RCCP) | Planning | MPS vs capacity check. Pairs with T4 (MRP) |
| T20 | Facility Break-Even / Crossover | Facility | Total cost curves for location alternatives |
| T21 | Green/Lean Carbon Estimator | Sustainability | Per-unit carbon footprint from energy/material/waste inputs |
| T22 | Time Study / Standard Time | IE | Observed times + rating + allowances → standard time. Extends existing Cycle Time Study |

---

## Implementation Notes

- All tools are client-side JS + Plotly — no new backend endpoints needed
- Each tool should include: inputs, interactive visualization, results, derivation (collapsible math), "Next Steps" linking to related tools
- Simulators should have: Start/Pause/Reset/Speed controls, real-time animated visuals, summary metrics
- Cross-calculator data bus (`SvendOps.publish/pull`) should connect new tools to existing ones where natural:
  - T2 (PM Optimization) pulls from MTBF/MTTR
  - T3 (Spare Parts) pulls from MTBF/MTTR
  - T6 (Newsvendor) relates to EOQ/Safety Stock
  - T10 (Line Balancing) pulls from Takt Time + Yamazumi
  - T9 (Learning Curve) exports to capacity planning

## Priority Order (suggested build sequence)

1. **S1-S5** — Ship the "coming soon" tools (they're already promised in the UI)
2. **T1** (Bathtub Curve) — quick win, pure Plotly, strong educational payoff
3. **T9** (Learning Curve) — simple calculator, no web equivalent, high demand
4. **T7+T8** (PERT/CPM + Monte Carlo) — pair build, fills a platform gap
5. **T10** (Assembly Line Balancing) — complements Yamazumi, strong IE demand
6. **T4** (MRP Explosion) — complex but the single most valuable for APICS audience
7. **T5** (Aggregate Planning) — pairs with MRP, great simulation
8. **T6** (Newsvendor) — rounds out inventory category
9. **T2+T3** (PM Optimization + Spare Parts) — pair build, extends MTBF/MTTR
10. **T11** (NIOSH Lifting) — standalone, simple, valuable for mfg users
11. **T12** (SPC Pattern Trainer) — gamified education, differentiator
12. **T13-T16** — Tier 2 as capacity allows
