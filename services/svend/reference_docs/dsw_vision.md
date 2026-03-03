# DSW Vision — The Three Pillars

**Status:** Discussion draft
**Date:** 2026-03-01

---

## Doctrine

The DSW is not a "do everything for me" tool. It is the most rigorous, most transparent, and most insightful statistical analysis platform that exists. The user does the thinking. The platform ensures they're thinking with trustworthy numbers, clear visuals, and connected explanations.

Three pillars. Nothing else.

### Inviolable Constraints

1. **Every test is independent.** The Bayesian shadow does not gate, suppress, override, or alter the frequentist result. It appears alongside it. The user ran a t-test — they get a t-test. They also get insurance. Both are complete, both are trustworthy on their own terms.
2. **Nothing is diluted.** Adding the Bayesian shadow and evidence grade does not compress, hide, or de-emphasize any existing output — narrative, diagnostics, charts, summary, what-if explorers. The existing diagnostic chain (normality check → cross-validation → effect size → practical significance) is untouched. Insurance is additive.
3. **No auto-chaining.** The system does not automatically run follow-up analyses. It surfaces the Bayesian shadow as a parallel computation, not as a sequential "and then we also ran..." chain. The action buttons ("Run Wilcoxon", "Run Spearman") remain one-click choices, not auto-fired pipelines.
4. **The analyst decides.** We don't auto-select tests, auto-exclude points, or auto-interpret results. We provide more information, more clearly, so the analyst can decide better.

---

## Pillar 1: Coverage (Tests on Tests, Bayesian Insurance)

### The Problem

Statistical tools hand you a p-value and leave you alone with it. Was the test appropriate? Were the assumptions met? Is the effect real or just statistically significant? Is p = 0.048 meaningfully different from p = 0.052? The user has to know to ask these questions. Most don't.

Minitab shows you the p-value. JMP shows you the p-value with a red/green indicator. Neither tells you whether the p-value is *trustworthy*.

### What We Already Do

We're already further than anyone else here. The diagnostic system in `common.py` runs automatically on every test:

- **Assumption verification**: `_check_normality()` runs Shapiro-Wilk on every parametric test. `_check_equal_variance()` runs Levene's. `_check_outliers()` flags Grubbs.
- **Cross-validation**: `_cross_validate()` auto-runs the nonparametric alternative (Wilcoxon for t-test, Kruskal-Wallis for ANOVA, Spearman for Pearson) and compares conclusions. Contradictions are flagged explicitly.
- **Effect size emphasis**: `_practical_block()` and `_effect_magnitude()` categorize every result into the 2x2 of statistical significance × practical significance. "Significant but trivial" gets a warning. "Not significant but large effect" suggests collecting more data.
- **Power explorer**: Client-side power calculation for 5 test types. Shows whether the test had enough power to detect the observed effect.
- **Actionable diagnostics**: Each diagnostic has an action button — "Run Wilcoxon Signed-Rank", "Run Spearman Correlation" — that fires the alternative with one click.

### What's Missing: Bayesian Insurance

The gap: we run the frequentist test, we cross-validate with a nonparametric alternative, but we don't automatically provide the Bayesian counterpart. We have 10+ Bayesian analyses (`bayes_ttest`, `bayes_anova`, `bayes_correlation`, `bayes_regression`, `bayes_proportion`, `bayes_chi2`, `bayes_poisson`, `bayes_equivalence`...) — they just run as separate analyses the user has to know to request.

**The vision: every frequentist result automatically includes its Bayesian shadow.**

Not as the primary result. Not as a replacement. As *insurance*. A parallel computation that arrives async, rendering below the existing result without displacing anything. The frequentist test is the headline. The Bayesian result is the fine print that says "and here's how confident you should be in that headline."

Concretely:

| Frequentist Test | Bayesian Shadow | What It Adds |
|-----------------|----------------|--------------|
| One-sample t-test (p = 0.03) | Bayes Factor + posterior | "BF₁₀ = 8.2 — the data are 8× more likely under H₁ than H₀. 95% credible interval: [0.12, 0.89]" |
| Two-sample t-test (p = 0.07) | Bayes Factor + posterior difference | "BF₁₀ = 1.8 — weak evidence. The data don't strongly favor either hypothesis. Collect more data before deciding." |
| ANOVA (p = 0.001) | Bayesian ANOVA + inclusion BFs | "BF for including Factor A: 342. The effect is real. BF for interaction: 0.4 — no evidence for interaction." |
| Correlation (r = 0.45, p = 0.02) | Bayesian correlation + posterior on ρ | "95% credible interval for ρ: [0.08, 0.72]. Even the lower bound suggests a meaningful relationship." |
| Proportion test (p = 0.04) | Bayesian proportion + posterior | "P(π₁ > π₂ | data) = 0.97. The posterior probability that treatment is better is 97%." |

**Why this matters more than anything else:** It elevates the discourse. When a user sees "p = 0.048" they stop. When they see "p = 0.048, but BF₁₀ = 2.1 (anecdotal evidence)" they understand that the result is borderline. When they see "p = 0.001 AND BF₁₀ = 342 AND Cohen's d = 1.2 AND the nonparametric test agrees" — that's a result you can act on. No tool on earth does this automatically.

### The Full Coverage Stack

For every hypothesis test, the DSW produces (existing items marked, new items noted):

```
1. PRIMARY RESULT (unchanged, untouched)
   - Test statistic, p-value, confidence interval
   - Effect size with named magnitude (Cohen's d = 0.82, "large")

2. ASSUMPTION VERIFICATION (existing)
   - Normality: Shapiro-Wilk
   - Variance equality: Levene's
   - Outliers: Grubbs
   - Sample size adequacy: power for observed effect

3. CROSS-VALIDATION (existing + extension)
   - Nonparametric alternative result + agreement check (existing)
   - Bayesian shadow: BF + credible interval (NEW — async, additive)
   - Contradiction resolution extended to 3-way if all three disagree (NEW)

4. PRACTICAL SIGNIFICANCE (existing)
   - Effect size interpretation
   - 2×2 classification: stat sig × practical sig
   - "So what?" sentence in plain language (via narrative)

5. EVIDENCE GRADE (NEW — synthesis, not replacement)
   - Composite confidence: "Strong evidence" / "Moderate" / "Weak" / "Inconclusive"
   - Based on: agreement across all tests, effect size, sample adequacy
   - Displayed as a distinct visual element — not buried in diagnostics
   - Does NOT suppress or override any individual result
```

The **Evidence Grade** is the capstone. It synthesizes everything above into a verdict. But every individual piece remains visible and independently interpretable. The grade is a convenience, not a gatekeeper.

### Architecture: Async Bayesian Shadow

The Bayesian shadow runs **async** — the frequentist result renders immediately, the shadow arrives moments later and appends below without disrupting anything.

**Server-side flow:**
1. Frequentist analysis completes → returns full result immediately
2. Separately (async), `_bayesian_shadow()` in `common.py` calls the matching `bayes_*` function from `bayesian.py`
3. Returns a `bayesian_shadow` dict via a second lightweight endpoint or as a deferred field

**Client-side flow:**
1. `renderStatsOutput()` renders the full frequentist result as today
2. After render, fires an async fetch for the Bayesian shadow (same pattern as what-if sections: `setTimeout` → `insertAdjacentHTML('beforeend')`)
3. Shadow panel fades in below diagnostics, above charts — or in a dedicated zone
4. Uses `Plotly.addTraces()` if posterior visualization is needed (preserves existing chart state)

**Why async:** Bayesian computation (especially MCMC-based) can add 200-500ms. The frequentist result should never wait for it. The user sees their result instantly; the insurance appears a moment later.

---

## Pillar 2: Visualization (Interactive, Cross-Linked, Self-Documenting)

### The Problem

Charts in most tools are illustrations. They show you what you already computed. JMP made them interactive — brush, link, exclude. But even JMP's charts don't *explain themselves*. You see the pattern. You still have to figure out what it means.

### What We Already Do

- **Click-to-inspect**: SPC (16 charts), regression diagnostics (4 plots), correlation heatmap, clustering. Click → see observation details, Nelson rules, Cook's D, data rows.
- **What-if explorers**: Capability (LSL/USL sliders), regression (per-factor sliders), power (effect × n × α), Monte Carlo (threshold).
- **Rich hovertemplates**: Every chart has domain-specific hover — p-values on correlation cells, at-risk counts on KM censored marks, rule violations on SPC OOC points.
- **Narratives**: Every analysis has a verdict + body + chart guidance + next steps. Charts-first philosophy — the narrative explains the chart, not the other way around.

### What's Next

**The principle: every chart interaction should either answer a question or raise a better one.**

#### 2a. Linked Brushing (Answer: "What else is true about these points?")

Select points in one chart → same observations highlight in all sibling charts. This is JMP's core feature and the single biggest gap we have. We already have shared `customdata[0]` = observation index across all diagnostic plots. The wiring is Plotly's `plotly_selected` → `Plotly.restyle()` on siblings. Pure frontend, no backend change.

Scope: Start with regression 4-panel diagnostics. Extend to any multi-chart analysis (SPC I-chart + MR-chart, DOE effects + residuals).

#### 2b. Selection → Explanation (Answer: "What do these points have in common?")

Lasso-select a cluster of points. Button appears: "Explain." Claude examines the selected rows and returns: "These 12 points are all from Machine B, Shift 3, batch 2024-Q3-007."

This is the feature no desktop tool can offer. JMP highlights points. We *explain* them. It converts a visual observation into a verbal insight. This is where the LLM earns its keep — not generating boilerplate, but pattern-matching across columns to find what distinguishes a selection from the rest of the data.

#### 2c. Click-to-Exclude with Delta Display (Answer: "Does this point change my conclusion?")

Click an influential point → temporarily remove it → show how R², coefficients, and diagnostic metrics change. Display as a delta: "Removing obs #47: R² 0.82 → 0.91 (+0.09), slope 2.3 → 1.8 (−22%)."

Not just a visual change — a quantified impact statement. The user sees immediately whether the point matters or not.

#### 2d. Range Slider for Time-Ordered Data (Answer: "What happened during this window?")

Plotly's native `rangeslider` on SPC charts, time series, forecasts. One layout flag. Enables zoom-to-window for investigation.

#### 2e. Charts That Explain Themselves

Extend the `chart_guidance` field in narratives to be context-sensitive. Not "Points above UCL are out of control" (static text) but "3 of your 7 OOC points are in the range 45–52, which coincides with subgroups where column 'Batch' = 'B-2024-Q3'. This suggests a batch-specific issue rather than random variation."

This is the anomaly narration concept — the chart doesn't just show the pattern, it tells you what the pattern *means* in the context of your data. Generated by cross-referencing statistical findings with the dataset's categorical and temporal columns.

---

## Pillar 3: Cross-Linked Explanations (The Ecosystem Moat)

### The Problem

Statistical analysis is not a single event. It's a campaign. You run a t-test, then a regression, then an SPC chart, then a DOE. Each produces a result. In every tool on the market, those results are disconnected islands. You screenshot the SPC chart and paste it into your A3 report. You remember the regression R² and type it into your FMEA justification. The narrative — how the evidence accumulated, what each result contributed to the conclusion — lives only in the analyst's head.

### What We Have That Nobody Else Has

- **Synara**: A Bayesian belief engine that tracks hypotheses, accumulates evidence with likelihood ratios, and maintains probability histories. No statistical tool has anything like this.
- **Workbench artifacts**: Every analysis result is a persistent object linked to a project. Not a disposable output — a piece of evidence.
- **Quality tools**: RCA, FMEA, A3, Hoshin Kanri, VSM — all in the same platform, all with their own data models.
- **Whiteboard**: A knowledge graph for causal reasoning.
- **`Hypothesis.probability_history`**: Tracks how belief evolved over time as evidence accumulated.

### The Vision: Analysis Results as Evidence Nodes

Every analysis result should know:
- What hypothesis it tested (or could test)
- What it proved or disproved
- What it contradicts or confirms from previous results
- Where it fits in the investigation narrative

Concretely:

#### 3a. Selection → Hypothesis (One Click from Pattern to Belief)

Identify an interesting pattern (OOC cluster, outlier group, unexpected correlation). One click creates a hypothesis in Synara: "Machine B produces higher defect rates on Shift 3." The analysis result is automatically linked as evidence. The prior comes from the Bayes Factor. The narrative writes itself.

Infrastructure largely exists: `appendLinkHypothesisPrompt` in the frontend, `add_finding_to_problem()` in the backend. Need to extend to accept selection context and pre-fill the hypothesis.

#### 3b. SPC → RCA Pipeline (One Click from Signal to Investigation)

Click an OOC point. "Investigate" button opens a pre-populated RCA session with: timestamp, measurement, Nelson rules violated, chart type, column name. The quality engineer doesn't transcribe anything — the system already knows.

This closes the detection-to-action loop that every quality standard requires and every company does manually.

#### 3c. Evidence Accumulation Timeline (The Campaign View)

A visualization showing how confidence in each hypothesis evolved as evidence accumulated:

```
Day 1: t-test → P(H₁) = 0.65 (p = 0.03, BF = 4.2)
Day 3: Regression → P(H₁) = 0.82 (Machine B coefficient significant)
Day 5: SPC confirms → P(H₁) = 0.94 (OOC cluster matches Machine B periods)
Day 8: DOE validates → P(H₁) = 0.99 (Machine B root cause confirmed)
```

`Hypothesis.probability_history` already tracks this. The visualization is a line chart with annotations at each evidence event. The narrative is auto-generated from the linked results.

This is the killer deliverable for quality audits, management reviews, and regulatory submissions. "Here's the evidence trail. Here's how our confidence evolved. Here's why we're acting."

#### 3d. FMEA ↔ SPC Closed Loop

An FMEA row identifies a failure mode. The recommended control is a control chart. When the chart detects OOC, the FMEA's Occurrence score updates. RPN recalculates. The quality engineer sees the change.

This is the holy grail of IATF 16949 / AS9100 compliance. Everyone does it in spreadsheets. We can automate it because both FMEA and SPC live in the same system.

#### 3e. A3 Auto-Population from Analysis Trail

After a chain of linked analyses, one click assembles an A3 report: Problem Statement (OOC event), Current Condition (SPC chart), Root Cause (regression), Countermeasure (DOE), Verification (confirmation run). Charts embedded, evidence linked, narrative generated.

---

## What This Is NOT

- **Not an autopilot.** The user chooses the analysis. The system validates, enriches, and connects. We don't auto-select tests or make decisions.
- **Not a dashboard builder.** We're not competing with Tableau or Power BI. Every chart earns its place by answering a statistical question.
- **Not a reporting tool.** The A3 auto-population and evidence timeline are outputs of the investigation workflow, not the purpose of the platform.
- **Not trying to replace the analyst.** The analyst's judgment is irreplaceable. What we replace is the tedious, error-prone work of assumption checking, cross-validation, evidence tracking, and report assembly. The analyst should spend 100% of their time on *interpretation*, not *verification*.
- **Not auto-chaining.** The Bayesian shadow is a parallel computation, not a sequential follow-up. The nonparametric cross-validation is a check, not a replacement. Nothing fires without the user requesting it except the insurance layer.

---

## Risk Analysis

### R1: Chart Library — Stay with Plotly or Switch?

**Decision: Stay with Plotly 2.27.0. Do not switch.**

| Concern | Assessment |
|---------|-----------|
| Linked brushing | No built-in cross-chart linking, but `plotly_selected` + `Plotly.restyle({selectedpoints: [[indices]]})` works. Manual wiring required — ~200 lines of JS for the event bus. This is a software design problem, not a library limitation. |
| Async chart updates | `Plotly.addTraces()` adds traces without re-rendering. `Plotly.react()` diffs efficiently. Both preserve user zoom/pan state. Proven pattern — all 5 what-if explorers already use `Plotly.react()`. |
| Selection events | `plotly_selected` returns `pointNumber` for each selected point. Works on scatter, scattergl, bar, box, histogram. Does NOT work on heatmap (must be programmatic) or scatter3d. |
| Range slider | Native `xaxis.rangeslider.visible = true`. One gotcha: y-axis doesn't auto-rescale (need manual `plotly_relayout` listener to fix). |
| WebGL | `scattergl` supports all events including lasso select. Threshold: useful above 10K points. Limit: 8-16 WebGL contexts per page (4-8 charts max with WebGL). Not needed at our current data sizes (<5K typical). |
| Performance | 4+ charts with event listeners is fine for selection-driven updates (not streaming). Debounce at 50ms, use `requestAnimationFrame` to coalesce restyle calls. Guard against circular event loops. |

**Risk of switching:** High. 200+ analyses generate Plotly trace dicts. Every backend module returns `{"type": "scatter", "x": [...], ...}`. Switching to Vega-Lite or ECharts means rewriting every trace in every analysis — thousands of lines across 8 Python modules. The gain (declarative cross-chart linking) does not justify the cost.

**Risk of staying:** Low. Every vision feature is implementable with Plotly's existing API. The missing piece is a ~200-line event bus for cross-chart coordination — a one-time frontend investment.

### R2: UI Layout — Making Space for Insurance

**Current layout:** Narrative → Diagnostics → Charts → Summary (collapsible) → What-if sections (async appended).

**What needs space:**
- Bayesian shadow panel (async, below diagnostics or below charts)
- Evidence grade badge (prominent, near narrative verdict)
- Linked brushing toolbar (selection mode toggle, clear selection)
- Inspect panels already work (positioned inside `.stats-plot` containers)

**Risk:** Visual clutter. The current output is already dense: narrative + diagnostics + 4 charts + summary + what-if explorer. Adding Bayesian shadow + evidence grade without redesign could overwhelm.

**Mitigation:**
- Evidence grade: compact badge/chip next to the narrative verdict, not a new section. One line: `Strong Evidence ● p < 0.001, BF₁₀ = 342, d = 1.2 (large), all tests agree`
- Bayesian shadow: collapsible panel (like `.dsw-details` toggle). Starts collapsed with one-line summary: `Bayesian: BF₁₀ = 8.2 (substantial), 95% CrI [0.12, 0.89]`. Expand for posterior plot and full interpretation.
- Progressive disclosure: insurance details hidden until the user wants them. The evidence grade is the visible surface; the shadow is the expandable depth.
- No new scrolling layers. Everything flows in the existing `.output-pane` scroll context.

**Required CSS additions:** ~40 lines for `.dsw-evidence-grade` badge, `.dsw-bayesian-shadow` collapsible panel. No structural HTML changes to the output block template.

### R3: Async Architecture — Bayesian Shadow Delivery

**Option A: Single endpoint, deferred field.**
The existing `/api/dsw/analysis/` endpoint returns the frequentist result immediately but fires the Bayesian computation in a background thread. A second lightweight endpoint `/api/dsw/analysis/shadow/` returns the shadow when ready. Frontend polls or uses SSE.

**Option B: Two-phase response.**
Return the frequentist result synchronously. Include a `shadow_task_id` in the response. Frontend calls `/api/dsw/analysis/shadow/{task_id}/` after render. If ready, returns immediately. If not, returns 202 and frontend retries once after 500ms.

**Option C: Inline with timeout.**
Attempt the Bayesian computation inline with a 300ms timeout. If it completes, include it in the response. If not, return without it and let the frontend fetch it separately.

**Recommendation: Option B.** Cleanest separation. The frequentist result is never delayed. The shadow arrives when it arrives. The existing `setTimeout` → `insertAdjacentHTML` pattern handles the frontend perfectly — same as what-if sections.

**Risk:** Additional API endpoint. Additional server load (every analysis now triggers two computations). Mitigation: Bayesian shadow computations are lightweight (our `bayes_ttest` etc. use conjugate priors, not MCMC — they complete in <50ms for typical datasets). The "async" is more about clean separation than actual latency.

### R4: Linked Brushing — Cross-Chart State Management

**The problem:** Plotly charts are independent DOM elements with no shared state. Selection on Chart A needs to propagate to Charts B, C, D.

**Architecture: Chart Registry + Event Bus.**

```
DSWChartRegistry {
    groups: {
        "output-5": [
            { el: <div#plot-5-0>, plot: {...}, type: "regression_diagnostic" },
            { el: <div#plot-5-1>, plot: {...}, type: "regression_diagnostic" },
            ...
        ]
    },

    onSelect(sourceEl, selectedIndices) {
        const group = this.getGroup(sourceEl);
        group.forEach(chart => {
            if (chart.el !== sourceEl) {
                Plotly.restyle(chart.el, {selectedpoints: [selectedIndices]}, [0]);
            }
        });
    }
}
```

**Risks:**
- Circular event loops: Chart A's `plotly_selected` updates Chart B → Chart B fires its own `plotly_selected` → infinite loop. **Mitigation:** Set a `_brushing` flag during programmatic updates. Check flag in event handler.
- `selectedpoints` API quirk: Must pass as `[[indices]]` (nested array), not `[indices]`. Flat array is misinterpreted as per-trace distribution. Well-documented gotcha.
- Lazy-rendered charts: Grouped/tabbed charts may not exist in DOM when selection fires. **Mitigation:** Store pending selection state in registry. Apply when chart renders on tab switch.
- Non-scatter traces: Histograms, heatmaps don't support lasso selection. **Mitigation:** Only attach `plotly_selected` to scatter-type traces. Other chart types respond to programmatic `selectedpoints` but don't source selections.

**Estimated code:** ~200 lines JS for the registry + event bus. No backend changes.

### R5: Backwards Compatibility — 200+ Analyses Must Keep Working

**Principle:** Every change is additive. No existing field renamed or removed.

| Change | Backwards Impact |
|--------|-----------------|
| `plot.interactive` property | Charts without it behave identically — `attachChartInteractivity` returns immediately on undefined. Already proven with current deployment. |
| `bayesian_shadow` response key | Frontend ignores unknown keys. If the shadow section renderer isn't present, nothing breaks. |
| `evidence_grade` response key | Same — additive key, ignored if frontend doesn't handle it. |
| `selectedpoints` on traces | Only set by the event bus during active brushing. Defaults to no selection (Plotly's default). |
| Range slider on SPC charts | Layout property. Existing charts without it are unaffected. New charts opt in via `layout.xaxis.rangeslider`. |
| Async shadow endpoint | New endpoint. Existing endpoint unchanged. |

**Risk:** Low. The additive-only constraint is already proven — the interactivity work (Phases 1-6) added `interactive` flags and `customdata` to all SPC/regression/correlation/clustering/KM charts with zero regressions.

### R6: LLM Dependency — Claude in the Analysis Loop

**Where Claude is invoked:**
- Selection → Explanation (Pillar 2b): User-initiated, not automatic
- Anomaly narration (Pillar 2e): Could be automatic, but should be opt-in
- A3 auto-population (Pillar 3e): User-initiated

**Risks:**
- Latency: Claude calls add 1-3 seconds. **Mitigation:** Always async. Never block the primary result.
- Cost: Each explanation call costs ~$0.01-0.05. **Mitigation:** Rate limit per user. Only fire on explicit user action (click "Explain"), not on every selection.
- Hallucination: Claude might claim patterns that don't exist. **Mitigation:** The prompt should include the actual data rows and instruct Claude to only state patterns verifiable in the provided data. Cross-reference against summary statistics before rendering.
- Availability: Claude API outage shouldn't break analysis. **Mitigation:** LLM features are enhancement layers. Core analysis (Pillars 1-2) works without them. Graceful degradation: if Claude is unavailable, the "Explain" button shows "Unavailable" instead of failing silently.

**Risk:** Low-medium. LLM features are opt-in and non-blocking. The core value proposition (coverage + visualization) works entirely without Claude.

### R7: Scope — Three Pillars Is a Lot of Work

**Total estimated scope:**

| Phase | Items | Effort | Dependencies |
|-------|-------|--------|-------------|
| A1: `_bayesian_shadow()` function | 1 function in common.py | Small | Existing `bayes_*` functions |
| A2: Wire shadow into 7 frequentist tests | 7 call sites in stats.py | Medium | A1 |
| A3: `_evidence_grade()` function | 1 function in common.py | Small | A1, A2 |
| A4: Async shadow endpoint | 1 new view | Small | A1 |
| A5: Frontend shadow renderer | ~100 lines JS + CSS | Small | A4 |
| A6: Evidence grade badge | ~50 lines JS + CSS | Small | A3 |
| B1: Chart registry + event bus | ~200 lines JS | Medium | None |
| B2: Linked brushing (regression) | ~100 lines JS | Medium | B1 |
| B3: Range slider for SPC/timeseries | Layout flag per chart | Trivial | None |
| B4: Selection → Explanation | ~150 lines JS + API call | Medium | B1, Claude |
| B5: Click-to-exclude | ~200 lines JS + server call | Medium | B1 |
| C1: Selection → Hypothesis | ~100 lines JS + API | Small | Synara integration |
| C2: SPC → RCA pipeline | ~80 lines JS + API | Small | RCA views |
| C3: Evidence timeline | ~300 lines JS + Plotly | Medium | Hypothesis model |
| C4: FMEA ↔ SPC | ~200 lines backend | Medium-Hard | FMEA + SPC models |
| C5: A3 auto-population | ~200 lines backend + LLM | Medium | A3 views + Claude |

**Mitigation:** Ship in phases. Each phase is independently deployable and revertable. Phase A (Bayesian insurance) delivers the coverage pillar. Phase B (visualization) delivers the interactivity pillar. Phase C (ecosystem) delivers the moat. Any phase can ship alone and provide value.

---

## Implementation Sequence

### Phase A: Bayesian Insurance (Pillar 1 completion)

1. Add `_bayesian_shadow()` to `common.py` — takes test type + data, calls matching `bayes_*` function, returns BF + credible interval + interpretation string
2. Add `_evidence_grade()` to `common.py` — pure function of p-value, BF, effect size magnitude, cross-validation agreement, power → returns grade label + one-line rationale
3. Wire `_bayesian_shadow()` into 7 frequentist tests (one-sample t, two-sample t, paired t, ANOVA, correlation, proportion, chi-square) — additive `bayesian_shadow` key in result dict
4. Add async shadow endpoint or inline the shadow into the existing response (since our Bayesian functions use conjugate priors, not MCMC, they're fast enough to run inline with minimal latency)
5. Frontend: render evidence grade as compact badge near narrative verdict
6. Frontend: render Bayesian shadow as collapsible panel (collapsed by default, one-line summary visible)

### Phase B: Visualization Cross-Linking (Pillar 2 gaps)

1. Chart registry + event bus (~200 lines JS)
2. Linked brushing for regression 4-panel diagnostics
3. Range slider for time-ordered charts (SPC, forecasts)
4. Selection → Explanation (Claude integration, user-initiated)
5. Click-to-exclude with delta display (regression)

### Phase C: Ecosystem Integration (Pillar 3)

1. Selection → Hypothesis (Synara link)
2. SPC → RCA pipeline
3. Evidence accumulation timeline
4. FMEA ↔ SPC closed loop
5. A3 auto-population

---

## The Competitive Narrative

**Minitab:** Shows you the p-value.
**JMP:** Shows you the p-value and lets you brush the chart.
**Svend:** Shows you the p-value, tells you whether to trust it, gives you the Bayesian alternative, flags when your assumptions are violated, cross-validates with the nonparametric backup, quantifies the practical significance, links the finding to your hypothesis, and connects it to your investigation — all automatically, all independently verifiable, all for 1/5th the price.

Coverage. Visualization. Cross-linked explanations. That's it. That's the moat.
