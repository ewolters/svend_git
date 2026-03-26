"""Learning content: Advanced Methods."""

from ._datasets import SHARED_DATASET  # noqa: F401

NONPARAMETRIC_TESTS = {
    "title": "Nonparametric Tests",
    "intro": "When your data is skewed, ordinal, or full of outliers, parametric tests break down. You'll learn the four core nonparametric tests and when to reach for each one.",
    "exercise": {
        "title": "Try It: Compare Two Suppliers",
        "steps": [
            "Load the supplier burst pressure data",
            "Run a Mann-Whitney U test comparing Supplier A vs B",
            "Check the rank-biserial correlation for effect size",
            "Compare medians between groups",
            "Try running a t-test on the same data and compare conclusions",
        ],
        "dsw_type": "stats:mann_whitney",
        "dsw_config": {"var": "diameter_mm", "group_var": "shift"},
    },
    "sample_data": {
        "supplier_a_burst_psi": [
            245,
            312,
            198,
            287,
            256,
            301,
            223,
            278,
            234,
            267,
            289,
            203,
            345,
            256,
            221,
            290,
            276,
            248,
            310,
            235,
        ],
        "supplier_b_burst_psi": [
            278,
            334,
            289,
            312,
            298,
            356,
            267,
            301,
            334,
            288,
            321,
            299,
            367,
            312,
            278,
            345,
            310,
            289,
            356,
            298,
        ],
        "shift_a_roughness": [
            2.3,
            1.8,
            2.1,
            3.5,
            1.9,
            2.0,
            2.4,
            1.7,
            2.8,
            1.6,
            2.2,
            3.1,
            1.9,
            2.5,
            2.0,
        ],
        "shift_b_roughness": [
            2.8,
            3.1,
            2.5,
            3.9,
            2.7,
            3.3,
            2.9,
            3.5,
            2.6,
            3.0,
            4.1,
            2.8,
            3.2,
            2.9,
            3.4,
        ],
        "shift_c_roughness": [
            2.1,
            1.9,
            2.3,
            2.0,
            1.8,
            2.5,
            2.2,
            1.7,
            2.4,
            2.1,
            1.9,
            2.6,
            2.0,
            2.3,
            1.8,
        ],
    },
    "content": """
## When Assumptions Break Down

Parametric tests (t-tests, ANOVA) assume your data is normally distributed with equal variances. In practice, these assumptions frequently fail:

- **Ordinal data** (Likert scales, ratings) — no true interval scale
- **Heavily skewed distributions** — income, response times, defect counts
- **Small samples** where normality can't be verified
- **Outliers** that can't be legitimately removed
- **Ranked data** — tournament results, preference orderings

Nonparametric tests make fewer assumptions — they work on **ranks** rather than raw values. The cost? Slightly less statistical power when parametric assumptions hold.

### The Tradeoff

| | Parametric | Nonparametric |
|---|---|---|
| **Assumptions** | Normality, equal variance | Few or none |
| **Power** | Higher when assumptions hold | ~95% of parametric when n > 20 |
| **Data types** | Continuous, interval/ratio | Ordinal, ranked, skewed continuous |
| **Outlier sensitivity** | High | Low (uses ranks) |

## The Big Four Nonparametric Tests

### 1. Mann-Whitney U Test (Two Independent Groups)

The nonparametric equivalent of the independent two-sample t-test. Tests whether one group tends to have larger values than the other.

**How it works:** Rank all observations from both groups together. If one group consistently ranks higher, the groups differ.

**Example:** Comparing customer satisfaction scores (1-5 scale) between two store locations. You can't assume normality with ordinal data.

**In DSW:** Select `mann_whitney` analysis type, provide two columns of data.

**Interpretation:** The U statistic represents the number of times a value from Group A beats a value from Group B. P-value tells you if the rank difference is beyond chance.

### 2. Wilcoxon Signed-Rank Test (Paired/Matched Data)

The nonparametric equivalent of the paired t-test. Tests whether paired differences tend to be positive or negative.

**How it works:** Calculate differences, rank the absolute differences, then compare sums of positive vs negative ranks.

**Example:** Before/after measurements where differences are skewed. Patient pain scores before and after treatment.

**In DSW:** Select `wilcoxon` analysis type, provide paired columns.

### 3. Kruskal-Wallis H Test (Three+ Independent Groups)

The nonparametric equivalent of one-way ANOVA. Tests whether any group differs from the others.

**How it works:** Like Mann-Whitney but extended to k groups. Ranks all observations, checks if mean ranks differ across groups.

**Example:** Comparing defect rates across 4 production shifts when data is highly skewed.

**In DSW:** Select `kruskal` analysis type, provide data and group column.

**Follow-up:** If significant, use Dunn's test for pairwise comparisons (the nonparametric Tukey).

### 4. Friedman Test (Repeated Measures, 3+ Conditions)

The nonparametric equivalent of repeated-measures ANOVA. Tests whether conditions differ when the same subjects are measured multiple times.

**How it works:** Rank within each subject, then check if mean ranks differ across conditions.

**Example:** Five judges rank three products. Does product ranking differ systematically?

**In DSW:** Select `friedman` analysis type, provide subject and condition columns.

## Post-Hoc Tests

When Kruskal-Wallis is significant, you need pairwise comparisons:

- **Dunn's test** — The standard nonparametric post-hoc. Adjusts for multiple comparisons. Use when group sizes are unequal.
- **Games-Howell** — Works when you can't assume equal variances across groups. More conservative but safer.

Both available in DSW: `dunn` and `games_howell` analysis types.

## Decision Guide: Parametric or Nonparametric?

Ask these questions in order:

1. **Is your data ordinal or ranked?** → Nonparametric (no debate)
2. **Is n < 15 per group?** → Nonparametric (can't verify normality)
3. **Is data heavily skewed (skewness > 1)?** → Nonparametric
4. **Do you have extreme outliers you can't remove?** → Nonparametric
5. **Are variances very unequal (ratio > 3:1)?** → Nonparametric or Welch's t-test
6. **None of the above?** → Parametric is fine

### Common Mistake

Don't test for normality and then decide. The Shapiro-Wilk test is too sensitive with large samples (always rejects) and too weak with small samples (never rejects). Use visual inspection (histogram, Q-Q plot) combined with domain knowledge.
""",
    "interactive": {
        "type": "nonparametric_demo",
        "config": {
            "title": "Try It: Mann-Whitney U Test",
            "description": "Compare burst pressure (psi) between two suppliers. Data is right-skewed — perfect for nonparametric testing.",
            "group_a_label": "Supplier A",
            "group_b_label": "Supplier B",
            "group_a": [
                245,
                312,
                198,
                287,
                256,
                301,
                223,
                278,
                234,
                267,
                289,
                203,
                345,
                256,
                221,
                290,
                276,
                248,
                310,
                235,
            ],
            "group_b": [
                278,
                334,
                289,
                312,
                298,
                356,
                267,
                301,
                334,
                288,
                321,
                299,
                367,
                312,
                278,
                345,
                310,
                289,
                356,
                298,
            ],
            "metric": "Burst Pressure (psi)",
        },
    },
    "key_takeaways": [
        "Nonparametric tests use ranks, not raw values — resistant to outliers and skew",
        "Mann-Whitney = two independent groups; Wilcoxon = paired/matched data",
        "Kruskal-Wallis = 3+ groups; Friedman = repeated measures",
        "Power loss is small (~5%) when n > 20, but assumptions violations cause bigger problems",
        "Don't test normality to decide — use domain knowledge and visual inspection",
    ],
    "practice_questions": [
        {
            "question": "A factory measures scratch counts on 30 items from Line A and 25 from Line B. Scratch counts are heavily right-skewed (most items have 0-2 scratches, a few have 10+). Which test and why?",
            "answer": "Mann-Whitney U test. Two independent groups with count data that's heavily right-skewed. A t-test would be dominated by the few high-scratch outliers. The Mann-Whitney compares whether one line tends to produce more scratches by working with ranks, which neutralizes the skew.",
            "hint": "Count data with heavy right skew violates normality assumption",
        },
        {
            "question": "After a Kruskal-Wallis test shows H=14.2, p=0.003 across 4 groups, a colleague runs 6 separate Mann-Whitney tests for all pairwise comparisons. What's wrong with this approach?",
            "answer": "Running 6 separate Mann-Whitney tests inflates the familywise error rate (6 × 0.05 = 30% chance of at least one false positive). The correct approach is Dunn's test, which is specifically designed for post-hoc pairwise comparisons after Kruskal-Wallis and includes an adjustment for multiple comparisons (typically Bonferroni or Holm).",
            "hint": "Multiple comparisons problem — same issue as running multiple t-tests after ANOVA",
        },
    ],
}


TIME_SERIES_ANALYSIS = {
    "title": "Time Series Analysis",
    "intro": "Time series data has memory: today depends on yesterday. Standard tests assume independence and fail here. You'll decompose trends, identify seasonality, and fit ARIMA models.",
    "exercise": {
        "title": "Try It: Decompose a Time Series",
        "steps": [
            "Load the monthly sales data",
            "Run decomposition with period=12",
            "Identify the trend, seasonal, and residual components",
            "Check the ACF/PACF of the residuals",
            "Determine if the series needs differencing for stationarity",
        ],
        "dsw_type": "stats:descriptive",
    },
    "sample_data": {
        "monthly_sales": [
            112,
            118,
            132,
            129,
            121,
            135,
            148,
            148,
            136,
            119,
            104,
            118,
            115,
            126,
            141,
            135,
            125,
            149,
            170,
            170,
            158,
            133,
            114,
            140,
            145,
            150,
            178,
            163,
            152,
            191,
            210,
            209,
            183,
            159,
            136,
            168,
        ],
        "months": [
            "Jan-Y1",
            "Feb-Y1",
            "Mar-Y1",
            "Apr-Y1",
            "May-Y1",
            "Jun-Y1",
            "Jul-Y1",
            "Aug-Y1",
            "Sep-Y1",
            "Oct-Y1",
            "Nov-Y1",
            "Dec-Y1",
            "Jan-Y2",
            "Feb-Y2",
            "Mar-Y2",
            "Apr-Y2",
            "May-Y2",
            "Jun-Y2",
            "Jul-Y2",
            "Aug-Y2",
            "Sep-Y2",
            "Oct-Y2",
            "Nov-Y2",
            "Dec-Y2",
            "Jan-Y3",
            "Feb-Y3",
            "Mar-Y3",
            "Apr-Y3",
            "May-Y3",
            "Jun-Y3",
            "Jul-Y3",
            "Aug-Y3",
            "Sep-Y3",
            "Oct-Y3",
            "Nov-Y3",
            "Dec-Y3",
        ],
    },
    "content": """
## Data With Memory

Time series data is fundamentally different from cross-sectional data: **observations are not independent.** Today's value depends on yesterday's. This breaks the core assumption of most statistical tests and requires specialized methods.

### Why Time Series Is Different

| Cross-Sectional | Time Series |
|---|---|
| Observations independent | Observations correlated |
| Order doesn't matter | Order is everything |
| One snapshot | Process over time |
| i.i.d. assumption | Autocorrelation structure |

## Key Concepts

### Stationarity

A time series is **stationary** if its statistical properties (mean, variance, autocorrelation) don't change over time. Most methods require stationarity.

**Non-stationary signals:**
- **Trend** — systematic upward or downward movement
- **Seasonality** — repeating patterns at fixed intervals
- **Changing variance** — heteroscedasticity over time

**How to achieve stationarity:**
- Differencing (first difference removes trend; seasonal difference removes seasonality)
- Log transformation (stabilizes variance)
- Detrending (subtract fitted trend)

### Autocorrelation

The correlation between a series and its lagged values. If temperature today is 30°C, tomorrow is likely around 30°C too — that's positive autocorrelation.

- **ACF (Autocorrelation Function)** — correlation at each lag
- **PACF (Partial ACF)** — correlation at lag k after removing effects of lags 1 to k-1

**In DSW:** `acf_pacf` analysis type shows both plots, essential for model identification.

### Decomposition

Every time series can be decomposed into components:

**Series = Trend + Seasonal + Residual** (additive)
**Series = Trend × Seasonal × Residual** (multiplicative)

**In DSW:** `decomposition` analysis type. Use additive when seasonal amplitude is constant; multiplicative when it grows with the level.

## ARIMA Models

The workhorse of time series forecasting: **AutoRegressive Integrated Moving Average.**

### Components

- **AR(p)** — AutoRegressive: current value depends on p previous values
- **I(d)** — Integrated: d differences needed for stationarity
- **MA(q)** — Moving Average: current value depends on q previous forecast errors

ARIMA(p,d,q) notation: ARIMA(1,1,1) means AR(1), one difference, MA(1).

### Model Identification (Box-Jenkins Method)

1. **Plot the series** — look for trend, seasonality, changing variance
2. **Difference until stationary** — this gives you d
3. **Examine ACF/PACF of differenced series:**
   - ACF cuts off at lag q → MA(q) term
   - PACF cuts off at lag p → AR(p) term
   - Both decay gradually → mixed ARMA model
4. **Fit model, check residuals** — residuals should be white noise
5. **Compare models with AIC/BIC** — lower is better

**In DSW:** `arima` analysis type handles model fitting. For seasonal data, use `sarima` which adds seasonal terms: SARIMA(p,d,q)(P,D,Q)s.

### Seasonal ARIMA (SARIMA)

Extends ARIMA for data with repeating seasonal patterns:

SARIMA(p,d,q)(P,D,Q)s where:
- (p,d,q) = non-seasonal terms
- (P,D,Q) = seasonal terms
- s = seasonal period (12 for monthly, 4 for quarterly, 7 for daily-weekly)

**Example:** Monthly ice cream sales with yearly seasonality → try SARIMA(1,1,1)(1,1,1)12

## Change Point Detection

Identifies moments where the underlying process shifted — a new mean, new variance, or new trend.

**Applications:**
- Manufacturing: when did the process shift?
- Clinical: when did a treatment start working?
- Business: when did the market change?

**In DSW:** `changepoint` analysis type. Also available as `bayes_changepoint` for Bayesian change point detection with uncertainty quantification.

## Granger Causality

Tests whether one time series helps predict another. **Not true causation** — it's predictive causation: does knowing X's past improve forecasts of Y beyond Y's own past?

**Example:** Do raw material prices Granger-cause finished goods prices? If material costs today predict product prices tomorrow (beyond what product price history alone predicts), that's Granger causality.

**In DSW:** `granger` analysis type. Requires choosing lag length — use AIC to select.

**Caution:** Granger causality can fail with:
- Confounders (Z causes both X and Y with different lags)
- Contemporaneous effects (X and Y move together, no lead/lag)
- Non-stationarity (spurious regression)

## Multi-Vari Analysis

A manufacturing-focused technique that visualizes variation across multiple factors simultaneously. Shows how variation is distributed across:
- Within-piece variation
- Piece-to-piece variation
- Time-to-time variation

**In DSW:** `multi_vari` analysis type. Essential for identifying the dominant source of variation before designing experiments.
""",
    "interactive": {
        "type": "timeseries_demo",
        "config": {
            "title": "Try It: Time Series Decomposition",
            "description": "Explore 3 years of monthly airline passenger data. Identify trend, seasonality, and residuals.",
            "data": [
                112,
                118,
                132,
                129,
                121,
                135,
                148,
                148,
                136,
                119,
                104,
                118,
                115,
                126,
                141,
                135,
                125,
                149,
                170,
                170,
                158,
                133,
                114,
                140,
                145,
                150,
                178,
                163,
                152,
                191,
                210,
                209,
                183,
                159,
                136,
                168,
            ],
            "labels": [
                "Jan-Y1",
                "Feb-Y1",
                "Mar-Y1",
                "Apr-Y1",
                "May-Y1",
                "Jun-Y1",
                "Jul-Y1",
                "Aug-Y1",
                "Sep-Y1",
                "Oct-Y1",
                "Nov-Y1",
                "Dec-Y1",
                "Jan-Y2",
                "Feb-Y2",
                "Mar-Y2",
                "Apr-Y2",
                "May-Y2",
                "Jun-Y2",
                "Jul-Y2",
                "Aug-Y2",
                "Sep-Y2",
                "Oct-Y2",
                "Nov-Y2",
                "Dec-Y2",
                "Jan-Y3",
                "Feb-Y3",
                "Mar-Y3",
                "Apr-Y3",
                "May-Y3",
                "Jun-Y3",
                "Jul-Y3",
                "Aug-Y3",
                "Sep-Y3",
                "Oct-Y3",
                "Nov-Y3",
                "Dec-Y3",
            ],
            "period": 12,
        },
    },
    "key_takeaways": [
        "Time series observations are NOT independent — standard tests don't apply directly",
        "Stationarity is required by most methods — achieve it via differencing and transformation",
        "ACF/PACF plots are the diagnostic tools for model identification",
        "ARIMA(p,d,q): AR = past values, I = differencing, MA = past errors",
        "Change point detection finds when a process shifted — critical for manufacturing and clinical",
        "Granger causality tests prediction, not true causation",
    ],
    "practice_questions": [
        {
            "question": "Monthly sales data shows a clear upward trend and higher variance in recent years. You difference once and the ACF of the differenced series shows a significant spike at lag 1 then cuts off. PACF decays gradually. What ARIMA model fits?",
            "answer": "ARIMA(0,1,1). The single differencing (d=1) removes the trend. The ACF cutting off at lag 1 suggests an MA(1) term. PACF decaying gradually is consistent with an MA model (not AR). You might also try a log transformation first to stabilize the increasing variance, then fit ARIMA(0,1,1) to the log-transformed series.",
            "hint": "ACF cutoff → MA order; PACF cutoff → AR order. One difference = d=1",
        },
        {
            "question": "A control chart shows a process running in control for 6 months, then a shift. Your manager wants to re-estimate control limits using all 6 months plus the shifted data. Why is this wrong?",
            "answer": "Including the shifted data contaminates the baseline estimates. Control limits should be estimated from the stable, in-control period only. Including the shift inflates the estimated standard deviation, making the limits too wide and potentially hiding the shift (and future shifts). Use change point detection to identify exactly when the shift occurred, then estimate limits from the pre-shift data only.",
            "hint": "Control limits estimate the voice of the process — only in-control data should be used",
        },
    ],
}


SURVIVAL_RELIABILITY = {
    "title": "Survival & Reliability Analysis",
    "intro": "How long until something fails? Survival analysis handles the unique challenge of censored data, where you know something lasted at least X hours but not how long it will ultimately last.",
    "exercise": {
        "title": "Try It: Build a Kaplan-Meier Curve",
        "steps": [
            "Load the bearing failure data (50 bearings, 20 failed, 30 censored)",
            "Plot the Kaplan-Meier survival curve",
            "Find the median survival time",
            "Note how censored observations appear as tick marks",
            "Estimate the B10 life (time for 10% failure)",
        ],
        "dsw_type": "stats:descriptive",
    },
    "sample_data": {
        "bearing_failure_hours": [
            120,
            245,
            310,
            389,
            412,
            456,
            498,
            534,
            567,
            601,
            623,
            678,
            712,
            745,
            789,
            823,
            856,
            890,
            934,
            967,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
        ],
        "bearing_censored": [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        "note": "1=failed, 0=still running at 1000hrs (censored). 20 of 50 bearings failed.",
    },
    "content": """
## Time-to-Event Data

Survival analysis (in medicine) and reliability analysis (in engineering) answer the same question: **how long until something happens?**

- How long until a patient relapses?
- How long until a machine fails?
- How long until a customer churns?
- How long until a light bulb burns out?

### Why Not Just Use Averages?

The critical challenge is **censoring**: you often don't observe the event for everyone.

- **Right censoring:** Patient is still alive when the study ends (you know they survived *at least* this long, but not how long they'll live)
- **Left censoring:** Failure happened before observation started
- **Interval censoring:** Event happened between two observation times

Ignoring censoring produces **biased results.** Dropping censored observations underestimates survival time. Including them as events overestimates failure rates.

## Kaplan-Meier Survival Curves

The fundamental tool for visualizing survival data.

### How It Works

At each event time:
1. Count subjects at risk (still alive and under observation)
2. Count events (deaths, failures)
3. Calculate conditional survival: S(t) = (at risk - events) / at risk
4. Multiply all conditional probabilities: cumulative S(t)

The result is a **step function** that decreases at each event time.

### Reading a KM Curve

- **Y-axis:** Probability of surviving past time t (starts at 1.0)
- **X-axis:** Time
- **Steps down:** Events occurred
- **Tick marks (|):** Censored observations
- **Median survival:** Time at which S(t) = 0.5
- **Confidence bands:** Wider = fewer subjects at risk

### Comparing Groups

The **log-rank test** compares survival curves between groups (e.g., treatment vs. control).

Null hypothesis: survival curves are identical.

**In DSW:** `kaplan_meier` analysis type. Provide time column, event indicator (1=event, 0=censored), and optional group column.

## Weibull Distribution

The go-to distribution for reliability engineering. Models time-to-failure with a **shape parameter** (β) that captures failure patterns:

- **β < 1:** Decreasing failure rate (infant mortality — early failures, then stability)
- **β = 1:** Constant failure rate (exponential distribution — random failures)
- **β > 1:** Increasing failure rate (wear-out — failures increase over time)

### Reliability Metrics from Weibull

- **MTTF (Mean Time to Failure):** Average lifetime
- **B10 life:** Time by which 10% of units fail (common warranty metric)
- **Reliability at time t:** R(t) = P(surviving past t)
- **Hazard function:** Instantaneous failure rate at time t

**In DSW:** `weibull` analysis type. Provide failure times and censoring indicators. Returns shape/scale parameters, reliability plots, and hazard function.

### Bathtub Curve

Real products often show all three phases:
1. **Infant mortality** (β < 1) — manufacturing defects, burn-in period
2. **Useful life** (β ≈ 1) — random failures
3. **Wear-out** (β > 1) — aging, fatigue, degradation

This is the classic "bathtub curve" and explains why burn-in testing and preventive maintenance are both necessary.

## Cox Proportional Hazards Model

The "regression" of survival analysis. Answers: **which factors affect survival time?**

### How It Works

Models the hazard (instantaneous risk) as a function of covariates:

h(t|X) = h₀(t) × exp(β₁X₁ + β₂X₂ + ...)

- **h₀(t):** Baseline hazard (unspecified — that's the "semi-parametric" part)
- **exp(βᵢ):** Hazard ratio for variable i
- **Hazard ratio > 1:** Increases risk (shortens survival)
- **Hazard ratio < 1:** Decreases risk (lengthens survival)

### Interpreting Hazard Ratios

HR = 2.0 means **twice the risk** at any point in time.
HR = 0.5 means **half the risk** at any point in time.

**Example:** Treatment group has HR = 0.65 vs. control → treatment reduces the hazard by 35% at any time point.

### The Proportional Hazards Assumption

The model assumes hazard ratios are **constant over time.** If treatment helps early but not late, the model is misspecified.

**Check it:** Plot log(-log(S(t))) vs. log(t) — lines should be parallel for different groups.

**In DSW:** `cox_ph` analysis type. Provide time, event, and covariate columns.

## Manufacturing Applications

### Warranty Analysis
Use Weibull to predict:
- What % of units will fail within the warranty period?
- How much should we budget for warranty claims?
- Should we extend/shorten the warranty?

### Preventive Maintenance
If β > 1 (increasing failure rate), preventive maintenance makes economic sense. Replace components before failure. The optimal replacement interval minimizes total cost (planned + unplanned maintenance).

### Accelerated Life Testing
Test products under harsh conditions to predict normal-use lifetime. Uses the Arrhenius model (temperature), power law (voltage), or Eyring model (multiple stresses) to extrapolate.
""",
    "interactive": {
        "type": "survival_demo",
        "config": {
            "title": "Try It: Kaplan-Meier Survival Curve",
            "description": "50 bearings tested to 1000 hours. 20 failed, 30 still running (censored). See how the survival curve accounts for censoring.",
            "times": [
                120,
                245,
                310,
                389,
                412,
                456,
                498,
                534,
                567,
                601,
                623,
                678,
                712,
                745,
                789,
                823,
                856,
                890,
                934,
                967,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
            ],
            "events": [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            "unit": "hours",
        },
    },
    "key_takeaways": [
        "Censoring is the defining challenge — ignoring it biases results",
        "Kaplan-Meier curves visualize survival probability over time; log-rank test compares groups",
        "Weibull shape parameter tells the failure story: beta<1 infant mortality, beta=1 random, beta>1 wear-out",
        "Cox PH models hazard as function of covariates — hazard ratio is the key output",
        "The proportional hazards assumption must be checked — violations invalidate the model",
    ],
    "practice_questions": [
        {
            "question": "A bearing manufacturer tests 50 bearings. After 1000 hours, 35 have failed and 15 are still running (test stopped). Weibull analysis gives β=2.3, η=850 hours. Interpret these results and recommend a maintenance interval.",
            "answer": "β=2.3 (>1) means an increasing failure rate — bearings wear out over time, so preventive maintenance is justified. η=850 hours (scale parameter) is the characteristic life — about 63.2% will fail by this point. For a maintenance strategy, calculate the B10 life (time for 10% failure): approximately 400 hours. A conservative maintenance interval would be around 350-400 hours, replacing bearings before the failure rate accelerates significantly. The 15 censored observations are properly handled by the Weibull analysis — they contribute information about surviving at least 1000 hours.",
            "hint": "β > 1 means wear-out; η is the characteristic life (63.2% failure point)",
        },
    ],
}


ML_ESSENTIALS = {
    "title": "Machine Learning Essentials",
    "intro": "Statistics asks if there is an effect. Machine learning asks if you can predict the outcome. You'll cluster customers, detect anomalies, rank feature importance, and learn the overfitting trap.",
    "exercise": {
        "title": "Try It: Cluster Customers",
        "steps": [
            "Load the customer spend vs frequency data",
            "Set k=3 clusters and run K-Means",
            "Review the silhouette score",
            "Change k to 4 and compare",
            "Characterize each cluster: who are these customers?",
        ],
        "dsw_type": "stats:descriptive",
    },
    "sample_data": {
        "clustering_example": {
            "description": "Customer purchase data: annual spend ($) vs visit frequency",
            "spend": [
                120,
                85,
                340,
                420,
                95,
                210,
                380,
                450,
                78,
                155,
                290,
                510,
                62,
                185,
                365,
                490,
                110,
                245,
                315,
                470,
            ],
            "frequency": [
                24,
                18,
                8,
                5,
                22,
                14,
                6,
                3,
                20,
                12,
                9,
                4,
                26,
                15,
                7,
                2,
                21,
                11,
                8,
                4,
            ],
        },
        "anomaly_example": {
            "description": "Sensor readings from 20 parts (2 are defective)",
            "temp": [
                72.1,
                71.8,
                72.3,
                71.9,
                72.0,
                85.4,
                71.7,
                72.2,
                71.6,
                72.4,
                72.1,
                71.5,
                72.0,
                71.8,
                72.3,
                71.9,
                92.1,
                72.1,
                71.7,
                72.2,
            ],
            "vibration": [
                0.15,
                0.12,
                0.18,
                0.14,
                0.13,
                0.45,
                0.16,
                0.11,
                0.17,
                0.14,
                0.15,
                0.13,
                0.12,
                0.16,
                0.15,
                0.14,
                0.62,
                0.13,
                0.15,
                0.11,
            ],
            "defective_indices": [5, 16],
        },
    },
    "content": """
## When Statistics Meets Prediction

Traditional statistics asks: **"Is there an effect?"** Machine learning asks: **"Can I predict the outcome?"**

These are different questions with different tools, but they complement each other. DSW provides 12+ ML methods that extend your analytical toolkit.

### When to Use ML vs. Traditional Statistics

| Use Statistics When | Use ML When |
|---|---|
| You need causal interpretation | You need prediction accuracy |
| You have a hypothesis to test | You have data to explore |
| Interpretability is essential | Performance matters more |
| Sample size is small | You have lots of data |
| Regulatory/legal context | Operational/business context |

## Clustering: Finding Natural Groups

**K-Means Clustering** partitions data into k groups based on similarity.

### How It Works
1. Choose k (number of clusters)
2. Randomly assign k initial centers
3. Assign each point to nearest center
4. Recalculate centers as cluster means
5. Repeat 3-4 until stable

### Choosing k
The **silhouette score** measures how well-separated clusters are:
- Close to +1: well-clustered
- Close to 0: near cluster boundary
- Negative: probably in wrong cluster

**In DSW:** `clustering` analysis type. Returns cluster assignments, silhouette analysis, and cluster profiles.

**Example:** Customer segmentation. Upload purchase data, cluster into 3-5 groups, then characterize each group (high-frequency/low-value, low-frequency/high-value, etc.).

## PCA: Dimensionality Reduction

**Principal Component Analysis** finds the directions of maximum variance in high-dimensional data.

### When to Use PCA
- Too many variables to visualize or model
- Many correlated variables (multicollinearity)
- Need to identify underlying patterns/factors
- Preprocessing for other ML methods

### How to Read PCA Output
- **Explained variance ratio:** PC1 explains X%, PC2 explains Y%, etc.
- **Loadings:** Which original variables contribute to each component
- **Scree plot:** Shows the "elbow" where additional PCs add little information
- **Biplot:** Visualizes both scores and loadings

**In DSW:** `pca` analysis type. Returns components, explained variance, loadings, and visualization.

**Rule of thumb:** Keep enough PCs to explain 80-90% of total variance.

## Feature Importance: What Matters Most?

**Random Forest feature importance** ranks variables by their contribution to prediction.

### How It Works
Random Forest builds hundreds of decision trees on random subsets of data and features. Importance is measured by how much prediction accuracy drops when a feature is randomly shuffled (permutation importance).

**In DSW:** `feature` analysis type. Provide target and feature columns. Returns ranked importance scores.

**Caution:** Correlated features split importance between them. If Temperature and Humidity are correlated, both may show moderate importance even if only one truly matters. Consider PCA first to handle this.

## Anomaly Detection

**Isolation Forest** identifies unusual observations — outliers that don't fit the normal pattern.

### How It Works
Randomly selects features and split values. Anomalies are isolated in fewer splits than normal points (they're "easy to separate"). Points requiring few splits get high anomaly scores.

**In DSW:** `isolation_forest` analysis type. Returns anomaly scores and flags for each observation.

**Applications:**
- Manufacturing: detecting unusual product measurements
- Quality control: identifying process deviations
- Fraud: flagging unusual transactions

## Regularized Regression

When standard regression overfits or has too many predictors:

- **Ridge (L2):** Shrinks coefficients toward zero. Good when all variables contribute.
- **Lasso (L1):** Can shrink coefficients exactly to zero — performs variable selection. Good when only some variables matter.
- **Elastic Net:** Combines Ridge and Lasso. Good when variables are correlated.

**In DSW:** `regularized_regression` analysis type. Specify penalty type and strength.

**When to use over standard regression:**
- More than ~20 predictors
- Correlated predictors (multicollinearity)
- Need automatic variable selection (Lasso)
- Want to prevent overfitting on small samples

## Gaussian Process Regression

A Bayesian approach that provides **uncertainty estimates** with every prediction.

- Predictions come with confidence bands
- Non-parametric: adapts to complex patterns
- Works well with small datasets
- Computationally expensive for large datasets

**In DSW:** `gaussian_process` analysis type. Excellent for process optimization where you need uncertainty-aware predictions.

## Practical ML Workflow

1. **Define the question:** What are you predicting? Why?
2. **Prepare data:** Handle missing values, encode categories, scale features
3. **Explore first:** Use EDA, PCA, and feature importance to understand structure
4. **Split data:** Train (70-80%) and test (20-30%). Never peek at test data
5. **Train model:** Start simple (regression), then try complex (Random Forest, GP)
6. **Validate:** Check performance on test data. If much worse than training → overfitting
7. **Interpret:** Use feature importance and partial dependence to explain predictions
8. **Monitor:** Model performance degrades over time (concept drift). Retrain periodically
""",
    "interactive": {
        "type": "clustering_demo",
        "config": {
            "title": "Try It: K-Means Clustering",
            "description": "Customer segments by annual spend vs visit frequency. Drag the slider to change k and see how clusters form.",
            "x_label": "Annual Spend ($)",
            "y_label": "Visit Frequency",
            "x_data": [
                120,
                85,
                340,
                420,
                95,
                210,
                380,
                450,
                78,
                155,
                290,
                510,
                62,
                185,
                365,
                490,
                110,
                245,
                315,
                470,
            ],
            "y_data": [
                24,
                18,
                8,
                5,
                22,
                14,
                6,
                3,
                20,
                12,
                9,
                4,
                26,
                15,
                7,
                2,
                21,
                11,
                8,
                4,
            ],
            "default_k": 3,
            "max_k": 6,
        },
    },
    "key_takeaways": [
        "ML is for prediction; statistics is for inference — use both where appropriate",
        "Clustering finds natural groups; PCA reduces dimensions; feature importance ranks variables",
        "Always split data into train/test — overfitting is the primary ML failure mode",
        "Regularization (Ridge/Lasso) prevents overfitting when you have many predictors",
        "Gaussian Process gives uncertainty estimates — valuable for process optimization",
        "Isolation Forest detects anomalies without needing labeled examples",
    ],
    "practice_questions": [
        {
            "question": "A manufacturing process has 45 sensor readings per part. You want to detect defective parts before final inspection. You have 10,000 good parts and 12 known defective parts. Which ML approach and why?",
            "answer": "Isolation Forest (anomaly detection). With only 12 defective examples out of 10,000, this is a severe class imbalance problem — supervised classification would struggle. Isolation Forest is unsupervised and excels at finding observations that differ from the majority. Train on the 10,000 good parts to learn the 'normal' pattern, then flag new parts with high anomaly scores. You could also use PCA first to reduce the 45 sensors to key components, then apply anomaly detection in the reduced space.",
            "hint": "12 out of 10,000 is extreme imbalance — supervised learning needs balanced classes",
        },
        {
            "question": "Your regression model has R²=0.95 on training data but R²=0.42 on test data. What happened and what should you try?",
            "answer": "Classic overfitting — the model memorized training noise rather than learning the true pattern. Try: (1) Regularized regression (Ridge or Lasso) to penalize complex models, (2) reduce the number of features using PCA or Lasso's built-in variable selection, (3) increase training data if possible, (4) use a simpler model (fewer terms). The gap between 0.95 and 0.42 is dramatic — the model is essentially useless for prediction despite looking great on training data.",
            "hint": "Big gap between train and test performance = overfitting",
        },
    ],
}


MEASUREMENT_SYSTEMS = {
    "title": "Measurement System Analysis",
    "intro": "If your measurement system is noisy, you can't tell process variation from measurement variation. You'll evaluate Gage R&R, acceptance criteria, and learn when to fix the gage before fixing the process.",
    "exercise": {
        "title": "Try It: Audit a Measurement System",
        "steps": [
            "Work through the measurement system checklist",
            "Check gage resolution against tolerance (10:1 rule)",
            "Review Gage R&R results: is GRR% below 10%?",
            "Determine if the number of distinct categories is at least 5",
            "Decide: fix the gage or proceed to process analysis?",
        ],
        "dsw_type": "stats:descriptive",
        "dsw_config": {},
    },
    "content": """
## Can You Trust Your Measurements?

Before optimizing a process, you must verify that your measurement system is adequate. If your measurements are noisy or biased, you can't tell process variation from measurement variation.

**The fundamental equation:**

Total Observed Variation = Process Variation + Measurement Variation

If measurement variation is large relative to process variation, you'll:
- Miss real process changes (too much noise)
- See phantom changes that aren't real
- Misestimate process capability (Cpk is wrong)

## Gage R&R (Repeatability & Reproducibility)

The standard method for evaluating measurement systems.

### Two Components

**Repeatability** — variation when the **same operator** measures the **same part** multiple times. This is the instrument's inherent precision.

**Reproducibility** — variation when **different operators** measure the **same part.** This captures operator technique differences.

**Gage R&R = Repeatability + Reproducibility**

### Study Design

Typical Gage R&R study:
- 10 parts (representing the process range)
- 3 operators
- 2-3 measurements per part per operator
- Total: 60-90 measurements

**Critical:** Parts must span the process range. If you only measure similar parts, you'll overestimate the %GRR.

### Acceptance Criteria

| %GRR of Total Variation | Assessment |
|---|---|
| < 10% | Excellent — measurement system is adequate |
| 10-30% | Marginal — may be acceptable depending on application |
| > 30% | Unacceptable — fix the measurement system before studying the process |

### Reading Gage R&R Output

**In DSW:** `gage_rr` analysis type. Returns:

- **%Contribution** — variance components as % of total
- **%Study Variation (GRR%)** — standard deviation ratio (the primary metric)
- **Number of Distinct Categories (ndc)** — how many process categories the gage can distinguish. Need ndc ≥ 5.
- **ANOVA table** — tests for part, operator, and interaction effects
- **By-operator plots** — shows whether specific operators are more variable

### What If GRR Is Too High?

1. **High repeatability** → Instrument issue
   - Calibrate or replace the gage
   - Improve measurement fixture (reduce positioning variation)
   - Increase resolution (need 10:1 ratio of tolerance to gage resolution)

2. **High reproducibility** → Operator issue
   - Standardize measurement procedure
   - Training on measurement technique
   - Automate the measurement

3. **Significant interaction** → Some operators measure some parts differently
   - Usually indicates unclear measurement instructions
   - Parts with features that are ambiguous to measure

## Acceptance Sampling

When 100% inspection is impossible or uneconomical, inspect a sample and decide whether to accept or reject the lot.

### Key Terms

- **AQL (Acceptable Quality Level):** The worst quality level that's still "acceptable" as a process average (producer's risk)
- **LTPD (Lot Tolerance Percent Defective):** The worst quality level the consumer will tolerate (consumer's risk)
- **Producer's risk (α):** Probability of rejecting a good lot (typically 5%)
- **Consumer's risk (β):** Probability of accepting a bad lot (typically 10%)
- **OC curve:** Shows probability of acceptance vs. actual defect rate

### Sampling Plan

A plan specifies:
- **n** — sample size
- **c** — acceptance number (accept lot if defects ≤ c)

**In DSW:** `acceptance_sampling` analysis type. Specify AQL, LTPD, producer's risk, and consumer's risk. Returns optimal n and c, plus the OC curve.

### Single vs. Double vs. Sequential

- **Single sampling:** Inspect n, decide once
- **Double sampling:** Inspect n₁, decide or inspect n₂ more, then decide
- **Sequential sampling:** Inspect one at a time, running decision boundary

Double and sequential plans reduce average sample size but are more complex to administer.

## Bias and Linearity

Beyond Gage R&R:

- **Bias:** Systematic difference between measured and true value (the gage reads consistently high or low)
- **Linearity:** Whether bias changes across the measurement range (accurate for small parts but biased for large parts)

Both should be checked when setting up a measurement system, especially for regulatory compliance.

## The Measurement System Audit

Before any process improvement project:

1. ✅ Gage R&R study — is the measurement system adequate?
2. ✅ Bias study — is the gage reading true values?
3. ✅ Linearity study — is bias constant across the range?
4. ✅ Stability study — does the gage drift over time?
5. ✅ Resolution check — 10:1 ratio to tolerance?

Only after all checks pass should you proceed with process capability studies.
""",
    "interactive": {
        "type": "bias_detector",
        "config": {
            "title": "Measurement System Checklist",
            "items": [
                "Gage resolution is ≤ 1/10 of tolerance",
                "Gage R&R study performed with ≥ 10 parts, ≥ 2 operators, ≥ 2 trials",
                "GRR% < 30% of total variation",
                "Number of distinct categories ≥ 5",
                "Bias study shows no systematic offset",
                "Linearity study shows constant bias across range",
                "Gage stability verified over time",
            ],
        },
    },
    "key_takeaways": [
        "Measurement variation adds to observed process variation — must quantify it first",
        "Gage R&R separates repeatability (instrument) from reproducibility (operator)",
        "GRR < 10% is excellent; 10-30% marginal; > 30% unacceptable",
        "Number of distinct categories (ndc ≥ 5) tells you if the gage can distinguish process variation",
        "Acceptance sampling balances inspection cost against quality risk with statistical rigor",
    ],
    "practice_questions": [
        {
            "question": "A Gage R&R study shows: Repeatability = 15% of total variation, Reproducibility = 22% of total variation, GRR = 27%. Your process Cpk is 1.1. Should you improve the process or the measurement system first?",
            "answer": "Fix the measurement system first. At 27% GRR, a significant portion of what you're measuring is noise, not signal. The reported Cpk of 1.1 is unreliable — the true Cpk could be higher (masked by measurement noise) or the measurement system could be making a capable process look marginal. Since reproducibility (22%) dominates, focus on operator training and standardizing the measurement procedure. Only after GRR < 10% can you trust the process capability assessment.",
            "hint": "If you can't trust the measurement, you can't trust any analysis based on it",
        },
    ],
}
