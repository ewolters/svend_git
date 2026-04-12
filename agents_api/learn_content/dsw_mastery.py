"""Learning content: Dsw Mastery."""

from ._datasets import SHARED_DATASET  # noqa: F401

DSW_OVERVIEW = {
    "id": "dsw-overview",
    "title": "DSW Architecture & Workflow",
    "intro": "DSW is your statistical analysis engine with built-in guardrails. You'll tour the analysis pipeline from data upload through validation, analysis, and export, and see how it prevents common mistakes.",
    "exercise": {
        "title": "Try It: Walk Through the Pipeline",
        "steps": [
            "Upload a sample CSV dataset",
            "Review the automatic validation report",
            "Select an analysis type and map your columns",
            "Run the analysis and review the plain-language interpretation",
        ],
        "dsw_type": "stats:descriptive",
        "dsw_config": {},
    },
    "content": """
## The Data Science Workbench

DSW is SVEND's analysis engine - a structured pipeline for running rigorous statistical analyses with guardrails against common mistakes.

### Architecture Overview

```
Data Input → Validation → Analysis → Interpretation → Report
    ↓           ↓            ↓            ↓            ↓
  CSV/API   Type check   Statistical   Plain-language  Export
            Missing %    computation   explanation     PDF/JSON
            Outliers                   Caveats
```

### Available Analysis Types

| Analysis | Use Case | Key Output |
|----------|----------|------------|
| **Bayesian A/B** | Compare two variants | P(B > A), expected lift |
| **SPC Charts** | Monitor process stability | Control limits, capability |
| **Regression** | Predict continuous outcome | Coefficients, R², diagnostics |
| **Correlation** | Assess relationships | Correlation matrix, significance |
| **Distribution** | Understand data shape | Normality tests, summary stats |

### Input Data Requirements

**Format:** CSV with headers, or JSON array

**Validation checks:**
- Column types (numeric, categorical, date)
- Missing data percentage (warning if >5%, error if >30%)
- Outlier detection (flagged, not auto-removed)
- Sample size adequacy for requested analysis

### The Analysis Pipeline

1. **Upload & Validate**
   - Data profiling: types, missing values, distributions
   - Automatic issue detection with recommendations

2. **Configure Analysis**
   - Select analysis type
   - Map columns to roles (outcome, groups, covariates)
   - Set parameters (priors, confidence level, etc.)

3. **Run & Review**
   - Results with uncertainty quantification
   - Diagnostic checks and assumption tests
   - Plain-language interpretation

4. **Export & Share**
   - PDF report with methodology
   - JSON for programmatic use
   - Shareable link with reproducibility info

### Guardrails

DSW prevents common mistakes:
- **Too-small samples:** Warns when power is inadequate
- **Multiple comparisons:** Flags when correction needed
- **Assumption violations:** Tests and warns automatically
- **Misleading visualizations:** Uses appropriate defaults
""",
    "interactive": {"type": "dsw_demo", "config": {}},
    "key_takeaways": [
        "DSW provides guardrails against common statistical mistakes",
        "Always review the validation report before trusting results",
        "The plain-language interpretation is a starting point, not gospel",
        "Export includes methodology for reproducibility",
    ],
    "practice_questions": [
        {
            "question": "You upload a CSV with 12% missing values in the outcome column. DSW flags a warning. Should you proceed with analysis?",
            "answer": "Investigate first. 12% is above the 5% warning threshold. Check WHY values are missing — is it MCAR (safe to impute/drop), MAR (adjust), or MNAR (dangerous)? DSW's warning is a signal to clean data before analyzing, not to ignore.",
            "hint": "The validation report tells you what to investigate, not what to ignore",
        },
        {
            "question": "DSW runs a t-test and reports p=0.02 but also flags 'assumption violation: non-normal distribution.' What should you do?",
            "answer": "Switch to a non-parametric alternative (Mann-Whitney U). The assumption violation warning means the t-test result may not be reliable. If n>30 per group, the t-test is fairly robust, but it's better practice to use the non-parametric test and compare results.",
            "hint": "Guardrails exist for a reason — follow up on assumption warnings",
        },
    ],
}


BAYESIAN_AB_HANDS_ON = {
    "id": "bayesian-ab-hands-on",
    "title": "Bayesian A/B Testing",
    "intro": "Frequentist tests tell you IF the null is unlikely. Bayesian tests tell you HOW LIKELY B is better than A. You'll run a Bayesian A/B test and see how priors affect the posterior.",
    "exercise": {
        "title": "Try It: Run a Bayesian A/B Test",
        "steps": [
            "Load the sample A/B test data",
            "Set an uninformative prior (Beta(1,1))",
            "Run the analysis and check P(B > A)",
            "Switch to a skeptical prior that favors small effects",
            "Compare posteriors and note how data overwhelms the prior",
        ],
        "dsw_type": "stats:descriptive",
        "dsw_config": {},
    },
    "content": """
## Why Bayesian for A/B Testing?

Frequentist A/B tests tell you: "If there's no difference, how likely is data this extreme?"

Bayesian tests tell you: "Given the data, what's the probability B is better than A?"

The second question is what you actually want to know.

### The Bayesian A/B Framework

#### 1. Start with a Prior

Your belief about the conversion rate before seeing data.

| Prior Type | When to Use |
|------------|-------------|
| **Uninformative** | No prior knowledge, let data speak |
| **Historical** | Use data from previous tests |
| **Conservative** | Skeptical of large effects |

Default: Beta(1, 1) - uniform prior (any conversion rate equally likely)

#### 2. Update with Data

Observe conversions and non-conversions. Posterior combines prior + data.

$$\\text{Posterior} \\propto \\text{Likelihood} \\times \\text{Prior}$$

#### 3. Calculate Probabilities

From the posterior, compute:
- **P(B > A)**: Probability treatment is better
- **Expected lift**: How much better, on average
- **Credible interval**: Range of plausible effect sizes

### Reading DSW Bayesian A/B Output

```
Variant A: 1,247/25,000 (4.99%)
Variant B: 1,372/25,000 (5.49%)

Probability B > A: 94.2%
Expected lift: +10.0% relative
95% Credible Interval: [+3.2%, +17.1%]
Expected loss if choosing B: 0.02%
Expected loss if choosing A: 0.48%
```

**Interpretation:**
- 94% confident B is better
- If B is better, it's about 10% better
- Even in worst case (CI lower bound), B is still +3% better
- If we're wrong about B, we lose very little (0.02%)

### Decision Rules

| P(B > A) | Recommendation |
|----------|----------------|
| < 50% | A is probably better |
| 50-75% | Inconclusive, need more data |
| 75-95% | B is likely better, consider risk |
| > 95% | Strong evidence for B |

But also consider:
- **Effect size:** Is +0.1% worth the change cost?
- **Expected loss:** What do we lose if wrong?
- **Reversibility:** Can we undo if needed?

### Early Stopping & Peeking

**Frequentist problem:** Peeking inflates false positives.

**Bayesian advantage:** P(B > A) is always valid. You can check anytime.

But: Your posterior is influenced by when you stopped. Pre-specify decision criteria.

### Hands-On Exercise

Try this in DSW:
1. Upload the sample A/B test dataset
2. Set an uninformative prior
3. Run the analysis
4. Change to a skeptical prior (e.g., assumes small effects likely)
5. Compare how the posterior changes

**Key learning:** Priors matter when data is limited. With lots of data, priors wash out.
""",
    "interactive": {"type": "ab_analyzer", "config": {}},
    "key_takeaways": [
        "Bayesian tests answer the question you actually care about",
        "P(B > A) is directly interpretable as probability of improvement",
        "Expected loss helps make decisions when uncertainty remains",
        "You can peek at Bayesian results without inflating error rates",
        "Priors matter with small samples, less so with large samples",
    ],
    "practice_questions": [
        {
            "question": "Your test shows P(B > A) = 78%. Should you ship variant B?",
            "answer": "It depends. 78% is suggestive but not conclusive. Consider: What's the expected lift? What's the expected loss if wrong? Is the change reversible? How costly is more data collection?",
            "hint": "There's no universal threshold - it depends on context.",
        },
        {
            "question": "Why might you use a skeptical prior?",
            "answer": "To require stronger evidence before believing in large effects. Most A/B tests show small effects; a skeptical prior encodes this expectation and protects against being fooled by noise.",
            "hint": "Think about base rates of effect sizes in your domain.",
        },
    ],
}


SPC_HANDS_ON = {
    "id": "spc-hands-on",
    "title": "SPC & Control Charts",
    "intro": "Is your process stable or drifting? Control charts separate signal from noise. You'll build an I-MR chart, identify out-of-control points, and calculate process capability.",
    "exercise": {
        "title": "Try It: Build a Control Chart",
        "steps": [
            "Load the manufacturing dataset",
            "Create an I-MR chart for the diameter column",
            "Identify points outside the control limits",
            "Check for Western Electric rule violations",
            "Calculate Cp and Cpk using spec limits of 25.00 +/- 0.15mm",
        ],
        "dsw_type": "spc:imr",
        "dsw_config": {"var": "diameter_mm"},
    },
    "content": """
## Statistical Process Control

SPC helps you distinguish between:
- **Common cause variation:** Normal fluctuation, built into the process
- **Special cause variation:** Something changed, investigate!

### The Core Idea

A stable process has predictable variation. If a point falls outside the expected range, something is different.

### Control Chart Types

| Chart | Data Type | What It Monitors |
|-------|-----------|------------------|
| **X-bar & R** | Continuous, subgroups | Process mean and range |
| **X-bar & S** | Continuous, subgroups (n>10) | Process mean and std dev |
| **I-MR** | Continuous, individuals | Individual values and moving range |
| **P chart** | Proportion defective | Defect rate |
| **C chart** | Count of defects | Defects per unit |

### Reading a Control Chart

```
UCL = Upper Control Limit (mean + 3σ)
 |
 |    *
 |  *   *  *      <- In control
 |*   *    *  *
CL = Center Line (mean)
 |    *  *   *
 |  *        *
 |           *
 |             *  <- Out of control!
LCL = Lower Control Limit (mean - 3σ)
```

### Control Limits vs Specification Limits

| | Control Limits | Spec Limits |
|-|----------------|-------------|
| **Based on** | Process data | Customer requirements |
| **Purpose** | Detect changes | Determine conformance |
| **Set by** | Statistics | Business/customer |

**A process can be:**
- In control but not capable (stable but doesn't meet spec)
- Out of control but capable (unstable but currently meeting spec - dangerous!)

### Capability Indices

**Cp:** Can the process meet spec if centered?
$$C_p = \\frac{USL - LSL}{6\\sigma}$$

**Cpk:** Is the process meeting spec given where it's centered?
$$C_{pk} = \\min\\left(\\frac{USL - \\mu}{3\\sigma}, \\frac{\\mu - LSL}{3\\sigma}\\right)$$

| Cpk | Interpretation |
|-----|----------------|
| < 1.0 | Not capable (significant defects) |
| 1.0 - 1.33 | Marginally capable |
| 1.33 - 1.67 | Capable |
| > 1.67 | Highly capable |

### Out-of-Control Patterns

**Western Electric Rules:**
1. One point beyond 3σ
2. Two of three consecutive points beyond 2σ (same side)
3. Four of five consecutive points beyond 1σ (same side)
4. Eight consecutive points on one side of center

### Hands-On Exercise

1. Upload the manufacturing dataset
2. Create an I-MR chart for the key dimension
3. Identify any out-of-control points
4. Calculate Cp and Cpk
5. Determine if the process is capable

### Next Level: Probabilistic Bayesian SPC

Classical control charts use fixed limits and binary rules. But what if your control chart could:
- Tell you the **probability** of a shift, not just "in or out"
- **Learn** from every observation and narrow its limits over time
- **Predict** where the process is heading, not just where it's been
- Give you a **credible interval** on Cpk, not just a point estimate

That's exactly what the **PBS (Probabilistic Bayesian SPC)** module does. After completing this section, head to **Module 11: PBS Mastery** to see the future of process monitoring.
""",
    "interactive": {"type": "spc_demo", "config": {}},
    "key_takeaways": [
        "Control charts separate common cause from special cause variation",
        "Control limits come from data, not specifications",
        "Cpk tells you if you're meeting requirements",
        "Out-of-control patterns indicate something changed - investigate!",
        "A process in control is predictable, not necessarily good",
    ],
    "practice_questions": [
        {
            "question": "Your process has Cp = 1.5 but Cpk = 0.8. What's happening?",
            "answer": "The process spread is small enough to meet spec (Cp = 1.5 is good), but the process is not centered. It's shifted toward one spec limit. Centering the process would dramatically improve Cpk.",
            "hint": "Cp ignores centering, Cpk accounts for it.",
        },
        {
            "question": "A point just outside the control limit could be random chance. Why do we investigate anyway?",
            "answer": "Control limits are set at 3σ, so random chance outside limits should occur only 0.3% of the time. While false alarms happen, the cost of missing a real problem usually exceeds the cost of investigation.",
            "hint": "What's the probability of a point beyond 3σ by chance?",
        },
    ],
}


REGRESSION_HANDS_ON = {
    "id": "regression-hands-on",
    "title": "Regression Analysis",
    "intro": "Regression is the Swiss Army knife of statistics. You'll build a model, check diagnostics, interpret coefficients, and learn when R-squared lies to you.",
    "exercise": {
        "title": "Try It: Build a Regression Model",
        "steps": [
            "Select a continuous outcome variable and 2-3 predictors",
            "Fit the model and review the coefficient table",
            "Check the residual vs fitted plot for patterns",
            "Inspect VIF values for multicollinearity",
            "Compare R-squared vs adjusted R-squared",
        ],
        "dsw_type": "stats:regression",
        "dsw_config": {
            "response": "diameter_mm",
            "predictors": ["weight_g", "roughness_ra"],
        },
    },
    "content": """
## Building and Interpreting Regression Models

Regression is the Swiss Army knife of statistics—used for prediction, explanation, and causal inference.

### Linear Regression Fundamentals

The linear regression model:
$$Y = \\beta_0 + \\beta_1 X_1 + \\beta_2 X_2 + ... + \\epsilon$$

**Interpretation:**
- $\\beta_0$: Expected Y when all X = 0 (intercept)
- $\\beta_1$: Change in Y for 1-unit increase in X₁, holding other X constant
- $\\epsilon$: Random error (what the model doesn't explain)

**Key assumptions (LINE):**
1. **L**inearity: Relationship is linear (check residual plot)
2. **I**ndependence: Observations are independent
3. **N**ormality: Residuals are normally distributed
4. **E**qual variance: Residuals have constant spread (homoscedasticity)

### Checking Assumptions in DSW

DSW provides diagnostic plots automatically:

**Residuals vs Fitted:**
- Should show random scatter (no pattern)
- Pattern = non-linearity or missing variable
- Funnel shape = heteroscedasticity

**Q-Q Plot of Residuals:**
- Points should follow diagonal line
- Departures indicate non-normality
- Focus on tails—middle is usually fine

**Scale-Location Plot:**
- Should be flat line with random scatter
- Upward slope = variance increases with fitted values

**Residuals vs Leverage:**
- Identifies influential points
- Points with high leverage AND high residual are problematic

### Interpreting Coefficients

**Continuous predictors:**
"For each 1-unit increase in X, Y increases by β (on average), holding other variables constant."

**Dummy variables (categorical):**
"Compared to the reference group, group A has Y that is β higher (on average)."

**Log-transformed outcome (log(Y)):**
"For each 1-unit increase in X, Y increases by approximately β × 100%."

**Standardized coefficients:**
"For each 1 SD increase in X, Y increases by β SDs."

### Understanding R²

$$R^2 = 1 - \\frac{SS_{residual}}{SS_{total}} = \\frac{SS_{explained}}{SS_{total}}$$

**Interpretation:** Proportion of variance in Y explained by the model.

**Caveats:**
- R² always increases with more predictors (use adjusted R²)
- High R² doesn't mean the model is correct
- Low R² doesn't mean predictors aren't important
- R² is for prediction, not causation

**Adjusted R²:**
$$R^2_{adj} = 1 - \\frac{(1-R^2)(n-1)}{n-k-1}$$

Penalizes adding unhelpful predictors.

### Multicollinearity

When predictors are highly correlated with each other:
- Coefficients become unstable (large standard errors)
- Individual effects are hard to separate
- Model still predicts well overall

**Detection:**
- Correlation matrix among predictors
- Variance Inflation Factor (VIF)
  - VIF > 5: Concerning
  - VIF > 10: Serious problem

**Solutions:**
- Remove one of the correlated predictors
- Combine into a single index
- Use regularization (ridge regression)

### Logistic Regression for Binary Outcomes

When Y is 0/1 (success/failure), use logistic regression:

$$\\log\\left(\\frac{p}{1-p}\\right) = \\beta_0 + \\beta_1 X_1 + ...$$

**Interpretation:**
- Coefficients are log-odds ratios
- $e^{\\beta_1}$ = odds ratio for 1-unit increase in X₁
- OR > 1: X increases probability of success
- OR < 1: X decreases probability

**Example:** If β₁ = 0.7 for smoking:
- e^0.7 ≈ 2.0
- Smokers have 2x the odds of the outcome

### Model Selection and Comparison

**Forward selection:** Start empty, add best predictor one at a time.

**Backward elimination:** Start full, remove worst predictor one at a time.

**Criteria:**
- **AIC/BIC:** Lower is better, penalizes complexity
- **Adjusted R²:** Higher is better
- **Cross-validation:** Most reliable for prediction

**Warning:** Stepwise methods can overfit and find spurious relationships. Theory-driven models are usually better.

### Hands-On: Building a Predictive Model

In DSW, a typical workflow:

1. **Explore data:** Scatterplot matrix, correlations
2. **Fit initial model:** All hypothesized predictors
3. **Check diagnostics:** Residual plots, VIF
4. **Address issues:** Transformations, remove predictors
5. **Compare models:** AIC, cross-validation
6. **Interpret final model:** Coefficients, R², prediction intervals

**Remember:** A model for prediction can differ from a model for explanation. Predictive models optimize accuracy; explanatory models optimize interpretability and causal validity.
""",
    "interactive": {"type": "regression_simulator", "config": {}},
    "key_takeaways": [
        "Coefficients show the change in Y per unit change in X, holding other variables constant",
        "Always check diagnostic plots—violations of assumptions invalidate inference",
        "R² tells you variance explained, not whether the model is correct",
        "VIF > 10 indicates serious multicollinearity",
        "Logistic regression coefficients are log-odds ratios; exponentiate to interpret",
    ],
    "practice_questions": [
        {
            "question": "Your regression has R² = 0.85 but the residuals vs fitted plot shows a clear U-shape. What does this mean and what should you do?",
            "answer": "The U-shaped pattern indicates non-linearity—the true relationship isn't captured by your linear terms. Despite high R², the model is misspecified. You should try adding polynomial terms (X²), transforming variables (log, square root), or using a non-linear model.",
            "hint": "High R² doesn't mean the model is correct if residuals show patterns.",
        },
        {
            "question": "You're building a model to predict house prices. You have 'number of bedrooms' and 'square footage' as predictors. VIF for both is 4.5. Should you be concerned?",
            "answer": "VIF of 4.5 is worth noting but not alarming (rule of thumb: concern at >5, serious at >10). Bedrooms and square footage are naturally correlated, so some collinearity is expected. If your goal is prediction, keep both—the model still predicts well. If your goal is understanding the separate effect of each, consider keeping only one or using theory to guide interpretation.",
            "hint": "What are the VIF thresholds for concern?",
        },
    ],
}


DOE_HANDS_ON = {
    "title": "DOE Hands-On",
    "intro": "Stop changing one factor at a time. DOE varies multiple factors simultaneously to find main effects, interactions, and optimal settings in a fraction of the runs.",
    "exercise": {
        "title": "Try It: Select a DOE Design",
        "steps": [
            "Answer the guided questions about your experiment",
            "Specify the number of factors and budget for runs",
            "Review the recommended design type",
            "Check the design resolution and what it can estimate",
            "Generate the randomized run order",
        ],
        "dsw_type": "stats:descriptive",
        "dsw_config": {},
    },
    "content": """
## Design of Experiments in SVEND

Design of Experiments (DOE) is one of SVEND's most powerful capabilities. Instead of changing one factor at a time (OFAT), DOE systematically varies multiple factors simultaneously to:

- Find main effects (which factors matter)
- Find interactions (which factor combinations matter)
- Build predictive models
- Optimize responses

### Accessing DOE

In the Experimenter module (`/api/experimenter/`), you can:
1. **Define factors** — what you want to vary (temperature, pressure, speed, etc.)
2. **Choose a design** — full factorial, fractional, screening, or response surface
3. **Generate run order** — randomized experimental plan
4. **Analyze results** — effects, interactions, model, optimization

## Available Design Types

### Full Factorial (2^k)

Tests every combination of factor levels. For k factors at 2 levels each, you need 2^k runs.

| Factors | Runs | Use When |
|---|---|---|
| 2 | 4 | Almost always feasible |
| 3 | 8 | Still manageable |
| 4 | 16 | Moderate cost |
| 5 | 32 | Getting expensive |
| 6+ | 64+ | Consider fractional |

**Advantage:** Estimates all main effects AND all interactions.
**Disadvantage:** Runs grow exponentially.

### Fractional Factorial (2^(k-p))

Runs only a fraction of the full factorial. Trades the ability to estimate high-order interactions for fewer runs.

**Resolution matters:**
- **Resolution III:** Main effects confounded with 2-factor interactions (screening only)
- **Resolution IV:** Main effects clear, but 2-factor interactions confounded with each other
- **Resolution V:** Main effects and 2-factor interactions clear

**Rule of thumb:** Use Resolution V or higher unless you're just screening.

### Plackett-Burman (Screening)

Ultra-efficient screening designs: test k factors in k+1 runs (rounded to multiple of 4).

**Example:** Screen 11 factors in just 12 runs to find the vital few.

**Caution:** Resolution III only — cannot separate main effects from interactions. Use for screening, then follow up with a more detailed design on the important factors.

### Definitive Screening Design (DSD)

A modern alternative that can estimate:
- All main effects
- All quadratic effects (curvature)
- Some two-factor interactions

In only 2k+1 runs. The best choice for initial experimentation with 4-12 factors.

### Response Surface Designs

For modeling curvature and finding optimal operating conditions:

**Central Composite Design (CCD):**
- Full/fractional factorial + star points + center points
- Most common response surface design
- Can be built sequentially (add star points to existing factorial)

**Box-Behnken:**
- 3-level design (no extreme corners)
- Slightly fewer runs than CCD
- Good when extreme conditions are impractical

### Taguchi Arrays

Specialized designs for robust design (making products insensitive to noise factors). Separate control factors from noise factors to find settings that work well across varying conditions.

## Hands-On Workflow

### Step 1: Define Your Problem

Before touching the DOE tool, answer:
- What response(s) are you measuring?
- What factors might affect it?
- What ranges are feasible for each factor?
- Are there any factors that must be held constant?
- What's your budget for experimental runs?

### Step 2: Choose the Right Design

| Situation | Design | Why |
|---|---|---|
| ≤4 factors, full understanding needed | Full factorial | Estimates everything |
| 5-7 factors, interactions matter | Fractional factorial (Res V) | Good balance |
| 8+ factors, screening | Plackett-Burman or DSD | Find vital few |
| Known important factors, need optimization | CCD or Box-Behnken | Models curvature |
| Robustness to noise | Taguchi | Separates control/noise |

### Step 3: Run and Analyze

1. **Randomize** run order (SVEND does this automatically)
2. **Execute** runs following the exact design
3. **Enter** results into the design matrix
4. **Analyze:**
   - Effect estimates (which factors are significant?)
   - Pareto chart (ranked effects)
   - Main effects plots (direction and magnitude)
   - Interaction plots (do factors depend on each other?)
   - Residual diagnostics (is the model adequate?)

### Step 4: Confirm

**Always run confirmation trials** at the predicted optimal settings. Never trust a model without confirmation.

## Power Analysis for DOE

Before running an experiment, check that you have enough power to detect meaningful effects.

**In DSW:** Use `power_doe` to determine the minimum effect size detectable for your design and error variance.

**Key input:** An estimate of experimental error (from prior data or pilot runs). Without this, power analysis is guesswork.
""",
    "interactive": {
        "type": "decision_framework",
        "config": {
            "title": "DOE Design Selector",
            "questions": [
                {
                    "q": "Are you screening (>6 factors) or optimizing (≤6 factors)?",
                    "yes_next": 1,
                    "no_next": 2,
                },
                {
                    "q": "Do you need to estimate curvature?",
                    "yes_next": "dsd",
                    "no_next": "screen",
                },
                {
                    "q": "Do you need to find the optimal point?",
                    "yes_next": 3,
                    "no_next": "factorial",
                },
                {
                    "q": "Can you run extreme factor combinations?",
                    "yes_next": "ccd",
                    "no_next": "bbd",
                },
            ],
            "outcomes": {
                "screen": "Plackett-Burman — Screen many factors in minimal runs",
                "dsd": "Definitive Screening Design — Screen with curvature estimation",
                "factorial": "Full or Fractional Factorial — Estimate effects and interactions",
                "ccd": "Central Composite Design — Full response surface with optimization",
                "bbd": "Box-Behnken — Response surface without extreme corners",
            },
        },
    },
    "key_takeaways": [
        "DOE varies multiple factors simultaneously — far more efficient than one-at-a-time",
        "Resolution determines what you can estimate: V+ for interactions, III for screening only",
        "Definitive Screening Designs are the modern go-to for initial experimentation",
        "Response surface designs (CCD, Box-Behnken) find optimal operating conditions",
        "Always randomize run order and run confirmation trials at predicted optimum",
        "Power analysis before the experiment prevents wasted effort",
    ],
    "practice_questions": [
        {
            "question": "You have 6 factors to investigate. Budget allows 20 experimental runs. Your boss insists on testing all two-factor interactions. What design do you recommend and why?",
            "answer": "A 2^(6-1) fractional factorial with Resolution VI requires 32 runs — over budget. A 2^(6-2) Resolution IV design needs 16 runs but confounds 2-factor interactions with each other, violating the boss's requirement. The best option: a Definitive Screening Design (DSD) with 2(6)+1 = 13 runs. It estimates all main effects, all quadratic effects, and can identify some 2-factor interactions. Use the remaining 7 runs for replicate center points or confirmation runs. Alternatively, if the boss truly needs ALL 15 two-factor interactions estimated cleanly, push for 32-run Resolution VI — but explain the 20-run budget simply can't do it.",
            "hint": "Count the 2-factor interactions: C(6,2) = 15. Any design estimating all 15 needs at least ~20 degrees of freedom",
        },
    ],
}


NONPARAMETRIC_HANDS_ON = {
    "title": "Nonparametric Tests in DSW",
    "intro": "Hands-on with Mann-Whitney, Kruskal-Wallis, Wilcoxon, and Friedman in DSW. You'll run each test, read the output, and learn when to follow up with Dunn's or Games-Howell post-hoc.",
    "exercise": {
        "title": "Try It: Run a Mann-Whitney Test",
        "steps": [
            "Enter the two-group sample data",
            "Run the Mann-Whitney U test",
            "Read the U statistic, p-value, and rank-biserial correlation",
            "Compare medians and decide if the groups differ",
            "Try the Kruskal-Wallis test with three groups",
        ],
        "dsw_type": "stats:mann_whitney",
        "dsw_config": {"var": "diameter_mm", "group_var": "shift"},
    },
    "content": """
## Running Nonparametric Tests in SVEND's DSW

This section walks through hands-on application of nonparametric tests using the DSW analysis engine. You'll learn the exact workflow for each test type.

### Data Requirements

Nonparametric tests in DSW accept:
- **Numeric columns** for continuous/ordinal data
- **Group columns** for between-group comparisons
- **Subject columns** for within-subject designs (Friedman)

Missing values are automatically excluded with a warning.

## Mann-Whitney U Test Walkthrough

**Scenario:** Two suppliers provide the same component. You measure burst pressure (psi) on 20 units from each. Data is right-skewed due to a few very strong units.

### DSW Setup
```
Analysis type: mann_whitney
Data: burst pressure column
Group: supplier column (A or B)
```

### Reading the Output

1. **U statistic** — Number of times a Supplier A value exceeds a Supplier B value. Maximum possible is n₁ × n₂ = 400.
2. **P-value** — Two-sided by default. If p < 0.05, suppliers differ.
3. **Median comparison** — Medians for each group (more robust than means for skewed data).
4. **Rank-biserial correlation** — Effect size for Mann-Whitney. Ranges from -1 to +1.
   - |r| < 0.3: small effect
   - |r| 0.3-0.5: medium
   - |r| > 0.5: large

### When to Use One-Sided

If your question is directional ("Is Supplier A *stronger* than B?"), use a one-sided test. The two-sided p-value divided by 2 gives the one-sided p-value (when the direction matches your hypothesis).

## Kruskal-Wallis Walkthrough

**Scenario:** Three shifts (A, B, C) produce the same part. You measure surface roughness (Ra) on 15 parts per shift. The data shows heavy tails.

### DSW Setup
```
Analysis type: kruskal
Data: roughness column
Group: shift column (A, B, C)
```

### Output Interpretation

1. **H statistic** — Chi-square approximation. Larger H = more group difference.
2. **P-value** — If significant, at least one shift differs.
3. **Mean ranks** by group — Shows which shift tends higher/lower.

### Post-Hoc: Dunn's Test

If Kruskal-Wallis is significant, follow up with:
```
Analysis type: dunn
Data: roughness column
Group: shift column
```

Returns pairwise comparisons with adjusted p-values. Tells you exactly which shifts differ.

## Wilcoxon Signed-Rank Walkthrough

**Scenario:** Measure surface finish before and after a process change. 25 paired observations. Differences are not normally distributed.

### DSW Setup
```
Analysis type: wilcoxon
Data: two columns (before, after)
```

### Output

1. **W statistic** — Sum of positive (or negative) ranks.
2. **P-value** — Tests whether differences are systematically positive or negative.
3. **Median difference** — Point estimate of the typical change.

### Interpreting Signed Ranks

The test ranks the absolute differences, then assigns the sign back. Large positive W means most large differences are positive (after > before). This is robust to outlier differences because it works with ranks.

## Friedman Test Walkthrough

**Scenario:** Five panelists rate four formulations on a 1-10 scale. Interest: do formulations differ in perceived quality?

### DSW Setup
```
Analysis type: friedman
Data: rating columns (one per formulation)
Subject: panelist column
```

### Output

1. **Friedman chi-square** — Test statistic.
2. **P-value** — Tests whether at least one formulation differs.
3. **Mean ranks** by formulation — Shows ordering.

Follow up with Dunn's post-hoc for pairwise comparisons if significant.

## Games-Howell Post-Hoc

When group variances are clearly unequal (common in real data):

```
Analysis type: games_howell
Data: response column
Group: group column
```

Returns pairwise comparisons that don't assume equal variances. More conservative than Dunn's but appropriate when variance heterogeneity is present.

## Practical Tips

1. **Always visualize first.** Box plots by group show you the story before the test confirms it.
2. **Report medians, not means** for nonparametric analyses. Medians match what the test is actually evaluating.
3. **Report effect sizes.** Rank-biserial correlation for two groups; Kendall's W for Friedman.
4. **Check sample sizes.** Mann-Whitney needs ≥8 per group for reliable p-values. Kruskal-Wallis needs ≥5 per group.
5. **Ties handling.** DSW uses mid-rank method for ties (standard approach). Many ties reduce power slightly.
""",
    "interactive": {
        "type": "dsw_demo",
        "config": {
            "analysis_type": "mann_whitney",
            "description": "Try running a Mann-Whitney test. Enter two groups of data to compare.",
            "sample_data": {
                "group_a": [23, 45, 12, 67, 34, 89, 15, 42, 56, 31],
                "group_b": [45, 67, 89, 34, 78, 92, 55, 71, 43, 61],
            },
        },
    },
    "key_takeaways": [
        "DSW provides mann_whitney, kruskal, wilcoxon, friedman, dunn, and games_howell",
        "Always follow up significant Kruskal-Wallis with Dunn's or Games-Howell post-hoc",
        "Report medians and rank-based effect sizes for nonparametric results",
        "Games-Howell is the safe post-hoc when variances are unequal",
        "Visualize with box plots before running the test — the plot often tells the whole story",
    ],
    "practice_questions": [
        {
            "question": "You run Kruskal-Wallis on defect counts across 4 suppliers and get H=11.3, p=0.010. Dunn's post-hoc shows: A-B p=0.003, A-C p=0.42, A-D p=0.15, B-C p=0.08, B-D p=0.31, C-D p=0.89. Summarize the findings for your quality manager.",
            "answer": "The overall test confirms supplier differences exist (p=0.010). The key finding: Supplier A differs significantly from Supplier B (p=0.003), while Suppliers C and D are indistinguishable from each other and from A and B individually. Check the median defect counts: if A has lower median defects than B, Supplier A is the better performer. Practically: Suppliers C and D perform similarly, somewhere between A and B. Recommend: if A is best, investigate what A does differently. If A is worst, it may need corrective action or replacement.",
            "hint": "Only A vs B is significant after multiple comparison adjustment",
        },
    ],
}


TIME_SERIES_HANDS_ON = {
    "title": "Time Series in DSW",
    "intro": "Walk through decomposition, ACF/PACF, ARIMA fitting, change point detection, and Granger causality in DSW. Each method gets a concrete scenario with step-by-step guidance.",
    "exercise": {
        "title": "Try It: Decompose and Forecast",
        "steps": [
            "Load the sample time series data",
            "Run decomposition with period=12 and additive model",
            "Examine the ACF/PACF of the residuals",
            "Fit an ARIMA model (auto or manual order)",
            "Generate a 6-month forecast with prediction intervals",
        ],
        "dsw_type": "stats:descriptive",
    },
    "content": """
## Running Time Series Analyses in SVEND's DSW

DSW provides a full suite of time series methods. This section walks through practical application of each.

### Data Format

Time series data in DSW needs:
- A **time/date column** or sequential index
- One or more **value columns**
- Data should be at **regular intervals** (daily, weekly, monthly)

Missing time points should be handled before analysis (interpolation or explicit NA).

## Decomposition Walkthrough

**Start here.** Decomposition reveals the structure of your time series before modeling.

### DSW Setup
```
Analysis type: decomposition
Data: value column
Period: seasonal period (12 for monthly, 7 for daily, 4 for quarterly)
Model: additive or multiplicative
```

### Output

1. **Observed** — Your raw data
2. **Trend** — Long-term direction (smoothed)
3. **Seasonal** — Repeating pattern within each period
4. **Residual** — What's left (should look random)

### Choosing Additive vs. Multiplicative

- **Additive:** Seasonal swings stay the same size regardless of level. Use when seasonal amplitude is constant.
- **Multiplicative:** Seasonal swings grow as the level increases. More common in business/economic data.

**Quick check:** If seasonal amplitude is proportional to the level, use multiplicative.

## ACF/PACF Analysis

Essential for ARIMA model identification.

### DSW Setup
```
Analysis type: acf_pacf
Data: value column
Max lags: typically 2-3 seasonal periods (e.g., 36 for monthly data)
```

### Reading the Plots

**ACF (Autocorrelation Function):**
- Blue shaded region = significance bounds (95% CI)
- Spikes outside the bounds are significant
- Slow decay → non-stationary (need differencing)
- Cuts off at lag q → MA(q) model

**PACF (Partial Autocorrelation Function):**
- Removes effect of intermediate lags
- Cuts off at lag p → AR(p) model

### Common Patterns

| ACF | PACF | Model |
|---|---|---|
| Exponential decay | Cuts off at lag 1 | AR(1) |
| Cuts off at lag 1 | Exponential decay | MA(1) |
| Exponential decay | Exponential decay | ARMA (mixed) |
| Slow decay, not dying | — | Non-stationary, difference first |
| Spike at lag s, 2s, 3s | — | Seasonal component |

## ARIMA Modeling

### DSW Setup
```
Analysis type: arima
Data: value column
Order: [p, d, q] (or let DSW auto-select)
```

### Auto-Selection

If you don't specify the order, DSW will search over reasonable (p,d,q) combinations and select based on AIC. This is usually a good starting point, but you should verify with residual diagnostics.

### Model Diagnostics

Check the residuals:
1. **Residual plot** — Should look random (no patterns)
2. **ACF of residuals** — No significant spikes (white noise)
3. **Normality** — Q-Q plot should be roughly linear
4. **Ljung-Box test** — Tests for remaining autocorrelation (p > 0.05 = good)

If diagnostics fail, try a different order or consider SARIMA.

## SARIMA for Seasonal Data

### DSW Setup
```
Analysis type: sarima
Data: value column
Order: [p, d, q]
Seasonal order: [P, D, Q, s]
```

**Example:** Monthly data with yearly seasonality:
- Non-seasonal: (1,1,1) — one AR, one difference, one MA
- Seasonal: (1,1,1,12) — one seasonal AR, one seasonal difference, one seasonal MA, period 12

### Forecasting

ARIMA/SARIMA in DSW returns:
- Point forecasts for specified horizon
- 95% prediction intervals
- Model summary (coefficients, AIC, BIC)

**Caution:** Prediction intervals widen rapidly. Forecasts beyond 2-3 seasonal periods are unreliable for most business data.

## Change Point Detection

### DSW Setup
```
Analysis type: changepoint
Data: value column
```

### Output

1. **Change point locations** — Times where the process shifted
2. **Segment means/variances** — Before and after each change
3. **Confidence** — How certain the algorithm is about each change point

### Bayesian Change Point Detection

For uncertainty quantification:
```
Analysis type: bayes_changepoint
Data: value column
```

Returns posterior probability of a change point at each time, rather than just a point estimate.

## Granger Causality

### DSW Setup
```
Analysis type: granger
Data: two columns (X and Y)
Max lags: typically 4-12 depending on data frequency
```

### Output

1. **F-statistics** at each lag
2. **P-values** — Does X Granger-cause Y? Does Y Granger-cause X?
3. **Optimal lag** (by AIC)

### Interpretation Checklist

- ✅ Both series are stationary (difference if needed)
- ✅ The relationship is specific (X→Y but not Y→X is most informative)
- ✅ No obvious confounders driving both series
- ✅ Results are robust to lag selection

## Multi-Vari Analysis

### DSW Setup
```
Analysis type: multi_vari
Data: measurement column
Factors: part, within-part position, time period
```

### Reading Multi-Vari Charts

The chart shows variation at three levels:
- **Within-piece** — variation across measurement positions on the same part
- **Piece-to-piece** — variation between different parts within the same time period
- **Time-to-time** — variation between different time periods

The level with the most variation is where you should focus improvement efforts.
""",
    "interactive": {
        "type": "dsw_demo",
        "config": {
            "analysis_type": "decomposition",
            "description": "Try time series decomposition. Upload or enter time series data.",
            "sample_data": {
                "values": [
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
                ],
                "period": 12,
            },
        },
    },
    "key_takeaways": [
        "Always decompose first — it reveals trend, seasonality, and residual structure",
        "ACF/PACF plots are the diagnostic tools for ARIMA model identification",
        "SARIMA adds seasonal terms — essential for monthly/quarterly business data",
        "Change point detection finds when a process shifted — both frequentist and Bayesian versions",
        "Granger causality tests prediction, not true causation — stationarity is required",
        "Multi-vari charts identify the dominant source of variation in manufacturing",
    ],
    "practice_questions": [
        {
            "question": "Monthly sales data: decomposition shows strong seasonality (December peak). After fitting SARIMA(1,1,1)(1,1,1)12, the residual ACF shows a significant spike at lag 1. What should you adjust?",
            "answer": "The significant spike at lag 1 in residual ACF suggests the MA(1) term isn't fully capturing short-term correlation. Try SARIMA(1,1,2)(1,1,1)12 (increase non-seasonal MA from 1 to 2) or SARIMA(2,1,1)(1,1,1)12 (increase AR instead). Compare models using AIC — lower is better. If neither fixes the spike, check whether there's a level shift or outlier causing it. The seasonal terms (1,1,1)12 seem adequate since there are no seasonal-lag spikes.",
            "hint": "Significant residual ACF at lag 1 means the non-seasonal component needs adjustment",
        },
    ],
}
