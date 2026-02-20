"""Learning content: Advanced Statistics."""

from ._datasets import SHARED_DATASET  # noqa: F401


MULTIVARIATE_ANALYSIS = {
    "id": "multivariate-analysis",
    "title": "Multivariate Analysis (PCA & Factor Analysis)",
    "intro": "When you have 20 variables, which ones actually matter? Multivariate analysis reveals the hidden structure in high-dimensional data — finding the handful of latent factors that explain most of the variation. Here you'll watch PCA compress dimensions and see which variables cluster together.",
    "exercise": {
        "title": "Try It: Find Hidden Structure",
        "steps": [
            "Load the clinical dataset — it has 8+ numeric variables per patient",
            "Run PCA and examine the scree plot — how many components matter?",
            "Look at the first 2 components — which variables load together?",
            "Rotate the factor loadings — does the interpretation change?",
            "Toggle variables on and off — see how each one contributes to the components",
            "Decide: how many dimensions capture the essential information?"
        ],
    },
    "content": """
## The Curse of Dimensionality

With many variables, data becomes sparse. Points that seem close in low dimensions are actually far apart in high dimensions. This breaks distance-based methods (clustering, KNN) and makes patterns harder to find.

**PCA** solves this by finding new axes that capture the most variation with the fewest dimensions.

### How PCA Works

1. **Standardize** all variables (mean=0, SD=1)
2. Compute the **covariance matrix** (how variables co-vary)
3. Find **eigenvectors** (directions of maximum variance) and **eigenvalues** (how much variance each direction captures)
4. **Project** data onto the top k eigenvectors

$$\\text{PC}_1 = w_{11}X_1 + w_{12}X_2 + \\ldots + w_{1p}X_p$$

Each PC is a linear combination of the original variables. The weights $w$ are the **loadings** — they tell you which variables contribute most.

### How Many Components to Keep?

**Kaiser's Rule:** Keep components with eigenvalue > 1 (captures more variance than a single original variable).

**Scree Plot:** Plot eigenvalues in descending order. Look for the "elbow" where the curve flattens.

**Cumulative Variance:** Keep enough components to explain 80-90% of total variance.

### PCA vs Factor Analysis

| | PCA | Factor Analysis |
|---|-----|-----------------|
| **Goal** | Maximize variance explained | Find latent factors causing correlations |
| **Model** | Components are linear combos of observed variables | Observed variables are caused by latent factors |
| **Rotation** | Not required | Often applied (varimax, promax) |
| **Use case** | Dimensionality reduction | Understanding latent constructs |

### Interpreting Loadings

A loading near +1 or -1 means that variable is strongly related to the component. Near 0 means it's irrelevant.

**Example from clinical data:**
- PC1 loads heavily on all score variables → "Overall severity"
- PC2 loads on adherence and adverse events → "Treatment compliance"
- PC3 loads on age and baseline → "Patient characteristics"

These interpretations help you understand what the components **mean** — they're not just mathematical abstractions.

### Stability of Components: Bootstrap PCA

Loadings are estimates, not ground truth. **Bootstrap PCA** — resample your data, recompute PCA, see how loadings vary — gives CIs on loadings. If a variable's loading on PC1 is 0.85 [0.72, 0.93], it reliably loads on that component. But if it's 0.45 [0.12, 0.71], the assignment is unstable — don't over-interpret it. Report confidence intervals on the variance explained by each component as well.
""",
    "interactive": {
        "type": "pca_explorer",
        "config": {
            "dataset": "clinical",
            "features": ["baseline_score", "week4_score", "week8_score", "week12_score",
                         "age", "adherence_pct", "adverse_events"],
            "show_scree_plot": True,
            "show_biplot": True,
            "show_loadings_table": True,
            "show_cumulative_variance": True,
        }
    },
    "key_takeaways": [
        "PCA finds directions of maximum variance — the first few components often capture most of the information",
        "Bootstrap PCA to get CIs on loadings — don't over-interpret unstable variable assignments",
        "Loadings tell you which original variables contribute most to each component",
        "Factor analysis assumes latent factors cause observed correlations; PCA just summarizes variance",
        "Always standardize variables before PCA — otherwise high-variance variables dominate",
    ],
    "practice_questions": [
        {
            "question": "PCA on 10 variables gives eigenvalues: 4.2, 2.1, 1.1, 0.8, 0.6, 0.5, 0.3, 0.2, 0.1, 0.1. How many components would you keep?",
            "answer": "Three components (eigenvalues 4.2, 2.1, 1.1), by Kaiser's rule (eigenvalue > 1). These three explain (4.2+2.1+1.1)/10 = 74% of total variance. If you need 80%+, you might include the 4th (0.8), but the scree plot shows a clear elbow after 3.",
            "hint": "Apply Kaiser's rule (eigenvalue > 1) and check cumulative variance."
        },
        {
            "question": "Two variables have loadings of [0.9, 0.1] and [0.85, 0.15] on PC1 and PC2 respectively. What does this suggest?",
            "answer": "Both variables are strongly associated with PC1 and nearly unrelated to PC2. They likely measure the same underlying construct and are highly correlated with each other. If you're doing feature selection, you could probably drop one without losing much information.",
            "hint": "High loadings on the same component mean the variables co-vary — they move together."
        }
    ]
}


CATEGORICAL_DATA = {
    "id": "categorical-data",
    "title": "Categorical Data Analysis",
    "intro": "Not everything is measured on a continuous scale. When your data is 'yes/no', 'red/blue/green', or 'agree/disagree/neutral', you need different tools. The chi-square test and its relatives let you find structure in tables of counts — but the common pitfalls are surprising.",
    "exercise": {
        "title": "Try It: Independence vs Association",
        "steps": [
            "Build a 2×2 table: churn × contract_type",
            "See the expected values under independence",
            "Compare observed to expected — where's the biggest gap?",
            "Watch the chi-square statistic and p-value update",
            "Adjust the cell counts — feel when the association becomes statistically significant",
            "Try a 3×2 table — does adding categories change the conclusion?"
        ],
    },
    "content": """
## Contingency Tables

A contingency table cross-tabulates two categorical variables. Each cell counts how many observations fall into that combination:

|  | Churned | Not Churned | Total |
|--|---------|-------------|-------|
| **Month-to-month** | 40 | 62 | 102 |
| **One-year** | 8 | 48 | 56 |
| **Two-year** | 6 | 36 | 42 |
| **Total** | 54 | 146 | 200 |

### Chi-Square Test of Independence

**Null hypothesis:** The two variables are independent (no association).

**Test statistic:**

$$\\chi^2 = \\sum \\frac{(O_i - E_i)^2}{E_i}$$

Where $E_i$ = (row total × column total) / grand total.

**If variables are independent**, observed counts should be close to expected counts, giving a small $\\chi^2$. Large $\\chi^2$ = evidence of association.

### Assumptions and When They Fail

1. **Expected counts ≥ 5** in every cell (rule of thumb). If not, use Fisher's exact test.
2. **Independent observations** — each subject appears in exactly one cell.
3. **Random sampling** from the population.

### Interpreting the Result

A significant chi-square tells you the variables are **associated** but not **how strongly**. Use effect size measures:

**Cramér's V:** $V = \\sqrt{\\frac{\\chi^2}{n \\cdot \\min(r-1, c-1)}}$

| V | Interpretation |
|---|---------------|
| 0.1 | Weak association |
| 0.3 | Moderate association |
| 0.5+ | Strong association |

### Beyond 2×2 Tables

For ordered categories (e.g., low/medium/high), the **Cochran-Armitage trend test** is more powerful than chi-square because it uses the ordering information.

For matched pairs (before/after on the same subjects), use **McNemar's test** instead.

### Mosaic Plots

A mosaic plot visualizes contingency tables. Each rectangle's width is proportional to the column total, and its height to the conditional proportion. Residuals are shown by color:
- **Blue:** more observations than expected (positive association)
- **Red:** fewer observations than expected (negative association)
""",
    "interactive": {
        "type": "contingency_explorer",
        "config": {
            "default_rows": ["month_to_month", "one_year", "two_year"],
            "default_cols": ["churned", "not_churned"],
            "editable_cells": True,
            "show_expected_values": True,
            "show_chi_square": True,
            "show_cramers_v": True,
            "show_mosaic_plot": True,
        }
    },
    "key_takeaways": [
        "Chi-square tests whether two categorical variables are independent (no association)",
        "Expected cell counts must be at least 5; use Fisher's exact test for small samples",
        "A significant result says 'there is an association' but not how strong — check Cramér's V",
        "Mosaic plots visualize the pattern of association in contingency tables",
        "For ordered categories, the Cochran-Armitage trend test is more powerful than chi-square",
    ],
    "practice_questions": [
        {
            "question": "A 2×2 table has chi-square = 12.4 with p = 0.0004, but Cramér's V = 0.08. How do you interpret this?",
            "answer": "Statistically significant but practically trivial. The large sample size (n = 12.4/0.08² ≈ 1,937) gives the test enough power to detect a tiny effect. The association is real but too small to matter for decision-making. With very large samples, almost any non-zero association becomes 'significant' — always check effect size alongside p-value.",
            "hint": "A small Cramér's V with a very small p-value typically means a large sample is detecting a trivial effect."
        },
        {
            "question": "You want to test whether patient improvement (none/some/major) is related to treatment group (drug/placebo). Which test should you use?",
            "answer": "The Cochran-Armitage trend test, because the outcome variable (improvement) has a natural ordering. A standard chi-square would treat 'none', 'some', and 'major' as unordered categories and lose the information that 'major' > 'some' > 'none'. The trend test is more powerful for detecting a dose-response or treatment-response gradient.",
            "hint": "The outcome categories have a natural order — is there a test that exploits this?"
        }
    ]
}


BAYESIAN_DEPTH = {
    "id": "bayesian-depth",
    "title": "Bayesian Methods in Depth",
    "intro": "You learned Bayesian thinking in Foundations. Now let's go deeper — into prior distributions, posterior updating with real data, and the practical question of when Bayesian methods give you something frequentist methods can't. You'll watch posteriors shift in real time as sample size grows.",
    "exercise": {
        "title": "Try It: Watch the Posterior Converge",
        "steps": [
            "Set a strong prior (narrow bell curve centered at your belief)",
            "Add 5 data points — the posterior barely moves",
            "Add 50 data points — the posterior starts to dominate over the prior",
            "Add 200 data points — the prior is almost irrelevant now",
            "Switch to a weak (flat) prior — see how even 5 data points drive the posterior",
            "Try a deliberately wrong prior — watch the data eventually overwhelm it"
        ],
    },
    "content": """
## From Point Estimates to Distributions

Frequentist statistics gives you a **point estimate** and a confidence interval. Bayesian statistics gives you a full **posterior distribution** — a complete picture of what you believe after seeing the data.

$$P(\\theta | \\text{data}) = \\frac{P(\\text{data} | \\theta) \\cdot P(\\theta)}{P(\\text{data})}$$

- **Prior** $P(\\theta)$: What you believed before data
- **Likelihood** $P(\\text{data} | \\theta)$: How probable is the observed data for each possible parameter value
- **Posterior** $P(\\theta | \\text{data})$: Your updated belief

### Choosing Priors

| Prior Type | When to Use | Example |
|-----------|-------------|---------|
| **Uninformative (flat)** | No prior knowledge | Uniform(0, 1) for a probability |
| **Weakly informative** | Ruling out absurd values | Normal(0, 10) — centered but wide |
| **Informative** | Real prior knowledge | Normal(25, 0.5) for a manufacturing dimension based on historical data |
| **Skeptical** | Testing a bold claim | Prior concentrated at the null hypothesis |

### How Much Does the Prior Matter?

The influence of the prior depends on the ratio of prior precision to data precision:

- **Small sample + strong prior:** Prior dominates → your belief barely changes
- **Large sample + strong prior:** Data overwhelms the prior → belief converges to the data
- **Small sample + weak prior:** Data drives everything → unstable estimates
- **Large sample + weak prior:** Data dominates anyway → same as frequentist

As $n \\to \\infty$, the posterior converges to the same answer regardless of prior (for well-behaved models). This is called **asymptotic consistency**.

### Credible Intervals vs Confidence Intervals

**95% Credible Interval (Bayesian):** "There is a 95% probability the parameter lies in this range."

**95% Confidence Interval (Frequentist):** "If we repeated this experiment many times, 95% of such intervals would contain the true value."

The Bayesian interpretation is what most people *think* confidence intervals mean — and it's actually the more useful statement for decision-making.

### When to Go Bayesian

- **Small samples** where prior information is genuinely available
- **Sequential analysis** where you update beliefs as data arrives (no need to pre-specify sample size)
- **Decision-making** where you need the probability of hypotheses, not just p-values
- **Hierarchical data** where Bayesian shrinkage improves estimates for small subgroups

### When Frequentist Is Fine

- **Large samples** where prior barely matters
- **Regulatory settings** where frequentist methods are required
- **Quick screening** where you need a fast yes/no answer
""",
    "interactive": {
        "type": "posterior_visualizer",
        "config": {
            "prior_types": ["flat", "weak", "informative", "skeptical"],
            "show_prior_curve": True,
            "show_likelihood_curve": True,
            "show_posterior_curve": True,
            "sample_size_slider": True,
            "min_n": 1,
            "max_n": 500,
            "show_credible_interval": True,
            "show_convergence": True,
        }
    },
    "key_takeaways": [
        "Bayesian analysis gives you a full posterior distribution, not just a point estimate",
        "The prior's influence decreases as sample size increases — with enough data, the prior doesn't matter",
        "Credible intervals have the intuitive interpretation people wrongly attribute to confidence intervals",
        "Informative priors based on real knowledge improve small-sample estimates via shrinkage",
        "Bayesian and frequentist methods converge with large samples — the difference matters most with small n",
    ],
    "practice_questions": [
        {
            "question": "You have historical data showing a process mean of 25.00mm with SD=0.05. A new batch of 10 measurements has mean 25.08mm. Should you use an informative prior or a flat prior?",
            "answer": "Use an informative prior — Normal(25.00, 0.05). You have genuine historical knowledge that the process typically centers at 25.00mm. The informative prior will shrink the estimate somewhat toward 25.00, which is appropriate with only 10 measurements. The posterior mean will be somewhere between 25.00 (prior) and 25.08 (data), weighted by their relative precisions. If the new batch truly shifted, more data will eventually overwhelm the prior.",
            "hint": "You have real historical data — that's exactly what informative priors are for."
        },
        {
            "question": "A colleague says 'I used a flat prior so my analysis is objective.' Is this true?",
            "answer": "No — flat priors are not truly 'objective.' First, a flat prior on one scale is not flat on transformed scales (e.g., flat on σ is not flat on σ²). Second, a flat prior on an unbounded parameter is improper (doesn't integrate to 1) and can give improper posteriors in some models. Third, choosing a flat prior is itself a choice — you're saying 'a parameter of 1000 is just as likely as 1,' which may be absurd. Weakly informative priors that rule out impossible values are often more principled than flat priors.",
            "hint": "Is saying 'every value is equally likely' truly objective? What happens on different scales?"
        }
    ]
}


MIXED_MODELS = {
    "id": "mixed-models",
    "title": "Mixed & Hierarchical Models",
    "intro": "Patients are nested within hospitals. Students within classrooms. Measurements within subjects. When your data has layers, ignoring the hierarchy gives wrong answers — standard errors are too small and effects are overestimated. Mixed models handle this correctly by modeling both fixed and random effects.",
    "exercise": {
        "title": "Try It: See Shrinkage in Action",
        "steps": [
            "View patient trajectories grouped by site — notice sites have different baselines",
            "Toggle 'fixed effects only' — see each site estimated independently",
            "Toggle 'random effects' — watch extreme site estimates shrink toward the grand mean",
            "Examine Site_D (lowest baseline) — its estimate gets pulled UP",
            "Examine Site_E (highest baseline) — its estimate gets pulled DOWN",
            "This is shrinkage: small subgroups borrow strength from the overall estimate"
        ],
    },
    "content": """
## The Hierarchy Problem

Consider our clinical dataset: patients nested within sites. Two wrong approaches:

**Approach 1: Ignore sites** — Pool all patients together. Problem: variation between sites inflates the residual, and you miss site-level effects.

**Approach 2: Separate analysis per site** — Analyze each site independently. Problem: small sites (n=20) give noisy estimates, and you can't compare treatment effects across sites.

### Mixed Models: The Right Answer

Mixed models include both:
- **Fixed effects:** Treatment group (the thing you care about, same across all sites)
- **Random effects:** Site intercept (different for each site, drawn from a distribution)

$$Y_{ij} = \\underbrace{(\\beta_0 + u_j)}_{\\text{site mean}} + \\underbrace{\\beta_1 \\cdot \\text{treatment}}_{\\text{fixed effect}} + \\epsilon_{ij}$$

Where $u_j \\sim N(0, \\sigma^2_u)$ is the random site effect.

### Why Shrinkage Is Good

The random effects model estimates each site's mean as a **weighted average** of:
- The site's own data (noisy for small sites)
- The grand mean across all sites (stable but ignores local information)

$$\\hat{u}_j = \\underbrace{\\frac{n_j \\sigma^2_u}{n_j \\sigma^2_u + \\sigma^2}}_{\\text{reliability}} \\cdot (\\bar{y}_j - \\bar{y})$$

Small sites (low $n_j$) get **shrunk** more toward the grand mean. Large sites are trusted more. This is **empirical Bayes** — each subgroup borrows strength from all others.

### When You Need Mixed Models

- **Repeated measures:** Multiple measurements per subject (time points, trials)
- **Clustered data:** Subjects within groups (students/classrooms, patients/hospitals)
- **Longitudinal studies:** Time points nested within subjects
- **Multi-site studies:** Our clinical dataset — patients within sites

### Fixed vs Random Effects

| Fixed Effects | Random Effects |
|---------------|----------------|
| Specific levels matter (treatment A vs B) | Levels are a sample from a population |
| You want to estimate each level's effect | You want to estimate the variance across levels |
| Few levels (2-5) | Many levels (10+) |
| Can be estimated with enough data per level | Estimated via their distribution |

### Report Effect Sizes, Not Just Significance

A mixed model might tell you the treatment effect is "significant" (p=0.02). But **how big is the effect?** Report:

- **Fixed effect estimate with CI:** Treatment effect = 4.2 points [0.8, 7.6]. The CI tells stakeholders whether the effect is clinically meaningful — 0.8 points might not justify the treatment's cost even though it's statistically significant.
- **ICC (Intraclass Correlation Coefficient):** What fraction of total variance is between sites? ICC = 0.15 means 15% of outcome variation is site-level — sites matter.
- **Marginal vs Conditional R²:** Marginal R² (fixed effects only) vs Conditional R² (fixed + random) shows how much the hierarchy explains.

### Common Mistakes

1. **Ignoring clustering** — Standard errors are too small (false positives!)
2. **Too many random effects** — Model doesn't converge with complex random structures
3. **Treating random effects as fixed** — Lose the shrinkage benefit and can't generalize
4. **Reporting p-values without effect sizes** — A significant treatment effect means nothing without knowing **how large** the effect is and whether CIs include clinically irrelevant values
""",
    "interactive": {
        "type": "random_effects_demo",
        "config": {
            "dataset": "clinical",
            "grouping_var": "site",
            "outcome_var": "week12_score",
            "fixed_effect": "treatment_group",
            "show_individual_trajectories": True,
            "show_fixed_only": True,
            "show_random_effects": True,
            "show_shrinkage_arrows": True,
            "show_variance_components": True,
        }
    },
    "key_takeaways": [
        "Mixed models handle hierarchical data by including both fixed and random effects",
        "Always report fixed effects with CIs — a significant treatment p-value without an effect size is incomplete",
        "Shrinkage pulls extreme subgroup estimates toward the grand mean — especially for small subgroups",
        "ICC quantifies how much variance lives between groups — this is the effect size of the hierarchy itself",
        "Use mixed models whenever observations are nested: repeated measures, multi-site, clustered designs",
    ],
    "practice_questions": [
        {
            "question": "Site A has 15 patients with a mean treatment effect of +12 points. The grand mean effect across all 5 sites is +5 points. What would a mixed model estimate for Site A's effect?",
            "answer": "Something between +5 and +12, likely closer to the grand mean (maybe +7 or +8). With only 15 patients, Site A's estimate is noisy — the mixed model 'shrinks' it toward the grand mean. If Site A had 500 patients, the shrinkage would be minimal and the estimate would stay close to +12. The exact amount depends on the ratio of within-site to between-site variance.",
            "hint": "Shrinkage depends on subgroup sample size — small groups get pulled more toward the mean."
        },
        {
            "question": "You have 20 patients each measured at 5 time points. A colleague runs a standard t-test comparing time 1 to time 5. What's wrong?",
            "answer": "The t-test assumes independent observations, but measurements from the same patient are correlated (repeated measures). This means: (1) The effective sample size is less than 100 — more like 20, (2) Standard errors are underestimated, (3) P-values are too small (inflated Type I error). A mixed model with patient as a random effect (or a paired t-test for just two time points) correctly accounts for the within-patient correlation.",
            "hint": "Are the 100 measurements truly independent? What happens to standard errors when observations are correlated?"
        }
    ]
}


RESPONSE_SURFACE = {
    "id": "response-surface",
    "title": "Response Surface Methodology",
    "intro": "DOE tells you which factors matter. RSM tells you where the optimum is. Response Surface Methodology fits curved surfaces to experimental data and navigates you toward the settings that maximize quality, minimize defects, or hit your target. You'll explore contour maps and find the sweet spot.",
    "exercise": {
        "title": "Try It: Navigate to the Optimum",
        "steps": [
            "See the contour plot for two factors — temperature and pressure",
            "Adjust factor sliders — watch the predicted response update",
            "Find the region where response is highest (the red zone on the contour map)",
            "Enable the 3D surface view — see the shape of the response surface",
            "Try the saddle point scenario — understand that not all surfaces have clean peaks",
            "Compare the linear model (flat plane) to quadratic model (curved surface)"
        ],
        "description": "For full RSM with CCD designs, use the Experimenter tool at /app/experimenter/",
    },
    "content": """
## From Screening to Optimization

**Factorial designs** tell you which factors affect the response. **RSM** goes further — it models the **curvature** of the response and finds the optimal settings.

### The Response Surface Model

The second-order (quadratic) model:

$$y = \\beta_0 + \\sum \\beta_i x_i + \\sum \\beta_{ii} x_i^2 + \\sum \\beta_{ij} x_i x_j + \\epsilon$$

This captures:
- **Linear effects** ($\\beta_i$): main trends
- **Quadratic effects** ($\\beta_{ii}$): curvature (is there a peak/valley?)
- **Interactions** ($\\beta_{ij}$): does the effect of one factor depend on another?

### Central Composite Design (CCD)

To fit a quadratic model, you need data at more than two levels per factor. The CCD adds:
- **Center points** (at the middle of each factor range)
- **Axial points** (star points beyond the factorial corners)

This gives you enough information to estimate all the curvature terms.

### Reading Contour Plots

A contour plot shows lines of equal response. Like a topographic map:
- **Circular contours** → clear peak or valley (pure quadratic)
- **Elliptical contours** → elongated ridge (one direction matters more)
- **Saddle shape** → optimal in one direction, pessimal in another (trade-off)

The **stationary point** (maximum, minimum, or saddle) is where the gradient equals zero.

### Ridge Analysis

When the stationary point is outside the experimental region (common!), ridge analysis traces the optimal response along the boundary of the design space.

### CIs on the Optimum — How Well Do You Know the Peak?

The predicted optimum (e.g., "210°C, 35 bar") is a point estimate. How confident are you? The **prediction interval** at the stationary point tells you:

$$\\hat{y}_{\\text{opt}} = 94.2\\% \\;\\; [91.8, 96.6]_{95\\%}$$

And the **confidence region for the optimal factor settings** — an ellipse in factor space — tells you how precisely you've located the peak. A narrow ellipse means you know where the optimum is; a wide one means you need more experiments.

Report the CI on the predicted response at your chosen operating conditions. A response of 94.2% with a PI of [85, 103] isn't very useful — but 94.2% [92, 96] is actionable.

### Practical RSM Workflow

1. **Screen** (fractional factorial) → identify the vital few factors
2. **Steepest ascent** → move toward the region of the optimum
3. **CCD** → fit the quadratic model
4. **Optimize** → find stationary point, check if it's a maximum
5. **Report CIs** → prediction interval at the optimum + confidence region for optimal settings
6. **Confirm** → run confirmation experiments at the predicted optimum
7. **Validate** → confirm result falls within the prediction interval
""",
    "interactive": {
        "type": "rsm_contour_explorer",
        "config": {
            "factors": ["temperature", "pressure"],
            "factor_ranges": {"temperature": [150, 250], "pressure": [10, 50]},
            "response": "yield",
            "show_contour_plot": True,
            "show_3d_surface": True,
            "show_factor_sliders": True,
            "show_predicted_response": True,
            "show_stationary_point": True,
        }
    },
    "key_takeaways": [
        "RSM models curved response surfaces to find optimal process settings",
        "Central Composite Designs provide the data needed to fit quadratic models",
        "Report prediction intervals at the optimum — the CI tells you how precisely you've located the peak",
        "The stationary point is the mathematical optimum but may be outside the experimental region",
        "Always confirm RSM predictions with follow-up experiments — the confirmation should fall within the PI",
    ],
    "practice_questions": [
        {
            "question": "Your contour plot shows a saddle point. Temperature increases yield at low pressure but decreases it at high pressure. What does this mean for optimization?",
            "answer": "A saddle point means there is no single maximum — the 'best' settings depend on constraints. You need to decide: optimize for maximum yield at a fixed pressure? Or find the best trade-off? You might also expand the experimental region to see if there's a true maximum beyond the saddle. The interaction between temperature and pressure is strong, so you can't optimize one without considering the other.",
            "hint": "A saddle point is a maximum in one direction and a minimum in another. What does that mean practically?"
        },
        {
            "question": "Your CCD has 5 center point replicates with responses: 82, 85, 83, 84, 83. What do these tell you?",
            "answer": "Two things: (1) Pure error estimate — the variance of center points (SD ≈ 1.1) estimates the experimental noise without any model assumption. (2) Lack of fit test — if the model predicts the center response poorly, the model is inadequate (significant curvature not captured). The center point mean (83.4) compared to the model prediction at the center tells you whether the model is missing curvature.",
            "hint": "Center point replicates serve two purposes: estimating pure error and testing for model adequacy."
        }
    ]
}


REGRESSION_DIAGNOSTICS = {
    "id": "regression-diagnostics",
    "title": "Regression Diagnostics Deep Dive",
    "intro": "Fitting a regression model is easy. Knowing if it's valid is hard. Diagnostics are the detective work of regression — checking assumptions, finding influential outliers, and detecting when your model is telling a story that's not supported by the data. Here you'll manipulate data points and watch the consequences.",
    "exercise": {
        "title": "Try It: Break a Regression (Then Fix It)",
        "steps": [
            "Fit a regression model to the manufacturing data (diameter vs weight + roughness)",
            "Examine the 4-panel diagnostic plot — what does each panel tell you?",
            "Toggle a high-leverage outlier on/off — watch the regression line shift",
            "Check VIF for multicollinearity — any variables competing?",
            "See Cook's distance — which points disproportionately influence the fit?",
            "Remove the most influential point — does the conclusion change?"
        ],
        "dsw_type": "stats:regression",
        "dsw_config": {
            "response": "diameter_mm",
            "predictors": ["weight_g", "roughness_ra"],
        },
    },
    "content": """
## The Four Diagnostic Plots

### 1. Residuals vs Fitted Values

**What to look for:** Random scatter around zero.

**Red flags:**
- **Funnel shape** → heteroscedasticity (non-constant variance). Try a log transform or weighted regression.
- **Curved pattern** → non-linearity. Add polynomial terms or use a different model.
- **Outlier cluster** → subgroup with different behavior. Investigate those data points.

### 2. Normal Q-Q Plot

**What to look for:** Points following the diagonal line.

**Red flags:**
- **S-curve** → heavy tails (more extreme values than expected)
- **Banana shape** → skewness. Try a log or Box-Cox transform.
- **Individual outliers** at the ends → possible data entry errors or genuine unusual cases.

### 3. Scale-Location (Spread-Level) Plot

**What to look for:** Horizontal band (constant spread).

**Red flags:**
- **Upward trend** → variance increases with fitted values. Classic heteroscedasticity.
- **Non-constant spread** → prediction intervals will be wrong (too wide for some, too narrow for others).

### 4. Residuals vs Leverage

**What to look for:** Points within Cook's distance contour lines.

**Key concepts:**
- **Leverage** = how far a point is from the center of the X-space. High leverage = potential to influence the fit.
- **Residual** = how far a point is from the fit. Large residual = poor fit.
- **Cook's distance** = leverage × residual². Combines both: how much does removing this point change the model?

Points with Cook's D > 4/n are considered influential.

### Multicollinearity: VIF

**Variance Inflation Factor** measures how much each predictor's coefficient variance is inflated due to correlation with other predictors:

$$VIF_j = \\frac{1}{1 - R^2_j}$$

Where $R^2_j$ is the R-squared from regressing predictor $j$ on all other predictors.

| VIF | Interpretation |
|-----|---------------|
| 1 | No collinearity |
| 1-5 | Moderate (usually acceptable) |
| 5-10 | High (coefficients becoming unreliable) |
| > 10 | Severe (don't trust individual coefficients) |

**What to do about high VIF:**
1. Remove one of the correlated predictors
2. Combine them (PCA, average)
3. Use regularization (Ridge regression)
4. Accept it if you only care about prediction, not individual coefficients

### CIs on Coefficients: The Real Diagnostic

Before checking residual plots, look at **coefficient CIs**. A coefficient of $\\beta_1 = 2.3$ with CI [0.1, 4.5] is barely significant — the effect might be trivially small. Meanwhile $\\beta_2 = 0.8$ with CI [0.6, 1.0] is precisely estimated and reliably nonzero.

When an influential point shifts $\\beta_1$ from 2.3 to 0.4, check both CIs: if the original CI was [0.1, 4.5] and the new one is [-1.2, 2.0], the coefficient was already fragile. The influential point isn't "creating" a problem — it's revealing one. **Report coefficients with CIs, and note when CIs are sensitive to individual observations.**

### Influential Points vs Outliers

Not all outliers are influential, and not all influential points are outliers:

- **High leverage, small residual:** On the fit but could pull it if wrong
- **Low leverage, large residual:** Far from fit but too central to influence it
- **High leverage, large residual:** DANGEROUS — influencing the fit in a bad direction
""",
    "interactive": {
        "type": "diagnostic_dashboard",
        "config": {
            "dataset": "manufacturing",
            "show_residuals_vs_fitted": True,
            "show_qq_plot": True,
            "show_scale_location": True,
            "show_residuals_vs_leverage": True,
            "show_vif_table": True,
            "show_cooks_distance": True,
            "interactive_point_toggle": True,
        }
    },
    "key_takeaways": [
        "Always check diagnostic plots before trusting regression results",
        "Report coefficients with CIs — a wide CI is a more honest diagnostic than a small p-value",
        "Cook's distance combines leverage and residual size to find influential points",
        "If removing one point changes the CI from excluding zero to including it, the result was always fragile",
        "VIF > 10 indicates severe multicollinearity — individual coefficients are untrustworthy",
        "An influential point is not necessarily 'bad' — investigate before removing",
    ],
    "practice_questions": [
        {
            "question": "Your residuals vs fitted plot shows a clear U-shape (curved pattern). What's wrong and how do you fix it?",
            "answer": "The model is missing a nonlinear relationship. The straight-line model underpredicts at the extremes and overpredicts in the middle. Solutions: (1) Add a quadratic term (x²) for the predictor causing the curvature, (2) Apply a transformation (log, sqrt) to the predictor or response, (3) Use a different model entirely (polynomial, GAM). Look at which predictor's partial residual plot shows the curvature to know which one to transform.",
            "hint": "A curved pattern in residuals means the model is missing something — what kind of term captures curves?"
        },
        {
            "question": "Removing one observation changes your regression coefficient from 2.3 (p=0.01) to 0.4 (p=0.65). What should you do?",
            "answer": "Your conclusion depends on a single observation — this is a serious problem. Steps: (1) Verify the observation is correct (not a data entry error), (2) Report both results (with and without the point) transparently, (3) Consider if this point represents a genuine subpopulation that behaves differently, (4) Collect more data to reduce the influence of any single point, (5) Use robust regression methods that are less sensitive to outliers. Never silently remove points to get the result you want.",
            "hint": "If one point changes your conclusion, how reliable is the conclusion? What's the right thing to do?"
        }
    ]
}
