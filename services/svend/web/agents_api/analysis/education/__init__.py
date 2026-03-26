"""DSW Education Content — hand-written education blocks for all 230 analyses.

Each entry is keyed by (analysis_type, analysis_id) and contains:
    title:   Display title for the collapsible education panel
    content: HTML definition list (<dl>/<dt>/<dd>) with Q&A pairs

CR: 5528303a — INIT-009 / E9-005
"""


def get_education(analysis_type, analysis_id):
    """Look up education content for an analysis. Returns dict or None."""
    return EDUCATION_CONTENT.get((analysis_type, analysis_id))


EDUCATION_CONTENT = {
    # ═══════════════════════════════════════════════════════════════════
    # STATS — Hypothesis Testing
    # ═══════════════════════════════════════════════════════════════════
    ("stats", "ttest"): {
        "title": "Understanding the One-Sample t-Test",
        "content": (
            "<dl>"
            "<dt>What is a one-sample t-test?</dt>"
            "<dd>It tests whether a sample mean differs significantly from a hypothesized "
            "value. For example: 'Is the average fill weight different from the target of 500g?'</dd>"
            "<dt>What is the p-value telling me?</dt>"
            "<dd>The probability of observing a result this extreme if the true mean really "
            "equals the hypothesized value. <strong>p &lt; 0.05</strong> is conventionally "
            "considered significant — but always pair it with the effect size.</dd>"
            "<dt>What is Cohen's d?</dt>"
            "<dd>The difference in means divided by the standard deviation. "
            "<strong>0.2</strong> = small, <strong>0.5</strong> = medium, "
            "<strong>0.8</strong> = large. A significant p-value with tiny d means "
            "the difference is real but practically irrelevant.</dd>"
            "<dt>When should I use this?</dt>"
            "<dd>When you have one group of continuous measurements and want to compare "
            "against a known standard or target. Data should be approximately normal "
            "(or n &gt; 30 for the CLT to apply).</dd>"
            "<dt>What about the Bayesian shadow?</dt>"
            "<dd>The Bayes Factor (BF₁₀) accompanies the p-value. Unlike p-values, it can "
            "quantify evidence <em>for</em> the null — BF₁₀ &lt; ⅓ means evidence supports "
            "no difference, which a p-value can never tell you.</dd>"
            "</dl>"
        ),
    },
    ("stats", "ttest2"): {
        "title": "Understanding the Two-Sample t-Test",
        "content": (
            "<dl>"
            "<dt>What is a two-sample t-test?</dt>"
            "<dd>It compares the means of two independent groups. For example: 'Do parts "
            "from Machine A have different dimensions than parts from Machine B?'</dd>"
            "<dt>Equal vs unequal variances?</dt>"
            "<dd>Welch's t-test (default) does not assume equal variances and is robust in "
            "almost all cases. The pooled (Student's) version assumes equal variances — "
            "only use it when Levene's test confirms homogeneity.</dd>"
            "<dt>What is Cohen's d here?</dt>"
            "<dd>The difference in group means divided by the pooled standard deviation. "
            "It lets you compare effect sizes across studies regardless of sample size.</dd>"
            "<dt>What if my data isn't normal?</dt>"
            "<dd>For large samples (n &gt; 30 per group), the t-test is robust to non-normality "
            "via the CLT. For small samples with non-normal data, consider the Mann-Whitney "
            "U test — it will appear in cross-validation diagnostics.</dd>"
            "</dl>"
        ),
    },
    ("stats", "paired_t"): {
        "title": "Understanding the Paired t-Test",
        "content": (
            "<dl>"
            "<dt>What is a paired t-test?</dt>"
            "<dd>It tests whether the mean difference between paired observations is zero. "
            "For example: 'Did the training program change test scores?' where each person "
            "is measured before and after.</dd>"
            "<dt>Why paired instead of two-sample?</dt>"
            "<dd>Pairing removes between-subject variability. Each subject serves as their own "
            "control, making the test more powerful for detecting real changes.</dd>"
            "<dt>What assumptions does it make?</dt>"
            "<dd>The <em>differences</em> (not the raw values) should be approximately normal. "
            "With n &gt; 30 pairs this is rarely a concern. For small samples with non-normal "
            "differences, consider the Wilcoxon signed-rank test.</dd>"
            "<dt>How do I interpret the confidence interval?</dt>"
            "<dd>The CI for the mean difference tells you the plausible range of the true "
            "average change. If it excludes zero, the change is statistically significant.</dd>"
            "</dl>"
        ),
    },
    ("stats", "anova"): {
        "title": "Understanding One-Way ANOVA",
        "content": (
            "<dl>"
            "<dt>What is ANOVA?</dt>"
            "<dd>Analysis of Variance tests whether the means of three or more groups differ. "
            "It answers: 'Is at least one group mean different from the others?'</dd>"
            "<dt>What is the F-statistic?</dt>"
            "<dd>The ratio of between-group variance to within-group variance. A large F means "
            "group means are more spread out than you'd expect from random variation alone.</dd>"
            "<dt>What is eta-squared (η²)?</dt>"
            "<dd>The proportion of total variance explained by the grouping factor. "
            "<strong>0.01</strong> = small, <strong>0.06</strong> = medium, "
            "<strong>0.14</strong> = large effect.</dd>"
            "<dt>ANOVA is significant — now what?</dt>"
            "<dd>ANOVA only tells you <em>something</em> differs. Use post-hoc tests "
            "(Tukey HSD, Games-Howell, Dunnett) to identify <em>which</em> pairs differ.</dd>"
            "<dt>What are the assumptions?</dt>"
            "<dd>Independence of observations, approximate normality within groups, and "
            "homogeneity of variances (check with Levene's test). ANOVA is robust to mild "
            "violations with balanced designs and n &gt; 20 per group.</dd>"
            "</dl>"
        ),
    },
    ("stats", "anova2"): {
        "title": "Understanding Two-Way ANOVA",
        "content": (
            "<dl>"
            "<dt>What is two-way ANOVA?</dt>"
            "<dd>It tests the effects of two factors simultaneously, plus their interaction. "
            "For example: 'Do temperature AND pressure affect yield, and does the effect of "
            "temperature depend on pressure level?'</dd>"
            "<dt>What is an interaction effect?</dt>"
            "<dd>When the effect of one factor depends on the level of the other. A significant "
            "interaction means you cannot interpret main effects in isolation — you must "
            "examine the combination.</dd>"
            "<dt>How do I read the interaction plot?</dt>"
            "<dd>Parallel lines = no interaction. Crossing or diverging lines = interaction "
            "present. The more the lines cross, the stronger the interaction.</dd>"
            "<dt>What is partial eta-squared?</dt>"
            "<dd>The variance explained by each factor after accounting for the other factor. "
            "Unlike regular η², it sums to more than 1.0 because each factor's contribution "
            "is measured independently.</dd>"
            "</dl>"
        ),
    },
    ("stats", "correlation"): {
        "title": "Understanding Pearson Correlation",
        "content": (
            "<dl>"
            "<dt>What is Pearson's r?</dt>"
            "<dd>A measure of the linear relationship between two continuous variables, "
            "ranging from −1 (perfect negative) to +1 (perfect positive). r = 0 means "
            "no linear relationship (but nonlinear patterns may exist).</dd>"
            "<dt>What is r²?</dt>"
            "<dd>The proportion of variance in one variable explained by the other. "
            "r = 0.7 means r² = 0.49 — about half the variance is shared.</dd>"
            "<dt>Correlation vs causation?</dt>"
            "<dd>Correlation measures association, not causation. Two variables can correlate "
            "because of a common cause, reverse causation, or coincidence. Use DOE or causal "
            "discovery to establish causal claims.</dd>"
            "<dt>What if the relationship is nonlinear?</dt>"
            "<dd>Pearson's r only captures linear relationships. For monotonic but nonlinear "
            "relationships, use Spearman's rank correlation instead.</dd>"
            "</dl>"
        ),
    },
    ("stats", "chi2"): {
        "title": "Understanding the Chi-Square Test",
        "content": (
            "<dl>"
            "<dt>What is the chi-square test?</dt>"
            "<dd>It tests whether the distribution of categorical observations differs from "
            "what you'd expect. Used for contingency tables (independence) and goodness-of-fit.</dd>"
            "<dt>What is Cramér's V?</dt>"
            "<dd>An effect size for chi-square, ranging from 0 (no association) to 1 (perfect "
            "association). <strong>0.1</strong> = small, <strong>0.3</strong> = medium, "
            "<strong>0.5</strong> = large for 2×2 tables.</dd>"
            "<dt>What are expected counts?</dt>"
            "<dd>The counts you'd see if the variables were independent. Chi-square compares "
            "observed vs expected. Cells with large residuals drive the result.</dd>"
            "<dt>When is chi-square unreliable?</dt>"
            "<dd>When expected counts are below 5 in more than 20% of cells. In that case, "
            "use Fisher's exact test or combine categories.</dd>"
            "</dl>"
        ),
    },
    ("stats", "prop_1sample"): {
        "title": "Understanding the One-Proportion Test",
        "content": (
            "<dl>"
            "<dt>What does this test?</dt>"
            "<dd>Whether an observed proportion differs from a hypothesized value. "
            "For example: 'Is our defect rate different from the 2% target?'</dd>"
            "<dt>What is Cohen's h?</dt>"
            "<dd>An effect size for proportions based on the arcsine transformation. "
            "<strong>0.2</strong> = small, <strong>0.5</strong> = medium, "
            "<strong>0.8</strong> = large difference between proportions.</dd>"
            "<dt>Normal approximation vs exact?</dt>"
            "<dd>The normal approximation works well when np and n(1−p) are both ≥ 10. "
            "For small samples or extreme proportions, the exact binomial test is used.</dd>"
            "<dt>How wide should my confidence interval be?</dt>"
            "<dd>The CI width depends on sample size and the proportion itself. Proportions "
            "near 0 or 1 need larger samples for the same precision as proportions near 0.5.</dd>"
            "</dl>"
        ),
    },
    ("stats", "prop_2sample"): {
        "title": "Understanding the Two-Proportion Test",
        "content": (
            "<dl>"
            "<dt>What does this test?</dt>"
            "<dd>Whether two groups have different proportions. For example: 'Is the defect "
            "rate on Line A different from Line B?'</dd>"
            "<dt>What is the test statistic?</dt>"
            "<dd>A z-statistic comparing the two proportions, pooled under the null hypothesis "
            "of equality. Large absolute z values indicate a significant difference.</dd>"
            "<dt>What about small samples?</dt>"
            "<dd>When expected counts are small, Fisher's exact test is more reliable. "
            "The normal approximation requires at least 10 successes and 10 failures per group.</dd>"
            "<dt>Absolute vs relative difference?</dt>"
            "<dd>Always report both. A shift from 1% to 2% is a 1 percentage point increase "
            "but a 100% relative increase. Context determines which matters more.</dd>"
            "</dl>"
        ),
    },
    ("stats", "fisher_exact"): {
        "title": "Understanding Fisher's Exact Test",
        "content": (
            "<dl>"
            "<dt>What is Fisher's exact test?</dt>"
            "<dd>An exact test of independence for 2×2 contingency tables that works "
            "regardless of sample size. Unlike chi-square, it does not rely on large-sample "
            "approximations.</dd>"
            "<dt>When should I use it?</dt>"
            "<dd>When any expected cell count is below 5, or when total sample size is small "
            "(n &lt; 30). It's always valid — chi-square is just faster for large samples.</dd>"
            "<dt>What is the odds ratio?</dt>"
            "<dd>The ratio of odds in one group vs another. OR = 1 means no association. "
            "OR = 3 means the odds are 3× higher in the first group.</dd>"
            "<dt>One-sided vs two-sided?</dt>"
            "<dd>Two-sided tests whether there is <em>any</em> association. One-sided tests "
            "a specific direction (e.g., 'Is treatment better than control?').</dd>"
            "</dl>"
        ),
    },
    ("stats", "normality"): {
        "title": "Understanding Normality Tests",
        "content": (
            "<dl>"
            "<dt>Why test for normality?</dt>"
            "<dd>Many statistical tests (t-tests, ANOVA, regression) assume approximately "
            "normal data. Normality testing checks this assumption before analysis.</dd>"
            "<dt>Which test is used?</dt>"
            "<dd>Shapiro-Wilk (preferred for n &lt; 5000, most powerful), Anderson-Darling "
            "(emphasizes tails), and Kolmogorov-Smirnov (less powerful but works for any n).</dd>"
            "<dt>My data 'failed' normality — what now?</dt>"
            "<dd>Consider: (1) transformation (Box-Cox, Johnson), (2) nonparametric alternatives "
            "(Mann-Whitney, Kruskal-Wallis), (3) ignoring it if n is large (CLT protects "
            "most tests at n &gt; 30). The Q-Q plot matters more than the p-value.</dd>"
            "<dt>Common pitfall</dt>"
            "<dd>With large samples, normality tests reject even trivial departures. "
            "With small samples, they have low power and may miss serious non-normality. "
            "Always examine the Q-Q plot visually alongside the test result.</dd>"
            "</dl>"
        ),
    },
    ("stats", "equivalence"): {
        "title": "Understanding Equivalence Testing (TOST)",
        "content": (
            "<dl>"
            "<dt>What is equivalence testing?</dt>"
            "<dd>Instead of testing 'are they different?' it tests 'are they equivalent "
            "within a practical margin?' Uses Two One-Sided Tests (TOST) to confirm the "
            "difference falls within ±Δ (the equivalence margin).</dd>"
            "<dt>Why not just use a regular t-test?</dt>"
            "<dd>A non-significant t-test means 'insufficient evidence of difference' — not "
            "'evidence of equivalence'. Equivalence testing directly tests the claim "
            "that the difference is small enough to ignore.</dd>"
            "<dt>How do I set the equivalence margin?</dt>"
            "<dd>The margin should reflect practical significance — the smallest difference "
            "that would matter operationally. For example, ±0.5mm for a machined part, "
            "or ±2% for a chemical assay.</dd>"
            "<dt>When is this used?</dt>"
            "<dd>Supplier qualification, method comparison, process transfer validation, "
            "bioequivalence studies. Any time you need to prove 'no meaningful difference'.</dd>"
            "</dl>"
        ),
    },
    ("stats", "variance_test"): {
        "title": "Understanding Variance Tests",
        "content": (
            "<dl>"
            "<dt>What does this test?</dt>"
            "<dd>Whether the variance (or standard deviation) of a sample differs from a "
            "target, or whether two groups have different variances. Important for process "
            "consistency evaluation.</dd>"
            "<dt>Which tests are used?</dt>"
            "<dd>Chi-square test for one-sample variance, F-test for two-sample variance "
            "comparison, and Levene's test (robust to non-normality) for equal variance checks.</dd>"
            "<dt>Why does variance matter?</dt>"
            "<dd>Two processes can have the same mean but very different consistency. In "
            "manufacturing, variance directly relates to defect rates — even a centered "
            "process produces defects if spread is too wide.</dd>"
            "<dt>Sensitivity to non-normality</dt>"
            "<dd>The chi-square and F-tests are very sensitive to non-normality — more so "
            "than t-tests. For non-normal data, prefer Levene's test or bootstrap-based "
            "variance comparison.</dd>"
            "</dl>"
        ),
    },
    ("stats", "f_test"): {
        "title": "Understanding the F-Test for Equal Variances",
        "content": (
            "<dl>"
            "<dt>What is the F-test?</dt>"
            "<dd>It compares the variances of two groups by taking their ratio. F = 1 "
            "means equal variances; large or small F indicates unequal variances.</dd>"
            "<dt>When is it used?</dt>"
            "<dd>As a preliminary check before choosing between pooled (Student's) and "
            "Welch's t-test. Also used in ANOVA decomposition and regression significance.</dd>"
            "<dt>Key limitation</dt>"
            "<dd>The F-test is extremely sensitive to non-normality — it tests normality "
            "as much as it tests equal variances. For robust variance comparison, "
            "use Levene's or Brown-Forsythe test instead.</dd>"
            "<dt>How to read the result</dt>"
            "<dd>If p &lt; 0.05, variances are significantly different. Use Welch's t-test "
            "rather than pooled. The F-ratio tells you how many times larger one variance "
            "is relative to the other.</dd>"
            "</dl>"
        ),
    },
    ("stats", "runs_test"): {
        "title": "Understanding the Runs Test",
        "content": (
            "<dl>"
            "<dt>What is the runs test?</dt>"
            "<dd>It tests whether a sequence of observations is random. A 'run' is a "
            "consecutive sequence of values above or below the median. Too few or too "
            "many runs indicates non-randomness.</dd>"
            "<dt>What patterns does it detect?</dt>"
            "<dd>Too few runs suggests trending or clustering. Too many runs suggests "
            "oscillation or alternation. Both indicate the process is not in a random state.</dd>"
            "<dt>When should I use it?</dt>"
            "<dd>Before SPC charting (to verify independence), after regression (to check "
            "residual randomness), or when validating that a process has no time-dependent "
            "patterns.</dd>"
            "<dt>Limitations</dt>"
            "<dd>The runs test has relatively low power — it detects gross non-randomness but "
            "can miss subtle patterns. Use autocorrelation analysis for more sensitive "
            "detection of serial dependence.</dd>"
            "</dl>"
        ),
    },
    ("stats", "sign_test"): {
        "title": "Understanding the Sign Test",
        "content": (
            "<dl>"
            "<dt>What is the sign test?</dt>"
            "<dd>The simplest nonparametric test for paired data. It counts how many "
            "differences are positive vs negative and tests whether the median "
            "difference is zero.</dd>"
            "<dt>When to use it?</dt>"
            "<dd>When paired data is ordinal or heavily non-normal and even the Wilcoxon "
            "signed-rank test assumptions are questionable. It only uses the direction "
            "of change, not the magnitude.</dd>"
            "<dt>Why is it less powerful?</dt>"
            "<dd>By ignoring the magnitude of differences, it discards information. "
            "The Wilcoxon signed-rank test is more powerful when differences are "
            "symmetric. Use the sign test only when magnitude is unreliable.</dd>"
            "<dt>Interpreting the result</dt>"
            "<dd>The test reports the number of positive and negative differences. "
            "If the split is close to 50/50, there's no evidence of a systematic "
            "directional change.</dd>"
            "</dl>"
        ),
    },
    ("stats", "poisson_1sample"): {
        "title": "Understanding the One-Sample Poisson Test",
        "content": (
            "<dl>"
            "<dt>What does this test?</dt>"
            "<dd>Whether an observed count rate differs from a hypothesized rate. "
            "For example: 'Is our defect count of 12 per shift different from the "
            "historical rate of 8 per shift?'</dd>"
            "<dt>What is the Poisson distribution?</dt>"
            "<dd>It models the number of events in a fixed interval when events occur "
            "independently at a constant average rate. Common for defect counts, "
            "arrivals, and rare events.</dd>"
            "<dt>Assumptions</dt>"
            "<dd>Events must be independent and occur at a constant rate. If the rate "
            "varies over time or events cluster, the Poisson model is inappropriate — "
            "consider negative binomial or time series approaches.</dd>"
            "<dt>Exact vs approximate</dt>"
            "<dd>For small counts, the exact Poisson test is used. For large counts "
            "(λ &gt; 20), the normal approximation is adequate.</dd>"
            "</dl>"
        ),
    },
    ("stats", "poisson_2sample"): {
        "title": "Understanding the Two-Sample Poisson Test",
        "content": (
            "<dl>"
            "<dt>What does this test?</dt>"
            "<dd>Whether two Poisson rates differ. For example: 'Does Line A have a "
            "different defect rate per hour than Line B?' Accounts for different "
            "exposure times or sample sizes.</dd>"
            "<dt>Rate vs count</dt>"
            "<dd>The test compares rates (counts per unit of exposure), not raw counts. "
            "This is critical when exposure differs — 10 defects in 100 hours vs 8 in "
            "50 hours are very different rates.</dd>"
            "<dt>When to use</dt>"
            "<dd>Comparing defect rates, incident rates, or event frequencies between "
            "two processes, time periods, or conditions when events follow a Poisson process.</dd>"
            "<dt>Alternatives</dt>"
            "<dd>If overdispersion is present (variance &gt; mean), use a negative binomial "
            "model instead. Check the dispersion statistic in the output.</dd>"
            "</dl>"
        ),
    },
    ("stats", "mood_median"): {
        "title": "Understanding Mood's Median Test",
        "content": (
            "<dl>"
            "<dt>What is Mood's median test?</dt>"
            "<dd>A nonparametric test comparing medians of two or more groups. It counts "
            "how many observations in each group fall above vs below the overall median.</dd>"
            "<dt>When to use it?</dt>"
            "<dd>When data is ordinal or heavily skewed and you want to compare central "
            "tendency without parametric assumptions. It's more robust than Kruskal-Wallis "
            "to outliers.</dd>"
            "<dt>Limitations</dt>"
            "<dd>Less powerful than Kruskal-Wallis because it only uses above/below "
            "information, discarding ranks. Use it when robustness to outliers matters "
            "more than power.</dd>"
            "<dt>Interpreting the result</dt>"
            "<dd>A significant result means at least one group's median differs. "
            "The contingency table shows which groups have disproportionate counts "
            "above or below the grand median.</dd>"
            "</dl>"
        ),
    },
}  # end EDUCATION_CONTENT — continued in _extend calls below


# ═══════════════════════════════════════════════════════════════════════════
# We split content across helper dicts and merge them to keep the file
# navigable. Each section adds to EDUCATION_CONTENT.
# ═══════════════════════════════════════════════════════════════════════════


def _extend(entries):
    """Merge a dict of education entries into the main registry."""
    EDUCATION_CONTENT.update(entries)


# ── Stats: Regression ──────────────────────────────────────────────────────

_extend(
    {
        ("stats", "regression"): {
            "title": "Understanding Linear Regression",
            "content": (
                "<dl>"
                "<dt>What is linear regression?</dt>"
                "<dd>It models the relationship between a response variable and one or more "
                "predictors as a straight line (or hyperplane). It estimates how much the "
                "response changes per unit change in each predictor.</dd>"
                "<dt>What is R²?</dt>"
                "<dd>The proportion of variance explained by the model. R² = 0.85 means the "
                "predictors account for 85% of the variation. Adjusted R² penalizes for "
                "adding predictors that don't genuinely help.</dd>"
                "<dt>What are residuals?</dt>"
                "<dd>The difference between observed and predicted values. Good models have "
                "residuals that are random, centered at zero, and constant in spread. "
                "Patterns in residuals indicate model problems.</dd>"
                "<dt>What is multicollinearity?</dt>"
                "<dd>When predictors are highly correlated with each other. It inflates standard "
                "errors and makes individual coefficient estimates unreliable. Check VIF — "
                "values above 5–10 are concerning.</dd>"
                "<dt>What is the what-if explorer?</dt>"
                "<dd>Adjust predictor values via sliders to see how the predicted response "
                "changes. This helps build intuition about the model and explore scenarios.</dd>"
                "</dl>"
            ),
        },
        ("stats", "logistic"): {
            "title": "Understanding Logistic Regression",
            "content": (
                "<dl>"
                "<dt>What is logistic regression?</dt>"
                "<dd>It models the probability of a binary outcome (pass/fail, yes/no) as a "
                "function of predictors. Unlike linear regression, the output is bounded "
                "between 0 and 1.</dd>"
                "<dt>What is the odds ratio?</dt>"
                "<dd>How much the odds of the outcome change per unit increase in the predictor. "
                "OR = 2 means the odds double. OR = 0.5 means they halve.</dd>"
                "<dt>What is the ROC curve?</dt>"
                "<dd>A plot of sensitivity vs (1 − specificity) across all thresholds. "
                "AUC (area under the curve) summarizes discrimination: 0.5 = random, "
                "0.7–0.8 = acceptable, &gt; 0.8 = good, &gt; 0.9 = excellent.</dd>"
                "<dt>Classification threshold</dt>"
                "<dd>The default 0.5 threshold may not be optimal. In quality contexts, "
                "you may want to lower it to catch more defects (higher sensitivity) at "
                "the cost of more false alarms.</dd>"
                "</dl>"
            ),
        },
        ("stats", "nominal_logistic"): {
            "title": "Understanding Nominal Logistic Regression",
            "content": (
                "<dl>"
                "<dt>What is nominal logistic regression?</dt>"
                "<dd>An extension of logistic regression for outcomes with more than two unordered "
                "categories (e.g., defect type: scratch, dent, discoloration). Each category "
                "is compared against a reference category.</dd>"
                "<dt>How to interpret coefficients</dt>"
                "<dd>Each predictor has a coefficient per category (relative to the reference). "
                "The odds ratio tells you how much the odds of that category vs the reference "
                "change per unit increase in the predictor.</dd>"
                "<dt>When to use</dt>"
                "<dd>When your outcome has 3+ unordered categories. If categories are ordered "
                "(e.g., severity: low/medium/high), use ordinal logistic regression instead.</dd>"
                "<dt>Key assumption</dt>"
                "<dd>Independence of Irrelevant Alternatives (IIA) — adding/removing a category "
                "shouldn't change the relative odds of the remaining ones. Violations suggest "
                "a multinomial probit or nested logit model.</dd>"
                "</dl>"
            ),
        },
        ("stats", "ordinal_logistic"): {
            "title": "Understanding Ordinal Logistic Regression",
            "content": (
                "<dl>"
                "<dt>What is ordinal logistic regression?</dt>"
                "<dd>Models an ordered categorical outcome (e.g., satisfaction: low/medium/high, "
                "severity grades). It respects the ordering that nominal logistic ignores.</dd>"
                "<dt>What is the proportional odds assumption?</dt>"
                "<dd>The effect of each predictor is the same across all cumulative splits. "
                "This is tested automatically — if violated, consider separate binary models "
                "or a partial proportional odds model.</dd>"
                "<dt>How to interpret</dt>"
                "<dd>A positive coefficient means higher predictor values shift the outcome "
                "toward higher categories. The odds ratio applies uniformly across all "
                "cut points (under proportional odds).</dd>"
                "<dt>When to use</dt>"
                "<dd>Rating scales, severity levels, ranked categories — any outcome where "
                "categories have a natural order but the distances between them are unknown.</dd>"
                "</dl>"
            ),
        },
        ("stats", "orthogonal_regression"): {
            "title": "Understanding Orthogonal Regression (Deming)",
            "content": (
                "<dl>"
                "<dt>What is orthogonal regression?</dt>"
                "<dd>Also called Deming regression — it accounts for measurement error in "
                "<em>both</em> variables, not just the response. Standard regression assumes "
                "the predictor is measured without error, which is often false.</dd>"
                "<dt>When to use</dt>"
                "<dd>Method comparison studies (comparing two measurement instruments), "
                "calibration where both x and y have measurement uncertainty, or when "
                "the error ratio between variables is known.</dd>"
                "<dt>What is the error ratio?</dt>"
                "<dd>The ratio of measurement variances (σ²_x / σ²_y). When both have equal "
                "precision, the ratio is 1. The slope estimate is sensitive to this ratio.</dd>"
                "<dt>Key difference from OLS</dt>"
                "<dd>OLS minimizes vertical distances; orthogonal regression minimizes "
                "perpendicular distances to the line. The slope is always steeper than OLS "
                "when measurement error exists in x.</dd>"
                "</dl>"
            ),
        },
        ("stats", "nonlinear_regression"): {
            "title": "Understanding Nonlinear Regression",
            "content": (
                "<dl>"
                "<dt>What is nonlinear regression?</dt>"
                "<dd>Fits a curve (exponential, logistic, power, polynomial, or custom) to data "
                "when the relationship is not linear. The model form must be specified — "
                "the algorithm finds the best-fitting parameters.</dd>"
                "<dt>How does it differ from polynomial regression?</dt>"
                "<dd>Polynomial regression is technically linear (linear in coefficients). "
                "True nonlinear regression has parameters inside nonlinear functions "
                "(e.g., y = a·e^(bx) — b is inside the exponential).</dd>"
                "<dt>Convergence issues</dt>"
                "<dd>Nonlinear optimization can fail to converge or find local optima. "
                "Good starting values are critical. If the fit looks wrong, try different "
                "initial parameter estimates.</dd>"
                "<dt>R² interpretation</dt>"
                "<dd>R² for nonlinear models is not as cleanly interpretable as for linear "
                "models. Use residual plots and prediction intervals to assess fit quality.</dd>"
                "</dl>"
            ),
        },
        ("stats", "poisson_regression"): {
            "title": "Understanding Poisson Regression",
            "content": (
                "<dl>"
                "<dt>What is Poisson regression?</dt>"
                "<dd>Models count data (defects, incidents, arrivals) as a function of predictors. "
                "It ensures predictions are non-negative and accounts for the discrete nature of counts.</dd>"
                "<dt>What is the link function?</dt>"
                "<dd>The log link: log(count) = β₀ + β₁x₁ + ... This means coefficients are "
                "on the log scale — exponentiate them (e^β) to get rate ratios.</dd>"
                "<dt>What is overdispersion?</dt>"
                "<dd>When the variance exceeds the mean (Poisson assumes they're equal). "
                "Check the deviance/df ratio — values &gt; 1.5 suggest overdispersion. "
                "Use negative binomial regression as the alternative.</dd>"
                "<dt>Offset terms</dt>"
                "<dd>When exposure differs across observations (e.g., different shift lengths), "
                "add log(exposure) as an offset. This converts the model from counts to rates.</dd>"
                "</dl>"
            ),
        },
        ("stats", "robust_regression"): {
            "title": "Understanding Robust Regression",
            "content": (
                "<dl>"
                "<dt>What is robust regression?</dt>"
                "<dd>It down-weights outliers and influential observations that distort ordinary "
                "least squares. The most common methods are M-estimation (Huber, bisquare) and "
                "least trimmed squares.</dd>"
                "<dt>When to use</dt>"
                "<dd>When your data has outliers or influential points that you don't want to "
                "remove but shouldn't let dominate the fit. Common in manufacturing data "
                "where occasional extreme values are expected.</dd>"
                "<dt>How does it differ from OLS?</dt>"
                "<dd>OLS gives equal weight to all points. Robust regression reduces the influence "
                "of points with large residuals. The weights column shows how much each observation "
                "contributes — low weights indicate potential outliers.</dd>"
                "<dt>Trade-off</dt>"
                "<dd>Robust regression is less efficient than OLS when data truly is normal "
                "(no outliers). It sacrifices a small amount of precision for protection "
                "against contamination.</dd>"
                "</dl>"
            ),
        },
        ("stats", "stepwise"): {
            "title": "Understanding Stepwise Regression",
            "content": (
                "<dl>"
                "<dt>What is stepwise regression?</dt>"
                "<dd>An automated variable selection method that adds or removes predictors based "
                "on statistical criteria (AIC, BIC, or p-value thresholds) to find a parsimonious model.</dd>"
                "<dt>Forward vs backward vs both</dt>"
                "<dd>Forward starts empty and adds the best predictor at each step. Backward starts "
                "full and removes the weakest. Stepwise (both) can add and remove at each step.</dd>"
                "<dt>Caution</dt>"
                "<dd>Stepwise selection inflates Type I error rates and can overfit. "
                "P-values from stepwise models are biased downward. Use cross-validation "
                "to verify the selected model generalizes.</dd>"
                "<dt>Better alternatives</dt>"
                "<dd>Consider LASSO (L1 regularization) for automated selection with built-in "
                "shrinkage, or best subsets regression for exhaustive comparison.</dd>"
                "</dl>"
            ),
        },
        ("stats", "best_subsets"): {
            "title": "Understanding Best Subsets Regression",
            "content": (
                "<dl>"
                "<dt>What is best subsets?</dt>"
                "<dd>It evaluates all possible combinations of predictors and ranks them by "
                "criteria like adjusted R², Mallows' Cp, AIC, or BIC. Unlike stepwise, it "
                "guarantees finding the globally best model at each subset size.</dd>"
                "<dt>Which criterion to use?</dt>"
                "<dd>Adjusted R² for explanatory models, Cp for prediction accuracy, "
                "BIC for the most parsimonious model (strongest penalty for extra predictors), "
                "AIC for balanced prediction.</dd>"
                "<dt>Computational limits</dt>"
                "<dd>With p predictors, there are 2^p possible models. Above ~15 predictors, "
                "exhaustive search becomes impractical — use LASSO or stepwise instead.</dd>"
                "<dt>Interpreting the output</dt>"
                "<dd>The output shows the best 1-predictor model, best 2-predictor model, etc. "
                "Look for where adding predictors stops improving the criterion — that's "
                "the right model complexity.</dd>"
                "</dl>"
            ),
        },
        ("stats", "glm"): {
            "title": "Understanding Generalized Linear Models (GLM)",
            "content": (
                "<dl>"
                "<dt>What is a GLM?</dt>"
                "<dd>A flexible framework that extends linear regression to non-normal response "
                "distributions. Specify a family (Gaussian, Binomial, Poisson, Gamma) and a "
                "link function to model different types of data.</dd>"
                "<dt>When to use which family?</dt>"
                "<dd><strong>Gaussian</strong>: continuous data (same as OLS). <strong>Binomial</strong>: "
                "binary or proportion data. <strong>Poisson</strong>: count data. "
                "<strong>Gamma</strong>: positive continuous data with variance proportional to mean².</dd>"
                "<dt>What is deviance?</dt>"
                "<dd>The GLM equivalent of residual sum of squares. The deviance/df ratio "
                "indicates goodness of fit — values near 1 suggest adequate fit.</dd>"
                "<dt>How to check fit</dt>"
                "<dd>Deviance residual plots, Q-Q plots of residuals, and the AIC allow model "
                "comparison. Overdispersion (deviance/df &gt; 1.5) may require a quasi-family "
                "or negative binomial model.</dd>"
                "</dl>"
            ),
        },
    }
)


# ── Stats: Nonparametric ──────────────────────────────────────────────────

_extend(
    {
        ("stats", "mann_whitney"): {
            "title": "Understanding the Mann-Whitney U Test",
            "content": (
                "<dl>"
                "<dt>What is the Mann-Whitney U test?</dt>"
                "<dd>The nonparametric alternative to the two-sample t-test. It tests whether "
                "one group tends to have larger values than the other by comparing ranks "
                "rather than means.</dd>"
                "<dt>What does it actually test?</dt>"
                "<dd>Strictly, it tests whether P(X &gt; Y) = 0.5 — i.e., whether a randomly "
                "chosen observation from group A is equally likely to be larger or smaller "
                "than one from group B.</dd>"
                "<dt>What is rank-biserial correlation?</dt>"
                "<dd>An effect size for Mann-Whitney, ranging from −1 to +1. It represents "
                "the proportion of favorable pairs minus unfavorable pairs. "
                "<strong>0.1</strong> = small, <strong>0.3</strong> = medium, "
                "<strong>0.5</strong> = large.</dd>"
                "<dt>When to choose this over a t-test?</dt>"
                "<dd>When data is ordinal, heavily skewed, has outliers, or the sample is too "
                "small to rely on the CLT. It makes no distributional assumptions.</dd>"
                "</dl>"
            ),
        },
        ("stats", "kruskal"): {
            "title": "Understanding the Kruskal-Wallis Test",
            "content": (
                "<dl>"
                "<dt>What is Kruskal-Wallis?</dt>"
                "<dd>The nonparametric alternative to one-way ANOVA. It tests whether three or "
                "more groups come from the same distribution by comparing ranks.</dd>"
                "<dt>What is the H statistic?</dt>"
                "<dd>Based on the sum of ranks within each group. Large H means the groups' "
                "rank distributions differ. Under the null, H follows a chi-square distribution.</dd>"
                "<dt>Post-hoc testing</dt>"
                "<dd>Like ANOVA, a significant Kruskal-Wallis only tells you <em>something</em> "
                "differs. Use Dunn's test with Bonferroni correction for pairwise comparisons.</dd>"
                "<dt>Assumptions</dt>"
                "<dd>Only requires independent observations and ordinal or continuous data. "
                "No normality or equal variance assumptions. However, it assumes similar "
                "distribution shapes across groups for a clean 'median comparison' interpretation.</dd>"
                "</dl>"
            ),
        },
        ("stats", "wilcoxon"): {
            "title": "Understanding the Wilcoxon Signed-Rank Test",
            "content": (
                "<dl>"
                "<dt>What is the Wilcoxon signed-rank test?</dt>"
                "<dd>The nonparametric alternative to the paired t-test. It tests whether the "
                "median of paired differences is zero, using both the sign and magnitude "
                "of differences (unlike the sign test which only uses direction).</dd>"
                "<dt>When to use it?</dt>"
                "<dd>When paired differences are not normally distributed and sample size is "
                "too small for the CLT. It's more powerful than the sign test because it "
                "uses rank information.</dd>"
                "<dt>Key assumption</dt>"
                "<dd>The distribution of differences should be approximately symmetric around "
                "the median. For highly asymmetric differences, the sign test is safer.</dd>"
                "<dt>Effect size</dt>"
                "<dd>The rank-biserial correlation r = 1 − (2T/n(n+1)) gives a standardized "
                "effect size comparable across studies.</dd>"
                "</dl>"
            ),
        },
        ("stats", "friedman"): {
            "title": "Understanding the Friedman Test",
            "content": (
                "<dl>"
                "<dt>What is the Friedman test?</dt>"
                "<dd>The nonparametric alternative to repeated measures ANOVA. It tests whether "
                "treatments differ when the same subjects are measured under multiple conditions.</dd>"
                "<dt>How does it work?</dt>"
                "<dd>It ranks observations within each subject (block), then tests whether the "
                "average ranks differ across treatments. This removes between-subject variability.</dd>"
                "<dt>When to use</dt>"
                "<dd>When you have repeated measures or matched groups with ordinal data or "
                "non-normal distributions. Common in sensory evaluation, before/during/after "
                "studies, and rating comparisons.</dd>"
                "<dt>Post-hoc</dt>"
                "<dd>A significant Friedman test indicates at least one treatment differs. "
                "Use Nemenyi or Conover post-hoc tests for pairwise comparisons.</dd>"
                "</dl>"
            ),
        },
        ("stats", "spearman"): {
            "title": "Understanding Spearman's Rank Correlation",
            "content": (
                "<dl>"
                "<dt>What is Spearman's ρ?</dt>"
                "<dd>A nonparametric measure of monotonic association. Unlike Pearson's r, it "
                "captures any monotonic relationship (not just linear) and is robust to outliers.</dd>"
                "<dt>How is it calculated?</dt>"
                "<dd>It's simply Pearson's r computed on the ranks of the data. This means it "
                "measures how well the relationship can be described by a monotonic function.</dd>"
                "<dt>When to choose Spearman over Pearson?</dt>"
                "<dd>When data is ordinal, the relationship is monotonic but nonlinear, or "
                "outliers are present. Spearman is the default safe choice when you're unsure "
                "about linearity.</dd>"
                "<dt>Interpreting the value</dt>"
                "<dd>Same scale as Pearson: −1 to +1. ρ = 0 means no monotonic relationship. "
                "The sign indicates direction. Effect size benchmarks are the same: "
                "0.1 small, 0.3 medium, 0.5 large.</dd>"
                "</dl>"
            ),
        },
    }
)


# ── Stats: ANOVA extensions ───────────────────────────────────────────────

_extend(
    {
        ("stats", "split_plot_anova"): {
            "title": "Understanding Split-Plot ANOVA",
            "content": (
                "<dl>"
                "<dt>What is a split-plot design?</dt>"
                "<dd>A design where one factor is hard to change (applied to whole plots) and "
                "another is easy to change (applied to subplots within each whole plot). "
                "Common when one factor requires batch processing.</dd>"
                "<dt>Why not just use regular two-way ANOVA?</dt>"
                "<dd>Split-plot designs have two error terms — one for whole-plot factors and "
                "one for subplot factors. Ignoring this structure inflates the F-test for "
                "the whole-plot factor, leading to false positives.</dd>"
                "<dt>Real-world example</dt>"
                "<dd>Temperature is set for an entire oven run (hard to change), while coating "
                "thickness varies within each run (easy to change). Temperature is the whole-plot "
                "factor; coating is the subplot factor.</dd>"
                "<dt>Interpreting the output</dt>"
                "<dd>Two separate F-tests with different error terms. The whole-plot F-test "
                "has fewer effective degrees of freedom and less power.</dd>"
                "</dl>"
            ),
        },
        ("stats", "repeated_measures_anova"): {
            "title": "Understanding Repeated Measures ANOVA",
            "content": (
                "<dl>"
                "<dt>What is repeated measures ANOVA?</dt>"
                "<dd>Tests whether the mean changes across conditions when the same subjects "
                "are measured multiple times. Each subject serves as their own control, "
                "reducing variability.</dd>"
                "<dt>What is sphericity?</dt>"
                "<dd>The assumption that the variances of all pairwise differences between "
                "conditions are equal. Mauchly's test checks this. If violated, use "
                "Greenhouse-Geisser or Huynh-Feldt corrections.</dd>"
                "<dt>Why is this more powerful?</dt>"
                "<dd>By removing between-subject variability, the error term shrinks and "
                "the test becomes more sensitive to treatment effects.</dd>"
                "<dt>Alternatives</dt>"
                "<dd>Mixed-effects models handle unbalanced designs, missing data, and complex "
                "correlation structures better than classical repeated measures ANOVA.</dd>"
                "</dl>"
            ),
        },
        ("stats", "nested_anova"): {
            "title": "Understanding Nested ANOVA",
            "content": (
                "<dl>"
                "<dt>What is nested ANOVA?</dt>"
                "<dd>Used when one factor is nested within another — e.g., operators nested "
                "within shifts, machines nested within plants. The levels of the nested factor "
                "are different at each level of the nesting factor.</dd>"
                "<dt>Crossed vs nested</dt>"
                "<dd>Crossed: every operator works on every machine. Nested: each operator "
                "works on only one machine. The design structure determines the appropriate "
                "error terms and F-tests.</dd>"
                "<dt>What does it tell me?</dt>"
                "<dd>It decomposes variance into between-group (nesting factor), within-group "
                "(nested factor), and residual components. You learn which level of hierarchy "
                "contributes most to variation.</dd>"
                "<dt>Common applications</dt>"
                "<dd>Gage R&R with operators nested in labs, multi-site studies with batches "
                "nested in sites, hierarchical sampling plans.</dd>"
                "</dl>"
            ),
        },
        ("stats", "anom"): {
            "title": "Understanding Analysis of Means (ANOM)",
            "content": (
                "<dl>"
                "<dt>What is ANOM?</dt>"
                "<dd>A graphical hypothesis test that compares each group mean against the "
                "overall mean. Unlike ANOVA (which tests 'any difference?'), ANOM identifies "
                "<em>which</em> groups differ from the average.</dd>"
                "<dt>How to read the chart</dt>"
                "<dd>Group means are plotted against the grand mean with decision limits. "
                "Means outside the limits are significantly different from the overall average.</dd>"
                "<dt>When to use ANOM vs ANOVA</dt>"
                "<dd>ANOM when you want to compare each group to the overall mean (like comparing "
                "each operator to the plant average). ANOVA when you want to compare groups "
                "to each other.</dd>"
                "<dt>Advantages</dt>"
                "<dd>Visual, intuitive, and controls the experiment-wise error rate. "
                "Particularly useful in quality improvement for identifying outlier performers.</dd>"
                "</dl>"
            ),
        },
        ("stats", "main_effects"): {
            "title": "Understanding Main Effects Plots",
            "content": (
                "<dl>"
                "<dt>What is a main effects plot?</dt>"
                "<dd>Shows the average response at each level of a factor, connected by lines. "
                "It visualizes how changing a factor affects the response, averaged over "
                "all other factors.</dd>"
                "<dt>How to interpret</dt>"
                "<dd>Steep lines indicate a strong effect; flat lines indicate no effect. "
                "The direction shows whether increasing the factor increases or decreases "
                "the response.</dd>"
                "<dt>Caution with interactions</dt>"
                "<dd>Main effects plots can be misleading when interactions are present. "
                "If an interaction is significant, the main effect depends on the level "
                "of the other factor — always check interaction plots too.</dd>"
                "<dt>Relationship to ANOVA</dt>"
                "<dd>ANOVA tests whether the main effect is statistically significant. "
                "The main effects plot shows the practical magnitude and direction.</dd>"
                "</dl>"
            ),
        },
        ("stats", "interaction"): {
            "title": "Understanding Interaction Plots",
            "content": (
                "<dl>"
                "<dt>What is an interaction plot?</dt>"
                "<dd>Shows the response at each combination of two factors. Lines for one factor "
                "are plotted across levels of the other. Non-parallel lines indicate interaction.</dd>"
                "<dt>Parallel vs crossing lines</dt>"
                "<dd><strong>Parallel</strong>: No interaction — factors act independently. "
                "<strong>Converging/diverging</strong>: Ordinal interaction — the effect changes "
                "magnitude but not direction. <strong>Crossing</strong>: Disordinal interaction — "
                "the effect reverses direction at different levels.</dd>"
                "<dt>Why do interactions matter?</dt>"
                "<dd>If you optimize factors independently (ignoring interaction), you may miss "
                "the true optimal combination. In DOE, interactions often reveal the most "
                "actionable insights.</dd>"
                "<dt>Statistical significance</dt>"
                "<dd>The interaction p-value from ANOVA tells you if the non-parallelism is "
                "statistically significant. Always test before interpreting visual patterns.</dd>"
                "</dl>"
            ),
        },
        ("stats", "variance_components"): {
            "title": "Understanding Variance Components Analysis",
            "content": (
                "<dl>"
                "<dt>What is variance components analysis?</dt>"
                "<dd>It decomposes total variability into contributions from different sources "
                "(e.g., batch, operator, measurement error). Essential for understanding "
                "<em>where</em> variation comes from.</dd>"
                "<dt>How to interpret</dt>"
                "<dd>Each component shows the estimated variance and its percentage of total. "
                "The largest component is your biggest opportunity for improvement.</dd>"
                "<dt>ANOVA vs REML estimation</dt>"
                "<dd>ANOVA-based estimates can be negative (set to zero); REML (restricted "
                "maximum likelihood) provides non-negative estimates and handles unbalanced "
                "designs better.</dd>"
                "<dt>Connection to Gage R&R</dt>"
                "<dd>Gage R&R is a specific application of variance components — decomposing "
                "measurement variance into repeatability (within-operator) and reproducibility "
                "(between-operator) components.</dd>"
                "</dl>"
            ),
        },
    }
)


# ── Stats: Post-hoc comparisons ──────────────────────────────────────────

_extend(
    {
        ("stats", "tukey_hsd"): {
            "title": "Understanding Tukey's HSD Test",
            "content": (
                "<dl>"
                "<dt>What is Tukey's HSD?</dt>"
                "<dd>An Honestly Significant Difference test for all pairwise comparisons after "
                "ANOVA. It controls the family-wise error rate — the probability of <em>any</em> "
                "false positive across all comparisons.</dd>"
                "<dt>When to use</dt>"
                "<dd>When all pairwise comparisons are of interest and group sizes are equal "
                "(or approximately equal). The most commonly used post-hoc test.</dd>"
                "<dt>How to interpret</dt>"
                "<dd>Pairs with adjusted p &lt; 0.05 differ significantly. The simultaneous "
                "confidence intervals show the plausible range for each pairwise difference.</dd>"
                "<dt>Assumptions</dt>"
                "<dd>Same as ANOVA: normality, equal variances, independence. For unequal "
                "variances, use Games-Howell instead.</dd>"
                "</dl>"
            ),
        },
        ("stats", "dunnett"): {
            "title": "Understanding Dunnett's Test",
            "content": (
                "<dl>"
                "<dt>What is Dunnett's test?</dt>"
                "<dd>Compares each treatment group against a single control group — not all "
                "pairs. This is more powerful than Tukey when you only care about comparisons "
                "to a reference.</dd>"
                "<dt>When to use</dt>"
                "<dd>When you have a clear control/reference group and want to know which "
                "treatments differ from it. Fewer comparisons = more power per comparison.</dd>"
                "<dt>One-sided vs two-sided</dt>"
                "<dd>Two-sided tests 'different from control'. One-sided tests 'better than "
                "control' or 'worse than control' — choose based on your research question.</dd>"
                "<dt>Example</dt>"
                "<dd>Comparing 5 new suppliers against the current supplier. You don't care "
                "how new suppliers compare to each other — only whether they beat the incumbent.</dd>"
                "</dl>"
            ),
        },
        ("stats", "games_howell"): {
            "title": "Understanding the Games-Howell Test",
            "content": (
                "<dl>"
                "<dt>What is Games-Howell?</dt>"
                "<dd>A post-hoc test for pairwise comparisons that does not assume equal "
                "variances or equal sample sizes. The nonparametric-robust counterpart to Tukey's HSD.</dd>"
                "<dt>When to use</dt>"
                "<dd>When Levene's test indicates unequal variances, or when group sizes "
                "differ substantially. It's the safe default when ANOVA assumptions "
                "are questionable.</dd>"
                "<dt>How it works</dt>"
                "<dd>It uses Welch's t-test for each pair with a Tukey-like correction for "
                "multiple comparisons. Degrees of freedom are adjusted per pair.</dd>"
                "<dt>Trade-off</dt>"
                "<dd>Slightly less powerful than Tukey's HSD when variances really are equal, "
                "but much more reliable when they're not.</dd>"
                "</dl>"
            ),
        },
        ("stats", "dunn"): {
            "title": "Understanding Dunn's Test",
            "content": (
                "<dl>"
                "<dt>What is Dunn's test?</dt>"
                "<dd>The post-hoc test for Kruskal-Wallis. It performs pairwise rank-sum "
                "comparisons with correction for multiple testing (Bonferroni or Holm).</dd>"
                "<dt>When to use</dt>"
                "<dd>After a significant Kruskal-Wallis test to identify which specific "
                "pairs differ. The nonparametric equivalent of Tukey's HSD.</dd>"
                "<dt>Bonferroni vs Holm correction</dt>"
                "<dd>Bonferroni multiplies each p-value by the number of comparisons — simple "
                "but conservative. Holm is a step-down method that's uniformly more powerful.</dd>"
                "<dt>Interpreting the results</dt>"
                "<dd>Each pair shows a z-statistic and adjusted p-value. Significant pairs "
                "have different rank distributions — one group tends to have larger values.</dd>"
                "</dl>"
            ),
        },
        ("stats", "scheffe_test"): {
            "title": "Understanding Scheffé's Test",
            "content": (
                "<dl>"
                "<dt>What is Scheffé's test?</dt>"
                "<dd>The most conservative post-hoc test. It controls the error rate for "
                "<em>all possible contrasts</em>, not just pairwise comparisons — including "
                "complex contrasts like 'Group A vs the average of B and C'.</dd>"
                "<dt>When to use</dt>"
                "<dd>When you want to test complex contrasts (not just simple pairs), or when "
                "you want maximum protection against false positives. Rarely needed in practice.</dd>"
                "<dt>Trade-off</dt>"
                "<dd>Very conservative — much wider confidence intervals and higher p-values "
                "than Tukey. Only use when you genuinely need to test arbitrary contrasts.</dd>"
                "<dt>Comparison to Tukey</dt>"
                "<dd>For simple pairwise comparisons, Tukey is always more powerful. Scheffé "
                "only wins when you need the flexibility to test data-driven contrasts.</dd>"
                "</dl>"
            ),
        },
        ("stats", "bonferroni_test"): {
            "title": "Understanding the Bonferroni Correction",
            "content": (
                "<dl>"
                "<dt>What is the Bonferroni correction?</dt>"
                "<dd>The simplest multiple comparison correction: divide α by the number of "
                "tests (or equivalently, multiply p-values by the number of tests). Controls "
                "the family-wise error rate.</dd>"
                "<dt>When to use</dt>"
                "<dd>When you have a small number of planned comparisons (not all pairs). "
                "For all-pairs testing, Tukey's HSD is more powerful.</dd>"
                "<dt>Is it too conservative?</dt>"
                "<dd>Yes, for large numbers of comparisons. With 20 tests, α becomes 0.0025 "
                "per test — very hard to reach significance. Consider Holm's step-down method "
                "or false discovery rate (FDR) control instead.</dd>"
                "<dt>When it's fine</dt>"
                "<dd>With 2–5 pre-planned comparisons, Bonferroni is simple, transparent, and "
                "not overly conservative. It's the right tool for a small number of specific tests.</dd>"
                "</dl>"
            ),
        },
        ("stats", "hsu_mcb"): {
            "title": "Understanding Hsu's MCB (Multiple Comparisons with the Best)",
            "content": (
                "<dl>"
                "<dt>What is Hsu's MCB?</dt>"
                "<dd>It identifies which treatments could plausibly be the best (largest or "
                "smallest mean). Each treatment gets a confidence interval for its difference "
                "from the best.</dd>"
                "<dt>When to use</dt>"
                "<dd>When you want to select the best treatment (supplier, machine, setting) "
                "rather than test all pairwise differences. Much more focused than Tukey.</dd>"
                "<dt>How to interpret</dt>"
                "<dd>Treatments whose interval includes zero cannot be ruled out as the best. "
                "Treatments whose entire interval is negative are significantly worse than "
                "the best.</dd>"
                "<dt>Practical application</dt>"
                "<dd>Supplier selection, process optimization, equipment comparison — any time "
                "the goal is 'pick the winner' rather than 'compare everything to everything'.</dd>"
                "</dl>"
            ),
        },
    }
)


# ── Stats: Power & sample size ────────────────────────────────────────────

_extend(
    {
        ("stats", "power_z"): {
            "title": "Understanding Power Analysis (z-test)",
            "content": (
                "<dl>"
                "<dt>What is statistical power?</dt>"
                "<dd>The probability of detecting a real effect if one exists. Power = 0.8 means "
                "an 80% chance of catching a true difference. Below 0.8, you risk missing real effects.</dd>"
                "<dt>What determines power?</dt>"
                "<dd>Four linked quantities: sample size, effect size, significance level (α), and "
                "power. Fix any three to solve for the fourth.</dd>"
                "<dt>What is the what-if explorer?</dt>"
                "<dd>Use the sliders to explore how changing effect size, sample size, or α "
                "affects power. This helps plan studies with adequate sensitivity.</dd>"
                "<dt>Common mistake</dt>"
                "<dd>Running a study without power analysis, then concluding 'no effect' when "
                "the test was simply underpowered. Always determine required sample size before "
                "collecting data.</dd>"
                "</dl>"
            ),
        },
        ("stats", "power_1prop"): {
            "title": "Understanding Power Analysis (One Proportion)",
            "content": (
                "<dl>"
                "<dt>What does this calculate?</dt>"
                "<dd>Sample size needed to detect a difference between an observed proportion "
                "and a target. For example: 'How many parts do I need to test to detect if "
                "defect rate exceeds 2%?'</dd>"
                "<dt>Key inputs</dt>"
                "<dd>Baseline proportion, detectable difference, desired power (typically 0.8), "
                "and significance level (typically 0.05).</dd>"
                "<dt>Why do proportions near 0 or 1 need more samples?</dt>"
                "<dd>The variance of a proportion peaks at p = 0.5. Near 0 or 1, the distribution "
                "is skewed and the normal approximation needs more data to be reliable.</dd>"
                "<dt>Practical tip</dt>"
                "<dd>If you need to detect small differences in low-defect-rate processes "
                "(e.g., 0.1% vs 0.2%), sample sizes can be enormous. Consider whether "
                "the practical difference justifies the testing cost.</dd>"
                "</dl>"
            ),
        },
        ("stats", "power_2prop"): {
            "title": "Understanding Power Analysis (Two Proportions)",
            "content": (
                "<dl>"
                "<dt>What does this calculate?</dt>"
                "<dd>Sample size per group to detect a difference between two proportions. "
                "For example: 'How many units per line to detect if Line A has a different "
                "pass rate than Line B?'</dd>"
                "<dt>Equal vs unequal allocation</dt>"
                "<dd>Equal sample sizes per group give maximum power. Unequal allocation (e.g., "
                "2:1 ratio) is common when one group is cheaper to sample.</dd>"
                "<dt>Effect size</dt>"
                "<dd>Cohen's h measures the difference on the arcsine-transformed scale. "
                "This accounts for the fact that a 5% difference matters more at 50% than at 95%.</dd>"
                "<dt>Use the sliders</dt>"
                "<dd>Adjust baseline proportion, difference to detect, and allocation ratio "
                "to find the smallest study that meets your power requirements.</dd>"
                "</dl>"
            ),
        },
        ("stats", "power_1variance"): {
            "title": "Understanding Power Analysis (One Variance)",
            "content": (
                "<dl>"
                "<dt>What does this calculate?</dt>"
                "<dd>Sample size to detect whether a population variance differs from a target. "
                "Used when process consistency is the quality characteristic of interest.</dd>"
                "<dt>Why is this harder than testing means?</dt>"
                "<dd>Variance tests require much larger samples than mean tests for the same "
                "power. Variance estimates are inherently noisier than mean estimates.</dd>"
                "<dt>Practical applications</dt>"
                "<dd>Validating that a new process has lower variability than the old one, "
                "verifying measurement system precision, or certifying that spread meets "
                "specification limits.</dd>"
                "<dt>Ratio vs absolute difference</dt>"
                "<dd>The effect size for variance is typically expressed as the ratio of true "
                "variance to hypothesized variance (or the ratio of standard deviations).</dd>"
                "</dl>"
            ),
        },
        ("stats", "power_2variance"): {
            "title": "Understanding Power Analysis (Two Variances)",
            "content": (
                "<dl>"
                "<dt>What does this calculate?</dt>"
                "<dd>Sample size to detect a difference in variances between two groups. "
                "For example: 'Is the old process more variable than the new process?'</dd>"
                "<dt>Key input: variance ratio</dt>"
                "<dd>The ratio σ₁²/σ₂² you want to detect. A ratio of 2 means one group "
                "has twice the variance. Larger ratios are easier to detect.</dd>"
                "<dt>Sensitivity to non-normality</dt>"
                "<dd>Power calculations for variance tests assume normality. Non-normal data "
                "can dramatically affect actual power. Consider bootstrap-based approaches "
                "for non-normal data.</dd>"
                "<dt>Use with F-test or Levene's</dt>"
                "<dd>This power analysis applies to the F-test. If you plan to use Levene's "
                "test (robust to non-normality), the actual power may differ slightly.</dd>"
                "</dl>"
            ),
        },
        ("stats", "power_equivalence"): {
            "title": "Understanding Power Analysis (Equivalence)",
            "content": (
                "<dl>"
                "<dt>What does this calculate?</dt>"
                "<dd>Sample size to demonstrate equivalence within a specified margin using TOST. "
                "Generally requires larger samples than superiority testing because you need to "
                "prove the difference is small, not just detect it.</dd>"
                "<dt>Why are equivalence studies larger?</dt>"
                "<dd>You must show the confidence interval falls entirely within ±Δ. This requires "
                "precise estimation (narrow CI), which demands more data.</dd>"
                "<dt>The equivalence margin</dt>"
                "<dd>The maximum difference considered practically unimportant. This is a "
                "domain-specific judgment — tighter margins need more data.</dd>"
                "<dt>Regulatory context</dt>"
                "<dd>Bioequivalence studies (FDA), method transfer validation, and process "
                "qualification all require equivalence testing with formal power calculations.</dd>"
                "</dl>"
            ),
        },
        ("stats", "power_doe"): {
            "title": "Understanding Power Analysis (DOE)",
            "content": (
                "<dl>"
                "<dt>What does this calculate?</dt>"
                "<dd>The power of a designed experiment to detect effects of a specified size. "
                "Helps determine whether your experiment has enough replicates.</dd>"
                "<dt>Key inputs</dt>"
                "<dd>Number of factors, levels per factor, number of replicates, significance "
                "level, and the minimum effect size you need to detect (in standard deviation units).</dd>"
                "<dt>Replicates vs repetitions</dt>"
                "<dd>Replicates are independent runs of the full experiment. Repetitions are "
                "multiple measurements within a single run. Only replicates increase power "
                "for detecting factor effects.</dd>"
                "<dt>What-if exploration</dt>"
                "<dd>Use the sliders to find the minimum replicates needed to achieve 80% power "
                "for your smallest important effect. This prevents wasted experimental resources.</dd>"
                "</dl>"
            ),
        },
        ("stats", "sample_size_ci"): {
            "title": "Understanding Sample Size for Confidence Intervals",
            "content": (
                "<dl>"
                "<dt>What does this calculate?</dt>"
                "<dd>The sample size needed to achieve a confidence interval of a specified width. "
                "For example: 'How many measurements to estimate the mean within ±0.5 units?'</dd>"
                "<dt>Width vs confidence level</dt>"
                "<dd>Narrower intervals and higher confidence levels both require more data. "
                "A 95% CI half as wide as a 90% CI requires roughly 4× the sample size.</dd>"
                "<dt>When to use this vs power analysis</dt>"
                "<dd>Use CI-based sizing when estimation precision matters more than hypothesis "
                "testing. Common for process characterization, survey design, and measurement "
                "validation.</dd>"
                "<dt>Practical consideration</dt>"
                "<dd>You need a preliminary estimate of the standard deviation. Use pilot data, "
                "historical data, or a worst-case estimate. The sample size is proportional to σ².</dd>"
                "</dl>"
            ),
        },
        ("stats", "sample_size_tolerance"): {
            "title": "Understanding Sample Size for Tolerance Intervals",
            "content": (
                "<dl>"
                "<dt>What is a tolerance interval?</dt>"
                "<dd>An interval that covers a specified proportion of the population with a "
                "given confidence level. For example: '95% of parts fall within this range, "
                "with 99% confidence.'</dd>"
                "<dt>Tolerance vs confidence interval</dt>"
                "<dd>A confidence interval estimates a parameter (like the mean). A tolerance "
                "interval covers a proportion of individual values. They answer different questions.</dd>"
                "<dt>What does this calculate?</dt>"
                "<dd>The sample size needed for a tolerance interval of a specified coverage "
                "and confidence level. Higher coverage and confidence both increase the "
                "required sample size.</dd>"
                "<dt>Application</dt>"
                "<dd>Process qualification, specification setting, supplier qualification — "
                "any time you need to characterize the range of individual values, not just "
                "the average.</dd>"
                "</dl>"
            ),
        },
    }
)


# ── Stats: Time series ────────────────────────────────────────────────────

_extend(
    {
        ("stats", "arima"): {
            "title": "Understanding ARIMA Modeling",
            "content": (
                "<dl>"
                "<dt>What is ARIMA?</dt>"
                "<dd>AutoRegressive Integrated Moving Average — a model for time series data that "
                "combines autoregression (past values), differencing (for stationarity), and "
                "moving average (past errors). Specified as ARIMA(p, d, q).</dd>"
                "<dt>What do p, d, q mean?</dt>"
                "<dd><strong>p</strong>: number of autoregressive terms (how many past values). "
                "<strong>d</strong>: differencing order (how many times to difference for stationarity). "
                "<strong>q</strong>: number of moving average terms (how many past errors).</dd>"
                "<dt>How are orders selected?</dt>"
                "<dd>ACF and PACF plots guide manual selection. AIC/BIC-based auto-selection "
                "searches over candidate orders. Lower information criteria = better model.</dd>"
                "<dt>Forecasting</dt>"
                "<dd>ARIMA forecasts extrapolate the learned patterns. Prediction intervals widen "
                "with forecast horizon — uncertainty grows the further ahead you predict.</dd>"
                "</dl>"
            ),
        },
        ("stats", "sarima"): {
            "title": "Understanding SARIMA (Seasonal ARIMA)",
            "content": (
                "<dl>"
                "<dt>What is SARIMA?</dt>"
                "<dd>ARIMA extended with seasonal terms: SARIMA(p,d,q)(P,D,Q)s. The seasonal "
                "component captures repeating patterns at period s (e.g., s=12 for monthly "
                "data with yearly seasonality).</dd>"
                "<dt>When to use SARIMA vs ARIMA?</dt>"
                "<dd>When the ACF shows significant spikes at seasonal lags (12, 24, 36... for "
                "monthly data). If no seasonal pattern exists, plain ARIMA is sufficient.</dd>"
                "<dt>Seasonal differencing</dt>"
                "<dd>D=1 differences at the seasonal lag (subtract last year's value). This "
                "removes the seasonal component so the remaining pattern can be modeled.</dd>"
                "<dt>Model diagnostics</dt>"
                "<dd>Check residuals for whiteness (no autocorrelation), normality, and constant "
                "variance. The Ljung-Box test formally tests for residual autocorrelation.</dd>"
                "</dl>"
            ),
        },
        ("stats", "decomposition"): {
            "title": "Understanding Time Series Decomposition",
            "content": (
                "<dl>"
                "<dt>What is decomposition?</dt>"
                "<dd>It separates a time series into trend, seasonal, and residual components. "
                "This reveals the underlying patterns hidden in the raw data.</dd>"
                "<dt>Additive vs multiplicative</dt>"
                "<dd><strong>Additive</strong>: Y = Trend + Seasonal + Residual. Use when seasonal "
                "variation is roughly constant. <strong>Multiplicative</strong>: Y = Trend × Seasonal × Residual. "
                "Use when seasonal variation grows with the level.</dd>"
                "<dt>STL decomposition</dt>"
                "<dd>Seasonal and Trend decomposition using LOESS — more robust than classical "
                "decomposition, handles outliers better, and allows the seasonal component to "
                "change over time.</dd>"
                "<dt>Practical use</dt>"
                "<dd>Deseasonalize data before applying SPC charts. Identify whether a trend is "
                "real or seasonal. Estimate the seasonal adjustment factor for planning.</dd>"
                "</dl>"
            ),
        },
        ("stats", "acf_pacf"): {
            "title": "Understanding ACF and PACF",
            "content": (
                "<dl>"
                "<dt>What is the ACF?</dt>"
                "<dd>The AutoCorrelation Function shows the correlation between a time series "
                "and its lagged versions. It reveals the memory and periodicity in the data.</dd>"
                "<dt>What is the PACF?</dt>"
                "<dd>The Partial AutoCorrelation Function shows the correlation at each lag "
                "after removing the effect of shorter lags. It isolates the direct relationship "
                "at each lag.</dd>"
                "<dt>How to use for ARIMA modeling</dt>"
                "<dd>ACF tailing off + PACF cutting off at lag p → AR(p). "
                "ACF cutting off at lag q + PACF tailing off → MA(q). "
                "Both tailing off → ARMA model needed.</dd>"
                "<dt>Significance bands</dt>"
                "<dd>The blue dashed lines show the 95% confidence bounds. Spikes outside "
                "these bands are statistically significant correlations.</dd>"
                "</dl>"
            ),
        },
        ("stats", "granger"): {
            "title": "Understanding Granger Causality",
            "content": (
                "<dl>"
                "<dt>What is Granger causality?</dt>"
                "<dd>It tests whether past values of one time series help predict another, "
                "beyond what the other series' own past provides. 'X Granger-causes Y' means "
                "X's history improves Y's forecast.</dd>"
                "<dt>Is it real causation?</dt>"
                "<dd>No — it's predictive causality, not structural causality. X may Granger-cause Y "
                "because of a shared driver. Use causal discovery methods for structural claims.</dd>"
                "<dt>Lag selection</dt>"
                "<dd>Test multiple lag orders. Too few lags may miss the relationship; too many "
                "reduce power. AIC or BIC can guide lag selection.</dd>"
                "<dt>Stationarity requirement</dt>"
                "<dd>Both series must be stationary. If not, difference them first. Non-stationary "
                "data can produce spurious Granger causality.</dd>"
                "</dl>"
            ),
        },
        ("stats", "changepoint"): {
            "title": "Understanding Changepoint Detection",
            "content": (
                "<dl>"
                "<dt>What is changepoint detection?</dt>"
                "<dd>It identifies points in time where the statistical properties (mean, variance, "
                "or distribution) of a time series change. Think of it as automated 'where did "
                "something shift?'</dd>"
                "<dt>Types of changes detected</dt>"
                "<dd>Mean shifts (level changes), variance shifts (stability changes), and "
                "distribution changes. Different algorithms specialize in different types.</dd>"
                "<dt>Methods</dt>"
                "<dd>PELT (fast, penalized), CUSUM (sequential, online-capable), Bayesian "
                "changepoint (posterior probability at each point). Each has different strengths.</dd>"
                "<dt>Practical use</dt>"
                "<dd>Process monitoring, identifying when a process changed state, root cause "
                "analysis (correlating changepoints with known events), and segmenting data "
                "for separate analysis.</dd>"
                "</dl>"
            ),
        },
        ("stats", "ccf"): {
            "title": "Understanding Cross-Correlation (CCF)",
            "content": (
                "<dl>"
                "<dt>What is the CCF?</dt>"
                "<dd>The Cross-Correlation Function measures the correlation between two time "
                "series at various lags. It reveals whether one series leads or follows another.</dd>"
                "<dt>How to read it</dt>"
                "<dd>Positive lags: X leads Y. Negative lags: Y leads X. The lag with the "
                "largest absolute correlation indicates the delay between the series.</dd>"
                "<dt>Pre-whitening</dt>"
                "<dd>For reliable CCF, both series should be pre-whitened (ARIMA-filtered to "
                "remove autocorrelation). Raw CCF can show spurious correlations due to "
                "shared trends.</dd>"
                "<dt>Applications</dt>"
                "<dd>Finding the time delay between an input variable and a quality outcome, "
                "identifying leading indicators, and lag analysis for transfer functions.</dd>"
                "</dl>"
            ),
        },
    }
)


# ── Stats: Survival ───────────────────────────────────────────────────────

_extend(
    {
        ("stats", "weibull"): {
            "title": "Understanding Weibull Analysis",
            "content": (
                "<dl>"
                "<dt>What is Weibull analysis?</dt>"
                "<dd>A reliability/lifetime modeling method that fits time-to-failure data to a "
                "Weibull distribution. It characterizes failure patterns and predicts future failures.</dd>"
                "<dt>What do the parameters mean?</dt>"
                "<dd><strong>Shape (β)</strong>: β &lt; 1 = decreasing failure rate (infant mortality), "
                "β = 1 = constant (random failures), β &gt; 1 = increasing (wear-out). "
                "<strong>Scale (η)</strong>: the characteristic life — 63.2% of units fail by this time.</dd>"
                "<dt>Censored data</dt>"
                "<dd>Units that haven't failed yet (right-censored) are included in the analysis. "
                "Ignoring censoring biases results toward shorter lifetimes.</dd>"
                "<dt>B-life values</dt>"
                "<dd>B10 is the time by which 10% of units fail. B1 is 1% failure time. These are "
                "standard reliability metrics for warranty and design decisions.</dd>"
                "</dl>"
            ),
        },
        ("stats", "kaplan_meier"): {
            "title": "Understanding Kaplan-Meier Survival Analysis",
            "content": (
                "<dl>"
                "<dt>What is the Kaplan-Meier estimator?</dt>"
                "<dd>A nonparametric method that estimates the survival function from time-to-event "
                "data, including censored observations. No distributional assumption required.</dd>"
                "<dt>How to read the survival curve</dt>"
                "<dd>The y-axis shows the probability of surviving past each time point. Steps down "
                "at each failure; flat sections indicate censored observations. Steeper drops = "
                "more failures concentrated in that period.</dd>"
                "<dt>Log-rank test</dt>"
                "<dd>Compares survival curves between groups. A significant result means the groups "
                "have different survival patterns. Does not tell you <em>how</em> they differ.</dd>"
                "<dt>Median survival time</dt>"
                "<dd>The time at which the survival probability crosses 50%. If fewer than half "
                "the subjects fail during follow-up, the median is undefined.</dd>"
                "</dl>"
            ),
        },
        ("stats", "cox_ph"): {
            "title": "Understanding Cox Proportional Hazards Regression",
            "content": (
                "<dl>"
                "<dt>What is the Cox model?</dt>"
                "<dd>A regression model for survival data that estimates how predictors affect "
                "the hazard (failure rate) without specifying the baseline hazard function.</dd>"
                "<dt>What is a hazard ratio?</dt>"
                "<dd>How much the failure rate changes per unit increase in a predictor. HR = 2 "
                "means double the failure rate; HR = 0.5 means half. HR = 1 means no effect.</dd>"
                "<dt>Proportional hazards assumption</dt>"
                "<dd>The effect of each predictor must be constant over time. Schoenfeld residuals "
                "test this — if violated, consider time-varying coefficients or stratification.</dd>"
                "<dt>Censoring</dt>"
                "<dd>Like Kaplan-Meier, the Cox model handles right-censored data naturally. "
                "The partial likelihood uses only the ordering of events, not absolute times.</dd>"
                "</dl>"
            ),
        },
    }
)


# ── Stats: MSA ─────────────────────────────────────────────────────────────

_extend(
    {
        ("stats", "gage_rr"): {
            "title": "Understanding Gage R&R (Crossed)",
            "content": (
                "<dl>"
                "<dt>What is Gage R&R?</dt>"
                "<dd>Gage Repeatability and Reproducibility study — decomposes measurement "
                "variation into equipment variation (repeatability) and operator variation "
                "(reproducibility) to assess measurement system adequacy.</dd>"
                "<dt>Key metrics</dt>"
                "<dd><strong>%StudyVar</strong>: Measurement system variation as a percentage of "
                "total study variation. &lt; 10% = acceptable, 10–30% = marginal, &gt; 30% = unacceptable. "
                "<strong>Number of Distinct Categories</strong>: ≥ 5 means the system can adequately "
                "discriminate between parts.</dd>"
                "<dt>ANOVA vs X̄&R method</dt>"
                "<dd>ANOVA-based Gage R&R separates the operator×part interaction. The X̄&R "
                "method is simpler but combines interaction with repeatability.</dd>"
                "<dt>What to do if %GRR is too high</dt>"
                "<dd>Investigate: Is it repeatability (equipment issue — maintenance, calibration)? "
                "Or reproducibility (operator issue — training, procedure clarity)? The breakdown "
                "guides your corrective action.</dd>"
                "</dl>"
            ),
        },
        ("stats", "gage_rr_nested"): {
            "title": "Understanding Nested Gage R&R",
            "content": (
                "<dl>"
                "<dt>When is nested Gage R&R needed?</dt>"
                "<dd>When each operator measures different parts (destructive testing, very large "
                "batches). In crossed designs, every operator measures every part. Nested designs "
                "handle the case where that's impossible.</dd>"
                "<dt>What changes?</dt>"
                "<dd>The ANOVA model is different — parts are nested within operators. The "
                "operator×part interaction cannot be estimated separately from repeatability.</dd>"
                "<dt>Interpreting results</dt>"
                "<dd>The same metrics apply (%StudyVar, NDC), but the decomposition is less "
                "detailed. If possible, prefer crossed designs for more informative results.</dd>"
                "<dt>Design requirements</dt>"
                "<dd>Each operator needs the same number of parts and repeated measurements. "
                "Minimum: 2 operators × 5 parts × 2 replicates.</dd>"
                "</dl>"
            ),
        },
        ("stats", "gage_rr_expanded"): {
            "title": "Understanding Expanded Gage R&R",
            "content": (
                "<dl>"
                "<dt>What is expanded Gage R&R?</dt>"
                "<dd>Adds additional factors beyond operator and part — such as fixture, time of day, "
                "or lab location. Decomposes measurement variation across all sources simultaneously.</dd>"
                "<dt>When to use</dt>"
                "<dd>When you suspect measurement variation comes from sources beyond operator and "
                "equipment. Common in multi-lab, multi-fixture, or multi-shift environments.</dd>"
                "<dt>Design considerations</dt>"
                "<dd>Each additional factor multiplies the study size. Plan carefully — too many "
                "factors make the study impractical. Focus on the factors most likely to contribute.</dd>"
                "<dt>Analysis</dt>"
                "<dd>Uses a general linear model with all factors and their interactions. "
                "Variance components are estimated for each source.</dd>"
                "</dl>"
            ),
        },
        ("stats", "gage_linearity_bias"): {
            "title": "Understanding Gage Linearity and Bias",
            "content": (
                "<dl>"
                "<dt>What is measurement bias?</dt>"
                "<dd>The systematic difference between the measured value and the true (reference) "
                "value. A biased gage consistently reads high or low.</dd>"
                "<dt>What is linearity?</dt>"
                "<dd>Whether the bias is constant across the measurement range. Poor linearity means "
                "the gage is accurate at some values but biased at others — it stretches or "
                "compresses the scale.</dd>"
                "<dt>How is it assessed?</dt>"
                "<dd>Measure reference standards spanning the operating range, multiple times each. "
                "Plot bias vs reference value. The slope of the regression line measures linearity.</dd>"
                "<dt>Acceptance criteria</dt>"
                "<dd>Bias should be statistically insignificant at each reference level. The "
                "linearity regression slope should not be significantly different from zero.</dd>"
                "</dl>"
            ),
        },
        ("stats", "gage_type1"): {
            "title": "Understanding Type 1 Gage Study",
            "content": (
                "<dl>"
                "<dt>What is a Type 1 gage study?</dt>"
                "<dd>Evaluates a single measurement system on a single reference part with many "
                "repeated measurements. Assesses bias and repeatability (Cg, Cgk) without "
                "involving multiple operators or parts.</dd>"
                "<dt>What are Cg and Cgk?</dt>"
                "<dd><strong>Cg</strong>: Capability of the gage — ratio of tolerance to 6σ of "
                "measurement variation. <strong>Cgk</strong>: Like Cpk, adjusts for bias. "
                "Both should be ≥ 1.33 for adequate measurement capability.</dd>"
                "<dt>When to use</dt>"
                "<dd>As a quick screening study before a full Gage R&R. Identifies whether the "
                "gage itself is adequate before involving operators and multiple parts.</dd>"
                "<dt>Minimum requirements</dt>"
                "<dd>One reference part measured at least 25 times (50+ preferred). The reference "
                "value must be known from a higher-accuracy method.</dd>"
                "</dl>"
            ),
        },
        ("stats", "attribute_gage"): {
            "title": "Understanding Attribute Gage Study",
            "content": (
                "<dl>"
                "<dt>What is an attribute gage study?</dt>"
                "<dd>Assesses a measurement system that produces categorical results (pass/fail, "
                "go/no-go) rather than continuous measurements. Evaluates whether inspectors "
                "agree with each other and with the known standard.</dd>"
                "<dt>Key metrics</dt>"
                "<dd><strong>Within-appraiser agreement</strong>: Does the same inspector give the "
                "same result on repeat measurements? <strong>Between-appraiser agreement</strong>: "
                "Do different inspectors agree? <strong>vs Standard</strong>: Do inspectors agree "
                "with the known correct answer?</dd>"
                "<dt>Minimum requirements</dt>"
                "<dd>At least 2 appraisers × 30 parts × 3 trials. Parts should span the "
                "pass/fail boundary — include known good, known bad, and borderline parts.</dd>"
                "<dt>If agreement is poor</dt>"
                "<dd>Clarify the acceptance criteria, improve lighting/fixtures, provide reference "
                "samples, and retrain inspectors. Poor attribute inspection often stems from "
                "ambiguous standards.</dd>"
                "</dl>"
            ),
        },
        ("stats", "attribute_agreement"): {
            "title": "Understanding Attribute Agreement Analysis",
            "content": (
                "<dl>"
                "<dt>What is attribute agreement analysis?</dt>"
                "<dd>Extends the attribute gage study with Cohen's Kappa and Fleiss' Kappa "
                "statistics for multi-category ratings. Measures agreement beyond chance.</dd>"
                "<dt>What is Kappa?</dt>"
                "<dd>Agreement corrected for chance. κ = 1 is perfect agreement, κ = 0 is "
                "chance-level agreement, κ &lt; 0 is worse than chance. "
                "0.61–0.80 = substantial, 0.81–1.0 = almost perfect agreement.</dd>"
                "<dt>Weighted vs unweighted Kappa</dt>"
                "<dd>Weighted Kappa gives partial credit for 'near misses' in ordinal categories "
                "(e.g., rating 3 vs 4 is a smaller disagreement than 1 vs 5).</dd>"
                "<dt>Common issue</dt>"
                "<dd>Kappa can be low even with high percent agreement when the marginal "
                "distributions are unbalanced (the 'Kappa paradox'). Report both metrics.</dd>"
                "</dl>"
            ),
        },
        ("stats", "icc"): {
            "title": "Understanding Intraclass Correlation (ICC)",
            "content": (
                "<dl>"
                "<dt>What is ICC?</dt>"
                "<dd>A measure of reliability (consistency) when multiple raters assess the same "
                "targets. Unlike Pearson's r, it accounts for both correlation and systematic "
                "differences between raters.</dd>"
                "<dt>Which ICC form?</dt>"
                "<dd><strong>ICC(1,1)</strong>: One-way random — raters are a random sample, each "
                "rates different targets. <strong>ICC(2,1)</strong>: Two-way random — each rater "
                "rates each target. <strong>ICC(3,1)</strong>: Two-way mixed — raters are fixed.</dd>"
                "<dt>Interpretation</dt>"
                "<dd>&lt; 0.50 = poor, 0.50–0.75 = moderate, 0.75–0.90 = good, &gt; 0.90 = excellent "
                "reliability.</dd>"
                "<dt>Single vs average measures</dt>"
                "<dd>Single measures ICC is for individual raters. Average measures ICC is for "
                "the mean of k raters — always higher. Report both and specify which you use.</dd>"
                "</dl>"
            ),
        },
        ("stats", "krippendorff_alpha"): {
            "title": "Understanding Krippendorff's Alpha",
            "content": (
                "<dl>"
                "<dt>What is Krippendorff's α?</dt>"
                "<dd>A versatile reliability measure that handles any number of raters, any "
                "measurement level (nominal, ordinal, interval, ratio), missing data, and "
                "unequal sample sizes. The most general agreement statistic.</dd>"
                "<dt>Interpretation</dt>"
                "<dd>α = 1 is perfect agreement, α = 0 is chance. Krippendorff recommends "
                "α ≥ 0.80 for reliable coding, α ≥ 0.667 for tentative conclusions.</dd>"
                "<dt>When to choose over Kappa</dt>"
                "<dd>When you have more than 2 raters, missing ratings, or ordinal/interval data. "
                "Kappa handles only nominal data with complete rating matrices cleanly.</dd>"
                "<dt>Bootstrap confidence interval</dt>"
                "<dd>The point estimate can be unstable with small samples. The bootstrap CI "
                "gives the range of plausible reliability values.</dd>"
                "</dl>"
            ),
        },
        ("stats", "bland_altman"): {
            "title": "Understanding Bland-Altman Analysis",
            "content": (
                "<dl>"
                "<dt>What is a Bland-Altman plot?</dt>"
                "<dd>A method comparison tool that plots the difference between two measurement "
                "methods against their mean. It reveals systematic bias and the limits of agreement.</dd>"
                "<dt>How to interpret</dt>"
                "<dd>The mean difference line shows systematic bias. The limits of agreement "
                "(mean ± 1.96 SD) show the range within which 95% of differences fall. "
                "Points outside suggest outliers or proportional bias.</dd>"
                "<dt>When to use</dt>"
                "<dd>Comparing a new measurement method against a reference method. Correlation "
                "alone is insufficient — two methods can be highly correlated but have poor "
                "agreement (systematic bias).</dd>"
                "<dt>Proportional bias</dt>"
                "<dd>If the difference increases with the magnitude of the measurement, the "
                "bias is proportional. A regression line through the Bland-Altman plot "
                "detects this pattern.</dd>"
                "</dl>"
            ),
        },
    }
)


# ── Stats: Multivariate ──────────────────────────────────────────────────

_extend(
    {
        ("stats", "hotelling_t2"): {
            "title": "Understanding Hotelling's T² Test",
            "content": (
                "<dl>"
                "<dt>What is Hotelling's T²?</dt>"
                "<dd>The multivariate generalization of the t-test. It tests whether the mean "
                "vector of multiple variables simultaneously differs from a hypothesized value "
                "or between two groups.</dd>"
                "<dt>Why multivariate?</dt>"
                "<dd>Running separate t-tests on each variable inflates Type I error and ignores "
                "correlations between variables. T² handles both issues simultaneously.</dd>"
                "<dt>Interpreting the result</dt>"
                "<dd>A significant T² means the multivariate mean differs. Follow up with "
                "individual t-tests (Bonferroni-corrected) to identify which variables contribute.</dd>"
                "<dt>Applications</dt>"
                "<dd>Multivariate process monitoring, batch release testing with multiple quality "
                "characteristics, comparing profiles across conditions.</dd>"
                "</dl>"
            ),
        },
        ("stats", "manova"): {
            "title": "Understanding MANOVA",
            "content": (
                "<dl>"
                "<dt>What is MANOVA?</dt>"
                "<dd>Multivariate Analysis of Variance — tests whether groups differ on a "
                "combination of dependent variables. The multivariate extension of ANOVA.</dd>"
                "<dt>Test statistics</dt>"
                "<dd><strong>Wilks' Lambda</strong>: Most common, based on ratio of determinants. "
                "<strong>Pillai's Trace</strong>: Most robust to violations. "
                "<strong>Hotelling-Lawley</strong>: Most powerful when assumptions hold.</dd>"
                "<dt>When to use</dt>"
                "<dd>When you have multiple correlated response variables. Running separate ANOVAs "
                "inflates error rates and misses multivariate patterns.</dd>"
                "<dt>Follow-up analysis</dt>"
                "<dd>Significant MANOVA → examine discriminant functions to understand which "
                "combination of variables best separates the groups.</dd>"
                "</dl>"
            ),
        },
        ("stats", "multi_vari"): {
            "title": "Understanding Multi-Vari Charts",
            "content": (
                "<dl>"
                "<dt>What is a multi-vari chart?</dt>"
                "<dd>A graphical tool that displays variation from multiple sources simultaneously — "
                "within-piece, piece-to-piece, and time-to-time (or any hierarchical nesting).</dd>"
                "<dt>How to read it</dt>"
                "<dd>Each column shows measurements on one piece across positions. The spread "
                "within columns shows within-piece variation. The spread between columns shows "
                "piece-to-piece variation. Panel changes show temporal variation.</dd>"
                "<dt>Why use it?</dt>"
                "<dd>It quickly reveals the dominant source of variation without formal statistical "
                "tests. If within-piece variation dominates, fix the process. If piece-to-piece "
                "dominates, investigate setup or material.</dd>"
                "<dt>Origin</dt>"
                "<dd>Developed by Leonard Seder in the 1950s. A staple of quality engineering "
                "for decomposing variation sources visually.</dd>"
                "</dl>"
            ),
        },
        ("stats", "copula"): {
            "title": "Understanding Copula Analysis",
            "content": (
                "<dl>"
                "<dt>What is a copula?</dt>"
                "<dd>A mathematical function that describes the dependence structure between "
                "variables separately from their individual distributions. It captures how "
                "variables move together, regardless of their marginal shapes.</dd>"
                "<dt>Why use copulas?</dt>"
                "<dd>Pearson correlation only captures linear dependence. Copulas capture "
                "nonlinear, asymmetric, and tail dependencies — critical for risk modeling.</dd>"
                "<dt>Common copula families</dt>"
                "<dd><strong>Gaussian</strong>: symmetric, no tail dependence. "
                "<strong>Clayton</strong>: lower tail dependence (joint failures). "
                "<strong>Gumbel</strong>: upper tail dependence (joint extremes). "
                "<strong>Frank</strong>: symmetric, moderate tail dependence.</dd>"
                "<dt>Applications</dt>"
                "<dd>Joint failure probability estimation, multivariate process capability, "
                "risk assessment when variables have complex dependency structures.</dd>"
                "</dl>"
            ),
        },
    }
)


# ── Stats: Exploratory & diagnostics ──────────────────────────────────────

_extend(
    {
        ("stats", "descriptive"): {
            "title": "Understanding Descriptive Statistics",
            "content": (
                "<dl>"
                "<dt>What are descriptive statistics?</dt>"
                "<dd>Summary measures that characterize a dataset: location (mean, median), "
                "spread (std dev, IQR), shape (skewness, kurtosis), and counts.</dd>"
                "<dt>Mean vs median</dt>"
                "<dd>Mean is sensitive to outliers; median is robust. When they differ substantially, "
                "the distribution is skewed. For skewed data, the median better represents "
                "the 'typical' value.</dd>"
                "<dt>Standard deviation vs IQR</dt>"
                "<dd>SD assumes approximate normality. IQR (Q3 − Q1) is robust to outliers. "
                "For symmetric data they're equivalent (IQR ≈ 1.35 SD). For skewed data, "
                "prefer IQR.</dd>"
                "<dt>Skewness and kurtosis</dt>"
                "<dd>Skewness measures asymmetry (0 = symmetric). Kurtosis measures tail weight "
                "(3 = normal; &gt; 3 = heavy tails). These guide the choice of downstream analysis.</dd>"
                "</dl>"
            ),
        },
        ("stats", "data_profile"): {
            "title": "Understanding Data Profiling",
            "content": (
                "<dl>"
                "<dt>What is data profiling?</dt>"
                "<dd>An automated overview of each column: data types, missing values, unique counts, "
                "distributions, correlations, and potential quality issues.</dd>"
                "<dt>What does it check?</dt>"
                "<dd>Type detection (numeric/categorical/datetime), missing data patterns, "
                "constant or near-constant columns, outlier prevalence, and basic distribution "
                "shape for each variable.</dd>"
                "<dt>When to use</dt>"
                "<dd>Before any analysis — as the first step in understanding a new dataset. "
                "It catches data quality issues (wrong types, missing values, duplicates) "
                "before they corrupt results.</dd>"
                "<dt>Action items</dt>"
                "<dd>High missing rate → investigate collection process. "
                "Constant column → remove from analysis. "
                "High skewness → consider transformation. "
                "High correlation → check for multicollinearity.</dd>"
                "</dl>"
            ),
        },
        ("stats", "auto_profile"): {
            "title": "Understanding Auto-Profile",
            "content": (
                "<dl>"
                "<dt>What is auto-profiling?</dt>"
                "<dd>An intelligent data characterization that automatically selects the most "
                "relevant analyses based on data types, distributions, and relationships found.</dd>"
                "<dt>What does it produce?</dt>"
                "<dd>Distribution tests, normality assessments, correlation matrices, outlier "
                "detection, and suggested follow-up analyses — all tailored to your actual data.</dd>"
                "<dt>When to use</dt>"
                "<dd>When you have a new dataset and want to quickly understand its structure "
                "without manually selecting analyses one by one.</dd>"
                "<dt>Relationship to data profile</dt>"
                "<dd>Data profile is descriptive (what's in the data). Auto-profile is prescriptive "
                "(what analyses to run next).</dd>"
                "</dl>"
            ),
        },
        ("stats", "graphical_summary"): {
            "title": "Understanding the Graphical Summary",
            "content": (
                "<dl>"
                "<dt>What is the graphical summary?</dt>"
                "<dd>A one-page overview combining histogram, boxplot, probability plot, and "
                "summary statistics. Equivalent to Minitab's Graphical Summary — a standard "
                "quality engineering tool.</dd>"
                "<dt>What to look for</dt>"
                "<dd>Distribution shape (normal? skewed? bimodal?), outliers (boxplot whiskers), "
                "goodness-of-fit (probability plot linearity), and key statistics (mean, StDev, "
                "confidence intervals).</dd>"
                "<dt>Anderson-Darling test</dt>"
                "<dd>A normality test emphasizing the tails. The p-value tells you whether the "
                "data is consistent with a normal distribution. Values &lt; 0.05 suggest "
                "non-normality.</dd>"
                "<dt>Confidence intervals</dt>"
                "<dd>CIs for the mean, median, and standard deviation give the precision of "
                "each estimate. Overlapping CIs for mean and median suggest symmetry.</dd>"
                "</dl>"
            ),
        },
        ("stats", "missing_data_analysis"): {
            "title": "Understanding Missing Data Analysis",
            "content": (
                "<dl>"
                "<dt>Why analyze missing data?</dt>"
                "<dd>Not all missingness is random. Patterns in missing data can bias results "
                "if not handled properly. Understanding the mechanism guides the right treatment.</dd>"
                "<dt>Types of missingness</dt>"
                "<dd><strong>MCAR</strong>: Missing Completely At Random — no pattern. "
                "<strong>MAR</strong>: Missing At Random — depends on observed data. "
                "<strong>MNAR</strong>: Missing Not At Random — depends on the missing value itself.</dd>"
                "<dt>Treatment options</dt>"
                "<dd>Listwise deletion (simple but loses data), imputation (mean, median, "
                "regression, multiple), or analysis methods that handle missingness natively.</dd>"
                "<dt>The analysis shows</dt>"
                "<dd>Missing data patterns, rates per variable, correlations between missingness "
                "indicators, and Little's MCAR test to assess the mechanism.</dd>"
                "</dl>"
            ),
        },
        ("stats", "outlier_analysis"): {
            "title": "Understanding Outlier Analysis",
            "content": (
                "<dl>"
                "<dt>What is an outlier?</dt>"
                "<dd>An observation that is unusually far from the main body of data. Outliers "
                "may be errors, special causes, or genuine extreme values — the response "
                "depends on which.</dd>"
                "<dt>Detection methods</dt>"
                "<dd><strong>IQR method</strong>: Beyond Q1 − 1.5×IQR or Q3 + 1.5×IQR. "
                "<strong>Z-score</strong>: |z| &gt; 3. <strong>Grubbs' test</strong>: Formal "
                "hypothesis test for a single outlier. <strong>Isolation Forest</strong>: "
                "ML-based, handles multivariate outliers.</dd>"
                "<dt>What to do with outliers</dt>"
                "<dd>Investigate first — never blindly remove. If they're errors, correct or remove. "
                "If they're special causes, analyze separately. If they're genuine, consider "
                "robust methods that down-weight their influence.</dd>"
                "<dt>Impact assessment</dt>"
                "<dd>The analysis shows how each outlier affects key statistics (mean, correlation, "
                "regression slope). Large influence = high leverage point.</dd>"
                "</dl>"
            ),
        },
        ("stats", "duplicate_analysis"): {
            "title": "Understanding Duplicate Analysis",
            "content": (
                "<dl>"
                "<dt>What does duplicate analysis detect?</dt>"
                "<dd>Exact and near-duplicate rows in your dataset. Duplicates can inflate sample "
                "sizes, bias statistics, and indicate data collection problems.</dd>"
                "<dt>Exact vs near duplicates</dt>"
                "<dd>Exact duplicates match on all columns. Near duplicates match on key columns "
                "but differ slightly on others (possible data entry variations).</dd>"
                "<dt>What to do</dt>"
                "<dd>Investigate why duplicates exist. Common causes: double data entry, "
                "system integration issues, copy-paste errors, or intentional replication. "
                "Remove unintentional duplicates before analysis.</dd>"
                "<dt>Impact on analysis</dt>"
                "<dd>Duplicates violate the independence assumption of most statistical tests, "
                "leading to inflated significance and artificially narrow confidence intervals.</dd>"
                "</dl>"
            ),
        },
        ("stats", "bootstrap_ci"): {
            "title": "Understanding Bootstrap Confidence Intervals",
            "content": (
                "<dl>"
                "<dt>What is bootstrapping?</dt>"
                "<dd>A resampling method that estimates the sampling distribution by repeatedly "
                "drawing samples with replacement from the data. No distributional assumptions required.</dd>"
                "<dt>When to use</dt>"
                "<dd>When parametric assumptions are questionable, for complex statistics without "
                "known sampling distributions, or for small samples where normal approximations fail.</dd>"
                "<dt>Bootstrap CI methods</dt>"
                "<dd><strong>Percentile</strong>: Simple quantiles of bootstrap distribution. "
                "<strong>BCa</strong>: Bias-corrected and accelerated — adjusts for bias and "
                "skewness. Generally preferred for accuracy.</dd>"
                "<dt>Number of resamples</dt>"
                "<dd>1000 is usually adequate for CIs, 10000 for precise p-values. More resamples "
                "reduce Monte Carlo error in the estimate.</dd>"
                "</dl>"
            ),
        },
        ("stats", "box_cox"): {
            "title": "Understanding Box-Cox Transformation",
            "content": (
                "<dl>"
                "<dt>What is the Box-Cox transformation?</dt>"
                "<dd>A family of power transformations (y^λ) that can make non-normal data "
                "approximately normal. The optimal λ is estimated from the data.</dd>"
                "<dt>Common λ values</dt>"
                "<dd>λ = 1: no transformation. λ = 0.5: square root. λ = 0: log. λ = −1: reciprocal. "
                "The algorithm finds the λ that maximizes normality.</dd>"
                "<dt>When to use</dt>"
                "<dd>When data is right-skewed (most common in manufacturing — times, costs, "
                "concentrations). Transform, analyze, then back-transform for interpretation.</dd>"
                "<dt>Limitation</dt>"
                "<dd>Only works for positive data. For data with zeros, add a small constant first. "
                "For data with negative values, use the Yeo-Johnson transformation instead.</dd>"
                "</dl>"
            ),
        },
        ("stats", "johnson_transform"): {
            "title": "Understanding Johnson Transformation",
            "content": (
                "<dl>"
                "<dt>What is the Johnson transformation?</dt>"
                "<dd>A system of three transformation families (SB, SL, SU) that can normalize "
                "virtually any distribution. More flexible than Box-Cox because it handles "
                "bounded, lognormal, and unbounded distributions.</dd>"
                "<dt>The three families</dt>"
                "<dd><strong>SB</strong>: Bounded — for data with natural limits. "
                "<strong>SL</strong>: Lognormal — equivalent to log transformation. "
                "<strong>SU</strong>: Unbounded — for heavy-tailed or platykurtic data.</dd>"
                "<dt>When to use</dt>"
                "<dd>When Box-Cox fails to achieve normality, or when data has natural bounds "
                "(e.g., percentage, proportion, or physically bounded measurements).</dd>"
                "<dt>Capability after transformation</dt>"
                "<dd>Transform the data, compute Cpk on the transformed scale, then report. "
                "The transformation makes capability estimates valid for non-normal data.</dd>"
                "</dl>"
            ),
        },
        ("stats", "run_chart"): {
            "title": "Understanding Run Charts",
            "content": (
                "<dl>"
                "<dt>What is a run chart?</dt>"
                "<dd>A simple time-ordered plot with the median line. It's the precursor to "
                "control charts — lighter on assumptions but less powerful.</dd>"
                "<dt>What do the runs tests check?</dt>"
                "<dd><strong>Clusters</strong>: Too few runs (points stay on one side too long). "
                "<strong>Mixtures</strong>: Too many runs (oscillation). "
                "<strong>Trends</strong>: Sustained increases or decreases. "
                "<strong>Oscillation</strong>: Regular up-down patterns.</dd>"
                "<dt>When to use over control charts</dt>"
                "<dd>Early in data collection when you don't yet have enough data for reliable "
                "control limits (typically need 20+ subgroups). Run charts work with any amount "
                "of time-ordered data.</dd>"
                "<dt>Limitations</dt>"
                "<dd>No control limits, so it can't distinguish common from special cause "
                "variation as precisely as control charts. It's a screening tool, not a "
                "monitoring tool.</dd>"
                "</dl>"
            ),
        },
        ("stats", "grubbs_test"): {
            "title": "Understanding Grubbs' Test",
            "content": (
                "<dl>"
                "<dt>What is Grubbs' test?</dt>"
                "<dd>A formal hypothesis test for detecting a single outlier in a dataset. It tests "
                "whether the most extreme value is significantly different from the rest.</dd>"
                "<dt>How it works</dt>"
                "<dd>The test statistic is the maximum absolute deviation from the mean divided "
                "by the standard deviation. Compared against critical values from the t-distribution.</dd>"
                "<dt>One vs two outliers</dt>"
                "<dd>Basic Grubbs' tests for one outlier. For multiple suspected outliers, apply "
                "iteratively or use ESD (Extreme Studentized Deviate) test instead.</dd>"
                "<dt>Caution</dt>"
                "<dd>Assumes normality of the underlying data. If the data is non-normal, "
                "outlier detection based on normal assumptions may be misleading.</dd>"
                "</dl>"
            ),
        },
        ("stats", "tolerance_interval"): {
            "title": "Understanding Tolerance Intervals",
            "content": (
                "<dl>"
                "<dt>What is a tolerance interval?</dt>"
                "<dd>An interval that contains a specified proportion of the population with a "
                "given confidence. For example: '99% of parts fall between 4.95 and 5.05 mm, "
                "with 95% confidence.'</dd>"
                "<dt>How it differs from confidence and prediction intervals</dt>"
                "<dd><strong>Confidence interval</strong>: Covers the population mean. "
                "<strong>Prediction interval</strong>: Covers the next single observation. "
                "<strong>Tolerance interval</strong>: Covers a specified proportion of all "
                "individuals.</dd>"
                "<dt>Parametric vs nonparametric</dt>"
                "<dd>Parametric assumes normality and uses k-factors. Nonparametric uses order "
                "statistics and requires larger samples but makes no distributional assumption.</dd>"
                "<dt>Applications</dt>"
                "<dd>Establishing natural process limits, setting specification limits, "
                "qualifying processes, and compliance demonstration.</dd>"
                "</dl>"
            ),
        },
        ("stats", "meta_analysis"): {
            "title": "Understanding Meta-Analysis",
            "content": (
                "<dl>"
                "<dt>What is meta-analysis?</dt>"
                "<dd>A statistical method for combining results from multiple independent studies "
                "to get a pooled estimate. More powerful than any single study because it "
                "aggregates evidence.</dd>"
                "<dt>Fixed vs random effects</dt>"
                "<dd><strong>Fixed</strong>: Assumes all studies estimate the same true effect. "
                "<strong>Random</strong>: Allows the true effect to vary across studies. "
                "Random effects is usually more appropriate and always more conservative.</dd>"
                "<dt>Heterogeneity (I²)</dt>"
                "<dd>Measures how much of the variation across studies is real (not sampling error). "
                "I² &lt; 25% = low, 25–75% = moderate, &gt; 75% = high heterogeneity.</dd>"
                "<dt>Forest plot</dt>"
                "<dd>Shows each study's estimate and CI, plus the pooled estimate. Studies "
                "with narrower CIs (larger samples) get more weight in the pool.</dd>"
                "</dl>"
            ),
        },
        ("stats", "effect_size_calculator"): {
            "title": "Understanding Effect Size Calculation",
            "content": (
                "<dl>"
                "<dt>What are effect sizes?</dt>"
                "<dd>Standardized measures of the magnitude of an effect, independent of sample "
                "size. They answer 'how big?' while p-values answer 'is it real?'</dd>"
                "<dt>Common effect sizes</dt>"
                "<dd><strong>Cohen's d</strong>: Standardized mean difference. "
                "<strong>r / R²</strong>: Correlation / variance explained. "
                "<strong>η²</strong>: Variance explained in ANOVA. "
                "<strong>Odds ratio</strong>: Effect on odds in categorical data.</dd>"
                "<dt>Why report effect sizes?</dt>"
                "<dd>P-values alone are misleading — large samples make trivial effects significant. "
                "Effect sizes + confidence intervals give the complete picture. Many journals "
                "now require them.</dd>"
                "<dt>Converting between effect sizes</dt>"
                "<dd>This calculator converts between different effect size metrics so you can "
                "compare results across studies that use different measures.</dd>"
                "</dl>"
            ),
        },
        ("stats", "distribution_fit"): {
            "title": "Understanding Distribution Fitting",
            "content": (
                "<dl>"
                "<dt>What is distribution fitting?</dt>"
                "<dd>Finding the probability distribution (normal, Weibull, lognormal, etc.) that "
                "best describes your data. The fitted distribution can be used for probability "
                "calculations, simulation, and capability analysis.</dd>"
                "<dt>How is fit assessed?</dt>"
                "<dd>Anderson-Darling and Kolmogorov-Smirnov tests measure goodness-of-fit. "
                "Lower AD values and higher p-values indicate better fit. Probability plots "
                "show fit visually — points should follow a straight line.</dd>"
                "<dt>Which distribution to try?</dt>"
                "<dd>Start with normal. If skewed right: lognormal, Weibull, gamma. If bounded: "
                "beta. If count data: Poisson, negative binomial. The analysis compares multiple "
                "candidates automatically.</dd>"
                "<dt>Practical impact</dt>"
                "<dd>Wrong distribution → wrong capability estimate, wrong control limits, wrong "
                "defect rate predictions. Correct distribution identification is the foundation "
                "of reliable process characterization.</dd>"
                "</dl>"
            ),
        },
        ("stats", "mixture_model"): {
            "title": "Understanding Mixture Models",
            "content": (
                "<dl>"
                "<dt>What is a mixture model?</dt>"
                "<dd>It models data as coming from a blend of two or more distributions. Bimodal "
                "or multimodal data often arises from mixing different process states, suppliers, "
                "or conditions.</dd>"
                "<dt>Gaussian Mixture Models (GMM)</dt>"
                "<dd>The most common type — fits a weighted sum of normal distributions. Each "
                "component has its own mean and standard deviation. The EM algorithm estimates "
                "parameters.</dd>"
                "<dt>Selecting the number of components</dt>"
                "<dd>BIC (Bayesian Information Criterion) penalizes complexity — the number of "
                "components that minimizes BIC is usually the best choice.</dd>"
                "<dt>Practical application</dt>"
                "<dd>Identifying mixed process states (e.g., two suppliers producing at different "
                "means), detecting contamination in otherwise clean data, and modeling "
                "multimodal distributions for accurate capability analysis.</dd>"
                "</dl>"
            ),
        },
        ("stats", "sprt"): {
            "title": "Understanding the Sequential Probability Ratio Test",
            "content": (
                "<dl>"
                "<dt>What is SPRT?</dt>"
                "<dd>A sequential hypothesis test that evaluates evidence as data arrives, one "
                "observation at a time. It stops sampling as soon as enough evidence accumulates "
                "for either hypothesis.</dd>"
                "<dt>Why sequential?</dt>"
                "<dd>SPRT typically requires fewer samples than fixed-sample tests (on average "
                "30-50% fewer). You stop early when the evidence is clear, saving resources.</dd>"
                "<dt>Decision boundaries</dt>"
                "<dd>Two boundaries: accept H₀ (no effect) or reject H₀ (effect exists). "
                "The log-likelihood ratio accumulates and triggers a decision when it crosses "
                "either boundary.</dd>"
                "<dt>Applications</dt>"
                "<dd>Lot acceptance sampling, online A/B testing, clinical trials, and any "
                "situation where sampling is expensive and you want to stop as early as possible.</dd>"
                "</dl>"
            ),
        },
    }
)


# ── Stats: Acceptance sampling ────────────────────────────────────────────

_extend(
    {
        ("stats", "acceptance_sampling"): {
            "title": "Understanding Acceptance Sampling",
            "content": (
                "<dl>"
                "<dt>What is acceptance sampling?</dt>"
                "<dd>A statistical method for deciding whether to accept or reject a lot based "
                "on inspecting a sample. Balances inspection cost against the risk of accepting "
                "bad lots or rejecting good ones.</dd>"
                "<dt>Key parameters</dt>"
                "<dd><strong>n</strong>: Sample size. <strong>c</strong>: Acceptance number (max "
                "defectives to accept the lot). <strong>AQL</strong>: Acceptable Quality Level. "
                "<strong>LTPD</strong>: Lot Tolerance Percent Defective (RQL).</dd>"
                "<dt>OC curve</dt>"
                "<dd>The Operating Characteristic curve shows the probability of accepting a lot "
                "as a function of the true defect rate. Steeper curves = better discrimination.</dd>"
                "<dt>Producer's vs consumer's risk</dt>"
                "<dd><strong>α (producer's risk)</strong>: Probability of rejecting a good lot. "
                "<strong>β (consumer's risk)</strong>: Probability of accepting a bad lot. "
                "The sampling plan should protect both parties.</dd>"
                "</dl>"
            ),
        },
        ("stats", "variable_acceptance_sampling"): {
            "title": "Understanding Variable Acceptance Sampling",
            "content": (
                "<dl>"
                "<dt>What is variable acceptance sampling?</dt>"
                "<dd>Instead of classifying items as pass/fail (attribute), it measures a continuous "
                "quality characteristic and uses the sample statistics to decide lot acceptance. "
                "More efficient than attribute sampling because it uses more information.</dd>"
                "<dt>Methods</dt>"
                "<dd><strong>k-method</strong>: Accept if x̄ − k·s ≥ LSL (or USL − x̄ ≥ k·s). "
                "<strong>M-method</strong>: Accept if estimated fraction nonconforming ≤ M.</dd>"
                "<dt>Advantages over attribute sampling</dt>"
                "<dd>Smaller sample sizes for the same protection because continuous measurements "
                "carry more information than pass/fail classifications.</dd>"
                "<dt>Assumption</dt>"
                "<dd>The quality characteristic must be approximately normally distributed within "
                "each lot. If not, attribute sampling is safer.</dd>"
                "</dl>"
            ),
        },
        ("stats", "multiple_plan_comparison"): {
            "title": "Understanding Multiple Sampling Plan Comparison",
            "content": (
                "<dl>"
                "<dt>What does this analysis do?</dt>"
                "<dd>Compares the OC curves, ASN (Average Sample Number), and ATI (Average Total "
                "Inspection) of multiple sampling plans side by side to help select the best one.</dd>"
                "<dt>What to compare</dt>"
                "<dd>Plans that offer similar protection (AQL/LTPD) but differ in sample size, "
                "number of stages, or sampling method. The comparison reveals trade-offs.</dd>"
                "<dt>Single vs double vs sequential</dt>"
                "<dd>Single plans are simplest. Double plans reduce average sample size. "
                "Sequential plans minimize average samples but are more complex to administer.</dd>"
                "<dt>ATI and AOQ</dt>"
                "<dd>ATI (Average Total Inspection) includes 100% inspection of rejected lots. "
                "AOQ (Average Outgoing Quality) shows the quality level after the sampling "
                "program — with the AOQL being the worst-case outgoing quality.</dd>"
                "</dl>"
            ),
        },
    }
)


# ── Stats: Capability ─────────────────────────────────────────────────────

_extend(
    {
        ("stats", "capability_sixpack"): {
            "title": "Understanding the Capability Sixpack",
            "content": (
                "<dl>"
                "<dt>What is the capability sixpack?</dt>"
                "<dd>A six-panel display combining process stability (I-MR control chart or Xbar-R) "
                "with capability analysis (histogram, normal probability plot, capability indices). "
                "The standard Minitab-equivalent capability report.</dd>"
                "<dt>Why stability before capability?</dt>"
                "<dd>Capability indices (Cp, Cpk) are meaningless if the process is unstable. "
                "The control charts verify stability first. Out-of-control points must be "
                "investigated before interpreting capability.</dd>"
                "<dt>Key indices</dt>"
                "<dd><strong>Cp</strong>: Potential capability (spread only). "
                "<strong>Cpk</strong>: Actual capability (spread + centering). "
                "<strong>Pp/Ppk</strong>: Performance indices using overall σ. "
                "Target: ≥ 1.33 for existing processes, ≥ 1.67 for new processes.</dd>"
                "<dt>What-if explorer</dt>"
                "<dd>Adjust the process mean or sigma via sliders to see how capability changes. "
                "This helps set improvement targets and evaluate the impact of centering vs "
                "spread reduction.</dd>"
                "</dl>"
            ),
        },
        ("stats", "nonnormal_capability_np"): {
            "title": "Understanding Non-Normal Capability Analysis",
            "content": (
                "<dl>"
                "<dt>Why non-normal capability?</dt>"
                "<dd>Standard Cpk assumes normality. If your data is skewed, heavy-tailed, or "
                "bounded, normal-based capability can be wildly wrong — typically overestimating "
                "capability and underestimating defect rates.</dd>"
                "<dt>Methods used</dt>"
                "<dd>Box-Cox or Johnson transformation to normalize, then compute standard "
                "indices on the transformed scale. Alternatively, fit a non-normal distribution "
                "and compute Ppk directly from the fitted distribution.</dd>"
                "<dt>Which method is better?</dt>"
                "<dd>Transformation is simpler and more widely accepted. Direct distribution "
                "fitting is more accurate but requires correct distribution identification. "
                "Both are reported for comparison.</dd>"
                "<dt>Practical impact</dt>"
                "<dd>For a right-skewed process, normal-based Ppk might say 1.5 (great) while "
                "the true Ppk is 0.9 (not capable). Always check normality before trusting "
                "capability indices.</dd>"
                "</dl>"
            ),
        },
        ("stats", "attribute_capability"): {
            "title": "Understanding Attribute Capability Analysis",
            "content": (
                "<dl>"
                "<dt>What is attribute capability?</dt>"
                "<dd>Assesses process capability for attribute (pass/fail) data. Instead of Cpk, "
                "it uses defect rates (DPU, DPO, DPMO) and sigma level as capability measures.</dd>"
                "<dt>Key metrics</dt>"
                "<dd><strong>DPU</strong>: Defects Per Unit. <strong>DPO</strong>: Defects Per "
                "Opportunity. <strong>DPMO</strong>: Defects Per Million Opportunities. "
                "<strong>Sigma level</strong>: The Z-value corresponding to the defect rate.</dd>"
                "<dt>Sigma level interpretation</dt>"
                "<dd>3σ = 66,807 DPMO (93.3% yield). 4σ = 6,210 DPMO (99.4%). "
                "5σ = 233 DPMO (99.98%). 6σ = 3.4 DPMO (99.99966%).</dd>"
                "<dt>Confidence intervals</dt>"
                "<dd>Attribute capability estimates have wide confidence intervals with small "
                "samples. Report the CI to show the precision of your capability estimate.</dd>"
                "</dl>"
            ),
        },
    }
)


# Advanced modules loaded via import (SPC, ML, Reliability, Viz, Bayesian, etc.)
from . import advanced  # noqa: F401, E402
