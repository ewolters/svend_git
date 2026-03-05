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

_extend({
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
})


# ── Stats: Nonparametric ──────────────────────────────────────────────────

_extend({
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
})


# ── Stats: ANOVA extensions ───────────────────────────────────────────────

_extend({
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
})


# ── Stats: Post-hoc comparisons ──────────────────────────────────────────

_extend({
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
})


# ── Stats: Power & sample size ────────────────────────────────────────────

_extend({
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
})


# ── Stats: Time series ────────────────────────────────────────────────────

_extend({
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
})


# ── Stats: Survival ───────────────────────────────────────────────────────

_extend({
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
})


# ── Stats: MSA ─────────────────────────────────────────────────────────────

_extend({
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
})


# ── Stats: Multivariate ──────────────────────────────────────────────────

_extend({
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
})


# ── Stats: Exploratory & diagnostics ──────────────────────────────────────

_extend({
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
})


# ── Stats: Acceptance sampling ────────────────────────────────────────────

_extend({
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
})


# ── Stats: Capability ─────────────────────────────────────────────────────

_extend({
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
})


# ═══════════════════════════════════════════════════════════════════════════
# SPC Module
# ═══════════════════════════════════════════════════════════════════════════

_extend({
    ("spc", "imr"): {
        "title": "Understanding I-MR Charts",
        "content": (
            "<dl>"
            "<dt>What is an I-MR chart?</dt>"
            "<dd>Individual-Moving Range chart — for monitoring a process when you have one "
            "measurement per time point (no subgroups). The I chart tracks individual values; "
            "the MR chart tracks consecutive differences.</dd>"
            "<dt>When to use</dt>"
            "<dd>Batch processes, expensive or destructive testing, slow processes where "
            "subgrouping isn't practical. The most common control chart type.</dd>"
            "<dt>Control limits</dt>"
            "<dd>Set at ±3σ from the mean. Points beyond limits signal special cause variation. "
            "σ is estimated from the average moving range (MR̄/1.128).</dd>"
            "<dt>Run rules</dt>"
            "<dd>Beyond 3σ limits, additional patterns signal special causes: 8 points on "
            "one side of center, 6 points trending, 2 of 3 beyond 2σ, etc.</dd>"
            "</dl>"
        ),
    },

    ("spc", "xbar_r"): {
        "title": "Understanding X̄-R Charts",
        "content": (
            "<dl>"
            "<dt>What is an X̄-R chart?</dt>"
            "<dd>Monitors the mean (X̄) and range (R) of subgroups. Best for subgroup "
            "sizes of 2–9. The X̄ chart detects mean shifts; the R chart detects "
            "spread changes.</dd>"
            "<dt>Why subgroups?</dt>"
            "<dd>Rational subgroups (short-term samples) separate within-group variation "
            "(common cause) from between-group variation. This makes the chart sensitive "
            "to process shifts.</dd>"
            "<dt>Reading the charts</dt>"
            "<dd>Always check the R chart first. If the R chart is out of control, the "
            "X̄ chart limits are unreliable. Investigate range signals before mean signals.</dd>"
            "<dt>Subgroup size</dt>"
            "<dd>Larger subgroups make the X̄ chart more sensitive but require more sampling. "
            "n = 5 is the classic default, balancing sensitivity and cost.</dd>"
            "</dl>"
        ),
    },

    ("spc", "xbar_s"): {
        "title": "Understanding X̄-S Charts",
        "content": (
            "<dl>"
            "<dt>What is an X̄-S chart?</dt>"
            "<dd>Like X̄-R but uses standard deviation instead of range. Preferred for "
            "subgroup sizes ≥ 10 where the range is an inefficient estimator of σ.</dd>"
            "<dt>X̄-R vs X̄-S</dt>"
            "<dd>For n &lt; 10, range and s are similarly efficient. For n ≥ 10, the range "
            "ignores most of the data (only min and max). Standard deviation uses all values.</dd>"
            "<dt>Interpreting the S chart</dt>"
            "<dd>Points above the UCL indicate increased variability (special cause). "
            "Points below the LCL indicate decreased variability — possibly good (process "
            "improvement) or suspicious (data truncation).</dd>"
            "<dt>Rational subgrouping</dt>"
            "<dd>Same principles as X̄-R: subgroups should represent short-term, within-group "
            "variation. Between-subgroup variation is what the X̄ chart monitors.</dd>"
            "</dl>"
        ),
    },

    ("spc", "p_chart"): {
        "title": "Understanding P Charts",
        "content": (
            "<dl>"
            "<dt>What is a P chart?</dt>"
            "<dd>Monitors the proportion of defective items per subgroup. Used when each item "
            "is classified as pass or fail and subgroup sizes may vary.</dd>"
            "<dt>Variable subgroup sizes</dt>"
            "<dd>P chart limits adjust with subgroup size — wider for smaller subgroups, "
            "narrower for larger ones. This is correct behavior, not a problem.</dd>"
            "<dt>Assumptions</dt>"
            "<dd>Each item is independent, and the probability of a defect is constant "
            "within each subgroup. Overdispersion (more variation than binomial predicts) "
            "is common — use Laney P' chart in that case.</dd>"
            "<dt>Minimum subgroup size</dt>"
            "<dd>The normal approximation works when np̄ ≥ 5 and n(1−p̄) ≥ 5. For very low "
            "defect rates, you need very large subgroups or should use a c/u chart instead.</dd>"
            "</dl>"
        ),
    },

    ("spc", "np_chart"): {
        "title": "Understanding NP Charts",
        "content": (
            "<dl>"
            "<dt>What is an NP chart?</dt>"
            "<dd>Monitors the count of defective items per subgroup (not the proportion). "
            "Simpler than P charts but requires equal subgroup sizes.</dd>"
            "<dt>NP vs P chart</dt>"
            "<dd>Use NP when subgroup sizes are constant (easier to interpret — counts "
            "are more intuitive than proportions). Use P when subgroup sizes vary.</dd>"
            "<dt>Interpreting signals</dt>"
            "<dd>Points above UCL: more defectives than expected (special cause). "
            "Points below LCL: fewer defectives (possibly improved process — investigate "
            "to sustain the improvement).</dd>"
            "<dt>Control limit calculation</dt>"
            "<dd>Based on the binomial distribution: CL = np̄, UCL = np̄ + 3√(np̄(1−p̄)), "
            "LCL = np̄ − 3√(np̄(1−p̄)).</dd>"
            "</dl>"
        ),
    },

    ("spc", "c_chart"): {
        "title": "Understanding C Charts",
        "content": (
            "<dl>"
            "<dt>What is a C chart?</dt>"
            "<dd>Monitors the count of defects per inspection unit. One unit can have "
            "multiple defects (unlike P charts which track defective items).</dd>"
            "<dt>Defects vs defectives</dt>"
            "<dd>A defective item fails overall. A defect is a single flaw — one item can "
            "have multiple defects. C charts track defects per unit when the inspection "
            "area/opportunity is constant.</dd>"
            "<dt>Assumptions</dt>"
            "<dd>Defects occur independently at a constant rate (Poisson process). "
            "Inspection area must be constant. If area varies, use U charts instead.</dd>"
            "<dt>Control limits</dt>"
            "<dd>Based on the Poisson distribution: CL = c̄, UCL = c̄ + 3√c̄, "
            "LCL = c̄ − 3√c̄.</dd>"
            "</dl>"
        ),
    },

    ("spc", "u_chart"): {
        "title": "Understanding U Charts",
        "content": (
            "<dl>"
            "<dt>What is a U chart?</dt>"
            "<dd>Monitors defects per unit when the inspection area or opportunity varies "
            "between subgroups. The rate-based counterpart to the C chart.</dd>"
            "<dt>U vs C chart</dt>"
            "<dd>Use U when inspection units differ in size (e.g., different length rolls, "
            "different area panels). Use C when units are identical.</dd>"
            "<dt>Overdispersion</dt>"
            "<dd>If the variation exceeds Poisson predictions, the Laney U' chart adjusts "
            "limits to avoid excessive false alarms.</dd>"
            "<dt>Rate interpretation</dt>"
            "<dd>The U chart plots defects per unit, making rates comparable across different "
            "sized inspection units. This is essential for fair comparison.</dd>"
            "</dl>"
        ),
    },

    ("spc", "cusum"): {
        "title": "Understanding CUSUM Charts",
        "content": (
            "<dl>"
            "<dt>What is a CUSUM chart?</dt>"
            "<dd>Cumulative Sum chart — accumulates deviations from a target value. More "
            "sensitive to small, sustained shifts than Shewhart charts.</dd>"
            "<dt>How it works</dt>"
            "<dd>Cumulates positive and negative deviations separately. When either cumulative "
            "sum exceeds a decision interval (H), a shift is signaled.</dd>"
            "<dt>When to use</dt>"
            "<dd>When detecting small shifts (0.5–2σ) is critical. CUSUM detects a 1σ shift "
            "about 5× faster than an I-MR chart.</dd>"
            "<dt>Parameters</dt>"
            "<dd><strong>k</strong>: Reference value (typically half the shift to detect). "
            "<strong>H</strong>: Decision interval (larger = fewer false alarms but slower "
            "detection). Common default: k = 0.5, H = 5.</dd>"
            "</dl>"
        ),
    },

    ("spc", "ewma"): {
        "title": "Understanding EWMA Charts",
        "content": (
            "<dl>"
            "<dt>What is an EWMA chart?</dt>"
            "<dd>Exponentially Weighted Moving Average — gives more weight to recent "
            "observations and less to older ones. Like CUSUM, it's sensitive to small shifts.</dd>"
            "<dt>The smoothing parameter (λ)</dt>"
            "<dd>Controls memory: small λ (0.05–0.1) for detecting small shifts, "
            "large λ (0.2–0.4) for larger shifts. λ = 1 reduces to a Shewhart chart.</dd>"
            "<dt>EWMA vs CUSUM vs Shewhart</dt>"
            "<dd>All detect large shifts similarly. For small shifts: CUSUM and EWMA are "
            "both much faster. EWMA is slightly easier to implement and interpret.</dd>"
            "<dt>Robustness</dt>"
            "<dd>EWMA is robust to non-normality due to the averaging effect. Good for "
            "individual measurements where normality is questionable.</dd>"
            "</dl>"
        ),
    },

    ("spc", "laney_p"): {
        "title": "Understanding Laney P' Charts",
        "content": (
            "<dl>"
            "<dt>What is a Laney P' chart?</dt>"
            "<dd>A modified P chart that adjusts for overdispersion (excess variation beyond "
            "what the binomial model predicts). Prevents excessive false alarms.</dd>"
            "<dt>When to use</dt>"
            "<dd>When large subgroups produce too many out-of-control signals on a standard "
            "P chart. Overdispersion is common with large samples because the binomial "
            "assumption of constant p is rarely exact.</dd>"
            "<dt>The sigma-Z factor</dt>"
            "<dd>Laney calculates a sigma-Z factor from the standardized residuals. "
            "If σ_Z ≈ 1, standard P chart is fine. If σ_Z &gt; 1, limits need widening.</dd>"
            "<dt>Real-world relevance</dt>"
            "<dd>Very common in high-volume manufacturing, healthcare (large patient counts), "
            "and any application where subgroup sizes are in the hundreds or thousands.</dd>"
            "</dl>"
        ),
    },

    ("spc", "laney_u"): {
        "title": "Understanding Laney U' Charts",
        "content": (
            "<dl>"
            "<dt>What is a Laney U' chart?</dt>"
            "<dd>A modified U chart that adjusts for overdispersion in defect rate data. "
            "The rate-based equivalent of the Laney P' chart.</dd>"
            "<dt>When to use</dt>"
            "<dd>When U chart limits are too tight due to large inspection areas, causing "
            "false alarms. The Poisson assumption (mean = variance) is violated.</dd>"
            "<dt>How it works</dt>"
            "<dd>Estimates the overdispersion factor from sigma-Z of standardized residuals, "
            "then inflates the control limits accordingly.</dd>"
            "<dt>Interpretation</dt>"
            "<dd>Same as U chart — signals indicate rate changes. But with Laney adjustment, "
            "signals are genuine process changes, not artifacts of Poisson model inadequacy.</dd>"
            "</dl>"
        ),
    },

    ("spc", "moving_average"): {
        "title": "Understanding Moving Average Charts",
        "content": (
            "<dl>"
            "<dt>What is a moving average chart?</dt>"
            "<dd>Plots the average of the last w observations at each time point. Smooths "
            "out noise to reveal trends and shifts that are hard to see in raw data.</dd>"
            "<dt>Window size (w)</dt>"
            "<dd>Larger windows = more smoothing but slower response. Smaller windows = "
            "less smoothing but faster response. w = 3–5 is common.</dd>"
            "<dt>Moving average vs EWMA</dt>"
            "<dd>Moving average gives equal weight to the last w observations and zero to "
            "all others. EWMA gives exponentially decreasing weights — smoother and more "
            "responsive to recent changes.</dd>"
            "<dt>Control limits</dt>"
            "<dd>Limits are tighter than I-MR because averaging reduces variation. They "
            "adjust for the number of observations in each average.</dd>"
            "</dl>"
        ),
    },

    ("spc", "zone_chart"): {
        "title": "Understanding Zone Charts",
        "content": (
            "<dl>"
            "<dt>What is a zone chart?</dt>"
            "<dd>Divides the control chart into zones (A, B, C — each 1σ wide) and assigns "
            "scores based on where points fall. Cumulates scores to detect patterns "
            "that individual run rules might miss.</dd>"
            "<dt>How scoring works</dt>"
            "<dd>Zone C (near center): 0 points. Zone B (1–2σ): 2 points. Zone A (2–3σ): "
            "4 points. Beyond 3σ: 8 points. Scores reset when the cumulative sum is "
            "signaled.</dd>"
            "<dt>When to use</dt>"
            "<dd>As a more sensitive alternative to standard Shewhart run rules. Zone charts "
            "detect small shifts faster because they accumulate evidence from successive "
            "near-misses.</dd>"
            "<dt>Interpretation</dt>"
            "<dd>A signal (cumulative score exceeds threshold) indicates the process has "
            "shifted. Investigate the sequence of scores to identify when the shift began.</dd>"
            "</dl>"
        ),
    },

    ("spc", "mewma"): {
        "title": "Understanding MEWMA Charts",
        "content": (
            "<dl>"
            "<dt>What is MEWMA?</dt>"
            "<dd>Multivariate EWMA — monitors multiple correlated variables simultaneously. "
            "It detects small shifts in the multivariate mean vector.</dd>"
            "<dt>Why multivariate?</dt>"
            "<dd>Running separate charts per variable misses shifts in the correlation "
            "structure. MEWMA catches shifts that no individual chart would detect.</dd>"
            "<dt>The smoothing parameter</dt>"
            "<dd>Like univariate EWMA, small r (0.05–0.1) detects small shifts; larger r "
            "detects larger shifts. The optimal r depends on the shift size you need to detect.</dd>"
            "<dt>Interpreting signals</dt>"
            "<dd>A signal means the multivariate mean has shifted, but doesn't tell you "
            "which variable changed. Follow up with individual charts or decomposition "
            "to identify the source.</dd>"
            "</dl>"
        ),
    },

    ("spc", "generalized_variance"): {
        "title": "Understanding Generalized Variance Charts",
        "content": (
            "<dl>"
            "<dt>What is a generalized variance chart?</dt>"
            "<dd>Monitors the determinant of the covariance matrix (|S|) over time. "
            "It detects changes in the overall multivariate spread — the volume of "
            "the data cloud in multivariate space.</dd>"
            "<dt>When to use</dt>"
            "<dd>When monitoring the consistency (not just the mean) of a multivariate "
            "process. Complements MEWMA which monitors the mean vector.</dd>"
            "<dt>Interpretation</dt>"
            "<dd>Increasing |S| means the process is becoming more variable (less consistent). "
            "Decreasing |S| means variables are becoming more tightly correlated or less variable.</dd>"
            "<dt>Practical application</dt>"
            "<dd>Monitoring multivariate process stability before computing multivariate "
            "capability indices. If the covariance structure changes, capability is undefined.</dd>"
            "</dl>"
        ),
    },

    ("spc", "capability"): {
        "title": "Understanding Process Capability Analysis",
        "content": (
            "<dl>"
            "<dt>What is process capability?</dt>"
            "<dd>A comparison of the voice of the process (natural variation) against the "
            "voice of the customer (specification limits). It answers: 'Can this process "
            "consistently meet requirements?'</dd>"
            "<dt>Cp vs Cpk</dt>"
            "<dd><strong>Cp</strong>: Potential capability — ignores centering, only considers "
            "spread. <strong>Cpk</strong>: Actual capability — accounts for how well the "
            "process is centered within specs. Cpk ≤ Cp always.</dd>"
            "<dt>Target values</dt>"
            "<dd>Cpk ≥ 1.33 for existing processes (4σ from nearest spec). Cpk ≥ 1.67 for "
            "new processes or safety-critical applications.</dd>"
            "<dt>Pp/Ppk vs Cp/Cpk</dt>"
            "<dd>Cp/Cpk use within-subgroup σ (process capability). Pp/Ppk use overall σ "
            "(process performance). The gap between them indicates how much variation comes "
            "from between-subgroup shifts.</dd>"
            "</dl>"
        ),
    },

    ("spc", "nonnormal_capability"): {
        "title": "Understanding Non-Normal Capability (SPC)",
        "content": (
            "<dl>"
            "<dt>Why non-normal capability in SPC?</dt>"
            "<dd>When control chart data is non-normal (skewed, bounded, or heavy-tailed), "
            "standard Cpk calculations based on ±3σ are wrong. The actual tail probabilities "
            "differ from what the normal distribution predicts.</dd>"
            "<dt>Methods</dt>"
            "<dd>Percentile-based (Cnpk): Uses actual data percentiles instead of ±3σ. "
            "Transformation-based: Box-Cox or Johnson transform to normality. "
            "Distribution fitting: Fit the correct distribution and compute from its CDF.</dd>"
            "<dt>Which to choose</dt>"
            "<dd>If the non-normal distribution is known (e.g., Weibull for time data), "
            "use distribution fitting. If unknown, percentile-based is safest. "
            "Transformation is a good middle ground.</dd>"
            "<dt>Impact</dt>"
            "<dd>For a skewed process, normal-based Cpk may be off by 0.3–0.5 or more. "
            "This directly affects defect rate predictions and acceptance decisions.</dd>"
            "</dl>"
        ),
    },

    ("spc", "between_within"): {
        "title": "Understanding Between/Within Capability",
        "content": (
            "<dl>"
            "<dt>What is between/within analysis?</dt>"
            "<dd>Decomposes total variation into between-subgroup and within-subgroup "
            "components. Shows whether variation comes from short-term repeatability or "
            "longer-term shifts.</dd>"
            "<dt>Why does this matter?</dt>"
            "<dd>If between-subgroup variation dominates, the process shifts over time — "
            "fix the assignable causes. If within-subgroup variation dominates, the "
            "inherent process noise is too high — fundamental process change needed.</dd>"
            "<dt>Cpk vs Ppk gap</dt>"
            "<dd>A large gap between Cpk (within) and Ppk (overall) quantifies the impact "
            "of between-subgroup variation. Closing this gap is a specific improvement target.</dd>"
            "<dt>Practical use</dt>"
            "<dd>Before launching improvement projects, this analysis tells you whether "
            "to focus on stabilizing the process (reducing shifts) or reducing inherent "
            "variability (process redesign).</dd>"
            "</dl>"
        ),
    },

    ("spc", "conformal_control"): {
        "title": "Understanding Conformal Control Charts",
        "content": (
            "<dl>"
            "<dt>What is a conformal control chart?</dt>"
            "<dd>A distribution-free control chart based on conformal prediction. It uses "
            "nonconformity scores rather than distributional assumptions to set control limits.</dd>"
            "<dt>When to use</dt>"
            "<dd>When data is non-normal and traditional charts are unreliable. Conformal "
            "methods provide valid coverage guarantees regardless of the data distribution.</dd>"
            "<dt>How it works</dt>"
            "<dd>Each new observation is compared against a calibration set to compute a "
            "p-value (conformal p-value). Points with very small p-values are out of control.</dd>"
            "<dt>Advantages</dt>"
            "<dd>Distribution-free, finite-sample valid, and handles multivariate data "
            "naturally. The main limitation is requiring a representative calibration set.</dd>"
            "</dl>"
        ),
    },

    ("spc", "conformal_monitor"): {
        "title": "Understanding Conformal Process Monitoring",
        "content": (
            "<dl>"
            "<dt>What is conformal monitoring?</dt>"
            "<dd>Continuous process monitoring using conformal prediction. Unlike one-time "
            "control charts, it adaptively updates as new data arrives and monitors "
            "for distributional changes in real time.</dd>"
            "<dt>Martingale approach</dt>"
            "<dd>Conformal p-values are combined via a betting martingale. When the "
            "martingale value exceeds a threshold, a change is detected — with "
            "mathematically guaranteed false alarm control.</dd>"
            "<dt>Advantages over traditional SPC</dt>"
            "<dd>No normality assumption, handles multivariate data, provides valid "
            "Type I error control in finite samples, and can detect distributional "
            "changes (not just mean shifts).</dd>"
            "<dt>When to use</dt>"
            "<dd>Complex processes with non-normal, multivariate, or non-stationary data "
            "where traditional SPC assumptions are violated.</dd>"
            "</dl>"
        ),
    },

    ("spc", "entropy_spc"): {
        "title": "Understanding Entropy-Based SPC",
        "content": (
            "<dl>"
            "<dt>What is entropy SPC?</dt>"
            "<dd>Monitors process entropy (information content) over time. Entropy measures "
            "the uncertainty or complexity of the process distribution — changes in entropy "
            "signal distributional shifts.</dd>"
            "<dt>Why entropy?</dt>"
            "<dd>Traditional SPC detects mean and variance shifts. Entropy detects any "
            "distributional change — including shape changes, modality changes, and "
            "tail behavior changes that mean/variance charts miss.</dd>"
            "<dt>How to interpret</dt>"
            "<dd>Increasing entropy: Process becoming more variable or unpredictable. "
            "Decreasing entropy: Process becoming more concentrated or predictable. "
            "Sudden changes: Distributional shift occurred.</dd>"
            "<dt>Applications</dt>"
            "<dd>Complex manufacturing processes where the failure mode affects the "
            "distribution shape rather than just the mean or variance.</dd>"
            "</dl>"
        ),
    },

    ("spc", "degradation_capability"): {
        "title": "Understanding Degradation Capability",
        "content": (
            "<dl>"
            "<dt>What is degradation capability?</dt>"
            "<dd>Assesses whether a degrading process will remain capable over its planned "
            "life. It models the degradation trend and projects when capability will drop "
            "below acceptable levels.</dd>"
            "<dt>How it works</dt>"
            "<dd>Fits a degradation model (linear, exponential, or power) to time-ordered "
            "capability measurements. Projects future capability with prediction intervals.</dd>"
            "<dt>Key output</dt>"
            "<dd>The time at which predicted Cpk drops below the threshold (e.g., 1.33). "
            "This drives maintenance scheduling and replacement planning.</dd>"
            "<dt>Applications</dt>"
            "<dd>Tool wear monitoring, equipment aging, filter degradation, battery capacity "
            "fade — any process where performance deteriorates predictably over time.</dd>"
            "</dl>"
        ),
    },
})


# ═══════════════════════════════════════════════════════════════════════════
# ML Module
# ═══════════════════════════════════════════════════════════════════════════

_extend({
    ("ml", "classification"): {
        "title": "Understanding Classification Models",
        "content": (
            "<dl>"
            "<dt>What is classification?</dt>"
            "<dd>Predicting a categorical outcome (pass/fail, defect type, severity level) "
            "from input features. Models learn the boundary between categories from "
            "labeled training data.</dd>"
            "<dt>Key metrics</dt>"
            "<dd><strong>Accuracy</strong>: Overall correct predictions. <strong>Precision</strong>: "
            "Of predicted positives, how many are correct. <strong>Recall</strong>: Of actual "
            "positives, how many were caught. <strong>F1</strong>: Harmonic mean of precision/recall.</dd>"
            "<dt>Confusion matrix</dt>"
            "<dd>Shows true positives, false positives, true negatives, false negatives. "
            "Reveals whether errors are missed defects (costly) or false alarms (wasteful).</dd>"
            "<dt>Which metric to optimize?</dt>"
            "<dd>In quality: usually recall (catching all defects) matters more than precision. "
            "The cost of missing a defect vs false alarm determines the right trade-off.</dd>"
            "</dl>"
        ),
    },

    ("ml", "regression_ml"): {
        "title": "Understanding ML Regression Models",
        "content": (
            "<dl>"
            "<dt>What is ML regression?</dt>"
            "<dd>Predicting a continuous outcome from input features using machine learning "
            "models (random forest, gradient boosting, neural networks). More flexible "
            "than linear regression but less interpretable.</dd>"
            "<dt>Key metrics</dt>"
            "<dd><strong>RMSE</strong>: Root Mean Square Error — in the units of the response. "
            "<strong>MAE</strong>: Mean Absolute Error — less sensitive to outliers. "
            "<strong>R²</strong>: Variance explained — same interpretation as linear regression.</dd>"
            "<dt>Train/test split</dt>"
            "<dd>Model performance on training data is optimistic. Always evaluate on held-out "
            "test data to assess generalization. Cross-validation gives more reliable estimates.</dd>"
            "<dt>Feature importance</dt>"
            "<dd>Shows which predictors matter most. Permutation importance is model-agnostic "
            "and reliable. Use SHAP values for detailed per-prediction explanations.</dd>"
            "</dl>"
        ),
    },

    ("ml", "model_compare"): {
        "title": "Understanding Model Comparison",
        "content": (
            "<dl>"
            "<dt>What does model comparison do?</dt>"
            "<dd>Trains multiple model types on the same data and compares their performance. "
            "Identifies which algorithm works best for your specific problem.</dd>"
            "<dt>Why compare?</dt>"
            "<dd>No single algorithm dominates. Random forest may beat gradient boosting on "
            "one dataset and lose on another. Systematic comparison prevents premature "
            "commitment to a suboptimal model.</dd>"
            "<dt>Fair comparison</dt>"
            "<dd>All models use the same train/test split and cross-validation folds. "
            "Performance is compared on the same held-out data to prevent information leakage.</dd>"
            "<dt>Beyond accuracy</dt>"
            "<dd>Consider training time, prediction speed, interpretability, and robustness "
            "to new data. The 'best' model balances all of these, not just test accuracy.</dd>"
            "</dl>"
        ),
    },

    ("ml", "xgboost"): {
        "title": "Understanding XGBoost",
        "content": (
            "<dl>"
            "<dt>What is XGBoost?</dt>"
            "<dd>Extreme Gradient Boosting — an ensemble method that sequentially builds "
            "decision trees, each correcting the errors of the previous ones. Often the "
            "best-performing algorithm for structured/tabular data.</dd>"
            "<dt>Key hyperparameters</dt>"
            "<dd><strong>n_estimators</strong>: Number of trees. <strong>max_depth</strong>: "
            "Tree complexity (3–6 typical). <strong>learning_rate</strong>: Step size "
            "(smaller = more trees needed but better generalization).</dd>"
            "<dt>Regularization</dt>"
            "<dd>Built-in L1 and L2 regularization prevents overfitting. XGBoost handles "
            "this better than random forests, especially with many features.</dd>"
            "<dt>Feature importance</dt>"
            "<dd>Gain-based importance shows which features contribute most to predictions. "
            "SHAP values provide more reliable and detailed feature attributions.</dd>"
            "</dl>"
        ),
    },

    ("ml", "lightgbm"): {
        "title": "Understanding LightGBM",
        "content": (
            "<dl>"
            "<dt>What is LightGBM?</dt>"
            "<dd>Light Gradient Boosting Machine — similar to XGBoost but faster on large "
            "datasets. Uses histogram-based splitting and leaf-wise tree growth.</dd>"
            "<dt>LightGBM vs XGBoost</dt>"
            "<dd>LightGBM is typically faster and more memory-efficient. XGBoost is more "
            "established with better documentation. Performance is usually similar — "
            "try both and compare.</dd>"
            "<dt>Handling categoricals</dt>"
            "<dd>LightGBM natively handles categorical features without one-hot encoding, "
            "which can improve both speed and accuracy.</dd>"
            "<dt>Overfitting risks</dt>"
            "<dd>Leaf-wise growth can overfit on small datasets. Use min_data_in_leaf and "
            "num_leaves to control complexity. Cross-validation is essential.</dd>"
            "</dl>"
        ),
    },

    ("ml", "shap_explain"): {
        "title": "Understanding SHAP Explanations",
        "content": (
            "<dl>"
            "<dt>What is SHAP?</dt>"
            "<dd>SHapley Additive exPlanations — a game-theoretic approach to explaining "
            "individual predictions. Each feature gets a SHAP value showing its contribution "
            "to moving the prediction from the average.</dd>"
            "<dt>Why SHAP over feature importance?</dt>"
            "<dd>SHAP values are consistent, locally accurate, and additive. They explain "
            "<em>each prediction</em> individually, not just global patterns. This reveals "
            "when and why a model makes mistakes.</dd>"
            "<dt>Reading the plots</dt>"
            "<dd><strong>Summary plot</strong>: Feature importance + direction of effect. "
            "<strong>Dependence plot</strong>: How SHAP values vary with feature values. "
            "<strong>Force plot</strong>: Contribution breakdown for a single prediction.</dd>"
            "<dt>Actionable insights</dt>"
            "<dd>SHAP reveals which factors drive predictions for specific cases, enabling "
            "targeted process adjustments rather than blanket changes.</dd>"
            "</dl>"
        ),
    },

    ("ml", "hyperparameter_tune"): {
        "title": "Understanding Hyperparameter Tuning",
        "content": (
            "<dl>"
            "<dt>What is hyperparameter tuning?</dt>"
            "<dd>Systematically searching for the best model configuration (tree depth, "
            "learning rate, regularization, etc.) to maximize performance. Like DOE for "
            "machine learning.</dd>"
            "<dt>Methods</dt>"
            "<dd><strong>Grid search</strong>: Try all combinations (exhaustive but slow). "
            "<strong>Random search</strong>: Random combinations (surprisingly effective). "
            "<strong>Bayesian optimization</strong>: Learns from past trials (most efficient).</dd>"
            "<dt>Cross-validation</dt>"
            "<dd>Each configuration is evaluated using k-fold cross-validation to prevent "
            "overfitting to the validation set. Never tune on the test set.</dd>"
            "<dt>Diminishing returns</dt>"
            "<dd>Default parameters often get you 90% of the way. Tuning typically improves "
            "performance by 1–5%. Focus on the most impactful hyperparameters first.</dd>"
            "</dl>"
        ),
    },

    ("ml", "clustering"): {
        "title": "Understanding Clustering Analysis",
        "content": (
            "<dl>"
            "<dt>What is clustering?</dt>"
            "<dd>Unsupervised learning that groups similar observations together without "
            "pre-defined labels. Discovers natural structure in data.</dd>"
            "<dt>K-Means vs hierarchical vs DBSCAN</dt>"
            "<dd><strong>K-Means</strong>: Fast, assumes spherical clusters, needs k specified. "
            "<strong>Hierarchical</strong>: Builds a tree of clusters, no k needed. "
            "<strong>DBSCAN</strong>: Finds arbitrary shapes, handles noise, no k needed.</dd>"
            "<dt>Choosing k</dt>"
            "<dd>Elbow method (diminishing returns in within-cluster variance), silhouette "
            "score (cluster separation quality), or domain knowledge.</dd>"
            "<dt>Applications in quality</dt>"
            "<dd>Identifying process modes, segmenting products, grouping failure patterns, "
            "and discovering hidden subpopulations in process data.</dd>"
            "</dl>"
        ),
    },

    ("ml", "pca"): {
        "title": "Understanding Principal Component Analysis",
        "content": (
            "<dl>"
            "<dt>What is PCA?</dt>"
            "<dd>Transforms correlated variables into uncorrelated principal components, "
            "ordered by variance explained. The first few PCs capture most of the information.</dd>"
            "<dt>How many PCs to keep?</dt>"
            "<dd>Keep enough to explain 80–95% of total variance. The scree plot shows "
            "diminishing returns — the 'elbow' indicates the right cutoff.</dd>"
            "<dt>Loading interpretation</dt>"
            "<dd>Loadings show how each original variable contributes to each PC. Large "
            "loadings (positive or negative) indicate strong contributors.</dd>"
            "<dt>Applications</dt>"
            "<dd>Dimensionality reduction for visualization, multicollinearity removal "
            "before regression, multivariate process monitoring (T² and SPE charts), "
            "and feature engineering.</dd>"
            "</dl>"
        ),
    },

    ("ml", "feature"): {
        "title": "Understanding Feature Selection/Engineering",
        "content": (
            "<dl>"
            "<dt>What is feature selection?</dt>"
            "<dd>Identifying which input variables are most relevant for prediction. Removing "
            "irrelevant features reduces overfitting, speeds training, and improves "
            "interpretability.</dd>"
            "<dt>Methods</dt>"
            "<dd><strong>Filter</strong>: Correlation, mutual information (fast, model-free). "
            "<strong>Wrapper</strong>: Forward/backward selection (model-specific). "
            "<strong>Embedded</strong>: LASSO, tree importance (built into model training).</dd>"
            "<dt>Feature engineering</dt>"
            "<dd>Creating new features from existing ones — interactions, polynomials, "
            "domain-specific transformations. Good features often matter more than model choice.</dd>"
            "<dt>Multicollinearity check</dt>"
            "<dd>Highly correlated features add noise without information. VIF &gt; 5 suggests "
            "removal. PCA can combine correlated features into independent components.</dd>"
            "</dl>"
        ),
    },

    ("ml", "bayesian_regression"): {
        "title": "Understanding Bayesian ML Regression",
        "content": (
            "<dl>"
            "<dt>What is Bayesian ML regression?</dt>"
            "<dd>Regression with full posterior distributions over predictions. Unlike point "
            "estimates, it quantifies prediction uncertainty — critical for decision-making.</dd>"
            "<dt>Prediction uncertainty</dt>"
            "<dd>Each prediction comes with a credible interval showing the range of "
            "plausible values. Wider intervals mean less certainty.</dd>"
            "<dt>Prior specification</dt>"
            "<dd>Priors encode existing knowledge about the relationship. Weakly informative "
            "priors regularize without dominating the data.</dd>"
            "<dt>When to use</dt>"
            "<dd>When prediction uncertainty matters (safety-critical, cost-sensitive), "
            "with small datasets, or when you have genuine prior knowledge to incorporate.</dd>"
            "</dl>"
        ),
    },

    ("ml", "gam"): {
        "title": "Understanding Generalized Additive Models (GAM)",
        "content": (
            "<dl>"
            "<dt>What is a GAM?</dt>"
            "<dd>A flexible regression model that uses smooth functions for each predictor: "
            "y = f₁(x₁) + f₂(x₂) + ... The smooth functions (splines) capture nonlinear "
            "effects while maintaining additive interpretability.</dd>"
            "<dt>GAM vs linear vs tree models</dt>"
            "<dd>More flexible than linear (captures curves). More interpretable than trees "
            "(each effect is a smooth function you can plot). Good middle ground.</dd>"
            "<dt>Partial dependence plots</dt>"
            "<dd>Each smooth function can be plotted individually — showing exactly how each "
            "predictor relates to the response. These are the GAM's greatest strength.</dd>"
            "<dt>Smoothing parameter</dt>"
            "<dd>Controls the wiggliness of each smooth function. Too smooth = underfitting. "
            "Too wiggly = overfitting. Cross-validation selects the optimal smoothness.</dd>"
            "</dl>"
        ),
    },

    ("ml", "isolation_forest"): {
        "title": "Understanding Isolation Forest",
        "content": (
            "<dl>"
            "<dt>What is Isolation Forest?</dt>"
            "<dd>An anomaly detection algorithm that isolates outliers by randomly partitioning "
            "data. Anomalies are easier to isolate (require fewer splits) than normal points.</dd>"
            "<dt>How it works</dt>"
            "<dd>Builds random trees by selecting random features and random split points. "
            "The average path length to isolate a point becomes its anomaly score — shorter "
            "paths = more anomalous.</dd>"
            "<dt>Contamination parameter</dt>"
            "<dd>The expected proportion of outliers. Setting it too high flags normal points; "
            "too low misses real anomalies. Start with domain knowledge of typical defect rates.</dd>"
            "<dt>Multivariate advantage</dt>"
            "<dd>Unlike z-scores, Isolation Forest detects multivariate anomalies — points that "
            "are normal on each variable individually but unusual in combination.</dd>"
            "</dl>"
        ),
    },

    ("ml", "gaussian_process"): {
        "title": "Understanding Gaussian Process Regression",
        "content": (
            "<dl>"
            "<dt>What is a Gaussian Process?</dt>"
            "<dd>A non-parametric Bayesian regression method that provides predictions with "
            "uncertainty estimates. It defines a distribution over functions, not just "
            "point predictions.</dd>"
            "<dt>Uncertainty quantification</dt>"
            "<dd>Predictions near training data have narrow uncertainty bands. Predictions "
            "far from training data have wide bands. This 'knows what it doesn't know'.</dd>"
            "<dt>Kernel functions</dt>"
            "<dd>The kernel defines the smoothness and structure of the learned function. "
            "RBF (squared exponential) is the default. Matérn kernels are common alternatives.</dd>"
            "<dt>Limitations</dt>"
            "<dd>Scales as O(n³) — impractical for &gt; 5000 data points without approximations. "
            "Best for smaller datasets where uncertainty quantification justifies the cost.</dd>"
            "</dl>"
        ),
    },

    ("ml", "pls"): {
        "title": "Understanding Partial Least Squares (PLS)",
        "content": (
            "<dl>"
            "<dt>What is PLS?</dt>"
            "<dd>A regression method designed for situations with many correlated predictors "
            "and few samples. It finds latent components that maximize covariance between "
            "predictors and response.</dd>"
            "<dt>PLS vs PCR vs OLS</dt>"
            "<dd>OLS fails with multicollinearity. PCR reduces predictors via PCA but ignores "
            "the response. PLS finds components that are relevant to both prediction and "
            "explaining the response — usually better.</dd>"
            "<dt>When to use</dt>"
            "<dd>Spectroscopy, chemometrics, sensor data — any domain where p &gt; n (more "
            "variables than observations) or variables are highly correlated.</dd>"
            "<dt>Number of components</dt>"
            "<dd>Cross-validation selects the optimal number. Too few = underfitting. "
            "Too many = overfitting (approaches OLS problems).</dd>"
            "</dl>"
        ),
    },

    ("ml", "sem"): {
        "title": "Understanding Structural Equation Modeling",
        "content": (
            "<dl>"
            "<dt>What is SEM?</dt>"
            "<dd>A framework for modeling complex relationships including latent variables "
            "(unobserved constructs), mediating effects, and simultaneous equations. "
            "Combines factor analysis with path analysis.</dd>"
            "<dt>Measurement vs structural model</dt>"
            "<dd>The measurement model defines how observed variables relate to latent "
            "constructs. The structural model defines relationships between constructs.</dd>"
            "<dt>Fit indices</dt>"
            "<dd><strong>CFI</strong> &gt; 0.95, <strong>RMSEA</strong> &lt; 0.06, "
            "<strong>SRMR</strong> &lt; 0.08 indicate good fit. No single index is sufficient — "
            "report multiple indices.</dd>"
            "<dt>When to use</dt>"
            "<dd>When you have theoretical models with latent variables (quality culture, "
            "employee engagement) or complex mediating/moderating relationships.</dd>"
            "</dl>"
        ),
    },

    ("ml", "regularized_regression"): {
        "title": "Understanding Regularized Regression",
        "content": (
            "<dl>"
            "<dt>What is regularization?</dt>"
            "<dd>Adding a penalty to the loss function to shrink coefficients toward zero. "
            "Prevents overfitting and handles multicollinearity.</dd>"
            "<dt>LASSO (L1) vs Ridge (L2) vs Elastic Net</dt>"
            "<dd><strong>LASSO</strong>: Shrinks some coefficients to exactly zero (feature selection). "
            "<strong>Ridge</strong>: Shrinks all coefficients but keeps all features. "
            "<strong>Elastic Net</strong>: Combines both — good default when unsure.</dd>"
            "<dt>Choosing the penalty strength (λ)</dt>"
            "<dd>Cross-validation selects the optimal λ. Larger λ = more shrinkage = simpler "
            "model. The CV curve shows the bias-variance trade-off.</dd>"
            "<dt>When to use</dt>"
            "<dd>When you have many predictors relative to observations, multicollinearity, "
            "or want automatic feature selection (LASSO).</dd>"
            "</dl>"
        ),
    },

    ("ml", "discriminant_analysis"): {
        "title": "Understanding Discriminant Analysis",
        "content": (
            "<dl>"
            "<dt>What is discriminant analysis?</dt>"
            "<dd>Finds linear combinations of features that best separate known groups. "
            "Both a classification method and a dimensionality reduction technique.</dd>"
            "<dt>LDA vs QDA</dt>"
            "<dd><strong>LDA</strong>: Assumes equal covariance matrices across groups (linear "
            "boundary). <strong>QDA</strong>: Allows different covariances (quadratic boundary). "
            "LDA is more stable with small samples.</dd>"
            "<dt>Discriminant functions</dt>"
            "<dd>The coefficients show which variables best discriminate between groups. "
            "The first function captures the most discrimination, the second captures "
            "the next most, etc.</dd>"
            "<dt>Applications</dt>"
            "<dd>Classifying products, defect typing, process state identification — any "
            "multivariate classification where interpretability matters.</dd>"
            "</dl>"
        ),
    },

    ("ml", "factor_analysis"): {
        "title": "Understanding Factor Analysis",
        "content": (
            "<dl>"
            "<dt>What is factor analysis?</dt>"
            "<dd>Identifies latent factors that explain the correlations among observed "
            "variables. Unlike PCA, it models measurement error explicitly.</dd>"
            "<dt>Factor analysis vs PCA</dt>"
            "<dd>PCA maximizes variance explained (data reduction). Factor analysis models "
            "the correlation structure (latent variable discovery). Use PCA for prediction, "
            "factor analysis for understanding constructs.</dd>"
            "<dt>Rotation</dt>"
            "<dd><strong>Varimax</strong>: Orthogonal rotation — factors are uncorrelated. "
            "<strong>Promax</strong>: Oblique rotation — factors can correlate. Oblique is "
            "usually more realistic for real-world data.</dd>"
            "<dt>Factor loadings</dt>"
            "<dd>Show how strongly each variable relates to each factor. Loadings &gt; 0.4 "
            "indicate a meaningful relationship. Cross-loadings suggest the factor "
            "structure isn't clean.</dd>"
            "</dl>"
        ),
    },

    ("ml", "correspondence_analysis"): {
        "title": "Understanding Correspondence Analysis",
        "content": (
            "<dl>"
            "<dt>What is correspondence analysis?</dt>"
            "<dd>A dimensionality reduction technique for categorical data (contingency tables). "
            "The categorical equivalent of PCA — visualizes the association between row and "
            "column categories.</dd>"
            "<dt>The biplot</dt>"
            "<dd>Row and column categories are plotted in the same space. Categories close "
            "together in the plot are associated. Distance from the origin indicates "
            "distinctiveness.</dd>"
            "<dt>Inertia</dt>"
            "<dd>The total inertia (chi-square/n) measures the overall association. The "
            "proportion explained by each dimension shows how well the 2D plot captures "
            "the full association structure.</dd>"
            "<dt>Applications</dt>"
            "<dd>Market research, defect-by-cause analysis, survey data exploration — any "
            "situation with two categorical variables and a contingency table.</dd>"
            "</dl>"
        ),
    },

    ("ml", "item_analysis"): {
        "title": "Understanding Item Analysis",
        "content": (
            "<dl>"
            "<dt>What is item analysis?</dt>"
            "<dd>Evaluates the quality of individual items in a test or survey. Identifies "
            "which items are good discriminators and which should be revised or removed.</dd>"
            "<dt>Key metrics</dt>"
            "<dd><strong>Cronbach's α</strong>: Internal consistency (reliability). ≥ 0.70 "
            "for research, ≥ 0.80 for applied use. <strong>Item-total correlation</strong>: "
            "How well each item correlates with the total score. &lt; 0.3 is weak.</dd>"
            "<dt>Difficulty and discrimination</dt>"
            "<dd>Difficulty: Proportion answering correctly (easy = high, hard = low). "
            "Discrimination: How well the item separates high and low performers.</dd>"
            "<dt>α if item deleted</dt>"
            "<dd>If removing an item increases Cronbach's α, the item is hurting reliability "
            "and should be revised. This is the most actionable metric.</dd>"
            "</dl>"
        ),
    },
})


# ═══════════════════════════════════════════════════════════════════════════
# Reliability Module
# ═══════════════════════════════════════════════════════════════════════════

_extend({
    ("reliability", "weibull"): {
        "title": "Understanding Weibull Reliability Analysis",
        "content": (
            "<dl>"
            "<dt>What is Weibull reliability analysis?</dt>"
            "<dd>The workhorse of reliability engineering. Fits time-to-failure data to a "
            "Weibull distribution to characterize failure patterns, estimate B-lives, and "
            "predict future failures.</dd>"
            "<dt>Shape parameter (β)</dt>"
            "<dd>β &lt; 1: Decreasing failure rate (infant mortality — burn-in helps). "
            "β = 1: Constant failure rate (random failures — no wear-out). "
            "β &gt; 1: Increasing failure rate (wear-out — scheduled replacement helps).</dd>"
            "<dt>Scale parameter (η)</dt>"
            "<dd>The characteristic life — 63.2% of units fail by time η. It sets the "
            "time scale of the failure process.</dd>"
            "<dt>Censored data handling</dt>"
            "<dd>Units still running (right-censored) provide partial information — they "
            "survived at least that long. Ignoring them biases results toward shorter lives.</dd>"
            "</dl>"
        ),
    },

    ("reliability", "lognormal"): {
        "title": "Understanding Lognormal Reliability",
        "content": (
            "<dl>"
            "<dt>When to use lognormal?</dt>"
            "<dd>When failure times are right-skewed and log(time) is approximately normal. "
            "Common for fatigue, crack growth, and degradation processes where failure "
            "results from multiplicative damage accumulation.</dd>"
            "<dt>Parameters</dt>"
            "<dd><strong>μ</strong>: Mean of log(time) — sets the median life. "
            "<strong>σ</strong>: Std dev of log(time) — controls the spread of failure times.</dd>"
            "<dt>Lognormal vs Weibull</dt>"
            "<dd>Lognormal has a non-monotonic hazard (increases then decreases). Weibull "
            "has a monotonic hazard. Use probability plots and goodness-of-fit tests to "
            "choose between them.</dd>"
            "<dt>Interpretation</dt>"
            "<dd>The median life (e^μ) is the time by which 50% of units fail. The spread "
            "σ determines how much variation exists in failure times.</dd>"
            "</dl>"
        ),
    },

    ("reliability", "exponential"): {
        "title": "Understanding Exponential Reliability",
        "content": (
            "<dl>"
            "<dt>When to use exponential?</dt>"
            "<dd>When the failure rate is constant over time — failures are purely random "
            "with no wear-out or break-in pattern. Equivalent to Weibull with β = 1.</dd>"
            "<dt>The memoryless property</dt>"
            "<dd>A used item has the same remaining life distribution as a new one. This is "
            "only realistic for electronic components and some truly random failure modes.</dd>"
            "<dt>MTBF</dt>"
            "<dd>Mean Time Between Failures = 1/λ (failure rate). The single parameter of "
            "the exponential distribution. MTBF is the expected life, not the median.</dd>"
            "<dt>Caution</dt>"
            "<dd>The constant failure rate assumption is rarely true for mechanical systems. "
            "Always verify with a probability plot or compare against Weibull before "
            "defaulting to exponential.</dd>"
            "</dl>"
        ),
    },

    ("reliability", "kaplan_meier"): {
        "title": "Understanding Kaplan-Meier Reliability",
        "content": (
            "<dl>"
            "<dt>What is Kaplan-Meier in reliability?</dt>"
            "<dd>A nonparametric reliability estimate — no distributional assumption needed. "
            "Estimates the survival function directly from failure and censoring times.</dd>"
            "<dt>Advantages</dt>"
            "<dd>No need to choose Weibull vs lognormal vs exponential. The KM estimator "
            "lets the data speak. Especially valuable when the failure distribution is "
            "unknown or complex.</dd>"
            "<dt>Comparison of populations</dt>"
            "<dd>Log-rank test compares survival curves between groups (e.g., two suppliers, "
            "two designs). Does not assume a specific distribution.</dd>"
            "<dt>Limitations</dt>"
            "<dd>Cannot extrapolate beyond the data range. For prediction at time horizons "
            "beyond the study, parametric models (Weibull, lognormal) are needed.</dd>"
            "</dl>"
        ),
    },

    ("reliability", "reliability_test_plan"): {
        "title": "Understanding Reliability Test Planning",
        "content": (
            "<dl>"
            "<dt>What is a reliability test plan?</dt>"
            "<dd>Determines sample size and test duration needed to demonstrate a reliability "
            "target with specified confidence. Balances testing cost against evidence quality.</dd>"
            "<dt>Test-to-failure vs time-terminated</dt>"
            "<dd>Test-to-failure: Run until all units fail (most information per unit). "
            "Time-terminated: Stop at a fixed time (known schedule, but censored data).</dd>"
            "<dt>Demonstration testing</dt>"
            "<dd>Tests n units for time T with no failures to demonstrate reliability R "
            "with confidence C. The formula: n = log(1−C) / log(R) (for exponential).</dd>"
            "<dt>Acceleration</dt>"
            "<dd>Run at elevated stress to cause failures faster. An acceleration factor "
            "relates test time to field time. Requires validated acceleration models.</dd>"
            "</dl>"
        ),
    },

    ("reliability", "distribution_id"): {
        "title": "Understanding Reliability Distribution Identification",
        "content": (
            "<dl>"
            "<dt>What is distribution identification?</dt>"
            "<dd>Systematically compares how well different lifetime distributions (Weibull, "
            "lognormal, exponential, normal, loglogistic) fit your failure data. Selects "
            "the best-fitting model for prediction.</dd>"
            "<dt>How to compare</dt>"
            "<dd>Probability plots (linearity indicates good fit), Anderson-Darling statistics "
            "(lower = better), and AIC/BIC for model comparison.</dd>"
            "<dt>Why does the distribution matter?</dt>"
            "<dd>Different distributions imply different failure patterns. Weibull with β &gt; 1 "
            "means scheduled replacement helps. Lognormal means median life is the key metric. "
            "Wrong distribution → wrong decisions.</dd>"
            "<dt>Multiple failure modes</dt>"
            "<dd>If data doesn't fit any single distribution well, consider mixture models — "
            "multiple failure mechanisms may be operating simultaneously.</dd>"
            "</dl>"
        ),
    },

    ("reliability", "accelerated_life"): {
        "title": "Understanding Accelerated Life Testing",
        "content": (
            "<dl>"
            "<dt>What is ALT?</dt>"
            "<dd>Accelerated Life Testing applies elevated stress (temperature, voltage, "
            "load) to induce failures faster, then extrapolates to normal operating "
            "conditions using physics-based models.</dd>"
            "<dt>Acceleration models</dt>"
            "<dd><strong>Arrhenius</strong>: Temperature-accelerated (chemical degradation). "
            "<strong>Inverse Power Law</strong>: Voltage/load-accelerated. "
            "<strong>Eyring</strong>: Temperature + humidity combined.</dd>"
            "<dt>Key assumption</dt>"
            "<dd>The failure mechanism must be the same at elevated and normal stress. If "
            "higher stress triggers a different failure mode, extrapolation is invalid.</dd>"
            "<dt>Planning</dt>"
            "<dd>At least 3 stress levels with adequate failures at each. The lowest stress "
            "should be close to normal use conditions for reliable extrapolation.</dd>"
            "</dl>"
        ),
    },

    ("reliability", "repairable_systems"): {
        "title": "Understanding Repairable Systems Analysis",
        "content": (
            "<dl>"
            "<dt>What are repairable systems?</dt>"
            "<dd>Systems that can be restored to operation after failure (unlike components "
            "that are replaced). Analysis focuses on failure patterns over the system's "
            "life — are failures getting more or less frequent?</dd>"
            "<dt>NHPP models</dt>"
            "<dd>Non-Homogeneous Poisson Process models capture changing failure rates. "
            "The power law (Crow-AMSAA) model is most common: m(t) = λ·t^β.</dd>"
            "<dt>Interpreting β</dt>"
            "<dd>β &lt; 1: Reliability growth (failures decreasing). β = 1: Constant rate. "
            "β &gt; 1: Reliability deterioration (failures increasing — system wearing out).</dd>"
            "<dt>MCF (Mean Cumulative Function)</dt>"
            "<dd>The nonparametric alternative — plots cumulative failures vs time. "
            "Concave = improving. Straight = constant. Convex = deteriorating.</dd>"
            "</dl>"
        ),
    },

    ("reliability", "warranty"): {
        "title": "Understanding Warranty Analysis",
        "content": (
            "<dl>"
            "<dt>What is warranty analysis?</dt>"
            "<dd>Predicts future warranty claims from current claim data and sales history. "
            "Accounts for reporting delays, IBNR (Incurred But Not Reported) claims, "
            "and fleet age distributions.</dd>"
            "<dt>The warranty data challenge</dt>"
            "<dd>Not all failures are reported during the warranty period — some occur "
            "after warranty expires, some aren't claimed. Analysis must adjust for these "
            "truncation and censoring effects.</dd>"
            "<dt>Forecasting</dt>"
            "<dd>Fit a lifetime distribution to warranty claims, estimate the total failure "
            "fraction, and project future claims by birth month (production cohort).</dd>"
            "<dt>Cost estimation</dt>"
            "<dd>Claims × average repair cost × remaining fleet exposure = financial reserve. "
            "This drives warranty accrual accounting and pricing decisions.</dd>"
            "</dl>"
        ),
    },

    ("reliability", "competing_risks"): {
        "title": "Understanding Competing Risks Analysis",
        "content": (
            "<dl>"
            "<dt>What are competing risks?</dt>"
            "<dd>When a unit can fail from multiple independent failure modes, and the "
            "first one to occur is what we observe. Each mode competes to be the cause "
            "of failure.</dd>"
            "<dt>Why separate analysis?</dt>"
            "<dd>Standard reliability analysis ignores the failure mode. Competing risks "
            "decomposes the overall reliability into mode-specific reliabilities, enabling "
            "targeted improvements.</dd>"
            "<dt>Cumulative incidence function</dt>"
            "<dd>The probability of failing from a specific mode by time t, accounting for "
            "the competing modes. Unlike mode-specific KM, it correctly accounts for the "
            "competition.</dd>"
            "<dt>Improvement prioritization</dt>"
            "<dd>Eliminating the dominant failure mode reveals the next mode's contribution. "
            "This guides R&D investment to the highest-impact improvement.</dd>"
            "</dl>"
        ),
    },
})


# ═══════════════════════════════════════════════════════════════════════════
# Visualization Module
# ═══════════════════════════════════════════════════════════════════════════

_extend({
    ("viz", "histogram"): {
        "title": "Understanding Histograms",
        "content": (
            "<dl>"
            "<dt>What is a histogram?</dt>"
            "<dd>A bar chart showing the distribution of continuous data by dividing values "
            "into bins and counting how many fall in each. The most fundamental tool for "
            "understanding data shape.</dd>"
            "<dt>Bin width matters</dt>"
            "<dd>Too few bins hide structure; too many create noise. The Freedman-Diaconis "
            "rule (bin width = 2·IQR·n^(−1/3)) usually works well.</dd>"
            "<dt>What to look for</dt>"
            "<dd>Shape (symmetric, skewed, bimodal), center (mean/median), spread, outliers, "
            "and gaps (which may indicate mixed populations).</dd>"
            "<dt>Histogram vs density plot</dt>"
            "<dd>Histograms are discrete (binned); density plots are smooth. Density plots "
            "avoid the bin width problem but are harder to interpret for small samples.</dd>"
            "</dl>"
        ),
    },

    ("viz", "boxplot"): {
        "title": "Understanding Boxplots",
        "content": (
            "<dl>"
            "<dt>What is a boxplot?</dt>"
            "<dd>A five-number summary visualization: minimum, Q1, median, Q3, maximum. "
            "The box spans the IQR; whiskers extend to the most extreme non-outlier values.</dd>"
            "<dt>Reading outliers</dt>"
            "<dd>Points beyond 1.5×IQR from the box are marked as outliers. They may be "
            "data errors, special causes, or genuine extreme values — always investigate.</dd>"
            "<dt>Comparing groups</dt>"
            "<dd>Side-by-side boxplots compare distributions across groups. Non-overlapping "
            "boxes suggest a significant difference. Overlapping medians with different "
            "spreads indicate a variance difference.</dd>"
            "<dt>Limitations</dt>"
            "<dd>Boxplots hide multimodality and detailed shape information. For small samples "
            "or complex distributions, add individual data points or use violin plots.</dd>"
            "</dl>"
        ),
    },

    ("viz", "scatter"): {
        "title": "Understanding Scatter Plots",
        "content": (
            "<dl>"
            "<dt>What is a scatter plot?</dt>"
            "<dd>Plots two continuous variables against each other to reveal relationships, "
            "clusters, outliers, and patterns.</dd>"
            "<dt>What to look for</dt>"
            "<dd><strong>Direction</strong>: Positive (both increase) or negative (one decreases). "
            "<strong>Strength</strong>: How tightly points cluster around a pattern. "
            "<strong>Form</strong>: Linear, curved, or no pattern. "
            "<strong>Outliers</strong>: Points far from the main cloud.</dd>"
            "<dt>Adding regression lines</dt>"
            "<dd>A trend line quantifies the relationship. But always check the scatter first — "
            "a regression line through a nonlinear pattern or clustered data is misleading.</dd>"
            "<dt>Overplotting</dt>"
            "<dd>With many points, individual values obscure each other. Use transparency, "
            "jitter, or 2D density (hexbin) plots for large datasets.</dd>"
            "</dl>"
        ),
    },

    ("viz", "heatmap"): {
        "title": "Understanding Heatmaps",
        "content": (
            "<dl>"
            "<dt>What is a heatmap?</dt>"
            "<dd>A matrix of values encoded as colors. Reveals patterns in two-dimensional "
            "data — correlations, clusters, and anomalies become visible as color patterns.</dd>"
            "<dt>Color scale</dt>"
            "<dd>Sequential (low-to-high) for magnitude. Diverging (two colors meeting at "
            "a midpoint) for deviations from a reference. Choose colorblind-safe palettes.</dd>"
            "<dt>Correlation heatmaps</dt>"
            "<dd>Show pairwise correlations between all variables. Hot spots indicate strong "
            "relationships. Blocks of correlated variables suggest underlying factors.</dd>"
            "<dt>Ordering matters</dt>"
            "<dd>Clustering rows and columns (dendrogram) groups similar items together, "
            "making patterns much easier to spot than random ordering.</dd>"
            "</dl>"
        ),
    },

    ("viz", "pareto"): {
        "title": "Understanding Pareto Charts",
        "content": (
            "<dl>"
            "<dt>What is a Pareto chart?</dt>"
            "<dd>A bar chart ordered by frequency plus a cumulative percentage line. Based "
            "on the Pareto principle: ~80% of effects come from ~20% of causes.</dd>"
            "<dt>How to read it</dt>"
            "<dd>Bars show individual category counts. The line shows cumulative percentage. "
            "Focus improvement efforts on the vital few (leftmost bars) that account for "
            "most of the total.</dd>"
            "<dt>Before/after comparison</dt>"
            "<dd>Create a Pareto chart before and after improvement. The rank order may "
            "change — previous top causes may drop, revealing new priorities.</dd>"
            "<dt>Stratified Pareto</dt>"
            "<dd>Break down by shift, machine, operator, or time period to find the root "
            "cause behind the top Pareto category.</dd>"
            "</dl>"
        ),
    },

    ("viz", "matrix"): {
        "title": "Understanding Matrix Plots",
        "content": (
            "<dl>"
            "<dt>What is a matrix plot?</dt>"
            "<dd>A grid of scatter plots for all pairs of variables. The fastest way to "
            "explore pairwise relationships in multivariate data.</dd>"
            "<dt>What to look for</dt>"
            "<dd>Strong linear patterns (potential predictors), clusters (subpopulations), "
            "outliers (unusual observations), and nonlinear relationships.</dd>"
            "<dt>Diagonal</dt>"
            "<dd>The diagonal shows each variable's own distribution (histogram or density). "
            "This combines univariate and bivariate exploration in one display.</dd>"
            "<dt>Practical use</dt>"
            "<dd>First step in regression analysis — identifies which predictors have "
            "relationships with the response and with each other (multicollinearity).</dd>"
            "</dl>"
        ),
    },

    ("viz", "timeseries"): {
        "title": "Understanding Time Series Plots",
        "content": (
            "<dl>"
            "<dt>What is a time series plot?</dt>"
            "<dd>Data plotted in time order — the most natural way to see process behavior "
            "over time. Reveals trends, cycles, shifts, and outliers.</dd>"
            "<dt>What to look for</dt>"
            "<dd>Trends (sustained increase/decrease), seasonality (repeating patterns), "
            "level shifts (sudden jumps), and increasing/decreasing spread.</dd>"
            "<dt>Vs control charts</dt>"
            "<dd>Time series plots show raw data. Control charts add statistical limits "
            "that distinguish common from special cause variation. Use time series for "
            "exploration, control charts for monitoring.</dd>"
            "<dt>Multiple series</dt>"
            "<dd>Overlay related variables to spot correlations, leads/lags, or common "
            "patterns. Normalize scales if comparing variables with different units.</dd>"
            "</dl>"
        ),
    },

    ("viz", "probability"): {
        "title": "Understanding Probability Plots",
        "content": (
            "<dl>"
            "<dt>What is a probability plot?</dt>"
            "<dd>Plots ordered data against theoretical quantiles from a reference "
            "distribution. If the data follows that distribution, the points form a "
            "straight line.</dd>"
            "<dt>Reading the plot</dt>"
            "<dd>Straight line = data matches the distribution. S-curve = heavy tails. "
            "Concave/convex departures = skewness. Gaps or clusters indicate data quality "
            "issues or mixed populations.</dd>"
            "<dt>Normal probability plot (Q-Q plot)</dt>"
            "<dd>The most common variant. Points above the line at the right end indicate "
            "right skew (heavy upper tail). Points below at both ends indicate light tails.</dd>"
            "<dt>Beyond normality</dt>"
            "<dd>Probability plots can use any reference distribution — Weibull, lognormal, "
            "exponential. The distribution giving the straightest line is the best fit.</dd>"
            "</dl>"
        ),
    },

    ("viz", "individual_value_plot"): {
        "title": "Understanding Individual Value Plots",
        "content": (
            "<dl>"
            "<dt>What is an individual value plot?</dt>"
            "<dd>Shows every data point for each group, with the mean and/or median marked. "
            "Unlike boxplots, no information is hidden — you see all the data.</dd>"
            "<dt>When to use</dt>"
            "<dd>Small to moderate samples where every observation matters. Boxplots can "
            "hide important patterns (bimodality, gaps) in small datasets.</dd>"
            "<dt>Jittering</dt>"
            "<dd>Points at the same value are offset horizontally to prevent overplotting. "
            "This reveals the density of the data at each value.</dd>"
            "<dt>Group comparison</dt>"
            "<dd>Side-by-side individual value plots show both the central tendency and "
            "the full spread of each group, making it easy to spot asymmetry and outliers.</dd>"
            "</dl>"
        ),
    },

    ("viz", "interval_plot"): {
        "title": "Understanding Interval Plots",
        "content": (
            "<dl>"
            "<dt>What is an interval plot?</dt>"
            "<dd>Shows the mean and confidence interval for each group. A clean visualization "
            "for comparing group means with their precision.</dd>"
            "<dt>Reading the intervals</dt>"
            "<dd>Non-overlapping intervals suggest a significant difference. However, "
            "overlapping intervals do NOT guarantee no difference — formal testing "
            "(ANOVA, t-test) is needed.</dd>"
            "<dt>CI width</dt>"
            "<dd>Wider intervals indicate less precision (smaller samples, more variability). "
            "Narrower intervals indicate more precise mean estimates.</dd>"
            "<dt>Best paired with</dt>"
            "<dd>Individual value plots (to see the raw data) and ANOVA/t-test results "
            "(for formal significance). Interval plots alone can be misleading.</dd>"
            "</dl>"
        ),
    },

    ("viz", "dotplot"): {
        "title": "Understanding Dot Plots",
        "content": (
            "<dl>"
            "<dt>What is a dot plot?</dt>"
            "<dd>Each observation is a dot, stacked vertically at its value. Shows the "
            "distribution shape without binning (unlike histograms). Best for small to "
            "moderate samples.</dd>"
            "<dt>Dot plot vs histogram</dt>"
            "<dd>Dot plots show every value exactly — no information loss from binning. "
            "Histograms are better for large datasets where individual points would overlap.</dd>"
            "<dt>Grouped dot plots</dt>"
            "<dd>Side-by-side grouped dot plots compare distributions across categories. "
            "Each dot is visible, making outliers and patterns immediately apparent.</dd>"
            "<dt>When to use</dt>"
            "<dd>Exploratory analysis with n &lt; 100, comparing a few groups, or when "
            "exact values matter (e.g., measurement data where rounding patterns "
            "reveal gage resolution issues).</dd>"
            "</dl>"
        ),
    },

    ("viz", "bubble"): {
        "title": "Understanding Bubble Charts",
        "content": (
            "<dl>"
            "<dt>What is a bubble chart?</dt>"
            "<dd>A scatter plot where a third variable is encoded as bubble size. Shows "
            "three dimensions of data in a 2D space.</dd>"
            "<dt>When to use</dt>"
            "<dd>When you want to visualize three continuous variables simultaneously — "
            "e.g., defect rate (x) vs cost (y) vs production volume (size).</dd>"
            "<dt>Perception issues</dt>"
            "<dd>Humans perceive area poorly. Encode the most important variable on the "
            "axes, not as size. Size should represent a secondary variable.</dd>"
            "<dt>Alternative</dt>"
            "<dd>For more than 3 variables, consider parallel coordinates or matrix plots "
            "instead of trying to encode too much in a single chart.</dd>"
            "</dl>"
        ),
    },

    ("viz", "parallel_coordinates"): {
        "title": "Understanding Parallel Coordinates",
        "content": (
            "<dl>"
            "<dt>What are parallel coordinates?</dt>"
            "<dd>Each variable gets a vertical axis; each observation is a line connecting "
            "its values across axes. Reveals multivariate patterns, clusters, and outliers "
            "in high-dimensional data.</dd>"
            "<dt>What to look for</dt>"
            "<dd>Lines that cross: negative correlation between adjacent axes. Lines that "
            "don't cross: positive correlation. Clusters of parallel lines: groups with "
            "similar profiles.</dd>"
            "<dt>Axis ordering</dt>"
            "<dd>The order of axes affects readability. Place related variables adjacent. "
            "Reordering can reveal hidden patterns.</dd>"
            "<dt>Use with clustering</dt>"
            "<dd>Color-code lines by cluster membership to see how clusters differ across "
            "all variables simultaneously.</dd>"
            "</dl>"
        ),
    },

    ("viz", "contour"): {
        "title": "Understanding Contour Plots",
        "content": (
            "<dl>"
            "<dt>What is a contour plot?</dt>"
            "<dd>Shows a 3D surface as 2D lines of equal value (like a topographic map). "
            "Essential for visualizing response surfaces in DOE.</dd>"
            "<dt>Reading contours</dt>"
            "<dd>Closely spaced lines = steep surface (rapid change). Widely spaced = flat "
            "region. Closed loops = peaks or valleys (optimal regions).</dd>"
            "<dt>In DOE context</dt>"
            "<dd>Contour plots show how two factors jointly affect a response. The optimal "
            "region (target contour) guides factor settings for process optimization.</dd>"
            "<dt>Filled vs line contours</dt>"
            "<dd>Filled contours use color gradients for easier reading. Line contours "
            "are better for overlay with data points.</dd>"
            "</dl>"
        ),
    },

    ("viz", "surface_3d"): {
        "title": "Understanding 3D Surface Plots",
        "content": (
            "<dl>"
            "<dt>What is a 3D surface plot?</dt>"
            "<dd>A three-dimensional visualization of how a response varies across two "
            "factors. Useful for seeing the overall shape of a response surface.</dd>"
            "<dt>When to use</dt>"
            "<dd>DOE response surface exploration, regression model visualization, "
            "probability density visualization. Best for presentations — contour plots "
            "are often more precise for analysis.</dd>"
            "<dt>Interaction</dt>"
            "<dd>Rotate the surface interactively to find the best viewing angle. "
            "Saddle shapes indicate interactions between factors.</dd>"
            "<dt>Limitations</dt>"
            "<dd>Can hide detail in the back. Perspective distorts distances. Always "
            "supplement with a contour plot for precise interpretation.</dd>"
            "</dl>"
        ),
    },

    ("viz", "contour_overlay"): {
        "title": "Understanding Contour Overlay Plots",
        "content": (
            "<dl>"
            "<dt>What is a contour overlay?</dt>"
            "<dd>Overlays contour plots for multiple responses on the same axes to find "
            "factor settings that satisfy all requirements simultaneously.</dd>"
            "<dt>The feasible region</dt>"
            "<dd>The area where all response contours meet their specifications. If no "
            "feasible region exists, the specs are mutually incompatible.</dd>"
            "<dt>Use in optimization</dt>"
            "<dd>After fitting response surfaces in DOE, the overlay plot shows the "
            "operating window — the range of factor settings that produce acceptable "
            "results on all responses.</dd>"
            "<dt>Multi-response optimization</dt>"
            "<dd>The desirability function approach quantifies how well each point "
            "satisfies all requirements. The overlay plot visualizes this trade-off.</dd>"
            "</dl>"
        ),
    },

    ("viz", "mosaic"): {
        "title": "Understanding Mosaic Plots",
        "content": (
            "<dl>"
            "<dt>What is a mosaic plot?</dt>"
            "<dd>A visual representation of a contingency table where tile areas are "
            "proportional to cell frequencies. Shows both marginal and conditional "
            "distributions simultaneously.</dd>"
            "<dt>Reading the plot</dt>"
            "<dd>Column widths show marginal proportions of one variable. Row heights "
            "within each column show conditional proportions of the other. Equal heights "
            "across columns = independence.</dd>"
            "<dt>Residual shading</dt>"
            "<dd>Tiles colored by Pearson residuals highlight cells that deviate from "
            "independence. Blue = more than expected. Red = fewer than expected.</dd>"
            "<dt>When to use</dt>"
            "<dd>Visualizing categorical associations before or alongside chi-square tests. "
            "Reveals which specific cells drive the overall association.</dd>"
            "</dl>"
        ),
    },

    ("viz", "bayes_spc_capability"): {
        "title": "Understanding Bayesian SPC Capability Visualization",
        "content": (
            "<dl>"
            "<dt>What does this show?</dt>"
            "<dd>Bayesian posterior distribution of capability indices (Cpk, Ppk) rather "
            "than single point estimates. Shows the full uncertainty in capability assessment.</dd>"
            "<dt>Why Bayesian?</dt>"
            "<dd>Traditional Cpk is a point estimate — it hides uncertainty. The posterior "
            "distribution shows the probability that true Cpk exceeds any threshold.</dd>"
            "<dt>Decision-making</dt>"
            "<dd>P(Cpk &gt; 1.33) = 0.95 is a much stronger statement than 'estimated Cpk = 1.5' "
            "because it accounts for sample size and estimation uncertainty.</dd>"
            "<dt>Prior influence</dt>"
            "<dd>With small samples, the prior matters. The analysis shows how the posterior "
            "shifts from prior to posterior as data accumulates.</dd>"
            "</dl>"
        ),
    },

    ("viz", "bayes_spc_changepoint"): {
        "title": "Understanding Bayesian SPC Changepoint Visualization",
        "content": (
            "<dl>"
            "<dt>What does this show?</dt>"
            "<dd>Posterior probability of a changepoint at each time position. Shows where "
            "the process most likely shifted, with uncertainty quantification.</dd>"
            "<dt>Posterior probability trace</dt>"
            "<dd>The height at each time point shows the probability that a change occurred "
            "there. Peaks indicate likely changepoints. Multiple peaks suggest multiple shifts.</dd>"
            "<dt>Vs classical changepoint</dt>"
            "<dd>Classical methods give point estimates. Bayesian methods give full "
            "probability distributions over changepoint locations and magnitudes.</dd>"
            "<dt>Segment analysis</dt>"
            "<dd>The data is partitioned at the most probable changepoint(s), and separate "
            "statistics are computed for each segment.</dd>"
            "</dl>"
        ),
    },

    ("viz", "bayes_spc_control"): {
        "title": "Understanding Bayesian SPC Control Visualization",
        "content": (
            "<dl>"
            "<dt>What does this show?</dt>"
            "<dd>Bayesian control limits derived from posterior predictive distributions "
            "rather than classical ±3σ limits. Valid for non-normal data and small samples.</dd>"
            "<dt>Predictive limits</dt>"
            "<dd>Based on where future observations are expected to fall given the data "
            "seen so far. They contract as more data arrives (uncertainty decreases).</dd>"
            "<dt>Advantages</dt>"
            "<dd>No normality assumption needed. Limits are honest about uncertainty with "
            "small samples — they start wide and narrow as evidence accumulates.</dd>"
            "<dt>Posterior predictive check</dt>"
            "<dd>Compares the observed data distribution against the posterior predictive "
            "distribution. Discrepancies indicate model misfit.</dd>"
            "</dl>"
        ),
    },

    ("viz", "bayes_spc_acceptance"): {
        "title": "Understanding Bayesian SPC Acceptance Visualization",
        "content": (
            "<dl>"
            "<dt>What does this show?</dt>"
            "<dd>Bayesian acceptance probability — the posterior probability that a lot "
            "meets quality requirements, given the sample data.</dd>"
            "<dt>Vs classical acceptance sampling</dt>"
            "<dd>Classical: accept if defects ≤ c (hard boundary). Bayesian: compute "
            "P(lot meets spec | data) and decide based on the probability.</dd>"
            "<dt>Prior information</dt>"
            "<dd>Bayesian acceptance can incorporate prior knowledge about the supplier's "
            "historical quality, updating as new lot data arrives.</dd>"
            "<dt>Risk quantification</dt>"
            "<dd>Directly computes the probability of accepting a bad lot (consumer's risk) "
            "and rejecting a good lot (producer's risk) for any decision threshold.</dd>"
            "</dl>"
        ),
    },
})


# ═══════════════════════════════════════════════════════════════════════════
# Simulation Module
# ═══════════════════════════════════════════════════════════════════════════

_extend({
    ("simulation", "tolerance_stackup"): {
        "title": "Understanding Tolerance Stackup Analysis",
        "content": (
            "<dl>"
            "<dt>What is tolerance stackup?</dt>"
            "<dd>Predicts the variation of an assembly dimension from the tolerances of "
            "its component parts. Answers: 'If each part is within spec, will the "
            "assembly be within spec?'</dd>"
            "<dt>Worst case vs RSS vs Monte Carlo</dt>"
            "<dd><strong>Worst case</strong>: Assumes all parts at their limits simultaneously "
            "(very conservative). <strong>RSS</strong>: Root Sum of Squares — assumes "
            "independent, centered, normal distributions. <strong>Monte Carlo</strong>: "
            "Simulates actual distributions (most realistic).</dd>"
            "<dt>Why Monte Carlo?</dt>"
            "<dd>Real parts aren't perfectly centered or normally distributed. Monte Carlo "
            "uses actual or fitted distributions, handling skewness, truncation, and "
            "correlation between dimensions.</dd>"
            "<dt>What-if exploration</dt>"
            "<dd>Adjust component tolerances to see which ones most affect assembly variation. "
            "This guides tolerance allocation — tighten where it matters, loosen where it doesn't.</dd>"
            "</dl>"
        ),
    },

    ("simulation", "variance_propagation"): {
        "title": "Understanding Variance Propagation",
        "content": (
            "<dl>"
            "<dt>What is variance propagation?</dt>"
            "<dd>Estimates how input variable uncertainties combine to affect an output "
            "through a mathematical model. Based on the delta method (Taylor series "
            "approximation) or Monte Carlo simulation.</dd>"
            "<dt>When to use</dt>"
            "<dd>When you have a transfer function Y = f(X₁, X₂, ...) and know the "
            "distributions of the inputs. Determines the output distribution without "
            "running physical experiments.</dd>"
            "<dt>Sensitivity analysis</dt>"
            "<dd>Partial derivatives (or Monte Carlo sensitivities) show which inputs "
            "contribute most to output variance. Focus improvement on the dominant "
            "contributors.</dd>"
            "<dt>Delta method vs Monte Carlo</dt>"
            "<dd>Delta method is fast but assumes linearity and normality. Monte Carlo "
            "handles any function and distribution but needs more computation.</dd>"
            "</dl>"
        ),
    },
})


# ═══════════════════════════════════════════════════════════════════════════
# D-Type Module (centralized copies of existing education)
# ═══════════════════════════════════════════════════════════════════════════

_extend({
    ("d_type", "d_chart"): {
        "title": "Understanding the D-Chart",
        "content": (
            "<dl>"
            "<dt>What is a D-Chart?</dt>"
            "<dd>A Divergence Chart monitors how much each factor level's distribution "
            "differs from the overall process distribution over time. Unlike traditional "
            "control charts that track means or ranges, it tracks <em>distributional shape</em> "
            "changes — catching shifts in spread, skew, or tails that mean/range charts miss.</dd>"
            "<dt>What is JSD (Jensen-Shannon Divergence)?</dt>"
            "<dd>A symmetric, bounded measure of how different two probability distributions "
            "are. JSD = 0 means identical; JSD = 1 (in bits) means completely different.</dd>"
            "<dt>What is the Noise Floor?</dt>"
            "<dd>The expected JSD from random sampling alone. Points above the noise floor "
            "indicate real, non-random divergence.</dd>"
            "<dt>What is the Information Score?</dt>"
            "<dd>A cumulative, recency-weighted sum of excess JSD above noise floor. Higher "
            "scores mean the factor consistently produces a different distribution.</dd>"
            "</dl>"
        ),
    },

    ("d_type", "d_cpk"): {
        "title": "Understanding D-Cpk",
        "content": (
            "<dl>"
            "<dt>What is D-Cpk?</dt>"
            "<dd>Attributes capability differences to specific factor levels. Standard Cpk "
            "tells you <em>if</em> the process is capable. D-Cpk tells you <em>which factors "
            "are dragging capability down</em>.</dd>"
            "<dt>Defect Efficiency</dt>"
            "<dd>The fraction of a factor's divergence that occurs in spec tails (where defects "
            "happen). High efficiency means the factor directly creates defects.</dd>"
            "<dt>Counterfactual Cpk</dt>"
            "<dd>The Cpk the process would have if that factor level were removed. The gap "
            "prioritizes improvement actions.</dd>"
            "<dt>Interpretation</dt>"
            "<dd>Factor above noise with high defect efficiency → priority target. Factor above "
            "noise with low defect efficiency → changes distribution but not defect rate.</dd>"
            "</dl>"
        ),
    },

    ("d_type", "d_nonnorm"): {
        "title": "Understanding D-NonNorm",
        "content": (
            "<dl>"
            "<dt>Why non-normal capability?</dt>"
            "<dd>Traditional Cpk assumes normality. Many real processes are skewed or bounded. "
            "KDE-based Ppk gives an honest capability estimate without distributional assumptions.</dd>"
            "<dt>Normality Penalty</dt>"
            "<dd>The difference between normal-assumption Ppk and KDE-based Ppk. Positive "
            "penalty means normal overstates capability.</dd>"
            "<dt>PPM comparison</dt>"
            "<dd>Compare KDE vs Normal PPM to see the real-world defect rate difference.</dd>"
            "<dt>When to use</dt>"
            "<dd>Any time you suspect non-normality. Always check the normality penalty — "
            "if it's large, normal-based capability is misleading.</dd>"
            "</dl>"
        ),
    },

    ("d_type", "d_equiv"): {
        "title": "Understanding D-Equiv",
        "content": (
            "<dl>"
            "<dt>What does D-Equiv test?</dt>"
            "<dd>Whether batches produce the same distribution of output — not just the same "
            "mean. Catches differences in spread, shape, and tails via JSD comparison.</dd>"
            "<dt>How is equivalence decided?</dt>"
            "<dd>Each batch's density is compared to a reference via JSD. Below the noise "
            "floor threshold = equivalent.</dd>"
            "<dt>Pairwise JSD heatmap</dt>"
            "<dd>Shows which batches are similar (cool colors) vs different (hot colors). "
            "Clusters of similar batches may indicate shared process conditions.</dd>"
            "<dt>Reference batch</dt>"
            "<dd>The 'known good' batch. All others are compared against it.</dd>"
            "</dl>"
        ),
    },

    ("d_type", "d_sig"): {
        "title": "Understanding D-Sig (Process Signatures)",
        "content": (
            "<dl>"
            "<dt>What is a Process Signature?</dt>"
            "<dd>A time-ordered measurement profile (e.g., temperature over a cycle). D-Sig "
            "compares profiles across groups to find where and how they diverge.</dd>"
            "<dt>Peak Divergence</dt>"
            "<dd>The time point where a group's distribution differs most from the reference — "
            "where to focus investigation.</dd>"
            "<dt>Interpretation</dt>"
            "<dd>Flat, low JSD trace: signatures match. Spike: localized divergence. "
            "Sustained elevation: fundamentally different process mode.</dd>"
            "<dt>Applications</dt>"
            "<dd>Batch process monitoring, forming press analysis, thermal profile comparison.</dd>"
            "</dl>"
        ),
    },

    ("d_type", "d_multi"): {
        "title": "Understanding D-Multi (Multivariate Capability)",
        "content": (
            "<dl>"
            "<dt>Why multivariate?</dt>"
            "<dd>Multiple correlated quality characteristics must be assessed jointly. A part "
            "can pass every individual spec but fail jointly.</dd>"
            "<dt>PCA and MCpk</dt>"
            "<dd>PCA rotates correlated variables into uncorrelated components. KDE-based "
            "capability on each PC gives MCpk — the minimum is the bottleneck.</dd>"
            "<dt>Hotelling's T²</dt>"
            "<dd>Multivariate distance from center. Points above UCL are multivariate "
            "outliers even if normal on individual variables.</dd>"
            "<dt>Interpreting MCpk</dt>"
            "<dd>≥ 1.33: Jointly capable. 1.0–1.33: Marginal — find the weak PC. "
            "&lt; 1.0: Not jointly capable.</dd>"
            "</dl>"
        ),
    },
})


# ═══════════════════════════════════════════════════════════════════════════
# Bayesian Module (centralized copies — originals remain in bayesian.py)
# ═══════════════════════════════════════════════════════════════════════════

_extend({
    ("bayesian", "bayes_regression"): {
        "title": "Understanding Bayesian Regression",
        "content": (
            "<dl>"
            "<dt>What is Bayesian regression?</dt>"
            "<dd>Regression that returns posterior distributions for each coefficient instead "
            "of point estimates. You get a full picture of uncertainty.</dd>"
            "<dt>Credible intervals</dt>"
            "<dd>A 95% credible interval means there is a 95% probability the true coefficient "
            "lies within that range — a direct probability statement.</dd>"
            "<dt>How do I know if a predictor matters?</dt>"
            "<dd>If the credible interval excludes zero, the predictor has a credible effect.</dd>"
            "<dt>Why Bayesian?</dt>"
            "<dd>Naturally handles small samples, avoids overfitting through regularizing "
            "priors, and gives probabilistic statements about parameters.</dd>"
            "</dl>"
        ),
    },

    ("bayesian", "bayes_ttest"): {
        "title": "Understanding the Bayesian t-Test",
        "content": (
            "<dl>"
            "<dt>What is a Bayesian t-test?</dt>"
            "<dd>Compares two groups using a Bayes Factor (BF₁₀) that quantifies evidence "
            "for a difference versus no difference.</dd>"
            "<dt>Bayes Factor interpretation</dt>"
            "<dd>BF₁₀ &gt; 3: moderate evidence for difference. BF₁₀ &gt; 10: strong. "
            "BF₁₀ &lt; ⅓: moderate evidence for no difference. Between ⅓–3: inconclusive.</dd>"
            "<dt>Cohen's d posterior</dt>"
            "<dd>The posterior distribution shows how uncertain we are about the standardized "
            "effect size.</dd>"
            "<dt>Why over classical?</dt>"
            "<dd>Classical t-tests cannot quantify evidence for the null. The Bayes Factor "
            "lets you say 'the data support no difference' with specific strength.</dd>"
            "</dl>"
        ),
    },

    ("bayesian", "bayes_ab"): {
        "title": "Understanding Bayesian A/B Testing",
        "content": (
            "<dl>"
            "<dt>What is Bayesian A/B testing?</dt>"
            "<dd>Compares two variants (A and B) using posterior distributions to compute "
            "the probability that one is better than the other.</dd>"
            "<dt>P(B &gt; A)</dt>"
            "<dd>The probability that variant B outperforms A. Unlike p-values, this is "
            "a direct probability statement that answers the decision-maker's question.</dd>"
            "<dt>Expected loss</dt>"
            "<dd>The expected cost of choosing the wrong variant. Even if P(B &gt; A) = 0.6, "
            "the expected loss may be negligible — making the decision low-risk.</dd>"
            "<dt>Stopping rules</dt>"
            "<dd>Bayesian testing allows peeking at results without inflating error rates. "
            "Stop when the expected loss is below your threshold.</dd>"
            "</dl>"
        ),
    },

    ("bayesian", "bayes_correlation"): {
        "title": "Understanding Bayesian Correlation",
        "content": (
            "<dl>"
            "<dt>What is Bayesian correlation?</dt>"
            "<dd>Estimates the correlation coefficient with a posterior distribution, giving "
            "both the best estimate and a credible interval.</dd>"
            "<dt>BF₁₀ for correlation</dt>"
            "<dd>Quantifies evidence for a non-zero correlation vs no relationship.</dd>"
            "<dt>Small sample advantage</dt>"
            "<dd>Especially useful with small samples where p-values are unreliable and "
            "you want honest uncertainty bounds.</dd>"
            "<dt>Evidence for independence</dt>"
            "<dd>BF₁₀ &lt; ⅓ provides positive evidence that there is no relationship — "
            "something classical tests can never tell you.</dd>"
            "</dl>"
        ),
    },

    ("bayesian", "bayes_anova"): {
        "title": "Understanding Bayesian ANOVA",
        "content": (
            "<dl>"
            "<dt>What is Bayesian ANOVA?</dt>"
            "<dd>Tests whether group means differ using Bayes Factors instead of p-values. "
            "Provides evidence for or against the null hypothesis of equal means.</dd>"
            "<dt>Inclusion Bayes Factor</dt>"
            "<dd>For each factor, the inclusion BF tells you how much the data support "
            "including that factor in the model vs excluding it.</dd>"
            "<dt>Posterior group means</dt>"
            "<dd>Each group mean gets a posterior distribution — showing the precision of "
            "the estimate for each group.</dd>"
            "<dt>Model comparison</dt>"
            "<dd>Compares all possible models (null, main effects, interaction) and "
            "reports their posterior probabilities.</dd>"
            "</dl>"
        ),
    },

    ("bayesian", "bayes_changepoint"): {
        "title": "Understanding Bayesian Changepoint Detection",
        "content": (
            "<dl>"
            "<dt>What is Bayesian changepoint detection?</dt>"
            "<dd>Identifies where a time series changed its statistical properties, with "
            "full posterior probability for each candidate changepoint location.</dd>"
            "<dt>Posterior probability</dt>"
            "<dd>Each time point gets a probability of being a changepoint. This quantifies "
            "uncertainty about <em>where</em> the change occurred, not just <em>if</em>.</dd>"
            "<dt>Multiple changepoints</dt>"
            "<dd>Can detect multiple changepoints simultaneously, with the posterior "
            "distribution of the number of changes.</dd>"
            "<dt>Applications</dt>"
            "<dd>Process monitoring, identifying regime changes, dating process shifts "
            "for root cause analysis.</dd>"
            "</dl>"
        ),
    },

    ("bayesian", "bayes_proportion"): {
        "title": "Understanding Bayesian Proportion Test",
        "content": (
            "<dl>"
            "<dt>What is a Bayesian proportion test?</dt>"
            "<dd>Estimates a proportion with a full posterior distribution (Beta distribution). "
            "The credible interval gives an honest range for the true proportion.</dd>"
            "<dt>Beta-Binomial model</dt>"
            "<dd>The conjugate prior (Beta) combined with binomial data gives an exact "
            "posterior. No normal approximation needed — works for any sample size.</dd>"
            "<dt>Comparison of proportions</dt>"
            "<dd>P(p₁ &gt; p₂) computed directly from the posteriors gives the probability "
            "that one proportion exceeds the other.</dd>"
            "<dt>Small sample advantage</dt>"
            "<dd>Classical proportion CIs are unreliable with small n or extreme proportions. "
            "The Bayesian interval is always valid.</dd>"
            "</dl>"
        ),
    },

    ("bayesian", "bayes_capability_prediction"): {
        "title": "Understanding Bayesian Capability Prediction",
        "content": (
            "<dl>"
            "<dt>What is Bayesian capability prediction?</dt>"
            "<dd>Estimates Cpk with full posterior uncertainty, then predicts future "
            "capability based on the current state of knowledge.</dd>"
            "<dt>P(Cpk &gt; target)</dt>"
            "<dd>The probability that true capability meets a specified target. More useful "
            "than a point estimate for decision-making.</dd>"
            "<dt>Predictive distribution</dt>"
            "<dd>Predicts where future individual measurements will fall, accounting for "
            "both process variation and parameter uncertainty.</dd>"
            "<dt>Sequential updating</dt>"
            "<dd>As new data arrives, the posterior updates — showing how capability "
            "knowledge improves with more samples.</dd>"
            "</dl>"
        ),
    },

    ("bayesian", "bayes_equivalence"): {
        "title": "Understanding Bayesian Equivalence Testing",
        "content": (
            "<dl>"
            "<dt>What is Bayesian equivalence testing?</dt>"
            "<dd>Tests whether two groups are equivalent within a practical margin using "
            "the posterior distribution of the difference, providing P(|δ| &lt; Δ).</dd>"
            "<dt>Vs TOST</dt>"
            "<dd>TOST gives a binary accept/reject. Bayesian equivalence gives the "
            "probability of equivalence — much more informative for decision-making.</dd>"
            "<dt>ROPE (Region of Practical Equivalence)</dt>"
            "<dd>The range of effect sizes considered practically equivalent to zero. "
            "The posterior probability within the ROPE determines equivalence.</dd>"
            "<dt>Applications</dt>"
            "<dd>Method comparison, supplier qualification, process transfer — any time "
            "you need to demonstrate that two things are equivalent.</dd>"
            "</dl>"
        ),
    },

    ("bayesian", "bayes_chi2"): {
        "title": "Understanding Bayesian Chi-Square Test",
        "content": (
            "<dl>"
            "<dt>What is a Bayesian chi-square test?</dt>"
            "<dd>Tests independence in contingency tables using Bayes Factors. Can provide "
            "evidence for independence (no association), unlike classical chi-square.</dd>"
            "<dt>Dirichlet-Multinomial model</dt>"
            "<dd>The conjugate model for categorical data. The posterior over cell "
            "probabilities is a Dirichlet distribution.</dd>"
            "<dt>BF interpretation</dt>"
            "<dd>BF₁₀ &gt; 3: evidence for association. BF₁₀ &lt; ⅓: evidence for independence.</dd>"
            "<dt>Small expected counts</dt>"
            "<dd>Unlike classical chi-square, the Bayesian version works well with small "
            "expected counts — no minimum sample size requirements.</dd>"
            "</dl>"
        ),
    },

    ("bayesian", "bayes_poisson"): {
        "title": "Understanding Bayesian Poisson Analysis",
        "content": (
            "<dl>"
            "<dt>What is Bayesian Poisson analysis?</dt>"
            "<dd>Estimates count rates with posterior uncertainty using the Gamma-Poisson "
            "conjugate model. The posterior rate has a Gamma distribution.</dd>"
            "<dt>Rate comparison</dt>"
            "<dd>P(λ₁ &gt; λ₂) computed from posterior gives the probability that one "
            "rate exceeds another.</dd>"
            "<dt>Prediction</dt>"
            "<dd>The posterior predictive distribution (negative binomial) predicts future "
            "counts accounting for rate uncertainty.</dd>"
            "<dt>Overdispersion</dt>"
            "<dd>If counts are overdispersed, a Gamma-Poisson mixture (negative binomial) "
            "model is used automatically.</dd>"
            "</dl>"
        ),
    },

    ("bayesian", "bayes_logistic"): {
        "title": "Understanding Bayesian Logistic Regression",
        "content": (
            "<dl>"
            "<dt>What is Bayesian logistic regression?</dt>"
            "<dd>Models binary outcomes with posterior distributions over coefficients and "
            "predictions. Quantifies uncertainty in odds ratios and predicted probabilities.</dd>"
            "<dt>Posterior odds ratios</dt>"
            "<dd>Each predictor's odds ratio has a credible interval. If the interval "
            "excludes 1, the predictor has a credible effect on the odds.</dd>"
            "<dt>Prediction uncertainty</dt>"
            "<dd>Each predicted probability has a credible interval — wider for predictions "
            "far from the training data.</dd>"
            "<dt>Regularization via priors</dt>"
            "<dd>Weakly informative priors naturally regularize, preventing the separation "
            "problem that plagues classical logistic regression with small samples.</dd>"
            "</dl>"
        ),
    },

    ("bayesian", "bayes_survival"): {
        "title": "Understanding Bayesian Survival Analysis",
        "content": (
            "<dl>"
            "<dt>What is Bayesian survival analysis?</dt>"
            "<dd>Estimates survival functions and hazard rates with full posterior "
            "uncertainty. Handles censored data and produces credible intervals for "
            "reliability predictions.</dd>"
            "<dt>Posterior survival curve</dt>"
            "<dd>The entire survival curve has a credible band — showing where the "
            "estimate is precise and where it's uncertain.</dd>"
            "<dt>B-life posteriors</dt>"
            "<dd>B10, B50 etc. have posterior distributions. P(B10 &gt; t) directly "
            "answers warranty and design life questions.</dd>"
            "<dt>Model comparison</dt>"
            "<dd>Compares Weibull, lognormal, and exponential fits using Bayes Factors "
            "to select the best lifetime distribution.</dd>"
            "</dl>"
        ),
    },

    ("bayesian", "bayes_meta"): {
        "title": "Understanding Bayesian Meta-Analysis",
        "content": (
            "<dl>"
            "<dt>What is Bayesian meta-analysis?</dt>"
            "<dd>Combines results across studies using hierarchical models. The posterior "
            "of the overall effect and between-study variance are estimated simultaneously.</dd>"
            "<dt>Heterogeneity</dt>"
            "<dd>τ² (between-study variance) gets a posterior distribution — you can compute "
            "P(τ² &gt; 0) to assess whether heterogeneity is real.</dd>"
            "<dt>Predictive distribution</dt>"
            "<dd>Predicts the effect in a new study, accounting for both the overall effect "
            "and the between-study variability.</dd>"
            "<dt>Advantages</dt>"
            "<dd>Handles small numbers of studies better than frequentist random-effects. "
            "Naturally incorporates prior information from domain expertise.</dd>"
            "</dl>"
        ),
    },

    ("bayesian", "bayes_demo"): {
        "title": "Understanding Bayesian Demonstration Testing",
        "content": (
            "<dl>"
            "<dt>What is Bayesian demonstration testing?</dt>"
            "<dd>Demonstrates that a reliability target is met using Bayesian analysis. "
            "Computes P(R &gt; target | data) — the probability the true reliability "
            "exceeds the requirement.</dd>"
            "<dt>Prior information</dt>"
            "<dd>Can incorporate prior test data or engineering judgment to reduce "
            "required sample sizes. The prior must be justified and documented.</dd>"
            "<dt>Zero-failure testing</dt>"
            "<dd>Even with zero failures, the posterior gives meaningful bounds on "
            "reliability — unlike classical tests which require at least one failure.</dd>"
            "<dt>Planning</dt>"
            "<dd>Determine the smallest test that achieves P(R &gt; target) ≥ 0.95 "
            "given the prior and expected failure rate.</dd>"
            "</dl>"
        ),
    },

    ("bayesian", "bayes_spares"): {
        "title": "Understanding Bayesian Spare Parts Planning",
        "content": (
            "<dl>"
            "<dt>What is Bayesian spare parts planning?</dt>"
            "<dd>Predicts spare parts demand using posterior predictive distributions. "
            "Accounts for uncertainty in failure rates and fleet size.</dd>"
            "<dt>Demand distribution</dt>"
            "<dd>The posterior predictive gives the full distribution of future demand — "
            "not just the expected value. This enables service level targeting.</dd>"
            "<dt>Service level</dt>"
            "<dd>Stock to meet P(demand ≤ stock) ≥ target service level. The Bayesian "
            "approach honestly accounts for rate uncertainty.</dd>"
            "<dt>Sequential updating</dt>"
            "<dd>As actual demands are observed, the posterior updates and inventory "
            "recommendations adjust automatically.</dd>"
            "</dl>"
        ),
    },

    ("bayesian", "bayes_system"): {
        "title": "Understanding Bayesian System Reliability",
        "content": (
            "<dl>"
            "<dt>What is Bayesian system reliability?</dt>"
            "<dd>Estimates the reliability of a system composed of multiple components, "
            "with full posterior uncertainty propagated from component-level data.</dd>"
            "<dt>System structure</dt>"
            "<dd>Series (all must work), parallel (at least one must work), or complex "
            "configurations. Component posteriors are combined according to the structure.</dd>"
            "<dt>Uncertainty propagation</dt>"
            "<dd>Component-level uncertainty flows through the system model to give "
            "system-level credible intervals.</dd>"
            "<dt>Importance measures</dt>"
            "<dd>Bayesian importance measures identify which component's uncertainty "
            "contributes most to system-level uncertainty — guiding test investments.</dd>"
            "</dl>"
        ),
    },

    ("bayesian", "bayes_warranty"): {
        "title": "Understanding Bayesian Warranty Analysis",
        "content": (
            "<dl>"
            "<dt>What is Bayesian warranty analysis?</dt>"
            "<dd>Predicts future warranty claims using posterior predictive distributions, "
            "accounting for uncertainty in failure rates and reporting patterns.</dd>"
            "<dt>IBNR estimation</dt>"
            "<dd>Incurred But Not Reported claims are estimated from the posterior, giving "
            "a distribution of the total claim count.</dd>"
            "<dt>Financial reserves</dt>"
            "<dd>The posterior predictive distribution of total cost drives warranty "
            "reserve requirements. Higher percentiles give more conservative reserves.</dd>"
            "<dt>Cohort analysis</dt>"
            "<dd>Different production cohorts may have different failure rates. The "
            "hierarchical model borrows strength across cohorts while allowing differences.</dd>"
            "</dl>"
        ),
    },

    ("bayesian", "bayes_repairable"): {
        "title": "Understanding Bayesian Repairable Systems",
        "content": (
            "<dl>"
            "<dt>What is Bayesian repairable systems analysis?</dt>"
            "<dd>Models the failure process of repairable systems using Bayesian NHPP "
            "(Non-Homogeneous Poisson Process) with posterior uncertainty.</dd>"
            "<dt>Growth or deterioration</dt>"
            "<dd>The posterior of the trend parameter β tells you P(β &gt; 1) = probability "
            "the system is deteriorating. P(β &lt; 1) = probability of reliability growth.</dd>"
            "<dt>Prediction</dt>"
            "<dd>Predicts the number of failures in the next time period with credible "
            "intervals, accounting for parameter uncertainty.</dd>"
            "<dt>Maintenance optimization</dt>"
            "<dd>The posterior hazard function guides optimal maintenance scheduling — "
            "repair when the hazard rate exceeds a cost-based threshold.</dd>"
            "</dl>"
        ),
    },

    ("bayesian", "bayes_rul"): {
        "title": "Understanding Bayesian Remaining Useful Life",
        "content": (
            "<dl>"
            "<dt>What is Bayesian RUL?</dt>"
            "<dd>Estimates the Remaining Useful Life of a degrading system using posterior "
            "predictive distributions of the degradation path.</dd>"
            "<dt>Degradation model</dt>"
            "<dd>A parametric degradation model is fitted with Bayesian inference. The "
            "posterior captures uncertainty in degradation rate and form.</dd>"
            "<dt>RUL distribution</dt>"
            "<dd>The time until degradation reaches a failure threshold has a posterior "
            "distribution — giving P(RUL &gt; t) for any time horizon.</dd>"
            "<dt>Condition monitoring</dt>"
            "<dd>As new degradation measurements arrive, the RUL estimate updates — "
            "becoming more precise as the system approaches failure.</dd>"
            "</dl>"
        ),
    },

    ("bayesian", "bayes_alt"): {
        "title": "Understanding Bayesian ALT",
        "content": (
            "<dl>"
            "<dt>What is Bayesian ALT?</dt>"
            "<dd>Bayesian Accelerated Life Testing combines prior knowledge with test data "
            "to estimate lifetime at normal conditions. Posterior uncertainty propagates "
            "through the acceleration model.</dd>"
            "<dt>Prior elicitation</dt>"
            "<dd>Engineering knowledge about activation energies, voltage coefficients, etc. "
            "is encoded as priors. This can substantially reduce required test samples.</dd>"
            "<dt>Model uncertainty</dt>"
            "<dd>The posterior includes uncertainty in both the lifetime distribution and "
            "the acceleration model parameters.</dd>"
            "<dt>Normal-use prediction</dt>"
            "<dd>Posterior predictive at normal stress gives the lifetime distribution "
            "with full uncertainty from extrapolation.</dd>"
            "</dl>"
        ),
    },

    ("bayesian", "bayes_comprisk"): {
        "title": "Understanding Bayesian Competing Risks",
        "content": (
            "<dl>"
            "<dt>What is Bayesian competing risks?</dt>"
            "<dd>Analyzes multiple failure modes with posterior distributions for each "
            "mode's parameters, accounting for the competition between modes.</dd>"
            "<dt>Mode-specific posteriors</dt>"
            "<dd>Each failure mode gets its own posterior lifetime distribution, enabling "
            "mode-specific reliability predictions.</dd>"
            "<dt>Elimination analysis</dt>"
            "<dd>P(system survives to time t | mode k eliminated) shows the benefit of "
            "eliminating each failure mode — directly quantifying improvement potential.</dd>"
            "<dt>Dependent modes</dt>"
            "<dd>Bayesian methods can model dependence between failure modes through "
            "copulas or shared frailty parameters.</dd>"
            "</dl>"
        ),
    },
})


# ═══════════════════════════════════════════════════════════════════════════
# External Modules (causal, drift, anytime, bayes_msa, quality_econ, pbs, ishap)
# ═══════════════════════════════════════════════════════════════════════════

_extend({
    ("causal", "causal_pc"): {
        "title": "Understanding PC Causal Discovery",
        "content": (
            "<dl>"
            "<dt>What is the PC algorithm?</dt>"
            "<dd>A constraint-based causal discovery method that builds a causal graph from "
            "observational data using conditional independence tests. Named after Peter "
            "Spirtes and Clark Glymour.</dd>"
            "<dt>How does it work?</dt>"
            "<dd>Starts with a fully connected graph and removes edges when conditional "
            "independence is detected. Then orients edges based on v-structures.</dd>"
            "<dt>What can it tell you?</dt>"
            "<dd>Which variables have direct causal relationships (edges), which are "
            "independent (no edge), and some edge directions. Not all directions can be "
            "determined from observational data alone.</dd>"
            "<dt>Assumptions</dt>"
            "<dd>Causal sufficiency (no hidden confounders), faithfulness (all independencies "
            "reflect the graph), and adequate sample size. Violations can produce false edges.</dd>"
            "</dl>"
        ),
    },

    ("causal", "causal_lingam"): {
        "title": "Understanding LiNGAM Causal Discovery",
        "content": (
            "<dl>"
            "<dt>What is LiNGAM?</dt>"
            "<dd>Linear Non-Gaussian Acyclic Model — discovers causal direction from "
            "observational data by exploiting non-Gaussianity of error terms.</dd>"
            "<dt>How is it different from PC?</dt>"
            "<dd>PC can leave some edge directions undetermined. LiNGAM can identify all "
            "causal directions — but requires non-Gaussian data and linear relationships.</dd>"
            "<dt>Causal ordering</dt>"
            "<dd>LiNGAM discovers the full causal ordering of variables — which variables "
            "cause which, and with what strength.</dd>"
            "<dt>Limitations</dt>"
            "<dd>Assumes linearity, no hidden confounders, and non-Gaussian distributions. "
            "For Gaussian data, the causal direction is not identifiable — use PC instead.</dd>"
            "</dl>"
        ),
    },

    ("drift", "drift_report"): {
        "title": "Understanding Drift Detection",
        "content": (
            "<dl>"
            "<dt>What is data drift?</dt>"
            "<dd>A change in the statistical properties of input data over time. Drift can "
            "degrade model performance and invalidate process assumptions.</dd>"
            "<dt>Types of drift</dt>"
            "<dd><strong>Covariate drift</strong>: Input distribution changes. "
            "<strong>Concept drift</strong>: The relationship between inputs and outputs changes. "
            "<strong>Prior drift</strong>: The target distribution changes.</dd>"
            "<dt>Detection methods</dt>"
            "<dd>Statistical tests (KS, PSI, chi-square) compare reference and current "
            "distributions. Alerts trigger when drift exceeds a threshold.</dd>"
            "<dt>Response to drift</dt>"
            "<dd>Investigate the cause (process change? data collection change?). Retrain "
            "models if necessary. Update control limits and baselines.</dd>"
            "</dl>"
        ),
    },

    ("anytime", "anytime_ab"): {
        "title": "Understanding Anytime-Valid A/B Testing",
        "content": (
            "<dl>"
            "<dt>What is anytime-valid testing?</dt>"
            "<dd>A/B testing with confidence sequences that remain valid no matter when you "
            "look at the data. No need to pre-specify a sample size or deal with peeking "
            "problems.</dd>"
            "<dt>Confidence sequences vs intervals</dt>"
            "<dd>A confidence sequence is a sequence of intervals that contains the true "
            "parameter at all time points simultaneously. Classical CIs are only valid "
            "at the planned stopping time.</dd>"
            "<dt>When to use</dt>"
            "<dd>Continuous monitoring of A/B tests where you want to stop as soon as "
            "the evidence is clear, without pre-committing to a fixed sample size.</dd>"
            "<dt>E-values</dt>"
            "<dd>The evidence measure used by anytime-valid tests. E-values can be "
            "multiplied across time (like betting) — large E-values indicate strong evidence.</dd>"
            "</dl>"
        ),
    },

    ("anytime", "anytime_onesample"): {
        "title": "Understanding Anytime-Valid One-Sample Test",
        "content": (
            "<dl>"
            "<dt>What is an anytime-valid one-sample test?</dt>"
            "<dd>Tests whether a process mean equals a target value with validity at any "
            "stopping time. No need to commit to a sample size in advance.</dd>"
            "<dt>Sequential monitoring</dt>"
            "<dd>Watch the confidence sequence as data arrives. When the sequence excludes "
            "the null value, you have evidence of a difference — at any time you choose to look.</dd>"
            "<dt>Type I error control</dt>"
            "<dd>The error rate is controlled across all possible stopping times, not just one. "
            "This is stronger than classical tests and prevents p-hacking by design.</dd>"
            "<dt>Applications</dt>"
            "<dd>Process monitoring where you want to detect shifts as quickly as possible "
            "while maintaining rigorous error control.</dd>"
            "</dl>"
        ),
    },

    ("bayes_msa", "bayes_msa"): {
        "title": "Understanding Bayesian Gage R&R",
        "content": (
            "<dl>"
            "<dt>What is Bayesian Gage R&R?</dt>"
            "<dd>A Bayesian approach to measurement system analysis that provides posterior "
            "distributions for all variance components instead of point estimates.</dd>"
            "<dt>Advantages over ANOVA-based MSA</dt>"
            "<dd>No negative variance components, valid with small samples, full uncertainty "
            "quantification, and P(%GRR &lt; threshold) for direct decision-making.</dd>"
            "<dt>Posterior %GRR</dt>"
            "<dd>The posterior distribution of %GRR (measurement system as % of total) gives "
            "P(%GRR &lt; 10%) = probability the system is acceptable.</dd>"
            "<dt>Prior information</dt>"
            "<dd>Previous MSA studies can be incorporated as informative priors, reducing "
            "required study size for subsequent assessments.</dd>"
            "</dl>"
        ),
    },

    ("quality_econ", "taguchi_loss"): {
        "title": "Understanding Taguchi Loss Function",
        "content": (
            "<dl>"
            "<dt>What is the Taguchi loss function?</dt>"
            "<dd>A quadratic model that quantifies the economic cost of deviation from a "
            "target value. Loss increases with the square of the deviation — not just "
            "beyond spec limits.</dd>"
            "<dt>Why it matters</dt>"
            "<dd>Traditional pass/fail ignores the cost of being close to spec limits. "
            "Taguchi loss shows that a process centered on target is economically "
            "superior to one barely meeting specs.</dd>"
            "<dt>Loss = k(y - T)²</dt>"
            "<dd>k is the loss coefficient (cost per unit deviation²), y is the measured "
            "value, T is the target. The expected loss depends on both the mean offset "
            "and the variance.</dd>"
            "<dt>Applications</dt>"
            "<dd>Cost of quality estimation, process optimization targeting, and justifying "
            "investment in variability reduction.</dd>"
            "</dl>"
        ),
    },

    ("quality_econ", "process_decision"): {
        "title": "Understanding Process Decision Analysis",
        "content": (
            "<dl>"
            "<dt>What is process decision analysis?</dt>"
            "<dd>Evaluates the economic trade-offs of process improvement decisions — whether "
            "the cost of changing the process is justified by the quality improvement.</dd>"
            "<dt>Cost-benefit framework</dt>"
            "<dd>Compares the cost of improvement (equipment, training, downtime) against "
            "the expected reduction in defect-related costs (scrap, rework, warranty).</dd>"
            "<dt>Break-even analysis</dt>"
            "<dd>The defect rate improvement needed to justify the investment. If the "
            "expected improvement exceeds break-even, proceed.</dd>"
            "<dt>Decision under uncertainty</dt>"
            "<dd>Monte Carlo simulation of costs and benefits gives a distribution of ROI, "
            "not just a point estimate.</dd>"
            "</dl>"
        ),
    },

    ("quality_econ", "lot_sentencing"): {
        "title": "Understanding Lot Sentencing Economics",
        "content": (
            "<dl>"
            "<dt>What is lot sentencing economics?</dt>"
            "<dd>Evaluates the total cost of an acceptance sampling program, including "
            "inspection costs, accepted defect costs, and rejected lot costs.</dd>"
            "<dt>Total cost model</dt>"
            "<dd>C_total = C_inspection × n + C_accept_defective × AOQ × N + "
            "C_reject × P(reject). Balances sampling cost against quality risk.</dd>"
            "<dt>Economic lot size</dt>"
            "<dd>The lot size that minimizes total cost per unit, considering sampling "
            "plan efficiency and defect cost.</dd>"
            "<dt>Optimization</dt>"
            "<dd>Finds the sampling plan (n, c) that minimizes total cost while meeting "
            "AQL and LTPD requirements.</dd>"
            "</dl>"
        ),
    },

    ("quality_econ", "cost_of_quality"): {
        "title": "Understanding Cost of Quality",
        "content": (
            "<dl>"
            "<dt>What is Cost of Quality (COQ)?</dt>"
            "<dd>The total cost of quality activities, categorized into prevention costs, "
            "appraisal costs, internal failure costs, and external failure costs.</dd>"
            "<dt>The four categories</dt>"
            "<dd><strong>Prevention</strong>: Training, process design, DOE. "
            "<strong>Appraisal</strong>: Inspection, testing, audits. "
            "<strong>Internal failure</strong>: Scrap, rework, downtime. "
            "<strong>External failure</strong>: Warranty, returns, lost customers.</dd>"
            "<dt>The optimum</dt>"
            "<dd>Total COQ is minimized when prevention + appraisal spending reduces "
            "failure costs more than it adds. The chart shows the trade-off.</dd>"
            "<dt>Hidden costs</dt>"
            "<dd>External failure costs are often underestimated. Lost customer lifetime "
            "value and reputation damage can dwarf direct warranty costs.</dd>"
            "</dl>"
        ),
    },

    ("pbs", "pbs_full"): {
        "title": "Understanding Process Behaviour Summary",
        "content": (
            "<dl>"
            "<dt>What is the PBS?</dt>"
            "<dd>A comprehensive process behaviour analysis that combines control chart "
            "assessment, capability evaluation, and distributional analysis with "
            "Bayesian uncertainty quantification.</dd>"
            "<dt>What does it produce?</dt>"
            "<dd>Stability assessment, capability indices with posteriors, distributional "
            "fit, and actionable recommendations in one integrated report.</dd>"
            "<dt>Bayesian advantage</dt>"
            "<dd>All estimates come with credible intervals. Small-sample capability "
            "assessments are honest about uncertainty.</dd>"
            "<dt>When to use</dt>"
            "<dd>Initial process characterization, periodic process reviews, or any time "
            "you need a complete process behaviour assessment.</dd>"
            "</dl>"
        ),
    },

    ("pbs", "pbs_belief"): {
        "title": "Understanding PBS Belief Assessment",
        "content": (
            "<dl>"
            "<dt>What is the belief assessment?</dt>"
            "<dd>Quantifies the probability that the process meets specified requirements "
            "using posterior predictive analysis.</dd>"
            "<dt>P(in spec)</dt>"
            "<dd>The posterior probability that a randomly selected unit will be within "
            "specification limits — directly answering the quality question.</dd>"
            "<dt>Evidence strength</dt>"
            "<dd>How much data supports the belief. With few samples, the belief is uncertain "
            "(wide credible interval). More data tightens the bound.</dd>"
            "<dt>Decision threshold</dt>"
            "<dd>Compare P(in spec) against your decision threshold (e.g., 0.9973 for 3σ). "
            "The posterior probability makes the decision transparent.</dd>"
            "</dl>"
        ),
    },

    ("pbs", "pbs_edetector"): {
        "title": "Understanding PBS E-Detector",
        "content": (
            "<dl>"
            "<dt>What is the E-detector?</dt>"
            "<dd>An anytime-valid sequential change detection method that monitors for "
            "process shifts using e-values — providing rigorous false alarm control.</dd>"
            "<dt>How it works</dt>"
            "<dd>E-values accumulate evidence of a change. When the e-value exceeds a "
            "threshold, a change is declared — valid at any stopping time.</dd>"
            "<dt>Vs CUSUM/EWMA</dt>"
            "<dd>E-detectors provide mathematically guaranteed false alarm rates without "
            "distributional assumptions. CUSUM/EWMA rely on parametric assumptions.</dd>"
            "<dt>Detection speed</dt>"
            "<dd>Typically detects shifts within a few observations of the change, with "
            "speed depending on the shift magnitude.</dd>"
            "</dl>"
        ),
    },

    ("pbs", "pbs_evidence"): {
        "title": "Understanding PBS Evidence Assessment",
        "content": (
            "<dl>"
            "<dt>What is the evidence assessment?</dt>"
            "<dd>Evaluates the strength of evidence that the process has shifted from its "
            "baseline behaviour, using Bayes Factors.</dd>"
            "<dt>Evidence grades</dt>"
            "<dd>Anecdotal, moderate, strong, very strong, decisive — based on the BF "
            "magnitude. The grade tells you how much to trust the shift signal.</dd>"
            "<dt>Sequential evidence</dt>"
            "<dd>Evidence accumulates as data arrives. You can monitor the BF trajectory "
            "to see when evidence became compelling.</dd>"
            "<dt>Two-sided evidence</dt>"
            "<dd>The BF can also provide evidence for stability (no shift) — important for "
            "confirming that a process hasn't changed after a suspected event.</dd>"
            "</dl>"
        ),
    },

    ("pbs", "pbs_predictive"): {
        "title": "Understanding PBS Predictive Analysis",
        "content": (
            "<dl>"
            "<dt>What is predictive analysis?</dt>"
            "<dd>Uses the posterior predictive distribution to forecast where future "
            "observations will fall, accounting for all sources of uncertainty.</dd>"
            "<dt>Prediction intervals</dt>"
            "<dd>Unlike control limits (which characterize past behaviour), prediction "
            "intervals give honest ranges for future values. They're always wider because "
            "they include parameter uncertainty.</dd>"
            "<dt>Calibration</dt>"
            "<dd>A well-calibrated model has prediction intervals that actually contain "
            "the specified proportion of future observations.</dd>"
            "<dt>Applications</dt>"
            "<dd>Forecasting process output, setting realistic expectations for customers, "
            "and planning for worst-case scenarios.</dd>"
            "</dl>"
        ),
    },

    ("pbs", "pbs_adaptive"): {
        "title": "Understanding PBS Adaptive Monitoring",
        "content": (
            "<dl>"
            "<dt>What is adaptive monitoring?</dt>"
            "<dd>Continuously updates the process model as new data arrives, adjusting "
            "control limits and capability estimates in real time.</dd>"
            "<dt>Learning rate</dt>"
            "<dd>Controls how quickly the model adapts. Too fast: overreacts to noise. "
            "Too slow: misses real shifts. The Bayesian framework optimizes this.</dd>"
            "<dt>Vs static charts</dt>"
            "<dd>Static charts use fixed limits from historical data. Adaptive monitoring "
            "evolves with the process — ideal for processes that improve or change over time.</dd>"
            "<dt>When to use</dt>"
            "<dd>New processes without historical baselines, processes undergoing improvement, "
            "or seasonal processes where the baseline shifts.</dd>"
            "</dl>"
        ),
    },

    ("pbs", "pbs_cpk"): {
        "title": "Understanding PBS Bayesian Cpk",
        "content": (
            "<dl>"
            "<dt>What is Bayesian Cpk?</dt>"
            "<dd>Capability index estimated with full posterior uncertainty rather than "
            "a point estimate. Gives P(Cpk &gt; 1.33) — the probability the process "
            "is truly capable.</dd>"
            "<dt>Small sample honesty</dt>"
            "<dd>With n = 30, a classical Cpk = 1.5 could easily have true value 1.1 or 2.0. "
            "The Bayesian posterior shows this uncertainty explicitly.</dd>"
            "<dt>Specification comparison</dt>"
            "<dd>Compare capability against multiple thresholds simultaneously — probability "
            "of meeting each one is computed from the posterior.</dd>"
            "<dt>Decision framework</dt>"
            "<dd>Accept/reject decisions based on P(Cpk &gt; threshold) rather than point "
            "estimates — reduces the risk of wrong decisions from sampling variability.</dd>"
            "</dl>"
        ),
    },

    ("pbs", "pbs_cpk_traj"): {
        "title": "Understanding PBS Cpk Trajectory",
        "content": (
            "<dl>"
            "<dt>What is a Cpk trajectory?</dt>"
            "<dd>Tracks how the Bayesian Cpk estimate evolves over time as more data "
            "accumulates. Shows whether capability is improving, stable, or degrading.</dd>"
            "<dt>Convergence</dt>"
            "<dd>The credible interval narrows as more data arrives. When it stabilizes, "
            "you have enough data for a reliable capability assessment.</dd>"
            "<dt>Trend detection</dt>"
            "<dd>A systematic downward trend signals degrading capability — triggering "
            "proactive intervention before the process becomes incapable.</dd>"
            "<dt>Monitoring frequency</dt>"
            "<dd>Update the trajectory at regular intervals (shift, day, batch) to maintain "
            "continuous visibility into capability health.</dd>"
            "</dl>"
        ),
    },

    ("pbs", "pbs_health"): {
        "title": "Understanding PBS Process Health",
        "content": (
            "<dl>"
            "<dt>What is process health?</dt>"
            "<dd>A composite assessment combining stability, capability, distributional fit, "
            "and trend analysis into an overall process health score.</dd>"
            "<dt>Health components</dt>"
            "<dd><strong>Stability</strong>: No special causes detected. "
            "<strong>Capability</strong>: Cpk meets target. <strong>Centering</strong>: "
            "Process centered on target. <strong>Trend</strong>: No degradation.</dd>"
            "<dt>Traffic light system</dt>"
            "<dd>Green: All components healthy. Yellow: One or more marginal. Red: One or "
            "more failing. Immediate visibility into process status.</dd>"
            "<dt>Actionable recommendations</dt>"
            "<dd>Each health component maps to specific improvement actions. The analysis "
            "prioritizes interventions by impact.</dd>"
            "</dl>"
        ),
    },

    ("ishap", "ishap"): {
        "title": "Understanding Interventional SHAP",
        "content": (
            "<dl>"
            "<dt>What is Interventional SHAP?</dt>"
            "<dd>An extension of SHAP that computes feature attributions based on causal "
            "interventions rather than observational conditioning. Gives causal "
            "explanations for model predictions.</dd>"
            "<dt>Interventional vs observational</dt>"
            "<dd>Standard SHAP conditions on features (observational). Interventional SHAP "
            "simulates what happens when you actively change a feature — reflecting the "
            "real-world effect of process adjustments.</dd>"
            "<dt>Why it matters</dt>"
            "<dd>Observational SHAP can give misleading feature importance when features are "
            "correlated. Interventional SHAP correctly attributes effects through the causal "
            "structure.</dd>"
            "<dt>Requirements</dt>"
            "<dd>A trained model and knowledge of the causal structure (which variables "
            "are causes vs effects). Without causal knowledge, falls back to standard SHAP.</dd>"
            "</dl>"
        ),
    },
})
