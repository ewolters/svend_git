"""Learning content: Statistical Inference."""

from ._datasets import SHARED_DATASET  # noqa: F401


CHOOSING_TESTS = {
    "id": "choosing-tests",
    "title": "Choosing the Right Test",
    "intro": "Picking the wrong statistical test is like using a hammer on a screw. Walk through the decision tree to match your question and data type to the right analysis.",
    "exercise": {
        "title": "Try It: Navigate the Test Selector",
        "steps": [
            "Select your outcome variable type (continuous or categorical)",
            "Choose the number of groups you are comparing",
            "Indicate whether your data is paired or independent",
            "Review the recommended test and its assumptions",
            "Toggle the non-normal option to see the nonparametric alternative"
        ],
    },
    "content": """
## Decision Tree for Statistical Tests

### Step 1: What's Your Question?

| Question Type | Test Category |
|---------------|---------------|
| Is there a difference? | Comparison tests |
| Is there a relationship? | Correlation/Regression |
| Does this fit a pattern? | Goodness of fit |
| Can I predict something? | Regression/Classification |

### Step 2: What's Your Data?

**Outcome Variable:**
- Continuous (height, revenue) → t-test, ANOVA, regression
- Categorical (yes/no, A/B/C) → Chi-square, logistic regression
- Count (events per day) → Poisson regression

**Groups:**
- 2 groups → t-test
- 3+ groups → ANOVA
- Paired/matched → Paired tests

### Common Tests Cheat Sheet

| Scenario | Test |
|----------|------|
| Compare 2 group means | Independent t-test |
| Compare same group before/after | Paired t-test |
| Compare 3+ group means | One-way ANOVA |
| Compare means with multiple factors | Two-way ANOVA |
| Compare 2 proportions | Chi-square test |
| Relationship between 2 continuous | Pearson correlation |
| Predict continuous from continuous | Linear regression |
| Predict binary outcome | Logistic regression |

### Step 3: Check Assumptions

**For t-tests and ANOVA:**
- Normality (or large n > 30)
- Equal variances (Levene's test)
- Independence

**If assumptions violated:**
- Use non-parametric alternatives (Mann-Whitney, Kruskal-Wallis)
- Transform data (log, sqrt)
- Use robust methods

## Example Decision Process

**Question:** "Did our new checkout flow increase conversion rate?"

1. Outcome: Categorical (converted yes/no)
2. Groups: 2 (old vs new)
3. Data: Unpaired (different users)

**Answer:** Chi-square test (or two-proportion z-test)

Or better: Bayesian A/B test (gives probability of improvement, not just "significant")
""",
    "interactive": {
        "type": "test_selector",
        "config": {
            "show_decision_tree": True,
            "allow_data_type_input": True,
            "recommend_test": True,
        }
    },
    "key_takeaways": [
        "Start with your question, not the test",
        "Match test to data type and structure",
        "Always check assumptions",
        "Non-parametric tests are safer when in doubt",
        "Bayesian tests often give more useful answers",
    ],
    "practice_questions": [
        {
            "question": "You have satisfaction scores (1-10) for 3 customer segments and suspect the data is heavily skewed. What test do you use?",
            "answer": "Kruskal-Wallis test. You have 3+ groups with a continuous-ish outcome, but skewed data violates the normality assumption for ANOVA. Kruskal-Wallis is the non-parametric alternative that compares medians across groups.",
            "hint": "3+ groups + non-normal data → non-parametric alternative to ANOVA"
        },
        {
            "question": "You want to know if there's a relationship between education level (high school / bachelor's / master's / PhD) and voting preference (candidate A / B / C). What test?",
            "answer": "Chi-square test of independence. Both variables are categorical. You'd create a 4×3 contingency table and test whether education and voting preference are independent. If any expected cell count is <5, use Fisher's exact test instead.",
            "hint": "Both variables categorical → contingency table test"
        },
    ]
}


INTERPRETING_RESULTS = {
    "id": "interpreting-results",
    "title": "Interpreting Results",
    "intro": "A p-value of 0.03 does NOT mean there's a 97% chance the effect is real. Most people get this wrong. You'll practice reading p-values, confidence intervals, and effect sizes correctly.",
    "exercise": {
        "title": "Try It: Decode a Statistical Result",
        "steps": [
            "Enter a sample size and observed effect",
            "Review the simulated p-value distribution",
            "Check the confidence interval width",
            "Calculate the effect size (Cohen's d)",
            "Decide: is this statistically AND practically significant?"
        ],
        "dsw_type": "stats:ttest",
        "dsw_config": {"var1": "diameter_mm", "mu": 25.0},
    },
    "content": """
## P-Values: What They Actually Mean

### The Definition

p-value = Probability of seeing results this extreme **if the null hypothesis were true**

### What p < 0.05 Does NOT Mean

- ❌ "There's a 95% chance the effect is real"
- ❌ "The effect is large"
- ❌ "The result is important"
- ❌ "There's only 5% chance we're wrong"

### What p < 0.05 DOES Mean

- ✓ "If there were no effect, we'd see data this extreme less than 5% of the time"

That's it. It says nothing about effect size or practical importance.

## Confidence Intervals: More Useful Than P-Values

A 95% CI gives a range of plausible values:

```
Effect: 5.2% increase
95% CI: [2.1%, 8.3%]
```

This tells you:
- Direction: Positive
- Magnitude: Between 2.1% and 8.3%
- Precision: Reasonably tight range

**A narrow CI around zero** is more informative than "p > 0.05":
- CI: [-0.3%, +0.2%] means "probably no meaningful effect"
- vs "p = 0.06" which could mean anything

## Effect Sizes

### Cohen's d (for mean differences)

$$d = \\frac{\\bar{x}_1 - \\bar{x}_2}{s_{pooled}}$$

| d | Interpretation |
|---|----------------|
| 0.2 | Small |
| 0.5 | Medium |
| 0.8 | Large |

### Relative Risk / Odds Ratio

For binary outcomes:
- RR = 2.0 means "twice as likely"
- OR = 2.0 means "twice the odds" (not quite the same!)

## Statistical vs Practical Significance

**Scenario:** A/B test with n=1,000,000

Result: New button increases clicks by 0.001%
p-value: 0.00001 (highly significant!)

**Question:** Is this worth implementing?

With enough data, any tiny effect becomes "statistically significant." Always ask:
- Is the effect big enough to matter?
- Does it justify the cost of change?
- Would users notice?
""",
    "interactive": {
        "type": "results_interpreter",
        "config": {
            "show_pvalue_simulator": True,
            "show_ci_visualizer": True,
            "show_effect_size_calculator": True,
        }
    },
    "key_takeaways": [
        "P-values answer a narrow question - understand what it is",
        "Confidence intervals are more informative than p-values",
        "Effect size tells you if the result matters",
        "Statistical significance ≠ practical significance",
        "With big data, everything is 'significant' - focus on magnitude",
    ],
    "practice_questions": [
        {
            "question": "An A/B test shows: treatment mean = 4.52, control mean = 4.50, p = 0.001, n = 2,000,000 per group, 95% CI for difference: [0.01, 0.03]. Is this result actionable?",
            "answer": "Probably not. While statistically significant (p=0.001), the effect is tiny: 0.02 units on a scale where the CI tops out at 0.03. Cohen's d would be minuscule. With 2M per group, even trivial differences reach significance. Ask: would anyone notice a 0.02 difference? Does it justify the cost of change?",
            "hint": "Look at the CI range, not just the p-value"
        },
        {
            "question": "A study reports 'p = 0.06, a trend toward significance.' How should you interpret this?",
            "answer": "There is no such thing as 'a trend toward significance.' p=0.06 is weak evidence against the null — not qualitatively different from p=0.04. Report the effect size and CI instead. If CI is [-0.5, 12.3], the data is simply inconclusive (wide CI spanning zero). The study may be underpowered.",
            "hint": "P-values are continuous — 0.06 and 0.04 provide similar evidence"
        },
    ]
}


MULTIPLE_COMPARISONS = {
    "id": "multiple-comparisons",
    "title": "Multiple Comparisons & P-Hacking",
    "intro": "Test 20 hypotheses and you'll find a 'significant' result by chance. This section exposes how multiple testing inflates false positives and why preregistration is the antidote.",
    "exercise": {
        "title": "Try It: Watch False Positives Accumulate",
        "steps": [
            "Set the number of simultaneous tests to 20",
            "Run the simulation with all null hypotheses true",
            "Count how many come back significant at p<0.05",
            "Apply Bonferroni correction and re-check",
            "Switch to FDR (Benjamini-Hochberg) and compare the results"
        ],
        "dsw_type": "stats:anova",
        "dsw_config": {"response": "diameter_mm", "factor": "line"},
    },
    "content": """
## How to Lie With Statistics (And How Not To)

The multiple comparisons problem is one of the main reasons published research often fails to replicate.

### The Problem Explained

If you test 20 hypotheses at α=0.05, you expect 1 false positive even if all null hypotheses are true.

**Example:**
Testing if a drug affects 20 different outcomes:
- All 20 nulls are true (drug does nothing)
- At α=0.05, expect 20 × 0.05 = 1 false positive
- You report the one "significant" finding
- This is wrong, but it happens constantly

### The Math

Probability of at least one false positive with m tests:

$$P(\\text{at least one FP}) = 1 - (1-\\alpha)^m$$

| Tests (m) | P(at least one FP) |
|-----------|-------------------|
| 1 | 5% |
| 5 | 23% |
| 10 | 40% |
| 20 | 64% |
| 100 | 99.4% |

With enough tests, a false positive is nearly guaranteed.

### Correction Methods

#### Bonferroni Correction
Simplest but most conservative.
$$\\alpha_{adjusted} = \\frac{\\alpha}{m}$$

For 20 tests at α=0.05: Use α=0.0025 per test.

- Pro: Simple, controls familywise error rate
- Con: Very conservative, low power

#### Holm-Bonferroni
Order p-values, apply sequential thresholds.
1. Smallest p-value: compare to α/m
2. Second smallest: compare to α/(m-1)
3. Continue until one fails to reject

- Pro: More powerful than Bonferroni
- Con: More complex

#### False Discovery Rate (FDR)
Controls expected proportion of false positives among rejected hypotheses.

Benjamini-Hochberg procedure:
1. Order p-values: p₁ ≤ p₂ ≤ ... ≤ pₘ
2. Find largest k where p_k ≤ (k/m)×α
3. Reject hypotheses 1 through k

- Pro: More powerful for many tests
- Con: Allows some false positives

### P-Hacking: The Dark Side

P-hacking = exploiting researcher degrees of freedom to achieve p<0.05.

**Common p-hacking techniques:**
- Test many outcomes, report only significant ones
- Test many subgroups, report only significant ones
- Try different exclusion criteria until significant
- Try different covariates until significant
- Collect more data until significant
- Transform data different ways until significant

**The result:** Published "significant" findings that don't replicate.

### The Garden of Forking Paths

Even without intentional manipulation, researcher decisions create multiple implicit tests:

"Should I exclude outliers?" (Yes/No = 2 paths)
"Use parametric or non-parametric?" (2 paths)
"Log-transform?" (2 paths)
"Include covariate X?" (2 paths)

2×2×2×2 = 16 different analyses, all "reasonable."

You only run one, but you chose the path after seeing the data.

### Preregistration: The Solution

**Preregistration** = Specify analysis plan before seeing data.

Register:
- Primary outcome
- Analysis method
- Sample size
- Exclusion criteria
- Secondary analyses (labeled as exploratory)

Then: You can't p-hack (no choices left to make).

### Detecting P-Hacking in Published Research

**Red flags:**
- Many outcomes tested, only some reported
- P-values clustered just below 0.05
- Effect disappears in replication
- "We also found that..." multiple discoveries
- No preregistration, no mention of analysis choices
- Sample size not justified by power analysis

**Caliper test:** Suspiciously many p-values between 0.01-0.05 vs 0.05-0.10.

### Exploratory vs Confirmatory Research

Both are valid, but different:

| Exploratory | Confirmatory |
|-------------|--------------|
| Generate hypotheses | Test hypotheses |
| Look for patterns | Test specific predictions |
| P-values are guides | P-values are evidence |
| Requires replication | Is replication |
| Don't correct | Do correct |

**The mistake:** Treating exploratory findings as confirmatory.
""",
    "interactive": {
        "type": "multiple_comparison_demo",
        "config": {
            "show_false_positive_accumulation": True,
            "show_correction_methods": True,
            "show_p_hacking_simulation": True,
        }
    },
    "key_takeaways": [
        "With many tests, false positives are expected, not rare",
        "Bonferroni is simplest correction; FDR is more powerful",
        "P-hacking exploits researcher choices to manufacture significance",
        "Preregistration eliminates flexibility that enables p-hacking",
        "Distinguish exploratory (hypothesis-generating) from confirmatory (hypothesis-testing)",
    ],
    "practice_questions": [
        {
            "question": "A study tests whether a drug affects any of 10 biomarkers. One shows p=0.03. Should we conclude the drug affects this biomarker?",
            "answer": "No. With 10 tests, Bonferroni-adjusted threshold is 0.05/10=0.005. The finding p=0.03 is not significant after correction. Expected ~0.5 false positives among 10 tests; this could easily be one.",
            "hint": "Apply Bonferroni correction."
        },
        {
            "question": "A paper reports p=0.048 and mentions they 'excluded 3 outliers.' What should you ask?",
            "answer": "Were the exclusion criteria specified before analysis? How were outliers defined? What's the p-value without exclusions? This pattern (p just under 0.05 after exclusions) suggests potential p-hacking.",
            "hint": "The p-value being just under 0.05 after a decision is suspicious."
        }
    ]
}


P_VALUES_DEEP_DIVE = {
    "id": "p-values-deep-dive",
    "title": "P-Values: The Deep Dive",
    "intro": "P-values are the most misunderstood concept in statistics. You'll simulate the 'dance of the p-values' and see why p=0.049 and p=0.051 are essentially the same evidence.",
    "exercise": {
        "title": "Try It: Simulate the Dance of P-Values",
        "steps": [
            "Set a true effect size of d=0.5",
            "Run the simulation 20 times with n=30 per group",
            "Count how many p-values fall below 0.05",
            "Observe the wide spread of p-values from the same truth",
            "Set effect to zero and see the uniform distribution of p-values"
        ],
        "dsw_type": "stats:ttest",
        "dsw_config": {"var1": "diameter_mm", "mu": 25.0},
    },
    "content": """
## What P-Values Actually Mean

P-values are the most misunderstood concept in statistics. Let's fix that.

### The Precise Definition

**P-value:** The probability of observing a result at least as extreme as yours, assuming the null hypothesis is true.

$$p = P(\\text{data this extreme} | H_0 \\text{ true})$$

That's it. It's a conditional probability about the DATA, not the hypothesis.

### What P < 0.05 Does NOT Mean

| Common Misinterpretation | Why It's Wrong |
|--------------------------|----------------|
| "95% chance effect is real" | P-value isn't P(H₀ false) |
| "Only 5% chance of being wrong" | P-value isn't error probability |
| "The effect is large" | P-value says nothing about size |
| "The result is important" | Significance ≠ importance |
| "The experiment worked" | Could be true null with bad luck |

### What P < 0.05 DOES Mean

"If there were truly no effect, we'd see data this extreme less than 5% of the time."

That's useful, but limited.

### The Coin Flip Analogy

You flip a coin 10 times and get 9 heads.

- P-value: P(9+ heads | fair coin) ≈ 0.01
- This means: If the coin is fair, 9+ heads is rare
- It does NOT mean: 99% chance coin is biased

The coin could still be fair—you just got unlucky (1% of the time).

### P-Values Under the Null

When H₀ is true, p-values are uniformly distributed:
- 5% of the time, p < 0.05 (by definition!)
- 1% of the time, p < 0.01
- This is why false positives happen

### The Dance of the P-Values

P-values from repeated experiments (same truth) vary wildly:

```
Experiment 1: p = 0.03 "Significant!"
Experiment 2: p = 0.12 "Not significant"
Experiment 3: p = 0.04 "Significant!"
Experiment 4: p = 0.07 "Not significant"
```

All from the same real effect. The p-value dances around.

### P-Value as Continuous Measure

Stop thinking binary. P = 0.049 and p = 0.051 are essentially identical evidence.

Better interpretation:
- p < 0.001: Strong evidence against null
- p ≈ 0.01: Moderate evidence
- p ≈ 0.05: Weak evidence
- p > 0.10: Little to no evidence

### Why P-Values Get Abused

1. **Binary thinking:** Significant/not significant dichotomy
2. **Publishing incentives:** Only p < 0.05 gets published
3. **Misunderstanding:** Treating as P(hypothesis)
4. **Flexibility:** Many ways to get p < 0.05 if you try

### Alternatives and Complements

| Approach | What It Tells You |
|----------|-------------------|
| Confidence intervals | Range of plausible effect sizes |
| Effect sizes | How big the effect is |
| Bayes factors | Relative evidence for H₁ vs H₀ |
| Posterior probability | P(hypothesis | data) — what you actually want |

### The Bottom Line

P-values answer a narrow question narrowly. They're not useless, but they're not what most people think they are.

Always report:
1. Effect size (how big)
2. Confidence interval (how precise)
3. P-value (if you must)
""",
    "interactive": {"type": "pvalue_simulator", "config": {}},
    "key_takeaways": [
        "P-value = P(data this extreme | null true), not P(null false | data)",
        "P < 0.05 doesn't mean 95% chance effect is real",
        "P-values vary wildly across replications of the same truth",
        "Treat p-values as continuous evidence, not binary",
        "Always report effect sizes and confidence intervals too",
    ],
    "practice_questions": [
        {
            "question": "A colleague says 'We got p=0.03 so there's a 97% chance the treatment works.' Correct their reasoning.",
            "answer": "This inverts the conditional. P=0.03 means: IF the null is true, there's a 3% chance of data this extreme. It is NOT the probability the null is false. To get P(treatment works | data), you need Bayes' theorem — which requires a prior probability. If the prior probability of the treatment working was low (say 10%), p=0.03 might only bring you to ~75% confidence, not 97%.",
            "hint": "P(data|null) ≠ P(null|data) — this is the prosecutor's fallacy"
        },
        {
            "question": "You run the same experiment 4 times with the same true effect (d=0.5). You get p-values of 0.001, 0.12, 0.04, and 0.23. Is this contradictory?",
            "answer": "No — this is completely normal. P-values dance around across replications. With a real effect of d=0.5, your power determines how often you'll get p<0.05. If power is ~60%, you'd expect roughly 2 of 4 replications to fail to reach significance. The pattern is consistent with a real but moderate effect.",
            "hint": "P-values from the same truth vary wildly — this is expected"
        },
    ]
}


CONFIDENCE_INTERVALS = {
    "id": "confidence-intervals",
    "title": "Confidence Intervals",
    "intro": "CIs tell you everything p-values do, plus the magnitude and precision of the effect. You'll visualize CIs and learn why overlapping intervals do NOT necessarily mean no difference.",
    "exercise": {
        "title": "Try It: Interpret a Confidence Interval",
        "steps": [
            "Enter two group means and sample sizes",
            "Review the 95% CI for the difference",
            "Check whether the CI includes zero",
            "Increase the sample size and watch the CI narrow",
            "Compare individual CIs vs the CI for the difference"
        ],
        "dsw_type": "stats:bootstrap_ci",
        "dsw_config": {"var": "diameter_mm", "statistic": "mean"},
    },
    "content": """
## More Useful Than P-Values

Confidence intervals tell you what you actually want to know: what values are plausible for the true effect.

### What a 95% CI Means

**Correct interpretation:** If we repeated this study many times, 95% of the calculated intervals would contain the true value.

**Practical interpretation:** Values inside the CI are plausible; values outside are implausible given the data.

### CI vs P-Value

| P-value | 95% CI | What It Tells You |
|---------|--------|-------------------|
| p = 0.03 | [2, 18] | Effect is likely between 2 and 18 |
| p = 0.03 | [0.5, 3] | Effect is likely between 0.5 and 3 |

Same p-value, very different information! The CI tells you magnitude.

### Narrow vs Wide CIs

**Narrow CI [4.8, 5.2]:**
- Precise estimate
- Large sample size
- Low variability

**Wide CI [-2, 12]:**
- Imprecise estimate
- Small sample size
- High variability

### CI for Different Parameters

**For means:**
$$\\bar{x} \\pm t_{\\alpha/2} \\times \\frac{s}{\\sqrt{n}}$$

**For proportions:**
$$\\hat{p} \\pm z_{\\alpha/2} \\times \\sqrt{\\frac{\\hat{p}(1-\\hat{p})}{n}}$$

**For differences:**
$$\\text{difference} \\pm t_{\\alpha/2} \\times SE_{difference}$$

### Overlapping CIs

**Common mistake:** "CIs overlap, so no significant difference"

**Reality:** CIs can overlap and difference still be significant.

For comparing two means, compute the CI for the *difference*, not whether individual CIs overlap.

### CI and Significance

If 95% CI excludes zero (for a difference) → p < 0.05
If 95% CI includes zero → p > 0.05

But the CI tells you MORE:
- How close to zero?
- What's the range of plausible effects?
- Is the effect practically meaningful even if CI excludes zero?

### Reporting CIs

**Good:** "Mean improvement was 5.2 points (95% CI: 2.1 to 8.3)"

**Bad:** "Results were significant (p < 0.05)"

The CI version tells you everything the p-value does, plus the effect size and precision.

### Bayesian Credible Intervals

Frequentist CI: "95% of intervals from repeated sampling contain truth"

Bayesian credible interval: "95% probability the true value is in this interval"

The Bayesian version is what most people think CIs mean (and what they want to know).
""",
    "interactive": {"type": "ci_visualizer", "config": {}},
    "key_takeaways": [
        "CIs show the range of plausible values for the true effect",
        "Narrow CI = precise estimate; wide CI = imprecise",
        "CIs tell you everything p-values do, plus effect size and precision",
        "Overlapping CIs don't necessarily mean no difference",
        "Always report CIs alongside point estimates",
    ],
    "practice_questions": [
        {
            "question": "Group A: mean=50, 95% CI [45, 55]. Group B: mean=54, 95% CI [49, 59]. The CIs overlap. Can you conclude there's no difference?",
            "answer": "No! Overlapping individual CIs don't prove no difference. The CI for the difference (B-A) might be [0.5, 7.5] — which excludes zero and is significant. Individual CIs overlap more readily than you'd expect. Always compute the CI for the difference directly.",
            "hint": "Overlapping CIs is not the same as CI-of-difference including zero"
        },
        {
            "question": "A study reports: 'Mean weight loss was 3.2 kg (95% CI: -1.1 to 7.5 kg), p=0.14.' What do you conclude?",
            "answer": "The data is inconclusive. The CI spans from a slight weight gain (-1.1) to a substantial loss (7.5 kg). While the point estimate is positive, the study can't distinguish between the treatment causing weight gain or meaningful weight loss. The study is likely underpowered — a larger sample would narrow the CI to resolve the question.",
            "hint": "A wide CI spanning zero means the study couldn't resolve the question"
        },
    ]
}


EFFECT_SIZES = {
    "id": "effect-sizes",
    "title": "Effect Sizes & Practical Significance",
    "intro": "Statistical significance tells you the effect probably isn't zero. Effect size tells you if anyone should care. You'll calculate Cohen's d, NNT, and odds ratios, and learn when a tiny significant effect is worthless.",
    "exercise": {
        "title": "Try It: Calculate Effect Sizes",
        "steps": [
            "Enter two group means and a pooled standard deviation",
            "Calculate Cohen's d and classify as small, medium, or large",
            "For a binary outcome, compute the NNT",
            "Compare: is d=0.08 with p<0.001 worth acting on?"
        ],
        "dsw_type": "stats:ttest",
        "dsw_config": {"var1": "diameter_mm", "mu": 25.0},
    },
    "content": """
## Does the Result Actually Matter?

Statistical significance tells you the effect probably isn't zero. Effect size tells you if it's big enough to care about.

### Why Effect Size Matters

With large n, even tiny effects are "significant":

| n per group | Effect | p-value | Meaningful? |
|-------------|--------|---------|-------------|
| 50 | d = 0.8 | 0.001 | Yes |
| 5,000 | d = 0.08 | 0.001 | Probably not |
| 500,000 | d = 0.008 | 0.001 | Definitely not |

The p-values are identical. The effects are wildly different.

### Cohen's d (Standardized Mean Difference)

$$d = \\frac{\\bar{x}_1 - \\bar{x}_2}{s_{pooled}}$$

| d | Interpretation |
|---|----------------|
| 0.2 | Small |
| 0.5 | Medium |
| 0.8 | Large |

**Example:** d = 0.5 means the groups differ by half a standard deviation.

### Correlation Coefficient (r)

$$r^2 = \\text{proportion of variance explained}$$

| r | r² | Interpretation |
|---|-----|----------------|
| 0.1 | 1% | Small |
| 0.3 | 9% | Medium |
| 0.5 | 25% | Large |

### Odds Ratio and Relative Risk

**For binary outcomes:**

$$RR = \\frac{P(outcome | treatment)}{P(outcome | control)}$$

$$OR = \\frac{odds_{treatment}}{odds_{control}}$$

| Value | Interpretation |
|-------|----------------|
| 1.0 | No effect |
| 1.5 | 50% increase |
| 2.0 | Doubled |
| 0.5 | Halved |

**Note:** OR ≈ RR only when outcome is rare. For common outcomes, OR exaggerates.

### Number Needed to Treat (NNT)

$$NNT = \\frac{1}{|risk_{treatment} - risk_{control}|}$$

**Interpretation:** Treat NNT patients to prevent one bad outcome.

| NNT | Interpretation |
|-----|----------------|
| 5 | Very effective (treat 5 to help 1) |
| 20 | Moderately effective |
| 100 | Weak effect (treat 100 to help 1) |

### Statistical vs Practical Significance

**Scenario:** New training improves test scores by 0.5 points (out of 100).
- p < 0.001 (statistically significant)
- d = 0.05 (tiny effect)
- Cost: $10,000 per employee

**Question:** Worth it?

Consider:
- What's the benefit of 0.5 points?
- What else could $10,000 buy?
- Would anyone notice the difference?

### Context Matters

d = 0.2 might be:
- **Small** for a therapy that's expensive and risky
- **Large** for a cheap, simple intervention at scale

Always interpret effect sizes in context:
- Cost of intervention
- Baseline rates
- Alternative uses of resources
- Whether effect is noticeable
""",
    "interactive": {"type": "effect_size_calculator", "config": {}},
    "key_takeaways": [
        "Effect size tells you how big, p-value just tells you not zero",
        "With big n, everything is significant—effect size matters more",
        "Cohen's d: 0.2 small, 0.5 medium, 0.8 large",
        "NNT translates effects into actionable terms",
        "Always interpret effect sizes in practical context",
    ],
    "practice_questions": [
        {
            "question": "A drug reduces heart attack risk from 2% to 1.5%. Calculate the NNT. Is this clinically meaningful?",
            "answer": "NNT = 1 / |0.02 - 0.015| = 1 / 0.005 = 200. You'd need to treat 200 patients to prevent one heart attack. Whether this is meaningful depends on: cost/side effects of the drug, severity of the outcome (heart attacks are severe), and available alternatives. For a cheap, safe drug preventing a fatal event, NNT=200 may be worthwhile. For an expensive drug with side effects, maybe not.",
            "hint": "NNT = 1 / absolute risk reduction"
        },
        {
            "question": "Study A reports d=0.3 for a free email intervention (n=10,000). Study B reports d=0.8 for a $50,000 training program (n=40). Which finding is more useful?",
            "answer": "Study A is likely more useful despite the smaller effect. d=0.3 for a free, scalable intervention affecting 10,000 people creates far more total impact than d=0.8 for an expensive program with n=40 (which may not replicate — small sample, large effect is a red flag). Effect sizes must be interpreted in context of cost, scalability, and sample reliability.",
            "hint": "Context matters: cost, scalability, and sample size all affect practical significance"
        },
    ]
}


