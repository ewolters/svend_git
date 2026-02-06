"""
Learning Module Content

Interactive educational content for SVEND Analyst Certification.
Each section has structured content with explanations, examples, and interactive elements.
"""

# =============================================================================
# Foundations Module
# =============================================================================

BAYESIAN_THINKING = {
    "id": "bayesian-thinking",
    "title": "Bayesian Thinking",
    "content": """
## What is Bayesian Thinking?

Bayesian thinking is a framework for updating beliefs based on evidence. Instead of asking "is this true or false?", we ask "how confident should I be given what I know?"

### The Core Equation

$$P(H|E) = \\frac{P(E|H) \\cdot P(H)}{P(E)}$$

In plain English:
- **P(H|E)** = Probability of hypothesis given evidence (posterior)
- **P(E|H)** = Probability of seeing this evidence if hypothesis is true (likelihood)
- **P(H)** = Probability of hypothesis before evidence (prior)
- **P(E)** = Probability of the evidence overall

### Likelihood Ratios: A Simpler Approach

Instead of computing full probabilities, we can use **likelihood ratios**:

$$LR = \\frac{P(E|H_{true})}{P(E|H_{false})}$$

- **LR > 1**: Evidence supports the hypothesis
- **LR < 1**: Evidence opposes the hypothesis
- **LR = 1**: Evidence is neutral

### Example: Medical Diagnosis

A patient has symptom X. You're considering Disease A.

- P(Symptom X | Disease A) = 0.9 (90% of patients with Disease A have this symptom)
- P(Symptom X | No Disease A) = 0.1 (10% of healthy people have this symptom)

Likelihood Ratio = 0.9 / 0.1 = **9**

This symptom is 9x more likely if the patient has Disease A. Strong evidence!

### The Confidence Adjustment

Not all evidence is equally reliable. We adjust the likelihood ratio based on confidence:

$$LR_{adjusted} = 1 + (LR - 1) \\times confidence$$

If our test has only 70% reliability:
- Raw LR = 9
- Adjusted LR = 1 + (9-1) × 0.7 = **6.6**

Still strong, but appropriately discounted.
""",
    "interactive": {
        "type": "bayesian_calculator",
        "config": {
            "show_prior": True,
            "show_likelihood_ratio": True,
            "show_confidence": True,
            "show_posterior": True,
        }
    },
    "key_takeaways": [
        "Probability is a measure of confidence, not truth",
        "Update beliefs incrementally as evidence arrives",
        "Likelihood ratios tell you how much evidence shifts your belief",
        "Discount weak evidence - don't treat all information equally",
    ],
    "practice_questions": [
        {
            "question": "You believe there's a 30% chance of rain. Weather radar shows clouds (LR=3 for rain). What's your updated probability?",
            "answer": "About 56%. Prior odds = 0.3/0.7 ≈ 0.43. Posterior odds = 0.43 × 3 = 1.29. Posterior probability = 1.29/(1+1.29) ≈ 0.56",
            "hint": "Convert probability to odds, multiply by LR, convert back"
        },
        {
            "question": "A test has LR=10 but only 50% reliability. What's the effective LR?",
            "answer": "5.5. Adjusted LR = 1 + (10-1) × 0.5 = 5.5",
            "hint": "Use the confidence adjustment formula"
        }
    ]
}

HYPOTHESIS_DRIVEN = {
    "id": "hypothesis-driven",
    "title": "Hypothesis-Driven Investigation",
    "content": """
## The Problem with Unstructured Investigation

Most people investigate by:
1. Looking at data
2. Noticing something interesting
3. Concluding that's the answer

This is backwards. It leads to:
- **Confirmation bias**: You find what you're looking for
- **Narrative fallacy**: You create stories that fit the data
- **Overfitting**: Your explanation is specific to this data, not generalizable

## The Hypothesis-Driven Approach

### Step 1: Generate Hypotheses FIRST

Before looking at data, ask: "What could explain this?"

Generate **multiple competing hypotheses**:
- Hypothesis A: The UI change caused the sales drop
- Hypothesis B: Seasonality caused the sales drop
- Hypothesis C: A competitor's promotion caused the drop
- Hypothesis D: A data collection bug is showing false numbers

### Step 2: Define What Evidence Would Distinguish Them

For each hypothesis, ask: "What would I expect to see if this were true?"

| Hypothesis | Expected Evidence |
|------------|-------------------|
| UI change | Drop correlates with change date |
| Seasonality | Same pattern last year |
| Competitor | Drop in traffic, not conversion |
| Data bug | Discrepancy between systems |

### Step 3: Gather Evidence Systematically

Now look at data - but with specific questions, not "let's see what's there."

### Step 4: Update Probabilities

As evidence comes in, update your confidence in each hypothesis:
- Evidence supports UI change → increase its probability
- Evidence contradicts seasonality → decrease its probability

### Step 5: Know When to Conclude

Stop when:
- One hypothesis has very high probability (>90%)
- Remaining hypotheses are ruled out (<10%)
- Additional evidence won't change the conclusion

## The "Murder Board" Technique

For each hypothesis, actively try to **disprove** it:
- What evidence would convince you this hypothesis is wrong?
- Have you looked for that evidence?
- If you can't disprove it, it survives

This prevents falling in love with your first idea.
""",
    "interactive": {
        "type": "hypothesis_tracker",
        "config": {
            "max_hypotheses": 5,
            "show_probability_bars": True,
            "allow_evidence_input": True,
        }
    },
    "key_takeaways": [
        "Generate hypotheses BEFORE looking at data",
        "Always maintain multiple competing explanations",
        "Define distinguishing evidence upfront",
        "Actively try to disprove your favorite hypothesis",
        "Know your stopping criteria",
    ],
    "practice_questions": [
        {
            "question": "Your website conversion rate dropped 20%. What are three competing hypotheses?",
            "answer": "Many valid answers. Examples: (1) Recent site change broke something, (2) Traffic source mix changed, (3) Seasonality, (4) Measurement error",
            "hint": "Think about different categories: technical, external, measurement"
        },
        {
            "question": "You think Hypothesis A is correct. What should you do next?",
            "answer": "Try to disprove it. Look for evidence that would contradict Hypothesis A. If you can't find any, your confidence is justified.",
            "hint": "Avoid confirmation bias"
        }
    ]
}

EVIDENCE_QUALITY = {
    "id": "evidence-quality",
    "title": "Evidence Quality",
    "content": """
## Not All Evidence is Equal

A peer-reviewed randomized controlled trial and your friend's anecdote are both "evidence." They are not equally valuable.

## Dimensions of Evidence Quality

### 1. Source Reliability

| Source Type | Typical Confidence |
|-------------|-------------------|
| Controlled experiment | 0.9 - 0.95 |
| Observational study | 0.6 - 0.8 |
| Expert opinion | 0.5 - 0.7 |
| Anecdote | 0.2 - 0.4 |

### 2. Sample Size

Larger samples = more reliable estimates

The confidence interval shrinks with √n:
- n=10: Wide uncertainty
- n=100: Moderate uncertainty
- n=1000: Narrow uncertainty

### 3. Reproducibility

Has this result been replicated?
- Single study: Discount heavily
- Replicated 2-3 times: Moderate confidence
- Meta-analysis of many studies: High confidence

### 4. Potential for Bias

Ask:
- Who conducted the study? (Conflicts of interest?)
- How was the sample selected? (Selection bias?)
- What wasn't measured? (Confounding variables?)

## The Confidence Discount

When entering evidence into your analysis, assign a confidence score:

```
Raw finding: "Drug X reduces symptoms by 50%"
Source: Single small study (n=30), industry-funded

Confidence adjustment:
- Small sample: 0.7
- Single study: 0.8
- Industry funding: 0.8

Combined confidence: 0.7 × 0.8 × 0.8 = 0.45

Use 45% confidence when applying this evidence
```

## Red Flags

Be skeptical when you see:
- Extraordinary claims without extraordinary evidence
- Results that seem too clean (real data is messy)
- No discussion of limitations
- Cherry-picked time periods or metrics
- "Correlation proves causation" reasoning
""",
    "interactive": {
        "type": "evidence_rater",
        "config": {
            "show_source_options": True,
            "show_sample_size": True,
            "show_replication_status": True,
            "calculate_confidence": True,
        }
    },
    "key_takeaways": [
        "Assign confidence scores to all evidence",
        "Discount for small samples, single studies, potential bias",
        "Extraordinary claims need extraordinary evidence",
        "Reproducibility is the gold standard",
        "Know the red flags of weak evidence",
    ],
    "practice_questions": [
        {
            "question": "A blog post claims a new productivity technique 'doubles output.' The author tried it for a week. What confidence would you assign?",
            "answer": "Very low, perhaps 0.2-0.3. Single person, short duration, self-reported, no control group.",
            "hint": "Consider sample size, measurement reliability, and potential bias"
        }
    ]
}

# =============================================================================
# Data Fundamentals Module
# =============================================================================

DATA_CLEANING = {
    "id": "data-cleaning",
    "title": "Data Cleaning",
    "content": """
## Garbage In, Garbage Out

No statistical technique can fix bad data. Cleaning is not glamorous, but it's essential.

## Missing Data

### Types of Missingness

1. **Missing Completely at Random (MCAR)**: Missingness unrelated to any variable
   - Example: Survey responses lost due to server crash
   - Safe to drop or impute

2. **Missing at Random (MAR)**: Missingness related to observed variables
   - Example: Older patients less likely to complete online surveys
   - Can adjust using observed variables

3. **Missing Not at Random (MNAR)**: Missingness related to the missing value itself
   - Example: High earners don't report income
   - Dangerous - can't fully correct

### Handling Strategies

| Strategy | When to Use | Caution |
|----------|-------------|---------|
| Drop rows | MCAR, small % missing | Loses information |
| Drop columns | >50% missing | May lose important variable |
| Mean imputation | MCAR, continuous | Underestimates variance |
| Mode imputation | MCAR, categorical | Distorts distribution |
| Regression imputation | MAR | More accurate but complex |
| Multiple imputation | MAR | Gold standard, more work |

## Outliers

### Detection Methods

1. **IQR Method**: Outlier if value < Q1 - 1.5×IQR or > Q3 + 1.5×IQR
2. **Z-Score**: Outlier if |z| > 3 (more than 3 standard deviations)
3. **Domain Knowledge**: "A human can't be 900 years old"

### Handling Strategies

- **Investigate first**: Is it an error or real extreme value?
- **Winsorize**: Cap at percentile (e.g., 99th)
- **Transform**: Log transform reduces outlier influence
- **Robust methods**: Use median instead of mean

## Data Validation

### Consistency Checks

```
# Cross-field validation
assert birth_date < hire_date, "Can't be hired before birth"

# Range checks
assert 0 <= probability <= 1, "Probability must be in [0,1]"

# Referential integrity
assert customer_id in customers_table, "Unknown customer"
```

### Duplicate Detection

- Exact duplicates: Easy to find
- Near duplicates: "John Smith" vs "Jon Smith" vs "J. Smith"
- Use fuzzy matching and domain knowledge
""",
    "interactive": {
        "type": "data_cleaner",
        "config": {
            "show_missing_analysis": True,
            "show_outlier_detection": True,
            "allow_strategy_selection": True,
        }
    },
    "key_takeaways": [
        "Understand WHY data is missing before choosing a strategy",
        "Investigate outliers - don't automatically remove them",
        "Validate data against domain constraints",
        "Document all cleaning decisions for reproducibility",
    ]
}

SAMPLING = {
    "id": "sampling",
    "title": "Sampling",
    "content": """
## Why Sampling Matters

You can't measure every customer, every transaction, every instance. Sampling lets you draw conclusions about populations from subsets - but only if done correctly.

## Sampling Methods

### Simple Random Sampling

Every member has equal probability of selection.

```python
import random
sample = random.sample(population, n=100)
```

Good for: Homogeneous populations
Bad for: Rare subgroups get underrepresented

### Stratified Sampling

Divide population into groups, sample from each proportionally.

```
Population: 70% customers from US, 30% from EU
Sample: 70 US customers, 30 EU customers (n=100)
```

Good for: Ensuring subgroup representation
Required for: Subgroup comparisons

### Cluster Sampling

Randomly select clusters, include all members.

```
Population: All students in a school district
Sample: Randomly select 5 schools, survey all students in those schools
```

Good for: Geographically dispersed populations
Caution: Clusters may not be representative

## Sample Size Determination

### For Proportions (e.g., conversion rate)

$$n = \\frac{z^2 \\cdot p(1-p)}{e^2}$$

Where:
- z = 1.96 for 95% confidence
- p = expected proportion (use 0.5 if unknown)
- e = margin of error

Example: For 95% CI with ±3% margin:
$$n = \\frac{1.96^2 \\cdot 0.5 \\cdot 0.5}{0.03^2} ≈ 1067$$

### For Means

$$n = \\frac{z^2 \\cdot \\sigma^2}{e^2}$$

Need to estimate σ from pilot study or prior data.

## Selection Bias

### Common Traps

1. **Survivorship Bias**: Only successful companies in your database
2. **Self-Selection**: Only motivated users take surveys
3. **Convenience Sampling**: Using whoever's easiest to reach
4. **Non-Response Bias**: Non-responders differ from responders

### Detection

Ask: "Who is systematically excluded from my sample?"

If your sample of "all customers" only includes those with email addresses, you're missing a population segment.
""",
    "interactive": {
        "type": "sample_size_calculator",
        "config": {
            "show_confidence_level": True,
            "show_margin_of_error": True,
            "show_population_size": True,
            "calculate_n": True,
        }
    },
    "key_takeaways": [
        "Random sampling is the foundation of valid inference",
        "Stratify when subgroups matter",
        "Calculate required sample size BEFORE collecting data",
        "Selection bias invalidates results - always ask who's missing",
    ]
}

# =============================================================================
# Statistical Tools Module
# =============================================================================

CHOOSING_TESTS = {
    "id": "choosing-tests",
    "title": "Choosing the Right Test",
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
    ]
}

INTERPRETING_RESULTS = {
    "id": "interpreting-results",
    "title": "Interpreting Results",
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
    ]
}

# =============================================================================
# New Foundations Content
# =============================================================================

BASE_RATE_NEGLECT = {
    "id": "base-rate-neglect",
    "title": "Base Rate Neglect",
    "content": """
## The Most Common Reasoning Error

Base rate neglect is the tendency to ignore general information (base rates) in favor of specific information (individual case details). It's arguably the most important reasoning error to understand.

### The Classic Example: The Taxi Problem

> A taxi was involved in a hit-and-run. 85% of taxis in the city are Green, 15% are Blue.
> A witness identified the taxi as Blue. In tests, the witness correctly identifies colors 80% of the time.
> What's the probability the taxi was actually Blue?

**Most people say:** 80% (the witness accuracy)

**Correct answer:** ~41%

Why? Let's work through it with 100 taxis:
- 85 Green taxis, 15 Blue taxis (base rates)
- Witness sees 85 Green: correctly says "Green" for 68, incorrectly says "Blue" for 17
- Witness sees 15 Blue: correctly says "Blue" for 12, incorrectly says "Green" for 3

Total "Blue" identifications: 17 + 12 = 29
Correct "Blue" identifications: 12

**P(Blue | Witness says Blue) = 12/29 ≈ 41%**

The witness being right 80% of the time isn't enough when Blue taxis are rare.

### Why This Matters in Practice

#### Medical Diagnosis

A test for a rare disease (prevalence 1 in 1000) has:
- 99% sensitivity (catches 99% of cases)
- 99% specificity (only 1% false positives)

Sounds great, right? Let's test 100,000 people:
- 100 have disease → 99 positive (true positives)
- 99,900 healthy → 999 positive (false positives)

**If you test positive: P(disease) = 99/(99+999) ≈ 9%**

A "99% accurate" test gives you only 9% confidence! This is why we don't screen for rare diseases.

#### Fraud Detection

Your model catches 95% of fraud with 5% false positive rate. Sounds great.
But if only 0.1% of transactions are fraud:
- 1,000 transactions: 1 fraud, 999 legitimate
- Fraud: 0.95 caught
- Legitimate: 50 flagged as fraud

**Precision = 0.95/(0.95+50) ≈ 2%**

Almost all your flags are false positives.

### The Formula

Use Bayes' theorem, but the intuitive version:

$$\\text{Posterior odds} = \\text{Prior odds} \\times \\text{Likelihood ratio}$$

Or work through the 100-person table method shown above.

### How to Avoid Base Rate Neglect

1. **Always ask: "What's the base rate?"** Before considering evidence
2. **Use the 100-person (or 1000-person) method** - Convert to frequencies
3. **Be suspicious when base rate isn't mentioned** - It's often omitted
4. **Build the habit** - Forcing yourself to consider base rates becomes automatic

### Common Traps

| Situation | Base Rate Often Ignored |
|-----------|------------------------|
| Medical diagnosis | Disease prevalence in population |
| Hiring decisions | Proportion of good candidates in applicant pool |
| Fraud detection | Actual fraud rate |
| Security screening | Actual threat rate |
| Predictive policing | Crime rate in area |
""",
    "interactive": {
        "type": "base_rate_calculator",
        "config": {
            "show_base_rate": True,
            "show_test_accuracy": True,
            "show_natural_frequencies": True,
        }
    },
    "key_takeaways": [
        "Base rate neglect is the tendency to ignore prior probabilities",
        "High accuracy tests can still give mostly false positives for rare events",
        "Always ask 'what's the base rate?' before considering evidence",
        "Convert to natural frequencies (100 people) to reason correctly",
        "If base rate isn't mentioned, be suspicious",
    ],
    "practice_questions": [
        {
            "question": "A test for a condition (prevalence 2%) has 90% sensitivity and 90% specificity. If you test positive, what's the probability you have the condition?",
            "answer": "About 15%. In 1000 people: 20 have it (18 test positive), 980 don't (98 test positive). P = 18/(18+98) ≈ 15.5%",
            "hint": "Use the 1000-person method. Calculate true positives and false positives separately."
        },
        {
            "question": "Why is base rate neglect worse for rare events?",
            "answer": "When something is rare, even a small false positive rate produces more false positives than true positives. The base rate dominates.",
            "hint": "Think about what happens when there are 1 true case and 1000 non-cases"
        }
    ]
}

REGRESSION_TO_MEAN = {
    "id": "regression-to-mean",
    "title": "Regression to the Mean",
    "content": """
## Why Extreme Results Don't Last

Regression to the mean is one of the most important and least understood statistical phenomena. It's not a causal force - it's a mathematical inevitability.

### What It Is

When you observe an extreme value of a variable with random variation, subsequent measurements will tend to be closer to the average.

**Key insight:** This happens automatically. No intervention required. No causal mechanism needed.

### The Sports Example

A baseball player hits .400 in the first half of the season (exceptional).
**Prediction:** They will hit closer to league average in the second half.

Why? Their first-half performance was partly skill, partly luck. The luck component won't repeat.

**This is NOT:**
- The player "choking under pressure"
- Teams figuring them out
- Physical decline

**It IS:**
- Random variation evening out over time

### The "Sophomore Slump"

Rookie of the Year often disappoints in year two. Why?
- To win Rookie of the Year, you need to be good AND lucky
- The luck won't repeat
- Performance regresses toward their true ability level

### The Dangerous Misinterpretation

#### Scenario: Speed Cameras

City installs speed cameras at intersections with the most accidents.
Next year: Accidents at those locations decrease 30%.
Conclusion: "Speed cameras work!"

**Problem:** Accidents would have decreased anyway. Those locations were selected *because* they were extreme. Extreme values regress to the mean.

The correct test: Compare camera locations to similar locations without cameras.

#### Scenario: Medical Treatment

Patients seek treatment when symptoms are worst.
After treatment: They improve.
Conclusion: "Treatment works!"

**Problem:** Patients often seek help at their worst point. They would have improved anyway (regression to mean + natural recovery).

This is why we need control groups.

### The Math

If correlation between Time 1 and Time 2 is r:

$$\\text{Expected T2} = \\bar{x} + r(T1 - \\bar{x})$$

If r = 0.5 and someone is 2 standard deviations above mean at T1:
- Expected T2 = mean + 0.5 × 2SD = 1 SD above mean

They regress halfway to the mean.

### How to Detect Regression Effects

1. **Ask: Was selection based on an extreme value?**
   - If yes, expect regression regardless of any intervention

2. **Use control groups**
   - If both treated and untreated groups regress equally, it's not the treatment

3. **Use regression discontinuity designs**
   - Compare just above vs just below a threshold

4. **Be suspicious of before/after comparisons without controls**

### Real-World Examples Where Regression Is Misinterpreted

| Situation | Misinterpretation |
|-----------|------------------|
| Students with worst grades improve after tutoring | "Tutoring works!" (maybe, but regression expected anyway) |
| Patients at peak symptoms improve after treatment | "Treatment works!" (maybe, but regression expected anyway) |
| Worst-performing employees improve after criticism | "Criticism motivates!" (regression expected) |
| Best-performing employees decline after praise | "Praise causes complacency!" (regression expected) |
| Accident hotspots improve after intervention | "Intervention works!" (regression expected) |
""",
    "interactive": {
        "type": "regression_simulator",
        "config": {
            "show_scatter_plot": True,
            "show_regression_line": True,
            "allow_correlation_adjustment": True,
        }
    },
    "key_takeaways": [
        "Extreme observations tend to be followed by less extreme ones",
        "This is mathematical, not causal - no intervention needed",
        "Selection based on extreme values guarantees regression",
        "Before/after comparisons without controls are misleading",
        "Control groups are essential to detect true treatment effects",
    ],
    "practice_questions": [
        {
            "question": "A company identifies its 10 worst-performing stores and provides extra training. Performance improves. What should you conclude?",
            "answer": "Very little. Those stores were selected because they were extreme. Regression to the mean would cause improvement even without training. You need to compare to similar stores without training.",
            "hint": "Think about why those stores were selected in the first place."
        },
        {
            "question": "Why do pilots often think punishment works better than praise for trainees?",
            "answer": "After exceptionally good performance (praised), regression makes next attempt worse. After exceptionally bad performance (punished), regression makes next attempt better. It looks like punishment works, but it's just regression.",
            "hint": "Consider what follows extreme performances, on average."
        }
    ]
}

# =============================================================================
# Experimental Design Module
# =============================================================================

RANDOMIZATION_CONTROLS = {
    "id": "randomization-controls",
    "title": "Randomization & Controls",
    "content": """
## The Foundation of Causal Inference

Randomization is the single most powerful tool for establishing causation. It's why randomized controlled trials (RCTs) sit atop the evidence hierarchy.

### Why Randomization Works

**The Problem:** We want to know if Treatment causes Outcome.
But people who choose treatment differ from those who don't (confounding).

**The Solution:** Don't let people choose. Randomly assign.

Randomization ensures that treatment and control groups are, on average, identical in:
- Measured variables (age, gender, etc.)
- **Unmeasured variables** (motivation, genetics, etc.)

This is the magic: randomization balances things we don't even know about.

### Types of Randomization

#### Simple Randomization
Flip a coin for each participant.
- Pro: Simplest, truly random
- Con: Can create unbalanced groups by chance (especially with small n)

#### Stratified Randomization
Randomize within strata (subgroups).
```
Within each age group (young/old) × gender (M/F):
  Randomly assign half to treatment, half to control
```
- Pro: Guarantees balance on stratification variables
- Con: Can only stratify on a few variables

#### Block Randomization
Randomize in blocks to ensure balance throughout.
```
Block of 4: Randomly assign 2 to treatment, 2 to control
Repeat for each block
```
- Pro: Ensures balance at any stopping point
- Con: Predictable at end of blocks

### Control Groups

#### Types of Controls

| Control Type | Description | When to Use |
|--------------|-------------|-------------|
| No treatment | Nothing given | When placebo effect unlikely |
| Placebo | Inert treatment | When patient expectation matters |
| Active control | Existing treatment | Comparing new vs standard |
| Waitlist | Delayed treatment | Ethical when treatment expected to work |
| Usual care | Continue normal practice | Pragmatic trials |

#### Placebo Effect

The act of being treated improves outcomes, even if treatment is inert.
- Real in subjective outcomes (pain, mood)
- Why blinding matters

### Blinding

| Type | Who's Blinded | Purpose |
|------|--------------|---------|
| Single-blind | Participants | Prevent placebo effect bias |
| Double-blind | Participants + researchers | Prevent measurement bias |
| Triple-blind | + data analysts | Prevent analysis bias |

**When blinding is impossible:**
- Surgery vs medication
- Exercise interventions
- Psychotherapy

Use objective outcomes and blinded assessors when possible.

### Threats to Randomization

#### Non-compliance
Assigned to treatment but doesn't take it.
Solutions:
- Intention-to-treat analysis (analyze as randomized)
- Per-protocol analysis (analyze as received) - biased but informative
- Instrumental variable analysis

#### Attrition
Participants drop out differentially.
- If 30% of treatment drops out vs 5% of control, groups no longer comparable
- Always report attrition and reasons

#### Randomization Failure
Check: Are baseline characteristics balanced?
- Statistical tests (but low power)
- Practical assessment (are differences meaningful?)

### When Randomization Is Impossible

Sometimes you can't randomize:
- Unethical (randomize to smoking)
- Impractical (randomize to education level)
- Too late (studying past events)

Then use:
- Quasi-experimental designs (next section)
- Match on observed confounders
- But acknowledge: unmeasured confounding remains
""",
    "interactive": {
        "type": "randomization_demo",
        "config": {
            "show_group_balance": True,
            "allow_stratification": True,
            "show_confounding_removal": True,
        }
    },
    "key_takeaways": [
        "Randomization balances both measured and unmeasured confounders",
        "Stratified randomization guarantees balance on key variables",
        "Blinding prevents expectation effects and measurement bias",
        "Intention-to-treat analysis preserves randomization benefits",
        "Check baseline balance to verify randomization worked",
    ],
    "practice_questions": [
        {
            "question": "You're testing a new teaching method. Students in the new method class score higher. Why might this NOT prove the method works?",
            "answer": "Students weren't randomized to classes. Motivated students may have chosen the new method. The classes may meet at different times attracting different students. Without randomization, confounding is likely.",
            "hint": "Think about how students ended up in each class."
        },
        {
            "question": "In a drug trial, 40% of patients in the treatment group stop taking the medication due to side effects. How should you analyze the results?",
            "answer": "Primary analysis should be intention-to-treat: analyze everyone according to their randomized group, regardless of compliance. This preserves randomization. You can do per-protocol as secondary analysis, acknowledging the selection bias.",
            "hint": "Why would analyzing only compliers be biased?"
        }
    ]
}

POWER_ANALYSIS = {
    "id": "power-analysis",
    "title": "Power Analysis",
    "content": """
## Determining Sample Size Before You Collect Data

Power analysis answers: "How many participants do I need?"

Getting this wrong is worse than it sounds:
- Too few: You won't detect real effects (wasted effort)
- Too many: Wasted resources, potential ethical issues

### The Four Quantities (Pick 3, Solve for 4th)

1. **Sample size (n)**: Number of participants
2. **Effect size (d)**: How big is the effect you're looking for
3. **Alpha (α)**: False positive rate (usually 0.05)
4. **Power (1-β)**: Probability of detecting a real effect (usually 0.80)

Typically you set α=0.05, power=0.80, estimate effect size, solve for n.

### Understanding Power

**Power = P(reject H₀ | H₀ is false)**

Or: If there IS a real effect, power is the probability you'll detect it.

| Power | Interpretation |
|-------|---------------|
| 0.50 | Coin flip - detect effect half the time |
| 0.80 | Standard - detect effect 80% of time |
| 0.90 | Conservative - detect effect 90% of time |
| 0.95 | Very conservative |

**The tragedy of underpowered studies:**
A study with 50% power that finds "no effect" tells you almost nothing. The effect might exist; you just couldn't detect it.

### Effect Size: The Critical Input

**Cohen's d** (for comparing means):
$$d = \\frac{\\mu_1 - \\mu_2}{\\sigma}$$

| d | Interpretation | Example |
|---|---------------|---------|
| 0.2 | Small | Barely noticeable |
| 0.5 | Medium | Obvious to careful observer |
| 0.8 | Large | Obvious to anyone |

**Where to get effect size estimates:**
1. Prior studies on same topic
2. Pilot study
3. Minimum effect that would be practically meaningful
4. When in doubt, assume small (d=0.2) to be safe

### Sample Size Formulas

#### For Comparing Two Means (t-test)

$$n = \\frac{2(z_{1-\\alpha/2} + z_{1-\\beta})^2}{d^2}$$

For α=0.05, power=0.80:
$$n ≈ \\frac{16}{d^2} \\text{ per group}$$

| Effect Size | n per group |
|-------------|-------------|
| d = 0.2 (small) | 400 |
| d = 0.5 (medium) | 64 |
| d = 0.8 (large) | 25 |

#### For Comparing Two Proportions

Need to specify both proportions (p₁ and p₂), not just difference.

Rough guide for detecting 5 percentage point difference:
- If baseline is 50%: ~800 per group
- If baseline is 10%: ~250 per group
- If baseline is 2%: ~1000 per group

### Why Underpowered Studies Are Harmful

1. **High false negative rate**: Real effects declared "not significant"
2. **Effect size inflation**: Significant results in underpowered studies overestimate effect
3. **Wasted resources**: Participants, time, money for uninformative results
4. **Ethical concerns**: Exposing participants to risk without adequate chance of learning

### The Power Analysis Workflow

1. **Define your primary outcome** - What are you measuring?
2. **Estimate effect size** - From literature, pilot, or practical significance
3. **Set alpha and power** - Usually 0.05 and 0.80
4. **Calculate required n** - Using formulas or software
5. **Sensitivity analysis** - What if effect is smaller than expected?
6. **Practical constraints** - Can you actually recruit this many?
7. **Adjust if needed** - Change design, accept lower power with documentation

### Common Mistakes

1. **Using observed effect from same data**: Circular reasoning
2. **Ignoring dropout**: Calculate n for completed participants, recruit more
3. **No sensitivity analysis**: What if your effect size estimate is wrong?
4. **Post-hoc power**: "We had 30% power" is meaningless after the fact
""",
    "interactive": {
        "type": "power_calculator",
        "config": {
            "show_effect_size_input": True,
            "show_alpha_beta_tradeoff": True,
            "calculate_sample_size": True,
            "show_power_curve": True,
        }
    },
    "key_takeaways": [
        "Always calculate required sample size BEFORE collecting data",
        "Effect size is the most important and uncertain input",
        "Underpowered studies waste resources and mislead",
        "80% power means 20% chance of missing a real effect",
        "Do sensitivity analysis: what if effect is smaller than expected?",
    ],
    "practice_questions": [
        {
            "question": "You expect a medium effect size (d=0.5). You want 80% power at α=0.05. Roughly how many participants per group do you need?",
            "answer": "About 64 per group. Using n ≈ 16/d² = 16/0.25 = 64",
            "hint": "Use the simplified formula n ≈ 16/d² per group"
        },
        {
            "question": "A study with n=20 per group found p=0.08 and concluded 'no effect.' What's wrong with this conclusion?",
            "answer": "The study was likely underpowered. With n=20, you can only reliably detect large effects (d≈0.8). A smaller effect might exist but be undetectable. The correct conclusion is 'inconclusive' not 'no effect.'",
            "hint": "Calculate what effect size this study was powered to detect."
        }
    ]
}

# =============================================================================
# Causal Inference Module
# =============================================================================

CONFOUNDING = {
    "id": "confounding",
    "title": "Confounding & How to Handle It",
    "content": """
## The Enemy of Causal Claims

Confounding occurs when a third variable influences both the treatment and the outcome, creating a spurious association.

### The Classic Example

**Observation:** People who carry lighters are more likely to get lung cancer.
**Naive conclusion:** Lighters cause lung cancer!
**Reality:** Smoking is a confounder.

```
    Smoking
    /    \\
   v      v
Lighters → Lung Cancer (spurious)
```

Lighters don't cause cancer. Smokers carry lighters AND get cancer.

### Formal Definition

A confounder C between treatment X and outcome Y:
1. C is associated with X (correlated with treatment)
2. C affects Y (causes outcome)
3. C is not on the causal path from X to Y (not a mediator)

### Measured vs Unmeasured Confounders

**Measured confounders:** You have data on them. Can adjust.
- Age, gender, income, etc.

**Unmeasured confounders:** You don't have data. Cannot fully adjust.
- Motivation, genetic factors, etc.

This is why observational studies can never fully prove causation.

### Adjustment Methods

#### 1. Stratification
Compare within strata where confounder is constant.
```
Overall: Drug users have worse outcomes
Within age 20-30: Drug vs no drug outcomes similar
Within age 60-70: Drug vs no drug outcomes similar
Conclusion: Age was confounding
```

#### 2. Regression
Include confounder as covariate.
```
Y = β₀ + β₁(Treatment) + β₂(Confounder) + ε
```
β₁ is treatment effect, adjusted for confounder.

#### 3. Matching
Match each treated unit with similar control unit.
```
Treated patient: Female, 65, diabetic
Matched control: Female, 67, diabetic (no treatment)
Compare outcomes
```

#### 4. Propensity Scores
Estimate P(Treatment | Confounders) for each person.
Match or weight by propensity score.

**Intuition:** Compare people who were equally likely to receive treatment.

### When Adjustment Makes Things WORSE

#### Collider Bias

A collider is caused by both X and Y:
```
    X → C ← Y
```

**Adjusting for a collider creates spurious association!**

**Example:**
- Talent → Hollywood success ← Attractiveness
- Among Hollywood stars, talent and attractiveness are negatively correlated
- This doesn't mean they're negatively correlated in the population
- Hollywood success is a collider; conditioning on it creates bias

#### Selection Bias as Collider Bias

Studying only hospitalized patients = conditioning on hospitalization.
If both disease A and disease B cause hospitalization:
```
Disease A → Hospitalized ← Disease B
```
Among hospitalized patients, diseases A and B appear negatively associated.

### Sensitivity Analysis

Since unmeasured confounding is always possible, ask:
"How strong would an unmeasured confounder need to be to explain away this effect?"

If the required confounder is implausibly strong, the finding is more credible.

### DAGs: Directed Acyclic Graphs

Draw the causal structure:
```
Smoking → Lung Cancer
   ↓           ↑
Yellow fingers  (direct)
```

DAGs help identify:
- What to adjust for (confounders)
- What NOT to adjust for (colliders, mediators)
- Whether causal identification is possible

### Red Flags for Confounding

- Observational data with "obvious" causal claims
- No mention of potential confounders
- Adjusting for everything (including potential colliders)
- Effect disappears with different adjustment sets
""",
    "interactive": {
        "type": "dag_builder",
        "config": {
            "show_confounder_detection": True,
            "show_collider_warning": True,
            "allow_adjustment_sets": True,
        }
    },
    "key_takeaways": [
        "Confounders create spurious associations that look causal",
        "Unmeasured confounders can never be fully addressed in observational data",
        "Adjusting for colliders makes things worse, not better",
        "Draw DAGs to understand what to adjust for",
        "Sensitivity analysis asks how strong a confounder would need to be",
    ],
    "practice_questions": [
        {
            "question": "Coffee drinking is associated with lower mortality in observational studies. What confounders might explain this?",
            "answer": "Many possibilities: Healthier people may drink more coffee (healthy user bias). Coffee drinkers may be more affluent (socioeconomic status). Coffee drinkers may be more social. Sicker people may avoid coffee.",
            "hint": "Think about what kind of person drinks coffee and what else affects mortality."
        },
        {
            "question": "A study adjusts for 'general health status' when looking at whether exercise prevents heart disease. What's the problem?",
            "answer": "General health status may be a collider (both exercise and genetics affect it) or a mediator (exercise improves health which prevents heart disease). Adjusting for a mediator blocks the causal path you're trying to measure. Adjusting for a collider creates bias.",
            "hint": "Draw the DAG. Where does 'general health status' fit?"
        }
    ]
}

MULTIPLE_COMPARISONS = {
    "id": "multiple-comparisons",
    "title": "Multiple Comparisons & P-Hacking",
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

# =============================================================================
# Case Studies
# =============================================================================

CASE_CLINICAL_TRIAL = {
    "id": "case-clinical-trial",
    "title": "Case: Clinical Trial Analysis",
    "content": """
## Scenario: The RECOVER Trial

You're analyzing a Phase III trial for a new rheumatoid arthritis drug. The study randomized 400 patients to drug (n=200) or placebo (n=200) for 24 weeks.

### The Data

**Primary outcome:** ACR50 response (50% improvement in symptoms)
- Drug group: 96/200 (48%) achieved ACR50
- Placebo group: 68/200 (34%) achieved ACR50

**The twist:**
- Drug group: 42 patients (21%) dropped out
- Placebo group: 18 patients (9%) dropped out

Most drug dropouts cited side effects (nausea, headache).

### Question 1: What's the Problem?

The differential dropout is concerning because:
1. Those who dropped out might have responded differently than completers
2. Drug dropouts may have been people who weren't responding AND had side effects
3. The groups are no longer balanced by randomization

### Question 2: How Should We Analyze?

**Intention-to-Treat (ITT):**
- Analyze everyone according to randomized group
- Assume dropouts = non-responders (conservative for drug)
- Drug: 96/200 = 48% → If dropouts are non-responders: 96/200 = 48%
- This preserves randomization but may underestimate effect

**Per-Protocol (PP):**
- Analyze only those who completed as assigned
- Drug: 96/158 = 61%
- Placebo: 68/182 = 37%
- Larger effect, but selection bias (completers differ)

**Composite Strategy:**
- Report ITT as primary (preserves randomization)
- Report PP as secondary (maximum effect if compliant)
- Discuss why they differ

### Question 3: Handling Missing Outcomes

For patients who dropped out, outcome is unknown. Options:

1. **Assume non-response** (ITT): Conservative, valid for regulatory approval
2. **Last observation carried forward**: Assume last status continued
3. **Multiple imputation**: Model likely outcomes based on observed patterns
4. **Sensitivity analysis**: What if all drug dropouts would have responded? What if none would have?

### Question 4: Statistical Testing

Chi-square test for ACR50 rates:
$$\\chi^2 = \\frac{(96-82)^2/82 + (104-118)^2/118 + (68-82)^2/82 + (132-118)^2/118}$$

Or calculate:
- Risk difference: 48% - 34% = 14 percentage points
- Relative risk: 48%/34% = 1.41 (41% more likely to respond)
- 95% CI for RD: [5%, 23%]
- p-value: 0.003

### Question 5: Subgroup Analyses

The sponsor wants to know: Does the drug work better for severe vs mild disease?

**Caution:** This was not prespecified.
- Severe: 52% vs 30% (n=120), p=0.008
- Mild: 44% vs 38% (n=280), p=0.28

Tempting to conclude "drug works for severe patients." But:
- Multiple comparisons: 2 tests, should adjust
- Interaction test: Is the difference in drug effect statistically significant?
- This is exploratory, not confirmatory

### Your Report Should Include

1. Primary ITT analysis with effect size and CI
2. PP analysis as secondary
3. Sensitivity analyses for different dropout assumptions
4. Clear statement that subgroup findings are exploratory
5. Discussion of dropout pattern and implications
""",
    "interactive": {
        "type": "trial_analyzer",
        "config": {
            "show_dropout_impact": True,
            "compare_itt_pp": True,
            "allow_subgroup_analysis": True,
        }
    },
    "key_takeaways": [
        "ITT analysis preserves randomization; PP analysis shows efficacy in compliers",
        "Differential dropout threatens validity - always report and address it",
        "Subgroup analyses should be prespecified or labeled exploratory",
        "Sensitivity analysis shows how conclusions depend on assumptions",
        "Report effect sizes with confidence intervals, not just p-values",
    ],
    "practice_questions": [
        {
            "question": "If we only analyze completers and find a bigger effect, is that good news for the drug?",
            "answer": "Not necessarily. Completers in the drug group may be systematically different (healthier, more tolerant of side effects) from completers in placebo group. The comparison is no longer randomized. The larger effect could be selection bias, not drug efficacy.",
            "hint": "What kind of person completes despite side effects?"
        },
        {
            "question": "The subgroup analysis suggests the drug works for severe but not mild disease. What would you tell the company?",
            "answer": "This finding is exploratory, not confirmatory. It could be a chance finding (multiple comparisons). The interaction test should be performed. A new trial specifically in severe patients would be needed to confirm. They should NOT claim efficacy only in severe patients based on this.",
            "hint": "Was this subgroup analysis prespecified?"
        }
    ]
}

CASE_AB_TEST = {
    "id": "case-ab-test",
    "title": "Case: E-commerce A/B Test",
    "content": """
## Scenario: The Checkout Flow Test

An e-commerce company tested a redesigned checkout flow. The test ran for 4 weeks with 50,000 users per variant.

### The Data

| Metric | Control | Treatment | Difference |
|--------|---------|-----------|------------|
| Conversion rate | 3.2% | 3.5% | +9.4% relative |
| Average order value | $85 | $82 | -3.5% |
| Revenue per user | $2.72 | $2.87 | +5.5% |

Statistical significance:
- Conversion: p = 0.02
- AOV: p = 0.08
- Revenue per user: p = 0.04

### Question 1: Which Metric Matters?

The team is debating:
- Product manager: "Conversion is up significantly!"
- Finance: "AOV is down, that's concerning"
- CEO: "Just tell me about revenue"

**The right answer:** Revenue per user is the primary metric.
- Conversion alone can be gamed (attract low-value customers)
- AOV alone ignores customer acquisition
- Revenue per user = Conversion × AOV = what actually matters

### Question 2: Is There a Novelty Effect?

Week-by-week data:

| Week | Control | Treatment | Difference |
|------|---------|-----------|------------|
| 1 | 3.1% | 3.9% | +26% |
| 2 | 3.2% | 3.6% | +13% |
| 3 | 3.3% | 3.4% | +3% |
| 4 | 3.2% | 3.3% | +3% |

**Problem:** The effect is decaying. By week 4, it's much smaller.

**Diagnosis:** Novelty effect. Users explored the new design, inflating early conversion.

**Implication:** The true long-term effect is ~3%, not 9.4%.

### Question 3: Segment Analysis

| Segment | Control | Treatment | Difference |
|---------|---------|-----------|------------|
| New users | 2.1% | 2.8% | +33% |
| Returning | 4.3% | 4.2% | -2% |

**Interpretation:**
- New users: Never saw old design, no novelty effect → real improvement
- Returning users: Confused by change → slight negative effect

**This is actually informative:**
- New design IS better (for new users)
- But transition cost for existing users
- Long-term prediction: positive (as more users are "new" to the design)

### Question 4: Multiple Testing Issues

The team ran tests on:
- Conversion rate
- Average order value
- Revenue per user
- Cart abandonment rate
- Time to purchase
- Mobile vs desktop
- New vs returning
- 5 geographic regions

That's 13+ comparisons. At least one false positive expected.

**The fix:**
- Prespecify ONE primary metric (revenue per user)
- Prespecify key segments (new vs returning)
- Label everything else as exploratory
- Apply correction if making claims about secondary metrics

### Question 5: Sample Ratio Mismatch

Let's check: Were users properly randomized?

| Variant | Expected | Observed |
|---------|----------|----------|
| Control | 50,000 | 50,127 |
| Treatment | 50,000 | 49,873 |

Chi-square test: p = 0.12

This looks fine (within expected variation).

**If it weren't:** Sample ratio mismatch indicates a bug in randomization → Results are invalid.

### Your Recommendation

"The new checkout flow shows a +5.5% improvement in revenue per user (p=0.04, 95% CI: [0.5%, 10.5%]).

However, there's clear evidence of a novelty effect: the benefit decreased from +26% in week 1 to +3% in week 4. Long-term effect is likely around +3%.

Segment analysis suggests the benefit is concentrated in new users (+33%) while returning users show slight negative (-2%). This is consistent with novelty effect interpretation.

Recommendation: Ship the change, but monitor metrics for the next month to confirm long-term effect stabilizes around +3%. The positive effect on new users suggests this is a real improvement, not just novelty."
""",
    "interactive": {
        "type": "ab_analyzer",
        "config": {
            "show_novelty_detection": True,
            "show_segment_breakdown": True,
            "show_sample_ratio_check": True,
        }
    },
    "key_takeaways": [
        "Revenue per user is usually the right primary metric",
        "Check for novelty effects by examining time trends",
        "Segment analysis reveals who's driving the effect",
        "Prespecify primary metric and key segments",
        "Check sample ratio to detect randomization bugs",
    ],
    "practice_questions": [
        {
            "question": "Conversion is up 9.4% but AOV is down 3.5%. Should we ship the new design?",
            "answer": "Look at revenue per user: $2.87 vs $2.72 = +5.5%. Net positive, so probably ship. But investigate WHY AOV dropped - are we attracting lower-value customers or just converting customers who would have bought less anyway?",
            "hint": "What metric captures the full picture?"
        },
        {
            "question": "Week 1 showed +26% lift but week 4 showed +3%. Which is the true effect?",
            "answer": "Week 4 is closer to the true long-term effect. Week 1 was inflated by novelty (users exploring the new design). Ideally, run longer and wait for stabilization. The 9.4% overall is an overestimate.",
            "hint": "What causes the decay pattern?"
        }
    ]
}

# =============================================================================
# DSW Mastery Module
# =============================================================================

DSW_OVERVIEW = {
    "id": "dsw-overview",
    "title": "DSW Architecture & Workflow",
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
    "interactive": {
        "type": "dsw_demo",
        "config": {}
    },
    "key_takeaways": [
        "DSW provides guardrails against common statistical mistakes",
        "Always review the validation report before trusting results",
        "The plain-language interpretation is a starting point, not gospel",
        "Export includes methodology for reproducibility",
    ],
}

BAYESIAN_AB_HANDS_ON = {
    "id": "bayesian-ab-hands-on",
    "title": "Bayesian A/B Testing",
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
    "interactive": {
        "type": "ab_analyzer",
        "config": {}
    },
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
            "hint": "There's no universal threshold - it depends on context."
        },
        {
            "question": "Why might you use a skeptical prior?",
            "answer": "To require stronger evidence before believing in large effects. Most A/B tests show small effects; a skeptical prior encodes this expectation and protects against being fooled by noise.",
            "hint": "Think about base rates of effect sizes in your domain."
        }
    ]
}

SPC_HANDS_ON = {
    "id": "spc-hands-on",
    "title": "SPC & Control Charts",
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
""",
    "interactive": {
        "type": "spc_demo",
        "config": {}
    },
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
            "hint": "Cp ignores centering, Cpk accounts for it."
        },
        {
            "question": "A point just outside the control limit could be random chance. Why do we investigate anyway?",
            "answer": "Control limits are set at 3σ, so random chance outside limits should occur only 0.3% of the time. While false alarms happen, the cost of missing a real problem usually exceeds the cost of investigation.",
            "hint": "What's the probability of a point beyond 3σ by chance?"
        }
    ]
}

# =============================================================================
# Data Fundamentals - Additional Content
# =============================================================================

DISTRIBUTIONS = {
    "id": "distributions",
    "title": "Distributions & Transformations",
    "content": """
## Understanding Data Shapes

The distribution of your data matters because:
- It determines which statistical tests are valid
- It reveals potential data quality issues
- It guides how to summarize and visualize

### The Normal Distribution

The bell curve. Mean = median = mode. Symmetric.

$$f(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}} e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}$$

**Why it matters:** Many statistical tests assume normality (or approximate normality).

**Where it comes from:** Central Limit Theorem - averages of many independent things tend toward normal.

### Checking for Normality

**Visual methods:**
- Histogram: Bell-shaped?
- Q-Q plot: Points follow diagonal line?

**Statistical tests:**
- Shapiro-Wilk (n < 50)
- Kolmogorov-Smirnov (n > 50)

**Warning:** With large n, tests reject even minor deviations. Visual check is often better.

### Common Non-Normal Distributions

| Shape | Common in | Example |
|-------|-----------|---------|
| Right-skewed | Income, time data | Response times |
| Left-skewed | Test scores near ceiling | Easy exam results |
| Bimodal | Mixed populations | Height (if mixing M/F) |
| Uniform | Random selection | Lottery numbers |
| Heavy-tailed | Finance, internet | Stock returns |

### Skewness and Kurtosis

**Skewness:** Asymmetry
- Positive: Right tail longer (mean > median)
- Negative: Left tail longer (mean < median)

**Kurtosis:** Tail heaviness
- High: More extreme values than normal
- Low: Fewer extreme values than normal

### When Normality Matters

**Tests that assume normality:**
- t-test (but robust to violations if n > 30)
- ANOVA (robust with equal group sizes)
- Pearson correlation

**Tests that don't:**
- Mann-Whitney, Kruskal-Wallis
- Spearman correlation
- Bootstrapped methods

### Transformations

When data is non-normal, sometimes transforming helps:

| Transformation | When to Use |
|----------------|-------------|
| Log | Right-skewed positive data |
| Square root | Count data, right-skewed |
| Box-Cox | Find optimal transformation |
| Rank | Severe non-normality |

**Example:** Income is heavily right-skewed. Log(income) is often approximately normal.

**Caution:** Interpret results on transformed scale, or back-transform carefully.

### When NOT to Transform

- If you need to interpret coefficients directly
- If the relationship is already linear
- If outliers are real and important
- If transforming makes less conceptual sense
""",
    "interactive": {
        "type": "distribution_explorer",
        "config": {}
    },
    "key_takeaways": [
        "Check distribution shape before choosing statistical tests",
        "Large samples make normality less critical (CLT)",
        "Visual checks often better than statistical tests for normality",
        "Transform only when it helps and interpretation remains sensible",
        "Non-parametric tests are safer when distribution is unclear",
    ],
}

EDA = {
    "id": "eda",
    "title": "Exploratory Data Analysis",
    "content": """
## The Art of Looking at Data

EDA is about understanding your data before modeling. It's hypothesis-generating, not hypothesis-testing.

### The EDA Mindset

1. **Curiosity without commitment:** Explore patterns without falling in love
2. **Visual first:** Plots reveal what summaries hide
3. **Document everything:** So you don't fool yourself later
4. **Separate from testing:** EDA findings need confirmation

### The EDA Workflow

#### 1. Univariate Exploration

For each variable:
- **Numeric:** Mean, median, SD, range, distribution shape
- **Categorical:** Counts, proportions, rare categories

```
Key questions:
- What's the center and spread?
- Are there outliers?
- What's the distribution shape?
- Are there impossible values?
- How much is missing?
```

#### 2. Bivariate Exploration

For pairs of variables:
- **Numeric vs Numeric:** Scatter plot, correlation
- **Numeric vs Categorical:** Box plots by group
- **Categorical vs Categorical:** Cross-tabulation

```
Key questions:
- Is there a relationship?
- Is it linear?
- Are there subgroups with different patterns?
- Are there influential points?
```

#### 3. Multivariate Exploration

- Correlation matrices
- Pair plots
- Faceted visualizations
- Dimension reduction (PCA) for many variables

### EDA Visualization Best Practices

| Data Type | Good Plot | Avoid |
|-----------|-----------|-------|
| Distribution | Histogram, density | Pie chart |
| Relationship | Scatter with trend | 3D plots |
| Comparison | Box plot, dot plot | Stacked bars |
| Time series | Line chart | Area charts (stacked) |

### The EDA vs Hypothesis Testing Boundary

**EDA is:** Looking for patterns, generating ideas, finding surprises

**EDA is not:** Statistical testing, confirmation, publication-ready analysis

**The bright line:** Once you've explored and formed a hypothesis, you need fresh data (or preregistration) to test it. Testing on the same data you explored is cheating.

### Documenting EDA

Keep notes on:
- What you looked at
- What patterns you noticed
- What you decided not to pursue
- Any data issues discovered

This prevents:
- HARKing (Hypothesizing After Results Known)
- Forgetting what you already tried
- Selective reporting of interesting patterns
""",
    "interactive": {
        "type": "eda_explorer",
        "config": {}
    },
    "key_takeaways": [
        "EDA generates hypotheses; it doesn't confirm them",
        "Visual inspection catches what summary statistics miss",
        "Document your exploration to prevent fooling yourself",
        "Patterns found in EDA need fresh data to confirm",
        "The EDA/testing boundary prevents p-hacking",
    ],
}

NATURAL_EXPERIMENTS = {
    "id": "natural-experiments",
    "title": "Natural Experiments",
    "content": """
## Finding Causation in Observational Data

When you can't randomize, nature sometimes provides quasi-random assignment.

### What Makes a Natural Experiment Valid?

1. **As-if random assignment:** The treatment/control split wasn't chosen based on outcomes
2. **No manipulation:** Assignment happened independently of the research
3. **Clear comparison:** Treatment and control groups are identifiable

### Instrumental Variables (IV)

An instrument Z is:
- Correlated with treatment X
- Affects outcome Y only through X (exclusion restriction)
- Not correlated with confounders

**Classic example:** Draft lottery and education
- Random draft number (Z) affects military service (X)
- Military service affects earnings (Y)
- Draft number itself doesn't affect earnings except through service

### Regression Discontinuity (RD)

When treatment is assigned based on a cutoff:
- Students scoring ≥70 get scholarship
- Compare students just below vs just above 70
- They're nearly identical except for treatment

**Key assumption:** No manipulation of the running variable near the cutoff.

**Example:** Drinking age and mortality
- Compare mortality just below vs above age 21
- Sharp increase at 21 suggests causal effect of legal drinking

### Difference-in-Differences (DiD)

Compare changes over time:
- Treatment group: Before vs after
- Control group: Before vs after
- DiD = (Treatment After - Before) - (Control After - Before)

**Key assumption:** Parallel trends - groups would have moved together without treatment

**Example:** Minimum wage and employment
- New Jersey raised minimum wage
- Compare NJ fast food employment before/after
- Use Pennsylvania (no change) as control
- DiD estimates causal effect of minimum wage

### Evaluating Natural Experiments

| Question | Why It Matters |
|----------|----------------|
| Was assignment really as-if random? | Confounding |
| Is the exclusion restriction plausible? | IV validity |
| Were there parallel trends? | DiD validity |
| Could people manipulate the cutoff? | RD validity |
| How local is the estimate? | Generalizability |

### Limitations

Natural experiments give:
- **Local** effects (at the cutoff, for the compliers)
- **Specific context** results
- Often weaker than RCTs

But they're far better than naive observational comparisons.
""",
    "interactive": {
        "type": "natural_experiment_demo",
        "config": {}
    },
    "key_takeaways": [
        "Natural experiments exploit as-if random variation",
        "IV requires an exclusion restriction that's often questionable",
        "RD compares just above and below a threshold",
        "DiD requires parallel trends assumption",
        "Natural experiments give local estimates, not universal effects",
    ],
}

# =============================================================================
# Content Registry
# =============================================================================

SECTION_CONTENT = {
    # Foundations
    "bayesian-thinking": BAYESIAN_THINKING,
    "base-rate-neglect": BASE_RATE_NEGLECT,
    "hypothesis-driven": HYPOTHESIS_DRIVEN,
    "evidence-quality": EVIDENCE_QUALITY,
    "regression-to-mean": REGRESSION_TO_MEAN,

    # Experimental Design
    "randomization-controls": RANDOMIZATION_CONTROLS,
    "power-analysis": POWER_ANALYSIS,

    # Data Fundamentals
    "data-cleaning": DATA_CLEANING,
    "sampling": SAMPLING,
    "distributions": DISTRIBUTIONS,
    "eda": EDA,

    # Statistical Inference
    "choosing-tests": CHOOSING_TESTS,
    "interpreting-results": INTERPRETING_RESULTS,
    "multiple-comparisons": MULTIPLE_COMPARISONS,

    # Causal Inference
    "confounding": CONFOUNDING,
    "natural-experiments": NATURAL_EXPERIMENTS,

    # DSW Mastery
    "dsw-overview": DSW_OVERVIEW,
    "bayesian-ab-hands-on": BAYESIAN_AB_HANDS_ON,
    "spc-hands-on": SPC_HANDS_ON,

    # Case Studies
    "case-clinical-trial": CASE_CLINICAL_TRIAL,
    "case-ab-test": CASE_AB_TEST,
}


def get_section_content(section_id: str) -> dict:
    """Get content for a section."""
    return SECTION_CONTENT.get(section_id, {})


def get_all_topics() -> list:
    """Get all topics across all sections for search."""
    topics = []
    for section_id, content in SECTION_CONTENT.items():
        if "key_takeaways" in content:
            topics.extend(content["key_takeaways"])
    return topics
