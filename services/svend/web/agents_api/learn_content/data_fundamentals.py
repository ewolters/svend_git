"""Learning content: Data Fundamentals."""

from ._datasets import SHARED_DATASET  # noqa: F401


DATA_CLEANING = {
    "id": "data-cleaning",
    "title": "Data Cleaning",
    "intro": "No statistical technique can rescue bad data. In this section, you'll diagnose missing data patterns, detect outliers, and choose the right cleaning strategy before analysis.",
    "exercise": {
        "title": "Try It: Diagnose a Messy Dataset",
        "steps": [
            "Review the missing data summary for each column",
            "Classify each missing pattern as MCAR, MAR, or MNAR",
            "Select a handling strategy for each column",
            "Run the outlier detection and inspect flagged values",
            "Document your cleaning decisions"
        ],
        "dsw_type": "stats:descriptive",
        "dsw_config": {},
    },
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
    ],
    "practice_questions": [
        {
            "question": "A survey has 15% missing income data. Higher earners are less likely to respond. What type of missingness is this, and what imputation strategy is best?",
            "answer": "MNAR — missingness depends on the value itself (high income). No imputation fully corrects MNAR. Multiple imputation with auxiliary variables (education, job title) can reduce bias but not eliminate it. Report the limitation.",
            "hint": "Ask: does the probability of being missing depend on the missing value itself?"
        },
        {
            "question": "You find a data point where a patient's age is listed as 3 but their occupation is 'engineer'. What should you do?",
            "answer": "This is a cross-field validation failure. Investigate: likely a data entry error (maybe 30 or 33). Check the source record. Don't auto-delete or auto-impute — document the decision either way.",
            "hint": "Cross-field consistency checks catch errors that univariate checks miss"
        },
    ]
}


SAMPLING = {
    "id": "sampling",
    "title": "Sampling",
    "intro": "You can't measure everyone. Sampling lets you draw conclusions from subsets, but only if done right. In this exercise, you'll calculate how many observations you actually need.",
    "exercise": {
        "title": "Try It: Calculate Sample Size",
        "steps": [
            "Set confidence level to 95%",
            "Set margin of error to 3%",
            "Leave population size at the default (large)",
            "Note the required sample size",
            "Change margin of error to 5% and see how n drops"
        ],
        "dsw_type": "stats:descriptive",
        "dsw_config": {},
    },
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
    ],
    "practice_questions": [
        {
            "question": "You want to estimate customer satisfaction within ±4% margin at 95% confidence. You have no prior data. How many customers do you need?",
            "answer": "n = (1.96² × 0.5 × 0.5) / 0.04² = 0.9604 / 0.0016 ≈ 601 customers. We use p=0.5 (worst case) since we have no prior estimate.",
            "hint": "Use the proportion sample size formula with p=0.5 for maximum conservatism"
        },
        {
            "question": "A company surveys users by emailing a link. Response rate is 8%. The survey finds 92% satisfaction. Should management trust this number?",
            "answer": "No — severe non-response bias. The 8% who respond are likely more engaged/satisfied. The 92% who didn't respond may have very different satisfaction levels. This is self-selection bias. Need to compare responders to non-responders on observable characteristics, or use follow-up methods to reach non-responders.",
            "hint": "Who chose not to respond, and why?"
        },
    ]
}


DISTRIBUTIONS = {
    "id": "distributions",
    "title": "Distributions & Transformations",
    "intro": "The shape of your data determines which tests are valid. You'll explore histograms, Q-Q plots, and normality tests, then practice transforming skewed data to meet assumptions.",
    "exercise": {
        "title": "Try It: Explore Data Shapes",
        "steps": [
            "Load the sample data and view the histogram",
            "Check the Q-Q plot for departures from normality",
            "Note the skewness and kurtosis values",
            "Apply a log transformation and re-check the Q-Q plot",
            "Compare the Shapiro-Wilk p-value before and after"
        ],
        "dsw_type": "stats:normality",
        "dsw_config": {"var": "diameter_mm"},
    },
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
    "practice_questions": [
        {
            "question": "Your data has a skewness of 2.1 and you plan to run a t-test with n=15 per group. Should you be concerned?",
            "answer": "Yes — very concerned. Skewness of 2.1 is heavily right-skewed, and with only n=15 per group, the CLT hasn't kicked in to normalize the sampling distribution. Options: (1) log-transform the data and check if skewness drops, (2) use Mann-Whitney U instead, (3) bootstrap the confidence interval.",
            "hint": "With small n, normality assumption matters more"
        },
        {
            "question": "A Shapiro-Wilk test on your data (n=5000) gives p<0.001. Does this mean you can't use parametric tests?",
            "answer": "No. With n=5000, Shapiro-Wilk detects trivial departures from normality. Check the Q-Q plot visually instead. If the data looks roughly bell-shaped with minor deviations, parametric tests are fine — they're robust with large n. This is a case where the visual check is more informative than the statistical test.",
            "hint": "With large n, normality tests reject even negligible deviations"
        },
    ]
}


EDA = {
    "id": "eda",
    "title": "Exploratory Data Analysis",
    "intro": "EDA is detective work: look at the data before you model it. You'll work through a structured exploration, from univariate summaries to bivariate relationships, and document what you find.",
    "exercise": {
        "title": "Try It: Explore a Dataset",
        "steps": [
            "Review univariate summaries for each variable",
            "Examine scatter plots and correlations between variable pairs",
            "Identify any outliers or unexpected patterns",
            "Note which findings are interesting but need confirmation",
            "Write down one hypothesis generated by your exploration"
        ],
        "dsw_type": "stats:descriptive",
        "dsw_config": {},
    },
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
    "practice_questions": [
        {
            "question": "During EDA you discover that customers who bought product A also tend to buy product B (correlation = 0.45). Your manager asks you to report this as a finding. What do you say?",
            "answer": "This is an EDA observation, not a confirmed finding. It should be treated as a hypothesis to test, not a conclusion. To confirm: (1) preregister the hypothesis, (2) test on held-out data or a new sample, (3) consider confounders (maybe both products appeal to the same demographic). Report it as 'pattern observed in exploratory analysis — requires confirmation.'",
            "hint": "EDA generates hypotheses; confirmation requires separate testing"
        },
        {
            "question": "You plot revenue by month and see a spike in March. Is this meaningful or noise?",
            "answer": "Can't tell from one data point. Check: (1) Is there a March spike in previous years? (seasonal pattern), (2) Was there a known event (promotion, holiday)? (3) How variable is month-to-month revenue normally? If the spike is within normal month-to-month variation, it's likely noise. If it's 3+ SD above the monthly mean, investigate further. One observation is never enough to conclude.",
            "hint": "A single unusual point needs context — compare to baseline variability"
        },
    ]
}


