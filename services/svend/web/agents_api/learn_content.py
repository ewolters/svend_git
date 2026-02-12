"""
Learning Module Content

Interactive learning content for SVEND — learn by doing.
Each section has structured content with explanations, examples, and interactive elements.
"""


# =============================================================================
# Shared Dataset — referenced across modules
# =============================================================================

SHARED_DATASET = {
    "name": "Widget Manufacturing Quality Data",
    "description": (
        "A plastic widget factory with 3 production lines (A, B, C) and 2 shifts "
        "(day, night). 200 parts measured for diameter (target: 25.00mm, tolerance: "
        "\u00b10.15mm), weight, surface roughness, and defect status. Line C has "
        "slightly worse quality (mean shifted +0.03mm). Night shift has slightly "
        "higher variance."
    ),
    "diameter_mm": [
        25.02, 24.99, 25.03, 25.08, 24.99, 24.99, 25.08, 25.05, 24.98, 25.03,
        24.97, 24.98, 25.01, 24.89, 24.91, 24.97, 24.94, 25.02, 24.95, 24.92,
        25.07, 24.99, 25.0, 24.93, 24.97, 25.01, 24.94, 25.02, 24.96, 24.99,
        24.97, 25.11, 25.0, 24.95, 25.05, 24.94, 25.01, 24.88, 24.93, 25.01,
        25.04, 25.01, 24.99, 24.98, 24.93, 24.96, 24.97, 25.05, 25.02, 24.89,
        25.02, 24.98, 24.96, 25.03, 25.05, 25.06, 24.96, 24.98, 25.02, 25.05,
        24.98, 24.99, 24.94, 24.94, 25.05, 25.07, 25.0, 25.06, 25.02, 24.97,
        25.02, 25.08, 25.0, 25.09, 24.87, 25.04, 25.01, 24.99, 25.0, 24.88,
        24.99, 25.02, 25.09, 24.97, 24.96, 24.97, 25.05, 25.02, 24.97, 25.03,
        25.0, 25.06, 24.96, 24.98, 24.98, 24.93, 25.01, 25.02, 25.0, 24.99,
        24.92, 24.98, 24.98, 24.95, 24.99, 25.02, 25.11, 25.01, 25.01, 25.0,
        24.9, 25.0, 25.0, 25.12, 24.99, 25.02, 25.0, 24.94, 25.07, 25.04,
        25.04, 24.95, 25.07, 24.93, 25.04, 25.11, 24.95, 24.97, 25.0, 24.97,
        24.91, 25.0, 24.95, 25.03, 24.98, 25.11, 24.98, 25.01, 25.07, 24.96,
        25.04, 25.1, 24.93, 25.04, 25.04, 25.08, 24.97, 24.96, 25.06, 25.04,
        25.04, 25.05, 25.0, 25.04, 25.05, 24.99, 25.12, 25.06, 24.97, 25.06,
        24.97, 25.07, 25.09, 24.98, 25.08, 25.05, 25.08, 25.12, 25.02, 24.98,
        24.99, 24.99, 25.03, 25.05, 25.04, 25.08, 25.03, 25.1, 25.01, 25.17,
        25.06, 24.98, 24.98, 25.05, 25.02, 25.07, 25.05, 25.03, 24.99, 24.95,
        25.0, 25.07, 25.04, 24.96, 25.04, 25.05, 24.98, 25.04, 25.03, 24.96,
    ],
    "weight_g": [
        12.61, 12.7, 12.82, 12.82, 12.0, 12.22, 12.65, 12.68, 12.65, 13.66,
        12.71, 12.84, 12.79, 12.73, 12.41, 12.73, 12.22, 12.43, 12.35, 12.53,
        13.19, 11.94, 12.75, 12.02, 12.36, 12.89, 12.52, 12.18, 12.24, 12.7,
        12.28, 12.58, 12.51, 12.3, 13.27, 12.69, 11.89, 12.57, 12.3, 12.76,
        12.21, 12.47, 12.65, 12.81, 12.14, 12.4, 12.33, 12.3, 13.03, 12.65,
        12.12, 12.78, 13.26, 12.81, 12.04, 12.33, 12.88, 12.29, 12.66, 12.73,
        12.22, 12.48, 11.53, 12.19, 12.41, 12.13, 12.99, 11.99, 12.37, 12.54,
        13.02, 12.07, 12.85, 12.5, 12.21, 12.64, 12.57, 12.32, 12.52, 12.36,
        12.53, 12.7, 13.07, 12.13, 13.14, 11.8, 12.45, 12.68, 12.6, 12.31,
        12.44, 12.32, 12.32, 12.75, 12.63, 12.29, 12.77, 12.61, 12.74, 12.69,
        12.2, 12.33, 12.72, 12.72, 12.49, 12.54, 12.96, 12.32, 12.66, 12.43,
        12.43, 12.83, 12.8, 12.74, 12.89, 12.51, 12.7, 12.41, 12.62, 12.46,
        12.53, 12.71, 12.25, 13.13, 12.14, 12.14, 12.85, 12.78, 12.69, 12.69,
        12.5, 12.23, 12.52, 12.26, 12.79, 12.46, 12.2, 12.4, 12.62, 12.3,
        12.25, 12.57, 12.59, 12.35, 12.36, 12.58, 12.07, 12.08, 12.24, 12.44,
        12.59, 13.03, 12.76, 12.45, 12.49, 12.2, 12.49, 12.4, 12.6, 12.25,
        12.69, 12.96, 12.47, 12.64, 12.71, 12.38, 12.58, 12.5, 12.53, 12.22,
        12.51, 12.65, 13.02, 12.79, 13.15, 12.22, 12.76, 12.56, 13.29, 12.26,
        12.25, 12.28, 11.86, 12.34, 12.23, 12.55, 12.6, 13.18, 12.79, 12.33,
        12.18, 12.65, 12.1, 13.16, 12.85, 12.36, 11.88, 12.91, 12.47, 12.95,
    ],
    "roughness_ra": [
        1.0, 1.26, 1.49, 1.51, 1.31, 1.74, 1.14, 1.43, 1.54, 1.7,
        1.83, 1.13, 1.02, 2.15, 1.62, 1.24, 2.33, 1.54, 2.0, 1.52,
        2.5, 2.31, 1.39, 1.9, 1.75, 2.21, 1.17, 1.77, 2.02, 0.96,
        1.11, 0.83, 1.39, 1.78, 2.3, 1.52, 2.24, 1.0, 0.97, 1.47,
        1.67, 1.48, 0.89, 1.45, 1.08, 1.76, 1.66, 1.18, 1.31, 1.1,
        1.47, 1.89, 1.12, 1.69, 1.31, 1.19, 1.45, 1.15, 1.27, 1.11,
        2.44, 1.51, 1.25, 1.57, 1.44, 1.41, 1.74, 1.85, 1.31, 1.29,
        1.38, 0.84, 1.02, 2.21, 2.25, 1.4, 1.76, 1.61, 3.22, 2.06,
        1.44, 1.17, 0.94, 1.57, 1.23, 0.99, 1.27, 1.14, 2.42, 1.86,
        1.49, 2.28, 1.52, 1.2, 2.31, 1.71, 1.15, 1.41, 1.2, 1.06,
        1.95, 2.4, 1.05, 1.75, 1.27, 1.32, 1.26, 1.2, 1.51, 1.17,
        1.6, 1.47, 1.39, 1.19, 1.29, 1.85, 1.69, 1.17, 1.54, 1.8,
        0.98, 1.74, 1.26, 1.72, 1.2, 0.95, 0.99, 1.51, 1.59, 1.19,
        1.79, 0.98, 1.47, 1.05, 1.33, 1.59, 1.22, 1.42, 2.02, 1.33,
        1.93, 1.18, 1.83, 2.25, 0.85, 1.25, 1.81, 1.49, 1.74, 1.35,
        1.6, 1.5, 2.1, 1.67, 1.73, 1.41, 1.39, 1.38, 1.73, 1.41,
        1.7, 2.63, 1.95, 1.43, 2.12, 1.42, 0.87, 1.22, 0.98, 1.42,
        1.58, 2.38, 1.72, 1.48, 1.93, 0.83, 1.66, 1.9, 1.03, 2.09,
        1.71, 1.39, 1.84, 2.77, 1.65, 1.67, 1.4, 1.23, 1.93, 1.27,
        1.6, 1.39, 1.77, 1.73, 2.03, 1.38, 1.45, 1.23, 1.4, 1.75,
    ],
    "line": (
        ["A"] * 67 + ["B"] * 67 + ["C"] * 66
    ),
    "shift": [
        "day", "night", "day", "day", "night", "day", "day", "night", "day",
        "day", "night", "day", "day", "night", "day", "day", "night", "day",
        "day", "night", "day", "day", "night", "day", "day", "night", "day",
        "day", "night", "day", "day", "night", "day", "day", "night", "day",
        "day", "night", "day", "day", "night", "day", "day", "night", "day",
        "day", "night", "day", "day", "night", "day", "day", "night", "day",
        "day", "night", "day", "day", "night", "day", "day", "night", "day",
        "day", "night", "day", "day", "night", "day", "day", "night", "day",
        "day", "night", "day", "day", "night", "day", "day", "night", "day",
        "day", "night", "day", "day", "night", "day", "day", "night", "day",
        "day", "night", "day", "day", "night", "day", "day", "night", "day",
        "day", "night", "day", "day", "night", "day", "day", "night", "day",
        "day", "night", "day", "day", "night", "day", "day", "night", "day",
        "day", "night", "day", "day", "night", "day", "day", "night", "day",
        "day", "night", "day", "day", "night", "day", "day", "night", "day",
        "day", "night", "day", "day", "night", "day", "day", "night", "day",
        "day", "night", "day", "day", "night", "day", "day", "night", "day",
        "day", "night", "day", "day", "night", "day", "day", "night", "day",
        "day", "night", "day", "day", "night", "day", "day", "night", "day",
        "day", "night", "day", "day", "night", "day", "day", "night", "day",
        "day", "night", "day", "day", "night", "day", "day", "night", "day",
        "day", "night", "day", "day", "night", "day", "day", "night", "day",
        "day", "night",
    ],
    "defect": [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    ],
}

# =============================================================================
# Foundations Module
# =============================================================================

BAYESIAN_THINKING = {
    "id": "bayesian-thinking",
    "title": "Bayesian Thinking",
    "intro": "Your gut has beliefs. Bayes' theorem tells you how to update them with evidence. In this section, you'll learn to quantify confidence and adjust it as new data arrives.",
    "exercise": {
        "title": "Try It: Update a Belief",
        "steps": [
            "Set a prior probability (your initial belief) to 30%",
            "Enter a likelihood ratio of 5 for your evidence",
            "Set confidence in the evidence to 80%",
            "Observe how the posterior probability updates",
            "Try extreme priors (5% and 95%) to see how they resist evidence"
        ],
        "dsw_type": "stats:descriptive",
        "dsw_config": {},
    },
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
    "intro": "Most people look at data and then explain what they see. That's backwards. In this exercise, you'll practice generating competing hypotheses before touching any data.",
    "exercise": {
        "title": "Try It: Build a Hypothesis Set",
        "steps": [
            "Enter a scenario: your website conversion rate dropped 20%",
            "Add at least 3 competing hypotheses before seeing data",
            "For each hypothesis, define what evidence would support or refute it",
            "Assign initial probabilities that sum to 100%",
            "Enter one piece of evidence and watch probabilities update"
        ],
        "dsw_type": "stats:ttest",
        "dsw_config": {"var1": "diameter_mm", "mu": 25.0},
    },
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
    "intro": "Not all evidence deserves equal weight. A randomized trial and a blog anecdote are both 'evidence' but vastly different in quality. You'll learn to assign confidence scores that reflect this.",
    "exercise": {
        "title": "Try It: Rate Evidence Quality",
        "steps": [
            "Select a source type (e.g., observational study)",
            "Set the sample size to 50",
            "Mark replication status as single study",
            "Review the calculated confidence score",
            "Change to a replicated RCT with n=500 and compare"
        ],
        "dsw_type": "stats:correlation",
        "dsw_config": {"vars": ["diameter_mm", "weight_g", "roughness_ra"]},
    },
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

# =============================================================================
# Statistical Tools Module
# =============================================================================

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

# =============================================================================
# New Foundations Content
# =============================================================================

BASE_RATE_NEGLECT = {
    "id": "base-rate-neglect",
    "title": "Base Rate Neglect",
    "intro": "A 99% accurate test sounds foolproof, but if the condition is rare, most positives are false. You'll calculate real-world probabilities and see why base rates dominate.",
    "exercise": {
        "title": "Try It: Calculate the True Positive Rate",
        "steps": [
            "Set disease prevalence (base rate) to 1%",
            "Set test sensitivity to 99% and specificity to 99%",
            "Read the natural frequency table for 10,000 people",
            "Note the positive predictive value",
            "Change prevalence to 10% and see how it transforms the result"
        ],
        "dsw_type": "stats:descriptive",
        "dsw_config": {},
    },
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
    "intro": "Extreme results rarely repeat. That's not a force; it's math. You'll simulate this effect and learn to stop attributing it to interventions that had nothing to do with the change.",
    "exercise": {
        "title": "Try It: Watch Regression in Action",
        "steps": [
            "Set the correlation between Time 1 and Time 2 to 0.5",
            "Observe how extreme scores regress toward the mean",
            "Increase correlation to 0.9 and see less regression",
            "Decrease correlation to 0.2 and see more regression",
            "Identify the lesson: extreme selection guarantees regression"
        ],
        "dsw_type": "stats:regression",
        "dsw_config": {"response": "diameter_mm", "predictors": ["weight_g"]},
    },
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
    "intro": "Randomization is the single most powerful tool for proving causation. You'll see how random assignment balances both measured and unmeasured confounders, and why alternatives always fall short.",
    "exercise": {
        "title": "Try It: Simulate Randomization",
        "steps": [
            "Run a random group assignment for 100 participants",
            "Check baseline balance between treatment and control",
            "Enable stratification by age and re-randomize",
            "Compare balance with and without stratification",
            "Observe how confounding disappears with proper randomization"
        ],
        "dsw_type": "stats:anova",
        "dsw_config": {"response": "diameter_mm", "factor": "line"},
    },
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
    "intro": "Running an experiment without power analysis is like driving blindfolded. You'll calculate exactly how many participants you need to detect a meaningful effect before collecting a single data point.",
    "exercise": {
        "title": "Try It: Size Your Experiment",
        "steps": [
            "Set expected effect size to medium (d=0.5)",
            "Set alpha to 0.05 and power to 0.80",
            "Read the required sample size per group",
            "Change effect size to small (d=0.2) and see n explode",
            "Adjust power to 0.90 and note the additional cost"
        ],
        "dsw_type": "stats:descriptive",
        "dsw_config": {},
    },
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

CAUSAL_THINKING = {
    "id": "causal-thinking",
    "title": "Causal Thinking",
    "intro": "Correlation is not causation, but causation does create correlation. You'll draw causal diagrams (DAGs) and learn to distinguish confounders from colliders, which completely changes what you should control for.",
    "exercise": {
        "title": "Try It: Draw a Causal DAG",
        "steps": [
            "Add three variables: Treatment, Outcome, and a Confounder",
            "Draw arrows showing assumed causal relationships",
            "Identify the backdoor path through the confounder",
            "Add a collider and observe the warning when you condition on it"
        ],
        "dsw_type": "stats:regression",
        "dsw_config": {"response": "diameter_mm", "predictors": ["weight_g", "roughness_ra"]},
    },
    "content": """
## The Framework for Causal Questions

Most questions worth asking are causal: Does this drug work? Did this marketing campaign increase sales? Will changing this process reduce defects? Answering them requires moving beyond correlation.

### Correlation vs Causation: The Full Picture

The mantra "correlation is not causation" is incomplete. It should be: "correlation is not causation, but causation does create correlation."

Three reasons for correlation without causation:
1. **Confounding:** A third variable causes both
2. **Reverse causation:** The effect causes the cause
3. **Selection/collider bias:** Conditioning on an effect of both

**Example:** Ice cream sales correlate with drowning deaths. Neither causes the other—heat causes both.

### The Potential Outcomes Framework

For any individual, there are two potential outcomes:
- $Y^1$: What would happen WITH treatment
- $Y^0$: What would happen WITHOUT treatment

The **causal effect** for that individual is $Y^1 - Y^0$.

**The fundamental problem:** We can only observe ONE of these. If you take the drug, we see $Y^1$ but not $Y^0$. This is why we need groups and averages.

**Average Treatment Effect (ATE):**
$$ATE = E[Y^1] - E[Y^0]$$

We estimate this by comparing treated and untreated groups—but ONLY if those groups are comparable on everything except treatment.

### Counterfactuals: What Would Have Happened?

Causal inference is fundamentally about counterfactuals:
- "What would have happened to the treated patients if they hadn't been treated?"
- "What would have happened to non-buyers if they'd seen the ad?"

We can never observe counterfactuals directly. The entire field is about finding clever ways to estimate them.

### Directed Acyclic Graphs (DAGs)

DAGs are diagrams that encode your causal assumptions:
- **Nodes:** Variables
- **Arrows:** Direct causal effects
- **Paths:** Chains of arrows (can be blocked)

```
     Age
    /   \\
   ↓     ↓
Treatment → Outcome
```

A DAG is "acyclic" because arrows can't loop back. If A causes B and B causes A, we need time subscripts.

### Key DAG Concepts

**Confounder:** A variable that causes both treatment and outcome.
```
     Z (confounder)
    /   \\
   ↓     ↓
  X  →  Y
```
Z creates a "backdoor path" from X to Y. To identify the causal effect of X on Y, you must block this path by adjusting for Z.

**Mediator:** A variable on the causal path from treatment to outcome.
```
X → M → Y
```
If you want the TOTAL effect of X on Y, DON'T adjust for M. If you want the DIRECT effect (not through M), you can adjust—but interpretation is tricky.

**Collider:** A variable caused by two other variables.
```
X → Z ← Y
```
X and Y are NOT causally related here. BUT if you condition on Z (adjust for it, select on it), you CREATE a spurious association between X and Y.

**Famous example:** Hollywood actors are either talented OR attractive (collider). Among actors (selecting on the collider), talent and attractiveness become negatively correlated—even if they're independent in the population.

### When Can We Estimate Causal Effects?

**Randomized experiment:** Randomization breaks confounding by making treatment independent of potential outcomes.

**Observational data with DAG justification:**
1. Draw your assumed DAG
2. Identify all backdoor paths from treatment to outcome
3. Find a set of variables that blocks all backdoor paths without opening new paths through colliders
4. If such a set exists AND you've measured it, you can estimate the causal effect

**Warning:** Your DAG is an ASSUMPTION. If you're wrong about the causal structure, your analysis may be invalid.
""",
    "interactive": {"type": "dag_builder", "config": {}},
    "key_takeaways": [
        "Causal questions ask about counterfactuals: what would have happened?",
        "Three sources of non-causal correlation: confounding, reverse causation, collider bias",
        "DAGs encode causal assumptions and help identify what to adjust for",
        "Confounders must be adjusted for; colliders must NOT be adjusted for",
        "Randomization is the gold standard because it eliminates confounding",
    ],
    "practice_questions": [
        {
            "question": "You're studying whether education affects income. You find that education correlates with income, but you suspect family wealth confounds this. Draw the DAG and explain what you'd need to adjust for.",
            "answer": "DAG: Family Wealth → Education, Family Wealth → Income, Education → Income. Family wealth is a confounder creating a backdoor path. You must adjust for family wealth to estimate the causal effect of education on income. But if you can't fully measure family wealth, residual confounding remains.",
            "hint": "Identify the backdoor path and what blocks it."
        },
        {
            "question": "Among successful entrepreneurs, you notice a negative correlation between technical skills and social skills. Does this mean technical skills hurt social development?",
            "answer": "No. 'Successful entrepreneur' is a collider—caused by both technical and social skills. Conditioning on success (only studying successful entrepreneurs) creates a spurious negative correlation. In the general population, these skills might be uncorrelated or even positively correlated.",
            "hint": "What causes someone to become a successful entrepreneur?"
        }
    ]
}

CONFOUNDING = {
    "id": "confounding",
    "title": "Confounding & How to Handle It",
    "intro": "Confounders create fake causal signals that feel real. You'll practice identifying them with DAGs and learn when adjusting helps, when it hurts, and why observational studies can never fully prove causation.",
    "exercise": {
        "title": "Try It: Unmask a Confounder",
        "steps": [
            "Build a DAG with treatment, outcome, and a suspected confounder",
            "Mark the confounder and observe the backdoor path",
            "Specify an adjustment set to block the confounding",
            "Add a collider variable and see the warning if you adjust for it"
        ],
        "dsw_type": "stats:regression",
        "dsw_config": {"response": "diameter_mm", "predictors": ["weight_g", "roughness_ra"]},
    },
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

# =============================================================================
# Case Studies
# =============================================================================

CASE_CLINICAL_TRIAL = {
    "id": "case-clinical-trial",
    "title": "Case: Clinical Trial Analysis",
    "intro": "A real Phase III trial with messy data: differential dropout, missing outcomes, unplanned subgroup analyses. You'll navigate the decision between ITT and per-protocol and decide what the data actually shows.",
    "exercise": {
        "title": "Try It: Analyze a Clinical Trial",
        "steps": [
            "Compare ITT vs per-protocol response rates",
            "Assess the impact of 21% vs 9% dropout rates",
            "Run a sensitivity analysis with different dropout assumptions",
            "Evaluate the unplanned subgroup analysis for severe vs mild disease"
        ],
        "dsw_type": "stats:ttest",
        "dsw_config": {"var1": "diameter_mm", "mu": 25.0},
    },
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
    "intro": "Conversion is up but AOV is down. Is the redesign a win? You'll dissect a real A/B test with novelty effects, segment differences, and multiple testing traps.",
    "exercise": {
        "title": "Try It: Evaluate an A/B Test",
        "steps": [
            "Review the headline metrics: conversion, AOV, and revenue per user",
            "Examine the week-over-week trend for novelty effects",
            "Break down results by new vs returning users",
            "Check the sample ratio mismatch diagnostic",
            "Draft your recommendation: ship, iterate, or kill"
        ],
        "dsw_type": "stats:descriptive",
        "dsw_config": {},
    },
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
    "intro": "DSW is your statistical analysis engine with built-in guardrails. You'll tour the analysis pipeline from data upload through validation, analysis, and export, and see how it prevents common mistakes.",
    "exercise": {
        "title": "Try It: Walk Through the Pipeline",
        "steps": [
            "Upload a sample CSV dataset",
            "Review the automatic validation report",
            "Select an analysis type and map your columns",
            "Run the analysis and review the plain-language interpretation"
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
    "practice_questions": [
        {
            "question": "You upload a CSV with 12% missing values in the outcome column. DSW flags a warning. Should you proceed with analysis?",
            "answer": "Investigate first. 12% is above the 5% warning threshold. Check WHY values are missing — is it MCAR (safe to impute/drop), MAR (adjust), or MNAR (dangerous)? DSW's warning is a signal to clean data before analyzing, not to ignore.",
            "hint": "The validation report tells you what to investigate, not what to ignore"
        },
        {
            "question": "DSW runs a t-test and reports p=0.02 but also flags 'assumption violation: non-normal distribution.' What should you do?",
            "answer": "Switch to a non-parametric alternative (Mann-Whitney U). The assumption violation warning means the t-test result may not be reliable. If n>30 per group, the t-test is fairly robust, but it's better practice to use the non-parametric test and compare results.",
            "hint": "Guardrails exist for a reason — follow up on assumption warnings"
        },
    ]
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
            "Compare posteriors and note how data overwhelms the prior"
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
    "intro": "Is your process stable or drifting? Control charts separate signal from noise. You'll build an I-MR chart, identify out-of-control points, and calculate process capability.",
    "exercise": {
        "title": "Try It: Build a Control Chart",
        "steps": [
            "Load the manufacturing dataset",
            "Create an I-MR chart for the diameter column",
            "Identify points outside the control limits",
            "Check for Western Electric rule violations",
            "Calculate Cp and Cpk using spec limits of 25.00 +/- 0.15mm"
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
            "Compare R-squared vs adjusted R-squared"
        ],
        "dsw_type": "stats:regression",
        "dsw_config": {"response": "diameter_mm", "predictors": ["weight_g", "roughness_ra"]},
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
            "hint": "High R² doesn't mean the model is correct if residuals show patterns."
        },
        {
            "question": "You're building a model to predict house prices. You have 'number of bedrooms' and 'square footage' as predictors. VIF for both is 4.5. Should you be concerned?",
            "answer": "VIF of 4.5 is worth noting but not alarming (rule of thumb: concern at >5, serious at >10). Bedrooms and square footage are naturally correlated, so some collinearity is expected. If your goal is prediction, keep both—the model still predicts well. If your goal is understanding the separate effect of each, consider keeping only one or using theory to guide interpretation.",
            "hint": "What are the VIF thresholds for concern?"
        }
    ]
}

# =============================================================================
# Data Fundamentals - Additional Content
# =============================================================================

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

NATURAL_EXPERIMENTS = {
    "id": "natural-experiments",
    "title": "Natural Experiments",
    "intro": "When you can't randomize, nature sometimes does it for you. You'll learn to spot and leverage instrumental variables, regression discontinuities, and difference-in-differences designs.",
    "exercise": {
        "title": "Try It: Identify a Natural Experiment",
        "steps": [
            "Review the three natural experiment types: IV, RD, DiD",
            "Select a scenario and identify the source of quasi-random variation",
            "Evaluate whether the key assumption (exclusion restriction, parallel trends, or no manipulation) holds",
            "Discuss the limitations of the local estimate"
        ],
        "dsw_type": "stats:regression",
        "dsw_config": {"response": "diameter_mm", "predictors": ["roughness_ra"]},
    },
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
    "practice_questions": [
        {
            "question": "A city raises the minimum drinking age from 18 to 21. You want to estimate the effect on traffic accidents. Which natural experiment design is most appropriate?",
            "answer": "Regression discontinuity — compare accident rates for people just under and just above 21 after the change. Those near the cutoff are similar except for legal drinking access. Alternatively, difference-in-differences using a neighboring city that didn't change the law as the control.",
            "hint": "A sharp age cutoff creates a natural experiment at the threshold"
        },
        {
            "question": "A researcher uses 'distance from a university' as an instrumental variable for education when studying the effect of education on earnings. What's the key assumption, and is it plausible?",
            "answer": "The exclusion restriction: distance to university affects earnings ONLY through education. This is questionable — people who grow up near universities may differ in many ways (urban vs rural, parental education, local job market). These factors could independently affect earnings, violating the exclusion restriction.",
            "hint": "The instrument must affect outcome only through the treatment variable"
        },
    ]
}

AB_TESTING_CAUSAL = {
    "id": "ab-testing-causal",
    "title": "A/B Testing as Causal Inference",
    "intro": "A/B tests are randomized experiments in disguise. But network effects, novelty bias, and metric gaming can still ruin them. You'll learn the threats and how to defend against them.",
    "exercise": {
        "title": "Try It: Audit an A/B Test",
        "steps": [
            "Check the sample ratio mismatch diagnostic",
            "Review the time trend for novelty/familiarity effects",
            "Examine segment-level results for heterogeneous treatment effects",
            "Count the number of metrics tested and assess multiple testing risk"
        ],
        "dsw_type": "stats:ttest",
        "dsw_config": {"var1": "diameter_mm", "mu": 25.0},
    },
    "content": """
## The Gold Standard and Its Limitations

A/B testing is the workhorse of causal inference in tech. When done right, it's the most reliable way to answer "did X cause Y?" But it has limitations that are often ignored.

### Why A/B Tests Establish Causation

Randomization solves the fundamental problem of causal inference:

**Without randomization:** Treatment group differs from control in systematic ways.
- People who click on ads are different from those who don't
- Customers who buy premium are different from those who don't
- Users who opt into a feature are different from those who opt out

**With randomization:** The ONLY systematic difference between groups is the treatment.
- Law of large numbers ensures balance on all variables (observed AND unobserved)
- Any difference in outcome can be attributed to treatment
- No confounding, no selection bias (if implemented correctly)

### The Math

For user $i$, let:
- $Y_i^1$ = outcome if treated
- $Y_i^0$ = outcome if control

Average Treatment Effect:
$$ATE = E[Y^1 - Y^0]$$

With random assignment:
$$\\hat{ATE} = \\bar{Y}_{treatment} - \\bar{Y}_{control}$$

The simple difference in means IS the causal effect (in expectation).

### Threats to A/B Test Validity

**1. Selection Effects**
- If users can opt out of treatment, you lose randomization
- If only engaged users trigger the treatment, you're measuring effect on engaged users only
- Solution: Intent-to-treat analysis

**2. Carryover Effects**
- In within-subject designs, earlier treatment affects later responses
- Solution: Sufficient washout period, between-subject design

**3. Novelty/Familiarity Effects**
- New features get attention (novelty) or resistance (familiarity)
- Short-term A/B results may not reflect long-term behavior
- Solution: Run longer, watch for effect decay

**4. Sample Ratio Mismatch (SRM)**
- If actual allocation differs from intended (e.g., 51/49 instead of 50/50), something is wrong
- Indicates bugs, bot contamination, or systematic exclusion
- Solution: Always check allocation ratios; investigate any mismatch

**5. Multiple Comparisons**
- Testing many metrics inflates false positive rate
- 20 metrics at α=0.05 → expect 1 false positive
- Solution: Pre-register primary metrics, adjust for multiplicity

### Network Effects and Interference

Standard A/B testing assumes **SUTVA** (Stable Unit Treatment Value Assumption):
- User A's outcome doesn't depend on whether User B is treated

This fails when:
- **Social networks:** Treated user shares feature with control user
- **Marketplaces:** Treated sellers affect prices for control sellers
- **Shared resources:** Treated group consumes resources, affecting control

**Solutions:**
- Cluster randomization (randomize groups, not individuals)
- Switchback designs (randomize time periods)
- Graph-based splitting

### Long-term Effects vs Short-term Metrics

A/B tests typically measure short-term proxies:
- Clicks, conversions, time on site
- These may not reflect long-term value

**Problems:**
- Feature improves clicks but hurts retention
- Change boosts revenue but annoys users (who leave later)
- Intervention works now but habituates

**Approaches:**
- Long-term holdouts (keep small % in control for months)
- Surrogate metrics validated against long-term outcomes
- User-level LTV modeling

### When A/B Tests Are Impossible or Unethical

**Impossible:**
- You can only launch once (major redesigns)
- Network effects make individual randomization meaningless
- Sample size too small for reliable inference

**Unethical:**
- Withholding known benefits
- Testing potentially harmful changes without safeguards
- Manipulating vulnerable populations

**Alternatives:**
- Observational causal inference with careful design
- Natural experiments
- Qualitative research + expert judgment
- Small-scale pilots with monitoring

### Combining A/B with Observational Methods

In practice, you often need both:

1. **A/B establishes causality** for changes you can randomize
2. **Observational studies** inform hypotheses and explain mechanisms
3. **Long-term holdouts** validate that short-term proxies predict long-term value
4. **Heterogeneous treatment effects** from A/B guide targeting

The best approach uses A/B testing as the final arbiter while using observational data to generate and prioritize hypotheses.
""",
    "interactive": {"type": "ab_analyzer", "config": {}},
    "key_takeaways": [
        "Randomization eliminates confounding, making A/B the gold standard",
        "Threats: selection effects, novelty, SRM, multiple comparisons",
        "Network effects violate SUTVA and require cluster randomization",
        "Short-term metrics may not reflect long-term value",
        "Some questions can't be answered with A/B tests—use alternatives wisely",
    ],
    "practice_questions": [
        {
            "question": "You run an A/B test for a social feature. The treatment group can share content with anyone, but control users might see shared content from treatment users. What problem does this create and how would you address it?",
            "answer": "This violates SUTVA—control users are partially 'treated' by exposure to the feature through treatment users. This spillover contaminates the control, biasing the effect toward zero. Solution: Cluster randomization at the network level (e.g., randomize friend groups or geographic clusters) so treated and control users don't interact.",
            "hint": "What happens when treatment 'leaks' to control?"
        },
        {
            "question": "Your A/B test shows a 2% lift in conversions (p=0.03). But you also checked 15 other metrics. Should you launch?",
            "answer": "With 16 metrics at α=0.05, you expect ~0.8 false positives by chance. A single significant result among 16 is suspicious. Apply Bonferroni (α/16 = 0.003) or Benjamini-Hochberg correction. If conversion was pre-registered as primary, you can give it more weight. Also check effect consistency across segments and practical significance of 2% lift.",
            "hint": "What's the expected false positive count with 16 tests?"
        }
    ]
}

# =============================================================================
# Experimental Design - Additional
# =============================================================================

BLOCKING_STRATIFICATION = {
    "id": "blocking-stratification",
    "title": "Blocking & Stratification",
    "intro": "Noise hides signal. Blocking removes known sources of unwanted variation so real effects become visible. You'll practice designing blocked experiments that are dramatically more efficient.",
    "exercise": {
        "title": "Try It: Design a Blocked Experiment",
        "steps": [
            "Identify the blocking variable in the scenario",
            "Randomize treatments within each block",
            "Compare the precision of blocked vs unblocked designs",
            "Check whether the blocking variable explains meaningful variance"
        ],
        "dsw_type": "stats:anova",
        "dsw_config": {"response": "diameter_mm", "factor": "line"},
    },
    "content": """
## Reducing Noise to Detect Signal

Blocking and stratification are techniques to reduce unwanted variation, making it easier to detect real effects.

### The Problem: Noise Obscures Signal

Imagine testing a new fertilizer on plant growth. But some plants are in sunny spots, others in shade. The sun/shade variation is noise that makes it harder to see the fertilizer effect.

### Blocking: The Solution

**Block** on variables that cause variation but aren't of interest.

```
Without blocking:
  Plants → Random to Fertilizer/Control → Measure growth
  (Sun/shade variation mixed in with treatment effect)

With blocking:
  Sunny plants → Random to Fertilizer/Control → Measure
  Shady plants → Random to Fertilizer/Control → Measure
  (Compare within blocks, then combine)
```

### When to Block

Block on variables that:
1. Affect the outcome (reduce noise)
2. Are known before randomization
3. Can be measured/categorized

**Common blocking variables:**
- Site/location in multi-site trials
- Time period (morning/afternoon)
- Baseline severity
- Operator/technician

### Randomized Block Design

1. Identify blocking variable (e.g., location)
2. Within each block, randomize to treatments
3. Analyze: treatment effect + block effect

$$Y_{ij} = \\mu + \\tau_i + \\beta_j + \\epsilon_{ij}$$

Where τ is treatment effect, β is block effect.

### Stratification vs Blocking

| | Stratification | Blocking |
|-|----------------|----------|
| **When** | Sampling | Randomization |
| **Purpose** | Ensure representation | Reduce variance |
| **Analysis** | Weight or analyze separately | Include block term |

**Stratified randomization:** Randomize within strata to guarantee balance.

### Matched Pairs Design

Extreme blocking: Each block has exactly 2 units.

- Match patients on key characteristics
- Randomly assign one to treatment, one to control
- Compare within pairs

**Advantage:** Very efficient, controls many confounders
**Disadvantage:** Finding good matches is hard

### When Blocking Helps (and When It Doesn't)

**Helps when:**
- Blocking variable explains substantial variance
- Blocks are relatively homogeneous within

**Doesn't help when:**
- Blocking variable unrelated to outcome (wastes df)
- Blocks are too small (can't estimate block effects)

**Rule of thumb:** If blocking variable explains >10% of variance, block on it.
""",
    "interactive": {"type": "blocking_demo", "config": {}},
    "key_takeaways": [
        "Blocking reduces noise by controlling known sources of variation",
        "Block on variables that affect outcome but aren't of interest",
        "Stratified randomization guarantees balance on key variables",
        "Matched pairs is blocking taken to the extreme",
        "Only block on variables that explain meaningful variance",
    ],
    "practice_questions": [
        {
            "question": "You're testing a new teaching method across 4 schools. Student ability varies a lot within each school. Should you block on school, student ability, or both?",
            "answer": "Block on school (since each school is a natural cluster with its own baseline), and consider stratifying students within schools by prior test scores (ability). School is the primary blocking variable because school-level factors (teacher quality, resources) affect outcomes but aren't your interest. Blocking on both maximizes precision.",
            "hint": "Block on the variable that introduces the most unwanted variation"
        },
        {
            "question": "A colleague blocks on hair color when testing a drug's effectiveness. Is this a good idea?",
            "answer": "Almost certainly not. Hair color is unlikely to explain meaningful variance in drug response (unless the drug targets a condition related to pigmentation). Blocking on irrelevant variables wastes degrees of freedom and can actually reduce power. Block on variables that explain >10% of outcome variance — like disease severity, age, or weight.",
            "hint": "Blocking on irrelevant variables hurts more than it helps"
        },
    ]
}

COMMON_DESIGN_FLAWS = {
    "id": "common-design-flaws",
    "title": "Common Design Flaws",
    "intro": "These mistakes make your results meaningless, no matter how sophisticated your analysis. You'll walk through a pre-study checklist that catches selection bias, survivorship bias, and other silent killers.",
    "exercise": {
        "title": "Try It: Spot the Bias",
        "steps": [
            "Review the eight-item bias checklist",
            "For each item, identify whether the scenario has this flaw",
            "Mark each bias as present, absent, or uncertain",
            "Determine which biases are fatal vs manageable"
        ],
    },
    "content": """
## Mistakes That Invalidate Your Study

These flaws can make your results meaningless, no matter how sophisticated your analysis.

### Selection Bias

**Problem:** Sample isn't representative of the population you care about.

**Examples:**
- Studying "all customers" but only have data on those with accounts
- Surveying people who answer phones during business hours
- Studying hospital patients to learn about disease (miss mild cases)

**Detection:** Ask "who's systematically missing?"

### Survivorship Bias

**Problem:** Only seeing the survivors, not the failures.

**Classic example:** WWII planes. Engineers studied bullet holes on returning planes to decide where to add armor. But they should armor where returning planes *weren't* hit—those planes didn't make it back.

**Modern examples:**
- Studying successful companies to find success factors
- Looking at published studies (file drawer problem)
- Analyzing completed projects (ignoring abandoned ones)

### Attrition Bias

**Problem:** People drop out non-randomly.

**Example:** Drug trial where sick patients drop out of treatment arm (side effects) and placebo arm (no improvement). Remaining groups aren't comparable.

**Detection:** Compare dropouts to completers on baseline characteristics.

### Measurement Bias

**Problem:** Systematic error in how you measure outcomes.

**Types:**
- **Recall bias:** Patients with disease remember exposures better
- **Observer bias:** Unblinded assessors rate treatment group higher
- **Instrument bias:** Measurement tool is miscalibrated

**Solution:** Blinding, objective measures, calibration

### Demand Characteristics

**Problem:** Participants guess the hypothesis and act accordingly.

**Example:** Participants in a "stress reduction" study report less stress because they think they should.

**Solution:** Deception (where ethical), implicit measures, active controls

### Hawthorne Effect

**Problem:** People change behavior when they know they're being observed.

**Example:** Factory workers are more productive during the study, regardless of intervention.

**Solution:** Long observation periods (effect fades), unobtrusive measures

### Confounding by Indication

**Problem:** Treatment is given to sicker patients, making treatment look harmful.

**Example:** Studying whether a drug causes death. But drug is given to the sickest patients. Drug appears to increase mortality, but it's actually severity causing both drug use and death.

**Solution:** Randomization, or careful adjustment for severity

### Immortal Time Bias

**Problem:** Time before treatment counted incorrectly.

**Example:** Studying if statin use prevents heart attack. Statin users had to survive long enough to get a prescription. Counting pre-prescription time as "statin exposed" biases results.

**Solution:** Time-varying exposure analysis, proper person-time calculation

### Checklist Before Starting

1. Who's in my sample? Who's missing?
2. Are there survivors I'm selecting on?
3. Will dropout be differential?
4. How objective are my measurements?
5. Do participants know the hypothesis?
6. Is observation itself an intervention?
7. Why do people receive treatment?
8. Am I counting time correctly?
""",
    "interactive": {"type": "bias_detector", "config": {}},
    "key_takeaways": [
        "Selection bias: your sample doesn't represent the population",
        "Survivorship bias: you only see what survived",
        "Attrition bias: dropouts aren't random",
        "Always ask 'who's missing?' and 'why did they receive treatment?'",
        "These biases can't be fixed with fancy statistics",
    ],
    "practice_questions": [
        {
            "question": "A study analyzes Yelp reviews to conclude that restaurant quality has improved over time. What bias might invalidate this conclusion?",
            "answer": "Survivorship bias. Restaurants that close (presumably the worst ones) disappear from Yelp. Over time, the remaining restaurants are increasingly a selected group of survivors. Their improving average rating may reflect bad restaurants dying, not existing restaurants getting better.",
            "hint": "What happened to the restaurants that aren't in the dataset?"
        },
        {
            "question": "In a 12-month weight loss trial, 40% of the treatment group drops out vs 15% of the control group. The remaining treatment participants lost significantly more weight. Is the treatment effective?",
            "answer": "Can't conclude that — severe differential attrition bias. The 40% who dropped out of treatment likely had poor results (side effects, didn't work). The remaining 60% are a selected, possibly more motivated group. Compare baseline characteristics of dropouts vs completers. Use intention-to-treat analysis (count dropouts as treatment failures).",
            "hint": "Differential dropout rates destroy the randomization balance"
        },
    ]
}

# =============================================================================
# Statistical Inference - Additional
# =============================================================================

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

# =============================================================================
# Critical Evaluation Module
# =============================================================================

READING_PAPERS = {
    "id": "reading-papers",
    "title": "Reading Research Papers",
    "intro": "Abstracts spin. Figures don't lie (as easily). You'll learn a reading order that starts with the data, not the claims, and a checklist that catches the most common methodological problems.",
    "exercise": {
        "title": "Try It: Evaluate a Paper",
        "steps": [
            "Read the figures and tables first",
            "Apply the methods checklist: sample, groups, outcome, size",
            "Check for red flags: missing data, unusual methods, no power analysis",
            "Compare the abstract claims to the actual results",
            "Rate your overall confidence in the findings"
        ],
    },
    "content": """
## Efficiently Evaluating Scientific Claims

You don't have time to read every paper thoroughly. Here's how to evaluate quickly and effectively.

### The Reading Order (Not Abstract First!)

1. **Title and authors** - What's the claim? Who did it?
2. **Figures and tables** - What did they actually find?
3. **Methods** - How did they do it? Is it valid?
4. **Results** - Do the numbers support the claims?
5. **Abstract** - Now check if their summary matches reality
6. **Discussion** - What do they think it means?

**Why not abstract first?** Abstracts spin findings. See the data first.

### Methods Checklist

| Question | Why It Matters |
|----------|----------------|
| What's the sample? | Generalizability |
| How were groups assigned? | Confounding |
| What's the comparison? | Appropriate baseline |
| How was outcome measured? | Bias |
| What's the sample size? | Power |
| What analysis was planned? | P-hacking risk |

### Red Flags in Methods

- "We excluded outliers" (which ones? why?)
- "Subgroup analysis revealed" (planned or discovered?)
- No power analysis or sample size justification
- Unusual statistical methods
- Multiple primary outcomes

### Reading Results

**Look for:**
- Effect sizes, not just p-values
- Confidence intervals
- Actual numbers, not just "significant"

**Be suspicious when:**
- Results section longer than methods
- Lots of "marginally significant" or "trending"
- Key results buried in supplements
- Different stats used for different outcomes

### What's NOT Reported

**Ask:** What would I expect to see that isn't here?

- Baseline characteristics by group
- Flow diagram (who dropped out)
- All outcomes mentioned in methods
- Sensitivity analyses
- Negative findings

### Figures That Mislead

| Trick | What to Check |
|-------|---------------|
| Truncated y-axis | Does axis start at zero? |
| Dual y-axes | Are scales comparable? |
| Cherry-picked time window | What happens outside this period? |
| 3D effects | Does it distort proportions? |
| Smoothed data | What do raw data look like? |

### The Discussion Trap

Authors interpret their findings generously. Ask:
- Do results actually support these conclusions?
- What alternative explanations exist?
- Are limitations adequately addressed?
- Is extrapolation beyond data justified?

### Quick Validity Heuristics

**Higher trust:**
- Preregistered study
- Large, well-powered sample
- Replication of prior finding
- Authors discuss limitations prominently
- Independent funding

**Lower trust:**
- First-ever finding
- Small n with big claims
- Industry-funded for industry product
- Limitations buried or dismissed
- Matches popular narrative perfectly
""",
    "interactive": {"type": "paper_evaluator", "config": {}},
    "key_takeaways": [
        "Read figures and methods before abstract",
        "Check what's NOT reported—missing data is often meaningful",
        "Effect sizes and CIs matter more than p-values",
        "Be suspicious of subgroup analyses unless preregistered",
        "Generous interpretation in discussion is normal—evaluate yourself",
    ],
    "practice_questions": [
        {
            "question": "A paper's abstract says 'Treatment significantly improved outcomes (p<0.05).' You look at the results table and see 12 outcomes tested, with only 1 reaching p<0.05. What's wrong?",
            "answer": "Multiple comparisons problem. Testing 12 outcomes at α=0.05 gives a ~46% chance of at least one false positive (1 - 0.95¹²). The one 'significant' result is likely a false positive. The abstract cherry-picked the one significant outcome from 12. Check if any correction (Bonferroni, FDR) was applied.",
            "hint": "How many outcomes were tested total? Were corrections applied?"
        },
        {
            "question": "A methods section says 'We excluded participants who did not complete the study.' The abstract reports 150 participants but methods mention 230 enrolled. Should you be concerned?",
            "answer": "Very concerned — 80 of 230 (35%) dropped out, and they were excluded from analysis. This is attrition bias. Were dropouts different from completers? Were they disproportionate between groups? This study should use intention-to-treat analysis. The 35% dropout rate also suggests the intervention may be unacceptable to many participants.",
            "hint": "What happened to the 80 missing participants, and were they different?"
        },
    ]
}

SPOTTING_BAD_SCIENCE = {
    "id": "spotting-bad-science",
    "title": "Spotting Bad Science",
    "intro": "P-values clustered at 0.049, impossibly consistent data, effects that are too large to be real. You'll learn the statistical red flags that signal unreliable research.",
    "exercise": {
        "title": "Try It: Run the Red Flag Checklist",
        "steps": [
            "Review the study summary for statistical red flags",
            "Check p-value distribution for suspicious clustering near 0.05",
            "Evaluate effect sizes against field norms",
            "Assess institutional red flags (funding, conflicts of interest)",
            "Assign an overall credibility rating"
        ],
    },
    "content": """
## Red Flags That Indicate Unreliable Research

Not all published research is reliable. Here's how to identify studies that shouldn't change your beliefs.

### Statistical Red Flags

**1. P-values just below 0.05**

Distribution of p-values should be smooth. A spike at 0.04-0.049 suggests manipulation.

**2. Too many significant results**

If testing 20 outcomes, expect ~1 significant by chance. If all 20 are significant, something's wrong.

**3. Impossibly consistent data**

Real data is messy. Perfect patterns suggest fabrication or selective reporting.

**4. Round numbers**

Means of 10.0, 20.0, 30.0 exactly? Unlikely in real data.

### The GRIM Test

For means of integer data, only certain values are possible.

**Example:** 27 participants, score range 1-7.
- Mean of 4.57 is possible (sum = 123.39... wait, can't have fractional sum of integers)
- Many reported means fail this basic test

### Effect Size Red Flags

**Too large:** Effect sizes in social science are usually small. d > 1 is rare and should be scrutinized.

**Too consistent across conditions:** Real effects vary. Suspiciously uniform effects suggest problems.

### Design Red Flags

| Red Flag | Why It's Concerning |
|----------|---------------------|
| No control group | Can't attribute effect to treatment |
| Before/after only | Regression to mean, time trends |
| Self-selected sample | Motivation confound |
| No blinding when possible | Expectation effects |
| Outcome changed from protocol | HARKing |
| Composite outcomes | Can hide null effects |

### Reporting Red Flags

- Results described qualitatively, not quantitatively
- Important outcomes in supplements
- "Data not shown" for key results
- Selective citation of prior work
- Conclusions stronger than data warrant

### Institutional Red Flags

**Conflicts of interest:**
- Industry funding for industry product
- Authors with financial stakes
- Undisclosed relationships

**Publication venue:**
- Predatory journal
- Unusual peer review timeline
- Journal outside the field

### Questions to Ask

1. If true, would this overturn well-established findings?
2. Has anyone tried to replicate it?
3. Do the authors have a track record?
4. Is there an obvious motive to find this result?
5. What would convince me this is wrong?

### The Pyramid of Skepticism

**Most skeptical of:**
- First finding ever
- Small sample
- Extraordinary claim
- Single lab
- Matches researcher's prior beliefs

**Less skeptical of:**
- Replication of prior work
- Large sample
- Incremental finding
- Multi-site study
- Finding against researcher's hypothesis

### It's Not About Dismissing Everything

Finding flaws ≠ Dismissing the research

The goal is calibrated confidence:
- Weak study → Weak update to beliefs
- Strong study → Stronger update
- Multiple strong studies → Much stronger update
""",
    "interactive": {"type": "study_evaluator", "config": {}},
    "key_takeaways": [
        "P-values clustered just below 0.05 suggest manipulation",
        "Too many significant results indicates selective reporting",
        "Real data is messy—too-clean data is suspicious",
        "Conflicts of interest don't prove fraud but warrant skepticism",
        "Finding flaws = calibrating confidence, not dismissing",
    ],
    "practice_questions": [
        {
            "question": "A paper tests 8 hypotheses and reports all 8 as significant at p<0.05. The sample size is n=30. Is this suspicious?",
            "answer": "Very suspicious. With n=30, statistical power for typical effect sizes is low (~50% for d=0.5). Getting 8 out of 8 significant results with low power is extremely unlikely unless: (1) all effects are huge (unlikely in most fields), (2) results are selectively reported, or (3) p-values were manipulated. The probability of 8/8 significant with 50% power per test is 0.5⁸ = 0.4%.",
            "hint": "Low power + all significant = mathematically implausible"
        },
        {
            "question": "A nutrition study funded by a chocolate company finds that chocolate consumption is associated with lower BMI. The study has 200,000 participants and p<0.001. Should you update your beliefs?",
            "answer": "Only slightly. Large n and small p don't overcome design concerns. Check: (1) observational study — can't prove causation, (2) industry funding — conflict of interest, (3) what confounders were controlled? (health-conscious people may eat dark chocolate AND exercise more), (4) what's the effect SIZE? (likely tiny). The study is evidence, but weak evidence given the conflict and design limitations.",
            "hint": "Large n and small p don't compensate for study design flaws or conflicts of interest"
        },
    ]
}

META_ANALYSIS_LITERACY = {
    "id": "meta-analysis-literacy",
    "title": "Meta-Analysis Literacy",
    "intro": "Meta-analyses pool multiple studies for a definitive answer, but garbage in still means garbage out. You'll learn to read forest plots, spot publication bias, and evaluate heterogeneity.",
    "exercise": {
        "title": "Try It: Read a Forest Plot",
        "steps": [
            "Identify individual study effects and their confidence intervals",
            "Find the combined diamond and its position relative to zero",
            "Check the I-squared value for heterogeneity",
            "Look for funnel plot asymmetry indicating publication bias"
        ],
    },
    "content": """
## Understanding Systematic Reviews and Meta-Analyses

Meta-analyses combine results from multiple studies. They're powerful but not infallible.

### What Meta-Analysis Does

1. **Systematic search** - Find all relevant studies
2. **Quality assessment** - Evaluate each study
3. **Effect extraction** - Get effect sizes and variances
4. **Statistical combination** - Pool effects with appropriate weights
5. **Heterogeneity assessment** - Are studies measuring the same thing?

### Reading a Forest Plot

```
Study          Effect    95% CI         Weight
─────────────────────────────────────────────────
Smith 2018    ──●──     [0.2, 0.8]      15%
Jones 2019       ──●──  [0.4, 1.0]      20%
Lee 2020      ●         [-0.2, 0.4]     10%
Chen 2021       ─●─     [0.3, 0.7]      25%
Brown 2022      ──●──   [0.3, 0.9]      30%
─────────────────────────────────────────────────
Combined         ◆      [0.35, 0.65]
                 |
              0  |  0.5   1.0
          ← Favors control | Favors treatment →
```

- Each line = one study
- Square position = point estimate
- Square size = study weight
- Diamond = combined effect

### Heterogeneity

**I² statistic:** Percentage of variation due to real differences (not sampling error)

| I² | Interpretation |
|----|----------------|
| 0-25% | Low heterogeneity |
| 25-50% | Moderate |
| 50-75% | Substantial |
| >75% | Considerable |

**High heterogeneity suggests:** Studies may be measuring different things. A single pooled estimate may be misleading.

### Publication Bias

Studies with significant results are more likely to be published. This biases meta-analyses.

**Funnel plot:** Plot effect size vs precision (1/SE)

```
Effect
  |    *
  |  *   *
  |*   *   *
  | *  *  *  *
  |* * *** * *
  └──────────── Precision
```

**Symmetric funnel:** No publication bias
**Asymmetric funnel:** Small studies missing on one side → bias

**Egger's test:** Statistical test for funnel asymmetry

### Quality of Included Studies

**Garbage in, garbage out.** Meta-analysis of bad studies = precise garbage.

**GRADE framework:**
- High: Further research unlikely to change confidence
- Moderate: Might change
- Low: Likely to change
- Very low: Very uncertain

### Fixed vs Random Effects

**Fixed effects:** Assumes one true effect; studies are samples from it
**Random effects:** Assumes distribution of true effects; studies are samples from that distribution

**When to use which:**
- Fixed: Studies are very similar, low heterogeneity
- Random: Studies differ in populations, interventions (more common)

### Limitations

1. **Can't fix bias:** Combines biased studies → biased result
2. **Apples and oranges:** Different interventions, populations, outcomes
3. **Publication bias:** Missing studies can't be found
4. **Assumes independence:** Same authors/data across studies violates this
5. **Ecological fallacy:** Combined effect may not apply to any specific population

### Reading Meta-Analyses Critically

Ask:
- How was the search conducted? Could studies be missed?
- How was quality assessed? Were bad studies downweighted?
- Is heterogeneity addressed? (Subgroups, meta-regression)
- Is publication bias assessed?
- Do sensitivity analyses change conclusions?
""",
    "interactive": {"type": "forest_plot_reader", "config": {}},
    "key_takeaways": [
        "Forest plots show individual studies and combined effect",
        "I² measures heterogeneity—high values mean studies differ",
        "Publication bias can make meta-analyses misleading",
        "Meta-analysis of bad studies gives precise but wrong answers",
        "Random effects models are usually more appropriate",
    ],
    "practice_questions": [
        {
            "question": "A meta-analysis of 20 studies finds a combined effect of d=0.4 (p<0.001), but I²=82%. Should you trust the combined estimate?",
            "answer": "Not as a single number. I²=82% means the studies are measuring substantially different effects — the true effect likely varies by context, population, or intervention variant. Report the range of effects across studies rather than a single pooled number. Investigate sources of heterogeneity through subgroup analysis or meta-regression. The combined d=0.4 is a misleading average of very different effects.",
            "hint": "High I² means the pooled estimate masks real differences between studies"
        },
        {
            "question": "A funnel plot for a meta-analysis shows that small studies cluster on the positive-effect side, while large studies cluster near zero. What does this suggest?",
            "answer": "Publication bias. Small studies with null or negative results are missing (the 'file drawer' problem). The small positive studies that were published likely overestimate the true effect. The large studies, which are published regardless of result, show the truer picture: near-zero effect. The combined meta-analytic estimate is probably inflated.",
            "hint": "Asymmetric funnel = small studies missing on one side = publication bias"
        },
    ]
}

WHEN_NOT_TO_USE_STATISTICS = {
    "id": "when-not-to-use-statistics",
    "title": "When NOT to Use Statistics",
    "intro": "The best analysts know when numbers can't help. You'll learn about the McNamara Fallacy, Goodhart's Law, and when qualitative judgment beats quantitative analysis.",
    "exercise": {
        "title": "Try It: Apply the Decision Framework",
        "steps": [
            "Review the scenario and identify the question type",
            "Determine whether the question is answerable with data",
            "Check for Goodhart's Law risk: is the metric a target?",
            "Decide: statistics, qualitative methods, or pure judgment?"
        ],
    },
    "content": """
## The Limits of Quantitative Analysis

Statistics is powerful but not universal. Knowing when NOT to use it is wisdom.

### Questions Statistics Can't Answer

| Question Type | Why Not |
|---------------|---------|
| Value questions | "Should we prioritize X over Y?" is ethical, not empirical |
| Definition questions | "What counts as success?" can't be measured before defining |
| Novel situations | No relevant data exists |
| Complex systems | Too many interactions to model |
| Individual predictions | Statistics are about groups |

### The McNamara Fallacy

> "The first step is to measure whatever can be easily measured. This is OK as far as it goes. The second step is to disregard that which can't be easily measured or to give it an arbitrary quantitative value. This is artificial and misleading. The third step is to presume that what can't be measured easily isn't important. This is blindness. The fourth step is to say that what can't be easily measured really doesn't exist. This is suicide."

**Translation:** Measuring what's easy, not what matters, leads to bad decisions.

### Goodhart's Law

> "When a measure becomes a target, it ceases to be a good measure."

**Examples:**
- Teaching to the test (scores up, learning unchanged)
- Hospital wait time targets (patients kept in ambulances)
- Police arrest quotas (minor offenses targeted)
- Publication counts (salami slicing, p-hacking)

**Implication:** Once you optimize for a metric, it no longer measures what you cared about.

### When Qualitative Methods Are Better

| Situation | Why Qualitative |
|-----------|-----------------|
| Exploring new phenomena | Don't know what to measure yet |
| Understanding "why" | Numbers show what, not why |
| Context-dependent meaning | Same behavior means different things |
| Complex social dynamics | Can't reduce to variables |
| Rare events | Not enough cases for statistics |

### Decision-Making Under Deep Uncertainty

When you can't estimate probabilities:
- Scenario planning, not expected value
- Robustness, not optimization
- Flexibility and optionality
- Satisficing, not maximizing

### The Role of Judgment

Statistics inform judgment; they don't replace it.

**Statistics can tell you:**
- Treatment A has higher response rate than B
- The difference is 5% with CI [2%, 8%]
- P-value is 0.002

**Statistics can't tell you:**
- Is 5% worth the extra cost?
- How does this apply to YOUR patients?
- What matters more: efficacy or side effects?
- Should we approve this drug?

### When to Step Back from Data

1. **When the data is the problem** - Garbage metrics mislead analysis
2. **When the model is the problem** - Elegant statistics on wrong model = wrong answer
3. **When the question is the problem** - Right answer to wrong question is useless
4. **When humans are the problem** - Incentives and politics override evidence

### The Wisdom of Knowing Limits

The best analysts know when to say:
- "I can't answer that with data"
- "The data we have won't help here"
- "This requires judgment, not calculation"
- "We need to think, not just compute"

Being willing to NOT use statistics when inappropriate is a sign of competence, not weakness.
""",
    "interactive": {"type": "decision_framework", "config": {}},
    "key_takeaways": [
        "Statistics can't answer value or definition questions",
        "McNamara fallacy: measuring what's easy, not what matters",
        "Goodhart's law: when a measure becomes a target, it's corrupted",
        "Qualitative methods are better for 'why' questions and new phenomena",
        "The best analysts know when NOT to use statistics",
    ],
    "practice_questions": [
        {
            "question": "A hospital sets a target of <4 hours for emergency department wait times. Over 6 months, reported wait times drop from 5.2 hours to 3.8 hours. Has patient care improved?",
            "answer": "Not necessarily — this is Goodhart's Law in action. Check HOW wait times dropped: Are patients being triaged differently? Held in ambulances before officially 'arriving'? Moved to hallways before being 'seen'? Rushed through without adequate assessment? The metric improved, but the underlying patient experience may not have. Need qualitative data: patient satisfaction, readmission rates, adverse outcomes.",
            "hint": "When a measure becomes a target, it ceases to be a good measure"
        },
        {
            "question": "Your CEO asks: 'What does the data say we should do about employee morale?' You have only eNPS scores. What's your response?",
            "answer": "eNPS alone can't answer 'what to do' — that's a question about causes and interventions, not just measurement. First, eNPS is one metric of a complex phenomenon (McNamara fallacy risk). Second, 'what to do' requires understanding WHY morale is the way it is — qualitative methods like interviews, focus groups, and open-ended surveys are needed. Recommend a mixed-methods approach: use eNPS to identify THAT there's a problem, then qualitative research to understand WHY.",
            "hint": "Numbers show what's happening; understanding why requires qualitative methods"
        },
    ]
}

# =============================================================================
# Case Studies - Additional
# =============================================================================

CASE_MANUFACTURING = {
    "id": "case-manufacturing",
    "title": "Case: Manufacturing Quality Control",
    "intro": "Defect rate jumped from 2% to 8%. Something changed. You'll use control charts to detect when the shift happened, test hypotheses about the cause, and verify the fix with capability analysis.",
    "exercise": {
        "title": "Try It: Investigate a Process Shift",
        "steps": [
            "Build a control chart and identify the shift point",
            "Correlate the timing with potential causes",
            "Compare Machine #3 output to other machines",
            "Calculate Cp and Cpk after the fix",
            "Set up ongoing monitoring with Western Electric rules"
        ],
        "dsw_type": "spc:capability",
        "dsw_config": {"var": "diameter_mm", "lsl": 24.85, "usl": 25.15, "target": 25.0},
    },
    "content": """
## Scenario: The Widget Problem

You're the quality manager at a manufacturing plant producing precision widgets. The specification is 10.00 ± 0.05 mm. Last week, the defect rate suddenly jumped from 2% to 8%.

### The Data

You have measurements from the past month:
- 30 measurements per day
- Last week's data shows shift in both mean and variance

```
Week 1-3 (stable):    Mean = 10.002, SD = 0.012
Week 4 (problem):     Mean = 10.018, SD = 0.021
```

### Question 1: Was There Really a Change?

**Control chart analysis:**

Before: UCL = 10.038, LCL = 9.966
Week 4: Multiple points above UCL, run of 8 above centerline

**Conclusion:** Yes, process is out of control. This isn't random variation.

### Question 2: When Did It Start?

Looking at the control chart day by day:
- Monday: In control
- Tuesday: One point near UCL
- Wednesday: Point above UCL + mean shift begins
- Thursday-Friday: Clearly out of control

**Conclusion:** Change occurred Tuesday night or Wednesday morning.

### Question 3: What Changed?

Check what happened Tuesday night:
- New batch of raw material arrived Tuesday
- Maintenance on Machine #3 Tuesday evening
- New operator started Wednesday

**Investigation:**
- Raw material: Tested, within spec
- Machine #3: Maintenance log shows recalibration
- New operator: Same training, different machines

**Hypothesis:** Machine #3 recalibration caused the shift.

### Question 4: Confirming the Cause

Run a test:
1. Measure 50 widgets from Machine #3
2. Measure 50 widgets from other machines
3. Compare

**Results:**
- Machine #3: Mean = 10.022, SD = 0.024
- Other machines: Mean = 10.001, SD = 0.011

**Conclusion:** Machine #3 is the problem.

### Question 5: Is It Fixed?

After recalibrating Machine #3:
- Run capability analysis
- Cp = 1.45, Cpk = 1.38

**Interpretation:**
- Cp > 1.33: Process spread is OK
- Cpk > 1.33: Process is centered well enough
- Both acceptable: Process is capable

### Question 6: Ongoing Monitoring

Set up:
- X-bar and R chart for daily monitoring
- 30 samples per day, subgroups of 5
- Western Electric rules for out-of-control detection
- Monthly capability recalculation

### Report to Management

"Investigation Summary:

The defect rate increase from 2% to 8% was caused by miscalibration of Machine #3 following routine maintenance on Tuesday.

Evidence:
- Control chart shows process shift beginning Wednesday
- Machine #3 output significantly different from other machines (p < 0.001)
- Recalibration restored process capability

Actions taken:
- Machine #3 recalibrated
- Maintenance procedure updated to include verification step
- Additional control chart monitoring implemented

Current status:
- Cpk = 1.38 (acceptable)
- Defect rate returned to <2%"
""",
    "interactive": {"type": "spc_demo", "config": {}},
    "key_takeaways": [
        "Control charts detect when a process changes",
        "Correlate timing of change with potential causes",
        "Test hypotheses with comparative data",
        "Capability indices confirm the fix worked",
        "Ongoing monitoring prevents recurrence",
    ],
    "practice_questions": [
        {
            "question": "After fixing Machine #3, the process shows Cp=1.45 but Cpk=0.95. What does this tell you and what action is needed?",
            "answer": "Cp=1.45 means the process spread is well within spec limits (process is capable in terms of variability). But Cpk=0.95 (less than 1.33) means the process is off-center — the mean is shifted toward one spec limit. Action: adjust the process mean to center it within the specification. Once centered, Cpk should approach Cp, and both will be above 1.33.",
            "hint": "Cp measures spread, Cpk measures spread + centering"
        },
        {
            "question": "Your control chart shows 7 consecutive points above the centerline but all within control limits. Is the process in control?",
            "answer": "No — this violates the Western Electric run rule (7+ consecutive points on one side of the centerline indicates a non-random pattern). Even though individual points are within control limits, the run signals a small but persistent shift in the process mean. Investigate what changed. Don't wait for a point to exceed the control limits.",
            "hint": "Western Electric rules detect non-random patterns even within control limits"
        },
    ]
}

CASE_OBSERVATIONAL = {
    "id": "case-observational",
    "title": "Case: Observational Study Critique",
    "intro": "Headlines say coffee drinkers live longer. But the study is observational with 500,000 people. You'll tear apart the methodology, identify confounders, and write a calibrated conclusion.",
    "exercise": {
        "title": "Try It: Critique an Observational Study",
        "steps": [
            "Identify the comparison group and its problems",
            "List measured and unmeasured confounders",
            "Evaluate the dose-response relationship",
            "Check the decaf finding for mechanistic clues",
            "Write a one-paragraph calibrated conclusion"
        ],
        "dsw_type": "stats:regression",
        "dsw_config": {"response": "diameter_mm", "predictors": ["weight_g", "roughness_ra"]},
    },
    "content": """
## Scenario: The Coffee Longevity Study

A news headline reads: "Coffee Drinkers Live Longer! Large Study Confirms Health Benefits"

Your task: Evaluate the underlying research.

### The Study

**Design:** Prospective cohort study
**Sample:** 500,000 adults from UK Biobank
**Follow-up:** 10 years
**Exposure:** Self-reported coffee consumption at baseline
**Outcome:** All-cause mortality

**Key finding:** Compared to non-drinkers, drinking 3-4 cups/day associated with:
- 12% lower mortality (HR = 0.88, 95% CI: 0.84-0.92)
- p < 0.001

### Question 1: Is There an Association?

**Yes.** With 500,000 participants, narrow CI, and p < 0.001, the association is real.

But association ≠ causation.

### Question 2: Who Are the Non-Drinkers?

This is the critical question. "Non-drinkers" include:
- Never drinkers (maybe healthier lifestyle)
- Former drinkers who quit (why? often health reasons)
- People too sick to drink coffee

**The "sick quitter" problem:** People often stop coffee when they feel unwell. Comparing to non-drinkers includes sick people in the comparison group.

### Question 3: What Was Controlled For?

The study adjusted for:
- Age, sex, BMI
- Smoking, alcohol
- Physical activity
- Education, income

**Not controlled for:**
- Overall diet quality
- Access to healthcare
- Stress levels
- Sleep quality
- Reasons for not drinking coffee

### Question 4: Healthy User Bias?

Coffee drinkers might be different in unmeasured ways:
- More likely to have stable routines
- More likely to be employed
- More social (coffee as social activity)
- More able to afford discretionary spending

These unmeasured confounders could explain the association.

### Question 5: What About Decaf?

If caffeine is the mechanism, decaf shouldn't help.

**Study finding:** Decaf showed similar benefit.

**Interpretation:** This suggests it's not caffeine. Maybe:
- Other compounds in coffee
- Or: confounding, not causation

### Question 6: Dose-Response?

| Cups/day | Hazard Ratio |
|----------|--------------|
| 0 | 1.00 (reference) |
| 1 | 0.94 |
| 2-3 | 0.88 |
| 4-5 | 0.87 |
| 6+ | 0.84 |

**Interpretation:** Dose-response pattern supports causation (but doesn't prove it—could also be confounding pattern).

### Your Critique

**Strengths:**
- Very large sample
- Long follow-up
- Prospective design
- Multiple sensitivity analyses

**Weaknesses:**
- Self-reported exposure (measurement error)
- Single baseline measurement (habits change)
- Healthy user bias likely
- "Non-drinker" group problematic
- Unmeasured confounding inevitable

### What We Can Conclude

**What we can say:**
- Coffee drinking is associated with lower mortality
- Association is robust across subgroups
- Not explained by measured confounders

**What we cannot say:**
- Coffee causes longer life
- You should drink more coffee to live longer
- Coffee is healthy

### The Bottom Line

"This well-conducted observational study finds an association between moderate coffee consumption and lower mortality. However, causal interpretation is limited by likely confounding from unmeasured lifestyle factors and the 'sick quitter' problem in the reference group. The finding is consistent with—but does not prove—a health benefit from coffee."
""",
    "interactive": {"type": "study_evaluator", "config": {}},
    "key_takeaways": [
        "Association ≠ causation, especially in observational studies",
        "The comparison group matters—who are the 'non-drinkers'?",
        "Healthy user bias: coffee drinkers may be healthier for other reasons",
        "Large sample size proves association, not causation",
        "Well-conducted observational studies show what to test, not what to conclude",
    ],
    "practice_questions": [
        {
            "question": "The study adjusts for 15 confounders (smoking, BMI, exercise, etc.) and the association persists. Does this prove causation?",
            "answer": "No. Adjustment for measured confounders doesn't rule out unmeasured confounders. There could be health-conscious behaviors not captured in the 15 variables (sleep quality, stress management, social connections). Also, residual confounding exists even for measured variables — BMI and exercise are measured imprecisely. The persistence after adjustment strengthens the association but doesn't establish causation.",
            "hint": "You can only adjust for what you measure, and even that's imperfect"
        },
        {
            "question": "How would you design a study that COULD establish whether coffee causes longer life?",
            "answer": "The gold standard would be a randomized controlled trial: randomly assign people to drink coffee vs not, follow for decades, measure mortality. This is practically impossible (decades of compliance, ethical concerns about forcing/prohibiting coffee). Pragmatic alternatives: (1) Mendelian randomization using genetic variants that affect coffee metabolism as instruments, (2) long-term randomized trial with a shorter-term biomarker proxy, (3) triangulation across multiple natural experiments with different sources of bias.",
            "hint": "What study design eliminates confounding by definition?"
        },
    ]
}

# =============================================================================
# Capstone
# =============================================================================

CAPSTONE_OVERVIEW = {
    "id": "capstone-overview",
    "title": "Capstone Overview",
    "intro": "This is where you put it all together. You'll plan a complete analysis from research question to final report, applying every skill from the course.",
    "exercise": {
        "title": "Try It: Plan Your Capstone",
        "steps": [
            "Select a dataset from the available options",
            "Write your research question and hypotheses",
            "Plan your analysis approach (before seeing data)",
            "Define your primary outcome and stopping criteria"
        ],
    },
    "content": """
## Demonstrating Your Skills

The capstone project is your opportunity to demonstrate everything you've learned. You'll conduct a complete analysis from question to conclusion.

### Project Requirements

**1. Question Formulation**
- Clear, answerable research question
- Defined hypotheses (before seeing data)
- Appropriate scope for available data

**2. Data Assessment**
- Document data source and quality
- Address missing data and outliers
- Check assumptions for planned analyses

**3. Analysis**
- Appropriate statistical methods
- Correct interpretation of results
- Sensitivity analyses where relevant

**4. Communication**
- Clear narrative structure
- Appropriate visualizations
- Honest acknowledgment of limitations

### Evaluation Rubric

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Question clarity | 15% | Is the question answerable and well-defined? |
| Data handling | 20% | Appropriate cleaning, documentation |
| Method selection | 20% | Right test for the question and data |
| Interpretation | 25% | Correct, nuanced, calibrated |
| Communication | 20% | Clear, honest, complete |

### Available Datasets

You may choose from:
1. **Clinical trial data** - Drug efficacy with complications
2. **A/B test results** - E-commerce conversion with segments
3. **Manufacturing data** - Process quality over time
4. **Survey data** - Observational study with confounding
5. **Your own data** - Subject to approval

### Common Pitfalls

**Avoid:**
- Changing hypotheses after seeing data (HARKing)
- Ignoring violated assumptions
- Overclaiming from correlational data
- Burying inconvenient results
- Skipping sensitivity analyses

**Embrace:**
- Acknowledging limitations prominently
- Reporting null findings honestly
- Showing what you don't know
- Alternative explanations

### Timeline

1. **Week 1:** Select dataset, formulate question, submit proposal
2. **Week 2:** Data exploration and cleaning, methods planning
3. **Week 3:** Analysis and interpretation
4. **Week 4:** Report writing and review
5. **Submission:** Complete report with code/methods

### Format

**Written report:** 2,000-3,000 words
**Sections:** Introduction, Methods, Results, Discussion, Limitations
**Appendix:** Technical details, additional analyses

### Peer Review Component

After submission, you will:
1. Receive two other capstone projects to review
2. Provide structured feedback using the rubric
3. Respond to feedback on your own project

This mirrors real scientific peer review and helps you see diverse approaches.
""",
    "interactive": {"type": "project_planner", "config": {}},
    "key_takeaways": [
        "Formulate hypotheses BEFORE analyzing data",
        "Document all data handling decisions",
        "Match statistical methods to question and data type",
        "Interpretation should be calibrated to evidence strength",
        "Limitations acknowledgment is strength, not weakness",
    ],
    "practice_questions": [
        {
            "question": "You're evaluating a peer's capstone. Their methods say 'We explored several tests and chose the one that gave the clearest results.' What feedback do you give?",
            "answer": "This is p-hacking / HARKing. Choosing the test that 'gives the clearest results' means choosing the test that gives the lowest p-value, inflating false positive risk. The analysis approach should be specified before seeing results. Recommend: state the planned analysis upfront, justify the choice based on data type and question, report results regardless of outcome. If multiple tests were run, apply corrections.",
            "hint": "Choosing the 'best' test after seeing results is a form of p-hacking"
        },
    ]
}

CAPSTONE_PROJECT = {
    "id": "capstone-project",
    "title": "Capstone Project",
    "intro": "Time to execute. Work through the full analysis pipeline: clean, explore, analyze, interpret, and write up. Your report will demonstrate calibrated reasoning under uncertainty.",
    "exercise": {
        "title": "Try It: Execute Your Analysis",
        "steps": [
            "Upload your selected dataset to DSW",
            "Document your data cleaning decisions",
            "Run your pre-specified primary analysis",
            "Conduct at least one sensitivity analysis",
            "Draft the results section with effect sizes and CIs"
        ],
        "dsw_type": "stats:descriptive",
        "dsw_config": {},
    },
    "content": """
## Complete Your Analysis

This is where you apply everything. Select a dataset, analyze it rigorously, and present your findings.

### Step 1: Select Your Dataset

Review the available options or propose your own:

**Option A: Clinical Trial**
- Randomized trial of arthritis treatment
- Significant dropout differential
- Subgroup data available
- Challenge: ITT vs per-protocol, missing data

**Option B: E-commerce A/B Test**
- Checkout flow redesign
- Conversion and revenue metrics
- Weekly time series available
- Challenge: Novelty effect, segment differences

**Option C: Manufacturing Process**
- Widget dimension measurements
- Before/after maintenance intervention
- Multiple machines
- Challenge: SPC, capability analysis, root cause

**Option D: Survey Study**
- Coffee consumption and health outcomes
- Observational data with covariates
- Challenge: Confounding, causal language

### Step 2: Formulate Your Question

Before looking at the data:
1. Write your primary research question
2. State your null and alternative hypotheses
3. Define your primary outcome
4. Specify your analysis approach

**Submit for approval before proceeding.**

### Step 3: Explore and Clean Data

Document:
- Sample size and characteristics
- Missing data patterns and handling
- Outlier detection and decisions
- Assumption checks

### Step 4: Conduct Analysis

- Run pre-specified primary analysis
- Report effect size with confidence interval
- Conduct sensitivity analyses
- Run any pre-specified secondary analyses

### Step 5: Interpret Results

- What does the evidence support?
- What alternative explanations exist?
- How should findings be qualified?
- What would change your conclusions?

### Step 6: Write Report

**Structure:**
```
1. Introduction (250-400 words)
   - Background and motivation
   - Research question
   - Hypotheses

2. Methods (400-600 words)
   - Data source and sample
   - Variables and measures
   - Statistical approach
   - Handling of missing data

3. Results (500-800 words)
   - Descriptive statistics
   - Primary analysis with effect size and CI
   - Secondary analyses
   - Sensitivity analyses

4. Discussion (500-800 words)
   - Summary of findings
   - Interpretation in context
   - Comparison to prior work
   - Implications

5. Limitations (200-400 words)
   - Study design limitations
   - Data limitations
   - Analysis limitations
   - What we can't conclude

6. Conclusion (100-200 words)
   - Main takeaway
   - Calibrated confidence
```

### Submission Checklist

☐ Research question clearly stated
☐ Hypotheses defined before analysis
☐ Data handling documented
☐ Appropriate statistical methods used
☐ Effect sizes and CIs reported
☐ Limitations prominently discussed
☐ Conclusions match evidence strength
☐ Report is 2,000-3,000 words
☐ Code/analysis files included

### After Submission

You will receive:
1. Two peer capstone projects to review
2. Feedback on your own project
3. Opportunity to revise based on feedback
4. Final review and feedback
""",
    "interactive": {"type": "capstone_workspace", "config": {}},
    "key_takeaways": [
        "Hypotheses must be stated before data analysis",
        "Document all decisions for reproducibility",
        "Effect sizes and CIs are mandatory, p-values optional",
        "Limitations section shows competence, not weakness",
        "Calibrate conclusions to actual evidence strength",
    ],
    "practice_questions": [
        {
            "question": "Your primary analysis shows p=0.08 (not significant at 0.05). In your sensitivity analysis, a slightly different exclusion criterion gives p=0.03. How do you report this?",
            "answer": "Report the primary analysis as the main result: the pre-specified analysis showed no significant effect (p=0.08). Report the sensitivity analysis transparently: 'A sensitivity analysis with [different criterion] yielded p=0.03.' Do NOT swap them or present the sensitivity as the main finding. The discrepancy should be discussed — it suggests the result is fragile and depends on analytical choices, which is itself informative.",
            "hint": "The pre-specified primary analysis is the main result, period"
        },
    ]
}

# =============================================================================
# Advanced Methods
# =============================================================================

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
            "Try running a t-test on the same data and compare conclusions"
        ],
        "dsw_type": "stats:mann_whitney",
        "dsw_config": {"var": "diameter_mm", "group_var": "shift"},
    },
    "sample_data": {
        "supplier_a_burst_psi": [245, 312, 198, 287, 256, 301, 223, 278, 234, 267, 289, 203, 345, 256, 221, 290, 276, 248, 310, 235],
        "supplier_b_burst_psi": [278, 334, 289, 312, 298, 356, 267, 301, 334, 288, 321, 299, 367, 312, 278, 345, 310, 289, 356, 298],
        "shift_a_roughness": [2.3, 1.8, 2.1, 3.5, 1.9, 2.0, 2.4, 1.7, 2.8, 1.6, 2.2, 3.1, 1.9, 2.5, 2.0],
        "shift_b_roughness": [2.8, 3.1, 2.5, 3.9, 2.7, 3.3, 2.9, 3.5, 2.6, 3.0, 4.1, 2.8, 3.2, 2.9, 3.4],
        "shift_c_roughness": [2.1, 1.9, 2.3, 2.0, 1.8, 2.5, 2.2, 1.7, 2.4, 2.1, 1.9, 2.6, 2.0, 2.3, 1.8],
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
            "group_a": [245, 312, 198, 287, 256, 301, 223, 278, 234, 267, 289, 203, 345, 256, 221, 290, 276, 248, 310, 235],
            "group_b": [278, 334, 289, 312, 298, 356, 267, 301, 334, 288, 321, 299, 367, 312, 278, 345, 310, 289, 356, 298],
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
            "hint": "Count data with heavy right skew violates normality assumption"
        },
        {
            "question": "After a Kruskal-Wallis test shows H=14.2, p=0.003 across 4 groups, a colleague runs 6 separate Mann-Whitney tests for all pairwise comparisons. What's wrong with this approach?",
            "answer": "Running 6 separate Mann-Whitney tests inflates the familywise error rate (6 × 0.05 = 30% chance of at least one false positive). The correct approach is Dunn's test, which is specifically designed for post-hoc pairwise comparisons after Kruskal-Wallis and includes an adjustment for multiple comparisons (typically Bonferroni or Holm).",
            "hint": "Multiple comparisons problem — same issue as running multiple t-tests after ANOVA"
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
            "Determine if the series needs differencing for stationarity"
        ],
        "dsw_type": "stats:descriptive",
    },
    "sample_data": {
        "monthly_sales": [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
                          115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
                          145, 150, 178, 163, 152, 191, 210, 209, 183, 159, 136, 168],
        "months": ["Jan-Y1", "Feb-Y1", "Mar-Y1", "Apr-Y1", "May-Y1", "Jun-Y1",
                   "Jul-Y1", "Aug-Y1", "Sep-Y1", "Oct-Y1", "Nov-Y1", "Dec-Y1",
                   "Jan-Y2", "Feb-Y2", "Mar-Y2", "Apr-Y2", "May-Y2", "Jun-Y2",
                   "Jul-Y2", "Aug-Y2", "Sep-Y2", "Oct-Y2", "Nov-Y2", "Dec-Y2",
                   "Jan-Y3", "Feb-Y3", "Mar-Y3", "Apr-Y3", "May-Y3", "Jun-Y3",
                   "Jul-Y3", "Aug-Y3", "Sep-Y3", "Oct-Y3", "Nov-Y3", "Dec-Y3"],
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
            "data": [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
                     115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
                     145, 150, 178, 163, 152, 191, 210, 209, 183, 159, 136, 168],
            "labels": ["Jan-Y1", "Feb-Y1", "Mar-Y1", "Apr-Y1", "May-Y1", "Jun-Y1",
                       "Jul-Y1", "Aug-Y1", "Sep-Y1", "Oct-Y1", "Nov-Y1", "Dec-Y1",
                       "Jan-Y2", "Feb-Y2", "Mar-Y2", "Apr-Y2", "May-Y2", "Jun-Y2",
                       "Jul-Y2", "Aug-Y2", "Sep-Y2", "Oct-Y2", "Nov-Y2", "Dec-Y2",
                       "Jan-Y3", "Feb-Y3", "Mar-Y3", "Apr-Y3", "May-Y3", "Jun-Y3",
                       "Jul-Y3", "Aug-Y3", "Sep-Y3", "Oct-Y3", "Nov-Y3", "Dec-Y3"],
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
            "hint": "ACF cutoff → MA order; PACF cutoff → AR order. One difference = d=1"
        },
        {
            "question": "A control chart shows a process running in control for 6 months, then a shift. Your manager wants to re-estimate control limits using all 6 months plus the shifted data. Why is this wrong?",
            "answer": "Including the shifted data contaminates the baseline estimates. Control limits should be estimated from the stable, in-control period only. Including the shift inflates the estimated standard deviation, making the limits too wide and potentially hiding the shift (and future shifts). Use change point detection to identify exactly when the shift occurred, then estimate limits from the pre-shift data only.",
            "hint": "Control limits estimate the voice of the process — only in-control data should be used"
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
            "Estimate the B10 life (time for 10% failure)"
        ],
        "dsw_type": "stats:descriptive",
    },
    "sample_data": {
        "bearing_failure_hours": [120, 245, 310, 389, 412, 456, 498, 534, 567, 601,
                                  623, 678, 712, 745, 789, 823, 856, 890, 934, 967,
                                  1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
                                  1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
                                  1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
        "bearing_censored": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "note": "1=failed, 0=still running at 1000hrs (censored). 20 of 50 bearings failed."
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
            "times": [120, 245, 310, 389, 412, 456, 498, 534, 567, 601,
                      623, 678, 712, 745, 789, 823, 856, 890, 934, 967,
                      1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
                      1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
                      1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
            "events": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
            "hint": "β > 1 means wear-out; η is the characteristic life (63.2% failure point)"
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
            "Characterize each cluster: who are these customers?"
        ],
        "dsw_type": "stats:descriptive",
    },
    "sample_data": {
        "clustering_example": {
            "description": "Customer purchase data: annual spend ($) vs visit frequency",
            "spend": [120, 85, 340, 420, 95, 210, 380, 450, 78, 155, 290, 510, 62, 185, 365, 490, 110, 245, 315, 470],
            "frequency": [24, 18, 8, 5, 22, 14, 6, 3, 20, 12, 9, 4, 26, 15, 7, 2, 21, 11, 8, 4],
        },
        "anomaly_example": {
            "description": "Sensor readings from 20 parts (2 are defective)",
            "temp": [72.1, 71.8, 72.3, 71.9, 72.0, 85.4, 71.7, 72.2, 71.6, 72.4, 72.1, 71.5, 72.0, 71.8, 72.3, 71.9, 92.1, 72.1, 71.7, 72.2],
            "vibration": [0.15, 0.12, 0.18, 0.14, 0.13, 0.45, 0.16, 0.11, 0.17, 0.14, 0.15, 0.13, 0.12, 0.16, 0.15, 0.14, 0.62, 0.13, 0.15, 0.11],
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
            "x_data": [120, 85, 340, 420, 95, 210, 380, 450, 78, 155, 290, 510, 62, 185, 365, 490, 110, 245, 315, 470],
            "y_data": [24, 18, 8, 5, 22, 14, 6, 3, 20, 12, 9, 4, 26, 15, 7, 2, 21, 11, 8, 4],
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
            "hint": "12 out of 10,000 is extreme imbalance — supervised learning needs balanced classes"
        },
        {
            "question": "Your regression model has R²=0.95 on training data but R²=0.42 on test data. What happened and what should you try?",
            "answer": "Classic overfitting — the model memorized training noise rather than learning the true pattern. Try: (1) Regularized regression (Ridge or Lasso) to penalize complex models, (2) reduce the number of features using PCA or Lasso's built-in variable selection, (3) increase training data if possible, (4) use a simpler model (fewer terms). The gap between 0.95 and 0.42 is dramatic — the model is essentially useless for prediction despite looking great on training data.",
            "hint": "Big gap between train and test performance = overfitting"
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
            "Decide: fix the gage or proceed to process analysis?"
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
            "hint": "If you can't trust the measurement, you can't trust any analysis based on it"
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
            "Generate the randomized run order"
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
                {"q": "Are you screening (>6 factors) or optimizing (≤6 factors)?", "yes_next": 1, "no_next": 2},
                {"q": "Do you need to estimate curvature?", "yes_next": "dsd", "no_next": "screen"},
                {"q": "Do you need to find the optimal point?", "yes_next": 3, "no_next": "factorial"},
                {"q": "Can you run extreme factor combinations?", "yes_next": "ccd", "no_next": "bbd"},
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
            "hint": "Count the 2-factor interactions: C(6,2) = 15. Any design estimating all 15 needs at least ~20 degrees of freedom"
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
            "Try the Kruskal-Wallis test with three groups"
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
            "hint": "Only A vs B is significant after multiple comparison adjustment"
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
            "Generate a 6-month forecast with prediction intervals"
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
                "values": [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
                           115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140],
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
            "hint": "Significant residual ACF at lag 1 means the non-seasonal component needs adjustment"
        },
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
    "blocking-stratification": BLOCKING_STRATIFICATION,
    "common-design-flaws": COMMON_DESIGN_FLAWS,

    # Data Fundamentals
    "data-cleaning": DATA_CLEANING,
    "sampling": SAMPLING,
    "distributions": DISTRIBUTIONS,
    "eda": EDA,

    # Statistical Inference
    "choosing-tests": CHOOSING_TESTS,
    "interpreting-results": INTERPRETING_RESULTS,
    "p-values-deep-dive": P_VALUES_DEEP_DIVE,
    "confidence-intervals": CONFIDENCE_INTERVALS,
    "effect-sizes": EFFECT_SIZES,
    "multiple-comparisons": MULTIPLE_COMPARISONS,

    # Causal Inference
    "causal-thinking": CAUSAL_THINKING,
    "confounding": CONFOUNDING,
    "natural-experiments": NATURAL_EXPERIMENTS,
    "ab-testing-causal": AB_TESTING_CAUSAL,

    # Critical Evaluation
    "reading-papers": READING_PAPERS,
    "spotting-bad-science": SPOTTING_BAD_SCIENCE,
    "meta-analysis-literacy": META_ANALYSIS_LITERACY,
    "when-not-to-use-statistics": WHEN_NOT_TO_USE_STATISTICS,

    # DSW Mastery
    "dsw-overview": DSW_OVERVIEW,
    "bayesian-ab-hands-on": BAYESIAN_AB_HANDS_ON,
    "spc-hands-on": SPC_HANDS_ON,
    "regression-hands-on": REGRESSION_HANDS_ON,

    # Case Studies
    "case-clinical-trial": CASE_CLINICAL_TRIAL,
    "case-ab-test": CASE_AB_TEST,
    "case-manufacturing": CASE_MANUFACTURING,
    "case-observational": CASE_OBSERVATIONAL,

    # Capstone
    "capstone-overview": CAPSTONE_OVERVIEW,
    "capstone-project": CAPSTONE_PROJECT,

    # Advanced Methods
    "nonparametric-tests": NONPARAMETRIC_TESTS,
    "time-series-analysis": TIME_SERIES_ANALYSIS,
    "survival-reliability": SURVIVAL_RELIABILITY,
    "ml-essentials": ML_ESSENTIALS,
    "measurement-systems": MEASUREMENT_SYSTEMS,

    # DSW Mastery (additional)
    "doe-hands-on": DOE_HANDS_ON,
    "nonparametric-hands-on": NONPARAMETRIC_HANDS_ON,
    "time-series-hands-on": TIME_SERIES_HANDS_ON,
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
