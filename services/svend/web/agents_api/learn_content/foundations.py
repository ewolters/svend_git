"""Learning content: Foundations."""

from ._datasets import SHARED_DATASET  # noqa: F401

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
            "Try extreme priors (5% and 95%) to see how they resist evidence",
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

### Bayesian Thinking in Practice: Process Monitoring

The same Bayesian update machinery you just learned powers SVEND's Process Belief System (PBS). Each new measurement on a production line updates a Normal-Gamma posterior — the same prior x likelihood → posterior logic, applied in O(1) per observation. See **Module 11: PBS Mastery** to see Bayesian thinking applied to real-time manufacturing.
""",
    "interactive": {
        "type": "bayesian_calculator",
        "config": {
            "show_prior": True,
            "show_likelihood_ratio": True,
            "show_confidence": True,
            "show_posterior": True,
        },
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
            "hint": "Convert probability to odds, multiply by LR, convert back",
        },
        {
            "question": "A test has LR=10 but only 50% reliability. What's the effective LR?",
            "answer": "5.5. Adjusted LR = 1 + (10-1) × 0.5 = 5.5",
            "hint": "Use the confidence adjustment formula",
        },
    ],
    # Interactive tutorial: Synara belief tracking
    "tool_steps": [
        {
            "id": "bt-1",
            "title": "Create a Hypothesis",
            "instruction": "Let's see Bayesian updating in action with Svend's belief engine. Create a hypothesis about whether a new drug reduces blood pressure. Set your prior to 30% — you're skeptical but open-minded.",
            "tool": "synara",
            "action": "add_hypothesis",
            "config": {
                "description": "",
                "behavior_class": "treatment_effect",
                "prior": 0.3,
            },
            "editable_fields": ["description", "prior"],
            "requires_input": True,
            "output_key": "drug_hypothesis",
            "validation": {"type": "field_present", "check": "result.id"},
        },
        {
            "id": "bt-2",
            "title": "Add Supporting Evidence",
            "instruction": "A clinical trial shows the drug lowered blood pressure by 8 mmHg (p=0.02). This is moderately strong evidence. Add it as evidence supporting your hypothesis with strength 0.8.",
            "tool": "synara",
            "action": "add_evidence",
            "config": {
                "event": "Clinical trial: 8 mmHg reduction, p=0.02",
                "strength": 0.8,
            },
            "editable_fields": ["event", "strength"],
            "output_key": "trial_evidence",
            "validation": {"type": "api_success"},
        },
        {
            "id": "bt-3",
            "title": "Observe the Update",
            "instruction": "Look at how the posterior probability changed. The belief engine applied Bayes' theorem — the same math you learned with the calculator above. Try adjusting the evidence strength in the previous step and re-running to see how it affects the update.",
            "tool": "synara",
            "action": "get_state",
            "config": {},
            "output_key": "final_state",
            "validation": {"type": "api_success"},
        },
    ],
    "sandbox_config": {
        "create_project": True,
        "project_title": "Bayesian Thinking Lab",
        "synara_enabled": True,
        "tools_available": ["synara"],
    },
    "workflow": {
        "type": "linear",
        "completion_requires": "all_steps",
    },
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
            "Enter one piece of evidence and watch probabilities update",
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
        },
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
            "hint": "Think about different categories: technical, external, measurement",
        },
        {
            "question": "You think Hypothesis A is correct. What should you do next?",
            "answer": "Try to disprove it. Look for evidence that would contradict Hypothesis A. If you can't find any, your confidence is justified.",
            "hint": "Avoid confirmation bias",
        },
    ],
    # Interactive tutorial: Synara hypothesis-driven investigation
    "tool_steps": [
        {
            "id": "hd-1",
            "title": "Create Competing Hypotheses",
            "instruction": "Scenario: A manufacturing line's defect rate jumped from 2% to 8% last week. Before looking at data, create your first hypothesis about what caused it. Assign a prior probability reflecting your initial belief.",
            "tool": "synara",
            "action": "add_hypothesis",
            "config": {
                "description": "",
                "behavior_class": "defect_increase",
                "prior": 0.4,
            },
            "editable_fields": ["description", "prior"],
            "requires_input": True,
            "output_key": "hypothesis_1",
            "validation": {"type": "field_present", "check": "result.id"},
        },
        {
            "id": "hd-2",
            "title": "Add a Second Hypothesis",
            "instruction": "Now add a competing explanation. Good investigations always maintain multiple hypotheses. Think about different categories: material, machine, method, measurement, man, environment.",
            "tool": "synara",
            "action": "add_hypothesis",
            "config": {
                "description": "",
                "behavior_class": "defect_increase",
                "prior": 0.3,
            },
            "editable_fields": ["description", "prior"],
            "requires_input": True,
            "output_key": "hypothesis_2",
            "validation": {"type": "field_present", "check": "result.id"},
        },
        {
            "id": "hd-3",
            "title": "Add a Third Hypothesis",
            "instruction": "Add one more competing explanation. The best investigators consider the non-obvious: measurement errors, seasonal effects, supplier changes. Make sure your priors across all three hypotheses reflect your relative confidence.",
            "tool": "synara",
            "action": "add_hypothesis",
            "config": {
                "description": "",
                "behavior_class": "defect_increase",
                "prior": 0.3,
            },
            "editable_fields": ["description", "prior"],
            "requires_input": True,
            "output_key": "hypothesis_3",
            "validation": {"type": "field_present", "check": "result.id"},
        },
        {
            "id": "hd-4",
            "title": "Generate Investigation Data",
            "instruction": "Now let's generate some factory data to investigate. This synthetic dataset has a hidden cause baked in — your job is to find it through evidence, not just looking at the data.",
            "tool": "forge",
            "action": "generate",
            "config": {
                "schema": {
                    "columns": [
                        {
                            "name": "diameter_mm",
                            "type": "numeric",
                            "mean": 25.0,
                            "std": 0.05,
                            "decimals": 3,
                        },
                        {
                            "name": "temperature_c",
                            "type": "numeric",
                            "mean": 22.0,
                            "std": 1.5,
                            "decimals": 1,
                        },
                        {
                            "name": "supplier",
                            "type": "categorical",
                            "values": ["Alpha", "Beta"],
                            "weights": [0.6, 0.4],
                        },
                        {
                            "name": "shift",
                            "type": "categorical",
                            "values": ["day", "night"],
                        },
                        {"name": "defect", "type": "binary", "prob": 0.08},
                    ],
                    "rows": 200,
                    "injections": [
                        {
                            "type": "mean_shift",
                            "column": "temperature_c",
                            "start_row": 140,
                            "delta": 3.0,
                        },
                    ],
                },
            },
            "editable_fields": ["schema.rows"],
            "output_key": "investigation_data",
            "validation": {"type": "api_success"},
        },
        {
            "id": "hd-5",
            "title": "Link Evidence to Hypotheses",
            "instruction": "Look at the data preview above. The temperature column has a shift starting at row 140. Link this observation as evidence. Which hypothesis does this support? Describe the evidence and set the strength based on how diagnostic it is (0.5 = weak, 1.0 = strong).",
            "tool": "synara",
            "action": "add_evidence",
            "config": {
                "event": "",
                "strength": 0.7,
            },
            "editable_fields": ["event", "strength"],
            "requires_input": True,
            "output_key": "investigation_evidence",
            "validation": {"type": "api_success"},
        },
        {
            "id": "hd-6",
            "title": "Review Updated Beliefs",
            "instruction": "Observe how the belief probabilities shifted after adding evidence. The hypothesis most consistent with the temperature shift should now have higher probability. This is hypothesis-driven investigation: you defined explanations first, then let evidence update your beliefs systematically.",
            "tool": "synara",
            "action": "get_state",
            "config": {},
            "output_key": "final_beliefs",
            "validation": {"type": "api_success"},
        },
    ],
    "sandbox_config": {
        "create_project": True,
        "project_title": "Hypothesis Investigation Lab",
        "synara_enabled": True,
        "tools_available": ["synara", "forge"],
    },
    "workflow": {
        "type": "linear",
        "completion_requires": "all_steps",
    },
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
            "Change to a replicated RCT with n=500 and compare",
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
        },
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
            "hint": "Consider sample size, measurement reliability, and potential bias",
        }
    ],
}


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
            "Change prevalence to 10% and see how it transforms the result",
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
        },
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
            "hint": "Use the 1000-person method. Calculate true positives and false positives separately.",
        },
        {
            "question": "Why is base rate neglect worse for rare events?",
            "answer": "When something is rare, even a small false positive rate produces more false positives than true positives. The base rate dominates.",
            "hint": "Think about what happens when there are 1 true case and 1000 non-cases",
        },
    ],
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
            "Identify the lesson: extreme selection guarantees regression",
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
        },
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
            "hint": "Think about why those stores were selected in the first place.",
        },
        {
            "question": "Why do pilots often think punishment works better than praise for trainees?",
            "answer": "After exceptionally good performance (praised), regression makes next attempt worse. After exceptionally bad performance (punished), regression makes next attempt better. It looks like punishment works, but it's just regression.",
            "hint": "Consider what follows extreme performances, on average.",
        },
    ],
}
