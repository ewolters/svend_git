"""Learning content: Experimental Design."""

from ._datasets import SHARED_DATASET  # noqa: F401


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
    ],
    "tool_steps": [
        {
            "id": "generate-confounded",
            "title": "Generate Confounded Data",
            "instruction": "First, let's see what happens when assignment ISN'T random. We'll use Forge to generate a dataset where treatment assignment is correlated with a confounder (severity). Sicker patients get treatment more often — this is confounding by indication.",
            "tool": "forge",
            "action": "generate",
            "config": {
                "n_rows": 200,
                "columns": [
                    {"name": "patient_id", "type": "integer", "min": 1, "max": 200},
                    {"name": "severity", "type": "numeric", "mean": 50, "std": 15},
                    {"name": "treatment", "type": "binary", "p": 0.5},
                    {"name": "outcome", "type": "numeric", "mean": 70, "std": 10},
                ],
                "injections": [
                    {"type": "correlation", "columns": ["severity", "treatment"], "strength": 0.6},
                    {"type": "correlation", "columns": ["severity", "outcome"], "strength": -0.5},
                ],
            },
            "editable_fields": [],
            "output_key": "confounded_data",
            "requires_input": False,
            "validation": {"type": "api_success"},
        },
        {
            "id": "hypothesize-bias",
            "title": "Create Hypothesis About Bias",
            "instruction": "Look at the data: sicker patients received treatment more often. Create a hypothesis in Synara that the observed treatment-outcome relationship is confounded by severity. Set a high prior — this is textbook confounding by indication.",
            "tool": "synara",
            "action": "create_hypothesis",
            "config": {
                "title": "Treatment-outcome association is confounded by severity",
                "description": "Sicker patients are more likely to receive treatment AND have worse outcomes, creating a spurious negative association between treatment and outcome.",
                "prior": 0.85,
            },
            "editable_fields": ["prior"],
            "output_key": "bias_hypothesis",
            "requires_input": False,
            "validation": {"type": "api_success"},
        },
        {
            "id": "generate-randomized",
            "title": "Generate Properly Randomized Data",
            "instruction": "Now generate a dataset where treatment is randomly assigned — no correlation with severity. The TRUE treatment effect is a +5 improvement in outcome. With random assignment, severity balances across groups automatically.",
            "tool": "forge",
            "action": "generate",
            "config": {
                "n_rows": 200,
                "columns": [
                    {"name": "patient_id", "type": "integer", "min": 1, "max": 200},
                    {"name": "severity", "type": "numeric", "mean": 50, "std": 15},
                    {"name": "treatment", "type": "binary", "p": 0.5},
                    {"name": "outcome", "type": "numeric", "mean": 72, "std": 10},
                ],
                "injections": [
                    {"type": "mean_shift", "column": "outcome", "condition": {"column": "treatment", "equals": 1}, "shift": 5},
                ],
            },
            "editable_fields": [],
            "output_key": "randomized_data",
            "requires_input": False,
            "validation": {"type": "api_success"},
        },
        {
            "id": "link-evidence",
            "title": "Compare & Link Evidence",
            "instruction": "Compare the two datasets. In the confounded data, treatment appears harmful (sicker patients got it). In the randomized data, the true +5 benefit is visible. Link this comparison as evidence to your hypothesis — the confounding was real, and randomization solved it.",
            "tool": "synara",
            "action": "add_evidence",
            "config": {
                "title": "Randomized vs confounded comparison",
                "description": "Confounded data shows spurious negative treatment effect due to severity-treatment correlation. Randomized data reveals true +5 benefit. Confirms confounding by indication.",
                "evidence_type": "experiment",
                "direction": "supports",
                "likelihood_ratio": 8.0,
            },
            "editable_fields": ["likelihood_ratio"],
            "input_from": "bias_hypothesis",
            "output_key": "comparison_evidence",
            "requires_input": False,
            "validation": {"type": "api_success"},
        },
    ],
    "sandbox_config": {
        "create_project": True,
        "project_title": "Randomization & Controls Lab",
        "synara_enabled": True,
        "tools_available": ["forge", "synara"],
    },
    "workflow": {
        "type": "linear",
        "completion_requires": "all_steps",
    },
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
    ],
    "tool_steps": [
        {
            "id": "configure-power",
            "title": "Configure Power Analysis",
            "instruction": "Set up a power analysis for a two-sample t-test. You're planning an experiment to detect a medium effect (d=0.5) with standard settings. Try adjusting the effect size to see how dramatically it changes the required sample size.",
            "tool": "experimenter",
            "action": "power_analysis",
            "config": {
                "test_type": "two_sample_t",
                "effect_size": 0.5,
                "alpha": 0.05,
                "power": 0.80,
            },
            "editable_fields": ["effect_size", "alpha", "power"],
            "output_key": "power_result",
            "requires_input": False,
            "validation": {"type": "api_success"},
        },
        {
            "id": "generate-adequate",
            "title": "Generate Adequately Powered Dataset",
            "instruction": "Now use Forge to generate a dataset at the recommended sample size. The true effect is a mean difference of 5 units (d≈0.5 with σ=10). With adequate n, the effect should be clearly detectable.",
            "tool": "forge",
            "action": "generate",
            "config": {
                "n_rows": 128,
                "columns": [
                    {"name": "participant_id", "type": "integer", "min": 1, "max": 128},
                    {"name": "group", "type": "categorical", "categories": ["control", "treatment"]},
                    {"name": "outcome", "type": "numeric", "mean": 50, "std": 10},
                ],
                "injections": [
                    {"type": "mean_shift", "column": "outcome", "condition": {"column": "group", "equals": "treatment"}, "shift": 5},
                ],
            },
            "editable_fields": ["n_rows"],
            "input_from": "power_result",
            "output_key": "adequate_data",
            "requires_input": False,
            "validation": {"type": "api_success"},
        },
        {
            "id": "generate-underpowered",
            "title": "Generate Underpowered Dataset",
            "instruction": "Now generate the same experiment but with only n=20 (10 per group). The SAME true effect exists, but watch how hard it is to detect with too few participants. This is why power analysis matters — you can waste an entire study by skipping this step.",
            "tool": "forge",
            "action": "generate",
            "config": {
                "n_rows": 20,
                "columns": [
                    {"name": "participant_id", "type": "integer", "min": 1, "max": 20},
                    {"name": "group", "type": "categorical", "categories": ["control", "treatment"]},
                    {"name": "outcome", "type": "numeric", "mean": 50, "std": 10},
                ],
                "injections": [
                    {"type": "mean_shift", "column": "outcome", "condition": {"column": "group", "equals": "treatment"}, "shift": 5},
                ],
            },
            "editable_fields": [],
            "output_key": "underpowered_data",
            "requires_input": False,
            "validation": {"type": "api_success"},
        },
    ],
    "sandbox_config": {
        "create_project": True,
        "project_title": "Power Analysis Lab",
        "synara_enabled": False,
        "tools_available": ["experimenter", "forge"],
    },
    "workflow": {
        "type": "linear",
        "completion_requires": "all_steps",
    },
}


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


