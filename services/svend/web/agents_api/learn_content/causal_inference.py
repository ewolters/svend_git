"""Learning content: Causal Inference."""

from ._datasets import SHARED_DATASET  # noqa: F401


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


