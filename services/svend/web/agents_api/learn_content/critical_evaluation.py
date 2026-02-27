"""Learning content: Critical Evaluation."""

from ._datasets import SHARED_DATASET  # noqa: F401


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


