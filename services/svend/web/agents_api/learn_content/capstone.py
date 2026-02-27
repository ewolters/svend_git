"""Learning content: Capstone."""

from ._datasets import SHARED_DATASET  # noqa: F401


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


