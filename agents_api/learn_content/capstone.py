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
            "Define your primary outcome and stopping criteria",
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
            "hint": "Choosing the 'best' test after seeing results is a form of p-hacking",
        },
    ],
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
            "Draft the results section with effect sizes and CIs",
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
            "hint": "The pre-specified primary analysis is the main result, period",
        },
    ],
    "tool_steps": [
        {
            "id": "generate-dataset",
            "title": "Generate Your Capstone Dataset",
            "instruction": "Use Forge to generate a manufacturing quality dataset — your capstone scenario. A factory produces widgets with spec 10.00 ± 0.05mm across 3 machines over 20 days. A process change occurred on day 12 affecting one machine. Your job: detect it, diagnose it, and report on it.",
            "tool": "forge",
            "action": "generate",
            "config": {
                "n_rows": 600,
                "columns": [
                    {"name": "measurement_id", "type": "integer", "min": 1, "max": 600},
                    {"name": "day", "type": "integer", "min": 1, "max": 20},
                    {
                        "name": "machine",
                        "type": "categorical",
                        "categories": ["A", "B", "C"],
                    },
                    {
                        "name": "operator",
                        "type": "categorical",
                        "categories": ["Smith", "Jones", "Garcia"],
                    },
                    {
                        "name": "diameter_mm",
                        "type": "numeric",
                        "mean": 10.001,
                        "std": 0.011,
                    },
                    {
                        "name": "surface_finish",
                        "type": "numeric",
                        "mean": 0.8,
                        "std": 0.15,
                    },
                ],
                "injections": [
                    {
                        "type": "mean_shift",
                        "column": "diameter_mm",
                        "condition": {
                            "column": "day",
                            "gte": 12,
                            "column2": "machine",
                            "equals2": "C",
                        },
                        "shift": 0.020,
                    },
                    {
                        "type": "variance_increase",
                        "column": "diameter_mm",
                        "condition": {
                            "column": "day",
                            "gte": 12,
                            "column2": "machine",
                            "equals2": "C",
                        },
                        "factor": 2.0,
                    },
                ],
            },
            "editable_fields": ["n_rows"],
            "output_key": "capstone_data",
            "requires_input": False,
            "validation": {"type": "api_success"},
        },
        {
            "id": "formulate-hypotheses",
            "title": "Formulate Hypotheses Before Analysis",
            "instruction": "BEFORE analyzing the data, create your primary hypothesis in Synara. What do you think caused the quality shift? Formalize it as a testable hypothesis with a prior probability reflecting your confidence. This mirrors real practice: hypotheses first, analysis second.",
            "tool": "synara",
            "action": "create_hypothesis",
            "config": {
                "title": "Machine C experienced a calibration shift after day 12",
                "description": "The process change (maintenance, tooling, or material batch) affected Machine C specifically, causing an upward mean shift and increased variability in diameter measurements.",
                "prior": 0.6,
            },
            "editable_fields": ["title", "description", "prior"],
            "output_key": "primary_hypothesis",
            "requires_input": True,
            "validation": {"type": "api_success"},
        },
        {
            "id": "link-evidence",
            "title": "Link Exploratory Evidence",
            "instruction": "After exploring the data (you can see the preview from step 1), link what you observe as evidence to your hypothesis. Does Machine C look different after day 12? How strong is the signal? Set the likelihood ratio based on how convincingly the data supports your hypothesis.",
            "tool": "synara",
            "action": "add_evidence",
            "config": {
                "title": "Exploratory data analysis findings",
                "description": "Comparing machine-level summary statistics before and after day 12 — checking for mean shift, variance change, and defect rate differences.",
                "evidence_type": "observation",
                "direction": "supports",
                "likelihood_ratio": 5.0,
            },
            "editable_fields": ["description", "direction", "likelihood_ratio"],
            "input_from": "primary_hypothesis",
            "output_key": "eda_evidence",
            "requires_input": True,
            "validation": {"type": "api_success"},
        },
        {
            "id": "create-fmea",
            "title": "Assess Failure Mode Risks",
            "instruction": "Build an FMEA row for the failure mode you identified. Score Severity (how bad is it if defective widgets ship?), Occurrence (how likely is this failure to recur?), and Detection (how quickly can current monitoring catch it?). The RPN guides where to focus improvement.",
            "tool": "fmea",
            "action": "create_row",
            "config": {
                "failure_mode": "Machine calibration drift",
                "effect": "Diameter out of spec, increased defect rate",
                "cause": "Unverified post-maintenance calibration",
                "severity": 7,
                "occurrence": 5,
                "detection": 5,
                "recommended_action": "Mandatory post-maintenance verification run with SPC check before returning to production",
            },
            "editable_fields": [
                "severity",
                "occurrence",
                "detection",
                "recommended_action",
            ],
            "output_key": "fmea_assessment",
            "requires_input": False,
            "validation": {"type": "api_success"},
        },
        {
            "id": "compile-a3",
            "title": "Compile A3 Report",
            "instruction": "Synthesize your entire capstone into an A3 report. This is the deliverable: background, current condition, root cause, countermeasures, and follow-up — all on one page. A3 thinking forces clarity.",
            "tool": "a3",
            "action": "create",
            "config": {
                "title": "Capstone: Manufacturing Quality Investigation",
                "background": "Widget manufacturing process with 3 machines, spec 10.00 ± 0.05mm. Quality metrics tracked over 20-day period.",
                "current_condition": "Data analysis revealed process shift after day 12, localized to one machine. Mean shift and variance increase detected.",
                "root_cause": "Machine calibration drift following maintenance event. No post-maintenance verification protocol in place.",
                "countermeasures": "1. Post-maintenance verification run (50 widgets, SPC check). 2. Updated calibration SOP. 3. Operator training on SPC rules.",
                "follow_up": "Weekly control chart review. Monthly capability analysis. Quarterly maintenance procedure audit.",
            },
            "editable_fields": [
                "current_condition",
                "root_cause",
                "countermeasures",
                "follow_up",
            ],
            "output_key": "a3_report",
            "requires_input": True,
            "validation": {"type": "api_success"},
        },
        {
            "id": "guide-review",
            "title": "Guide Agent Review",
            "instruction": "Submit your completed analysis for review by the Guide agent. It will evaluate your reasoning, check for common pitfalls (HARKing, overclaiming, missing limitations), and provide structured feedback on your capstone work.",
            "tool": "guide",
            "action": "review",
            "config": {
                "context": "capstone_review",
                "prompt": "Review this student's capstone analysis. They investigated a manufacturing quality problem using data generation, hypothesis formulation, evidence linking, FMEA risk assessment, and A3 reporting. Evaluate: (1) Was the hypothesis formulated before analysis? (2) Is the evidence interpretation calibrated? (3) Are limitations acknowledged? (4) Is the A3 report complete and actionable? Provide structured feedback.",
            },
            "editable_fields": [],
            "output_key": "guide_feedback",
            "requires_input": False,
            "validation": {"type": "api_success"},
        },
    ],
    "sandbox_config": {
        "create_project": True,
        "project_title": "Capstone Project",
        "synara_enabled": True,
        "tools_available": ["forge", "synara", "fmea", "a3", "guide"],
    },
    "workflow": {
        "type": "linear",
        "completion_requires": "all_steps",
    },
}
