"""Learning content: Case Studies."""

from ._datasets import SHARED_DATASET  # noqa: F401

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
            "Evaluate the unplanned subgroup analysis for severe vs mild disease",
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
        },
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
            "hint": "What kind of person completes despite side effects?",
        },
        {
            "question": "The subgroup analysis suggests the drug works for severe but not mild disease. What would you tell the company?",
            "answer": "This finding is exploratory, not confirmatory. It could be a chance finding (multiple comparisons). The interaction test should be performed. A new trial specifically in severe patients would be needed to confirm. They should NOT claim efficacy only in severe patients based on this.",
            "hint": "Was this subgroup analysis prespecified?",
        },
    ],
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
            "Draft your recommendation: ship, iterate, or kill",
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
        },
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
            "hint": "What metric captures the full picture?",
        },
        {
            "question": "Week 1 showed +26% lift but week 4 showed +3%. Which is the true effect?",
            "answer": "Week 4 is closer to the true long-term effect. Week 1 was inflated by novelty (users exploring the new design). Ideally, run longer and wait for stabilization. The 9.4% overall is an overestimate.",
            "hint": "What causes the decay pattern?",
        },
    ],
    "tool_steps": [
        {
            "id": "generate-ab-data",
            "title": "Generate A/B Test Data",
            "instruction": "Use Forge to generate an A/B test dataset with 1000 users. Control converts at 3.2% and treatment at 3.5% — the kind of small but real difference you'd see in production. The data also includes a 'user_type' column (new vs returning) to reveal segment differences.",
            "tool": "forge",
            "action": "generate",
            "config": {
                "n_rows": 1000,
                "columns": [
                    {"name": "user_id", "type": "integer", "min": 1, "max": 1000},
                    {
                        "name": "variant",
                        "type": "categorical",
                        "categories": ["control", "treatment"],
                    },
                    {
                        "name": "user_type",
                        "type": "categorical",
                        "categories": ["new", "returning"],
                    },
                    {"name": "converted", "type": "binary", "p": 0.032},
                    {"name": "order_value", "type": "numeric", "mean": 85, "std": 30},
                ],
                "injections": [
                    {
                        "type": "mean_shift",
                        "column": "converted",
                        "condition": {"column": "variant", "equals": "treatment"},
                        "shift": 0.003,
                    },
                    {
                        "type": "mean_shift",
                        "column": "order_value",
                        "condition": {"column": "variant", "equals": "treatment"},
                        "shift": -3,
                    },
                ],
            },
            "editable_fields": ["n_rows"],
            "output_key": "ab_data",
            "requires_input": False,
            "validation": {"type": "api_success"},
        },
        {
            "id": "hypothesis-conversion",
            "title": "Create Competing Hypotheses",
            "instruction": "The data shows conversion up but AOV down. Create two competing hypotheses in Synara: (1) the redesign genuinely improves the user experience, and (2) the conversion lift is a novelty effect that will fade. Give the genuine-improvement hypothesis a moderate prior — you have evidence both ways.",
            "tool": "synara",
            "action": "create_hypothesis",
            "config": {
                "title": "Redesign genuinely improves user experience",
                "description": "The checkout flow redesign removes friction and improves conversion durably. The AOV drop is explained by converting previously-lost low-intent users.",
                "prior": 0.55,
            },
            "editable_fields": ["prior"],
            "output_key": "genuine_hypothesis",
            "requires_input": False,
            "validation": {"type": "api_success"},
        },
        {
            "id": "hypothesis-novelty",
            "title": "Create Novelty Effect Hypothesis",
            "instruction": "Now create the competing hypothesis: the lift is mostly a novelty effect. Users are exploring the new design, which inflates early conversion but will decay. The AOV drop supports this — curious browsers convert but buy less.",
            "tool": "synara",
            "action": "create_hypothesis",
            "config": {
                "title": "Conversion lift is primarily a novelty effect",
                "description": "Users explore the new checkout flow out of curiosity, inflating early conversion. Effect will decay over 2-4 weeks as novelty wears off. Lower AOV is consistent with curiosity-driven low-intent conversions.",
                "prior": 0.45,
            },
            "editable_fields": ["prior"],
            "output_key": "novelty_hypothesis",
            "requires_input": False,
            "validation": {"type": "api_success"},
        },
        {
            "id": "link-evidence",
            "title": "Link Data as Evidence",
            "instruction": "Link the A/B test data as evidence. The segment breakdown (new vs returning users) is key: if new users show durable lift but returning users don't, that supports the genuine-improvement hypothesis. Update your beliefs accordingly.",
            "tool": "synara",
            "action": "add_evidence",
            "config": {
                "title": "A/B test results with segment breakdown",
                "description": "Conversion +9.4% overall but decaying weekly. New users +33%, returning users -2%. AOV down 3.5%. Revenue per user +5.5%.",
                "evidence_type": "experiment",
                "direction": "supports",
                "likelihood_ratio": 3.0,
            },
            "editable_fields": ["likelihood_ratio", "direction"],
            "input_from": "genuine_hypothesis",
            "output_key": "ab_evidence",
            "requires_input": False,
            "validation": {"type": "api_success"},
        },
    ],
    "sandbox_config": {
        "create_project": True,
        "project_title": "A/B Test Analysis Lab",
        "synara_enabled": True,
        "tools_available": ["forge", "synara"],
    },
    "workflow": {
        "type": "linear",
        "completion_requires": "all_steps",
    },
}


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
            "Set up ongoing monitoring with Western Electric rules",
        ],
        "dsw_type": "spc:capability",
        "dsw_config": {
            "var": "diameter_mm",
            "lsl": 24.85,
            "usl": 25.15,
            "target": 25.0,
        },
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
            "hint": "Cp measures spread, Cpk measures spread + centering",
        },
        {
            "question": "Your control chart shows 7 consecutive points above the centerline but all within control limits. Is the process in control?",
            "answer": "No — this violates the Western Electric run rule (7+ consecutive points on one side of the centerline indicates a non-random pattern). Even though individual points are within control limits, the run signals a small but persistent shift in the process mean. Investigate what changed. Don't wait for a point to exceed the control limits.",
            "hint": "Western Electric rules detect non-random patterns even within control limits",
        },
    ],
    "tool_steps": [
        {
            "id": "generate-factory-data",
            "title": "Generate Factory Measurement Data",
            "instruction": "Use Forge to generate 300 widget measurements over 10 days. The first 7 days are stable (mean=10.002, SD=0.012). Day 8-10 show a process shift — mean drifts to 10.018 with increased variance. This simulates Machine #3 going out of calibration after maintenance.",
            "tool": "forge",
            "action": "generate",
            "config": {
                "n_rows": 300,
                "columns": [
                    {"name": "measurement_id", "type": "integer", "min": 1, "max": 300},
                    {"name": "day", "type": "integer", "min": 1, "max": 10},
                    {
                        "name": "machine",
                        "type": "categorical",
                        "categories": ["Machine_1", "Machine_2", "Machine_3"],
                    },
                    {
                        "name": "diameter_mm",
                        "type": "numeric",
                        "mean": 10.002,
                        "std": 0.012,
                    },
                ],
                "injections": [
                    {
                        "type": "mean_shift",
                        "column": "diameter_mm",
                        "condition": {"column": "day", "gte": 8},
                        "shift": 0.016,
                    },
                    {
                        "type": "variance_increase",
                        "column": "diameter_mm",
                        "condition": {"column": "day", "gte": 8},
                        "factor": 1.75,
                    },
                ],
            },
            "editable_fields": [],
            "output_key": "factory_data",
            "requires_input": False,
            "validation": {"type": "api_success"},
        },
        {
            "id": "create-rca",
            "title": "Start Root Cause Analysis",
            "instruction": "Launch an RCA session to investigate the defect spike. Use the 5-Why technique: Why did defects increase? → Why is Machine #3 out of spec? → Why was the calibration wrong? → Why wasn't it verified? → Why is there no post-maintenance check?",
            "tool": "rca",
            "action": "create_session",
            "config": {
                "title": "Widget Defect Rate Increase — 2% to 8%",
                "description": "Defect rate jumped from 2% to 8% in week 4. Control charts show process shift starting around day 8, coinciding with Machine #3 maintenance.",
                "methodology": "five_why",
            },
            "editable_fields": [],
            "output_key": "rca_session",
            "requires_input": False,
            "validation": {"type": "api_success"},
        },
        {
            "id": "build-causal-chain",
            "title": "Build the Causal Chain",
            "instruction": "Add the 5-Why chain to the RCA session. Each step drills deeper into the root cause: symptom → proximate cause → procedural gap → systemic gap → root cause.",
            "tool": "rca",
            "action": "add_chain_step",
            "config": {
                "steps": [
                    {
                        "why": "Why did defect rate increase?",
                        "because": "Widget diameters shifted above spec (mean 10.018 vs 10.002)",
                    },
                    {
                        "why": "Why did diameters shift?",
                        "because": "Machine #3 was producing oversized widgets after Tuesday maintenance",
                    },
                    {
                        "why": "Why was Machine #3 out of calibration?",
                        "because": "Recalibration during maintenance introduced a systematic offset",
                    },
                    {
                        "why": "Why wasn't the calibration error caught?",
                        "because": "No post-maintenance verification run was performed",
                    },
                    {
                        "why": "Why is there no post-maintenance verification?",
                        "because": "Maintenance procedure lacks a mandatory verification step with control chart check",
                    },
                ],
            },
            "editable_fields": [],
            "input_from": "rca_session",
            "output_key": "causal_chain",
            "requires_input": False,
            "validation": {"type": "api_success"},
        },
        {
            "id": "formalize-hypothesis",
            "title": "Formalize Root Cause as Hypothesis",
            "instruction": "Take the RCA conclusion and formalize it as a Synara hypothesis: Machine #3 recalibration is the root cause. Set a high prior — the 5-Why analysis and data strongly point here. Link the factory data as evidence.",
            "tool": "synara",
            "action": "create_hypothesis",
            "config": {
                "title": "Machine #3 recalibration caused the defect spike",
                "description": "Tuesday maintenance recalibration introduced a +0.016mm offset on Machine #3, increasing variance and shifting the mean above UCL. Root cause: no post-maintenance verification step.",
                "prior": 0.90,
            },
            "editable_fields": ["prior"],
            "output_key": "root_cause_hypothesis",
            "requires_input": False,
            "validation": {"type": "api_success"},
        },
        {
            "id": "create-fmea",
            "title": "Create FMEA for Failure Mode",
            "instruction": "Build an FMEA row for this failure mode. Rate Severity (defective widgets reaching customers), Occurrence (how often maintenance causes this), and Detection (how quickly control charts catch it). The RPN tells you how urgently to fix the procedure.",
            "tool": "fmea",
            "action": "create_row",
            "config": {
                "failure_mode": "Machine calibration drift after maintenance",
                "effect": "Widget diameter out of spec, defect rate increase from 2% to 8%",
                "cause": "Maintenance recalibration without post-verification run",
                "severity": 7,
                "occurrence": 4,
                "detection": 6,
                "recommended_action": "Add mandatory 50-widget verification run after any maintenance recalibration, with control chart check before returning to production",
            },
            "editable_fields": ["severity", "occurrence", "detection"],
            "output_key": "fmea_row",
            "requires_input": False,
            "validation": {"type": "api_success"},
        },
        {
            "id": "compile-a3",
            "title": "Compile A3 Report",
            "instruction": "Pull everything together into an A3 report — the standard one-page problem-solving format. It synthesizes: background (defect spike), current condition (data), root cause (RCA), countermeasures (verification step), and follow-up (monitoring plan).",
            "tool": "a3",
            "action": "create",
            "config": {
                "title": "Widget Defect Rate Investigation — Machine #3",
                "background": "Defect rate increased from 2% to 8% during week 4, detected via routine SPC monitoring.",
                "current_condition": "Control charts show process shift starting day 8. Machine #3 output significantly different from Machines 1-2 (mean 10.022 vs 10.001, p<0.001).",
                "root_cause": "Machine #3 recalibration during Tuesday maintenance introduced systematic offset. No post-maintenance verification was performed.",
                "countermeasures": "1. Mandatory 50-widget verification run after any recalibration. 2. Control chart check required before returning machine to production. 3. Updated maintenance SOP.",
                "follow_up": "Monthly capability recalculation. Weekly control chart review. Audit maintenance compliance quarterly.",
            },
            "editable_fields": ["countermeasures", "follow_up"],
            "output_key": "a3_report",
            "requires_input": False,
            "validation": {"type": "api_success"},
        },
    ],
    "sandbox_config": {
        "create_project": True,
        "project_title": "Manufacturing Quality Investigation",
        "synara_enabled": True,
        "tools_available": ["forge", "rca", "synara", "fmea", "a3"],
    },
    "workflow": {
        "type": "linear",
        "completion_requires": "all_steps",
    },
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
            "Write a one-paragraph calibrated conclusion",
        ],
        "dsw_type": "stats:regression",
        "dsw_config": {
            "response": "diameter_mm",
            "predictors": ["weight_g", "roughness_ra"],
        },
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
            "hint": "You can only adjust for what you measure, and even that's imperfect",
        },
        {
            "question": "How would you design a study that COULD establish whether coffee causes longer life?",
            "answer": "The gold standard would be a randomized controlled trial: randomly assign people to drink coffee vs not, follow for decades, measure mortality. This is practically impossible (decades of compliance, ethical concerns about forcing/prohibiting coffee). Pragmatic alternatives: (1) Mendelian randomization using genetic variants that affect coffee metabolism as instruments, (2) long-term randomized trial with a shorter-term biomarker proxy, (3) triangulation across multiple natural experiments with different sources of bias.",
            "hint": "What study design eliminates confounding by definition?",
        },
    ],
}
