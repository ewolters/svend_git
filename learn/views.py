"""Learning module views.

Interactive learning integrated with DSW — learn by doing.
Guided exercises with sample data, inline analysis, progress tracking.
"""

import json
import logging
import uuid

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.http import require_http_methods

from agents_api.learn_content import SHARED_DATASET, get_section_content
from agents_api.llm_service import llm_service

logger = logging.getLogger(__name__)


# =============================================================================
# Course Content Structure
# =============================================================================

COURSE_MODULES = {
    # =========================================================================
    # MODULE 1: FOUNDATIONS OF RIGOROUS THINKING
    # =========================================================================
    "foundations": {
        "id": "foundations",
        "title": "Foundations of Rigorous Thinking",
        "description": "The mental models that separate rigorous analysts from everyone else",
        "order": 1,
        "sections": [
            {
                "id": "bayesian-thinking",
                "title": "Bayesian Thinking",
                "description": "How to update beliefs with evidence, not intuition",
                "topics": [
                    "Why probability is a measure of uncertainty, not frequency",
                    "Priors: what you believe before seeing evidence",
                    "Likelihood ratios: how much evidence shifts belief",
                    "The confidence adjustment for weak evidence",
                    "Odds form vs probability form",
                    "Common Bayesian pitfalls and how to avoid them",
                ],
                "interactive": True,
                "estimated_minutes": 45,
            },
            {
                "id": "base-rate-neglect",
                "title": "Base Rate Neglect",
                "description": "The most common reasoning error in the world",
                "topics": [
                    "What base rates are and why they matter",
                    "The taxi cab problem and medical diagnosis",
                    "Why even experts ignore base rates",
                    "Techniques to force base rate consideration",
                    "Real-world examples: cancer screening, fraud detection",
                ],
                "interactive": True,
                "estimated_minutes": 40,
            },
            {
                "id": "hypothesis-driven",
                "title": "Hypothesis-Driven Investigation",
                "description": "The discipline of structured inquiry",
                "topics": [
                    "Why 'exploring the data' leads to false discoveries",
                    "Generate hypotheses BEFORE looking at data",
                    "Multiple competing hypotheses (ACH method)",
                    "Defining distinguishing evidence upfront",
                    "The murder board: actively trying to disprove",
                    "When to conclude vs keep investigating",
                    "Preregistration and analysis plans",
                ],
                "interactive": True,
                "estimated_minutes": 50,
            },
            {
                "id": "evidence-quality",
                "title": "Evidence Quality & Source Credibility",
                "description": "Not all evidence is created equal",
                "topics": [
                    "The evidence hierarchy: RCTs to anecdotes",
                    "Sample size and the law of large numbers",
                    "Reproducibility and replication crisis",
                    "Conflicts of interest and funding bias",
                    "The confidence discount formula",
                    "Red flags: too-clean data, cherry-picking, HARKing",
                ],
                "interactive": True,
                "estimated_minutes": 40,
            },
            {
                "id": "regression-to-mean",
                "title": "Regression to the Mean",
                "description": "Why extreme results don't persist",
                "topics": [
                    "What regression to the mean actually is",
                    "Why it's not a causal force",
                    "Sports examples: sophomore slump, hot hand",
                    "Medical examples: why patients 'improve' without treatment",
                    "Business examples: why top performers disappoint",
                    "How to account for it in analysis",
                ],
                "interactive": True,
                "estimated_minutes": 35,
            },
        ],
    },
    # =========================================================================
    # MODULE 2: EXPERIMENTAL DESIGN
    # =========================================================================
    "experimental-design": {
        "id": "experimental-design",
        "title": "Experimental Design",
        "description": "How to set up studies that can actually answer your question",
        "order": 2,
        "sections": [
            {
                "id": "randomization-controls",
                "title": "Randomization & Controls",
                "description": "The foundation of causal inference",
                "topics": [
                    "Why randomization eliminates confounding",
                    "Simple vs stratified randomization",
                    "Control groups: placebo, active, waitlist",
                    "Blinding: single, double, triple",
                    "When randomization is impossible: quasi-experiments",
                    "Common randomization failures and how to detect them",
                ],
                "interactive": True,
                "estimated_minutes": 50,
            },
            {
                "id": "power-analysis",
                "title": "Power Analysis",
                "description": "Determining sample size BEFORE you collect data",
                "topics": [
                    "What statistical power is (and isn't)",
                    "Type I and Type II errors: the tradeoff",
                    "Effect size: the most important input",
                    "Calculating required sample size",
                    "Why underpowered studies are worse than no studies",
                    "Power for different test types",
                    "Sensitivity analysis: what if effect is smaller?",
                ],
                "interactive": True,
                "estimated_minutes": 45,
            },
            {
                "id": "blocking-stratification",
                "title": "Blocking & Stratification",
                "description": "Reducing noise to detect signal",
                "topics": [
                    "Why blocking increases precision",
                    "Choosing blocking variables",
                    "Randomized block designs",
                    "Stratified sampling vs stratified randomization",
                    "Matched pairs designs",
                    "When blocking helps and when it doesn't",
                ],
                "interactive": True,
                "estimated_minutes": 40,
            },
            {
                "id": "common-design-flaws",
                "title": "Common Design Flaws",
                "description": "Mistakes that invalidate your study before it starts",
                "topics": [
                    "Selection bias: who's in your sample?",
                    "Attrition bias: who drops out?",
                    "Measurement bias: are you measuring what you think?",
                    "Demand characteristics: do subjects know the hypothesis?",
                    "Hawthorne effect: does observation change behavior?",
                    "Survivorship bias: who's missing from the data?",
                    "Publication bias: the file drawer problem",
                ],
                "interactive": True,
                "estimated_minutes": 45,
            },
        ],
    },
    # =========================================================================
    # MODULE 3: DATA FUNDAMENTALS
    # =========================================================================
    "data-fundamentals": {
        "id": "data-fundamentals",
        "title": "Data Fundamentals",
        "description": "Working with real-world data that's never as clean as textbooks suggest",
        "order": 3,
        "sections": [
            {
                "id": "data-cleaning",
                "title": "Data Cleaning",
                "description": "Garbage in, garbage out - how to prevent it",
                "topics": [
                    "Types of missingness: MCAR, MAR, MNAR",
                    "Imputation strategies and their tradeoffs",
                    "Outlier detection: IQR, z-score, domain knowledge",
                    "When to remove vs winsorize vs transform",
                    "Data validation and consistency checks",
                    "Duplicate detection and resolution",
                    "Documenting cleaning decisions for reproducibility",
                ],
                "interactive": True,
                "estimated_minutes": 50,
            },
            {
                "id": "sampling",
                "title": "Sampling Methods",
                "description": "How samples represent (or misrepresent) populations",
                "topics": [
                    "Simple random sampling",
                    "Stratified sampling for subgroup analysis",
                    "Cluster sampling for efficiency",
                    "Systematic sampling and its hidden dangers",
                    "Sample size determination formulas",
                    "Selection bias: the silent killer",
                    "Non-response bias and what to do about it",
                ],
                "interactive": True,
                "estimated_minutes": 45,
            },
            {
                "id": "distributions",
                "title": "Distributions & Transformations",
                "description": "Understanding data shapes and when they matter",
                "topics": [
                    "The normal distribution and central limit theorem",
                    "Skewness, kurtosis, and what they tell you",
                    "Common non-normal distributions in practice",
                    "When normality matters and when it doesn't",
                    "Log, sqrt, Box-Cox transformations",
                    "Checking assumptions visually and statistically",
                ],
                "interactive": True,
                "estimated_minutes": 40,
            },
            {
                "id": "eda",
                "title": "Exploratory Data Analysis",
                "description": "Looking at data without fooling yourself",
                "topics": [
                    "The EDA mindset: curiosity without commitment",
                    "Univariate exploration: distributions and outliers",
                    "Bivariate exploration: relationships and patterns",
                    "Visualization best practices",
                    "EDA vs hypothesis testing: the bright line",
                    "Documenting EDA to prevent HARKing",
                ],
                "interactive": True,
                "estimated_minutes": 35,
            },
        ],
    },
    # =========================================================================
    # MODULE 4: STATISTICAL INFERENCE
    # =========================================================================
    "statistical-inference": {
        "id": "statistical-inference",
        "title": "Statistical Inference",
        "description": "Drawing valid conclusions from data",
        "order": 4,
        "sections": [
            {
                "id": "choosing-tests",
                "title": "Choosing the Right Test",
                "description": "A decision framework for statistical tests",
                "topics": [
                    "The decision tree: outcome type, groups, pairing",
                    "Comparing means: t-test vs ANOVA vs non-parametric",
                    "Comparing proportions: chi-square, Fisher's exact",
                    "Correlation: Pearson vs Spearman vs Kendall",
                    "Regression: linear, logistic, Poisson",
                    "When to go non-parametric",
                    "Checking assumptions before running tests",
                ],
                "interactive": True,
                "estimated_minutes": 50,
            },
            {
                "id": "p-values-deep-dive",
                "title": "P-Values: The Deep Dive",
                "description": "What p-values actually mean (and don't mean)",
                "topics": [
                    "The precise definition of a p-value",
                    "What p < 0.05 does NOT mean",
                    "Why p-values are uniformly distributed under null",
                    "P-value as continuous measure vs binary threshold",
                    "The replication crisis and p-value abuse",
                    "Alternatives: confidence intervals, effect sizes, Bayes factors",
                ],
                "interactive": True,
                "estimated_minutes": 45,
            },
            {
                "id": "confidence-intervals",
                "title": "Confidence Intervals",
                "description": "More useful than p-values in almost every situation",
                "topics": [
                    "What a 95% CI actually means",
                    "CI width and what it tells you",
                    "CIs for means, proportions, differences",
                    "Overlapping CIs don't mean 'no difference'",
                    "Reporting CIs alongside point estimates",
                    "Bayesian credible intervals vs frequentist CIs",
                ],
                "interactive": True,
                "estimated_minutes": 40,
            },
            {
                "id": "effect-sizes",
                "title": "Effect Sizes & Practical Significance",
                "description": "Does the result actually matter?",
                "topics": [
                    "Cohen's d for mean differences",
                    "Correlation coefficients as effect sizes",
                    "Odds ratios and relative risk",
                    "Number needed to treat (NNT)",
                    "Benchmarks: small, medium, large",
                    "Statistical vs practical significance",
                    "When tiny effects matter (and when large effects don't)",
                ],
                "interactive": True,
                "estimated_minutes": 40,
            },
            {
                "id": "multiple-comparisons",
                "title": "Multiple Comparisons & P-Hacking",
                "description": "How to lie with statistics (and how not to)",
                "topics": [
                    "The multiple comparisons problem explained",
                    "Bonferroni, Holm, FDR corrections",
                    "P-hacking: what it is and how it happens",
                    "Garden of forking paths",
                    "Preregistration as protection",
                    "Detecting p-hacking in published research",
                ],
                "interactive": True,
                "estimated_minutes": 45,
            },
        ],
    },
    # =========================================================================
    # MODULE 5: CAUSAL INFERENCE
    # =========================================================================
    "causal-inference": {
        "id": "causal-inference",
        "title": "Causal Inference",
        "description": "Moving from correlation to causation",
        "order": 5,
        "sections": [
            {
                "id": "causal-thinking",
                "title": "Causal Thinking",
                "description": "The framework for causal questions",
                "topics": [
                    "Correlation vs causation: the full picture",
                    "The potential outcomes framework",
                    "Counterfactuals: what would have happened?",
                    "The fundamental problem of causal inference",
                    "Directed acyclic graphs (DAGs)",
                    "Confounders, mediators, colliders",
                ],
                "interactive": True,
                "estimated_minutes": 50,
            },
            {
                "id": "confounding",
                "title": "Confounding & How to Handle It",
                "description": "The enemy of causal claims",
                "topics": [
                    "What confounding is and how to spot it",
                    "Measured vs unmeasured confounders",
                    "Adjustment: regression, matching, stratification",
                    "Propensity scores",
                    "When adjustment makes things worse (collider bias)",
                    "Sensitivity analysis for unmeasured confounding",
                ],
                "interactive": True,
                "estimated_minutes": 50,
            },
            {
                "id": "natural-experiments",
                "title": "Natural Experiments",
                "description": "Finding causation in observational data",
                "topics": [
                    "What makes a natural experiment valid",
                    "Instrumental variables",
                    "Regression discontinuity",
                    "Difference-in-differences",
                    "Synthetic control methods",
                    "Evaluating natural experiment quality",
                ],
                "interactive": True,
                "estimated_minutes": 55,
            },
            {
                "id": "ab-testing-causal",
                "title": "A/B Testing as Causal Inference",
                "description": "The gold standard and its limitations",
                "topics": [
                    "Why A/B tests establish causation",
                    "Threats to A/B test validity",
                    "Network effects and interference",
                    "Long-term effects vs short-term metrics",
                    "When A/B tests are impossible or unethical",
                    "Combining A/B with observational methods",
                ],
                "interactive": True,
                "estimated_minutes": 45,
            },
        ],
    },
    # =========================================================================
    # MODULE 6: CRITICAL EVALUATION
    # =========================================================================
    "critical-evaluation": {
        "id": "critical-evaluation",
        "title": "Critical Evaluation",
        "description": "Reading and evaluating research like a skeptic",
        "order": 6,
        "sections": [
            {
                "id": "reading-papers",
                "title": "Reading Research Papers",
                "description": "How to efficiently evaluate scientific claims",
                "topics": [
                    "Anatomy of a research paper",
                    "What to read first (not the abstract)",
                    "Evaluating methods before looking at results",
                    "Checking for hidden researcher degrees of freedom",
                    "Looking for what's NOT reported",
                    "The difference between finding flaws and dismissing research",
                ],
                "interactive": True,
                "estimated_minutes": 45,
            },
            {
                "id": "spotting-bad-science",
                "title": "Spotting Bad Science",
                "description": "Red flags that indicate unreliable research",
                "topics": [
                    "Sample size too small for claimed effect",
                    "No preregistration for exploratory findings",
                    "Suspiciously round numbers",
                    "P-values just below 0.05",
                    "GRIM and SPRITE tests for data anomalies",
                    "Too many significant results",
                    "Claims that defy prior knowledge",
                    "Conflicts of interest",
                ],
                "interactive": True,
                "estimated_minutes": 40,
            },
            {
                "id": "meta-analysis-literacy",
                "title": "Meta-Analysis Literacy",
                "description": "Understanding systematic reviews and meta-analyses",
                "topics": [
                    "What meta-analysis is and isn't",
                    "Forest plots and how to read them",
                    "Heterogeneity: I² and what it means",
                    "Publication bias detection: funnel plots, Egger's test",
                    "Garbage in, garbage out: quality of included studies",
                    "When meta-analyses disagree",
                ],
                "interactive": True,
                "estimated_minutes": 45,
            },
            {
                "id": "when-not-to-use-statistics",
                "title": "When NOT to Use Statistics",
                "description": "The limits of quantitative analysis",
                "topics": [
                    "Questions that can't be answered with data",
                    "When qualitative methods are more appropriate",
                    "The McNamara fallacy: measuring the measurable",
                    "Goodhart's law: when metrics become targets",
                    "Decision-making under deep uncertainty",
                    "The role of judgment and expertise",
                ],
                "interactive": True,
                "estimated_minutes": 35,
            },
        ],
    },
    # =========================================================================
    # MODULE 7: DSW MASTERY (Hands-on)
    # =========================================================================
    "dsw-mastery": {
        "id": "dsw-mastery",
        "title": "DSW Mastery",
        "description": "Hands-on proficiency with SVEND's analysis tools",
        "order": 7,
        "sections": [
            {
                "id": "dsw-overview",
                "title": "DSW Architecture & Workflow",
                "description": "Understanding the Data Science Workbench",
                "topics": [
                    "Analysis pipeline architecture",
                    "Input data requirements and formats",
                    "Available analysis types",
                    "Configuring analysis parameters",
                    "Interpreting output structure",
                    "Exporting and sharing results",
                ],
                "interactive": True,
                "estimated_minutes": 30,
            },
            {
                "id": "bayesian-ab-hands-on",
                "title": "Bayesian A/B Testing",
                "description": "Running and interpreting Bayesian A/B tests",
                "topics": [
                    "Why Bayesian beats frequentist for A/B",
                    "Setting informative priors",
                    "Reading posterior distributions",
                    "Probability of being best",
                    "Expected loss and decision rules",
                    "Early stopping and peeking",
                    "Hands-on: analyze a real A/B test",
                ],
                "interactive": True,
                "estimated_minutes": 50,
                "hands_on": True,
            },
            {
                "id": "spc-hands-on",
                "title": "SPC & Control Charts",
                "description": "Process monitoring and capability analysis",
                "topics": [
                    "X-bar, R, S, and individuals charts",
                    "Setting control limits correctly",
                    "Control limits vs specification limits",
                    "Capability indices: Cp, Cpk, Pp, Ppk",
                    "Detecting out-of-control conditions",
                    "Western Electric rules",
                    "Hands-on: analyze process data",
                ],
                "interactive": True,
                "estimated_minutes": 50,
                "hands_on": True,
            },
            {
                "id": "regression-hands-on",
                "title": "Regression Analysis",
                "description": "Building and interpreting regression models",
                "topics": [
                    "Linear regression in DSW",
                    "Checking assumptions with diagnostics",
                    "Interpreting coefficients and R²",
                    "Multicollinearity detection",
                    "Logistic regression for binary outcomes",
                    "Model comparison and selection",
                    "Hands-on: build a predictive model",
                ],
                "interactive": True,
                "estimated_minutes": 55,
                "hands_on": True,
            },
            {
                "id": "doe-hands-on",
                "title": "Design of Experiments (DOE)",
                "description": "Creating and analyzing experimental designs",
                "topics": [
                    "Full factorial, fractional, and screening designs",
                    "Choosing the right design for your question",
                    "Response surface methods for optimization",
                    "Power analysis for DOE",
                    "Analyzing effects and interactions",
                    "Confirmation runs",
                    "Hands-on: design and analyze an experiment",
                ],
                "interactive": True,
                "estimated_minutes": 60,
                "hands_on": True,
            },
            {
                "id": "nonparametric-hands-on",
                "title": "Nonparametric Tests in DSW",
                "description": "Running and interpreting rank-based tests",
                "topics": [
                    "Mann-Whitney U test walkthrough",
                    "Kruskal-Wallis with Dunn's post-hoc",
                    "Wilcoxon signed-rank for paired data",
                    "Friedman test for repeated measures",
                    "Effect sizes for nonparametric tests",
                    "Hands-on: compare groups with real data",
                ],
                "interactive": True,
                "estimated_minutes": 45,
                "hands_on": True,
            },
            {
                "id": "time-series-hands-on",
                "title": "Time Series in DSW",
                "description": "Decomposition, ARIMA, and change detection",
                "topics": [
                    "Decomposition walkthrough",
                    "ACF/PACF for model identification",
                    "ARIMA and SARIMA modeling",
                    "Change point detection",
                    "Granger causality testing",
                    "Multi-vari analysis for manufacturing",
                    "Hands-on: forecast and detect changes",
                ],
                "interactive": True,
                "estimated_minutes": 55,
                "hands_on": True,
            },
        ],
    },
    # =========================================================================
    # MODULE 8: ADVANCED METHODS
    # =========================================================================
    "advanced-methods": {
        "id": "advanced-methods",
        "title": "Advanced Methods",
        "description": "Specialized techniques for complex real-world problems",
        "order": 8,
        "sections": [
            {
                "id": "nonparametric-tests",
                "title": "Nonparametric Tests",
                "description": "When assumptions fail, ranks prevail",
                "topics": [
                    "When to use nonparametric over parametric",
                    "Mann-Whitney U: two independent groups",
                    "Wilcoxon signed-rank: paired comparisons",
                    "Kruskal-Wallis: three or more groups",
                    "Friedman: repeated measures",
                    "Post-hoc tests: Dunn's and Games-Howell",
                    "Power considerations and effect sizes",
                ],
                "interactive": True,
                "estimated_minutes": 45,
            },
            {
                "id": "time-series-analysis",
                "title": "Time Series Analysis",
                "description": "Methods for data with temporal dependence",
                "topics": [
                    "Stationarity and why it matters",
                    "ACF/PACF for model identification",
                    "Decomposition: trend, seasonal, residual",
                    "ARIMA and SARIMA models",
                    "Change point detection",
                    "Granger causality",
                    "Multi-vari analysis",
                ],
                "interactive": True,
                "estimated_minutes": 55,
            },
            {
                "id": "survival-reliability",
                "title": "Survival & Reliability Analysis",
                "description": "Time-to-event data with censoring",
                "topics": [
                    "Censoring: why it matters and how to handle it",
                    "Kaplan-Meier survival curves",
                    "Log-rank test for group comparisons",
                    "Weibull distribution and failure patterns",
                    "Cox proportional hazards regression",
                    "Warranty analysis and preventive maintenance",
                ],
                "interactive": True,
                "estimated_minutes": 50,
            },
            {
                "id": "ml-essentials",
                "title": "Machine Learning Essentials",
                "description": "Prediction, clustering, and anomaly detection",
                "topics": [
                    "When ML vs traditional statistics",
                    "K-Means clustering and silhouette analysis",
                    "PCA for dimensionality reduction",
                    "Feature importance with Random Forest",
                    "Anomaly detection with Isolation Forest",
                    "Regularized regression (Ridge, Lasso, Elastic Net)",
                    "Gaussian Process regression with uncertainty",
                ],
                "interactive": True,
                "estimated_minutes": 55,
            },
            {
                "id": "measurement-systems",
                "title": "Measurement System Analysis",
                "description": "Verifying your measurements before trusting your data",
                "topics": [
                    "Why measurement systems need evaluation",
                    "Gage R&R: repeatability and reproducibility",
                    "Acceptance criteria for measurement systems",
                    "Acceptance sampling plans",
                    "Bias and linearity studies",
                    "The measurement system audit",
                ],
                "interactive": True,
                "estimated_minutes": 40,
            },
        ],
    },
    # =========================================================================
    # MODULE 9: CASE STUDIES
    # =========================================================================
    "case-studies": {
        "id": "case-studies",
        "title": "Case Studies",
        "description": "Apply everything you've learned to real scenarios",
        "order": 9,
        "sections": [
            {
                "id": "case-clinical-trial",
                "title": "Case: Clinical Trial Analysis",
                "description": "Analyze a drug efficacy trial with complications",
                "topics": [
                    "Scenario: Phase III trial with dropout",
                    "Handling missing outcome data",
                    "Intent-to-treat vs per-protocol analysis",
                    "Subgroup analyses and multiplicity",
                    "Interpreting hazard ratios",
                    "Writing up findings",
                ],
                "interactive": True,
                "estimated_minutes": 60,
                "case_study": True,
            },
            {
                "id": "case-ab-test",
                "title": "Case: E-commerce A/B Test",
                "description": "Analyze an A/B test that's not as simple as it looks",
                "topics": [
                    "Scenario: checkout flow test with novelty effect",
                    "Detecting novelty/learning effects",
                    "Segmentation: new vs returning users",
                    "Revenue vs conversion tradeoffs",
                    "Long-term vs short-term metrics",
                    "Making a recommendation",
                ],
                "interactive": True,
                "estimated_minutes": 55,
                "case_study": True,
            },
            {
                "id": "case-manufacturing",
                "title": "Case: Manufacturing Quality Control",
                "description": "Investigate a process gone wrong",
                "topics": [
                    "Scenario: sudden increase in defect rate",
                    "Using control charts to detect change point",
                    "Root cause investigation with data",
                    "Capability analysis after fix",
                    "Setting up ongoing monitoring",
                    "Presenting to operations team",
                ],
                "interactive": True,
                "estimated_minutes": 55,
                "case_study": True,
            },
            {
                "id": "case-observational",
                "title": "Case: Observational Study Critique",
                "description": "Evaluate a published study making causal claims",
                "topics": [
                    "Scenario: news headline claims X causes Y",
                    "Reading the original paper",
                    "Identifying threats to validity",
                    "Assessing confounding adjustment",
                    "Alternative explanations",
                    "What we can and can't conclude",
                ],
                "interactive": True,
                "estimated_minutes": 50,
                "case_study": True,
            },
        ],
    },
    # =========================================================================
    # MODULE 10: CAPSTONE PROJECT
    # =========================================================================
    "capstone": {
        "id": "capstone",
        "title": "Capstone Project",
        "description": "Demonstrate your skills with a comprehensive analysis",
        "order": 10,
        "sections": [
            {
                "id": "capstone-overview",
                "title": "Capstone Overview",
                "description": "Requirements and evaluation criteria",
                "topics": [
                    "Project requirements and scope",
                    "Evaluation rubric",
                    "Common pitfalls to avoid",
                    "Documentation standards",
                    "Submission process",
                ],
                "interactive": False,
                "estimated_minutes": 20,
            },
            {
                "id": "capstone-project",
                "title": "Capstone Project",
                "description": "Complete a full analysis from question to conclusion",
                "topics": [
                    "Choose from provided datasets",
                    "Define research question and hypotheses",
                    "Conduct appropriate analysis",
                    "Document methodology and limitations",
                    "Present findings with appropriate uncertainty",
                    "Peer review component",
                ],
                "interactive": True,
                "estimated_minutes": 180,
                "capstone": True,
            },
        ],
    },
    # =========================================================================
    # MODULE 11: PROBABILISTIC BAYESIAN SPC (PBS)
    # =========================================================================
    "pbs-mastery": {
        "id": "pbs-mastery",
        "title": "Probabilistic Bayesian SPC (PBS)",
        "description": "The paradigm shift from classical SPC to probabilistic process monitoring — your process has beliefs, learn to update them",
        "order": 11,
        "sections": [
            {
                "id": "pbs-paradigm-shift",
                "title": "The Paradigm Shift: From Rules to Beliefs",
                "description": "Why classical SPC with fixed limits and binary rules is fundamentally broken — and how Bayesian beliefs fix it",
                "topics": [
                    "The 5 fatal flaws of classical SPC",
                    "The Normal-Gamma posterior: encoding process beliefs in four parameters",
                    "O(1) belief updates: each observation refines the belief, not recomputes",
                    "The prior: encoding domain knowledge before seeing data",
                    "PBS vs classical SPC: side-by-side on real data",
                    "Why Minitab and JMP cannot offer this",
                ],
                "interactive": True,
                "estimated_minutes": 40,
                "hands_on": True,
            },
            {
                "id": "pbs-change-detection",
                "title": "Detecting Change: BOCPD & E-Detector",
                "description": "Two independent methods that answer 'has the process shifted?' with a probability, not a binary flag",
                "topics": [
                    "Bayesian Online Changepoint Detection (Adams & MacKay 2007)",
                    "Shift probability: P(change) from 0 to 1, not binary in/out",
                    "Alert cascade: nominal → watch → alert → alarm",
                    "Empirical Bayes over the hazard rate grid",
                    "The E-Detector: distribution-free sequential changepoint detection",
                    "Corroboration: when two independent detectors agree",
                    "Dm-BOCD: handling outliers without discarding data",
                    "Hands-on: detect the hidden shift in manufacturing data",
                ],
                "interactive": True,
                "estimated_minutes": 55,
                "hands_on": True,
            },
            {
                "id": "pbs-evidence-accumulation",
                "title": "Accumulating Evidence: Anytime-Valid E-Values",
                "description": "Evidence that grows over time without the statistical sin of peeking",
                "topics": [
                    "The peeking problem: why checking p-values inflates error rates",
                    "E-values: evidence ratios that are always valid (Grunwald 2024)",
                    "Accumulation: evidence compounds multiplicatively over time",
                    "Evidence levels: none → notable → strong → decisive",
                    "The critical difference from p-values: check anytime without correction",
                    "Practical application: when to escalate based on evidence strength",
                    "Hands-on: watch evidence accumulate as data streams in",
                ],
                "interactive": True,
                "estimated_minutes": 45,
                "hands_on": True,
            },
            {
                "id": "pbs-predictive-adaptive",
                "title": "Seeing the Future: Prediction Fans & Adaptive Limits",
                "description": "Classical charts only look backward — PBS looks forward with calibrated uncertainty",
                "topics": [
                    "The Predictive Chart: Bayesian linear trend on a rolling window",
                    "Prediction fans: where will the process be in 10, 25 observations?",
                    "Spec exceedance probability: P(out of spec) at future horizons",
                    "Slope posterior: is the process trending, and how sure are we?",
                    "Adaptive Control Limits: posterior predictive limits that narrow over time",
                    "Why adaptive limits are better: wide when uncertain, tight when confident",
                    "Hands-on: predict future values and compare adaptive vs classical limits",
                ],
                "interactive": True,
                "estimated_minutes": 50,
                "hands_on": True,
            },
            {
                "id": "pbs-bayesian-capability",
                "title": "Capability with Honesty: Bayesian Cpk",
                "description": "Classical Cpk gives you a number — Bayesian Cpk gives you a distribution with the uncertainty your decisions deserve",
                "topics": [
                    "The problem with point-estimate Cpk at small sample sizes",
                    "Bayesian Cpk: posterior distribution via ancestral sampling",
                    "Credible intervals: 'we are 90% sure Cpk is between X and Y'",
                    "P(Cpk > 1.33): the probability your process meets the standard",
                    "Cpk Trajectory: tracking capability over time with trend detection",
                    "P(Cpk declining): early warning before capability drops",
                    "Hands-on: compare classical point estimate to Bayesian credible interval",
                ],
                "interactive": True,
                "estimated_minutes": 50,
                "hands_on": True,
            },
            {
                "id": "pbs-health-fusion",
                "title": "The Full Picture: Health Fusion & Narrative",
                "description": "Combining SPC, capability, gage, trend, and material signals into one actionable health score",
                "topics": [
                    "MultiStream Health: log-linear fusion of six signal streams",
                    "Why fusion matters: a process can be in-control but failing on capability",
                    "Uncertainty Fusion: accounting for measurement system error in every reading",
                    "The Process Narrative: deterministic summary (no LLM, no hallucinations)",
                    "Investigation Timeline: linking changepoints, evidence, and regimes",
                    "Taguchi Loss: expected quality cost decomposed into bias vs variance",
                    "Hands-on: run a full PBS analysis and interpret the multi-panel output",
                ],
                "interactive": True,
                "estimated_minutes": 50,
                "hands_on": True,
            },
            {
                "id": "pbs-advanced",
                "title": "Advanced PBS: Genealogy & Decision-Theoretic Alarms",
                "description": "Building organizational learning and economic rationality into process monitoring",
                "topics": [
                    "Chart Genealogy: inheriting priors from parent processes",
                    "Transfer factors: how much to trust the parent posterior",
                    "Multi-parent priors: combining knowledge from similar processes",
                    "Probabilistic Alarms: cost-based investigation thresholds",
                    "The optimal threshold formula: balancing missed shifts vs false alarms",
                    "Tuning alarms for safety-critical vs general manufacturing",
                    "The complete PBS workflow: setup → monitor → detect → decide → learn",
                ],
                "interactive": True,
                "estimated_minutes": 40,
                "hands_on": True,
            },
        ],
    },
}

# Assessment settings
PASSING_SCORE = 0.80
ASSESSMENT_QUESTIONS = 25
ASSESSMENT_TIME_MINUTES = 45


# =============================================================================
# Course Content Views
# =============================================================================


@login_required
@require_http_methods(["GET"])
def list_modules(request):
    """List all course modules with user progress."""
    modules = []
    user_progress = _get_user_progress(request.user)

    for module_id, module in sorted(COURSE_MODULES.items(), key=lambda x: x[1]["order"]):
        module_progress = user_progress.get(module_id, {})
        completed_sections = sum(
            1 for s in module["sections"] if module_progress.get(s["id"], {}).get("completed", False)
        )

        modules.append(
            {
                "id": module["id"],
                "title": module["title"],
                "description": module["description"],
                "order": module["order"],
                "section_count": len(module["sections"]),
                "completed_sections": completed_sections,
                "progress_pct": round(completed_sections / len(module["sections"]) * 100),
            }
        )

    return JsonResponse({"modules": modules})


@login_required
@require_http_methods(["GET"])
def get_module(request, module_id: str):
    """Get module details including sections."""
    module = COURSE_MODULES.get(module_id)
    if not module:
        return JsonResponse({"error": "Module not found"}, status=404)

    user_progress = _get_user_progress(request.user)
    module_progress = user_progress.get(module_id, {})

    sections = []
    for section in module["sections"]:
        section_progress = module_progress.get(section["id"], {})
        sections.append(
            {
                **section,
                "completed": section_progress.get("completed", False),
                "completed_at": section_progress.get("completed_at"),
            }
        )

    return JsonResponse(
        {
            "id": module["id"],
            "title": module["title"],
            "description": module["description"],
            "sections": sections,
        }
    )


@login_required
@require_http_methods(["GET"])
def get_section(request, module_id: str, section_id: str):
    """Get section content with rich educational material."""
    module = COURSE_MODULES.get(module_id)
    if not module:
        return JsonResponse({"error": "Module not found"}, status=404)

    section = next((s for s in module["sections"] if s["id"] == section_id), None)
    if not section:
        return JsonResponse({"error": "Section not found"}, status=404)

    user_progress = _get_user_progress(request.user)
    section_progress = user_progress.get(module_id, {}).get(section_id, {})

    # Get rich content if available
    rich_content = get_section_content(section_id)

    response_data = {
        "module_id": module_id,
        "module_title": module["title"],
        **section,
        "completed": section_progress.get("completed", False),
        "completed_at": section_progress.get("completed_at"),
    }

    # Add rich content nested under rich_content key (for frontend)
    if rich_content:
        section_data = rich_content.get("sample_data", {})
        response_data["rich_content"] = {
            "content": rich_content.get("content", ""),
            "intro": rich_content.get("intro", ""),
            "exercise": rich_content.get("exercise", {}),
            "sample_data": section_data if section_data else SHARED_DATASET,
            "key_takeaways": rich_content.get("key_takeaways", []),
            "practice_questions": rich_content.get("practice_questions", []),
            "interactive": rich_content.get("interactive", {}),
        }

    # Include tool_steps and active session for interactive tutorials
    if rich_content and rich_content.get("tool_steps"):
        response_data["tool_steps"] = rich_content["tool_steps"]
        response_data["sandbox_config"] = rich_content.get("sandbox_config", {})
        response_data["workflow"] = rich_content.get("workflow", {})

        session = LearnSession.objects.filter(user=request.user, module_id=module_id, section_id=section_id).first()
        if session:
            response_data["active_session"] = {
                "id": str(session.id),
                "project_id": str(session.project_id) if session.project_id else None,
                "steps_completed": session.steps_completed,
                "state": _sanitize_session_state(session.state),
                "started_at": session.started_at.isoformat(),
            }

    return JsonResponse(response_data)


# =============================================================================
# Progress Tracking
# =============================================================================


@login_required
@require_http_methods(["GET"])
def get_progress(request):
    """Get user's overall course progress."""
    user_progress = _get_user_progress(request.user)

    total_sections = sum(len(m["sections"]) for m in COURSE_MODULES.values())
    completed_sections = 0
    module_progress = {}

    for module_id, module in COURSE_MODULES.items():
        mp = user_progress.get(module_id, {})
        completed_in_module = sum(1 for s in module["sections"] if mp.get(s["id"], {}).get("completed", False))
        completed_sections += completed_in_module
        module_progress[module_id] = {
            "total": len(module["sections"]),
            "completed": completed_in_module,
        }

    # Check if eligible for assessment
    eligible_for_assessment = completed_sections >= total_sections * 0.8  # 80% completion required

    # Get assessment history
    assessment_data = _get_assessment_data(request.user)
    best_score = assessment_data.get("best_score", 0)
    attempts = assessment_data.get("attempts", 0)
    certified = assessment_data.get("certified", False)

    return JsonResponse(
        {
            "total_sections": total_sections,
            "completed_sections": completed_sections,
            "overall_progress_pct": round(completed_sections / total_sections * 100),
            "module_progress": module_progress,
            "eligible_for_assessment": eligible_for_assessment,
            "assessment": {
                "attempts": attempts,
                "best_score": best_score,
                "passing_score": PASSING_SCORE,
                "certified": certified,
            },
        }
    )


@login_required
@require_http_methods(["POST"])
def mark_section_complete(request, module_id: str):
    """Mark a section as complete."""
    try:
        data = json.loads(request.body)
        section_id = data.get("section_id")
    except (json.JSONDecodeError, KeyError):
        return JsonResponse({"error": "section_id required"}, status=400)

    module = COURSE_MODULES.get(module_id)
    if not module:
        return JsonResponse({"error": "Module not found"}, status=404)

    section = next((s for s in module["sections"] if s["id"] == section_id), None)
    if not section:
        return JsonResponse({"error": "Section not found"}, status=404)

    # Enforce workflow completion for tool-integrated sections
    rich_content = get_section_content(section_id)
    if rich_content.get("tool_steps") and rich_content.get("workflow"):
        workflow = rich_content["workflow"]
        if workflow.get("completion_requires") == "all_steps":
            session = LearnSession.objects.filter(user=request.user, module_id=module_id, section_id=section_id).first()
            required = {s["id"] for s in rich_content["tool_steps"]}
            completed = set(session.steps_completed) if session else set()
            remaining = required - completed
            if remaining:
                return JsonResponse(
                    {
                        "error": "Complete all tool steps first",
                        "remaining": sorted(remaining),
                    },
                    status=400,
                )

    # Update progress
    _mark_section_complete(request.user, module_id, section_id)

    return JsonResponse({"success": True, "completed_at": timezone.now().isoformat()})


# =============================================================================
# Assessment
# =============================================================================


@login_required
@require_http_methods(["POST"])
def generate_assessment(request):
    """Generate a new assessment with LLM-generated questions.

    Questions are generated dynamically to prevent answer sharing.
    Each user gets unique questions based on the course content.
    """
    # Check eligibility
    progress = _get_user_progress(request.user)
    total_sections = sum(len(m["sections"]) for m in COURSE_MODULES.values())
    completed = sum(1 for mid, mp in progress.items() for sid, sp in mp.items() if sp.get("completed", False))

    if completed < total_sections * 0.8:
        return JsonResponse(
            {
                "error": "Complete at least 80% of course content before taking assessment",
                "completed": completed,
                "required": int(total_sections * 0.8),
            },
            status=400,
        )

    # Generate questions using Claude
    questions = _generate_assessment_questions(request.user)

    if not questions:
        return JsonResponse(
            {
                "error": "Failed to generate assessment. Please try again.",
            },
            status=500,
        )

    # Store assessment session
    assessment_id = str(uuid.uuid4())
    _store_assessment_session(request.user, assessment_id, questions)

    # Return questions without answers
    client_questions = []
    for i, q in enumerate(questions):
        client_questions.append(
            {
                "id": i,
                "question": q["question"],
                "options": q["options"],
                "topic": q["topic"],
            }
        )

    return JsonResponse(
        {
            "assessment_id": assessment_id,
            "questions": client_questions,
            "time_limit_minutes": ASSESSMENT_TIME_MINUTES,
            "passing_score": PASSING_SCORE,
        }
    )


@login_required
@require_http_methods(["POST"])
def submit_assessment(request):
    """Submit assessment answers and get results."""
    try:
        data = json.loads(request.body)
        assessment_id = data.get("assessment_id")
        answers = data.get("answers", {})  # {question_id: selected_option_index}
    except (json.JSONDecodeError, KeyError):
        return JsonResponse({"error": "Invalid request"}, status=400)

    # Get stored assessment
    assessment = _get_assessment_session(request.user, assessment_id)
    if not assessment:
        return JsonResponse({"error": "Assessment not found or expired"}, status=404)

    # Grade
    questions = assessment["questions"]
    correct = 0
    results = []

    for i, q in enumerate(questions):
        user_answer = answers.get(str(i))
        is_correct = user_answer == q["correct_index"]
        if is_correct:
            correct += 1

        results.append(
            {
                "id": i,
                "question": q["question"],
                "your_answer": (q["options"][user_answer] if user_answer is not None else None),
                "correct_answer": q["options"][q["correct_index"]],
                "is_correct": is_correct,
                "explanation": q.get("explanation", ""),
                "topic": q["topic"],
            }
        )

    score = correct / len(questions)
    passed = score >= PASSING_SCORE

    # Update assessment history
    _record_assessment_attempt(request.user, score, passed)

    return JsonResponse(
        {
            "score": score,
            "correct": correct,
            "total": len(questions),
            "passed": passed,
            "passing_score": PASSING_SCORE,
            "results": results,
        }
    )


@login_required
@require_http_methods(["GET"])
def assessment_history(request):
    """Get user's assessment history."""
    data = _get_assessment_data(request.user)
    return JsonResponse(data)


# =============================================================================
# Helper Functions — backed by SectionProgress, AssessmentAttempt
# =============================================================================

from .models import AssessmentAttempt, LearnSession, SectionProgress


def _get_user_progress(user) -> dict:
    """Get user's course progress as {module_id: {section_id: {completed, completed_at}}}."""
    rows = SectionProgress.objects.filter(user=user, is_completed=True)
    progress = {}
    for row in rows:
        progress.setdefault(row.module_id, {})[row.section_id] = {
            "completed": True,
            "completed_at": row.completed_at.isoformat() if row.completed_at else None,
        }
    return progress


def _mark_section_complete(user, module_id: str, section_id: str):
    """Mark a section as complete for a user."""
    SectionProgress.objects.update_or_create(
        user=user,
        module_id=module_id,
        section_id=section_id,
        defaults={
            "is_completed": True,
            "completed_at": timezone.now(),
        },
    )
    logger.info(f"User {user.id} completed {module_id}/{section_id}")


def _get_assessment_data(user) -> dict:
    """Get user's assessment history."""
    attempts = AssessmentAttempt.objects.filter(user=user, score__isnull=False)
    best = attempts.order_by("-score").first()

    history = [
        {
            "id": str(a.id),
            "score": a.score,
            "passed": a.is_passed,
            "date": a.started_at.isoformat(),
        }
        for a in attempts[:10]
    ]

    return {
        "attempts": attempts.count(),
        "best_score": best.score if best else 0,
        "history": history,
    }


def _store_assessment_session(user, assessment_id: str, questions: list):
    """Store assessment session for grading."""
    AssessmentAttempt.objects.create(
        id=assessment_id,
        user=user,
        questions=questions,
    )
    logger.info(f"Stored assessment {assessment_id} for user {user.id}")


def _get_assessment_session(user, assessment_id: str) -> dict | None:
    """Retrieve assessment session."""
    try:
        attempt = AssessmentAttempt.objects.get(id=assessment_id, user=user)
        # Only return if not yet graded (score is null)
        if attempt.score is not None:
            return None
        return {"questions": attempt.questions}
    except AssessmentAttempt.DoesNotExist:
        return None


def _record_assessment_attempt(user, score: float, passed: bool):
    """Record score on the most recent pending assessment."""
    attempt = AssessmentAttempt.objects.filter(user=user, score__isnull=True).order_by("-started_at").first()
    if attempt:
        attempt.score = score
        attempt.is_passed = passed
        attempt.completed_at = timezone.now()
        attempt.save(update_fields=["score", "is_passed", "completed_at"])
    logger.info(f"User {user.id} scored {score:.0%} on assessment (passed={passed})")


def _generate_assessment_questions(user) -> list:
    """Generate assessment questions using Claude.

    Questions are generated dynamically based on course content
    to prevent answer sharing between users.
    """
    # Build prompt for question generation
    topics = []
    for module in COURSE_MODULES.values():
        for section in module["sections"]:
            if not section.get("coming_soon"):
                topics.extend(section["topics"])

    # Select random subset of topics
    import random

    selected_topics = random.sample(topics, min(ASSESSMENT_QUESTIONS, len(topics)))

    prompt = f"""Generate {ASSESSMENT_QUESTIONS} multiple-choice assessment questions for a data science certification exam.

Topics to cover (one question per topic):
{chr(10).join(f"- {t}" for t in selected_topics)}

Requirements:
1. Questions should test understanding, not memorization
2. Include scenario-based questions where appropriate
3. Each question should have exactly 4 options (A, B, C, D)
4. Include a brief explanation for the correct answer
5. Vary difficulty: 40% medium, 40% hard, 20% very hard

Format as JSON array:
[
  {{
    "topic": "topic name",
    "question": "Question text?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_index": 0,
    "explanation": "Why this answer is correct..."
  }}
]

Return ONLY the JSON array, no other text."""

    result = llm_service.chat(
        user,
        prompt,
        system="You are an expert educator creating rigorous assessment questions for a data science certification. Questions should be challenging and test real understanding.",
        context="generation",
        temperature=0.8,
    )

    if not result.success:
        logger.error("Failed to generate assessment questions - LLM error: %s", result.error)
        return None

    try:
        content = result.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        questions = json.loads(content.strip())
        return questions

    except (json.JSONDecodeError, IndexError, KeyError) as e:
        logger.error(f"Failed to parse assessment questions: {e}")
        return None


# =============================================================================
# Interactive Tutorial Sessions
# =============================================================================


def _sanitize_session_state(state: dict) -> dict:
    """Strip large data blobs from session state for API responses.

    Keeps summaries, hypothesis info, and small results but drops
    raw data arrays that could be hundreds of rows.
    """
    sanitized = {}
    for key, value in state.items():
        if not isinstance(value, dict):
            sanitized[key] = value
            continue
        clean = {}
        for k, v in value.items():
            # Skip raw data arrays (Forge output, inline datasets)
            if k == "data" and isinstance(v, (dict, list)):
                clean[k] = {
                    "_truncated": True,
                    "row_count": (len(v) if isinstance(v, list) else len(next(iter(v.values()), []))),
                }
            else:
                clean[k] = v
        sanitized[key] = clean
    return sanitized


def _merge_edits(config: dict, edits: dict, editable_fields: list) -> dict:
    """Merge student edits into step config, respecting editable_fields."""
    import copy

    merged = copy.deepcopy(config)
    for field in editable_fields:
        if field not in edits:
            continue
        # Support dotted paths like "schema.rows"
        parts = field.split(".")
        target = merged
        for part in parts[:-1]:
            if part.isdigit():
                target = target[int(part)]
            else:
                target = target.setdefault(part, {})
        final_key = parts[-1]
        if final_key.isdigit():
            target[int(final_key)] = edits[field]
        else:
            target[final_key] = edits[field]
    return merged


@login_required
@require_http_methods(["POST"])
def start_session(request):
    """Start or resume a learning session for a tool-integrated section.

    Creates a sandbox core.Project and initializes Synara if configured.
    If a session already exists for this user/module/section, resumes it.

    Request: {"module_id": "foundations", "section_id": "hypothesis-driven"}
    Response: {"session_id": "uuid", "project_id": "uuid", "tool_steps": [...], "state": {}, "steps_completed": []}
    """
    try:
        data = json.loads(request.body)
        module_id = data.get("module_id")
        section_id = data.get("section_id")
    except (json.JSONDecodeError, KeyError):
        return JsonResponse({"error": "module_id and section_id required"}, status=400)

    if not module_id or not section_id:
        return JsonResponse({"error": "module_id and section_id required"}, status=400)

    # Validate the section exists and has tool_steps
    rich_content = get_section_content(section_id)
    if not rich_content or not rich_content.get("tool_steps"):
        return JsonResponse({"error": "Section does not support interactive sessions"}, status=400)

    sandbox_config = rich_content.get("sandbox_config", {})

    # Resume existing session
    session = LearnSession.objects.filter(user=request.user, module_id=module_id, section_id=section_id).first()

    if session:
        return JsonResponse(
            {
                "session_id": str(session.id),
                "project_id": str(session.project_id) if session.project_id else None,
                "tool_steps": rich_content["tool_steps"],
                "state": _sanitize_session_state(session.state),
                "steps_completed": session.steps_completed,
                "resumed": True,
            }
        )

    # Create sandbox project if configured
    project = None
    if sandbox_config.get("create_project"):
        from core.models import Project

        project = Project.objects.create(
            user=request.user,
            title=sandbox_config.get("project_title", f"Learn: {section_id}"),
            description=f"Sandbox project for learning session: {section_id}",
            tags=["learn-sandbox", module_id, section_id],
        )
        logger.info(f"Created sandbox project {project.id} for learn session {section_id}")

        # Initialize Synara if needed
        if sandbox_config.get("synara_enabled") and project:
            from agents_api.synara_views import get_synara, save_synara

            synara = get_synara(str(project.id), user=request.user)
            save_synara(str(project.id), synara, user=request.user)

    # Create session
    session = LearnSession.objects.create(
        user=request.user,
        module_id=module_id,
        section_id=section_id,
        project=project,
        state={},
        steps_completed=[],
    )

    return JsonResponse(
        {
            "session_id": str(session.id),
            "project_id": str(project.id) if project else None,
            "tool_steps": rich_content["tool_steps"],
            "state": {},
            "steps_completed": [],
            "resumed": False,
        }
    )


@login_required
@require_http_methods(["POST"])
def execute_step(request, session_id, step_id):
    """Execute a tool step within a learning session.

    Merges student edits into step config, resolves input_from references,
    dispatches to the appropriate tool handler, stores result in session
    state, runs validation, and returns the result.

    Request: {"edits": {"description": "Night shift temperature drift"}}
    Response: {
        "step_id": "step-3",
        "status": "completed" | "failed",
        "result": {... tool result ...},
        "validation": {"passed": true, "message": "..."},
        "next_step": "step-4" | null,
        "steps_completed": ["step-1", "step-2", "step-3"],
        "state": {... sanitized session state ...}
    }
    """
    try:
        session = LearnSession.objects.get(id=session_id, user=request.user)
    except LearnSession.DoesNotExist:
        return JsonResponse({"error": "Session not found"}, status=404)

    # Get the section content and find the step
    rich_content = get_section_content(session.section_id)
    if not rich_content or not rich_content.get("tool_steps"):
        return JsonResponse({"error": "Section has no tool steps"}, status=400)

    tool_steps = rich_content["tool_steps"]
    step = next((s for s in tool_steps if s["id"] == step_id), None)
    if not step:
        return JsonResponse({"error": f"Step {step_id} not found"}, status=404)

    # Check step ordering (linear workflow)
    workflow = rich_content.get("workflow", {})
    if workflow.get("type") == "linear":
        step_idx = next(i for i, s in enumerate(tool_steps) if s["id"] == step_id)
        if step_idx > 0:
            prev_step_id = tool_steps[step_idx - 1]["id"]
            if prev_step_id not in (session.steps_completed or []):
                return JsonResponse({"error": f"Complete step '{prev_step_id}' first"}, status=400)

    # Already completed — return cached result
    if step_id in (session.steps_completed or []):
        output_key = step.get("output_key")
        cached = session.state.get(output_key, {}) if output_key else {}
        next_step = _get_next_step(tool_steps, step_id)
        return JsonResponse(
            {
                "step_id": step_id,
                "status": "completed",
                "result": cached,
                "validation": {"passed": True, "message": "Already completed"},
                "next_step": next_step,
                "steps_completed": session.steps_completed,
                "state": _sanitize_session_state(session.state),
            }
        )

    # Parse student edits
    try:
        body = json.loads(request.body)
        edits = body.get("edits", {})
    except json.JSONDecodeError:
        edits = {}

    # Validate required input
    if step.get("requires_input"):
        editable = step.get("editable_fields", [])
        # Check that at least one editable field was provided
        if not any(edits.get(f) for f in editable):
            missing = [f for f in editable if not edits.get(f)]
            return JsonResponse(
                {
                    "error": "This step requires your input",
                    "missing_fields": missing,
                },
                status=400,
            )

    # Merge edits into config
    config = step.get("config", {})
    editable_fields = step.get("editable_fields", [])
    merged_config = _merge_edits(config, edits, editable_fields) if editable_fields else config

    # Apply auto_fill from prior step outputs
    if step.get("auto_fill"):
        for target_field, source_expr in step["auto_fill"].items():
            value = _resolve_expression(source_expr, session.state)
            if value is not None:
                merged_config[target_field] = value

    # Dispatch to tool handler
    tool = step.get("tool")
    handler = TOOL_DISPATCH.get(tool)
    if not handler:
        return JsonResponse({"error": f"Tool '{tool}' not supported"}, status=400)

    try:
        result = handler(session, step, merged_config, request.user)
    except Exception as e:
        logger.exception(f"Tool step execution failed: {tool}/{step_id}")
        return JsonResponse(
            {
                "step_id": step_id,
                "status": "failed",
                "result": {},
                "validation": {"passed": False, "message": str(e)},
                "next_step": None,
                "steps_completed": session.steps_completed or [],
                "state": _sanitize_session_state(session.state),
            },
            status=500,
        )

    # Run validation
    validation = _validate_step(step, result)

    # Store result and mark step completed
    output_key = step.get("output_key")
    if output_key and validation["passed"]:
        state = session.state or {}
        state[output_key] = result
        session.state = state
        completed = list(session.steps_completed or [])
        if step_id not in completed:
            completed.append(step_id)
        session.steps_completed = completed
        session.save(update_fields=["state", "steps_completed"])

    # Check if all steps are done
    all_step_ids = {s["id"] for s in tool_steps}
    if all_step_ids.issubset(set(session.steps_completed or [])):
        session.completed_at = timezone.now()
        session.save(update_fields=["completed_at"])

    next_step = _get_next_step(tool_steps, step_id) if validation["passed"] else None

    return JsonResponse(
        {
            "step_id": step_id,
            "status": "completed" if validation["passed"] else "failed",
            "result": result,
            "validation": validation,
            "next_step": next_step,
            "steps_completed": session.steps_completed or [],
            "state": _sanitize_session_state(session.state),
        }
    )


@login_required
@require_http_methods(["POST"])
def reset_session(request, session_id):
    """Reset a learning session, clearing all step progress.

    Deletes the sandbox project and creates a fresh one.
    """
    try:
        session = LearnSession.objects.get(id=session_id, user=request.user)
    except LearnSession.DoesNotExist:
        return JsonResponse({"error": "Session not found"}, status=404)

    rich_content = get_section_content(session.section_id)
    sandbox_config = rich_content.get("sandbox_config", {}) if rich_content else {}

    # Delete old sandbox project
    if session.project:
        old_project = session.project
        session.project = None
        session.save(update_fields=["project"])
        old_project.delete()

    # Create fresh sandbox project
    project = None
    if sandbox_config.get("create_project"):
        from core.models import Project

        project = Project.objects.create(
            user=request.user,
            title=sandbox_config.get("project_title", f"Learn: {session.section_id}"),
            description=f"Sandbox project for learning session: {session.section_id}",
            tags=["learn-sandbox", session.module_id, session.section_id],
        )
        if sandbox_config.get("synara_enabled"):
            from agents_api.synara_views import get_synara, save_synara

            synara = get_synara(str(project.id), user=request.user)
            save_synara(str(project.id), synara, user=request.user)

    session.project = project
    session.state = {}
    session.steps_completed = []
    session.completed_at = None
    session.save(update_fields=["project", "state", "steps_completed", "completed_at"])

    return JsonResponse(
        {
            "session_id": str(session.id),
            "project_id": str(project.id) if project else None,
            "state": {},
            "steps_completed": [],
        }
    )


# =============================================================================
# Tool Dispatch — internal Python calls, not HTTP
# =============================================================================


def _execute_synara_step(session, step, config, user):
    """Execute a Synara belief engine step."""
    from agents_api.synara_views import get_synara, save_synara

    if not session.project_id:
        raise ValueError("Synara requires a sandbox project")

    project_id = str(session.project_id)
    synara = get_synara(project_id, user=user)
    action = step.get("action", "")

    if action == "add_hypothesis":
        h = synara.create_hypothesis(
            description=config.get("description", ""),
            domain_conditions=config.get("domain_conditions", {}),
            behavior_class=config.get("behavior_class", ""),
            latent_causes=config.get("latent_causes", []),
            prior=config.get("prior", 0.5),
            source="learn_session",
        )
        save_synara(project_id, synara, user=user)
        return {
            "type": "hypothesis",
            "id": h.id,
            "description": h.description,
            "prior": h.prior,
            "all_hypotheses": [
                {"id": hh.id, "description": hh.description, "probability": hh.prior}
                for hh in synara.get_all_hypotheses()
            ],
        }

    elif action == "add_evidence":
        supports = config.get("supports", [])
        weakens = config.get("weakens", [])
        # Auto-resolve: if no explicit supports/weakens, use all hypotheses
        if not supports and not weakens:
            all_h = synara.get_all_hypotheses()
            supports = [h.id for h in all_h]

        result = synara.create_evidence(
            event=config.get("event", config.get("summary", "")),
            context=config.get("context", {}),
            supports=supports,
            weakens=weakens,
            strength=config.get("strength", config.get("confidence", 0.7)),
            source="learn_session",
            data=config.get("data"),
        )
        save_synara(project_id, synara, user=user)

        return {
            "type": "evidence_update",
            "posteriors": result.posteriors if hasattr(result, "posteriors") else {},
            "most_supported": (result.most_supported if hasattr(result, "most_supported") else None),
            "most_weakened": (result.most_weakened if hasattr(result, "most_weakened") else None),
            "all_hypotheses": [
                {"id": h.id, "description": h.description, "probability": h.prior} for h in synara.get_all_hypotheses()
            ],
        }

    elif action == "add_link":
        link = synara.create_link(
            from_id=config.get("from_id", ""),
            to_id=config.get("to_id", ""),
            mechanism=config.get("mechanism", ""),
            strength=config.get("strength", 0.7),
        )
        save_synara(project_id, synara, user=user)
        return {
            "type": "causal_link",
            "from_id": (link.from_id if hasattr(link, "from_id") else config.get("from_id")),
            "to_id": link.to_id if hasattr(link, "to_id") else config.get("to_id"),
            "mechanism": config.get("mechanism", ""),
        }

    elif action == "get_state":
        hypotheses = synara.get_all_hypotheses()
        return {
            "type": "synara_state",
            "hypotheses": [{"id": h.id, "description": h.description, "probability": h.prior} for h in hypotheses],
            "most_likely": (synara.get_most_likely_cause().id if synara.get_most_likely_cause() else None),
        }

    else:
        raise ValueError(f"Unknown Synara action: {action}")


def _execute_experimenter_step(session, step, config, user):
    """Execute an Experimenter DOE/power analysis step."""
    action = step.get("action", "")

    if action == "power_analysis":
        from agents.experimenter.stats import PowerAnalyzer, interpret_effect_size

        analyzer = PowerAnalyzer()
        test_type = config.get("test_type", "ttest_ind")
        effect_size = config.get("effect_size", 0.5)
        alpha = config.get("alpha", 0.05)
        power = config.get("power", 0.80)
        groups = config.get("groups", 2)

        # Run power analysis based on test type
        if test_type in ("ttest_ind", "ttest_paired"):
            pa = analyzer.power_ttest_ind(effect_size, alpha=alpha, power=power)
        elif test_type == "anova":
            pa = analyzer.power_anova(effect_size, groups=groups, alpha=alpha, power=power)
        elif test_type == "correlation":
            pa = analyzer.power_correlation(effect_size, alpha=alpha, power=power)
        else:
            pa = analyzer.power_ttest_ind(effect_size, alpha=alpha, power=power)

        interpretation = interpret_effect_size(effect_size, test_type)

        return {
            "type": "power_analysis",
            "sample_size": (pa.sample_size if hasattr(pa, "sample_size") else pa.get("sample_size")),
            "sample_size_per_group": getattr(pa, "sample_size_per_group", None),
            "effect_size": effect_size,
            "alpha": alpha,
            "power": power,
            "test_type": test_type,
            "interpretation": (interpretation if isinstance(interpretation, str) else str(interpretation)),
        }

    elif action == "design_experiment":
        from agents.experimenter.doe import DOEGenerator, Factor

        factors = []
        for f in config.get("factors", []):
            factors.append(
                Factor(
                    name=f["name"],
                    levels=f.get("levels", [f.get("low", -1), f.get("high", 1)]),
                    units=f.get("units", ""),
                    categorical=f.get("categorical", False),
                )
            )

        generator = DOEGenerator(seed=config.get("seed", 42))
        design_type = config.get("design_type", "full_factorial")

        if design_type == "full_factorial":
            design = generator.full_factorial(factors, replicates=config.get("replicates", 1))
        elif design_type == "fractional_factorial":
            design = generator.fractional_factorial(factors, resolution=config.get("resolution", 3))
        elif design_type == "ccd":
            design = generator.central_composite(factors)
        elif design_type == "plackett_burman":
            design = generator.plackett_burman(factors)
        else:
            design = generator.full_factorial(factors, replicates=1)

        return {
            "type": "doe_design",
            "design_type": design_type,
            "num_runs": (design.num_runs if hasattr(design, "num_runs") else len(design.runs)),
            "factors": [{"name": f.name, "levels": f.levels} for f in factors],
            "runs": design.runs if hasattr(design, "runs") else [],
            "markdown": design.to_markdown() if hasattr(design, "to_markdown") else "",
        }

    else:
        raise ValueError(f"Unknown experimenter action: {action}")


def _execute_forge_step(session, step, config, user):
    """Execute a Forge synthetic data generation step.

    For learning, we generate small datasets synchronously.
    """
    import numpy as np
    import pandas as pd

    schema = config.get("schema", {})
    n_rows = min(schema.get("rows", 200), 1000)  # Cap at 1000 for learn mode
    columns = schema.get("columns", [])
    injections = schema.get("injections", [])

    if not columns:
        raise ValueError("Schema must define at least one column")

    # Generate data in-process (avoid Forge task queue for small learn datasets)
    data = {}
    for col in columns:
        name = col["name"]
        col_type = col.get("type", "numeric")

        if col_type == "numeric":
            mean = col.get("mean", 0)
            std = col.get("std", 1)
            data[name] = np.random.normal(mean, std, n_rows).tolist()
        elif col_type == "categorical":
            values = col.get("values", ["A", "B"])
            weights = col.get("weights")
            if weights:
                data[name] = np.random.choice(values, n_rows, p=weights).tolist()
            else:
                data[name] = np.random.choice(values, n_rows).tolist()
        elif col_type == "integer":
            low = col.get("low", 0)
            high = col.get("high", 100)
            data[name] = np.random.randint(low, high + 1, n_rows).tolist()
        elif col_type == "binary":
            prob = col.get("prob", 0.5)
            data[name] = np.random.binomial(1, prob, n_rows).tolist()

    # Apply injections (mean shifts, trends, etc.)
    for injection in injections:
        inj_type = injection.get("type")
        col_name = injection.get("column")
        if col_name not in data:
            continue

        if inj_type == "mean_shift":
            start = injection.get("start_row", 0)
            delta = injection.get("delta", 1.0)
            for i in range(start, n_rows):
                data[col_name][i] += delta

        elif inj_type == "trend":
            slope = injection.get("slope", 0.01)
            start = injection.get("start_row", 0)
            for i in range(start, n_rows):
                data[col_name][i] += slope * (i - start)

        elif inj_type == "variance_increase":
            start = injection.get("start_row", 0)
            factor = injection.get("factor", 2.0)
            mean_val = np.mean(data[col_name][:start]) if start > 0 else 0
            for i in range(start, n_rows):
                data[col_name][i] = mean_val + (data[col_name][i] - mean_val) * factor

    # Round numeric columns
    for col in columns:
        if col.get("type") == "numeric" and col.get("decimals") is not None:
            data[col["name"]] = [round(v, col["decimals"]) for v in data[col["name"]]]

    df = pd.DataFrame(data)
    preview = df.head(10).to_dict(orient="records")

    return {
        "type": "generated_data",
        "data": data,
        "row_count": n_rows,
        "columns": [{"name": c["name"], "type": c.get("type", "numeric")} for c in columns],
        "preview": preview,
        "summary": {
            col: {
                "mean": (round(float(np.mean(data[col])), 4) if isinstance(data[col][0], (int, float)) else None),
                "std": (round(float(np.std(data[col])), 4) if isinstance(data[col][0], (int, float)) else None),
                "unique": len(set(data[col])),
            }
            for col in data
        },
    }


def _execute_rca_step(session, step, config, user):
    """Execute an RCA step."""
    from agents_api.models import RCASession as RCASessionModel

    action = step.get("action", "")

    if action == "create_session":
        rca = RCASessionModel.objects.create(
            owner=user,
            title=config.get("title", "Learn: Root Cause Analysis"),
            event=config.get("event", ""),
            chain=[],
            status="in_progress",
        )
        # Link to sandbox project if available
        if session.project:
            rca.project = session.project
            rca.save(update_fields=["project"])

        return {
            "type": "rca_session",
            "id": str(rca.id),
            "title": rca.title,
            "event": rca.event,
            "chain": [],
        }

    elif action == "add_chain_step":
        rca_id = config.get("rca_id") or _get_from_state(session.state, "rca_session", "id")
        if not rca_id:
            raise ValueError("No RCA session found. Run create_session first.")

        rca = RCASessionModel.objects.get(id=rca_id, owner=user)
        chain = list(rca.chain or [])
        chain.append(
            {
                "claim": config.get("claim", ""),
                "supporting_evidence": config.get("supporting_evidence", ""),
                "accepted": True,
            }
        )
        rca.chain = chain
        rca.save(update_fields=["chain"])

        return {
            "type": "rca_chain_step",
            "rca_id": str(rca.id),
            "chain": chain,
            "depth": len(chain),
        }

    elif action == "set_root_cause":
        rca_id = config.get("rca_id") or _get_from_state(session.state, "rca_session", "id")
        if not rca_id:
            raise ValueError("No RCA session found.")

        rca = RCASessionModel.objects.get(id=rca_id, owner=user)
        rca.root_cause = config.get("root_cause", "")
        rca.status = "completed"
        rca.save(update_fields=["root_cause", "status"])

        return {
            "type": "rca_root_cause",
            "rca_id": str(rca.id),
            "root_cause": rca.root_cause,
            "chain": rca.chain,
        }

    else:
        raise ValueError(f"Unknown RCA action: {action}")


def _execute_fmea_step(session, step, config, user):
    """Execute an FMEA step."""
    from agents_api.models import FMEA, FMEARow

    action = step.get("action", "")

    if action == "create_fmea":
        fmea = FMEA.objects.create(
            owner=user,
            project=session.project,
            title=config.get("title", "Learn: FMEA"),
            description=config.get("description", ""),
            fmea_type=config.get("fmea_type", "process"),
        )
        return {
            "type": "fmea",
            "id": str(fmea.id),
            "title": fmea.title,
            "fmea_type": fmea.fmea_type,
            "rows": [],
        }

    elif action == "add_row":
        fmea_id = config.get("fmea_id") or _get_from_state(session.state, "fmea", "id")
        if not fmea_id:
            raise ValueError("No FMEA found. Run create_fmea first.")

        fmea = FMEA.objects.get(id=fmea_id, owner=user)
        row_count = FMEARow.objects.filter(fmea=fmea).count()

        severity = max(1, min(10, int(config.get("severity", 5))))
        occurrence = max(1, min(10, int(config.get("occurrence", 5))))
        detection = max(1, min(10, int(config.get("detection", 5))))

        row = FMEARow.objects.create(
            fmea=fmea,
            sort_order=row_count + 1,
            process_step=config.get("process_step", ""),
            failure_mode=config.get("failure_mode", ""),
            effect=config.get("effect", ""),
            severity=severity,
            cause=config.get("cause", ""),
            occurrence=occurrence,
            current_controls=config.get("current_controls", ""),
            detection=detection,
            recommended_action=config.get("recommended_action", ""),
        )

        return {
            "type": "fmea_row",
            "fmea_id": str(fmea.id),
            "row_id": str(row.id),
            "failure_mode": row.failure_mode,
            "rpn": severity * occurrence * detection,
            "severity": severity,
            "occurrence": occurrence,
            "detection": detection,
        }

    else:
        raise ValueError(f"Unknown FMEA action: {action}")


def _execute_a3_step(session, step, config, user):
    """Execute an A3 report step."""
    from agents_api.models import A3Report

    action = step.get("action", "")

    if action == "create_a3":
        a3 = A3Report.objects.create(
            owner=user,
            project=session.project,
            title=config.get("title", "Learn: A3 Report"),
            background=config.get("background", ""),
            current_condition=config.get("current_condition", ""),
            goal=config.get("goal", ""),
            root_cause=config.get("root_cause", ""),
            countermeasures=config.get("countermeasures", ""),
            implementation_plan=config.get("implementation_plan", ""),
            follow_up=config.get("follow_up", ""),
        )
        return {
            "type": "a3_report",
            "id": str(a3.id),
            "title": a3.title,
            "sections_filled": sum(
                1
                for f in [
                    a3.background,
                    a3.current_condition,
                    a3.goal,
                    a3.root_cause,
                    a3.countermeasures,
                ]
                if f
            ),
        }

    elif action == "update_a3":
        a3_id = config.get("a3_id") or _get_from_state(session.state, "a3_report", "id")
        if not a3_id:
            raise ValueError("No A3 report found. Run create_a3 first.")

        a3 = A3Report.objects.get(id=a3_id, owner=user)
        update_fields = []
        for field in [
            "background",
            "current_condition",
            "goal",
            "root_cause",
            "countermeasures",
            "implementation_plan",
            "follow_up",
        ]:
            if field in config:
                setattr(a3, field, config[field])
                update_fields.append(field)
        if update_fields:
            a3.save(update_fields=update_fields)

        return {
            "type": "a3_report",
            "id": str(a3.id),
            "title": a3.title,
            "updated_sections": update_fields,
            "sections_filled": sum(
                1
                for f in [
                    a3.background,
                    a3.current_condition,
                    a3.goal,
                    a3.root_cause,
                    a3.countermeasures,
                ]
                if f
            ),
        }

    else:
        raise ValueError(f"Unknown A3 action: {action}")


def _execute_vsm_step(session, step, config, user):
    """Execute a VSM step."""
    from agents_api.models import ValueStreamMap

    action = step.get("action", "")

    if action == "create_vsm":
        vsm = ValueStreamMap.objects.create(
            owner=user,
            project=session.project,
            title=config.get("title", "Learn: Value Stream Map"),
        )
        return {
            "type": "vsm",
            "id": str(vsm.id),
            "title": vsm.title,
        }

    else:
        raise ValueError(f"Unknown VSM action: {action}")


def _execute_guide_step(session, step, config, user):
    """Execute a Guide agent chat step."""
    message = config.get("message", "")
    config.get("context", "project")

    # Build context data from session state
    data = {}
    if session.project:
        data["project"] = {
            "title": session.project.title,
            "description": session.project.description or "",
        }

    # Include session state summary as context
    if session.state:
        data["session_state"] = {
            k: v.get("type", "unknown") if isinstance(v, dict) else str(v) for k, v in session.state.items()
        }

    result = llm_service.chat(
        user,
        message,
        system="You are the Guide agent for a learning session. The student is working through an interactive tutorial and needs guidance. Be concise, supportive, and educational.",
        context="chat",
        max_tokens=1024,
    )

    if not result.success:
        return {
            "type": "guide_response",
            "response": "Guide is currently unavailable. Please continue with the next step.",
        }

    return {
        "type": "guide_response",
        "response": result.content,
    }


# Tool dispatch table
TOOL_DISPATCH = {
    "synara": _execute_synara_step,
    "experimenter": _execute_experimenter_step,
    "forge": _execute_forge_step,
    "rca": _execute_rca_step,
    "fmea": _execute_fmea_step,
    "a3": _execute_a3_step,
    "vsm": _execute_vsm_step,
    "guide": _execute_guide_step,
}


# =============================================================================
# Step Helpers
# =============================================================================


def _get_next_step(tool_steps: list, current_step_id: str) -> str | None:
    """Get the next step ID after the current one, or None if last."""
    for i, step in enumerate(tool_steps):
        if step["id"] == current_step_id and i + 1 < len(tool_steps):
            return tool_steps[i + 1]["id"]
    return None


def _validate_step(step: dict, result: dict) -> dict:
    """Run validation on a step result."""
    validation = step.get("validation", {})
    val_type = validation.get("type", "api_success")

    if val_type == "api_success":
        # Step passes if we got a result without exception
        return {"passed": True, "message": "Step completed successfully"}

    elif val_type == "field_present":
        check = validation.get("check", "")
        # Simple field presence check like "result.id"
        parts = check.replace("result.", "").split(".")
        obj = result
        for part in parts:
            if isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return {
                    "passed": False,
                    "message": f"Expected field '{check}' not found in result",
                }
        return {"passed": True, "message": "Validation passed"}

    elif val_type == "result_check":
        # JS-style expression — for now, just check truthiness of the referenced field
        check = validation.get("check", "")
        # Simple parsing: "result.field && result.field.length > 0"
        # We just check the field exists and is truthy
        field = check.split("&&")[0].strip().replace("result.", "")
        parts = field.split(".")
        obj = result
        for part in parts:
            if isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return {"passed": False, "message": f"Validation check failed: {check}"}
        if obj:
            return {"passed": True, "message": "Validation passed"}
        return {"passed": False, "message": f"Validation check failed: {check}"}

    return {"passed": True, "message": "No validation configured"}


def _resolve_expression(expr: str, state: dict):
    """Resolve a simple dot-path expression against session state.

    Example: "spc_result.guide_observation" → state["spc_result"]["guide_observation"]
    Supports || for fallback: "spc_result.guide_observation || spc_result.summary"
    """
    alternatives = [e.strip() for e in expr.split("||")]
    for alt in alternatives:
        parts = alt.split(".")
        obj = state
        resolved = True
        for part in parts:
            if isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                resolved = False
                break
        if resolved and obj:
            return obj
    return None


def _get_from_state(state: dict, result_type: str, field: str):
    """Find a value in session state by result type and field.

    Searches through state values for one matching the given type.
    """
    for key, value in state.items():
        if isinstance(value, dict) and value.get("type") == result_type:
            return value.get(field)
    return None
