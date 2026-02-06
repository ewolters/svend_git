"""Learning module views.

SVEND Analyst Certification - educational content on:
- Bayesian thinking and hypothesis-driven investigation
- Statistical analysis and DSW tools
- Data cleaning and sampling
- Evidence quality and experimental design

Features:
- LLM-generated assessment questions (prevents answer sharing)
- Interactive testing with immediate feedback
- Shareable certificate upon completion
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.utils import timezone

from .llm_manager import LLMManager
from .learn_content import get_section_content, SECTION_CONTENT

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
        ],
    },

    # =========================================================================
    # MODULE 8: CASE STUDIES
    # =========================================================================
    "case-studies": {
        "id": "case-studies",
        "title": "Case Studies",
        "description": "Apply everything you've learned to real scenarios",
        "order": 8,
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
    # MODULE 9: CAPSTONE PROJECT
    # =========================================================================
    "capstone": {
        "id": "capstone",
        "title": "Capstone Project",
        "description": "Demonstrate your skills with a comprehensive analysis",
        "order": 9,
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
    # MODULE 10: SYNARA & VERIFIED REASONING (Future)
    # =========================================================================
    "synara": {
        "id": "synara",
        "title": "Synara & Verified Reasoning",
        "description": "Using AI-assisted reasoning with symbolic verification",
        "order": 10,
        "sections": [
            {
                "id": "reasoning-protocol",
                "title": "The Reasoning Protocol",
                "description": "How Synara structures reasoning",
                "topics": [
                    "OBSERVE → HYPOTHESIZE → TEST → UPDATE → CONCLUDE",
                    "Why structured reasoning beats free-form",
                    "Reading and validating reasoning traces",
                    "When to trust vs verify",
                ],
                "interactive": True,
                "estimated_minutes": 35,
                "coming_soon": True,
            },
            {
                "id": "symbolic-verification",
                "title": "Symbolic Verification",
                "description": "How verification prevents hallucination",
                "topics": [
                    "What symbolic verification means",
                    "Contradiction detection",
                    "Logical consistency checking",
                    "Why this matters for high-stakes decisions",
                ],
                "interactive": True,
                "estimated_minutes": 30,
                "coming_soon": True,
            },
            {
                "id": "human-ai-collaboration",
                "title": "Human-AI Collaboration",
                "description": "Best practices for working with Synara",
                "topics": [
                    "What AI reasoning is good at",
                    "What still requires human judgment",
                    "Effective prompting for reasoning tasks",
                    "Interpreting and validating AI output",
                ],
                "interactive": True,
                "estimated_minutes": 25,
                "coming_soon": True,
            },
        ],
    },
}

# Minimum passing score for certification
PASSING_SCORE = 0.80  # 80% required to pass
ASSESSMENT_QUESTIONS = 25  # More questions for comprehensive assessment
ASSESSMENT_TIME_MINUTES = 45  # More time for harder questions

# Certification levels
CERTIFICATION_LEVELS = {
    "analyst": {
        "title": "SVEND Certified Analyst",
        "description": "Demonstrates proficiency in rigorous analytical thinking and statistical methods",
        "min_modules_completed": 6,  # Must complete 6 of first 8 modules
        "passing_score": 0.80,
    },
    "senior_analyst": {
        "title": "SVEND Senior Analyst",
        "description": "Demonstrates mastery including causal inference and advanced methods",
        "min_modules_completed": 8,  # Must complete 8 modules
        "passing_score": 0.85,
        "requires_capstone": True,
    },
}


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
            1 for s in module["sections"]
            if module_progress.get(s["id"], {}).get("completed", False)
        )

        modules.append({
            "id": module["id"],
            "title": module["title"],
            "description": module["description"],
            "order": module["order"],
            "section_count": len(module["sections"]),
            "completed_sections": completed_sections,
            "progress_pct": round(completed_sections / len(module["sections"]) * 100),
        })

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
        sections.append({
            **section,
            "completed": section_progress.get("completed", False),
            "completed_at": section_progress.get("completed_at"),
        })

    return JsonResponse({
        "id": module["id"],
        "title": module["title"],
        "description": module["description"],
        "sections": sections,
    })


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
        response_data["rich_content"] = {
            "content": rich_content.get("content", ""),
            "key_takeaways": rich_content.get("key_takeaways", []),
            "practice_questions": rich_content.get("practice_questions", []),
            "interactive": rich_content.get("interactive", {}),
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
        completed_in_module = sum(
            1 for s in module["sections"]
            if mp.get(s["id"], {}).get("completed", False)
        )
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

    return JsonResponse({
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
    })


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
    completed = sum(
        1 for mid, mp in progress.items()
        for sid, sp in mp.items()
        if sp.get("completed", False)
    )

    if completed < total_sections * 0.8:
        return JsonResponse({
            "error": "Complete at least 80% of course content before taking assessment",
            "completed": completed,
            "required": int(total_sections * 0.8),
        }, status=400)

    # Generate questions using Claude
    questions = _generate_assessment_questions(request.user)

    if not questions:
        return JsonResponse({
            "error": "Failed to generate assessment. Please try again.",
        }, status=500)

    # Store assessment session
    assessment_id = str(uuid.uuid4())
    _store_assessment_session(request.user, assessment_id, questions)

    # Return questions without answers
    client_questions = []
    for i, q in enumerate(questions):
        client_questions.append({
            "id": i,
            "question": q["question"],
            "options": q["options"],
            "topic": q["topic"],
        })

    return JsonResponse({
        "assessment_id": assessment_id,
        "questions": client_questions,
        "time_limit_minutes": ASSESSMENT_TIME_MINUTES,
        "passing_score": PASSING_SCORE,
    })


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

        results.append({
            "id": i,
            "question": q["question"],
            "your_answer": q["options"][user_answer] if user_answer is not None else None,
            "correct_answer": q["options"][q["correct_index"]],
            "is_correct": is_correct,
            "explanation": q.get("explanation", ""),
            "topic": q["topic"],
        })

    score = correct / len(questions)
    passed = score >= PASSING_SCORE

    # Update assessment history
    _record_assessment_attempt(request.user, score, passed)

    # If passed, generate certificate
    certificate_id = None
    if passed:
        certificate_id = _generate_certificate(request.user)

    return JsonResponse({
        "score": score,
        "correct": correct,
        "total": len(questions),
        "passed": passed,
        "passing_score": PASSING_SCORE,
        "results": results,
        "certificate_id": certificate_id,
    })


@login_required
@require_http_methods(["GET"])
def assessment_history(request):
    """Get user's assessment history."""
    data = _get_assessment_data(request.user)
    return JsonResponse(data)


# =============================================================================
# Certificate
# =============================================================================

@login_required
@require_http_methods(["GET"])
def get_certificate(request):
    """Get user's certificate if they've passed."""
    cert_data = _get_certificate_data(request.user)
    if not cert_data:
        return JsonResponse({"error": "No certificate found. Pass the assessment first."}, status=404)

    return JsonResponse(cert_data)


@require_http_methods(["GET"])
def verify_certificate(request, cert_id: str):
    """Public endpoint to verify a certificate is valid."""
    cert_data = _verify_certificate(cert_id)
    if not cert_data:
        return JsonResponse({"valid": False, "error": "Certificate not found"}, status=404)

    return JsonResponse({
        "valid": True,
        "holder_name": cert_data["holder_name"],
        "issued_date": cert_data["issued_date"],
        "certificate_id": cert_id,
    })


# =============================================================================
# Helper Functions (would use database models in production)
# =============================================================================

def _get_user_progress(user) -> dict:
    """Get user's course progress from their profile."""
    # In production: CourseProgress.objects.filter(user=user)
    # For now, use user profile JSON field or cache
    try:
        if hasattr(user, 'profile') and hasattr(user.profile, 'course_progress'):
            return user.profile.course_progress or {}
    except Exception:
        pass
    return {}


def _mark_section_complete(user, module_id: str, section_id: str):
    """Mark a section as complete for a user."""
    # In production: CourseProgress.objects.update_or_create(...)
    # For now, would update user profile
    logger.info(f"User {user.id} completed {module_id}/{section_id}")


def _get_assessment_data(user) -> dict:
    """Get user's assessment history."""
    # In production: AssessmentAttempt.objects.filter(user=user)
    return {
        "attempts": 0,
        "best_score": 0,
        "certified": False,
        "history": [],
    }


def _store_assessment_session(user, assessment_id: str, questions: list):
    """Store assessment session for grading."""
    # In production: cache or database
    # For now, would use Django cache
    logger.info(f"Stored assessment {assessment_id} for user {user.id}")


def _get_assessment_session(user, assessment_id: str) -> Optional[dict]:
    """Retrieve assessment session."""
    # In production: cache.get(f"assessment:{assessment_id}")
    return None


def _record_assessment_attempt(user, score: float, passed: bool):
    """Record an assessment attempt."""
    logger.info(f"User {user.id} scored {score:.0%} on assessment (passed={passed})")


def _generate_certificate(user) -> str:
    """Generate a certificate for a passing user."""
    # Generate unique certificate ID
    cert_id = hashlib.sha256(
        f"{user.id}:{user.email}:{timezone.now().isoformat()}".encode()
    ).hexdigest()[:16].upper()

    # Format: SVND-XXXX-XXXX-XXXX
    cert_id = f"SVND-{cert_id[:4]}-{cert_id[4:8]}-{cert_id[8:12]}"

    logger.info(f"Generated certificate {cert_id} for user {user.id}")
    return cert_id


def _get_certificate_data(user) -> Optional[dict]:
    """Get certificate data for a user."""
    # In production: Certificate.objects.filter(user=user).first()
    return None


def _verify_certificate(cert_id: str) -> Optional[dict]:
    """Verify a certificate exists and is valid."""
    # In production: Certificate.objects.filter(id=cert_id).first()
    return None


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

    response = LLMManager.chat(
        user=user,
        messages=[{"role": "user", "content": prompt}],
        system="You are an expert educator creating rigorous assessment questions for a data science certification. Questions should be challenging and test real understanding.",
        temperature=0.8,  # Some variation for uniqueness
    )

    if not response:
        logger.error("Failed to generate assessment questions - no response from LLM")
        return None

    try:
        # Parse JSON from response
        content = response["content"]
        # Handle potential markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        questions = json.loads(content.strip())
        return questions

    except (json.JSONDecodeError, IndexError, KeyError) as e:
        logger.error(f"Failed to parse assessment questions: {e}")
        return None
