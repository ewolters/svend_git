"""Content registry — maps section IDs to content dicts."""

from .advanced_methods import (
    MEASUREMENT_SYSTEMS,
    ML_ESSENTIALS,
    NONPARAMETRIC_TESTS,
    SURVIVAL_RELIABILITY,
    TIME_SERIES_ANALYSIS,
)
from .advanced_statistics import (
    BAYESIAN_DEPTH,
    CATEGORICAL_DATA,
    MIXED_MODELS,
    MULTIVARIATE_ANALYSIS,
    REGRESSION_DIAGNOSTICS,
    RESPONSE_SURFACE,
)
from .capstone import CAPSTONE_OVERVIEW, CAPSTONE_PROJECT
from .case_studies import (
    CASE_AB_TEST,
    CASE_CLINICAL_TRIAL,
    CASE_MANUFACTURING,
    CASE_OBSERVATIONAL,
)
from .causal_inference import (
    AB_TESTING_CAUSAL,
    CAUSAL_THINKING,
    CONFOUNDING,
    NATURAL_EXPERIMENTS,
)
from .critical_evaluation import (
    META_ANALYSIS_LITERACY,
    READING_PAPERS,
    SPOTTING_BAD_SCIENCE,
    WHEN_NOT_TO_USE_STATISTICS,
)
from .data_fundamentals import DATA_CLEANING, DISTRIBUTIONS, EDA, SAMPLING
from .dsw_mastery import (
    BAYESIAN_AB_HANDS_ON,
    DOE_HANDS_ON,
    DSW_OVERVIEW,
    NONPARAMETRIC_HANDS_ON,
    REGRESSION_HANDS_ON,
    SPC_HANDS_ON,
    TIME_SERIES_HANDS_ON,
)
from .experimental_design import (
    BLOCKING_STRATIFICATION,
    COMMON_DESIGN_FLAWS,
    POWER_ANALYSIS,
    RANDOMIZATION_CONTROLS,
)
from .foundations import (
    BASE_RATE_NEGLECT,
    BAYESIAN_THINKING,
    EVIDENCE_QUALITY,
    HYPOTHESIS_DRIVEN,
    REGRESSION_TO_MEAN,
)
from .machine_learning import (
    ML_ENSEMBLE_METHODS,
    ML_FEATURE_ENGINEERING,
    ML_INTERPRETABILITY,
    ML_MODEL_VALIDATION,
    ML_SUPERVISED_CLASSIFICATION,
    ML_SUPERVISED_REGRESSION,
    ML_UNSUPERVISED,
)
from .pbs_mastery import (
    PBS_ADVANCED,
    PBS_BAYESIAN_CAPABILITY,
    PBS_CHANGE_DETECTION,
    PBS_EVIDENCE_ACCUMULATION,
    PBS_HEALTH_FUSION,
    PBS_PARADIGM_SHIFT,
    PBS_PREDICTIVE_ADAPTIVE,
)
from .statistical_inference import (
    CHOOSING_TESTS,
    CONFIDENCE_INTERVALS,
    EFFECT_SIZES,
    INTERPRETING_RESULTS,
    MULTIPLE_COMPARISONS,
    P_VALUES_DEEP_DIVE,
)

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
    # Machine Learning
    "ml-supervised-classification": ML_SUPERVISED_CLASSIFICATION,
    "ml-supervised-regression": ML_SUPERVISED_REGRESSION,
    "ml-unsupervised": ML_UNSUPERVISED,
    "ml-model-validation": ML_MODEL_VALIDATION,
    "ml-feature-engineering": ML_FEATURE_ENGINEERING,
    "ml-ensemble-methods": ML_ENSEMBLE_METHODS,
    "ml-interpretability": ML_INTERPRETABILITY,
    # Advanced Statistics
    "multivariate-analysis": MULTIVARIATE_ANALYSIS,
    "categorical-data": CATEGORICAL_DATA,
    "bayesian-depth": BAYESIAN_DEPTH,
    "mixed-models": MIXED_MODELS,
    "response-surface": RESPONSE_SURFACE,
    "regression-diagnostics": REGRESSION_DIAGNOSTICS,
    # PBS Mastery
    "pbs-paradigm-shift": PBS_PARADIGM_SHIFT,
    "pbs-change-detection": PBS_CHANGE_DETECTION,
    "pbs-evidence-accumulation": PBS_EVIDENCE_ACCUMULATION,
    "pbs-predictive-adaptive": PBS_PREDICTIVE_ADAPTIVE,
    "pbs-bayesian-capability": PBS_BAYESIAN_CAPABILITY,
    "pbs-health-fusion": PBS_HEALTH_FUSION,
    "pbs-advanced": PBS_ADVANCED,
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
