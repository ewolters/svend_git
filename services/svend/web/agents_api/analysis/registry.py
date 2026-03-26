"""DSW Analysis Registry — canonical catalog of all 230 analyses.

Every analysis dispatched through dispatch.py is registered here with metadata
used by the post-processor (standardize.py), education system, and compliance
checks.

Schema per entry:
    module:       Source module (e.g. "stats", "spc", "bayesian")
    category:     Functional category for grouping
    has_pvalue:   Whether the analysis produces a p-value
    effect_type:  Effect size family (None if not applicable)
    shadow_type:  Bayesian shadow type from common._bayesian_shadow()
    what_if_tier: 0 = none, 1 = full explorer, 2 = sensitivity display
    has_narrative: Whether the analysis currently returns narrative
    has_education: Whether education content exists in education.py
    has_charts:   Whether the analysis returns plots

CR: 5528303a — INIT-009 / E9-001
"""

# ── Categories ──────────────────────────────────────────────────────────────

CAT_HYPOTHESIS = "hypothesis_testing"
CAT_REGRESSION = "regression"
CAT_NONPARAM = "nonparametric"
CAT_POWER = "power_sample_size"
CAT_TIMESERIES = "time_series"
CAT_SURVIVAL = "survival"
CAT_MSA = "measurement_system"
CAT_POSTHOC = "post_hoc"
CAT_MULTIVARIATE = "multivariate"
CAT_PROPORTION = "proportion"
CAT_EXPLORATORY = "exploratory"
CAT_ACCEPTANCE = "acceptance_sampling"
CAT_SPC = "spc"
CAT_CAPABILITY = "capability"
CAT_ML = "machine_learning"
CAT_BAYESIAN = "bayesian"
CAT_RELIABILITY = "reliability"
CAT_VIZ = "visualization"
CAT_SIMULATION = "simulation"
CAT_DIMENSIONAL = "dimensional"
CAT_CAUSAL = "causal"
CAT_DRIFT = "drift"
CAT_ANYTIME = "anytime_valid"
CAT_QUALITY_ECON = "quality_economics"
CAT_PBS = "process_behaviour"
CAT_ISHAP = "interventional_shap"
CAT_SIOP = "siop"

# ── Effect types ────────────────────────────────────────────────────────────

EFF_COHEN_D = "cohens_d"
EFF_COHENS_F = "cohens_f"
EFF_ETA_SQ = "eta_squared"
EFF_R = "r"
EFF_R2 = "r_squared"
EFF_CRAMERS_V = "cramers_v"
EFF_ODDS_RATIO = "odds_ratio"
EFF_COHENS_H = "cohens_h"
EFF_RANK_BISERIAL = "rank_biserial"
EFF_COHENS_W = "cohens_w"

# ── Shadow types (maps to common._bayesian_shadow) ─────────────────────────

SH_TTEST = "ttest"
SH_TTEST2 = "ttest2"
SH_PAIRED = "paired"
SH_ANOVA = "anova"
SH_CORRELATION = "correlation"
SH_CHI2 = "chi2"
SH_PROPORTION = "proportion"
SH_REGRESSION = "regression"
SH_VARIANCE = "variance"
SH_NONPARAMETRIC = "nonparametric"


def _entry(
    module,
    category,
    has_pvalue=False,
    effect_type=None,
    shadow_type=None,
    what_if_tier=0,
    has_narrative=True,
    has_education=False,
    has_charts=True,
):
    """Build a registry entry dict."""
    return {
        "module": module,
        "category": category,
        "has_pvalue": has_pvalue,
        "effect_type": effect_type,
        "shadow_type": shadow_type,
        "what_if_tier": what_if_tier,
        "has_narrative": has_narrative,
        "has_education": has_education,
        "has_charts": has_charts,
    }


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS_REGISTRY — keyed by (analysis_type, analysis_id)
# ═══════════════════════════════════════════════════════════════════════════

ANALYSIS_REGISTRY = {
    # ── stats.py — hypothesis testing ───────────────────────────────────
    ("stats", "ttest"): _entry("stats", CAT_HYPOTHESIS, True, EFF_COHEN_D, SH_TTEST),
    ("stats", "ttest2"): _entry("stats", CAT_HYPOTHESIS, True, EFF_COHEN_D, SH_TTEST2),
    ("stats", "paired_t"): _entry(
        "stats", CAT_HYPOTHESIS, True, EFF_COHEN_D, SH_PAIRED
    ),
    ("stats", "anova"): _entry("stats", CAT_HYPOTHESIS, True, EFF_ETA_SQ, SH_ANOVA),
    ("stats", "anova2"): _entry("stats", CAT_HYPOTHESIS, True, EFF_ETA_SQ, SH_ANOVA),
    ("stats", "correlation"): _entry(
        "stats", CAT_HYPOTHESIS, True, EFF_R, SH_CORRELATION
    ),
    ("stats", "chi2"): _entry("stats", CAT_HYPOTHESIS, True, EFF_CRAMERS_V, SH_CHI2),
    ("stats", "prop_1sample"): _entry(
        "stats", CAT_PROPORTION, True, EFF_COHENS_H, SH_PROPORTION
    ),
    ("stats", "prop_2sample"): _entry(
        "stats", CAT_PROPORTION, True, EFF_COHENS_H, SH_PROPORTION
    ),
    ("stats", "fisher_exact"): _entry(
        "stats", CAT_HYPOTHESIS, True, EFF_ODDS_RATIO, SH_CHI2
    ),
    ("stats", "poisson_1sample"): _entry(
        "stats", CAT_HYPOTHESIS, True, effect_type=None, shadow_type=None
    ),
    ("stats", "poisson_2sample"): _entry(
        "stats", CAT_HYPOTHESIS, True, effect_type=None, shadow_type=None
    ),
    ("stats", "variance_test"): _entry(
        "stats", CAT_HYPOTHESIS, True, effect_type=None, shadow_type=SH_VARIANCE
    ),
    ("stats", "f_test"): _entry(
        "stats", CAT_HYPOTHESIS, True, effect_type=None, shadow_type=SH_VARIANCE
    ),
    ("stats", "equivalence"): _entry(
        "stats", CAT_HYPOTHESIS, True, EFF_COHEN_D, SH_TTEST
    ),
    ("stats", "normality"): _entry("stats", CAT_EXPLORATORY, True),
    ("stats", "runs_test"): _entry("stats", CAT_HYPOTHESIS, True),
    ("stats", "sign_test"): _entry(
        "stats", CAT_NONPARAM, True, EFF_RANK_BISERIAL, SH_NONPARAMETRIC
    ),
    # ── stats.py — regression ───────────────────────────────────────────
    ("stats", "regression"): _entry(
        "stats", CAT_REGRESSION, True, EFF_R2, SH_REGRESSION, what_if_tier=1
    ),
    ("stats", "logistic"): _entry("stats", CAT_REGRESSION, True, EFF_ODDS_RATIO),
    ("stats", "nominal_logistic"): _entry(
        "stats", CAT_REGRESSION, True, EFF_ODDS_RATIO
    ),
    ("stats", "ordinal_logistic"): _entry(
        "stats", CAT_REGRESSION, True, EFF_ODDS_RATIO
    ),
    ("stats", "orthogonal_regression"): _entry(
        "stats", CAT_REGRESSION, True, EFF_R2, SH_REGRESSION
    ),
    ("stats", "nonlinear_regression"): _entry("stats", CAT_REGRESSION, True, EFF_R2),
    ("stats", "poisson_regression"): _entry("stats", CAT_REGRESSION, True),
    ("stats", "robust_regression"): _entry("stats", CAT_REGRESSION, True, EFF_R2),
    ("stats", "stepwise"): _entry("stats", CAT_REGRESSION, True, EFF_R2, SH_REGRESSION),
    ("stats", "best_subsets"): _entry("stats", CAT_REGRESSION, True, EFF_R2),
    ("stats", "glm"): _entry("stats", CAT_REGRESSION, True),
    # ── stats.py — nonparametric ────────────────────────────────────────
    ("stats", "mann_whitney"): _entry(
        "stats", CAT_NONPARAM, True, EFF_RANK_BISERIAL, SH_NONPARAMETRIC
    ),
    ("stats", "kruskal"): _entry(
        "stats", CAT_NONPARAM, True, EFF_ETA_SQ, SH_NONPARAMETRIC
    ),
    ("stats", "wilcoxon"): _entry(
        "stats", CAT_NONPARAM, True, EFF_RANK_BISERIAL, SH_NONPARAMETRIC
    ),
    ("stats", "friedman"): _entry("stats", CAT_NONPARAM, True, EFF_COHENS_W),
    ("stats", "spearman"): _entry("stats", CAT_NONPARAM, True, EFF_R, SH_CORRELATION),
    ("stats", "mood_median"): _entry("stats", CAT_NONPARAM, True),
    # ── stats.py — ANOVA extensions ────────────────────────────────────
    ("stats", "split_plot_anova"): _entry(
        "stats", CAT_HYPOTHESIS, True, EFF_ETA_SQ, SH_ANOVA
    ),
    ("stats", "repeated_measures_anova"): _entry(
        "stats", CAT_HYPOTHESIS, True, EFF_ETA_SQ, SH_ANOVA
    ),
    ("stats", "nested_anova"): _entry(
        "stats", CAT_HYPOTHESIS, True, EFF_ETA_SQ, SH_ANOVA
    ),
    ("stats", "anom"): _entry("stats", CAT_HYPOTHESIS, True),
    ("stats", "main_effects"): _entry("stats", CAT_HYPOTHESIS, True, EFF_ETA_SQ),
    ("stats", "interaction"): _entry("stats", CAT_HYPOTHESIS, True, EFF_ETA_SQ),
    ("stats", "variance_components"): _entry("stats", CAT_HYPOTHESIS, True),
    # ── stats.py — post-hoc comparisons ────────────────────────────────
    ("stats", "tukey_hsd"): _entry("stats", CAT_POSTHOC, True, EFF_COHEN_D),
    ("stats", "dunnett"): _entry("stats", CAT_POSTHOC, True, EFF_COHEN_D),
    ("stats", "games_howell"): _entry("stats", CAT_POSTHOC, True, EFF_COHEN_D),
    ("stats", "dunn"): _entry("stats", CAT_POSTHOC, True),
    ("stats", "scheffe_test"): _entry("stats", CAT_POSTHOC, True),
    ("stats", "bonferroni_test"): _entry("stats", CAT_POSTHOC, True),
    ("stats", "hsu_mcb"): _entry("stats", CAT_POSTHOC, True),
    # ── stats.py — power & sample size ─────────────────────────────────
    ("stats", "power_z"): _entry("stats", CAT_POWER, False, what_if_tier=1),
    ("stats", "power_1prop"): _entry("stats", CAT_POWER, False, what_if_tier=1),
    ("stats", "power_2prop"): _entry("stats", CAT_POWER, False, what_if_tier=1),
    ("stats", "power_1variance"): _entry("stats", CAT_POWER, False, what_if_tier=1),
    ("stats", "power_2variance"): _entry("stats", CAT_POWER, False, what_if_tier=1),
    ("stats", "power_equivalence"): _entry("stats", CAT_POWER, False, what_if_tier=1),
    ("stats", "power_doe"): _entry("stats", CAT_POWER, False, what_if_tier=1),
    ("stats", "sample_size_ci"): _entry("stats", CAT_POWER, False, what_if_tier=1),
    ("stats", "sample_size_tolerance"): _entry(
        "stats", CAT_POWER, False, what_if_tier=1
    ),
    # ── stats.py — time series ─────────────────────────────────────────
    ("stats", "arima"): _entry("stats", CAT_TIMESERIES, True),
    ("stats", "sarima"): _entry("stats", CAT_TIMESERIES, True),
    ("stats", "decomposition"): _entry("stats", CAT_TIMESERIES, False),
    ("stats", "acf_pacf"): _entry("stats", CAT_TIMESERIES, False),
    ("stats", "granger"): _entry("stats", CAT_TIMESERIES, True),
    ("stats", "changepoint"): _entry("stats", CAT_TIMESERIES, True),
    ("stats", "ccf"): _entry("stats", CAT_TIMESERIES, False),
    # ── stats.py — survival ────────────────────────────────────────────
    ("stats", "weibull"): _entry("stats", CAT_SURVIVAL, True),
    ("stats", "kaplan_meier"): _entry("stats", CAT_SURVIVAL, True),
    ("stats", "cox_ph"): _entry("stats", CAT_SURVIVAL, True),
    # ── stats.py — measurement system analysis ─────────────────────────
    ("stats", "gage_rr"): _entry("stats", CAT_MSA, False),
    ("stats", "gage_rr_nested"): _entry("stats", CAT_MSA, False),
    ("stats", "gage_rr_expanded"): _entry("stats", CAT_MSA, False),
    ("stats", "gage_linearity_bias"): _entry("stats", CAT_MSA, True),
    ("stats", "gage_type1"): _entry("stats", CAT_MSA, False),
    ("stats", "attribute_gage"): _entry("stats", CAT_MSA, False),
    ("stats", "attribute_agreement"): _entry("stats", CAT_MSA, False),
    ("stats", "icc"): _entry("stats", CAT_MSA, True),
    ("stats", "krippendorff_alpha"): _entry("stats", CAT_MSA, False),
    ("stats", "bland_altman"): _entry("stats", CAT_MSA, False),
    # ── stats.py — multivariate ────────────────────────────────────────
    ("stats", "hotelling_t2"): _entry("stats", CAT_MULTIVARIATE, True),
    ("stats", "manova"): _entry("stats", CAT_MULTIVARIATE, True, EFF_ETA_SQ),
    ("stats", "multi_vari"): _entry("stats", CAT_MULTIVARIATE, False),
    # ── stats.py — exploratory & diagnostics ───────────────────────────
    ("stats", "descriptive"): _entry(
        "stats", CAT_EXPLORATORY, False, has_narrative=False
    ),
    ("stats", "data_profile"): _entry("stats", CAT_EXPLORATORY, False),
    ("stats", "auto_profile"): _entry("stats", CAT_EXPLORATORY, False),
    ("stats", "graphical_summary"): _entry("stats", CAT_EXPLORATORY, False),
    ("stats", "missing_data_analysis"): _entry("stats", CAT_EXPLORATORY, False),
    ("stats", "outlier_analysis"): _entry("stats", CAT_EXPLORATORY, False),
    ("stats", "duplicate_analysis"): _entry("stats", CAT_EXPLORATORY, False),
    ("stats", "bootstrap_ci"): _entry("stats", CAT_EXPLORATORY, False),
    ("stats", "box_cox"): _entry("stats", CAT_EXPLORATORY, False),
    ("stats", "johnson_transform"): _entry("stats", CAT_EXPLORATORY, False),
    ("stats", "run_chart"): _entry("stats", CAT_EXPLORATORY, False),
    ("stats", "grubbs_test"): _entry("stats", CAT_EXPLORATORY, True),
    ("stats", "tolerance_interval"): _entry("stats", CAT_EXPLORATORY, False),
    # ── stats.py — acceptance sampling ─────────────────────────────────
    ("stats", "acceptance_sampling"): _entry(
        "stats", CAT_ACCEPTANCE, False, what_if_tier=1
    ),
    ("stats", "variable_acceptance_sampling"): _entry(
        "stats", CAT_ACCEPTANCE, False, what_if_tier=1
    ),
    ("stats", "multiple_plan_comparison"): _entry(
        "stats", CAT_ACCEPTANCE, False, what_if_tier=2
    ),
    # ── stats.py — capability ──────────────────────────────────────────
    ("stats", "capability_sixpack"): _entry(
        "stats", CAT_CAPABILITY, False, what_if_tier=1
    ),
    ("stats", "nonnormal_capability_np"): _entry("stats", CAT_CAPABILITY, False),
    ("stats", "attribute_capability"): _entry("stats", CAT_CAPABILITY, False),
    # ── stats.py — meta-analysis ───────────────────────────────────────
    ("stats", "meta_analysis"): _entry("stats", CAT_HYPOTHESIS, True),
    ("stats", "effect_size_calculator"): _entry("stats", CAT_EXPLORATORY, False),
    # ── stats.py — distribution ────────────────────────────────────────
    ("stats", "distribution_fit"): _entry("stats", CAT_EXPLORATORY, True),
    ("stats", "mixture_model"): _entry("stats", CAT_EXPLORATORY, False),
    ("stats", "sprt"): _entry("stats", CAT_HYPOTHESIS, True),
    ("stats", "copula"): _entry("stats", CAT_MULTIVARIATE, False),
    # ── spc.py ─────────────────────────────────────────────────────────
    ("spc", "imr"): _entry("spc", CAT_SPC, False),
    ("spc", "xbar_r"): _entry("spc", CAT_SPC, False),
    ("spc", "xbar_s"): _entry("spc", CAT_SPC, False),
    ("spc", "p_chart"): _entry("spc", CAT_SPC, False),
    ("spc", "np_chart"): _entry("spc", CAT_SPC, False),
    ("spc", "c_chart"): _entry("spc", CAT_SPC, False),
    ("spc", "u_chart"): _entry("spc", CAT_SPC, False),
    ("spc", "cusum"): _entry("spc", CAT_SPC, False),
    ("spc", "ewma"): _entry("spc", CAT_SPC, False),
    ("spc", "laney_p"): _entry("spc", CAT_SPC, False),
    ("spc", "laney_u"): _entry("spc", CAT_SPC, False),
    ("spc", "moving_average"): _entry("spc", CAT_SPC, False),
    ("spc", "zone_chart"): _entry("spc", CAT_SPC, False, has_narrative=False),
    ("spc", "mewma"): _entry("spc", CAT_SPC, False),
    ("spc", "generalized_variance"): _entry("spc", CAT_SPC, False),
    ("spc", "capability"): _entry("spc", CAT_CAPABILITY, False, what_if_tier=1),
    ("spc", "nonnormal_capability"): _entry(
        "spc", CAT_CAPABILITY, False, has_narrative=False
    ),
    ("spc", "between_within"): _entry(
        "spc", CAT_CAPABILITY, False, has_narrative=False
    ),
    ("spc", "conformal_control"): _entry("spc", CAT_SPC, False),
    ("spc", "conformal_monitor"): _entry("spc", CAT_SPC, False),
    ("spc", "entropy_spc"): _entry("spc", CAT_SPC, False),
    ("spc", "degradation_capability"): _entry("spc", CAT_CAPABILITY, False),
    # ── ml.py ──────────────────────────────────────────────────────────
    ("ml", "classification"): _entry("ml", CAT_ML, False, has_narrative=False),
    ("ml", "regression_ml"): _entry("ml", CAT_ML, False, has_narrative=False),
    ("ml", "model_compare"): _entry("ml", CAT_ML, False, has_narrative=False),
    ("ml", "xgboost"): _entry("ml", CAT_ML, False, has_narrative=False),
    ("ml", "lightgbm"): _entry("ml", CAT_ML, False, has_narrative=False),
    ("ml", "shap_explain"): _entry("ml", CAT_ML, False, has_narrative=False),
    ("ml", "hyperparameter_tune"): _entry("ml", CAT_ML, False, has_narrative=False),
    ("ml", "clustering"): _entry("ml", CAT_ML, False, has_narrative=False),
    ("ml", "pca"): _entry("ml", CAT_ML, False, has_narrative=False),
    ("ml", "feature"): _entry("ml", CAT_ML, False, has_narrative=False),
    ("ml", "bayesian_regression"): _entry("ml", CAT_ML, False, has_narrative=False),
    ("ml", "gam"): _entry("ml", CAT_ML, False, has_narrative=False),
    ("ml", "isolation_forest"): _entry("ml", CAT_ML, False, has_narrative=False),
    ("ml", "gaussian_process"): _entry("ml", CAT_ML, False, has_narrative=False),
    ("ml", "pls"): _entry("ml", CAT_ML, False, has_narrative=False),
    ("ml", "sem"): _entry("ml", CAT_ML, False, has_narrative=False),
    ("ml", "regularized_regression"): _entry("ml", CAT_ML, False, has_narrative=False),
    ("ml", "discriminant_analysis"): _entry("ml", CAT_ML, False, has_narrative=False),
    ("ml", "factor_analysis"): _entry("ml", CAT_ML, False, has_narrative=False),
    ("ml", "correspondence_analysis"): _entry("ml", CAT_ML, False, has_narrative=False),
    ("ml", "item_analysis"): _entry("ml", CAT_ML, False, has_narrative=False),
    # ── bayesian.py ────────────────────────────────────────────────────
    ("bayesian", "bayes_regression"): _entry(
        "bayesian", CAT_BAYESIAN, False, has_education=True
    ),
    ("bayesian", "bayes_ttest"): _entry(
        "bayesian", CAT_BAYESIAN, False, has_education=True
    ),
    ("bayesian", "bayes_ab"): _entry(
        "bayesian", CAT_BAYESIAN, False, has_education=True
    ),
    ("bayesian", "bayes_correlation"): _entry(
        "bayesian", CAT_BAYESIAN, False, has_education=True
    ),
    ("bayesian", "bayes_anova"): _entry(
        "bayesian", CAT_BAYESIAN, False, has_education=True
    ),
    ("bayesian", "bayes_changepoint"): _entry(
        "bayesian", CAT_BAYESIAN, False, has_education=True
    ),
    ("bayesian", "bayes_proportion"): _entry(
        "bayesian", CAT_BAYESIAN, False, has_education=True
    ),
    ("bayesian", "bayes_capability_prediction"): _entry(
        "bayesian", CAT_BAYESIAN, False, has_education=True
    ),
    ("bayesian", "bayes_equivalence"): _entry(
        "bayesian", CAT_BAYESIAN, False, has_education=True
    ),
    ("bayesian", "bayes_chi2"): _entry(
        "bayesian", CAT_BAYESIAN, False, has_education=True
    ),
    ("bayesian", "bayes_poisson"): _entry(
        "bayesian", CAT_BAYESIAN, False, has_education=True
    ),
    ("bayesian", "bayes_logistic"): _entry(
        "bayesian", CAT_BAYESIAN, False, has_education=True
    ),
    ("bayesian", "bayes_survival"): _entry(
        "bayesian", CAT_BAYESIAN, False, has_education=True
    ),
    ("bayesian", "bayes_meta"): _entry(
        "bayesian", CAT_BAYESIAN, False, has_education=True
    ),
    ("bayesian", "bayes_demo"): _entry(
        "bayesian", CAT_BAYESIAN, False, has_education=True
    ),
    ("bayesian", "bayes_spares"): _entry(
        "bayesian", CAT_BAYESIAN, False, has_education=True
    ),
    ("bayesian", "bayes_system"): _entry(
        "bayesian", CAT_BAYESIAN, False, has_education=True
    ),
    ("bayesian", "bayes_warranty"): _entry(
        "bayesian", CAT_BAYESIAN, False, has_education=True
    ),
    ("bayesian", "bayes_repairable"): _entry(
        "bayesian", CAT_BAYESIAN, False, has_education=True
    ),
    ("bayesian", "bayes_rul"): _entry(
        "bayesian", CAT_BAYESIAN, False, has_education=True
    ),
    ("bayesian", "bayes_alt"): _entry(
        "bayesian", CAT_BAYESIAN, False, has_education=True
    ),
    ("bayesian", "bayes_comprisk"): _entry(
        "bayesian", CAT_BAYESIAN, False, has_education=True
    ),
    ("bayesian", "bayes_ewma"): _entry(
        "bayesian", CAT_BAYESIAN, False, has_education=True
    ),
    # ── reliability.py ─────────────────────────────────────────────────
    ("reliability", "weibull"): _entry("reliability", CAT_RELIABILITY, True),
    ("reliability", "lognormal"): _entry("reliability", CAT_RELIABILITY, True),
    ("reliability", "exponential"): _entry("reliability", CAT_RELIABILITY, True),
    ("reliability", "kaplan_meier"): _entry("reliability", CAT_RELIABILITY, True),
    ("reliability", "reliability_test_plan"): _entry(
        "reliability", CAT_RELIABILITY, False
    ),
    ("reliability", "distribution_id"): _entry("reliability", CAT_RELIABILITY, True),
    ("reliability", "accelerated_life"): _entry("reliability", CAT_RELIABILITY, True),
    ("reliability", "repairable_systems"): _entry("reliability", CAT_RELIABILITY, True),
    ("reliability", "warranty"): _entry("reliability", CAT_RELIABILITY, False),
    ("reliability", "competing_risks"): _entry("reliability", CAT_RELIABILITY, True),
    # ── viz.py ─────────────────────────────────────────────────────────
    ("viz", "histogram"): _entry("viz", CAT_VIZ, False, has_narrative=False),
    ("viz", "boxplot"): _entry("viz", CAT_VIZ, False, has_narrative=False),
    ("viz", "scatter"): _entry("viz", CAT_VIZ, False, has_narrative=False),
    ("viz", "heatmap"): _entry("viz", CAT_VIZ, False, has_narrative=False),
    ("viz", "pareto"): _entry("viz", CAT_VIZ, False, has_narrative=False),
    ("viz", "matrix"): _entry("viz", CAT_VIZ, False, has_narrative=False),
    ("viz", "timeseries"): _entry("viz", CAT_VIZ, False, has_narrative=False),
    ("viz", "probability"): _entry("viz", CAT_VIZ, False, has_narrative=False),
    ("viz", "individual_value_plot"): _entry(
        "viz", CAT_VIZ, False, has_narrative=False
    ),
    ("viz", "interval_plot"): _entry("viz", CAT_VIZ, False, has_narrative=False),
    ("viz", "dotplot"): _entry("viz", CAT_VIZ, False, has_narrative=False),
    ("viz", "bubble"): _entry("viz", CAT_VIZ, False, has_narrative=False),
    ("viz", "parallel_coordinates"): _entry("viz", CAT_VIZ, False, has_narrative=False),
    ("viz", "contour"): _entry("viz", CAT_VIZ, False, has_narrative=False),
    ("viz", "surface_3d"): _entry("viz", CAT_VIZ, False, has_narrative=False),
    ("viz", "contour_overlay"): _entry("viz", CAT_VIZ, False, has_narrative=False),
    ("viz", "mosaic"): _entry("viz", CAT_VIZ, False, has_narrative=False),
    ("viz", "bayes_spc_capability"): _entry(
        "viz", CAT_VIZ, False, has_narrative=False, has_education=True
    ),
    ("viz", "bayes_spc_changepoint"): _entry(
        "viz", CAT_VIZ, False, has_narrative=False, has_education=True
    ),
    ("viz", "bayes_spc_control"): _entry(
        "viz", CAT_VIZ, False, has_narrative=False, has_education=True
    ),
    ("viz", "bayes_spc_acceptance"): _entry(
        "viz", CAT_VIZ, False, has_narrative=False, has_education=True
    ),
    # ── simulation.py ──────────────────────────────────────────────────
    ("simulation", "tolerance_stackup"): _entry(
        "simulation", CAT_SIMULATION, False, what_if_tier=1
    ),
    ("simulation", "variance_propagation"): _entry(
        "simulation", CAT_SIMULATION, False, what_if_tier=1
    ),
    # ── d_type.py ──────────────────────────────────────────────────────
    ("d_type", "d_chart"): _entry("d_type", CAT_DIMENSIONAL, False, has_education=True),
    ("d_type", "d_cpk"): _entry("d_type", CAT_DIMENSIONAL, False, has_education=True),
    ("d_type", "d_nonnorm"): _entry(
        "d_type", CAT_DIMENSIONAL, False, has_education=True
    ),
    ("d_type", "d_equiv"): _entry("d_type", CAT_DIMENSIONAL, True, has_education=True),
    ("d_type", "d_sig"): _entry("d_type", CAT_DIMENSIONAL, True, has_education=True),
    ("d_type", "d_multi"): _entry("d_type", CAT_DIMENSIONAL, False, has_education=True),
    # ── causal_discovery.py ────────────────────────────────────────────
    ("causal", "causal_pc"): _entry("causal", CAT_CAUSAL, False, has_narrative=False),
    ("causal", "causal_lingam"): _entry(
        "causal", CAT_CAUSAL, False, has_narrative=False
    ),
    # ── drift_detection.py ─────────────────────────────────────────────
    ("drift", "drift_report"): _entry("drift", CAT_DRIFT, True, has_narrative=False),
    # ── anytime_valid.py ───────────────────────────────────────────────
    ("anytime", "anytime_ab"): _entry(
        "anytime", CAT_ANYTIME, True, has_narrative=False
    ),
    ("anytime", "anytime_onesample"): _entry(
        "anytime", CAT_ANYTIME, True, has_narrative=False
    ),
    # ── msa_bayes.py (single function, no analysis_id dispatch) ────────
    ("bayes_msa", "bayes_msa"): _entry(
        "bayes_msa", CAT_MSA, False, has_narrative=False
    ),
    # ── quality_economics.py ───────────────────────────────────────────
    ("quality_econ", "taguchi_loss"): _entry(
        "quality_econ", CAT_QUALITY_ECON, False, has_narrative=False
    ),
    ("quality_econ", "process_decision"): _entry(
        "quality_econ", CAT_QUALITY_ECON, False, has_narrative=False
    ),
    ("quality_econ", "lot_sentencing"): _entry(
        "quality_econ", CAT_QUALITY_ECON, False, has_narrative=False
    ),
    ("quality_econ", "cost_of_quality"): _entry(
        "quality_econ", CAT_QUALITY_ECON, False, has_narrative=False
    ),
    # ── pbs_engine.py ──────────────────────────────────────────────────
    ("pbs", "pbs_full"): _entry("pbs", CAT_PBS, False, has_education=True),
    ("pbs", "pbs_belief"): _entry("pbs", CAT_PBS, False, has_education=True),
    ("pbs", "pbs_edetector"): _entry("pbs", CAT_PBS, False, has_education=True),
    ("pbs", "pbs_evidence"): _entry("pbs", CAT_PBS, False, has_education=True),
    ("pbs", "pbs_predictive"): _entry("pbs", CAT_PBS, False, has_education=True),
    ("pbs", "pbs_adaptive"): _entry("pbs", CAT_PBS, False, has_education=True),
    ("pbs", "pbs_cpk"): _entry("pbs", CAT_PBS, False, has_education=True),
    ("pbs", "pbs_cpk_traj"): _entry("pbs", CAT_PBS, False, has_education=True),
    ("pbs", "pbs_health"): _entry("pbs", CAT_PBS, False, has_education=True),
    # ── interventional_shap.py (single function) ──────────────────────
    ("ishap", "ishap"): _entry("ishap", CAT_ISHAP, False, has_narrative=False),
    # ── siop.py — S&OP / inventory / supply chain ────────────────────
    ("siop", "abc_analysis"): _entry("siop", CAT_SIOP, False, what_if_tier=1),
    ("siop", "eoq"): _entry("siop", CAT_SIOP, False, what_if_tier=1),
    ("siop", "safety_stock"): _entry("siop", CAT_SIOP, False, what_if_tier=1),
    ("siop", "inventory_turns"): _entry("siop", CAT_SIOP, False),
    ("siop", "service_level"): _entry("siop", CAT_SIOP, False),
    ("siop", "demand_profile"): _entry("siop", CAT_SIOP, has_pvalue=True),
    ("siop", "kanban_sizing"): _entry("siop", CAT_SIOP, False),
    ("siop", "epei"): _entry("siop", CAT_SIOP, False),
    ("siop", "rop_simulation"): _entry("siop", CAT_SIOP, False),
    ("siop", "mrp_netting"): _entry("siop", CAT_SIOP, False),
    ("siop", "inventory_policy_wizard"): _entry("siop", CAT_SIOP, False),
}


# ── Convenience accessors ──────────────────────────────────────────────────


def get_entry(analysis_type, analysis_id):
    """Look up registry entry, returns None if unregistered."""
    return ANALYSIS_REGISTRY.get((analysis_type, analysis_id))


def get_all_by_module(module):
    """Return all (type, id) pairs for a module."""
    return [(t, a) for (t, a), e in ANALYSIS_REGISTRY.items() if e["module"] == module]


def get_all_with_pvalue():
    """Return all (type, id) pairs that produce p-values."""
    return [(t, a) for (t, a), e in ANALYSIS_REGISTRY.items() if e["has_pvalue"]]


def get_all_needing_shadow():
    """Return (type, id) pairs with shadow_type assigned but no current implementation."""
    return [
        (t, a)
        for (t, a), e in ANALYSIS_REGISTRY.items()
        if e["shadow_type"] and not e.get("has_shadow_impl")
    ]


def get_what_if_analyses(tier=None):
    """Return analyses with what-if interactivity. Filter by tier (1 or 2)."""
    if tier:
        return [
            (t, a)
            for (t, a), e in ANALYSIS_REGISTRY.items()
            if e["what_if_tier"] == tier
        ]
    return [(t, a) for (t, a), e in ANALYSIS_REGISTRY.items() if e["what_if_tier"] > 0]


def registry_stats():
    """Summary statistics for compliance reporting."""
    total = len(ANALYSIS_REGISTRY)
    return {
        "total": total,
        "with_pvalue": len(get_all_with_pvalue()),
        "with_education": sum(
            1 for e in ANALYSIS_REGISTRY.values() if e["has_education"]
        ),
        "with_narrative": sum(
            1 for e in ANALYSIS_REGISTRY.values() if e["has_narrative"]
        ),
        "with_charts": sum(1 for e in ANALYSIS_REGISTRY.values() if e["has_charts"]),
        "what_if_tier1": len(get_what_if_analyses(1)),
        "what_if_tier2": len(get_what_if_analyses(2)),
        "with_shadow_type": sum(
            1 for e in ANALYSIS_REGISTRY.values() if e["shadow_type"]
        ),
    }
