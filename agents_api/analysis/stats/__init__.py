"""Statistical analysis dispatcher.

Routes analysis_id to the appropriate sub-module.
Forge-backed handlers are tried first; legacy handlers are the fallback.
"""

import logging

from ..exploratory import run_exploratory
from ..forge_stats import run_forge_stats
from .advanced import _run_advanced
from .nonparametric import _run_nonparametric
from .parametric import _run_parametric
from .posthoc import _run_posthoc
from .quality import _run_quality
from .regression import _run_regression

logger = logging.getLogger(__name__)

_PARAMETRIC = {
    "anova",
    "anova2",
    "correlation",
    "equivalence",
    "f_test",
    "normality",
    "paired_t",
    "repeated_measures_anova",
    "sign_test",
    "split_plot_anova",
    "ttest",
    "ttest2",
}
_NONPARAMETRIC = {
    "friedman",
    "kruskal",
    "mann_whitney",
    "mood_median",
    "multi_vari",
    "runs_test",
    "spearman",
    "wilcoxon",
}
_REGRESSION = {
    "best_subsets",
    "glm",
    "logistic",
    "nominal_logistic",
    "nonlinear_regression",
    "ordinal_logistic",
    "orthogonal_regression",
    "poisson_regression",
    "regression",
    "robust_regression",
    "stepwise",
}
_POSTHOC = {
    "bonferroni_test",
    "dunn",
    "dunnett",
    "games_howell",
    "hsu_mcb",
    "interaction",
    "main_effects",
    "scheffe_test",
    "tukey_hsd",
}
_QUALITY = {
    "acceptance_sampling",
    "anom",
    "attribute_capability",
    "capability_sixpack",
    "multiple_plan_comparison",
    "nonnormal_capability_np",
    "variable_acceptance_sampling",
    "variance_components",
    "variance_test",
}
_ADVANCED = {
    "acf_pacf",
    "arima",
    "attribute_agreement",
    "attribute_gage",
    "bland_altman",
    "ccf",
    "changepoint",
    "cox_ph",
    "decomposition",
    "gage_linearity_bias",
    "gage_rr",
    "gage_rr_expanded",
    "gage_rr_nested",
    "gage_type1",
    "granger",
    "icc",
    "kaplan_meier",
    "krippendorff_alpha",
    "power_1prop",
    "power_1variance",
    "power_2prop",
    "power_2variance",
    "power_doe",
    "power_equivalence",
    "power_z",
    "sample_size_ci",
    "sample_size_tolerance",
    "sarima",
    "weibull",
}
_EXPLORATORY = {
    "auto_profile",
    "bootstrap_ci",
    "box_cox",
    "chi2",
    "copula",
    "data_profile",
    "descriptive",
    "distribution_fit",
    "duplicate_analysis",
    "effect_size_calculator",
    "fisher_exact",
    "graphical_summary",
    "grubbs_test",
    "hotelling_t2",
    "johnson_transform",
    "manova",
    "meta_analysis",
    "missing_data_analysis",
    "mixture_model",
    "nested_anova",
    "outlier_analysis",
    "poisson_1sample",
    "poisson_2sample",
    "prop_1sample",
    "prop_2sample",
    "run_chart",
    "sprt",
    "tolerance_interval",
}


def run_statistical_analysis(df, analysis_id, config):
    """Run statistical analysis — routes to sub-module by analysis_id.

    Forge-backed handlers (forgestat + ForgeViz) are tried first.
    If the analysis_id isn't ported yet, or the forge handler fails,
    falls back to the legacy inline implementation.
    """
    # Forge intercept DISABLED — forge handlers return ForgeViz specs,
    # but live workbench expects raw Plotly format. Re-enable only after
    # workbench renderer is updated or forge handlers output Plotly.
    # forge_result = run_forge_stats(analysis_id, df, config)
    # if forge_result is not None:
    #     return forge_result

    # Legacy fallback
    if analysis_id in _PARAMETRIC:
        return _run_parametric(analysis_id, df, config)
    elif analysis_id in _NONPARAMETRIC:
        return _run_nonparametric(analysis_id, df, config)
    elif analysis_id in _REGRESSION:
        return _run_regression(analysis_id, df, config)
    elif analysis_id in _POSTHOC:
        return _run_posthoc(analysis_id, df, config)
    elif analysis_id in _QUALITY:
        return _run_quality(analysis_id, df, config)
    elif analysis_id in _ADVANCED:
        return _run_advanced(analysis_id, df, config)
    elif analysis_id in _EXPLORATORY:
        return run_exploratory(analysis_id, df, config)
    else:
        logger.warning(f"Unknown analysis_id: {analysis_id}")
        return {
            "plots": [],
            "summary": f"Unknown analysis: {analysis_id}",
            "guide_observation": "",
        }
