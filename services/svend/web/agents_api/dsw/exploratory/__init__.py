"""DSW Exploratory analysis package — drop-in replacement for stats_exploratory._run_exploratory."""

from .comparative import (
    run_bootstrap_ci,
    run_chi2,
    run_fisher_exact,
    run_poisson_2sample,
    run_prop_2sample,
)
from .meta import (
    run_copula,
    run_effect_size_calculator,
    run_meta_analysis,
)
from .multivariate import (
    run_hotelling_t2,
    run_manova,
    run_nested_anova,
)
from .quality import (
    run_auto_profile,
    run_data_profile,
    run_duplicate_analysis,
    run_graphical_summary,
    run_missing_data_analysis,
    run_outlier_analysis,
)
from .sequential import (
    run_run_chart,
    run_sprt,
    run_tolerance_interval,
)
from .univariate import (
    run_box_cox,
    run_descriptive,
    run_distribution_fit,
    run_grubbs_test,
    run_johnson_transform,
    run_mixture_model,
    run_poisson_1sample,
    run_prop_1sample,
)

# Dispatch map: analysis_id -> function(df, config)
_DISPATCH = {
    "descriptive": run_descriptive,
    "chi2": run_chi2,
    "prop_1sample": run_prop_1sample,
    "prop_2sample": run_prop_2sample,
    "fisher_exact": run_fisher_exact,
    "poisson_1sample": run_poisson_1sample,
    "poisson_2sample": run_poisson_2sample,
    "bootstrap_ci": run_bootstrap_ci,
    "box_cox": run_box_cox,
    "run_chart": run_run_chart,
    "grubbs_test": run_grubbs_test,
    "johnson_transform": run_johnson_transform,
    "tolerance_interval": run_tolerance_interval,
    "hotelling_t2": run_hotelling_t2,
    "manova": run_manova,
    "nested_anova": run_nested_anova,
    "data_profile": run_data_profile,
    "auto_profile": run_auto_profile,
    "graphical_summary": run_graphical_summary,
    "missing_data_analysis": run_missing_data_analysis,
    "outlier_analysis": run_outlier_analysis,
    "duplicate_analysis": run_duplicate_analysis,
    "meta_analysis": run_meta_analysis,
    "effect_size_calculator": run_effect_size_calculator,
    "distribution_fit": run_distribution_fit,
    "mixture_model": run_mixture_model,
    "sprt": run_sprt,
    "copula": run_copula,
}


def _run_exploratory(analysis_id, df, config):
    """Run exploratory analysis — drop-in replacement dispatcher."""
    handler = _DISPATCH.get(analysis_id)
    if handler is None:
        return None
    return handler(df, config)
