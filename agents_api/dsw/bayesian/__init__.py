"""DSW Bayesian Analysis — split into logical submodules.

Submodules:
    tests       — hypothesis tests (t-test, A/B, proportion, ANOVA, chi2, correlation, equivalence)
    regression  — linear, logistic, Poisson regression
    reliability — survival, repairable, RUL, ALT, competing risks, demo
    operations  — warranty, spares, system reliability
    special     — changepoint, capability prediction, meta-analysis, EWMA
"""

from scipy import stats

from .operations import (
    run_bayes_spares,
    run_bayes_system,
    run_bayes_warranty,
)
from .regression import (
    run_bayes_logistic,
    run_bayes_poisson,
    run_bayes_regression,
)
from .reliability import (
    run_bayes_alt,
    run_bayes_comprisk,
    run_bayes_demo,
    run_bayes_repairable,
    run_bayes_rul,
    run_bayes_survival,
)
from .special import (
    run_bayes_capability_prediction,
    run_bayes_changepoint,
    run_bayes_ewma,
    run_bayes_meta,
)
from .tests import (
    run_bayes_ab,
    run_bayes_anova,
    run_bayes_chi2,
    run_bayes_correlation,
    run_bayes_equivalence,
    run_bayes_proportion,
    run_bayes_ttest,
)

# Dispatch map: analysis_id -> function
_DISPATCH = {
    "bayes_regression": run_bayes_regression,
    "bayes_ttest": run_bayes_ttest,
    "bayes_ab": run_bayes_ab,
    "bayes_correlation": run_bayes_correlation,
    "bayes_anova": run_bayes_anova,
    "bayes_proportion": run_bayes_proportion,
    "bayes_changepoint": run_bayes_changepoint,
    "bayes_capability_prediction": run_bayes_capability_prediction,
    "bayes_equivalence": run_bayes_equivalence,
    "bayes_chi2": run_bayes_chi2,
    "bayes_poisson": run_bayes_poisson,
    "bayes_logistic": run_bayes_logistic,
    "bayes_survival": run_bayes_survival,
    "bayes_meta": run_bayes_meta,
    "bayes_demo": run_bayes_demo,
    "bayes_spares": run_bayes_spares,
    "bayes_system": run_bayes_system,
    "bayes_warranty": run_bayes_warranty,
    "bayes_repairable": run_bayes_repairable,
    "bayes_rul": run_bayes_rul,
    "bayes_alt": run_bayes_alt,
    "bayes_comprisk": run_bayes_comprisk,
    "bayes_ewma": run_bayes_ewma,
}


def run_bayesian_analysis(df, analysis_id, config):
    """Run Bayesian inference analyses — drop-in replacement for the monolith."""
    ci_level = float(config.get("ci", 0.95))
    z = stats.norm.ppf((1 + ci_level) / 2)

    handler = _DISPATCH.get(analysis_id)
    if handler:
        return handler(df, config, ci_level, z)

    return {
        "plots": [],
        "summary": f"Unknown Bayesian analysis: {analysis_id}",
        "guide_observation": "",
    }
