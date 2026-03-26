"""SPC analysis package."""

from .advanced import (
    run_cusum,
    run_ewma,
    run_laney_p,
    run_laney_u,
    run_moving_average,
    run_zone_chart,
)
from .capability import (
    run_between_within,
    run_capability,
    run_degradation_capability,
    run_nonnormal_capability,
)
from .conformal import (
    run_conformal_control,
    run_conformal_monitor,
    run_entropy_spc,
)
from .multivariate import (
    run_generalized_variance,
    run_mewma,
)
from .shewhart import (
    run_c_chart,
    run_imr,
    run_np_chart,
    run_p_chart,
    run_u_chart,
    run_xbar_r,
    run_xbar_s,
)

_DISPATCH = {
    "imr": run_imr,
    "xbar_r": run_xbar_r,
    "xbar_s": run_xbar_s,
    "p_chart": run_p_chart,
    "np_chart": run_np_chart,
    "c_chart": run_c_chart,
    "u_chart": run_u_chart,
    "cusum": run_cusum,
    "ewma": run_ewma,
    "laney_p": run_laney_p,
    "laney_u": run_laney_u,
    "moving_average": run_moving_average,
    "zone_chart": run_zone_chart,
    "capability": run_capability,
    "nonnormal_capability": run_nonnormal_capability,
    "degradation_capability": run_degradation_capability,
    "between_within": run_between_within,
    "mewma": run_mewma,
    "generalized_variance": run_generalized_variance,
    "conformal_control": run_conformal_control,
    "conformal_monitor": run_conformal_monitor,
    "entropy_spc": run_entropy_spc,
}


def run_spc_analysis(df, analysis_id, config):
    """Run SPC analysis."""
    # Bridge: Bayesian SPC suite
    if analysis_id.startswith("bayes_spc_"):
        from ..viz import run_visualization

        return run_visualization(df, analysis_id, config)

    # Bridge: Bayesian DOE suite
    if analysis_id.startswith("bayes_doe_"):
        from agents_api.bayes_doe import run_bayesian_doe

        return run_bayesian_doe(df, analysis_id, config)

    # G/T chart (rare events)
    if analysis_id in ("g_chart", "t_chart"):
        from .helpers import run_g_t_chart

        return run_g_t_chart(df, config)

    handler = _DISPATCH.get(analysis_id)
    if handler is not None:
        return handler(df, config)

    return {
        "plots": [],
        "summary": f"Unknown SPC analysis: {analysis_id}",
        "guide_observation": "",
    }
