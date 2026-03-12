"""D-Type Process Intelligence dispatcher.

Routes analysis_id to the appropriate D-Type handler.

CR: 3c0d0e53
"""

from .chart import run_d_chart
from .cpk import run_d_cpk
from .equiv import run_d_equiv

# Re-export helpers that may be used externally
from .helpers import (  # noqa: F401
    _bernoulli_jsd,
    _build_grid,
    _compute_cpk,
    _cpk_noise_floor,
    _decompose_divergence,
    _jsd,
    _jsd_tail,
    _kde_density,
    _noise_floor,
    _p_within_spec,
)
from .multi import run_d_multi
from .nonnorm import run_d_nonnorm
from .sig import run_d_sig


def run_d_type(df, analysis_id, config):
    """Dispatcher for D-Type analyses."""
    if analysis_id == "d_chart":
        return run_d_chart(df, config)
    elif analysis_id == "d_cpk":
        return run_d_cpk(df, config)
    elif analysis_id == "d_nonnorm":
        return run_d_nonnorm(df, config)
    elif analysis_id == "d_equiv":
        return run_d_equiv(df, config)
    elif analysis_id == "d_sig":
        return run_d_sig(df, config)
    elif analysis_id == "d_multi":
        return run_d_multi(df, config)
    else:
        return {
            "plots": [],
            "summary": f"<<COLOR:danger>>Unknown D-Type analysis: {analysis_id}<</COLOR>>",
            "guide_observation": "",
        }
