"""Visualization engine — spec-driven chart rendering + custom handlers.

Standard charts are rendered from declarative specs (specs.py) by the
engine (engine.py). Complex charts (matrix, parallel_coordinates, mosaic)
and Bayesian SPC have their own handler functions.

CR: 3c0d0e53
"""

from .bayesian_spc import (
    _BAYES_SPC_DISPATCH,
    _cpk_from_params,
    _nig_posterior_update,
    _nig_sample,
)
from .custom import _CUSTOM_DISPATCH
from .engine import render_chart
from .hooks import HOOKS
from .specs import _CHART_SPECS


def run_visualization(df, analysis_id, config):
    """Create visualizations — routes to engine, custom handlers, or Bayesian SPC.

    DSW-001 §4.3: Single entry point through dispatch.
    """
    # Tier 3: Bayesian SPC handlers (computation-heavy)
    handler = _BAYES_SPC_DISPATCH.get(analysis_id)
    if handler:
        return handler(df, config)

    # Tier 3: Custom handlers (complex layout/traces)
    handler = _CUSTOM_DISPATCH.get(analysis_id)
    if handler:
        return handler(df, config)

    # Tier 1+2: Engine-rendered from spec
    spec = _CHART_SPECS.get(analysis_id)
    if spec:
        return render_chart(df, config, spec, hooks=HOOKS)

    return {"plots": [], "summary": f"Unknown visualization: {analysis_id}"}


# Re-export NIG helpers for backward compatibility (tests import these)
__all__ = [
    "run_visualization",
    "_nig_posterior_update",
    "_nig_sample",
    "_cpk_from_params",
]
