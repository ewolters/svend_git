"""Analysis engine — clean backend for the analysis workbench.

Canonical analysis package (formerly agents_api.dsw) with proper package structure.
Each sub-package owns its domain; dispatch.py routes requests.
"""

from .bayesian import run_bayesian_analysis
from .ml import run_ml_analysis
from .spc import run_spc_analysis
from .stats import run_statistical_analysis
from .viz import run_visualization

__all__ = [
    "run_statistical_analysis",
    "run_ml_analysis",
    "run_bayesian_analysis",
    "run_spc_analysis",
    "run_visualization",
]
