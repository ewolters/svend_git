"""DSW Analysis modules.

This package organizes the statistical, ML, Bayesian, SPC, and visualization
analysis functions that were previously in dsw_views.py.

The split improves maintainability:
- stats.py: Statistical tests (t-tests, ANOVA, regression, etc.)
- ml.py: Machine learning (classification, clustering, PCA, etc.)
- bayesian.py: Bayesian inference (Bayes t-test, A/B, changepoint, etc.)
- spc.py: Statistical process control (capability, control charts, etc.)
- viz.py: Visualization (scatter, heatmap, pareto, etc.)

Usage:
    from agents_api.analysis import run_statistical_analysis, run_ml_analysis
    # or
    from agents_api.analysis.stats import run_statistical_analysis
"""

# Phase 2: Import from dsw/ sub-modules instead of monolith
from agents_api.dsw.bayesian import run_bayesian_analysis
from agents_api.dsw.ml import run_ml_analysis
from agents_api.dsw.spc import run_spc_analysis
from agents_api.dsw.stats import run_statistical_analysis
from agents_api.dsw.viz import run_visualization

__all__ = [
    "run_statistical_analysis",
    "run_ml_analysis",
    "run_bayesian_analysis",
    "run_spc_analysis",
    "run_visualization",
]
