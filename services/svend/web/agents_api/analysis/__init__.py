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

# For backwards compatibility, import main functions
# These will be migrated to individual modules over time
from agents_api.dsw_views import (
    run_statistical_analysis,
    run_ml_analysis,
    run_bayesian_analysis,
    run_spc_analysis,
    run_visualization,
)

__all__ = [
    'run_statistical_analysis',
    'run_ml_analysis',
    'run_bayesian_analysis',
    'run_spc_analysis',
    'run_visualization',
]
