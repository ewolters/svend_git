"""
DSW (Decision Science Workbench) — split modules.

This package contains the DSW analysis engine, split from the monolithic
dsw_views.py for maintainability. Each module handles one analysis category:

- common: Shared utilities (model cache, logging, ML helpers)
- dispatch: Main run_analysis() router
- stats: Statistical analysis (200+ tests)
- ml: Machine learning analysis
- bayesian: Bayesian inference
- reliability: Reliability & survival analysis
- spc: Statistical process control
- viz: Visualization + Bayesian SPC suite
- simulation: Monte Carlo simulation
- endpoints_ml: ML lab HTTP endpoints
- endpoints_data: Data/code/assistant HTTP endpoints
"""
