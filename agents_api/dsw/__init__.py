"""
DSW (Decision Science Workbench) — split modules.

This package contains the DSW analysis engine, split from the monolithic
dsw_views.py for maintainability. Each module handles one analysis category:

- common: Shared utilities (model cache, logging, ML helpers)
- dispatch: Main run_analysis() router
- stats: Statistical analysis facade (routes to sub-modules)
- stats_parametric: t-tests, ANOVA, correlation, normality
- stats_nonparametric: Mann-Whitney, Kruskal-Wallis, Wilcoxon, Friedman
- stats_regression: OLS, logistic, GLM, stepwise, robust
- stats_posthoc: Tukey, Dunnett, Games-Howell, Dunn, Scheffé
- stats_quality: Capability, acceptance sampling, ANOM, variance
- stats_advanced: Power, MSA/Gage R&R, survival, time series
- stats_exploratory: Descriptive, profiling, distributions, multivariate
- ml: Machine learning analysis
- bayesian: Bayesian inference
- reliability: Reliability & survival analysis
- spc: Statistical process control
- viz: Visualization + Bayesian SPC suite
- simulation: Monte Carlo simulation
- endpoints_ml: ML lab HTTP endpoints
- endpoints_data: Data/code/assistant HTTP endpoints
"""
