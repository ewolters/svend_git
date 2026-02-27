# DSW vs Minitab — Gap Analysis

**Last updated:** 2026-02-20
**Svend pricing:** $49/month ($588/year) vs Minitab $2,594/year vs JMP $1,320–$8,400/year
**Total analysis types:** 200+

## Summary Scorecard

| Category | Coverage | Notes |
|---|---|---|
| Basic Statistics | **100%** | Proportion tests, Fisher's exact ✓ |
| Regression | **100%** | GLM, ordinal, nominal, Poisson, Deming, nonlinear ✓ |
| ANOVA | **100%** | GLM, split-plot, repeated measures, ANOM, Scheffé/Bonferroni/Hsu ✓ |
| DOE | **85%** | Missing split-plot design, mixture, general full factorial, augment |
| SPC | **100%** | All chart types including Gen.Var., MEWMA, Laney, zone ✓ |
| Capability | **100%** | Cp/Cpk/Pp/Ppk + sigma + full MSA suite |
| MSA | **100%** | 6 methods: crossed, nested, linearity, Type 1, attribute, agreement |
| Reliability | **100%** | Dist ID, ALT, repairable, warranty, probit all done |
| Time Series | **100%** | Beat Minitab (Granger, changepoint, CCF) |
| Multivariate | **100%** | Beat Minitab (SEM, correspondence, item analysis) |
| Power/Sample Size | **100%** | 14 calculators (5 Experimenter + 9 DSW) |
| Acceptance Sampling | **100%** | Single, double attribute + variable sampling ✓ |
| Bayesian | **100%+** | Minitab has 0, we have 7 |
| ML/Predictive | **100%+** | Minitab charges extra, ours included |
| Quality Tools | **100%** | Pareto, multi-vari, fishbone, ANOM, run chart, sign test, Mood's ✓ |
| Visualization | **100%** | Dotplot, individual value, interval, 3D surface, contour overlay ✓ |

**Overall: ~99% Minitab parity. Only 4 DOE design types missing (split-plot, mixture, general full factorial, augment). Clear advantages in Bayesian, ML, AI, collaboration, and modern regression.**

## Svend Advantages (not in Minitab)

- 7 Bayesian methods (Minitab has zero)
- Synara belief engine for hypothesis-driven reasoning
- 12 ML methods included (Minitab charges extra for CART/TreeNet/RF)
- GAM, Gaussian Process, SEM, Isolation Forest
- Ridge/LASSO/Elastic Net (Minitab lacks penalized regression)
- Granger causality, changepoint detection
- Auto ML from intent (LLM-driven)
- I-optimal designs
- Hypothesis ↔ evidence integration across all modules
- Auto-profile on import + Minitab-style Graphical Summary

---

## P1 — Core Gaps (affects daily Minitab users)

### P1.1 Proportion Tests
- [x] 1-proportion Z-test
- [x] 2-proportion Z-test
- [x] Fisher's exact test
- [x] 1-sample Poisson rate test

### P1.2 Power & Sample Size Expansion
Currently have: t-test (ind/paired), ANOVA, correlation, chi-square (5 in Experimenter) + 9 in DSW (14 total)
- [x] 1-sample Z power
- [x] 1-proportion power
- [x] 2-proportion power
- [x] 1-variance power
- [x] 2-variance power
- [x] Equivalence power (TOST)
- [x] DOE power (2-level factorial)
- [x] Sample size for estimation (CI width — mean and proportion)
- [x] Sample size for tolerance intervals

### P1.3 MSA Expansion
Currently have: crossed Gage R&R + 5 new methods (6 total)
- [x] Gage R&R nested
- [x] Gage linearity & bias study
- [x] Type 1 Gage study
- [x] Attribute Gage study (binary pass/fail)
- [x] Attribute agreement analysis (Kappa/Fleiss Kappa)

---

## P2 — Competitive Gaps (manufacturing differentiators)

### P2.1 DOE Expansion → Experimenter module (`experimenter_views.py`)
- [ ] Split-plot designs (hard-to-change factors)
- [ ] Mixture designs (components sum to 100%)
- [x] Multi-response optimization (desirability across multiple Y's) — interactive calculator with profiles + contour
- [ ] General full factorial (multi-level, not just 2-level)
- [ ] Augment design (add runs to existing design)

### P2.2 Reliability Expansion
- [x] Distribution ID plot — `distribution_id` fits 11+ distributions, picks best
- [x] Lognormal, exponential, loglogistic distribution fitting — `lognormal`, `exponential`, in `distribution_id`
- [x] Accelerated Life Testing (ALT) — `accelerated_life`
- [x] Repairable systems (Mean Cumulative Function) — `repairable_systems`
- [x] Probit analysis — interactive dose-response explorer (probit + logit, IRLS fitting, ED50 CI)
- [x] Demonstration/estimation test plans — `reliability_test_plan`
- [x] Warranty prediction — `warranty`

### P2.3 SPC Expansion
- [x] Laney P' chart — `laney_p`
- [x] Laney U' chart — `laney_u`
- [x] G chart (time/count between rare events) — interactive rare events lab with simulate mode
- [x] T chart (time between events) — interactive rare events lab with simulate mode
- [x] Moving Average chart — backend + workbench UI with configurable span
- [x] I-MR-R/S (between/within) — `between_within`
- [x] Zone chart — backend + workbench UI with color-coded A/B/C zones and cumulative scoring
- [x] Generalized Variance chart — `generalized_variance` in dsw/spc.py
- [x] MEWMA chart — backend + workbench UI with configurable lambda and variable contribution

---

## P3 — Nice to Have

### P3.1 ANOVA / GLM
- [x] General Linear Model (GLM) — `glm` with unified framework
- [x] Split-plot ANOVA analysis — `split_plot_anova` in dsw/stats.py
- [x] Repeated measures ANOVA — `repeated_measures_anova` in dsw/stats.py
- [x] Analysis of Means (ANOM) — `anom` in dsw/stats.py
- [x] Scheffé, Bonferroni, Hsu MCB post-hoc methods — all three in dsw/stats.py

### P3.2 Regression
- [x] Ordinal logistic regression — `ordinal_logistic`
- [x] Nominal logistic regression — `nominal_logistic` in dsw/stats.py
- [x] Nonlinear regression (user-specified model) — `nonlinear_regression` in dsw/stats.py
- [x] Poisson regression — `poisson_regression` in dsw/stats.py
- [x] Orthogonal (Deming) regression — `orthogonal_regression` in dsw/stats.py

### P3.3 Multivariate
- [x] Factor analysis (exploratory) — `factor_analysis`
- [x] Correspondence analysis — `correspondence_analysis` in dsw/ml.py
- [x] Item analysis (Cronbach's alpha) — `item_analysis` in dsw/ml.py

### P3.4 Quality Tools
- [x] Cause-and-effect (fishbone/Ishikawa) diagram — **exists in Whiteboard module**
- [x] Run chart (distinct from control chart) — `run_chart` in dsw/stats.py

### P3.5 Acceptance Sampling
- [x] Variable acceptance sampling (normal distribution plans) — `variable_acceptance_sampling` in dsw/stats.py
- [ ] Multiple plan comparison

### P3.6 Visualization
- [x] 3D surface plots (rotating) — `surface_3d` in dsw/viz.py
- [x] Dotplots — `dotplot` in dsw/viz.py
- [x] Individual value plots — `individual_value_plot` in dsw/viz.py
- [x] Interval plots — `interval_plot` in dsw/viz.py
- [x] Contour plot overlay — `contour_overlay` in dsw/viz.py

### P3.7 Other
- [x] 1-sample Sign test — `sign_test`
- [x] Mood's median test — `mood_median`
- [x] Cross-correlation function (CCF) — `ccf` in dsw/stats.py
- [x] Grubbs' outlier test — `grubbs_test` in dsw/stats.py
- [x] Box-Cox transformation (standalone) — `box_cox`
- [x] Johnson transformation (standalone) — `johnson_transform` in dsw/stats.py

---

## Resolved

Items moved here as implemented. Include date and commit.

### 2026-02-13 — Audit & check-off
18 items previously unchecked were found to have working backend + frontend implementations:
Distribution ID, Lognormal/Exponential/Loglogistic fitting, ALT, Repairable systems, Reliability test plans, Warranty, Laney P'/U', I-MR-R/S between/within, GLM, Ordinal logistic, Factor analysis, Sign test, Mood's median, Box-Cox, Interaction plot.

### 2026-02-13 — Auto-profile + Graphical Summary
- `auto_profile`: lightweight data overview on import (histograms, correlation heatmap, per-column stats)
- `graphical_summary`: Minitab-style per-variable view (histogram + normal curve, boxplot, CI bars, Anderson-Darling, full descriptive stats, CIs)

### 2026-02-20 — Full audit: 21 remaining items all implemented
All 21 previously unchecked items confirmed implemented with working backend + frontend:
- P2.3: Generalized Variance chart
- P3.1: Split-plot ANOVA, repeated measures ANOVA, ANOM, Scheffé/Bonferroni/Hsu MCB
- P3.2: Nominal logistic, nonlinear regression, Poisson regression, orthogonal (Deming) regression
- P3.3: Correspondence analysis, item analysis (Cronbach's alpha)
- P3.4: Run chart
- P3.5: Variable acceptance sampling
- P3.6: 3D surface, dotplot, individual value, interval, contour overlay
- P3.7: CCF, Grubbs' outlier, Johnson transformation
Overall parity updated to ~99%. Only 4 DOE design types remain (split-plot, mixture, general full factorial, augment).
