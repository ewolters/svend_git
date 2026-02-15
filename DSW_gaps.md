# DSW vs Minitab — Gap Analysis

**Last updated:** 2026-02-13
**Svend pricing:** $49/month ($588/year) vs Minitab $1,851/year vs JMP $1,320–$8,400/year

## Summary Scorecard

| Category | Coverage | Notes |
|---|---|---|
| Basic Statistics | **100%** | Proportion tests, Fisher's exact ✓ |
| Regression | **95%** | GLM, ordinal logistic, modern methods ✓ |
| ANOVA | **90%** | GLM ✓, missing split-plot, repeated measures |
| DOE | **85%** | Missing split-plot, mixture; multi-response done |
| SPC | **98%** | Laney P'/U', I-MR-R/S, MEWMA all done; missing Gen.Var. only |
| Capability | **90%** | Solid (Cp/Cpk/Pp/Ppk + sigma) |
| MSA | **90%** | 6 methods: crossed, nested, linearity, Type 1, attribute, agreement |
| Reliability | **95%** | Dist ID, ALT, repairable, warranty, probit all done |
| Time Series | **90%** | Beat Minitab (Granger, changepoint) |
| Multivariate | **95%** | Beat Minitab (SEM), factor analysis ✓ |
| Power/Sample Size | **85%** | 14 calculators (5 Experimenter + 9 DSW) |
| Acceptance Sampling | **80%** | Single/double attribute, missing variable plans |
| Bayesian | **100%+** | Minitab has 0, we have 7 |
| ML/Predictive | **100%+** | Minitab charges extra, ours included |
| Quality Tools | **90%** | Pareto, multi-vari, fishbone, sign test, Mood's ✓ |
| Visualization | **80%** | Plotly-based, missing dotplots, ind.value, interval |

**Overall: ~95% Minitab parity, with clear advantages in Bayesian, ML, and modern regression.**

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
- [ ] Generalized Variance chart
- [x] MEWMA chart — backend + workbench UI with configurable lambda and variable contribution

---

## P3 — Nice to Have

### P3.1 ANOVA / GLM
- [x] General Linear Model (GLM) — `glm` with unified framework
- [ ] Split-plot ANOVA analysis
- [ ] Repeated measures ANOVA
- [ ] Analysis of Means (ANOM)
- [ ] Scheffé, Bonferroni, Hsu MCB post-hoc methods

### P3.2 Regression
- [x] Ordinal logistic regression — `ordinal_logistic`
- [ ] Nominal logistic regression
- [ ] Nonlinear regression (user-specified model)
- [ ] Poisson regression
- [ ] Orthogonal (Deming) regression

### P3.3 Multivariate
- [x] Factor analysis (exploratory) — `factor_analysis`
- [ ] Correspondence analysis
- [ ] Item analysis (Cronbach's alpha)

### P3.4 Quality Tools
- [x] Cause-and-effect (fishbone/Ishikawa) diagram — **exists in Whiteboard module**
- [ ] Run chart (distinct from control chart)

### P3.5 Acceptance Sampling
- [ ] Variable acceptance sampling (normal distribution plans)
- [ ] Multiple plan comparison

### P3.6 Visualization
- [ ] 3D surface plots (rotating)
- [ ] Dotplots
- [ ] Individual value plots
- [ ] Interval plots
- [ ] Contour plot overlay

### P3.7 Other
- [x] 1-sample Sign test — `sign_test`
- [x] Mood's median test — `mood_median`
- [ ] Cross-correlation function (CCF)
- [ ] Grubbs' outlier test
- [x] Box-Cox transformation (standalone) — `box_cox`
- [ ] Johnson transformation (standalone)

---

## Resolved

Items moved here as implemented. Include date and commit.

### 2026-02-13 — Audit & check-off
18 items previously unchecked were found to have working backend + frontend implementations:
Distribution ID, Lognormal/Exponential/Loglogistic fitting, ALT, Repairable systems, Reliability test plans, Warranty, Laney P'/U', I-MR-R/S between/within, GLM, Ordinal logistic, Factor analysis, Sign test, Mood's median, Box-Cox, Interaction plot.

### 2026-02-13 — Auto-profile + Graphical Summary
- `auto_profile`: lightweight data overview on import (histograms, correlation heatmap, per-column stats)
- `graphical_summary`: Minitab-style per-variable view (histogram + normal curve, boxplot, CI bars, Anderson-Darling, full descriptive stats, CIs)
