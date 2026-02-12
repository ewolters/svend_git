# DSW vs Minitab — Gap Analysis

**Last updated:** 2026-02-07
**Svend pricing:** $29/month ($348/year) vs Minitab $1,851/year vs JMP $1,320–$8,400/year

## Summary Scorecard

| Category | Coverage | Notes |
|---|---|---|
| Basic Statistics | **100%** | Proportion tests, Fisher's exact ✓ |
| Regression | **90%** | Exceed on modern methods, missing GLM variants |
| ANOVA | **85%** | Missing GLM, split-plot, repeated measures |
| DOE | **80%** | Missing split-plot, mixture, multi-response |
| SPC | **85%** | Core complete, missing specialty charts |
| Capability | **90%** | Solid (Cp/Cpk/Pp/Ppk + sigma) |
| MSA | **90%** | 6 methods: crossed, nested, linearity, Type 1, attribute, agreement |
| Reliability | **60%** | Big 3 present, missing deep reliability |
| Time Series | **90%** | Beat Minitab (Granger, changepoint) |
| Multivariate | **90%** | Beat Minitab (SEM) |
| Power/Sample Size | **85%** | 14 calculators (5 Experimenter + 9 DSW) |
| Acceptance Sampling | **80%** | Single/double attribute, missing variable plans |
| Bayesian | **100%+** | Minitab has 0, we have 7 |
| ML/Predictive | **100%+** | Minitab charges extra, ours included |
| Quality Tools | **85%** | Pareto, multi-vari, fishbone (in Whiteboard) |
| Visualization | **80%** | Plotly-based, missing 3D surface, dotplots |

**Overall: ~89% Minitab parity, with clear advantages in Bayesian, ML, and modern regression.**

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
- [ ] Multi-response optimization (desirability across multiple Y's)
- [ ] General full factorial (multi-level, not just 2-level)
- [ ] Augment design (add runs to existing design)

### P2.2 Reliability Expansion
Currently have: Weibull, Kaplan-Meier, Cox PH
Missing:
- [ ] Distribution ID plot (fit 11+ distributions, pick best)
- [ ] Lognormal, exponential, loglogistic distribution fitting
- [ ] Accelerated Life Testing (ALT)
- [ ] Repairable systems (Mean Cumulative Function)
- [ ] Probit analysis
- [ ] Demonstration/estimation test plans
- [ ] Warranty prediction

### P2.3 SPC Expansion
Currently have: I-MR, X-bar R/S, P, NP, C, U, CUSUM, EWMA, T², capability
Missing:
- [ ] Laney P' chart (overdispersion-adjusted proportions)
- [ ] Laney U' chart (overdispersion-adjusted rates)
- [ ] G chart (time/count between rare events)
- [ ] T chart (time between events)
- [ ] Moving Average chart
- [ ] I-MR-R/S (between/within)
- [ ] Zone chart
- [ ] Generalized Variance chart
- [ ] MEWMA chart

---

## P3 — Nice to Have

### P3.1 ANOVA / GLM
- [ ] General Linear Model (GLM) — unified framework
- [ ] Split-plot ANOVA analysis
- [ ] Repeated measures ANOVA
- [ ] Analysis of Means (ANOM)
- [ ] Scheffé, Bonferroni, Hsu MCB post-hoc methods

### P3.2 Regression
- [ ] Ordinal logistic regression
- [ ] Nominal logistic regression
- [ ] Nonlinear regression (user-specified model)
- [ ] Poisson regression
- [ ] Orthogonal (Deming) regression

### P3.3 Multivariate
- [ ] Factor analysis (exploratory)
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
- [ ] 1-sample Sign test
- [ ] Mood's median test
- [ ] Cross-correlation function (CCF)
- [ ] Grubbs' outlier test
- [ ] Box-Cox / Johnson transformations (standalone)

---

## Resolved

Items moved here as implemented. Include date and commit.

*(none yet)*
