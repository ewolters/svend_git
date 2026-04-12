# Svend — Competitive Positioning Reference

**Last updated:** 2026-02-20

---

## Analysis Type Count

Svend offers **200+** analysis types across the Decision Science Workbench (DSW). Breakdown:

| Category | Count | Examples |
|---|---|---|
| Statistical Tests | 99 | t-tests, ANOVA, chi-square, proportion tests, nonparametric, GLM, regression (linear, logistic, ordinal, nominal, Poisson, Deming, nonlinear), sign test, Mood's median, Grubbs' outlier, Box-Cox, Johnson transform, CCF |
| Machine Learning | 21 | Random Forest, Gradient Boosting, SVM, Neural Net, k-NN, Decision Tree, Naive Bayes, Logistic, Ridge, LASSO, Elastic Net, GAM, Gaussian Process, Isolation Forest, DBSCAN, PCA, LDA, t-SNE, UMAP, K-Means, Hierarchical Clustering |
| Bayesian Methods | 7 | Bayesian t-test, Bayesian proportion, Bayesian regression, Bayesian ANOVA, Bayesian correlation, Bayesian A/B, Bayesian meta-analysis |
| Reliability | 10 | Distribution ID, ALT, repairable systems, probit, warranty prediction, reliability test plans, Weibull/lognormal/exponential fitting |
| SPC / Control Charts | 20+ | I-MR, X-bar R, X-bar S, P, NP, C, U, Laney P'/U', G, T, Zone, Moving Average, I-MR-R/S, MEWMA, CUSUM, EWMA, Generalized Variance, capability (Cp/Cpk/Pp/Ppk), Gage R&R (crossed/nested/linearity/Type 1/attribute/agreement) |
| Visualization | 21 | Histogram, boxplot, scatter, matrix plot, heatmap, Pareto, control chart, probability plot, contour, surface 3D, dotplot, individual value, interval, graphical summary, auto-profile, bubble, marginal, run chart, multi-vari, pie, treemap |
| DOE | 8 | 2-level factorial, fractional factorial, Plackett-Burman, Box-Behnken, CCD, D-optimal, I-optimal, response surface |
| Power / Sample Size | 14 | t-test, ANOVA, proportion, variance, equivalence, correlation, chi-square, DOE, estimation, tolerance intervals |
| Acceptance Sampling | 3 | Single attribute, double attribute, variable sampling |
| Quality Tools | 6 | Fishbone (via Whiteboard), ANOM, multi-vari, run chart, cause-and-effect, process map |
| Other | ~10 | Granger causality, changepoint detection, correspondence analysis, item analysis (Cronbach's alpha), factor analysis, SEM |

**Total: ~200+ distinct analysis types**

---

## Competitor Pricing Comparison

| Platform | Annual Cost | Notes |
|---|---|---|
| **Svend Professional** | **$588/yr** ($49/mo) | All features included |
| **Svend Team** | **$1,188/yr** ($99/mo) | Multi-tenant, collaboration |
| **Svend Enterprise** | **$3,588/yr** ($299/mo) | Hoshin Kanri, SSO, audit trails |
| Minitab Statistical | $2,594/yr | Stats only, no ML |
| Minitab + Predictive Analytics | $4,000+/yr | Adds CART, TreeNet, RF |
| Minitab + Workspace + PA | $5,000+/yr | Full stack |
| Companion by Minitab | $1,199/yr | Lightweight, limited |
| JMP Standard | $1,320/yr | Desktop only |
| JMP Pro | $8,400/yr | Adds advanced modeling |
| InfinityQS Enact | ~$4,000+/yr | SPC-only cloud |

**Svend Professional saves 77% vs Minitab Statistical ($588 vs $2,594).**

---

## Switching Triggers (Why Users Leave Competitors)

1. **Price shock** — Minitab license renewal at $2,594+/yr per seat; teams multiply fast
2. **Fragmentation fatigue** — Minitab split into 4-5 products (Statistical, Workspace, Connect, Engage, Model Ops); JMP licensing tiers
3. **Collaboration needs** — Desktop tools don't support real-time multi-user work
4. **AI expectations** — Users want AI-assisted analysis, not just manual test selection
5. **Modern UX** — Engineers under 35 expect browser-based, mobile-friendly tools
6. **Platform consolidation** — Separate tools for SPC, DOE, FMEA, A3, VSM is expensive
7. **Regulatory pressure** — 21 CFR Part 11 / ISO 13485 driving cloud adoption for audit trails

## Switching Barriers (Why Users Stay with Competitors)

1. **Regulatory validation** — Validated environments (pharma, medical device) resist tool changes
2. **Training alignment** — Six Sigma / ASQ curriculum references Minitab by name
3. **Institutional inertia** — "We've always used Minitab" — procurement path of least resistance
4. **Data migration** — Years of Minitab project files (.mpx) with no export path
5. **IT policy** — Some enterprises mandate desktop-only or approved vendor lists
6. **Feature muscle memory** — Power users know exact Minitab menu paths

---

## Svend Unique Differentiators

1. **AI-first platform** — LLM-driven analysis selection, interpretation, and code generation (Synara, Guide, Auto ML)
2. **Web-native** — Browser-based, zero install, cross-platform, real-time collaboration
3. **Bayesian methods** — 7 Bayesian analyses (Minitab and JMP have zero)
4. **Unified platform** — DSW + SPC + DOE + FMEA + A3 + VSM + Hoshin + RCA in one product
5. **Hypothesis-driven workflow** — Evidence links to hypotheses with Bayesian probability updates
6. **Knowledge graph** — Collaborative whiteboard with causal reasoning
7. **Modern ML included** — Random Forest, Gradient Boosting, GAM, GP, Isolation Forest — no upsell
8. **Free tools funnel** — 9 public calculators driving SEO traffic (Cpk, OEE, sample size, sigma, takt time, kanban, control chart, Gage R&R, Pareto)
9. **Penalized regression** — Ridge, LASSO, Elastic Net (Minitab lacks these entirely)
10. **Time series advantage** — Granger causality, changepoint detection beyond Minitab's offering

---

## Remaining Gaps (vs Minitab)

Only **4 DOE types** are truly missing:

1. **Split-plot designs** — Hard-to-change factor support in DOE (ANOVA split-plot analysis exists)
2. **Mixture designs** — Components summing to 100%
3. **General full factorial** — Multi-level factors (not just 2-level)
4. **Augment design** — Add runs to existing design

These are all in the Experimenter DOE module. Split-plot ANOVA (analysis side) is implemented; only the design generation is missing.

**Minitab parity: ~99%** (all P1, P2, and P3 items implemented except 4 DOE design types).

---

## SEO Keyword Targets

| Keyword | Monthly Volume (est.) | Current Page |
|---|---|---|
| minitab alternative | 500-1,000 | None (need /svend-vs-minitab/) |
| minitab free alternative | 200-500 | None |
| cpk calculator | 2,000+ | /tools/cpk-calculator/ |
| oee calculator | 3,000+ | /tools/oee-calculator/ |
| gage r&r calculator | 1,000+ | /tools/gage-rr-calculator/ |
| sample size calculator | 5,000+ | /tools/sample-size-calculator/ |
| control chart generator | 1,000+ | /tools/control-chart-generator/ |
| pareto chart maker | 500+ | /tools/pareto-chart-generator/ |
| statistical analysis software | 1,000+ | Landing page |
| six sigma software | 500+ | Landing page |
