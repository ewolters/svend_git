# Change Log

All edits to the kjerne codebase are logged here. Each entry records what changed, why, and how to verify.

## Format

```
### YYYY-MM-DD — Summary
**Debt item:** DEBT.md reference (if applicable)
**Files changed:**
- `path/to/file` — what changed
**Verification:** how to confirm it worked
**Commit:** git hash
```

---

### 2026-02-15 — Concept Drift Detection (3-lane diagnostic)

**What:** Three-lane drift diagnostic suite for ML model health monitoring. Detects what drifted, how, and whether it matters to performance.

**Architecture:**
- `drift_detection.py` — new module with three detectors:
  - **PSI** (Population Stability Index): quantile-based binning from reference, epsilon smoothing, per-feature table. Thresholds: negligible/<0.10/low/<0.20/moderate/<0.25/high
  - **ADWIN** (Adaptive Windowing): normalizes stream to [0,1], scans all possible cut points, Hoeffding bound ε=√(1/(2m)·ln(4n/δ)). Reports change index, mean before/after, shift magnitude
  - **Page-Hinkley**: cumulative deviation from running mean on standardized stream, two-sided (up+down), reports direction, cumulative stat, threshold

**Three lanes:**
- Lane A (Data Drift): PSI per feature + ADWIN on top 3 drifting features
- Lane B (Prediction Drift): PSI on predicted scores + ADWIN on prediction stream mean
- Lane C (Error Drift): ADWIN on squared-error loss + Page-Hinkley (up/down) on loss. Only available when both target and prediction columns are specified

**Frontend:** "Health" group in ML ribbon with Drift button. Config dialog: feature selection, prediction col, target col, reference/current split %. Three-lane explanation shown before running.

**Files changed:**
- `services/svend/web/agents_api/drift_detection.py` — NEW, ~420 lines
- `services/svend/web/agents_api/dsw_views.py` — added `elif analysis_type == "drift"` route
- `services/svend/web/templates/workbench_new.html` — Health ribbon group + openDriftDialog()

**Verification:** Upload time-ordered CSV → ML tab → Health → Drift → select features, set split % → Run. Should produce PSI bar chart, distribution comparison, rolling loss plot with annotated change points.

---

### 2026-02-15 — Causal Discovery (PC + LiNGAM)

**What:** New causal discovery module — discover causal structure from observational data. Two algorithms: PC (constraint-based via conditional independence) and LiNGAM (ICA-based, non-Gaussian). Outputs: directed DAG (Plotly), edge stability (bootstrap), separating-set explanations, assumptions panel.

**Architecture:**
- `causal_discovery.py` — clean wrapper around `causal-learn` (CMU). Functions: `run_causal_discovery()`, `_run_pc_core()`, `_run_lingam_analysis()`, `_bootstrap_pc()`, `_bootstrap_lingam()`
- Partial correlation CI test via regression residuals (not single-variable formula), Fisher z-transform with n > |S| + 3 guard
- Bootstrap stability: edge frequency + coefficient CIs (LiNGAM), configurable 10-500 resamples
- Separating-set explanations: "Removed X–Y because X ⊥ Y | {Z1, Z4} (p=0.21, ρ=0.03)"
- Gaussianity warnings (LiNGAM): Shapiro-Wilk test per variable, flags those that may violate non-Gaussian assumption
- Routed via `analysis_type='causal'` in dsw_views.py dispatcher

**Frontend:** Two buttons in ML tab ribbon "Causal" group (PC DAG, LiNGAM). Config dialogs expose α, max conditioning set, bootstrap count. Assumptions panel shown in dialog before running. Dispatches via `runStatsAnalysis('causal', 'causal_pc'|'causal_lingam', config)`.

**Files changed:**
- `services/svend/web/agents_api/causal_discovery.py` — NEW, ~460 lines
- `services/svend/web/agents_api/dsw_views.py` — added `elif analysis_type == "causal"` route
- `services/svend/web/templates/workbench_new.html` — Causal ribbon group + openCausalDialog()

**Verification:** Upload CSV with correlated numeric columns → ML tab → Causal group → PC (DAG) or LiNGAM → config → Run. Should produce DAG plot, stability bar chart, and detailed summary with edge explanations.

---

### 2026-02-15 — Conformal Prediction Intervals on ML Hub

**What:** Every supervised ML model (classification, regression_ml, xgboost, lightgbm, model_compare) now wraps predictions in split conformal prediction intervals/sets with finite-sample marginal coverage guarantees under exchangeability — no distributional assumptions.

**Architecture:**
- `conformal.py` — `_conformal_qhat()` (order statistic, NOT np.quantile interpolation), `ConformalRegressor` (absolute residual scores), `ConformalClassifier` (softmax nonconformity scores), `compute_conformal()` wrapper
- Data split changed from 80/20 train/test → 70/15/15 train/calibration/test. Added `_stratified_split_3way()` helper
- Calibration predictions computed on X_cal only (no leakage). Metrics still on test set only
- Conformal state persisted as compact `{type, method, n_cal, qhats}` — not full scores

**Regression output:** Interval half-width (±qhat), empirical test coverage, conformal interval scatter plot (green=inside, red=outside)
**Classification output:** Average prediction set size, single-class %, empirical coverage, prediction set size histogram
**Inference (`run_model()`):** Conformal intervals/sets from saved state. Tree-quantile fallback for pre-conformal models

**Files changed:**
- `agents_api/conformal.py` — NEW: split conformal prediction module
- `agents_api/dsw_views.py` — 3-way split, conformal hooks in 5 ML blocks, `run_model()` updated
**Verification:** Train any supervised model → summary shows "Conformal Prediction Intervals/Sets (90% nominal)" with empirical test coverage. Save model → run inference → conformal intervals in response.

---

### 2026-02-15 — Bayesian DOE Suite (5 tools)

**What:** Added Bayesian DOE suite to Bayesian Sigma — 5 tools for designed experiments using conjugate Normal-Inverse-Gamma linear model. All closed-form (no MCMC), O(p³). No competitor offers Bayesian DOE as click-and-run.

**Architecture:**
- `bayes_core.py` — 4 shared posterior routines (precision form, Cholesky decomposition): `bayesian_linear_posterior()`, `contrast_posterior()`, `predictive_posterior()`, `marginal_log_likelihood()`
- `bayes_doe.py` — Design matrix builder with stored coding metadata + 5 analysis handlers
- Separated from SPC: DOE operates on multivariate linear models, SPC on scalar process statistics

**Tools:**
1. **Effect Screening** — P(practical significance) per factor via Student-t contrast posteriors. Verdicts: ACTIVE/POSSIBLY ACTIVE/INERT. 3 plots.
2. **Model Selection** — Compare main effects / +interactions / +quadratic via exact marginal likelihood (not AIC/BIC). Log-softmax for model probabilities.
3. **Sample Size** — Pre-posterior simulation (no data needed). Finds minimum n for 90% detection probability.
4. **Response Optimization** — Grid search with full posterior predictive uncertainty propagation. Not frequentist point estimates.
5. **Next Experiment** — Sequential DOE via expected predictive variance reduction. Rank-1 precision updates.

**Files changed:**
- `agents_api/bayes_core.py` — NEW: shared conjugate posterior math (Zellner g-prior, Cholesky, precision form)
- `agents_api/bayes_doe.py` — NEW: design matrix builder + 5 DOE tool handlers
- `agents_api/dsw_views.py` — Added `bayes_doe_*` dispatch bridge to `run_bayesian_doe()`
- `agents_api/spc_views.py` — Added DOE bridge for uploaded file path (multi-column extraction)
- `templates/dsw.html` — Added 5 DOE sub-tabs to Bayesian Sigma ribbon (with CSS divider), `runBayesDOE()` JS function, dynamic table inputs

**Verification:** Django check: 0 issues. Smoke tests pass: effect screening correctly identifies A=2.0 as ACTIVE, B=0 as INERT. Model selection prefers main effects (97.2%). Sample size recommends 8 runs for large effect (σ=1, effect=2). Optimizer finds correct corner.

---

### 2026-02-15 — Add Bayesian Sigma + Conformal SPC to workbench ribbon, cleanup duplicates

**What:** Added 6 new analysis buttons to Quality ribbon in workbench_new.html (the live DSW template): 4 Bayesian Sigma (Cpk, Change Point, Control, Acceptance) and 2 Conformal SPC (Control Chart, P-Value Monitor). Each opens a config dialog and dispatches through `runStatsAnalysis('spc', ...)`. Deleted orphan `dsw.html` (not served by any URL). Removed 529 lines of duplicate Bayesian SPC blocks from `run_spc_analysis()`.

**Files changed:**
- `templates/workbench_new.html` — 2 new ribbon groups in Quality tab: Bayesian Sigma (4 buttons) and Conformal SPC (2 buttons); `openBayesSigmaDialog()` and `openConformalDialog()` JS functions with full config dialogs
- `agents_api/dsw_views.py` — Removed duplicate bayes_changepoint/control/acceptance from run_spc_analysis(); kept bridge routing for bayes_spc_*
- `templates/dsw.html` — **Deleted** (orphan, not served anywhere)

**Verification:** `/app/dsw/` → Quality tab → scroll right to see Bayesian Sigma and Conformal SPC groups. Import data, click button, config dialog opens.

---

### 2026-02-15 — Bayesian SPC Blocks + Conformal Prediction Control Charts

**What:** Added 3 Bayesian SPC analysis blocks and 2 conformal prediction control chart blocks to `run_spc_analysis()` in dsw_views.py.

Bayesian SPC (Bayesian Sigma suite):
- `bayes_changepoint` — Adams & MacKay (2007) BOCPD with NIG conjugate, run-length heatmap, segment means
- `bayes_control` — Two-state HMM forward filter with sequential NIG posterior, credible interval ribbon
- `bayes_acceptance` — Beta-Binomial conjugate with sequential decision boundaries, posterior PDF

Conformal Prediction SPC (Burger et al., Dec 2025, arXiv:2512.23602):
- `conformal_control` — Distribution-free control chart with conformal prediction intervals. Phase I/II split, nonconformity scores, adaptive prediction intervals, uncertainty spike detection, Shewhart comparison overlay. Supports individuals, subgroup mean, subgroup range.
- `conformal_monitor` — Multivariate conformal p-value chart. Isolation Forest or Mahalanobis anomaly scoring, conformal p-values with guaranteed false alarm rate, variable contribution heatmap.

**Files changed:**
- `agents_api/dsw_views.py` — 5 new `elif analysis_id ==` blocks in `run_spc_analysis()`

**Verification:** `python3 -m py_compile dsw_views.py` passes. Each block accepts standard SPC dispatch config.

---

### 2026-02-15 — Bayesian SPC Landing Section + Free Bayesian Cpk Calculator

**What:** Added Bayesian SPC section to the landing page for SEO and product positioning. Created a free Bayesian Cpk Calculator tool — Monte Carlo posterior sampling, side-by-side traditional vs Bayesian comparison, probability bars for P(Cpk > threshold), posterior histogram with credible intervals. Full structured data and FAQ for SEO.

**Files changed:**
- `templates/landing.html` — New Bayesian SPC section with 4 cards (Cpk, Control Chart, Change Point, Acceptance Sampling), grouped "Also in the suite" chips (Inference/Modeling/Evidence), SEO keywords, FAQ, structured data
- `templates/tools/bayesian_cpk_calculator.html` — New free tool: Bayesian Cpk calculator with Normal-Inverse-Gamma conjugate model, Monte Carlo sampling (10K draws), probability bars, posterior histogram, insight interpretation, comprehensive SEO
- `templates/tools/index.html` — Added Bayesian Cpk Calculator card + structured data entry + updated meta keywords
- `templates/tool_base.html` — Added Bayesian Cpk Calculator to footer links
- `svend/urls.py` — Added route `/tools/bayesian-cpk-calculator/` + sitemap entry

**Verification:** Visit `/tools/bayesian-cpk-calculator/` — calculator loads with default data, shows traditional vs Bayesian comparison, probability bars, and posterior histogram.

---

### 2026-02-15 — 5S Cleanup: Remove Reasoning-Era Artifacts

**What:** Removed all stale references to Svend's old identity as a "reasoning system" / "tool-augmented reasoning system." Svend is a decision science platform. Deleted old mockups, stale docs, and updated descriptions in active config files.

**Files deleted:**
- `svend-sea-india-distribution-plan.html` — old distribution plan with "reasoning system" branding
- `svend-whitepaper-insight-spine.html` — old whitepaper draft with "reasoning system" branding
- `svend-seo-strategy.html` — old SEO strategy document
- `services/svend/site/mockups/` — 25 HTML files of old reasoning chat mockups (math, chemistry, physics, logic)
- `services/svend/reference_docs/ROADMAP.md` — stale roadmap describing training a 374M reasoning model
- `services/svend/agents/agents/site/` — 9 old pre-Django agent site templates
- `services/svend/data/` — empty directory from prior cleanup

**Files updated:**
- `services/svend/web/pyproject.toml` — "Tool-augmented reasoning system" → "Hypothesis-driven decision science platform"
- `services/svend/web/svend/__init__.py` — same
- `services/svend/requirements.txt` — "Tool-Augmented Reasoning System" → "Decision Science Platform"
- `services/svend/web/.env.example` — "Synara reasoning engine" → "Synara belief engine"
- `services/svend/site/STYLE_GUIDE.md` — updated brand identity, tool indicators, file structure
- `services/svend/reference_docs/ARCHITECTURE.md` — "Reasoning Engine" → "Belief Engine (Synara)"
- `services/svend/reference_docs/SYNARA_WHITEPAPER.md` — acknowledgments updated
- `services/svend/web/agents_api/experimenter_views.py` — "reasoning model" → "language model"

**Left alone (legitimate uses):**
- `core/synara.py` — "Bayesian reasoning engine" describes what Synara actually does
- `log.md` — historical entries preserved
- `agents/agents/docs/*.py` — clarifying comments ("not reasoning engines")
- `core/llm.py` — technical description of r1-1.5b model category

**Verification:** `grep -ri "reasoning system\|reasoning model\|tool-augmented reasoning" --include="*.py" --include="*.toml" --include="*.md" services/svend/ | grep -v log.md | grep -v SYNARA_WHITEPAPER`

---

### 2026-02-13 — New Statistics: Confidence Intervals for 14 Core DSW Analyses

**What:** Added confidence intervals to the 14 most-used statistical analyses in the DSW, embracing New Statistics (estimation over naked p-values). Every core analysis now reports CIs alongside effect sizes and p-values.

**Files changed:**
- `agents_api/dsw_views.py` — 14 analysis blocks updated:
  - `ttest2`: 95% CI for mean difference (pooled SE)
  - `paired_t`: 95% CI for mean difference (paired SE)
  - `anova`: 95% CI for each group mean (t-based)
  - `regression`: 95% CI for every coefficient (t-critical × SE)
  - `correlation`: 95% CI for r via Fisher z-transform
  - `chi2`: Odds ratio + 95% CI for 2×2 tables (Woolf log-OR method)
  - `mann_whitney`: Hodges-Lehmann median difference + 95% CI
  - `kruskal`: 95% CI for each group median (order statistic method)
  - `logistic`: 95% CI for odds ratios (Fisher information matrix)
  - `f_test`: 95% CI for variance ratio + log variance ratio effect size
  - `equivalence`: Explicit 90% CI (TOST standard) + 95% CI for difference
  - `ordinal_logistic`: 95% CI for odds ratios (Wald SE)
  - `glm` (fixed + mixed): CIs for all coefficients (model.conf_int / Wald)
  - `nominal_logistic`: Approximate 95% CI for odds ratios (Fisher info per class)

**Verification:** Run any of the 14 analyses — output now includes CI lines.

---

### 2026-02-15 — Learn Module Phase 1: ML + Advanced Statistics Expansion

**What:** Expanded the Learn module from 10 modules / 47 sections to 12 modules / 58 sections. Added Machine Learning (7 sections) and Advanced Statistics (6 sections) with full interactive widgets, hybrid datasets, and Forge integration.

**Architecture change:** Refactored monolithic `learn_content.py` (6,511 lines) into a `learn_content/` package with 14 topic files. Backward-compatible shim preserves existing imports.

**Files changed:**
- `agents_api/learn_content.py` — converted to thin re-export shim (imports from package)
- `agents_api/learn_content/__init__.py` — new package entry point
- `agents_api/learn_content/_registry.py` — central SECTION_CONTENT dict (58 sections), get_section_content(), get_all_topics()
- `agents_api/learn_content/_datasets.py` — SHARED_DATASET (original) + CHURN_DATASET (200 telecom records, 27% churn) + CLINICAL_DATASET (200 patients, 5 sites, 15% dropout)
- `agents_api/learn_content/machine_learning.py` — 7 sections: supervised classification/regression, unsupervised, model validation, feature engineering, ensemble methods, interpretability
- `agents_api/learn_content/advanced_statistics.py` — 6 sections: multivariate analysis, categorical data, Bayesian depth, mixed models, response surface, regression diagnostics
- `agents_api/learn_content/foundations.py` — extracted (5 sections, unchanged content)
- `agents_api/learn_content/experimental_design.py` — extracted (4 sections)
- `agents_api/learn_content/data_fundamentals.py` — extracted (4 sections)
- `agents_api/learn_content/statistical_inference.py` — extracted (6 sections)
- `agents_api/learn_content/causal_inference.py` — extracted (4 sections)
- `agents_api/learn_content/critical_evaluation.py` — extracted (4 sections)
- `agents_api/learn_content/dsw_mastery.py` — extracted (7 sections)
- `agents_api/learn_content/case_studies.py` — extracted (4 sections)
- `agents_api/learn_content/advanced_methods.py` — extracted (5 sections)
- `agents_api/learn_content/capstone.py` — extracted (2 sections)
- `agents_api/learn_views.py` — added Machine Learning (module 11) and Advanced Statistics (module 12) to COURSE_MODULES
- `templates/learn.html` — added 13 new interactive widget render functions (classifier_playground, regression_playground, validation_visualizer, feature_transformer, ensemble_builder, shap_explorer, pca_explorer, contingency_explorer, posterior_visualizer, random_effects_demo, rsm_contour_explorer, diagnostic_dashboard, clustering_explorer update), "Practice with Fresh Data" Forge button, getForgeTemplate() mapping, generateFreshData() async function
- `forge/migrations/0005_learning_templates.py` — 6 new Forge schema templates (customer_churn, clinical_trial, manufacturing_quality, assembly_line_production, supply_chain_orders, product_launches)

**Verification:**
- `python3 -c "import sys; sys.path.insert(0,'.'); from agents_api.learn_content import SECTION_CONTENT; print(len(SECTION_CONTENT))"` → 58
- Navigate to Learn page → verify 12 modules render → click ML/Stats sections → verify content + widgets load
- Click "Practice with Fresh Data" → verify Forge generates and reloads widget data

---

### 2026-02-15 — ML Hub: Class Imbalance Handling & Reliability Governance

**What:** A 97/3 imbalanced dataset produced 1.000 accuracy and 1.000 F1 — the model was either predicting all-majority or the test set had no minority samples. Fixed the pipeline to produce honest, decision-grade metrics.

**Root causes fixed:**
1. `train_test_split` had no `stratify=y` — minority class could vanish from test set
2. `RandomForestClassifier` used default `class_weight=None` — no rebalancing
3. `f1_score(average="weighted")` masks the problem (≈0.97 even predicting all-majority)
4. Warnings existed post-hoc but didn't change metrics or model behavior

**Files changed:**
- `agents_api/dsw_views.py`:
  - Added `_stratified_split()` — StratifiedShuffleSplit with up to 10 retries to ensure all classes appear in test set, falls back to plain stratified, then unstratified
  - Added `_classification_reliability()` — shared function computing balanced_accuracy, f1_macro, recall_macro, per-class breakdown, baseline_accuracy, and reliability_warnings list
  - Updated `_auto_train()` — uses stratified split, auto-enables `class_weight="balanced"` when majority > 75%, adds average_precision (PR AUC) for binary
  - Updated XGBoost section — stratified split, `scale_pos_weight` for binary / `compute_sample_weight("balanced")` for multiclass when imbalanced
  - Updated LightGBM section — stratified split, `is_unbalance=True` when imbalanced
  - All three paths now call `_classification_reliability()` for enriched metrics

- `agents_api/autopilot_views.py`:
  - Enhanced `_build_training_interpretation()` — prepends "Reliability: LOW" when HIGH warnings exist, shows balanced_accuracy and f1_macro context, per-class minority recall with failure flags

- `templates/models.html`:
  - Added `result-warnings` div — red-bordered banner for HIGH warnings, orange for medium
  - Updated metrics grid — filters out objects, adds Baseline/Balanced Acc/Lift cards, colors accuracy orange when HIGH warnings exist
  - Added `result-perclass` — collapsible per-class metrics table (precision/recall/F1/support), rows with recall < 0.5 highlighted orange
  - Model card color override — shows primary metric in orange instead of green when HIGH reliability warnings stored in model artifact

**New metrics added to classification results:**
- `balanced_accuracy` — mean per-class recall (honest metric for imbalanced data)
- `f1_macro` — punishes minority neglect
- `recall_macro` — mean per-class recall
- `average_precision` — PR AUC (binary only, most honest for 97/3 land)
- `baseline_accuracy` — majority class proportion
- `per_class` — per-class precision/recall/f1/support dict
- `reliability_warnings` — list of {level, msg} dicts persisted with model artifact

**Reliability warning rules:**
- Test split missing classes (HIGH)
- Perfect/near-perfect accuracy ≥ 0.99 (HIGH)
- Accuracy matches baseline within 1-2% (HIGH)
- Severe class imbalance > 80% (MEDIUM)
- High accuracy but low balanced accuracy gap > 25% (HIGH)
- Minority class recall < 50% (HIGH)

**Verification:** Upload 97/3 imbalanced dataset → Quick Train should show: class_weight="balanced" in recipe, balanced accuracy < standard accuracy, red reliability warnings banner, per-class table, baseline comparison. Balanced dataset should show no warnings.

---

### 2026-02-15 — Analysis Workbench UX Round 2: Icons, Grid, Next Steps, Smart Select

**What:** Continued UX polish pass. Uniform analysis grid items, category icons, next-steps suggestions, smart column pre-selection, toolbar expansion.

**Changes:**
1. **Uniform grid items:** Fixed height (48px), consistent `minmax(150px, 1fr)` columns, text overflow with ellipsis + title tooltips
2. **Category icons:** Each analysis item gets a small 14x14 SVG icon matching its category (stats=bar chart, SPC=pulse, viz=chart, ML=sun, Bayesian=layers, etc.)
3. **Next steps suggestions:** After ~15 common analyses, shows clickable "Next:" links for logical follow-up analyses (e.g., ANOVA → Tukey HSD / Main Effects / Interval Plot). Includes "Re-run" link.
4. **Smart column pre-selection:** When no saved config exists, auto-picks columns by name pattern matching (e.g., "response" → response field, "operator" → operator field, "part" → part field)
5. **Toolbar expansion:** Added Reliability, MSA, and Time Series shortcuts to toolbar ribbon (previously only Stats, SPC, Viz, ML, Bayesian had toolbar buttons)
6. **Removed inline SVG style** on chart settings gear (CSS rule already handles it)

**Files changed:**
- `services/svend/web/templates/analysis_workbench.html` — grid CSS, icon rendering, next steps, smart select, toolbar

**Verification:** JS braces balanced (1386 pairs), div tags balanced (747 pairs).

---

### 2026-02-15 — Analysis Workbench UX: Config Forms + Convenience Features

**What:** Comprehensive UX pass on the analysis workbench. Added config forms for all 36 Sprint 2+3 analyses that were missing frontend forms, plus 8 convenience features.

**Config forms added (36 analyses):**
split_plot_anova, repeated_measures_anova, anom, glm, hsu_mcb, nominal_logistic, ordinal_logistic, poisson_regression, orthogonal_regression, nonlinear_regression, variable_acceptance_sampling, multiple_plan_comparison, gage_rr_expanded, capability_sixpack, laney_p, laney_u, between_within, mewma, generalized_variance, dotplot, individual_value_plot, interval_plot, contour, contour_overlay, surface_3d, run_chart, factor_analysis, correspondence_analysis, item_analysis, box_cox, johnson_transform, grubbs_test, graphical_summary, auto_profile. Existing post-hoc group extended to include scheffe_test, bonferroni_test.

**UX convenience features:**
1. Search box with real-time filtering in analysis dialog
2. Favorites system (localStorage star toggle)
3. Chart export (Plotly mode bar on hover with download button)
4. Copy button on result summaries (strips color tags)
5. Chart tab labels now show chart titles instead of "1-4"
6. Remember last config per analysis (localStorage save/restore)
7. 30+ missing menu items added to frontend analysis grid
8. Color map fixed (title, highlight, good tags now render)

**Bug fixes:**
- `run_spc_analysis()` crashed on multivariate analyses (mewma, generalized_variance) because it unconditionally called `df[measurement]` before reaching the analysis block. Now gracefully falls back to first numeric column.
- `executeAnalysis()` now collects checkbox values with IDs (interactions, show_mean)
- Duplicate CSS rule for `.aw-analysis-item .name` merged

**Files changed:**
- `services/svend/web/templates/analysis_workbench.html` — config forms, UX features, CSS
- `services/svend/web/agents_api/dsw_views.py` — SPC null-measurement guard

**Verification:** `py_compile` passes. JS brace balance verified (1337 pairs). All form IDs verified against backend config.get() keys.

---

### 2026-02-15 — Minitab Parity Sprint 3: Final 11 Items — 99% Parity Achieved

**What:** Closed all remaining Minitab parity gaps. Every item in the DSW_gaps.md audit is now checked off.

**New DSW analyses (in `dsw_views.py`):**
1. `poisson_regression` — GLM with Poisson family, log link, IRR with 95% CI, deviance goodness-of-fit, overdispersion detection, optional offset/exposure variable.
2. `split_plot_anova` — Split-plot ANOVA with proper error term separation. WP factors tested against WP error, SP factors against residual. Residual + main effects plots.
3. `repeated_measures_anova` — Within-subject ANOVA with Mauchly sphericity test + Greenhouse-Geisser/Huynh-Feldt epsilon corrections. Profile plot + spaghetti plot.
4. `scheffe_test` — Scheffé's post-hoc (most conservative, controls for all contrasts). Pairwise CIs + forest plot.
5. `bonferroni_test` — Bonferroni post-hoc (pairwise t with adjusted α). CI forest plot.
6. `hsu_mcb` — Hsu's Multiple Comparisons with the Best. Identifies which groups could be best at given confidence level.
7. `correspondence_analysis` — SVD-based decomposition of contingency table. Biplot, inertia scree, chi-squared test, row/column coordinates.
8. `item_analysis` — Cronbach's α (overall + if-deleted), corrected item-total correlations, inter-item correlation heatmap, standardized alpha.
9. `multiple_plan_comparison` — Compare OC curves, AOQ curves, producer/consumer risk across multiple sampling plans side-by-side.
10. `generalized_variance` — |S| (covariance determinant) control chart for multivariate process variability. Subgroup-based with UCL/LCL/OOC detection.
11. `contour_overlay` — Overlay contour lines from multiple responses on one plot for DOE optimization.

**DOE augment design (in `doe.py` + `experimenter_views.py`):**
- `augment_design()` method with 5 modes: axial (CCD upgrade), foldover (de-alias 2FIs), replicate, center points, D-optimal (greedy exchange)
- All 5 modes smoke-tested. Routing + design_types endpoint updated.

**Files changed:**
- `services/svend/web/agents_api/dsw_views.py` — 11 new analysis blocks
- `services/svend/agents/agents/experimenter/doe.py` — `augment_design()` method
- `services/svend/web/agents_api/experimenter_views.py` — augment routing + design type entry
- `DSW_gaps.md` — All items checked, scorecard updated to ~99% parity

**Verification:** `py_compile` passes on all 3 modified Python files. DOE augment smoke-tested all 5 modes.

---

### 2026-02-15 — Minitab Parity Sprint 2: 8 New Analyses + DOE Expansion

**What:** Continued Minitab/JMP competitive parity closure. Added 6 new DSW analyses, 1 expanded MSA study, and 4 new DOE design types.

**New DSW analyses (in `dsw_views.py`):**
1. `orthogonal_regression` — Deming/orthogonal regression with Bland-Altman plot. Bootstrap CI, OLS comparison. Supports method comparison studies.
2. `nonlinear_regression` — 9 preset curve models (exponential, power, logistic, logarithmic, polynomial2/3, Michaelis-Menten, Gompertz, Hill). Returns parameters, SE, R², AIC/BIC, residual plot.
3. `variable_acceptance_sampling` — k-method variables sampling plan (MIL-STD-414/ANSI Z1.9 style) with OC curve. Evaluates sample data against spec limits.
4. `capability_sixpack` — 6-panel Minitab-style capability display: I/MR (or Xbar/R), run chart, histogram with specs, normal probability plot, capability stats. Supports subgrouping.
5. `anom` — Analysis of Means with Bonferroni-corrected decision limits. ANOM chart with UDL/LDL. Supports balanced and unbalanced designs, factor format.
6. `gage_rr_expanded` — Multi-factor MSA with up to 8 factors beyond standard part/operator. GLM-style variance decomposition. Identifies largest reproducibility source.

**New DOE design types (in `doe.py` + `experimenter_views.py`):**
7. `simplex_centroid` — Mixture design with vertices, edge midpoints, face centroids (all points sum to 1).
8. `simplex_lattice` — {q, m} lattice design on the simplex with configurable degree.
9. `extreme_vertices` — Constrained mixture regions with lower/upper bounds on each component.
10. `split_plot` — Hard-to-change (whole-plot) and easy-to-change (sub-plot) factors with restricted randomization.

**Files changed:**
- `agents_api/dsw_views.py` — Added 6 analyses: orthogonal_regression, nonlinear_regression, variable_acceptance_sampling, capability_sixpack, anom, gage_rr_expanded
- `agents/agents/experimenter/doe.py` — Added 4 methods: simplex_centroid, simplex_lattice, extreme_vertices, split_plot
- `agents_api/experimenter_views.py` — Routing for mixture designs + split-plot, design_types endpoint additions

---

### 2026-02-15 — Minitab Parity Sprint 1: 7 New Analyses

**What:** Competitive gap closure targeting Minitab/JMP parity for LSS practitioners. Added 7 missing analysis types across DSW, SPC, and DOE.

**New analyses:**
1. `variance_test` — One/two-variance tests (chi-square, F-test, Bartlett's + Levene's). Supports wide and factor formats.
2. `poisson_2sample` — Two-sample Poisson rate test with exact conditional test, rate ratio + CI.
3. `nominal_logistic` — Nominal logistic regression for 3+ categories. Auto-excludes response from predictors.
4. `attribute_capability` — Attribute capability (DPU, DPO, DPMO, yield %, sigma level with 1.5σ shift).
5. `nonnormal_capability_np` — Nonparametric capability using percentile method with AD normality comparison.
6. `g_chart` / `t_chart` — Rare events SPC charts (geometric / Weibull). Auto-detect chart type.
7. `general_full_factorial` — Multi-level factorial DOE with >1000 run warning.

**Files changed:**
- `agents_api/dsw_views.py` — Added 5 stats + 2 SPC implementations
- `agents_api/experimenter_views.py` — Added general_full_factorial alias

---

### 2026-02-15 — Wire Study/Projects Evidence Pipeline to 100%

**What:** Audited and completed the evidence pipeline so all tools feed findings into the core `Evidence` → `EvidenceLink` → Bayesian update system. Previously ~70% wired; now 100%.

**Files changed:**
- `core/views.py` — Fixed `evidence_list` POST to call `synara.apply_evidence(link)` after creating EvidenceLink (was missing Bayesian update)
- `templates/projects.html` — Changed `addEvidence()` to call `/api/core/projects/{id}/evidence/` instead of `/api/core/evidence/from-code/`
- `agents_api/models.py` — Added `self.sync_evidence_to_core(evidence)` in `Problem.add_evidence()` so all legacy paths auto-sync to core
- `agents_api/views.py` — Added `record_tool_evidence()` utility function accepting `project_id` (direct core) or `problem_id` (legacy sync)
- `agents_api/spc_views.py` — Updated all 6 evidence spots (control_chart, chart_recommendation, capability_study, statistical_summary, analyze_data, gage_rr) to use `record_tool_evidence()` with `project_id` support
- `agents_api/experimenter_views.py` — Updated all 6 evidence spots (power_analysis, generate_design, full_experiment, analyze_results, contour_plot, optimization) with same pattern
- `agents_api/rca_views.py` — Added auto-evidence in `update_session()` when root_cause is set on a project-linked session
- `agents_api/a3_views.py` — Added auto-evidence in `update_a3_report()` when root_cause or countermeasures are updated on project-linked A3
- `agents_api/guide_views.py` — Added `save_finding` endpoint for manual evidence capture from guide chat
- `agents_api/guide_urls.py` — Added `save-finding/` route
- `agents_api/whiteboard_views.py` — Added `save_finding` endpoint for whiteboard observations as evidence
- `agents_api/whiteboard_urls.py` — Added `save-finding/` route

**New API endpoints:**
- `POST /api/guide/save-finding/` — Save guide observation as evidence (project_id, summary, optional hypothesis_ids + likelihood_ratio)
- `POST /api/whiteboard/boards/<room_code>/save-finding/` — Save whiteboard finding as evidence (summary, optional element_ids, hypothesis_ids + likelihood_ratio)

**Verification:** `python manage.py check` → 0 issues. All tools now create `core.Evidence` records and trigger Bayesian updates when linked to projects.

---

### 2026-02-15 — Standardize Operations Widgets to Svend Theme

**Files changed:**
- `templates/calculators.html` — Added themed CSS for `input[type="range"]` (green accent thumb, dark track), `input[type="checkbox"]` (accent-primary when checked, white checkmark), and `.svend-toggle` component (replaces inline-styled toggle). Updated Yamazumi toggle from inline styles to CSS classes. Removed manual JS style manipulation for toggle (CSS `:checked` handles it). Removed inline size on breakdowns checkbox. Affects 13 range sliders, 1 checkbox, 1 toggle across Line Simulator, Queue Simulator, Kanban, Beer Game, TOC/DBR, Reliability, and Desirability calculators.
**Verification:** Open /app/calculators/, check any simulator — sliders should show green thumb on dark track, checkbox shows green with white checkmark when checked.

---

### 2026-02-15 — Training Partner Student Program

**What:** Full training partner flow for India LSS certification programs. Training centers (e.g., Contiprove) send students to a partner-specific registration link. Students get Pro access with 100% Stripe coupon for the program duration, then convert to ₹749/mo (50% of India Pro) after the free period ends.

**Flow:** `/register/?partner=contiprove` → register → auto-subscribe with Stripe coupon → Pro access with countdown timer → 14-day and 3-day reminder emails → upgrade CTA in settings at discounted rate

**Files changed:**
- `accounts/models.py` — Added `partner_code`, `partner_discount_ends_at` to User model
- `accounts/migrations/0011_user_partner_fields.py` — AddField x2
- `accounts/billing.py` — Added `TRAINING_PARTNERS` config, `get_partner_config()`, `partner_activate()` view, partner-aware checkout
- `accounts/urls.py` — Added `/billing/partner-activate/` route
- `api/views.py` — `register()` captures partner code + sets expiry + schedules reminders; `me()` returns partner fields
- `api/tasks.py` — Added `send_partner_expiry_reminder()` tempora task
- `templates/register.html` — Captures `?partner=` param, redirects to partner-activate
- `templates/settings.html` — Partner info section with countdown timer + upgrade CTA
- `templates/base_app.html` — Gold partner expiry banner (14-day warning, dismissable)

**Verification:** Register at `/register/?partner=contiprove`, check `me()` returns partner fields, verify settings shows timer

---

### 2026-02-15 — Partnerships Page (Education, Stiykist, Consulting)

**Files changed:**
- `templates/partnerships.html` — New page extending tool_base.html with three sections: Education partnerships (training company licensing + direct training), Project Stiykist (Ukrainian defense innovation, entirely in Ukrainian), Consulting placeholder (March 2026)
- `svend/urls.py` — Added /partnerships/ route and sitemap entry
- `templates/landing.html` — Added Partnerships link to footer Company column
**Verification:** `curl -s -o /dev/null -w "%{http_code}" -H "X-Forwarded-Proto: https" http://127.0.0.1:8000/partnerships/` → 200

---

### 2026-02-15 — Regional PPP Pricing via Cloudflare Geo-IP

**What:** Added purchasing power parity pricing for 6 regions using Cloudflare's `CF-IPCountry` header. Landing page and Stripe checkout both resolve the same geo-IP, so displayed price always matches checkout price. No external dependencies — Cloudflare adds the header automatically through the tunnel.

**Regions:**
- India (INR): ₹1,499 / ₹3,499 / ₹9,999
- Indochina — VN, TH, KH (VND): ₫349,000 / ₫799,000 / ₫2,499,000
- Philippines (PHP): ₱1,290 / ₱2,990 / ₱8,990
- Malaysia (MYR): RM99 / RM229 / RM699
- Indonesia (IDR): Rp249,000 / Rp579,000 / Rp1,799,000
- Ukraine (UAH): ₴349 / ₴899 / ₴2,999
- Default / everyone else (USD): $49 / $99 / $299

**billing.py:**
- Added `COUNTRY_TO_REGION`, `REGIONAL_PRICES` (7 regions × 3 tiers = 21 price IDs), `REGIONAL_DISPLAY`
- Added `get_region(request)`, `get_regional_price(plan, region)`, `get_pricing_context(request)` helpers
- `create_checkout_session()` now resolves regional price via `get_region()` — consistent with landing page display
- All 25 price IDs (4 legacy + 21 regional) registered in `PRICE_TO_TIER` for webhook tier resolution

**Landing page:**
- Replaced `TemplateView` with `landing_view()` function that passes pricing context
- Pricing cards use `{{ currency_symbol }}{{ price_pro }}` etc. instead of hardcoded `$49`

**Files changed:**
- `accounts/billing.py` — regional pricing config + routing
- `svend/urls.py` — landing_view replaces TemplateView
- `templates/landing.html` — template variables in pricing cards

**Verification:** `curl -H "CF-IPCountry: IN"` returns ₹ prices; `curl -H "CF-IPCountry: UA"` returns ₴ prices; no header returns $ prices

---

### 2026-02-15 — VSM Enterprise Tenant Support + Enterprise Rollout Debt Documentation

**What:** Added dual owner/tenant ownership to ValueStreamMap, mirroring the `core.Project` pattern. Personal VSMs use `owner` FK, enterprise VSMs use `tenant` FK. A CheckConstraint ensures exactly one is set. Created `get_user_vsms(user)` helper and replaced all 15 `owner=request.user` queries across vsm_views.py. Documented the remaining 6 module rollout as technical debt with a breadcrumb trail.

**Model (agents_api/models.py):**
- Added `tenant` FK (nullable, CASCADE) to `ValueStreamMap`
- Added `created_by` FK (SET_NULL) for tracking who created tenant-owned VSMs
- Made `owner` nullable (was required) for tenant-owned records
- Added `CheckConstraint` `vsm_has_single_owner`: owner XOR tenant
- Added indexes on `(owner, status)` and `(tenant, status)`
- Updated `to_dict()` with `owner_id` and `tenant_id`

**Views (vsm_views.py):**
- Added `get_user_vsms(user)` helper: `Q(owner=user) | Q(tenant_id__in=user_tenant_ids)` via Membership
- Replaced 15 instances of `owner=request.user` with `get_user_vsms(request.user)`
- `create_vsm()` accepts optional `tenant_id` for enterprise VSM creation
- `create_future_state()` inherits ownership model (owner or tenant) from parent VSM

**Debt (.kjerne/DEBT.md):**
- Documented enterprise rollout plan for 6 remaining modules: FMEA, RCA, A3, Reports, Whiteboards, Hoshin
- Included proven pattern, priority ordering, and breadcrumb trail referencing VSM as template

**Files changed:**
- `agents_api/models.py` — tenant FK, created_by FK, owner nullable, constraint, indexes
- `agents_api/migrations/0033_vsm_enterprise_tenant.py` — schema changes
- `agents_api/vsm_views.py` — `get_user_vsms()` helper + 15 query replacements
- `.kjerne/DEBT.md` — enterprise rollout debt section

**Verification:** Create VSM as personal user → appears in list. Create VSM with `tenant_id` → other tenant members see it via `get_user_vsms()`. CheckConstraint prevents setting both owner and tenant.

---

### 2026-02-15 — VSM Packet System: Current↔Future Linking, Diff UI, Calculator Tracking, Hoshin Bridge

**What:** The VSM evolved into a composable primitive feeding 11+ calculators. This change explicitly links current↔future state VSMs, adds a station-by-station diff with suggested improvement events, tracks which VSM sourced each calculator session, and surfaces VSM links in Hoshin projects.

**Model:**
- Added `parent_vsm` self-FK to `ValueStreamMap` (with `related_name="future_states"`)
- Updated `to_dict()` to include `parent_vsm_id`
- Migration `0032` with `RunPython` backfill for existing future-state VSMs

**Views (vsm_views.py):**
- `create_future_state()` now sets `parent_vsm=current` on the new future VSM
- `get_vsm()` returns `linked` dict with parent info and child future states for navigation
- `compare_vsm()` uses FK-based lookup (with reverse + legacy fallback), returns `station_diffs` with per-station deltas and deterministic event suggestions (SMED/Kaizen/TPM/Standard Work/Elimination)
- `generate_proposals()` uses same FK-based lookup

**VSM Template (vsm.html):**
- Status badge below VSM name: green "Current State" or purple "Future State"
- Navigation links: "View Current State" ← / → "View Future State"
- Full diff modal replacing `alert()`: 3 summary metric cards + station comparison table + color-coded event pills + events summary
- Save-before-create on future state creation with confirmation

**Calculator Template (calculators.html):**
- `_lastImportedVsmId` / `_lastImportedVsmName` tracked on VSM import
- Persisted via `collectState()` / `restoreState()` in localStorage
- Source indicator next to VSM import button links back to source VSM

**Hoshin Template (hoshin.html):**
- Source VSM link in project detail info grid
- Source VSM link in quick links sidebar

**Files changed:**
- `agents_api/models.py` — `parent_vsm` FK + `to_dict()`
- `agents_api/migrations/0032_valuestreammap_parent_vsm.py` — field + backfill
- `agents_api/vsm_views.py` — 4 view fixes
- `templates/vsm.html` — status badge, nav links, diff modal
- `templates/calculators.html` — source tracking + indicator
- `templates/hoshin.html` — VSM link in detail + sidebar

**Verification:** Create current-state VSM → "Create Future State" → edit future (change changeover/cycle time) → "Compare States" → see diff modal with SMED/Kaizen pills → navigate between states via links

---

### 2026-02-14 — Landing Page Repositioning: "Experimentation Platform for Quality & Operations"

**What:** Full repositioning of svend.ai landing page from "Statistical Analysis & SPC Software | Minitab Alternative" to "Continuous Improvement & DOE Software | Experiment Tracking for Quality Teams." Motivated by SEO research showing head terms (statistical analysis software, SPC software) are unwinnable against Minitab/JMP/SAS with decades of backlinks, while "continuous improvement software" ($1.85B market), "DOE software" (new entrants breaking through), and "hypothesis tracking" (zero competition) are realistic targets.

**Changes:**
- Title/meta/OG/Twitter repositioned to target CI, DOE, experiment tracking keywords
- Hero: "Minitab + AI at 68% less" → "From signal to verified improvement"
- Features grid reframed as experimentation lifecycle (Measure, Design, Analyze, Investigate, Track Hypotheses, Deploy)
- "How it works" 4-card grid replaced with 7-step Insight Spine vertical timeline (CSS already existed)
- Pricing: "Choose your plan" → "One platform, not five subscriptions" with tool stack cost comparison ($3K-$10K vs $588/yr)
- FAQ expanded from 4→6 questions targeting "continuous improvement software" and "hypothesis tracking" search intent
- Footer expanded from 3-link row to 4-column grid with Product/Resources/Company sections
- All 11 Minitab/JMP name references removed
- Structured data: lowPrice 19→0, highPrice 199→299, aggregateRating deleted (unverifiable)
- Dead CSS cleaned up (tool-badge--python/sympy/z3/verify, tools-grid, tool-card)
- Contact email unified to hello@svend.ai

**Files changed:** `templates/landing.html`

---

### 2026-02-14 — Whitepaper Management + Content Tab Split

**What:** Added whitepaper CRUD and analytics to the Internal dashboard, separated from blog management with sub-tab navigation.

**Backend:**
- `WhitePaper` model (uuid PK, title, slug, description, body, meta_description, topic, status, gated, author, timestamps)
- `WhitePaperDownload` model (paper FK, downloaded_at, referrer_domain, ip_hash, user_agent, email, is_bot)
- 6 API endpoints: list (with download_count annotation), get, save, publish, delete, analytics
- Analytics: daily downloads, top papers, referrer breakdown, totals (downloads, unique, emails, bots)

**Frontend (internal_dashboard.html):**
- Content tab split into Blog / Whitepapers sub-tabs via `switchContentSub()`
- Full whitepaper editor: title, slug, topic, meta description, abstract, markdown body with live preview, gated checkbox
- Whitepaper list with status badges, gated indicator, download counts
- Whitepaper analytics: 4 KPI cards + 3 charts (downloads over time, top papers, traffic sources)
- CRUD functions: newWhitepaper, saveWhitepaper, publishWhitepaper, deleteWhitepaper, loadWhitepaper

**Files changed:**
- `api/models.py` — WhitePaper + WhitePaperDownload models
- `api/internal_views.py` — 6 whitepaper endpoints + download_count annotation
- `api/urls.py` — 6 URL routes under internal/whitepapers/
- `templates/internal_dashboard.html` — sub-tab toggle, whitepaper editor/analytics HTML + all JS functions
- `api/migrations/0010_whitepaper_models.py` — migration (applied)

**Verification:** Internal dashboard > Content tab > click "Whitepapers" sub-tab > create/save/publish a whitepaper, analytics charts render

---

### 2026-02-14 — Content BI Suite (Device/Browser/OS/Hourly/Sessions/Referrer)

**What:** Added rich content analytics to both Blog and Whitepaper analytics sections. Parses user agents for device/browser/OS breakdown, builds hourly activity heatmap, tracks visitor sessions with device fingerprinting, and extracts full referrer page paths and UTM parameters.

**Backend (internal_views.py):**
- `_parse_ua(ua)` — regex-based UA parser returning device (Mobile/Desktop/Tablet), browser (Chrome/Firefox/Safari/Edge/Opera/curl), OS (Windows/macOS/iOS/Android/Linux)
- `_extract_utm(url)` — extracts utm_source/medium/campaign/content/term from referrer query strings
- `_content_bi(qs, ua_field, time_field, referrer_field)` — shared analytics builder producing devices, browsers, os, hourly (0-23), visitor sessions (grouped by ip_hash, top 50), referrer_pages (full domain+path), and utm_sources
- Both `api_blog_analytics()` and `api_whitepaper_analytics()` enhanced to include BI data
- `WhitePaperDownload` model gained `referrer` (URLField) and `path` (CharField) fields

**Frontend (internal_dashboard.html):**
- 6 new chart containers added to both Blog and Whitepaper analytics sections
- `renderContentBI(data, prefix)` shared JS function renders: devices doughnut, browsers horizontal bar, OS doughnut, hourly activity bar (peak hour highlighted), visitor sessions table (hash/hits/device/browser/os/source/first_seen), referrer pages table (full path + count)
- Wired to `loadBlogAnalytics()` and `loadWhitepaperAnalytics()`

**Files changed:**
- `api/internal_views.py` — 3 helper functions + 2 endpoint enhancements
- `api/models.py` — WhitePaperDownload referrer/path fields
- `api/migrations/0011_whitepaper_download_referrer_path.py` — migration (applied)
- `templates/internal_dashboard.html` — 12 chart containers (6 per section) + renderContentBI function + wiring

**Verification:** Internal dashboard > Content tab > Blog analytics shows device/browser/OS doughnut/bar charts, hourly activity, visitor sessions table, referrer pages. Same for Whitepapers sub-tab.

---

### 2026-02-14 — Free SEO tool pages (/tools/)

**What:** Built 5 standalone calculator pages as SEO landing pages. Each is a fully functional client-side tool with no login required, targeting high-intent search keywords.

**Pages created:**
- `/tools/` — Index page linking to all calculators (CollectionPage structured data)
- `/tools/cpk-calculator/` — Cpk/Ppk from pasted measurements + spec limits (histogram, normal overlay)
- `/tools/sample-size-calculator/` — Sample size for t-tests, proportions, ANOVA (power curve)
- `/tools/oee-calculator/` — OEE from production data (ring gauges, six big losses, world-class benchmarks)
- `/tools/sigma-calculator/` — DPMO ↔ sigma level ↔ yield ↔ Cpk bidirectional converter
- `/tools/takt-time-calculator/` — Takt time + line balance chart with station cycle times

**SEO features per page:**
- Unique `<title>`, `<meta description>`, `<meta keywords>`, canonical URL
- Open Graph + Twitter Card meta tags
- WebApplication structured data (schema.org)
- FAQPage structured data (schema.org) with 3-5 questions each
- All pages added to sitemap.xml via StaticSitemap

**Files changed:**
- `templates/tool_base.html` — New shared base template (Svend branding, nav, footer, CSS, FAQ toggle)
- `templates/tools/index.html` — Tools index with cards linking to each calculator
- `templates/tools/cpk_calculator.html` — Cpk/Ppk calculator
- `templates/tools/sample_size_calculator.html` — Sample size calculator
- `templates/tools/oee_calculator.html` — OEE calculator
- `templates/tools/sigma_calculator.html` — Sigma/DPMO converter
- `templates/tools/takt_time_calculator.html` — Takt time calculator
- `svend/urls.py` — Added 6 tool routes + sitemap entries
- `templates/landing.html` — Added "Calculators" link to nav

**Verification:** Visit https://svend.ai/tools/ and each sub-page. All calculators compute results with default data on page load. No auth required.

---

### 2026-02-14 — Split Analysis ribbon into Statistics + Quality tabs

**What:** The Analysis ribbon tab was dense (37 buttons in one horizontal scroll). Split into two focused tabs:
- **Statistics** — Hypothesis Tests (t-Test, ANOVA, Non-Param, Post-Hoc), Modeling (Regression, GLM, Multivariate, Factor), Diagnostics (Normality, Survival), All Tests
- **Quality** — Control Charts (14 chart types), Capability (Cp/Cpk, B/W, NN Cap), MSA (Gage R&R, Sampling), Reliability (Weibull, Dist ID, K-M, CIF, ALT, Repair, Warranty, Plan)

Tab order: Data | Prepare | Statistics | Quality | Experiment | ML | Simulate | Synara

**Files changed:**
- `templates/workbench_new.html` — replaced `ribbon-analysis` with `ribbon-statistics` + `ribbon-quality`

**Verification:** Click Statistics tab → see tests/modeling/diagnostics. Click Quality tab → see control charts/capability/MSA/reliability.

---

### 2026-02-14 — Cross-Simulator Station Sharing

**What:** Added 4 pull connections between the discrete-event simulators so they can share station layouts:

1. **Line Sim ← Kanban Sim**: Pull stations (same `{name, cycleTime}` structure, direct copy)
2. **Kanban Sim ← Line Sim**: Pull stations (bidirectional — same structure)
3. **TOC/DBR ← Line Sim**: Pull stations with conversion (`cycleTime` sec → `capacity` units/hr via `3600/CT`)
4. **TOC/DBR ← Bottleneck**: Pull stations with same CT→capacity conversion

Now a user can: VSM → Line Sim → Kanban Sim → TOC/DBR as a connected pipeline without re-entering station data.

**Files changed:**
- `templates/calculators.html` — 4 pull buttons in HTML + 4 JS functions (`pullLinesToKanban`, `pullKanbanToLine`, `pullLinesToTOC`, `pullBottleneckToTOC`)

**Verification:** Open Line Sim with stations → switch to Kanban Sim → click "← Line Sim" → stations populate. Same for TOC/DBR ← Line Sim and ← Bottleneck.

---

### 2026-02-14 — Operations Calculator Hardening: Flow Gaps + VSM Import + Financial Layer

**What:** 10 changes to `calculators.html` addressing disconnections documented in `new_flows.md`:

**Flow Gaps (4):**
1. Changeover Matrix → Sequence Optimizer: Added "← Matrix" pull button. The code already checked `changeoverMatrix[from][to]` but users couldn't see it. Now surfaced with toast feedback.
2. Mixed-Model → Line Sim: Added "← Mixed-Model" pull button in Line Sim order queue. Calls existing `pushMixedToLineSim()`.
3. Due Date Risk → Actions: High-risk orders (<80% on-time) now show Reschedule and Flag buttons. Reschedule prompts for new due date and re-runs simulation. Flag adds visual indicator.
4. Before/After → Synara: Added "Log to Synara" button. Posts improvement summary as evidence to `/api/synara/{wbId}/evidence/add/`. Falls back to sessionStorage if no workbench.

**VSM Import Expansion (3):**
5. Capacity Load: New `loadVSMIntoCapacityLoad()` — creates work orders per station (CT × demand / 3600 = hours). Added `capacity-load` case to dispatcher.
6. RTO Staffing: New `loadVSMIntoRTO()` — sums station cycle times → rto-cycle, imports takt_time → rto-takt. Added `rto` case to dispatcher.
7. OEE: Expanded `loadVSMIntoOEE()` to also import `cycle_time` as ideal cycle time (was only importing `uptime`).

**Financial Layer (3):**
8. TOC/DBR: Added "Value per Unit ($)" input + "Throughput Value $/hr" result card. `updateTocMetrics()` now calculates `throughput × valuePerUnit`.
9. Changeover Matrix: Added "Hourly Cost Rate ($)" input + "Avg Cost" and "Best Sequence Cost" result cards. `calcChangeover()` converts minutes to $.
10. Sequence Optimizer: Added "Hourly Cost Rate ($)" input. Improvement summary now shows "= $X saved" alongside time savings.

**Files changed:**
- `templates/calculators.html` — all 10 changes (~175 lines added)

**Verification:** See plan file for 10-step verification checklist. Test each calculator individually.

---

### 2026-02-14 — Operations Calculator Flows & Surface Area Documentation

**What:** Created `new_flows.md` documenting the complete surface area of the 54 Operations calculators (49 active + 5 coming soon). Maps all SvendOps publish-pull data flows (38 keys, 6 pull connections, 2 push connections), all 16 simulation engines (6 discrete-event, 7 Monte Carlo, 3 statistical), financial capabilities (6 calculators with $ output), and critical flow gaps (4 high-priority, 6 medium, 5 low). Includes multi-material kanban v2 architecture proposal, VSM import surface analysis, and cross-simulator integration opportunities.

**Files changed:**
- `new_flows.md` — new file, full surface area documentation

**Verification:** Read `new_flows.md` — should contain 12 sections covering inventory, data bus, flow graph, simulations, financial, scheduling, kanban v2, gaps, coming-soon, cross-sim, VSM import, and summary counts.

---

### 2026-02-14 — Response + Factor (Stacked) Data Format for Two-Sample & Paired Tests

**What:** Added "Response + Factor" data format option to ttest2, paired_t, and wilcoxon. Users can now run these tests on stacked/long-format data (one measurement column + one grouping column) instead of requiring two separate numeric columns. This is standard practice in statistics — ANOVA, Mann-Whitney, Kruskal-Wallis, and F-test already supported it.

**Backend** (`agents_api/dsw_views.py`):
- `ttest2`: detects `data_format: "factor"`, splits response by factor levels (exactly 2 required), labels with group names
- `paired_t`: same detection, pairs by row order within each group (`reset_index(drop=True)`)
- `wilcoxon`: same pattern as paired_t

**Frontend** (`templates/workbench_new.html`):
- t-Test dialog: "Data Format" radio toggle (Two Columns / Response + Factor), shown for ttest2 and paired_t. Swaps labels and dropdowns.
- Generic "more" dialog: same toggle, shown for wilcoxon and equivalence tests
- `updateTTestDialog()` and `updateMoreDialog()` handle show/hide logic

**Verification:** t-Test dialog → Two-Sample → toggle to Response + Factor → select numeric response + categorical factor → Run

---

### 2026-02-14 — Interactive DSW: What-If Replay, Power Explorer, Monte Carlo Simulation

**What:** Made DSW statistical analysis interactive. After every analysis, users can explore "what if?" with client-side sliders — zero server round-trips. Plus a full Monte Carlo simulation engine that competitors charge $1,500+/yr for.

**Phase 1 — What-If Replay** (`dsw_views.py` + `workbench_new.html`):
- **Capability What-If**: After capability analysis, LSL/USL sliders update Cpk/Cp/DPMO/Yield/Sigma Level + Plotly histogram with movable spec lines — all client-side.
- **Regression What-If**: After linear regression, sliders per predictor update predicted value + 95% PI via dot product `ŷ = intercept + Σ(coef × val)`.
- Backend augments capability and regression responses with `what_if_data` (sufficient statistics).
- Client-side `normalCDF()` (Abramowitz & Stegun) and `normalPPF()` (Acklam rational approximation).

**Phase 2 — Interactive Power Explorer** (`dsw_views.py` + `workbench_new.html`):
- Auto-appends below every hypothesis test (t-test, two-sample t, paired t, ANOVA, chi-square).
- Three sliders: effect size, sample size, alpha → live power curve + marker + 80% threshold line.
- "Need n = X for 80% power (Y more samples)" message.
- Backend adds `power_explorer` metadata (Cohen's d, observed n/std, alpha) to 5 test branches.
- Client-side power via z-approximation: `computePowerJS()` handles ttest/ttest2/anova/chi2.

**Phase 3 — Monte Carlo Simulation Engine** (`dsw_views.py` + `workbench_new.html`):
- New `run_simulation()` function + `simulation` type in analysis dispatcher.
- 8 input distributions (Normal, Uniform, Lognormal, Weibull, Exponential, Gamma, Triangular, Beta) + "Fit from Data" (auto-selects best via `_fit_best_distribution()` helper).
- Transfer function via safe `eval()` with AST validation (rejects imports, restricts names) OR saved ML model prediction.
- Output: histogram, sensitivity tornado (±1σ), input-output correlation chart, percentiles, threshold probabilities.
- "Simulate" ribbon tab with full Monte Carlo dialog: dynamic variable builder, distribution parameter inputs, formula/model toggle, threshold inputs.
- Interactive threshold slider on output histogram — drag to see P(output > X) instantly.
- Security: `__builtins__: {}`, AST walk rejects forbidden names, cap 100k iterations × 20 variables.

**Files changed:**
- `agents_api/dsw_views.py` — `_fit_best_distribution()`, `run_simulation()`, `what_if_data` on capability/regression, `power_explorer` on 5 hypothesis tests, `simulation` route
- `templates/workbench_new.html` — normalCDF/PPF, appendCapabilityWhatIf, appendRegressionWhatIf, computePowerJS, appendPowerExplorer, openMonteCarloDialog, appendMCThresholdSlider, Simulate ribbon tab, renderStatsOutput hooks

**Verification:** Run capability analysis → What-If sliders appear. Run t-test → Power Explorer appears. Simulate tab → Monte Carlo dialog → Run → output histogram + tornado + threshold slider.

---

### 2026-02-14 — Constrained Optimization: Density, Bounds, Costs, Diminishing Returns

**What:** Made the optimizer reality-aware. Users can now define real-world constraints (feature bounds, sum limits, change costs) and the optimizer respects them via penalty terms. Joint-distribution density penalty keeps solutions in high-density regions. Diminishing returns analysis shows where to stop pushing.

**Backend — `optimize_model()` rewrite** (`agents_api/dsw_views.py`):
- **Joint density penalty**: Mahalanobis distance penalty ramps quadratically beyond the "ok" threshold, scaling with prediction magnitude. Keeps optimizer in observed data regions.
- **User-defined feature bounds**: `feature_bounds` dict overrides training-data min/max with tighter real-world limits (e.g., temperature 20-80 instead of 0-100). Applied directly to `differential_evolution` bounds.
- **Sum constraints**: `sum_constraints` array (e.g., sleep+study+gaming ≤ 24). Enforced via quadratic penalty scaled 10× prediction magnitude.
- **Cost weights**: `cost_weights` dict (1=easy, 10=hard to change). Penalizes changing expensive features proportional to fractional change × cost weight.
- **Diminishing returns**: Post-optimization, sweeps each feature from current→optimal in 11 steps. Finds knee point where 80% of total gain is achieved. Returns per-feature `{knee_pct, knee_value, total_gain}`.
- **Constraint satisfaction**: Reports whether each sum constraint was satisfied at optimal point.
- Extracted `_predict_numeric()` helper to module level (eliminates closure over changing variables).

**Frontend — Optimization Settings Panel** (`templates/models.html`):
- "Settings" button next to Optimize in topbar → toggles collapsible panel
- **Feature table**: Each numeric feature shows min/max override inputs (placeholders = training range) + cost weight input (1-10)
- **Sum constraint builder**: Click "+ Add constraint" → feature pill selector + operator (≤/≥) + limit input. Features toggle on/off with visual feedback.
- **Density checkbox**: "Stay near observed data" toggle (checked by default)
- `getOptSettings()` collects all settings → sent with optimize request
- `populateOptSettings()` rebuilds table on profiler init

**Frontend — Results Panel additions**:
- **Diminishing returns bars**: Per-feature horizontal bars showing solid green (80% of gain) + faded green (remaining 20%) + yellow knee marker. Shows where to stop pushing.
- **Constraint status**: Per constraint checkmark/cross with actual vs limit values.
- **Cost column**: When cost weights > 1 exist, prescription table adds a "Cost" column with dot indicators.

**Verification:** Profiler → Settings → set feature min/max overrides → add sum constraint → set cost on expensive feature → click Optimize → results show constrained optimal with diminishing returns bars and constraint status.

---

### 2026-02-14 — Smart Optimization: Decision Intelligence in Profiler

**What:** Enhanced the ML Profiler's Optimize feature from naive slider-moving into a full decision intelligence toolkit. When a user clicks "Optimize", they now get not just optimal values but a comprehensive assessment of whether those values are achievable and what to change.

**Backend — `optimize_model()` enrichment** (`agents_api/dsw_views.py`):
- **Prescription**: Per-feature actions (increase/decrease/switch/hold) with magnitude as % of range and from→to values
- **Feasibility scoring**: Mahalanobis distance from optimal point to training data centroid. Three-tier classification (high/moderate/low) using χ²-based thresholds scaled by feature count
- **Correlation violation detection**: Checks whether optimal values violate observed feature correlations (|r| > 0.5). Flags when optimizer pushes positively-correlated features apart or negatively-correlated features together
- **Edge warnings**: Flags features where optimal value is within 5% of observed min/max boundary
- **Sensitivity analysis**: Numerical gradient ∂prediction/∂feature at optimal point, ranked by total impact across feature range
- **Prediction interval**: 90% interval from tree ensemble at optimal point (RandomForest models)

**Backend — `_compute_feature_stats()`** (`agents_api/autopilot_views.py`):
- New helper storing means, stds, covariance matrix, and strong correlations at training time
- Wired into all 4 autopilot endpoints (clean_train, full_pipeline, augment_train, retrain)
- Stored in `training_config.feature_stats` for use by optimize endpoint

**Frontend — Optimization Results Panel** (`templates/models.html`):
- New `#optimize-results` panel in profiler main area (between response plot and PDP curves)
- Feasibility badge: green/amber/red with Mahalanobis distance and plain-language interpretation
- Prescription table: feature-by-feature actions with directional arrows and magnitude indicators
- Edge warning badges: amber chips for features at boundary of observed data
- Correlation conflict warnings: red text flagging violated correlations
- Sensitivity bars: horizontal bar chart ranking features by impact at optimal point
- Prediction interval at optimum: 90% interval with range and standard deviation
- Dismissible panel with "Dismiss" button

**Verification:** ML Hub → train model → Profiler → click Optimize → results panel appears with feasibility badge, prescription table, sensitivity bars. Try with data that has correlated features to see correlation warnings.

---

### 2026-02-14 — Fix CategoricalDtype Crash in ML Pipeline

**What:** ML pipeline ("From Intent" and "From Data") crashed with `Cannot interpret 'CategoricalDtype(...)' as a data type` when datasets contained categorical columns (e.g., Gender with Female/Male/Other). The root cause: `pd.Categorical(col).codes` returns int8 codes but the column can retain CategoricalDtype metadata, which numpy/sklearn can't interpret.

**Fix:** Added `.astype(int)` to all 9 occurrences of `.codes` across `dsw_views.py`. Also hardened `_clean_for_ml()`:
- Added `hasattr(y.dtype, 'categories')` check for CategoricalDtype detection
- Added `y.map(label_map).astype(np.int32)` explicit cast for target
- Added final safety cast: `X.apply(pd.to_numeric, errors='coerce').fillna(0)`

**Files changed:**
- `agents_api/dsw_views.py` — Fixed `_clean_for_ml()` (lines 767-799), `_auto_train()` dtype check, and 9 `.codes` calls across classification, regression, SHAP, and model comparison blocks

**Verification:** Upload CSV with categorical columns → "From Data" or "From Intent" → pipeline completes without CategoricalDtype error

---

### 2026-02-14 — Practical Significance: Effect Sizes + Decision Language

**What:** Enhanced 8 core statistical analyses to report effect sizes, practical significance classification, and prescriptive action language. Every analysis now answers "does this matter?" not just "is this significant?"

**Files changed:**
- `agents_api/dsw_views.py` — Added `_effect_magnitude()` and `_practical_block()` helper functions before `run_statistical_analysis()`. Enhanced:
  - **One-sample t-test**: Cohen's d, magnitude classification, prescriptive summary
  - **Two-sample t-test**: Cohen's d (pooled), unit-level context ("X is 15 units higher, 0.8 SDs")
  - **Paired t-test**: Cohen's d (within-subject), direction language ("values improved by...")
  - **One-way ANOVA**: Eta-squared + omega-squared, "factor explains X% of variation"
  - **Two-way ANOVA**: Partial eta-squared per factor, strongest effect highlighted
  - **Chi-square**: Cramér's V, association strength classification
  - **Regression**: R² reframed as practical effect size, RMSE in plain language, guide_observation with significant predictors
  - **Correlation**: P-values per pair (were missing), "KEY RELATIONSHIPS" section ranked by shared variance %

**Effect size thresholds (Cohen's conventions):**
- Cohen's d: <0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, ≥0.8 large
- Eta-squared: <0.01 negligible, 0.01-0.06 small, 0.06-0.14 medium, ≥0.14 large
- Cramér's V: <0.1 negligible, 0.1-0.3 small, 0.3-0.5 medium, ≥0.5 large
- R-squared: <0.02 negligible, 0.02-0.13 small, 0.13-0.26 medium, ≥0.26 large

**Key decisions:**
- Four-way interpretation matrix: {significant + meaningful → act}, {significant + small → consider cost}, {significant + negligible → ignore despite p-value}, {not significant + large effect → need more data}
- guide_observation now includes effect size for Synara evidence linking
- statistics dict includes effect_size_label for downstream consumers
- Correlation now computes per-pair p-values (Pearson/Spearman/Kendall) and highlights pairs with |r| ≥ 0.3

**Verification:** Load data in DSW → run any t-test, ANOVA, chi-square, regression, or correlation → "PRACTICAL SIGNIFICANCE" section appears after p-value conclusion with effect size, magnitude, and action recommendation.

---

### 2026-02-14 — JMP Visualization Parity: New Chart Types + Prediction Profiler

**What:** Added 5 new chart types to DSW and built an interactive Prediction Profiler in ML Hub, closing visualization gaps vs JMP.

**New DSW Chart Types** (`agents_api/dsw_views.py` `run_visualization()`):
- **Bubble Chart** (backend) — scatter with marker size mapped to 3rd variable, optional color grouping
- **Parallel Coordinates** — Plotly `parcoords` trace, supports numeric + categorical dimensions, color line
- **Contour Plot** — 2D contour from 3 variables using `scipy.interpolate.griddata`, cubic with linear fallback for NaN regions
- **3D Surface Plot** — Plotly `surface` trace, same griddata interpolation, interactive rotation/zoom
- **Mosaic Plot** — proportional rectangle tiles from `pd.crosstab()`, built with Plotly shapes + annotations, capped at 15 levels per variable

**Workbench Ribbon** (`templates/workbench_new.html`):
- 4 new buttons in Data tab Graph group: Parallel, Contour, Surface, Mosaic (with SVG icons)
- 4 new dialog cases in `openGraphDialog()`: parcoords (checkbox dimension picker), contour/surface (x/y/z dropdowns), mosaic (row/col dropdowns)
- Client-side `renderGraph('parcoords', ...)` case for parallel coordinates

**ML Hub Prediction Profiler** (`templates/models.html`):
- New "Profiler" tab in model detail modal
- Feature sliders: numeric = range slider with min/max from training data, categorical = dropdown
- Predicted value updates in real-time via debounced `POST /api/dsw/models/<uuid>/run/` with JSON body
- "Show Partial Dependence" button: sweeps each feature across 20 points, batch predicts, renders PDP curves as Plotly line charts
- CSS for profiler controls (grid layout, slider styling)

**Feature Info Storage** (`agents_api/autopilot_views.py`):
- New `_compute_feature_info(df, feature_names)` helper
- All 4 autopilot endpoints (clean_train, full_pipeline, augment_train, retrain) now store `feature_info` in `training_config`
- Contains `{type, min, max}` for numeric features and `{type, categories}` for categorical

**Model Inference Fix** (`agents_api/dsw_views.py` `run_model()`):
- Fixed categorical encoding mismatch: single-row `pd.Categorical().codes` produced different codes than training. Now uses stored training categories with `pd.Categorical(X[col], categories=sorted(train_cats))` for consistent encoding
- Decodes classification predictions via `label_map` (returns original labels, not integer codes)
- Added `label_map` to full_pipeline's `training_config` (was missing — other endpoints had it via `train_with_recipe`)

**Profiler → Full-Page Workbench** (`templates/models.html`):
- Profiler removed from modal tab → now a full-page pane (`#profiler-view`, `position:fixed; inset:0`)
- Layout: topbar (back button + model name/metrics) → sidebar (280px, prediction card + sliders) → main area (4-col PDP grid)
- "Profile" button on model cards (skip modal) and in modal footer (from detail view)
- `openProfiler()` hides `.ml-page`, shows profiler; `exitProfiler()` reverses; no new routes needed
- PDP curves auto-render on open, "Refresh" button for recompute
- Green cursor dot + vertical dotted line on each PDP curve tracks current slider value
- Cursors update instantly on slider drag via `Plotly.restyle`/`relayout` (no API call)
- For classification, PDP y-axis uses class probabilities for smooth curves
- Sliders styled with Svend green thumb/track (replaces browser default blue)
- `.pdp-cell` has border + border-radius for visual separation
- Responsive: 4-col > 1200px, 3-col > 900px, 2-col below
- Error state shown in UI instead of silent console.warn

**Verification:** `python3 manage.py check` → 0 issues. DSW Data tab → new chart buttons functional. ML page → train model → click model → Profiler tab → sliders update prediction → expand button → full-viewport with 4-col PDP grid → cursor dots track slider positions.

---

### 2026-02-13 — ML Hub Platform (Layers 6-11)

**What:** Transformed the ML subsystem from disposable models into a full lifecycle platform with altitude control (manual / guided / autopilot). Six layers built on top of ML Lab Layers 1-5.

**Layer 6: SavedModel Schema Evolution**
- `agents_api/models.py` — Added `project` FK (→ core.Project), `training_config` JSONField, `data_lineage` JSONField, `version` IntegerField, `parent_model` self-FK for version chains
- `agents_api/dsw_views.py` — Extended `save_model_to_disk()`, `list_models()`, `save_model_from_cache()`, `dsw_from_intent()`, `dsw_from_data()` with project linking, recipe capture, and lineage tracking
- Migration `0031_savedmodel_project_training_config.py` applied

**Layer 7: Synara Evidence Bridge**
- `agents_api/dsw_views.py` — Added `_create_ml_evidence()` function that auto-creates `core.Evidence` from ML results when project_id is provided. Wired into `dsw_from_intent()` and `dsw_from_data()`

**Layer 8: In-Memory Pipeline Engine**
- `agents_api/ml_pipeline.py` — **NEW** — `triage_clean_df()` (in-memory scrub), `forge_augment_df()` (in-memory Forge tabular), `_infer_forge_schema()` (auto-detect types from DataFrame), `train_with_recipe()` (train + capture full recipe)

**Layer 9: Autopilot Pipelines**
- `agents_api/autopilot_views.py` — **NEW** — 4 endpoints: `autopilot_clean_train` (Triage+Train), `autopilot_full_pipeline` (Clean→Compare→SHAP→Tune), `autopilot_augment_train` (Forge+Train), `retrain_model` (replay recipe)
- `agents_api/dsw_views.py` — Added `models_summary()` and `model_versions()` endpoints
- `agents_api/dsw_urls.py` — Added 6 URL patterns for autopilot, retrain, summary, versions

**Layer 10: ML Hub Frontend**
- `templates/models.html` — Rewrote from 371-line card grid to ~510-line ML Hub with: stats bar, project grouping, search/filter, enhanced model cards (color-coded metrics, version badges, quick actions), tabbed detail modal (Overview/Metrics/Recipe/Lineage/Inference), version history, retrain modal, prediction download

**Layer 11: Ribbon Integration + Dialogs**
- `templates/workbench_new.html` — Added "Autopilot" ribbon group (Clean+Train, Full Pipeline, Augment+Train buttons) between Auto ML and Time Series. Added "Hub" button to Models group. Added `openAutopilotDialog()` with CSV header parsing for target selector, mode-specific options (triage toggles, CV folds, Optuna trials, synthetic row slider). Added `renderAutopilotOutput()` for multi-stage pipeline results.
- `templates/models.html` — Complete rewrite. Page renamed "Machine Learning" (was "ML Hub" / "Saved Models"). Three-section layout: (1) Train section with drag-and-drop CSV upload, live data preview table (headers + first 8 rows), target column picker, three training mode cards (Quick Train / Full Pipeline / Augment+Train) with mode-specific options, Train button; (2) Results section with pipeline stage badges, summary cards (Triage/Forge/Optuna/Model), metrics grid, AI insight card, model comparison table, Plotly diagnostic plots grid, collapsible recipe; (3) Saved Models section with stats line, project filter, search, grouped model cards. Page is fully self-contained ML experience. Plotly CDN added.
- `templates/workbench_new.html` — Renamed "Hub" button to "ML Page" in Models ribbon group.

**Recallable Training Reports**
- `agents_api/dsw_views.py` — Added `model_report()` endpoint (`GET /api/dsw/models/<uuid>/report/`). Fetches stored `DSWResult` linked to a SavedModel, transforms the raw result_data into the frontend-expected format (pipeline_stages, cleaning, augmentation, comparison, tuning, metrics, shap_plots, plots, recipe).
- `agents_api/dsw_urls.py` — Added `models/<uuid:model_id>/report/` URL pattern.
- `templates/models.html` — Added "Report" tab to model detail modal. New `renderReport(data, container)` function renders full training results (pipeline stages, summary cards, metrics, comparison table, plots via Plotly, recipe) inside the modal. `loadReport(modelId)` fetches report data from the API. Report tab auto-loads when model is opened. Modal widened to 960px for plots.

**Verification:** `python3 manage.py check` → 0 issues. ML page at `/app/models/` — upload data, see preview, pick target, choose mode, train, see results with plots, manage saved models. Click any saved model → Report tab shows full training report with plots and metrics. Workbench ribbon has Autopilot shortcuts + ML Page link.

---

### 2026-02-13 — Synara Tab + Belief Engine Hardening

**What:** Major hardening of the Synara hypothesis tracking system. Added the missing Synara tab to the workbench, delete endpoints for evidence/links, and expansion signal UI.

**1. Synara tab in workbench ribbon**
New "Synara" tab alongside Data/Prepare/Analysis/Experiment/ML. Shows:
- Hypothesis list with posterior probability bars (color-coded: red <30%, yellow 30-70%, green >70%)
- Evidence list (last 10, with supports/weakens indicators and source labels)
- Expansion signals with amber alert cards, "Dismiss" and "Add Hypothesis" actions
- Add Hypothesis button (inline prompt for description + prior)
- Delete buttons on hypotheses and evidence
- Auto-refreshes on tab click and workbench load

**2. Delete endpoints for evidence and causal links**
- `DELETE /api/synara/{wb}/evidence/{id}/delete/` — removes evidence from belief graph
- `DELETE /api/synara/{wb}/links/delete/?from_id=...&to_id=...` — removes causal link
- Both save updated Synara state after deletion

**3. Expansion signals surfaced in UI**
Backend already detected gaps in the causal surface but the frontend never showed them. Now rendered as amber-bordered cards with signal type, description, and dismiss/resolve actions. Resolving with "new_hypothesis" prompts for description and creates the hypothesis.

**4. hypotheses.html confirmed as dead code**
URL `/app/hypotheses/` already routes to `projects.html`. The standalone `hypotheses.html` template calls legacy `/api/problems/` endpoints and is completely disconnected. No action needed.

**Files changed:**
- `templates/workbench_new.html` — Synara tab HTML + ribbon content, `synaraRefresh()`, `renderSynaraHypotheses()`, `renderSynaraEvidence()`, `renderSynaraSignals()`, `synaraAddHypothesis()`, `synaraDeleteHypothesis()`, `synaraDeleteEvidence()`, `synaraResolveSignal()`, auto-refresh on tab click and workbench load
- `agents_api/synara_views.py` — Added `delete_evidence()` and `delete_link()` endpoints
- `agents_api/synara_urls.py` — Added URL patterns for delete endpoints

**Verification:**
1. Open workbench → Synara tab visible in ribbon → shows hypotheses/evidence/signals
2. Add hypothesis → appears in list with probability bar
3. Delete hypothesis → removed from list
4. Delete evidence → removed from list
5. Expansion signals show amber cards → dismiss or resolve works
6. Link DSW result to hypothesis → switch to Synara tab → evidence appears

---

### 2026-02-13 — Studies & Hypothesis Tracking Polish

**What:** Fixed two broken features in the Studies (projects) and hypothesis tracking system.

**Fix 1: Hypothesis editing was a stub**
`editHypothesis()` was `alert('Edit functionality coming soon')`. Replaced with full implementation that reuses the create modal — pre-fills all fields (If/Then/Because, variables, testing plan, prior), switches button text to "Save Changes", and PUTs to the existing `/api/core/projects/{id}/hypotheses/{id}/` endpoint. `closeHypothesisModal()` resets edit state.

**Fix 2: Workbench → Study navigation was broken**
"Open study" link from workbench always went to `/app/projects/` (generic list) instead of the specific project. Fixed by: (1) removing the `onclick` override that discarded the hash, (2) adding hash-based deep link support to projects.html `DOMContentLoaded` — if URL has `#<uuid>`, auto-navigates to that project.

**Files changed:**
- `templates/projects.html` — `editHypothesis()` implementation, `saveHypothesisEdit()`, `closeHypothesisModal()` reset, form onsubmit wiring, hash-based routing on load
- `templates/workbench_new.html` — `updateProjectLink()` fixed to use href instead of hardcoded onclick

**Verification:**
1. Studies → view hypothesis → click "Edit" → modal opens with pre-filled fields → save → updates
2. DSW workbench with project selected → click "Open study" → navigates to specific project

---

### 2026-02-13 — Fix Stale "96% Less" Pricing Claims

**What:** Four references still said "96% less than Minitab" (from the old $19/mo Founder price). Updated to "68% less" to reflect current $49/mo Professional pricing ($588/yr vs Minitab's $1,851/yr).

**Files changed:**
- `templates/landing.html` — Hero title, Twitter meta description, structured data FAQ answer
- `templates/blog_detail.html` — CTA box text

**Verification:** View landing page → hero says "68% less". View page source → no "96%" references remain.

---

### 2026-02-13 — DSW → Synara Evidence Bridge + VSM Monte Carlo Savings

**What:** Two backend improvements to close product differentiation gaps.

**Feature 1: DSW → Synara "Link to hypothesis" button**
After any DSW analysis renders, a "Link to hypothesis" button appears below results. Users can pick a hypothesis from their Synara session, choose supports/weakens, and evidence is auto-created via the existing Synara API. No backend changes needed — purely frontend wiring.

**Feature 2: VSM Monte Carlo savings simulation**
VSM savings estimates now use 1,000 Monte Carlo simulations with three uncertainty sources (volume volatility ±15%, cost variation ±10%, improvement realization risk via Beta(4,2)). Proposal cards show 90% CI ranges and P(positive ROI) instead of single point estimates.

**Files changed:**
- `templates/workbench_new.html` — Added `appendLinkHypothesisPrompt()`, `loadHypothesesForLink()`, `submitHypothesisLink()` functions; call site in `renderStatsOutput()`
- `agents_api/hoshin_calculations.py` — Added `estimate_savings_monte_carlo()` function (wraps deterministic function with MC simulation)
- `agents_api/vsm_views.py` — Updated `generate_proposals()` to use Monte Carlo; added CI fields to proposal response
- `templates/vsm.html` — Updated proposal card to show savings range ($low — $high/yr, 90% CI, median, P(positive))

**Verification:**
1. DSW: Run analysis in workbench with Synara hypotheses → "Link to hypothesis" button appears → link works
2. VSM: Generate proposals → savings show CI range instead of single number

---

### 2026-02-13 — In-App Documentation for Key Features

**What:** Added contextual help panels to VSM, Synara (Hypotheses), and FMEA templates. Collapsible `<details>` sections explain workflows without cluttering the UI.

**Files changed:**
- `templates/vsm.html` — "How to build a VSM" guide, field tooltips (C/T, C/O, Uptime, Batch, Scrap), metric tooltips (Lead Time, PCE, Takt)
- `templates/hypotheses.html` — "How hypothesis tracking works" panel explaining Bayesian reasoning and expansion signals
- `templates/fmea.html` — S/O/D scoring guide, hypothesis linking explanation, enhanced empty state

**Verification:** Visit each module → collapsible help visible at top (closed by default). VSM property labels show tooltips on hover.

---

### 2026-02-13 — Pricing Restructure: $49/$99/$299

**What:** Restructured pricing tiers. Killed Founder tier (legacy users grandfathered), renamed Pro→Professional at $49/mo, Team→$99/mo, Enterprise→$299/mo. Positions Svend as serious tool at 68% less than Minitab ($588/yr vs $1,851/yr).

**Files changed:**
- `accounts/constants.py` — Updated tier labels, prices, comments
- `accounts/billing.py` — New Stripe price IDs (Pro: price_1T0Y13, Team: price_1T0Y36, Enterprise: price_1T0Y42), legacy prices kept for existing subscribers, founder checkout redirects to pro
- `accounts/permissions.py` — Updated tier comments and upgrade messages
- `templates/landing.html` — New 4-card pricing (Free/Professional/Team/Enterprise), removed founder availability JS, updated meta descriptions and FAQ
- `templates/settings.html` — Removed hardcoded seat price from invite text
- `api/tasks.py` — Updated all email campaigns from "Founder $19" → "Professional $49"
- `api/internal_views.py` — Updated system prompt pricing
- `agents_api/llm_manager.py` — Updated docstring (Pro→Professional)
- `svend_config/config.py` — Updated price description
- `CLAUDE.md`, `DSW_gaps.md`, `STANDARD.md` — Updated pricing references
- `reference_docs/ARCHITECTURE.md`, `LAUNCH_PLAN.md` — Updated pricing tables and market position
- `TECH_DEBT.md`, `services/svend/__init__.py`, `agents/agents/CLAUDE.md` — Updated pricing
- `site/site/index.html`, `agents/site/templates/landing.html`, `agents/site/templates/base.html`, `site/mockups/landing.html` — Updated "$19/month" → "$49/month"

**Verification:** Visit svend.ai → pricing section shows Free/$49/$99/$299. Billing checkout routes to new Stripe prices. Existing founder/pro subscribers unaffected (legacy price IDs still mapped).

---

### 2026-02-13 — ML Lab Diagnostic Engine: Full visualization suite for all ML features

**Context:** Pricing increase (Professional $49, Team $99, Enterprise $299) requires every ML feature to have JMP-competitive diagnostics. Previously From Intent/From Data had zero plots, XGBoost/LightGBM had 2 plots each.

**Files changed:**
- `agents_api/dsw_views.py` — New shared `_build_ml_diagnostics()` engine:
  - Classification (6 plots): confusion matrix (counts + %), ROC curve (Youden's J optimal, multiclass one-vs-rest), precision-recall curve (per-class), feature importance, predicted probability distribution (per-class histogram), calibration curve
  - Regression (6 plots): actual vs predicted (R² annotated), residuals vs predicted (color-coded magnitude), residual histogram + normal + Shapiro-Wilk p, Q-Q plot + reference line, feature importance, scale-location homoscedasticity
  - `_auto_train()` now returns test split + predictions for diagnostics
  - Wired into: `dsw_from_intent`, `dsw_from_data`, `xgboost`, `lightgbm`
  - `model_compare` enhanced: precision/recall/F1 for classification, MAE for regression, multi-metric heatmap, training time bar chart
- `templates/workbench_new.html` — `renderMLOutput()` renders Plotly plots via stats-plots container

**Verification:** From Intent/Data/XGBoost/LightGBM → 6 diagnostic charts. Compare → heatmap + timing.

---

### 2026-02-13 — ML Lab Layer 5: Hyperparameter Tuning (Optuna)

**Context:** Auto-search for best hyperparameters — closes the last major ML gap vs JMP Pro.

**Files changed:**
- `agents_api/dsw_views.py` — New `hyperparameter_tune` analysis ID. Supports RF, XGBoost, LightGBM, Ridge, LASSO with predefined search spaces. Optuna study with configurable trials (10-50) and CV folds (3/5). 2-minute timeout. Output: best params, optimization history plot (trial scores + running best), parameter importance plot. Final model trained with best params and cached for saving.
- `templates/workbench_new.html` — "Tune" button (gear icon) in Auto ML group. `openTuneDialog()` with model type, task, trials, CV folds selectors.

**Package installed:** optuna 4.7.0

**Verification:** `/app/dsw/` → import CSV → Tune → select RF, 30 trials → optimization history + best params + parameter importance.

---

### 2026-02-13 — ML Lab Layer 4: SHAP Explainability

**Context:** SHAP (SHapley Additive exPlanations) gives transparent, per-feature explanations for any model.

**Files changed:**
- `agents_api/dsw_views.py` — New `shap_explain` analysis ID. TreeExplainer for tree models (RF, XGBoost, LightGBM), KernelExplainer fallback (capped at 100 background samples). Four Plotly charts: (1) Feature importance bar (mean |SHAP|), (2) Beeswarm plot (top 10 features, colored by feature value), (3) Waterfall for single prediction, (4) Dependence plot (top feature vs SHAP value). Works on any cached model from previous analysis.
- `templates/workbench_new.html` — "Explain" button in Auto ML group. `openShapDialog()` with model key display, target/features, global vs single mode, sample index selector.

**Package installed:** shap 0.49.1 (+ numba, llvmlite, slicer, cloudpickle)

**Verification:** `/app/dsw/` → train any model → Explain → beeswarm plot + feature importance + waterfall renders.

---

### 2026-02-13 — ML Lab Layer 3: XGBoost + LightGBM + GPU Training

**Context:** XGBoost and LightGBM are the industry standard for gradient boosting. GPU training uses the RTX 3090 with configurable allocation.

**Files changed:**
- `agents_api/gpu_manager.py` — **NEW**. Thread-safe GPU context manager with `_gpu_lock` mutex. Configurable via `SVEND_GPU_TRAINING_PERCENT` env var (default 50%). Falls back to CPU if GPU busy or unavailable. Returns model-specific params: `xgb_params()` (tree_method=hist, device=cuda) and `lgb_params()` (device=gpu).
- `agents_api/dsw_views.py` — Two new analysis IDs: `xgboost` and `lightgbm`. Both: auto-detect task type, encode categoricals, GPU-accelerated via GPUTrainingContext, exposed hyperparameters (n_estimators, max_depth/num_leaves, learning_rate, subsample), feature importance plots, actual-vs-predicted or confusion matrix, model caching. Also integrated into `model_compare` roster (auto-added when installed).
- `templates/workbench_new.html` — "XGBoost" and "LightGBM" buttons in Advanced ML group. `openGBMDialog()` with target/features/task/trees/depth/lr selectors.

**Packages installed:** xgboost 3.2.0, lightgbm 4.6.0

**Verification:** `/app/dsw/` → import CSV → XGBoost button → train. `nvidia-smi` should show GPU activity. Compare button now includes XGBoost + LightGBM in roster.

---

### 2026-02-13 — ML Lab Layer 2: Model Comparison + Cross-Validation

**Context:** JMP Pro's killer feature is running 6+ models and comparing with CV. Svend previously only trained one model at a time.

**Files changed:**
- `agents_api/dsw_views.py` — New `model_compare` analysis ID in the ML dispatch chain (after `regression_ml`, before `clustering`). Classification roster: RF, LogReg (Pipeline w/ Scaler), LDA, NaiveBayes. Regression roster: RF, Linear, Ridge, LASSO, ElasticNet, BayesianRidge. Auto-adds XGBoost/LightGBM if installed. Outputs: comparison table (CV mean ± std, train score, fit time), bar chart with error bars, ROC curves (binary classification), actual-vs-predicted overlay (regression). Best model auto-trained on full data and cached for saving. Gated by `can_use_ml()`.
- `templates/workbench_new.html` — "Compare" button added to Auto ML ribbon group. `openModelCompareDialog()` function with target, feature checkboxes, task type (auto/classification/regression), CV folds (3/5/10) selectors. Calls `model_compare` analysis via existing `run_analysis` dispatch.

**Verification:** `/app/dsw/` → import CSV → Compare button → select target + features → should show comparison table, bar chart, and ROC/actual-vs-predicted plots.

---

### 2026-02-13 — ML Lab Layer 1: Claude-powered From Intent + From Data

**Context:** Competitive analysis showed JMP Pro ($8,400/yr) as the primary ML gap. The existing `dsw_from_intent()` and `dsw_from_data()` both imported a non-existent `dsw` module and fell back to hardcoded mock results. Replaced with Claude-powered pipelines that generate real data, train real models, and produce AI interpretations.

**Files changed:**
- `accounts/constants.py` — `can_use_anthropic()` widened from Enterprise-only to all paid tiers. LLMManager already handles tier-based model selection (FOUNDER→haiku, PRO/TEAM→sonnet, ENTERPRISE→opus).
- `agents_api/dsw_views.py` — 5 new helper functions + 2 system prompts added before From Intent:
  - `_claude_generate_schema()` — Claude designs dataset schema from natural language
  - `_generate_data_from_schema()` — numpy/pandas synthetic data from schema
  - `_clean_for_ml()` — encode categoricals, handle NaN, split X/y
  - `_auto_train()` — auto-detect classification/regression, train RF, return metrics+importances
  - `_claude_interpret_results()` — Claude narrative of ML results
  - `dsw_from_intent()` rewritten: 4-step pipeline (schema→data→train→interpret), `@gated_paid` decorator
  - `dsw_from_data()` rewritten: upload CSV→clean→train→interpret, `@gated_paid` decorator, preserves `add_finding_to_problem` Synara integration
- `templates/workbench_new.html` — `renderMLOutput()` updated: AI Insight card for Claude interpretation, Plotly horizontal bar chart for feature importance, save model triggered by `model_key` (no longer requires `can_save` flag), shows task type and data shape details

**Verification:** `/app/dsw/` → From Intent: type "predict widget defects from temperature and pressure" → should get real schema, synthetic data, trained RF, Claude interpretation. From Data: upload CSV + target → real model + interpretation. Non-paid users → 403.

---

### 2026-02-13 — Whiteboard UX gaps: multi-select, copy/paste, resize, SVG export, image drop

**Context:** Competitive gap analysis identified 5 table-stakes UX features missing from the whiteboard. Connection cleanup on delete was also listed but was already implemented. All changes are client-side only in `whiteboard.html`.

**Files changed:**
- `templates/whiteboard.html` — 5 features added:
  1. **Multi-select**: `selectedElements` Set, Shift+Click additive toggle, marquee drag-box selection (dashed rect on empty canvas), multi-drag (delta-based, all selected move together), multi-delete (removes all selected + their connections), Ctrl+A select all.
  2. **Copy/paste**: Ctrl+C copies selected elements + inter-connections, Ctrl+V pastes with new IDs and +40,+40 offset, Ctrl+D duplicates in-place. Connections between copied elements are preserved.
  3. **Resize handles**: 4 corner handles (nw/ne/sw/se) on selected element. Supports post-its, shapes, groups, images. Diamond constrained to square, images lock aspect ratio. Minimum sizes enforced. Width/height stored in element data and restored on undo/load.
  4. **SVG export**: Client-side SVG generation (`exportBoardSVG()`). Renders all element types including images, connections with arrowheads, causal IF/THEN labels. New toolbar button next to PNG export.
  5. **Image drop/paste**: New `image` element type stored as base64 data URL. Drag-and-drop from desktop, Ctrl+V from clipboard. Auto-downscale to max 800px, JPEG compression. 5MB file / 2MB data URL cap. Included in PNG export (async image preload) and SVG export. Resize handles with aspect ratio lock.

**Verification:** Load /app/whiteboard/. Shift+click multiple elements, drag marquee. Ctrl+C/V. Drag image from desktop. Click SVG export. Resize handles on selected element.
**Commit:** pending

---

### 2026-02-13 — Whiteboard guest invite system

**Context:** Board owners need to share whiteboards with non-users (clients, stakeholders, contractors) without requiring a Svend account. Adds token-based guest access scoped to a single board with tier-based invite limits and owner-controlled permissions (view / edit / edit+vote).

**Files changed:**
- `agents_api/models.py` — Added `BoardGuestInvite` model (UUID PK, 64-char token, permission choices, expiry, presence fields). Made `BoardVote.user` nullable, added `guest_invite` FK, replaced `unique_together` with conditional `UniqueConstraint`s.
- `accounts/constants.py` — Added `GUEST_INVITE_LIMITS` (Free=0, Founder=2, Pro=5, Team=15, Enterprise=unlimited) and `GUEST_INVITE_EXPIRY_DAYS` (Founder/Pro=7 days, Team/Enterprise=permanent).
- `accounts/permissions.py` — Added `@allow_guest` decorator: checks `X-Guest-Token` header first, falls back to `@gated_paid` if no token.
- `agents_api/whiteboard_views.py` — Switched 6 endpoints to `@allow_guest` (get_board, update_board, update_cursor, add_vote, remove_vote, export_svg) with guest branching. Added 5 new views: `create_guest_invite`, `list_guest_invites`, `revoke_guest_invite`, `set_guest_name`, `guest_board_view`. Added `_build_participants_list()` helper.
- `agents_api/whiteboard_urls.py` — Added 4 guest API routes (list, create, revoke, set-name).
- `svend/urls.py` — Added guest page route `app/whiteboard/guest/<token>/`.
- `templates/base_guest.html` — New minimal base template (no nav/auth, just logo + Guest badge).
- `templates/guest_invalid.html` — New error page for invalid/expired/revoked tokens.
- `templates/whiteboard.html` — Variable extends for guest/normal base. Guest mode JS: fetch override for token injection, modified initCollaboration(), applyGuestMode(), name entry modal, invite management modal, Guests toolbar button for owners.
- `agents_api/migrations/0030_*.py` — Auto-generated migration for above model changes.

**Verification:** `manage.py check` passes. Migration applies cleanly. Guest page route serves HTML. API endpoints accept `X-Guest-Token` header. Owner sees Guests button in collaborative mode. Guest link opens board with minimal UI.
**Commit:** pending

---

### 2026-02-13 — Internal Dashboard improvements (4 tiers)

**Context:** After fixing the Internal Dashboard dropdown theming, performed a deep audit of the ~2149-line dashboard. Identified 13 improvements across 4 tiers. All implemented in this session.

**Tier 1 — Quick Wins:**
- Added "New Rule" form to Automation tab (name, trigger, config JSON, action, cooldown)
- Added toast notification system (CSS + JS `showToast()`) for error feedback — wired into KPI, draft, and campaign loaders
- Added feedback analytics summary: status counts + category Chart.js doughnut above the feedback table; extended `api_feedback` to return `summary` object
- Audited JS template literals for broken CSS variables — all clean

**Tier 2 — Marketing:**
- Added behavioral email segmentation: 14 segments (active:7d/30d, inactive:14d/30d, has_queries, no_queries, new:7d, domain:dsw/spc/doe, plus tiers) with `<optgroup>` organized dropdown and live "X users match" preview via new `api_email_preview` endpoint
- Added blog content calendar: month-view grid with colored dots (green=published, amber=scheduled, grey=draft) above the post list in the Content tab
- Added campaign-to-conversion attribution: counts recipients who upgraded within 7 days of campaign, shows "Conversions" column in campaign history table

**Tier 3 — Insights:**
- Added KPI anomaly alerts: week-over-week change computation in `api_overview`, colored arrow + percentage indicators (green up/red down if >10% change)
- Added cohort retention endpoint: `api_cohort_retention` returns monthly signup cohorts with retention percentages; rendered as heatmap table in Business tab
- Added churn risk flags: paid users inactive 14+ days shown in Users tab with severity-colored days-inactive column

**Tier 4 — Automation:**
- Added feedback-to-action workflow: Reply button (switches to Email tab with user pre-filled), Note button (saves internal annotations); added `internal_notes` field to Feedback model
- Added compound automation triggers: `trigger_2`, `trigger_2_config`, `trigger_logic` fields on AutomationRule; refactored `process_automations` to use `_evaluate_trigger` helper with AND/OR logic; updated rule form UI with optional 2nd trigger
- Added autopilot recommendation tracking: approved recommendations now store result object IDs; UI shows action status (Experiment created / Blog draft created / Email queued) instead of just checkmark

**Files changed:**
- `api/internal_views.py` — Added `_resolve_recipients()` helper, `api_email_preview()`, `api_cohort_retention()` endpoints. Extended `api_overview` with WoW changes, `api_users` with churn_risk, `api_feedback` with summary + internal_notes, `api_email_campaigns` with conversions, `api_automation_rules` with compound trigger fields, `api_autopilot_approve` with result tracking
- `templates/internal_dashboard.html` — Toast system, new rule form, feedback summary + doughnut, email segmentation dropdown + preview count, content calendar, campaign conversions column, KPI change indicators, retention heatmap, churn risk table, feedback action buttons, compound trigger UI, autopilot result display
- `api/models.py` — Added `Feedback.internal_notes`, `AutomationRule.trigger_2/trigger_2_config/trigger_logic`
- `api/tasks.py` — Refactored `process_automations` with `_evaluate_trigger()` helper for compound trigger evaluation
- `api/urls.py` — Added `email-preview/` and `cohort-retention/` routes
- `api/migrations/0008_feedback_internal_notes.py` — New field
- `api/migrations/0009_automation_compound_triggers.py` — New fields

**Verification:** Load /internal/ — all 10 tabs should work. KPI cards show WoW arrows. Email tab shows segments with user count. Content tab shows calendar. Business tab shows retention heatmap. Users tab shows churn risk. Feedback tab shows summary + Reply/Note buttons. Automation tab supports compound rules. Autopilot shows result status.
**Commit:** pending

---

### 2026-02-13 — Fix Internal Dashboard dropdown theming

**Context:** Dropdowns and form inputs in the Internal Dashboard were unreadable in dark/themed modes. Root cause: `var(--surface)` CSS variable doesn't exist in the Svend theme system, and `var(--bg-card)` was the wrong variable name (`--card-bg` is the actual name). Email dropdown options also had hardcoded dark-mode colors.

**Files changed:**
- `templates/internal_dashboard.html` — Replaced all `var(--surface)` → `var(--bg-tertiary)`, all `var(--bg-card)` → `var(--card-bg)`. Removed hardcoded `style="color:#e8efe8;background:#121a12;"` from email `<option>` elements. Added global `select, option` CSS rule to inherit theme colors.

**Verification:** Load /internal/ — all dropdowns (email To, experiment type/status, feedback filter, inline feedback status) should be readable in all 6 themes.
**Commit:** pending

---

### 2026-02-13 — Replace VSM emojis with Svend-style SVG icons

**Context:** Value Stream Map palette, canvas rendering, and landing page showcase used emoji characters (factory, people, truck, package, etc.) which render inconsistently across platforms. Replaced all with inline SVG stroke icons matching the Svend design system (24x24 viewBox, stroke-based, currentColor).

**Files changed:**
- `templates/vsm.html` — Replaced all 13 palette icon entities with inline SVGs. Added `VSM_ICONS` constant and `createSvgIcon()` helper. Updated `renderEntityBox()` to render SVG icons instead of emoji text. Updated inventory delay type icon rendering (queue/transport/batch) to use SVG icons.
- `templates/landing.html` — Replaced supplier (factory emoji) and customer (people emoji) in the VSM showcase SVG with inline SVG icon paths.

**Verification:** Load /app/vsm/ — palette icons should render as clean monochrome stroke icons. Create customer/supplier entities on canvas — icons should appear as SVG. Load landing page — VSM showcase should show stroke icons for supplier/customer. Check all 6 themes.
**Commit:** pending

---

### 2026-02-13 — Close DSW gaps: 7 new analyses + gap audit

**Context:** Gap analysis showed ~91% Minitab parity but was stale — audit found 18 items already implemented. After check-off, true coverage ~95%. Built 7 new analyses to close remaining visible holes.

**Files changed:**
- `DSW_gaps.md` — Checked off 18 already-implemented items (reliability, SPC, GLM, ordinal logistic, factor analysis, sign test, Mood's, Box-Cox, etc.). Updated scorecard to ~95%. Added Resolved section.
- `agents_api/dsw_views.py` — Added 7 new analysis blocks:
  - `individual_value_plot` (viz): jittered points by group with mean diamonds + CI bars
  - `interval_plot` (viz): group means with t-interval CI error bars + overall mean line
  - `dotplot` (viz): stacked dot display with optional grouping
  - `run_chart` (stats): time-ordered values + median line + runs tests (clustering/mixtures p-values)
  - `grubbs_test` (stats): formal single-outlier test with G statistic + critical value + highlight plot
  - `ccf` (stats): cross-correlation function with lag bars + significance bands
  - `johnson_transform` (stats): SB/SL/SU family fitting, before/after histograms, best family selection
- `templates/workbench_new.html` — Added 3 graph ribbon buttons (Ind Value, Interval, Dotplot) + Run Chart button in Prepare ribbon. Added dialogs: openIndValueDialog, openIntervalDialog, openDotplotDialog, openRunChartDialog. Added Grubbs + Johnson to Diagnostics dialog. Added CCF to All Tests time series group.

**Verification:** Upload CSV → Graph ribbon shows 12 chart types. Prepare ribbon has Run Chart. Diagnostics has Grubbs + Johnson. All Tests has CCF under Time Series.
**Commit:** pending

---

### 2026-02-13 — Graphical Summary + Auto-Profile on Import

**Context:** DSW had 129 analysis types but no automatic data exploration on import. Users upload data and see a raw worksheet. Added Minitab-style Graphical Summary and auto-profiling on import.

**Files changed:**
- `agents_api/dsw_views.py` — Added `auto_profile` analysis (lightweight overview: column stats, correlation heatmap, distribution histograms for up to 12 numeric columns, missing data bar chart). Added `graphical_summary` analysis (Minitab-style per-variable view: histogram + normal PDF overlay, boxplot, CI bars for mean/median, Anderson-Darling normality test, full descriptive stats, CIs for mean/median/StDev using scipy.stats)
- `templates/workbench_new.html` — Added `autoRunProfile()` function, called after triage scan (clean data) and after triage fix (committed data). Added "Graphical" button in Prepare ribbon with column-select dialog + confidence level input. Added `openGraphicalSummaryDialog()` function.

**Verification:** Upload CSV → "Data Overview" tab appears automatically. Click Graphical → select columns → histogram + normal curve, boxplot, CI bars, Anderson-Darling test, full stats.
**Commit:** pending

---

### 2026-02-13 — Fix triage auto-fix silently dropping data

**Context:** User reported triage auto-fix dropping data entirely on import. Three root causes found:

1. **Response format mismatch (primary):** `workbench_new.html` (served at `/app/dsw/`) expects `data.cleaned_csv` in the triage response to re-upload as a new dataset. The `triage_data()` endpoint never included this field. So `data.cleaned_csv` was `undefined`, the frontend created a CSV blob containing the literal string "undefined", re-uploaded it, and the dataset became empty — 0 usable rows.
2. **DSW triage `drop_rows` was nuclear:** `df.dropna()` drops every row with even ONE missing value. With real-world data (e.g. 70% missing in a notes column), this destroyed 91/100 rows.
3. **Standalone triage (separate code path):** `triage_views.py` + `scrub` module at `/api/triage/clean/` had its own bug where `DataCleaner` ignored config's `drop_threshold`, auto-dropping columns >50% missing.

**Files changed:**
- `agents_api/dsw_views.py` — Added `cleaned_csv` string to response (the actual fix); rewrote `drop_rows` to only drop rows >80% empty then impute the rest; type conversion no longer silently creates NaN; added `warnings` and `cols_removed` to response
- `templates/analysis_workbench.html` — Shows warnings in session output; `drop_rows` label updated
- `templates/workbench_new.html` — Shows warnings in toast; `drop_rows` label updated
- `services/scrub/missing.py` — Removed auto-DROP from `analyze()`; added transparency fields to MissingResult; fixed median/mean for Int64 columns
- `services/scrub/cleaner.py` — Passes `config.drop_threshold` to MissingHandler; surfaces warnings
- `agents_api/triage_views.py` — Response includes `columns_dropped`, `rows_dropped`
- `templates/triage.html` — 6-stat summary, "Changes Made" section, delta indicators

**Verification:** Upload water potability CSV (3276 rows, 3 columns with missing) → all rows preserved, missing imputed
**Commit:** pending

---

### 2026-02-13 — Security hardening: field-level encryption, infrastructure, privacy policy

**Files changed:**
- `.env`, `.env.production` — chmod 600 (was world-readable 644)
- `/home/eric/.svend_encryption_key` — New Fernet key file (chmod 600)
- `pyproject.toml` — Added `cryptography>=41.0` dependency
- `Caddyfile` — Added HSTS (2yr, preload), CSP, Permissions-Policy headers; fixed stale static path
- `svend/settings.py` — Removed BasicAuthentication from REST_FRAMEWORK; added SECURE_HSTS_*, SECURE_SSL_REDIRECT, SECURE_PROXY_SSL_HEADER, FIELD_ENCRYPTION_KEY
- `svend_config/config.py` — Added `field_encryption_key` setting
- `start_prod.sh` — Loads encryption key from keyfile before gunicorn start
- `core/encryption.py` — New: EncryptedTextField, EncryptedCharField, EncryptedJSONField, encrypt/decrypt helpers, hash_token()
- `core/encrypted_storage.py` — New: EncryptedFileSystemStorage (encrypts files on disk)
- `chat/models.py` — Message.content, reasoning_trace, tool_calls; TraceLog.input_text, reasoning_trace, tool_calls, reasoner_raw_output, lm_prompt, lm_raw_output, response; TrainingCandidate.input_text, reasoning_trace, model_response, corrected_response → encrypted fields
- `accounts/models.py` — stripe_customer_id → EncryptedCharField + stripe_customer_id_hash (SHA-256, indexed); verification token → stored as SHA-256 hash; generate_verification_token() returns plaintext but stores hash; verify_email() compares hashes
- `accounts/billing.py` — Stripe customer lookup uses hash column; populates hash on create
- `agents_api/models.py` — DSWResult.data, TriageResult.cleaned_csv/report_markdown/summary_json → EncryptedTextField
- `files/models.py` — UserFile.file uses EncryptedFileSystemStorage
- `chat/migrations/0005_*`, `0006_encrypt_existing_data.py` — Schema + data migration
- `accounts/migrations/0009_*`, `0010_encrypt_existing_data.py` — Schema + data migration (hash column + encryption)
- `agents_api/migrations/0028_*`, `0029_encrypt_existing_data.py` — Schema + data migration
- `files/migrations/0002_*` — Storage backend migration
- `files/management/commands/encrypt_existing_files.py` — New: one-time command to encrypt files on disk
- `forge/management/commands/purge_old_data.py` — Extended with retention policy: TraceLog 30d, AgentLog 30d, TrainingCandidate 30/7d, EventLog 90d, SharedConversation expired, BlogView 180d
- `backup_db.sh` — New: pg_dump | gzip | AES-256 encrypted backups with 30d retention
- `svend-backup.service` + `svend-backup.timer` — New: systemd daily backup at 03:00
- `templates/privacy.html` — Full rewrite: 10 sections covering data collection, encryption at rest, retention schedule, third-party services, rights, cookies, email

**Verification:**
1. Raw DB query shows Fernet ciphertext (gAAAAA...) for stripe_customer_id, dsw_result.data — confirmed
2. ORM reads decrypt transparently — confirmed
3. `curl -I https://svend.ai` should show HSTS, CSP, Permissions-Policy headers
4. `python manage.py encrypt_existing_files --dry-run` shows 0 unencrypted files
5. `python manage.py purge_old_data --dry-run` runs without error
6. Privacy policy at /privacy/ reflects all actual data practices

---

### 2026-02-13 — DSW session save/load + model save/load + UX fixes

**Files changed:**
- `templates/base_app.html` — Logo href changed from `/` to `/app/` for logged-in users. Internal nav checks `is_internal` instead of `is_staff`.
- `api/internal_views.py` — Added `INTERNAL_TENANT_SLUGS`, `can_access_internal()`, `IsInternalUser` permission class. Replaced `IsAdminUser` with `IsInternalUser` on all internal endpoints.
- `api/views.py` — Added `is_internal` field to `/api/auth/me/` response.
- `templates/workbench_new.html` — Added Save/Load Model buttons to ML ribbon. Added `openSaveModelDialog()`, `openLoadModelDialog()`, `saveModel()`, `appendSaveModelPrompt()`. Renamed top-right button to "Save Session". Added `collectSessionState()`, `collectOutputTabs()`. Rewrote `saveWorkbench()` to persist full session (output tabs, data ref, cache key). Updated `loadWorkbench()` to restore saved session state (output tabs, data panel, variables). Removed redundant `apiPost` in `openLoadModelDialog`. Both `renderStatsOutput` and `renderMLOutput` now track `lastModelKey` and show inline save prompts.
- `workbench/views.py` — `update_workbench` accepts `datasets` and `guide_observations` on PATCH.
- `workbench/models.py` — Added `layout` to `to_json()` serialization.
**Verification:** Save a DSW session with data + analysis output, reload from list — output tabs and data panel should restore. Save/Load model from ML ribbon should work.

---

### 2026-02-13 — Exclude rtWzrd from internal analytics

**Files changed:**
- `api/internal_views.py` — Added `INTERNAL_USERNAMES = {"rtWzrd"}` constant. Updated `_customers()` and `_staff_ids()` helpers to exclude internal accounts. Added `.exclude(user__username__in=INTERNAL_USERNAMES)` to all 10 direct queryset filters (UsageLog, EventLog, Subscription, OnboardingSurvey, OnboardingEmail, TraceLog via `_staff_ids`).
**Verification:** Check internal dashboard — rtWzrd should not appear in user counts, tier distribution, MRR, usage, activity, or onboarding metrics.

---

### 2026-02-13 — Remove invite code requirement from all account types

**Files changed:**
- `svend/settings.py` — Removed `InviteRequiredMiddleware` from MIDDLEWARE, removed `REQUIRE_INVITE` setting
- `svend_config/config.py` — Removed `require_invite` config field
- `accounts/middleware.py` — Removed `InviteRequiredMiddleware` class
- `api/views.py` — Removed invite code validation, import, and usage from `register()` view
- `templates/register.html` — Removed invite code form field and related CSS/JS
- `templates/chat.html` — Removed invite code input, formatting JS, validation, and CSS; updated copy
**Note:** `InviteCode` model, admin, migrations, and management command kept for historical records.
**Verification:** Register a new free account without an invite code — should succeed.

---

### 2026-02-13 — Unified Action Items: connect A3, RCA, FMEA to ActionItem model

**Files changed:**
- `agents_api/models.py` — Added `source_type` (CharField, max_length=20) and `source_id` (UUIDField, nullable) to ActionItem. Updated `to_dict()` to include both fields. Allows tracking where each action item originated (hoshin, a3, rca, fmea, report).
- `agents_api/migrations/0027_action_item_source.py` — Migration for the two new fields.
- `agents_api/a3_views.py` — Added `list_a3_actions` (GET) and `create_a3_action` (POST) endpoints. Updated `get_a3_report` to include `action_items` in response.
- `agents_api/a3_urls.py` — Added routes for `<report_id>/actions/` and `<report_id>/actions/create/`.
- `agents_api/rca_views.py` — Added `list_rca_actions` (GET) and `create_rca_action` (POST) endpoints. Updated `get_session` to include `action_items` in response.
- `agents_api/rca_urls.py` — Added routes for `sessions/<session_id>/actions/` and `sessions/<session_id>/actions/create/`.
- `agents_api/fmea_views.py` — Added `list_fmea_actions` (GET) and `promote_fmea_action` (POST, idempotent) endpoints. Updated `get_fmea` to include `action_items` in response.
- `agents_api/fmea_urls.py` — Added routes for `<fmea_id>/actions/` and `<fmea_id>/rows/<row_id>/promote-action/`.
- `agents_api/hoshin_views.py` — Updated `create_action_item` to set `source_type="hoshin"` and `source_id=hoshin.id`.
- `agents_api/action_views.py` (new) — Shared `update_action_item` (PUT/PATCH) and `delete_action_item` (DELETE) views. Auth: project__user must match request.user.
- `agents_api/action_urls.py` (new) — Routes for `<action_id>/update/` and `<action_id>/delete/`.
- `svend/urls.py` — Mounted `api/actions/` → `agents_api.action_urls`.
- `templates/a3.html` — Added Action Items panel below A3 paper with add form, status cycling (click to cycle not_started→in_progress→completed), delete, CSS styling.
- `templates/rca.html` — Added Action Items section below countermeasure field with add form, status cycling, delete, CSS styling.
- `templates/fmea.html` — Added "Track" button on FMEA rows (promotes recommended_action to ActionItem), "Tracked" badge for already-promoted rows, action items summary panel with status dropdown.
- `templates/hoshin.html` — Added Source column to action items table with color-coded badges (hoshin=green, a3=blue, rca=yellow, fmea=purple, report=pink).
**Verification:** `python3 manage.py check` passes. Migration 0027 applied. Create action items from A3/RCA/FMEA → they appear in the respective tool AND on the Hoshin dashboard with source badges.

---

### 2026-02-12 — Operations Workbench: semi-smart guide widget

**Files changed:**
- `services/svend/web/templates/calculators.html` — Added collapsible guide widget system. Single `#calc-guide` container dynamically populated by `renderGuide(id)` on calculator switch. `calcGuide` data for 13 calculators (Takt, RTO, OEE, Bottleneck, SMED, EPEI, Kanban, EOQ, Little's Law, Pitch, MTBF, Erlang C, Risk Matrix). `buildSmartPrompts()` reads SvendOps bus to suggest pull actions. Gracefully hides for unconfigured calculators.

**Verification:** Open Operations, expand Guide on Takt. Run Takt calc, switch to RTO — should show "Takt Time (Xs) available — pull it in" prompt.

---

### 2026-02-12 — Operations Workbench: search bar + 3 new calculators + nav cleanup

**Files changed:**
- `services/svend/web/templates/calculators.html` — Added reactive search bar to left sidebar nav (filters by name, description, and ID). Added three new calculators: MTBF/MTTR + Availability, Erlang C Staffing, Risk Matrix 5x5. Placed under existing nav groups (Line Performance, Queuing Lab, Risk & Quality) instead of creating new groups.

**Verification:** Navigate to Operations. Search bar filters as you type. New calculators in Line Performance, Queuing Lab, and Risk & Quality groups.

---

### 2026-02-13 — Codebase cleanup: deduplicate agents, remove dead code, consolidate sys.path

**Files removed (git rm):**
- `services/svend/data/` — reasoning engine training data (tokenizer, datasets, seeds, generated JSONL)
- `services/svend/shared_context/` — reasoning context sharing (problem JSON files)
- `services/svend/agents/{coder,researcher,writer,reviewer,guide,experimenter,analyst}/` — duplicate copies (canonical is agents/agents/)
- `services/svend/agents/agents/core/` — duplicate of root core/ shared library
- `services/svend/{docs,tools,dsw,workflow}/` — duplicates of agents/agents/{docs,tools,dsw,workflow}/
- `lab/` — empty synara placeholder
**Directories cleaned (rm -rf, pycache-only):**
- `services/svend/{server,models,pipeline,evaluation,inference_tools}/` — hollow dirs from prior reasoning engine removal
- 66 `__pycache__/` directories repo-wide
**Files modified:**
- `services/svend/web/svend/settings.py` — centralized agent sys.path setup (one insert for agents/agents/)
- `agents_api/views.py` — removed 30-line importlib bootstrap hack for agents/agents/core/ shadowing
- `agents_api/experimenter_views.py` — removed sys.path.insert
- `agents_api/problem_views.py` — removed 2 sys.path.insert calls
- `agents_api/workflow_views.py` — removed 7 sys.path.insert calls
- `agents_api/dsw_views.py` — removed 5 sys.path.insert calls (including Desktop/agents and services/ refs)
- `agents_api/triage_views.py` — removed 2 sys.path.insert calls
**Verification:** `python3 manage.py check` passes. `grep -r sys.path.insert services/svend/web/` returns only settings.py + one test. Zero references to Desktop/agents in Django views.

---

### 2026-02-13 — Remove stale reasoning engine code and docs

**Files removed:**
- `services/svend/server/` (5 files) — Standalone FastAPI reasoning API server, never imported by Django
- `services/svend/models/` (6 files) — Custom transformer architecture and training code (RoPE, GQA, SwiGLU, distillation)
- `services/svend/pipeline/` (5 files) — Training pipeline orchestration (runner, config, checkpoints, validation)
- `services/svend/evaluation/` (8 files) — Evaluation harness and benchmarking (adversarial, scaling diagnostics)
- `services/svend/inference_tools/` (24 files) — Tool-calling infrastructure for reasoning models (35 domain solvers)
- `services/svend/README.md` — Described "training custom reasoning models from scratch"
- 10 stale reference docs: CLAUDE.md, PIPELINE_GUIDE.md, experiment_001_reasoning.md, RETRAINING_STRATEGY.md, SAFETY_EVAL_GUIDE.md, TOOL_SPEC_V0.md, TOOL_TRACE_GENERATION.md, TRAINING_STRATEGY.md, BETA_DEPLOYMENT.md, SVEND_PRODUCTION_PLAN.md, README.md
**Kept:** `core/reasoning.py` (CodeReasoner used by agents), `agents/agents/CLAUDE.md` (current agent docs), reference_docs/ARCHITECTURE.md, SYNARA_WHITEPAPER.md, DATA_SOURCES.md, POLICIES.md, ROADMAP.md, LAUNCH_PLAN.md
**Verification:** `grep -r "from.*server\|from.*pipeline\|from.*inference_tools\|from.*evaluation.harness" services/svend/web/` returns nothing — zero production imports.

---

### 2026-02-13 — Whiteboard: snap-to-grid, VSM removal, diamond connection points

**Files changed:**
- `templates/whiteboard.html` — Three changes:
  1. **Snap-to-grid:** Added `GRID_SIZE = 20` and `snapToGrid()` helper. Applied to all 4 element placement paths (click postit, click text, drag move, drag-and-drop from sidebar). Grid size matches the visual background pattern.
  2. **VSM removal:** Removed `vsm-process` and `vsm-inventory` element types from `createElement()`, removed ~40 lines of VSM CSS, removed VSM case from `loadTemplate()`. Replaced VSM template button with link to `/app/vsm/` since VSM is now its own standalone module.
  3. **Diamond connection points at vertices:** Repositioned CSS connection point dots from edge midpoints to the corners of the unrotated 80×80 box (which become the visual vertices after 45deg rotation). Restored `cornerExt = 0.21` in `getConnectionPoint()` JS so SVG connector paths reach the actual diamond tips instead of the bounding box edges.
**Verification:** Open whiteboard, drag elements — they snap to 20px grid. Diamond shapes show connection dots at all 4 pointy tips. Connectors attach at diamond vertices. VSM shapes no longer in sidebar.

---

### 2026-02-13 — Hoshin Kanri: site access control, isolation fixes, calendar view

**Files changed:**
- `agents_api/models.py` — Added `SiteAccess` model (viewer/member/admin roles per site). Standalone join table between User and Site with `granted_by` tracking and unique constraint on (site, user).
- `agents_api/migrations/0026_site_access.py` — Migration for SiteAccess table (`hoshin_site_access`).
- `agents_api/hoshin_views.py` — Major changes:
  - Added 4 access helpers: `_get_accessible_sites()`, `_check_site_read()`, `_check_site_write()`, `_is_site_admin()`
  - **Security fix:** `update_action_item` and `delete_action_item` now authorize via query (tenant in filter) instead of load-then-check
  - **Security fix:** `create_from_proposals` validates VSM tenant ownership before use
  - Applied site access filtering to all 12+ views — org admins see all sites, others only see sites they have SiteAccess entries for
  - Added 3 site member management endpoints: `list_site_members`, `grant_site_access`, `revoke_site_access`
  - Added `hoshin_calendar_view` endpoint — returns projects grouped by site with monthly target/actual/pct data
- `agents_api/hoshin_urls.py` — Added routes for site member management (`/members/`, `/members/grant/`, `/members/<id>/revoke/`) and calendar (`/calendar/`)
- `templates/hoshin.html` — Added Calendar tab (projects-by-month grid with color-coded performance bars), site member management UI on Sites tab (add/remove members, role assignment), CSS for both features
**Verification:** Django system check passes. Migration 0026 applied. Navigate to Hoshin → Calendar tab shows project grid. Sites tab shows member management for admins. Non-admin users only see sites they're assigned to.

---

### 2026-02-12 — Update CLAUDE.md, delete stale services/svend/CLAUDE.md

**Files changed:**
- `CLAUDE.md` — Full rewrite. Added: all 9 Django apps to architecture tree, 25-row API surface table (was 10), multi-tenancy section, KnowledgeGraph model, frontend section, new libraries (WhiteNoise, Stripe). Updated: dual-write Phase 2 status, researcher agent re-enabled, coder still disabled. Added pointer to ARCHITECTURE.md.
- `services/svend/CLAUDE.md` — **Deleted.** Described a stale custom transformer training project (reasoning-lab/) that no longer matches the product. The root CLAUDE.md is now the single source of truth.
**Verification:** Read CLAUDE.md, confirm architecture tree lists all apps (core, agents_api, api, chat, workbench, forge, files, tempora, accounts). Confirm services/svend/CLAUDE.md no longer exists.

---

### 2026-02-12 — Remove duplicate email feedback from settings

**Files changed:**
- `templates/settings.html` — Removed email feedback section (HTML + CSS). The global "?" button in base_app.html is the single feedback path now.
**Verification:** Visit /settings, confirm no "Send Feedback" / email section. Confirm "?" button still works on all pages.

---

### 2026-02-12 — Unsubscribe + In-App Feedback System

**Files changed:**
- `accounts/models.py` — Added `email_opted_out` field
- `accounts/migrations/0008_email_opt_out.py` — Migration
- `api/models.py` — Added `Feedback` model (category, message, page, status)
- `api/migrations/0007_feedback.py` — Migration
- `api/views.py` — `email_unsubscribe` endpoint (signed token), `make_unsubscribe_url()`, `submit_feedback` endpoint
- `api/internal_views.py` — EMAIL_TEMPLATE footer now includes unsubscribe link, `api_send_email` skips opted-out users, `api_feedback` staff endpoint
- `api/tasks.py` — All email senders check `email_opted_out`, all automation queries filter `email_opted_out=False`, feedback fed into Claude autopilot data
- `api/urls.py` — Routes for unsubscribe, feedback (public + staff)
- `templates/base_app.html` — Floating feedback button + modal on all app pages
- `templates/internal_dashboard.html` — Feedback section in Automation tab with status management

**Verification:**
- Every email footer has "Unsubscribe" link with signed token
- Clicking unsubscribe shows confirmation page and sets `email_opted_out=True`
- Opted-out users are skipped by manual sends, onboarding drip, and automation rules
- Feedback button appears on all app pages (bottom-right "?")
- Feedback shows in dashboard Automation tab with category, message, status dropdown
- Claude autopilot weekly review includes user feedback in its data

---

### 2026-02-12 — Solo Founder Growth Engine: A/B Testing, Automation & Claude Autopilot

**Files changed:**
- `services/svend/web/api/models.py` — Added 5 models: Experiment, ExperimentAssignment, AutomationRule, AutomationLog, AutopilotReport
- `services/svend/web/api/migrations/0006_automation_framework.py` — Migration for new models
- `services/svend/web/api/experiments.py` — New file: experiment engine (assign_variant, record_conversion, evaluate_experiment, get_variant)
- `services/svend/web/api/tasks.py` — 7 lifecycle email templates + 3 Tempora tasks (process_automations every 30min, evaluate_experiments daily, claude_growth_review weekly)
- `services/svend/web/api/apps.py` — Register 3 new schedules + seed 7 default automation rules
- `services/svend/web/api/internal_views.py` — 8 new endpoints (experiments CRUD, rules list/toggle, automation log, autopilot reports/approve/run) + email A/B test integration
- `services/svend/web/api/urls.py` — 9 new routes for automation endpoints
- `services/svend/web/templates/internal_dashboard.html` — Automation tab with 4 sections: Experiments, Rules, Autopilot, Log

**Verification:**
- Dashboard Automation tab shows 7 seeded rules with on/off toggles
- New Experiment form creates experiments, evaluate/conclude buttons work
- Rules can be toggled on/off
- Claude Autopilot "Run Growth Review" schedules a Tempora task
- Automation log shows fired events

---

### 2026-02-12 — Standardize calculator integration buttons

**Files changed:** `services/svend/web/templates/calculators.html`
**What:** Unified two inconsistent pull button styles into one standard `.calc-pull-btn` class:
- Replaced 11 icon-only `.calc-link-btn` buttons (24×24, hard to discover) with labeled buttons showing source: "← Takt", "← SMED", "← Bottleneck", etc.
- Replaced 5 inline-styled text buttons ("Pull from Line Sim", etc.) with `.calc-pull-btn.standalone` class
- Added visual feedback: green flash on target input when data is pulled (via `SvendOps.pull()`)
- Added toast notification when pull fails (no data available)
- Added 3 new pull buttons: OEE ideal CT ← Bottleneck, Queue Priority μ ← Bottleneck, Line Sim changeover ← SMED
- Kept `.calc-dsw-pull` unchanged (separate system for DSW external data)
**Total:** 19 pull buttons, all using consistent `.calc-pull-btn` class
**Verify:** Load any calculator with pull buttons — should show "← Source" text labels, green pulse when data available

---

### 2026-02-12 — Remove alpha access badge, update docs to reflect live production status

**Debt item:** N/A
**Files changed:**
- `templates/landing.html` — Removed "Alpha Access" hero badge and its CSS
- `CLAUDE.md` — "launching May 2026" → "live in production"
- `services/svend/CLAUDE.md` — Updated status from "Target launch: May 2026" to "Live in production at svend.ai"
- `services/svend/agents/agents/CLAUDE.md` — "Alpha Notes / alpha release" → "Production Notes"
- `services/svend/reference_docs/CLAUDE.md` — "launching May 2026" → "live in production"
- `services/svend/reference_docs/ROADMAP.md` — "Target launch: May 2026" → "Launched February 2026, live in production"
**Verification:** Visit svend.ai — no alpha badge on hero. Grep for "alpha" in CLAUDE.md files returns no hits.
**Commit:** pending

---

### 2026-02-12 — Hoshin: remove duplicate custom card + add {{fieldname}} extraction for custom formulas

**Debt item:** N/A (feature)
**Files changed:**
- `agents_api/hoshin_calculations.py` — Added `extract_formula_fields()` and `normalize_formula()`. Updated `_custom()` to merge arbitrary `custom_vars` dict into eval variables and strip `{{}}` before AST evaluation.
- `agents_api/hoshin_views.py` — Updated `test_formula` to return extracted fields and handle `{{}}` syntax. Updated `update_monthly_actual` to accept and store `custom_vars` dict on monthly entries, passed through to calculation.
- `templates/hoshin.html` — Removed duplicate custom card from calc library (filtered `custom` from API-sourced cards, kept purple hardcoded card). Updated purple card to document `{{fieldname}}` syntax. Formula tester dynamically generates inputs from `{{}}` fields. Monthly data entry (both calc tab and overview modal) shows custom field inputs when formula uses `{{}}` syntax. Added `extractFormulaFields()` and `updateFormulaFields()` JS helpers.
**Verification:** Open Hoshin > Calc Library: only one custom card (purple, dashed border). Create a project with custom formula using `{{field}}` syntax — monthly data entry should show named inputs instead of baseline/actual/volume/cost. Formula tester should auto-generate inputs when typing `{{fields}}`.
**Commit:** pending

---

### 2026-02-12 — Landing page: replace chat demo with live simulator, Cpk study, and VSM showcases

**Debt item:** N/A (marketing)
**Files changed:**
- `templates/landing.html` — Removed chat bubble demo. Added 3-panel showcase carousel: (1) live line simulator with animated WIP flow, throughput tracking, bottleneck highlighting, and utilization; (2) static Cpk study with histogram, spec limits, and capability stats; (3) inline SVG VSM matching actual VSM tool rendering (process boxes with green headers, yellow inventory triangles, blue entity boxes, material flow arrows, info flow dashed line, kaizen burst, lead time ladder). Auto-rotates every 12s. Cleaned up dead CSS from old HTML-based VSM approach. All pure HTML/CSS/JS in Svend colors.
**Verification:** Visit svend.ai — simulator should be running live, tabs switch between Simulator/Cpk/VSM. VSM panel should show proper SVG with process boxes, arrows, and timeline matching the real tool.
**Commit:** pending

---

### 2026-02-12 — DOE ANOVA audit: fix JSON serialization + saturated model handling + session persistence

**Debt item:** N/A (bug fix)
**Files changed:**
- `agents_api/experimenter_views.py` — Fixed 3 critical bugs in `analyze_results()`:
  1. **numpy.bool_ not JSON serializable**: scipy returns numpy types that Django's JsonResponse can't serialize. Added `_sanitize()` helper, wrapped all response data. Cast all numpy types to Python natives.
  2. **Saturated model crash**: 2-factor full factorial with interactions (n=p=4) produced `float('inf')` t-stats and `nan` p-values → invalid JSON (`Infinity`/`NaN`). Now returns `null` for untestable values with `saturated: true` flag. Added interpretation explaining why p-values are unavailable and recommending replicates/center points.
  3. **Anderson-Darling skipped for saturated**: Residuals are all ~0 in saturated models, skip AD normality test.
- `templates/experimenter.html` — Fixed 3 issues:
  1. **Session persistence**: Added `saveState()`/`restoreState()` using `sessionStorage`. Design, analysis, and entered response values survive page refresh. Previously, any page refresh lost `currentDesign` and showed "Generate a design first".
  2. **showSubTab crash**: `event.target.classList.add('active')` used implicit `event` which doesn't exist when called programmatically. Now finds button by `onclick` attribute.
  3. **Saturated model UI**: Shows warning banner when model is saturated. Coefficient table displays "-" for null p-values/t-values instead of "undefined".
**Verification:** Create 2-factor full factorial design, enter responses, click Analyze → ANOVA table renders with null p-values and saturated warning. Refresh page → design and data persist.
**Commit:** pending

---

### 2026-02-12 — Email campaign tracking: sent/opened/clicked traceability + draft save/reset

**Debt item:** N/A (email feature)
**Files changed:**
- `api/models.py` — Added `EmailCampaign` (subject, body, target, sent_by) and `EmailRecipient` (campaign FK, user FK, email, sent_at, opened_at, clicked_at, failed) models
- `api/internal_views.py` — Rewrote `api_send_email` to create campaign records with tracking pixel and link rewriting; added `api_save_email_draft`, `api_get_email_draft`, `api_email_campaigns` endpoints
- `api/views.py` — Added `email_track_open` (1x1 GIF pixel) and `email_track_click` (redirect with timestamp) public endpoints
- `api/urls.py` — Added routes: email-draft/save/, email-draft/, email-campaigns/, email/open/<id>/, email/click/<id>/
- `templates/internal_dashboard.html` — Added darker dropdown text, Save Draft/Reset buttons, Campaign History table with sent/opened/clicked/open-rate columns, JS functions (saveEmailDraft, resetEmailForm, loadEmailDraft, loadEmailCampaigns, loadEmail)
- `api/migrations/0005_add_email_campaign_tracking.py` — Migration for email_campaigns and email_recipients tables
**Verification:** Email tab loads saved drafts, Save/Reset buttons work, sending creates campaign records, Campaign History table shows sent/opened/clicked stats
**Commit:** pending

---

### 2026-02-12 — Blog analytics: view tracking with referrer/source data + dashboard charts

**Debt item:** N/A (analytics feature)
**Files changed:**
- `api/models.py` — Added `BlogView` model (post FK, referrer, referrer_domain, ip_hash, user_agent, is_bot)
- `api/blog_views.py` — Added `_record_view()` to log each blog detail page hit with referrer, hashed IP, bot detection
- `api/internal_views.py` — Added `api_blog_analytics` endpoint: daily views, top posts, referrer domains, traffic source split
- `api/urls.py` — Added `/api/internal/blog/analytics/` route
- `templates/internal_dashboard.html` — Added blog analytics section to Content tab: totals, views-over-time line chart, top posts bar, traffic sources doughnut, referrer domains bar
- `api/migrations/0004_blog_view_analytics.py` — Migration for `blog_views` table
**Verification:** Visit a blog post, then check Content tab in internal dashboard — analytics charts should appear
**Commit:** pending

---

### 2026-02-12 — Onboarding system with survey, personalized email drip, and dashboard analytics

**Debt item:** N/A (growth feature)
**Files changed:**
- `accounts/models.py` — Added `onboarding_completed_at` DateTimeField to User model
- `api/models.py` — Created `OnboardingSurvey` (demographics, goals, self-assessment, learning path) and `OnboardingEmail` (drip email tracking) models
- `api/views.py` — Added `onboarding_status` (GET) and `onboarding_complete` (POST) endpoints; added `onboarding_completed` to `me()` response
- `api/tasks.py` — Added 5 personalized email builders (welcome, getting_started, tips, learning_path, checkin) with content tailored by survey responses (goal, confidence level, learning path); added `send_onboarding_email` and `process_onboarding_drip` Tempora tasks
- `api/apps.py` — Registered `process_onboarding_drip` recurring schedule (every 10 minutes via Tempora)
- `api/urls.py` — Added onboarding API routes and internal onboarding analytics route
- `api/internal_views.py` — Added `api_onboarding` endpoint (funnel, survey distributions, email stats, challenges, completion over time)
- `templates/onboarding.html` — New multi-step survey page (4 steps: About You, Goals, Self-Assessment, Completion) with progress bar, chip selectors, slider inputs, learning path assignment
- `templates/register.html` — Updated redirect to `/app/onboarding/` for new free signups
- `templates/internal_dashboard.html` — Added Onboarding tab with funnel chart, learning path distribution, goal/experience/industry/role/tools charts, email stats, completion timeline, challenges feed
- `svend/urls.py` — Added `/app/onboarding/` route
- `accounts/migrations/0007_add_onboarding_completed_at.py` — Applied
- `api/migrations/0003_add_onboarding_models.py` — Applied
**Verification:**
1. New signup → redirected to `/app/onboarding/` → 4-step survey → completion screen → `/app/`
2. Survey syncs demographics to User profile + computes learning path
3. Welcome email fires immediately via Tempora; drip emails at 1h, 24h, 3d, 7d
4. Email content personalized by goal, confidence level, and learning path
5. Internal dashboard Onboarding tab shows funnel, distributions, email stats
6. `python manage.py check` — clean

---

### 2026-02-12 — Blog charts + scheduled publishing

**Debt item:** N/A (content feature)
**Files changed:**
- `api/models.py` — Added `scheduled_at` DateTimeField and `SCHEDULED` status to BlogPost
- `api/migrations/0002_blogpost_scheduled_at.py` — Applied
- `api/tasks.py` — **CREATED** Tempora task `api.publish_scheduled_posts` — checks for due scheduled posts every 15min and publishes them
- `api/apps.py` — Added `ready()` hook to register Tempora tasks and create recurring schedule (idempotent)
- `api/management/commands/publish_scheduled.py` — **CREATED** Fallback management command for manual publish
- `api/internal_views.py` — Updated blog endpoints: list returns `scheduled_at`/`scheduled` counts, get returns `scheduled_at`, publish supports `action: "schedule"` with datetime
- `templates/internal_dashboard.html` — Content tab: added datetime picker + Schedule/Unschedule button, "Insert Chart" button for markdown editor, status badges show scheduled date
- `templates/blog_detail.html` — Added Chart.js + custom marked.js renderer: ` ```chart ` fenced code blocks render as interactive Chart.js charts with dark theme, auto-colored datasets, and optional captions
- `templates/base_app.html` — Added marked.js CDN for dashboard markdown preview
**Verification:** Content tab → write post with ` ```chart ` block → preview renders chart. Schedule for future date → status shows "scheduled". Tempora publishes it when due.

---

### 2026-02-12 — Blog + SEO + Content Generator

**Debt item:** N/A (marketing/SEO feature)
**Files changed:**
- `api/models.py` — **CREATED** BlogPost model (title, slug, body markdown, meta_description, status draft/published, author FK, timestamps). Auto-slug generation with uniqueness.
- `api/blog_views.py` — **CREATED** Public blog views: `blog_list` (all published posts) and `blog_detail` (single post by slug). No auth required.
- `api/internal_views.py` — Added 6 blog management endpoints: `api_blog_list`, `api_blog_get`, `api_blog_save`, `api_blog_publish`, `api_blog_delete`, `api_blog_generate`. Generate endpoint uses Anthropic API to create SEO-optimized drafts with meta descriptions.
- `api/urls.py` — Added 6 blog management API routes under `/api/internal/blog/`.
- `api/migrations/0001_blogpost.py` — BlogPost migration, applied.
- `svend/urls.py` — Added `/blog/`, `/blog/<slug>/`, `/robots.txt`, `/sitemap.xml` routes. Added Django sitemaps (StaticSitemap + BlogSitemap).
- `svend/settings.py` — Added `django.contrib.sitemaps` to INSTALLED_APPS.
- `templates/blog_list.html` — **CREATED** Public blog listing with SEO meta tags, OG tags, Svend branding.
- `templates/blog_detail.html` — **CREATED** Blog post detail with Article schema (JSON-LD), OG article tags, client-side markdown rendering (marked.js), CTA box.
- `templates/landing.html` — Added "Blog" link to nav bar.
- `templates/robots.txt` — Serves at /robots.txt (Allow /, /blog/; Disallow /app/, /api/, /admin/, /login/, /register/, /internal/; Sitemap reference).
- `templates/internal_dashboard.html` — Added "Content" tab (8th tab). Two-column layout: left has AI draft generator + post list, right has full markdown editor with live preview. Generate/save/publish/unpublish/delete workflow.
- `templates/base_app.html` — Added marked.js CDN for markdown preview.
**Verification:** Visit /blog/ (public, no auth). Visit /robots.txt and /sitemap.xml. Internal dashboard Content tab → generate, edit, save, publish a post → appears on /blog/.

---

### 2026-02-12 — Email composer in internal dashboard

**Debt item:** N/A (staff-only feature)
**Files changed:**
- `api/internal_views.py` — Added `api_send_email` POST endpoint + inline HTML email template with Svend branding. Supports: custom email, tier-based, all customers, and test mode. Markdown body → HTML via `markdown` lib. Per-user personalization with `{{name}}`, `{{email}}`, `{{tier}}`. Staff excluded from recipients.
- `api/urls.py` — Added `/api/internal/send-email/` route
- `templates/internal_dashboard.html` — Added Email tab (7th tab). Compose + live preview layout. "Send Test to Me" for proofing, "Send" with confirmation for bulk.
**Verification:** Email tab → write markdown, see preview. Test sends to your inbox from hello@svend.ai.

---

### 2026-02-12 — Staff exclusion from analytics + event tracking

**Files changed:**
- `api/internal_views.py` — Added `_customers()` and `_staff_ids()` helpers. All dashboard queries now exclude `is_staff=True`. Added `api_activity()` endpoint.
- `chat/models.py` — Added `EventLog` model. `chat/migrations/0004_eventlog.py` applied.
- `api/views.py` — Added `track_event()` POST endpoint at `/api/events/`
- `templates/base_app.html` — Added `svendTrack()` JS function, auto page_view + session_start
- 10 templates instrumented: workbench_new, spc, forecast, a3, experimenter, learn, rca, vsm, models, chat
- `templates/internal_dashboard.html` — Added Activity tab with page popularity, feature heatmap, daily sessions, user journeys
**Verification:** Browse any page → events recorded. Dashboard Activity tab shows customer-only data. Staff invisible in all analytics.

---

### 2026-02-12 — Calculator charts + Monte Carlo (Batches 3-5)

**Files changed:**
- `services/svend/web/templates/calculators.html` — Batch 3: Added Plotly gauge charts to Takt (zone-colored: red/green/yellow), DPMO (sigma 0-6 range), Inventory Turns (benchmark zones). Batch 4: Added Kanban pipeline visual (HTML/CSS supplier→cards→customer flow diagram), Little's Law bar chart (3 bars with L=λW annotation). Batch 5: Added Monte Carlo simulations to Safety Stock (varies demand/σ/LT/σLT), Kanban (varies demand/LT/safety%), Cpk (varies mean/σ, fixed specs). Each MC includes toggle button, 4-stat summary, histogram.
**Verification:** All 3 gauges render with correct zones. Kanban shows colored card tokens. Little's bars update with solve mode. MC toggles open/close correctly, histograms render 2000 runs.

---

### 2026-02-12 — Calculator cross-links: pull buttons + next steps (Batch 2)

**Files changed:**
- `services/svend/web/templates/calculators.html` — Added 2 pull buttons (EPEI←SMED changeover, Queue←Bottleneck throughput); added 8 "Next Steps" card containers (Takt, OEE, Safety Stock, Cpk, DPMO, SMED, EPEI, RTY) with `renderNextSteps()` calls wiring 24 cross-calculator navigation links; fixed `navigateToCalc()` to use correct `.ops-nav-item` selector.
**Verification:** After calculating any of the 8 calculators, clickable Next Steps cards appear below the derivation. Clicking navigates to the linked calculator.

---

### 2026-02-12 — Calculator cross-link infrastructure (Batch 1)

**Files changed:**
- `services/svend/web/templates/calculators.html` — Added `.calc-next-steps`/`.calc-next-step` CSS classes; `renderNextSteps()` and `navigateToCalc()` helper functions; `SvendOps.publish()` calls to 11 calculators (RTO, Kanban, EPEI, Safety Stock, EOQ, OEE, Bottleneck, Little's Law, DPMO, SMED, Cpk) publishing 18 new keys to shared state.
**Verification:** Page loads without console errors. After running any calculator, `SvendOps.values` contains the published keys.

---

### 2026-02-12 — Event tracking system for product analytics

**Debt item:** N/A (new feature — product improvement infrastructure)
**Files changed:**
- `chat/models.py` — Added `EventLog` model (event_type, category, action, label, page, session_id, metadata). 3 composite indexes for query performance
- `chat/migrations/0004_eventlog.py` — Migration applied
- `api/views.py` — Added `track_event()` POST endpoint at `/api/events/`. Supports batch (up to 20). Validates event_type against choices. Uses `bulk_create`
- `api/urls.py` — Added event tracking route + activity internal route
- `templates/base_app.html` — Added `svendTrack()` global JS function. Auto-logs `page_view` on every page load and `session_start` once per browser session. Uses `sessionStorage` for session ID (crypto.randomUUID). Fire-and-forget (non-blocking)
- `templates/workbench_new.html` — Added tracking: `dsw` / analysis type
- `templates/spc.html` — Added tracking: `spc` / chart type
- `templates/forecast.html` — Added tracking: `forecast` / method + symbol
- `templates/a3.html` — Added tracking: `a3` / save_report
- `templates/experimenter.html` — Added tracking: `experimenter` / design type
- `templates/learn.html` — Added tracking: `learn` / complete_section
- `templates/rca.html` — Added tracking: `rca` / evaluate
- `templates/vsm.html` — Added tracking: `vsm` / create
- `templates/models.html` — Added tracking: `models` / inference
- `templates/chat.html` — Added tracking: `chat` / send_message + mode
- `api/internal_views.py` — Added `api_activity()` endpoint: page popularity, feature heatmap, daily sessions, user journeys, feature use over time
- `templates/internal_dashboard.html` — Added Activity tab (6th tab) with KPI cards (events/pageviews/feature uses/sessions), 4 charts, user journey timeline with color-coded event tags
**Verification:** Browse any page → EventLog records created. Visit `/internal/dashboard/` → Activity tab shows page popularity, feature heatmap, session counts, user journeys. `svendTrack('feature_use', {category:'test'})` in console creates a record.

---

### 2026-02-12 — Calculator cross-link infrastructure (Batch 1)

**Files changed:**
- `services/svend/web/templates/calculators.html` — Added `.calc-next-steps`/`.calc-next-step` CSS classes; `renderNextSteps()` and `navigateToCalc()` helper functions; `SvendOps.publish()` calls to 11 calculators (RTO, Kanban, EPEI, Safety Stock, EOQ, OEE, Bottleneck, Little's Law, DPMO, SMED, Cpk) publishing 18 new keys to shared state.
**Verification:** Page loads without console errors. After running any calculator, `SvendOps.values` contains the published keys.

---

### 2026-02-12 — Multi-tenancy org management + auto-expand seat billing

**Debt item:** N/A (Enterprise feature — org member management + Stripe seat billing)
**Files changed:**
- `core/models/tenant.py` — Added `OrgInvitation` model (email, tenant FK, role, UUID token, status [pending/accepted/expired/cancelled], expires_at 7-day default). Added `stripe_seat_item_id` to Tenant for Stripe subscription item tracking.
- `core/models/__init__.py` — Export `OrgInvitation`
- `core/migrations/0005_org_invitation.py` — OrgInvitation model
- `core/migrations/0006_tenant_stripe_seat_item.py` — stripe_seat_item_id field
- `accounts/permissions.py` — Added `@require_org_admin` decorator (checks Membership.can_admin, NOT Django is_staff)
- `accounts/billing.py` — Added `SEAT_PRICE_ID` placeholder, `add_org_seat(tenant)` (auto-adds seat line item to owner's Stripe subscription with proration), `remove_org_seat(tenant)` (decrements/removes seat item), `_sync_seat_count()` (syncs Stripe seat quantity → tenant.max_members on webhook). Graceful fallback when SEAT_PRICE_ID not yet configured.
- `core/views.py` — 8 org management endpoints. `org_invite` calls `add_org_seat` (auto-expand, returns 402 on payment failure). `org_remove_member` and `org_cancel_invitation` call `remove_org_seat`.
- `core/urls.py` — 8 URL patterns under `org/` prefix
- `templates/settings.html` — Account/Organization tab system. Seat bar, members table with role change/remove, invite form (shows prorated charge messaging), pending invitations with cancel. Handles 402 payment errors. No separate "purchase seat" button — seats auto-expand on invite like Slack/GitHub.
**Verification:** `python manage.py check` passes. Set SEAT_PRICE_ID after creating $129/month/seat product in Stripe dashboard.

---

### 2026-02-12 — Internal telemetry dashboard

**Debt item:** N/A (new feature — staff-only)
**Files changed:**
- `api/internal_views.py` — **NEW** — 7 endpoints: dashboard_view (template render), api_overview (KPI cards), api_users (signups, tiers, demographics, DAU), api_usage (requests/day, domains, tokens, errors), api_performance (latency, pipeline stages, gate rates, error stages), api_business (revenue, funnel, churn, founder slots, feature adoption), api_insights (POST — sends anonymized data snapshot to Anthropic API, returns AI analysis)
- `templates/internal_dashboard.html` — **NEW** — Full single-page dashboard. KPI card row, 5 tabs (Users/Usage/Performance/Business/AI Insights), Chart.js visualizations (line, bar, doughnut), time range selector (7d/30d/90d), lazy-loaded tabs, AI chat interface with quick prompts. Theme-aware via SvendTheme.chartColors
- `api/urls.py` — Added 6 internal API routes under `/api/internal/`
- `svend/urls.py` — Added `/internal/dashboard/` page route
- `templates/base_app.html` — Added hidden "Internal" nav link, shown via JS for `is_staff` users
- `api/views.py` — Added `is_staff` to `me()` response (done in prior session)
**Verification:** Visit `/internal/dashboard/` as staff user → KPI cards, all 5 tabs render with real DB data. Non-staff → redirected. Time range selector updates all charts. AI Insights tab → sends prompt to Claude, displays response.

---

### 2026-02-12 — Hoshin Kanri subsystem expansion

**Debt item:** N/A (Enterprise feature expansion)
**Files changed:**
- `services/svend/web/templates/base_app.html` — Replaced hidden hoshin link in Methods dropdown with top-level "Hoshin Kanri" nav dropdown (enterprise-only) with Dashboard/Projects/Sites/Calculations links
- `services/svend/web/templates/hoshin.html` — Expanded from 1461 to 2730 lines. Added hash-based SPA router (#/dashboard, #/projects, #/sites, #/project/:id, #/project/:id/charter, #/project/:id/plan, #/project/:id/calculations, #/calc-library). New views: project detail with bowler chart + sidebar, kaizen charter form, project plan with Gantt chart + action items CRUD, calculations with baseline data entry + monthly operational data + formula editor, calculation method library with formula tester
- `agents_api/hoshin_calculations.py` — Added safe custom formula evaluator (AST-based, restricted to arithmetic + abs/min/max/round/sqrt/pow). Added `custom` to CALCULATION_METHODS and calculate_savings() dispatch
- `agents_api/models.py` — Added `custom_formula` and `custom_formula_desc` fields to HoshinProject
- `agents_api/hoshin_views.py` — Added `test_formula` endpoint (POST /api/hoshin/test-formula/), handle custom_formula fields in create/update, pass formula to calculate_savings for custom method
- `agents_api/hoshin_urls.py` — Added test-formula/ URL pattern
- `agents_api/migrations/0023_hoshin_custom_formula.py` — Migration for new model fields
**Verification:** Enterprise user sees Hoshin Kanri dropdown in nav. Navigate to #/dashboard, #/projects, click project row to see detail. Test charter form, plan/Gantt, calculations with baseline. Test formula at #/calc-library. `python manage.py check` passes.

---

### 2026-02-12 — Add "Show Derivation" to 24 calculator tools

**Debt item:** N/A (Feature parity)
**Files changed:**
- `services/svend/web/templates/calculators.html` — Added collapsible "Show Derivation" sections to 24 formula-based calculators (rto, kanban, epei, safety, oee, littles, pitch, rty, dpmo, turns, coq, smed, fmea, cpk, samplesize, lineeff, ole, cycletime, heijunka, capacity-load, queue-finite, queue-priority, queue-optimizer, queue-tandem). Each shows step-by-step formula work with substituted values. Reuses existing CSS and toggleDerivation() function from takt/eoq/queue. Simulators and interactive tools excluded as not appropriate.
**Verification:** Open calculators page, navigate to any modified calculator, verify "Show Derivation" appears and shows correct math when expanded.

---

### 2026-02-12 — Housekeeping: STANDARD.md update + user profile fields

**Debt item:** N/A (Foundation for personalized onboarding)
**Files changed:**
- `STANDARD.md` — Full rewrite to v2.0: updated directory tree, added sections for subscription tiers, feature gating (backend + frontend), theme system, template pattern, API surface table (19 routes), data model migration state, user profile fields, production environment docs, emergency procedures, key commands. Preserved 5S framework structure.
- `accounts/constants.py` — Added 4 TextChoices enums: Industry (8 options), Role (8 options), ExperienceLevel (3 options), OrganizationSize (4 options).
- `accounts/models.py` — Added 4 CharField fields to User model: industry, role, experience_level, organization_size. All blank=True for backwards compatibility.
- `accounts/migrations/0006_user_profile_fields.py` — Migration adding the 4 new fields.
- `api/views.py` — Fixed bug: `me()` was missing `bio` in response (settings page couldn't load it). Added 4 new profile fields to `me()` response. Expanded `update_profile()` allowed list with validation against TextChoices. Added Industry/Role/ExperienceLevel/OrganizationSize imports.
- `templates/settings.html` — Added "About You" section between Profile and Password with 4 dropdowns (industry, role, experience level, org size). Added `.section-desc` CSS. Added form submit handler + data loading in JS.
- `.kjerne/config.json` — Updated versions: lab 1.0.0→2.0.0, svend 1.0.0→2.0.0.
**Verification:**
- `python3 manage.py makemigrations accounts --check` — no pending changes
- Settings page → "About You" section visible, dropdowns save and persist
- `/api/auth/me/` returns bio + industry + role + experience_level + organization_size
- STANDARD.md accurately reflects current architecture

---

### 2026-02-12 — Theme system overhaul: contrast fixes + 3 new themes

**Debt item:** N/A (UX improvement — WCAG contrast compliance + expanded theme options)
**Files changed:**
- `templates/base_app.html` — Fixed contrast failures in Forest/Light/Midnight themes (`--text-dim`, `--error`, `--accent-purple`, `--accent-blue`). Added 3 new themes: Nordic Frost (light cool-blue), Sandstone (light warm), High Contrast (dark OLED). Added 4 semantic vars per theme (`--error-dim`, `--error-border`, `--warning-dim`, `--warning-border`). Updated SvendTheme JS fallback colors.
- `templates/settings.html` — Added Nordic Frost, Sandstone, High Contrast to theme selector dropdown. Replaced hardcoded rgba(159,74,74,...) with `var(--error-dim/border/error)`.
- `templates/dsw.html` — Replaced 5 instances of hardcoded `#9f4a4a` / `rgba(159,74,74,...)` with CSS variables.
- `templates/spc.html` — Replaced 6 instances of hardcoded error colors with CSS variables.
- `templates/forecast.html` — Replaced rgba error colors with CSS variables.
- `templates/models.html` — Replaced rgba error colors + modal overrides with `var(--card-bg)`.
- `templates/hoshin.html` — Replaced rgba error background with `var(--error-dim)`.
- `templates/chat.html` — Fixed `--accent-red` and `--text-dim` CSS vars, replaced rgba instances.
- `templates/learn.html` — Updated JS rgba to new #d06060-based values.
- `templates/workbench_new.html` — Replaced rgba in CSS and JS chart colors.
- `templates/analysis_workbench.html` — Updated `--aw-text-muted` (#5a6a5a→#7a8f7a), `--aw-danger` (#9f4a4a→#d06060), fixed ~20 inline hex references, updated rgba.
- `templates/login.html`, `register.html`, `privacy.html`, `terms.html`, `landing.html`, `verify_email.html` — Updated `--text-dim` (#5a6a5a→#7a8f7a) and `--error` where defined.
- `templates/problems.html`, `hypotheses.html`, `projects.html`, `a3.html` — Replaced per-theme modal `[data-theme="light/midnight"]` overrides with universal `var(--card-bg)` / `var(--border)`. Removed inline `background-color: #121a12` from modal HTML elements.
**Verification:**
- Settings → cycle all 6 themes, each applies instantly and looks cohesive
- `grep -r '#5a6a5a\|#9f4a4a\|rgba(159' templates/` returns 0 matches
- Modals open with correct background in all themes
- DSW/SPC error indicators clearly visible in all themes

---

### 2026-02-12 — Hoshin Kanri CI module (Enterprise-only)

**Debt item:** N/A (Enterprise tier feature — CI project tracking with savings calculations)
**Files changed:**
- `services/svend/web/accounts/constants.py` — Added `hoshin_kanri` feature flag to all 5 tier dicts (only `True` for ENTERPRISE)
- `services/svend/web/agents_api/models.py` — Added 3 models: `Site` (manufacturing plant), `HoshinProject` (OneToOne wrapper on core.Project for CI tracking), `ActionItem` (task/Gantt for any project)
- `services/svend/web/agents_api/migrations/0022_hoshin_kanri.py` — Migration creating `hoshin_sites`, `hoshin_projects`, `action_items` tables
- `services/svend/web/agents_api/hoshin_calculations.py` — NEW: 8 savings calculation methods (waste_pct, time_reduction, headcount, claims, layout, freight, energy, direct) + VSM delta estimator
- `services/svend/web/agents_api/hoshin_views.py` — NEW: 18 API endpoints for sites CRUD, hoshin projects CRUD, monthly actuals, batch creation from VSM proposals, dashboard rollup, action items
- `services/svend/web/agents_api/hoshin_urls.py` — NEW: URL routing for all hoshin endpoints
- `services/svend/web/agents_api/vsm_views.py` — Added `generate_proposals` view: diffs current/future VSM kaizen bursts, estimates savings per burst
- `services/svend/web/agents_api/vsm_urls.py` — Added generate-proposals URL
- `services/svend/web/svend/urls.py` — Added `api/hoshin/` and `app/hoshin/` routes
- `services/svend/web/templates/hoshin.html` — NEW: Enterprise dashboard with savings rollup, project management, site management, VSM proposal workflow
- `services/svend/web/templates/vsm.html` — Added "Generate CI Proposals" button (enterprise-only) with review modal for approving proposals and creating hoshin projects
**Verification:** Django check passes. Non-enterprise users see no hoshin UI. Enterprise users: create site, create hoshin project, update monthly actuals, generate proposals from VSM.
**Commit:** pending

---

### 2026-02-12 — Feature tiering: gate paid tools from free users

**Debt item:** N/A (Product differentiation / monetization)
**Files changed:**
- `services/svend/web/api/views.py` — Added `features` dict from `TIER_FEATURES` to `/api/auth/me/` response (single source of truth for frontend gating)
- `services/svend/web/accounts/permissions.py` — Added `@gated_paid` decorator (auth + `full_tools` feature check + rate limiting; returns 403 with upgrade prompt for free users)
- `services/svend/web/agents_api/whiteboard_views.py` — 11 endpoints: `@require_auth` → `@gated_paid`
- `services/svend/web/agents_api/a3_views.py` — 9 endpoints: `@require_auth` → `@gated_paid`
- `services/svend/web/agents_api/vsm_views.py` — 10 endpoints: `@require_auth` → `@gated_paid`
- `services/svend/web/agents_api/rca_views.py` — 11 endpoints: `@require_auth`/`@rate_limited` → `@gated_paid`
- `services/svend/web/agents_api/experimenter_views.py` — 9 endpoints: `@gated` → `@gated_paid`
- `services/svend/web/agents_api/synara_views.py` — 26 endpoints: `@gated`/`@require_auth` → `@gated_paid`
- `services/svend/web/agents_api/forecast_views.py` — 2 endpoints: `@gated` → `@gated_paid`
- `services/svend/web/agents_api/guide_views.py` — `guide_chat`/`summarize_project` → `@require_enterprise`; fixed missing `require_auth` import that crashed entire site
- `services/svend/web/workbench/graph_views.py` — 20 endpoints: replaced inline `is_authenticated` checks with `@require_auth` decorator
- `services/svend/web/templates/base_app.html` — Added `window.svendUser` global, upgrade modal HTML/CSS, global 403 interceptor, `svendUserReady` custom event
- `services/svend/web/templates/dashboard.html` — Added `data-feature="full_tools"` to 6 paid tool cards; JS gating adds `.locked` class + PRO badge + click-to-upgrade for free users; `loadRecent()` skips paid-API fetches for free users
- `services/svend/web/templates/experimenter.html` — Page-level gate check (upgrade modal on load for free users)
- `services/svend/web/templates/forecast.html` — Page-level gate check
- `services/svend/web/templates/a3.html` — Page-level gate check
- `services/svend/web/templates/rca.html` — Page-level gate check
- `services/svend/web/templates/vsm.html` — Page-level gate check
- `services/svend/web/templates/whiteboard.html` — Page-level gate check
**Verification:** Log in as free user → dashboard shows PRO badges on 6 tools → clicking locked card shows upgrade modal → navigating directly to `/app/whiteboard/` shows upgrade modal → API calls to paid endpoints return 403. Log in as paid user → all tools unlocked. Free tools (DSW, SPC, Projects, Learn, Calculators) remain accessible to all.
**Commit:** pending

---

### 2026-02-12 — Subscription system debug audit (17 bugs fixed)

**Debt item:** N/A (Critical bug fixes across billing/subscription system)
**Files changed:**
- `services/svend/web/accounts/models.py` — Fixed `timezone.timedelta` crash (AttributeError on daily reset), added `total_queries` increment, expanded Stripe `Status` choices to include `incomplete_expired`, `unpaid`, `paused`
- `services/svend/web/accounts/billing.py` — Payment failure now downgrades user tier; unknown Stripe price IDs default to FREE (not PRO); founder slot limit enforced at checkout; checkout success URL trailing slash fixed; success/cancel redirects go to `/app/` not `/`; session ownership verified on checkout success; Stripe error messages no longer leaked in redirect URLs; `subscription_ends_at` cleared on subscription deletion
- `services/svend/web/accounts/middleware.py` — Fixed stale "beta" tier reference (now uses `is_paid_tier()`); `last_active_at` DB writes throttled to 5-minute intervals; invite code casing normalized for POST/GET (was only normalized for JSON body)
- `services/svend/web/accounts/permissions.py` — Fixed tier limits docstring (was 10x actual values: 500→50, 1000→200, 5000→1000)
- `services/svend/web/accounts/constants.py` — Removed unused `TIER_STRIPE_PRICES` (had placeholder IDs disconnected from real Stripe config in billing.py)
- `services/svend/web/api/views.py` — `user_info` endpoint now reads `subscription_active` from Subscription model (consistent with `/api/auth/me/`)
**Verification:** All 6 files pass `ast.parse()`. Full endpoint flow should be tested: checkout → webhook → status → portal → payment failure → cancellation.
**Commit:** pending

---

### 2026-02-12 — P2 Gap Closure: Interactive Quality & DOE Tools + Backend SPC Charts

**Debt item:** DSW_gaps.md P2.1 (Multi-response optimization), P2.2 (Probit analysis), P2.3 (G chart, T chart, Moving Average, Zone, MEWMA)

**Files changed:**

**Phase A: Interactive Calculator Tools (calculators.html)**
- `services/svend/web/templates/calculators.html`:
  - Added "Quality & DOE" nav group with 3 new interactive tools
  - **Multi-Response Desirability Optimizer**: Define 2-4 responses with goal (maximize/minimize/target), bounds, weight/importance sliders. Define factors with ranges and linear response model coefficients. Client-side grid search optimization (up to 4 factors). Plotly desirability profile plots per response, composite D contour/surface plot, optimal settings with star marker. Sensitivity analysis insight panel (factor perturbation, binding response identification, improvement suggestions). Load Example with pharmaceutical formulation (Yield/Purity/Cost vs Temperature/Pressure).
  - **SPC Rare Events Lab (G + T Chart)**: Toggle between G chart (geometric, count between events) and T chart (exponential, time between events). Configurable baseline event rate, sample size, shift injection point and magnitude slider. Two modes: Generate All (instant) or Simulate (timer-based point-by-point like Kanban/Beer Game/TOC simulators). Pause/resume and speed controls in simulate mode. Control chart with UCL/LCL, OOC diamond markers, shift annotation line. Distribution fit panel (histogram + geometric/exponential PDF overlay). Insight panel with ARL analysis, shift detection delay, chart selection guidance.
  - **Probit / Dose-Response Explorer**: Editable data table (dose, n_tested, n_responding). Model toggle (Probit/Logit). Client-side IRLS fitting (Newton-Raphson on log-likelihood, Abramowitz & Stegun normal CDF). Plotly S-curve with fitted model, confidence bands (delta method), ED10/ED50/ED90 vertical marker lines. Auto log-scale when dose range > 10x. Fieller's theorem CI on ED50. Pearson chi-squared goodness of fit. Insight panel with model summary, slope interpretation, ED90/ED10 ratio analysis. Load Example with LD50 toxicology data (7 dose levels).
  - Added `calcMeta` entries and `ops-nav-item` elements for all 3 tools

**Phase B: Backend SPC Charts (dsw_views.py + workbench_new.html)**
- `services/svend/web/agents_api/dsw_views.py`:
  - Added `moving_average` to `run_spc_analysis()`: configurable span (window size), variable-width control limits that tighten as window fills, individual data points shown faded behind MA line, OOC detection, summary with effective shift detection size
  - Added `zone_chart` to `run_spc_analysis()`: color-coded A/B/C zone bands (green/yellow/red Plotly shapes), per-point zone scoring (A=8, B=4, C=2), cumulative score tracking with side-change reset, signal at cumulative ≥ 8, data points colored by zone, zone labels, separate cumulative score chart
  - Added `mewma` to `run_spc_analysis()`: multivariate EWMA with configurable lambda, chi-squared UCL, time-varying covariance matrix for T² statistic, auto-select numeric columns if none specified, covariance regularization for near-singular matrices, variable contribution bar chart at first OOC point

- `services/svend/web/templates/workbench_new.html`:
  - Added 3 ribbon buttons (MA, Zone, MEWMA) to Control Charts group with custom SVG icons
  - Added 3 dialog cases in `openSPCExtDialog()`: Moving Average (measurement + span), Zone Chart (measurement), MEWMA (multi-select variables + lambda)

**Verification:**
1. Navigate to `/app/calculators/` → "Quality & DOE" group visible with 3 tools
2. **Desirability**: Click "Load Example" → 3 profile plots + contour → drag weight sliders in response config → re-run → contour updates → insight shows sensitivity
3. **SPC Rare Events**: Set rate=0.02, shift at sample 30, magnitude 3x → select "Simulate" mode → click Generate → watch chart build live → shift detected → insight shows delay
4. **Probit**: Click "Load Example" → S-curve fits → ED50 shown → toggle Probit/Logit → curve shape changes → CI band shown
5. **DSW Workbench → Analysis → Control Charts**: MA, Zone, MEWMA buttons visible → run each with data → charts render with OOC detection

---

### 2026-02-11 — VSM ↔ Calculator Integration + Work Centers

**Debt item:** N/A (Major feature — cross-page data flow + work center grouping)

**Files changed:**

**Phase A: VSM ↔ Calculator Data Flow**
- `services/svend/web/templates/calculators.html`:
  - Added "VSM" import button to calculators header bar
  - Added VSM import modal (VSM selector dropdown, step preview table, import button)
  - Added `currentCalcId` tracking to `showCalc()` for context-aware import
  - Added `openVSMImport()`, `closeVSMImport()`, `previewVSMImport()`, `doVSMImport()` functions
  - 8 calculator-specific import functions: `loadVSMIntoLineSim()`, `loadVSMIntoKanbanSim()`, `loadVSMIntoTocSim()`, `loadVSMIntoBottleneck()`, `loadVSMIntoYamazumi()`, `loadVSMIntoTakt()`, `loadVSMIntoOEE()`, `loadVSMIntoKanbanSizing()`
  - Added `exportTaktToVSM()` function + "Export to VSM" button in takt calculator results
  - Added `buildEffectiveStations()` to collapse work center members into effective stations for simulators

**Phase B: Work Centers in VSM**
- `services/svend/web/agents_api/models.py`:
  - Added `work_centers` JSONField (each: `{id, name, x, y, width, height}`)
  - Updated `calculate_metrics()` with parallel machine logic: effective CT = 1 / Σ(1/CT_i)
  - Updated `to_dict()` to include work_centers

- `services/svend/web/agents_api/vsm_views.py`:
  - Added `work_centers` to structured data update whitelist
  - Added `work_centers` to future state copy

- `services/svend/web/agents_api/migrations/0021_add_vsm_work_centers.py`:
  - Migration: AddField work_centers to ValueStreamMap

- `services/svend/web/templates/vsm.html`:
  - Added "Work Center" to Process & Entities palette (draggable)
  - `renderWorkCenter()`: dotted-line rectangle (stroke-dasharray), subtle fill, name label, effective CT badge, resize handle
  - `associateStepsToWorkCenters()`: auto-links process steps whose center falls inside a work center box
  - `getWorkCenterEffectiveCT()` and `getWorkCenterMembers()` helper functions
  - Resize via corner handle: `startResizeWorkCenter()`, `resizeWorkCenterMove()`, `resizeWorkCenterEnd()`
  - `showWorkCenterProperties()`: name, width, height, effective CT, member list
  - Work center membership indicator (accent dot on process boxes inside a work center)
  - Updated `renderVSM()` to render work centers first (behind process steps)
  - Updated `renderLeadTimeLadder()`: work center members get one combined timeline segment with "(eff.)" label
  - Updated `updateMetrics()` with parallel machine effective CT logic
  - Updated undo/redo snapshot to include work_centers
  - Updated `dragElementEnd()` to re-associate after drag
  - Updated `deleteSelected()` to handle work centers (clears member associations)
  - All show*Properties() functions hide prop-wc-group when not relevant

**Phase C: Simulator Work Center Awareness**
- `services/svend/web/templates/calculators.html`:
  - `buildEffectiveStations(steps, workCenters)`: collapses parallel machines to single effective station
  - Updated `doVSMImport()` to use effective stations for all simulator imports
  - Updated preview table to show [WC] badge, effective CT, and machine count
  - Updated meta line to show work center count

**Verification:**
1. Open VSM → drag "Work Center" from palette → dotted box appears. Drag two process steps inside → they auto-associate (accent dot appears). Effective CT shown in top-right of box. Timeline shows single combined segment.
2. Open calculators → click "VSM" button → modal shows VSMs → select one → preview shows effective stations with [WC] markers → Import → stations populate in simulator.
3. Calculate takt → click "Export to VSM" → takt_time writes back to selected VSM.
4. Resize work center via corner handle → steps re-associate. Delete work center → member steps become standalone.

---

### 2026-02-11 — Calculators: 8 Simulator Placeholders + 3 Tier 1 Simulators
**Debt item:** N/A (Major feature)
**Files changed:**
- `services/svend/web/templates/calculators.html` — Added 8 new simulator entries to calculators nav (Kanban Sim, Beer Game, TOC/DBR, Safety Stock Sim, Heijunka Sim, SMED Sim, Cell Design Sim, FMEA Monte Carlo). 5 show "Coming Soon" placeholders with descriptions. 3 are fully built interactive simulators:
  - **Kanban Pull System Simulator**: PUSH vs PULL toggle, supermarket buffers with fill gauges, kanban card circulation, station states, WIP/throughput/lead time metrics, time-series chart
  - **Beer Game (Bullwhip Effect)**: 4-tier supply chain (Factory→Distributor→Wholesaler→Retailer→Customer), demand patterns (constant/step/seasonal/random), ordering policies (order-up-to/match-demand), inventory gauges, order thickness arrows, bullwhip ratio calculation, multi-line chart showing order amplification
  - **TOC / Drum-Buffer-Rope**: Uncontrolled vs DBR toggle, constraint auto-detection with drum label, visual buffer before constraint, rope from constraint to release gate, station utilization coloring, dual-axis WIP+throughput chart
- Added `.coming-soon` CSS for dimmed nav items with "SOON" badge
- Added `calcMeta` entries for all 8 simulators
- Added dynamic Insights panels to all 3 simulators (matching line simulator's bottleneck analysis pattern):
  - **Kanban**: System analysis (WIP, throughput, constraint, stockouts) + improvement suggestions (push→pull comparison, blocked/starved station detection, variability cost)
  - **Beer Game**: Bullwhip analysis (per-tier amplification ratios, cost breakdown, order vs demand variance) + improvement suggestions (policy comparison, lead time, factory over-ordering, stabilization timeline, countermeasures list)
  - **TOC/DBR**: Constraint analysis (utilization bars per station, WIP distribution before/after constraint, buffer status) + improvement suggestions (buffer sizing, constraint starvation, Goldratt's 5 Focusing Steps)
**Verification:** Open calculators, check all 8 appear in nav. Test Kanban sim (push vs pull), Beer Game (step demand), TOC (uncontrolled vs DBR). Verify pause/resume, speed slider, reset. Check Insights panel updates dynamically after ~30 seconds of simulation.

---

### 2026-02-11 — Calculators: OLE Layout Reorganization
**Debt item:** N/A (UI improvement)
**Files changed:**
- `services/svend/web/templates/calculators.html` — Split OLE Results section into three sections matching OEE layout: standalone OLE Score, Three Pillars with labor-specific loss descriptions, Loss Breakdown with centered donut chart + stats + commentary cards (Reading OLE, OLE vs OEE). Updated calcOLE() to populate new breakdown stat elements.
**Verification:** Open calculators > OLE, verify three distinct sections, centered donut, commentary cards render correctly.

---

### 2026-02-11 — VSM: Multiple Suppliers/Customers + Undo/Redo
**Debt item:** N/A (Feature)
**Files changed:**
- `services/svend/web/agents_api/models.py`:
  - Added `customers` and `suppliers` JSONField to ValueStreamMap model
  - Each entry: `{id, name, detail, x, y}`
  - Updated `to_dict()` to include new fields

- `services/svend/web/agents_api/vsm_views.py`:
  - Added `customers` and `suppliers` to structured data update list in `update_vsm()`

- `services/svend/web/agents_api/migrations/0020_add_vsm_customers_suppliers.py`:
  - Migration: AddField customers/suppliers to ValueStreamMap

- `services/svend/web/templates/vsm.html`:
  **Multiple customers/suppliers:**
  - Rewrote `renderCustomerSupplier()` to render from `customers`/`suppliers` arrays
  - New `renderEntityBox()` helper: draggable, editable (dblclick), deletable
  - Drag customer/supplier from palette → drops on canvas as new entity
  - Legacy migration: existing single-field data auto-migrated to arrays on load
  - `addElement()` handles `customer`/`supplier` types (client-side, saved via saveVSM)
  - `deleteSelected()` handles removing from customers/suppliers arrays

  **Undo/Redo:**
  - Added `vsmHistory[]` stack with `vsmHistoryIndex` (max 50 snapshots)
  - `saveVSMState()` captures deep-copy snapshot before each mutation
  - `undoVSM()` / `redoVSM()` restore snapshots and re-render
  - Keyboard: Ctrl+Z (undo), Ctrl+Shift+Z or Ctrl+Y (redo)
  - Toolbar: undo/redo buttons with arrow icons
  - State saved at: addElement, saveProperties, deleteSelected, addMaterialFlow, takt changes

**Verification:** Drag multiple suppliers/customers from palette onto canvas. Double-click to edit name/detail. Delete with Delete key. Ctrl+Z to undo, Ctrl+Shift+Z to redo.

---

### 2026-02-11 — VSM: Fix Kaizen, Customer/Supplier, Flow UX, and Takt Time
**Debt item:** N/A (UX fixes + feature)
**Files changed:**
- `services/svend/web/templates/vsm.html`:

  **Kaizen burst readability + editability:**
  - Changed text fill from white to dark (#1a1a2e) for contrast on orange/red
  - Added bold weight, word-wrapping into two lines for longer text
  - Added dblclick → showKaizenProperties() to edit text and priority

  **Customer/Supplier editable + draggable:**
  - Rewrote renderCustomerSupplier() to render as interactive SVG groups
  - Both now draggable (mousedown drag handler, positions in _customer_x/_y, _supplier_x/_y)
  - Both now dblclick-editable via showEntityProperties() → properties panel
  - Properties panel gains entity-specific fields (name + demand/frequency)

  **Material flow (push/pull) UX:**
  - Added step-by-step hint text in sidebar (#flow-hint) that updates as user progresses
  - Source process box gets dashed highlight (.flow-source-highlight) during selection
  - Success confirmation message after flow is created

  **Takt time from data:**
  - Added "Set Takt Time" section in metrics sidebar
  - Direct entry: type takt time in seconds and click Set
  - Calculate: enter available time (sec/day) and demand (units/day), calculates takt = avail/demand
  - Persists via existing takt_time model field and saveVSM()

**Verification:** Open VSM, double-click kaizen burst to edit, double-click customer/supplier to rename, click Push then two process boxes, use takt time calculator in sidebar.

---

### 2026-02-11 — Whiteboard: Fix Inverted Arrowheads
**Debt item:** N/A (Bug fix from connector refactor)
**Files changed:**
- `services/svend/web/templates/whiteboard.html`:
  - Flipped arrowhead polygons: `0 0, 12 4, 0 8` (tip at x=12 pointing in path direction)
  - Updated refX to 11/13 to place tip at path endpoint
  - Required because new path calculators end going inward toward target (opposite of old code)

**Verification:** Draw connections between shapes, arrowheads should point toward the destination.

---

### 2026-02-11 — Whiteboard: Connector Style System (Straight, Orthogonal, Curved)
**Debt item:** N/A (Feature + bug fix)
**Files changed:**
- `services/svend/web/templates/whiteboard.html`:

  **New connector style system:**
  - Added 3 connector routing styles: straight, orthogonal (90-degree), curved
  - Orthogonal is the default — clean right-angle paths like MS Office connectors
  - Style selector buttons appear in toolbar when connect/causal tool is active
  - Style stored per-connection in `conn.style` field, persists through save/export

  **Fixed curved connector curling bug:**
  - Replaced complex 3-branch bezier logic with simpler approach
  - cp1 extends from source port, cp2 extends from target port (both outward)
  - Removed `arrowSegment` hack that caused kinks at terminal end
  - Curve now arrives cleanly from the correct direction at both endpoints

  **New functions:**
  - `calculateStraightPath()` — direct line between ports
  - `calculateOrthogonalPath()` — Manhattan routing with right angles
  - `calculateCurvedPath()` — clean bezier that never loops

  **Temp connection preview respects selected style during drag**

**Verification:** Open whiteboard, select connect tool, use style buttons to switch between straight/orthogonal/curved. Test all port combinations (top↔bottom, left↔right, same-side, etc.)

---

### 2026-02-11 — Whiteboard: Connection z-index, Diamond Corners, Arrow Orientation
**Debt item:** N/A (Visual fixes)
**Files changed:**
- `services/svend/web/templates/whiteboard.html`:

  **Connections above shapes:**
  - Changed `.wb-connections` z-index from 1 to 10
  - Connection lines now render above shapes, not behind them

  **Diamond connection points at corners:**
  - Added CSS for `.wb-process-shape.diamond .wb-connection-point.*`
  - Positions moved to -21% (corner extension due to 45° rotation)
  - Counter-rotation applied (-45deg) to keep dots oriented correctly
  - Hover states updated to include counter-rotation
  - Updated `getConnectionPoint()` function to calculate correct coordinates for diamond corners

  **Arrowhead orientation fix:**
  - Modified `calculateCurvePath()` to add 12px straight segment at end
  - Bezier curve ends slightly before target, then straight line to target
  - Guarantees arrowhead always points toward the target shape regardless of curve path

**Verification:** Whiteboard → Add shapes and diamond → Connect them → Lines visible above shapes, diamond connections at corners, arrows point correctly

---

### 2026-02-10 — Custom Stepper Widget: Cockpit Rule for Number Inputs
**Debt item:** N/A (UX consistency)
**Files changed:**
- `services/svend/web/templates/calculators.html` — Custom stepper widget implementation:

  **CSS:**
  - `.stepper` container with flexbox layout
  - `.stepper-btn` for +/- buttons with hover/active states
  - `.stepper-value` for the input field (clean, centered)
  - `.stepper-sm` variant for smaller inline use
  - Global spinner hiding for all number inputs (`::-webkit-outer-spin-button`, `-moz-appearance: textfield`)

  **JavaScript:**
  - `createStepper(input, options)` — converts number input to stepper widget
  - Preserves min/max/step attributes
  - Hold-to-repeat for fast adjustment
  - `initializeSteppers()` — runs on DOMContentLoaded

  **Behavior:**
  - Main calculator inputs (`.calc-input`) get full stepper widgets
  - Inline/table number inputs just have spinners hidden (clean numbers)
  - "Cockpit rule": familiar widget shape = expected behavior

**Verification:** Ops Workbench → Any calculator → Number inputs show +/- buttons, numbers clearly visible

---

### 2026-02-10 — Product Flow Analysis (PFA) & Workflow Analysis (WFA)
**Debt item:** N/A (Shingo's two perspectives now accessible)
**Files changed:**
- `services/svend/web/templates/calculators.html` — Two flow analysis tools added under Flow section:

  **PFA — Product Flow Analysis (TIPS):**
  - Follow the PRODUCT through the process
  - Categories: Transport, Inspect, Process, Storage (B=Between, L=Lot, W=Within)
  - Records: step description, category, time (min), distance (m)
  - Metrics: Process Ratio %, total time, total distance, step count
  - Breakdown by category with color-coded display
  - Pie chart visualization
  - Flow diagram showing step sequence
  - Before/after comparison with baseline capture
  - Example data button

  **WFA — Workflow Analysis (Therbligs):**
  - Follow the WORKER through the task
  - Categories: VA (value-add), RW (required work), P (parts), T (tools), I (inspection), MH (material handling), UW (unnecessary work), IT (idle time)
  - NVA taxonomy: NVA/R (required, target later) vs NVA/N (unnecessary, eliminate now)
  - Records: element description, category, time (sec)
  - Metrics: VA Ratio %, total time, NVA/R %, NVA/N %
  - Bar chart visualization
  - Separate lists for NVA/R and NVA/N items
  - Before/after comparison with baseline capture
  - Example data button

  **Integration:**
  - Full persistence (auto-save, scenarios, export/import)
  - Consistent UI with rest of workbench

**Verification:** Ops Workbench → Flow → Product Flow (PFA) or Workflow (WFA) → Load Example → See analysis

---

### 2026-02-10 — House of Quality (QFD): Full Four-Phase Deployment under 3P
**Debt item:** N/A (First usable QFD since Y2K)
**Files changed:**
- `services/svend/web/templates/calculators.html` — Complete QFD implementation:

  **Four Cascading Phases:**
  1. House of Quality: Customer WHATs → Engineering HOWs
  2. Part Deployment: Engineering Characteristics → Part Characteristics
  3. Process Planning: Part Characteristics → Process Parameters
  4. Production Control: Process Parameters → Control Points

  **Phase 1 Features (House of Quality):**
  - Customer requirements with importance ratings (1-5)
  - Engineering characteristics with units and targets
  - Relationship matrix with click-to-cycle (●=9, ○=3, △=1)
  - Correlation roof showing HOW-to-HOW relationships (++, +, -, --)
  - Priority scores calculated automatically
  - Coverage % (what % of WHATs have strong relationships)
  - Conflict detection (negative correlations)
  - Priority bar chart

  **Phase 2-4 Features:**
  - Inputs cascade from previous phase (shown as chips)
  - Add/remove items dynamically
  - Relationship matrices with click-to-cycle
  - Navigation between phases

  **Traceability:**
  - Export function traces controls back through all phases
  - Every production control links to customer requirement

  **UI:**
  - Tab navigation for four phases
  - Cascade buttons to move data forward
  - Load Example button with realistic sample data
  - Fully persistent (auto-save, scenarios)

**Verification:** Ops Workbench → Method → House of Quality → Load Example → Click cells → Cascade through all phases

---

### 2026-02-10 — Scheduling Tools: Job Sequencer, Optimizer, Capacity, Mixed-Model, Due Date Risk
**Debt item:** N/A (MAJOR FEATURE — bridges ops and scheduling worlds)
**Files changed:**
- `services/svend/web/templates/calculators.html` — Five new scheduling calculators:

  **1. Job Sequencer (Visual Foundation)**
  - Drag-and-drop Gantt chart for job scheduling
  - Live metrics: makespan, total flow time, total setup, jobs late, tardiness
  - Setup groups with sequence-dependent changeover times
  - Pulls from Changeover Matrix for setup times
  - Push to Line Simulator creates orders from sequence

  **2. Sequence Optimizer**
  - Four algorithms: Nearest Neighbor, 2-Opt, EDD, SPT
  - Four objectives: minimize setup, tardiness, makespan, avg flow time
  - Before/After comparison with improvement percentage
  - Apply optimized sequence back to Job Sequencer

  **3. Capacity Load Chart**
  - Work orders with hours required and start day
  - Stacked bar chart showing load vs capacity by day
  - Red highlighting for overloaded days
  - Efficiency factor for realistic capacity
  - Metrics: total load, available capacity, utilization, overload days

  **4. Mixed-Model Sequencer**
  - Toyota-style heijunka sequencing
  - Three methods: Ratio-Based (Toyota), Goal Chasing, Batched
  - Visual sequence with color-coded product blocks
  - Smoothness index and max consecutive same product
  - Comparison chart: leveled vs batched cumulative production
  - Push to Line Simulator with grouped orders

  **5. Due Date Risk Simulator**
  - Monte Carlo simulation (100-5000 runs)
  - Parameters: CV of processing time, breakdown probability, breakdown duration
  - Per-order on-time probability with color coding
  - Histogram of completion times with due date marker
  - Overall OTD %, average delta, worst case (P95)

  **Integration Points:**
  - Job Sequencer ↔ Line Simulator (bidirectional)
  - Job Sequencer → Sequence Optimizer
  - Job Sequencer → Capacity Load
  - Job Sequencer → Due Date Risk
  - Heijunka → Mixed-Model
  - Mixed-Model → Line Simulator
  - Changeover Matrix → Setup times everywhere

  **Persistence:**
  - All scheduling data included in auto-save and scenarios
  - sequencerJobs, sequencerOrder, capacityOrders, mixedProducts, ddsOrders

**Verification:** Ops Workbench → Scheduling section → all five tools functional with cross-links

---

### 2026-02-10 — Scenario Persistence: LocalStorage + Export/Import
**Debt item:** N/A (Critical infrastructure)
**Files changed:**
- `services/svend/web/templates/calculators.html` — Full persistence system:

  **Auto-Save:**
  - Every state change auto-saves to localStorage (1s debounce)
  - Restores automatically on page load
  - Covers: Line Sim, SMED, Yamazumi, Changeover Matrix, FMEA, RTY, Cycle Time, Before/After, Heijunka, Priority Queue, Multi-Stage Queue

  **Named Scenarios:**
  - Save current state with custom name
  - Dropdown in header to switch between scenarios
  - Rename and delete scenarios
  - Scenarios persist across browser sessions

  **Export/Import:**
  - Export all scenarios to JSON file
  - Import scenarios from JSON
  - Enables backup and team sharing
  - Includes version for future compatibility

  **State Captured:**
  - `lineStations`, `lineProducts`, `lineOrders`, Line Sim settings
  - `smedData`, `smedBaseline`, impact calculator inputs
  - `yamazumiData`, takt time
  - `changeoverMatrix`, products
  - `bottleneckData`, `fmeaData`, `rtyData`
  - `cycleData`, `baData`, `heijunkaData`
  - `tandemStages`, `priorityClasses`

  **UI:**
  - Scenario dropdown + Save button in header
  - Three-dot menu for Save As, Rename, Delete, Export, Import
  - Toast notifications for feedback

**Verification:** Open Ops Workbench → make changes → refresh page → changes persist. Save as scenario → switch scenarios → changes preserved.

---

### 2026-02-10 — SMED Calculator Enhancement: Before/After Tracking & Line Sim Integration
**Debt item:** N/A (Feature enhancement)
**Files changed:**
- `services/svend/web/templates/calculators.html` — Enhanced SMED calculator:

  **Before/After Comparison:**
  - Capture Baseline button saves current internal time
  - Real-time comparison shows improvement percentage
  - Clear visual: Before → After with delta

  **Conversion Suggestions:**
  - Pattern-based analysis of element names
  - Suggests specific kaizen for each internal element:
    - Pre-staging for "get/fetch/find" activities
    - Quick-change for "remove/install/mount" activities
    - Pre-conditioning for "heat/warm/cool" activities
    - Parallel work for "check/inspect" activities
  - Impact rating (high/medium/low) for prioritization

  **Impact Calculator:**
  - Inputs: changeovers/day, operating days/year, hourly cost
  - Outputs: hours recovered/year, capacity gain %, annual value $
  - Shows current loss before baseline, savings after

  **Line Simulator Integration:**
  - "Apply & See Impact" button pushes internal time to Line Sim
  - Converts minutes to seconds automatically
  - Navigates to Line Sim for immediate simulation
  - Toast notification confirms the value set

  **New Functions:**
  - `captureBaseline()` - snapshots current state
  - `clearBaseline()` - removes baseline
  - `suggestConversions()` - AI-like kaizen suggestions
  - `calcSMEDImpact()` - annual value calculations
  - `applySMEDToLineSim()` - cross-calculator integration

**Verification:** Go to Ops Workbench → SMED Analysis → Capture Baseline → Convert elements → See improvement & apply to Line Sim

---

### 2026-02-10 — Line Simulator: Order-Driven Value Stream Simulation
**Debt item:** N/A (MAJOR — This is Arena/Simul8 for $29/month instead of $50K)
**Files changed:**
- `services/svend/web/templates/calculators.html` — Full Line Simulator with Order-Driven Mode:

  **Simulation Modes:**
  - **Infinite Supply:** Continuous production (original mode)
  - **Order-Driven:** Process customer orders with due dates, track on-time delivery

  **Order-Driven Features:**
  - **Product Types:** Define multiple products with different cycle time multipliers
  - **Changeover Time:** Time to switch between products (connects to SMED)
  - **Order Queue:** Orders with product, quantity, due date
  - **Generate Sample Orders:** Quick setup for demos
  - **Order Tracking:** Status (pending/in-progress/complete), completion time

  **Delivery Metrics:**
  - On-Time Delivery % (color-coded: green >95%, yellow >80%, red <80%)
  - Orders Complete counter
  - Average Lead Time
  - Changeover Loss (total time spent changing over)

  **Root Cause Analysis for Late Orders:**
  - Traces back to exact cause of lateness
  - "Changeover to Product B started at t=340s"
  - "Breakdown at Station 3 (t=180s)"
  - "Blocking at Station 2 (buffer full, t=420s)"
  - "Cumulative delays exceeded buffer"

  **Visual Enhancements:**
  - Input node shows current order progress (#3: 4/8)
  - Product color coding
  - Changeover indicator with countdown (pulsing orange)
  - Current product letter badge

  **Core Simulation (from earlier):**
  - Station config, CoV variability, WIP buffers
  - One-piece vs batch flow
  - Random breakdowns with downtime tracking
  - Import from Yamazumi
  - A/B scenario comparison
  - Scenario save/load to localStorage
  - Export report for kaizen events

**The Insight:**
VSM, Yamazumi, and Line Sim are the same underlying system. Now they're connected:
- Define stations in Yamazumi → Import to Line Sim
- Add products and orders → Simulate delivery performance
- Add changeovers → See SMED impact on OTD
- Enable breakdowns → See TPM impact on OTD

"When an order is late, highlight the moment in the simulation where it became inevitable."

**Verification:** Navigate to /app/calculators/, Line Simulator. Switch to "Order-Driven" mode, generate sample orders, click Start. Watch orders flow, see changeovers, check on-time delivery. Enable breakdowns to see root cause analysis.

---

### 2026-02-10 — New "Prepare" Ribbon Tab: Data Cleaning, Profiling & Meta-Analysis

**Files**: `workbench_new.html`, `dsw_views.py`

New dedicated Prepare tab between Data and Analysis with 4 groups (11 buttons):
- **Clean**: Triage (moved from Data tab), Profile (summary stats + correlation heatmap + distribution grid), Missing (pattern matrix, MCAR test, row completeness), Duplicates (exact/subset mode)
- **Detect**: Outliers (IQR, Z-score, Modified Z-score/MAD, Mahalanobis with consensus)
- **Transform**: Encode (one-hot/label), Scale (z-score/min-max/robust), Bin (equal-width/frequency/custom breakpoints)
- **Meta-Analysis**: Meta (fixed+random effects, DerSimonian-Laird, forest plot, funnel plot, I²/Q/tau²), Effect Size (Cohen's d, Hedges' g, Glass's delta, OR, RR with 95% CI)

Backend: 6 new analysis_ids in run_statistical_analysis + 3 new tools in transform_data. All smoke-tested.

---

### 2026-02-10 — Queuing Lab: Comprehensive Queuing Theory Suite
**Debt item:** N/A (Major differentiator — competes with $5-20K/yr simulation software)
**Files changed:**
- `services/svend/web/templates/calculators.html` — Full Queuing Lab with 7 tools:
  - **M/M/c Basic:** Enhanced with Monte Carlo simulation and full Erlang C derivation
  - **M/M/c/K Finite Queue:** Limited capacity systems (drive-throughs, ERs). Shows blocking probability, effective throughput, lost customers. Chart shows blocking vs capacity tradeoff. Monte Carlo for variability.
  - **Priority Queue:** Multi-class priority system (ER triage, tiered support). Dynamic class management with color coding. Shows wait times by priority class with visualization.
  - **Staffing Optimizer:** Find optimal server count given costs. Inputs: arrival rate, service rate, server cost/hr, wait cost/hr, optional SLA target. Output: optimal staffing with cost breakdown chart and comparison table. Uses total cost minimization (server cost + wait cost).
  - **Live Queue Simulator:** Real-time animated queue visualization. Watch customers arrive (blue dots), get served (server icons turn red), and queue length fluctuate. Adjustable variability (CoV slider 0-100%). Live stats + "What Broke?" burst analysis that identifies when/why queues exploded.
  - **A/B Scenario Compare:** Run two simulations side-by-side with identical random arrivals. Current state vs proposed change. Real-time verdict showing % improvement. Perfect for "should we add a server?" decisions.
  - **Multi-Stage (Tandem) Queue:** Model sequential processes — ER: Triage→Doctor→Checkout. Manufacturing: Assembly→QC→Pack. Shows per-stage metrics, identifies bottleneck, calculates end-to-end time. Monte Carlo for total system variability.
  - All tools designed for real-world use: hospitals, call centers, Chipotle, manufacturing
**Verification:** Navigate to /app/calculators/, see "Queuing Lab" nav group with 7 items. Try A/B Compare, Multi-Stage, and the Live Simulator with burst analysis.

---

### 2026-02-10 — Operations Workbench: Cross-Calculator Intelligence
**Debt item:** N/A (Feature expansion)
**Files changed:**
- `services/svend/web/templates/calculators.html` — Added three interconnected systems:
  - **SvendOps Shared State:** Calculators publish their results to a shared data bus. Other calculators can pull these values via link buttons (chain icon). Takt Time publishes to 'takt' and 'taktMin', available to RTO and Pitch calculators.
  - **Monte Carlo Simulation:** Added simulation toggle to EOQ and Queuing (M/M/c) calculators. Runs 2000 iterations with ±10-15% input variability, displays histogram with 5th/95th percentiles and mean. Shows confidence intervals for inventory decisions and wait time predictions.
  - **Show Derivation:** Expandable sections showing step-by-step math with actual values. Added to Takt Time, EOQ, and Queuing calculators. Educational for senseis and builds trust.
  - **SMED Waterfall:** Changed from pie chart to cascading waterfall chart with internal (red), external (green), and total (Svend Gold) bars.
  - **UI Polish:** Fixed nav alignment issue (added align-items:stretch to flex containers), standardized chart heights to 350px.
**Verification:** Navigate to /app/calculators/. Calculate Takt Time, then go to RTO and click the link icon to pull the value. Toggle "Run Monte Carlo" on EOQ to see the simulation. Expand "Show Derivation" on Takt to see the math.

---

### 2026-02-10 — Operations Workbench Expansion (Batch 2: 10 Calculators)
**Debt item:** N/A (Feature expansion)
**Files changed:**
- `services/svend/web/templates/calculators.html` — Added 10 new lean/six sigma calculators with interactive visualizations:
  - **Changeover group:** SMED Analysis (dynamic activity table, internal/external/waste breakdown Sankey), Changeover Matrix (product×product heatmap for setup times)
  - **Risk & Quality group:** FMEA/RPN Calculator (dynamic failure mode table, RPN pareto chart with 80% threshold), Cp/Cpk Process Capability (histogram with spec limits + normal fit), Sample Size Calculator (Type I/II error tradeoff curves)
  - **Line Performance group:** Line Efficiency (planned vs actual bars with efficiency gauge), OLE Calculator (Overall Labor Effectiveness with donut breakdown)
  - **Analysis group:** Cycle Time Study (multi-observation table with box plot variability viz), Before/After Comparison (grouped bar chart with improvement percentage), Heijunka Box Calculator (leveled production schedule heatmap)
  - All calculators feature real-time updates as inputs change
**Verification:** Navigate to /app/calculators/, all 10 new nav items visible and functional with interactive Plotly charts.

---

### 2026-02-10 — Operations Workbench Expansion (Batch 1: 7 Calculators + Visualizations)
**Debt item:** N/A (Feature expansion)
**Files changed:**
- `services/svend/web/templates/calculators.html` — Enhanced existing calculators with visualizations (OEE donut chart, EOQ cost curve, Safety Stock distribution). Added 7 new calculators: Little's Law (WIP/Throughput/Cycle Time relationship plot), M/M/c Queuing (full Erlang C with wait time vs utilization curve), Pitch (takt × pack quantity), RTY (multi-step yield cascade with waterfall viz), DPMO/Sigma Level (defect rate curve), Inventory Turns (months of supply bar chart), Cost of Quality (PAF model pie chart). New nav groups: Flow Analysis, Quality Metrics, Financial.
**Verification:** Navigate to /app/calculators/, all visualizations render, calculations update in real-time.

---

### 2026-02-07 — UX Polish, Full Nelson Rules, Non-Parametric + Non-Normal Capability
**Debt item:** N/A (Minitab gap closure, UX improvement)
**Files changed:**
- `services/svend/web/templates/workbench_new.html` — (1) Gage R&R dialog: replaced textarea-based manual data entry with column dropdown selectors (measurement, part, operator, study type), now routes through DSW backend via `runStatsAnalysis`. (2) Analysis ribbon restructured: Control Charts and Quality groups use 2-row compact layout with `flex-direction:column`, ribbon-content now `flex-wrap` enabled. Chart labels shortened (P', U', K-M, NN Cap). (3) Added Non-Normal Capability button + `openNonNormalCapDialog()` to Quality group. (4) Added sign_test and mood_median to both Non-Parametric dialog and All Tests dialog.
- `services/svend/web/agents_api/dsw_views.py` — (1) `_spc_nelson_rules()` expanded from 3 rules to all 8 Nelson rules: Rule 3 (6 trending), Rule 4 (14 alternating), Rule 6 (4/5 beyond 1σ), Rule 7 (15 within 1σ stratification), Rule 8 (8 beyond 1σ mixture). (2) Added `sign_test` (one-sample median test with binomial CI) and `mood_median` (k-sample median test with chi-squared contingency). (3) Added `nonnormal_capability` to `run_spc_analysis` — fits Normal/Lognormal/Weibull/Exponential, auto-selects best fit by KS p-value, computes equivalent Pp/Ppk, histogram with PDF overlay, probability plot.
**Verification:** All 8 Nelson rules unit tested. Sign test and Mood's median smoke tested. Non-normal capability tested with lognormal data (correctly identifies Lognormal as best fit). Template loads without errors.

---

### 2026-02-07 — SPC Nelson Rules, Laney Charts, B/W Capability, Reliability Suite
**Debt item:** N/A (Minitab gap closure)
**Files changed:**
- `services/svend/web/agents_api/dsw_views.py` — (1) Added `_spc_nelson_rules()` and `_spc_add_ooc_markers()` helpers checking Rules 1, 2, 5 with red diamond OOC markers. Applied to all 10 SPC charts: I-MR, X-bar R, X-bar S, P, NP, C, U, CUSUM, EWMA, and Nelson rule violation text in summaries. (2) Added `laney_p` and `laney_u` chart types with σz overdispersion correction. (3) Added `between_within` capability analysis with nested variance decomposition (within/between/overall σ), Cp/Cpk/Pp/Ppk, variance bar chart, and histogram with within vs overall normal fits. (4) Added `run_reliability_analysis()` function with 5 analyses: Weibull (probability plot, reliability curve, B-life), Lognormal (probability plot, reliability curve), Exponential (probability plot, MTTF CI), Kaplan-Meier survival (step function with 95% CI and censored markers), Reliability Test Planning (sample size calculator for demo testing).
- `services/svend/web/templates/workbench_new.html` — (1) Added 2 Laney chart buttons (P', U') and B/W Capability button to SPC ribbon section with dialog cases in `openSPCExtDialog()`. (2) Added new Reliability ribbon group with 5 buttons (Weibull, Lognormal, Exponential, Kaplan-Meier, Test Plan) and `openReliabilityDialog()` function with custom dialogs per analysis type.
**Verification:** All 10 SPC charts, 3 new SPC analyses, and 5 reliability analyses smoke-tested via Django shell. Template loads without errors.

---

### 2026-02-07 — Analysis Ribbon Restructure (2-row layout)
**Debt item:** N/A (UX improvement)
**Files changed:**
- `services/svend/web/templates/workbench_new.html` — Restructured the Analysis tab ribbon from a single overcrowded row (37 buttons) into two conceptual rows separated by a subtle border: **Row 1** = Quality Engineering (Control Charts 11btn, Quality 5btn, Reliability 8btn), **Row 2** = Statistical Modeling (Modeling 4btn, Advanced 6btn, All Tests 1btn). Total 34 buttons. Consolidated Reliability group from 10 to 8 buttons (merged Lognorm/Expon into Dist ID pathway, renamed Compete→CIF, Test Plan→Plan). Added descriptive `title` tooltips to every button. Tightened button gaps with `gap:0.15rem`.
**Verification:** Template loads OK. All 34 button onclick handlers resolve to existing functions.

---

### 2026-02-07 — GLM Enhancement (Full ANCOVA/Multivariate Regression)
**Debt item:** N/A (Minitab parity — GLM is the workhorse)
**Files changed:**
- `services/svend/web/agents_api/dsw_views.py` — Rewrote GLM from scratch as unified engine for ANOVA/ANCOVA/regression/mixed models. Key additions: (1) Factor*covariate interactions for ANCOVA homogeneity-of-slopes test. (2) LS-Means (estimated marginal means) — covariate-adjusted group means at covariate mean, with raw vs adjusted comparison. (3) Partial eta-squared (η²p) effect sizes in ANOVA table. (4) Full 4-panel residual diagnostics (vs fitted, normal QQ, histogram, vs order). (5) Interaction plots for factor×factor. (6) ANCOVA covariate scatter with per-group regression lines. (7) Auto-detection of model type label (ANOVA, ANCOVA, Mixed, Regression). (8) 95% CI error bars on main effects plots with grand mean reference.
- `services/svend/web/templates/workbench_new.html` — Updated GLM dialog: multi-select covariates, factor×covariate interaction checkbox, mode hint (ANOVA/ANCOVA/Regression), dynamic output title.
**Verification:** All 5 GLM modes smoke-tested: Pure ANOVA (5 plots, η²p), ANCOVA (6 plots, LS-Means, homogeneity test, covariate plot), Two-way (7 plots, interaction plot), Mixed (5 plots, ICC), Regression (4 plots, R²). Template loads OK.

---

### 2026-02-07 — GLM, MANOVA, Factor Analysis, Tolerance Intervals, Variance Components, Ordinal Logistic, Competing Risks
**Debt item:** N/A (Minitab gap closure — closing remaining ~8% gap)
**Files changed:**
- `services/svend/web/agents_api/dsw_views.py` — Added 8 new analyses: (1) `glm` in run_statistical_analysis — General Linear Model with fixed/random factors, covariates, interactions, Type III ANOVA table, effects plots, residual diagnostics. Supports OLS for pure fixed and mixedlm for random effects. (2) `manova` — Multivariate ANOVA with Pillai's trace, Wilks' lambda, Hotelling-Lawley, Roy's greatest root, univariate F-tests per response. (3) `tolerance_interval` — Normal and non-parametric tolerance bounds with coverage/confidence, histogram with bound lines. (4) `variance_components` — ANOVA-based or REML variance decomposition, pie chart + bar chart of components. (5) `ordinal_logistic` — Proportional odds model via statsmodels OrderedModel, predicted probability curves. (6) `factor_analysis` in run_ml_analysis — Exploratory factor analysis with varimax rotation, scree plot, loading heatmap, communalities, Kaiser criterion auto-selection. Added to unsupervised_analyses list. (7) `competing_risks` in run_reliability_analysis — Aalen-Johansen cumulative incidence functions for multiple failure modes, CIF plot, stacked area plot.
- `services/svend/web/templates/workbench_new.html` — Added GLM button (Parametric group) with `openGLMDialog()` (multi-factor select, random factor, covariate). Added Factor Analysis button (Multivariate group) with `openFactorAnalysisDialog()` (variable multi-select, rotation, n_factors). Added Competing Risks button ("Compete") to Reliability group. Added GLM, ordinal_logistic, variance_components, factor_analysis to multivar and All Tests dialog dropdowns. Updated generic dialog dispatch to route ML tests correctly.
- Installed `statsmodels` 0.14.6 (was missing from venv).
**Verification:** All 8 new analyses smoke-tested: GLM fixed (4 plots), GLM mixed (3 plots), MANOVA (3 plots), Tolerance Interval (1 plot), Variance Components (2 plots), Ordinal Logistic (1 plot), Factor Analysis (2 plots, correctly finds 2 factors in synthetic data), Competing Risks (2 plots). Template loads OK.

---

### 2026-02-07 — Reliability Expansion + Holt-Winters Forecasting
**Debt item:** N/A (Minitab gap closure)
**Files changed:**
- `services/svend/web/agents_api/dsw_views.py` — Added 4 new reliability analyses to `run_reliability_analysis()`: (1) `distribution_id` — fits 6 distributions (Normal, Lognormal, Weibull, Exponential, Gamma, Loglogistic), ranks by KS p-value, probability plots for top 3, density comparison. (2) `accelerated_life` — Arrhenius/Inverse Power Law models, fits Weibull at each stress level, extrapolates to use conditions. (3) `repairable_systems` — Crow-AMSAA power law NHPP, Laplace trend test, MCF plot, failure intensity (ROCOF) plot. (4) `warranty` — fits Weibull to return times, projects future returns, cumulative return rate + monthly incremental return plots.
- `services/svend/web/templates/workbench_new.html` — Added 4 new buttons (Dist ID, ALT, Repair, Warranty) to Reliability ribbon group in 3-row layout. Added dialog cases in `openReliabilityDialog()`.
- `services/svend/web/agents_api/forecast_views.py` — Added `holt_winters_forecast()` with additive/multiplicative seasonality. Falls back to simple exponential if insufficient data. Added dispatch case `elif method == "holt_winters"` in `forecast()` view.
**Verification:** All 4 reliability analyses + Holt-Winters (additive, multiplicative, short-data fallback) smoke-tested via Django shell. Template loads without errors.

---

### 2026-02-10 — RCA Similar Incidents Feature (#5)
**Debt item:** N/A (new feature)
**Files changed:**
- `services/svend/web/agents_api/embeddings.py` — New embedding service using sentence-transformers (all-MiniLM-L6-v2, 384 dims). Functions: `generate_embedding()`, `generate_rca_embedding()`, `cosine_similarity()`, `find_similar_in_memory()`. Model cached as singleton, uses GPU if available.
- `services/svend/web/agents_api/models.py` — Added `embedding` BinaryField to RCASession for storing vectors. Added `generate_embedding()` and `get_embedding()` helper methods.
- `services/svend/web/agents_api/rca_views.py` — Added embedding generation on session create/update. New endpoints: `find_similar()` (POST /api/rca/similar/) searches for matching past incidents, `reindex_embeddings()` (POST /api/rca/reindex/) regenerates all user embeddings.
- `services/svend/web/agents_api/rca_urls.py` — Added routes for `/similar/` and `/reindex/`.
- `services/svend/web/templates/rca.html` — Added Similar Incidents section that appears when entering event description. Uses debounced search (800ms). Shows top 3 matches with similarity percentage. Click to load past session.
- `services/svend/web/agents_api/migrations/0019_add_rca_embedding.py` — Migration for embedding field.
**Verification:** Go to RCA tool, type an incident description (20+ chars). After 800ms, similar past incidents appear with % match. Tested: related events show ~56% similarity, unrelated show ~16%.

---

### 2026-02-10 — OpEx Calculators (Crewing, Inventory, OEE)
**Debt item:** N/A (new feature)
**Files changed:**
- `services/svend/web/templates/calculators.html` — New page with 8 calculators:
  - **Crewing**: Takt Time, RTO (Required to Operate) with CoV margin, Yamazumi line balance chart
  - **Inventory**: Kanban quantity, EPEI, Safety Stock (with demand/lead time variation), EOQ
  - **Capacity**: OEE (with A×P×Q breakdown), Bottleneck identifier
  - All client-side instant calculation, Plotly visualizations for Yamazumi/Bottleneck, DSW pull buttons (stub)
- `services/svend/web/svend/urls.py` — Added `/app/calculators/` route
- `services/svend/web/templates/base_app.html` — Added Calculators link to Methods nav dropdown
**Verification:** Go to Methods → Calculators. Takt, RTO, Kanban, OEE calculators all compute instantly. Yamazumi shows stacked bar with takt line.

---

### 2026-02-10 — Graph expansion, Forge removal, Triage auto-open
**Debt item:** N/A (UX improvements)
**Files changed:**
- `services/svend/web/templates/workbench_new.html` — Removed Forge button from Data ribbon. Added 4 new graph types to Graph section (Violin, Bar, Heatmap, Bubble) with dialog configs and Plotly renderers. Changed Triage button to call inline `openTriagePanel()` instead of opening new tab. Added `autoTriageScan()` call after file upload — scans for missing values, outliers, type issues, Excel errors and renders inline triage panel with issue badges, column breakdown, and one-click fix options. Added `runTriageFixFromPanel()` that calls `/api/dsw/triage/`, re-uploads cleaned data, and refreshes the grid.
**Verification:** Upload CSV with missing data → triage panel auto-opens. Click Auto-Fix → data cleaned and reloaded. Graph section shows 9 chart types.

---

### 2026-02-10 — Learning: fix & expand "Run in DSW" integration
**Debt item:** N/A (feature fix + expansion)
**Files changed:**
- `services/svend/web/agents_api/learn_views.py` — Added `intro`, `exercise`, `sample_data` fields to `get_section()` API response (were missing — exercise blocks, Run in DSW buttons never rendered). SHARED_DATASET served as fallback when section has no sample_data.
- `services/svend/web/agents_api/dsw_views.py` — Added inline data acceptance to `run_analysis()`. New Source 0: if `body["data"]` is a dict, converts to DataFrame directly (capped at 10k rows). Existing data_id flow untouched.
- `services/svend/web/templates/learn.html` — Added Plotly 2.27.0 CDN. Rewrote `runInDSW()`: sends correct `{type, analysis, config, data}` format (was `{analysis_type, data}`). Parses `"type:analysis"` colon format. Added `formatDSWSummary()` for color tag rendering. Results now show formatted summary + Plotly charts instead of raw JSON.
- `services/svend/web/agents_api/learn_content.py` — Updated all 10 existing `dsw_type` values to colon format (`"stats:descriptive"` etc). Added `dsw_type` + `dsw_config` to 28 more sections (38 total, 7 conceptual sections skipped). Added `dsw_config` dicts specifying column names and parameters for each analysis.
**Verification:** `python3 manage.py check` — 0 issues. 38/45 sections have dsw_type, all in colon format. 7 conceptual sections correctly skipped.
**Commit:** pending

---

### 2026-02-10 — DSW diagnostic plots audit & gap closure
**Debt item:** N/A (quality gap)
**Files changed:**
- `services/svend/web/agents_api/dsw_views.py` — Added missing diagnostic plots to 9 analyses:
  - ttest: histogram with mean line, CI band, H₀ reference
  - ttest2: side-by-side box plots + statistics dict
  - paired_t: differences histogram with mean/zero lines + statistics dict
  - f_test: variance comparison bars + distribution box plots
  - normality: histogram with fitted normal curve overlay (alongside existing Q-Q)
  - box_cox: lambda vs log-likelihood profile (alongside existing before/after histograms)
  - classification: confusion matrix heatmap + ROC curve (alongside existing feature importance)
  - regression_ml: feature importance + residuals vs predicted (alongside existing actual vs predicted)
  - clustering: elbow plot with silhouette scores + best-k marker (alongside existing cluster scatter)
**Verification:** Run any t-test, f-test, normality, box-cox, classification, regression ML, or clustering analysis — all should produce diagnostic plots below the summary.

---

### 2026-02-10 — GP freeze fix & GAM chart limit fix
**Debt item:** N/A (bug fix)
**Files changed:**
- `services/svend/web/agents_api/dsw_views.py` — GP: Added 500-row subsample cap (was O(n³) with no limit), reduced n_restarts_optimizer to 2 for >300 rows. GAM: Removed hardcoded `features[:4]` limit on partial dependence plots, wrapped each plot in try/except for robustness.
**Verification:** GP with 1000+ rows should complete in ~2s. GAM should produce plots for all features, not just first 4.

---

### 2026-02-10 — A3 embedded diagrams from whiteboard
**Debt item:** N/A (feature)
**Files changed:**
- `services/svend/web/agents_api/models.py` — Added `embedded_diagrams` JSONField to A3Report model for storing SVG snapshots.
- `services/svend/web/agents_api/migrations/0017_add_a3_embedded_diagrams.py` — Migration for new field.
- `services/svend/web/agents_api/whiteboard_views.py` — Added `export_svg()` endpoint that renders whiteboard elements as inline SVG. Includes renderers for post-its, rectangles, ovals, diamonds, text, groups, fishbone diagrams, and connections.
- `services/svend/web/agents_api/whiteboard_urls.py` — Added `/boards/<code>/svg/` route.
- `services/svend/web/agents_api/a3_views.py` — Added `embed_diagram()` and `remove_diagram()` endpoints.
- `services/svend/web/agents_api/a3_urls.py` — Added embed-diagram and diagram removal routes.
- `services/svend/web/templates/a3.html` — Added "+ Diagram" buttons to sections (current_condition, root_cause, countermeasures). Added embed modal, diagram container CSS, and JavaScript for embedding/removing diagrams.
**Verification:** Create a whiteboard with elements → Create A3 for same project → Click "+ Diagram" in Root Cause section → Select whiteboard → Diagram should appear as embedded SVG.

---

### 2026-02-10 — A3 status dropdown for demo readiness
**Debt item:** N/A (feature gap)
**Files changed:**
- `services/svend/web/templates/a3.html` — Replaced static status badge with interactive dropdown. Added styling for `.a3-status-select`. Added `updateStatus()` function to persist status changes via API. Updated `loadReport()` to set dropdown value from report data.
**Verification:** Open an A3 report, change status from "Draft" to "In Progress" using the dropdown — should persist on page reload.

---

### 2026-02-07 — Learning section: "learn by doing" restructure
**Debt item:** N/A (UX overhaul)
**Files changed:**
- `services/svend/web/agents_api/models.py` — Removed Certificate model
- `services/svend/web/agents_api/learn_views.py` — Removed certificate system (CERTIFICATION_LEVELS, _generate_certificate, _get_certificate_data, _verify_certificate, get_certificate view, verify_certificate view). Updated docstring. Simplified assessment to not generate certificates.
- `services/svend/web/agents_api/learn_urls.py` — Removed certificate/ and certificate/verify/ URL routes
- `services/svend/web/agents_api/learn_content.py` — Removed certification references. Added `intro` and `exercise` fields to all 45 sections. Added SHARED_DATASET (200 manufacturing observations: diameter_mm, weight_g, roughness_ra, line, shift, defect). 10 sections have `dsw_type` for inline "Run in DSW" button. 6449 lines.
- `services/svend/web/agents_api/migrations/0016_remove_certificate.py` — Migration to drop learn_certificate table. Applied.
- `services/svend/web/templates/learn.html` — Major restructure:
  - New rendering pipeline: intro → exercise block → interactive widget (prominent) → "Run in DSW" button → collapsible "Go Deeper" → key takeaways → practice questions
  - Auto-extracts intro from first paragraph if no explicit intro field
  - Added exercise-block CSS, deep-dive collapsible, widget-prominent wrapper
  - Added toggleDeepDive(), runInDSW() (calls /api/dsw/analysis/ inline), markInteracted()
  - Interaction gating: complete button shows "Try the exercise first" until widget interaction
  - Removed all certificate HTML, CSS, and JS (showCertificate, shareCertificate, downloadCertificate)
  - Updated welcome text from "Certification" to "Learn by Doing"
  - Updated assessment header from "Certification Assessment" to "Knowledge Check"
**Verification:** `python3 manage.py check` — 0 issues. All 4 script blocks parse in Node.js. 4159 lines.

---

### 2026-02-07 — DOE/DSW Unification

**Debt item:** N/A (feature work)
**Files changed:**
- `services/svend/web/templates/workbench_new.html` — Unified DOE Experiment tab:
  - Expanded ribbon from 2 groups to 4: Create Design, Analyze DOE, Power, Assistant
  - Expanded openDOEDialog with all 11 design types (full factorial, fractional, PB, DSD, CCD, Box-Behnken, Taguchi, Latin Square, RCBD, D-optimal, I-optimal) plus conditional fields per type
  - Added editable Response column to DOE output table with Analyze Results, Import to Workbench, Export CSV buttons
  - Added 12 new JS functions: analyzeDOEResults, renderExperimenterAnalysis, importDOEToWorkbench (bridge to DSW), exportDOECSV, openDOEAnalysisDialog (main effects/interaction via DSW), openDOEContourDialog, openDOEOptimizeDialog, openDOEChatDialog, updateDOEConditionalFields, currentDoeDesign state
- `services/svend/web/agents_api/experimenter_views.py` — Bugfixes + deprecation:
  - Added deprecation comment on power_analysis (superseded by DSW 9-type calculator)
  - Fixed string-to-float conversion bugs in _find_optimal_settings, contour_plot, optimize_response (levels were strings, arithmetic failed)
  - Fixed desirability function to handle None bounds with sensible defaults
**Verification:** Navigate to /app/dsw/ → Experiment tab → all 4 ribbon groups visible; Create Design → all 11 types in dropdown; generate design → Response column editable; Analyze Results / Import to Workbench / Export CSV buttons work
**Commit:** pending

---

### 2026-02-08 — DOE JSON serialization fix
**Debt item:** N/A (bugfix)
**Files changed:**
- `services/svend/agents/experimenter/doe.py` — Added `_to_python()` helper to convert numpy int64/float64 to native Python types. Updated `to_dict()` to use it on all numeric fields (run_id, levels, coded, resolution, etc.)
- `services/svend/agents/agents/experimenter/doe.py` — Synced with same fix (duplicate directory)
**Verification:** `cd /home/eric/kjerne/services/svend/agents && python3 -c "from experimenter.doe import DOEGenerator, Factor; import json; json.dumps(DOEGenerator(42).full_factorial([Factor('T', [100.0, 150.0])]).to_dict())"` — no error
**Commit:** pending

---

### 2026-02-07 — A3 UI theme fix
**Debt item:** N/A (UI fix)
**Files changed:**
- `services/svend/web/templates/a3.html` — Fixed hardcoded white/light colors that didn't respect theme:
  - Changed CSS variables to use theme vars (--bg-card, --bg-secondary, --border, --text-primary)
  - Added light theme overrides for paper look when appropriate
  - Fixed status badges to use semi-transparent theme-aware colors
  - Fixed modal to use same pattern as other modals (#121a12 dark, #ffffff light, #12121f midnight)
  - Fixed import items, section headers, textareas to use theme colors
  - Added placeholder color styling
**Verification:** Navigate to /app/a3/ - should match app theme (dark/light/midnight)
**Commit:** pending

---

### 2026-02-07 — Learning section: 8 new content sections + interactive widgets
**Debt item:** N/A (feature expansion)
**Files changed:**
- `services/svend/web/agents_api/learn_content.py` — Added 8 new section content dicts: NONPARAMETRIC_TESTS, TIME_SERIES_ANALYSIS, SURVIVAL_RELIABILITY, ML_ESSENTIALS, MEASUREMENT_SYSTEMS, DOE_HANDS_ON, NONPARAMETRIC_HANDS_ON, TIME_SERIES_HANDS_ON. 4 sections include sample_data with fake datasets. Total sections: 45 (up from 37). Total practice questions: 82 (up from 71). Registered all 8 in SECTION_CONTENT.
- `services/svend/web/agents_api/learn_views.py` — Added "Advanced Methods" module (Module 8) with 5 sections (nonparametric, time series, survival, ML, measurement systems). Added 3 new hands-on sections to DSW Mastery module (DOE, nonparametric, time series). Renumbered Case Studies→9, Capstone→10. Removed Synara module (not public). Updated certification thresholds. Total: 10 modules, 47 sections.
- `services/svend/web/templates/learn.html` — Added 4 new interactive widget types with render/update functions: nonparametric_demo (Mann-Whitney U with fake data, box plots, p-value), timeseries_demo (decomposition with sparkline visualization), survival_demo (Kaplan-Meier SVG curve with censoring), clustering_demo (K-Means with scatter plot and silhouette score). Added widget cases to switch and initializeWidgets. Set window.currentSectionData for widget config access. File: 4070 lines, all 4 script blocks parse OK in Node.js.
**Verification:** `python3 manage.py check` — 0 issues. Content imports clean. Node.js parses all script blocks.

---

### 2026-02-07 — Projects UI: Charter Form and Structured Hypothesis
**Debt item:** N/A (UI update for charter structure)
**Files changed:**
- `services/svend/web/templates/projects.html` — Complete overhaul of project creation and display:
  - New "Create Project" modal now a full charter form with collapsible sections:
    - Problem Definition (5W2H): What/Where/When multi-input lists, magnitude, trend, since
    - Business Impact: financial, customer, quality, delivery, safety, regulatory
    - Goal Statement (SMART): metric, unit, baseline, target, deadline
    - Scope: in/out scope lists, constraints, assumptions
    - Team: champion, leader, team members with roles
    - Timeline: target completion, can experiment checkbox
  - New "Add Hypothesis" modal with structured format:
    - If/Then/Because clause inputs with auto-generated statement preview
    - Variables section: independent (X), dependent (Y), direction, magnitude
    - Testing plan: rationale, test method, success criteria
  - Project detail view now renders charter cards showing all structured fields
  - Hypothesis detail view shows structured clauses, variables, and testing info
  - Added helper functions: toggleSection, addListItem, getListValues, addTeamMember, updateHypothesisPreview
  - Added CSS for charter forms, clause labels, multi-input lists, charter display cards
- `services/svend/web/core/serializers.py` — Updated for new fields:
  - HypothesisSerializer: Added if_clause, then_clause, because_clause, variables, testing fields
  - ProjectListSerializer: Changed description to problem_statement
  - ProjectDetailSerializer: Added all charter fields (5W2H, impacts, goal, scope, team, timeline)
**Verification:** Navigate to /app/projects/, click "+ New Project" to see charter form. Create project and view detail.
**Commit:** pending

---

### 2026-02-07 — P1 gap closure: proportion tests, power calculators, MSA expansion
**Debt item:** DSW_gaps.md P1.1, P1.2, P1.3
**Files changed:**
- `services/svend/web/agents_api/dsw_views.py` — Added 18 new analysis methods: 4 proportion tests (prop_1sample, prop_2sample, fisher_exact, poisson_1sample), 9 power/sample-size calculators (power_z, power_1prop, power_2prop, power_1variance, power_2variance, power_equivalence, power_doe, sample_size_ci, sample_size_tolerance), 5 MSA methods (gage_rr_nested, gage_linearity_bias, gage_type1, attribute_gage, attribute_agreement). Added `import math`.
- `services/svend/web/templates/workbench_new.html` — Added proportion tests to more_nonparam and more dialogs; added power/MSA to more dialog; replaced old 3-option power dialog with comprehensive 9-calculator dialog
- `services/svend/web/templates/dsw.html` — Added all 18 methods to dropdown, form labels, needsVar2, and config builders
- `services/svend/web/templates/analysis_workbench.html` — Added all 18 methods to items arrays and config form builders
- `DSW_gaps.md` — Updated: Basic Statistics 95→100%, Power 50→85%, MSA 40→90%, overall 82→89%
**Verification:** `DJANGO_SETTINGS_MODULE=svend.settings python3 -c "..."` — 18/18 pass
**Commit:** pending

---

### 2026-02-07 — Learning section: wire up all interactive widgets
**Debt item:** N/A (feature)
**Files changed:**
- `services/svend/web/templates/learn.html` — Added 16 missing widget render functions (DSW Demo, SPC Demo, P-Value Simulator, CI Visualizer, Effect Size Calculator, Blocking Demo, Bias Detector, Distribution Explorer, EDA Explorer, Natural Experiment Demo, Paper Evaluator, Study Evaluator, Forest Plot Reader, Decision Framework, Project Planner, Capstone Workspace). Added helper functions (randNormal, normalCDF). Updated renderInteractiveWidget switch to dispatch all 32 widget types. Updated initializeWidgets to initialize new dynamic widgets. DSW Demo connects to live /api/dsw/analysis/ with client-side fallback. File grew from 2839 to 3542 lines, all script blocks balanced.
**Verification:** `python3 manage.py check` — pre-existing core.admin issue only. All 4 JS script blocks have balanced braces/parens/brackets.

---

### 2026-02-07 — Remove Knowledge page (prototype only)
**Debt item:** N/A (cleanup)
**Files changed:**
- `services/svend/web/templates/base_app.html` — Removed Knowledge link from navigation
- `services/svend/web/svend/urls.py` — Commented out /app/knowledge/ route
**Verification:** Navigation no longer shows "Knowledge" link
**Commit:** pending

---

### 2026-02-07 — Remove Coder agent from UI
**Debt item:** N/A (cleanup)
**Files changed:**
- `services/svend/web/templates/projects.html` — Removed "Open in Coder" buttons, "Explore in Coder" button, openCoder() function, updated text to remove Coder references
- `services/svend/web/templates/workbench.html` — Removed Coder tab, Coder form, coder switch case in runAgent, formatCoderResult function
- `services/svend/web/templates/workflows.html` — Removed Coder from step type dropdown, getStepConfigHtml, collectSteps, typeIcons
- `services/svend/web/agents_api/urls.py` — Commented out coder route
**Verification:** Navigate to /app/projects/ — no Coder references. Workbench has no Coder tab.
**Commit:** pending

---

### 2026-02-07 — Restructure Project as Charter, Hypothesis as If/Then/Because
**Debt item:** N/A (schema redesign)
**Files changed:**
- `services/svend/web/core/models/project.py` — Complete rewrite as Project Charter with ~50 fields:
  - Problem Definition (5W2H): problem_whats, problem_wheres, problem_whens (JSONField lists), problem_magnitude, problem_trend, problem_since
  - Business Impact: impact_financial, impact_customer, impact_safety, impact_quality, impact_regulatory, impact_delivery, impact_other
  - Goal Statement (SMART): goal_statement, goal_metric, goal_baseline, goal_target, goal_unit, goal_deadline
  - Scope: scope_in, scope_out (JSONField lists), constraints, assumptions
  - Team: champion_name, champion_title, leader_name, leader_title, team_members (JSONField)
  - Timeline: milestones (JSONField), target_completion, phase_history
  - Resolution: resolution_summary, resolution_actions, resolution_verification
  - Removed: description, available_data, effect_description, effect_magnitude, stakeholders
  - Added helper methods: generate_problem_statement(), generate_goal_statement()
- `services/svend/web/core/models/hypothesis.py` — Restructured with If/Then/Because format:
  - Structured: if_clause, then_clause, because_clause (TextField)
  - Variables: independent_variable, independent_var_values, dependent_variable, dependent_var_unit, predicted_direction, predicted_magnitude
  - Testing: rationale, test_method, success_criteria, data_requirements (JSONField)
  - Removed: mechanism field
  - Added generate_statement() method
  - Added project FK to Evidence model for easier querying
- `services/svend/web/core/admin.py` — Registered all core models (Project, Hypothesis, Evidence, EvidenceLink, Dataset, ExperimentDesign)
- `services/svend/web/core/migrations/0004_charter_structure.py` — Migration with all field changes
**Verification:** `python3 manage.py migrate core` — applied successfully. Check admin at /admin/core/
**Commit:** pending

---

### 2026-02-07 — Learning section: practice questions for all 37 sections
**Debt item:** N/A (content enhancement)
**Files changed:**
- `services/svend/web/agents_api/learn_content.py` — Added practice questions to all 22 sections that lacked them. Total practice questions: 71 (up from 16). Every section now has 1-2 scenario-based practice questions with detailed answers and hints.
**Verification:** `python3 -c "from agents_api.learn_content import SECTION_CONTENT"` loads cleanly. All 37 sections have practice_questions.

---

### 2026-02-07 — Learning section persistence models
**Debt item:** N/A (new feature)
**Files changed:**
- `services/svend/web/agents_api/models.py` — Added SectionProgress, AssessmentAttempt, Certificate models with UUID PKs, indexes, constraints
- `services/svend/web/agents_api/learn_views.py` — Replaced all stub helper functions with real ORM-backed implementations
- `services/svend/web/agents_api/migrations/0015_learning_models.py` — Migration created and applied
**Verification:** `python3 manage.py check` — 0 issues. Migration applied successfully.

---

### 2026-02-07 — VSM delay types, supermarket, FIFO, and push/pull flow arrows
**Debt item:** N/A (feature enhancement)
**Files changed:**
- `services/svend/web/templates/vsm.html`:
  - Added new palette sections: "Delays & Buffers" (Inventory, Queue, Transport, Batch Wait, Supermarket, FIFO) and "Material Flow" (Push, Pull)
  - Added CSS for .flow-palette, .flow-item, .sidebar-hint, supermarket and FIFO elements
  - Updated renderInventory() to use delay type colors (inventory=warning, queue=amber, transport=purple, batch=pink) with icons
  - Added renderSupermarket() - shelves icon with horizontal lines
  - Added renderFIFO() - horizontal lane with arrow and "FIFO" label
  - Added setFlowTool() and handleFlowClick() for drawing push/pull connections
  - Added addMaterialFlow() to save connections to material_flow array
  - Updated renderConnections() to show push (striped gray arrow) vs pull (solid green with kanban signal)
  - Updated startDragElement() to accept element type and handle flow clicks
  - Added showInventoryProperties() for editing delay/buffer elements
  - Added delay type selector and days of supply input to properties panel
  - Updated saveProperties() to handle both process and inventory elements
  - Lead time ladder now color-codes by delay type
**Verification:**
- Drag different delay types from palette → see different colored triangles with icons
- Drag Supermarket → see shelf icon
- Drag FIFO Lane → see horizontal box with arrow
- Click Push/Pull in Material Flow, then click two process boxes → see connection
- Push = striped gray arrow, Pull = solid green with "K" signal
- Double-click inventory → see delay type dropdown and days of supply input
**Commit:** pending

---

### 2026-02-07 — VSM tool refinement: data points and lead time ladder
**Debt item:** N/A (feature enhancement)
**Files changed:**
- `services/svend/web/templates/vsm.html`:
  - Enhanced properties panel with 2-column layout for: C/T, C/O, Uptime, Operators, Batch Size, Scrap Rate, Available Time, Shifts
  - Updated showProperties/saveProperties to handle new fields
  - Expanded process box from 120x100 to 130x140 to display 7 metrics
  - Added formatTime() helper for human-readable time display
  - Added renderLeadTimeLadder() function that draws timeline below process flow:
    - Elevated rectangles (orange) for wait/inventory time
    - Depressed rectangles (green) for cycle/value-add time
    - Shows time labels for each segment
    - Displays totals: Lead Time, Process Time, PCE%
**Verification:**
- Double-click a process box → see all 8 property fields
- Process boxes show C/T, C/O, Uptime, Batch, Scrap, Ops, Shifts
- Lead time ladder appears below process flow
- Elevated = wait time (inventory days), Depressed = cycle time
**Commit:** pending

---

### 2026-02-07 — Enhanced AI Guide with project context
**Debt item:** N/A (feature enhancement)
**Files changed:**
- `services/svend/web/templates/workbench_new.html` — Added currentProjectData variable, loadProjectData() function to fetch full project details when project selected, enhanced buildAIContext() to include project title, problem statement, hypotheses with probabilities, and evidence counts
- `services/svend/web/agents_api/guide_views.py` — Updated DSW system prompt to mention hypothesis evaluation and likelihood ratios, enhanced context handling to structure project data with hypotheses for LLM
**Verification:**
- Select a project in DSW with hypotheses defined
- Open AI Guide panel and ask about your data
- Assistant should reference project hypotheses and help evaluate evidence
**Commit:** pending

---

### 2026-02-07 — Project linkages across all tools
**Debt item:** N/A (integration feature)
**Files changed:**
- `services/svend/web/templates/workbench_new.html` — Added project selector dropdown in header with currentProjectId tracking, URL param reading, and project linking when running analyses
- `services/svend/web/templates/a3.html` — Added URL param reading for ?project= to auto-select project when creating new A3
- `services/svend/web/templates/vsm.html` — Added project selector in sidebar with CSS, currentProjectId tracking, URL param reading, project linking on create/save
- `services/svend/web/templates/whiteboard.html` — Added project selector in toolbar with CSS, currentProjectId tracking, URL param reading, project linking on create/update
- `services/svend/web/agents_api/whiteboard_views.py` — Added project_id handling in update_board()
- `services/svend/web/agents_api/vsm_views.py` — Added project_id handling in update_vsm()
**Verification:**
- Each tool (DSW, Whiteboard, A3, VSM) shows project selector
- Selecting a project updates URL param and saves link
- Creating new artifacts from project hub (via ?project=) auto-selects project
- Linked artifacts appear in project hub
**Commit:** pending

---

### 2026-02-07 — Project Hub/Dashboard with linked tools
**Debt item:** N/A (integration feature)
**Files changed:**
- `services/svend/web/templates/projects.html` — Updated viewProject to fetch from /hub/ endpoint instead of detail. Added 4 new tool sections (DSW Analyses, Whiteboards, A3 Reports, VSM Maps) with tool-card UI. Added renderLinkedTools() and per-tool render functions. Updated Knowledge Graph summary to show tool counts. Added tool-list CSS.
- `services/svend/web/core/views.py` — (previously) Added project_hub endpoint returning project details + linked tools + counts
- `services/svend/web/core/urls.py` — (previously) Added projects/<id>/hub/ route
**Verification:**
- Navigate to /app/projects/, click a project → should see DSW Analyses, Whiteboards, A3 Reports, VSM Maps sections
- Knowledge Graph summary shows counts for all tool types
- Tool cards clickable, navigate to respective tools
**Commit:** pending

---

### 2026-02-07 — Workbench ribbon reorganization
**Debt item:** N/A (UI cleanup)
**Files changed:**
- `services/svend/web/templates/workbench_new.html` — removed Thinking and Process tabs from ribbon (now 4 tabs: Data, Analysis, Experiment, ML). Redesigned Analysis tab into 7 groups with proper SVG icons on every button: Control Charts (X-bar R, I-MR, P, C) | Capability (Cp/Cpk, Gage R&R, Sampling) | Parametric (Regression, ANOVA, t-Test, Post-Hoc) | Non-Parametric (Rank Tests, Diagnostics) | Multivariate (MANOVA, Survival) | All Tests. Replaced flat 24-item "More..." dropdown with 6 categorized sub-dialogs (Non-Parametric, Post-Hoc, Multivariate, Survival, Acceptance, Diagnostics) sharing a single dialogConfigs handler. Full optgroup-organized "All Tests" dialog as catch-all. Custom SVG icons: step-function for Survival, overlapping ellipses for Multivariate, gauge for Diagnostics, rank dots for Non-Parametric, bar comparison for Post-Hoc, grid for All Tests.
**Verification:**
- `python3 manage.py check` — 0 issues
- Ribbon tabs: Data | Analysis | Experiment | ML
- Analysis groups: Control Charts | Capability | Parametric | Non-Parametric | Multivariate | All Tests
**Commit:** pending

---

### 2026-02-07 — Acceptance sampling + Multivariate SPC (Hotelling T²)
**Debt item:** Minitab feature parity — quality/SPC gaps
**Files changed:**
- `services/svend/web/agents_api/dsw_views.py` — added acceptance sampling (`acceptance_sampling`): single/double sampling plans, OC curve, AOQ curve with AOQL, producer/consumer risk, ATI calculation. No dataset required.
- `services/svend/web/agents_api/spc.py` — added `hotelling_t_squared_chart()`: T² statistic per observation, F-distribution UCL, variable contribution analysis, correlation matrix, out-of-control detection
- `services/svend/web/agents_api/spc_views.py` — added T-squared dispatch in both `control_chart()` and `analyze_uploaded()` endpoints; added T² to `chart_types()` registry
- `services/svend/web/templates/spc.html` — added T² to chart type dropdown, help text, and multivariate parseData
- `services/svend/web/templates/dsw.html` — added acceptance sampling dropdown, labels, config
- `services/svend/web/templates/analysis_workbench.html` — added acceptance sampling catalog + config form (plan type, n, Ac, lot size, AQL/LTPD)
- `services/svend/web/templates/workbench_new.html` — added acceptance sampling to dropdown
**Verification:**
- `python3 manage.py check` — 0 issues
- T²: 50 obs × 3 vars, 2 injected outliers detected, UCL=8.94, correct correlation matrix
- Acceptance (single): n=50, Ac=2, Pa@AQL=0.986, Pa@LTPD=0.540, 2 plots (OC + AOQ)
- Acceptance (double): n1=30/c1=1/r1=4/n2=30/c2=4, Pa@AQL=0.9996
**Commit:** pending

---

### 2026-02-07 — Survival analysis (Kaplan-Meier + Cox PH) and Discriminant Analysis (LDA/QDA)
**Debt item:** Minitab feature parity — survival/reliability (was ~50% parity), classification (new)
**Files changed:**
- `services/svend/web/agents_api/dsw_views.py` — replaced basic KM with full implementation: Greenwood CIs, log-rank test, backwards-compat config keys. Added Cox PH using statsmodels PHReg: hazard ratios, forest plot, concordance index, automatic categorical dummy coding. Added discriminant analysis (LDA/QDA) in `run_ml_analysis()`: confusion matrix, discriminant space projection, classification report, CV accuracy
- `services/svend/web/templates/dsw.html` — added dropdown options, needsVar2, label updates, config mapping for all 3 new analyses
- `services/svend/web/templates/analysis_workbench.html` — added catalog entries + config forms: KM (time, event, group selectors), Cox PH (time, event, covariate checkboxes), discriminant (group target, predictor checkboxes, LDA/QDA selector)
- `services/svend/web/templates/workbench_new.html` — added all 3 to More Tests dropdown
**Verification:**
- `python3 manage.py check` — 0 issues
- KM single: n=100, median=28.06, 1 plot with CI bands
- KM grouped: log-rank p=0.0001 (correctly detects exp(20) vs exp(40) difference)
- Cox PH: age HR=1.031 (p=0.002), treatment HR=0.456 (p<0.001), C-index=0.634
- LDA: test accuracy=0.967, CV accuracy=0.987, 2 plots (confusion matrix + LD space)
- QDA: test accuracy=0.967, CV accuracy=0.987
- Old KM config keys ('time'/'event') still work (backwards compat)
**Commit:** pending

---

### 2026-02-07 — DSW + LLM integration, A3 import from DSW
**Debt item:** N/A (new feature — Tools→Methods→Knowledge architecture)
**Files changed:**
- `services/svend/web/agents_api/models.py` — added `project` FK and `title` field to DSWResult, plus `get_summary()` method for import previews
- `services/svend/web/agents_api/dsw_views.py` — `run_analysis()` now accepts `project_id`, `title`, `save_result` params; saves DSWResult when linked to project
- `services/svend/web/agents_api/a3_views.py` — added DSWResult import: `get_a3_report()` returns `dsw_results` in available_imports; `import_to_a3()` handles `source_type="dsw"`
- `services/svend/web/templates/a3.html` — added DSW import buttons to Current Condition and Root Cause sections; added DSW handler in `showImport()`
- `services/svend/web/templates/workbench_new.html` — added collapsible AI Assistant panel: chat interface, context-aware prompts (sends data summary + recent analyses), rate limit display; 180 lines of CSS + 130 lines of JS
- `services/svend/web/agents_api/migrations/0014_dsw_result_project_link.py` — migration for DSWResult.project and .title
**Verification:**
- `python3 manage.py check` — 0 issues
- Migrations applied successfully
**Commit:** pending

---

### 2026-02-07 — Navigation reorganization + VSM tool
**Debt item:** N/A (new feature)
**Files changed:**
- `services/svend/web/templates/base_app.html` — reorganized nav into dropdown menus: Analysis (DSW, SPC, DOE, Forecast, Models), Visual (Whiteboard, VSM), Methods (A3, DMAIC, 8D, 5-Why); added CSS for disabled menu items
- `services/svend/web/agents_api/models.py` — added ValueStreamMap model with process steps, inventory, information/material flow, kaizen bursts, and metrics calculation
- `services/svend/web/agents_api/vsm_views.py` — new file: CRUD endpoints for VSM, add process step/inventory/kaizen, create future state, compare states
- `services/svend/web/agents_api/vsm_urls.py` — new file: VSM API routes
- `services/svend/web/svend/urls.py` — added VSM template routes (/app/vsm/) and API routes (/api/vsm/)
- `services/svend/web/templates/vsm.html` — new file: VSM editor with drag-drop elements, process boxes with metrics, inventory triangles, kaizen bursts, timeline metrics (lead time, process time, PCE)
- `services/svend/web/agents_api/migrations/0013_add_value_stream_map.py` — migration for ValueStreamMap model
**Verification:**
- `python3 manage.py check` — 0 issues
- Migration applied successfully
**Commit:** pending

---

### 2026-02-07 — Regularized regression (Ridge/LASSO/Elastic Net)
**Debt item:** Minitab feature parity — ML/regression (was ~75% parity)
**Files changed:**
- `services/svend/web/agents_api/dsw_views.py` — added regularized regression (`regularized_regression`) in `run_ml_analysis()` using sklearn RidgeCV, LassoCV, ElasticNetCV: cross-validated alpha selection, coefficient bar plot, actual vs predicted scatter, R²/MSE/MAE metrics, feature importance ranking
- `services/svend/web/templates/analysis_workbench.html` — added catalog entry in ML menu with config form (response selector, predictor checkboxes, method dropdown: Ridge/LASSO/Elastic Net)
- `services/svend/web/templates/dsw.html` — added dropdown option
- `services/svend/web/templates/workbench_new.html` — added to More Tests dropdown
**Verification:**
- `python3 manage.py check` — 0 issues
- End-to-end: 200-row synthetic data with 5 true + 5 noise features. LASSO R²=0.933, CV R²=0.911, correctly identified all 5 true features, α=0.0221
**Commit:** pending

---

### 2026-02-07 — SARIMA seasonal forecasting
**Debt item:** Minitab feature parity — time series (was 70% parity)
**Files changed:**
- `services/svend/web/agents_api/dsw_views.py` — added SARIMA (`sarima`) using statsmodels SARIMAX: (p,d,q)(P,D,Q)[m] seasonal orders, ADF stationarity test, parameter table, Ljung-Box residual test, forecast with 95% CI, residual diagnostics plot
- `services/svend/web/templates/analysis_workbench.html` — added SARIMA to timeseries catalog with full config form (p,d,q,P,D,Q,m dropdowns, seasonal period selector)
- `services/svend/web/templates/dsw.html` — added dropdown option
- `services/svend/web/templates/workbench_new.html` — added to More Tests dropdown
**Verification:**
- `python3 manage.py check` — 0 issues
- End-to-end: 72 months synthetic data with trend + 12-month seasonality, SARIMA(1,0,1)(1,1,1)[12]: AIC=200.7, Ljung-Box p=0.20 (good fit), 2 plots
**Commit:** pending

---

### 2026-02-07 — Nested ANOVA (mixed-effects model)
**Debt item:** Minitab feature parity — hierarchical/mixed-effects models
**Files changed:**
- `services/svend/web/agents_api/dsw_views.py` — added nested ANOVA (`nested_anova`) using statsmodels mixedlm: fixed effects table, variance components (random + residual), ICC (intraclass correlation), REML estimation, convergence check
- `services/svend/web/templates/dsw.html` — added dropdown option, label updates, config mapping
- `services/svend/web/templates/analysis_workbench.html` — added catalog entry + 3-field config form (response, fixed factor, random factor)
- `services/svend/web/templates/workbench_new.html` — added to More Tests dropdown
**Verification:**
- `python3 manage.py check` — 0 issues
- End-to-end: 3 machines × 4 operators × 5 replicates, nested design: ICC=0.59 (operators account for 59% variance), machine effect not significant after nesting
**Commit:** pending

---

### 2026-02-07 — A3 Report method (Toyota-style problem solving)

**Debt item:** N/A (new feature - Methods architecture)

**Files changed:**
- `services/svend/web/agents_api/models.py` — added A3Report model with 7 sections (background, current_condition, goal, root_cause, countermeasures, implementation_plan, follow_up), status tracking, import references
- `services/svend/web/agents_api/a3_views.py` (new) — A3 API: list, create, get, update, delete, import_to_a3, auto_populate_a3
- `services/svend/web/agents_api/a3_urls.py` (new) — URL routing for A3 API
- `services/svend/web/templates/a3.html` (new) — A3 report UI with paper-like layout, import modal, auto-fill with AI
- `services/svend/web/svend/urls.py` — added `/api/a3/` and `/app/a3/` routes
- `services/svend/web/agents_api/migrations/0012_a3_report.py` — A3Report migration
- `services/svend/web/agents_api/whiteboard_views.py` — fixed to use `core.Hypothesis` instead of `workbench.Hypothesis`
- `services/svend/web/agents_api/guide_views.py` — fixed to use `core.Project` instead of `workbench.Project`

**A3 features:**
- CRUD operations for A3 reports
- Import from: hypotheses → root_cause, whiteboard → root_cause/countermeasures, project → background
- Auto-populate sections using LLM (rate-limited)
- Print-friendly CSS for PDF export
- Linked to `core.Project` (consistent with Board, Hypothesis)

**API endpoints:**
- `GET /api/a3/` — list reports
- `POST /api/a3/create/` — create new
- `GET /api/a3/<id>/` — get with available imports
- `PUT /api/a3/<id>/update/` — update sections
- `DELETE /api/a3/<id>/delete/` — delete
- `POST /api/a3/<id>/import/` — import from tool to section
- `POST /api/a3/<id>/auto-populate/` — AI fill sections

**Verification:**
- `python3 manage.py check` — 0 issues
- Created test A3 "Seal Failure Investigation" linked to test project
- Hypotheses and boards available for import

**Commit:** pending

---

### 2026-02-07 — Multivariate tests: Hotelling's T² and MANOVA
**Debt item:** Minitab feature parity — multivariate analysis (was 10% parity, biggest gap)
**Files changed:**
- `services/svend/web/agents_api/dsw_views.py` — added 2 multivariate analysis types:
  1. **Hotelling's T²** (`hotelling_t2`): multivariate two-sample test, pooled covariance, F-approximation, radar/profile plot of group means
  2. **MANOVA** (`manova`): one-way multivariate ANOVA with all 4 test statistics (Wilks' Lambda, Pillai's Trace, Hotelling-Lawley Trace, Roy's Largest Root), F-approximations, eigenvalue decomposition, centroid scatter plot
- `services/svend/web/templates/dsw.html` — added options, updated form logic for multi-response selection
- `services/svend/web/templates/analysis_workbench.html` — added catalog entries + checkbox-based multi-response config form
- `services/svend/web/templates/workbench_new.html` — added options to More Tests dropdown
**Verification:**
- `python3 manage.py check` — 0 issues
- End-to-end: 3-group × 3-response synthetic data
  - Hotelling's T²: T² = 126.63, F = 40.76, p < 0.001
  - MANOVA: Wilks' Λ = 0.400, Pillai's V = 0.635, all p < 0.001
**Commit:** pending

---

### 2026-02-07 — Post-hoc tests: Tukey HSD, Dunnett, Games-Howell, Dunn's
**Debt item:** Minitab feature parity — ANOVA post-hoc comparisons
**Files changed:**
- `services/svend/web/agents_api/dsw_views.py` — added 4 post-hoc analysis types before `return result` in `run_statistical_analysis()`:
  1. **Tukey HSD** (`tukey_hsd`): pairwise comparisons with family-wise error control, CI plot
  2. **Dunnett's** (`dunnett`): each treatment vs control group, uses `scipy.stats.dunnett` with Bonferroni fallback
  3. **Games-Howell** (`games_howell`): pairwise comparisons without equal variance assumption, Studentized Range distribution
  4. **Dunn's** (`dunn`): non-parametric post-hoc for Kruskal-Wallis, rank-based with Bonferroni correction and tie correction
  - Also added post-hoc suggestion hint to one-way ANOVA significant results
- `services/svend/web/templates/dsw.html` — added 4 options to test type dropdown, updated `needsVar2`, labels, and config mapping
- `services/svend/web/templates/analysis_workbench.html` — added 4 items to stats analysis catalog, added config form builders
- `services/svend/web/templates/workbench_new.html` — added 4 options to More Tests dropdown
**Verification:**
- `python3 manage.py check` — 0 issues
- `py_compile` — passes
- End-to-end: 4-group synthetic data (means 50/55/52/60), all 4 tests produce correct results:
  - Tukey: 3/6 significant (D differs from A, B, C)
  - Dunnett vs A: B and D differ from control
  - Games-Howell: 4/6 significant (more sensitive with unequal variances)
  - Dunn's: 3/6 significant (rank-based, Bonferroni-adjusted)
**Commit:** pending

---

### 2026-02-07 — Guide API with rate-limited LLM access

**Debt item:** N/A (new feature)

**Files changed:**
- `services/svend/web/agents_api/models.py` — added LLMUsage model for tracking requests/tokens per user per day, LLM_RATE_LIMITS dict, check_rate_limit() function
- `services/svend/web/agents_api/llm_manager.py` — updated chat() to enforce rate limits and track usage
- `services/svend/web/agents_api/guide_views.py` (new) — Guide API endpoints:
  - `guide_chat()` — general chat with context (dsw, whiteboard, project, general)
  - `summarize_project()` — generate CAPA/8D/A3 reports from project data
  - `rate_limit_status()` — check remaining requests
- `services/svend/web/agents_api/guide_urls.py` (new) — URL routing for Guide API
- `services/svend/web/agents_api/migrations/0011_llm_usage_tracking.py` — LLMUsage migration
- `services/svend/web/svend/urls.py` — added guide API route

**Rate limits by tier:**
| Tier | Model | Requests/day |
|------|-------|--------------|
| Free | Haiku | 10 |
| Founder | Haiku | 50 |
| Pro | Sonnet | 200 |
| Team | Sonnet | 500 |
| Enterprise | Opus | 10000 |

**API endpoints:**
- `POST /api/guide/chat/` — chat with context
- `POST /api/guide/summarize/` — project → report (CAPA, 8D, A3, custom)
- `GET /api/guide/rate-limit/` — check usage/remaining

**Verification:**
- `python3 manage.py check` — 0 issues
- Guide views import successfully
- Migration applied

**Commit:** pending

---

### 2026-02-07 — Whiteboard collaboration + If-Then causal + Tools → Methods architecture

**Debt item:** N/A (new feature + architecture documentation)

**Files changed:**
- `services/svend/web/agents_api/models.py` — added Board, BoardParticipant, BoardVote models for collaborative whiteboards with room codes, version tracking, and dot voting
- `services/svend/web/agents_api/whiteboard_views.py` (new) — complete whiteboard API: create_board, get_board, update_board, toggle_voting, add_vote, remove_vote, list_boards, delete_board, export_hypotheses
- `services/svend/web/agents_api/whiteboard_urls.py` (new) — URL routing for whiteboard API including export-hypotheses endpoint
- `services/svend/web/svend/urls.py` — added whiteboard API and room code URL patterns
- `services/svend/web/templates/whiteboard.html`:
  - Added collaboration UI (room code display, participant cursors, voting badges, share button)
  - Added collaboration JS (polling sync, conflict detection, voting)
  - Added If-Then causal connections (orange, thicker, IF/THEN labels)
  - Added AND/OR gate elements for compound logic
  - Added causal connection tool (keyboard shortcut: I)
  - Added `getCausalRelationships()` and `exportCausalAsHypotheses()` for hypothesis export
  - Added export button for causal → hypothesis conversion
- `services/svend/web/agents_api/migrations/0009_whiteboard_models.py` — Board, BoardParticipant, BoardVote migration
- `services/svend/web/agents_api/migrations/0010_board_project_link.py` — Board.project FK migration
- `services/svend/reference_docs/ARCHITECTURE.md` — added "Tools → Methods → Knowledge Architecture" section documenting separation of Tools (DSW, Whiteboard) from Methods (A3, DMAIC, 5-Why) with import/export flows

**If-Then causal connections:**
- New tool: If-Then connection (I key) - distinct from regular arrows
- Visual: orange color, thicker stroke, IF/THEN labels at endpoints
- AND/OR gates: compound condition elements (multiple inputs → one output)
- Export function: extracts causal relationships as hypothesis candidates
- API endpoint: `POST /api/whiteboard/boards/<room_code>/export-hypotheses/` creates Hypothesis objects linked to board's project
- Frontend calls API with confirmation dialog, handles duplicates gracefully
- Supports the Whiteboard → Knowledge flow in architecture

**Architecture vision:**
- Tools (DSW=quantitative, Whiteboard=qualitative) generate Knowledge
- Methods (A3, DMAIC, 5-Why, 8D, Kaizen) orchestrate and structure
- Knowledge artifacts (Hypotheses, Evidence, Conclusions) persist at project level
- Import/export flows enable bi-directional movement (e.g., Whiteboard if-then → Hypothesis, DSW summary → A3 Analysis)
- LLM summarization layer (Qwen) translates raw tool output to method-consumable summaries

**Verification:**
- `python3 manage.py check` — 0 issues
- Whiteboard models import successfully
- ARCHITECTURE.md section readable

**Commit:** pending

---

### 2026-02-06 — Lock down repo for public push
**Debt item:** [REPO] svend.db + snapshot tar.gz tracked in git
**Files changed:**
- `.gitignore` — added `*.db`, `*.tar.gz`
- `.kjerne/DEBT.md` — added 22 tracked debt items from full audit
- `services/svend/agents/agents/site/data/svend.db` — removed from git tracking (file kept on disk)
- `.kjerne/snapshots/**/*.tar.gz` (10 files) — removed from git tracking (files kept on disk)
**Verification:** `git status` shows clean, `git ls-files '*.db' '*.tar.gz'` returns empty
**Commit:** 9c9396e

---

### 2026-02-06 — Add project documentation and debt closure process
**Debt item:** N/A (infrastructure)
**Files changed:**
- `CLAUDE.md` (new) — root-level architecture documentation: module map, data model (both current + target), API surface, integration pattern, serving config, working conventions
- `log.md` (new) — change log for all edits
- `DEBT-001.md` (new) — repeatable process for closing technical debt: pick → document → change → test → log → update DEBT.md → commit → push. Includes P1 dependency map.
**Verification:** files exist and are readable
**Commit:** 2a3c2b6

---

### 2026-02-06 — P1: DSW ↔ Evidence integration
**Debt item:** [DSW] No integration with Projects/Evidence
**Files changed:**
- `services/svend/web/agents_api/dsw_views.py` — added `problem_id` support to `run_analysis()` (line ~1038) and `dsw_from_data()` (line ~399). When `problem_id` is in the request body, analysis results are linked as evidence via `add_finding_to_problem()`. Uses `guide_observation` for summary (falls back to cleaned `summary` text). Maps analysis types to evidence types (stats/ml/bayesian/spc → data_analysis, viz → observation).
- `services/svend/web/agents_api/tests.py` — added `EvidenceIntegrationTest` class with 6 tests: Problem.add_evidence(), add_finding_to_problem() helper, invalid/empty ID handling, DSW with/without problem_id.
**Verification:**
- `python3 manage.py check` — 0 issues
- `py_compile` — both files pass
- End-to-end test: created Problem → added evidence via add_finding_to_problem() → verified 2 evidence items → cleaned up. PASSED.
**Commit:** 0eef3fb

---

### 2026-02-06 — P1: Experimenter ↔ Evidence integration
**Debt item:** [EXPERIMENTER] Only 2/9 endpoints create evidence
**Files changed:**
- `services/svend/web/agents_api/experimenter_views.py` — added `problem_id` support to 4 additional endpoints:
  - `power_analysis()` — "Power analysis (test_type): need N=X for effect d=Y"
  - `design_experiment()` — "Generated {type} design: N runs, K factors"
  - `contour_plot()` — "Response surface: optimal at X=val, Y=val (predicted=Z)"
  - `optimize_response()` — "DOE optimization: desirability=X, settings: ..."
  - Skipped `doe_guidance_chat` (chat interface, not analysis results), `design_types` and `available_models` (read-only metadata).
**Verification:**
- `python3 manage.py check` — 0 issues
- `py_compile` — passes
- All 4 endpoints follow the exact same pattern as existing `full_experiment` and `analyze_results`
**Commit:** 0eef3fb

---

### 2026-02-06 — P1: Phase 1 model migration (Problem → core.Project dual-write)
**Debt item:** [CORE] agents_api.Problem → core.Project migration
**Files changed:**
- `services/svend/web/agents_api/models.py` — added `core_project` FK field to Problem, 4 sync methods: `ensure_core_project()`, `sync_hypothesis_to_core()`, `sync_evidence_to_core()`, `_find_core_hypothesis()`
- `services/svend/web/agents_api/migrations/0008_add_core_project_fk.py` — migration adding core_project FK column
- `services/svend/web/agents_api/problem_views.py` — added dual-write calls to 6 write paths: `problems_list()` POST, `add_hypothesis()`, `add_evidence()`, `reject_hypothesis()`, `resolve_problem()`, `generate_hypotheses()`
- `services/svend/web/agents_api/tests.py` — added `DualWriteMigrationTest` class with 4 tests: ensure_core_project, sync_hypothesis, sync_evidence_with_links, find_core_hypothesis
**Data migration:**
- Existing "Employee Turnover" Problem (5 hypotheses, 0 evidence) migrated to core.Project with 5 core.Hypothesis records
**Verification:**
- `python3 manage.py check` — 0 issues
- `py_compile` — all files pass
- End-to-end test: created Problem → ensure_core_project → sync_hypothesis → sync_evidence → verified EvidenceLink + Bayesian update (0.6 → 0.73) → cleaned up. PASSED.
- Verified all 6 view write paths have dual-write wired in via `inspect.getsource()`
- Employee Turnover: core.Project created, 5 hypotheses synced
**Commit:** f4fb8db

---

### 2026-02-06 — P1: Synara persistence to Django ORM
**Debt item:** [SYNARA] In-memory only — state lost on server restart
**Files changed:**
- `services/svend/web/core/models/project.py` — added `synara_state` JSONField to Project model
- `services/svend/web/core/migrations/0003_add_synara_state.py` — migration adding synara_state column
- `services/svend/web/agents_api/synara_views.py` — replaced in-memory `_synara_instances` dict with DB-backed `_synara_cache` + `save_synara()`. Added `_resolve_project()` to resolve both Project and Problem UUIDs. Added `save_synara()` calls to all 9 mutating endpoints.
- `services/svend/web/agents_api/tests.py` — added `SynaraPersistenceTest` class with 3 tests: save/load round-trip, Problem UUID resolution, evidence-belief persistence.
**Verification:**
- `python3 manage.py check` — 0 issues
- `py_compile` — all files pass
- End-to-end test: created Synara → add hypothesis → add evidence → save → clear cache → reload → verified hypothesis/evidence/posterior survived round-trip. PASSED.
- Problem-to-Project resolution: Problem UUID → follow FK → save to core.Project. PASSED.
**Commit:** 841af3d

---

### 2026-02-06 — P2: SPC evidence integration + re-enable agents
**Debt items:** [SPC] 3/7 endpoints, [AGENTS] Coder/Researcher disabled
**Files changed:**
- `services/svend/web/agents_api/spc_views.py` — added `problem_id` support to `statistical_summary()` and `recommend_chart()`. Updated existing 3 endpoints to use `write_context_file()` and `evidence_type="data_analysis"` for consistency.
- `services/svend/web/agents_api/urls.py` — uncommented researcher and coder agent routes
- `services/svend/web/agents_api/views.py` — added `importlib.util` shim to pre-load agent core modules (`core.intent`, `core.search`, `core.verifier`, etc.) in dependency order, fixing namespace collision with Django's `core` app. All 3 agents (researcher, coder, writer) now import successfully.
**Verification:**
- `python3 manage.py check` — 0 issues
- `py_compile` — all files pass
- Agent imports: ResearchAgent ✓, CodingAgent ✓, WriterAgent ✓
- URL resolution: `/api/agents/researcher/` ✓, `/api/agents/coder/` ✓
- Researcher endpoint made actual search API calls (arXiv, Semantic Scholar) confirming full integration
**Commit:** 2888c32

---

### 2026-02-06 — P2: Synara DSL parser and belief engine test coverage
**Debt item:** [SYNARA] No test coverage for DSL parser or belief engine
**Files changed:**
- `services/svend/web/agents_api/tests.py` — added 46 unit tests across 9 test classes:
  - `KernelHypothesisRegionTest` (4 tests): matches_context full/partial/neutral, to_dict/from_dict roundtrip
  - `KernelEvidenceTest` (1 test): to_dict/from_dict roundtrip
  - `KernelCausalGraphTest` (8 tests): roots/terminals, upstream/downstream, ancestors/descendants, paths, link references, diamond graph, to_dict
  - `BeliefEngineComputeLikelihoodTest` (6 tests): explicit support/weaken, neutral, strength scaling, behavior alignment positive/conflicting
  - `BeliefEngineUpdatePosteriorsTest` (4 tests): supporting evidence increases posterior, normalization, clamping, evidence tracking
  - `BeliefEnginePropagationTest` (3 tests): chain propagation, no downstream, nonexistent hypothesis
  - `BeliefEngineExpansionTest` (3 tests): expansion signal generation, no expansion above threshold, empty likelihoods
  - `DSLParserBasicTest` (11 tests): comparison, string comparison, implication, quantifiers (ALWAYS/NEVER), logical AND/OR, WHEN domain, empty input, tautology detection, variable extraction
  - `DSLParserToDictTest` (3 tests): comparison/implication/quantified serialization
  - `DSLFormatTest` (3 tests): natural/formal/code formatting
**Verification:**
- `python3 manage.py check` — 0 issues
- `py_compile` — passes
- All 46 tests pass (13 kernel + 16 belief + 17 DSL)
**Commit:** afd60e0

---

### 2026-02-06 — P2: Wire Synara LLM interface to Anthropic API
**Debt item:** [SYNARA] LLM interface stubbed — prompts generated but never call API
**Files changed:**
- `services/svend/web/agents_api/synara/llm_interface.py` — added 6 methods to `SynaraLLMInterface`:
  - `_call_llm(user, prompt)` — calls Claude via `LLMManager.chat()`, tier-aware model selection
  - `_extract_json(text)` — robust JSON extraction from LLM responses (direct parse, ```json blocks, brace matching)
  - `validate_graph_llm(user)` — full round-trip: prompt → Claude → parse → `GraphAnalysis`
  - `generate_hypotheses_llm(user, signal)` — prompt → Claude → parse → `list[HypothesisRegion]` (auto-added to graph)
  - `interpret_evidence_llm(user, evidence, result)` — prompt → Claude → plain text interpretation
  - `document_findings_llm(user, format_type)` — prompt → Claude → formatted document (summary/a3/8d/technical)
- `services/svend/web/agents_api/synara_views.py` — added 4 server-side LLM endpoints:
  - `llm_validate` — validates causal graph via Claude
  - `llm_generate_hypotheses` — generates hypotheses from expansion signal via Claude
  - `llm_interpret_evidence` — interprets evidence update via Claude
  - `llm_document` — documents findings via Claude
  - All return 503 with fallback prompt if API key not set
- `services/svend/web/agents_api/synara_urls.py` — registered 4 new URL routes under `/api/synara/<wb_id>/llm/`
**Verification:**
- `python3 manage.py check` — 0 issues
- `py_compile` — all files pass
- URL resolution: all 4 endpoints resolve correctly
- Prompt generation + JSON extraction: tested in Django shell, all pass
- Graceful degradation: returns 503 with fallback_prompt when ANTHROPIC_API_KEY not set
**Commit:** fd16c67

---

### 2026-02-06 — P2: Researcher hallucination detection — fuzzy threshold tuning
**Debt item:** [CORE] Researcher hallucination detection needs fuzzy threshold tuning
**Files changed:**
- `services/svend/agents/agents/researcher/validator.py` — 3 improvements to `_validate_claim()`:
  1. **Windowed fuzzy matching**: `_fuzzy_similarity()` now slides a claim-sized window across source text instead of comparing whole strings. Claim "crispr can edit genes" vs 200-word source: old=0.25, new=0.71.
  2. **Bigram overlap**: new `_extract_bigrams()` adds phrase-level matching (word pairs) alongside single-term coverage. Combined score weights: 40% term coverage, 30% bigram overlap, 30% windowed similarity.
  3. **Smooth confidence curve**: replaced stepwise formula (`count * 0.3 + 0.4`) with `1 - 0.5^n` (0 sources→0.0, 1→0.5, 2→0.75, 3→0.88), blended 70/30 with best match quality.
- `services/svend/agents/researcher/validator.py` — synced duplicate copy
**Verification:**
- `py_compile` — both copies pass
- Windowed similarity: 0.706 for embedded claim (vs ~0.25 with old method)
- Bigram extraction: correct word pairs
- Confidence curve: monotonically increasing, properly scaled
- Claim validation: "CRISPR enables precise gene editing" correctly supported with confidence 0.60
**Commit:** 04fae5c

---

### 2026-02-06 — P3: Synara fallacy detection — implement pattern checks
**Debt item:** [SYNARA] Fallacy detection mostly stubbed
**Files changed:**
- `services/svend/web/agents_api/synara/logic_engine.py` — replaced `_check_fallacy_patterns()` stub (returned `[]`) with 5 structural pattern detectors:
  1. **Affirming the consequent**: shared variables between consequent/antecedent across multiple implications
  2. **Denying the antecedent**: negation of an implication's antecedent found in AST
  3. **False dichotomy**: XOR with exactly 2 options, or overlapping NEVER constraints on same variable
  4. **Hasty generalization**: universal quantifier (ALWAYS/NEVER) without WHEN domain restriction
  5. **Overgeneralization**: nested quantifiers
- Added 3 helper methods: `_collect_nodes()`, `_get_variables()`, `_contains_negation_of()`
- `services/svend/web/agents_api/tests.py` — added `FallacyDetectionTest` class with 13 tests covering all 5 fallacy types, helper methods, and `validate_hypothesis()` convenience function
**Verification:**
- `python3 manage.py check` — 0 issues
- All 13 fallacy detection tests pass
- Django shell verification: hasty generalization, XOR false dichotomy, WHEN clause suppression all correct
**Commit:** 0ba85e8

---

### 2026-02-06 — P3: Extend non-parametric battery — Friedman, Wilcoxon, Spearman
**Debt item:** [DSW] Non-parametric battery limited to Mann-Whitney + Kruskal
**Files changed:**
- `services/svend/web/agents_api/dsw_views.py` — added 3 new analysis types after Kruskal-Wallis:
  1. **Wilcoxon signed-rank** (`wilcoxon`): paired non-parametric test with effect size r, difference histogram
  2. **Friedman test** (`friedman`): repeated measures non-parametric ANOVA with Kendall's W, 3+ column checkbox selection
  3. **Spearman correlation** (`spearman`): rank correlation with p-value, 95% CI (Fisher z-transform), scatter plot
- `services/svend/web/templates/dsw.html` — added 3 options to dropdown, updated needsVar2/labels/config JS
- `services/svend/web/templates/analysis_workbench.html` — added 3 items to analysis catalog, form configs with checkboxes for Friedman
- `services/svend/web/templates/workbench_new.html` — added 3 options to More Tests dropdown
**Verification:**
- `python3 manage.py check` — 0 issues
- `py_compile` — passes
- End-to-end: Wilcoxon p=0.0020, Friedman p=0.0003, Spearman rho=0.95 — all correct
**Commit:** bfe3956

---

### 2026-02-06 — P2: Phase 2 model cutover — read paths from core.Project FKs
**Debt item:** [CORE] Phase 2 model cutover
**Files changed:**
- `services/svend/web/agents_api/models.py` — added 6 reader methods to Problem:
  - `get_hypotheses()` → reads from core.Hypothesis FKs, falls back to JSON blob
  - `get_evidence()` → reads from core.Evidence via EvidenceLinks, falls back to JSON blob
  - `get_dead_ends()` → reads from core.Hypothesis status=rejected, falls back to JSON blob
  - `get_probable_causes()` → reads from top core.Hypothesis by probability, falls back to JSON blob
  - `get_hypothesis_count()` → ORM count or JSON len
  - `get_evidence_count()` → ORM count or JSON len
- `services/svend/web/agents_api/problem_views.py` — switched 8 read paths:
  - `problem_to_dict()` — hypotheses, evidence, dead_ends, probable_causes
  - `write_context_file()` — hypotheses, evidence, dead_ends, probable_causes
  - `problems_list()` GET — hypothesis_count, evidence_count, top_cause
  - `add_evidence()` response — updated_hypotheses, probable_causes
  - `reject_hypothesis()` response — dead_ends, probable_causes
  - `generate_hypotheses()` — prompt context + response
- `services/svend/web/agents_api/views.py` — `get_problem_context_for_agent()` switched to `get_hypotheses()`
**Design:** All methods read from core.Project FKs when `core_project` FK exists, falling back to JSON blobs when not. API response shape unchanged — templates require no modifications. Fields without core equivalents (`key_uncertainties`, `recommended_next_steps`, `bias_warnings`) stay on Problem.
**Verification:**
- `python3 manage.py check` — 0 issues
- `py_compile` — all 3 files pass
- problem_to_dict(): 5 hypotheses from core FKs, correct dict shape (id, cause, probability, status, etc.)
- write_context_file(): context JSON has 5 hypotheses + 3 probable causes from core FKs
- get_problem_context_for_agent(): hypothesis text from core.Hypothesis
- Fallback: clearing core_project falls back to JSON blob
**Commit:** 98a1628

---

### 2026-02-07 — Complete learning module content for certification program
**Debt item:** N/A (feature completion)
**Files changed:**
- `services/svend/web/agents_api/learn_content.py` — added 3 missing educational sections:
  1. **CAUSAL_THINKING** (Causal Inference module): potential outcomes framework, DAGs, confounders vs colliders
  2. **AB_TESTING_CAUSAL** (Causal Inference module): A/B testing as causal inference, SUTVA, threats to validity, network effects
  3. **REGRESSION_HANDS_ON** (DSW Mastery module): linear regression, diagnostics, R², multicollinearity, logistic regression
- Updated SECTION_CONTENT registry: 34 → 37 sections total
**Verification:**
- `python3 -c "from agents_api.learn_content import SECTION_CONTENT; print(len(SECTION_CONTENT))"` → 37
- All sections include: content (markdown+LaTeX), interactive config, key_takeaways, practice_questions
- Only Synara module (3 sections) remains "coming soon" — intentionally skipped
**Commit:** pending

---

### 2026-02-07 — Rewrite landing page to match actual product
**Debt item:** N/A (positioning fix)
**Files changed:**
- `services/svend/site/index.html` — complete rewrite from "reasoning system" to "statistical analysis tool"
  - Old: "A reasoning system that shows its work" (described unreleased AI model)
  - New: "Statistical analysis without the $1,800/year price tag" (describes actual DSW product)
  - Added price comparison: Minitab $1,851/yr, JMP $1,320-8,400/yr, Svend $5/mo
  - Listed concrete features: SPC, DOE, Bayesian A/B, 60+ statistical tests
  - Defined target audience: quality engineers, startup data scientists, grad students, consultants
  - Simplified design, removed ambient animations
**Verification:** Visual review of new page
**Commit:** pending

---

### 2026-02-07 — Add collaboration and dot voting to Whiteboard
**Debt item:** N/A (new feature)
**Files changed:**
- `services/svend/web/agents_api/models.py` — added 3 models:
  - `Board`: collaborative whiteboard with room code, elements/connections as JSON, voting state
  - `BoardParticipant`: tracks who's in a session with color and cursor position
  - `BoardVote`: dot votes on elements with user limit
- `services/svend/web/agents_api/whiteboard_views.py` (new) — API endpoints:
  - `POST /api/whiteboard/boards/create/` — create new board, get room code
  - `GET /api/whiteboard/boards/<code>/` — get board state, participants, vote counts
  - `PUT /api/whiteboard/boards/<code>/update/` — save board state with version check
  - `POST /api/whiteboard/boards/<code>/voting/` — toggle voting mode (owner only)
  - `POST /api/whiteboard/boards/<code>/vote/` — add vote to element
  - `DELETE /api/whiteboard/boards/<code>/vote/<id>/` — remove vote
- `services/svend/web/agents_api/whiteboard_urls.py` (new) — URL routing
- `services/svend/web/svend/urls.py` — added whiteboard API route and room code URL
- `services/svend/web/templates/whiteboard.html` — added:
  - Collaboration UI: room code display, participant avatars, share button
  - Voting UI: toggle button, vote count badges, remaining votes display
  - CSS for collaboration/voting elements
  - JS for polling-based sync, voting, presence
- `services/svend/web/agents_api/migrations/0009_whiteboard_models.py` — migration applied
**Verification:**
- `python3 manage.py check` — 0 issues
- Models import correctly in Django shell
- Whiteboard accessible at `/app/whiteboard/` and `/app/whiteboard/<ROOM_CODE>/`
**Commit:** pending

## 2026-02-14 - Bayesian SPC Suite (4 tools)

**What:** Added 4 Bayesian SPC tools to DSW — no competitor offers these as click-and-run.
- **Bayesian Capability (`bayes_spc_capability`)** — eliminates the 1.5σ shift assumption. Uses NIG conjugate posterior + Monte Carlo for Cpk posterior distribution, predictive DPMO, and probability-driven verdicts. Supports one-sided specs (USL-only or LSL-only). 4 plots.
- **Bayesian Change Point Detection (`bayes_spc_changepoint`)** — Adams & MacKay (2007) BOCPD with NIG predictive. Run-length posterior heatmap, change probability timeline, annotated process data. 3 plots.
- **Bayesian Control Chart (`bayes_spc_control`)** — Two-state HMM forward filter with proper shift model (marginalized ±δ). Sequential NIG posterior for μ. Works with short runs (n=10-15). 3 plots.
- **Bayesian Acceptance Sampling (`bayes_spc_acceptance`)** — Beta-Binomial conjugate. Sequential P(p<AQL), earliest stopping, decision boundaries. 3 plots.
- 3 shared NIG helpers: `_nig_posterior_update()`, `_nig_sample()`, `_cpk_from_params()`
- SPC bridge in `spc_views.py` routes Bayesian types through `run_spc_analysis()`
- Frontend sub-tab "Bayesian SPC" in dsw.html with analysis selector, conditional config forms, prior presets

**Mathematical notes:**
- Weakly informative prior uses α₀=2 (finite σ² mean), β₀=(α₀-1)·s² (centered on sample variance)
- Predictive P(OOS) computed via Monte Carlo (not hardcoded Student-t parameterization)
- σ posterior sanity check warns about non-normal/mixed/outlier data
- Prior presets with plain English: "Default (weak)", "Historical", "Engineering"

**Files:**
- `agents_api/dsw_views.py` — 3 helpers + 4 analysis blocks (~480 lines)
- `agents_api/spc_views.py` — Bayesian SPC bridge in `analyze_uploaded()`
- `templates/dsw.html` — Bayesian SPC sub-tab, `updateBayesSPCForm()`, `runBayesSPC()`
**Update:** Moved to standalone "Bayesian Sigma" ribbon component (separate from SPC). Each tool gets its own sub-tab with dedicated form and results panel. Fixed `<<COLOR:error>>` handler in `displayStatsResults()`.
**Verified:** `python manage.py check` — 0 issues, gunicorn reloaded

## 2026-02-14 - Whitepaper PDF download infrastructure (WeasyPrint)

**What:** Built end-to-end whitepaper system with web reading + PDF download via WeasyPrint.

**Files added:**
- `api/whitepaper_views.py` — public views: `whitepaper_list`, `whitepaper_detail`, `whitepaper_pdf`
- `templates/whitepaper_detail.html` — web-readable view (extends tool_base.html, SEO metadata, TechArticle schema)
- `templates/whitepaper_print.html` — print-optimized standalone HTML for WeasyPrint (cover page, A4 pagination, page numbers, back cover CTA)

**Files modified:**
- `svend/urls.py` — added `/whitepapers/<slug>/` and `/whitepapers/<slug>/pdf/` routes, `WhitePaperSitemap` class, changed whitepapers index from TemplateView to `whitepaper_list` view
- `templates/whitepapers.html` — dynamic listing from DB with Read + PDF buttons, `{% empty %}` fallback

**Architecture:**
- Uses existing `WhitePaper` model (body field stores HTML content)
- `WhitePaperDownload` tracking on every view/download
- WeasyPrint installed system-wide (`pip3 install weasyprint`) — needed because gunicorn uses `/usr/bin/python3`
- PDF has cover page (title, topic tag, date, Svend branding), paginated body, page numbers, back cover CTA
- Web detail page has marked.js fallback for markdown bodies + native HTML rendering
- Sitemap auto-includes published whitepapers

**Verify:**
- `curl -H "X-Forwarded-Proto: https" http://127.0.0.1:8000/whitepapers/` — 200
- Create a WhitePaper via admin with status=published, verify detail + PDF routes work
- `python3 manage.py check` — 0 issues

## 2026-02-10 - Fixed founder registration invite bypass

**Issue:** Founder registration was still showing "Invite code required" error even after updating the middleware.

**Root cause:** Two issues:
1. `request.content_type` comparison was exact match but Django includes charset (e.g., `application/json; charset=utf-8`)
2. `request.body` needs explicit decoding when it's bytes

**Fix:** Updated `accounts/middleware.py`:
- Changed `request.content_type == "application/json"` to `request.content_type.startswith("application/json")`
- Added explicit `decode('utf-8')` for request body parsing

**Files changed:**
- `accounts/middleware.py` - Fixed JSON body parsing for paid plan detection

## 2026-02-10 - SEO: Black logo and Google search integration

**Changes:**
- Updated favicon.svg fill color from #4a9f6e (green) to #000000 (black)
- Added Organization schema to landing.html with logo property for Google search
- Created logo.png (512x512) from SVG for structured data
- Ran collectstatic to deploy new assets

**Files changed:**
- `static/favicon.svg` - Changed fill to black
- `static/logo.png` - New 512x512 PNG logo for Google
- `templates/landing.html` - Added Organization structured data with logo

## 2026-02-10 - Operations page visualizations

Added three new Plotly visualizations to the Operations Workbench:

1. **OEE Donut Chart** - Shows loss breakdown (availability, performance, quality losses) with OEE percentage in center
2. **EOQ Cost Curve** - Classic U-shape showing order cost, holding cost, and total cost curves with optimal EOQ marked
3. **Safety Stock Distribution** - Normal distribution curve showing demand during lead time, with shaded service level area and reorder point line

**Files changed:**
- `templates/calculators.html` - Added chart containers and Plotly rendering code in calcOEE(), calcEOQ(), calcSafety()

## 2026-02-10 - Added new Operations calculators

Added 7 new calculators to Methods > Operations:

**Flow:**
- **Little's Law** - WIP = Throughput × Lead Time, solve for any variable
- **Queuing (M/M/c)** - Full M/M/c queue theory with wait times, queue lengths, utilization, P(wait). Includes wait time vs utilization curve
- **Pitch** - Takt × pack-out for paced withdrawal intervals

**Quality:**
- **RTY (Rolled Throughput Yield)** - Multi-step first-pass yield with waterfall chart
- **DPMO / Sigma Level** - Convert between defects, DPMO, yield, and sigma level

**Financial:**
- **Inventory Turns** - Turnover rate and days/weeks on hand
- **Cost of Quality** - Prevention, appraisal, and failure cost breakdown with pie chart

All calculators include real-time updates and visualizations where applicable.

**Files changed:**
- `templates/calculators.html` - Added nav items, layouts, and JavaScript functions

---

## ML Interpretation & Decision Language — 2025-02-13

Added "last mile" interpretation to all ML outputs — every model training result now answers "does this matter?" and "what should I do?" instead of just showing raw metrics.

### Statistical Analyses (dsw_views.py)

Added `_effect_magnitude()` and `_practical_block()` helpers. Enhanced 8 analyses with effect sizes and four-way interpretation matrix (significant+meaningful → act, significant+small → consider cost, significant+negligible → ignore, not significant+large → need more data):

- **One-sample t-test** — Cohen's d
- **Two-sample t-test** — Cohen's d (pooled std), direction context
- **Paired t-test** — Cohen's d, improvement language
- **One-way ANOVA** — eta-squared, omega-squared
- **Two-way ANOVA** — partial eta-squared per factor, strongest effect ID
- **Chi-square** — Cramér's V
- **Regression** — R² as effect size, RMSE in plain language, significant predictors
- **Correlation** — per-pair p-values, KEY RELATIONSHIPS ranked by shared variance %

All enhanced analyses include `guide_observation` with effect sizes for downstream Synara Bayesian updates.

### ML Analyses (dsw_views.py)

Added `_ml_interpretation()` helper. Enhanced 4 ML analysis types:

- **Classification** — majority class baseline comparison, lift %, class imbalance warning (>80%), confusion matrix highlights
- **Regression ML** — RMSE as % of data range, CV(RMSE), R² practical significance, predict-mean baseline
- **Model Compare** — overfitting diagnosis (train-test gap), winner margin analysis, baseline comparison, deployment recommendation
- **Clustering** — silhouette score interpretation (strong/reasonable/weak/none), cluster size imbalance warning

### Autopilot Endpoints (autopilot_views.py)

Added `_build_training_interpretation()` and `_build_retrain_interpretation()` helpers. All 4 autopilot endpoints now return interpretation:

- **Clean+Train** — baseline comparison, data quality impact, top drivers, next steps
- **Full Pipeline** — same + Optuna tuning context
- **Augment+Train** — same + synthetic data impact
- **Retrain** — previous vs current comparison, degradation flags, notable changes

### Frontend (models.html)

Updated both rendering paths (inline results + modal report) to display interpretation with formatted newlines, colored checkmarks (✓ green), and warning icons (⚠ amber).

**Files changed:**
- `agents_api/dsw_views.py` — `_effect_magnitude()`, `_practical_block()`, `_ml_interpretation()`, enhanced 12 analysis types
- `agents_api/autopilot_views.py` — `_build_training_interpretation()`, `_build_retrain_interpretation()`, all 4 endpoints + retrain
- `templates/models.html` — interpretation rendering in both display paths

---

## Interactive ML Insights — 2025-02-13

Four features that close the gap between ML numbers and actionable decisions.

### 1. Subgroup Diagnostics (autopilot_views.py + models.html)

After training, the system automatically slices the test set by each categorical feature and reports per-segment metrics. Color-coded badges show where the model excels (green) and where it struggles (amber). Flags segments where accuracy drops >15% below overall. Renders in both inline results and modal report views.

### 2. Cost-Sensitive Threshold Analysis (autopilot_views.py + models.html)

For binary classification models: sweeps thresholds from 0.05 to 0.95, computing precision, recall, F1, and accuracy at each point. Reports optimal thresholds for F1, accuracy, and Youden's J. Includes a Plotly chart with precision/recall/F1 curves. In the Profiler, a threshold slider lets users adjust the decision boundary in real-time and see how reclassification changes. Stored in `training_config.threshold_analysis` for profiler access.

### 3. Prediction Intervals (dsw_views.py + models.html)

`run_model()` now accepts `intervals: true` parameter. For RandomForest models (most common), computes 90% prediction intervals from individual tree predictions (5th/95th percentiles of estimators). Profiler prediction card shows "± X (90% interval)". PDP curves display a shaded confidence band around the main line.

### 4. Optimization Profiler (dsw_views.py + dsw_urls.py + models.html)

New endpoint: `POST /api/dsw/models/<uuid>/optimize/` using `scipy.optimize.differential_evolution`. Supports maximize, minimize, and target-value goals. Optimizes over numeric features while brute-forcing categorical combinations (capped at 3 categorical features × 10 categories). "Optimize" button in profiler top bar with goal selector — sets all sliders to optimal values on completion.

**Files changed:**
- `agents_api/dsw_views.py` — `optimize_model()` endpoint, prediction intervals in `run_model()`
- `agents_api/dsw_urls.py` — optimize route
- `agents_api/autopilot_views.py` — `_compute_subgroup_diagnostics()`, `_compute_threshold_analysis()`, stored in all 4 endpoints + `training_config`
- `templates/models.html` — subgroup badges, threshold chart+slider, interval display, PDP confidence bands, optimize UI
