**STAT-001: STATISTICAL METHODOLOGY STANDARD**

**Version:** 1.2
**Status:** APPROVED
**Date:** 2026-03-03
**Supersedes:** None
**Author:** Eric + Claude (Systems Architect)
**Compliance:**
- DOC-001 ≥ 1.1 (Documentation Structure)
- XRF-001 ≥ 1.0 (Cross-Reference Syntax)
- DSW-001 ≥ 1.0 (DSW Architecture — stateless dispatch, evidence bridge)
- SOC 2 CC4.1 (Monitoring of Controls)
- ISO 9001:2015 §8.5.1 (Production and Service Provision — process control)
**Related Standards:**
- QMS-001 ≥ 1.2 (Quality Management System — SPC integration, evidence linking)
- TST-001 ≥ 1.0 (Testing Patterns — statistical test verification)
- MAP-001 ≥ 1.0 (Architecture Map — STAT-001 registry entry)

---

## **1. SCOPE AND PURPOSE**

### **1.1 Purpose**

STAT-001 defines the mathematical ideology, statistical methods, and correctness requirements for all quantitative analysis in Svend — DSW, SPC, DOE, Bayesian inference, and the evidence synthesis pipeline.

**Core Principle:**

> A p-value is a starting point, not a conclusion. Every analysis must answer three questions: Is there a real effect? How large is it? Would a different method agree?

This standard codifies eleven methodological principles that distinguish Svend from tools that treat statistics as a black box. These principles are not academic preferences — they are engineering requirements with machine-readable assertions and traceable implementations.

### **1.2 Scope**

**Applies to:**
- DSW statistical tests (`agents_api/dsw/stats.py`)
- DSW common statistical helpers (`agents_api/dsw/common.py`)
- Bayesian analysis module (`agents_api/dsw/bayesian.py`)
- SPC engine (`agents_api/dsw/spc.py`, `agents_api/spc.py`)
- Power analysis and DOE (`agents_api/experimenter_views.py`)
- Evidence grading and synthesis pipeline
- Guide observation narrative generation
- Any future analysis module that produces statistical output

**Does NOT apply to:**
- DSW architecture, dispatch, caching, persistence (DSW-001)
- ML model training and evaluation (`agents_api/dsw/ml.py`) — uses different correctness criteria
- Visualization rendering (`agents_api/dsw/viz.py`) — chart correctness, not math
- Frontend display of results (`templates/workbench_new.html`) — presentation, not computation
- Monte Carlo simulation (`agents_api/dsw/simulation.py`) — stochastic methods have separate validation needs

### **1.3 Mathematical Lineage**

The statistical methodology follows the **New Statistics** movement (Cumming 2012, 2014) and Bayesian insurance approach. Core references:

| Reference | Contribution | Used In |
|-----------|-------------|---------|
| Cohen (1988) | Effect size benchmarks (d, η², V, R²) | `_effect_magnitude()` |
| Cumming (2012, 2014) | New Statistics: estimation over testing | Ideology §4 |
| Rouder et al. (2009) | JZS Bayes Factor for t-tests | `_bayesian_shadow()` |
| Wagenmakers (2007) | BIC-approximated Bayes Factor | `_bayesian_shadow()` |
| Ly et al. (2016) | Bayes Factor for correlations | `_bayesian_shadow()` |
| Jeffreys (1961), Lee & Wagenmakers (2013) | BF interpretation scale | `_bayesian_shadow()` |
| Nelson (1984) | 8 SPC rules for process stability | `_spc_nelson_rules()` |
| Wilson (1927) | Score confidence interval for proportions | `stats.py` proportion tests |

---

## **2. NORMATIVE REFERENCES**

### **2.1 Kjerne Standards**

| Standard | Section | Requirement |
|----------|---------|-------------|
| **DSW-001** | §4 | Stateless dispatch — analysis functions receive data, return results |
| **DSW-001** | §6.1 | Evidence bridge — analysis results link to hypotheses as evidence |
| **QMS-001** | §4-5 | SPC→FMEA integration, evidence→hypothesis linking |
| **ERR-001** | §4 | Error handling for computation failures |

### **2.2 External Standards**

| Standard | Clause | Requirement |
|----------|--------|-------------|
| **ISO 9001:2015** | §8.5.1 | Process control — statistical methods must be valid |
| **SOC 2** | CC4.1 | Monitoring — analysis outputs inform compliance decisions |
| **APA 7th Edition** | §6.1 | Effect sizes reported alongside significance tests |

---

## **3. TERMINOLOGY**

| Term | Definition |
|------|------------|
| **New Statistics** | Methodological approach emphasizing effect sizes, confidence intervals, and estimation over null-hypothesis significance testing (Cumming 2012) |
| **Effect Size** | Standardized measure of the magnitude of a phenomenon. Not affected by sample size. |
| **Practical Significance** | Whether an effect is large enough to matter in context, independent of statistical significance |
| **Bayesian Insurance** | Pattern of computing a Bayes Factor alongside every frequentist test to provide convergent evidence |
| **Shadow BF** | A Bayes Factor computed silently alongside a frequentist test, using the same data |
| **Cross-Validation** | Running a non-parametric test in parallel with a parametric test to check agreement |
| **Evidence Grade** | Composite quality rating (Strong/Moderate/Weak/Inconclusive) synthesizing p-value, BF, effect size, and cross-validation |
| **Guide Observation** | 1-2 sentence narrative interpretation of an analysis result, displayed in the UI |
| **Nelson Rules** | Eight tests for non-random patterns in control chart data (Nelson 1984) |
| **JZS Prior** | Jeffreys-Zellner-Siow prior for Bayesian t-tests — a Cauchy prior on effect size with scale r = √2/2 |
| **BIC Approximation** | Using Bayesian Information Criterion difference to approximate Bayes Factor (Wagenmakers 2007) |
| **Credible Interval** | Bayesian analogue of confidence interval — probability statement about parameter location |

---

## **4. MATHEMATICAL IDEOLOGY**

### **4.1 The Eleven Principles**

<!-- rule: mandatory -->

<!-- assert: All DSW statistical analyses implement the eleven methodological principles defined in STAT-001 §4 | check=stat-ideology-implemented -->
<!-- impl: agents_api/dsw/common.py -->
<!-- impl: agents_api/dsw/stats.py -->
<!-- test: syn.audit.tests.test_statistics.MethodologyPrinciplesTest.test_effect_magnitude_exists -->
<!-- test: syn.audit.tests.test_statistics.MethodologyPrinciplesTest.test_practical_block_exists -->
<!-- test: syn.audit.tests.test_statistics.MethodologyPrinciplesTest.test_bayesian_shadow_exists -->
<!-- test: syn.audit.tests.test_statistics.MethodologyPrinciplesTest.test_check_normality_exists -->
<!-- test: syn.audit.tests.test_statistics.MethodologyPrinciplesTest.test_check_equal_variance_exists -->
<!-- test: syn.audit.tests.test_statistics.MethodologyPrinciplesTest.test_check_outliers_exists -->
<!-- test: syn.audit.tests.test_statistics.MethodologyPrinciplesTest.test_cross_validate_exists -->
<!-- test: syn.audit.tests.test_statistics.MethodologyPrinciplesTest.test_evidence_grade_exists -->
<!-- test: syn.audit.tests.test_statistics.MethodologyPrinciplesTest.test_narrative_exists -->
<!-- test: syn.audit.tests.test_statistics.MethodologyPrinciplesTest.test_guide_observation_set_in_stats -->
<!-- test: syn.audit.tests.test_statistics.MethodologyPrinciplesTest.test_run_statistical_analysis_exists -->
<!-- impl: agents_api/dsw/bayesian.py -->
<!-- test: agents_api.dsw_engine_tests.BayesianAnalysisScenarioTest.test_bayesian_regression -->
<!-- test: agents_api.dsw_engine_tests.StatisticalAnalysisScenarioTest.test_descriptive_stats -->
<!-- test: agents_api.dsw_engine_tests.StatisticalAnalysisScenarioTest.test_anova -->
<!-- test: agents_api.dsw_engine_tests.StatisticalAnalysisScenarioTest.test_regression -->

Svend's statistical methodology is built on eleven principles. These are not optional enhancements — they are the methodological foundation that differentiates Svend from tools that stop at p-values.

| # | Principle | Requirement | Implementation |
|---|-----------|-------------|----------------|
| 1 | **Effect sizes over p-values** | Every test that produces a p-value MUST also produce at least one effect size metric | `_effect_magnitude()`, `_practical_block()` |
| 2 | **Confidence intervals as estimators** | Every effect size MUST include a confidence or credible interval | Wilson CI, Fisher z CI, bootstrap CI |
| 3 | **Bayesian insurance** | Every frequentist test SHOULD compute a shadow Bayes Factor | `_bayesian_shadow()` |
| 4 | **Practical ≠ statistical significance** | Results MUST distinguish between statistical and practical significance using the four-region model | `_practical_block()` |
| 5 | **Cross-validation** | Parametric tests MUST run a non-parametric parallel and report agreement/contradiction | `_cross_validate()` |
| 6 | **Assumption checking is not optional** | Normality, equal variance, and outlier checks MUST run automatically for parametric tests | `_check_normality()`, `_check_equal_variance()`, `_check_outliers()` |
| 7 | **Power analysis as context** | Post-hoc power metadata SHOULD accompany results for interactive exploration | `power_explorer` metadata |
| 8 | **Narrative interpretation** | Every analysis MUST produce a `guide_observation` — a human-readable verdict | `_narrative()`, `guide_observation` field |
| 9 | **Multiple effect size metrics** | Where applicable, tests SHOULD report more than one effect size | Cohen's d + r²_pb, η² + Cohen's f, etc. |
| 10 | **SPC with Nelson rules** | Control charts MUST apply all 8 Nelson rules, not just 3σ limits | `_spc_nelson_rules()` |
| 11 | **DOE with power curves** | Design of experiments MUST include power curve context | `_compute_power_curve()` |

### **4.2 Principle Hierarchy**

Principles 1, 4, 5, 6, and 8 are **MANDATORY** — every applicable analysis must implement them. Principles 2, 3, 7, 9, 10, and 11 are **SHOULD** — implemented where mathematically appropriate.

---

## **5. EFFECT SIZE FRAMEWORK**

### **5.1 Effect Size Thresholds**

<!-- assert: Effect size classification uses Cohen (1988) benchmarks for d, η², V, and R² | check=stat-effect-thresholds -->
<!-- impl: agents_api/dsw/common.py:_effect_magnitude -->
<!-- test: syn.audit.tests.test_statistics.EffectSizeThresholdsTest.test_cohens_d_thresholds -->
<!-- test: syn.audit.tests.test_statistics.EffectSizeThresholdsTest.test_eta_squared_thresholds -->
<!-- test: syn.audit.tests.test_statistics.EffectSizeThresholdsTest.test_cramers_v_thresholds -->
<!-- test: syn.audit.tests.test_statistics.EffectSizeThresholdsTest.test_r_squared_thresholds -->
<!-- test: syn.audit.tests.test_statistics.EffectSizeThresholdsTest.test_returns_label_and_meaningful -->

`_effect_magnitude(value, effect_type)` classifies effect sizes using standard Cohen (1988) benchmarks:

| Effect Type | Negligible | Small | Medium | Large | Source |
|-------------|-----------|-------|--------|-------|--------|
| Cohen's d | < 0.2 | 0.2–0.5 | 0.5–0.8 | ≥ 0.8 | Cohen (1988) |
| η² (eta-squared) | < 0.01 | 0.01–0.06 | 0.06–0.14 | ≥ 0.14 | Cohen (1988) |
| Cramér's V | < 0.1 | 0.1–0.3 | 0.3–0.5 | ≥ 0.5 | Cohen (1988) |
| R² | < 0.02 | 0.02–0.13 | 0.13–0.26 | ≥ 0.26 | Cohen (1988) |

Returns tuple `(label, is_meaningful)` where `is_meaningful = True` for medium and large effects.

### **5.2 Four-Region Decision Model**

<!-- assert: Practical significance uses four-region model separating statistical and practical significance | check=stat-four-regions -->
<!-- impl: agents_api/dsw/common.py:_practical_block -->
<!-- test: syn.audit.tests.test_statistics.FourRegionModelTest.test_practical_block_uses_alpha -->
<!-- test: syn.audit.tests.test_statistics.FourRegionModelTest.test_practical_block_uses_effect -->
<!-- test: syn.audit.tests.test_statistics.FourRegionModelTest.test_practical_block_in_stats -->

`_practical_block()` maps every test result into one of four decision regions:

| Region | p-value | Effect Size | Verdict | Action |
|--------|---------|-------------|---------|--------|
| **1** | < α | Medium/Large | Both statistically and practically significant | Act on the finding |
| **2** | < α | Small | Statistically significant but small effect | Consider cost-benefit |
| **3** | < α | Negligible | Statistically significant but negligible effect | Do not act — sample size inflated significance |
| **4** | ≥ α | Medium/Large | Not significant but meaningful effect size | Collect more data — possible power issue |

Region 3 is the most important for practitioners: it catches the "p < 0.05 but who cares" pattern that traditional tools ignore.

### **5.3 Multiple Effect Size Metrics**

Where applicable, analyses report multiple effect sizes from the same data:

| Test | Primary | Secondary |
|------|---------|-----------|
| Two-sample t-test | Cohen's d | Point-biserial r² |
| One-way ANOVA | η² (eta-squared) | Cohen's f, ω² (omega-squared) |
| Chi-square | Cramér's V | Contingency coefficient |
| Correlation | Pearson r | R² (coefficient of determination) |
| Paired t-test | Cohen's d (within) | — |

---

## **6. BAYESIAN INSURANCE**

### **6.1 Shadow Bayes Factor**

<!-- assert: Bayesian shadow computes JZS Bayes Factor for t-tests using Rouder et al. (2009) integrand | check=stat-jzs-bf -->
<!-- impl: agents_api/dsw/common.py:_bayesian_shadow -->
<!-- test: syn.audit.tests.test_statistics.BayesianInsuranceTest.test_shadow_types_supported -->
<!-- test: syn.audit.tests.test_statistics.BayesianInsuranceTest.test_jzs_prior_scale -->
<!-- test: syn.audit.tests.test_statistics.BayesianInsuranceTest.test_uses_scipy_integrate -->
<!-- test: syn.audit.tests.test_statistics.BayesianInsuranceTest.test_shadow_called_in_stats -->
<!-- test: syn.audit.tests.test_statistics.BayesianInsuranceTest.test_bf10_in_result -->

`_bayesian_shadow(shadow_type, **kwargs)` computes a Bayes Factor alongside frequentist tests. The BF provides evidence strength on a continuous scale rather than the binary sig/not-sig of p-values.

**Supported shadow types:**

| Shadow Type | Method | Prior | Reference |
|-------------|--------|-------|-----------|
| `ttest_1samp` | JZS BF₁₀ via numerical integration | Cauchy(0, r=√2/2) on δ | Rouder et al. (2009) |
| `ttest_2samp` | JZS BF₁₀ via numerical integration | Cauchy(0, r=√2/2) on δ | Rouder et al. (2009) |
| `ttest_paired` | JZS BF₁₀ via numerical integration | Cauchy(0, r=√2/2) on δ | Rouder et al. (2009) |
| `anova` | BIC-approximated BF₁₀ | Implicit BIC prior | Wagenmakers (2007) |
| `correlation` | BF₁₀ via integral under uniform prior on ρ | Uniform(-1, 1) | Ly et al. (2016) |
| `proportion` | Savage-Dickey BF under Beta(1,1) prior | Beta(1, 1) | — |
| `chi2` | BIC-approximated BF₁₀ | Implicit BIC prior | Wagenmakers (2007) |

### **6.2 JZS Integrand (t-tests)**

The JZS Bayes Factor (Rouder et al. 2009, Equation 2) integrates over the prior on effect size:

```
BF₁₀ = ∫₀^∞ (1 + n·r²·g)^(-½) · [(1 + t²/((1+n·r²·g)·ν))^(-(ν+1)/2) / (1 + t²/ν)^(-(ν+1)/2)] · (2π)^(-½) · g^(-3/2) · exp(-1/(2g)) dg
```

Where:
- `t` = t-statistic, `ν` = degrees of freedom, `n` = effective sample size
- `r = √2/2 ≈ 0.707` (standard JZS default scale)
- `g` = integration variable (half-Cauchy prior on δ²)

Integration uses `scipy.integrate.quad` with bounds `[1e-10, ∞)`.

### **6.3 Bayes Factor Interpretation Scale**

<!-- assert: Bayes Factor interpretation follows Jeffreys (1961) / Lee & Wagenmakers (2013) evidence categories | check=stat-bf-scale -->
<!-- impl: agents_api/dsw/common.py:_bayesian_shadow -->
<!-- test: syn.audit.tests.test_statistics.BayesianInsuranceTest.test_bf_interpretation_categories -->

| BF₁₀ Range | Evidence Category | Direction |
|-------------|-------------------|-----------|
| > 100 | Extreme | For H₁ |
| 30–100 | Very strong | For H₁ |
| 10–30 | Strong | For H₁ |
| 3–10 | Moderate | For H₁ |
| 1–3 | Weak | For H₁ |
| 1/3–1 | Weak | For H₀ |
| 1/10–1/3 | Moderate | For H₀ |
| ≤ 1/10 | Strong | For H₀ |

### **6.4 Credible Intervals**

Shadow computations also return credible intervals when applicable:

| Shadow Type | Parameter | CI Method |
|-------------|-----------|-----------|
| t-tests | Cohen's d | Normal approximation: d ± 1.96 × SE_d |
| Correlation | Pearson r | Fisher z-transform: tanh(z_r ± 1.96/√(n-3)) |
| Proportion | p | Beta posterior: Beta(1+x, 1+n-x) quantiles |

**Cohen's d SE formula:** `SE_d = √(1/n + d²/(2n))` (one-sample/paired) or `SE_d = √((n₁+n₂)/(n₁n₂) + d²/(2(n₁+n₂)))` (two-sample).

---

## **7. ROBUSTNESS FRAMEWORK**

### **7.1 Assumption Verification**

<!-- assert: Parametric tests automatically run normality, equal variance, and outlier checks without user request | check=stat-auto-assumptions -->
<!-- impl: agents_api/dsw/common.py:_check_normality -->
<!-- impl: agents_api/dsw/common.py:_check_equal_variance -->
<!-- impl: agents_api/dsw/common.py:_check_outliers -->
<!-- test: syn.audit.tests.test_statistics.AssumptionVerificationTest.test_normality_uses_shapiro -->
<!-- test: syn.audit.tests.test_statistics.AssumptionVerificationTest.test_normality_dagostino_fallback -->
<!-- test: syn.audit.tests.test_statistics.AssumptionVerificationTest.test_equal_variance_uses_levene -->
<!-- test: syn.audit.tests.test_statistics.AssumptionVerificationTest.test_outlier_uses_iqr -->
<!-- test: syn.audit.tests.test_statistics.AssumptionVerificationTest.test_stats_calls_normality -->
<!-- test: syn.audit.tests.test_statistics.AssumptionVerificationTest.test_stats_calls_equal_variance -->
<!-- test: syn.audit.tests.test_statistics.AssumptionVerificationTest.test_diagnostics_in_stats -->
<!-- test: agents_api.dsw_engine_tests.StatisticalAnalysisScenarioTest.test_descriptive_stats -->
<!-- test: agents_api.dsw_engine_tests.StatisticalAnalysisScenarioTest.test_anova -->

Assumption checks are **automatic** — users do not request them. They run on every parametric test and their results appear in the `diagnostics` array.

| Check | Function | Test Used | Threshold | Minimum n |
|-------|----------|-----------|-----------|-----------|
| Normality | `_check_normality()` | Shapiro-Wilk (n ≤ 5000), D'Agostino-Pearson (n > 5000) | p < 0.05 | 8 |
| Equal Variance | `_check_equal_variance()` | Levene's test (median) | p < 0.05 | 3 per group |
| Outliers | `_check_outliers()` | 1.5 × IQR rule | > 1% of data | 10 |

**Diagnostic severity:**
- `"warning"` — assumption violated, results may be affected (outliers < 5%)
- `"error"` — severe violation, results unreliable (outliers ≥ 5%)
- `"info"` — informational (cross-validation agreement)
- `"contradiction"` — parametric and non-parametric tests disagree

### **7.2 Cross-Validation**

<!-- assert: Parametric tests run a non-parametric parallel test and report agreement or contradiction | check=stat-cross-validation -->
<!-- impl: agents_api/dsw/common.py:_cross_validate -->
<!-- impl: agents_api/dsw/stats.py:run_statistical_analysis -->
<!-- test: syn.audit.tests.test_statistics.CrossValidationTest.test_cross_validate_accepts_both_pvalues -->
<!-- test: syn.audit.tests.test_statistics.CrossValidationTest.test_stats_calls_cross_validate -->
<!-- test: syn.audit.tests.test_statistics.CrossValidationTest.test_cross_validate_reports_agreement -->
<!-- test: syn.audit.tests.test_statistics.CrossValidationTest.test_nonparametric_parallels_exist -->

`_cross_validate(primary_p, alt_p, primary_name, alt_name, alpha, normality_failed)` compares a parametric test against its non-parametric counterpart:

| Parametric Test | Non-Parametric Cross-Check |
|----------------|---------------------------|
| One-sample t-test | Wilcoxon signed-rank |
| Two-sample t-test | Mann-Whitney U |
| Paired t-test | Wilcoxon signed-rank |
| One-way ANOVA | Kruskal-Wallis |

When the two tests disagree (one significant, one not), the diagnostic:
1. Reports the contradiction with both p-values
2. If normality failed, notes that non-normality may affect the parametric test
3. If either p-value is within 0.02 of α, notes the borderline result

Cross-validation agreement/disagreement feeds into the evidence grade (§8).

### **7.3 Welch's Correction**

When Levene's test detects unequal variances, Welch's t-test (default in `scipy.stats.ttest_ind`) handles this automatically. The diagnostic notes: "Welch's t-test (default) handles this correctly."

---

## **8. EVIDENCE SYNTHESIS**

### **8.1 Evidence Grading Algorithm**

<!-- assert: Evidence grade synthesizes p-value, Bayes Factor, effect magnitude, and cross-validation into a composite quality rating | check=stat-evidence-grade -->
<!-- impl: agents_api/dsw/common.py:_evidence_grade -->
<!-- test: syn.audit.tests.test_statistics.EvidenceSynthesisTest.test_evidence_grade_accepts_components -->
<!-- test: syn.audit.tests.test_statistics.EvidenceSynthesisTest.test_grade_categories_defined -->
<!-- test: syn.audit.tests.test_statistics.EvidenceSynthesisTest.test_evidence_grade_called_in_stats -->

`_evidence_grade(p_value, bf10, effect_magnitude, cross_val_agrees)` combines all available evidence signals into a single quality grade:

**Scoring:**

| Component | Condition | Score |
|-----------|-----------|-------|
| p-value | < 0.001 | +3 |
| p-value | < 0.01 | +2 |
| p-value | < 0.05 | +1 |
| p-value | ≥ 0.05 | +0 |
| BF₁₀ | > 10 | +3 |
| BF₁₀ | > 3 | +2 |
| BF₁₀ | > 1 | +1 |
| BF₁₀ | ≤ 1 | +0 |
| Effect size | Large | +2 |
| Effect size | Medium | +1 |
| Effect size | Small/Negligible | +0 |
| Cross-validation | Agrees | +1 |
| Cross-validation | Disagrees | -1 |

**Maximum possible score:** 3 + 3 + 2 + 1 = **9**

**Grade mapping:**

| Score | Grade | Meaning |
|-------|-------|---------|
| ≥ 8 | **Strong** | Convergent evidence from all signals |
| 5–7 | **Moderate** | Most signals agree, some missing or weak |
| 2–4 | **Weak** | Limited evidence, p-value alone or conflicting signals |
| < 2 | **Inconclusive** | No clear evidence or contradictory signals |

### **8.2 Integration Pattern**

Every test that computes p-value, effect size, and Bayesian shadow calls `_evidence_grade()` to synthesize:

```python
_shadow = _bayesian_shadow("ttest_2samp", x=x.values, y=y.values)
_grade = _evidence_grade(
    pval,
    bf10=_shadow.get("bf10") if _shadow else None,
    effect_magnitude=label,
    cross_val_agrees=_cv_agrees
)
if _grade:
    result["evidence_grade"] = _grade
```

The grade travels through the evidence bridge (DSW-001 §6.1) and appears in Synara's belief update pipeline.

---

## **9. STATISTICAL PROCESS CONTROL**

### **9.1 Nelson Rules**

<!-- assert: SPC control charts apply all 8 Nelson rules for out-of-control detection | check=stat-nelson-rules -->
<!-- impl: agents_api/dsw/spc.py:_spc_nelson_rules -->
<!-- test: syn.audit.tests.test_statistics.NelsonRulesTest.test_nelson_rules_function_exists -->
<!-- test: syn.audit.tests.test_statistics.NelsonRulesTest.test_rule_1_beyond_3sigma -->
<!-- test: syn.audit.tests.test_statistics.NelsonRulesTest.test_rule_2_nine_consecutive -->
<!-- test: syn.audit.tests.test_statistics.NelsonRulesTest.test_rule_7_fifteen_within_1sigma -->
<!-- test: syn.audit.tests.test_statistics.NelsonRulesTest.test_rule_8_eight_beyond_1sigma -->
<!-- test: syn.audit.tests.test_statistics.NelsonRulesTest.test_sigma_derived_from_limits -->
<!-- impl: agents_api/spc.py -->
<!-- test: agents_api.dsw_engine_tests.SPCEngineTest.test_control_chart_computation -->
<!-- test: agents_api.dsw_engine_tests.SPCEngineTest.test_capability_indices -->
<!-- test: agents_api.scenario_tests.SPCScenarioTest.test_imr_control_chart -->
<!-- test: agents_api.scenario_tests.SPCScenarioTest.test_capability_study -->
<!-- test: agents_api.scenario_tests.SPCScenarioTest.test_auth_required -->

`_spc_nelson_rules(data, cl, ucl, lcl)` implements all 8 Nelson rules. σ is derived from control limits: `σ = (UCL - CL) / 3`.

| Rule | Pattern | Window | Detection |
|------|---------|--------|-----------|
| **1** | Point beyond 3σ | 1 | `data[i] > UCL or data[i] < LCL` |
| **2** | 9 consecutive same side of CL | 9 | All above or all below center line |
| **3** | 6 consecutive trending | 6 | All differences same sign (monotonic) |
| **4** | 14 consecutive alternating | 14 | Every consecutive pair reverses direction |
| **5** | 2 of 3 beyond 2σ (same side) | 3 | Count > 2σ violations in window |
| **6** | 4 of 5 beyond 1σ (same side) | 5 | Count > 1σ violations in window |
| **7** | 15 consecutive within 1σ | 15 | All within ±1σ (stratification — too little variation) |
| **8** | 8 consecutive beyond 1σ (both sides) | 8 | All outside ±1σ (mixture — bimodal process) |

**Application:** Applied to I-MR, X̄-R, X̄-S, NP, and C chart types. Per-point rule annotations are built by `_spc_build_point_rules()` for interactive hover inspection.

<!-- assert: SPC per-point rule annotations built for interactive hover inspection | check=stat-point-annotations -->
<!-- impl: agents_api/dsw/spc.py -->
<!-- test: syn.audit.tests.test_statistics.NelsonRulesTest.test_nelson_rules_function_exists -->
<!-- test: syn.audit.tests.test_statistics.NelsonRulesTest.test_sigma_derived_from_limits -->

### **9.2 Control Chart Types**

| Chart | Use Case | Statistics |
|-------|----------|------------|
| I-MR | Individual measurements | Individual values + moving range |
| X̄-R | Subgroup means (n ≤ 10) | Subgroup mean + range |
| X̄-S | Subgroup means (n > 10) | Subgroup mean + standard deviation |
| NP | Defective count (constant n) | Count of nonconforming |
| C | Defect count (constant area) | Count of defects |
| P | Proportion defective | Fraction nonconforming |

### **9.3 Advanced Engines**

<!-- assert: DSW dispatch routes causal discovery (PC, LiNGAM), drift detection (PSI), and anytime-valid inference (e-process) to their respective sub-modules | check=stat-advanced-engines -->
<!-- impl: agents_api/causal_discovery.py -->
<!-- impl: agents_api/drift_detection.py -->
<!-- impl: agents_api/anytime_valid.py -->
<!-- test: agents_api.engine_tests.CausalDiscoveryScenarioTest.test_pc_algorithm -->
<!-- test: agents_api.engine_tests.CausalDiscoveryScenarioTest.test_lingam -->
<!-- test: agents_api.engine_tests.DriftDetectionScenarioTest.test_drift_report -->
<!-- test: agents_api.engine_tests.DriftDetectionScenarioTest.test_psi_calculation -->
<!-- test: agents_api.engine_tests.AnytimeValidScenarioTest.test_gaussian_e_process -->
<!-- test: agents_api.engine_tests.AnytimeValidScenarioTest.test_anytime_via_endpoint -->

---

## **10. POWER ANALYSIS & DESIGN OF EXPERIMENTS**

### **10.1 Sample Size Computation**

<!-- assert: Power analysis computes required sample size for five test types given effect size, alpha, and power | check=stat-power-analysis -->
<!-- impl: agents_api/experimenter_views.py:_compute_power_curve -->
<!-- test: syn.audit.tests.test_statistics.PowerAnalysisTest.test_power_curve_function_exists -->
<!-- test: syn.audit.tests.test_statistics.PowerAnalysisTest.test_supports_ttest -->
<!-- test: syn.audit.tests.test_statistics.PowerAnalysisTest.test_supports_anova -->
<!-- test: syn.audit.tests.test_statistics.PowerAnalysisTest.test_supports_correlation -->
<!-- test: syn.audit.tests.test_statistics.PowerAnalysisTest.test_power_explorer_metadata -->

`_compute_power_curve(test_type, alpha, power, groups)` computes sample sizes across effect sizes 0.10 → 2.00 (step 0.05).

| Test Type | Formula | Notes |
|-----------|---------|-------|
| Independent t-test | `n₁ = ⌈2((z_{α/2} + z_β)/d)²⌉` | Per group; total = 2n₁ |
| Paired t-test | `n = ⌈((z_{α/2} + z_β)/d)²⌉` | Total (pairs) |
| One-way ANOVA | Iterative search using non-central F | `λ = d² × n_total`, tries n 2→5000 |
| Correlation | `n = ⌈((z_{α/2} + z_β)/z_r)²⌉ + 3` | Fisher z-transform of r |
| Chi-square | `n = ⌈((z_α + z_β)/d)²⌉` | One-tailed z |

<!-- assert: Post-hoc power metadata attached to analysis results for interactive exploration | check=stat-power-metadata -->
<!-- impl: agents_api/experimenter_views.py -->
<!-- test: syn.audit.tests.test_statistics.PowerAnalysisTest.test_power_explorer_metadata -->

### **10.2 Post-Hoc Power Metadata**

Analyses that produce effect sizes attach `power_explorer` metadata for interactive exploration:

```python
result["power_explorer"] = {
    "test_type": "anova",
    "observed_effect": float(cohens_f),
    "observed_n": int(n_total),
    "alpha": 0.05,
    "n_groups": int(k),
}
```

The frontend uses this to let users explore "what if I had more data?" scenarios without re-running the analysis.

### **10.3 ANOVA Power via Non-Central F**

For ANOVA, power is computed iteratively:

1. Set `λ = d² × n_total`
2. Compute `f_crit = F_{1-α}(df₁, df₂)`
3. Compute `power = 1 - F_{nc}(f_crit; df₁, df₂, λ)` where F_nc is the non-central F CDF
4. Increase n until achieved power ≥ requested power

### **10.4 Bayesian DOE**

<!-- assert: Bayesian DOE computes design matrices, posterior distributions, and full effect analysis for factorial experiments | check=stat-bayes-doe -->
<!-- impl: agents_api/bayes_doe.py:build_doe_design_matrix -->
<!-- impl: agents_api/bayes_core.py:bayesian_linear_posterior -->
<!-- impl: agents_api/bayes_doe.py:run_bayesian_doe -->
<!-- test: agents_api.engine_tests.BayesDOEScenarioTest.test_build_design_matrix -->
<!-- test: agents_api.engine_tests.BayesDOEScenarioTest.test_bayesian_linear_posterior -->
<!-- test: agents_api.engine_tests.BayesDOEScenarioTest.test_run_bayesian_doe -->

---

## **11. NARRATIVE INTERPRETATION**

### **11.1 Guide Observations**

<!-- assert: Every DSW analysis produces a guide_observation field with a human-readable 1-2 sentence verdict | check=stat-guide-observation -->
<!-- impl: agents_api/dsw/stats.py:run_statistical_analysis -->
<!-- test: syn.audit.tests.test_statistics.GuideObservationTest.test_guide_observation_set -->
<!-- test: syn.audit.tests.test_statistics.GuideObservationTest.test_guide_observation_not_empty -->

Every analysis result includes a `guide_observation` string that:

1. Names the test and key statistics in one phrase
2. States the practical conclusion (significant/not, effect size label)
3. Is concise enough for UI display above detailed output

**Pattern:**
```python
obs = [f"Two-sample t-test: t={stat:.4f}, p={pval:.4f}, d={abs(d):.3f}"]
if pval < alpha and meaningful:
    obs.append(f"'{var1}' and '{var2}' differ significantly.")
result["guide_observation"] = " ".join(obs)
```

<!-- assert: Narrative blocks provide verdict, body, chart guidance, and next steps | check=stat-narrative-blocks -->
<!-- impl: agents_api/dsw/common.py:_narrative -->
<!-- test: syn.audit.tests.test_statistics.GuideObservationTest.test_guide_observation_set -->
<!-- test: syn.audit.tests.test_statistics.MethodologyPrinciplesTest.test_narrative_exists -->

### **11.2 Narrative Blocks**

`_narrative(verdict, body, next_steps, chart_guidance)` builds HTML blocks for charts-first output, providing:

- **Verdict** — one-line conclusion (e.g., "Groups differ significantly")
- **Body** — explanation of what the result means
- **Chart guidance** — what to look for in the visualization
- **Next steps** — what analysis to run next

---

## **12. CONFIDENCE INTERVAL METHODS**

### **12.1 Interval Types by Context**

<!-- assert: Proportion confidence intervals use Wilson score method, not Wald method | check=stat-wilson-ci -->
<!-- impl: agents_api/dsw/stats.py:run_statistical_analysis -->
<!-- test: syn.audit.tests.test_statistics.ConfidenceIntervalTest.test_wilson_ci_used -->
<!-- test: syn.audit.tests.test_statistics.ConfidenceIntervalTest.test_fisher_z_for_correlations -->
<!-- test: syn.audit.tests.test_statistics.StatAntiPatternTest.test_no_pvalue_only_results -->
<!-- test: syn.audit.tests.test_statistics.StatAntiPatternTest.test_alpha_parameterized -->

| Context | Method | Formula |
|---------|--------|---------|
| Proportions | Wilson score interval | `center = (p̂ + z²/(2n)) / (1 + z²/n)` |
| Correlations | Fisher z-transform | `tanh(arctanh(r) ± z/√(n-3))` |
| Means | Standard t-interval | `x̄ ± t_{α/2} × s/√n` |
| Cohen's d | Normal approximation | `d ± z × SE_d` |
| Bayesian posteriors | Credible interval | Beta/Normal quantiles |

**Wilson over Wald:** The Wald interval (`p̂ ± z√(p̂(1-p̂)/n)`) is known to have poor coverage near 0 and 1. Wilson maintains nominal coverage across all proportions and is strictly preferred.

---

## **13. STATISTICAL CALIBRATION**

### **13.1 Calibration System**

<!-- assert: Statistical analysis functions are calibrated using known reference data with analytically correct answers, with daily rotation of reference cases | check=stat-calibration-system -->
<!-- impl: agents_api/calibration.py:run_calibration -->
<!-- impl: agents_api/calibration.py:CalibrationCase -->
<!-- impl: agents_api/calibration.py:Expectation -->
<!-- test: syn.audit.tests.test_compliance_system.CalibrationTest.test_reference_pool_has_cases -->
<!-- test: syn.audit.tests.test_compliance_system.CalibrationTest.test_calibration_runner_returns_results -->
<!-- test: syn.audit.tests.test_compliance_system.CalibrationTest.test_known_null_ttest_passes -->

Statistical analysis functions are treated as measurement devices requiring periodic calibration. The calibration system:

1. Maintains a **reference pool** of 15+ calibration cases across 6 categories (inference, Bayesian, SPC, reliability, ML, simulation)
2. Each case feeds known reference data with analytically correct answers through the corresponding DSW analysis function
3. Verifies that outputs fall within specified tolerances
4. Flags **drift** when any case fails — outputs have deviated from known correct answers

### **13.2 Reference Case Rotation**

<!-- assert: Calibration cases are selected daily using a date-seeded RNG for reproducible rotation across the reference pool | check=stat-calibration-rotation -->
<!-- impl: agents_api/calibration.py:run_calibration -->
<!-- test: syn.audit.tests.test_compliance_system.CalibrationTest.test_date_seeded_reproducibility -->

To avoid path-dependency and ensure all cases are exercised:
- **Seed**: `date.today().toordinal()` — same seed = same selection for that day
- **Subset**: 8 of 17 cases per run (configurable)
- **Full coverage**: every case runs approximately every 2 days

### **13.3 Drift Detection**

<!-- assert: Failed calibration cases create DriftViolation records with enforcement_check="CAL" for audit trail | check=stat-calibration-drift -->
<!-- impl: syn/audit/compliance.py:check_statistical_calibration -->
<!-- test: syn.audit.tests.test_compliance_system.CalibrationTest.test_drift_violation_on_failure -->

When a calibration case fails:
1. A `DriftViolation` is created with `enforcement_check="CAL"` and severity based on failure count
2. The compliance check status degrades: 100% pass = "pass", ≥80% = "warning", <80% = "fail"
3. Results are logged in `ComplianceCheck.details` with per-case expected/actual/deviation data
4. The dashboard displays per-case results sorted with failures first

### **13.4 Reference Categories**

| Category | Cases | What Is Calibrated |
|----------|-------|--------------------|
| **Inference** | 7 | t-tests, ANOVA, correlation, chi-square, paired t |
| **Bayesian** | 3 | Estimation, A/B testing, regression |
| **SPC** | 3 | I-MR control charts, capability analysis |
| **Reliability** | 1 | Weibull distribution fitting |
| **ML** | 2 | Random forest regression, classification |
| **Simulation** | 1 | Monte Carlo simulation |

---

## **14. ANTI-PATTERNS**

### **14.1 P-Value as Sole Decision Criterion**

<!-- rule: mandatory -->

**PROHIBITED:** Reporting statistical significance without effect size or practical interpretation.

```python
# Wrong
if pval < 0.05:
    result["summary"] = "Significant difference found."

# Correct
label, meaningful = _effect_magnitude(d, "cohens_d")
result["summary"] += _practical_block("Cohen's d", d, "cohens_d", pval, alpha)
```

### **14.2 Skipping Assumption Checks**

<!-- rule: mandatory -->

**PROHIBITED:** Running parametric tests without automatic normality, variance, and outlier diagnostics.

```python
# Wrong
stat, pval = stats.ttest_ind(x, y)
result["summary"] = f"p = {pval:.4f}"

# Correct
diagnostics = []
_norm1 = _check_normality(x, label="Group 1")
_norm2 = _check_normality(y, label="Group 2")
if _norm1: diagnostics.append(_norm1)
if _norm2: diagnostics.append(_norm2)
_eq = _check_equal_variance(x, y, labels=["Group 1", "Group 2"])
if _eq: diagnostics.append(_eq)
stat, pval = stats.ttest_ind(x, y)
result["diagnostics"] = diagnostics
```

### **14.3 Bayesian Without Frequentist (or Vice Versa)**

<!-- rule: recommended -->

**DISCOURAGED:** Running only one paradigm when both are available. The Bayesian shadow exists to provide convergent evidence — omitting it loses information.

### **14.4 Missing Guide Observation**

<!-- rule: mandatory -->

**PROHIBITED:** Returning analysis results without a `guide_observation` string. The guide observation is the primary UI-facing interpretation and must always be populated.

### **14.5 Wald Confidence Interval for Proportions**

<!-- rule: mandatory -->

**PROHIBITED:** Using the Wald interval (`p̂ ± z√(p̂(1-p̂)/n)`) for proportion confidence intervals. Wilson score interval is required.

### **14.6 Hardcoded Alpha Without Parameterization**

<!-- rule: recommended -->

**DISCOURAGED:** Hardcoding α = 0.05 without accepting it as a parameter. Tests should accept `alpha` from config and default to 0.05.

---

## **15. ACCEPTANCE CRITERIA**

| Criterion | Validation Method |
|-----------|-------------------|
| Every parametric test in stats.py calls `_check_normality()` | Code review / grep |
| Every parametric test in stats.py calls `_cross_validate()` | Code review / grep |
| Every test producing a p-value also produces an effect size | Code review |
| `_practical_block()` called for all tests with effect sizes | Code review / grep |
| `_bayesian_shadow()` called for t-tests, ANOVA, correlation, chi-square, proportion | Code review |
| `_evidence_grade()` called after shadow computation | Code review |
| All 8 Nelson rules implemented in `_spc_nelson_rules()` | Unit test |
| Nelson rules applied to I-MR, X̄-R, X̄-S, NP, C charts | Code review |
| Wilson CI used for proportion intervals (not Wald) | Code review |
| Every analysis sets `guide_observation` to a non-empty string | Code review / grep |
| Cohen (1988) thresholds match §5.1 table | Unit test |
| BF interpretation scale matches §6.3 table | Unit test |
| Evidence grade scoring matches §8.1 tables | Unit test |
| Power analysis produces correct sample sizes for known inputs | Unit test |
| STAT-001 assertion tags parse correctly | `python3 manage.py run_compliance --standards` |
| Calibration reference pool has ≥15 cases across ≥5 categories | Unit test |
| Calibration runner returns per-case results with expected fields | Unit test |
| Same date seed produces same case selection | Unit test |
| Known-null t-test case passes calibration (N(100,15) vs μ₀=100) | Unit test |
| Failed calibration case creates DriftViolation with enforcement_check="CAL" | Unit test |

---

## **16. COMPLIANCE MAPPING**

<!-- table: compliance-mapping -->
<!-- control: ISO 9001:2015 §8.5.1 -->
<!-- control: SOC 2 CC4.1 -->

| Requirement | External Standard | STAT-001 Section |
|-------------|-------------------|------------------|
| Valid statistical methods for process control | ISO 9001:2015 §8.5.1 | §4 (Methodology principles) |
| Effect size reporting with significance tests | APA 7th Edition §6.1 | §5 (Effect size framework) |
| Monitoring and evaluation of controls | SOC 2 CC4.1 | §8 (Evidence synthesis), §9 (SPC) |
| Process stability monitoring | ISO 9001:2015 §8.5.1 | §9 (Nelson rules) |
| Calibration of analysis functions | ISO 9001:2015 §7.1.5 | §13 (Statistical calibration) |

---

## **REVISION HISTORY**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-03 | Eric + Claude | Initial release — 11 principles, effect size framework, Bayesian insurance, robustness framework, evidence synthesis, SPC, power analysis, CI methods |
| 1.1 | 2026-03-04 | Eric + Claude | Added §13 Statistical Calibration — reference pool, daily rotation, drift detection, dashboard visualization |
| 1.2 | 2026-03-04 | Eric + Claude | Fixed DOC-001 §4 compliance — moved calibration to §13 (domain content), renumbered terminal trio to §14/§15/§16, added calibration acceptance criteria |
