"""Coverage tests — exercise every stats analysis_id with valid input.

One test per analysis_id not already covered by golden files.
Verifies output schema: plots list, summary string, statistics bounds.

Standard: CAL-001 §6 (Statistical Correctness Verification)
Compliance: SOC 2 CC4.1
<!-- test: agents_api.tests.test_stats_coverage -->
"""

import numpy as np
import pandas as pd
from django.test import TestCase


def _run(analysis_id, config, data_dict):
    """Run stats analysis — no exception masking (TST-001 §11.6)."""
    from agents_api.dsw.stats import run_statistical_analysis

    df = pd.DataFrame(data_dict)
    return run_statistical_analysis(df, analysis_id, config)


def _check_schema(tc, r):
    """Verify output schema per CAL-001 §6 / TST-001 §10.6."""
    aid = tc._testMethodName
    tc.assertIsInstance(r, dict, f"{aid} did not return a dict")
    tc.assertIn("plots", r, f"{aid} missing 'plots' key")
    tc.assertIsInstance(r["plots"], list, f"{aid} plots is not a list")
    tc.assertIn("summary", r, f"{aid} missing 'summary' key")
    # Validate statistical bounds if present
    stats = r.get("statistics", {})
    if isinstance(stats, dict):
        if "p_value" in stats and stats["p_value"] is not None:
            tc.assertGreaterEqual(stats["p_value"], 0, f"{aid} p_value < 0")
            tc.assertLessEqual(stats["p_value"], 1, f"{aid} p_value > 1")


# --- Shared test data ---

NORMAL_50 = list(np.random.RandomState(42).normal(100, 15, 50))
NORMAL_50B = list(np.random.RandomState(99).normal(105, 15, 50))
GROUPS_50 = ["A"] * 25 + ["B"] * 25
GROUPS_ABC = ["A"] * 17 + ["B"] * 17 + ["C"] * 16
BINARY_50 = [0] * 25 + [1] * 25
COUNT_50 = list(np.random.RandomState(42).poisson(5, 50))
TIME_50 = list(range(50))
FACTOR_A = ["L1", "L1", "L2", "L2", "L3", "L3"] * 8 + ["L1", "L2"]
FACTOR_B = ["X", "Y"] * 25


# ===========================================================================
# Parametric — uncovered analysis_ids
# ===========================================================================


class ParametricCoverageTest(TestCase):
    """Parametric analyses missing from golden files."""

    def test_anova2(self):
        r = _run(
            "anova2",
            {"response": "y", "factor1": "a", "factor2": "b"},
            {"y": NORMAL_50, "a": FACTOR_A, "b": FACTOR_B},
        )
        _check_schema(self, r)

    def test_repeated_measures_anova(self):
        # Within-subject design: same subjects measured under 3 conditions
        subj = list(range(15)) * 3
        cond = ["c1"] * 15 + ["c2"] * 15 + ["c3"] * 15
        vals = NORMAL_50[:45]
        r = _run(
            "repeated_measures_anova",
            {"response": "y", "subject": "subj", "within_factor": "cond"},
            {"y": vals, "subj": subj, "cond": cond},
        )
        _check_schema(self, r)

    def test_split_plot_anova(self):
        subj = list(range(10)) * 4
        whole = (["W1"] * 10 + ["W2"] * 10) * 2
        sub = ["S1"] * 20 + ["S2"] * 20
        vals = list(np.random.RandomState(42).normal(50, 10, 40))
        r = _run(
            "split_plot_anova",
            {"response": "y", "whole_plot": "w", "sub_plot": "s", "subject": "subj"},
            {"y": vals, "w": whole, "s": sub, "subj": subj},
        )
        _check_schema(self, r)

    def test_normality(self):
        r = _run("normality", {"var": "x"}, {"x": NORMAL_50})
        _check_schema(self, r)

    def test_f_test(self):
        r = _run(
            "f_test",
            {"var": "x", "group_var": "g"},
            {"x": NORMAL_50 + NORMAL_50B, "g": ["A"] * 50 + ["B"] * 50},
        )
        _check_schema(self, r)

    def test_equivalence(self):
        r = _run(
            "equivalence",
            {"var": "x", "group_var": "g", "margin": 10},
            {"x": NORMAL_50 + NORMAL_50B, "g": ["A"] * 50 + ["B"] * 50},
        )
        _check_schema(self, r)

    def test_sign_test(self):
        r = _run(
            "sign_test",
            {"var": "x", "hypothesized_median": 100},
            {"x": NORMAL_50},
        )
        _check_schema(self, r)


# ===========================================================================
# Nonparametric — uncovered
# ===========================================================================


class NonparametricCoverageTest(TestCase):
    """Nonparametric analyses missing from golden files."""

    def test_runs_test(self):
        r = _run("runs_test", {"var": "x"}, {"x": NORMAL_50})
        _check_schema(self, r)

    def test_multi_vari(self):
        r = _run(
            "multi_vari",
            {"response": "y", "factors": ["a", "b"]},
            {"y": NORMAL_50, "a": FACTOR_A, "b": FACTOR_B},
        )
        _check_schema(self, r)


# ===========================================================================
# Regression — uncovered
# ===========================================================================


class RegressionCoverageTest(TestCase):
    """Regression analyses missing from golden files."""

    def test_glm(self):
        r = _run(
            "glm",
            {"response": "y", "predictors": ["x1", "x2"], "family": "gaussian"},
            {
                "y": NORMAL_50,
                "x1": list(np.random.RandomState(1).normal(0, 1, 50)),
                "x2": list(np.random.RandomState(2).normal(0, 1, 50)),
            },
        )
        _check_schema(self, r)

    def test_nominal_logistic(self):
        cats = np.random.RandomState(42).choice(["A", "B", "C"], 50).tolist()
        r = _run(
            "nominal_logistic",
            {"response": "y", "predictors": ["x"]},
            {"y": cats, "x": NORMAL_50},
        )
        _check_schema(self, r)

    def test_ordinal_logistic(self):
        cats = np.random.RandomState(42).choice([1, 2, 3, 4], 50).tolist()
        r = _run(
            "ordinal_logistic",
            {"response": "y", "predictors": ["x"]},
            {"y": cats, "x": NORMAL_50},
        )
        _check_schema(self, r)

    def test_orthogonal_regression(self):
        r = _run(
            "orthogonal_regression",
            {"var1": "x", "var2": "y"},
            {"x": NORMAL_50, "y": NORMAL_50B},
        )
        _check_schema(self, r)

    def test_nonlinear_regression(self):
        x = list(np.linspace(0.1, 10, 50))
        y = [3 * np.exp(-0.5 * xi) + np.random.RandomState(42).normal(0, 0.1) for xi in x]
        r = _run(
            "nonlinear_regression",
            {"var_x": "x", "var_y": "y", "model": "exponential"},
            {"x": x, "y": y},
        )
        _check_schema(self, r)

    def test_poisson_regression(self):
        r = _run(
            "poisson_regression",
            {"response": "y", "predictors": ["x"]},
            {"y": COUNT_50, "x": NORMAL_50},
        )
        _check_schema(self, r)

    def test_robust_regression(self):
        r = _run(
            "robust_regression",
            {"response": "y", "predictors": ["x"]},
            {"y": NORMAL_50, "x": NORMAL_50B},
        )
        _check_schema(self, r)

    def test_best_subsets(self):
        r = _run(
            "best_subsets",
            {"response": "y", "predictors": ["x1", "x2", "x3"]},
            {
                "y": NORMAL_50,
                "x1": list(np.random.RandomState(1).normal(0, 1, 50)),
                "x2": list(np.random.RandomState(2).normal(0, 1, 50)),
                "x3": list(np.random.RandomState(3).normal(0, 1, 50)),
            },
        )
        _check_schema(self, r)


# ===========================================================================
# Posthoc — uncovered
# ===========================================================================


class PosthocCoverageTest(TestCase):
    """Post-hoc analyses missing from golden files."""

    def test_main_effects(self):
        r = _run(
            "main_effects",
            {"response": "y", "factors": ["a", "b"]},
            {"y": NORMAL_50, "a": FACTOR_A, "b": FACTOR_B},
        )
        _check_schema(self, r)

    def test_interaction(self):
        r = _run(
            "interaction",
            {"response": "y", "factor1": "a", "factor2": "b"},
            {"y": NORMAL_50, "a": FACTOR_A, "b": FACTOR_B},
        )
        _check_schema(self, r)

    def test_dunnett(self):
        r = _run(
            "dunnett",
            {"response": "y", "factor": "g", "control": "A"},
            {"y": NORMAL_50, "g": GROUPS_ABC},
        )
        _check_schema(self, r)

    def test_games_howell(self):
        r = _run(
            "games_howell",
            {"response": "y", "factor": "g"},
            {"y": NORMAL_50, "g": GROUPS_ABC},
        )
        _check_schema(self, r)

    def test_dunn(self):
        r = _run(
            "dunn",
            {"response": "y", "factor": "g"},
            {"y": NORMAL_50, "g": GROUPS_ABC},
        )
        _check_schema(self, r)

    def test_scheffe_test(self):
        r = _run(
            "scheffe_test",
            {"response": "y", "factor": "g"},
            {"y": NORMAL_50, "g": GROUPS_ABC},
        )
        _check_schema(self, r)

    def test_bonferroni_test(self):
        r = _run(
            "bonferroni_test",
            {"response": "y", "factor": "g"},
            {"y": NORMAL_50, "g": GROUPS_ABC},
        )
        _check_schema(self, r)

    def test_hsu_mcb(self):
        r = _run(
            "hsu_mcb",
            {"response": "y", "factor": "g"},
            {"y": NORMAL_50, "g": GROUPS_ABC},
        )
        _check_schema(self, r)


# ===========================================================================
# Quality — 0 golden files, all need coverage
# ===========================================================================


class QualityCoverageTest(TestCase):
    """Quality analyses — zero golden file coverage currently."""

    def test_variance_test_one_sample(self):
        r = _run(
            "variance_test",
            {"var1": "x", "sigma0": 15},
            {"x": NORMAL_50},
        )
        _check_schema(self, r)

    def test_variance_test_two_col(self):
        r = _run(
            "variance_test",
            {"var1": "x", "var2": "y"},
            {"x": NORMAL_50, "y": NORMAL_50B},
        )
        _check_schema(self, r)

    def test_variance_components(self):
        r = _run(
            "variance_components",
            {"response": "y", "factors": ["a"]},
            {"y": NORMAL_50, "a": GROUPS_50},
        )
        _check_schema(self, r)

    def test_capability_sixpack(self):
        r = _run(
            "capability_sixpack",
            {"var1": "x", "lsl": 70, "usl": 130},
            {"x": NORMAL_50},
        )
        _check_schema(self, r)

    def test_nonnormal_capability_np(self):
        r = _run(
            "nonnormal_capability_np",
            {"var1": "x", "lsl": 70, "usl": 130},
            {"x": NORMAL_50},
        )
        _check_schema(self, r)

    def test_attribute_capability(self):
        defects = [0, 1, 0, 0, 1, 0, 0, 0, 1, 0] * 5
        r = _run(
            "attribute_capability",
            {"var1": "x", "n_inspected": 10},
            {"x": defects},
        )
        _check_schema(self, r)

    def test_acceptance_sampling(self):
        r = _run(
            "acceptance_sampling",
            {"lot_size": 1000, "aql": 1.0, "ltpd": 5.0},
            {"x": NORMAL_50},  # dummy — acceptance sampling uses config, not data
        )
        _check_schema(self, r)

    def test_variable_acceptance_sampling(self):
        r = _run(
            "variable_acceptance_sampling",
            {"lot_size": 500, "aql": 1.0, "var1": "x", "lsl": 70, "usl": 130},
            {"x": NORMAL_50},
        )
        _check_schema(self, r)

    def test_multiple_plan_comparison(self):
        r = _run(
            "multiple_plan_comparison",
            {"plans": [{"n": 50, "c": 2}, {"n": 80, "c": 3}]},
            {"x": NORMAL_50},
        )
        _check_schema(self, r)

    def test_anom(self):
        r = _run(
            "anom",
            {"response": "y", "factor": "g"},
            {"y": NORMAL_50, "g": GROUPS_50},
        )
        _check_schema(self, r)


# ===========================================================================
# Advanced — power, MSA, time series, survival
# ===========================================================================


class AdvancedPowerTest(TestCase):
    """Power and sample size analyses."""

    def test_power_z(self):
        r = _run("power_z", {"delta": 0.5, "sigma": 1.0, "alpha": 0.05, "power": 0.80}, {"x": [1]})
        _check_schema(self, r)

    def test_power_1prop(self):
        r = _run("power_1prop", {"p0": 0.5, "pa": 0.7, "alpha": 0.05, "power": 0.80}, {"x": [1]})
        _check_schema(self, r)

    def test_power_2prop(self):
        r = _run("power_2prop", {"p1": 0.5, "p2": 0.7, "alpha": 0.05, "power": 0.80}, {"x": [1]})
        _check_schema(self, r)

    def test_power_1variance(self):
        r = _run("power_1variance", {"sigma0": 1.0, "sigma1": 1.5, "alpha": 0.05, "power": 0.80}, {"x": [1]})
        _check_schema(self, r)

    def test_power_2variance(self):
        r = _run("power_2variance", {"sigma1": 1.0, "sigma2": 1.5, "alpha": 0.05, "power": 0.80}, {"x": [1]})
        _check_schema(self, r)

    def test_power_equivalence(self):
        r = _run(
            "power_equivalence",
            {"delta": 0.5, "sigma": 1.0, "alpha": 0.05, "power": 0.80, "margin": 1.0},
            {"x": [1]},
        )
        _check_schema(self, r)

    def test_power_doe(self):
        r = _run(
            "power_doe",
            {"n_factors": 3, "n_levels": 2, "effect_size": 1.0, "sigma": 1.0},
            {"x": [1]},
        )
        _check_schema(self, r)

    def test_sample_size_ci(self):
        r = _run(
            "sample_size_ci",
            {"sigma": 1.0, "margin": 0.5, "conf": 95},
            {"x": [1]},
        )
        _check_schema(self, r)

    def test_sample_size_tolerance(self):
        r = _run(
            "sample_size_tolerance",
            {"coverage": 0.95, "conf": 0.95},
            {"x": [1]},
        )
        _check_schema(self, r)


class AdvancedMSATest(TestCase):
    """Measurement System Analysis / Gage R&R."""

    def test_gage_rr(self):
        # 3 operators × 10 parts × 2 replicates = 60
        parts = list(range(1, 11)) * 6
        ops = (["Op1"] * 10 + ["Op2"] * 10 + ["Op3"] * 10) * 2
        vals = list(np.random.RandomState(42).normal(50, 2, 60))
        r = _run(
            "gage_rr",
            {"measurement": "y", "part": "part", "operator": "op"},
            {"y": vals, "part": parts, "op": ops},
        )
        _check_schema(self, r)

    def test_gage_rr_expanded(self):
        parts = list(range(1, 11)) * 6
        ops = (["Op1"] * 10 + ["Op2"] * 10 + ["Op3"] * 10) * 2
        fixture = ["F1"] * 30 + ["F2"] * 30
        vals = list(np.random.RandomState(42).normal(50, 2, 60))
        r = _run(
            "gage_rr_expanded",
            {"measurement": "y", "part": "part", "operator": "op", "factors": ["fixture"]},
            {"y": vals, "part": parts, "op": ops, "fixture": fixture},
        )
        _check_schema(self, r)

    def test_gage_rr_nested(self):
        parts = list(range(1, 11)) * 6
        ops = (["Op1"] * 10 + ["Op2"] * 10 + ["Op3"] * 10) * 2
        vals = list(np.random.RandomState(42).normal(50, 2, 60))
        r = _run(
            "gage_rr_nested",
            {"measurement": "y", "part": "part", "operator": "op"},
            {"y": vals, "part": parts, "op": ops},
        )
        _check_schema(self, r)

    def test_gage_type1(self):
        vals = list(np.random.RandomState(42).normal(50, 0.5, 25))
        r = _run(
            "gage_type1",
            {"measurement": "y", "reference": 50.0, "tolerance": 2.0},
            {"y": vals},
        )
        _check_schema(self, r)

    def test_gage_linearity_bias(self):
        refs = list(np.linspace(10, 100, 25))
        measured = [r + np.random.RandomState(42).normal(0.1, 0.3) for r in refs]
        r = _run(
            "gage_linearity_bias",
            {"measurement": "y", "reference": "ref"},
            {"y": measured, "ref": refs},
        )
        _check_schema(self, r)

    def test_attribute_gage(self):
        ratings = np.random.RandomState(42).choice([0, 1], 50).tolist()
        refs = np.random.RandomState(99).choice([0, 1], 50).tolist()
        r = _run(
            "attribute_gage",
            {"measurement": "rate", "reference": "ref", "appraiser": "op"},
            {"rate": ratings, "ref": refs, "op": GROUPS_50},
        )
        _check_schema(self, r)

    def test_attribute_agreement(self):
        # Stacked format: appraiser, part, rating columns
        parts = list(range(1, 16)) * 2
        appraisers = ["A1"] * 15 + ["A2"] * 15
        ratings = np.random.RandomState(42).choice(["P", "F"], 30).tolist()
        r = _run(
            "attribute_agreement",
            {"appraiser": "appraiser", "part": "part", "rating": "rating"},
            {"appraiser": appraisers, "part": parts, "rating": ratings},
        )
        _check_schema(self, r)

    def test_icc(self):
        # Stacked format: rater, subject, value
        subjects = list(range(1, 16)) * 2
        raters = ["R1"] * 15 + ["R2"] * 15
        vals = list(np.random.RandomState(42).normal(5, 1, 15)) + [
            x + np.random.RandomState(43).normal(0, 0.5) for x in np.random.RandomState(42).normal(5, 1, 15)
        ]
        r = _run(
            "icc",
            {"rater": "rater", "subject": "subject", "value": "value"},
            {"rater": raters, "subject": subjects, "value": vals},
        )
        _check_schema(self, r)

    def test_krippendorff_alpha(self):
        # Stacked format: rater, subject, value
        subjects = list(range(1, 11)) * 3
        raters = ["C1"] * 10 + ["C2"] * 10 + ["C3"] * 10
        vals = (
            np.random.RandomState(42).choice([1, 2, 3], 10).tolist()
            + np.random.RandomState(43).choice([1, 2, 3], 10).tolist()
            + np.random.RandomState(44).choice([1, 2, 3], 10).tolist()
        )
        r = _run(
            "krippendorff_alpha",
            {"rater": "rater", "subject": "subject", "value": "value"},
            {"rater": raters, "subject": subjects, "value": vals},
        )
        _check_schema(self, r)

    def test_bland_altman(self):
        m1 = list(np.random.RandomState(42).normal(100, 10, 30))
        m2 = [x + np.random.RandomState(43).normal(0.5, 2) for x in m1]
        r = _run(
            "bland_altman",
            {"method1": "m1", "method2": "m2"},
            {"m1": m1, "m2": m2},
        )
        _check_schema(self, r)


class AdvancedTimeSeriesTest(TestCase):
    """Time series analyses."""

    def test_acf_pacf(self):
        ts = list(np.random.RandomState(42).normal(0, 1, 100))
        r = _run("acf_pacf", {"var": "y"}, {"y": ts})
        _check_schema(self, r)

    def test_arima(self):
        ts = list(np.cumsum(np.random.RandomState(42).normal(0, 1, 60)))
        r = _run("arima", {"var": "y", "p": 1, "d": 1, "q": 1}, {"y": ts})
        _check_schema(self, r)

    def test_sarima(self):
        ts = list(np.cumsum(np.random.RandomState(42).normal(0, 1, 60)))
        r = _run(
            "sarima",
            {"var": "y", "p": 1, "d": 1, "q": 1, "P": 0, "D": 0, "Q": 0, "s": 12},
            {"y": ts},
        )
        _check_schema(self, r)

    def test_decomposition(self):
        ts = list(np.sin(np.linspace(0, 4 * np.pi, 48)) * 10 + np.random.RandomState(42).normal(0, 1, 48) + 50)
        r = _run("decomposition", {"var": "y", "period": 12}, {"y": ts})
        _check_schema(self, r)

    def test_granger(self):
        x = list(np.random.RandomState(42).normal(0, 1, 50))
        y = [0] + [0.5 * x[i - 1] + np.random.RandomState(43).normal(0, 0.5) for i in range(1, 50)]
        r = _run("granger", {"var_x": "x", "var_y": "y", "max_lag": 3}, {"x": x, "y": y})
        _check_schema(self, r)

    def test_changepoint(self):
        seg1 = list(np.random.RandomState(42).normal(10, 1, 30))
        seg2 = list(np.random.RandomState(43).normal(15, 1, 30))
        r = _run("changepoint", {"var": "y"}, {"y": seg1 + seg2})
        _check_schema(self, r)

    def test_ccf(self):
        x = list(np.random.RandomState(42).normal(0, 1, 50))
        y = list(np.random.RandomState(43).normal(0, 1, 50))
        r = _run("ccf", {"var1": "x", "var2": "y"}, {"x": x, "y": y})
        _check_schema(self, r)


class AdvancedSurvivalTest(TestCase):
    """Survival / reliability analyses via stats_advanced."""

    def test_kaplan_meier(self):
        times = list(np.random.RandomState(42).exponential(10, 40))
        events = np.random.RandomState(43).choice([0, 1], 40, p=[0.3, 0.7]).tolist()
        r = _run(
            "kaplan_meier",
            {"time": "t", "event": "e"},
            {"t": times, "e": events},
        )
        _check_schema(self, r)

    def test_cox_ph(self):
        times = list(np.random.RandomState(42).exponential(10, 40))
        events = np.random.RandomState(43).choice([0, 1], 40, p=[0.3, 0.7]).tolist()
        covar = list(np.random.RandomState(44).normal(0, 1, 40))
        r = _run(
            "cox_ph",
            {"time": "t", "event": "e", "covariates": ["x"]},
            {"t": times, "e": events, "x": covar},
        )
        _check_schema(self, r)

    def test_weibull(self):
        times = list(np.random.RandomState(42).exponential(10, 40))
        r = _run(
            "weibull",
            {"var": "t"},
            {"t": times},
        )
        _check_schema(self, r)


# ===========================================================================
# Exploratory — 0 golden files, all need coverage
# ===========================================================================


class ExploratoryCoverageTest(TestCase):
    """Exploratory analyses — zero golden file coverage currently."""

    def test_descriptive(self):
        r = _run("descriptive", {"vars": ["x", "y"]}, {"x": NORMAL_50, "y": NORMAL_50B})
        _check_schema(self, r)

    def test_data_profile(self):
        r = _run("data_profile", {}, {"x": NORMAL_50, "y": NORMAL_50B, "g": GROUPS_50})
        _check_schema(self, r)

    def test_auto_profile(self):
        r = _run("auto_profile", {}, {"x": NORMAL_50, "y": NORMAL_50B, "g": GROUPS_50})
        _check_schema(self, r)

    def test_graphical_summary(self):
        r = _run("graphical_summary", {"var1": "x"}, {"x": NORMAL_50})
        _check_schema(self, r)

    def test_missing_data_analysis(self):
        data_with_nan = NORMAL_50[:40] + [None] * 10
        r = _run("missing_data_analysis", {}, {"x": data_with_nan, "y": NORMAL_50})
        _check_schema(self, r)

    def test_outlier_analysis(self):
        r = _run("outlier_analysis", {"var1": "x"}, {"x": NORMAL_50})
        _check_schema(self, r)

    def test_duplicate_analysis(self):
        data_dup = NORMAL_50 + NORMAL_50[:10]
        r = _run("duplicate_analysis", {}, {"x": data_dup})
        _check_schema(self, r)

    def test_bootstrap_ci(self):
        r = _run("bootstrap_ci", {"var": "x", "statistic": "mean"}, {"x": NORMAL_50})
        _check_schema(self, r)

    def test_box_cox(self):
        pos_data = [abs(x) + 1 for x in NORMAL_50]
        r = _run("box_cox", {"var": "x"}, {"x": pos_data})
        _check_schema(self, r)

    def test_johnson_transform(self):
        r = _run("johnson_transform", {"var": "x"}, {"x": NORMAL_50})
        _check_schema(self, r)

    def test_run_chart(self):
        r = _run("run_chart", {"var": "x"}, {"x": NORMAL_50})
        _check_schema(self, r)

    def test_grubbs_test(self):
        r = _run("grubbs_test", {"var": "x"}, {"x": NORMAL_50})
        _check_schema(self, r)

    def test_tolerance_interval(self):
        r = _run(
            "tolerance_interval",
            {"var": "x", "proportion": 0.95, "confidence": 0.95},
            {"x": NORMAL_50},
        )
        _check_schema(self, r)

    def test_effect_size_calculator(self):
        r = _run(
            "effect_size_calculator",
            {"var1": "x", "var2": "y"},
            {"x": NORMAL_50, "y": NORMAL_50B},
        )
        _check_schema(self, r)

    def test_meta_analysis(self):
        effects = [0.3, 0.5, 0.4, 0.6, 0.35]
        ses = [0.1, 0.15, 0.12, 0.08, 0.11]
        r = _run(
            "meta_analysis",
            {"effects": "effect", "standard_errors": "se"},
            {"effect": effects, "se": ses},
        )
        _check_schema(self, r)

    def test_distribution_fit(self):
        r = _run("distribution_fit", {"var1": "x"}, {"x": NORMAL_50})
        _check_schema(self, r)

    def test_mixture_model(self):
        # Bimodal data
        mix = list(np.random.RandomState(42).normal(50, 5, 25)) + list(np.random.RandomState(43).normal(80, 5, 25))
        r = _run("mixture_model", {"var1": "x", "n_components": 2}, {"x": mix})
        _check_schema(self, r)

    def test_copula(self):
        r = _run(
            "copula",
            {"var1": "x", "var2": "y"},
            {"x": NORMAL_50, "y": NORMAL_50B},
        )
        _check_schema(self, r)

    def test_sprt(self):
        r = _run(
            "sprt",
            {"var1": "x", "p0": 0.05, "p1": 0.10},
            {"x": BINARY_50},
        )
        _check_schema(self, r)

    def test_prop_1sample(self):
        r = _run(
            "prop_1sample",
            {"var1": "x", "p0": 0.5},
            {"x": BINARY_50},
        )
        _check_schema(self, r)

    def test_prop_2sample(self):
        r = _run(
            "prop_2sample",
            {"var1": "x", "group_var": "g"},
            {"x": BINARY_50, "g": GROUPS_50},
        )
        _check_schema(self, r)

    def test_hotelling_t2(self):
        r = _run(
            "hotelling_t2",
            {"responses": ["x", "y"], "group_var": "g"},
            {"x": NORMAL_50, "y": NORMAL_50B, "g": GROUPS_50},
        )
        _check_schema(self, r)

    def test_manova(self):
        r = _run(
            "manova",
            {"responses": ["x", "y"], "factor": "g"},
            {"x": NORMAL_50, "y": NORMAL_50B, "g": GROUPS_50},
        )
        _check_schema(self, r)

    def test_nested_anova(self):
        r = _run(
            "nested_anova",
            {"response": "y", "factor": "a", "nested": "b"},
            {"y": NORMAL_50, "a": GROUPS_50, "b": FACTOR_B},
        )
        _check_schema(self, r)

    def test_poisson_1sample(self):
        r = _run(
            "poisson_1sample",
            {"var1": "x", "mu0": 5},
            {"x": COUNT_50},
        )
        _check_schema(self, r)

    def test_poisson_2sample(self):
        counts2 = list(np.random.RandomState(99).poisson(7, 50))
        r = _run(
            "poisson_2sample",
            {"var1": "x", "var2": "y"},
            {"x": COUNT_50, "y": counts2},
        )
        _check_schema(self, r)

    def test_fisher_exact(self):
        # Already in golden files but also in exploratory
        r = _run(
            "fisher_exact",
            {"var1": "x", "var2": "y"},
            {"x": BINARY_50, "y": np.random.RandomState(42).choice([0, 1], 50).tolist()},
        )
        _check_schema(self, r)

    def test_chi2(self):
        cat1 = np.random.RandomState(42).choice(["A", "B", "C"], 50).tolist()
        cat2 = np.random.RandomState(43).choice(["X", "Y"], 50).tolist()
        r = _run(
            "chi2",
            {"var1": "c1", "var2": "c2"},
            {"c1": cat1, "c2": cat2},
        )
        _check_schema(self, r)
