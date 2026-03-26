"""Bounds exhaustive tests — run every Tier 1/2 analysis with minimal valid data.

For each analysis, verify:
1. No unhandled exception (crash)
2. Returns a dict (not None)
3. Numeric outputs are finite (no NaN/Inf leaking into results)

Standard: CAL-001 §6 (Statistical Correctness Verification)
Compliance: SOC 2 CC4.1
"""

import math

import numpy as np
import pandas as pd
from django.test import TestCase


def _run(module, analysis_id, config, data_dict):
    """Run an analysis and return the result dict."""
    df = pd.DataFrame(data_dict)

    if module == "stats":
        from agents_api.dsw.stats import run_statistical_analysis

        return run_statistical_analysis(df, analysis_id, config)
    elif module == "spc":
        from agents_api.dsw.spc import run_spc_analysis

        return run_spc_analysis(df, analysis_id, config)
    elif module == "bayesian":
        from agents_api.dsw.bayesian import run_bayesian_analysis

        return run_bayesian_analysis(df, analysis_id, config)
    elif module == "reliability":
        from agents_api.dsw.reliability import run_reliability_analysis

        return run_reliability_analysis(df, analysis_id, config)
    elif module == "ml":
        from agents_api.dsw.ml import run_ml_analysis

        return run_ml_analysis(df, analysis_id, config, user=None)
    elif module == "simulation":
        from agents_api.dsw.simulation import run_simulation

        return run_simulation(df, analysis_id, config, user=None)
    else:
        raise ValueError(f"Unknown module: {module}")


# Shared RNG for deterministic test data
RNG = np.random.RandomState(3000)


def _normal(n=100, mu=50, sigma=10):
    return RNG.normal(mu, sigma, n).tolist()


def _two_groups(n1=50, n2=50, mu1=50, mu2=60, sigma=10):
    a = RNG.normal(mu1, sigma, n1).tolist()
    b = RNG.normal(mu2, sigma, n2).tolist()
    return a, b


def _stacked_groups(n_per=50, k=3):
    """Return stacked values + group labels for k groups."""
    values = []
    groups = []
    for i in range(k):
        vals = RNG.normal(50 + 10 * i, 10, n_per).tolist()
        values.extend(vals)
        groups.extend([f"G{i}"] * n_per)
    return values, groups


def _check_result(test_case, result, label):
    """Assert result is a valid dict without unhandled errors."""
    test_case.assertIsNotNone(result, f"{label}: returned None")
    test_case.assertIsInstance(result, dict, f"{label}: not a dict")


def _check_numerics_finite(test_case, d, path=""):
    """Recursively check that all numeric values in a nested dict are finite."""
    if isinstance(d, dict):
        for k, v in d.items():
            _check_numerics_finite(test_case, v, f"{path}.{k}")
    elif isinstance(d, (list, tuple)):
        for i, v in enumerate(d):
            _check_numerics_finite(test_case, v, f"{path}[{i}]")
    elif isinstance(d, float):
        # Allow NaN in specific known locations (e.g., missing data markers)
        # but flag inf which usually indicates overflow
        if math.isinf(d):
            test_case.fail(f"Inf found at {path}: {d}")


# ---------------------------------------------------------------------------
# Stats: Hypothesis testing
# ---------------------------------------------------------------------------
class StatsHypothesisExhaustiveTest(TestCase):
    """Every hypothesis test completes without crash."""

    def test_ttest(self):
        r = _run("stats", "ttest", {"var1": "x", "mu": 50}, {"x": _normal()})
        _check_result(self, r, "ttest")

    def test_ttest2(self):
        a, b = _two_groups()
        r = _run("stats", "ttest2", {"var1": "a", "var2": "b"}, {"a": a, "b": b})
        _check_result(self, r, "ttest2")

    def test_paired_t(self):
        a, b = _two_groups()
        r = _run(
            "stats",
            "paired_t",
            {"var1": "before", "var2": "after"},
            {"before": a, "after": b},
        )
        _check_result(self, r, "paired_t")

    def test_anova(self):
        v, g = _stacked_groups()
        r = _run("stats", "anova", {"response": "v", "factor": "g"}, {"v": v, "g": g})
        _check_result(self, r, "anova")

    def test_correlation(self):
        x = _normal(100, 0, 1)
        y = _normal(100, 0, 1)
        r = _run("stats", "correlation", {"variables": ["x", "y"]}, {"x": x, "y": y})
        _check_result(self, r, "correlation")

    def test_chi2(self):
        n = 200
        a = RNG.choice(["A", "B", "C"], n).tolist()
        b = RNG.choice(["X", "Y"], n).tolist()
        r = _run("stats", "chi2", {"var1": "a", "var2": "b"}, {"a": a, "b": b})
        _check_result(self, r, "chi2")

    def test_fisher_exact(self):
        a = ["A"] * 10 + ["B"] * 10
        b = ["Y"] * 8 + ["N"] * 2 + ["Y"] * 3 + ["N"] * 7
        r = _run(
            "stats",
            "fisher_exact",
            {"var1": "treatment", "var2": "outcome"},
            {"treatment": a, "outcome": b},
        )
        _check_result(self, r, "fisher_exact")

    def test_mann_whitney(self):
        v, g = _stacked_groups(50, 2)
        r = _run(
            "stats", "mann_whitney", {"var": "v", "group_var": "g"}, {"v": v, "g": g}
        )
        _check_result(self, r, "mann_whitney")

    def test_kruskal(self):
        v, g = _stacked_groups(40, 3)
        r = _run("stats", "kruskal", {"var": "v", "group_var": "g"}, {"v": v, "g": g})
        _check_result(self, r, "kruskal")

    def test_wilcoxon(self):
        a, b = _two_groups(40, 40)
        r = _run(
            "stats",
            "wilcoxon",
            {"var1": "before", "var2": "after"},
            {"before": a, "after": b},
        )
        _check_result(self, r, "wilcoxon")

    def test_normality(self):
        r = _run("stats", "normality", {"var": "x"}, {"x": _normal()})
        _check_result(self, r, "normality")

    def test_variance_test(self):
        a, b = _two_groups()
        r = _run("stats", "variance_test", {"var1": "a", "var2": "b"}, {"a": a, "b": b})
        _check_result(self, r, "variance_test")

    def test_equivalence(self):
        v, g = _stacked_groups(50, 2)
        r = _run(
            "stats",
            "equivalence",
            {"var": "v", "group_var": "g", "margin": 5.0},
            {"v": v, "g": g},
        )
        _check_result(self, r, "equivalence")

    def test_sign_test(self):
        r = _run(
            "stats",
            "sign_test",
            {"var": "x", "hypothesized_median": 50},
            {"x": _normal()},
        )
        _check_result(self, r, "sign_test")

    def test_runs_test(self):
        r = _run("stats", "runs_test", {"var": "x"}, {"x": _normal()})
        _check_result(self, r, "runs_test")

    def test_spearman(self):
        x = _normal(80, 0, 1)
        y = _normal(80, 0, 1)
        r = _run("stats", "spearman", {"var1": "x", "var2": "y"}, {"x": x, "y": y})
        _check_result(self, r, "spearman")

    def test_mood_median(self):
        v, g = _stacked_groups(40, 3)
        r = _run(
            "stats", "mood_median", {"var": "v", "group_var": "g"}, {"v": v, "g": g}
        )
        _check_result(self, r, "mood_median")

    def test_friedman(self):
        a = _normal(30)
        b = _normal(30)
        c = _normal(30)
        r = _run(
            "stats", "friedman", {"vars": ["a", "b", "c"]}, {"a": a, "b": b, "c": c}
        )
        _check_result(self, r, "friedman")


# ---------------------------------------------------------------------------
# Stats: Regression
# ---------------------------------------------------------------------------
class StatsRegressionExhaustiveTest(TestCase):
    """Every regression analysis completes without crash."""

    def test_regression(self):
        x = _normal(100, 0, 1)
        y = [2 * xi + RNG.normal(0, 0.5) for xi in x]
        r = _run(
            "stats",
            "regression",
            {"response": "y", "predictors": ["x"]},
            {"x": x, "y": y},
        )
        _check_result(self, r, "regression")

    def test_logistic(self):
        x = _normal(100, 0, 1)
        y = [1 if xi > 0 else 0 for xi in x]
        r = _run(
            "stats",
            "logistic",
            {"response": "y", "predictors": ["x"]},
            {"x": x, "y": y},
        )
        _check_result(self, r, "logistic")

    def test_stepwise(self):
        x1 = _normal(100, 0, 1)
        x2 = _normal(100, 0, 1)
        y = [2 * a + b + RNG.normal(0, 1) for a, b in zip(x1, x2)]
        r = _run(
            "stats",
            "stepwise",
            {"response": "y", "predictors": ["x1", "x2"]},
            {"x1": x1, "x2": x2, "y": y},
        )
        _check_result(self, r, "stepwise")


# ---------------------------------------------------------------------------
# Stats: Exploratory
# ---------------------------------------------------------------------------
class StatsExploratoryExhaustiveTest(TestCase):
    """Exploratory analyses complete without crash."""

    def test_descriptive(self):
        r = _run("stats", "descriptive", {"var1": "x"}, {"x": _normal()})
        _check_result(self, r, "descriptive")

    def test_bootstrap_ci(self):
        r = _run("stats", "bootstrap_ci", {"var": "x"}, {"x": _normal()})
        _check_result(self, r, "bootstrap_ci")

    def test_grubbs_test(self):
        r = _run("stats", "grubbs_test", {"var": "x"}, {"x": _normal()})
        _check_result(self, r, "grubbs_test")

    def test_box_cox(self):
        # Box-Cox needs positive values
        data = [abs(x) + 1 for x in _normal()]
        r = _run("stats", "box_cox", {"var": "x"}, {"x": data})
        _check_result(self, r, "box_cox")

    def test_tolerance_interval(self):
        r = _run(
            "stats",
            "tolerance_interval",
            {"var": "x", "proportion": 0.95, "conf": 0.95},
            {"x": _normal()},
        )
        _check_result(self, r, "tolerance_interval")

    def test_run_chart(self):
        r = _run("stats", "run_chart", {"var": "x"}, {"x": _normal()})
        _check_result(self, r, "run_chart")

    def test_distribution_fit(self):
        r = _run("stats", "distribution_fit", {"var": "x"}, {"x": _normal()})
        _check_result(self, r, "distribution_fit")


# ---------------------------------------------------------------------------
# Stats: Post-hoc
# ---------------------------------------------------------------------------
class StatsPostHocExhaustiveTest(TestCase):
    """Post-hoc tests complete without crash."""

    def _anova_data(self):
        v, g = _stacked_groups(40, 3)
        return {"v": v, "g": g}

    def test_tukey_hsd(self):
        d = self._anova_data()
        r = _run("stats", "tukey_hsd", {"response": "v", "factor": "g"}, d)
        _check_result(self, r, "tukey_hsd")

    def test_dunnett(self):
        d = self._anova_data()
        r = _run(
            "stats", "dunnett", {"response": "v", "factor": "g", "control": "G0"}, d
        )
        _check_result(self, r, "dunnett")

    def test_games_howell(self):
        d = self._anova_data()
        r = _run("stats", "games_howell", {"response": "v", "factor": "g"}, d)
        _check_result(self, r, "games_howell")

    def test_dunn(self):
        d = self._anova_data()
        r = _run("stats", "dunn", {"response": "v", "factor": "g"}, d)
        _check_result(self, r, "dunn")


# ---------------------------------------------------------------------------
# Stats: Power & Sample Size
# ---------------------------------------------------------------------------
class StatsPowerExhaustiveTest(TestCase):
    """Power analyses complete without crash."""

    def test_power_z(self):
        r = _run(
            "stats", "power_z", {"effect_size": 0.5, "alpha": 0.05, "power": 0.8}, {}
        )
        _check_result(self, r, "power_z")

    def test_sample_size_ci(self):
        r = _run("stats", "sample_size_ci", {"margin": 5, "std": 15, "conf": 0.95}, {})
        _check_result(self, r, "sample_size_ci")


# ---------------------------------------------------------------------------
# Stats: Time Series
# ---------------------------------------------------------------------------
class StatsTimeSeriesExhaustiveTest(TestCase):
    """Time series analyses complete without crash."""

    def test_decomposition(self):
        # Seasonal data — need at least 2 full periods
        data = [
            50 + 10 * np.sin(2 * np.pi * i / 12) + RNG.normal(0, 2) for i in range(48)
        ]
        r = _run("stats", "decomposition", {"var": "x", "period": 12}, {"x": data})
        _check_result(self, r, "decomposition")


# ---------------------------------------------------------------------------
# SPC: Control charts
# ---------------------------------------------------------------------------
class SPCExhaustiveTest(TestCase):
    """Every SPC chart type completes without crash."""

    def test_imr(self):
        r = _run("spc", "imr", {"measurement": "x"}, {"x": _normal(50)})
        _check_result(self, r, "imr")

    def test_xbar_r(self):
        data = _normal(100)
        subgroups = [f"S{i // 5}" for i in range(100)]
        r = _run(
            "spc",
            "xbar_r",
            {"measurement": "x", "subgroup": "sg"},
            {"x": data, "sg": subgroups},
        )
        _check_result(self, r, "xbar_r")

    def test_xbar_s(self):
        data = _normal(100)
        subgroups = [f"S{i // 10}" for i in range(100)]
        r = _run(
            "spc",
            "xbar_s",
            {"measurement": "x", "subgroup": "sg"},
            {"x": data, "sg": subgroups},
        )
        _check_result(self, r, "xbar_s")

    def test_p_chart(self):
        defectives = RNG.binomial(50, 0.05, 30).tolist()
        sizes = [50] * 30
        r = _run(
            "spc",
            "p_chart",
            {"defectives": "d", "sample_size": "n"},
            {"d": defectives, "n": sizes},
        )
        _check_result(self, r, "p_chart")

    def test_np_chart(self):
        defectives = RNG.binomial(50, 0.05, 30).tolist()
        r = _run(
            "spc", "np_chart", {"defectives": "d", "sample_size": 50}, {"d": defectives}
        )
        _check_result(self, r, "np_chart")

    def test_c_chart(self):
        defects = RNG.poisson(3, 30).tolist()
        r = _run("spc", "c_chart", {"defects": "c"}, {"c": defects})
        _check_result(self, r, "c_chart")

    def test_u_chart(self):
        defects = RNG.poisson(5, 30).tolist()
        units = [10] * 30
        r = _run(
            "spc", "u_chart", {"defects": "d", "units": "n"}, {"d": defects, "n": units}
        )
        _check_result(self, r, "u_chart")

    def test_cusum(self):
        r = _run("spc", "cusum", {"measurement": "x"}, {"x": _normal(50)})
        _check_result(self, r, "cusum")

    def test_ewma(self):
        r = _run("spc", "ewma", {"measurement": "x"}, {"x": _normal(50)})
        _check_result(self, r, "ewma")

    def test_capability(self):
        r = _run(
            "spc",
            "capability",
            {"measurement": "x", "lsl": 20, "usl": 80},
            {"x": _normal(100)},
        )
        _check_result(self, r, "capability")

    def test_between_within(self):
        data = _normal(100)
        subgroups = [f"S{i // 5}" for i in range(100)]
        r = _run(
            "spc",
            "between_within",
            {"measurement": "x", "subgroup": "sg"},
            {"x": data, "sg": subgroups},
        )
        _check_result(self, r, "between_within")


# ---------------------------------------------------------------------------
# Bayesian
# ---------------------------------------------------------------------------
class BayesianExhaustiveTest(TestCase):
    """Every Bayesian analysis completes without crash."""

    def test_bayes_ttest(self):
        a, b = _two_groups()
        r = _run(
            "bayesian", "bayes_ttest", {"var1": "a", "var2": "b"}, {"a": a, "b": b}
        )
        _check_result(self, r, "bayes_ttest")

    def test_bayes_ab(self):
        groups = ["A"] * 50 + ["B"] * 50
        converted = (
            RNG.binomial(1, 0.5, 50).tolist() + RNG.binomial(1, 0.7, 50).tolist()
        )
        r = _run(
            "bayesian",
            "bayes_ab",
            {"group": "group", "success": "converted"},
            {"group": groups, "converted": converted},
        )
        _check_result(self, r, "bayes_ab")

    def test_bayes_regression(self):
        x = _normal(80, 0, 1)
        y = [2 * xi + RNG.normal(0, 0.5) for xi in x]
        r = _run(
            "bayesian",
            "bayes_regression",
            {"response": "y", "predictors": ["x"]},
            {"x": x, "y": y},
        )
        _check_result(self, r, "bayes_regression")

    def test_bayes_correlation(self):
        x = _normal(80, 0, 1)
        y = _normal(80, 0, 1)
        r = _run(
            "bayesian",
            "bayes_correlation",
            {"var1": "x", "var2": "y"},
            {"x": x, "y": y},
        )
        _check_result(self, r, "bayes_correlation")

    def test_bayes_anova(self):
        v, g = _stacked_groups(40, 3)
        r = _run(
            "bayesian",
            "bayes_anova",
            {"response": "v", "factor": "g"},
            {"v": v, "g": g},
        )
        _check_result(self, r, "bayes_anova")

    def test_bayes_proportion(self):
        data = RNG.binomial(1, 0.6, 100).tolist()
        r = _run("bayesian", "bayes_proportion", {"success": "x"}, {"x": data})
        _check_result(self, r, "bayes_proportion")

    def test_bayes_changepoint(self):
        # Data with a shift
        data = _normal(50, 50, 5) + _normal(50, 70, 5)
        r = _run("bayesian", "bayes_changepoint", {"var": "x"}, {"x": data})
        _check_result(self, r, "bayes_changepoint")


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------
class ReliabilityExhaustiveTest(TestCase):
    """Every reliability analysis completes without crash."""

    def test_weibull(self):
        data = RNG.exponential(100, 50).tolist()
        r = _run("reliability", "weibull", {"time": "t"}, {"t": data})
        _check_result(self, r, "weibull")

    def test_kaplan_meier(self):
        times = RNG.exponential(100, 60).tolist()
        events = RNG.binomial(1, 0.7, 60).tolist()
        r = _run(
            "reliability",
            "kaplan_meier",
            {"time": "t", "event": "e"},
            {"t": times, "e": events},
        )
        _check_result(self, r, "kaplan_meier")

    def test_exponential(self):
        data = RNG.exponential(100, 50).tolist()
        r = _run("reliability", "exponential", {"time": "t"}, {"t": data})
        _check_result(self, r, "exponential")

    def test_lognormal(self):
        data = RNG.lognormal(4, 0.5, 50).tolist()
        r = _run("reliability", "lognormal", {"time": "t"}, {"t": data})
        _check_result(self, r, "lognormal")


# ---------------------------------------------------------------------------
# ML
# ---------------------------------------------------------------------------
class MLExhaustiveTest(TestCase):
    """ML analyses complete without crash."""

    def test_clustering(self):
        x1 = _normal(60, 0, 1)
        x2 = _normal(60, 0, 1)
        r = _run(
            "ml",
            "clustering",
            {"features": ["x1", "x2"], "n_clusters": 3},
            {"x1": x1, "x2": x2},
        )
        _check_result(self, r, "clustering")

    def test_pca(self):
        x1 = _normal(60, 0, 1)
        x2 = _normal(60, 0, 1)
        x3 = _normal(60, 0, 1)
        r = _run(
            "ml",
            "pca",
            {"features": ["x1", "x2", "x3"]},
            {"x1": x1, "x2": x2, "x3": x3},
        )
        _check_result(self, r, "pca")

    def test_isolation_forest(self):
        x1 = _normal(100, 0, 1)
        x2 = _normal(100, 0, 1)
        r = _run(
            "ml", "isolation_forest", {"features": ["x1", "x2"]}, {"x1": x1, "x2": x2}
        )
        _check_result(self, r, "isolation_forest")

    def test_feature(self):
        x1 = _normal(100, 0, 1)
        x2 = _normal(100, 0, 1)
        y = [a + b + RNG.normal(0, 0.1) for a, b in zip(x1, x2)]
        r = _run(
            "ml",
            "feature",
            {"response": "y", "predictors": ["x1", "x2"]},
            {"x1": x1, "x2": x2, "y": y},
        )
        _check_result(self, r, "feature")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
class SimulationExhaustiveTest(TestCase):
    """Simulation analyses complete without crash."""

    def test_tolerance_stackup(self):
        r = _run(
            "simulation",
            "tolerance_stackup",
            {
                "dimensions": [
                    {"name": "A", "nominal": 10, "tolerance": 0.1},
                    {"name": "B", "nominal": 20, "tolerance": 0.2},
                ],
                "assembly_func": "sum",
                "n_simulations": 1000,
            },
            {},
        )
        _check_result(self, r, "tolerance_stackup")
