"""Coverage tests for Bayesian analysis engine — exercise all analysis_ids.

Golden files cover: bayes_ttest, bayes_ab, bayes_regression (5 files).
This file covers the remaining ~19 analysis IDs.

Standard: CAL-001 §6 (Statistical Correctness Verification)
Compliance: SOC 2 CC4.1
<!-- test: agents_api.tests.test_bayesian_coverage -->
"""

import numpy as np
import pandas as pd
from django.test import TestCase


def _run(analysis_id, config, data_dict):
    """Run Bayesian analysis — no exception masking (TST-001 §11.6)."""
    from agents_api.dsw.bayesian import run_bayesian_analysis

    df = pd.DataFrame(data_dict)
    return run_bayesian_analysis(df, analysis_id, config)


def _check_schema(tc, r):
    """Verify output schema per CAL-001 §6 / TST-001 §10.6."""
    aid = tc._testMethodName
    tc.assertIsInstance(r, dict, f"{aid} did not return a dict")
    tc.assertIn("plots", r, f"{aid} missing 'plots' key")
    tc.assertIsInstance(r["plots"], list, f"{aid} plots is not a list")
    tc.assertIn("summary", r, f"{aid} missing 'summary' key")
    # Bayesian analyses should have statistics with posterior info
    stats = r.get("statistics", {})
    if isinstance(stats, dict):
        if "bf10" in stats and stats["bf10"] is not None:
            tc.assertGreater(stats["bf10"], 0, f"{aid} bf10 must be positive")


# Shared test data
NORMAL_40 = list(np.random.RandomState(42).normal(100, 15, 40))
NORMAL_40B = list(np.random.RandomState(99).normal(105, 15, 40))
GROUPS = ["A"] * 20 + ["B"] * 20
GROUPS_3 = ["A"] * 14 + ["B"] * 13 + ["C"] * 13
BINARY_40 = [0] * 20 + [1] * 20
COUNTS_40 = list(np.random.RandomState(42).poisson(5, 40))
TIME_40 = list(np.random.RandomState(42).exponential(10, 40))
EVENTS_40 = np.random.RandomState(43).choice([0, 1], 40, p=[0.3, 0.7]).tolist()


class BayesianInferenceCoverageTest(TestCase):
    """Bayesian inference analyses beyond golden file coverage."""

    def test_bayes_correlation(self):
        r = _run(
            "bayes_correlation",
            {"var1": "x", "var2": "y"},
            {"x": NORMAL_40, "y": NORMAL_40B},
        )
        _check_schema(self, r)

    def test_bayes_anova(self):
        r = _run(
            "bayes_anova",
            {"response": "y", "factor": "g"},
            {"y": NORMAL_40, "g": GROUPS},
        )
        _check_schema(self, r)

    def test_bayes_changepoint(self):
        seg1 = list(np.random.RandomState(42).normal(10, 1, 20))
        seg2 = list(np.random.RandomState(43).normal(15, 1, 20))
        r = _run(
            "bayes_changepoint",
            {"var": "y"},
            {"y": seg1 + seg2},
        )
        _check_schema(self, r)

    def test_bayes_proportion(self):
        r = _run(
            "bayes_proportion",
            {"success": "x", "prior": "uniform"},
            {"x": BINARY_40},
        )
        _check_schema(self, r)

    def test_bayes_capability_prediction(self):
        r = _run(
            "bayes_capability_prediction",
            {"var1": "x", "lsl": 70, "usl": 130},
            {"x": NORMAL_40},
        )
        _check_schema(self, r)

    def test_bayes_equivalence(self):
        r = _run(
            "bayes_equivalence",
            {"var1": "x", "var2": "y", "margin": 10},
            {"x": NORMAL_40, "y": NORMAL_40B},
        )
        _check_schema(self, r)

    def test_bayes_chi2(self):
        cat1 = np.random.RandomState(42).choice(["A", "B", "C"], 40).tolist()
        cat2 = np.random.RandomState(43).choice(["X", "Y"], 40).tolist()
        r = _run(
            "bayes_chi2",
            {"var1": "c1", "var2": "c2"},
            {"c1": cat1, "c2": cat2},
        )
        _check_schema(self, r)

    def test_bayes_poisson(self):
        r = _run(
            "bayes_poisson",
            {"var1": "x"},
            {"x": COUNTS_40},
        )
        _check_schema(self, r)

    def test_bayes_logistic(self):
        r = _run(
            "bayes_logistic",
            {"response": "y", "predictors": ["x"]},
            {"y": BINARY_40, "x": NORMAL_40},
        )
        _check_schema(self, r)

    def test_bayes_survival(self):
        r = _run(
            "bayes_survival",
            {"var1": "t", "var2": "e"},
            {"t": TIME_40, "e": EVENTS_40},
        )
        _check_schema(self, r)

    def test_bayes_meta(self):
        effects = [0.3, 0.5, 0.4, 0.6, 0.35]
        ses = [0.1, 0.15, 0.12, 0.08, 0.11]
        r = _run(
            "bayes_meta",
            {"var1": "effect", "var2": "se"},
            {"effect": effects, "se": ses},
        )
        _check_schema(self, r)


class BayesianGoldenFileCoverageTest(TestCase):
    """Tests for Bayesian analysis IDs covered by golden files — ensures they run."""

    def test_bayes_ttest(self):
        r = _run(
            "bayes_ttest",
            {"var1": "x", "var2": "y"},
            {"x": NORMAL_40, "y": NORMAL_40B},
        )
        _check_schema(self, r)

    def test_bayes_ttest_factor_mode(self):
        """Factor-split mode: one measurement column split by a categorical grouping column."""
        r = _run(
            "bayes_ttest",
            {"mode": "factor", "response": "y", "factor": "g"},
            {"y": NORMAL_40 + NORMAL_40B, "g": ["A"] * 40 + ["B"] * 40},
        )
        _check_schema(self, r)
        self.assertIn("cohens_d", r["statistics"])
        self.assertIn("bf10", r["statistics"])

    def test_bayes_ab(self):
        groups = ["A"] * 20 + ["B"] * 20
        successes = [
            1,
            0,
            1,
            1,
            0,
            1,
            0,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            1,
            0,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            0,
        ]
        r = _run(
            "bayes_ab",
            {"group": "g", "success": "s"},
            {"g": groups, "s": successes},
        )
        _check_schema(self, r)

    def test_bayes_regression(self):
        x1 = list(np.random.RandomState(42).normal(0, 1, 40))
        y = [2 * xi + np.random.RandomState(43).normal(0, 0.5) for xi in x1]
        r = _run(
            "bayes_regression",
            {"target": "y", "features": ["x1"]},
            {"y": y, "x1": x1},
        )
        _check_schema(self, r)


class BayesianReliabilityCoverageTest(TestCase):
    """Bayesian reliability and demo analyses."""

    def test_bayes_demo(self):
        r = _run(
            "bayes_demo",
            {"n_tested": 30, "n_failures": 2, "target_reliability": 0.95},
            {"x": NORMAL_40},
        )
        _check_schema(self, r)
        # Verify posterior updates with actual config (not defaults)
        # With 30 tested, 2 failures, prior Beta(1,1): posterior Beta(29, 3)
        self.assertAlmostEqual(r["statistics"]["posterior_alpha"], 29.0, places=0)
        self.assertAlmostEqual(r["statistics"]["posterior_beta"], 3.0, places=0)

    def test_bayes_spares(self):
        r = _run(
            "bayes_spares",
            {"failure_rate": 0.01, "mission_time": 1000, "n_spares": 3},
            {"x": [1]},
        )
        _check_schema(self, r)

    def test_bayes_system(self):
        r = _run(
            "bayes_system",
            {
                "components": [
                    {"name": "A", "reliability": 0.95},
                    {"name": "B", "reliability": 0.90},
                ],
                "structure": "series",
            },
            {"x": [1]},
        )
        _check_schema(self, r)

    def test_bayes_warranty(self):
        r = _run(
            "bayes_warranty",
            {"failure_times": "t", "warranty_period": 12},
            {"t": TIME_40},
        )
        _check_schema(self, r)

    def test_bayes_repairable(self):
        r = _run(
            "bayes_repairable",
            {"failure_times": "t"},
            {"t": sorted(TIME_40[:20])},
        )
        _check_schema(self, r)

    def test_bayes_rul(self):
        # Remaining useful life prediction
        deg = [100 - 0.5 * i + np.random.RandomState(42).normal(0, 1) for i in range(40)]
        r = _run(
            "bayes_rul",
            {"measurement": "y", "threshold": 50},
            {"y": deg},
        )
        _check_schema(self, r)

    def test_bayes_alt(self):
        # Accelerated life testing
        times = list(np.random.RandomState(42).exponential(100, 30))
        temps = [50.0] * 10 + [75.0] * 10 + [100.0] * 10
        events = [1] * 30
        r = _run(
            "bayes_alt",
            {"time": "t", "event": "e", "stress": "temp"},
            {"t": times, "e": events, "temp": temps},
        )
        _check_schema(self, r)

    def test_bayes_comprisk(self):
        times = list(np.random.RandomState(42).exponential(10, 40))
        causes = np.random.RandomState(43).choice([1, 2, 3], 40).tolist()
        r = _run(
            "bayes_comprisk",
            {"time": "t", "cause": "c"},
            {"t": times, "c": causes},
        )
        _check_schema(self, r)

    def test_bayes_ewma(self):
        r = _run(
            "bayes_ewma",
            {"measurement": "x", "lambda_param": 0.2, "L": 3},
            {"x": NORMAL_40},
        )
        _check_schema(self, r)
