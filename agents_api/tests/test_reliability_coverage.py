"""Coverage tests for reliability analyses — CAL-001 §6 / TST-001 §10.

Standard: CAL-001 §6 (Statistical Correctness Verification)
Compliance: SOC 2 CC4.1
<!-- test: agents_api.tests.test_reliability_coverage -->
"""

import numpy as np
import pandas as pd
from django.test import TestCase

# Shared test data
RNG = np.random.RandomState(42)
TIMES_40 = list(RNG.exponential(100, 40))
EVENTS_40 = np.random.RandomState(43).choice([0, 1], 40, p=[0.3, 0.7]).tolist()
CENSOR_40 = np.random.RandomState(44).choice([0, 1], 40, p=[0.2, 0.8]).tolist()
STRESS_40 = [50.0] * 14 + [75.0] * 13 + [100.0] * 13
CAUSES_40 = np.random.RandomState(45).choice([1, 2, 3], 40).tolist()


def _run(analysis_id, config, data_dict):
    """Run reliability analysis — no exception masking (TST-001 §11.6)."""
    from agents_api.analysis.reliability import run_reliability_analysis

    df = pd.DataFrame(data_dict)
    return run_reliability_analysis(df, analysis_id, config)


def _check_schema(tc, r):
    """Verify output schema per CAL-001 §6 / TST-001 §10.6."""
    aid = tc._testMethodName
    tc.assertIsInstance(r, dict, f"{aid} did not return a dict")
    tc.assertIn("plots", r, f"{aid} missing 'plots' key")
    tc.assertIsInstance(r["plots"], list, f"{aid} plots is not a list")
    tc.assertIn("summary", r, f"{aid} missing 'summary' key")


class ReliabilityDistributionTest(TestCase):
    """Distribution-based reliability analyses."""

    def test_weibull(self):
        r = _run(
            "weibull",
            {"time": "t"},
            {"t": TIMES_40},
        )
        _check_schema(self, r)

    def test_lognormal(self):
        r = _run(
            "lognormal",
            {"time": "t"},
            {"t": TIMES_40},
        )
        _check_schema(self, r)

    def test_exponential(self):
        r = _run(
            "exponential",
            {"time": "t"},
            {"t": TIMES_40},
        )
        _check_schema(self, r)

    def test_distribution_id(self):
        r = _run(
            "distribution_id",
            {"time": "t"},
            {"t": TIMES_40},
        )
        _check_schema(self, r)


class ReliabilitySurvivalTest(TestCase):
    """Survival and life data analyses."""

    def test_kaplan_meier(self):
        r = _run(
            "kaplan_meier",
            {"time": "t", "event": "e"},
            {"t": TIMES_40, "e": EVENTS_40},
        )
        _check_schema(self, r)

    def test_reliability_test_plan(self):
        r = _run(
            "reliability_test_plan",
            {"target_reliability": 0.90, "confidence": 0.95, "test_duration": 1000},
            {"t": [1]},  # dummy data
        )
        _check_schema(self, r)

    def test_accelerated_life(self):
        times_alt = list(np.random.RandomState(42).exponential(100, 39))
        events_alt = [1] * 39
        r = _run(
            "accelerated_life",
            {"time": "t", "event": "e", "stress": "stress", "model": "arrhenius"},
            {"t": times_alt, "e": events_alt, "stress": STRESS_40[:39]},
        )
        _check_schema(self, r)

    def test_repairable_systems(self):
        # Cumulative failure times for a repairable system
        cum_times = sorted(np.random.RandomState(42).uniform(0, 1000, 20).tolist())
        r = _run(
            "repairable_systems",
            {"time": "t"},
            {"t": cum_times},
        )
        _check_schema(self, r)

    def test_warranty(self):
        return_times = list(np.random.RandomState(42).exponential(60, 40))
        r = _run(
            "warranty",
            {"time": "t", "warranty_period": 365},
            {"t": return_times},
        )
        _check_schema(self, r)

    def test_competing_risks(self):
        r = _run(
            "competing_risks",
            {"time": "t", "event": "cause"},
            {"t": TIMES_40, "cause": CAUSES_40},
        )
        _check_schema(self, r)
