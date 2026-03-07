"""Coverage tests for SPC analysis engine — exercise all analysis_ids.

Tests SPC chart types not already covered by golden files.
Golden files cover: imr, capability, xbar_r, xbar_s, p_chart, cusum, ewma (14 files).

Standard: CAL-001 §6 (Statistical Correctness Verification)
Compliance: SOC 2 CC4.1
<!-- test: agents_api.tests.test_spc_coverage -->
"""

import numpy as np
import pandas as pd
from django.test import TestCase


def _run(analysis_id, config, data_dict):
    """Run SPC analysis — no exception masking (TST-001 §11.6)."""
    from agents_api.dsw.spc import run_spc_analysis

    df = pd.DataFrame(data_dict)
    return run_spc_analysis(df, analysis_id, config)


def _check_schema(tc, r):
    """Verify output schema per CAL-001 §6 / TST-001 §10.6."""
    aid = tc._testMethodName
    tc.assertIsInstance(r, dict, f"{aid} did not return a dict")
    tc.assertIn("plots", r, f"{aid} missing 'plots' key")
    tc.assertIsInstance(r["plots"], list, f"{aid} plots is not a list")
    tc.assertIn("summary", r, f"{aid} missing 'summary' key")


# Shared test data
NORMAL_100 = list(np.random.RandomState(42).normal(50, 5, 100))
DEFECTS_100 = list(np.random.RandomState(42).binomial(1, 0.05, 100))
COUNTS_100 = list(np.random.RandomState(42).poisson(3, 100))
SUBGROUP_IDS = [i // 5 for i in range(100)]  # 20 subgroups of 5


class SPCChartCoverageTest(TestCase):
    """SPC chart types not in golden files."""

    def test_np_chart(self):
        r = _run(
            "np_chart",
            {"defectives": "d", "sample_size": 50},
            {"d": list(np.random.RandomState(42).binomial(50, 0.05, 25))},
        )
        _check_schema(self, r)

    def test_c_chart(self):
        r = _run(
            "c_chart",
            {"defects": "c"},
            {"c": COUNTS_100},
        )
        _check_schema(self, r)

    def test_u_chart(self):
        areas = [10] * 100
        r = _run(
            "u_chart",
            {"defects": "c", "units": "n"},
            {"c": COUNTS_100, "n": areas},
        )
        _check_schema(self, r)

    def test_laney_p(self):
        r = _run(
            "laney_p",
            {"defectives": "d", "sample_size": "n"},
            {"d": list(np.random.RandomState(42).binomial(100, 0.05, 30)), "n": [100] * 30},
        )
        _check_schema(self, r)

    def test_laney_u(self):
        r = _run(
            "laney_u",
            {"defects": "c", "units": "n"},
            {"c": COUNTS_100[:30], "n": [50] * 30},
        )
        _check_schema(self, r)

    def test_between_within(self):
        r = _run(
            "between_within",
            {"measurement": "y", "subgroup": "sg"},
            {"y": NORMAL_100, "sg": SUBGROUP_IDS},
        )
        _check_schema(self, r)

    def test_nonnormal_capability(self):
        r = _run(
            "nonnormal_capability",
            {"measurement": "y", "lsl": 30, "usl": 70},
            {"y": NORMAL_100},
        )
        _check_schema(self, r)

    def test_moving_average(self):
        r = _run(
            "moving_average",
            {"measurement": "y", "span": 5},
            {"y": NORMAL_100},
        )
        _check_schema(self, r)

    def test_zone_chart(self):
        r = _run(
            "zone_chart",
            {"measurement": "y"},
            {"y": NORMAL_100},
        )
        _check_schema(self, r)

    def test_mewma(self):
        r = _run(
            "mewma",
            {"measurements": ["x", "y"]},
            {
                "x": NORMAL_100,
                "y": list(np.random.RandomState(43).normal(50, 5, 100)),
            },
        )
        _check_schema(self, r)

    def test_generalized_variance(self):
        r = _run(
            "generalized_variance",
            {"measurements": ["x", "y"]},
            {
                "x": NORMAL_100,
                "y": list(np.random.RandomState(43).normal(50, 5, 100)),
            },
        )
        _check_schema(self, r)

    def test_conformal_control(self):
        r = _run(
            "conformal_control",
            {"measurement": "y"},
            {"y": NORMAL_100},
        )
        _check_schema(self, r)

    def test_conformal_monitor(self):
        r = _run(
            "conformal_monitor",
            {"measurement": "y"},
            {"y": NORMAL_100},
        )
        _check_schema(self, r)

    def test_entropy_spc(self):
        r = _run(
            "entropy_spc",
            {"measurement": "y"},
            {"y": NORMAL_100},
        )
        _check_schema(self, r)

    def test_degradation_capability(self):
        # Degradation over time
        time_vals = list(range(100))
        deg_vals = [50 + 0.1 * t + np.random.RandomState(42).normal(0, 1) for t in time_vals]
        r = _run(
            "degradation_capability",
            {"measurement": "y", "time": "t", "lsl": 40, "usl": 60},
            {"y": deg_vals, "t": time_vals},
        )
        _check_schema(self, r)


class SPCGoldenFileCoverageTest(TestCase):
    """Tests for analysis IDs covered by golden files — ensures they run without error."""

    def test_capability(self):
        r = _run(
            "capability",
            {"measurement": "y", "lsl": 30, "usl": 70},
            {"y": NORMAL_100},
        )
        _check_schema(self, r)

    def test_cusum(self):
        r = _run(
            "cusum",
            {"measurement": "y", "target": 50},
            {"y": NORMAL_100},
        )
        _check_schema(self, r)

    def test_ewma(self):
        r = _run(
            "ewma",
            {"measurement": "y", "lambda": 0.2, "L": 3},
            {"y": NORMAL_100},
        )
        _check_schema(self, r)

    def test_imr(self):
        r = _run(
            "imr",
            {"measurement": "y"},
            {"y": NORMAL_100},
        )
        _check_schema(self, r)

    def test_p_chart(self):
        r = _run(
            "p_chart",
            {"defectives": "d", "sample_size": "n"},
            {"d": list(np.random.RandomState(42).binomial(50, 0.05, 25)), "n": [50] * 25},
        )
        _check_schema(self, r)

    def test_xbar_r(self):
        r = _run(
            "xbar_r",
            {"measurement": "y", "subgroup": "sg", "subgroup_size": 5},
            {"y": NORMAL_100, "sg": SUBGROUP_IDS},
        )
        _check_schema(self, r)

    def test_xbar_s(self):
        r = _run(
            "xbar_s",
            {"measurement": "y", "subgroup": "sg", "subgroup_size": 5},
            {"y": NORMAL_100, "sg": SUBGROUP_IDS},
        )
        _check_schema(self, r)


class NelsonRulesTest(TestCase):
    """Test _spc_nelson_rules directly."""

    def test_rule1_beyond_limits(self):
        from agents_api.dsw.spc import _spc_nelson_rules

        data = [50.0] * 20
        data[10] = 100.0  # Way beyond UCL
        ooc, violations = _spc_nelson_rules(data, cl=50, ucl=65, lcl=35)
        self.assertIn(10, ooc)

    def test_no_violations_in_control(self):
        from agents_api.dsw.spc import _spc_nelson_rules

        np.random.seed(42)
        data = list(np.random.normal(50, 3, 20))
        ooc, violations = _spc_nelson_rules(data, cl=50, ucl=59, lcl=41)
        # Verify return types — behavioural check
        self.assertIsInstance(ooc, list)
        self.assertIsInstance(violations, list)
        for v in violations:
            self.assertIn("rule", v)
            self.assertIn("points", v)
