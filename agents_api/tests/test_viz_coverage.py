"""Coverage tests for visualization analyses — CAL-001 §6 / TST-001 §10.

Standard: CAL-001 §6 (Statistical Correctness Verification)
Compliance: SOC 2 CC4.1
<!-- test: agents_api.tests.test_viz_coverage -->
"""

import numpy as np
import pandas as pd
from django.test import TestCase

# Shared test data
RNG = np.random.RandomState(42)
NORMAL_60 = list(RNG.normal(100, 15, 60))
NORMAL_60B = list(np.random.RandomState(43).normal(105, 15, 60))
NORMAL_60C = list(np.random.RandomState(44).normal(95, 10, 60))
GROUPS_60 = ["A"] * 30 + ["B"] * 30
CAT1_60 = np.random.RandomState(45).choice(["X", "Y", "Z"], 60).tolist()
CAT2_60 = np.random.RandomState(46).choice(["P", "Q"], 60).tolist()
TIME_60 = list(range(60))


def _run(analysis_id, config, data_dict):
    """Run visualization — no exception masking (TST-001 §11.6)."""
    from agents_api.analysis.viz import run_visualization

    df = pd.DataFrame(data_dict)
    return run_visualization(df, analysis_id, config)


def _check_schema(tc, r):
    """Verify output schema per CAL-001 §6 / TST-001 §10.6."""
    aid = tc._testMethodName
    tc.assertIsInstance(r, dict, f"{aid} did not return a dict")
    tc.assertIn("plots", r, f"{aid} missing 'plots' key")
    tc.assertIsInstance(r["plots"], list, f"{aid} plots is not a list")
    tc.assertIn("summary", r, f"{aid} missing 'summary' key")


class BasicVisualizationTest(TestCase):
    """Basic plot types."""

    def test_histogram(self):
        r = _run(
            "histogram",
            {"var": "x"},
            {"x": NORMAL_60},
        )
        _check_schema(self, r)

    def test_boxplot(self):
        r = _run(
            "boxplot",
            {"var": "x", "by": "g"},
            {"x": NORMAL_60, "g": GROUPS_60},
        )
        _check_schema(self, r)

    def test_scatter(self):
        r = _run(
            "scatter",
            {"x": "x", "y": "y"},
            {"x": NORMAL_60, "y": NORMAL_60B},
        )
        _check_schema(self, r)

    def test_heatmap(self):
        r = _run(
            "heatmap",
            {"variables": ["x", "y", "z"]},
            {"x": NORMAL_60, "y": NORMAL_60B, "z": NORMAL_60C},
        )
        _check_schema(self, r)

    def test_pareto(self):
        categories = (
            np.random.RandomState(42).choice(["Defect_A", "Defect_B", "Defect_C", "Defect_D", "Defect_E"], 100).tolist()
        )
        r = _run(
            "pareto",
            {"var": "defect"},
            {"defect": categories},
        )
        _check_schema(self, r)

    def test_matrix(self):
        r = _run(
            "matrix",
            {"variables": ["x", "y", "z"]},
            {"x": NORMAL_60, "y": NORMAL_60B, "z": NORMAL_60C},
        )
        _check_schema(self, r)

    def test_timeseries(self):
        r = _run(
            "timeseries",
            {"y": ["x"], "x": "t"},
            {"x": NORMAL_60, "t": TIME_60},
        )
        _check_schema(self, r)

    def test_probability(self):
        r = _run(
            "probability",
            {"var": "x"},
            {"x": NORMAL_60},
        )
        _check_schema(self, r)


class AdvancedVisualizationTest(TestCase):
    """Advanced plot types."""

    def test_individual_value_plot(self):
        r = _run(
            "individual_value_plot",
            {"var": "x", "group": "g"},
            {"x": NORMAL_60, "g": GROUPS_60},
        )
        _check_schema(self, r)

    def test_interval_plot(self):
        r = _run(
            "interval_plot",
            {"var": "x", "group": "g"},
            {"x": NORMAL_60, "g": GROUPS_60},
        )
        _check_schema(self, r)

    def test_dotplot(self):
        r = _run(
            "dotplot",
            {"var": "x"},
            {"x": NORMAL_60},
        )
        _check_schema(self, r)

    def test_bubble(self):
        sizes = list(np.random.RandomState(47).uniform(5, 50, 60))
        r = _run(
            "bubble",
            {"x": "x", "y": "y", "size": "s"},
            {"x": NORMAL_60, "y": NORMAL_60B, "s": sizes},
        )
        _check_schema(self, r)

    def test_parallel_coordinates(self):
        r = _run(
            "parallel_coordinates",
            {"dimensions": ["x", "y", "z"]},
            {"x": NORMAL_60, "y": NORMAL_60B, "z": NORMAL_60C},
        )
        _check_schema(self, r)

    def test_contour(self):
        r = _run(
            "contour",
            {"x": "x", "y": "y", "z": "z"},
            {"x": NORMAL_60, "y": NORMAL_60B, "z": NORMAL_60C},
        )
        _check_schema(self, r)

    def test_surface_3d(self):
        r = _run(
            "surface_3d",
            {"x": "x", "y": "y", "z": "z"},
            {"x": NORMAL_60, "y": NORMAL_60B, "z": NORMAL_60C},
        )
        _check_schema(self, r)

    def test_contour_overlay(self):
        r = _run(
            "contour_overlay",
            {"x": "x", "y": "y", "z_cols": ["z1", "z2"]},
            {"x": NORMAL_60, "y": NORMAL_60B, "z1": NORMAL_60C, "z2": NORMAL_60},
        )
        _check_schema(self, r)

    def test_mosaic(self):
        r = _run(
            "mosaic",
            {"row_var": "c1", "col_var": "c2"},
            {"c1": CAT1_60, "c2": CAT2_60},
        )
        _check_schema(self, r)


class BayesSPCVisualizationTest(TestCase):
    """Bayesian SPC visualization suite."""

    def test_bayes_spc_capability(self):
        r = _run(
            "bayes_spc_capability",
            {"measurement": "x", "lsl": 60, "usl": 140},
            {"x": NORMAL_60},
        )
        _check_schema(self, r)

    def test_bayes_spc_changepoint(self):
        seg1 = list(np.random.RandomState(42).normal(100, 10, 30))
        seg2 = list(np.random.RandomState(43).normal(120, 10, 30))
        r = _run(
            "bayes_spc_changepoint",
            {"measurement": "x"},
            {"x": seg1 + seg2},
        )
        _check_schema(self, r)

    def test_bayes_spc_control(self):
        r = _run(
            "bayes_spc_control",
            {"measurement": "x"},
            {"x": NORMAL_60},
        )
        _check_schema(self, r)

    def test_bayes_spc_acceptance(self):
        r = _run(
            "bayes_spc_acceptance",
            {"measurement": "x", "lsl": 60, "usl": 140},
            {"x": NORMAL_60},
        )
        _check_schema(self, r)
