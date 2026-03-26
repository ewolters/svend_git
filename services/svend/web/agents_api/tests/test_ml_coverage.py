"""Coverage tests for ML analysis engine — exercise all analysis_ids.

Golden files cover: classification, regression_ml, clustering, pca (4 files).
This file covers the remaining ~17 analysis IDs.

Standard: CAL-001 §6 (Statistical Correctness Verification)
Compliance: SOC 2 CC4.1
<!-- test: agents_api.tests.test_ml_coverage -->
"""

import importlib
import unittest

import numpy as np
import pandas as pd
from django.test import TestCase

_has_xgboost = importlib.util.find_spec("xgboost") is not None
_has_lightgbm = importlib.util.find_spec("lightgbm") is not None
_has_optuna = importlib.util.find_spec("optuna") is not None


def _run(analysis_id, config, data_dict):
    """Run ML analysis — no exception masking (TST-001 §11.6)."""
    from agents_api.analysis.ml import run_ml_analysis

    df = pd.DataFrame(data_dict)
    return run_ml_analysis(df, analysis_id, config, user=None)


def _check_schema(tc, r):
    """Verify output schema per CAL-001 §6 / TST-001 §10.6."""
    aid = tc._testMethodName
    tc.assertIsInstance(r, dict, f"{aid} did not return a dict")
    tc.assertIn("plots", r, f"{aid} missing 'plots' key")
    tc.assertIsInstance(r["plots"], list, f"{aid} plots is not a list")
    tc.assertIn("summary", r, f"{aid} missing 'summary' key")


# Shared test data
N = 60
X1 = list(np.random.RandomState(42).normal(0, 1, N))
X2 = list(np.random.RandomState(43).normal(0, 1, N))
X3 = list(np.random.RandomState(44).normal(0, 1, N))
Y_REG = [2 * x1 + 0.5 * x2 + np.random.RandomState(45).normal(0, 0.5) for x1, x2 in zip(X1, X2)]
Y_CLS = [1 if y > 0 else 0 for y in Y_REG]
Y_MULTI = np.random.RandomState(46).choice(["A", "B", "C"], N).tolist()


class MLSupervised(TestCase):
    """Supervised ML analyses beyond golden files."""

    def test_model_compare(self):
        r = _run(
            "model_compare",
            {"target": "y", "features": ["x1", "x2"]},
            {"y": Y_REG, "x1": X1, "x2": X2},
        )
        _check_schema(self, r)

    @unittest.skipUnless(_has_xgboost, "xgboost not installed")
    def test_xgboost(self):
        r = _run(
            "xgboost",
            {"target": "y", "features": ["x1", "x2"], "task": "regression"},
            {"y": Y_REG, "x1": X1, "x2": X2},
        )
        _check_schema(self, r)

    @unittest.skipUnless(_has_lightgbm, "lightgbm not installed")
    def test_lightgbm(self):
        r = _run(
            "lightgbm",
            {"target": "y", "features": ["x1", "x2"], "task": "regression"},
            {"y": Y_REG, "x1": X1, "x2": X2},
        )
        _check_schema(self, r)

    def test_shap_explain(self):
        r = _run(
            "shap_explain",
            {"target": "y", "features": ["x1", "x2"]},
            {"y": Y_REG, "x1": X1, "x2": X2},
        )
        _check_schema(self, r)

    @unittest.skipUnless(_has_optuna, "optuna not installed")
    def test_hyperparameter_tune(self):
        r = _run(
            "hyperparameter_tune",
            {"target": "y", "features": ["x1", "x2"], "algorithm": "rf"},
            {"y": Y_REG, "x1": X1, "x2": X2},
        )
        _check_schema(self, r)

    def test_bayesian_regression(self):
        r = _run(
            "bayesian_regression",
            {"target": "y", "features": ["x1", "x2"]},
            {"y": Y_REG, "x1": X1, "x2": X2},
        )
        _check_schema(self, r)

    def test_gam(self):
        r = _run(
            "gam",
            {"target": "y", "features": ["x1", "x2"]},
            {"y": Y_REG, "x1": X1, "x2": X2},
        )
        _check_schema(self, r)

    def test_gaussian_process(self):
        r = _run(
            "gaussian_process",
            {"target": "y", "features": ["x1"]},
            {"y": Y_REG[:30], "x1": X1[:30]},  # Smaller for GP
        )
        _check_schema(self, r)

    def test_pls(self):
        r = _run(
            "pls",
            {"target": "y", "features": ["x1", "x2", "x3"]},
            {"y": Y_REG, "x1": X1, "x2": X2, "x3": X3},
        )
        _check_schema(self, r)

    def test_sem(self):
        r = _run(
            "sem",
            {"target": "y", "features": ["x1", "x2"]},
            {"y": Y_REG, "x1": X1, "x2": X2},
        )
        _check_schema(self, r)

    def test_item_analysis(self):
        items = {f"q{i}": np.random.RandomState(i).choice([0, 1], N).tolist() for i in range(5)}
        r = _run(
            "item_analysis",
            {"features": [f"q{i}" for i in range(5)]},
            items,
        )
        _check_schema(self, r)


class MLGoldenFileCoverageTest(TestCase):
    """Tests for ML analysis IDs covered by golden files — ensures they run."""

    def test_classification(self):
        r = _run(
            "classification",
            {"target": "y", "features": ["x1", "x2"]},
            {"y": Y_CLS, "x1": X1, "x2": X2},
        )
        _check_schema(self, r)

    def test_regression_ml(self):
        r = _run(
            "regression_ml",
            {"target": "y", "features": ["x1", "x2"]},
            {"y": Y_REG, "x1": X1, "x2": X2},
        )
        _check_schema(self, r)

    def test_clustering(self):
        r = _run(
            "clustering",
            {"features": ["x1", "x2"]},
            {"x1": X1, "x2": X2},
        )
        _check_schema(self, r)

    def test_pca(self):
        r = _run(
            "pca",
            {"features": ["x1", "x2", "x3"]},
            {"x1": X1, "x2": X2, "x3": X3},
        )
        _check_schema(self, r)


class MLUnsupervised(TestCase):
    """Unsupervised ML analyses."""

    def test_feature(self):
        r = _run(
            "feature",
            {"target": "y", "features": ["x1", "x2", "x3"]},
            {"y": Y_REG, "x1": X1, "x2": X2, "x3": X3},
        )
        _check_schema(self, r)

    def test_isolation_forest(self):
        r = _run(
            "isolation_forest",
            {"features": ["x1", "x2"]},
            {"x1": X1, "x2": X2},
        )
        _check_schema(self, r)

    def test_regularized_regression(self):
        r = _run(
            "regularized_regression",
            {"response": "y", "predictors": ["x1", "x2", "x3"]},
            {"y": Y_REG, "x1": X1, "x2": X2, "x3": X3},
        )
        _check_schema(self, r)

    def test_discriminant_analysis(self):
        r = _run(
            "discriminant_analysis",
            {"features": ["x1", "x2"]},
            {"x1": X1, "x2": X2},
        )
        _check_schema(self, r)

    def test_factor_analysis(self):
        r = _run(
            "factor_analysis",
            {"features": ["x1", "x2", "x3"]},
            {"x1": X1, "x2": X2, "x3": X3},
        )
        _check_schema(self, r)

    def test_correspondence_analysis(self):
        cat1 = np.random.RandomState(42).choice(["A", "B", "C"], N).tolist()
        cat2 = np.random.RandomState(43).choice(["X", "Y", "Z"], N).tolist()
        r = _run(
            "correspondence_analysis",
            {"features": ["c1", "c2"]},
            {"c1": cat1, "c2": cat2},
        )
        _check_schema(self, r)
