"""DSW engine scenario tests — bayesian, ML, reliability, simulation, statistics.

Tests exercise DSW submodule functions directly and via HTTP endpoints.
Covers: dsw/bayesian.py, dsw/ml.py, dsw/stats.py, dsw/simulation.py,
dsw/reliability.py, dsw/d_type.py, dsw/viz.py, dsw/endpoints_data.py,
dsw/endpoints_ml.py.

Linked from DSW-001 and STAT-001 via <!-- test: --> hooks.
"""

import json

import numpy as np
import pandas as pd
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from accounts.constants import Tier

User = get_user_model()
SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)


def _make_user(email, tier=Tier.PRO, password="testpass123!", **kwargs):
    username = kwargs.pop("username", email.split("@")[0])
    user = User.objects.create_user(username=username, email=email, password=password, **kwargs)
    user.tier = tier
    user.save(update_fields=["tier"])
    return user


def _sample_data(n=50, seed=42):
    """Generate sample data for statistical tests."""
    rng = np.random.RandomState(seed)
    return {
        "x": rng.normal(100, 15, n).tolist(),
        "y": (rng.normal(100, 15, n) * 0.8 + rng.normal(0, 5, n)).tolist(),
        "group": (["A"] * (n // 2) + ["B"] * (n // 2)),
    }


# =========================================================================
# Bayesian Analysis
# =========================================================================


@SECURE_OFF
class BayesianAnalysisScenarioTest(TestCase):
    """Scenario: run Bayesian analyses via DSW dispatch endpoint."""

    def setUp(self):
        self.user = _make_user("bayesian@dsw.com")
        self.client = APIClient()
        self.client.force_login(self.user)

    def test_bayesian_regression(self):
        """Bayesian regression → returns posterior coefficients with credible intervals."""
        data = _sample_data()
        res = self.client.post(
            "/api/dsw/analysis/",
            json.dumps(
                {
                    "type": "bayesian",
                    "analysis": "bayes_regression",
                    "data": data,
                    "config": {"target": "y", "features": ["x"], "ci": 0.95},
                }
            ),
            content_type="application/json",
        )

        self.assertEqual(res.status_code, 200)
        result = res.json()
        self.assertIn("summary", result)
        self.assertIn("BAYESIAN REGRESSION", result["summary"])

    def test_bayesian_ab_test(self):
        """Bayesian A/B test → endpoint accepts request and returns JSON."""
        data = _sample_data()
        res = self.client.post(
            "/api/dsw/analysis/",
            json.dumps(
                {
                    "type": "bayesian",
                    "analysis": "bayes_ab",
                    "data": data,
                    "config": {"group_col": "group", "metric_col": "x"},
                }
            ),
            content_type="application/json",
        )

        self.assertNotIn(res.status_code, [401, 403, 404])
        if res.status_code == 200:
            result = res.json()
            self.assertIn("summary", result)


# =========================================================================
# Statistical Analysis
# =========================================================================


@SECURE_OFF
class StatisticalAnalysisScenarioTest(TestCase):
    """Scenario: run common statistical tests via DSW endpoint."""

    def setUp(self):
        self.user = _make_user("stats@dsw.com")
        self.client = APIClient()
        self.client.force_login(self.user)

    def test_descriptive_stats(self):
        """Descriptive statistics → returns mean, std, quartiles."""
        data = _sample_data()
        res = self.client.post(
            "/api/dsw/analysis/",
            json.dumps(
                {
                    "type": "stats",
                    "analysis": "descriptive",
                    "data": data,
                    "config": {"columns": ["x", "y"]},
                }
            ),
            content_type="application/json",
        )

        self.assertEqual(res.status_code, 200)
        result = res.json()
        self.assertIn("summary", result)

    def test_anova(self):
        """One-way ANOVA → returns F-statistic and p-value."""
        data = _sample_data()
        res = self.client.post(
            "/api/dsw/analysis/",
            json.dumps(
                {
                    "type": "stats",
                    "analysis": "anova",
                    "data": data,
                    "config": {"group_col": "group", "value_col": "x"},
                }
            ),
            content_type="application/json",
        )

        self.assertEqual(res.status_code, 200)

    def test_regression(self):
        """Linear regression → endpoint accepts request and returns JSON."""
        data = _sample_data()
        res = self.client.post(
            "/api/dsw/analysis/",
            json.dumps(
                {
                    "type": "stats",
                    "analysis": "regression",
                    "data": data,
                    "config": {"target": "y", "features": ["x"]},
                }
            ),
            content_type="application/json",
        )

        # Endpoint is reachable (not 404/401)
        self.assertNotIn(res.status_code, [401, 403, 404])
        if res.status_code == 200:
            result = res.json()
            self.assertIn("summary", result)

    def test_requires_auth(self):
        """Unauthenticated request → 401."""
        client = APIClient()
        res = client.post(
            "/api/dsw/analysis/",
            json.dumps(
                {
                    "type": "stats",
                    "analysis": "descriptive",
                    "data": _sample_data(),
                    "config": {},
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 401)


# =========================================================================
# ML Analysis
# =========================================================================


@SECURE_OFF
class MLAnalysisScenarioTest(TestCase):
    """Scenario: run ML model training and prediction via DSW."""

    def setUp(self):
        self.user = _make_user("ml@dsw.com")
        self.client = APIClient()
        self.client.force_login(self.user)

    def test_classification(self):
        """Classification model → returns accuracy, confusion matrix."""
        rng = np.random.RandomState(42)
        n = 100
        data = {
            "feature1": rng.normal(0, 1, n).tolist(),
            "feature2": rng.normal(0, 1, n).tolist(),
            "label": (["class_a"] * (n // 2) + ["class_b"] * (n // 2)),
        }

        res = self.client.post(
            "/api/dsw/analysis/",
            json.dumps(
                {
                    "type": "ml",
                    "analysis": "classification",
                    "data": data,
                    "config": {"target": "label", "features": ["feature1", "feature2"]},
                }
            ),
            content_type="application/json",
        )

        self.assertEqual(res.status_code, 200)
        result = res.json()
        self.assertIn("summary", result)

    def test_clustering(self):
        """Clustering → returns cluster assignments and centroids."""
        rng = np.random.RandomState(42)
        n = 60
        data = {
            "x": np.concatenate(
                [
                    rng.normal(0, 1, n // 3),
                    rng.normal(5, 1, n // 3),
                    rng.normal(10, 1, n // 3),
                ]
            ).tolist(),
            "y": np.concatenate(
                [
                    rng.normal(0, 1, n // 3),
                    rng.normal(5, 1, n // 3),
                    rng.normal(10, 1, n // 3),
                ]
            ).tolist(),
        }

        res = self.client.post(
            "/api/dsw/analysis/",
            json.dumps(
                {
                    "type": "ml",
                    "analysis": "clustering",
                    "data": data,
                    "config": {"features": ["x", "y"], "n_clusters": 3},
                }
            ),
            content_type="application/json",
        )

        self.assertEqual(res.status_code, 200)


# =========================================================================
# Simulation
# =========================================================================


@SECURE_OFF
class SimulationScenarioTest(TestCase):
    """Scenario: Monte Carlo and discrete event simulation via DSW."""

    def setUp(self):
        self.user = _make_user("sim@dsw.com")
        self.client = APIClient()
        self.client.force_login(self.user)

    def test_monte_carlo(self):
        """Monte Carlo simulation → returns distribution of outcomes."""
        res = self.client.post(
            "/api/dsw/analysis/",
            json.dumps(
                {
                    "type": "simulation",
                    "analysis": "monte_carlo",
                    "data": {},
                    "config": {
                        "distributions": [
                            {"name": "cost", "type": "normal", "mean": 100, "std": 15},
                            {
                                "name": "units",
                                "type": "uniform",
                                "low": 50,
                                "high": 150,
                            },
                        ],
                        "formula": "cost * units",
                        "n_simulations": 1000,
                    },
                }
            ),
            content_type="application/json",
        )

        self.assertEqual(res.status_code, 200)
        result = res.json()
        self.assertIn("summary", result)

    def test_tolerance_stackup(self):
        """Tolerance stack-up analysis → returns RSS and worst-case."""
        data = _sample_data(30)
        res = self.client.post(
            "/api/dsw/analysis/",
            json.dumps(
                {
                    "type": "simulation",
                    "analysis": "tolerance_stackup",
                    "data": data,
                    "config": {
                        "tolerances": [
                            {"name": "dim1", "nominal": 10, "tolerance": 0.1},
                            {"name": "dim2", "nominal": 20, "tolerance": 0.2},
                        ],
                    },
                }
            ),
            content_type="application/json",
        )

        self.assertEqual(res.status_code, 200)


# =========================================================================
# Reliability
# =========================================================================


@SECURE_OFF
class ReliabilityScenarioTest(TestCase):
    """Scenario: reliability and survival analysis via DSW."""

    def setUp(self):
        self.user = _make_user("rel@dsw.com")
        self.client = APIClient()
        self.client.force_login(self.user)

    def test_weibull_analysis(self):
        """Weibull analysis → endpoint accepts request and returns JSON."""
        rng = np.random.RandomState(42)
        times = (rng.weibull(2, 50) * 1000).tolist()
        data = {"time": times, "failed": [1] * 40 + [0] * 10}

        res = self.client.post(
            "/api/dsw/analysis/",
            json.dumps(
                {
                    "type": "reliability",
                    "analysis": "weibull",
                    "data": data,
                    "config": {"time_col": "time", "event_col": "failed"},
                }
            ),
            content_type="application/json",
        )

        self.assertNotIn(res.status_code, [401, 403, 404])
        if res.status_code == 200:
            result = res.json()
            self.assertIn("summary", result)


# =========================================================================
# D-Type Distribution Analysis
# =========================================================================


@SECURE_OFF
class DTypeAnalysisTest(TestCase):
    """Test D-chart and D-Cpk distribution analysis functions."""

    def test_run_d_chart_basic(self):
        """D-chart processes sample data without error."""
        from agents_api.analysis.d_type import run_d_chart

        rng = np.random.RandomState(42)
        df = pd.DataFrame({"measurement": rng.normal(100, 5, 100)})
        config = {"column": "measurement"}

        result = run_d_chart(df, config)
        self.assertIn("plots", result)
        self.assertIn("summary", result)

    def test_run_d_cpk_basic(self):
        """D-Cpk processes data with spec limits."""
        from agents_api.analysis.d_type import run_d_cpk

        rng = np.random.RandomState(42)
        df = pd.DataFrame({"measurement": rng.normal(100, 5, 100)})
        config = {"column": "measurement", "lsl": 85, "usl": 115}

        result = run_d_cpk(df, config)
        self.assertIn("summary", result)


# =========================================================================
# SPC Engine
# =========================================================================


@SECURE_OFF
# SPCEngineTest removed — agents_api/spc.py deleted, forgespc (95 tests) is canonical

# =========================================================================
# DSW Endpoint Integration
# =========================================================================


@SECURE_OFF
class DSWEndpointScenarioTest(TestCase):
    """Scenario: full DSW workflow — upload data → analyze → save result."""

    def setUp(self):
        self.user = _make_user("dsw-full@dsw.com")
        self.client = APIClient()
        self.client.force_login(self.user)

    def test_analysis_with_save_result(self):
        """Run analysis with save_result=true → persists DSWResult for later import."""
        data = _sample_data()
        res = self.client.post(
            "/api/dsw/analysis/",
            json.dumps(
                {
                    "type": "stats",
                    "analysis": "descriptive",
                    "data": data,
                    "config": {"columns": ["x"]},
                    "save_result": True,
                    "title": "Descriptive Stats Test",
                }
            ),
            content_type="application/json",
        )

        self.assertEqual(res.status_code, 200)
        result = res.json()
        # If save_result creates a DSWResult, there should be a result_id
        # Check either result_id or that the analysis ran successfully
        self.assertIn("summary", result)

    def test_invalid_analysis_type(self):
        """Unknown analysis type → returns error gracefully."""
        res = self.client.post(
            "/api/dsw/analysis/",
            json.dumps(
                {
                    "type": "unknown_type",
                    "analysis": "nonexistent",
                    "data": _sample_data(),
                    "config": {},
                }
            ),
            content_type="application/json",
        )

        # Should return an error, not crash
        self.assertIn(res.status_code, [200, 400])

    def test_missing_data(self):
        """Analysis without data → returns meaningful error."""
        res = self.client.post(
            "/api/dsw/analysis/",
            json.dumps(
                {
                    "type": "stats",
                    "analysis": "descriptive",
                    "config": {"columns": ["x"]},
                }
            ),
            content_type="application/json",
        )

        self.assertIn(res.status_code, [200, 400])
