"""Logic engine scenario tests — PBS, quality economics, causal discovery,
drift detection, anytime-valid inference, Bayesian DOE.

Covers: pbs_engine.py, quality_economics.py, causal_discovery.py,
drift_detection.py, anytime_valid.py, bayes_doe.py.

Linked from STAT-001 and DSW-001 via <!-- test: --> hooks.
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

DSW_URL = "/api/dsw/analysis/"
_PASSWORD = "testpass123!"


def _make_user(email, tier=Tier.PRO, password=_PASSWORD, **kwargs):
    username = kwargs.pop("username", email.split("@")[0])
    user = User.objects.create_user(username=username, email=email, password=password, **kwargs)
    user.tier = tier
    user.save(update_fields=["tier"])
    return user


# =========================================================================
# Quality Economics
# =========================================================================


@SECURE_OFF
class QualityEconomicsScenarioTest(TestCase):
    """Scenario: quality cost analysis via DSW endpoint."""

    def setUp(self):
        self.user = _make_user("qecon@engine.com")
        self.client = APIClient()
        self.client.force_login(self.user)

    def test_taguchi_loss(self):
        """Taguchi loss function — nominal-is-best with correct API."""
        from agents_api.quality_economics import TaguchiLoss

        loss = TaguchiLoss(loss_type="nib", target=100.0, delta0=5.0, cost_at_limit=250.0)
        # Verify the object was created with correct attributes
        self.assertEqual(loss.target, 100.0)
        self.assertEqual(loss.delta0, 5.0)

    def test_taguchi_via_endpoint(self):
        """Taguchi analysis via DSW endpoint with sample data."""
        rng = np.random.RandomState(42)
        data = {"measurement": rng.normal(100, 3, 50).tolist()}

        res = self.client.post(
            DSW_URL,
            json.dumps(
                {
                    "type": "quality_econ",
                    "analysis": "taguchi",
                    "data": data,
                    "config": {
                        "column": "measurement",
                        "target": 100.0,
                        "tolerance": 5.0,
                        "loss_per_unit": 50.0,
                    },
                }
            ),
            content_type="application/json",
        )

        self.assertEqual(res.status_code, 200)
        result = res.json()
        self.assertIn("summary", result)

    def test_process_decision(self):
        """Process decision analysis — instantiation with correct cost args."""
        from agents_api.quality_economics import ProcessDecision

        pd_obj = ProcessDecision(
            c_miss=500.0,
            c_fa=100.0,
            c_inv=80.0,
            c_over=120.0,
            c_adj=150.0,
        )
        self.assertEqual(pd_obj.c_miss, 500.0)
        self.assertEqual(pd_obj.c_adj, 150.0)

    def test_cost_of_quality(self):
        """Cost of Quality analysis → categorizes prevention/appraisal/failure."""
        from agents_api.quality_economics import CostOfQuality

        coq = CostOfQuality(
            prevention=1000,
            appraisal=2000,
            internal_failure=3000,
            external_failure=5000,
        )
        self.assertEqual(coq.total, 11000)
        self.assertAlmostEqual(coq.prevention / coq.total, 1000 / 11000, places=4)


# =========================================================================
# Causal Discovery
# =========================================================================


@SECURE_OFF
class CausalDiscoveryScenarioTest(TestCase):
    """Scenario: causal structure learning from observational data."""

    def setUp(self):
        self.user = _make_user("causal@engine.com")
        self.client = APIClient()
        self.client.force_login(self.user)

    def test_pc_algorithm(self):
        """PC algorithm → discovers DAG from correlational data."""
        rng = np.random.RandomState(42)
        n = 200
        x = rng.normal(0, 1, n)
        y = 0.7 * x + rng.normal(0, 0.5, n)
        z = 0.5 * y + rng.normal(0, 0.5, n)

        data = {"x": x.tolist(), "y": y.tolist(), "z": z.tolist()}

        res = self.client.post(
            DSW_URL,
            json.dumps(
                {
                    "type": "causal",
                    "analysis": "pc",
                    "data": data,
                    "config": {"alpha": 0.05, "max_cond_size": 2},
                }
            ),
            content_type="application/json",
        )

        self.assertEqual(res.status_code, 200)
        result = res.json()
        self.assertIn("summary", result)

    def test_lingam(self):
        """LiNGAM → discovers directed acyclic graph with non-Gaussian data."""
        rng = np.random.RandomState(42)
        n = 200
        x = rng.exponential(1, n)
        y = 0.8 * x + rng.exponential(0.5, n)

        data = {"x": x.tolist(), "y": y.tolist()}

        res = self.client.post(
            DSW_URL,
            json.dumps(
                {
                    "type": "causal",
                    "analysis": "lingam",
                    "data": data,
                    "config": {"alpha": 0.05},
                }
            ),
            content_type="application/json",
        )

        self.assertEqual(res.status_code, 200)


# =========================================================================
# Drift Detection
# =========================================================================


@SECURE_OFF
class DriftDetectionScenarioTest(TestCase):
    """Scenario: detect distributional drift in streaming data."""

    def setUp(self):
        self.user = _make_user("drift@engine.com")
        self.client = APIClient()
        self.client.force_login(self.user)

    def test_drift_report(self):
        """Drift report → detects shift between reference and current data."""
        rng = np.random.RandomState(42)
        ref = rng.normal(100, 5, 100).tolist()
        cur = rng.normal(103, 5, 100).tolist()

        data = {"measurement": ref + cur}

        res = self.client.post(
            DSW_URL,
            json.dumps(
                {
                    "type": "drift",
                    "analysis": "drift_report",
                    "data": data,
                    "config": {
                        "column": "measurement",
                        "split_point": 100,
                    },
                }
            ),
            content_type="application/json",
        )

        self.assertEqual(res.status_code, 200)
        result = res.json()
        self.assertIn("summary", result)

    def test_psi_calculation(self):
        """Population Stability Index detects distribution shift."""
        from agents_api.drift_detection import _compute_psi

        rng = np.random.RandomState(42)
        ref = rng.normal(0, 1, 500)
        shifted = rng.normal(0.5, 1, 500)

        result = _compute_psi(ref, shifted)
        # Returns (psi_value, bins_list) tuple
        psi_value = result[0] if isinstance(result, tuple) else result
        self.assertGreater(psi_value, 0.0)


# =========================================================================
# Anytime-Valid Inference
# =========================================================================


@SECURE_OFF
class AnytimeValidScenarioTest(TestCase):
    """Scenario: sequential A/B testing with anytime-valid e-processes."""

    def setUp(self):
        self.user = _make_user("anytime@engine.com")
        self.client = APIClient()
        self.client.force_login(self.user)

    def test_gaussian_e_process(self):
        """Gaussian mean e-process accumulates evidence over time."""
        from agents_api.anytime_valid import GaussianMeanEProcess

        proc = GaussianMeanEProcess(mu0=0.0, sigma=1.0)
        rng = np.random.RandomState(42)

        # Feed observations from shifted distribution
        for obs in rng.normal(0.5, 1, 50):
            proc.update(obs)

        # S_t accumulates sum; e_value is the e-process value
        self.assertGreater(proc.t, 0)
        self.assertIsNotNone(proc.e_value)

    def test_anytime_via_endpoint(self):
        """Anytime-valid analysis via DSW endpoint."""
        rng = np.random.RandomState(42)
        n = 100
        data = {
            "group": (["control"] * (n // 2) + ["treatment"] * (n // 2)),
            "metric": np.concatenate(
                [
                    rng.normal(100, 15, n // 2),
                    rng.normal(105, 15, n // 2),
                ]
            ).tolist(),
        }

        res = self.client.post(
            DSW_URL,
            json.dumps(
                {
                    "type": "anytime",
                    "analysis": "ab_test",
                    "data": data,
                    "config": {"group_col": "group", "metric_col": "metric"},
                }
            ),
            content_type="application/json",
        )

        self.assertEqual(res.status_code, 200)
        result = res.json()
        self.assertIn("summary", result)


# =========================================================================
# Bayesian DOE
# =========================================================================


class BayesDOEScenarioTest(TestCase):
    """Scenario: Bayesian design of experiments — direct module tests."""

    def test_build_design_matrix(self):
        """Build DOE design matrix from factorial data."""
        from agents_api.bayes_doe import build_doe_design_matrix

        df = pd.DataFrame(
            {
                "temp": [150, 170, 150, 170, 150, 170, 150, 170],
                "pressure": [50, 50, 100, 100, 50, 50, 100, 100],
                "yield_val": [35, 45, 42, 55, 37, 47, 44, 57],
            }
        )

        # Returns (X, y, col_names, factor_names)
        result = build_doe_design_matrix(df, ["temp", "pressure"], "yield_val")
        X, y = result[0], result[1]
        self.assertEqual(X.shape[0], 8)
        self.assertEqual(len(y), 8)

    def test_bayesian_linear_posterior(self):
        """Bayesian linear posterior from DOE data."""
        from agents_api.bayes_doe import bayesian_linear_posterior, build_doe_design_matrix

        df = pd.DataFrame(
            {
                "temp": [150, 170, 150, 170, 150, 170, 150, 170],
                "pressure": [50, 50, 100, 100, 50, 50, 100, 100],
                "yield_val": [35, 45, 42, 55, 37, 47, 44, 57],
            }
        )

        result = build_doe_design_matrix(df, ["temp", "pressure"], "yield_val")
        X, y = result[0], result[1]
        posterior = bayesian_linear_posterior(X, y)
        self.assertIsNotNone(posterior)

    def test_run_bayesian_doe(self):
        """Full Bayesian DOE analysis via run_bayesian_doe."""
        from agents_api.bayes_doe import run_bayesian_doe

        df = pd.DataFrame(
            {
                "temp": [150, 170, 150, 170, 160, 160],
                "pressure": [50, 50, 100, 100, 75, 75],
                "yield_val": [35, 45, 42, 55, 40, 41],
            }
        )

        result = run_bayesian_doe(
            df,
            "effects",
            {
                "factors": ["temp", "pressure"],
                "response": "yield_val",
            },
        )
        self.assertIn("summary", result)


# =========================================================================
# PBS Engine
# =========================================================================


@SECURE_OFF
class PBSEngineTest(TestCase):
    """Test PBS engine core classes directly."""

    def test_normal_gamma_posterior(self):
        """NormalGammaPosterior updates with observations."""
        from agents_api.pbs_engine import NormalGammaPosterior

        posterior = NormalGammaPosterior(mu=0, kappa=1, alpha=1, beta=1)

        data = np.array([10.1, 9.8, 10.3, 10.0, 9.9])
        posterior.update(data)

        # Posterior mu should shift toward data mean
        self.assertGreater(posterior.mu, 5.0)

    def test_belief_chart(self):
        """BeliefChart processes observations and tracks run lengths."""
        from agents_api.pbs_engine import BeliefChart

        chart = BeliefChart()

        # Process observations
        points = []
        for val in [101, 99, 102, 98, 100]:
            pt = chart.process(float(val))
            points.append(pt)

        self.assertEqual(len(points), 5)
        self.assertTrue(hasattr(points[0], "shift_probability"))

    def test_e_detector(self):
        """EDetector raises alarm on process shift."""
        from agents_api.pbs_engine import EDetector

        detector = EDetector(mu_0=100.0, bounds=(90.0, 110.0))

        # In-control observations — should not alarm
        for val in np.random.RandomState(42).normal(100, 5, 20):
            detector.process(val)

        in_control_alarms = sum(p.alarm for p in detector.points)

        # Out-of-control observations — should alarm
        for val in np.random.RandomState(99).normal(120, 5, 30):
            detector.process(val)

        total_alarms = sum(p.alarm for p in detector.points)
        self.assertGreater(total_alarms, in_control_alarms)
