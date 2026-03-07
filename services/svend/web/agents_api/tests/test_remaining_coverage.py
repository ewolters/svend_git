"""Coverage tests for remaining DSW modules — CAL-001 §6 / TST-001 §10.

Covers: quality_economics, d_type, causal_discovery, simulation,
        anytime_valid, drift_detection, msa_bayes, interventional_shap.

Standard: CAL-001 §6 (Statistical Correctness Verification)
Compliance: SOC 2 CC4.1
<!-- test: agents_api.tests.test_remaining_coverage -->
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


def _check_schema(tc, r):
    """Verify output schema per CAL-001 §6 / TST-001 §10.6."""
    aid = tc._testMethodName
    tc.assertIsInstance(r, dict, f"{aid} did not return a dict")
    tc.assertIn("plots", r, f"{aid} missing 'plots' key")
    tc.assertIsInstance(r["plots"], list, f"{aid} plots is not a list")
    tc.assertIn("summary", r, f"{aid} missing 'summary' key")


# ===========================================================================
# Quality Economics
# ===========================================================================


class QualityEconCoverageTest(TestCase):
    """Quality economics analyses."""

    def _run(self, analysis_id, config, data_dict):
        from agents_api.quality_economics import run_quality_econ

        df = pd.DataFrame(data_dict)
        return run_quality_econ(df, analysis_id, config)

    def test_taguchi_loss(self):
        r = self._run(
            "taguchi_loss",
            {"column": "x", "target": 100, "delta0": 15, "cost_at_limit": 50, "loss_type": "nib"},
            {"x": NORMAL_60},
        )
        _check_schema(self, r)

    def test_process_decision(self):
        r = self._run(
            "process_decision",
            {"p_ooc": 0.3, "c_miss": 500, "c_fa": 100, "c_inv": 80},
            {"x": [1]},
        )
        _check_schema(self, r)

    def test_lot_sentencing(self):
        r = self._run(
            "lot_sentencing",
            {"p_defect": 0.02, "lot_size": 500, "c_external": 50, "c_internal": 5, "c_inspection": 0.5},
            {"x": [1]},
        )
        _check_schema(self, r)

    def test_cost_of_quality(self):
        r = self._run(
            "cost_of_quality",
            {
                "prevention": 10000,
                "appraisal": 8000,
                "internal_failure": 15000,
                "external_failure": 25000,
                "revenue": 1000000,
            },
            {"x": [1]},
        )
        _check_schema(self, r)


# ===========================================================================
# D-Type Process Intelligence
# ===========================================================================


class DTypeCoverageTest(TestCase):
    """D-Type analyses — factor divergence via Jensen-Shannon Divergence."""

    def _run(self, analysis_id, config, data_dict):
        from agents_api.dsw.d_type import run_d_type

        df = pd.DataFrame(data_dict)
        return run_d_type(df, analysis_id, config)

    def test_d_chart(self):
        # Need time column and factor column
        n = 100
        r = self._run(
            "d_chart",
            {"variable": "y", "factor": "shift", "time_col": "t", "window_size": 30},
            {
                "y": list(np.random.RandomState(42).normal(50, 5, n)),
                "shift": (["Day"] * 50 + ["Night"] * 50),
                "t": list(range(n)),
            },
        )
        _check_schema(self, r)

    def test_d_cpk(self):
        n = 100
        r = self._run(
            "d_cpk",
            {"variable": "y", "factor": "machine", "lsl": 30, "usl": 70},
            {
                "y": list(np.random.RandomState(42).normal(50, 5, n)),
                "machine": (["M1"] * 50 + ["M2"] * 50),
            },
        )
        _check_schema(self, r)

    def test_d_nonnorm(self):
        r = self._run(
            "d_nonnorm",
            {"variable": "y", "lsl": 60, "usl": 140},
            {"y": NORMAL_60},
        )
        _check_schema(self, r)

    def test_d_equiv(self):
        n = 90
        r = self._run(
            "d_equiv",
            {"variable": "y", "batch": "batch"},
            {
                "y": list(np.random.RandomState(42).normal(50, 5, n)),
                "batch": ["B1"] * 30 + ["B2"] * 30 + ["B3"] * 30,
            },
        )
        _check_schema(self, r)

    def test_d_sig(self):
        # Interleave groups so each time window has enough per-group data for KDE
        n = 200
        rng = np.random.RandomState(42)
        groups = (["M1", "M2"] * (n // 2))[:n]
        vals = rng.normal(50, 5, n).tolist()
        r = self._run(
            "d_sig",
            {"variable": "y", "time_col": "t", "group": "machine", "window_size": 50},
            {
                "y": vals,
                "t": list(range(n)),
                "machine": groups,
            },
        )
        _check_schema(self, r)

    def test_d_multi(self):
        r = self._run(
            "d_multi",
            {"variables": ["x", "y", "z"], "lsl": [60, 60, 60], "usl": [140, 140, 140]},
            {"x": NORMAL_60, "y": NORMAL_60B, "z": NORMAL_60C},
        )
        _check_schema(self, r)


# ===========================================================================
# Causal Discovery
# ===========================================================================


class CausalDiscoveryCoverageTest(TestCase):
    """Causal discovery — PC and LiNGAM algorithms."""

    def _run(self, analysis_id, config, data_dict):
        from agents_api.causal_discovery import run_causal_discovery

        df = pd.DataFrame(data_dict)
        return run_causal_discovery(df, analysis_id, config)

    def test_causal_pc(self):
        # Use independent noise to avoid singular correlation matrix
        rng = np.random.RandomState(42)
        n = 80
        x = rng.normal(0, 1, n)
        noise_y = np.random.RandomState(43).normal(0, 1, n)
        noise_z = np.random.RandomState(44).normal(0, 1, n)
        y = 0.5 * x + noise_y
        z = 0.3 * x + 0.4 * y + noise_z
        r = self._run(
            "causal_pc",
            {"variables": ["x", "y", "z"], "alpha": 0.05, "n_bootstraps": 10},
            {"x": x.tolist(), "y": y.tolist(), "z": z.tolist()},
        )
        _check_schema(self, r)

    def test_causal_lingam(self):
        rng = np.random.RandomState(42)
        n = 80
        x = rng.normal(0, 1, n)
        noise_y = np.random.RandomState(43).normal(0, 1, n)
        noise_z = np.random.RandomState(44).normal(0, 1, n)
        y = 0.5 * x + noise_y
        z = 0.3 * x + 0.4 * y + noise_z
        r = self._run(
            "causal_lingam",
            {"variables": ["x", "y", "z"], "n_bootstraps": 10},
            {"x": x.tolist(), "y": y.tolist(), "z": z.tolist()},
        )
        _check_schema(self, r)


# ===========================================================================
# Simulation
# ===========================================================================


class SimulationCoverageTest(TestCase):
    """Simulation — tolerance stackup and variance propagation."""

    def _run(self, analysis_id, config, data_dict):
        from agents_api.dsw.simulation import run_simulation

        df = pd.DataFrame(data_dict)
        return run_simulation(df, analysis_id, config, user=None)

    def test_tolerance_stackup(self):
        r = self._run(
            "tolerance_stackup",
            {
                "components": [
                    {"name": "A", "nominal": 10.0, "tolerance": 0.1, "distribution": "normal"},
                    {"name": "B", "nominal": 5.0, "tolerance": 0.05, "distribution": "normal"},
                    {"name": "C", "nominal": 3.0, "tolerance": 0.08, "distribution": "normal"},
                ],
                "n_simulations": 5000,
            },
            {"x": [1]},
        )
        _check_schema(self, r)

    def test_variance_propagation(self):
        r = self._run(
            "variance_propagation",
            {
                "variables": [
                    {"name": "x", "mean": 10.0, "std": 0.5},
                    {"name": "y", "mean": 5.0, "std": 0.3},
                ],
                "formula": "x + y",
            },
            {"x": [1]},
        )
        _check_schema(self, r)


# ===========================================================================
# Anytime-Valid Inference
# ===========================================================================


class AnytimeValidCoverageTest(TestCase):
    """Anytime-valid inference — e-processes and confidence sequences."""

    def _run(self, analysis_id, config, data_dict):
        from agents_api.anytime_valid import run_anytime_valid

        df = pd.DataFrame(data_dict)
        return run_anytime_valid(df, analysis_id, config)

    def test_anytime_ab(self):
        n = 60
        vals = list(np.random.RandomState(42).normal(50, 5, n))
        groups = ["A"] * 30 + ["B"] * 30
        r = self._run(
            "anytime_ab",
            {"value_col": "y", "group_col": "g", "group_a": "A", "group_b": "B", "alpha": 0.05},
            {"y": vals, "g": groups},
        )
        _check_schema(self, r)

    def test_anytime_onesample(self):
        vals = list(np.random.RandomState(42).normal(50, 5, 40))
        r = self._run(
            "anytime_onesample",
            {"value_col": "y", "mu0": 48, "alpha": 0.05},
            {"y": vals},
        )
        _check_schema(self, r)


# ===========================================================================
# Drift Detection
# ===========================================================================


class DriftDetectionCoverageTest(TestCase):
    """Drift detection — 3-lane diagnostic."""

    def _run(self, analysis_id, config, data_dict):
        from agents_api.drift_detection import run_drift_detection

        df = pd.DataFrame(data_dict)
        return run_drift_detection(df, analysis_id, config)

    def test_drift_report(self):
        # Create data with a shift at the midpoint
        seg1 = list(np.random.RandomState(42).normal(50, 5, 30))
        seg2 = list(np.random.RandomState(43).normal(55, 5, 30))
        r = self._run(
            "drift_report",
            {"features": ["x"], "split_pct": 50},
            {"x": seg1 + seg2},
        )
        _check_schema(self, r)


# ===========================================================================
# Bayesian MSA (Gage R&R)
# ===========================================================================


class BayesMSACoverageTest(TestCase):
    """Bayesian Measurement System Analysis."""

    def _run(self, analysis_id, config, data_dict):
        from agents_api.msa_bayes import run_bayes_msa

        df = pd.DataFrame(data_dict)
        return run_bayes_msa(df, analysis_id, config)

    def test_bayes_msa(self):
        # 3 operators x 5 parts x 2 replicates = 30
        parts = list(range(1, 6)) * 6
        ops = (["Op1"] * 5 + ["Op2"] * 5 + ["Op3"] * 5) * 2
        vals = list(np.random.RandomState(42).normal(50, 2, 30))
        r = self._run(
            "bayes_msa",
            {"part": "part", "operator": "op", "measurement": "y", "tolerance": 10},
            {"part": parts, "op": ops, "y": vals},
        )
        _check_schema(self, r)


# ===========================================================================
# Interventional SHAP
# ===========================================================================


class InterventionalSHAPCoverageTest(TestCase):
    """Interventional SHAP — feature attribution under do-calculus."""

    def _run(self, analysis_id, config, data_dict, model=None, model_features=None):
        from agents_api.interventional_shap import run_interventional_shap

        df = pd.DataFrame(data_dict)
        return run_interventional_shap(df, analysis_id, config, model=model, model_features=model_features)

    def test_ishap_no_model(self):
        """Without a model, ishap should return an error message gracefully."""
        r = self._run(
            "ishap",
            {"features": ["x1", "x2"], "target": "y"},
            {"x1": NORMAL_60, "x2": NORMAL_60B, "y": NORMAL_60C},
            model=None,
        )
        _check_schema(self, r)
        # Should contain an error about no model
        self.assertIn("Error", r["summary"])

    def test_ishap_with_model(self):
        """With a trained model, ishap should produce attribution results."""
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            self.skipTest("sklearn not installed")

        n = 60
        x1 = np.random.RandomState(42).normal(0, 1, n)
        x2 = np.random.RandomState(43).normal(0, 1, n)
        y = 2 * x1 + 0.5 * x2 + np.random.RandomState(44).normal(0, 0.3, n)

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(np.column_stack([x1, x2]), y)

        r = self._run(
            "ishap",
            {"features": ["x1", "x2"], "target": "y", "n_bg": 10, "n_explain": 5, "max_perm": 20},
            {"x1": x1.tolist(), "x2": x2.tolist(), "y": y.tolist()},
            model=model,
            model_features=["x1", "x2"],
        )
        _check_schema(self, r)
