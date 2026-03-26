"""T2-COV smoke tests — view module coverage for Revenue tier.

Covers: synara_views, spc_views, experimenter_views, dsw_views endpoints
not already tested in test_endpoint_smoke.py.

Also covers internal-only engines: viz.py, simulation.py (called directly).

Standard: CAL-001 §7 (Endpoint Coverage), TST-001 §10.6
Compliance: SOC 2 CC4.1, CC7.2
<!-- test: agents_api.tests.test_t2_views_smoke -->
"""

import numpy as np
import pandas as pd
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from accounts.models import Tier, User

SMOKE_SETTINGS = {"RATELIMIT_ENABLE": False, "SECURE_SSL_REDIRECT": False}


def _make_user(email, tier=Tier.PRO, staff=False):
    username = email.split("@")[0].replace(".", "_")
    u = User.objects.create_user(username=username, email=email, password="testpass123")
    u.tier = tier
    u.is_staff = staff
    u.email_verified = True
    u.save()
    return u


# ---------------------------------------------------------------------------
# Synara — belief engine CRUD
# ---------------------------------------------------------------------------


@override_settings(**SMOKE_SETTINGS)
class SynaraViewsSmokeTest(TestCase):
    """Synara endpoints require a workbench_id (Problem UUID)."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _make_user("smoke-synara@test.com")

    def setUp(self):
        self.auth = APIClient()
        self.auth.force_authenticate(self.user)
        self.anon = APIClient()

    def _get_or_create_project(self):
        from core.models import Project

        p, _ = Project.objects.get_or_create(
            user=self.user,
            defaults={"title": "Test Project for Synara"},
        )
        return str(p.id)

    def test_hypotheses_list_unauth(self):
        res = self.anon.get("/api/synara/00000000-0000-0000-0000-000000000000/hypotheses/")
        self.assertIn(res.status_code, [401, 403])

    def test_hypotheses_list_auth(self):
        pid = self._get_or_create_project()
        res = self.auth.get(f"/api/synara/{pid}/hypotheses/")
        self.assertNotEqual(res.status_code, 500)

    def test_add_hypothesis(self):
        pid = self._get_or_create_project()
        res = self.auth.post(
            f"/api/synara/{pid}/hypotheses/add/",
            {"text": "Test hypothesis", "prior": 0.5},
            format="json",
        )
        self.assertNotEqual(res.status_code, 500)

    def test_evidence_list(self):
        pid = self._get_or_create_project()
        res = self.auth.get(f"/api/synara/{pid}/evidence/")
        self.assertNotEqual(res.status_code, 500)

    def test_state_endpoint(self):
        pid = self._get_or_create_project()
        res = self.auth.get(f"/api/synara/{pid}/state/")
        self.assertNotEqual(res.status_code, 500)

    def test_dsl_parse(self):
        pid = self._get_or_create_project()
        res = self.auth.post(
            f"/api/synara/{pid}/dsl/parse/",
            {"expression": "H1: p > 0.5"},
            format="json",
        )
        self.assertNotEqual(res.status_code, 500)

    def test_export(self):
        pid = self._get_or_create_project()
        res = self.auth.get(f"/api/synara/{pid}/export/")
        self.assertNotEqual(res.status_code, 500)


# ---------------------------------------------------------------------------
# SPC — extended endpoint tests
# ---------------------------------------------------------------------------


@override_settings(**SMOKE_SETTINGS)
class SPCViewsSmokeTest(TestCase):
    """SPC endpoints beyond existing smoke tests."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _make_user("smoke-spc@test.com")

    def setUp(self):
        self.auth = APIClient()
        self.auth.force_authenticate(self.user)

    def test_chart_types(self):
        res = self.auth.get("/api/spc/chart/types/")
        self.assertNotEqual(res.status_code, 500)

    def test_chart_recommend_unauth(self):
        anon = APIClient()
        res = anon.post("/api/spc/chart/recommend/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])

    def test_summary_unauth(self):
        anon = APIClient()
        res = anon.post("/api/spc/summary/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])

    def test_gage_rr_unauth(self):
        anon = APIClient()
        res = anon.post("/api/spc/gage-rr/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])


# ---------------------------------------------------------------------------
# Experimenter — DOE extended tests
# ---------------------------------------------------------------------------


@override_settings(**SMOKE_SETTINGS)
class ExperimenterViewsSmokeTest(TestCase):
    """Experimenter endpoints beyond existing smoke tests."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _make_user("smoke-exp@test.com")

    def setUp(self):
        self.auth = APIClient()
        self.auth.force_authenticate(self.user)
        self.anon = APIClient()

    def test_design_types(self):
        res = self.auth.get("/api/experimenter/design/types/")
        self.assertNotEqual(res.status_code, 500)

    def test_models_list(self):
        res = self.auth.get("/api/experimenter/models/")
        self.assertNotEqual(res.status_code, 500)

    def test_full_unauth(self):
        res = self.anon.post("/api/experimenter/full/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])

    def test_analyze_unauth(self):
        res = self.anon.post("/api/experimenter/analyze/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])

    def test_contour_unauth(self):
        res = self.anon.post("/api/experimenter/contour/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])

    def test_optimize_unauth(self):
        res = self.anon.post("/api/experimenter/optimize/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])


# ---------------------------------------------------------------------------
# DSW — additional endpoint tests
# ---------------------------------------------------------------------------


@override_settings(**SMOKE_SETTINGS)
class DSWViewsSmokeTest(TestCase):
    """DSW endpoints beyond existing smoke tests."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _make_user("smoke-dsw2@test.com")

    def setUp(self):
        self.auth = APIClient()
        self.auth.force_authenticate(self.user)
        self.anon = APIClient()

    def test_upload_data_unauth(self):
        res = self.anon.post("/api/dsw/upload-data/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])

    def test_transform_unauth(self):
        res = self.anon.post("/api/dsw/transform/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])

    def test_generate_code_unauth(self):
        res = self.anon.post("/api/dsw/generate-code/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])

    def test_execute_unauth(self):
        res = self.anon.post("/api/dsw/execute/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])

    def test_models_summary(self):
        res = self.auth.get("/api/dsw/models/summary/")
        self.assertNotEqual(res.status_code, 500)

    def test_triage_unauth(self):
        res = self.anon.post("/api/dsw/triage/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])


# ---------------------------------------------------------------------------
# viz.py — internal engine (direct call tests)
# ---------------------------------------------------------------------------


class VizEngineCoverageTest(TestCase):
    """Direct tests for run_visualization() — internal engine, not HTTP."""

    def _run_viz(self, analysis_id, config, data_dict):
        """Run viz analysis — no exception masking (TST-001 §11.6)."""
        from agents_api.analysis.viz import run_visualization

        df = pd.DataFrame(data_dict)
        return run_visualization(df, analysis_id, config)

    def test_histogram(self):
        r = self._run_viz(
            "histogram",
            {"var": "x"},
            {"x": list(np.random.RandomState(42).normal(0, 1, 50))},
        )
        self.assertIsInstance(r, dict)

    def test_boxplot(self):
        r = self._run_viz(
            "boxplot",
            {"var": "x"},
            {"x": list(np.random.RandomState(42).normal(0, 1, 50))},
        )
        self.assertIsInstance(r, dict)

    def test_scatter(self):
        r = self._run_viz(
            "scatter",
            {"x": "x", "y": "y"},
            {
                "x": list(np.random.RandomState(42).normal(0, 1, 50)),
                "y": list(np.random.RandomState(43).normal(0, 1, 50)),
            },
        )
        self.assertIsInstance(r, dict)

    def test_heatmap(self):
        r = self._run_viz(
            "heatmap",
            {"vars": ["x", "y", "z"]},
            {
                "x": list(np.random.RandomState(42).normal(0, 1, 30)),
                "y": list(np.random.RandomState(43).normal(0, 1, 30)),
                "z": list(np.random.RandomState(44).normal(0, 1, 30)),
            },
        )
        self.assertIsInstance(r, dict)

    def test_qq(self):
        r = self._run_viz("qq", {"var1": "x"}, {"x": list(np.random.RandomState(42).normal(0, 1, 50))})
        self.assertIsInstance(r, dict)


# ---------------------------------------------------------------------------
# simulation.py — internal engine (direct call tests)
# ---------------------------------------------------------------------------


class SimulationEngineCoverageTest(TestCase):
    """Direct tests for run_simulation() — internal engine, not HTTP."""

    def _run_sim(self, analysis_id, config, data_dict=None):
        """Run simulation — no exception masking (TST-001 §11.6)."""
        from agents_api.analysis.simulation import run_simulation

        df = pd.DataFrame(data_dict) if data_dict else pd.DataFrame()
        return run_simulation(df, analysis_id, config, user=None)

    def test_monte_carlo(self):
        r = self._run_sim(
            "monte_carlo",
            {
                "expression": "x + y",
                "variables": {
                    "x": {"distribution": "normal", "mean": 10, "std": 1},
                    "y": {"distribution": "normal", "mean": 5, "std": 0.5},
                },
                "n_simulations": 1000,
            },
        )
        self.assertIsInstance(r, dict)

    def test_tolerance_stackup(self):
        r = self._run_sim(
            "tolerance_stackup",
            {
                "dimensions": [
                    {"name": "A", "nominal": 10.0, "tolerance": 0.1},
                    {"name": "B", "nominal": 5.0, "tolerance": 0.05},
                ],
                "n_simulations": 1000,
            },
        )
        self.assertIsInstance(r, dict)

    def test_variance_propagation(self):
        r = self._run_sim(
            "variance_propagation",
            {
                "expression": "x * y",
                "variables": {
                    "x": {"mean": 10, "std": 1},
                    "y": {"mean": 5, "std": 0.5},
                },
            },
        )
        self.assertIsInstance(r, dict)
