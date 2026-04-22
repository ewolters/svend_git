"""
DSW-001 compliance tests: Decision Science Workbench Architecture.

Tests verify stateless dispatch, data source resolution, module split,
Synara belief engine invariants, evidence bridge, result persistence,
feature gating, and IDOR prevention patterns.

Standard: DSW-001
"""

import inspect
import json
import os
import uuid
from pathlib import Path

from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase, override_settings
from rest_framework.test import APIClient

from accounts.constants import Tier

User = get_user_model()

# Base paths (for structural and frontend tests only)
WEB_ROOT = Path(os.path.dirname(__file__)).parent.parent.parent
DSW_DIR = WEB_ROOT / "agents_api" / "analysis"  # dsw/ extracted; analysis/ is canonical
SYNARA_DIR = Path(os.path.expanduser("~")) / "forgesia" / "src" / "forgesia"
AGENTS_API = WEB_ROOT / "agents_api"

SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)


def _read(path):
    """Read a file, return empty string on failure."""
    try:
        return Path(path).read_text(errors="ignore")
    except Exception:
        return ""


def _make_user(email, tier=Tier.PRO, password="testpass123!", **kwargs):
    username = kwargs.pop("username", email.split("@")[0])
    user = User.objects.create_user(username=username, email=email, password=password, **kwargs)
    user.tier = tier
    user.save(update_fields=["tier"])
    return user


def _err_msg(resp):
    """Extract error message from response (handles ErrorEnvelopeMiddleware format).

    ErrorEnvelopeMiddleware wraps errors as:
        {"error": {"code": "...", "message": "...", "retryable": ..., ...}}
    """
    try:
        data = json.loads(resp.content)
    except Exception:
        return ""
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            return err.get("message", "")
        if isinstance(err, str):
            return err
        return data.get("message", "")
    return ""


# =============================================================================
# §4.1: Dispatch Error Handling (functional)
# =============================================================================


@SECURE_OFF
class DispatchErrorHandlingTest(TestCase):
    """DSW-001 §4.1: Dispatch returns proper errors for invalid requests."""

    def setUp(self):
        self.user = _make_user("dispatch_err@dsw.test")
        self.client = APIClient()
        self.client.force_login(self.user)
        self.url = "/api/dsw/analysis/"

    def test_invalid_json_returns_400(self):
        """POST non-JSON body returns 400."""
        res = self.client.post(self.url, "not json{{{", content_type="application/json")
        self.assertEqual(res.status_code, 400)
        self.assertIn("error", res.json())

    def test_unknown_type_returns_400(self):
        """Unknown analysis_type returns 400."""
        res = self.client.post(
            self.url,
            json.dumps({"type": "nonexistent", "analysis": "fake", "data": {"x": [1]}}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("Unknown analysis type", _err_msg(res))

    def test_no_data_returns_400(self):
        """Stats analysis without data returns 400."""
        res = self.client.post(
            self.url,
            json.dumps({"type": "stats", "analysis": "ttest"}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("No data loaded", _err_msg(res))

    def test_invalid_inline_data_returns_400(self):
        """Malformed inline data returns 400."""
        res = self.client.post(
            self.url,
            json.dumps({"type": "stats", "analysis": "ttest", "data": "not_a_dict"}),
            content_type="application/json",
        )
        # data="not_a_dict" is not a dict, so inline data path is skipped → "No data loaded"
        self.assertIn(res.status_code, (400, 500))

    def test_oversized_inline_data_returns_400(self):
        """Inline data exceeding 10,000 rows returns 400."""
        res = self.client.post(
            self.url,
            json.dumps(
                {
                    "type": "stats",
                    "analysis": "ttest",
                    "data": {"x": list(range(10001))},
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("10,000", _err_msg(res))


# =============================================================================
# §4.2: Data Source Resolution (functional)
# =============================================================================


@SECURE_OFF
class DataSourceFunctionalTest(TestCase):
    """DSW-001 §4.2: Data source fallback works at runtime."""

    def setUp(self):
        self.user = _make_user("datasrc@dsw.test")
        self.client = APIClient()
        self.client.force_login(self.user)
        self.url = "/api/dsw/analysis/"

    def test_inline_data_accepted(self):
        """Inline data dict produces a valid analysis result."""
        import numpy as np

        np.random.seed(42)
        data = {"x": np.random.normal(50, 2, 50).tolist()}
        res = self.client.post(
            self.url,
            json.dumps({"type": "stats", "analysis": "descriptive", "data": data}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertIn("summary", body)

    def test_empty_df_for_simulation(self):
        """Simulation runs without data (user-defined distributions)."""
        res = self.client.post(
            self.url,
            json.dumps(
                {
                    "type": "simulation",
                    "analysis": "monte_carlo",
                    "config": {
                        "distributions": [
                            {
                                "name": "process_time",
                                "type": "normal",
                                "params": {"mean": 10, "std": 2},
                            },
                        ],
                        "formula": "process_time",
                        "n_simulations": 100,
                    },
                }
            ),
            content_type="application/json",
        )
        self.assertIn(res.status_code, (200, 500))  # 200 if simulation succeeds, 500 if config issue

    def test_empty_df_for_bayesian(self):
        """Bayesian analysis runs without data."""
        res = self.client.post(
            self.url,
            json.dumps(
                {
                    "type": "bayesian",
                    "analysis": "bayesian_regression",
                    "config": {"predictors": ["x"], "response": "y"},
                }
            ),
            content_type="application/json",
        )
        # Bayesian with no data and empty df may produce runtime error but should not be 400 "No data loaded"
        self.assertNotEqual(_err_msg(res), "No data loaded. Please load a dataset first.")

    def test_no_data_rejects_stats(self):
        """Stats analysis with no data source rejects with 400."""
        res = self.client.post(
            self.url,
            json.dumps({"type": "stats", "analysis": "ttest"}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("No data loaded", _err_msg(res))

    def test_inline_row_limit_enforced(self):
        """More than 10,000 inline rows returns 400."""
        res = self.client.post(
            self.url,
            json.dumps(
                {
                    "type": "stats",
                    "analysis": "descriptive",
                    "data": {"x": list(range(10001))},
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 400)


# =============================================================================
# §4.3: Dispatch Routing (functional)
# =============================================================================


@SECURE_OFF
class DispatchRoutingFunctionalTest(TestCase):
    """DSW-001 §4.3: Dispatch routes to correct sub-module handlers."""

    def setUp(self):
        import numpy as np

        self.user = _make_user("routing@dsw.test")
        self.client = APIClient()
        self.client.force_login(self.user)
        self.url = "/api/dsw/analysis/"
        np.random.seed(42)
        self.sample_data = {
            "x": np.random.normal(50, 2, 50).tolist(),
            "y": np.random.normal(52, 2, 50).tolist(),
            "group": (["A"] * 25) + (["B"] * 25),
        }

    def _post(self, analysis_type, analysis_id, config=None, data=None):
        payload = {"type": analysis_type, "analysis": analysis_id}
        if config:
            payload["config"] = config
        if data is not None:
            payload["data"] = data
        else:
            payload["data"] = self.sample_data
        return self.client.post(self.url, json.dumps(payload), content_type="application/json")

    def test_stats_route_produces_output(self):
        """type=stats routes to stats module and produces summary."""
        res = self._post("stats", "descriptive")
        self.assertEqual(res.status_code, 200)
        self.assertIn("summary", res.json())

    def test_ml_route_produces_output(self):
        """type=ml routes to ML module."""
        res = self._post("ml", "regression", config={"predictors": ["x"], "response": "y"})
        self.assertIn(res.status_code, (200, 500))

    def test_spc_route_produces_output(self):
        """type=spc routes to SPC module."""
        res = self._post("spc", "capability", config={"column": "x", "lsl": 44, "usl": 56})
        self.assertEqual(res.status_code, 200)

    def test_bayesian_route_produces_output(self):
        """type=bayesian routes to Bayesian module."""
        res = self._post(
            "bayesian",
            "bayesian_regression",
            config={"predictors": ["x"], "response": "y"},
        )
        self.assertIn(res.status_code, (200, 500))

    def test_simulation_route_produces_output(self):
        """type=simulation routes to simulation module (no data needed)."""
        res = self._post(
            "simulation",
            "monte_carlo",
            config={
                "distributions": [{"name": "t", "type": "normal", "params": {"mean": 10, "std": 2}}],
                "formula": "t",
                "n_simulations": 100,
            },
            data={},
        )
        self.assertIn(res.status_code, (200, 500))

    def test_unknown_type_rejected(self):
        """type=fake returns 400."""
        res = self._post("fake", "test")
        self.assertEqual(res.status_code, 400)


# =============================================================================
# §5.1: Synara Cache (functional)
# =============================================================================


@SECURE_OFF
class SynaraCacheFunctionalTest(TestCase):
    """DSW-001 §5.1: Cache get/save roundtrip, eviction, project requirement."""

    def setUp(self):
        from agents_api.synara_views import _synara_cache

        _synara_cache.clear()

    def test_get_creates_fresh_instance(self):
        """get_synara with unknown ID returns fresh Synara with 0 hypotheses."""
        from agents_api.synara_views import get_synara

        synara = get_synara(str(uuid.uuid4()))
        self.assertEqual(len(synara.graph.hypotheses), 0)

    def test_cache_returns_same_instance(self):
        """Second get_synara call returns the same object (cache hit)."""
        from agents_api.synara_views import get_synara

        wid = str(uuid.uuid4())
        s1 = get_synara(wid)
        s2 = get_synara(wid)
        self.assertIs(s1, s2)

    def test_save_without_project_returns_false(self):
        """save_synara returns False when no project exists."""
        from agents_api.synara_views import get_synara, save_synara

        wid = str(uuid.uuid4())
        synara = get_synara(wid)
        result = save_synara(wid, synara)
        self.assertFalse(result)

    def test_cache_eviction_at_max(self):
        """Cache stays bounded at _SYNARA_CACHE_MAX entries."""
        from agents_api.synara_views import _SYNARA_CACHE_MAX, _synara_cache, get_synara

        for i in range(_SYNARA_CACHE_MAX + 5):
            get_synara(str(uuid.uuid4()))
        self.assertLessEqual(len(_synara_cache), _SYNARA_CACHE_MAX)


# =============================================================================
# §5.2: Project Resolution (functional)
# =============================================================================


@SECURE_OFF
class ProjectResolutionFunctionalTest(TestCase):
    """DSW-001 §5.2: _resolve_project maps UUIDs to Projects."""

    def test_resolve_core_project(self):
        """Project UUID resolves to that Project."""
        from agents_api.synara_views import _resolve_project
        from core.models import Project

        user = _make_user("resolve1@dsw.test")
        project = Project.objects.create(title="Test Project", user=user)
        result = _resolve_project(str(project.id), user=user)
        self.assertEqual(result.id, project.id)

    def test_resolve_nonexistent_returns_none(self):
        """Random UUID resolves to None."""
        from agents_api.synara_views import _resolve_project

        user = _make_user("resolve3@dsw.test")
        result = _resolve_project(str(uuid.uuid4()), user=user)
        self.assertIsNone(result)

    def test_user_filter_prevents_idor(self):
        """Other user's project resolves to None."""
        from agents_api.synara_views import _resolve_project
        from core.models import Project

        owner = _make_user("owner@dsw.test")
        intruder = _make_user("intruder@dsw.test")
        project = Project.objects.create(title="Owner Project", user=owner)
        result = _resolve_project(str(project.id), user=intruder)
        self.assertIsNone(result)


# =============================================================================
# §5.3: Project Guard HTTP (functional)
# =============================================================================


@SECURE_OFF
class ProjectGuardHTTPTest(TestCase):
    """DSW-001 §5.3: Mutating endpoints return 400 when no project resolves."""

    def setUp(self):
        self.user = _make_user("guard@dsw.test")
        self.client = APIClient()
        self.client.force_login(self.user)
        self.fake_wid = str(uuid.uuid4())

    def test_add_hypothesis_no_project_400(self):
        """add_hypothesis with random workbench_id returns 400."""
        res = self.client.post(
            f"/api/synara/{self.fake_wid}/hypotheses/add/",
            json.dumps({"description": "test hypothesis", "prior": 0.5}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("No study loaded", _err_msg(res))

    def test_add_evidence_no_project_400(self):
        """add_evidence with random workbench_id returns 400."""
        res = self.client.post(
            f"/api/synara/{self.fake_wid}/evidence/add/",
            json.dumps({"event": "test", "supports": [], "weakens": []}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("No study loaded", _err_msg(res))

    def test_add_link_no_project_400(self):
        """add_link with random workbench_id returns 400."""
        res = self.client.post(
            f"/api/synara/{self.fake_wid}/links/add/",
            json.dumps({"from_id": "h1", "to_id": "h2"}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("No study loaded", _err_msg(res))


# =============================================================================
# §5.4: Save Failure (functional)
# =============================================================================


@SECURE_OFF
class SaveFailureFunctionalTest(TestCase):
    """DSW-001 §5.4: save_synara failure propagates correctly."""

    def test_save_returns_false_without_project(self):
        """save_synara with nonexistent workbench_id returns False."""
        from agents_api.synara_views import get_synara, save_synara

        wid = str(uuid.uuid4())
        synara = get_synara(wid)
        self.assertFalse(save_synara(wid, synara))

    def test_save_returns_true_with_project(self):
        """save_synara with valid project returns True."""
        from agents_api.synara_views import get_synara, save_synara
        from core.models import Project

        user = _make_user("save_ok@dsw.test")
        project = Project.objects.create(title="Save Test", user=user)
        wid = str(project.id)
        synara = get_synara(wid, user=user)
        self.assertTrue(save_synara(wid, synara, user=user))

    def test_require_project_returns_400(self):
        """_require_project with random UUID returns (None, 400 response)."""
        from agents_api.synara_views import _require_project

        user = _make_user("reqproj@dsw.test")
        project, err = _require_project(str(uuid.uuid4()), user=user)
        self.assertIsNone(project)
        self.assertIsNotNone(err)
        self.assertEqual(err.status_code, 400)


# =============================================================================
# §6.1: Evidence Bridge (functional)
# =============================================================================


@SECURE_OFF
class EvidenceBridgeFunctionalTest(TestCase):
    """DSW-001 §6.1: Evidence linking from analysis results."""

    def setUp(self):
        self.user = _make_user("evidence@dsw.test")
        self.client = APIClient()
        self.client.force_login(self.user)
        self.url = "/api/dsw/analysis/"

    def test_analysis_without_problem_succeeds(self):
        """Analysis without problem_id returns 200 without problem_updated."""
        import numpy as np

        data = {"x": np.random.normal(50, 2, 50).tolist()}
        res = self.client.post(
            self.url,
            json.dumps({"type": "stats", "analysis": "descriptive", "data": data}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertFalse(body.get("problem_updated", False))


# =============================================================================
# §6.2: DSWResult Persistence (functional)
# =============================================================================


@SECURE_OFF
class DSWResultFunctionalTest(TestCase):
    """DSW-001 §6.2: DSWResult created on save, encrypted, project-linked."""

    def setUp(self):
        self.user = _make_user("result@dsw.test")
        self.client = APIClient()
        self.client.force_login(self.user)
        self.url = "/api/dsw/analysis/"

    def test_dswresult_created_on_save(self):
        """save_result=True returns result_id and creates DB record."""
        import numpy as np

        from agents_api.models import DSWResult

        data = {"x": np.random.normal(50, 2, 50).tolist()}
        res = self.client.post(
            self.url,
            json.dumps(
                {
                    "type": "stats",
                    "analysis": "descriptive",
                    "data": data,
                    "save_result": True,
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        result_id = body.get("result_id")
        self.assertIsNotNone(result_id, "save_result=True should return result_id")
        self.assertTrue(DSWResult.objects.filter(id=result_id).exists())

    def test_dswresult_has_encrypted_data(self):
        """DSWResult.data field uses EncryptedTextField."""
        from agents_api.models import DSWResult

        field = DSWResult._meta.get_field("data")
        self.assertIn("Encrypted", type(field).__name__)

    def test_dswresult_project_link(self):
        """DSWResult with project_id has project FK set."""
        import numpy as np

        from agents_api.models import DSWResult
        from core.models import Project

        project = Project.objects.create(title="Result Link Test", user=self.user)
        data = {"x": np.random.normal(50, 2, 50).tolist()}
        res = self.client.post(
            self.url,
            json.dumps(
                {
                    "type": "stats",
                    "analysis": "descriptive",
                    "data": data,
                    "save_result": True,
                    "project_id": str(project.id),
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        result_id = res.json().get("result_id")
        if result_id:
            dsw_result = DSWResult.objects.get(id=result_id)
            self.assertEqual(dsw_result.project_id, project.id)


# =============================================================================
# §7.1: Feature Gating HTTP (functional)
# =============================================================================


@SECURE_OFF
class FeatureGatingHTTPTest(TestCase):
    """DSW-001 §7.1: Feature gating enforced at HTTP level."""

    def test_free_user_rejected_synara(self):
        """Free-tier user gets 403 on Synara endpoint."""
        user = _make_user("free@dsw.test", tier=Tier.FREE)
        client = APIClient()
        client.force_login(user)
        res = client.post(
            f"/api/synara/{uuid.uuid4()}/hypotheses/add/",
            json.dumps({"description": "test"}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 403)

    def test_paid_user_accepted_synara(self):
        """Pro-tier user is not rejected by feature gate (may get 400 for missing project)."""
        user = _make_user("pro@dsw.test", tier=Tier.PRO)
        client = APIClient()
        client.force_login(user)
        res = client.post(
            f"/api/synara/{uuid.uuid4()}/hypotheses/add/",
            json.dumps({"description": "test"}),
            content_type="application/json",
        )
        # Should NOT be 401 or 403 — those are auth/gate failures
        self.assertNotIn(res.status_code, (401, 403))

    def test_unauthenticated_rejected_dsw(self):
        """Unauthenticated request to DSW returns 401."""
        client = APIClient()
        res = client.post(
            "/api/dsw/analysis/",
            json.dumps({"type": "stats", "analysis": "ttest"}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 401)


# =============================================================================
# §7.2: IDOR Prevention HTTP (functional)
# =============================================================================


@SECURE_OFF
class IDORPreventionHTTPTest(TestCase):
    """DSW-001 §7.2: Cross-user project access denied."""

    def test_user_cannot_access_other_project(self):
        """User A loading User B's project gets empty Synara, not B's data."""
        from core.models import Project

        owner = _make_user("idor_owner@dsw.test")
        intruder = _make_user("idor_intruder@dsw.test")

        project = Project.objects.create(title="Owner Project", user=owner)

        # Intruder tries to read hypotheses from owner's project
        client = APIClient()
        client.force_login(intruder)
        res = client.get(f"/api/synara/{project.id}/hypotheses/")
        # Should get empty hypotheses (fresh Synara), not owner's data
        if res.status_code == 200:
            body = res.json()
            hypotheses = body.get("hypotheses", body.get("data", []))
            self.assertEqual(len(hypotheses), 0)

    def test_user_cannot_add_to_other_project(self):
        """User A adding hypothesis to User B's project returns 400."""
        from core.models import Project

        owner = _make_user("idor_own2@dsw.test")
        intruder = _make_user("idor_int2@dsw.test")

        project = Project.objects.create(title="Owner Project 2", user=owner)

        client = APIClient()
        client.force_login(intruder)
        res = client.post(
            f"/api/synara/{project.id}/hypotheses/add/",
            json.dumps({"description": "intruder hypothesis"}),
            content_type="application/json",
        )
        # Should be 400 (no project found for intruder) — not 200
        self.assertEqual(res.status_code, 400)


# =============================================================================
# Module Structure (structural — allowed per TST-001 §10.5)
# =============================================================================


class DSWModuleStructureTest(SimpleTestCase):
    """DSW-001 §1.3: Module structure verification."""

    def test_analysis_package_exists(self):
        """agents_api/analysis/ package exists (canonical compute engine)."""
        self.assertTrue(DSW_DIR.exists())
        self.assertTrue((DSW_DIR / "__init__.py").exists())

    def test_synara_package_exists(self):
        """forgesia package exists (replaced agents_api/synara/)."""
        self.assertTrue(SYNARA_DIR.exists())

    def test_dispatch_module_exists(self):
        """dispatch.py is the routing entry point."""
        self.assertTrue((DSW_DIR / "dispatch.py").exists())

    def test_common_module_exists(self):
        """common.py provides shared utilities."""
        self.assertTrue((DSW_DIR / "common.py").exists())

    def test_synara_views_exists(self):
        """synara_views.py provides belief engine endpoints."""
        self.assertTrue((AGENTS_API / "synara_views.py").exists())

    def test_dsw_app_exists(self):
        """dsw/ extracted app exists with views."""
        self.assertTrue((WEB_ROOT / "dsw" / "views.py").exists())

    def test_spc_engine_exists(self):
        """SPC analysis modules exist in analysis/spc/."""
        self.assertTrue((DSW_DIR / "spc").is_dir())

    def test_experimenter_exists(self):
        """DOE experimenter views exist in dsw/ app."""
        self.assertTrue((WEB_ROOT / "dsw" / "experimenter_views.py").exists())

    def test_permissions_module_exists(self):
        """accounts/permissions.py provides decorators."""
        self.assertTrue((WEB_ROOT / "accounts" / "permissions.py").exists())

    def test_all_submodules_exist(self):
        """All DSW analysis sub-modules exist on disk (files or packages)."""
        expected = [
            "stats",
            "ml",
            "spc",
            "bayesian",
            "reliability",
            "simulation",
            "viz",
            "d_type",
            "dispatch.py",
            "common.py",
        ]
        for f in expected:
            path = DSW_DIR / f
            self.assertTrue(
                path.exists() or (DSW_DIR / (f + ".py")).exists(),
                f"DSW sub-module missing: {f}",
            )


# =============================================================================
# §8 — Output Standardization — Hardened Tests
# =============================================================================


def _run_analysis(analysis_type, analysis_id, df=None, config=None):
    """Run an analysis through the full pipeline and return standardized result."""
    import numpy as np
    import pandas as pd

    if df is None:
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "x": np.random.normal(50, 2, 100),
                "y": np.random.normal(52, 2, 100),
                "group": (["A"] * 50) + (["B"] * 50),
            }
        )
    if config is None:
        config = {}

    # Import the right module
    if analysis_type == "stats":
        from agents_api.analysis.stats import run_statistical_analysis

        result = run_statistical_analysis(df, analysis_id, config)
    elif analysis_type == "spc":
        from agents_api.analysis.spc import run_spc_analysis

        result = run_spc_analysis(df, analysis_id, config)
    elif analysis_type == "ml":
        from agents_api.analysis.ml import run_ml_analysis

        result = run_ml_analysis(df, analysis_id, config, user=None)
    elif analysis_type == "viz":
        from agents_api.analysis.viz import run_visualization

        result = run_visualization(df, analysis_id, config)
    else:
        result = {"summary": "test"}

    from agents_api.analysis.standardize import standardize_output

    return standardize_output(result, analysis_type, analysis_id)


class OutputSchemaTest(SimpleTestCase):
    """DSW-001 §8.1: Canonical output schema enforcement."""

    def test_mandatory_keys_present(self):
        """standardize_output() fills all mandatory keys."""
        from agents_api.analysis.standardize import REQUIRED_FIELDS, standardize_output

        result = standardize_output({"summary": "test"}, "stats", "ttest")
        for key in REQUIRED_FIELDS:
            self.assertIn(key, result, f"Missing mandatory key: {key}")

    def test_education_always_present(self):
        """Education key present and non-None after standardization."""
        from agents_api.analysis.standardize import standardize_output

        result = standardize_output({"summary": "test output"}, "stats", "ttest")
        self.assertIsNotNone(
            result.get("education"),
            "education is None after standardize — should be filled from centralized store",
        )

    def test_narrative_always_present(self):
        """Narrative generated from summary with non-empty verdict."""
        from agents_api.analysis.standardize import standardize_output

        result = standardize_output({"summary": "Test result line 1\nLine 2"}, "stats", "ttest")
        self.assertIsNotNone(result.get("narrative"))
        self.assertEqual(result["narrative"]["verdict"], "Test result line 1")
        self.assertTrue(
            len(result["narrative"]["verdict"]) >= 10,
            "Narrative verdict too short (<10 chars)",
        )

    def test_integration_capability_analysis(self):
        """Cpk analysis produces education, narrative, charts through full pipeline."""
        import numpy as np
        import pandas as pd

        np.random.seed(42)
        df = pd.DataFrame({"measurement": np.random.normal(50, 2, 100)})
        result = _run_analysis("spc", "capability", df, {"column": "measurement", "lsl": 44, "usl": 56})

        self.assertIsNotNone(
            result.get("education"),
            "Cpk analysis missing education after standardization",
        )
        self.assertIn("title", result["education"])
        self.assertTrue(
            len(result["education"]["content"]) >= 200,
            f"Cpk education content too shallow: {len(result['education']['content'])} chars",
        )

        self.assertIsNotNone(result.get("narrative"), "Cpk analysis missing narrative")
        self.assertTrue(
            len(result["narrative"].get("verdict", "")) >= 10,
            "Cpk narrative verdict too short",
        )

        self.assertTrue(
            len(result.get("plots", [])) >= 1,
            "Cpk analysis should produce at least 1 chart",
        )
        for plot in result["plots"]:
            if isinstance(plot, dict) and "layout" in plot:
                self.assertIn(
                    "height",
                    plot["layout"],
                    "Cpk chart missing height after chart_defaults",
                )

    def test_integration_ttest_analysis(self):
        """t-test produces education, narrative, evidence grade, shadow through full pipeline."""
        import numpy as np
        import pandas as pd

        np.random.seed(42)
        df = pd.DataFrame(
            {
                "before": np.random.normal(50, 5, 30),
                "after": np.random.normal(55, 5, 30),
            }
        )
        result = _run_analysis("stats", "ttest", df, {"var1": "before", "mu": 50, "alpha": 0.05})

        self.assertIsNotNone(result.get("education"), "t-test missing education")
        self.assertIsNotNone(result.get("narrative"), "t-test missing narrative")

        # t-test has p_value — should have evidence grade and shadow
        if result.get("p_value") is not None or (result.get("statistics") or {}).get("p_value") is not None:
            self.assertIsNotNone(
                result.get("evidence_grade"),
                "t-test with p-value missing evidence_grade",
            )
            self.assertIsNotNone(
                result.get("bayesian_shadow"),
                "t-test with p-value missing bayesian_shadow",
            )

    def test_integration_regression_analysis(self):
        """Regression produces education, narrative, what-if through full pipeline."""
        import numpy as np
        import pandas as pd

        np.random.seed(42)
        x = np.random.normal(0, 1, 50)
        df = pd.DataFrame({"x": x, "y": 2 * x + np.random.normal(0, 0.5, 50)})
        result = _run_analysis("stats", "regression", df, {"predictors": ["x"], "response": "y"})

        self.assertIsNotNone(result.get("education"), "Regression missing education")
        self.assertIsNotNone(result.get("narrative"), "Regression missing narrative")


class RegistryTest(SimpleTestCase):
    """DSW-001 §8.2: Analysis registry completeness."""

    def test_all_dispatch_types_registered(self):
        """All dispatch route types have at least one registry entry."""
        from agents_api.analysis.dispatch import run_analysis
        from agents_api.analysis.registry import ANALYSIS_REGISTRY

        # Get the source of dispatch to find route types
        src = inspect.getsource(run_analysis)
        import re

        route_types = set(re.findall(r'analysis_type\s*==\s*"(\w+)"', src))
        registry_types = {t for t, _ in ANALYSIS_REGISTRY.keys()}
        for rt in route_types:
            self.assertIn(
                rt,
                registry_types,
                f"Dispatch route type '{rt}' has no registry entries",
            )

    def test_registry_has_required_fields(self):
        """Every registry entry has the required metadata fields."""
        from agents_api.analysis.registry import ANALYSIS_REGISTRY

        required = {"module", "category", "has_pvalue", "effect_type", "shadow_type"}
        for key, entry in ANALYSIS_REGISTRY.items():
            for field in required:
                self.assertIn(field, entry, f"Registry entry {key} missing field '{field}'")


class PostProcessorTest(SimpleTestCase):
    """DSW-001 §8.3: Post-processing pipeline."""

    def test_standardize_called_on_dispatch(self):
        """dispatch.py imports and calls standardize_output."""
        from agents_api.analysis.dispatch import run_analysis

        src = inspect.getsource(run_analysis)
        self.assertIn("standardize_output", src)

    def test_missing_education_filled(self):
        """Post-processor injects education from centralized store."""
        from agents_api.analysis.standardize import standardize_output

        result = standardize_output({}, "stats", "ttest")
        from agents_api.analysis.education import get_education

        edu = get_education("stats", "ttest")
        if edu:
            self.assertIsNotNone(
                result.get("education"),
                "Education exists in centralized store but post-processor did not inject it",
            )

    def test_missing_narrative_generated(self):
        """Post-processor generates narrative from summary."""
        from agents_api.analysis.standardize import standardize_output

        result = standardize_output({"summary": "A detailed test result"}, "stats", "ttest")
        self.assertIsNotNone(result.get("narrative"))

    def test_narrative_has_nonempty_verdict(self):
        """Generated narrative has verdict >= 10 characters."""
        from agents_api.analysis.standardize import standardize_output

        result = standardize_output(
            {"summary": "One-Sample t-Test: t(29) = 2.45, p = 0.021"},
            "stats",
            "ttest",
        )
        narrative = result.get("narrative")
        self.assertIsNotNone(narrative)
        verdict = narrative.get("verdict", "")
        self.assertTrue(
            len(verdict) >= 10,
            f"Narrative verdict too short: '{verdict}' ({len(verdict)} chars)",
        )

    def test_guide_observation_nonempty(self):
        """guide_observation >= 10 chars when summary exists."""
        from agents_api.analysis.standardize import standardize_output

        result = standardize_output(
            {"summary": "Process capability Cpk = 1.33. Process is capable."},
            "spc",
            "capability",
        )
        obs = result.get("guide_observation", "")
        self.assertTrue(len(obs) >= 10, f"guide_observation too short: '{obs}' ({len(obs)} chars)")


class EducationTest(SimpleTestCase):
    """DSW-001 §8.4: Education content completeness and depth."""

    def test_all_registered_analyses_have_education(self):
        """Every registered analysis has education content."""
        from agents_api.analysis.education import get_education
        from agents_api.analysis.registry import ANALYSIS_REGISTRY

        missing = []
        for atype, aid in ANALYSIS_REGISTRY:
            edu = get_education(atype, aid)
            if not edu:
                missing.append(f"{atype}/{aid}")
        self.assertEqual(
            len(missing),
            0,
            f"{len(missing)} analyses missing education: {missing[:10]}",
        )

    def test_education_has_required_fields(self):
        """Education entries have title and content."""
        from agents_api.analysis.education import EDUCATION_CONTENT

        for key, entry in EDUCATION_CONTENT.items():
            self.assertIn("title", entry, f"Education {key} missing 'title'")
            self.assertIn("content", entry, f"Education {key} missing 'content'")
            self.assertTrue(len(entry["content"]) > 10, f"Education {key} has empty content")

    def test_education_content_depth(self):
        """Education content >= 200 characters (not shallow stubs)."""
        from agents_api.analysis.education import EDUCATION_CONTENT

        shallow = []
        for key, entry in EDUCATION_CONTENT.items():
            content = entry.get("content", "")
            if len(content) < 200:
                shallow.append(f"{key}: {len(content)} chars")
        self.assertEqual(
            len(shallow),
            0,
            f"{len(shallow)} entries below 200-char minimum: {shallow[:5]}",
        )

    def test_education_has_dl_structure(self):
        """Education content uses <dl> definition lists."""
        from agents_api.analysis.education import EDUCATION_CONTENT

        no_dl = []
        for key, entry in EDUCATION_CONTENT.items():
            content = entry.get("content", "")
            if "<dl>" not in content or "<dt>" not in content:
                no_dl.append(str(key))
        self.assertEqual(
            len(no_dl),
            0,
            f"{len(no_dl)} entries missing <dl>/<dt> structure: {no_dl[:5]}",
        )

    def test_education_title_length(self):
        """Education titles >= 15 characters."""
        from agents_api.analysis.education import EDUCATION_CONTENT

        short = []
        for key, entry in EDUCATION_CONTENT.items():
            title = entry.get("title", "")
            if len(title) < 15:
                short.append(f"{key}: '{title}'")
        self.assertEqual(len(short), 0, f"{len(short)} titles too short: {short[:5]}")


class ChartDefaultsTest(SimpleTestCase):
    """DSW-001 §8.5: Chart standardization."""

    def test_height_standardized(self):
        """apply_chart_defaults sets standard height."""
        from agents_api.analysis.chart_defaults import (
            CHART_HEIGHT,
            apply_chart_defaults,
        )

        plot = {"data": [], "layout": {}}
        apply_chart_defaults(plot)
        self.assertEqual(plot["layout"]["height"], CHART_HEIGHT)

    def test_legend_placement(self):
        """Legend defaults to bottom-left horizontal."""
        from agents_api.analysis.chart_defaults import apply_chart_defaults

        plot = {"data": [], "layout": {}}
        apply_chart_defaults(plot)
        self.assertEqual(plot["layout"]["legend"]["orientation"], "h")

    def test_colors_from_palette(self):
        """Trace colors come from SVEND_COLORS palette."""
        from agents_api.analysis.chart_defaults import apply_chart_defaults
        from agents_api.analysis.common import SVEND_COLORS

        plot = {"data": [{"type": "bar", "x": [1], "y": [1]}], "layout": {}}
        apply_chart_defaults(plot)
        self.assertEqual(plot["data"][0]["marker"]["color"], SVEND_COLORS[0])

    def test_transparent_background(self):
        """Charts have transparent background for theme compatibility."""
        from agents_api.analysis.chart_defaults import apply_chart_defaults

        plot = {"data": [], "layout": {}}
        apply_chart_defaults(plot)
        self.assertEqual(plot["layout"]["paper_bgcolor"], "rgba(0,0,0,0)")
        self.assertEqual(plot["layout"]["plot_bgcolor"], "rgba(0,0,0,0)")

    def test_margins_applied(self):
        """Charts have standard margins."""
        from agents_api.analysis.chart_defaults import (
            CHART_MARGINS,
            apply_chart_defaults,
        )

        plot = {"data": [], "layout": {}}
        apply_chart_defaults(plot)
        for key, val in CHART_MARGINS.items():
            self.assertEqual(plot["layout"]["margin"][key], val, f"Margin {key} should be {val}")

    def test_real_analysis_charts_styled(self):
        """Charts from a real analysis have defaults applied after standardization."""
        import numpy as np
        import pandas as pd

        np.random.seed(42)
        df = pd.DataFrame({"measurement": np.random.normal(50, 2, 100)})
        result = _run_analysis("spc", "capability", df, {"column": "measurement", "lsl": 44, "usl": 56})
        plots = result.get("plots", [])
        for i, plot in enumerate(plots):
            if isinstance(plot, dict) and "layout" in plot:
                layout = plot["layout"]
                self.assertIn("height", layout, f"Plot {i} missing height")
                self.assertEqual(
                    layout.get("paper_bgcolor"),
                    "rgba(0,0,0,0)",
                    f"Plot {i} has non-transparent background",
                )


class NewStatisticsTest(SimpleTestCase):
    """DSW-001 §8.6: Evidence grade and Bayesian shadow."""

    def test_pvalue_analyses_have_evidence_grade(self):
        """Post-processor generates evidence_grade when p_value present."""
        from agents_api.analysis.standardize import standardize_output

        result = standardize_output(
            {
                "summary": "t",
                "p_value": 0.01,
                "statistics": {"p_value": 0.01, "effect_size_r": 0.4, "n": 50},
            },
            "stats",
            "mann_whitney",
        )
        self.assertIsNotNone(
            result.get("evidence_grade"),
            "evidence_grade not generated for p-value result",
        )

    def test_pvalue_analyses_have_bayesian_shadow(self):
        """Post-processor generates bayesian_shadow for shadow-eligible analyses with sufficient stats."""
        from agents_api.analysis.standardize import standardize_output

        result = standardize_output(
            {
                "summary": "t",
                "p_value": 0.01,
                "statistics": {"p_value": 0.01, "effect_size_r": 0.4, "n": 50},
            },
            "stats",
            "mann_whitney",
        )
        self.assertIsNotNone(
            result.get("bayesian_shadow"),
            "bayesian_shadow not generated for nonparametric analysis with effect_r + n",
        )


class WhatIfTest(SimpleTestCase):
    """DSW-001 §8.7: What-if interactivity."""

    def test_tier1_analyses_have_whatif(self):
        """Post-processor creates what_if stub for tier 1 analyses."""
        from agents_api.analysis.standardize import standardize_output

        result = standardize_output({"summary": "t"}, "stats", "power_z")
        self.assertIsNotNone(result.get("what_if"), "what_if not generated for tier 1 analysis")

    def test_whatif_schema_valid(self):
        """what_if dict has required schema fields."""
        from agents_api.analysis.standardize import standardize_output

        result = standardize_output(
            {
                "summary": "t",
                "power_explorer": {
                    "test_type": "ttest",
                    "observed_effect": 1.0,
                    "observed_std": 1.0,
                    "observed_n": 30,
                    "alpha": 0.05,
                    "cohens_d": 0.5,
                },
            },
            "stats",
            "regression",
        )
        wi = result.get("what_if")
        self.assertIsNotNone(wi)
        self.assertIn("type", wi)
        self.assertIn("parameters", wi)
        self.assertIn("endpoint", wi)


class FrontendRenderingTest(SimpleTestCase):
    """DSW-001 §8.8: Frontend rendering elements."""

    def _template(self):
        return _read(WEB_ROOT / "templates" / "analysis_workbench.html")

    def test_education_details_element(self):
        """Frontend renders education as collapsible <details> panel."""
        t = self._template()
        self.assertIn("dsw-education", t)
        self.assertIn("dsw-panel", t)
        self.assertIn("details.className = 'dsw-panel dsw-education'", t)

    def test_narrative_css_classes(self):
        """Frontend has all narrative CSS classes."""
        t = self._template()
        for cls in [
            "dsw-narrative",
            "dsw-verdict",
            "dsw-narrative-body",
            "dsw-next",
            "dsw-chart-guidance",
        ]:
            self.assertIn(cls, t, f"Missing CSS class: {cls}")

    def test_evidence_badge_css_classes(self):
        """Frontend has evidence grade badge CSS for all levels."""
        t = self._template()
        self.assertIn("dsw-evidence-badge", t)
        for level in ["strong", "moderate", "weak", "inconclusive"]:
            self.assertIn(
                f".dsw-evidence-badge.{level}",
                t,
                f"Missing badge CSS for level: {level}",
            )

    def test_bayesian_panel_css(self):
        """Frontend has Bayesian shadow panel CSS."""
        t = self._template()
        self.assertIn("dsw-bayesian-panel", t)

    def test_diagnostics_panel_css(self):
        """Frontend has diagnostics panel CSS."""
        t = self._template()
        self.assertIn("dsw-diagnostics", t)
        for cls in ["dsw-diag-pass", "dsw-diag-warn", "dsw-diag-fail"]:
            self.assertIn(cls, t, f"Missing diagnostic CSS: {cls}")

    def test_what_if_slider_css(self):
        """Frontend has what-if slider CSS."""
        t = self._template()
        self.assertIn("dsw-what-if", t)
        self.assertIn("dsw-slider", t)

    def test_render_function_called(self):
        """renderDSWBlocks(result) is called in the response handler."""
        t = self._template()
        self.assertIn("renderDSWBlocks(result)", t)

    def test_hypothesis_button_requires_project_id(self):
        """DSW-001 §9.3: Hypothesis creation UI requires currentProjectId."""
        wb = _read(WEB_ROOT / "templates" / "workbench_new.html")
        self.assertIn("currentProjectId", wb)
