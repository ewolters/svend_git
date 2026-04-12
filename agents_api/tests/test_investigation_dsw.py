"""
Tests for DSW → investigation bridge integration — CANON-002 §12.

All tests exercise real behavior per TST-001 §10.6.
Verifies that DSW endpoints with investigation_id call connect_tool()
and update the Synara graph. Existing behavior without investigation_id
is preserved.

<!-- test: agents_api.tests.test_investigation_dsw.DSWBridgeStatsTest -->
<!-- test: agents_api.tests.test_investigation_dsw.DSWBridgeMLTest -->
<!-- test: agents_api.tests.test_investigation_dsw.DSWBridgeMetricsTest -->
<!-- test: agents_api.tests.test_investigation_dsw.DSWNoBridgeWithoutIdTest -->
"""

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.test.client import Client

from accounts.constants import Tier
from agents_api.investigation_bridge import HypothesisSpec, connect_tool, load_synara
from agents_api.models import DSWResult
from core.models import Investigation, InvestigationToolLink, MeasurementSystem

User = get_user_model()

SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)


def _make_user(email, tier=Tier.TEAM):
    username = email.split("@")[0]
    user = User.objects.create_user(username=username, email=email, password="testpass123")
    user.tier = tier
    user.save(update_fields=["tier"])
    return user


def _make_active_investigation(user):
    """Create an active investigation with a hypothesis for evidence to support."""
    inv = Investigation.objects.create(
        title="DSW Bridge Test",
        description="Testing DSW integration",
        owner=user,
        status="active",
    )
    tool = MeasurementSystem.objects.create(name="DSW Test Gage", system_type="variable", owner=user)
    spec = HypothesisSpec(description="Statistical hypothesis", prior=0.5)
    connect_tool(
        investigation_id=str(inv.id),
        tool_output=tool,
        tool_type="rca",
        user=user,
        spec=spec,
    )
    return inv


def _authed_client(user):
    client = Client()
    client.force_login(user)
    return client


def _count_evidence(inv):
    """Count evidence nodes in investigation's Synara graph."""
    inv.refresh_from_db()
    synara = load_synara(inv)
    graph = synara.to_dict().get("graph", {})
    return len(graph.get("evidence", []))


def _dsw_payload(analysis_type, analysis_id, config=None, investigation_id=None, data=None):
    """Build a DSW analysis request payload with inline data."""
    payload = {
        "type": analysis_type,
        "analysis": analysis_id,
        "config": config or {},
    }
    if data is not None:
        payload["data"] = data
    if investigation_id:
        payload["investigation_id"] = str(investigation_id)
    return payload


# Default test datasets — column names must match config keys
_TTEST_DATA = {
    "x": [
        10.1,
        10.2,
        10.0,
        9.9,
        10.3,
        10.1,
        10.0,
        9.8,
        10.2,
        10.1,
        10.5,
        10.6,
        10.4,
        10.7,
        10.8,
        10.5,
        10.6,
        10.3,
        10.7,
        10.6,
    ],
}

_ANOVA_DATA = {
    "value": [10, 11, 12, 9, 10, 15, 16, 17, 14, 15, 20, 21, 22, 19, 20],
    "group": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
}

_DESCRIPTIVE_DATA = {
    "x": [
        10.1,
        10.2,
        10.0,
        9.9,
        10.3,
        10.1,
        10.0,
        9.8,
        10.2,
        10.1,
        10.5,
        10.6,
        10.4,
        10.7,
        10.8,
        10.5,
        10.6,
        10.3,
        10.7,
        10.6,
    ],
}


@SECURE_OFF
class DSWBridgeStatsTest(TestCase):
    """CANON-002 §12 — stats analyses with investigation_id update graph."""

    def setUp(self):
        self.user = _make_user("dsw-stats@test.com")
        self.client = _authed_client(self.user)
        self.inv = _make_active_investigation(self.user)

    def test_ttest_updates_graph(self):
        """Two-sample t-test with investigation_id creates evidence in Synara graph."""
        initial = _count_evidence(self.inv)
        resp = self.client.post(
            "/api/dsw/analysis/",
            _dsw_payload(
                "stats",
                "ttest",
                config={"var1": "x", "mu": 10},
                investigation_id=self.inv.id,
                data=_TTEST_DATA,
            ),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertGreater(_count_evidence(self.inv), initial)

    def test_ttest_creates_tool_link(self):
        """Two-sample t-test with investigation_id creates InvestigationToolLink."""
        resp = self.client.post(
            "/api/dsw/analysis/",
            _dsw_payload(
                "stats",
                "ttest",
                config={"var1": "x", "mu": 10},
                investigation_id=self.inv.id,
                data=_TTEST_DATA,
            ),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(InvestigationToolLink.objects.filter(investigation=self.inv, tool_type="dsw").exists())

    def test_ttest_bridge_result_in_response(self):
        """Response includes investigation_bridge when investigation_id provided."""
        resp = self.client.post(
            "/api/dsw/analysis/",
            _dsw_payload(
                "stats",
                "ttest",
                config={"var1": "x", "mu": 10},
                investigation_id=self.inv.id,
                data=_TTEST_DATA,
            ),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("investigation_bridge", body)
        bridge = body["investigation_bridge"]
        self.assertTrue(bridge.get("linked"))
        self.assertTrue(bridge.get("graph_updated"))
        self.assertIn("evidence_weight", bridge)

    def test_anova_updates_graph(self):
        """One-way ANOVA with investigation_id creates evidence."""
        initial = _count_evidence(self.inv)
        resp = self.client.post(
            "/api/dsw/analysis/",
            _dsw_payload(
                "stats",
                "anova",
                config={"response": "value", "factor": "group"},
                investigation_id=self.inv.id,
                data=_ANOVA_DATA,
            ),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertGreater(_count_evidence(self.inv), initial)

    def test_descriptive_updates_graph(self):
        """Descriptive stats with investigation_id creates evidence."""
        initial = _count_evidence(self.inv)
        resp = self.client.post(
            "/api/dsw/analysis/",
            _dsw_payload(
                "stats",
                "descriptive",
                config={"vars": ["x"]},
                investigation_id=self.inv.id,
                data=_DESCRIPTIVE_DATA,
            ),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertGreater(_count_evidence(self.inv), initial)


@SECURE_OFF
class DSWBridgeMLTest(TestCase):
    """CANON-002 §12 — ML analyses with investigation_id update graph."""

    def setUp(self):
        self.user = _make_user("dsw-ml@test.com")
        self.client = _authed_client(self.user)
        self.inv = _make_active_investigation(self.user)

    def test_clustering_updates_graph(self):
        """Clustering with investigation_id creates evidence in Synara graph."""
        initial = _count_evidence(self.inv)
        data = {
            "x": [1, 2, 3, 10, 11, 12, 20, 21, 22],
            "y": [1, 2, 1, 10, 11, 10, 20, 21, 20],
        }
        resp = self.client.post(
            "/api/dsw/analysis/",
            _dsw_payload(
                "ml",
                "clustering",
                config={"features": ["x", "y"], "n_clusters": 3},
                investigation_id=self.inv.id,
                data=data,
            ),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertGreater(_count_evidence(self.inv), initial)


@SECURE_OFF
class DSWBridgeMetricsTest(TestCase):
    """CANON-002 §12 — verify evidence weight and DSWResult creation."""

    def setUp(self):
        self.user = _make_user("dsw-metrics@test.com")
        self.client = _authed_client(self.user)
        self.inv = _make_active_investigation(self.user)

    def test_bridge_creates_measurement_system(self):
        """Bridge creates a MeasurementSystem record for the tool link."""
        resp = self.client.post(
            "/api/dsw/analysis/",
            _dsw_payload(
                "stats",
                "ttest",
                config={"var1": "x", "mu": 10},
                investigation_id=self.inv.id,
                data=_TTEST_DATA,
            ),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        # Bridge uses MeasurementSystem as the tool_output for InvestigationToolLink
        self.assertTrue(MeasurementSystem.objects.filter(name="DSW stats", owner=self.user).exists())

    def test_evidence_weight_is_positive(self):
        """Evidence weight in bridge result is a positive float."""
        resp = self.client.post(
            "/api/dsw/analysis/",
            _dsw_payload(
                "stats",
                "ttest",
                config={"var1": "x", "mu": 10},
                investigation_id=self.inv.id,
                data=_TTEST_DATA,
            ),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        bridge = resp.json().get("investigation_bridge", {})
        self.assertGreater(bridge.get("evidence_weight", 0), 0)

    def test_posteriors_returned(self):
        """Bridge result includes posteriors dict for hypothesis updates."""
        resp = self.client.post(
            "/api/dsw/analysis/",
            _dsw_payload(
                "stats",
                "ttest",
                config={"var1": "x", "mu": 10},
                investigation_id=self.inv.id,
                data=_TTEST_DATA,
            ),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        bridge = resp.json().get("investigation_bridge", {})
        self.assertIn("posteriors", bridge)
        self.assertIsInstance(bridge["posteriors"], dict)

    def test_study_quality_factors_forwarded(self):
        """Config blinding/pre_registration are forwarded as study quality factors."""
        resp = self.client.post(
            "/api/dsw/analysis/",
            _dsw_payload(
                "stats",
                "ttest",
                config={
                    "var1": "x",
                    "mu": 10,
                    "blinding": True,
                    "pre_registration": True,
                },
                investigation_id=self.inv.id,
                data=_TTEST_DATA,
            ),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        bridge = resp.json().get("investigation_bridge", {})
        # With quality factors, weight should still be valid
        self.assertGreater(bridge.get("evidence_weight", 0), 0)


@SECURE_OFF
class DSWNoBridgeWithoutIdTest(TestCase):
    """CANON-002 §12 — DSW without investigation_id does NOT touch bridge."""

    def setUp(self):
        self.user = _make_user("dsw-nobridge@test.com")
        self.client = _authed_client(self.user)

    def test_ttest_no_investigation(self):
        """T-test without investigation_id works normally, no tool links created."""
        resp = self.client.post(
            "/api/dsw/analysis/",
            _dsw_payload(
                "stats",
                "ttest",
                config={"var1": "x", "mu": 10},
                data=_TTEST_DATA,
            ),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(InvestigationToolLink.objects.count(), 0)
        self.assertNotIn("investigation_bridge", resp.json())

    def test_no_dsw_result_created_without_bridge(self):
        """Without investigation_id and save_result, no DSWResult is created."""
        resp = self.client.post(
            "/api/dsw/analysis/",
            _dsw_payload(
                "stats",
                "ttest",
                config={"var1": "x", "mu": 10},
                data=_TTEST_DATA,
            ),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(DSWResult.objects.count(), 0)
