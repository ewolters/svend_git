"""
Tests for SPC → investigation bridge integration — CANON-002 §12.

All tests exercise real behavior per TST-001 §10.6.
Verifies that SPC endpoints with investigation_id call connect_tool()
and update the Synara graph. Existing behavior without investigation_id
is preserved (tested in test_spc_views_gaps.py).

<!-- test: agents_api.tests.test_investigation_spc.ControlChartBridgeTest -->
<!-- test: agents_api.tests.test_investigation_spc.CapabilityBridgeTest -->
<!-- test: agents_api.tests.test_investigation_spc.SummaryBridgeTest -->
<!-- test: agents_api.tests.test_investigation_spc.GageRRBridgeTest -->
<!-- test: agents_api.tests.test_investigation_spc.RecommendBridgeTest -->
<!-- test: agents_api.tests.test_investigation_spc.NoBridgeWithoutIdTest -->
"""

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.test.client import Client

from accounts.constants import Tier
from agents_api.investigation_bridge import HypothesisSpec, connect_tool, load_synara
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
        title="SPC Bridge Test",
        description="Testing SPC integration",
        owner=user,
        status="active",
    )
    tool = MeasurementSystem.objects.create(name="SPC Test Gage", system_type="variable", owner=user)
    spec = HypothesisSpec(description="Process drift hypothesis", prior=0.5)
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


@SECURE_OFF
class ControlChartBridgeTest(TestCase):
    """CANON-002 §12 — control_chart() with investigation_id updates graph."""

    def setUp(self):
        self.user = _make_user("spc-chart@test.com")
        self.client = _authed_client(self.user)
        self.inv = _make_active_investigation(self.user)

    def test_control_chart_updates_graph(self):
        """Control chart with investigation_id creates evidence in Synara graph."""
        initial = _count_evidence(self.inv)
        resp = self.client.post(
            "/api/spc/chart/",
            {
                "chart_type": "I-MR",
                "data": [10, 12, 11, 13, 12, 15, 14, 11],
                "investigation_id": str(self.inv.id),
            },
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertGreater(_count_evidence(self.inv), initial)

    def test_control_chart_creates_tool_link(self):
        """Control chart with investigation_id creates InvestigationToolLink."""
        resp = self.client.post(
            "/api/spc/chart/",
            {
                "chart_type": "I-MR",
                "data": [10, 12, 11, 13, 12, 15, 14, 11],
                "investigation_id": str(self.inv.id),
            },
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(
            InvestigationToolLink.objects.filter(investigation=self.inv, tool_type="spc_control_chart").exists()
        )


@SECURE_OFF
class CapabilityBridgeTest(TestCase):
    """CANON-002 §12 — capability_study() with investigation_id updates graph."""

    def setUp(self):
        self.user = _make_user("spc-cap@test.com")
        self.client = _authed_client(self.user)
        self.inv = _make_active_investigation(self.user)

    def test_capability_updates_graph(self):
        """Capability study with investigation_id creates evidence in Synara graph."""
        initial = _count_evidence(self.inv)
        resp = self.client.post(
            "/api/spc/capability/",
            {
                "data": [10.1, 10.2, 10.0, 9.9, 10.3, 10.1, 10.0, 9.8, 10.2, 10.1],
                "usl": 10.5,
                "lsl": 9.5,
                "investigation_id": str(self.inv.id),
            },
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertGreater(_count_evidence(self.inv), initial)


@SECURE_OFF
class SummaryBridgeTest(TestCase):
    """CANON-002 §12 — statistical_summary() with investigation_id updates graph."""

    def setUp(self):
        self.user = _make_user("spc-sum@test.com")
        self.client = _authed_client(self.user)
        self.inv = _make_active_investigation(self.user)

    def test_summary_updates_graph(self):
        """Statistical summary with investigation_id creates evidence in Synara graph."""
        initial = _count_evidence(self.inv)
        resp = self.client.post(
            "/api/spc/summary/",
            {
                "data": [10, 12, 11, 13, 12, 15, 14, 11, 10, 13],
                "investigation_id": str(self.inv.id),
            },
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertGreater(_count_evidence(self.inv), initial)


@SECURE_OFF
class GageRRBridgeTest(TestCase):
    """CANON-002 §12 — gage_rr() with investigation_id updates graph."""

    def setUp(self):
        self.user = _make_user("spc-grr@test.com")
        self.client = _authed_client(self.user)
        self.inv = _make_active_investigation(self.user)

    def test_gage_rr_updates_graph(self):
        """Gage R&R with investigation_id creates evidence in Synara graph."""
        initial = _count_evidence(self.inv)
        resp = self.client.post(
            "/api/spc/gage-rr/",
            {
                "parts": ["A", "A", "B", "B", "C", "C", "A", "A", "B", "B", "C", "C"],
                "operators": [
                    "Op1",
                    "Op1",
                    "Op1",
                    "Op1",
                    "Op1",
                    "Op1",
                    "Op2",
                    "Op2",
                    "Op2",
                    "Op2",
                    "Op2",
                    "Op2",
                ],
                "measurements": [
                    10.1,
                    10.2,
                    10.5,
                    10.4,
                    10.8,
                    10.9,
                    10.0,
                    10.3,
                    10.6,
                    10.5,
                    10.7,
                    10.8,
                ],
                "tolerance": 1.0,
                "investigation_id": str(self.inv.id),
            },
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertGreater(_count_evidence(self.inv), initial)


@SECURE_OFF
class RecommendBridgeTest(TestCase):
    """CANON-002 §12 — recommend_chart() with investigation_id updates graph."""

    def setUp(self):
        self.user = _make_user("spc-rec@test.com")
        self.client = _authed_client(self.user)
        self.inv = _make_active_investigation(self.user)

    def test_recommend_updates_graph(self):
        """Chart recommendation with investigation_id creates evidence in Synara graph."""
        initial = _count_evidence(self.inv)
        resp = self.client.post(
            "/api/spc/chart/recommend/",
            {
                "data_type": "continuous",
                "subgroup_size": 5,
                "investigation_id": str(self.inv.id),
            },
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertGreater(_count_evidence(self.inv), initial)


@SECURE_OFF
class NoBridgeWithoutIdTest(TestCase):
    """CANON-002 §12 — SPC without investigation_id does NOT touch bridge."""

    def setUp(self):
        self.user = _make_user("spc-nobridge@test.com")
        self.client = _authed_client(self.user)

    def test_control_chart_no_investigation(self):
        """Control chart without investigation_id works normally, no tool links created."""
        resp = self.client.post(
            "/api/spc/chart/",
            {"chart_type": "I-MR", "data": [10, 12, 11, 13, 12, 15, 14, 11]},
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(InvestigationToolLink.objects.count(), 0)

    def test_capability_no_investigation(self):
        """Capability study without investigation_id works normally."""
        resp = self.client.post(
            "/api/spc/capability/",
            {
                "data": [10.1, 10.2, 10.0, 9.9, 10.3, 10.1, 10.0, 9.8, 10.2, 10.1],
                "usl": 10.5,
                "lsl": 9.5,
            },
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(InvestigationToolLink.objects.count(), 0)
