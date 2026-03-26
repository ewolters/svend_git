"""
Tests for Layer 1 → investigation bridge integration — CANON-002 §12.

All tests exercise real behavior per TST-001 §10.6.
Verifies that DOE, forecast endpoints with investigation_id call
connect_tool() and update the Synara graph.

<!-- test: agents_api.tests.test_investigation_layer1.DOEDesignBridgeTest -->
<!-- test: agents_api.tests.test_investigation_layer1.DOEResultsBridgeTest -->
<!-- test: agents_api.tests.test_investigation_layer1.ForecastBridgeTest -->
<!-- test: agents_api.tests.test_investigation_layer1.NoBridgeWithoutIdTest -->
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
    user = User.objects.create_user(
        username=username, email=email, password="testpass123"
    )
    user.tier = tier
    user.save(update_fields=["tier"])
    return user


def _make_active_investigation(user):
    """Create an active investigation with a hypothesis."""
    inv = Investigation.objects.create(
        title="Layer 1 Bridge Test",
        description="Testing Layer 1 integration",
        owner=user,
        status="active",
    )
    tool = MeasurementSystem.objects.create(
        name="Layer 1 Test Gage", system_type="variable", owner=user
    )
    spec = HypothesisSpec(description="Layer 1 test hypothesis", prior=0.5)
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
class DOEDesignBridgeTest(TestCase):
    """CANON-002 §12 — design_experiment() with investigation_id updates graph."""

    def setUp(self):
        self.user = _make_user("doe-design@test.com")
        self.client = _authed_client(self.user)
        self.inv = _make_active_investigation(self.user)

    def test_doe_design_creates_tool_link(self):
        """DOE design with investigation_id creates InvestigationToolLink."""
        resp = self.client.post(
            "/api/experimenter/design/",
            {
                "design_type": "full_factorial",
                "factors": [
                    {"name": "Temperature", "levels": [100, 200]},
                    {"name": "Pressure", "levels": [50, 100]},
                ],
                "investigation_id": str(self.inv.id),
            },
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(
            InvestigationToolLink.objects.filter(
                investigation=self.inv, tool_type="doe_design"
            ).exists()
        )


@SECURE_OFF
class DOEResultsBridgeTest(TestCase):
    """CANON-002 §12 — analyze_results() with investigation_id updates graph."""

    def setUp(self):
        self.user = _make_user("doe-results@test.com")
        self.client = _authed_client(self.user)
        self.inv = _make_active_investigation(self.user)

    def test_doe_analysis_updates_graph(self):
        """DOE analysis with investigation_id and significant factors creates evidence."""
        initial = _count_evidence(self.inv)
        resp = self.client.post(
            "/api/experimenter/analyze/",
            {
                "design": {
                    "factors": [
                        {"name": "Temperature", "levels": [100, 200]},
                        {"name": "Pressure", "levels": [50, 100]},
                    ],
                    "runs": [
                        {
                            "run_id": 1,
                            "run_order": 1,
                            "coded": {"Temperature": -1, "Pressure": -1},
                            "levels": {"Temperature": 100, "Pressure": 50},
                        },
                        {
                            "run_id": 2,
                            "run_order": 2,
                            "coded": {"Temperature": 1, "Pressure": -1},
                            "levels": {"Temperature": 200, "Pressure": 50},
                        },
                        {
                            "run_id": 3,
                            "run_order": 3,
                            "coded": {"Temperature": -1, "Pressure": 1},
                            "levels": {"Temperature": 100, "Pressure": 100},
                        },
                        {
                            "run_id": 4,
                            "run_order": 4,
                            "coded": {"Temperature": 1, "Pressure": 1},
                            "levels": {"Temperature": 200, "Pressure": 100},
                        },
                        {
                            "run_id": 5,
                            "run_order": 5,
                            "coded": {"Temperature": -1, "Pressure": -1},
                            "levels": {"Temperature": 100, "Pressure": 50},
                        },
                        {
                            "run_id": 6,
                            "run_order": 6,
                            "coded": {"Temperature": 1, "Pressure": -1},
                            "levels": {"Temperature": 200, "Pressure": 50},
                        },
                        {
                            "run_id": 7,
                            "run_order": 7,
                            "coded": {"Temperature": -1, "Pressure": 1},
                            "levels": {"Temperature": 100, "Pressure": 100},
                        },
                        {
                            "run_id": 8,
                            "run_order": 8,
                            "coded": {"Temperature": 1, "Pressure": 1},
                            "levels": {"Temperature": 200, "Pressure": 100},
                        },
                    ],
                },
                "results": [
                    {"run_id": 1, "response": 85},
                    {"run_id": 2, "response": 92},
                    {"run_id": 3, "response": 88},
                    {"run_id": 4, "response": 95},
                    {"run_id": 5, "response": 84},
                    {"run_id": 6, "response": 93},
                    {"run_id": 7, "response": 87},
                    {"run_id": 8, "response": 96},
                ],
                "response_name": "Yield",
                "investigation_id": str(self.inv.id),
            },
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertGreater(_count_evidence(self.inv), initial)


@SECURE_OFF
class ForecastBridgeTest(TestCase):
    """CANON-002 §12 — forecast() with investigation_id updates graph."""

    def setUp(self):
        self.user = _make_user("forecast-bridge@test.com")
        self.client = _authed_client(self.user)
        self.inv = _make_active_investigation(self.user)

    def test_forecast_updates_graph(self):
        """Forecast with investigation_id creates evidence in Synara graph."""
        initial = _count_evidence(self.inv)
        # Use custom data to avoid yfinance dependency
        resp = self.client.post(
            "/api/forecast/",
            {
                "data": [
                    100,
                    102,
                    101,
                    103,
                    105,
                    104,
                    106,
                    108,
                    107,
                    109,
                    110,
                    112,
                    111,
                    113,
                    115,
                    114,
                    116,
                    118,
                    117,
                    119,
                    120,
                    122,
                    121,
                    123,
                    125,
                    124,
                    126,
                    128,
                    127,
                    129,
                ],
                "days": 5,
                "method": "sma",
                "investigation_id": str(self.inv.id),
            },
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertGreater(_count_evidence(self.inv), initial)

    def test_forecast_creates_tool_link(self):
        """Forecast with investigation_id creates InvestigationToolLink."""
        resp = self.client.post(
            "/api/forecast/",
            {
                "data": [
                    100,
                    102,
                    101,
                    103,
                    105,
                    104,
                    106,
                    108,
                    107,
                    109,
                    110,
                    112,
                    111,
                    113,
                    115,
                    114,
                    116,
                    118,
                    117,
                    119,
                    120,
                    122,
                    121,
                    123,
                    125,
                    124,
                    126,
                    128,
                    127,
                    129,
                ],
                "days": 5,
                "method": "sma",
                "investigation_id": str(self.inv.id),
            },
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(
            InvestigationToolLink.objects.filter(
                investigation=self.inv, tool_type="forecast"
            ).exists()
        )


@SECURE_OFF
class NoBridgeWithoutIdTest(TestCase):
    """CANON-002 §12 — Layer 1 tools without investigation_id do NOT touch bridge."""

    def setUp(self):
        self.user = _make_user("l1-nobridge@test.com")
        self.client = _authed_client(self.user)

    def test_forecast_no_investigation(self):
        """Forecast without investigation_id creates no tool links."""
        resp = self.client.post(
            "/api/forecast/",
            {
                "data": [
                    100,
                    102,
                    101,
                    103,
                    105,
                    104,
                    106,
                    108,
                    107,
                    109,
                    110,
                    112,
                    111,
                    113,
                    115,
                    114,
                    116,
                    118,
                    117,
                    119,
                    120,
                    122,
                    121,
                    123,
                    125,
                    124,
                    126,
                    128,
                    127,
                    129,
                ],
                "days": 5,
                "method": "sma",
            },
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(InvestigationToolLink.objects.count(), 0)

    def test_doe_design_no_investigation(self):
        """DOE design without investigation_id creates no tool links."""
        resp = self.client.post(
            "/api/experimenter/design/",
            {
                "design_type": "full_factorial",
                "factors": [
                    {"name": "Temperature", "levels": [100, 200]},
                    {"name": "Pressure", "levels": [50, 100]},
                ],
            },
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(InvestigationToolLink.objects.count(), 0)
