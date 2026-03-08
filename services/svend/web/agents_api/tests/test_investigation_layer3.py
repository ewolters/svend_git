"""
Tests for Layer 3 → investigation bridge integration — CANON-002 §12.

All tests exercise real behavior per TST-001 §10.6.
Verifies that NCR and CAPA endpoints with investigation_id call
connect_tool() and update the Synara graph.

<!-- test: agents_api.tests.test_investigation_layer3.NCRBridgeTest -->
<!-- test: agents_api.tests.test_investigation_layer3.CAPABridgeTest -->
<!-- test: agents_api.tests.test_investigation_layer3.NoBridgeWithoutIdTest -->
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
    inv = Investigation.objects.create(
        title="Layer 3 Bridge Test", description="Testing Layer 3 integration", owner=user, status="active"
    )
    tool = MeasurementSystem.objects.create(name="Layer 3 Test Gage", system_type="variable", owner=user)
    spec = HypothesisSpec(description="Layer 3 test hypothesis", prior=0.5)
    connect_tool(investigation_id=str(inv.id), tool_output=tool, tool_type="rca", user=user, spec=spec)
    return inv


def _authed_client(user):
    client = Client()
    client.force_login(user)
    return client


def _count_hypotheses(inv):
    inv.refresh_from_db()
    synara = load_synara(inv)
    graph = synara.to_dict().get("graph", {})
    return len(graph.get("hypotheses", {}))


def _create_ncr(client):
    """Create an NCR via the API and return its ID."""
    resp = client.post(
        "/api/iso/ncrs/",
        {
            "title": "Test NCR for bridge",
            "description": "Bridge test nonconformance",
            "category": "process",
            "severity": "minor",
        },
        content_type="application/json",
    )
    return resp.json().get("id") or resp.json().get("ncr", {}).get("id")


def _create_capa(client):
    """Create a CAPA via the API and return its ID."""
    resp = client.post(
        "/api/capa/",
        {
            "title": "Test CAPA for bridge",
            "description": "Bridge test corrective action",
            "type": "corrective",
        },
        content_type="application/json",
    )
    return resp.json().get("id") or resp.json().get("capa", {}).get("id")


@SECURE_OFF
class NCRBridgeTest(TestCase):
    """CANON-002 §12 — NCR update with investigation_id updates graph."""

    def setUp(self):
        self.user = _make_user("ncr-bridge@test.com")
        self.client = _authed_client(self.user)
        self.inv = _make_active_investigation(self.user)
        self.ncr_id = _create_ncr(self.client)

    def test_ncr_root_cause_updates_graph(self):
        """NCR update with root_cause + investigation_id creates hypothesis."""
        initial = _count_hypotheses(self.inv)
        resp = self.client.put(
            f"/api/iso/ncrs/{self.ncr_id}/",
            {
                "root_cause": "Weld parameter drift causing incomplete fusion",
                "investigation_id": str(self.inv.id),
            },
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertGreater(_count_hypotheses(self.inv), initial)

    def test_ncr_creates_tool_link(self):
        """NCR update with investigation_id creates InvestigationToolLink."""
        resp = self.client.put(
            f"/api/iso/ncrs/{self.ncr_id}/",
            {
                "root_cause": "Material defect",
                "investigation_id": str(self.inv.id),
            },
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(InvestigationToolLink.objects.filter(investigation=self.inv, tool_type="ncr").exists())


@SECURE_OFF
class CAPABridgeTest(TestCase):
    """CANON-002 §12 — CAPA update with investigation_id updates graph."""

    def setUp(self):
        self.user = _make_user("capa-bridge@test.com")
        self.client = _authed_client(self.user)
        self.inv = _make_active_investigation(self.user)
        self.capa_id = _create_capa(self.client)

    def test_capa_root_cause_updates_graph(self):
        """CAPA update with root_cause + investigation_id creates hypothesis."""
        initial = _count_hypotheses(self.inv)
        resp = self.client.put(
            f"/api/capa/{self.capa_id}/",
            {
                "root_cause": "Inadequate incoming inspection procedure",
                "investigation_id": str(self.inv.id),
            },
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertGreater(_count_hypotheses(self.inv), initial)

    def test_capa_creates_tool_link(self):
        """CAPA update with investigation_id creates InvestigationToolLink."""
        resp = self.client.put(
            f"/api/capa/{self.capa_id}/",
            {
                "corrective_action": "Revise inspection checklist",
                "investigation_id": str(self.inv.id),
            },
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(InvestigationToolLink.objects.filter(investigation=self.inv, tool_type="capa").exists())


@SECURE_OFF
class NoBridgeWithoutIdTest(TestCase):
    """CANON-002 §12 — Layer 3 tools without investigation_id do NOT touch bridge."""

    def setUp(self):
        self.user = _make_user("l3-nobridge@test.com")
        self.client = _authed_client(self.user)

    def test_ncr_no_investigation(self):
        """NCR update without investigation_id creates no tool links."""
        ncr_id = _create_ncr(self.client)
        resp = self.client.put(
            f"/api/iso/ncrs/{ncr_id}/",
            {"root_cause": "Some root cause"},
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(InvestigationToolLink.objects.count(), 0)

    def test_capa_no_investigation(self):
        """CAPA update without investigation_id creates no tool links."""
        capa_id = _create_capa(self.client)
        resp = self.client.put(
            f"/api/capa/{capa_id}/",
            {"root_cause": "Some root cause"},
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(InvestigationToolLink.objects.count(), 0)
