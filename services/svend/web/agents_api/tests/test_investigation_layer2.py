"""
Tests for Layer 2 → investigation bridge integration — CANON-002 §12.

All tests exercise real behavior per TST-001 §10.6.
Verifies that RCA, Ishikawa, C&E Matrix, and FMEA endpoints with
investigation_id call connect_tool() and update the Synara graph.

<!-- test: agents_api.tests.test_investigation_layer2.RCABridgeTest -->
<!-- test: agents_api.tests.test_investigation_layer2.IshikawaBridgeTest -->
<!-- test: agents_api.tests.test_investigation_layer2.CEMatrixBridgeTest -->
<!-- test: agents_api.tests.test_investigation_layer2.FMEABridgeTest -->
<!-- test: agents_api.tests.test_investigation_layer2.NoBridgeWithoutIdTest -->
"""

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.test.client import Client

from accounts.constants import Tier
from agents_api.investigation_bridge import HypothesisSpec, connect_tool, load_synara
from agents_api.models import FMEA, CEMatrix, FMEARow, IshikawaDiagram, RCASession
from core.models import (
    Hypothesis,
    Investigation,
    InvestigationToolLink,
    MeasurementSystem,
    Project,
)

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
    """Create an active investigation with a hypothesis for evidence to support."""
    inv = Investigation.objects.create(
        title="Layer 2 Bridge Test",
        description="Testing Layer 2 integration",
        owner=user,
        status="active",
    )
    tool = MeasurementSystem.objects.create(
        name="Layer 2 Test Gage", system_type="variable", owner=user
    )
    spec = HypothesisSpec(description="Layer 2 test hypothesis", prior=0.5)
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


def _count_hypotheses(inv):
    """Count hypotheses in investigation's Synara graph."""
    inv.refresh_from_db()
    synara = load_synara(inv)
    graph = synara.to_dict().get("graph", {})
    return len(graph.get("hypotheses", {}))


@SECURE_OFF
class RCABridgeTest(TestCase):
    """CANON-002 §12 — RCA update_session() with investigation_id updates graph."""

    def setUp(self):
        self.user = _make_user("rca-bridge@test.com")
        self.client = _authed_client(self.user)
        self.inv = _make_active_investigation(self.user)
        self.session = RCASession.objects.create(
            owner=self.user,
            title="RCA Bridge Test",
            event="Test event for bridge",
        )

    def test_rca_with_root_cause_updates_graph(self):
        """RCA update with root_cause + investigation_id creates hypothesis in Synara graph."""
        initial = _count_hypotheses(self.inv)
        resp = self.client.patch(
            f"/api/rca/sessions/{self.session.id}/update/",
            {
                "root_cause": "Bearing failure due to inadequate lubrication",
                "investigation_id": str(self.inv.id),
            },
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertGreater(_count_hypotheses(self.inv), initial)

    def test_rca_creates_tool_link(self):
        """RCA update with investigation_id creates InvestigationToolLink."""
        resp = self.client.patch(
            f"/api/rca/sessions/{self.session.id}/update/",
            {
                "root_cause": "Operator training gap",
                "investigation_id": str(self.inv.id),
            },
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(
            InvestigationToolLink.objects.filter(
                investigation=self.inv, tool_type="rca"
            ).exists()
        )

    def test_rca_chain_steps_create_hypotheses(self):
        """RCA update with chain steps + investigation_id creates multiple hypotheses."""
        initial = _count_hypotheses(self.inv)
        resp = self.client.patch(
            f"/api/rca/sessions/{self.session.id}/update/",
            {
                "root_cause": "System design flaw",
                "chain": [
                    {"claim": "Bearing overheated", "accepted": True},
                    {"claim": "Lubrication schedule missed", "accepted": True},
                    {"claim": "Work order system down", "accepted": False},
                ],
                "investigation_id": str(self.inv.id),
            },
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        # Root cause + 2 accepted chain steps = 3 new hypotheses
        self.assertGreaterEqual(_count_hypotheses(self.inv) - initial, 3)


@SECURE_OFF
class IshikawaBridgeTest(TestCase):
    """CANON-002 §12 — Ishikawa update_diagram() with investigation_id updates graph."""

    def setUp(self):
        self.user = _make_user("ishikawa-bridge@test.com")
        self.client = _authed_client(self.user)
        self.inv = _make_active_investigation(self.user)
        self.diagram = IshikawaDiagram.objects.create(
            owner=self.user,
            title="Ishikawa Bridge Test",
            effect="Test defect",
            branches=[
                {"category": "Man", "causes": [{"text": "Insufficient training"}]},
                {"category": "Machine", "causes": [{"text": "Worn tooling"}]},
            ],
            status="draft",
        )

    def test_ishikawa_complete_updates_graph(self):
        """Ishikawa set to complete with investigation_id creates hypotheses."""
        initial = _count_hypotheses(self.inv)
        resp = self.client.patch(
            f"/api/ishikawa/sessions/{self.diagram.id}/update/",
            {
                "status": "complete",
                "investigation_id": str(self.inv.id),
            },
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        # 2 causes → 2 new hypotheses
        self.assertGreaterEqual(_count_hypotheses(self.inv) - initial, 2)

    def test_ishikawa_draft_no_bridge(self):
        """Ishikawa staying draft with investigation_id does NOT update graph."""
        initial = _count_hypotheses(self.inv)
        resp = self.client.patch(
            f"/api/ishikawa/sessions/{self.diagram.id}/update/",
            {
                "title": "Updated title",
                "investigation_id": str(self.inv.id),
            },
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(_count_hypotheses(self.inv), initial)


@SECURE_OFF
class CEMatrixBridgeTest(TestCase):
    """CANON-002 §12 — C&E update_matrix() with investigation_id updates graph."""

    def setUp(self):
        self.user = _make_user("ce-bridge@test.com")
        self.client = _authed_client(self.user)
        self.inv = _make_active_investigation(self.user)
        self.matrix = CEMatrix.objects.create(
            owner=self.user,
            title="CE Bridge Test",
            outputs=[{"name": "Defect Rate", "weight": 10}],
            inputs=[
                {"name": "Temperature"},
                {"name": "Pressure"},
                {"name": "Speed"},
            ],
            scores={"0": {"0": 9}, "1": {"0": 7}, "2": {"0": 3}},
            status="draft",
        )

    def test_ce_complete_updates_graph(self):
        """C&E Matrix set to complete with investigation_id creates hypotheses."""
        initial = _count_hypotheses(self.inv)
        resp = self.client.patch(
            f"/api/ce/sessions/{self.matrix.id}/update/",
            {
                "status": "complete",
                "investigation_id": str(self.inv.id),
            },
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertGreater(_count_hypotheses(self.inv), initial)


@SECURE_OFF
class FMEABridgeTest(TestCase):
    """CANON-002 §12 — FMEA link_to_hypothesis() with investigation_id updates graph."""

    def setUp(self):
        self.user = _make_user("fmea-bridge@test.com")
        self.client = _authed_client(self.user)
        self.inv = _make_active_investigation(self.user)
        self.project = Project.objects.create(
            user=self.user,
            title="FMEA Bridge Project",
            project_class="investigation",
        )
        self.hypothesis = Hypothesis.objects.create(
            project=self.project,
            statement="Bearing failure hypothesis",
            prior_probability=0.5,
            current_probability=0.5,
        )
        self.fmea = FMEA.objects.create(
            owner=self.user,
            title="FMEA Bridge Test",
            project=self.project,
        )
        self.row = FMEARow.objects.create(
            fmea=self.fmea,
            process_step="Assembly",
            failure_mode="Bearing misalignment",
            effect="Premature wear",
            cause="Incorrect torque",
            current_controls="Visual inspection",
            severity=8,
            occurrence=5,
            detection=6,
        )

    def test_fmea_link_updates_graph(self):
        """FMEA link_to_hypothesis with investigation_id creates hypothesis in graph."""
        initial = _count_hypotheses(self.inv)
        resp = self.client.post(
            f"/api/fmea/{self.fmea.id}/rows/{self.row.id}/link/",
            {
                "hypothesis_id": str(self.hypothesis.id),
                "investigation_id": str(self.inv.id),
            },
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertGreater(_count_hypotheses(self.inv), initial)

    def test_fmea_creates_tool_link(self):
        """FMEA link with investigation_id creates InvestigationToolLink."""
        resp = self.client.post(
            f"/api/fmea/{self.fmea.id}/rows/{self.row.id}/link/",
            {
                "hypothesis_id": str(self.hypothesis.id),
                "investigation_id": str(self.inv.id),
            },
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(
            InvestigationToolLink.objects.filter(
                investigation=self.inv, tool_type="fmea"
            ).exists()
        )


@SECURE_OFF
class NoBridgeWithoutIdTest(TestCase):
    """CANON-002 §12 — Layer 2 tools without investigation_id do NOT touch bridge."""

    def setUp(self):
        self.user = _make_user("l2-nobridge@test.com")
        self.client = _authed_client(self.user)

    def test_rca_no_investigation(self):
        """RCA update without investigation_id creates no tool links."""
        session = RCASession.objects.create(
            owner=self.user,
            title="No Bridge RCA",
            event="Test event",
        )
        resp = self.client.patch(
            f"/api/rca/sessions/{session.id}/update/",
            {"root_cause": "Test root cause"},
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(InvestigationToolLink.objects.count(), 0)

    def test_ishikawa_no_investigation(self):
        """Ishikawa update without investigation_id creates no tool links."""
        diagram = IshikawaDiagram.objects.create(
            owner=self.user,
            title="No Bridge Ishikawa",
            effect="Test effect",
            branches=[{"category": "Man", "causes": [{"text": "Test"}]}],
            status="draft",
        )
        resp = self.client.patch(
            f"/api/ishikawa/sessions/{diagram.id}/update/",
            {"status": "complete"},
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(InvestigationToolLink.objects.count(), 0)

    def test_ce_no_investigation(self):
        """C&E Matrix update without investigation_id creates no tool links."""
        matrix = CEMatrix.objects.create(
            owner=self.user,
            title="No Bridge CE",
            outputs=[{"name": "Quality", "weight": 10}],
            inputs=[{"name": "Input1"}],
            scores={"0": {"0": 5}},
            status="draft",
        )
        resp = self.client.patch(
            f"/api/ce/sessions/{matrix.id}/update/",
            {"status": "complete"},
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(InvestigationToolLink.objects.count(), 0)
