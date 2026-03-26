"""
Tests for agents_api.investigation_bridge — CANON-002 §12.3.

All tests exercise real behavior per TST-001 §10.6.
Tests use DB + real Synara engine to verify connect_tool(),
permission checks, auto-transitions, and idempotent linking.

<!-- test: agents_api.tests.test_investigation_bridge.ConnectToolInformationTest -->
<!-- test: agents_api.tests.test_investigation_bridge.ConnectToolInferenceTest -->
<!-- test: agents_api.tests.test_investigation_bridge.ConnectToolIntentTest -->
<!-- test: agents_api.tests.test_investigation_bridge.ConnectToolReportTest -->
<!-- test: agents_api.tests.test_investigation_bridge.AutoTransitionTest -->
<!-- test: agents_api.tests.test_investigation_bridge.PermissionTest -->
<!-- test: agents_api.tests.test_investigation_bridge.IdempotentLinkTest -->
<!-- test: agents_api.tests.test_investigation_bridge.LoadSaveSynaraTest -->
"""

from django.test import TestCase

from agents_api.investigation_bridge import (
    HypothesisSpec,
    InferenceSpec,
    IntentSpec,
    connect_tool,
    get_investigation,
    load_synara,
    save_synara,
)
from core.models import (
    Investigation,
    InvestigationMembership,
    InvestigationToolLink,
    MeasurementSystem,
)


def _make_user(email="test@example.com"):
    from django.contrib.auth import get_user_model

    User = get_user_model()
    return User.objects.create_user(username=email, email=email, password="testpass123")


def _make_investigation(user, **kwargs):
    defaults = {
        "title": "Bridge test investigation",
        "description": "Testing bridge module",
        "owner": user,
    }
    defaults.update(kwargs)
    return Investigation.objects.create(**defaults)


def _make_tool_output(user):
    """Create a MeasurementSystem as a generic tool output for linking."""
    return MeasurementSystem.objects.create(
        name="Test Gage", system_type="variable", owner=user
    )


class ConnectToolInformationTest(TestCase):
    """CANON-002 §12.3 — information function creates hypotheses."""

    def setUp(self):
        self.user = _make_user()
        self.inv = _make_investigation(self.user, status="active")
        self.tool_output = _make_tool_output(self.user)

    def test_single_hypothesis(self):
        """Single HypothesisSpec creates one hypothesis."""
        spec = HypothesisSpec(description="Machine vibration causes defects", prior=0.6)
        result = connect_tool(
            investigation_id=str(self.inv.id),
            tool_output=self.tool_output,
            tool_type="rca",
            user=self.user,
            spec=spec,
        )
        self.assertTrue(result["linked"])
        self.assertTrue(result["graph_updated"])
        self.assertEqual(result["hypotheses_added"], 1)

    def test_multiple_hypotheses(self):
        """List of HypothesisSpec creates multiple hypotheses."""
        specs = [
            HypothesisSpec(description="Cause A", prior=0.5),
            HypothesisSpec(description="Cause B", prior=0.3),
            HypothesisSpec(description="Cause C", prior=0.7),
        ]
        result = connect_tool(
            investigation_id=str(self.inv.id),
            tool_output=self.tool_output,
            tool_type="ishikawa",
            user=self.user,
            spec=specs,
        )
        self.assertEqual(result["hypotheses_added"], 3)

    def test_hypothesis_persisted_to_synara(self):
        """Created hypothesis is saved in investigation.synara_state."""
        spec = HypothesisSpec(description="Root cause hypothesis")
        connect_tool(
            investigation_id=str(self.inv.id),
            tool_output=self.tool_output,
            tool_type="rca",
            user=self.user,
            spec=spec,
        )
        self.inv.refresh_from_db()
        synara = load_synara(self.inv)
        graph = synara.to_dict().get("graph", {})
        self.assertTrue(len(graph.get("hypotheses", {})) > 0)


class ConnectToolInferenceTest(TestCase):
    """CANON-002 §12.3 — inference function computes evidence weight and creates evidence."""

    def setUp(self):
        self.user = _make_user("inference@test.com")
        self.inv = _make_investigation(self.user, status="active")
        self.tool_output = _make_tool_output(self.user)

    def test_inference_computes_weight(self):
        """InferenceSpec produces evidence_weight in result."""
        # First add a hypothesis so inference has something to support
        h_spec = HypothesisSpec(description="Hypothesis for inference test")
        connect_tool(
            investigation_id=str(self.inv.id),
            tool_output=self.tool_output,
            tool_type="rca",
            user=self.user,
            spec=h_spec,
        )
        self.inv.refresh_from_db()
        synara = load_synara(self.inv)
        h_ids = list(synara.to_dict()["graph"]["hypotheses"].keys())

        # Now connect SPC inference
        spc_output = MeasurementSystem.objects.create(
            name="SPC Output", system_type="variable", owner=self.user
        )
        spec = InferenceSpec(
            event_description="SPC X-bar: 3 points above UCL",
            supports=h_ids[:1],
            sample_size=30,
        )
        result = connect_tool(
            investigation_id=str(self.inv.id),
            tool_output=spc_output,
            tool_type="spc",
            user=self.user,
            spec=spec,
        )
        self.assertTrue(result["graph_updated"])
        self.assertIn("evidence_weight", result)
        self.assertGreater(result["evidence_weight"], 0)
        self.assertIn("posteriors", result)

    def test_inference_returns_posteriors(self):
        """InferenceSpec returns posteriors dict."""
        spec = InferenceSpec(event_description="Statistical test result p=0.02")
        result = connect_tool(
            investigation_id=str(self.inv.id),
            tool_output=self.tool_output,
            tool_type="dsw",
            user=self.user,
            spec=spec,
        )
        self.assertIn("posteriors", result)
        self.assertIsInstance(result["posteriors"], dict)


class ConnectToolIntentTest(TestCase):
    """CANON-002 §12.3 — intent function annotates hypotheses."""

    def setUp(self):
        self.user = _make_user("intent@test.com")
        self.inv = _make_investigation(self.user, status="active")
        self.tool_output = _make_tool_output(self.user)

    def test_intent_marks_graph_updated(self):
        """IntentSpec results in graph_updated=True."""
        spec = IntentSpec(
            target_hypothesis_ids=[],
            design_metadata={"design_type": "full_factorial", "runs": 16},
        )
        result = connect_tool(
            investigation_id=str(self.inv.id),
            tool_output=self.tool_output,
            tool_type="doe_design",
            user=self.user,
            spec=spec,
        )
        self.assertTrue(result["linked"])
        self.assertTrue(result["graph_updated"])


class ConnectToolReportTest(TestCase):
    """CANON-002 §12.3 — report function does not modify graph."""

    def setUp(self):
        self.user = _make_user("report@test.com")
        self.inv = _make_investigation(self.user, status="active")
        self.tool_output = _make_tool_output(self.user)

    def test_report_no_graph_change(self):
        """Report tools link but don't update the graph."""
        spec = HypothesisSpec(description="Ignored — reports don't use spec")
        result = connect_tool(
            investigation_id=str(self.inv.id),
            tool_output=self.tool_output,
            tool_type="a3",
            user=self.user,
            spec=spec,
        )
        self.assertTrue(result["linked"])
        self.assertFalse(result["graph_updated"])


class AutoTransitionTest(TestCase):
    """CANON-002 §12.3 — auto-transition open → active on first connection."""

    def setUp(self):
        self.user = _make_user("transition@test.com")

    def test_open_becomes_active(self):
        """First tool connection auto-transitions open → active."""
        inv = _make_investigation(self.user, status="open")
        tool_output = _make_tool_output(self.user)
        spec = HypothesisSpec(description="First tool connection")
        connect_tool(
            investigation_id=str(inv.id),
            tool_output=tool_output,
            tool_type="rca",
            user=self.user,
            spec=spec,
        )
        inv.refresh_from_db()
        self.assertEqual(inv.status, "active")

    def test_active_stays_active(self):
        """Already-active investigation stays active."""
        inv = _make_investigation(self.user, status="active")
        tool_output = _make_tool_output(self.user)
        spec = HypothesisSpec(description="Second tool connection")
        connect_tool(
            investigation_id=str(inv.id),
            tool_output=tool_output,
            tool_type="rca",
            user=self.user,
            spec=spec,
        )
        inv.refresh_from_db()
        self.assertEqual(inv.status, "active")


class PermissionTest(TestCase):
    """CANON-002 §12.3 — non-members get PermissionError."""

    def setUp(self):
        self.owner = _make_user("owner@test.com")
        self.stranger = _make_user("stranger@test.com")
        self.member = _make_user("member@test.com")

    def test_owner_allowed(self):
        """Investigation owner can connect tools."""
        inv = _make_investigation(self.owner, status="active")
        tool_output = _make_tool_output(self.owner)
        spec = HypothesisSpec(description="Owner test")
        result = connect_tool(
            investigation_id=str(inv.id),
            tool_output=tool_output,
            tool_type="rca",
            user=self.owner,
            spec=spec,
        )
        self.assertTrue(result["linked"])

    def test_member_allowed(self):
        """Investigation member can connect tools."""
        inv = _make_investigation(self.owner, status="active")
        InvestigationMembership.objects.create(
            investigation=inv, user=self.member, role="contributor"
        )
        tool_output = _make_tool_output(self.member)
        spec = HypothesisSpec(description="Member test")
        result = connect_tool(
            investigation_id=str(inv.id),
            tool_output=tool_output,
            tool_type="rca",
            user=self.member,
            spec=spec,
        )
        self.assertTrue(result["linked"])

    def test_stranger_rejected(self):
        """Non-member gets PermissionError."""
        inv = _make_investigation(self.owner, status="active")
        tool_output = _make_tool_output(self.stranger)
        spec = HypothesisSpec(description="Stranger test")
        with self.assertRaises(PermissionError):
            connect_tool(
                investigation_id=str(inv.id),
                tool_output=tool_output,
                tool_type="rca",
                user=self.stranger,
                spec=spec,
            )

    def test_get_investigation_not_found(self):
        """Non-existent investigation raises DoesNotExist."""
        import uuid

        with self.assertRaises(Investigation.DoesNotExist):
            get_investigation(str(uuid.uuid4()), self.owner)


class IdempotentLinkTest(TestCase):
    """CANON-002 §12.3 — idempotent tool link creation."""

    def setUp(self):
        self.user = _make_user("idempotent@test.com")
        self.inv = _make_investigation(self.user, status="active")
        self.tool_output = _make_tool_output(self.user)

    def test_double_connect_creates_one_link(self):
        """Connecting the same tool twice creates only one InvestigationToolLink."""
        spec = HypothesisSpec(description="First call")
        connect_tool(
            investigation_id=str(self.inv.id),
            tool_output=self.tool_output,
            tool_type="rca",
            user=self.user,
            spec=spec,
        )
        spec2 = HypothesisSpec(description="Second call")
        connect_tool(
            investigation_id=str(self.inv.id),
            tool_output=self.tool_output,
            tool_type="rca",
            user=self.user,
            spec=spec2,
        )
        self.assertEqual(
            InvestigationToolLink.objects.filter(investigation=self.inv).count(), 1
        )

    def test_different_tools_create_separate_links(self):
        """Different tool outputs create separate links."""
        tool2 = MeasurementSystem.objects.create(
            name="Second Gage", system_type="variable", owner=self.user
        )
        spec = HypothesisSpec(description="Tool 1")
        connect_tool(
            investigation_id=str(self.inv.id),
            tool_output=self.tool_output,
            tool_type="rca",
            user=self.user,
            spec=spec,
        )
        spec2 = HypothesisSpec(description="Tool 2")
        connect_tool(
            investigation_id=str(self.inv.id),
            tool_output=tool2,
            tool_type="ishikawa",
            user=self.user,
            spec=spec2,
        )
        self.assertEqual(
            InvestigationToolLink.objects.filter(investigation=self.inv).count(), 2
        )


class LoadSaveSynaraTest(TestCase):
    """CANON-002 §12.3 — load_synara/save_synara round-trip."""

    def setUp(self):
        self.user = _make_user("synara@test.com")

    def test_empty_state_loads(self):
        """Empty synara_state produces a fresh Synara."""
        inv = _make_investigation(self.user)
        synara = load_synara(inv)
        self.assertIsNotNone(synara)
        graph = synara.to_dict()
        self.assertIn("graph", graph)

    def test_round_trip(self):
        """Save then load preserves graph state."""
        inv = _make_investigation(self.user)
        synara = load_synara(inv)
        synara.create_hypothesis(description="Test hypothesis", prior=0.6)
        save_synara(inv, synara)

        inv.refresh_from_db()
        loaded = load_synara(inv)
        hypotheses = loaded.to_dict()["graph"]["hypotheses"]
        self.assertEqual(len(hypotheses), 1)
        h = next(iter(hypotheses.values()))
        self.assertEqual(h["description"], "Test hypothesis")

    def test_existing_state_preserved(self):
        """Existing synara_state is loaded correctly."""
        inv = _make_investigation(self.user)
        synara = load_synara(inv)
        synara.create_hypothesis(description="H1")
        synara.create_hypothesis(description="H2")
        save_synara(inv, synara)

        inv.refresh_from_db()
        loaded = load_synara(inv)
        hypotheses = loaded.to_dict()["graph"]["hypotheses"]
        self.assertEqual(len(hypotheses), 2)
