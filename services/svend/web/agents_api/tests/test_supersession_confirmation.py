"""
Tests for supersession detection and confirmation thresholds — CANON-002 §8.4, §10.1.

All tests exercise real behavior per TST-001 §10.6.
Tests use real Synara engine and DB models.

<!-- test: agents_api.tests.test_supersession_confirmation.ConfirmationThresholdTest -->
<!-- test: agents_api.tests.test_supersession_confirmation.RejectionThresholdTest -->
<!-- test: agents_api.tests.test_supersession_confirmation.ReversalTest -->
<!-- test: agents_api.tests.test_supersession_confirmation.SupersessionDetectionTest -->
"""

from django.test import TestCase

from agents_api.investigation_bridge import (
    CONFIRMED_THRESHOLD,
    REJECTED_THRESHOLD,
    _apply_confirmation_thresholds,
    _detect_and_apply_supersession,
)
from agents_api.synara.synara import Synara
from core.models import Evidence, Investigation, Project


def _make_user(email="test@example.com"):
    from django.contrib.auth import get_user_model

    User = get_user_model()
    return User.objects.create_user(username=email, email=email, password="testpass123")


def _make_investigation(user, **kwargs):
    defaults = {
        "title": "Threshold test",
        "description": "Testing thresholds",
        "owner": user,
    }
    defaults.update(kwargs)
    return Investigation.objects.create(**defaults)


class ConfirmationThresholdTest(TestCase):
    """CANON-002 §10.1 — posterior >= 0.85 sets confirmed."""

    def setUp(self):
        self.user = _make_user("confirm@test.com")
        self.inv = _make_investigation(self.user, status="active")

    def test_threshold_value(self):
        """CONFIRMED_THRESHOLD is 0.85."""
        self.assertEqual(CONFIRMED_THRESHOLD, 0.85)

    def test_confirmation_at_threshold(self):
        """Posterior exactly at 0.85 triggers confirmation."""
        synara = Synara()
        h = synara.create_hypothesis(description="Test hypothesis", prior=0.5)

        events = _apply_confirmation_thresholds(self.inv, synara, {h.id: 0.85})
        self.assertEqual(len(events), 1)
        self.assertIn("confirmed", events[0]["transition"])

    def test_confirmation_above_threshold(self):
        """Posterior above 0.85 triggers confirmation."""
        synara = Synara()
        h = synara.create_hypothesis(description="Strong hypothesis", prior=0.5)

        events = _apply_confirmation_thresholds(self.inv, synara, {h.id: 0.95})
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["posterior"], 0.95)

    def test_no_confirmation_below_threshold(self):
        """Posterior below 0.85 does not trigger confirmation."""
        synara = Synara()
        h = synara.create_hypothesis(description="Uncertain hypothesis", prior=0.5)

        events = _apply_confirmation_thresholds(self.inv, synara, {h.id: 0.80})
        self.assertEqual(len(events), 0)

    def test_already_confirmed_no_duplicate(self):
        """Already-confirmed hypothesis does not trigger again."""
        synara = Synara()
        h = synara.create_hypothesis(description="Confirmed hypothesis", prior=0.5)
        h.confirmation_status = "confirmed"

        events = _apply_confirmation_thresholds(self.inv, synara, {h.id: 0.90})
        self.assertEqual(len(events), 0)

    def test_nonexistent_hypothesis_ignored(self):
        """Unknown hypothesis ID is silently ignored."""
        synara = Synara()
        events = _apply_confirmation_thresholds(self.inv, synara, {"fake-id": 0.95})
        self.assertEqual(len(events), 0)


class RejectionThresholdTest(TestCase):
    """CANON-002 §10.1 — posterior <= 0.15 sets rejected."""

    def setUp(self):
        self.user = _make_user("reject@test.com")
        self.inv = _make_investigation(self.user, status="active")

    def test_threshold_value(self):
        """REJECTED_THRESHOLD is 0.15."""
        self.assertEqual(REJECTED_THRESHOLD, 0.15)

    def test_rejection_at_threshold(self):
        """Posterior exactly at 0.15 triggers rejection."""
        synara = Synara()
        h = synara.create_hypothesis(description="Weak hypothesis", prior=0.5)

        events = _apply_confirmation_thresholds(self.inv, synara, {h.id: 0.15})
        self.assertEqual(len(events), 1)
        self.assertIn("rejected", events[0]["transition"])

    def test_rejection_below_threshold(self):
        """Posterior below 0.15 triggers rejection."""
        synara = Synara()
        h = synara.create_hypothesis(description="Very weak hypothesis", prior=0.5)

        events = _apply_confirmation_thresholds(self.inv, synara, {h.id: 0.05})
        self.assertEqual(len(events), 1)

    def test_rejection_deactivates_links(self):
        """Rejected hypothesis has outgoing causal links set to strength 0."""
        synara = Synara()
        h1 = synara.create_hypothesis(description="Will be rejected", prior=0.5)
        h2 = synara.create_hypothesis(description="Downstream", prior=0.5)
        synara.create_link(from_id=h1.id, to_id=h2.id, strength=0.7, mechanism="test")

        _apply_confirmation_thresholds(self.inv, synara, {h1.id: 0.10})

        # Find the link and check strength
        for causal_link in synara.graph.links:
            if causal_link.from_id == h1.id:
                self.assertEqual(causal_link.strength, 0.0)

    def test_no_rejection_above_threshold(self):
        """Posterior above 0.15 does not trigger rejection."""
        synara = Synara()
        h = synara.create_hypothesis(description="Uncertain hypothesis", prior=0.5)

        events = _apply_confirmation_thresholds(self.inv, synara, {h.id: 0.20})
        self.assertEqual(len(events), 0)


class ReversalTest(TestCase):
    """CANON-002 §10.1 — contradictory evidence reverses confirmation/rejection."""

    def setUp(self):
        self.user = _make_user("reversal@test.com")
        self.inv = _make_investigation(self.user, status="active")

    def test_confirmed_to_uncertain(self):
        """Confirmed hypothesis reverts to uncertain when posterior drops below 0.85."""
        synara = Synara()
        h = synara.create_hypothesis(description="Was confirmed", prior=0.5)
        h.confirmation_status = "confirmed"

        events = _apply_confirmation_thresholds(self.inv, synara, {h.id: 0.60})
        self.assertEqual(len(events), 1)
        self.assertIn("uncertain", events[0]["transition"])
        self.assertEqual(h.confirmation_status, "uncertain")

    def test_rejected_to_uncertain(self):
        """Rejected hypothesis reverts to uncertain when posterior rises above 0.15."""
        synara = Synara()
        h = synara.create_hypothesis(description="Was rejected", prior=0.5)
        h.confirmation_status = "rejected"

        events = _apply_confirmation_thresholds(self.inv, synara, {h.id: 0.40})
        self.assertEqual(len(events), 1)
        self.assertIn("uncertain", events[0]["transition"])
        self.assertEqual(h.confirmation_status, "uncertain")

    def test_reversal_restores_links(self):
        """Reversal from rejected restores deactivated causal links."""
        synara = Synara()
        h1 = synara.create_hypothesis(description="Will flip", prior=0.5)
        h2 = synara.create_hypothesis(description="Downstream", prior=0.5)
        synara.create_link(from_id=h1.id, to_id=h2.id, strength=0.7, mechanism="test")

        # Reject — link strength goes to 0
        _apply_confirmation_thresholds(self.inv, synara, {h1.id: 0.10})
        for causal_link in synara.graph.links:
            if causal_link.from_id == h1.id:
                self.assertEqual(causal_link.strength, 0.0)

        # Reversal — link strength restored to 0.7
        _apply_confirmation_thresholds(self.inv, synara, {h1.id: 0.50})
        for causal_link in synara.graph.links:
            if causal_link.from_id == h1.id:
                self.assertEqual(causal_link.strength, 0.7)

    def test_uncertain_stays_uncertain(self):
        """Already-uncertain hypothesis at 0.50 produces no event."""
        synara = Synara()
        h = synara.create_hypothesis(description="Steady state", prior=0.5)

        events = _apply_confirmation_thresholds(self.inv, synara, {h.id: 0.50})
        self.assertEqual(len(events), 0)

    def test_multiple_hypotheses(self):
        """Multiple hypotheses can trigger events in one call."""
        synara = Synara()
        h1 = synara.create_hypothesis(description="Will confirm", prior=0.5)
        h2 = synara.create_hypothesis(description="Will reject", prior=0.5)
        h3 = synara.create_hypothesis(description="Will stay uncertain", prior=0.5)

        events = _apply_confirmation_thresholds(
            self.inv, synara, {h1.id: 0.90, h2.id: 0.10, h3.id: 0.50}
        )
        self.assertEqual(len(events), 2)
        transitions = {e["hypothesis_id"]: e["transition"] for e in events}
        self.assertIn("confirmed", transitions[h1.id])
        self.assertIn("rejected", transitions[h2.id])


class SupersessionDetectionTest(TestCase):
    """CANON-002 §8.4 — re-run detection creates supersedes FK."""

    def setUp(self):
        self.user = _make_user("supersession@test.com")
        self.inv = _make_investigation(self.user, status="active")
        self.project = Project.objects.create(title="Test", user=self.user)

    def test_supersession_on_matching_source(self):
        """Same source_tool:source_id creates supersedes FK."""
        e1 = Evidence.objects.create(
            project=self.project,
            summary="SPC run 1",
            source_type="analysis",
            source_description="spc:tool-123",
        )
        e2 = Evidence.objects.create(
            project=self.project,
            summary="SPC run 2",
            source_type="analysis",
            source_description="spc:tool-123",
        )

        _detect_and_apply_supersession(
            investigation=self.inv,
            source_tool="spc",
            source_id="tool-123",
            new_evidence_id=str(e2.id),
        )
        e2.refresh_from_db()
        self.assertEqual(e2.supersedes, e1)

    def test_no_supersession_different_source(self):
        """Different source IDs do not create supersession."""
        Evidence.objects.create(
            project=self.project,
            summary="SPC run 1",
            source_type="analysis",
            source_description="spc:tool-111",
        )
        e2 = Evidence.objects.create(
            project=self.project,
            summary="SPC run 2",
            source_type="analysis",
            source_description="spc:tool-222",
        )

        _detect_and_apply_supersession(
            investigation=self.inv,
            source_tool="spc",
            source_id="tool-222",
            new_evidence_id=str(e2.id),
        )
        e2.refresh_from_db()
        self.assertIsNone(e2.supersedes)

    def test_no_supersession_when_no_prior(self):
        """First evidence from a source has no supersession."""
        e1 = Evidence.objects.create(
            project=self.project,
            summary="First ever",
            source_type="analysis",
            source_description="rca:session-1",
        )

        _detect_and_apply_supersession(
            investigation=self.inv,
            source_tool="rca",
            source_id="session-1",
            new_evidence_id=str(e1.id),
        )
        e1.refresh_from_db()
        self.assertIsNone(e1.supersedes)

    def test_nonexistent_evidence_id_no_crash(self):
        """Non-existent evidence ID does not crash."""
        import uuid

        _detect_and_apply_supersession(
            investigation=self.inv,
            source_tool="spc",
            source_id="whatever",
            new_evidence_id=str(uuid.uuid4()),
        )
        # No crash = pass
