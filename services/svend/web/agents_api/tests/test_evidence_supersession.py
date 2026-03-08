"""
Tests for Evidence supersession — CANON-002 §8.3.

All tests exercise real behavior per TST-001 §10.6.
Tests verify the supersedes FK, reverse relation, chain walk,
and active evidence filtering.

<!-- test: agents_api.tests.test_evidence_supersession.SupersessionFKTest -->
<!-- test: agents_api.tests.test_evidence_supersession.SupersessionChainTest -->
<!-- test: agents_api.tests.test_evidence_supersession.ActiveEvidenceQueryTest -->
"""

from django.test import TestCase

from core.models import Evidence, Project


def _make_user(email="test@example.com"):
    from django.contrib.auth import get_user_model

    User = get_user_model()
    return User.objects.create_user(username=email, email=email, password="testpass123")


def _make_project(user, title="Test Project"):
    return Project.objects.create(title=title, user=user)


def _make_evidence(project, summary="Test evidence", **kwargs):
    defaults = {
        "project": project,
        "summary": summary,
        "source_type": "analysis",
        "confidence": 0.8,
    }
    defaults.update(kwargs)
    return Evidence.objects.create(**defaults)


class SupersessionFKTest(TestCase):
    """CANON-002 §8.3 — supersedes FK basic behavior."""

    def setUp(self):
        self.user = _make_user()
        self.project = _make_project(self.user)

    def test_default_supersedes_is_none(self):
        """New evidence has no supersession by default."""
        e = _make_evidence(self.project)
        self.assertIsNone(e.supersedes)

    def test_set_supersedes(self):
        """Can set supersedes to another Evidence."""
        e1 = _make_evidence(self.project, summary="SPC run 1")
        e2 = _make_evidence(self.project, summary="SPC run 2", supersedes=e1)
        self.assertEqual(e2.supersedes, e1)

    def test_superseded_by_reverse(self):
        """superseded_by reverse relation returns the newer evidence."""
        e1 = _make_evidence(self.project, summary="Old analysis")
        e2 = _make_evidence(self.project, summary="New analysis", supersedes=e1)
        self.assertIn(e2, e1.superseded_by.all())

    def test_supersedes_set_null_on_delete(self):
        """If superseded evidence is deleted, FK is set to NULL."""
        e1 = _make_evidence(self.project, summary="Will be deleted")
        e2 = _make_evidence(self.project, summary="Successor", supersedes=e1)
        e1.delete()
        e2.refresh_from_db()
        self.assertIsNone(e2.supersedes)

    def test_multiple_evidence_can_supersede_same(self):
        """Multiple newer evidence can each supersede the same old evidence."""
        e1 = _make_evidence(self.project, summary="Original")
        e2 = _make_evidence(self.project, summary="Version A", supersedes=e1)
        e3 = _make_evidence(self.project, summary="Version B", supersedes=e1)
        self.assertEqual(e1.superseded_by.count(), 2)
        self.assertIn(e2, e1.superseded_by.all())
        self.assertIn(e3, e1.superseded_by.all())


class SupersessionChainTest(TestCase):
    """CANON-002 §8.3 — chain navigation through supersession."""

    def setUp(self):
        self.user = _make_user("chain@test.com")
        self.project = _make_project(self.user)

    def test_chain_of_three(self):
        """Can walk a chain: e3 supersedes e2 supersedes e1."""
        e1 = _make_evidence(self.project, summary="Run 1")
        e2 = _make_evidence(self.project, summary="Run 2", supersedes=e1)
        e3 = _make_evidence(self.project, summary="Run 3", supersedes=e2)

        # Walk forward (newest to oldest)
        self.assertEqual(e3.supersedes, e2)
        self.assertEqual(e2.supersedes, e1)
        self.assertIsNone(e1.supersedes)

    def test_find_chain_head(self):
        """Can find the head (most recent) of a supersession chain."""
        e1 = _make_evidence(self.project, summary="Original")
        e2 = _make_evidence(self.project, summary="Update 1", supersedes=e1)
        e3 = _make_evidence(self.project, summary="Update 2", supersedes=e2)

        # Walk from any point to head
        current = e1
        while current.superseded_by.exists():
            current = current.superseded_by.first()
        self.assertEqual(current, e3)


class ActiveEvidenceQueryTest(TestCase):
    """CANON-002 §8.5 — graph walk query excludes superseded evidence."""

    def setUp(self):
        self.user = _make_user("active@test.com")
        self.project = _make_project(self.user)

    def test_active_evidence_excludes_superseded(self):
        """Evidence that has been superseded is excluded from active query."""
        e1 = _make_evidence(self.project, summary="Superseded")
        e2 = _make_evidence(self.project, summary="Current", supersedes=e1)
        e3 = _make_evidence(self.project, summary="Standalone")

        # Active = not superseded by anything
        active = Evidence.objects.filter(project=self.project, superseded_by__isnull=True)
        self.assertIn(e2, active)
        self.assertIn(e3, active)
        self.assertNotIn(e1, active)

    def test_all_active_when_no_supersession(self):
        """Without supersession, all evidence is active."""
        e1 = _make_evidence(self.project, summary="A")
        e2 = _make_evidence(self.project, summary="B")

        active = Evidence.objects.filter(project=self.project, superseded_by__isnull=True)
        self.assertEqual(active.count(), 2)

    def test_chain_only_head_is_active(self):
        """In a chain of 3, only the head is active."""
        e1 = _make_evidence(self.project, summary="v1")
        e2 = _make_evidence(self.project, summary="v2", supersedes=e1)
        e3 = _make_evidence(self.project, summary="v3", supersedes=e2)

        active = Evidence.objects.filter(project=self.project, superseded_by__isnull=True)
        self.assertEqual(list(active), [e3])

    def test_existing_evidence_unaffected(self):
        """Pre-existing evidence without supersedes remains active."""
        e1 = _make_evidence(self.project, summary="Legacy evidence")
        self.assertIsNone(e1.supersedes)
        active = Evidence.objects.filter(project=self.project, superseded_by__isnull=True)
        self.assertIn(e1, active)
