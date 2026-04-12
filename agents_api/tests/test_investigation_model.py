"""
Tests for core.models.investigation — CANON-002 §7, §11.

All tests exercise real behavior per TST-001 §10.6.
Tests use DB fixtures (model creation) to verify state machine,
versioning, membership, and tool linkage.

<!-- test: agents_api.tests.test_investigation_model.InvestigationStateMachineTest -->
<!-- test: agents_api.tests.test_investigation_model.InvestigationReopenTest -->
<!-- test: agents_api.tests.test_investigation_model.InvestigationMembershipTest -->
<!-- test: agents_api.tests.test_investigation_model.InvestigationToolLinkTest -->
"""

from django.contrib.contenttypes.models import ContentType
from django.test import TestCase

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


def _make_investigation(user, title="Root cause of defect X", **kwargs):
    defaults = {
        "title": title,
        "description": "Investigating high scrap rate on Line 3",
        "owner": user,
    }
    defaults.update(kwargs)
    return Investigation.objects.create(**defaults)


class InvestigationStateMachineTest(TestCase):
    """CANON-002 §7.2 — state machine transitions."""

    def setUp(self):
        self.user = _make_user()

    def test_default_status_is_open(self):
        """New investigations default to open status."""
        inv = _make_investigation(self.user)
        self.assertEqual(inv.status, Investigation.Status.OPEN)

    def test_open_to_active(self):
        """Open → active is a valid transition."""
        inv = _make_investigation(self.user)
        inv.transition_to("active", self.user)
        self.assertEqual(inv.status, "active")

    def test_active_to_concluded(self):
        """Active → concluded is valid and sets concluded_at."""
        inv = _make_investigation(self.user, status="active")
        inv.transition_to("concluded", self.user)
        self.assertEqual(inv.status, "concluded")
        self.assertIsNotNone(inv.concluded_at)

    def test_concluded_to_exported(self):
        """Concluded → exported is valid and sets exported_at."""
        inv = _make_investigation(self.user, status="concluded")
        inv.transition_to("exported", self.user)
        self.assertEqual(inv.status, "exported")
        self.assertIsNotNone(inv.exported_at)

    def test_exported_is_terminal(self):
        """Exported is a terminal state — no transitions allowed."""
        inv = _make_investigation(self.user, status="exported")
        with self.assertRaises(ValueError):
            inv.transition_to("open", self.user)

    def test_invalid_transition_raises(self):
        """Open → concluded skips active — not allowed."""
        inv = _make_investigation(self.user)
        with self.assertRaises(ValueError):
            inv.transition_to("concluded", self.user)

    def test_backward_transition_raises(self):
        """Active → open is not allowed (no backward transitions)."""
        inv = _make_investigation(self.user, status="active")
        with self.assertRaises(ValueError):
            inv.transition_to("open", self.user)

    def test_transition_persists_to_db(self):
        """transition_to() calls save() — changes are persisted."""
        inv = _make_investigation(self.user)
        inv.transition_to("active", self.user)
        inv.refresh_from_db()
        self.assertEqual(inv.status, "active")

    def test_str_includes_version_and_status(self):
        """String representation includes title, version, and status."""
        inv = _make_investigation(self.user, title="Scrap analysis")
        s = str(inv)
        self.assertIn("Scrap analysis", s)
        self.assertIn("v1", s)
        self.assertIn("open", s)


class InvestigationReopenTest(TestCase):
    """CANON-002 §7.3 — reopen creates a new version."""

    def setUp(self):
        self.user = _make_user("reopen@test.com")
        self.other_user = _make_user("other@test.com")

    def test_reopen_concluded_creates_new_version(self):
        """Reopening a concluded investigation creates version+1."""
        inv = _make_investigation(self.user)
        inv.transition_to("active", self.user)
        inv.transition_to("concluded", self.user)

        new_inv = inv.reopen(self.other_user)
        self.assertEqual(new_inv.version, 2)
        self.assertEqual(new_inv.status, "active")
        self.assertEqual(new_inv.owner, self.other_user)
        self.assertEqual(new_inv.parent_version, inv)

    def test_reopen_exported_creates_new_version(self):
        """Reopening an exported investigation also works."""
        inv = _make_investigation(self.user)
        inv.transition_to("active", self.user)
        inv.transition_to("concluded", self.user)
        inv.transition_to("exported", self.user)

        new_inv = inv.reopen(self.user)
        self.assertEqual(new_inv.version, 2)
        self.assertEqual(new_inv.parent_version, inv)

    def test_reopen_open_raises(self):
        """Cannot reopen an investigation that is still open."""
        inv = _make_investigation(self.user)
        with self.assertRaises(ValueError):
            inv.reopen(self.user)

    def test_reopen_active_raises(self):
        """Cannot reopen an active investigation."""
        inv = _make_investigation(self.user, status="active")
        with self.assertRaises(ValueError):
            inv.reopen(self.user)

    def test_reopen_copies_synara_state(self):
        """Reopened investigation gets a deep copy of synara_state."""
        inv = _make_investigation(self.user, synara_state={"nodes": [{"id": "n1"}]})
        inv.transition_to("active", self.user)
        inv.transition_to("concluded", self.user)

        new_inv = inv.reopen(self.user)
        self.assertEqual(new_inv.synara_state, {"nodes": [{"id": "n1"}]})
        # Verify deep copy — mutating new doesn't affect original
        new_inv.synara_state["nodes"].append({"id": "n2"})
        inv.refresh_from_db()
        self.assertEqual(len(inv.synara_state["nodes"]), 1)

    def test_reopen_copies_membership(self):
        """Reopened investigation copies all members."""
        inv = _make_investigation(self.user)
        InvestigationMembership.objects.create(investigation=inv, user=self.user, role="owner")
        InvestigationMembership.objects.create(investigation=inv, user=self.other_user, role="contributor")
        inv.transition_to("active", self.user)
        inv.transition_to("concluded", self.user)

        new_inv = inv.reopen(self.user)
        self.assertEqual(new_inv.investigationmembership_set.count(), 2)

    def test_reopen_preserves_title_and_description(self):
        """Reopened investigation keeps title and description."""
        inv = _make_investigation(self.user, title="Thermal issue", description="Press overheating")
        inv.transition_to("active", self.user)
        inv.transition_to("concluded", self.user)

        new_inv = inv.reopen(self.user)
        self.assertEqual(new_inv.title, "Thermal issue")
        self.assertEqual(new_inv.description, "Press overheating")


class InvestigationMembershipTest(TestCase):
    """CANON-002 §7.4 — investigation membership through table."""

    def setUp(self):
        self.user = _make_user("member@test.com")
        self.inv = _make_investigation(self.user)

    def test_add_member(self):
        """Can add a member with a role."""
        m = InvestigationMembership.objects.create(investigation=self.inv, user=self.user, role="owner")
        self.assertEqual(m.role, "owner")
        self.assertIsNotNone(m.joined_at)

    def test_default_role_is_contributor(self):
        """Default role is contributor."""
        m = InvestigationMembership.objects.create(investigation=self.inv, user=self.user)
        self.assertEqual(m.role, "contributor")

    def test_unique_user_per_investigation(self):
        """Same user cannot be added twice to the same investigation."""
        InvestigationMembership.objects.create(investigation=self.inv, user=self.user, role="owner")
        from django.db import IntegrityError

        with self.assertRaises(IntegrityError):
            InvestigationMembership.objects.create(investigation=self.inv, user=self.user, role="viewer")

    def test_members_m2m_access(self):
        """Investigation.members M2M provides user access."""
        InvestigationMembership.objects.create(investigation=self.inv, user=self.user, role="owner")
        self.assertIn(self.user, self.inv.members.all())


class InvestigationToolLinkTest(TestCase):
    """CANON-002 §11.1 — generic FK tool linkage."""

    def setUp(self):
        self.user = _make_user("toollink@test.com")
        self.inv = _make_investigation(self.user)

    def test_link_tool_output(self):
        """Can link any model via generic FK."""
        # Use MeasurementSystem as a concrete tool output
        ms = MeasurementSystem.objects.create(name="Test Gage", system_type="variable", owner=self.user)
        ct = ContentType.objects.get_for_model(MeasurementSystem)

        link = InvestigationToolLink.objects.create(
            investigation=self.inv,
            content_type=ct,
            object_id=ms.id,
            tool_type="spc",
            tool_function="inference",
            linked_by=self.user,
        )
        self.assertEqual(link.tool_output, ms)
        self.assertEqual(link.tool_type, "spc")

    def test_unique_tool_per_investigation(self):
        """Same tool output cannot be linked twice to the same investigation."""
        ms = MeasurementSystem.objects.create(name="Gage A", system_type="variable", owner=self.user)
        ct = ContentType.objects.get_for_model(MeasurementSystem)

        InvestigationToolLink.objects.create(
            investigation=self.inv,
            content_type=ct,
            object_id=ms.id,
            tool_type="spc",
            tool_function="inference",
            linked_by=self.user,
        )
        from django.db import IntegrityError

        with self.assertRaises(IntegrityError):
            InvestigationToolLink.objects.create(
                investigation=self.inv,
                content_type=ct,
                object_id=ms.id,
                tool_type="spc",
                tool_function="inference",
                linked_by=self.user,
            )

    def test_tool_links_related_name(self):
        """Investigation.tool_links gives access to all linked tools."""
        ms = MeasurementSystem.objects.create(name="Gage B", system_type="variable", owner=self.user)
        ct = ContentType.objects.get_for_model(MeasurementSystem)
        InvestigationToolLink.objects.create(
            investigation=self.inv,
            content_type=ct,
            object_id=ms.id,
            tool_type="rca",
            tool_function="information",
            linked_by=self.user,
        )
        self.assertEqual(self.inv.tool_links.count(), 1)
        self.assertEqual(self.inv.tool_links.first().tool_type, "rca")

    def test_tool_function_choices(self):
        """tool_function must be one of the valid choices."""
        ms = MeasurementSystem.objects.create(name="Gage C", system_type="variable", owner=self.user)
        ct = ContentType.objects.get_for_model(MeasurementSystem)
        link = InvestigationToolLink(
            investigation=self.inv,
            content_type=ct,
            object_id=ms.id,
            tool_type="fmea",
            tool_function="information",
            linked_by=self.user,
        )
        # Validate choices are enforced at model validation level
        link.full_clean()  # Should not raise
