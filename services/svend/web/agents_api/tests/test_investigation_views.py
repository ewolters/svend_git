"""
Tests for Investigation API views — CANON-002 §13.

All tests exercise real behavior per TST-001 §10.6.
Follows TST-001: Django TestCase + DRF APIClient, force_authenticate,
@override_settings(SECURE_SSL_REDIRECT=False).

<!-- test: agents_api.tests.test_investigation_views.ListCreateTest -->
<!-- test: agents_api.tests.test_investigation_views.DetailTest -->
<!-- test: agents_api.tests.test_investigation_views.TransitionTest -->
<!-- test: agents_api.tests.test_investigation_views.ReopenTest -->
<!-- test: agents_api.tests.test_investigation_views.ExportTest -->
<!-- test: agents_api.tests.test_investigation_views.MemberTest -->
<!-- test: agents_api.tests.test_investigation_views.GraphTest -->
<!-- test: agents_api.tests.test_investigation_views.ToolsTest -->
"""

import uuid

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.test.client import Client

from accounts.constants import Tier
from core.models import (
    Investigation,
    InvestigationMembership,
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


def _make_investigation(user, **kwargs):
    defaults = {
        "title": "API test investigation",
        "description": "Testing views",
        "owner": user,
    }
    defaults.update(kwargs)
    return Investigation.objects.create(**defaults)


def _authed_client(user):
    client = Client()
    client.force_login(user)
    return client


@SECURE_OFF
class ListCreateTest(TestCase):
    """CANON-002 §13 — list and create investigations via API."""

    def setUp(self):
        self.user = _make_user("listcreate@test.com")
        self.client = _authed_client(self.user)

    def test_list_empty(self):
        """GET returns empty list when no investigations exist."""
        resp = self.client.get("/api/investigations/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["investigations"], [])

    def test_create_investigation(self):
        """POST creates an investigation and returns 201."""
        resp = self.client.post(
            "/api/investigations/",
            {"title": "New Investigation", "description": "Testing"},
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data["investigation"]["title"], "New Investigation")
        self.assertEqual(data["investigation"]["status"], "open")

    def test_create_auto_membership(self):
        """POST auto-creates owner membership."""
        self.client.post(
            "/api/investigations/",
            {"title": "Membership Test"},
            content_type="application/json",
        )
        inv = Investigation.objects.get(title="Membership Test")
        self.assertTrue(
            InvestigationMembership.objects.filter(
                investigation=inv, user=self.user, role="owner"
            ).exists()
        )

    def test_create_requires_title(self):
        """POST without title returns 400."""
        resp = self.client.post(
            "/api/investigations/",
            {"description": "No title"},
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_list_shows_owned(self):
        """GET returns owned investigations."""
        _make_investigation(self.user, title="My Investigation")
        resp = self.client.get("/api/investigations/")
        titles = [inv["title"] for inv in resp.json()["investigations"]]
        self.assertIn("My Investigation", titles)

    def test_list_shows_member_of(self):
        """GET returns investigations where user is a member."""
        other = _make_user("other@test.com")
        inv = _make_investigation(other, title="Shared Investigation")
        InvestigationMembership.objects.create(
            investigation=inv, user=self.user, role="contributor"
        )
        resp = self.client.get("/api/investigations/")
        titles = [i["title"] for i in resp.json()["investigations"]]
        self.assertIn("Shared Investigation", titles)


@SECURE_OFF
class DetailTest(TestCase):
    """CANON-002 §13 — investigation detail and delete."""

    def setUp(self):
        self.user = _make_user("detail@test.com")
        self.client = _authed_client(self.user)
        self.inv = _make_investigation(self.user)

    def test_get_detail(self):
        """GET returns investigation with graph."""
        resp = self.client.get(f"/api/investigations/{self.inv.id}/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()["investigation"]
        self.assertEqual(data["id"], str(self.inv.id))
        self.assertIn("graph", data)

    def test_get_not_found(self):
        """GET unknown ID returns 404."""
        resp = self.client.get(f"/api/investigations/{uuid.uuid4()}/")
        self.assertEqual(resp.status_code, 404)

    def test_get_non_member_403(self):
        """GET by non-member returns 403."""
        stranger = _make_user("stranger@test.com")
        client = _authed_client(stranger)
        resp = client.get(f"/api/investigations/{self.inv.id}/")
        self.assertEqual(resp.status_code, 403)

    def test_delete_open(self):
        """DELETE on open investigation succeeds."""
        resp = self.client.delete(f"/api/investigations/{self.inv.id}/")
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(Investigation.objects.filter(id=self.inv.id).exists())

    def test_delete_active_rejected(self):
        """DELETE on active investigation returns 400."""
        self.inv.status = "active"
        self.inv.save()
        resp = self.client.delete(f"/api/investigations/{self.inv.id}/")
        self.assertEqual(resp.status_code, 400)

    def test_delete_non_owner_rejected(self):
        """DELETE by non-owner member returns 403."""
        member = _make_user("member@test.com")
        InvestigationMembership.objects.create(
            investigation=self.inv, user=member, role="contributor"
        )
        client = _authed_client(member)
        resp = client.delete(f"/api/investigations/{self.inv.id}/")
        self.assertEqual(resp.status_code, 403)


@SECURE_OFF
class TransitionTest(TestCase):
    """CANON-002 §13 — state machine transitions via API."""

    def setUp(self):
        self.user = _make_user("transition@test.com")
        self.client = _authed_client(self.user)

    def test_open_to_active(self):
        """Transition open → active succeeds."""
        inv = _make_investigation(self.user, status="open")
        resp = self.client.post(
            f"/api/investigations/{inv.id}/transition/",
            {"target_status": "active"},
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        inv.refresh_from_db()
        self.assertEqual(inv.status, "active")

    def test_invalid_transition_rejected(self):
        """Invalid transition (open → concluded) returns 400."""
        inv = _make_investigation(self.user, status="open")
        resp = self.client.post(
            f"/api/investigations/{inv.id}/transition/",
            {"target_status": "concluded"},
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_missing_target_status(self):
        """POST without target_status returns 400."""
        inv = _make_investigation(self.user)
        resp = self.client.post(
            f"/api/investigations/{inv.id}/transition/",
            {},
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)


@SECURE_OFF
class ReopenTest(TestCase):
    """CANON-002 §13 — reopen creates a new version."""

    def setUp(self):
        self.user = _make_user("reopen@test.com")
        self.client = _authed_client(self.user)

    def test_reopen_concluded(self):
        """Reopen concluded investigation creates new version."""
        inv = _make_investigation(self.user, status="concluded")
        resp = self.client.post(f"/api/investigations/{inv.id}/reopen/")
        self.assertEqual(resp.status_code, 201)
        data = resp.json()["investigation"]
        self.assertEqual(data["status"], "active")
        self.assertEqual(data["version"], 2)

    def test_reopen_open_rejected(self):
        """Reopen open investigation returns 400."""
        inv = _make_investigation(self.user, status="open")
        resp = self.client.post(f"/api/investigations/{inv.id}/reopen/")
        self.assertEqual(resp.status_code, 400)


@SECURE_OFF
class ExportTest(TestCase):
    """CANON-002 §13 — export investigation via API."""

    def setUp(self):
        self.user = _make_user("export@test.com")
        self.client = _authed_client(self.user)
        self.project = Project.objects.create(title="Export Target", user=self.user)

    def test_export_non_concluded_rejected(self):
        """Export active investigation returns 400."""
        inv = _make_investigation(self.user, status="active")
        resp = self.client.post(
            f"/api/investigations/{inv.id}/export/",
            {"target_project_id": str(self.project.id)},
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_export_missing_project(self):
        """Export with missing target_project_id returns 400."""
        inv = _make_investigation(self.user, status="concluded")
        resp = self.client.post(
            f"/api/investigations/{inv.id}/export/",
            {},
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_export_nonexistent_project(self):
        """Export to non-existent project returns 404."""
        inv = _make_investigation(self.user, status="concluded")
        resp = self.client.post(
            f"/api/investigations/{inv.id}/export/",
            {"target_project_id": str(uuid.uuid4())},
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 404)


@SECURE_OFF
class MemberTest(TestCase):
    """CANON-002 §13 — member management via API."""

    def setUp(self):
        self.owner = _make_user("owner@test.com")
        self.member = _make_user("member2@test.com")
        self.client = _authed_client(self.owner)
        self.inv = _make_investigation(self.owner)

    def test_list_members(self):
        """GET members returns list."""
        InvestigationMembership.objects.create(
            investigation=self.inv, user=self.owner, role="owner"
        )
        resp = self.client.get(f"/api/investigations/{self.inv.id}/members/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.json()["members"]), 1)

    def test_add_member(self):
        """POST adds a member."""
        resp = self.client.post(
            f"/api/investigations/{self.inv.id}/members/",
            {"user_id": str(self.member.id), "role": "contributor"},
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 201)
        self.assertTrue(
            InvestigationMembership.objects.filter(
                investigation=self.inv, user=self.member
            ).exists()
        )

    def test_remove_member(self):
        """DELETE removes a member."""
        InvestigationMembership.objects.create(
            investigation=self.inv, user=self.member, role="contributor"
        )
        resp = self.client.delete(
            f"/api/investigations/{self.inv.id}/members/",
            {"user_id": str(self.member.id)},
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(
            InvestigationMembership.objects.filter(
                investigation=self.inv, user=self.member
            ).exists()
        )

    def test_non_owner_cannot_add(self):
        """Non-owner POST to members returns 403."""
        InvestigationMembership.objects.create(
            investigation=self.inv, user=self.member, role="contributor"
        )
        client = _authed_client(self.member)
        other = _make_user("other2@test.com")
        resp = client.post(
            f"/api/investigations/{self.inv.id}/members/",
            {"user_id": str(other.id)},
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 403)


@SECURE_OFF
class GraphTest(TestCase):
    """CANON-002 §13 — graph retrieval via API."""

    def setUp(self):
        self.user = _make_user("graph@test.com")
        self.client = _authed_client(self.user)

    def test_empty_graph(self):
        """GET graph on fresh investigation returns empty structures."""
        inv = _make_investigation(self.user)
        resp = self.client.get(f"/api/investigations/{inv.id}/graph/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["hypotheses"], {})
        self.assertEqual(data["links"], [])
        self.assertEqual(data["evidence"], [])

    def test_graph_with_hypothesis(self):
        """GET graph after adding hypothesis returns it."""
        from agents_api.investigation_bridge import HypothesisSpec, connect_tool

        inv = _make_investigation(self.user, status="active")
        tool = MeasurementSystem.objects.create(
            name="Test Gage", system_type="variable", owner=self.user
        )
        spec = HypothesisSpec(description="Test hypothesis", prior=0.6)
        connect_tool(
            investigation_id=str(inv.id),
            tool_output=tool,
            tool_type="rca",
            user=self.user,
            spec=spec,
        )

        resp = self.client.get(f"/api/investigations/{inv.id}/graph/")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(len(resp.json()["hypotheses"]) > 0)


@SECURE_OFF
class ToolsTest(TestCase):
    """CANON-002 §13 — tool listing via API."""

    def setUp(self):
        self.user = _make_user("tools@test.com")
        self.client = _authed_client(self.user)

    def test_empty_tools(self):
        """GET tools on fresh investigation returns empty list."""
        inv = _make_investigation(self.user)
        resp = self.client.get(f"/api/investigations/{inv.id}/tools/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["tools"], [])

    def test_tools_after_connect(self):
        """GET tools after connect_tool shows the linked tool."""
        from agents_api.investigation_bridge import HypothesisSpec, connect_tool

        inv = _make_investigation(self.user, status="active")
        tool = MeasurementSystem.objects.create(
            name="Test Gage", system_type="variable", owner=self.user
        )
        spec = HypothesisSpec(description="Tool link test", prior=0.5)
        connect_tool(
            investigation_id=str(inv.id),
            tool_output=tool,
            tool_type="rca",
            user=self.user,
            spec=spec,
        )

        resp = self.client.get(f"/api/investigations/{inv.id}/tools/")
        self.assertEqual(resp.status_code, 200)
        tools = resp.json()["tools"]
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["tool_type"], "rca")
