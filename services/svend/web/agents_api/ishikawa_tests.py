"""Tests for Ishikawa (Fishbone) diagram API endpoints.

Follows TST-001: Django TestCase + DRF APIClient, force_authenticate,
explicit helpers, @override_settings(SECURE_SSL_REDIRECT=False).
"""

import json

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from core.models import Project

from .models import IshikawaDiagram

User = get_user_model()


def _make_user(username="ishiuser", tier="pro"):
    user = User.objects.create_user(
        username=username,
        email=f"{username}@example.com",
        password="testpass123",
    )
    user.tier = tier
    user.save(update_fields=["tier"])
    return user


def _auth_client(user):
    client = APIClient()
    client.force_login(user)
    return client


API = "/api/ishikawa/sessions/"


# =========================================================================
# CRUD Tests
# =========================================================================


@override_settings(SECURE_SSL_REDIRECT=False)
class IshikawaCRUDTests(TestCase):
    def setUp(self):
        self.user = _make_user()
        self.client = _auth_client(self.user)

    def test_create_diagram(self):
        """POST /api/ishikawa/sessions/create/ creates a diagram."""
        resp = self.client.post(
            API + "create/",
            data=json.dumps({"effect": "High defect rate on Line 3", "title": "Line 3 Analysis"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertIn("diagram", data)
        self.assertEqual(data["diagram"]["effect"], "High defect rate on Line 3")
        self.assertEqual(data["diagram"]["title"], "Line 3 Analysis")
        self.assertEqual(data["diagram"]["status"], "draft")

    def test_create_initializes_6m_branches(self):
        """New diagrams are initialized with 6M category branches."""
        resp = self.client.post(
            API + "create/",
            data=json.dumps({"effect": "Test effect"}),
            content_type="application/json",
        )
        branches = resp.json()["diagram"]["branches"]
        self.assertEqual(len(branches), 6)
        categories = [b["category"] for b in branches]
        self.assertEqual(categories, ["Man", "Machine", "Method", "Material", "Measurement", "Mother Nature"])
        # Each should have empty causes
        for b in branches:
            self.assertEqual(b["causes"], [])

    def test_create_requires_effect(self):
        """Effect is required to create a diagram."""
        resp = self.client.post(
            API + "create/",
            data=json.dumps({"title": "No effect"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_get_diagram(self):
        """GET /api/ishikawa/sessions/<id>/ returns diagram."""
        create = self.client.post(
            API + "create/",
            data=json.dumps({"effect": "Test"}),
            content_type="application/json",
        )
        diagram_id = create.json()["diagram"]["id"]
        resp = self.client.get(API + diagram_id + "/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["diagram"]["id"], diagram_id)

    def test_update_branches(self):
        """PUT updates branches with recursive causes."""
        create = self.client.post(
            API + "create/",
            data=json.dumps({"effect": "Test"}),
            content_type="application/json",
        )
        diagram_id = create.json()["diagram"]["id"]
        branches = create.json()["diagram"]["branches"]

        # Add a cause to Man
        branches[0]["causes"].append(
            {
                "text": "Operator Training",
                "children": [{"text": "Only on one desktop", "children": []}],
            }
        )

        resp = self.client.put(
            API + diagram_id + "/update/",
            data=json.dumps({"branches": branches}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        updated = resp.json()["diagram"]["branches"]
        self.assertEqual(len(updated[0]["causes"]), 1)
        self.assertEqual(updated[0]["causes"][0]["text"], "Operator Training")
        self.assertEqual(len(updated[0]["causes"][0]["children"]), 1)

    def test_delete_diagram(self):
        """DELETE removes diagram."""
        create = self.client.post(
            API + "create/",
            data=json.dumps({"effect": "To delete"}),
            content_type="application/json",
        )
        diagram_id = create.json()["diagram"]["id"]
        resp = self.client.delete(API + diagram_id + "/delete/")
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(IshikawaDiagram.objects.filter(id=diagram_id).exists())

    def test_list_diagrams(self):
        """GET /api/ishikawa/sessions/ lists user's diagrams."""
        for i in range(3):
            self.client.post(
                API + "create/",
                data=json.dumps({"effect": f"Effect {i}"}),
                content_type="application/json",
            )
        resp = self.client.get(API)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.json()["diagrams"]), 3)

    def test_update_status(self):
        """Status can be updated to valid values."""
        create = self.client.post(
            API + "create/",
            data=json.dumps({"effect": "Test"}),
            content_type="application/json",
        )
        diagram_id = create.json()["diagram"]["id"]
        resp = self.client.put(
            API + diagram_id + "/update/",
            data=json.dumps({"status": "analyzing"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["diagram"]["status"], "analyzing")


# =========================================================================
# Permission Tests
# =========================================================================


@override_settings(SECURE_SSL_REDIRECT=False)
class IshikawaPermissionTests(TestCase):
    def test_unauthenticated_blocked(self):
        """Unauthenticated requests are rejected."""
        client = APIClient()
        resp = client.get(API)
        self.assertIn(resp.status_code, [401, 403])

    def test_user_isolation(self):
        """Users cannot access other users' diagrams."""
        user1 = _make_user("user1")
        user2 = _make_user("user2")
        client1 = _auth_client(user1)
        client2 = _auth_client(user2)

        create = client1.post(
            API + "create/",
            data=json.dumps({"effect": "User 1 effect"}),
            content_type="application/json",
        )
        diagram_id = create.json()["diagram"]["id"]

        # User 2 cannot get user 1's diagram
        resp = client2.get(API + diagram_id + "/")
        self.assertEqual(resp.status_code, 404)

        # User 2's list doesn't show user 1's diagram
        resp = client2.get(API)
        self.assertEqual(len(resp.json()["diagrams"]), 0)


# =========================================================================
# Auto-Project Tests
# =========================================================================


@override_settings(SECURE_SSL_REDIRECT=False)
class IshikawaProjectTests(TestCase):
    def setUp(self):
        self.user = _make_user()
        self.client = _auth_client(self.user)

    def test_auto_project_creation(self):
        """Creating a diagram auto-creates a linked project."""
        resp = self.client.post(
            API + "create/",
            data=json.dumps({"effect": "Auto project test"}),
            content_type="application/json",
        )
        diagram_id = resp.json()["diagram"]["id"]
        diagram = IshikawaDiagram.objects.get(id=diagram_id)
        self.assertIsNotNone(diagram.project)
        self.assertIn("ishikawa", diagram.project.tags)
        self.assertIn("auto-created", diagram.project.tags)

    def test_project_class_investigation(self):
        """Auto-created project has project_class='investigation'."""
        resp = self.client.post(
            API + "create/",
            data=json.dumps({"effect": "Class test"}),
            content_type="application/json",
        )
        diagram = IshikawaDiagram.objects.get(id=resp.json()["diagram"]["id"])
        self.assertEqual(diagram.project.project_class, "investigation")

    def test_existing_project_linked(self):
        """If project_id provided, links to existing project."""
        project = Project.objects.create(user=self.user, title="Existing")
        resp = self.client.post(
            API + "create/",
            data=json.dumps({"effect": "Link test", "project_id": str(project.id)}),
            content_type="application/json",
        )
        diagram = IshikawaDiagram.objects.get(id=resp.json()["diagram"]["id"])
        self.assertEqual(diagram.project_id, project.id)


# =========================================================================
# Evidence Tests
# =========================================================================


@override_settings(SECURE_SSL_REDIRECT=False, EVIDENCE_INTEGRATION_ENABLED=True)
class IshikawaEvidenceTests(TestCase):
    def setUp(self):
        self.user = _make_user()
        self.client = _auth_client(self.user)

    def test_evidence_created_on_complete(self):
        """Completing an Ishikawa creates evidence for causes."""
        from core.models import Evidence

        create = self.client.post(
            API + "create/",
            data=json.dumps({"effect": "Evidence test"}),
            content_type="application/json",
        )
        diagram_id = create.json()["diagram"]["id"]
        branches = create.json()["diagram"]["branches"]

        # Add causes
        branches[0]["causes"].append({"text": "Training gap", "children": []})
        branches[1]["causes"].append({"text": "Machine wear", "children": []})

        # Update with causes and set to complete
        self.client.put(
            API + diagram_id + "/update/",
            data=json.dumps({"branches": branches, "status": "complete"}),
            content_type="application/json",
        )

        diagram = IshikawaDiagram.objects.get(id=diagram_id)
        evidence = Evidence.objects.filter(project=diagram.project, source_description__startswith="ishikawa:")
        self.assertEqual(evidence.count(), 2)

    def test_no_evidence_when_draft(self):
        """Draft diagrams don't create evidence."""
        from core.models import Evidence

        create = self.client.post(
            API + "create/",
            data=json.dumps({"effect": "Draft test"}),
            content_type="application/json",
        )
        diagram_id = create.json()["diagram"]["id"]
        branches = create.json()["diagram"]["branches"]
        branches[0]["causes"].append({"text": "Some cause", "children": []})

        self.client.put(
            API + diagram_id + "/update/",
            data=json.dumps({"branches": branches}),
            content_type="application/json",
        )

        diagram = IshikawaDiagram.objects.get(id=diagram_id)
        evidence = Evidence.objects.filter(project=diagram.project, source_description__startswith="ishikawa:")
        self.assertEqual(evidence.count(), 0)
