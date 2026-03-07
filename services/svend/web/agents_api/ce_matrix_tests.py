"""Tests for Cause & Effect (C&E) Matrix API endpoints.

Follows TST-001: Django TestCase + DRF APIClient, force_authenticate,
explicit helpers, @override_settings(SECURE_SSL_REDIRECT=False).
"""

import json

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from core.models import Project

from .models import CEMatrix

User = get_user_model()


def _make_user(username="ceuser", tier="pro"):
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


API = "/api/ce/sessions/"


# =========================================================================
# CRUD Tests
# =========================================================================


@override_settings(SECURE_SSL_REDIRECT=False)
class CEMatrixCRUDTests(TestCase):
    def setUp(self):
        self.user = _make_user()
        self.client = _auth_client(self.user)

    def test_create_matrix(self):
        """POST /api/ce/sessions/create/ creates a matrix."""
        resp = self.client.post(
            API + "create/",
            data=json.dumps({"title": "Line 3 C&E"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertIn("matrix", data)
        self.assertEqual(data["matrix"]["title"], "Line 3 C&E")
        self.assertEqual(data["matrix"]["status"], "draft")

    def test_create_empty(self):
        """Creating with no data still works (title optional)."""
        resp = self.client.post(
            API + "create/",
            data=json.dumps({}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(resp.json()["matrix"]["title"], "")

    def test_create_with_initial_data(self):
        """Creating with outputs and inputs pre-populated."""
        resp = self.client.post(
            API + "create/",
            data=json.dumps(
                {
                    "title": "Pre-filled",
                    "outputs": [{"name": "Defect Rate", "weight": 9}],
                    "inputs": [{"name": "Temperature"}],
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 201)
        matrix = resp.json()["matrix"]
        self.assertEqual(len(matrix["outputs"]), 1)
        self.assertEqual(len(matrix["inputs"]), 1)
        self.assertEqual(matrix["outputs"][0]["name"], "Defect Rate")

    def test_get_matrix(self):
        """GET /api/ce/sessions/<id>/ returns matrix with totals."""
        create = self.client.post(
            API + "create/",
            data=json.dumps({"title": "Get test"}),
            content_type="application/json",
        )
        matrix_id = create.json()["matrix"]["id"]
        resp = self.client.get(API + matrix_id + "/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["matrix"]["id"], matrix_id)
        self.assertIn("totals", resp.json()["matrix"])

    def test_update_matrix(self):
        """PUT updates matrix fields."""
        create = self.client.post(
            API + "create/",
            data=json.dumps({"title": "Update test"}),
            content_type="application/json",
        )
        matrix_id = create.json()["matrix"]["id"]

        resp = self.client.put(
            API + matrix_id + "/update/",
            data=json.dumps(
                {
                    "title": "Updated Title",
                    "outputs": [
                        {"name": "Defect Rate", "weight": 9},
                        {"name": "Cycle Time", "weight": 7},
                    ],
                    "inputs": [
                        {"name": "Temperature"},
                        {"name": "Pressure"},
                    ],
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        matrix = resp.json()["matrix"]
        self.assertEqual(matrix["title"], "Updated Title")
        self.assertEqual(len(matrix["outputs"]), 2)
        self.assertEqual(len(matrix["inputs"]), 2)

    def test_delete_matrix(self):
        """DELETE removes matrix."""
        create = self.client.post(
            API + "create/",
            data=json.dumps({"title": "To delete"}),
            content_type="application/json",
        )
        matrix_id = create.json()["matrix"]["id"]
        resp = self.client.delete(API + matrix_id + "/delete/")
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(CEMatrix.objects.filter(id=matrix_id).exists())

    def test_list_matrices(self):
        """GET /api/ce/sessions/ lists user's matrices."""
        for i in range(3):
            self.client.post(
                API + "create/",
                data=json.dumps({"title": f"Matrix {i}"}),
                content_type="application/json",
            )
        resp = self.client.get(API)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.json()["matrices"]), 3)

    def test_update_status(self):
        """Status can be updated to valid values."""
        create = self.client.post(
            API + "create/",
            data=json.dumps({"title": "Status test"}),
            content_type="application/json",
        )
        matrix_id = create.json()["matrix"]["id"]
        resp = self.client.put(
            API + matrix_id + "/update/",
            data=json.dumps({"status": "scoring"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["matrix"]["status"], "scoring")


# =========================================================================
# Scoring Tests
# =========================================================================


@override_settings(SECURE_SSL_REDIRECT=False)
class CEMatrixScoringTests(TestCase):
    def setUp(self):
        self.user = _make_user()
        self.client = _auth_client(self.user)

    def _create_scored_matrix(self):
        """Helper: create a matrix with outputs, inputs, and scores."""
        create = self.client.post(
            API + "create/",
            data=json.dumps(
                {
                    "title": "Scored Matrix",
                    "outputs": [
                        {"name": "Defect Rate", "weight": 9},
                        {"name": "Cycle Time", "weight": 7},
                    ],
                    "inputs": [
                        {"name": "Temperature"},
                        {"name": "Pressure"},
                        {"name": "Humidity"},
                    ],
                }
            ),
            content_type="application/json",
        )
        matrix_id = create.json()["matrix"]["id"]

        # Set scores: Temperature=high, Pressure=medium, Humidity=low
        scores = {
            "0": {"0": 9, "1": 3},  # Temperature: 9*9 + 3*7 = 81+21 = 102
            "1": {"0": 3, "1": 9},  # Pressure: 3*9 + 9*7 = 27+63 = 90
            "2": {"0": 1, "1": 1},  # Humidity: 1*9 + 1*7 = 9+7 = 16
        }
        self.client.put(
            API + matrix_id + "/update/",
            data=json.dumps({"scores": scores}),
            content_type="application/json",
        )
        return matrix_id

    def test_weighted_totals_computation(self):
        """Totals correctly apply weight * score formula."""
        matrix_id = self._create_scored_matrix()
        resp = self.client.get(API + matrix_id + "/")
        totals = resp.json()["matrix"]["totals"]

        # Sorted descending by total
        self.assertEqual(totals[0]["input_name"], "Temperature")
        self.assertEqual(totals[0]["total"], 102)  # 9*9 + 3*7

        self.assertEqual(totals[1]["input_name"], "Pressure")
        self.assertEqual(totals[1]["total"], 90)  # 3*9 + 9*7

        self.assertEqual(totals[2]["input_name"], "Humidity")
        self.assertEqual(totals[2]["total"], 16)  # 1*9 + 1*7

    def test_totals_sorted_descending(self):
        """Totals are returned sorted highest first."""
        matrix_id = self._create_scored_matrix()
        resp = self.client.get(API + matrix_id + "/")
        totals = resp.json()["matrix"]["totals"]
        scores = [t["total"] for t in totals]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_scores_zero_totals(self):
        """Matrix with inputs but no scores returns zero totals."""
        create = self.client.post(
            API + "create/",
            data=json.dumps(
                {
                    "title": "No scores",
                    "inputs": [{"name": "X1"}, {"name": "X2"}],
                    "outputs": [{"name": "Y1", "weight": 5}],
                }
            ),
            content_type="application/json",
        )
        matrix_id = create.json()["matrix"]["id"]
        resp = self.client.get(API + matrix_id + "/")
        totals = resp.json()["matrix"]["totals"]
        self.assertEqual(len(totals), 2)
        for t in totals:
            self.assertEqual(t["total"], 0)

    def test_default_weight_is_one(self):
        """Outputs without explicit weight default to 1."""
        create = self.client.post(
            API + "create/",
            data=json.dumps(
                {
                    "title": "Default weight",
                    "outputs": [{"name": "Y1"}],  # No weight specified
                    "inputs": [{"name": "X1"}],
                }
            ),
            content_type="application/json",
        )
        matrix_id = create.json()["matrix"]["id"]

        # Score 9 for X1 vs Y1
        self.client.put(
            API + matrix_id + "/update/",
            data=json.dumps({"scores": {"0": {"0": 9}}}),
            content_type="application/json",
        )

        resp = self.client.get(API + matrix_id + "/")
        totals = resp.json()["matrix"]["totals"]
        self.assertEqual(totals[0]["total"], 9)  # 9 * 1 (default weight)


# =========================================================================
# Permission Tests
# =========================================================================


@override_settings(SECURE_SSL_REDIRECT=False)
class CEMatrixPermissionTests(TestCase):
    def test_unauthenticated_blocked(self):
        """Unauthenticated requests are rejected."""
        client = APIClient()
        resp = client.get(API)
        self.assertIn(resp.status_code, [401, 403])

    def test_user_isolation(self):
        """Users cannot access other users' matrices."""
        user1 = _make_user("user1")
        user2 = _make_user("user2")
        client1 = _auth_client(user1)
        client2 = _auth_client(user2)

        create = client1.post(
            API + "create/",
            data=json.dumps({"title": "User 1 matrix"}),
            content_type="application/json",
        )
        matrix_id = create.json()["matrix"]["id"]

        # User 2 cannot get user 1's matrix
        resp = client2.get(API + matrix_id + "/")
        self.assertEqual(resp.status_code, 404)

        # User 2's list doesn't show user 1's matrix
        resp = client2.get(API)
        self.assertEqual(len(resp.json()["matrices"]), 0)


# =========================================================================
# Auto-Project Tests
# =========================================================================


@override_settings(SECURE_SSL_REDIRECT=False)
class CEMatrixProjectTests(TestCase):
    def setUp(self):
        self.user = _make_user()
        self.client = _auth_client(self.user)

    def test_auto_project_creation(self):
        """Creating a matrix auto-creates a linked project."""
        resp = self.client.post(
            API + "create/",
            data=json.dumps({"title": "Auto project test"}),
            content_type="application/json",
        )
        matrix_id = resp.json()["matrix"]["id"]
        matrix = CEMatrix.objects.get(id=matrix_id)
        self.assertIsNotNone(matrix.project)
        self.assertIn("ce-matrix", matrix.project.tags)
        self.assertIn("auto-created", matrix.project.tags)

    def test_project_class_investigation(self):
        """Auto-created project has project_class='investigation'."""
        resp = self.client.post(
            API + "create/",
            data=json.dumps({"title": "Class test"}),
            content_type="application/json",
        )
        matrix = CEMatrix.objects.get(id=resp.json()["matrix"]["id"])
        self.assertEqual(matrix.project.project_class, "investigation")

    def test_existing_project_linked(self):
        """If project_id provided, links to existing project."""
        project = Project.objects.create(user=self.user, title="Existing")
        resp = self.client.post(
            API + "create/",
            data=json.dumps({"title": "Link test", "project_id": str(project.id)}),
            content_type="application/json",
        )
        matrix = CEMatrix.objects.get(id=resp.json()["matrix"]["id"])
        self.assertEqual(matrix.project_id, project.id)


# =========================================================================
# Evidence Tests
# =========================================================================


@override_settings(SECURE_SSL_REDIRECT=False, EVIDENCE_INTEGRATION_ENABLED=True)
class CEMatrixEvidenceTests(TestCase):
    def setUp(self):
        self.user = _make_user()
        self.client = _auth_client(self.user)

    def test_evidence_created_on_complete(self):
        """Completing a C&E matrix creates evidence for top inputs."""
        from core.models import Evidence

        create = self.client.post(
            API + "create/",
            data=json.dumps(
                {
                    "title": "Evidence test",
                    "outputs": [{"name": "Defect Rate", "weight": 9}],
                    "inputs": [
                        {"name": "Temperature"},
                        {"name": "Pressure"},
                    ],
                }
            ),
            content_type="application/json",
        )
        matrix_id = create.json()["matrix"]["id"]

        # Set scores and complete
        self.client.put(
            API + matrix_id + "/update/",
            data=json.dumps(
                {
                    "scores": {"0": {"0": 9}, "1": {"0": 3}},
                    "status": "complete",
                }
            ),
            content_type="application/json",
        )

        matrix = CEMatrix.objects.get(id=matrix_id)
        evidence = Evidence.objects.filter(
            project=matrix.project,
            source_description__startswith="ce_matrix:",
        )
        self.assertEqual(evidence.count(), 2)

    def test_no_evidence_when_draft(self):
        """Draft matrices don't create evidence."""
        from core.models import Evidence

        create = self.client.post(
            API + "create/",
            data=json.dumps(
                {
                    "title": "Draft test",
                    "outputs": [{"name": "Y1", "weight": 5}],
                    "inputs": [{"name": "X1"}],
                }
            ),
            content_type="application/json",
        )
        matrix_id = create.json()["matrix"]["id"]

        # Update scores but stay in draft
        self.client.put(
            API + matrix_id + "/update/",
            data=json.dumps({"scores": {"0": {"0": 9}}}),
            content_type="application/json",
        )

        matrix = CEMatrix.objects.get(id=matrix_id)
        evidence = Evidence.objects.filter(
            project=matrix.project,
            source_description__startswith="ce_matrix:",
        )
        self.assertEqual(evidence.count(), 0)
