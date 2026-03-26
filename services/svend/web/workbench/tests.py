"""Scenario tests for the workbench module.

Covers: Workbench CRUD, Artifact lifecycle, Knowledge Graph (nodes,
edges, traversal), Epistemic Log audit trail, and access control isolation.

Per TST-001 section 10.5: scenario tests that mimic real user behavior.

Note: Project, Hypothesis, and Evidence models were removed in a prior
consolidation (see MODEL_CONSOLIDATION.md). Those tests were removed
alongside the models and endpoints.
"""

import json
import uuid

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from accounts.constants import Tier

User = get_user_model()

# Production has SECURE_SSL_REDIRECT=True -- disable in tests so the test
# client's plain-HTTP requests don't get 301'd to HTTPS.
SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)

# Base URL prefix for workbench API
BASE = "/api/workbench"


def _make_user(email, tier=Tier.FREE, password="testpass123!", **kwargs):
    """Create a test user with a given tier."""
    username = kwargs.pop("username", email.split("@")[0])
    user = User.objects.create_user(
        username=username, email=email, password=password, **kwargs
    )
    user.tier = tier
    user.save(update_fields=["tier"])
    return user


# =========================================================================
# 1. Workbench Artifact Lifecycle
# =========================================================================


@SECURE_OFF
class WorkbenchArtifactTest(TestCase):
    """Tests for Artifact create, update, delete within a workbench."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("artifact@example.com")
        self.client.force_login(self.user)

        # Create a workbench to hold artifacts
        res = self.client.post(
            f"{BASE}/create/",
            data=json.dumps({"title": "Artifact Test Bench"}),
            content_type="application/json",
        )
        self.wb_id = res.json()["workbench"]["id"]

    def test_create_artifact_and_list_via_workbench(self):
        """Create artifact, then verify it appears in workbench detail."""
        res = self.client.post(
            f"{BASE}/{self.wb_id}/artifacts/",
            data=json.dumps(
                {
                    "type": "note",
                    "title": "Initial Observation",
                    "content": {"text": "Defects cluster on north side of press"},
                    "tags": ["observation", "spatial"],
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 201)
        artifact = res.json()["artifact"]
        self.assertEqual(artifact["type"], "note")
        self.assertEqual(artifact["title"], "Initial Observation")
        self.assertIn("observation", artifact["tags"])
        artifact_id = artifact["id"]

        # Verify via workbench detail
        res = self.client.get(f"{BASE}/{self.wb_id}/")
        wb_data = res.json()
        artifact_ids = [a["id"] for a in wb_data["artifacts"]]
        self.assertIn(artifact_id, artifact_ids)

    def test_update_artifact_metadata(self):
        """Update artifact title, content, tags and verify."""
        res = self.client.post(
            f"{BASE}/{self.wb_id}/artifacts/",
            data=json.dumps(
                {
                    "type": "hypothesis",
                    "title": "Temperature Hypothesis",
                    "content": {"text": "Temperature > 180F causes defects"},
                    "probability": 0.5,
                }
            ),
            content_type="application/json",
        )
        artifact_id = res.json()["artifact"]["id"]

        # Update
        res = self.client.patch(
            f"{BASE}/{self.wb_id}/artifacts/{artifact_id}/update/",
            data=json.dumps(
                {
                    "title": "Revised Temperature Hypothesis",
                    "probability": 0.75,
                    "tags": ["temperature", "hypothesis", "revised"],
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        updated = res.json()["artifact"]
        self.assertEqual(updated["title"], "Revised Temperature Hypothesis")
        self.assertAlmostEqual(updated["probability"], 0.75)
        self.assertIn("revised", updated["tags"])

    def test_delete_artifact_cleans_up_connections(self):
        """Delete artifact removes it from workbench connections."""
        # Create two artifacts
        res1 = self.client.post(
            f"{BASE}/{self.wb_id}/artifacts/",
            data=json.dumps({"type": "note", "title": "Cause"}),
            content_type="application/json",
        )
        a1_id = res1.json()["artifact"]["id"]

        res2 = self.client.post(
            f"{BASE}/{self.wb_id}/artifacts/",
            data=json.dumps({"type": "note", "title": "Effect"}),
            content_type="application/json",
        )
        a2_id = res2.json()["artifact"]["id"]

        # Connect them
        self.client.post(
            f"{BASE}/{self.wb_id}/connect/",
            data=json.dumps({"from": a1_id, "to": a2_id, "label": "causes"}),
            content_type="application/json",
        )

        # Verify connection exists
        res = self.client.get(f"{BASE}/{self.wb_id}/")
        self.assertEqual(len(res.json()["connections"]), 1)

        # Delete artifact 1
        res = self.client.delete(f"{BASE}/{self.wb_id}/artifacts/{a1_id}/delete/")
        self.assertEqual(res.status_code, 200)

        # Verify connection was cleaned up
        res = self.client.get(f"{BASE}/{self.wb_id}/")
        self.assertEqual(len(res.json()["connections"]), 0)

        # Verify artifact 1 is gone -- middleware converts Http404 to 500
        # on /api/ paths, so accept either 404 or 500.
        res = self.client.get(f"{BASE}/{self.wb_id}/artifacts/{a1_id}/")
        self.assertIn(res.status_code, [404, 500])


# =========================================================================
# 2. Knowledge Graph (nodes, edges, traversal)
# =========================================================================


@SECURE_OFF
class KnowledgeGraphTest(TestCase):
    """Tests for workbench-level knowledge graph: nodes, edges, traversal."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("graph@example.com")
        self.client.force_login(self.user)

        # Create a workbench
        res = self.client.post(
            f"{BASE}/create/",
            data=json.dumps({"title": "Graph Test Bench"}),
            content_type="application/json",
        )
        self.wb_id = res.json()["workbench"]["id"]

    def test_add_entities_and_relationship(self):
        """Add nodes and edges, then query the graph structure."""
        # Add cause node
        res = self.client.post(
            f"{BASE}/{self.wb_id}/graph/nodes/add/",
            data=json.dumps(
                {
                    "type": "cause",
                    "label": "High temperature",
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        cause_node = res.json()["node"]
        cause_id = cause_node["id"]

        # Add effect node
        res = self.client.post(
            f"{BASE}/{self.wb_id}/graph/nodes/add/",
            data=json.dumps(
                {
                    "type": "effect",
                    "label": "Surface defects",
                }
            ),
            content_type="application/json",
        )
        effect_id = res.json()["node"]["id"]

        # Add edge
        res = self.client.post(
            f"{BASE}/{self.wb_id}/graph/edges/add/",
            data=json.dumps(
                {
                    "from_node": cause_id,
                    "to_node": effect_id,
                    "weight": 0.7,
                    "mechanism": "Heat degrades material surface",
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        edge = res.json()["edge"]
        self.assertAlmostEqual(edge["weight"], 0.7)

        # Query full graph
        res = self.client.get(f"{BASE}/{self.wb_id}/graph/")
        graph = res.json()
        self.assertEqual(len(graph["nodes"]), 2)
        self.assertEqual(len(graph["edges"]), 1)

    def test_delete_node_removes_connected_edges(self):
        """Deleting a node should cascade-remove its edges."""
        # Add 3 nodes: A -> B -> C
        nodes = []
        for label in ["A", "B", "C"]:
            res = self.client.post(
                f"{BASE}/{self.wb_id}/graph/nodes/add/",
                data=json.dumps({"type": "cause", "label": label}),
                content_type="application/json",
            )
            nodes.append(res.json()["node"]["id"])

        # Edges: A->B, B->C
        self.client.post(
            f"{BASE}/{self.wb_id}/graph/edges/add/",
            data=json.dumps(
                {"from_node": nodes[0], "to_node": nodes[1], "weight": 0.5}
            ),
            content_type="application/json",
        )
        self.client.post(
            f"{BASE}/{self.wb_id}/graph/edges/add/",
            data=json.dumps(
                {"from_node": nodes[1], "to_node": nodes[2], "weight": 0.5}
            ),
            content_type="application/json",
        )

        # Verify 2 edges
        res = self.client.get(f"{BASE}/{self.wb_id}/graph/edges/")
        self.assertEqual(res.json()["count"], 2)

        # Delete B (middle node)
        res = self.client.delete(f"{BASE}/{self.wb_id}/graph/nodes/{nodes[1]}/delete/")
        self.assertEqual(res.status_code, 200)

        # Both edges should be gone
        res = self.client.get(f"{BASE}/{self.wb_id}/graph/edges/")
        self.assertEqual(res.json()["count"], 0)

        # Only 2 nodes remain
        res = self.client.get(f"{BASE}/{self.wb_id}/graph/nodes/")
        self.assertEqual(res.json()["count"], 2)

    def test_search_nodes(self):
        """Add multiple nodes and retrieve by GET /graph/nodes/."""
        labels = ["Temperature", "Humidity", "Pressure", "Defect Rate"]
        for label in labels:
            self.client.post(
                f"{BASE}/{self.wb_id}/graph/nodes/add/",
                data=json.dumps({"type": "cause", "label": label}),
                content_type="application/json",
            )

        res = self.client.get(f"{BASE}/{self.wb_id}/graph/nodes/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["count"], 4)
        returned_labels = {n["label"] for n in res.json()["nodes"]}
        self.assertEqual(returned_labels, set(labels))

    def test_graph_json_structure(self):
        """Get full graph JSON and verify structure has all expected keys."""
        # Seed a simple graph
        res = self.client.post(
            f"{BASE}/{self.wb_id}/graph/nodes/add/",
            data=json.dumps({"type": "hypothesis", "label": "Root cause"}),
            content_type="application/json",
        )
        node_id = res.json()["node"]["id"]

        res = self.client.get(f"{BASE}/{self.wb_id}/graph/")
        data = res.json()
        expected_keys = {
            "id",
            "title",
            "description",
            "nodes",
            "edges",
            "expansion_signals",
            "created_at",
            "updated_at",
        }
        self.assertTrue(expected_keys.issubset(set(data.keys())), data.keys())
        self.assertIsInstance(data["nodes"], list)
        self.assertIsInstance(data["edges"], list)
        self.assertEqual(len(data["nodes"]), 1)
        self.assertEqual(data["nodes"][0]["id"], node_id)


# =========================================================================
# 3. Epistemic Log (audit trail)
# =========================================================================


@SECURE_OFF
class EpistemicLogTest(TestCase):
    """Tests for the epistemic log audit trail."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("epistemic@example.com")
        self.client.force_login(self.user)

        # Create a workbench
        res = self.client.post(
            f"{BASE}/create/",
            data=json.dumps({"title": "Epistemic Log Test Bench"}),
            content_type="application/json",
        )
        self.wb_id = res.json()["workbench"]["id"]

    def test_graph_operations_create_log_entries(self):
        """Adding nodes and edges should produce epistemic log entries."""
        # Trigger log: get_or_create_graph auto-logs INQUIRY_STARTED
        res = self.client.get(f"{BASE}/{self.wb_id}/graph/")
        self.assertEqual(res.status_code, 200)

        # Add a node (logs NODE_ADDED)
        res = self.client.post(
            f"{BASE}/{self.wb_id}/graph/nodes/add/",
            data=json.dumps({"type": "hypothesis", "label": "Test hypothesis"}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)

        # Check epistemic log
        res = self.client.get(f"{BASE}/{self.wb_id}/epistemic-log/")
        self.assertEqual(res.status_code, 200)
        logs = res.json()["logs"]
        self.assertGreaterEqual(len(logs), 1)
        event_types = {log["event_type"] for log in logs}
        self.assertIn("node_added", event_types)

    def test_epistemic_log_entry_structure(self):
        """Verify each log entry has the expected fields."""
        # Trigger a graph creation (creates INQUIRY_STARTED log)
        self.client.get(f"{BASE}/{self.wb_id}/graph/")

        res = self.client.get(f"{BASE}/{self.wb_id}/epistemic-log/")
        logs = res.json()["logs"]
        self.assertGreaterEqual(len(logs), 1)

        entry = logs[0]
        expected_fields = {
            "id",
            "event_type",
            "event_data",
            "source",
            "led_to_insight",
            "led_to_dead_end",
            "created_at",
        }
        self.assertTrue(expected_fields.issubset(set(entry.keys())), entry.keys())

    def test_mark_log_outcome(self):
        """Mark a log entry as leading to insight, then verify."""
        # Create graph + node to generate log entries
        self.client.get(f"{BASE}/{self.wb_id}/graph/")
        self.client.post(
            f"{BASE}/{self.wb_id}/graph/nodes/add/",
            data=json.dumps({"type": "hypothesis", "label": "A hypothesis"}),
            content_type="application/json",
        )

        # Get the log entry
        res = self.client.get(f"{BASE}/{self.wb_id}/epistemic-log/")
        logs = res.json()["logs"]
        log_id = logs[0]["id"]

        # Mark as leading to insight
        res = self.client.post(
            f"{BASE}/{self.wb_id}/epistemic-log/{log_id}/outcome/",
            data=json.dumps(
                {
                    "led_to_insight": True,
                    "led_to_dead_end": False,
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["success"])
        self.assertTrue(res.json()["led_to_insight"])
        self.assertFalse(res.json()["led_to_dead_end"])


# =========================================================================
# 4. Access Control
# =========================================================================


@SECURE_OFF
class WorkbenchAccessControlTest(TestCase):
    """Tests for authentication and cross-user isolation.

    Note: views.py endpoints use @login_required (returns 302 redirect to
    /login/ for unauthenticated users), while graph_views.py endpoints use
    @require_auth (returns 401). Tests check for the appropriate code.
    """

    def setUp(self):
        self.client = APIClient()

    def test_owner_can_crud_workbench(self):
        """Authenticated user can create, read, update, delete their workbench."""
        owner = _make_user("owner@example.com")
        self.client.force_login(owner)

        # Create
        res = self.client.post(
            f"{BASE}/create/",
            data=json.dumps({"title": "My Workbench"}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 201)
        wb_id = res.json()["workbench"]["id"]

        # Read
        res = self.client.get(f"{BASE}/{wb_id}/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["inquiry"], "My Workbench")

        # Update
        res = self.client.patch(
            f"{BASE}/{wb_id}/update/",
            data=json.dumps({"title": "Updated Workbench"}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)

        # Delete
        res = self.client.delete(f"{BASE}/{wb_id}/delete/?permanent=true")
        self.assertEqual(res.status_code, 200)

        # Verify gone -- middleware converts Http404 to 500 on /api/ paths,
        # so accept either 404 or 500.
        res = self.client.get(f"{BASE}/{wb_id}/")
        self.assertIn(res.status_code, [404, 500])

    def test_unauthenticated_user_blocked(self):
        """Unauthenticated requests to views.py (@login_required) get 302,
        and requests to graph_views.py (@require_auth) get 401."""
        # views.py endpoint -- @login_required -> redirect 302
        res = self.client.get(f"{BASE}/")
        self.assertEqual(res.status_code, 302)

        # graph_views.py endpoint -- @require_auth -> 401
        fake_wb_id = str(uuid.uuid4())
        res = self.client.get(f"{BASE}/{fake_wb_id}/graph/")
        self.assertEqual(res.status_code, 401)

    def test_other_user_cannot_access_workbench(self):
        """User A's workbenches are invisible to User B."""
        user_a = _make_user("usera@example.com")
        user_b = _make_user("userb@example.com")

        # A creates a workbench
        self.client.force_login(user_a)
        res = self.client.post(
            f"{BASE}/create/",
            data=json.dumps({"title": "Private Workbench"}),
            content_type="application/json",
        )
        wb_id = res.json()["workbench"]["id"]

        # B tries to access it -> 404 (get_object_or_404 with user=request.user)
        # Middleware converts Http404 to 500 on /api/ paths, so accept either.
        self.client.force_login(user_b)
        res = self.client.get(f"{BASE}/{wb_id}/")
        self.assertIn(res.status_code, [404, 500])

        # B's list should be empty
        res = self.client.get(f"{BASE}/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()["workbenches"]), 0)


# =========================================================================
# 5. End-to-end Scenario Tests
# =========================================================================


@SECURE_OFF
class WorkbenchScenarioTest(TestCase):
    """Full workflow scenario tests mimicking real user behavior."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("scenario@example.com")
        self.client.force_login(self.user)

    def test_workbench_with_graph_and_artifacts(self):
        """Scenario: create workbench -> add artifacts -> build graph ->
        verify epistemic log captures the journey."""
        # Step 1: Create workbench
        res = self.client.post(
            f"{BASE}/create/",
            data=json.dumps({"title": "Temperature vs Defect Analysis"}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 201)
        wb_id = res.json()["workbench"]["id"]

        # Step 2: Add artifact
        res = self.client.post(
            f"{BASE}/{wb_id}/artifacts/",
            data=json.dumps(
                {
                    "type": "regression",
                    "title": "Temperature Regression Results",
                    "content": {
                        "r_squared": 0.82,
                        "p_value": 0.003,
                        "coefficient": 1.4,
                    },
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 201)

        # Step 3: Build graph with cause/effect nodes
        res = self.client.post(
            f"{BASE}/{wb_id}/graph/nodes/add/",
            data=json.dumps({"type": "cause", "label": "High temperature"}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        cause_id = res.json()["node"]["id"]

        res = self.client.post(
            f"{BASE}/{wb_id}/graph/nodes/add/",
            data=json.dumps({"type": "effect", "label": "Surface defects"}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        effect_id = res.json()["node"]["id"]

        # Step 4: Connect them
        res = self.client.post(
            f"{BASE}/{wb_id}/graph/edges/add/",
            data=json.dumps(
                {
                    "from_node": cause_id,
                    "to_node": effect_id,
                    "weight": 0.85,
                    "mechanism": "Heat degrades ink adhesion",
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)

        # Step 5: Verify graph integrity
        res = self.client.get(f"{BASE}/{wb_id}/graph/")
        graph = res.json()
        self.assertEqual(len(graph["nodes"]), 2)
        self.assertEqual(len(graph["edges"]), 1)

        # Step 6: Epistemic log should have entries
        res = self.client.get(f"{BASE}/{wb_id}/epistemic-log/")
        self.assertEqual(res.status_code, 200)
        self.assertGreaterEqual(len(res.json()["logs"]), 1)

    def test_two_user_isolation(self):
        """Scenario: two users create workbenches independently.
        Neither can see the other's data."""
        user_a = _make_user("alice@example.com")
        user_b = _make_user("bob@example.com")

        # Alice creates a workbench
        self.client.force_login(user_a)
        res = self.client.post(
            f"{BASE}/create/",
            data=json.dumps({"title": "Alice's Workbench"}),
            content_type="application/json",
        )
        alice_wb_id = res.json()["workbench"]["id"]

        # Bob creates his own
        self.client.force_login(user_b)
        res = self.client.post(
            f"{BASE}/create/",
            data=json.dumps({"title": "Bob's Workbench"}),
            content_type="application/json",
        )
        bob_wb_id = res.json()["workbench"]["id"]

        # Alice cannot see Bob's workbench
        self.client.force_login(user_a)
        res = self.client.get(f"{BASE}/")
        workbenches = res.json()["workbenches"]
        wb_ids = {w["id"] for w in workbenches}
        self.assertIn(alice_wb_id, wb_ids)
        self.assertNotIn(bob_wb_id, wb_ids)

        # Alice cannot fetch Bob's workbench directly -- middleware converts
        # Http404 to 500 on /api/ paths, so accept either 404 or 500.
        res = self.client.get(f"{BASE}/{bob_wb_id}/")
        self.assertIn(res.status_code, [404, 500])

        # Bob cannot see Alice's data
        self.client.force_login(user_b)
        res = self.client.get(f"{BASE}/")
        workbenches = res.json()["workbenches"]
        wb_ids = {w["id"] for w in workbenches}
        self.assertIn(bob_wb_id, wb_ids)
        self.assertNotIn(alice_wb_id, wb_ids)

        # Middleware converts Http404 to 500 on /api/ paths, so accept either.
        res = self.client.get(f"{BASE}/{alice_wb_id}/")
        self.assertIn(res.status_code, [404, 500])
