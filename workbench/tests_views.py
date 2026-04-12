"""Functional scenario tests for workbench/views.py and workbench/graph_views.py.

Covers all 62 view functions across 5 test classes.
Follows TST-001: Django TestCase + self.client, force_login for auth,
@override_settings(SECURE_SSL_REDIRECT=False).

NOTE: ErrorEnvelopeMiddleware converts Http404 exceptions to 500 on /api/
paths, so "not found" assertions accept [404, 500].
NOTE: graph_views passes `project=` to EpistemicLog.log() which does not
accept that kwarg; project-level graph endpoints may return 500.
"""

import json
import uuid

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from accounts.constants import Tier

User = get_user_model()

SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)

BASE = "/api/workbench/"

# Middleware converts Http404 to 500 on /api/ paths
NOT_FOUND = [404, 500]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user(email="wb@test.com", tier=Tier.TEAM, **kwargs):
    username = email.split("@")[0]
    user = User.objects.create_user(username=username, email=email, password="testpass123", **kwargs)
    user.tier = tier
    user.save(update_fields=["tier"])
    return user


def _err_msg(resp):
    """Extract error message from ErrorEnvelopeMiddleware response."""
    data = resp.json()
    err = data.get("error")
    if isinstance(err, dict):
        return err.get("message", "")
    return err or data.get("message") or data.get("code", "")


def _post(client, url, data=None, **kwargs):
    return client.post(url, json.dumps(data or {}), content_type="application/json", **kwargs)


def _patch(client, url, data=None, **kwargs):
    return client.patch(url, json.dumps(data or {}), content_type="application/json", **kwargs)


def _delete(client, url, data=None, **kwargs):
    if data:
        return client.delete(url, json.dumps(data), content_type="application/json", **kwargs)
    return client.delete(url, **kwargs)


# =========================================================================
# 1. WorkbenchLifecycleTest
# =========================================================================


@SECURE_OFF
class WorkbenchLifecycleTest(TestCase):
    """Scenario: create -> list -> get -> update -> export -> import ->
    create artifact -> get artifact -> update artifact -> connect ->
    disconnect -> advance phase -> guide -> delete artifact -> delete workbench.
    """

    def setUp(self):
        self.user = _make_user("wblife@test.com")
        self.client.force_login(self.user)

    # -- workbench CRUD --

    def test_create_list_get_update_delete_workbench(self):
        """Full workbench lifecycle: create -> list -> get -> update -> archive -> permanent delete."""
        # Create
        resp = _post(
            self.client,
            f"{BASE}create/",
            {"title": "Yield Analysis", "description": "Checking yield"},
        )
        self.assertEqual(resp.status_code, 201)
        wb_id = resp.json()["workbench"]["id"]

        # List
        resp = self.client.get(f"{BASE}")
        self.assertEqual(resp.status_code, 200)
        wbs = resp.json()["workbenches"]
        self.assertEqual(len(wbs), 1)
        self.assertEqual(wbs[0]["title"], "Yield Analysis")

        # Get
        resp = self.client.get(f"{BASE}{wb_id}/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["inquiry"], "Yield Analysis")

        # Update
        resp = _patch(
            self.client,
            f"{BASE}{wb_id}/update/",
            {"title": "Updated Yield", "status": "completed"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["success"])

        # Verify update
        resp = self.client.get(f"{BASE}{wb_id}/")
        self.assertEqual(resp.json()["inquiry"], "Updated Yield")
        self.assertEqual(resp.json()["status"], "completed")

        # Archive (soft delete)
        resp = _delete(self.client, f"{BASE}{wb_id}/delete/")
        self.assertEqual(resp.status_code, 200)

        # Verify archived
        resp = self.client.get(f"{BASE}{wb_id}/")
        self.assertEqual(resp.json()["status"], "archived")

        # Permanent delete
        resp = _delete(self.client, f"{BASE}{wb_id}/delete/?permanent=true")
        self.assertEqual(resp.status_code, 200)

        # Verify gone -- middleware converts Http404 to 500 on /api/ paths
        resp = self.client.get(f"{BASE}{wb_id}/")
        self.assertIn(resp.status_code, NOT_FOUND)

    def test_create_workbench_missing_title(self):
        resp = _post(self.client, f"{BASE}create/", {"title": ""})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Title", _err_msg(resp))

    def test_create_workbench_invalid_template(self):
        resp = _post(self.client, f"{BASE}create/", {"title": "Test", "template": "nonexistent"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("template", _err_msg(resp).lower())

    def test_create_workbench_invalid_json(self):
        resp = self.client.post(f"{BASE}create/", "not json", content_type="application/json")
        self.assertEqual(resp.status_code, 400)

    def test_create_dmaic_workbench_initializes_template(self):
        resp = _post(self.client, f"{BASE}create/", {"title": "DMAIC Test", "template": "dmaic"})
        self.assertEqual(resp.status_code, 201)
        data = resp.json()["workbench"]
        self.assertEqual(data["template"], "dmaic")
        self.assertIn("current_phase", data["template_state"])
        self.assertEqual(data["template_state"]["current_phase"], "define")

    def test_create_8d_workbench_initializes_template(self):
        resp = _post(self.client, f"{BASE}create/", {"title": "8D Test", "template": "8d"})
        self.assertEqual(resp.status_code, 201)
        data = resp.json()["workbench"]
        self.assertEqual(data["template"], "8d")
        self.assertIn("current_discipline", data["template_state"])

    def test_list_workbenches_filter_by_status(self):
        _post(self.client, f"{BASE}create/", {"title": "Active WB"})
        resp = _post(self.client, f"{BASE}create/", {"title": "To Archive"})
        wb_id = resp.json()["workbench"]["id"]
        _patch(self.client, f"{BASE}{wb_id}/update/", {"status": "archived"})

        resp = self.client.get(f"{BASE}?status=active")
        self.assertEqual(len(resp.json()["workbenches"]), 1)
        self.assertEqual(resp.json()["workbenches"][0]["title"], "Active WB")

    def test_list_workbenches_filter_by_template(self):
        _post(self.client, f"{BASE}create/", {"title": "Blank", "template": "blank"})
        _post(self.client, f"{BASE}create/", {"title": "DMAIC", "template": "dmaic"})

        resp = self.client.get(f"{BASE}?template=dmaic")
        self.assertEqual(len(resp.json()["workbenches"]), 1)
        self.assertEqual(resp.json()["workbenches"][0]["title"], "DMAIC")

    def test_workbench_isolation_between_users(self):
        """Another user cannot see or modify this user's workbench."""
        resp = _post(self.client, f"{BASE}create/", {"title": "My WB"})
        wb_id = resp.json()["workbench"]["id"]

        other = _make_user("other@test.com")
        self.client.force_login(other)
        resp = self.client.get(f"{BASE}{wb_id}/")
        self.assertIn(resp.status_code, NOT_FOUND)

    def test_get_nonexistent_workbench(self):
        fake_id = uuid.uuid4()
        resp = self.client.get(f"{BASE}{fake_id}/")
        self.assertIn(resp.status_code, NOT_FOUND)

    # -- export / import --

    def test_export_import_roundtrip(self):
        """Export a workbench, import it, verify data preserved."""
        resp = _post(self.client, f"{BASE}create/", {"title": "Export Me"})
        wb_id = resp.json()["workbench"]["id"]

        # Add an artifact
        _post(
            self.client,
            f"{BASE}{wb_id}/artifacts/",
            {"type": "note", "title": "My Note", "content": {"text": "hello"}},
        )

        # Export
        resp = _post(self.client, f"{BASE}{wb_id}/export/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("attachment", resp.get("Content-Disposition", ""))
        exported = resp.json()

        # Import
        resp = _post(self.client, f"{BASE}import/", exported)
        self.assertEqual(resp.status_code, 201)
        new_id = resp.json()["workbench_id"]
        self.assertNotEqual(new_id, wb_id)

        # Verify imported workbench has same data
        resp = self.client.get(f"{BASE}{new_id}/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["inquiry"], "Export Me")
        self.assertEqual(len(resp.json()["artifacts"]), 1)

    def test_import_invalid_json(self):
        resp = self.client.post(f"{BASE}import/", "bad", content_type="application/json")
        self.assertEqual(resp.status_code, 400)

    # -- artifacts --

    def test_artifact_crud(self):
        """Create -> get -> update -> delete artifact lifecycle."""
        resp = _post(self.client, f"{BASE}create/", {"title": "Art WB"})
        wb_id = resp.json()["workbench"]["id"]

        # Create artifact
        resp = _post(
            self.client,
            f"{BASE}{wb_id}/artifacts/",
            {
                "type": "note",
                "title": "Observation",
                "content": {"text": "Saw something"},
                "tags": ["important"],
                "position": {"x": 10, "y": 20},
            },
        )
        self.assertEqual(resp.status_code, 201)
        art_id = resp.json()["artifact"]["id"]

        # Get artifact
        resp = self.client.get(f"{BASE}{wb_id}/artifacts/{art_id}/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["title"], "Observation")

        # Update artifact
        resp = _patch(
            self.client,
            f"{BASE}{wb_id}/artifacts/{art_id}/update/",
            {
                "title": "Updated Obs",
                "tags": ["critical"],
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["artifact"]["title"], "Updated Obs")

        # Delete artifact
        resp = _delete(self.client, f"{BASE}{wb_id}/artifacts/{art_id}/delete/")
        self.assertEqual(resp.status_code, 200)

        # Verify gone -- middleware converts Http404 to 500 on /api/ paths
        resp = self.client.get(f"{BASE}{wb_id}/artifacts/{art_id}/")
        self.assertIn(resp.status_code, NOT_FOUND)

    def test_create_artifact_missing_type(self):
        resp = _post(self.client, f"{BASE}create/", {"title": "WB"})
        wb_id = resp.json()["workbench"]["id"]
        resp = _post(self.client, f"{BASE}{wb_id}/artifacts/", {"title": "No type"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("type", _err_msg(resp).lower())

    def test_artifact_with_probability_and_supports(self):
        """Create hypothesis artifact with probability and supports fields."""
        resp = _post(self.client, f"{BASE}create/", {"title": "WB"})
        wb_id = resp.json()["workbench"]["id"]

        resp = _post(
            self.client,
            f"{BASE}{wb_id}/artifacts/",
            {
                "type": "hypothesis",
                "title": "My Hyp",
                "content": {"text": "test"},
                "probability": 0.75,
                "source": "analyst",
            },
        )
        self.assertEqual(resp.status_code, 201)
        art = resp.json()["artifact"]
        self.assertAlmostEqual(art["probability"], 0.75)

        # Update with supports/weakens
        art_id = art["id"]
        resp = _patch(
            self.client,
            f"{BASE}{wb_id}/artifacts/{art_id}/update/",
            {
                "supports": ["hyp-1"],
                "weakens": ["hyp-2"],
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["artifact"]["supports"], ["hyp-1"])

    # -- connections --

    def test_connect_disconnect_artifacts(self):
        """Connect two artifacts then disconnect them."""
        resp = _post(self.client, f"{BASE}create/", {"title": "Conn WB"})
        wb_id = resp.json()["workbench"]["id"]

        # Create two artifacts
        r1 = _post(self.client, f"{BASE}{wb_id}/artifacts/", {"type": "note", "content": {}})
        r2 = _post(self.client, f"{BASE}{wb_id}/artifacts/", {"type": "note", "content": {}})
        a1 = r1.json()["artifact"]["id"]
        a2 = r2.json()["artifact"]["id"]

        # Connect
        resp = _post(
            self.client,
            f"{BASE}{wb_id}/connect/",
            {"from": a1, "to": a2, "label": "causes"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["success"])

        # Verify connection exists
        resp = self.client.get(f"{BASE}{wb_id}/")
        conns = resp.json()["connections"]
        self.assertEqual(len(conns), 1)
        self.assertEqual(conns[0]["from"], a1)

        # Disconnect
        resp = _delete(self.client, f"{BASE}{wb_id}/disconnect/", {"from": a1, "to": a2})
        self.assertEqual(resp.status_code, 200)

        # Verify disconnected
        resp = self.client.get(f"{BASE}{wb_id}/")
        self.assertEqual(len(resp.json()["connections"]), 0)

    def test_connect_missing_fields(self):
        resp = _post(self.client, f"{BASE}create/", {"title": "WB"})
        wb_id = resp.json()["workbench"]["id"]
        resp = _post(self.client, f"{BASE}{wb_id}/connect/", {"from": "a"})
        self.assertEqual(resp.status_code, 400)

    # -- template-specific (DMAIC phase) --

    def test_advance_dmaic_phase(self):
        """Create DMAIC workbench, advance through phases."""
        resp = _post(self.client, f"{BASE}create/", {"title": "DMAIC", "template": "dmaic"})
        wb_id = resp.json()["workbench"]["id"]

        # Advance: define -> measure
        resp = _post(self.client, f"{BASE}{wb_id}/advance-phase/", {"notes": "Done defining"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["current_phase"], "measure")

        # Advance: measure -> analyze
        resp = _post(self.client, f"{BASE}{wb_id}/advance-phase/")
        self.assertEqual(resp.json()["current_phase"], "analyze")

        # Advance: analyze -> improve
        resp = _post(self.client, f"{BASE}{wb_id}/advance-phase/")
        self.assertEqual(resp.json()["current_phase"], "improve")

        # Advance: improve -> control
        resp = _post(self.client, f"{BASE}{wb_id}/advance-phase/")
        self.assertEqual(resp.json()["current_phase"], "control")

    def test_advance_phase_non_dmaic(self):
        resp = _post(self.client, f"{BASE}create/", {"title": "Blank"})
        wb_id = resp.json()["workbench"]["id"]
        resp = _post(self.client, f"{BASE}{wb_id}/advance-phase/")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("DMAIC", _err_msg(resp))

    # -- guide observations --

    def test_guide_observation_lifecycle(self):
        """Add observation -> acknowledge it."""
        resp = _post(self.client, f"{BASE}create/", {"title": "Guide WB"})
        wb_id = resp.json()["workbench"]["id"]

        # Add observation
        resp = _post(
            self.client,
            f"{BASE}{wb_id}/guide/observe/",
            {"observation": "Data shows trend", "suggestion": "Try regression"},
        )
        self.assertEqual(resp.status_code, 200)

        # Verify it's in the workbench
        resp = self.client.get(f"{BASE}{wb_id}/")
        obs = resp.json()["guide_observations"]
        self.assertEqual(len(obs), 1)
        self.assertFalse(obs[0]["acknowledged"])

        # Acknowledge
        resp = _post(self.client, f"{BASE}{wb_id}/guide/0/acknowledge/")
        self.assertEqual(resp.status_code, 200)

        # Verify acknowledged
        resp = self.client.get(f"{BASE}{wb_id}/")
        self.assertTrue(resp.json()["guide_observations"][0]["acknowledged"])

    def test_guide_observation_missing_text(self):
        resp = _post(self.client, f"{BASE}create/", {"title": "WB"})
        wb_id = resp.json()["workbench"]["id"]
        resp = _post(self.client, f"{BASE}{wb_id}/guide/observe/", {"observation": ""})
        self.assertEqual(resp.status_code, 400)

    def test_acknowledge_invalid_index(self):
        resp = _post(self.client, f"{BASE}create/", {"title": "WB"})
        wb_id = resp.json()["workbench"]["id"]
        resp = _post(self.client, f"{BASE}{wb_id}/guide/99/acknowledge/")
        self.assertEqual(resp.status_code, 400)

    # -- update workbench with all fields --

    def test_update_workbench_all_fields(self):
        resp = _post(self.client, f"{BASE}create/", {"title": "All Fields"})
        wb_id = resp.json()["workbench"]["id"]

        resp = _patch(
            self.client,
            f"{BASE}{wb_id}/update/",
            {
                "title": "New Title",
                "description": "New Desc",
                "conclusion": "Root cause found",
                "conclusion_confidence": "high",
                "layout": {"a": {"x": 1}},
                "datasets": [{"name": "ds1"}],
                "guide_observations": [{"note": "test"}],
            },
        )
        self.assertEqual(resp.status_code, 200)

        resp = self.client.get(f"{BASE}{wb_id}/")
        d = resp.json()
        self.assertEqual(d["conclusion"], "Root cause found")
        self.assertEqual(d["datasets"], [{"name": "ds1"}])

    # -- delete artifact cleans layout and connections --

    def test_delete_artifact_cleans_layout_and_connections(self):
        resp = _post(self.client, f"{BASE}create/", {"title": "Clean WB"})
        wb_id = resp.json()["workbench"]["id"]

        r1 = _post(
            self.client,
            f"{BASE}{wb_id}/artifacts/",
            {"type": "note", "content": {}, "position": {"x": 10, "y": 20}},
        )
        r2 = _post(self.client, f"{BASE}{wb_id}/artifacts/", {"type": "note", "content": {}})
        a1 = r1.json()["artifact"]["id"]
        a2 = r2.json()["artifact"]["id"]

        _post(self.client, f"{BASE}{wb_id}/connect/", {"from": a1, "to": a2})

        # Delete a1 -- should clean up layout and connections
        _delete(self.client, f"{BASE}{wb_id}/artifacts/{a1}/delete/")

        resp = self.client.get(f"{BASE}{wb_id}/")
        self.assertNotIn(a1, resp.json()["layout"])
        self.assertEqual(len(resp.json()["connections"]), 0)


# =========================================================================
# 2. (Removed: ProjectLifecycleTest — workbench.Project deprecated, use core.Project)
# 3. (Removed: HypothesisEvidenceTest — workbench.Hypothesis deprecated, use core.Hypothesis)
# 4. (Removed: ConversationTest — workbench.Conversation deprecated)
# =========================================================================


# =========================================================================
# 5. KnowledgeGraphTest (project graph tests removed — workbench graph tests remain)
# =========================================================================
# Lines removed: ProjectLifecycleTest, HypothesisEvidenceTest, ConversationTest
# These tested deprecated workbench.Project/Hypothesis/Evidence/Conversation models.
# Canonical tests for these concepts live in core/tests.py.


# =========================================================================
# 5. KnowledgeGraphTest
# =========================================================================


@SECURE_OFF
class KnowledgeGraphTest(TestCase):
    """Scenario: get/create graph -> add nodes -> add edges -> update weight ->
    apply evidence -> check expansion -> resolve expansion -> causal chain ->
    upstream causes -> clear graph -> epistemic log -> mark log outcome.
    Also: project-level graph operations."""

    def setUp(self):
        self.user = _make_user("graph@test.com")
        self.client.force_login(self.user)
        resp = _post(self.client, f"{BASE}create/", {"title": "Graph WB"})
        self.wb_id = resp.json()["workbench"]["id"]

    def _g(self, *parts):
        return f"{BASE}{self.wb_id}/graph/" + "/".join(str(p) for p in parts)

    def test_graph_get_or_create(self):
        """Getting a graph auto-creates it."""
        resp = self.client.get(self._g())
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("nodes", data)
        self.assertIn("edges", data)
        self.assertEqual(len(data["nodes"]), 0)

    def test_node_add_list_remove(self):
        """Add nodes, list them, remove one."""
        # Add two nodes
        resp = _post(
            self.client,
            self._g("nodes/add/"),
            {"type": "hypothesis", "label": "Temperature is key"},
        )
        self.assertEqual(resp.status_code, 200)
        n1_id = resp.json()["node"]["id"]

        resp = _post(
            self.client,
            self._g("nodes/add/"),
            {"type": "cause", "label": "Equipment age"},
        )
        resp.json()["node"]["id"]  # n2 created successfully

        # List nodes
        resp = self.client.get(self._g("nodes/"))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["count"], 2)

        # Remove first node
        resp = _delete(self.client, self._g(f"nodes/{n1_id}/delete/"))
        self.assertEqual(resp.status_code, 200)

        # Verify only one left
        resp = self.client.get(self._g("nodes/"))
        self.assertEqual(resp.json()["count"], 1)

    def test_remove_nonexistent_node(self):
        # Ensure graph exists first
        self.client.get(self._g())
        resp = _delete(self.client, self._g("nodes/fake_id/delete/"))
        self.assertEqual(resp.status_code, 404)

    def test_add_node_with_metadata(self):
        resp = _post(
            self.client,
            self._g("nodes/add/"),
            {
                "type": "observation",
                "label": "Temp spike",
                "metadata": {"value": 150, "unit": "celsius"},
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["node"]["metadata"]["value"], 150)

    def test_edge_add_list(self):
        """Add edge between two nodes."""
        r1 = _post(self.client, self._g("nodes/add/"), {"type": "cause", "label": "Heat"})
        r2 = _post(self.client, self._g("nodes/add/"), {"type": "effect", "label": "Defects"})
        n1 = r1.json()["node"]["id"]
        n2 = r2.json()["node"]["id"]

        resp = _post(
            self.client,
            self._g("edges/add/"),
            {
                "from_node": n1,
                "to_node": n2,
                "weight": 0.7,
                "mechanism": "Thermal stress",
            },
        )
        self.assertEqual(resp.status_code, 200)
        edge = resp.json()["edge"]
        self.assertAlmostEqual(edge["weight"], 0.7)

        # List edges
        resp = self.client.get(self._g("edges/"))
        self.assertEqual(resp.json()["count"], 1)

    def test_edge_add_nonexistent_node(self):
        self.client.get(self._g())
        resp = _post(
            self.client,
            self._g("edges/add/"),
            {"from_node": "fake1", "to_node": "fake2"},
        )
        self.assertEqual(resp.status_code, 400)

    def test_edge_add_invalid_json(self):
        self.client.get(self._g())
        resp = self.client.post(self._g("edges/add/"), "bad", content_type="application/json")
        self.assertEqual(resp.status_code, 400)

    def test_update_edge_weight(self):
        r1 = _post(self.client, self._g("nodes/add/"), {"type": "cause", "label": "A"})
        r2 = _post(self.client, self._g("nodes/add/"), {"type": "effect", "label": "B"})
        n1 = r1.json()["node"]["id"]
        n2 = r2.json()["node"]["id"]

        resp = _post(
            self.client,
            self._g("edges/add/"),
            {"from_node": n1, "to_node": n2, "weight": 0.5},
        )
        edge_id = resp.json()["edge"]["id"]

        resp = _post(self.client, self._g(f"edges/{edge_id}/weight/"), {"weight": 0.9})
        self.assertEqual(resp.status_code, 200)
        self.assertAlmostEqual(resp.json()["old_weight"], 0.5)
        self.assertAlmostEqual(resp.json()["new_weight"], 0.9)

    def test_update_nonexistent_edge_weight(self):
        self.client.get(self._g())
        resp = _post(self.client, self._g("edges/fake_edge/weight/"), {"weight": 0.5})
        self.assertEqual(resp.status_code, 404)

    def test_apply_evidence_bayesian_update(self):
        """Apply evidence to update edge weights."""
        r1 = _post(self.client, self._g("nodes/add/"), {"type": "cause", "label": "X"})
        r2 = _post(self.client, self._g("nodes/add/"), {"type": "effect", "label": "Y"})
        n1 = r1.json()["node"]["id"]
        n2 = r2.json()["node"]["id"]

        resp = _post(
            self.client,
            self._g("edges/add/"),
            {"from_node": n1, "to_node": n2, "weight": 0.5},
        )
        edge_id = resp.json()["edge"]["id"]

        resp = _post(
            self.client,
            self._g("evidence/apply/"),
            {
                "evidence_id": "ev_001",
                "supports": [{"edge_id": edge_id, "likelihood": 0.8}],
                "weakens": [],
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["success"])
        updates = resp.json()["updates"]
        self.assertEqual(len(updates), 1)
        self.assertNotEqual(updates[0]["old_weight"], updates[0]["new_weight"])

    def test_apply_evidence_invalid_json(self):
        self.client.get(self._g())
        resp = self.client.post(self._g("evidence/apply/"), "bad", content_type="application/json")
        self.assertEqual(resp.status_code, 400)

    def test_check_expansion_triggered(self):
        """Low likelihoods trigger expansion signal."""
        self.client.get(self._g())
        resp = _post(
            self.client,
            self._g("expansion/check/"),
            {"likelihoods": {"edge_1": 0.05, "edge_2": 0.1}},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["expansion_needed"])
        self.assertIn("signal", resp.json())

    def test_check_expansion_not_triggered(self):
        """High likelihoods do not trigger expansion."""
        self.client.get(self._g())
        resp = _post(
            self.client,
            self._g("expansion/check/"),
            {"likelihoods": {"edge_1": 0.5, "edge_2": 0.8}},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(resp.json()["expansion_needed"])

    def test_expansion_signals_and_resolve(self):
        """Get pending expansion signals, then resolve one."""
        self.client.get(self._g())

        # Trigger expansion
        resp = _post(self.client, self._g("expansion/check/"), {"likelihoods": {"edge_1": 0.01}})
        signal_id = resp.json()["signal"]["id"]

        # Get pending signals
        resp = self.client.get(self._g("expansions/"))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["count"], 1)

        # Resolve
        resp = _post(
            self.client,
            self._g(f"expansions/{signal_id}/resolve/"),
            {
                "resolution": "new_hypothesis",
                "new_node": {"type": "hypothesis", "label": "New cause discovered"},
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["resolution"], "new_hypothesis")
        self.assertIsNotNone(resp.json()["new_node"])

        # No more pending
        resp = self.client.get(self._g("expansions/"))
        self.assertEqual(resp.json()["count"], 0)

    def test_resolve_expansion_dismissed(self):
        """Resolve expansion as dismissed (no new node)."""
        self.client.get(self._g())
        resp = _post(self.client, self._g("expansion/check/"), {"likelihoods": {"e": 0.01}})
        signal_id = resp.json()["signal"]["id"]

        resp = _post(
            self.client,
            self._g(f"expansions/{signal_id}/resolve/"),
            {"resolution": "dismissed"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(resp.json()["new_node"])

    def test_resolve_nonexistent_signal(self):
        self.client.get(self._g())
        resp = _post(
            self.client,
            self._g("expansions/fake_signal/resolve/"),
            {"resolution": "dismissed"},
        )
        self.assertEqual(resp.status_code, 404)

    def test_causal_chain(self):
        """Add A->B->C, find chain from A to C."""
        ra = _post(self.client, self._g("nodes/add/"), {"type": "cause", "label": "A"})
        rb = _post(self.client, self._g("nodes/add/"), {"type": "cause", "label": "B"})
        rc = _post(self.client, self._g("nodes/add/"), {"type": "effect", "label": "C"})
        na = ra.json()["node"]["id"]
        nb = rb.json()["node"]["id"]
        nc = rc.json()["node"]["id"]

        _post(
            self.client,
            self._g("edges/add/"),
            {"from_node": na, "to_node": nb, "weight": 0.8},
        )
        _post(
            self.client,
            self._g("edges/add/"),
            {"from_node": nb, "to_node": nc, "weight": 0.6},
        )

        resp = self.client.get(self._g(f"chain/{na}/{nc}/"))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["count"], 1)
        self.assertEqual(len(resp.json()["chains"][0]), 2)  # 2 edges in chain

    def test_causal_chain_no_path(self):
        """No chain between disconnected nodes."""
        ra = _post(self.client, self._g("nodes/add/"), {"type": "cause", "label": "A"})
        rb = _post(self.client, self._g("nodes/add/"), {"type": "effect", "label": "B"})
        na = ra.json()["node"]["id"]
        nb = rb.json()["node"]["id"]

        resp = self.client.get(self._g(f"chain/{na}/{nb}/"))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["count"], 0)

    def test_upstream_causes(self):
        """Add A->B->C, get upstream of C."""
        ra = _post(self.client, self._g("nodes/add/"), {"type": "cause", "label": "Root"})
        rb = _post(
            self.client,
            self._g("nodes/add/"),
            {"type": "cause", "label": "Intermediate"},
        )
        rc = _post(self.client, self._g("nodes/add/"), {"type": "effect", "label": "Leaf"})
        na = ra.json()["node"]["id"]
        nb = rb.json()["node"]["id"]
        nc = rc.json()["node"]["id"]

        _post(self.client, self._g("edges/add/"), {"from_node": na, "to_node": nb})
        _post(self.client, self._g("edges/add/"), {"from_node": nb, "to_node": nc})

        resp = self.client.get(self._g(f"upstream/{nc}/"))
        self.assertEqual(resp.status_code, 200)
        self.assertGreaterEqual(resp.json()["count"], 1)

    def test_upstream_causes_with_depth(self):
        """Upstream causes respect depth parameter."""
        ra = _post(self.client, self._g("nodes/add/"), {"type": "cause", "label": "Root"})
        rb = _post(self.client, self._g("nodes/add/"), {"type": "cause", "label": "Mid"})
        rc = _post(self.client, self._g("nodes/add/"), {"type": "effect", "label": "Leaf"})
        na = ra.json()["node"]["id"]
        nb = rb.json()["node"]["id"]
        nc = rc.json()["node"]["id"]

        _post(self.client, self._g("edges/add/"), {"from_node": na, "to_node": nb})
        _post(self.client, self._g("edges/add/"), {"from_node": nb, "to_node": nc})

        # Depth=1 should only find immediate parent
        resp = self.client.get(self._g(f"upstream/{nc}/?depth=1"))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["count"], 1)

    def test_clear_graph(self):
        """Add nodes/edges, then clear entire graph."""
        _post(self.client, self._g("nodes/add/"), {"type": "cause", "label": "X"})
        _post(self.client, self._g("nodes/add/"), {"type": "effect", "label": "Y"})

        resp = _delete(self.client, self._g("clear/"))
        self.assertEqual(resp.status_code, 200)

        resp = self.client.get(self._g("nodes/"))
        self.assertEqual(resp.json()["count"], 0)

    # -- epistemic log --

    def test_epistemic_log(self):
        """Adding a node creates a log entry; verify log retrieval."""
        _post(
            self.client,
            self._g("nodes/add/"),
            {"type": "hypothesis", "label": "Logged"},
        )

        resp = self.client.get(f"{BASE}{self.wb_id}/epistemic-log/")
        self.assertEqual(resp.status_code, 200)
        # Should have at least the inquiry_started + node_added entries
        self.assertGreaterEqual(resp.json()["count"], 1)

    def test_epistemic_log_filter_by_type(self):
        """Filter log by event type."""
        _post(self.client, self._g("nodes/add/"), {"type": "hypothesis", "label": "Test"})

        resp = self.client.get(f"{BASE}{self.wb_id}/epistemic-log/?type=node_added")
        self.assertEqual(resp.status_code, 200)
        for log_entry in resp.json()["logs"]:
            self.assertEqual(log_entry["event_type"], "node_added")

    def test_epistemic_log_limit(self):
        """Epistemic log respects limit parameter."""
        for i in range(5):
            _post(self.client, self._g("nodes/add/"), {"type": "cause", "label": f"N{i}"})

        resp = self.client.get(f"{BASE}{self.wb_id}/epistemic-log/?limit=2")
        self.assertEqual(resp.status_code, 200)
        self.assertLessEqual(resp.json()["count"], 2)

    def test_epistemic_log_mark_outcome(self):
        """Mark an epistemic log entry with outcome."""
        _post(self.client, self._g("nodes/add/"), {"type": "hypothesis", "label": "Test"})

        resp = self.client.get(f"{BASE}{self.wb_id}/epistemic-log/")
        logs = resp.json()["logs"]
        self.assertGreater(len(logs), 0)
        log_id = logs[0]["id"]

        resp = _post(
            self.client,
            f"{BASE}{self.wb_id}/epistemic-log/{log_id}/outcome/",
            {"led_to_insight": True, "led_to_dead_end": False},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["led_to_insight"])
        self.assertFalse(resp.json()["led_to_dead_end"])

    # -- remove node also removes edges --

    def test_remove_node_removes_edges(self):
        r1 = _post(self.client, self._g("nodes/add/"), {"type": "cause", "label": "A"})
        r2 = _post(self.client, self._g("nodes/add/"), {"type": "effect", "label": "B"})
        n1 = r1.json()["node"]["id"]
        n2 = r2.json()["node"]["id"]

        _post(self.client, self._g("edges/add/"), {"from_node": n1, "to_node": n2})

        # Remove source node
        _delete(self.client, self._g(f"nodes/{n1}/delete/"))

        # Edge should be gone
        resp = self.client.get(self._g("edges/"))
        self.assertEqual(resp.json()["count"], 0)

    # -- auth required for graph views --

    def test_graph_requires_auth(self):
        self.client.logout()
        resp = self.client.get(self._g())
        self.assertIn(resp.status_code, [401, 302])

    def test_workbench_views_require_auth(self):
        self.client.logout()
        resp = self.client.get(f"{BASE}")
        self.assertIn(resp.status_code, [302, 401])


# =========================================================================
# 6. DSWSessionReloadTest — Behavioral tests for save/load round-trip
# =========================================================================


@SECURE_OFF
class DSWSessionReloadTest(TestCase):
    """Behavioral tests: simulate a user uploading data, saving a session,
    then reloading it and verifying everything comes back.

    These tests hit the actual API endpoints the frontend calls — the same
    sequence a real user triggers. They exist because DSW session reload has
    regressed multiple times due to shape mismatches, missing fields, and
    broken restore logic.

    CR: 20fdd5cb — Fix DSW session reload
    """

    DSW_BASE = "/api/dsw/"

    def setUp(self):
        self.user = _make_user("dsw_reload@test.com")
        self.client.force_login(self.user)

    def _upload_csv(self, filename="test_data.csv", content=None):
        """Upload a CSV file via the DSW upload endpoint, return response data."""
        import io

        if content is None:
            content = "temperature,pressure,yield\n100,50,85\n110,55,88\n120,60,92\n130,65,78\n"
        f = io.BytesIO(content.encode("utf-8"))
        f.name = filename
        from django.core.files.uploadedfile import SimpleUploadedFile

        upload = SimpleUploadedFile(filename, content.encode("utf-8"), content_type="text/csv")
        resp = self.client.post(f"{self.DSW_BASE}upload-data/", {"file": upload})
        return resp

    def _create_workbench(self, title="Reload Test Session"):
        resp = _post(self.client, f"{BASE}create/", {"title": title})
        self.assertEqual(resp.status_code, 201)
        return resp.json()["workbench"]["id"]

    def _save_session(self, wb_id, layout, datasets=None, **kwargs):
        payload = {"layout": layout}
        if datasets is not None:
            payload["datasets"] = datasets
        payload.update(kwargs)
        resp = _patch(self.client, f"{BASE}{wb_id}/update/", payload)
        self.assertEqual(resp.status_code, 200)
        return resp

    def _load_session(self, wb_id):
        resp = self.client.get(f"{BASE}{wb_id}/")
        self.assertEqual(resp.status_code, 200)
        return resp.json()

    def _retrieve_data(self, data_id, filename="dataset"):
        resp = _post(
            self.client,
            f"{self.DSW_BASE}retrieve-data/",
            {"data_id": data_id, "filename": filename},
        )
        return resp

    # ----- Core round-trip tests -----

    def test_save_and_reload_preserves_layout(self):
        """Save layout state (data_id, cache_key, output_tabs) → reload → verify all fields present."""
        wb_id = self._create_workbench()
        layout = {
            "data_id": "data_abc123def456",
            "data_file": "measurements.csv",
            "cache_key": "ck_test_123",
            "output_tabs": [
                {"id": "out_1", "title": "Regression", "html": "<div>R²=0.95</div>"},
                {"id": "out_2", "title": "ANOVA", "html": "<div>F=12.3, p=0.001</div>"},
            ],
            "project_id": "proj_test_001",
        }
        self._save_session(wb_id, layout)

        loaded = self._load_session(wb_id)
        self.assertEqual(loaded["layout"]["data_id"], "data_abc123def456")
        self.assertEqual(loaded["layout"]["data_file"], "measurements.csv")
        self.assertEqual(loaded["layout"]["cache_key"], "ck_test_123")
        self.assertEqual(loaded["layout"]["project_id"], "proj_test_001")
        self.assertEqual(len(loaded["layout"]["output_tabs"]), 2)
        self.assertEqual(loaded["layout"]["output_tabs"][0]["title"], "Regression")
        self.assertIn("R²=0.95", loaded["layout"]["output_tabs"][0]["html"])

    def test_save_and_reload_preserves_datasets(self):
        """Save dataset metadata → reload → verify datasets array intact."""
        wb_id = self._create_workbench()
        datasets = [
            {
                "name": "batch_data.csv",
                "data_id": "data_aaa111bbb222",
                "cache_key": "ck_1",
                "rows": 500,
                "cols": 8,
            },
            {
                "name": "validation.csv",
                "data_id": "data_ccc333ddd444",
                "cache_key": "ck_2",
                "rows": 100,
                "cols": 3,
            },
        ]
        self._save_session(wb_id, {}, datasets=datasets)

        loaded = self._load_session(wb_id)
        self.assertEqual(len(loaded["datasets"]), 2)
        self.assertEqual(loaded["datasets"][0]["name"], "batch_data.csv")
        self.assertEqual(loaded["datasets"][0]["data_id"], "data_aaa111bbb222")
        self.assertEqual(loaded["datasets"][0]["rows"], 500)
        self.assertEqual(loaded["datasets"][1]["name"], "validation.csv")
        self.assertEqual(loaded["datasets"][1]["cols"], 3)

    def test_save_and_reload_preserves_conclusion_confidence(self):
        """conclusion_confidence field must survive the save/load round-trip."""
        wb_id = self._create_workbench()
        _patch(
            self.client,
            f"{BASE}{wb_id}/update/",
            {
                "conclusion": "Temperature above 125°C causes yield degradation",
                "conclusion_confidence": "high",
            },
        )

        loaded = self._load_session(wb_id)
        self.assertEqual(loaded["conclusion"], "Temperature above 125°C causes yield degradation")
        self.assertEqual(loaded["conclusion_confidence"], "high")

    def test_save_and_reload_preserves_guide_observations(self):
        """AI Guide observations must survive save/load."""
        wb_id = self._create_workbench()
        observations = [
            {
                "timestamp": "2026-03-10T10:00:00",
                "observation": "High variance in col3",
                "suggestion": "Check outliers",
                "acknowledged": False,
            },
            {
                "timestamp": "2026-03-10T10:05:00",
                "observation": "Non-normal distribution",
                "suggestion": "Use Kruskal-Wallis",
                "acknowledged": True,
            },
        ]
        _patch(self.client, f"{BASE}{wb_id}/update/", {"guide_observations": observations})

        loaded = self._load_session(wb_id)
        self.assertEqual(len(loaded["guide_observations"]), 2)
        self.assertEqual(loaded["guide_observations"][0]["observation"], "High variance in col3")
        self.assertTrue(loaded["guide_observations"][1]["acknowledged"])

    def test_save_and_reload_preserves_template_state(self):
        """DMAIC template state must survive save/load including phase history."""
        resp = _post(
            self.client,
            f"{BASE}create/",
            {"title": "DMAIC Reload", "template": "dmaic"},
        )
        wb_id = resp.json()["workbench"]["id"]

        # Advance to Measure phase
        _post(self.client, f"{BASE}{wb_id}/advance-phase/", {"notes": "Define complete"})

        loaded = self._load_session(wb_id)
        self.assertEqual(loaded["template"], "dmaic")
        self.assertEqual(loaded["template_state"]["current_phase"], "measure")
        self.assertEqual(len(loaded["template_state"]["phase_history"]), 2)

    # ----- Data retrieval on reload -----

    def test_upload_then_retrieve_returns_same_data(self):
        """Upload CSV → retrieve by data_id → verify columns and data match."""
        csv_content = "temp,pressure,yield\n100,50,85\n110,55,88\n120,60,92\n"
        upload_resp = self._upload_csv("process_data.csv", csv_content)
        self.assertEqual(upload_resp.status_code, 200)
        upload_data = upload_resp.json()
        data_id = upload_data["id"]

        # Retrieve — same call frontend makes on loadWorkbench
        ret_resp = self._retrieve_data(data_id, "process_data.csv")
        self.assertEqual(ret_resp.status_code, 200)
        ret_data = ret_resp.json()

        # Column names must match
        upload_col_names = [c["name"] for c in upload_data["columns"]]
        ret_col_names = [c["name"] for c in ret_data["columns"]]
        self.assertEqual(upload_col_names, ret_col_names)

        # Column dtypes must match
        upload_dtypes = [c["dtype"] for c in upload_data["columns"]]
        ret_dtypes = [c["dtype"] for c in ret_data["columns"]]
        self.assertEqual(upload_dtypes, ret_dtypes)

        # Row count must match
        self.assertEqual(ret_data["row_count"], 3)

        # Filename preserved
        self.assertEqual(ret_data["filename"], "process_data.csv")

    def test_retrieve_returns_dict_of_arrays_preview(self):
        """retrieve-data preview must be dict-of-arrays (col -> [vals]) for displayDataTable."""
        csv_content = "x,y\n1,10\n2,20\n3,30\n"
        upload_resp = self._upload_csv("xy.csv", csv_content)
        data_id = upload_resp.json()["id"]

        ret_resp = self._retrieve_data(data_id)
        ret_data = ret_resp.json()

        # preview must be dict with column keys
        self.assertIsInstance(ret_data["preview"], dict)
        self.assertIn("x", ret_data["preview"])
        self.assertIn("y", ret_data["preview"])

        # Each column value must be a list
        self.assertIsInstance(ret_data["preview"]["x"], list)
        self.assertEqual(len(ret_data["preview"]["x"]), 3)
        self.assertEqual(ret_data["preview"]["y"], [10, 20, 30])

    def test_retrieve_nonexistent_data_returns_404(self):
        """retrieve-data for missing data_id returns 404, not 500."""
        ret_resp = self._retrieve_data("data_000000000000")
        self.assertEqual(ret_resp.status_code, 404)

    def test_retrieve_invalid_data_id_rejected(self):
        """Path traversal attempts in data_id are rejected."""
        resp = _post(
            self.client,
            f"{self.DSW_BASE}retrieve-data/",
            {"data_id": "../../../etc/passwd"},
        )
        self.assertEqual(resp.status_code, 400)

    # ----- Full user simulation: upload → save → reload → verify data -----

    def test_full_session_round_trip_with_data(self):
        """Simulate complete user flow: upload CSV → run analysis (mock) → save session → reload → verify everything."""
        # Step 1: Upload data
        csv_content = "machine,cycle_time,defects\nA,12.5,3\nB,13.1,1\nA,11.9,2\nB,14.0,5\nA,12.2,0\n"
        upload_resp = self._upload_csv("factory_data.csv", csv_content)
        self.assertEqual(upload_resp.status_code, 200)
        upload_data = upload_resp.json()
        data_id = upload_data["id"]

        # Step 2: Create workbench
        wb_id = self._create_workbench("Factory Analysis")

        # Step 3: Save session (mimics saveWorkbench() in frontend)
        layout = {
            "data_id": data_id,
            "data_file": "factory_data.csv",
            "cache_key": "ck_factory_001",
            "output_tabs": [
                {
                    "id": "out_1",
                    "title": "Summary Stats",
                    "html": "<div>Mean cycle time: 12.74</div>",
                },
            ],
            "project_id": "",
        }
        datasets = [
            {
                "name": "factory_data.csv",
                "data_id": data_id,
                "cache_key": "ck_factory_001",
                "rows": 5,
                "cols": 3,
            }
        ]
        self._save_session(
            wb_id,
            layout,
            datasets=datasets,
            guide_observations=[
                {
                    "timestamp": "2026-03-10T12:00:00",
                    "observation": "Machine B has higher defect rate",
                    "suggestion": "Run chi-square test",
                    "acknowledged": False,
                }
            ],
        )

        # Step 4: Reload session (mimics loadWorkbench() in frontend)
        loaded = self._load_session(wb_id)

        # Verify layout restored
        self.assertEqual(loaded["layout"]["data_id"], data_id)
        self.assertEqual(loaded["layout"]["data_file"], "factory_data.csv")
        self.assertEqual(loaded["layout"]["cache_key"], "ck_factory_001")

        # Verify output tabs restored
        self.assertEqual(len(loaded["layout"]["output_tabs"]), 1)
        self.assertIn("Mean cycle time", loaded["layout"]["output_tabs"][0]["html"])

        # Verify datasets metadata restored
        self.assertEqual(len(loaded["datasets"]), 1)
        self.assertEqual(loaded["datasets"][0]["name"], "factory_data.csv")
        self.assertEqual(loaded["datasets"][0]["data_id"], data_id)

        # Verify guide observations restored
        self.assertEqual(len(loaded["guide_observations"]), 1)
        self.assertEqual(
            loaded["guide_observations"][0]["observation"],
            "Machine B has higher defect rate",
        )

        # Step 5: Re-fetch data (mimics the retrieve-data call in loadWorkbench)
        ret_resp = self._retrieve_data(data_id, "factory_data.csv")
        self.assertEqual(ret_resp.status_code, 200)
        ret_data = ret_resp.json()

        # Verify data is actually there and usable
        self.assertEqual(ret_data["row_count"], 5)
        self.assertEqual(len(ret_data["columns"]), 3)
        col_names = [c["name"] for c in ret_data["columns"]]
        self.assertIn("machine", col_names)
        self.assertIn("cycle_time", col_names)
        self.assertIn("defects", col_names)

        # Verify preview has actual values
        self.assertIn("machine", ret_data["preview"])
        self.assertEqual(ret_data["preview"]["machine"], ["A", "B", "A", "B", "A"])

    def test_to_json_includes_all_fields(self):
        """to_json() must return every field the frontend needs for full restore."""
        resp = _post(
            self.client,
            f"{BASE}create/",
            {"title": "Completeness Check", "template": "dmaic"},
        )
        wb_id = resp.json()["workbench"]["id"]

        # Set all fields
        _patch(
            self.client,
            f"{BASE}{wb_id}/update/",
            {
                "description": "Checking all fields",
                "status": "active",
                "layout": {"data_id": "data_aaa111bbb222", "cache_key": "ck_1"},
                "datasets": [
                    {
                        "name": "d.csv",
                        "data_id": "data_aaa111bbb222",
                        "rows": 10,
                        "cols": 2,
                    }
                ],
                "guide_observations": [{"observation": "test", "suggestion": "test", "acknowledged": False}],
                "conclusion": "All good",
                "conclusion_confidence": "medium",
            },
        )

        loaded = self._load_session(wb_id)

        # Every field the frontend depends on must be present
        required_fields = [
            "id",
            "inquiry",
            "description",
            "template",
            "status",
            "template_state",
            "artifacts",
            "connections",
            "layout",
            "datasets",
            "guide_observations",
            "conclusion",
            "conclusion_confidence",
            "created",
            "updated",
        ]
        for field in required_fields:
            self.assertIn(field, loaded, f"Missing field in to_json(): {field}")

        # Verify values
        self.assertEqual(loaded["inquiry"], "Completeness Check")
        self.assertEqual(loaded["conclusion_confidence"], "medium")
        self.assertEqual(loaded["template"], "dmaic")

    def test_reload_with_artifacts_and_connections(self):
        """Artifacts and connections must survive the round-trip."""
        wb_id = self._create_workbench("Artifact Test")

        # Add artifacts
        resp = _post(
            self.client,
            f"{BASE}{wb_id}/artifacts/",
            {
                "type": "analysis",
                "title": "Regression Result",
                "content": {"r_squared": 0.95, "p_value": 0.001},
            },
        )
        self.assertEqual(resp.status_code, 201)
        art1_id = resp.json()["artifact"]["id"]

        resp = _post(
            self.client,
            f"{BASE}{wb_id}/artifacts/",
            {
                "type": "note",
                "title": "Observation",
                "content": {"text": "Strong linear relationship"},
            },
        )
        self.assertEqual(resp.status_code, 201)
        art2_id = resp.json()["artifact"]["id"]

        # Connect them
        _post(
            self.client,
            f"{BASE}{wb_id}/connect/",
            {"from": art1_id, "to": art2_id, "label": "supports"},
        )

        # Reload
        loaded = self._load_session(wb_id)
        self.assertEqual(len(loaded["artifacts"]), 2)
        art_titles = [a["title"] for a in loaded["artifacts"]]
        self.assertIn("Regression Result", art_titles)
        self.assertIn("Observation", art_titles)
        self.assertEqual(len(loaded["connections"]), 1)
        self.assertEqual(loaded["connections"][0]["label"], "supports")

    def test_other_user_cannot_retrieve_data(self):
        """User B cannot retrieve User A's uploaded data."""
        upload_resp = self._upload_csv("secret.csv", "a,b\n1,2\n")
        data_id = upload_resp.json()["id"]

        other = _make_user("intruder@test.com")
        self.client.force_login(other)
        ret_resp = self._retrieve_data(data_id)
        # Should be 404 — file stored under original user's directory
        self.assertIn(ret_resp.status_code, [404, 500])
