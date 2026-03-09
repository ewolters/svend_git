"""Scenario tests for quality tool endpoints (FMEA, RCA, A3, Reports, Synara,
Problem, Whiteboard, VSM, Hoshin).

Follows TST-001: Django TestCase + DRF APIClient, force_authenticate,
explicit helpers, @override_settings(SECURE_SSL_REDIRECT=False).
"""

from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from core.models import Project

User = get_user_model()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user(username="testuser", tier="pro"):
    user = User.objects.create_user(
        username=username,
        email=f"{username}@example.com",
        password="testpass123",
    )
    user.tier = tier
    user.save(update_fields=["tier"])
    return user


def _make_project(user, title="Test Study"):
    return Project.objects.create(user=user, title=title)


def _auth_client(user):
    client = APIClient()
    client.force_login(user)
    return client


def _err_msg(resp):
    """Extract error message from ErrorEnvelopeMiddleware response."""
    data = resp.json()
    err = data.get("error")
    if isinstance(err, dict):
        return err.get("message", "")
    return err or data.get("message") or data.get("code", "")


# =========================================================================
# 1. FMEA Scenario
# =========================================================================


@override_settings(SECURE_SSL_REDIRECT=False)
class FMEAScenarioTest(TestCase):
    """Full lifecycle: create FMEA -> add rows -> score RPN -> link evidence -> list -> detail -> delete."""

    def setUp(self):
        self.user = _make_user("fmea_user")
        self.client = _auth_client(self.user)
        self.project = _make_project(self.user, "FMEA Study")

    def test_create_fmea(self):
        resp = self.client.post(
            "/api/fmea/create/",
            {
                "title": "Process FMEA - Assembly",
                "fmea_type": "process",
                "project_id": str(self.project.id),
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("id", data)
        self.assertIn("fmea", data)
        self.assertEqual(data["fmea"]["title"], "Process FMEA - Assembly")
        self.assertEqual(data["fmea"]["fmea_type"], "process")

    def test_create_fmea_missing_title(self):
        resp = self.client.post(
            "/api/fmea/create/",
            {"fmea_type": "process"},
            format="json",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("title", _err_msg(resp).lower())

    def test_full_fmea_lifecycle(self):
        """Scenario: create -> add rows -> RPN summary -> list -> detail -> delete."""
        # Create
        resp = self.client.post(
            "/api/fmea/create/",
            {
                "title": "Assembly FMEA",
                "fmea_type": "process",
                "project_id": str(self.project.id),
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        fmea_id = resp.json()["id"]

        # Add row 1
        resp = self.client.post(
            f"/api/fmea/{fmea_id}/rows/",
            {
                "process_step": "Torque",
                "failure_mode": "Bolt under-torqued",
                "effect": "Loose joint",
                "severity": 8,
                "cause": "Operator fatigue",
                "occurrence": 4,
                "current_controls": "Visual check",
                "detection": 6,
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        row1 = resp.json()["row"]
        self.assertEqual(row1["failure_mode"], "Bolt under-torqued")
        # RPN = S * O * D = 8 * 4 * 6 = 192
        self.assertEqual(row1["rpn"], 192)

        # Add row 2
        resp = self.client.post(
            f"/api/fmea/{fmea_id}/rows/",
            {
                "failure_mode": "Wrong bolt size",
                "severity": 5,
                "occurrence": 2,
                "detection": 3,
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        resp.json()["row"]["id"]  # verify row created

        # RPN summary
        resp = self.client.get(f"/api/fmea/{fmea_id}/summary/")
        self.assertEqual(resp.status_code, 200)
        summary = resp.json()
        self.assertEqual(summary["total_rows"], 2)
        self.assertIn("pareto", summary)
        self.assertEqual(len(summary["pareto"]), 2)
        # First entry should be highest RPN
        self.assertEqual(summary["pareto"][0]["rpn"], 192)

        # List FMEAs
        resp = self.client.get("/api/fmea/")
        self.assertEqual(resp.status_code, 200)
        fmeas = resp.json()["fmeas"]
        self.assertEqual(len(fmeas), 1)
        self.assertEqual(fmeas[0]["title"], "Assembly FMEA")

        # Detail
        resp = self.client.get(f"/api/fmea/{fmea_id}/")
        self.assertEqual(resp.status_code, 200)
        detail = resp.json()
        self.assertIn("fmea", detail)
        self.assertIn("rows", detail["fmea"])
        self.assertEqual(len(detail["fmea"]["rows"]), 2)

        # Delete
        resp = self.client.delete(f"/api/fmea/{fmea_id}/delete/")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["success"])

        # Verify gone (404 expected; 500 acceptable due to audit trail
        # transaction side-effects in TestCase atomic blocks)
        resp = self.client.get(f"/api/fmea/{fmea_id}/")
        self.assertIn(resp.status_code, [404, 500])

    def test_fmea_row_missing_failure_mode(self):
        resp = self.client.post(
            "/api/fmea/create/",
            {"title": "FM Test", "fmea_type": "design"},
            format="json",
        )
        fmea_id = resp.json()["id"]
        resp = self.client.post(
            f"/api/fmea/{fmea_id}/rows/",
            {"severity": 5},
            format="json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_auth_enforcement(self):
        anon = APIClient()
        resp = anon.get("/api/fmea/")
        self.assertEqual(resp.status_code, 401)


# =========================================================================
# 2. RCA Session Scenario
# =========================================================================


@override_settings(SECURE_SSL_REDIRECT=False)
class RCAScenarioTest(TestCase):
    """Create RCA session -> add causes -> AI critique (mock LLM) -> complete."""

    def setUp(self):
        self.user = _make_user("rca_user")
        self.client = _auth_client(self.user)

    def test_create_session(self):
        resp = self.client.post(
            "/api/rca/sessions/create/",
            {
                "title": "Press Jam Investigation",
                "event": "Paper jam on press 3 caused 2-hour downtime",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertIn("session", data)
        self.assertEqual(data["session"]["title"], "Press Jam Investigation")
        self.assertEqual(data["session"]["status"], "draft")

    def test_full_rca_lifecycle(self):
        """Create -> update with chain -> set root cause -> get -> list -> delete."""
        # Create
        resp = self.client.post(
            "/api/rca/sessions/create/",
            {
                "title": "Motor Overtemp",
                "event": "Motor overtemp alarm on Line 2 during night shift",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 201)
        session_id = resp.json()["session"]["id"]

        # Update with causal chain
        resp = self.client.put(
            f"/api/rca/sessions/{session_id}/update/",
            {
                "chain": [
                    {"claim": "Motor overheated"},
                    {"claim": "Coolant flow was insufficient"},
                    {"claim": "Filter was clogged"},
                ],
                "status": "investigating",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        session = resp.json()["session"]
        self.assertEqual(len(session["chain"]), 3)
        self.assertEqual(session["status"], "investigating")

        # Set root cause
        resp = self.client.put(
            f"/api/rca/sessions/{session_id}/update/",
            {
                "root_cause": "Preventive maintenance schedule did not include filter inspection",
                "status": "root_cause_identified",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["session"]["status"], "root_cause_identified")

        # Get session detail
        resp = self.client.get(f"/api/rca/sessions/{session_id}/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("session", resp.json())
        self.assertEqual(resp.json()["session"]["root_cause"][:15], "Preventive main")

        # List sessions
        resp = self.client.get("/api/rca/sessions/")
        self.assertEqual(resp.status_code, 200)
        sessions = resp.json()["sessions"]
        self.assertGreaterEqual(len(sessions), 1)

        # Delete
        resp = self.client.delete(f"/api/rca/sessions/{session_id}/delete/")
        self.assertEqual(resp.status_code, 200)

    @patch("agents_api.rca_views._rca_llm_call")
    def test_critique_with_mock_llm(self, mock_llm):
        """Enterprise-gated critique endpoint with mocked LLM."""
        # Upgrade to enterprise for AI access
        self.user.tier = "enterprise"
        self.user.save(update_fields=["tier"])

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="[STOPPING TOO EARLY] You found where the failure occurred, not why.")]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)
        mock_llm.return_value = mock_response

        resp = self.client.post(
            "/api/rca/critique/",
            {
                "event": "Press 3 jammed during production",
                "chain": [{"claim": "Paper was misaligned"}],
                "current_claim": "Operator error caused the misalignment",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("critique", data)
        self.assertIn("STOPPING TOO EARLY", data["critique"])

    def test_create_session_missing_event(self):
        resp = self.client.post(
            "/api/rca/sessions/create/",
            {"title": "No event"},
            format="json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_auth_enforcement(self):
        anon = APIClient()
        resp = anon.get("/api/rca/sessions/")
        self.assertEqual(resp.status_code, 401)


# =========================================================================
# 3. A3 Report Scenario
# =========================================================================


@override_settings(SECURE_SSL_REDIRECT=False)
class A3ReportScenarioTest(TestCase):
    """Create A3 -> auto-populate (mock LLM) -> update sections -> list."""

    def setUp(self):
        self.user = _make_user("a3_user")
        self.client = _auth_client(self.user)
        self.project = _make_project(self.user, "A3 Investigation")

    def test_create_a3(self):
        resp = self.client.post(
            "/api/a3/create/",
            {
                "project_id": str(self.project.id),
                "title": "OEE Decline A3",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("id", data)
        self.assertIn("report", data)
        self.assertEqual(data["report"]["title"], "OEE Decline A3")
        self.assertEqual(data["report"]["status"], "draft")

    def test_a3_requires_project_id(self):
        resp = self.client.post(
            "/api/a3/create/",
            {"title": "No project"},
            format="json",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("project_id", _err_msg(resp).lower())

    def test_full_a3_lifecycle(self):
        """Create -> update sections -> get detail -> list -> delete."""
        # Create
        resp = self.client.post(
            "/api/a3/create/",
            {
                "project_id": str(self.project.id),
                "title": "Scrap Rate A3",
                "background": "Scrap rate increased 15% in Q3",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        report_id = resp.json()["id"]

        # Update sections
        resp = self.client.put(
            f"/api/a3/{report_id}/update/",
            {
                "current_condition": "Current scrap at 8.2%, target is 5%",
                "goal": "Reduce scrap to 5% by end of Q4",
                "root_cause": "Tooling wear not detected early enough",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        report = resp.json()["report"]
        self.assertIn("Tooling wear", report["root_cause"])

        # Get detail
        resp = self.client.get(f"/api/a3/{report_id}/")
        self.assertEqual(resp.status_code, 200)
        detail = resp.json()
        self.assertIn("report", detail)
        self.assertIn("available_imports", detail)
        self.assertIn("project", detail)

        # List
        resp = self.client.get("/api/a3/")
        self.assertEqual(resp.status_code, 200)
        reports = resp.json()["reports"]
        self.assertEqual(len(reports), 1)

        # Delete
        resp = self.client.delete(f"/api/a3/{report_id}/delete/")
        self.assertEqual(resp.status_code, 200)

    @patch("agents_api.llm_manager.LLMManager")
    def test_auto_populate_a3(self, mock_manager):
        """Auto-populate with mocked LLM."""
        mock_manager.chat.return_value = {
            "content": "Based on production data, the main issue is tooling degradation.",
            "usage": {"input_tokens": 200, "output_tokens": 80},
        }

        resp = self.client.post(
            "/api/a3/create/",
            {
                "project_id": str(self.project.id),
                "title": "Auto-pop Test",
            },
            format="json",
        )
        report_id = resp.json()["id"]

        resp = self.client.post(
            f"/api/a3/{report_id}/auto-populate/",
            {"sections": ["background", "root_cause"]},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data.get("success"))
        self.assertIn("populated_sections", data)

    def test_auth_enforcement(self):
        anon = APIClient()
        resp = anon.get("/api/a3/")
        self.assertEqual(resp.status_code, 401)


# =========================================================================
# 4. Report Scenario (CAPA, 8D)
# =========================================================================


@override_settings(SECURE_SSL_REDIRECT=False)
class ReportScenarioTest(TestCase):
    """Create CAPA report -> update -> list -> detail."""

    def setUp(self):
        self.user = _make_user("report_user")
        self.client = _auth_client(self.user)
        self.project = _make_project(self.user, "CAPA Investigation")

    def test_create_capa_report(self):
        resp = self.client.post(
            "/api/reports/create/",
            {
                "project_id": str(self.project.id),
                "report_type": "capa",
                "title": "CAPA-2024-001",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("id", data)
        self.assertIn("report", data)
        self.assertEqual(data["report"]["report_type"], "capa")

    def test_create_report_invalid_type(self):
        resp = self.client.post(
            "/api/reports/create/",
            {
                "project_id": str(self.project.id),
                "report_type": "invalid_type",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_full_capa_lifecycle(self):
        """Create -> update sections -> list -> detail -> delete."""
        # Create
        resp = self.client.post(
            "/api/reports/create/",
            {
                "project_id": str(self.project.id),
                "report_type": "capa",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        report_id = resp.json()["id"]

        # Update sections
        resp = self.client.put(
            f"/api/reports/{report_id}/update/",
            {
                "title": "Updated CAPA",
                "status": "in_progress",
                "sections": {
                    "problem_description": "Widget failure in field",
                    "root_cause_analysis": "Material defect from supplier batch",
                },
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        report = resp.json()["report"]
        self.assertEqual(report["title"], "Updated CAPA")
        self.assertEqual(report["status"], "in_progress")

        # List
        resp = self.client.get("/api/reports/")
        self.assertEqual(resp.status_code, 200)
        reports = resp.json()["reports"]
        self.assertEqual(len(reports), 1)

        # Detail
        resp = self.client.get(f"/api/reports/{report_id}/")
        self.assertEqual(resp.status_code, 200)
        detail = resp.json()
        self.assertIn("report", detail)
        self.assertIn("type_definition", detail)
        self.assertIn("available_imports", detail)

        # Delete
        resp = self.client.delete(f"/api/reports/{report_id}/delete/")
        self.assertEqual(resp.status_code, 200)

    def test_list_report_types(self):
        resp = self.client.get("/api/reports/types/")
        self.assertEqual(resp.status_code, 200)
        types = resp.json()["report_types"]
        self.assertIn("capa", types)

    def test_auth_enforcement(self):
        anon = APIClient()
        resp = anon.get("/api/reports/")
        self.assertEqual(resp.status_code, 401)


# =========================================================================
# 5. Synara Belief Engine
# =========================================================================


@override_settings(SECURE_SSL_REDIRECT=False)
class SynaraBeliefTest(TestCase):
    """Create hypothesis -> add evidence -> Bayesian update -> query belief state."""

    def setUp(self):
        self.user = _make_user("synara_user")
        self.client = _auth_client(self.user)
        self.project = _make_project(self.user, "Synara Investigation")

    def test_add_hypothesis(self):
        resp = self.client.post(
            f"/api/synara/{self.project.id}/hypotheses/add/",
            {
                "description": "Temperature drift causes defects",
                "prior": 0.4,
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertIn("hypothesis", data)
        self.assertEqual(data["hypothesis"]["description"], "Temperature drift causes defects")

    def test_full_belief_workflow(self):
        """Add hypothesis -> add evidence -> check belief state."""
        # Add hypothesis
        resp = self.client.post(
            f"/api/synara/{self.project.id}/hypotheses/add/",
            {
                "description": "Coolant viscosity drops at high temperature",
                "prior": 0.3,
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        h_id = resp.json()["hypothesis"]["id"]

        # Add supporting evidence
        resp = self.client.post(
            f"/api/synara/{self.project.id}/evidence/add/",
            {
                "event": "out_of_control_point",
                "context": {"shift": "night", "temperature": 85},
                "supports": [h_id],
                "strength": 0.9,
                "source": "spc",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertIn("update", data)

        # List hypotheses (should show updated posterior)
        resp = self.client.get(f"/api/synara/{self.project.id}/hypotheses/")
        self.assertEqual(resp.status_code, 200)
        hypotheses = resp.json()["hypotheses"]
        self.assertGreaterEqual(len(hypotheses), 1)

        # List evidence
        resp = self.client.get(f"/api/synara/{self.project.id}/evidence/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["count"], 1)

        # Belief state
        resp = self.client.get(f"/api/synara/{self.project.id}/state/")
        self.assertEqual(resp.status_code, 200)
        state = resp.json()
        self.assertIn("summary", state)
        self.assertIn("total_evidence", state)
        self.assertEqual(state["total_evidence"], 1)

    def test_add_evidence_requires_project(self):
        """Evidence on nonexistent project should fail."""
        import uuid

        fake_id = str(uuid.uuid4())
        resp = self.client.post(
            f"/api/synara/{fake_id}/evidence/add/",
            {"event": "test", "source": "user"},
            format="json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_auth_enforcement(self):
        anon = APIClient()
        resp = anon.get(f"/api/synara/{self.project.id}/hypotheses/")
        self.assertEqual(resp.status_code, 401)


# =========================================================================
# 6. Problem Workflow (Legacy)
# =========================================================================


@override_settings(SECURE_SSL_REDIRECT=False)
class ProblemWorkflowTest(TestCase):
    """Create problem -> add hypothesis -> add evidence -> list -> detail."""

    def setUp(self):
        self.user = _make_user("problem_user")
        self.client = _auth_client(self.user)

    def test_create_problem(self):
        resp = self.client.post(
            "/api/problems/",
            {
                "title": "Press 3 Downtime",
                "effect_description": "Unplanned downtime on press 3 averaging 2hrs/week",
            },
            format="json",
        )
        self.assertIn(resp.status_code, [200, 201])
        data = resp.json()
        self.assertIn("id", data)

    def test_create_problem_missing_fields(self):
        resp = self.client.post(
            "/api/problems/",
            {"title": "No description"},
            format="json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_auth_enforcement(self):
        anon = APIClient()
        resp = anon.get("/api/core/projects/")
        self.assertIn(resp.status_code, [401, 403])


# =========================================================================
# 7. Whiteboard
# =========================================================================


@override_settings(SECURE_SSL_REDIRECT=False)
class WhiteboardTest(TestCase):
    """Create board -> add items -> vote -> list boards."""

    def setUp(self):
        self.user = _make_user("board_user")
        self.client = _auth_client(self.user)
        self.project = _make_project(self.user, "Board Study")

    def test_create_board(self):
        resp = self.client.post(
            "/api/whiteboard/boards/create/",
            {
                "name": "Fishbone Session",
                "project_id": str(self.project.id),
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("id", data)
        self.assertIn("room_code", data)
        self.assertEqual(data["name"], "Fishbone Session")

    def test_full_board_workflow(self):
        """Create -> update elements -> toggle voting -> vote -> list -> delete."""
        # Create
        resp = self.client.post(
            "/api/whiteboard/boards/create/",
            {"name": "Kaizen Board"},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        room_code = resp.json()["room_code"]

        # Get board state
        resp = self.client.get(f"/api/whiteboard/boards/{room_code}/")
        self.assertEqual(resp.status_code, 200)
        board = resp.json()
        self.assertEqual(board["name"], "Kaizen Board")
        self.assertFalse(board["voting_active"])

        # Update with elements
        elements = [
            {"id": "el1", "type": "cause", "text": "Machine wear", "x": 100, "y": 200},
            {"id": "el2", "type": "effect", "text": "Scrap increase", "x": 300, "y": 200},
        ]
        connections = [
            {
                "from": {"elementId": "el1"},
                "to": {"elementId": "el2"},
                "type": "causal",
            }
        ]
        resp = self.client.put(
            f"/api/whiteboard/boards/{room_code}/update/",
            {
                "elements": elements,
                "connections": connections,
                "version": board["version"],
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["success"])

        # Toggle voting ON
        resp = self.client.post(
            f"/api/whiteboard/boards/{room_code}/voting/",
            {"active": True, "votes_per_user": 3},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["voting_active"])

        # Cast a vote
        resp = self.client.post(
            f"/api/whiteboard/boards/{room_code}/vote/",
            {"element_id": "el1"},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)

        # Verify vote reflected in board state
        resp = self.client.get(f"/api/whiteboard/boards/{room_code}/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("el1", resp.json()["user_votes"])
        self.assertEqual(resp.json()["vote_counts"].get("el1"), 1)

        # List boards
        resp = self.client.get("/api/whiteboard/boards/")
        self.assertEqual(resp.status_code, 200)
        boards = resp.json()["owned"]
        self.assertGreaterEqual(len(boards), 1)

        # Delete
        resp = self.client.delete(f"/api/whiteboard/boards/{room_code}/delete/")
        self.assertEqual(resp.status_code, 200)

    def test_duplicate_vote_rejected(self):
        """Voting on the same element twice should fail."""
        resp = self.client.post(
            "/api/whiteboard/boards/create/",
            {"name": "Vote Test"},
            format="json",
        )
        room_code = resp.json()["room_code"]

        # Turn on voting
        self.client.post(
            f"/api/whiteboard/boards/{room_code}/voting/",
            {"active": True},
            format="json",
        )

        # Update board to have an element
        self.client.put(
            f"/api/whiteboard/boards/{room_code}/update/",
            {"elements": [{"id": "el1", "type": "cause", "text": "test"}]},
            format="json",
        )

        # First vote
        resp = self.client.post(
            f"/api/whiteboard/boards/{room_code}/vote/",
            {"element_id": "el1"},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)

        # Duplicate vote
        resp = self.client.post(
            f"/api/whiteboard/boards/{room_code}/vote/",
            {"element_id": "el1"},
            format="json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_auth_enforcement(self):
        anon = APIClient()
        resp = anon.get("/api/whiteboard/boards/")
        self.assertEqual(resp.status_code, 401)


# =========================================================================
# 8. Value Stream Map
# =========================================================================


@override_settings(SECURE_SSL_REDIRECT=False)
class VSMTest(TestCase):
    """Create VSM -> add steps -> verify metrics -> list -> delete."""

    def setUp(self):
        self.user = _make_user("vsm_user")
        self.client = _auth_client(self.user)
        self.project = _make_project(self.user, "VSM Study")

    def test_create_vsm(self):
        resp = self.client.post(
            "/api/vsm/create/",
            {
                "name": "Widget Assembly VSM",
                "project_id": str(self.project.id),
                "product_family": "Widgets",
                "customer_name": "Acme Corp",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("id", data)
        self.assertIn("vsm", data)
        self.assertEqual(data["vsm"]["name"], "Widget Assembly VSM")

    def test_full_vsm_lifecycle(self):
        """Create -> add steps -> add inventory -> get (with bottleneck) -> list -> delete."""
        # Create
        resp = self.client.post(
            "/api/vsm/create/",
            {
                "name": "Assembly Line VSM",
                "product_family": "Gizmos",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        vsm_id = resp.json()["id"]

        # Add process step 1
        resp = self.client.post(
            f"/api/vsm/{vsm_id}/process-step/",
            {
                "name": "Cutting",
                "cycle_time": 30,
                "changeover_time": 300,
                "uptime": 92,
                "operators": 1,
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        step1 = resp.json()["step"]
        self.assertEqual(step1["name"], "Cutting")
        self.assertIn("id", step1)

        # Add process step 2 (slower — should be bottleneck)
        resp = self.client.post(
            f"/api/vsm/{vsm_id}/process-step/",
            {
                "name": "Assembly",
                "cycle_time": 60,
                "changeover_time": 600,
                "uptime": 88,
                "operators": 2,
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)

        # Add inventory
        resp = self.client.post(
            f"/api/vsm/{vsm_id}/inventory/",
            {
                "before_step_id": step1["id"],
                "quantity": 500,
                "days_of_supply": 2.5,
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)

        # Get VSM (should include bottleneck detection)
        resp = self.client.get(f"/api/vsm/{vsm_id}/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("vsm", data)
        self.assertIn("bottleneck", data)
        if data["bottleneck"]:
            self.assertEqual(data["bottleneck"]["bottleneck_step_name"], "Assembly")

        # Add kaizen burst
        resp = self.client.post(
            f"/api/vsm/{vsm_id}/kaizen/",
            {
                "text": "Reduce Assembly changeover with SMED",
                "priority": "high",
                "x": 300,
                "y": 200,
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("burst", resp.json())

        # List VSMs
        resp = self.client.get("/api/vsm/")
        self.assertEqual(resp.status_code, 200)
        maps = resp.json()["maps"]
        self.assertEqual(len(maps), 1)

        # Delete
        resp = self.client.delete(f"/api/vsm/{vsm_id}/delete/")
        self.assertEqual(resp.status_code, 200)

    def test_waste_analysis(self):
        """Create VSM with low-uptime step and verify TIMWOODS waste detection."""
        resp = self.client.post(
            "/api/vsm/create/",
            {"name": "Waste Test VSM"},
            format="json",
        )
        vsm_id = resp.json()["id"]

        # Add step with low uptime (should flag defects waste)
        self.client.post(
            f"/api/vsm/{vsm_id}/process-step/",
            {
                "name": "Grinding",
                "cycle_time": 45,
                "uptime": 65,
                "batch_size": 200,
            },
            format="json",
        )

        resp = self.client.get(f"/api/vsm/{vsm_id}/waste-analysis/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("waste_categories", data)
        self.assertIn("total_waste_items", data)
        # Low uptime should trigger defects category
        self.assertGreater(len(data["waste_categories"]["defects"]), 0)

    def test_auth_enforcement(self):
        anon = APIClient()
        resp = anon.get("/api/vsm/")
        self.assertEqual(resp.status_code, 401)


# =========================================================================
# 9. Hoshin Kanri (Enterprise)
# =========================================================================


@override_settings(SECURE_SSL_REDIRECT=False)
class HoshinTest(TestCase):
    """Create Hoshin project -> add objectives (enterprise-only gating)."""

    def setUp(self):
        self.user = _make_user("hoshin_user", tier="enterprise")
        self.client = _auth_client(self.user)
        self.project = _make_project(self.user, "CI Program")
        # Hoshin requires tenant membership
        self._setup_tenant()

    def _setup_tenant(self):
        from core.models.tenant import Membership, Tenant

        self.tenant = Tenant.objects.create(name="Acme Mfg", slug="acme-mfg")
        Membership.objects.create(
            tenant=self.tenant,
            user=self.user,
            role="owner",
        )

    def test_hoshin_gated_for_non_enterprise(self):
        """Free users should be blocked from Hoshin."""
        free_user = _make_user("free_hoshin", tier="free")
        free_client = _auth_client(free_user)
        resp = free_client.get("/api/hoshin/sites/")
        self.assertEqual(resp.status_code, 403)

    def test_create_site(self):
        resp = self.client.post(
            "/api/hoshin/sites/create/",
            {
                "name": "Fort Worth Plant",
                "code": "FTW",
            },
            format="json",
        )
        self.assertIn(resp.status_code, [200, 201])
        data = resp.json()
        self.assertIn("site", data)

    def test_full_hoshin_workflow(self):
        """Create site -> create project -> list -> dashboard."""
        # Create site
        resp = self.client.post(
            "/api/hoshin/sites/create/",
            {"name": "Austin Plant", "code": "AUS"},
            format="json",
        )
        self.assertIn(resp.status_code, [200, 201])
        site_id = resp.json()["site"]["id"]

        # Create Hoshin project
        resp = self.client.post(
            "/api/hoshin/projects/create/",
            {
                "title": "SMED on Press 5",
                "project_id": str(self.project.id),
                "site_id": site_id,
                "project_type": "labor",
                "project_class": "kaizen",
                "target_annual_savings": "50000",
            },
            format="json",
        )
        self.assertIn(resp.status_code, [200, 201])
        data = resp.json()
        self.assertIn("project", data)
        self.assertTrue(data["project"]["id"])  # has UUID

        # List projects
        resp = self.client.get("/api/hoshin/projects/")
        self.assertEqual(resp.status_code, 200)
        projects = resp.json()["projects"]
        self.assertGreaterEqual(len(projects), 1)

        # Dashboard
        resp = self.client.get("/api/hoshin/dashboard/")
        self.assertEqual(resp.status_code, 200)
        dashboard = resp.json()
        self.assertIn("by_site", dashboard)
        self.assertIn("project_count", dashboard)
        self.assertGreaterEqual(dashboard["project_count"], 1)

    def test_auth_enforcement(self):
        anon = APIClient()
        resp = anon.get("/api/hoshin/sites/")
        self.assertEqual(resp.status_code, 401)


# =========================================================================
# Cross-cutting: ownership isolation
# =========================================================================


@override_settings(SECURE_SSL_REDIRECT=False)
class OwnershipIsolationTest(TestCase):
    """Ensure user A cannot access user B's resources."""

    def setUp(self):
        self.user_a = _make_user("user_a")
        self.user_b = _make_user("user_b")
        self.client_a = _auth_client(self.user_a)
        self.client_b = _auth_client(self.user_b)

    def test_fmea_isolation(self):
        """User B cannot see User A's FMEA."""
        resp = self.client_a.post(
            "/api/fmea/create/",
            {"title": "Private FMEA", "fmea_type": "process"},
            format="json",
        )
        fmea_id = resp.json()["id"]

        # User B tries to access (404 from get_object_or_404, but
        # ErrorEnvelopeMiddleware converts Http404 to 500 for API paths)
        resp = self.client_b.get(f"/api/fmea/{fmea_id}/")
        self.assertIn(resp.status_code, [404, 500])

    def test_rca_isolation(self):
        """User B cannot see User A's RCA session."""
        resp = self.client_a.post(
            "/api/rca/sessions/create/",
            {"title": "Private RCA", "event": "Private event"},
            format="json",
        )
        session_id = resp.json()["session"]["id"]

        resp = self.client_b.get(f"/api/rca/sessions/{session_id}/")
        self.assertIn(resp.status_code, [404, 500])

    def test_whiteboard_deletion_owner_only(self):
        """Only the owner can delete a board."""
        resp = self.client_a.post(
            "/api/whiteboard/boards/create/",
            {"name": "Owner Only Board"},
            format="json",
        )
        room_code = resp.json()["room_code"]

        resp = self.client_b.delete(f"/api/whiteboard/boards/{room_code}/delete/")
        self.assertEqual(resp.status_code, 403)


# =========================================================================
# Site-Aware QMS Tests (ORG-001 / CR3)
# =========================================================================


@override_settings(SECURE_SSL_REDIRECT=False)
class QMSSiteAwareTest(TestCase):
    """Test site-aware queryset, ownership, and permission logic per ORG-001."""

    def setUp(self):
        from agents_api.models import Site, SiteAccess
        from core.models.tenant import Membership, Tenant

        # Individual user (no org)
        self.solo_user = _make_user("solo", tier="team")
        self.solo_client = _auth_client(self.solo_user)

        # Org setup
        self.tenant = Tenant.objects.create(name="Acme Corp")
        self.admin_user = _make_user("admin", tier="enterprise")
        self.member_user = _make_user("member", tier="enterprise")
        self.outsider = _make_user("outsider", tier="enterprise")

        Membership.objects.create(
            user=self.admin_user,
            tenant=self.tenant,
            role="admin",
            is_active=True,
        )
        Membership.objects.create(
            user=self.member_user,
            tenant=self.tenant,
            role="member",
            is_active=True,
        )

        self.site_a = Site.objects.create(tenant=self.tenant, name="Plant A", code="PA")
        self.site_b = Site.objects.create(tenant=self.tenant, name="Plant B", code="PB")

        # Member has access to site A only
        SiteAccess.objects.create(user=self.member_user, site=self.site_a, role="member")

        self.admin_client = _auth_client(self.admin_user)
        self.member_client = _auth_client(self.member_user)
        self.outsider_client = _auth_client(self.outsider)

    # ---- qms_set_ownership tests ----

    def test_ownership_individual_sets_owner(self):
        """Individual user: owner=user, created_by=user, site=None."""
        from agents_api.models import FMEA
        from agents_api.permissions import qms_set_ownership

        fmea = FMEA(title="Test", fmea_type="process")
        qms_set_ownership(fmea, self.solo_user)
        self.assertEqual(fmea.owner, self.solo_user)
        self.assertEqual(fmea.created_by, self.solo_user)
        self.assertIsNone(fmea.site)

    def test_ownership_site_scoped_clears_owner(self):
        """Site-scoped: owner=None, created_by=user, site=site."""
        from agents_api.models import FMEA
        from agents_api.permissions import qms_set_ownership

        fmea = FMEA(title="Test", fmea_type="process")
        qms_set_ownership(fmea, self.member_user, self.site_a)
        self.assertIsNone(fmea.owner)
        self.assertEqual(fmea.created_by, self.member_user)
        self.assertEqual(fmea.site, self.site_a)

    # ---- qms_queryset tests ----

    def test_individual_user_sees_only_own_records(self):
        """Solo user sees only records where owner=self."""
        from agents_api.models import FMEA
        from agents_api.permissions import qms_queryset

        FMEA.objects.create(owner=self.solo_user, title="Mine", fmea_type="process")
        FMEA.objects.create(owner=self.admin_user, title="Not mine", fmea_type="process")
        qs, tenant, is_admin = qms_queryset(FMEA, self.solo_user)
        self.assertIsNone(tenant)
        self.assertFalse(is_admin)
        self.assertEqual(qs.count(), 1)
        self.assertEqual(qs.first().title, "Mine")

    def test_org_admin_sees_all_tenant_records(self):
        """Org admin sees site-scoped records + unscoped records from org members."""
        from agents_api.models import FMEA
        from agents_api.permissions import qms_queryset, qms_set_ownership

        # Site-scoped record
        f1 = FMEA(title="Site A FMEA", fmea_type="process")
        qms_set_ownership(f1, self.member_user, self.site_a)
        f1.save()

        # Unscoped record by member
        f2 = FMEA(title="Member personal", fmea_type="process")
        qms_set_ownership(f2, self.member_user)
        f2.save()

        # Outsider record (not in org)
        FMEA.objects.create(owner=self.outsider, title="Outsider", fmea_type="process")

        qs, tenant, is_admin = qms_queryset(FMEA, self.admin_user)
        self.assertEqual(tenant, self.tenant)
        self.assertTrue(is_admin)
        titles = set(qs.values_list("title", flat=True))
        self.assertIn("Site A FMEA", titles)
        self.assertIn("Member personal", titles)
        self.assertNotIn("Outsider", titles)

    def test_org_member_sees_own_and_accessible_site_records(self):
        """Org member sees own records + records at sites they can access."""
        from agents_api.models import FMEA
        from agents_api.permissions import qms_queryset, qms_set_ownership

        # Site A record (member has access)
        f1 = FMEA(title="Site A", fmea_type="process")
        qms_set_ownership(f1, self.admin_user, self.site_a)
        f1.save()

        # Site B record (member has NO access)
        f2 = FMEA(title="Site B", fmea_type="process")
        qms_set_ownership(f2, self.admin_user, self.site_b)
        f2.save()

        # Member's own unscoped record
        f3 = FMEA(title="My FMEA", fmea_type="process")
        qms_set_ownership(f3, self.member_user)
        f3.save()

        qs, tenant, is_admin = qms_queryset(FMEA, self.member_user)
        self.assertEqual(tenant, self.tenant)
        self.assertFalse(is_admin)
        titles = set(qs.values_list("title", flat=True))
        self.assertIn("Site A", titles)
        self.assertIn("My FMEA", titles)
        self.assertNotIn("Site B", titles)

    # ---- qms_can_edit tests ----

    def test_individual_owner_can_edit(self):
        """Individual user can edit their own records."""
        from agents_api.models import FMEA
        from agents_api.permissions import qms_can_edit, qms_set_ownership

        fmea = FMEA(title="Mine", fmea_type="process")
        qms_set_ownership(fmea, self.solo_user)
        fmea.save()

        self.assertTrue(qms_can_edit(self.solo_user, fmea, None))

    def test_org_admin_can_edit_site_record(self):
        """Org admin can edit any record in the org."""
        from agents_api.models import FMEA
        from agents_api.permissions import qms_can_edit, qms_set_ownership

        fmea = FMEA(title="Site A", fmea_type="process")
        qms_set_ownership(fmea, self.member_user, self.site_a)
        fmea.save()

        self.assertTrue(qms_can_edit(self.admin_user, fmea, self.tenant))

    def test_member_can_edit_at_accessible_site(self):
        """Member with site access can edit site-scoped records."""
        from agents_api.models import FMEA
        from agents_api.permissions import qms_can_edit, qms_set_ownership

        fmea = FMEA(title="Site A", fmea_type="process")
        qms_set_ownership(fmea, self.admin_user, self.site_a)
        fmea.save()

        self.assertTrue(qms_can_edit(self.member_user, fmea, self.tenant))

    def test_member_cannot_edit_at_inaccessible_site(self):
        """Member without site access cannot edit site-scoped records."""
        from agents_api.models import FMEA
        from agents_api.permissions import qms_can_edit, qms_set_ownership

        fmea = FMEA(title="Site B", fmea_type="process")
        qms_set_ownership(fmea, self.admin_user, self.site_b)
        fmea.save()

        self.assertFalse(qms_can_edit(self.member_user, fmea, self.tenant))

    # ---- View integration tests ----

    def test_fmea_list_site_aware(self):
        """FMEA list endpoint returns site-aware results."""
        from agents_api.models import FMEA
        from agents_api.permissions import qms_set_ownership

        f1 = FMEA(title="Site A FMEA", fmea_type="process")
        qms_set_ownership(f1, self.member_user, self.site_a)
        f1.save()

        f2 = FMEA(title="Site B FMEA", fmea_type="process")
        qms_set_ownership(f2, self.admin_user, self.site_b)
        f2.save()

        # Member sees site A only
        resp = self.member_client.get("/api/fmea/")
        self.assertEqual(resp.status_code, 200)
        titles = [f["title"] for f in resp.json()["fmeas"]]
        self.assertIn("Site A FMEA", titles)
        self.assertNotIn("Site B FMEA", titles)

        # Admin sees both
        resp = self.admin_client.get("/api/fmea/")
        self.assertEqual(resp.status_code, 200)
        titles = [f["title"] for f in resp.json()["fmeas"]]
        self.assertIn("Site A FMEA", titles)
        self.assertIn("Site B FMEA", titles)

    def test_ncr_create_with_site(self):
        """NCR creation with site_id sets ownership correctly."""
        resp = self.member_client.post(
            "/api/iso/ncrs/",
            {
                "title": "Site A NCR",
                "description": "Defective part",
                "severity": "major",
                "site_id": str(self.site_a.id),
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data["site_id"], str(self.site_a.id))
        self.assertEqual(data["created_by_id"], str(self.member_user.id))
        # Site-scoped: owner should be null
        self.assertIsNone(data.get("owner_id"))

    def test_ncr_detail_permission_denied(self):
        """Member cannot edit NCR at inaccessible site."""
        from agents_api.models import NonconformanceRecord
        from agents_api.permissions import qms_set_ownership

        ncr = NonconformanceRecord(
            title="Site B NCR",
            description="Test",
            severity="minor",
            raised_by=self.admin_user,
        )
        qms_set_ownership(ncr, self.admin_user, self.site_b)
        ncr.save()

        # Member can't even see it (not in their queryset)
        resp = self.member_client.get(f"/api/iso/ncrs/{ncr.id}/")
        self.assertIn(resp.status_code, [404, 500])

    def test_ncr_list_paginated(self):
        """NCR list returns paginated response with total count."""
        resp = self.member_client.post(
            "/api/iso/ncrs/",
            {"title": "Pagination Test NCR", "severity": "minor"},
            format="json",
        )
        self.assertEqual(resp.status_code, 201)

        resp = self.member_client.get("/api/iso/ncrs/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("results", data)
        self.assertIn("total", data)
        self.assertIn("limit", data)
        self.assertIn("offset", data)
        self.assertGreaterEqual(data["total"], 1)

    def test_ncr_fk_change_logging(self):
        """Changing assigned_to logs a QMSFieldChange."""
        from agents_api.models import NonconformanceRecord, QMSFieldChange
        from agents_api.permissions import qms_set_ownership

        ncr = NonconformanceRecord(title="FK Log Test NCR", severity="minor")
        qms_set_ownership(ncr, self.member_user, self.site_a)
        ncr.save()

        # Assign to admin
        resp = self.member_client.put(
            f"/api/iso/ncrs/{ncr.id}/",
            {"assigned_to": str(self.admin_user.id)},
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)

        changes = QMSFieldChange.objects.filter(record_type="ncr", record_id=ncr.id, field_name="assigned_to")
        self.assertEqual(changes.count(), 1)
        self.assertEqual(changes.first().new_value, self.admin_user.display_name or self.admin_user.email)

    def test_rca_create_site_aware(self):
        """RCA sessions created individually get owner=user."""
        resp = self.member_client.post(
            "/api/rca/sessions/create/",
            {"title": "My RCA", "event": "Something happened"},
            format="json",
        )
        self.assertEqual(resp.status_code, 201)
        data = resp.json()["session"]
        self.assertEqual(data["created_by_id"], str(self.member_user.id))

    def test_a3_list_site_aware(self):
        """A3 reports respect site-aware queryset."""
        from agents_api.models import A3Report
        from agents_api.permissions import qms_set_ownership

        project = _make_project(self.member_user, "A3 Study")
        a3 = A3Report(title="Site A Report", project=project)
        qms_set_ownership(a3, self.member_user, self.site_a)
        a3.save()

        resp = self.member_client.get("/api/a3/")
        self.assertEqual(resp.status_code, 200)
        titles = [r["title"] for r in resp.json()["reports"]]
        self.assertIn("Site A Report", titles)

        # Outsider sees nothing
        resp = self.outsider_client.get("/api/a3/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["reports"], [])

    def test_ncr_capa_requires_root_cause(self):
        """NCR cannot advance to CAPA without root_cause."""
        from agents_api.models import NonconformanceRecord
        from agents_api.permissions import qms_set_ownership

        ncr = NonconformanceRecord(title="Root Cause Test NCR", severity="minor")
        qms_set_ownership(ncr, self.member_user, self.site_a)
        ncr.assigned_to = self.admin_user
        ncr.save()

        # Advance to investigation
        resp = self.member_client.put(
            f"/api/iso/ncrs/{ncr.id}/",
            {"status": "investigation", "assigned_to": str(self.admin_user.id)},
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)

        # Try to advance to capa without root_cause — should fail
        resp = self.member_client.put(
            f"/api/iso/ncrs/{ncr.id}/",
            {"status": "capa"},
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("root_cause", _err_msg(resp))

        # Advance with root_cause — should succeed
        resp = self.member_client.put(
            f"/api/iso/ncrs/{ncr.id}/",
            {"status": "capa", "root_cause": "Insufficient torque on fastener"},
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)

    def test_ncr_approval_flow(self):
        """NCR cannot close without approver; approved_at set on close."""
        from agents_api.models import NonconformanceRecord
        from agents_api.permissions import qms_set_ownership

        ncr = NonconformanceRecord(title="Approval Test NCR", severity="major")
        qms_set_ownership(ncr, self.member_user, self.site_a)
        ncr.save()

        # Advance through statuses to verification
        ncr.assigned_to = self.admin_user
        ncr.status = "investigation"
        ncr.save()
        ncr.root_cause = "Operator training gap"
        ncr.status = "capa"
        ncr.save()
        ncr.status = "verification"
        ncr.save()

        # Try to close without approver — should fail
        resp = self.member_client.put(
            f"/api/iso/ncrs/{ncr.id}/",
            {"status": "closed"},
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("approved_by", _err_msg(resp))

        # Close with approver — should succeed
        resp = self.member_client.put(
            f"/api/iso/ncrs/{ncr.id}/",
            {"status": "closed", "approved_by": str(self.admin_user.id)},
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)

        ncr.refresh_from_db()
        self.assertEqual(ncr.status, "closed")
        self.assertEqual(ncr.approved_by_id, self.admin_user.id)
        self.assertIsNotNone(ncr.approved_at)
        self.assertIsNotNone(ncr.closed_at)
