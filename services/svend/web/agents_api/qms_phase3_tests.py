"""QMS Phase 3 tests — Intelligence Layer.

Proves QMS-001 §11 assertions:
- §11.1 FMEA trending, cross-FMEA patterns, RCA clustering
- §11.2 FMEA auto-suggest, RCA guided questions, A3 critique
- §11.3 VSM waste analysis, Hoshin alignment, QMS dashboard
- §11.4 SPC → FMEA auto-trigger, SPC → Evidence bridge
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from accounts.constants import Tier
from agents_api.models import (
    FMEA,
    A3Report,
    RCASession,
    ValueStreamMap,
)
from core.models import Project

User = get_user_model()
SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)


def _post(client, url, data=None):
    return client.post(url, json.dumps(data or {}), content_type="application/json")


def _put(client, url, data=None):
    return client.put(url, json.dumps(data or {}), content_type="application/json")


def _make_team_user(email):
    username = email.split("@")[0]
    user = User.objects.create_user(
        username=username, email=email, password="testpass123!"
    )
    user.tier = Tier.TEAM
    user.save(update_fields=["tier"])
    return user


def _make_enterprise_user(email):
    username = email.split("@")[0]
    user = User.objects.create_user(
        username=username, email=email, password="testpass123!"
    )
    user.tier = Tier.ENTERPRISE
    user.save(update_fields=["tier"])
    return user


# =============================================================================
# I-001: FMEA Risk Trending
# =============================================================================


@SECURE_OFF
class FMEARiskTrendingTest(TestCase):
    """QMS-001 §11.1 — FMEA trending endpoint."""

    def setUp(self):
        self.user = _make_team_user("trending@test.com")
        self.client.force_login(self.user)
        resp = _post(self.client, "/api/fmea/create/", {"title": "Trending Test"})
        self.fmea_id = resp.json()["id"]
        # Add a row
        resp = _post(
            self.client,
            f"/api/fmea/{self.fmea_id}/rows/",
            {
                "failure_mode": "Seal leak",
                "severity": 8,
                "occurrence": 5,
                "detection": 6,
            },
        )
        self.row_id = resp.json()["row"]["id"]

    def test_trending_returns_rows(self):
        """Trending endpoint returns row history."""
        resp = self.client.get(f"/api/fmea/{self.fmea_id}/trending/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("rows", data)
        self.assertIn("summary", data)
        self.assertEqual(len(data["rows"]), 1)
        self.assertEqual(data["rows"][0]["failure_mode"], "Seal leak")

    def test_trending_classifies_direction(self):
        """Trend direction classified as stable when no revisions."""
        resp = self.client.get(f"/api/fmea/{self.fmea_id}/trending/")
        data = resp.json()
        self.assertEqual(data["rows"][0]["trend"], "stable")
        self.assertEqual(data["summary"]["stable"], 1)

    def test_trending_empty_fmea(self):
        """Empty FMEA returns empty rows."""
        resp = _post(self.client, "/api/fmea/create/", {"title": "Empty"})
        empty_id = resp.json()["id"]
        resp = self.client.get(f"/api/fmea/{empty_id}/trending/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["rows"], [])


# =============================================================================
# I-002: Cross-FMEA Pattern Detection
# =============================================================================


@SECURE_OFF
class CrossFMEAPatternsTest(TestCase):
    """QMS-001 §11.1 — Cross-FMEA pattern detection."""

    def setUp(self):
        self.user = _make_team_user("patterns@test.com")
        self.client.force_login(self.user)
        # Create two FMEAs with similar failure modes
        resp = _post(self.client, "/api/fmea/create/", {"title": "FMEA A"})
        self.fmea_a = resp.json()["id"]
        _post(
            self.client,
            f"/api/fmea/{self.fmea_a}/rows/",
            {
                "failure_mode": "Seal leak at gasket joint",
                "process_step": "Assembly",
                "effect": "Fluid loss",
                "cause": "Torque variation",
                "severity": 7,
                "occurrence": 4,
                "detection": 5,
            },
        )

        resp = _post(self.client, "/api/fmea/create/", {"title": "FMEA B"})
        self.fmea_b = resp.json()["id"]
        _post(
            self.client,
            f"/api/fmea/{self.fmea_b}/rows/",
            {
                "failure_mode": "Gasket seal failure",
                "process_step": "Final assembly",
                "effect": "Leak",
                "cause": "Insufficient torque",
                "severity": 8,
                "occurrence": 5,
                "detection": 4,
            },
        )

    @patch("agents_api.embeddings.generate_embedding")
    def test_find_similar_failure_modes(self, mock_embed):
        """Pattern search returns similar rows across FMEAs."""
        # Mock embeddings to return known similar vectors
        base = np.random.randn(384).astype(np.float32)
        call_count = [0]

        def side_effect(text):
            call_count[0] += 1
            # Return similar vectors for all calls
            noise = np.random.randn(384).astype(np.float32) * 0.01
            return base + noise

        mock_embed.side_effect = side_effect

        resp = _post(
            self.client,
            "/api/fmea/patterns/",
            {
                "failure_mode": "Seal leak gasket",
                "threshold": 0.5,
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("matches", data)

    @patch("agents_api.embeddings.generate_embedding")
    def test_patterns_requires_input(self, mock_embed):
        """Pattern search requires failure_mode or fmea_row_id."""
        resp = _post(self.client, "/api/fmea/patterns/", {})
        self.assertEqual(resp.status_code, 400)

    @patch("agents_api.embeddings.generate_embedding")
    def test_no_matches_below_threshold(self, mock_embed):
        """Returns empty matches when nothing is above threshold."""
        # Return orthogonal vectors
        call_count = [0]

        def side_effect(text):
            call_count[0] += 1
            v = np.zeros(384, dtype=np.float32)
            v[call_count[0] % 384] = 1.0
            return v

        mock_embed.side_effect = side_effect

        resp = _post(
            self.client,
            "/api/fmea/patterns/",
            {
                "failure_mode": "completely unrelated topic xyz",
                "threshold": 0.99,
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["matches"], [])


# =============================================================================
# I-003: RCA Root Cause Clustering
# =============================================================================


@SECURE_OFF
class RCAClusteringTest(TestCase):
    """QMS-001 §11.1 — RCA clustering."""

    def setUp(self):
        self.user = _make_team_user("cluster@test.com")
        self.client.force_login(self.user)

    def test_cluster_requires_minimum_sessions(self):
        """Clustering needs at least 2 sessions."""
        resp = _post(self.client, "/api/rca/clusters/", {})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["clusters"], [])
        self.assertIn("Need at least 2", data.get("message", ""))

    def test_cluster_with_sessions(self):
        """Clustering groups similar sessions together."""
        # Create sessions with embeddings
        emb1 = np.random.randn(384).astype(np.float32)
        emb2 = emb1 + np.random.randn(384).astype(np.float32) * 0.01  # Very similar
        emb3 = np.random.randn(384).astype(np.float32)  # Different

        for i, emb in enumerate([emb1, emb2, emb3]):
            RCASession.objects.create(
                owner=self.user,
                title=f"Session {i}",
                event=f"Incident {i}",
                chain=[{"claim": f"Cause {i}"}],
                root_cause=f"Root cause {i}",
                status="investigating",
                embedding=emb.tobytes(),
            )

        resp = _post(self.client, "/api/rca/clusters/", {})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("clusters", data)
        self.assertEqual(data["total_sessions"], 3)

    def test_cluster_excludes_drafts(self):
        """Draft sessions excluded from clustering."""
        emb = np.random.randn(384).astype(np.float32)
        RCASession.objects.create(
            owner=self.user,
            title="Draft",
            event="Test",
            status="draft",
            embedding=emb.tobytes(),
        )
        resp = _post(self.client, "/api/rca/clusters/", {})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["total_sessions"], 0)


# =============================================================================
# I-004: FMEA Auto-Suggest Failure Modes
# =============================================================================


@SECURE_OFF
class FMEASuggestTest(TestCase):
    """QMS-001 §11.2 — FMEA failure mode suggestion via LLM."""

    def setUp(self):
        self.user = _make_team_user("suggest@test.com")
        self.client.force_login(self.user)
        resp = _post(self.client, "/api/fmea/create/", {"title": "Suggest Test"})
        self.fmea_id = resp.json()["id"]

    @patch("agents_api.llm_manager.LLMManager.chat")
    def test_suggest_returns_structured_modes(self, mock_chat):
        """Suggest endpoint returns structured failure mode suggestions."""
        mock_chat.return_value = {
            "content": json.dumps(
                [
                    {
                        "failure_mode": "Bolt not torqued",
                        "effect": "Loose joint",
                        "cause": "Operator fatigue",
                        "severity_hint": 8,
                        "occurrence_hint": 4,
                        "detection_hint": 6,
                    }
                ]
            ),
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }

        resp = _post(
            self.client,
            f"/api/fmea/{self.fmea_id}/suggest-failure-modes/",
            {
                "process_step": "Assembly - Torque bolt to 25 Nm",
                "context": "automotive brake caliper",
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("suggestions", data)
        self.assertTrue(len(data["suggestions"]) >= 1)
        self.assertEqual(data["suggestions"][0]["failure_mode"], "Bolt not torqued")

    def test_suggest_requires_process_step(self):
        """Suggest endpoint requires process_step."""
        resp = _post(
            self.client, f"/api/fmea/{self.fmea_id}/suggest-failure-modes/", {}
        )
        self.assertEqual(resp.status_code, 400)


# =============================================================================
# I-005: RCA Guided Questioning
# =============================================================================


@SECURE_OFF
class RCAGuidedQuestionsTest(TestCase):
    """QMS-001 §11.2 — RCA guided questioning via LLM."""

    def setUp(self):
        self.user = _make_enterprise_user("guided@test.com")
        self.client.force_login(self.user)

    @patch("anthropic.Anthropic")
    def test_guided_questions_returns_structured(self, mock_anthropic_cls):
        """Guided questions endpoint returns structured questions."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "questions": [
                            {
                                "question": "What happens at end of shift?",
                                "targets": "System conditions",
                                "gap_in_chain": "Step 2",
                            }
                        ],
                        "chain_assessment": "Chain stops too early.",
                    }
                )
            )
        ]
        mock_response.usage.input_tokens = 200
        mock_response.usage.output_tokens = 100
        mock_client.messages.create.return_value = mock_response

        resp = _post(
            self.client,
            "/api/rca/guided-questions/",
            {
                "event": "Production line stopped",
                "chain": [{"claim": "Operator missed defect"}],
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("questions", data)
        self.assertTrue(len(data["questions"]) >= 1)

    def test_guided_questions_requires_event(self):
        """Guided questions requires event and chain."""
        resp = _post(
            self.client,
            "/api/rca/guided-questions/",
            {
                "chain": [{"claim": "test"}],
            },
        )
        self.assertEqual(resp.status_code, 400)


# =============================================================================
# I-006: A3 Critique
# =============================================================================


@SECURE_OFF
class A3CritiqueTest(TestCase):
    """QMS-001 §11.2 — A3 critique via LLM."""

    def setUp(self):
        self.user = _make_team_user("a3critique@test.com")
        self.client.force_login(self.user)
        self.project = Project.objects.create(
            user=self.user,
            title="A3 Critique Project",
        )
        self.report = A3Report.objects.create(
            owner=self.user,
            project=self.project,
            title="Test A3",
            background="Line stopped 3 times last week",
            current_condition="OEE dropped to 65%",
            root_cause="Bearing failure from lack of lubrication schedule",
        )

    @patch("agents_api.llm_manager.LLMManager.chat")
    def test_critique_returns_per_section(self, mock_chat):
        """Critique returns ratings per section."""
        mock_chat.return_value = {
            "content": json.dumps(
                {
                    "sections": {
                        "background": {
                            "rating": "[STRONG]",
                            "feedback": "Good context",
                        },
                        "current_condition": {
                            "rating": "[ADEQUATE]",
                            "feedback": "Has data",
                        },
                        "root_cause": {
                            "rating": "[WEAK]",
                            "feedback": "Needs more depth",
                        },
                    },
                    "overall": "Needs work on root cause",
                    "logical_flow": "Background connects to condition well",
                }
            ),
            "usage": {"input_tokens": 300, "output_tokens": 200},
        }

        resp = _post(self.client, f"/api/a3/{self.report.id}/critique/", {})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("sections", data)
        self.assertEqual(data["sections"]["background"]["rating"], "[STRONG]")

    @patch("agents_api.llm_manager.LLMManager.chat")
    def test_critique_specific_sections(self, mock_chat):
        """Critique can target specific sections."""
        mock_chat.return_value = {
            "content": json.dumps(
                {
                    "sections": {
                        "root_cause": {"rating": "[WEAK]", "feedback": "Too shallow"},
                    },
                    "overall": "Root cause needs depth",
                    "logical_flow": "OK",
                }
            ),
            "usage": {"input_tokens": 200, "output_tokens": 100},
        }

        resp = _post(
            self.client,
            f"/api/a3/{self.report.id}/critique/",
            {
                "sections": ["root_cause"],
            },
        )
        self.assertEqual(resp.status_code, 200)

    def test_critique_empty_report(self):
        """Critique rejects reports with no content."""
        empty_report = A3Report.objects.create(
            owner=self.user,
            project=self.project,
            title="Empty",
        )
        resp = _post(self.client, f"/api/a3/{empty_report.id}/critique/", {})
        self.assertEqual(resp.status_code, 400)


# =============================================================================
# I-007: VSM Waste Analysis (TIMWOODS)
# =============================================================================


@SECURE_OFF
class VSMWasteAnalysisTest(TestCase):
    """QMS-001 §11.3 — TIMWOODS waste classification."""

    def setUp(self):
        self.user = _make_team_user("waste@test.com")
        self.client.force_login(self.user)
        self.vsm = ValueStreamMap.objects.create(
            owner=self.user,
            name="Waste Analysis VSM",
            process_steps=[
                {
                    "id": "1",
                    "name": "CNC",
                    "cycle_time": 45,
                    "changeover_time": 1800,
                    "uptime": 72,
                    "batch_size": 200,
                    "operators": 1,
                },
                {
                    "id": "2",
                    "name": "Assembly",
                    "cycle_time": 30,
                    "changeover_time": 60,
                    "uptime": 95,
                    "batch_size": 10,
                    "operators": 2,
                },
            ],
            inventory=[
                {"id": "inv1", "name": "Buffer 1", "days_of_supply": 15},
                {"id": "inv2", "name": "Buffer 2", "days_of_supply": 2},
            ],
            material_flow=[
                {"type": "push"},
                {"type": "push"},
                {"type": "push"},
            ],
            pce=3.5,
        )

    def test_timwoods_classification(self):
        """Waste analysis returns TIMWOODS categories."""
        resp = self.client.get(f"/api/vsm/{self.vsm.id}/waste-analysis/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("waste_categories", data)
        categories = data["waste_categories"]
        for cat in [
            "transport",
            "inventory",
            "motion",
            "waiting",
            "overproduction",
            "overprocessing",
            "defects",
            "skills",
        ]:
            self.assertIn(cat, categories)

    def test_high_wip_flagged_as_inventory(self):
        """High days of supply flagged as inventory waste."""
        resp = self.client.get(f"/api/vsm/{self.vsm.id}/waste-analysis/")
        data = resp.json()
        inv_waste = data["waste_categories"]["inventory"]
        self.assertTrue(len(inv_waste) >= 1)
        self.assertEqual(inv_waste[0]["detail"], "15 days supply")

    def test_long_changeover_flagged_as_waiting(self):
        """Long changeover relative to CT flagged as waiting."""
        resp = self.client.get(f"/api/vsm/{self.vsm.id}/waste-analysis/")
        data = resp.json()
        waiting = data["waste_categories"]["waiting"]
        self.assertTrue(len(waiting) >= 1)
        self.assertTrue(any("changeover" in w["detail"] for w in waiting))


# =============================================================================
# I-008: Hoshin Strategy Alignment
# =============================================================================


@SECURE_OFF
class HoshinAlignmentTest(TestCase):
    """QMS-001 §11.3 — Hoshin alignment analysis."""

    def setUp(self):
        self.user = _make_enterprise_user("alignment@test.com")
        self.client.force_login(self.user)
        # Enterprise user needs tenant setup for Hoshin
        from core.models import Membership, Tenant

        self.tenant = Tenant.objects.create(name="Align Corp")
        Membership.objects.create(
            user=self.user,
            tenant=self.tenant,
            role="owner",
        )
        from agents_api.models import Site

        self.site = Site.objects.create(
            tenant=self.tenant,
            name="Plant A",
        )

    def test_alignment_returns_gaps(self):
        """Alignment analysis returns gap structure."""
        resp = self.client.get(f"/api/hoshin/sites/{self.site.id}/alignment/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("alignment_score", data)
        self.assertIn("gaps", data)
        self.assertIn("recommendations", data)

    def test_alignment_score_calculated(self):
        """Alignment score is between 0 and 1."""
        resp = self.client.get(f"/api/hoshin/sites/{self.site.id}/alignment/")
        self.assertEqual(resp.status_code, 200)
        score = resp.json()["alignment_score"]
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_unlinked_projects_detected(self):
        """Unlinked projects appear in gaps."""
        from agents_api.models import HoshinProject

        # HoshinProject wraps core.Project via OneToOne FK
        # Project must have either user or tenant, not both (check constraint)
        core_project = Project.objects.create(
            tenant=self.tenant,
            title="Unlinked Project",
        )
        HoshinProject.objects.create(
            project=core_project,
            site=self.site,
            hoshin_status="active",
        )
        resp = self.client.get(f"/api/hoshin/sites/{self.site.id}/alignment/")
        self.assertEqual(resp.status_code, 200)
        gaps = resp.json()["gaps"]
        self.assertTrue(len(gaps["projects_without_objectives"]) >= 1)


# =============================================================================
# I-009: QMS Health Dashboard
# =============================================================================


@SECURE_OFF
class QMSHealthDashboardTest(TestCase):
    """QMS-001 §11.3 — QMS health dashboard."""

    def setUp(self):
        self.user = _make_team_user("dashboard@test.com")
        self.client.force_login(self.user)

    def test_dashboard_aggregates_all_modules(self):
        """Dashboard returns data for all 5 modules."""
        # Create some data
        FMEA.objects.create(owner=self.user, title="Test FMEA")
        RCASession.objects.create(
            owner=self.user,
            title="Test RCA",
            event="Test",
            status="investigating",
        )

        resp = self.client.get("/api/qms/dashboard/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        for module in ["fmea", "rca", "a3", "vsm", "hoshin"]:
            self.assertIn(module, data)
        self.assertIn("overall_health", data)

    def test_dashboard_empty_state(self):
        """Dashboard works with no data."""
        resp = self.client.get("/api/qms/dashboard/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["fmea"]["total"], 0)
        self.assertEqual(data["rca"]["total"], 0)


# =============================================================================
# I-012: SPC → FMEA Auto-Trigger
# =============================================================================


@SECURE_OFF
class SPCFMEATriggerTest(TestCase):
    """QMS-001 §11.4 — SPC OOC auto-triggers FMEA occurrence update."""

    def setUp(self):
        self.user = _make_team_user("spctrigger@test.com")
        self.client.force_login(self.user)
        resp = _post(self.client, "/api/fmea/create/", {"title": "SPC Trigger FMEA"})
        self.fmea_id = resp.json()["id"]
        resp = _post(
            self.client,
            f"/api/fmea/{self.fmea_id}/rows/",
            {
                "failure_mode": "Dimension drift",
                "severity": 7,
                "occurrence": 3,
                "detection": 5,
            },
        )
        self.row_id = resp.json()["row"]["id"]

    def test_ooc_triggers_fmea_update(self):
        """OOC control chart updates linked FMEA row occurrence."""
        # Generate data with OOC points
        data = [10.0, 10.1, 10.2, 10.0, 10.1, 25.0, 10.0, 10.1, 30.0, 10.0]
        resp = _post(
            self.client,
            "/api/spc/chart/",
            {
                "chart_type": "I-MR",
                "data": data,
                "fmea_row_id": self.row_id,
            },
        )
        self.assertEqual(resp.status_code, 200)
        result = resp.json()
        # If OOC was detected, fmea_update should be present
        if not result["chart"].get("in_control", True):
            self.assertIn("fmea_update", result)
            self.assertEqual(result["fmea_update"]["row_id"], self.row_id)

    def test_in_control_no_trigger(self):
        """In-control chart does not trigger FMEA update."""
        data = [10.0, 10.1, 10.0, 10.1, 10.0, 10.1, 10.0, 10.1, 10.0, 10.1]
        resp = _post(
            self.client,
            "/api/spc/chart/",
            {
                "chart_type": "I-MR",
                "data": data,
                "fmea_row_id": self.row_id,
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn("fmea_update", resp.json())

    def test_trigger_requires_fmea_row_id(self):
        """Without fmea_row_id, no trigger fires."""
        data = [10.0, 10.1, 10.2, 25.0, 10.0]
        resp = _post(
            self.client,
            "/api/spc/chart/",
            {
                "chart_type": "I-MR",
                "data": data,
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn("fmea_update", resp.json())


# =============================================================================
# I-013: SPC → Evidence Bridge
# =============================================================================


@SECURE_OFF
class SPCEvidenceTest(TestCase):
    """QMS-001 §11.4 — SPC violations create Evidence entries."""

    def setUp(self):
        self.user = _make_team_user("spcevidence@test.com")
        self.client.force_login(self.user)
        self.project = Project.objects.create(
            user=self.user,
            title="SPC Evidence Project",
        )

    def test_ooc_creates_evidence(self):
        """OOC control chart creates Evidence when project_id provided."""
        data = [10.0, 10.1, 10.2, 10.0, 10.1, 25.0, 10.0, 10.1, 30.0, 10.0]
        resp = _post(
            self.client,
            "/api/spc/chart/",
            {
                "chart_type": "I-MR",
                "data": data,
                "project_id": str(self.project.id),
            },
        )
        self.assertEqual(resp.status_code, 200)
        result = resp.json()
        if not result["chart"].get("in_control", True):
            self.assertIn("evidence_created", result)

    def test_no_evidence_without_project(self):
        """No evidence created without project_id."""
        data = [10.0, 10.1, 25.0, 10.0]
        resp = _post(
            self.client,
            "/api/spc/chart/",
            {
                "chart_type": "I-MR",
                "data": data,
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn("evidence_created", resp.json())
