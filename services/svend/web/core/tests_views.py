"""Functional scenario tests for core/views.py.

Covers all 37 public view symbols across 11 test classes.
Follows TST-001: Django TestCase + DRF APIClient, force_authenticate,
explicit helpers, @override_settings(SECURE_SSL_REDIRECT=False).
"""

from datetime import timedelta
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.utils import timezone
from rest_framework.test import APIClient

from accounts.constants import Tier
from core.models import (
    Dataset,
    Entity,
    Evidence,
    EvidenceLink,
    ExperimentDesign,
    Hypothesis,
    KnowledgeGraph,
    Membership,
    OrgInvitation,
    Project,
    Tenant,
)
from core.synara import ConsistencyIssue, UpdateResult

User = get_user_model()

SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user(email, tier=Tier.TEAM, **kwargs):
    """Create a user with given tier (default TEAM to pass rate_limited checks)."""
    username = email.split("@")[0]
    user = User.objects.create_user(
        username=username, email=email, password="testpass123", **kwargs
    )
    user.tier = tier
    user.save(update_fields=["tier"])
    return user


def _make_project(user, title="Yield Investigation", **kwargs):
    """Create a personal project owned by the given user."""
    return Project.objects.create(user=user, title=title, **kwargs)


def _make_hypothesis(project, statement="Temperature affects yield", **kwargs):
    """Create a hypothesis in the given project."""
    defaults = {
        "project": project,
        "statement": statement,
        "prior_probability": 0.5,
    }
    defaults.update(kwargs)
    return Hypothesis.objects.create(**defaults)


def _make_evidence(project, summary="Observed 5% yield drop", **kwargs):
    """Create evidence in the given project."""
    defaults = {
        "project": project,
        "summary": summary,
        "source_type": Evidence.SourceType.OBSERVATION,
        "confidence": 0.8,
    }
    defaults.update(kwargs)
    return Evidence.objects.create(**defaults)


def _make_tenant(name="Test Org", slug="test-org", plan=Tenant.Plan.TEAM, **kwargs):
    return Tenant.objects.create(name=name, slug=slug, plan=plan, **kwargs)


def _make_membership(tenant, user, role=Membership.Role.MEMBER, **kwargs):
    return Membership.objects.create(
        tenant=tenant, user=user, role=role, joined_at=timezone.now(), **kwargs
    )


def _make_graph(user):
    """Get or create a knowledge graph for the user."""
    graph, _ = KnowledgeGraph.objects.get_or_create(
        user=user, defaults={"name": f"{user.username}'s KG"}
    )
    return graph


# =========================================================================
# 1. ProjectLifecycleTest
# =========================================================================


@SECURE_OFF
class ProjectLifecycleTest(TestCase):
    """Scenario: create project -> list -> detail -> comment -> advance phase -> recalculate."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("lifecycle@example.com", Tier.TEAM)
        self.client.force_authenticate(self.user)

    def test_full_project_lifecycle(self):
        """Multi-step lifecycle: create, list, detail, comment, advance, recalculate."""
        # Step 1: Create project
        res = self.client.post(
            "/api/core/projects/",
            {
                "title": "Scrap Rate Reduction",
                "domain": "manufacturing",
                "methodology": "dmaic",
                "problem_statement": "Scrap rate is 12%, target 5%",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()
        project_id = data["id"]
        self.assertEqual(data["title"], "Scrap Rate Reduction")
        self.assertEqual(data["domain"], "manufacturing")
        self.assertEqual(data["methodology"], "dmaic")
        self.assertEqual(data["current_phase"], "define")
        self.assertEqual(data["status"], "active")
        self.assertTrue(Project.objects.filter(id=project_id).exists())

        # Step 2: Verify it appears in list
        res = self.client.get("/api/core/projects/")
        self.assertEqual(res.status_code, 200)
        titles = [p["title"] for p in res.json()]
        self.assertIn("Scrap Rate Reduction", titles)

        # Step 3: Get detail
        res = self.client.get(f"/api/core/projects/{project_id}/")
        self.assertEqual(res.status_code, 200)
        detail = res.json()
        self.assertEqual(detail["title"], "Scrap Rate Reduction")
        self.assertIn("hypotheses", detail)
        self.assertIn("datasets", detail)
        self.assertIn("experiment_designs", detail)

        # Step 4: Add comment
        res = self.client.post(
            f"/api/core/projects/{project_id}/comment/",
            {"text": "Initial stakeholder alignment complete."},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        changelog = res.json()["changelog"]
        self.assertTrue(len(changelog) >= 1)
        self.assertEqual(changelog[-1]["action"], "comment")
        self.assertIn("Initial stakeholder", changelog[-1]["detail"])

        # Step 5: Advance phase (define -> measure)
        res = self.client.post(
            f"/api/core/projects/{project_id}/advance-phase/",
            {"phase": "measure", "notes": "Define gate review passed"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["current_phase"], "measure")
        project = Project.objects.get(id=project_id)
        self.assertEqual(project.current_phase, "measure")
        self.assertTrue(len(project.phase_history) >= 1)

        # Step 6: Recalculate (mocked synara)
        with patch("core.views.synara") as mock_synara:
            mock_synara.recalculate_project.return_value = {
                "hypotheses_updated": 0,
                "status_changes": [],
            }
            res = self.client.post(f"/api/core/projects/{project_id}/recalculate/")
            self.assertEqual(res.status_code, 200)
            self.assertTrue(res.json()["success"])
            self.assertIn("project", res.json())

    def test_project_update_and_delete(self):
        """PUT to update and DELETE to remove a project."""
        project = _make_project(self.user, title="Old Title")
        pid = str(project.id)

        # Update
        res = self.client.put(
            f"/api/core/projects/{pid}/",
            {"title": "New Title", "status": "on_hold"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["title"], "New Title")
        self.assertEqual(res.json()["status"], "on_hold")

        # Delete
        res = self.client.delete(f"/api/core/projects/{pid}/")
        self.assertEqual(res.status_code, 204)
        self.assertFalse(Project.objects.filter(id=pid).exists())

    def test_project_list_filters_by_status(self):
        """GET /projects/?status=active returns only active projects."""
        _make_project(self.user, title="Active One", status="active")
        _make_project(self.user, title="Resolved One", status="resolved")

        res = self.client.get("/api/core/projects/?status=active")
        self.assertEqual(res.status_code, 200)
        titles = [p["title"] for p in res.json()]
        self.assertIn("Active One", titles)
        self.assertNotIn("Resolved One", titles)

    def test_advance_phase_rejects_skip(self):
        """Cannot skip from define to analyze (skipping measure)."""
        project = _make_project(self.user)
        res = self.client.post(
            f"/api/core/projects/{project.id}/advance-phase/",
            {"phase": "analyze"},
            format="json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("Cannot skip", res.json()["error"]["message"])

    def test_comment_requires_text(self):
        """Comment endpoint rejects empty text."""
        project = _make_project(self.user)
        res = self.client.post(
            f"/api/core/projects/{project.id}/comment/",
            {"text": ""},
            format="json",
        )
        self.assertEqual(res.status_code, 400)

    def test_unauthenticated_project_list_rejected(self):
        """Anonymous user gets 401/403 on project list."""
        self.client.force_authenticate(None)
        res = self.client.get("/api/core/projects/")
        self.assertIn(res.status_code, [401, 403])


# =========================================================================
# 2. ProjectHubTest
# =========================================================================


@SECURE_OFF
class ProjectHubTest(TestCase):
    """Test project_hub returns a unified view of all linked tools."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("hub@example.com", Tier.TEAM)
        self.client.force_authenticate(self.user)
        self.project = _make_project(self.user, title="Hub Project")

    def test_hub_returns_unified_structure(self):
        """project_hub returns project, tools, counts, evidence_summary, changelog."""
        res = self.client.get(f"/api/core/projects/{self.project.id}/hub/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("project", data)
        self.assertIn("tools", data)
        self.assertIn("counts", data)
        self.assertIn("evidence_summary", data)
        self.assertIn("changelog", data)
        self.assertIn("study_actions", data)

        # Verify tools sub-keys
        tools = data["tools"]
        for key in [
            "whiteboards",
            "dsw_analyses",
            "a3_reports",
            "vsm_maps",
            "ncrs",
            "rca_sessions",
            "reports",
            "fmeas",
        ]:
            self.assertIn(key, tools)

        # Verify counts sub-keys
        counts = data["counts"]
        for key in [
            "hypotheses",
            "datasets",
            "experiments",
            "whiteboards",
            "dsw_analyses",
            "a3_reports",
            "vsm_maps",
            "ncrs",
            "rca_sessions",
            "reports",
            "fmeas",
        ]:
            self.assertIn(key, counts)

    def test_hub_unauthenticated(self):
        """Hub endpoint requires authentication."""
        self.client.force_authenticate(None)
        res = self.client.get(f"/api/core/projects/{self.project.id}/hub/")
        self.assertIn(res.status_code, [401, 403])


# =========================================================================
# 3. HypothesisWorkflowTest
# =========================================================================


@SECURE_OFF
class HypothesisWorkflowTest(TestCase):
    """Scenario: create hypothesis -> list -> update -> link evidence -> recalculate."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("hyp@example.com", Tier.TEAM)
        self.client.force_authenticate(self.user)
        self.project = _make_project(self.user, title="Hypothesis Project")
        self.pid = str(self.project.id)

    def test_hypothesis_crud_and_bayesian_update(self):
        """Full hypothesis lifecycle with Bayesian evidence application."""
        # Step 1: Create hypothesis
        res = self.client.post(
            f"/api/core/projects/{self.pid}/hypotheses/",
            {
                "statement": "If temperature rises, then yield drops",
                "if_clause": "temperature rises above 200C",
                "then_clause": "yield drops below 90%",
                "because_clause": "thermal degradation of catalyst",
                "prior_probability": 0.5,
            },
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        hyp_data = res.json()
        hyp_id = hyp_data["id"]
        self.assertEqual(
            hyp_data["statement"], "If temperature rises, then yield drops"
        )
        self.assertAlmostEqual(float(hyp_data["prior_probability"]), 0.5)
        self.assertEqual(hyp_data["status"], "active")

        # Step 2: List hypotheses
        res = self.client.get(f"/api/core/projects/{self.pid}/hypotheses/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()), 1)
        self.assertEqual(res.json()[0]["id"], hyp_id)

        # Step 3: Update hypothesis
        res = self.client.put(
            f"/api/core/projects/{self.pid}/hypotheses/{hyp_id}/",
            {"rationale": "Historical data shows temperature sensitivity"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        # Detail serializer includes evidence_links
        self.assertIn("evidence_links", res.json())

        # Step 4: Create evidence in project
        ev_res = self.client.post(
            f"/api/core/projects/{self.pid}/evidence/",
            {
                "summary": "Temp correlation r=0.85",
                "source_type": "analysis",
                "confidence": 0.9,
            },
            format="json",
        )
        self.assertEqual(ev_res.status_code, 201)
        ev_id = ev_res.json()["id"]

        # Step 5: Link evidence to hypothesis (with apply=False to skip synara)
        res = self.client.post(
            f"/api/core/projects/{self.pid}/hypotheses/{hyp_id}/link-evidence/",
            {
                "evidence_id": ev_id,
                "likelihood_ratio": 3.0,
                "reasoning": "Strong correlation supports causal link",
                "apply": False,
            },
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        self.assertIn("link", res.json())
        self.assertTrue(res.json()["created"])

        # Verify EvidenceLink was created in DB
        self.assertTrue(
            EvidenceLink.objects.filter(
                hypothesis_id=hyp_id, evidence_id=ev_id
            ).exists()
        )
        link = EvidenceLink.objects.get(hypothesis_id=hyp_id, evidence_id=ev_id)
        self.assertAlmostEqual(link.likelihood_ratio, 3.0)
        self.assertEqual(link.direction, "supports")

        # Step 6: Link evidence with apply=True (mock synara)
        with patch("core.views.synara") as mock_synara:
            mock_synara.apply_evidence.return_value = UpdateResult(
                hypothesis_id=str(hyp_id),
                prior_probability=0.5,
                posterior_probability=0.75,
                likelihood_ratio=3.0,
                adjusted_lr=2.8,
                evidence_id=str(ev_id),
                status_changed=False,
                new_status=None,
            )
            res = self.client.post(
                f"/api/core/projects/{self.pid}/hypotheses/{hyp_id}/link-evidence/",
                {
                    "evidence_id": ev_id,
                    "likelihood_ratio": 3.0,
                    "apply": True,
                },
                format="json",
            )
            self.assertEqual(res.status_code, 200)
            update = res.json()["update_result"]
            self.assertAlmostEqual(update["prior"], 0.5)
            self.assertAlmostEqual(update["posterior"], 0.75)
            self.assertIn("hypothesis", res.json())

        # Step 7: Recalculate single hypothesis (mock synara)
        with patch("core.views.synara") as mock_synara:
            mock_synara.recalculate_hypothesis.return_value = 0.8
            res = self.client.post(
                f"/api/core/projects/{self.pid}/hypotheses/{hyp_id}/recalculate/"
            )
            self.assertEqual(res.status_code, 200)
            self.assertTrue(res.json()["success"])
            self.assertIn("hypothesis", res.json())

    def test_hypothesis_detail_returns_evidence_links(self):
        """GET hypothesis detail includes evidence_links, supporting, opposing."""
        hyp = _make_hypothesis(self.project)
        ev = _make_evidence(self.project, summary="Supporting observation")
        EvidenceLink.objects.create(
            hypothesis=hyp, evidence=ev, likelihood_ratio=2.0, is_manual=True
        )

        res = self.client.get(f"/api/core/projects/{self.pid}/hypotheses/{hyp.id}/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("evidence_links", data)
        self.assertIn("supporting_evidence", data)
        self.assertIn("opposing_evidence", data)
        self.assertEqual(len(data["evidence_links"]), 1)
        self.assertEqual(len(data["supporting_evidence"]), 1)

    def test_hypothesis_delete(self):
        """DELETE a hypothesis removes it and logs event."""
        hyp = _make_hypothesis(self.project)
        res = self.client.delete(f"/api/core/projects/{self.pid}/hypotheses/{hyp.id}/")
        self.assertEqual(res.status_code, 204)
        self.assertFalse(Hypothesis.objects.filter(id=hyp.id).exists())
        # Verify changelog
        self.project.refresh_from_db()
        actions = [e["action"] for e in self.project.changelog]
        self.assertIn("hypothesis_removed", actions)

    def test_unauthenticated_hypothesis_list(self):
        """Anonymous user gets rejected from hypothesis endpoints."""
        self.client.force_authenticate(None)
        res = self.client.get(f"/api/core/projects/{self.pid}/hypotheses/")
        self.assertIn(res.status_code, [401, 403])


# =========================================================================
# 4. EvidenceManagementTest
# =========================================================================


@SECURE_OFF
class EvidenceManagementTest(TestCase):
    """CRUD for evidence + linking to hypotheses."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("evidence@example.com", Tier.TEAM)
        self.client.force_authenticate(self.user)
        self.project = _make_project(self.user, title="Evidence Project")
        self.pid = str(self.project.id)

    def test_evidence_create_list_detail(self):
        """Create evidence, verify in list, retrieve detail."""
        # Create
        res = self.client.post(
            f"/api/core/projects/{self.pid}/evidence/",
            {
                "summary": "SPC chart shows out-of-control process",
                "source_type": "observation",
                "confidence": 0.85,
                "details": "X-bar chart violation on subgroup 12",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        ev_id = res.json()["id"]
        self.assertEqual(
            res.json()["summary"], "SPC chart shows out-of-control process"
        )
        self.assertAlmostEqual(float(res.json()["confidence"]), 0.85)

        # Verify in DB
        self.assertTrue(Evidence.objects.filter(id=ev_id).exists())

        # Detail
        res = self.client.get(f"/api/core/projects/{self.pid}/evidence/{ev_id}/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(
            res.json()["summary"], "SPC chart shows out-of-control process"
        )

    def test_evidence_create_with_hypothesis_link(self):
        """Create evidence pre-linked to a hypothesis."""
        hyp = _make_hypothesis(self.project)
        res = self.client.post(
            f"/api/core/projects/{self.pid}/evidence/",
            {
                "summary": "Regression p-value < 0.001",
                "source_type": "analysis",
                "confidence": 0.95,
                "hypothesis_ids": [str(hyp.id)],
                "likelihood_ratios": {str(hyp.id): 5.0},
            },
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        ev_id = res.json()["id"]

        # Verify EvidenceLink created
        self.assertTrue(
            EvidenceLink.objects.filter(hypothesis=hyp, evidence_id=ev_id).exists()
        )
        link = EvidenceLink.objects.get(hypothesis=hyp, evidence_id=ev_id)
        self.assertAlmostEqual(link.likelihood_ratio, 5.0)

    def test_evidence_update(self):
        """PUT updates evidence fields."""
        ev = _make_evidence(self.project)
        res = self.client.put(
            f"/api/core/projects/{self.pid}/evidence/{ev.id}/",
            {"confidence": 0.95, "details": "Updated methodology"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        ev.refresh_from_db()
        self.assertAlmostEqual(ev.confidence, 0.95)

    def test_evidence_delete(self):
        """DELETE evidence removes it."""
        ev = _make_evidence(self.project)
        res = self.client.delete(f"/api/core/projects/{self.pid}/evidence/{ev.id}/")
        self.assertEqual(res.status_code, 204)
        self.assertFalse(Evidence.objects.filter(id=ev.id).exists())

    def test_evidence_list_includes_linked_evidence(self):
        """GET /evidence/ returns evidence linked to project hypotheses."""
        hyp = _make_hypothesis(self.project)
        ev = _make_evidence(self.project, summary="Linked evidence")
        EvidenceLink.objects.create(hypothesis=hyp, evidence=ev, likelihood_ratio=1.5)

        res = self.client.get(f"/api/core/projects/{self.pid}/evidence/")
        self.assertEqual(res.status_code, 200)
        summaries = [e["summary"] for e in res.json()]
        self.assertIn("Linked evidence", summaries)

    def test_link_evidence_requires_evidence_id(self):
        """link_evidence rejects missing evidence_id."""
        hyp = _make_hypothesis(self.project)
        res = self.client.post(
            f"/api/core/projects/{self.pid}/hypotheses/{hyp.id}/link-evidence/",
            {"likelihood_ratio": 2.0},
            format="json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("evidence_id", res.json()["error"]["message"])


# =========================================================================
# 5. DatasetWorkflowTest
# =========================================================================


@SECURE_OFF
class DatasetWorkflowTest(TestCase):
    """Create dataset -> list -> detail -> get data."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("dataset@example.com", Tier.TEAM)
        self.client.force_authenticate(self.user)
        self.project = _make_project(self.user, title="Dataset Project")
        self.pid = str(self.project.id)

    def test_create_dataset_with_inline_data(self):
        """Create a dataset with inline JSON data."""
        inline_data = [
            {"temperature": 200, "yield": 92},
            {"temperature": 210, "yield": 88},
            {"temperature": 220, "yield": 85},
        ]
        res = self.client.post(
            f"/api/core/projects/{self.pid}/datasets/",
            {
                "name": "Temperature Study",
                "description": "Temp vs yield data",
                "data_type": "json",
                "data": inline_data,
            },
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()
        ds_id = data["id"]
        self.assertEqual(data["name"], "Temperature Study")
        self.assertEqual(data["row_count"], 3)
        self.assertTrue(Dataset.objects.filter(id=ds_id).exists())

        # List
        res = self.client.get(f"/api/core/projects/{self.pid}/datasets/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()), 1)
        self.assertEqual(res.json()[0]["name"], "Temperature Study")

        # Detail
        res = self.client.get(f"/api/core/projects/{self.pid}/datasets/{ds_id}/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["name"], "Temperature Study")
        # Inline data gets a preview
        self.assertIn("preview", res.json())

        # Get full data
        res = self.client.get(f"/api/core/projects/{self.pid}/datasets/{ds_id}/data/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("data", res.json())
        self.assertIn("columns", res.json())
        self.assertEqual(len(res.json()["data"]), 3)

    def test_dataset_with_no_data_returns_empty(self):
        """Dataset with no file and no data returns empty arrays."""
        ds = Dataset.objects.create(
            project=self.project,
            name="Empty Dataset",
            data_type="csv",
            uploaded_by=self.user,
        )
        res = self.client.get(f"/api/core/projects/{self.pid}/datasets/{ds.id}/data/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["data"], [])
        self.assertEqual(res.json()["columns"], [])

    def test_dataset_delete(self):
        """DELETE a dataset removes it."""
        ds = Dataset.objects.create(
            project=self.project, name="To Delete", uploaded_by=self.user
        )
        res = self.client.delete(f"/api/core/projects/{self.pid}/datasets/{ds.id}/")
        self.assertEqual(res.status_code, 204)
        self.assertFalse(Dataset.objects.filter(id=ds.id).exists())


# =========================================================================
# 6. ExperimentDesignTest
# =========================================================================


@SECURE_OFF
class ExperimentDesignTest(TestCase):
    """Create design -> list -> detail -> review execution."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("design@example.com", Tier.TEAM)
        self.client.force_authenticate(self.user)
        self.project = _make_project(self.user, title="DOE Project")
        self.pid = str(self.project.id)

    def test_design_crud(self):
        """Create, list, detail, update, delete experiment designs."""
        # Create
        res = self.client.post(
            f"/api/core/projects/{self.pid}/designs/",
            {
                "name": "2^3 Full Factorial",
                "description": "Temperature, pressure, time",
                "design_type": "full_factorial",
                "num_runs": 8,
                "factors": [
                    {"name": "Temperature", "levels": [180, 220]},
                    {"name": "Pressure", "levels": [50, 100]},
                    {"name": "Time", "levels": [30, 60]},
                ],
                "responses": [{"name": "Yield"}],
            },
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()
        design_id = data["id"]
        self.assertEqual(data["name"], "2^3 Full Factorial")
        self.assertEqual(data["design_type"], "full_factorial")
        self.assertEqual(data["num_runs"], 8)
        self.assertEqual(data["status"], "planned")

        # List
        res = self.client.get(f"/api/core/projects/{self.pid}/designs/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()), 1)

        # Detail
        res = self.client.get(f"/api/core/projects/{self.pid}/designs/{design_id}/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("result_datasets", res.json())

        # Update
        res = self.client.put(
            f"/api/core/projects/{self.pid}/designs/{design_id}/",
            {"status": "in_progress"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "in_progress")

        # Delete
        res = self.client.delete(f"/api/core/projects/{self.pid}/designs/{design_id}/")
        self.assertEqual(res.status_code, 204)
        self.assertFalse(ExperimentDesign.objects.filter(id=design_id).exists())

    def test_review_design_execution(self):
        """Review execution with inline data."""
        design = ExperimentDesign.objects.create(
            project=self.project,
            name="Screening Design",
            design_type="full_factorial",
            num_runs=4,
            factors=[
                {"name": "Temperature", "levels": [180, 220]},
                {"name": "Pressure", "levels": [50, 100]},
            ],
            responses=[{"name": "Yield"}],
            design_spec={
                "runs": [
                    {"Temperature": 180, "Pressure": 50},
                    {"Temperature": 220, "Pressure": 50},
                    {"Temperature": 180, "Pressure": 100},
                    {"Temperature": 220, "Pressure": 100},
                ]
            },
        )
        actual_data = [
            {"Temperature": 180, "Pressure": 50, "Yield": 92},
            {"Temperature": 220, "Pressure": 50, "Yield": 88},
            {"Temperature": 180, "Pressure": 100, "Yield": 95},
            {"Temperature": 220, "Pressure": 100, "Yield": 90},
        ]
        res = self.client.post(
            f"/api/core/projects/{self.pid}/designs/{design.id}/review/",
            {"data": actual_data},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertIn("review", data)
        review = data["review"]
        self.assertIn("overall_score", review)
        self.assertIn("grade", review)
        self.assertIn("scores", review)
        self.assertIn("coverage", review["scores"])
        self.assertIn("balance", review["scores"])

        # Verify design was updated in DB
        design.refresh_from_db()
        self.assertEqual(design.status, "reviewed")
        self.assertIsNotNone(design.execution_review)
        self.assertIsNotNone(design.execution_score)

    def test_review_requires_data(self):
        """Review endpoint rejects requests with no data."""
        design = ExperimentDesign.objects.create(
            project=self.project,
            name="Empty Review",
            design_type="full_factorial",
            num_runs=4,
            design_spec={"runs": [{"A": 1}, {"A": 2}]},
        )
        res = self.client.post(
            f"/api/core/projects/{self.pid}/designs/{design.id}/review/",
            {},
            format="json",
        )
        self.assertEqual(res.status_code, 400)


# =========================================================================
# 7. OrgManagementScenarioTest
# =========================================================================


@SECURE_OFF
class OrgManagementScenarioTest(TestCase):
    """Scenario: create org -> invite -> accept -> list members -> change role -> remove."""

    def setUp(self):
        self.client = APIClient()
        self.owner = _make_user("owner@example.com", Tier.TEAM)
        self.invitee = _make_user("invitee@example.com", Tier.TEAM)
        self.invitee.is_email_verified = True
        self.invitee.save(update_fields=["is_email_verified"])

    def test_full_org_lifecycle(self):
        """End-to-end org management workflow."""
        # Step 1: Create org
        self.client.force_authenticate(self.owner)
        res = self.client.post(
            "/api/core/org/create/",
            {"name": "Acme Manufacturing", "slug": "acme-mfg"},
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        org_data = res.json()
        self.assertTrue(org_data["success"])
        self.assertEqual(org_data["org"]["name"], "Acme Manufacturing")
        self.assertEqual(org_data["org"]["slug"], "acme-mfg")
        tenant = Tenant.objects.get(slug="acme-mfg")
        owner_membership = Membership.objects.get(tenant=tenant, user=self.owner)
        self.assertEqual(owner_membership.role, "owner")

        # Step 2: Verify org info
        res = self.client.get("/api/core/org/")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["has_org"])
        self.assertEqual(res.json()["org"]["name"], "Acme Manufacturing")
        self.assertEqual(res.json()["membership"]["role"], "owner")
        self.assertTrue(res.json()["membership"]["can_admin"])

        # Step 3: Invite member (mock billing)
        with patch("accounts.billing.add_org_seat", return_value=11):
            res = self.client.post(
                "/api/core/org/invite/",
                {"email": "invitee@example.com", "role": "member"},
                format="json",
            )
            self.assertEqual(res.status_code, 201)
            inv_data = res.json()
            self.assertTrue(inv_data["success"])
            token = inv_data["invitation"]["token"]
            self.assertTrue(inv_data["invitation"]["id"])  # has UUID

        # Step 4: List invitations
        res = self.client.get("/api/core/org/invitations/")
        self.assertEqual(res.status_code, 200)
        invitations = res.json()["invitations"]
        self.assertTrue(len(invitations) >= 1)
        self.assertEqual(invitations[0]["email"], "invitee@example.com")
        self.assertEqual(invitations[0]["status"], "pending")

        # Step 5: Accept invitation (as invitee)
        self.client.force_authenticate(self.invitee)
        res = self.client.post(
            "/api/core/org/accept-invite/",
            {"token": token},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["success"])
        self.assertEqual(res.json()["org_name"], "Acme Manufacturing")
        self.assertEqual(res.json()["role"], "member")

        # Verify membership in DB
        invitee_membership = Membership.objects.get(
            tenant=tenant, user=self.invitee, is_active=True
        )
        self.assertEqual(invitee_membership.role, "member")

        # Step 6: List members (as owner)
        self.client.force_authenticate(self.owner)
        res = self.client.get("/api/core/org/members/")
        self.assertEqual(res.status_code, 200)
        members = res.json()["members"]
        self.assertEqual(len(members), 2)
        emails = [m["email"] for m in members]
        self.assertIn("owner@example.com", emails)
        self.assertIn("invitee@example.com", emails)

        # Step 7: Change invitee role to admin
        res = self.client.put(
            f"/api/core/org/members/{invitee_membership.id}/role/",
            {"role": "admin"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["success"])
        invitee_membership.refresh_from_db()
        self.assertEqual(invitee_membership.role, "admin")

        # Step 8: Remove member
        with patch("accounts.billing.remove_org_seat"):
            res = self.client.delete(
                f"/api/core/org/members/{invitee_membership.id}/remove/"
            )
            self.assertEqual(res.status_code, 200)
            self.assertTrue(res.json()["success"])
        invitee_membership.refresh_from_db()
        self.assertFalse(invitee_membership.is_active)

    def test_free_user_cannot_create_org(self):
        """FREE tier user is blocked from org creation."""
        free_user = _make_user("freeuser@example.com", Tier.FREE)
        self.client.force_authenticate(free_user)
        res = self.client.post(
            "/api/core/org/create/",
            {"name": "No Way", "slug": "nope"},
            format="json",
        )
        self.assertEqual(res.status_code, 403)

    def test_cannot_create_second_org(self):
        """User already in an org cannot create another."""
        self.client.force_authenticate(self.owner)
        tenant = _make_tenant(name="First Org", slug="first-org")
        _make_membership(tenant, self.owner, Membership.Role.OWNER)

        res = self.client.post(
            "/api/core/org/create/",
            {"name": "Second Org", "slug": "second-org"},
            format="json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("already belong", res.json()["error"]["message"])

    def test_cannot_remove_self(self):
        """Owner cannot remove themselves."""
        self.client.force_authenticate(self.owner)
        tenant = _make_tenant(name="Self Org", slug="self-org")
        membership = _make_membership(tenant, self.owner, Membership.Role.OWNER)

        res = self.client.delete(f"/api/core/org/members/{membership.id}/remove/")
        self.assertEqual(res.status_code, 400)
        self.assertIn("Cannot remove yourself", res.json()["error"]["message"])


# =========================================================================
# 8. OrgInvitationEdgeCasesTest
# =========================================================================


@SECURE_OFF
class OrgInvitationEdgeCasesTest(TestCase):
    """Edge cases: cancel invitation, wrong email, expired invitation."""

    def setUp(self):
        self.client = APIClient()
        self.owner = _make_user("edgeowner@example.com", Tier.TEAM)
        self.tenant = _make_tenant(name="Edge Org", slug="edge-org")
        _make_membership(self.tenant, self.owner, Membership.Role.OWNER)
        self.client.force_authenticate(self.owner)

    def test_cancel_invitation(self):
        """Cancel a pending invitation."""
        inv = OrgInvitation.objects.create(
            tenant=self.tenant,
            email="cancel@example.com",
            role="member",
            invited_by=self.owner,
        )
        with patch("accounts.billing.remove_org_seat"):
            res = self.client.post(f"/api/core/org/invitations/{inv.id}/cancel/")
            self.assertEqual(res.status_code, 200)
            self.assertTrue(res.json()["success"])
        inv.refresh_from_db()
        self.assertEqual(inv.status, "cancelled")

    def test_wrong_email_rejects_accept(self):
        """User with different email cannot accept invitation."""
        inv = OrgInvitation.objects.create(
            tenant=self.tenant,
            email="someone@example.com",
            role="member",
            invited_by=self.owner,
        )
        wrong_user = _make_user("wrong@example.com", Tier.TEAM)
        wrong_user.is_email_verified = True
        wrong_user.save(update_fields=["is_email_verified"])
        self.client.force_authenticate(wrong_user)
        res = self.client.post(
            "/api/core/org/accept-invite/",
            {"token": str(inv.token)},
            format="json",
        )
        self.assertEqual(res.status_code, 403)
        self.assertIn("different email", res.json()["error"]["message"])

    def test_expired_invitation_rejected(self):
        """Expired invitation returns 400."""
        inv = OrgInvitation.objects.create(
            tenant=self.tenant,
            email="expired@example.com",
            role="member",
            invited_by=self.owner,
            expires_at=timezone.now() - timedelta(days=1),
        )
        expired_user = _make_user("expired@example.com", Tier.TEAM)
        self.client.force_authenticate(expired_user)
        res = self.client.post(
            "/api/core/org/accept-invite/",
            {"token": str(inv.token)},
            format="json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("expired", res.json()["error"]["message"].lower())

    def test_accept_requires_token(self):
        """Accept endpoint rejects missing token."""
        user = _make_user("notoken@example.com", Tier.TEAM)
        self.client.force_authenticate(user)
        res = self.client.post("/api/core/org/accept-invite/", {}, format="json")
        self.assertEqual(res.status_code, 400)
        self.assertIn("Token", res.json()["error"]["message"])

    def test_duplicate_invitation_blocked(self):
        """Cannot send a second pending invitation to the same email."""
        OrgInvitation.objects.create(
            tenant=self.tenant,
            email="dup@example.com",
            role="member",
            invited_by=self.owner,
            status=OrgInvitation.Status.PENDING,
        )
        with patch("accounts.billing.add_org_seat", return_value=11):
            res = self.client.post(
                "/api/core/org/invite/",
                {"email": "dup@example.com", "role": "member"},
                format="json",
            )
        self.assertEqual(res.status_code, 400)
        self.assertIn("already pending", res.json()["error"]["message"])

    def test_last_owner_cannot_change_own_role(self):
        """Sole owner cannot demote themselves."""
        res = self.client.put(
            f"/api/core/org/members/{Membership.objects.get(tenant=self.tenant, user=self.owner).id}/role/",
            {"role": "member"},
            format="json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("only owner", res.json()["error"]["message"])


# =========================================================================
# 9. KnowledgeGraphTest
# =========================================================================


@SECURE_OFF
class KnowledgeGraphTest(TestCase):
    """Knowledge graph: get graph, create entities, create relationships, check consistency."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("graph@example.com", Tier.TEAM)
        self.client.force_authenticate(self.user)

    def test_get_knowledge_graph(self):
        """GET /graph/ returns or creates the user's knowledge graph."""
        res = self.client.get("/api/core/graph/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("id", data)
        self.assertIn("name", data)
        self.assertIn("entities", data)
        self.assertIn("relationships", data)
        self.assertEqual(data["entity_count"], 0)
        self.assertEqual(data["relationship_count"], 0)
        # Verify graph was created in DB
        self.assertTrue(KnowledgeGraph.objects.filter(user=self.user).exists())

    def test_entity_crud(self):
        """Create, list, detail, update, delete entities."""
        # Create
        res = self.client.post(
            "/api/core/graph/entities/",
            {
                "name": "Temperature",
                "entity_type": "variable",
                "description": "Process temperature in Celsius",
                "unit": "C",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        ent_id = res.json()["id"]
        self.assertEqual(res.json()["name"], "Temperature")
        self.assertEqual(res.json()["entity_type"], "variable")
        self.assertEqual(res.json()["unit"], "C")

        # List
        res = self.client.get("/api/core/graph/entities/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()), 1)

        # Filter by type
        res = self.client.get("/api/core/graph/entities/?type=variable")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()), 1)

        res = self.client.get("/api/core/graph/entities/?type=concept")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()), 0)

        # Detail
        res = self.client.get(f"/api/core/graph/entities/{ent_id}/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["name"], "Temperature")

        # Update
        res = self.client.put(
            f"/api/core/graph/entities/{ent_id}/",
            {"description": "Updated description"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)

        # Delete
        res = self.client.delete(f"/api/core/graph/entities/{ent_id}/")
        self.assertEqual(res.status_code, 204)
        self.assertFalse(Entity.objects.filter(id=ent_id).exists())

    def test_relationship_crud(self):
        """Create and list relationships between entities."""
        graph = _make_graph(self.user)
        source = Entity.objects.create(
            graph=graph,
            name="Temperature",
            entity_type="variable",
            created_by=self.user,
        )
        target = Entity.objects.create(
            graph=graph,
            name="Yield",
            entity_type="variable",
            created_by=self.user,
        )

        # Create relationship
        res = self.client.post(
            "/api/core/graph/relationships/",
            {
                "source": str(source.id),
                "target": str(target.id),
                "relation_type": "causes",
                "strength": 0.8,
                "confidence": 0.7,
            },
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["relation_type"], "causes")
        self.assertAlmostEqual(float(data["strength"]), 0.8)
        self.assertEqual(data["source_name"], "Temperature")
        self.assertEqual(data["target_name"], "Yield")

        # List relationships
        res = self.client.get("/api/core/graph/relationships/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()), 1)

        # Filter by type
        res = self.client.get("/api/core/graph/relationships/?type=causes")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()), 1)

        res = self.client.get("/api/core/graph/relationships/?type=correlates_with")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()), 0)

    def test_unauthenticated_graph_access(self):
        """Anonymous user gets rejected from graph endpoints."""
        self.client.force_authenticate(None)
        res = self.client.get("/api/core/graph/")
        self.assertIn(res.status_code, [401, 403])


# =========================================================================
# 10. EvidenceFromAnalysisTest
# =========================================================================


@SECURE_OFF
class EvidenceFromAnalysisTest(TestCase):
    """Test create_evidence_from_analysis and create_evidence_from_code."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("analysis@example.com", Tier.PRO)
        self.client.force_authenticate(self.user)
        self.project = _make_project(self.user, title="Analysis Project")
        self.hyp = _make_hypothesis(self.project, statement="X causes Y")

    def test_create_evidence_from_code(self):
        """POST /evidence/from-code/ creates evidence and links to hypotheses."""
        with patch("core.views.synara") as mock_synara:
            mock_synara.apply_evidence.return_value = UpdateResult(
                hypothesis_id=str(self.hyp.id),
                prior_probability=0.5,
                posterior_probability=0.65,
                likelihood_ratio=2.0,
                adjusted_lr=1.8,
                evidence_id="dummy",
                status_changed=False,
            )
            res = self.client.post(
                "/api/core/evidence/from-code/",
                {
                    "project_id": str(self.project.id),
                    "hypothesis_ids": [str(self.hyp.id)],
                    "summary": "Monte Carlo simulation shows 15% effect",
                    "details": "10000 iterations, p < 0.01",
                    "source_type": "simulation",
                    "code": "import numpy as np\nnp.random.seed(42)",
                    "output": {"mean": 15.2, "std": 2.1},
                    "p_value": 0.005,
                    "effect_size": 0.8,
                    "sample_size": 10000,
                    "confidence": 0.9,
                    "likelihood_ratios": {str(self.hyp.id): 2.0},
                },
                format="json",
            )
            self.assertEqual(res.status_code, 201)
            data = res.json()
            self.assertIn("evidence", data)
            self.assertIn("links", data)
            self.assertEqual(
                data["evidence"]["summary"],
                "Monte Carlo simulation shows 15% effect",
            )

            # Verify evidence in DB
            ev = Evidence.objects.get(id=data["evidence"]["id"])
            self.assertAlmostEqual(ev.p_value, 0.005)
            self.assertEqual(ev.source_description, "Coder")
            self.assertTrue(ev.is_reproducible)
            self.assertEqual(
                ev.code_reference, "import numpy as np\nnp.random.seed(42)"
            )

            # Verify link
            self.assertTrue(
                EvidenceLink.objects.filter(hypothesis=self.hyp, evidence=ev).exists()
            )

    def test_create_evidence_from_analysis(self):
        """POST /evidence/from-analysis/ creates evidence from DSW results."""
        with patch("core.views.synara") as mock_synara:
            mock_synara.apply_evidence.return_value = UpdateResult(
                hypothesis_id=str(self.hyp.id),
                prior_probability=0.5,
                posterior_probability=0.7,
                likelihood_ratio=3.0,
                adjusted_lr=2.8,
                evidence_id="dummy",
                status_changed=False,
            )
            mock_synara.suggest_likelihood_ratio.return_value = (3.0, "Strong p-value")

            res = self.client.post(
                "/api/core/evidence/from-analysis/",
                {
                    "project_id": str(self.project.id),
                    "hypothesis_ids": [str(self.hyp.id)],
                    "summary": "Linear regression R2=0.82",
                    "analysis_type": "regression",
                    "results": {
                        "r2": 0.82,
                        "coefficients": {"temperature": -0.15},
                        "p_value": 0.003,
                    },
                    "metrics": {
                        "p_value": 0.003,
                        "r2": 0.82,
                        "sample_size": 50,
                    },
                    "confidence": 0.85,
                },
                format="json",
            )
            self.assertEqual(res.status_code, 201)
            data = res.json()
            self.assertIn("evidence", data)
            ev = Evidence.objects.get(id=data["evidence"]["id"])
            self.assertEqual(ev.source_type, "analysis")
            self.assertIn("DSW", ev.source_description)
            self.assertAlmostEqual(ev.p_value, 0.003)
            self.assertTrue(ev.is_reproducible)

    def test_from_code_validation_error(self):
        """Missing required fields returns 400."""
        res = self.client.post(
            "/api/core/evidence/from-code/",
            {"project_id": str(self.project.id)},
            format="json",
        )
        self.assertEqual(res.status_code, 400)

    def test_free_user_blocked_from_analysis(self):
        """FREE tier cannot use create_evidence_from_analysis (require_ml)."""
        free_user = _make_user("freeml@example.com", Tier.FREE)
        self.client.force_authenticate(free_user)
        res = self.client.post(
            "/api/core/evidence/from-analysis/",
            {
                "project_id": str(self.project.id),
                "summary": "test",
                "analysis_type": "regression",
                "results": {},
            },
            format="json",
        )
        self.assertEqual(res.status_code, 403)


# =========================================================================
# 11. ConsistencyCheckTest
# =========================================================================


@SECURE_OFF
class ConsistencyCheckTest(TestCase):
    """Test check_consistency endpoint."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("consistency@example.com", Tier.TEAM)
        self.client.force_authenticate(self.user)

    def test_check_consistency_no_issues(self):
        """Empty graph has no consistency issues."""
        with patch("core.views.synara") as mock_synara:
            mock_synara.check_consistency.return_value = []
            res = self.client.post("/api/core/graph/check-consistency/")
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["total_issues"], 0)
            self.assertFalse(data["has_errors"])
            self.assertEqual(data["issues"], [])

    def test_check_consistency_with_issues(self):
        """Consistency check returns structured issue data."""
        issue = ConsistencyIssue(
            issue_type="contradiction",
            severity="error",
            description="A causes B and A prevents B",
            entities_involved=["entity-1", "entity-2"],
            suggestions=["Remove one of the contradictory relationships"],
        )
        with patch("core.views.synara") as mock_synara:
            mock_synara.check_consistency.return_value = [issue]
            res = self.client.post("/api/core/graph/check-consistency/")
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["total_issues"], 1)
            self.assertTrue(data["has_errors"])
            issue_data = data["issues"][0]
            self.assertEqual(issue_data["type"], "contradiction")
            self.assertEqual(issue_data["severity"], "error")
            self.assertIn("causes B", issue_data["description"])
            self.assertEqual(len(issue_data["entities"]), 2)
            self.assertEqual(len(issue_data["suggestions"]), 1)

    def test_unauthenticated_consistency_check(self):
        """Anonymous user gets rejected."""
        self.client.force_authenticate(None)
        res = self.client.post("/api/core/graph/check-consistency/")
        self.assertIn(res.status_code, [401, 403])


# =========================================================================
# Cross-Cutting: Suggest Likelihood Ratio
# =========================================================================


@SECURE_OFF
class SuggestLikelihoodRatioTest(TestCase):
    """Test suggest_likelihood_ratio endpoint (requires ML tier)."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("suggest@example.com", Tier.PRO)
        self.client.force_authenticate(self.user)
        self.project = _make_project(self.user, title="Suggest LR Project")
        self.hyp = _make_hypothesis(self.project)
        self.ev = _make_evidence(self.project)

    def test_suggest_lr(self):
        """suggest_likelihood_ratio returns suggested LR and reasoning."""
        with patch("core.views.synara") as mock_synara:
            mock_synara.suggest_likelihood_ratio.return_value = (
                3.5,
                "Strong statistical result with p < 0.01",
            )
            res = self.client.post(
                f"/api/core/projects/{self.project.id}/suggest-lr/",
                {
                    "evidence_id": str(self.ev.id),
                    "hypothesis_id": str(self.hyp.id),
                },
                format="json",
            )
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertAlmostEqual(data["suggested_likelihood_ratio"], 3.5)
            self.assertIn("reasoning", data)
            self.assertEqual(data["evidence_id"], str(self.ev.id))
            self.assertEqual(data["hypothesis_id"], str(self.hyp.id))

    def test_suggest_lr_requires_both_ids(self):
        """Missing evidence_id or hypothesis_id returns 400."""
        res = self.client.post(
            f"/api/core/projects/{self.project.id}/suggest-lr/",
            {"evidence_id": str(self.ev.id)},
            format="json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("required", res.json()["error"]["message"])

    def test_free_user_blocked(self):
        """FREE tier blocked from suggest LR (require_ml)."""
        free_user = _make_user("freesuggest@example.com", Tier.FREE)
        self.client.force_authenticate(free_user)
        res = self.client.post(
            f"/api/core/projects/{self.project.id}/suggest-lr/",
            {
                "evidence_id": str(self.ev.id),
                "hypothesis_id": str(self.hyp.id),
            },
            format="json",
        )
        self.assertEqual(res.status_code, 403)
