"""Cross-module integration tests.

Tests the integration surfaces between modules:
- Projects → Hypotheses → Evidence (Bayesian pipeline)
- Evidence from Code (Coder → Core)
- Evidence from Analysis (DSW → Core)
- Knowledge Graph (entities, relationships, consistency)
- File uploads (quota, security)
- FMEA → Hypothesis linking
- Project Hub (cross-tool aggregation)
"""

import uuid

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.utils import timezone
from rest_framework.test import APIClient

from accounts.constants import Tier

User = get_user_model()
SECURE_OFF = override_settings(
    SECURE_SSL_REDIRECT=False,
    FIELD_ENCRYPTION_KEY="aX2iPiqaHO32rF439HMExl3UBAxhqCRMErnn10pFrvU=",
)


def _make_user(email, tier=Tier.FREE, password="testpass123!", **kwargs):
    username = kwargs.pop("username", email.split("@")[0])
    user = User.objects.create_user(username=username, email=email, password=password, **kwargs)
    user.tier = tier
    user.save(update_fields=["tier"])
    return user


def _create_project(client, title="Test Project"):
    """Helper: create a project via API, return response data."""
    res = client.post(
        "/api/core/projects/",
        {
            "title": title,
            "problem_statement": "Does X cause Y?",
            "domain": "manufacturing",
            "methodology": "dmaic",
        },
        format="json",
    )
    return res


def _create_hypothesis(client, project_id, statement="If X then Y"):
    """Helper: create hypothesis via API, return response data."""
    res = client.post(
        f"/api/core/projects/{project_id}/hypotheses/",
        {
            "statement": statement,
            "prior_probability": 0.5,
        },
        format="json",
    )
    return res


# =========================================================================
# Project → Hypothesis → Evidence Pipeline
# =========================================================================


@SECURE_OFF
class ProjectHypothesisEvidencePipelineTest(TestCase):
    """Test the core Bayesian pipeline: project → hypothesis → evidence."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("pipeline@example.com", Tier.PRO)
        self.client.force_authenticate(self.user)

    def test_create_project(self):
        res = _create_project(self.client)
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertIn("id", data)
        self.assertEqual(data["title"], "Test Project")
        self.assertEqual(data["domain"], "manufacturing")

    def test_create_hypothesis_in_project(self):
        proj = _create_project(self.client).json()
        res = _create_hypothesis(self.client, proj["id"])
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["statement"], "If X then Y")
        self.assertAlmostEqual(float(data["prior_probability"]), 0.5)

    def test_add_evidence_to_hypothesis(self):
        proj = _create_project(self.client).json()
        hyp = _create_hypothesis(self.client, proj["id"]).json()

        # Create evidence linked to hypothesis
        res = self.client.post(
            f"/api/core/projects/{proj['id']}/evidence/",
            {
                "summary": "Observation shows X correlates with Y",
                "source_type": "observation",
                "confidence": 0.8,
                "hypothesis_ids": [hyp["id"]],
                "likelihood_ratios": {hyp["id"]: 2.0},
            },
            format="json",
        )
        self.assertEqual(res.status_code, 201)

    def test_evidence_link_created(self):
        """Evidence linked to hypothesis creates EvidenceLink record."""
        from core.models import EvidenceLink

        proj = _create_project(self.client).json()
        hyp = _create_hypothesis(self.client, proj["id"]).json()

        self.client.post(
            f"/api/core/projects/{proj['id']}/evidence/",
            {
                "summary": "Strong evidence for X",
                "source_type": "analysis",
                "confidence": 0.9,
                "hypothesis_ids": [hyp["id"]],
                "likelihood_ratios": {hyp["id"]: 3.0},
            },
            format="json",
        )

        links = EvidenceLink.objects.filter(hypothesis_id=hyp["id"])
        self.assertEqual(links.count(), 1)
        self.assertAlmostEqual(float(links.first().likelihood_ratio), 3.0)

    def test_link_evidence_with_bayesian_update(self):
        """Linking evidence with apply=True triggers Bayesian probability update."""

        proj = _create_project(self.client).json()
        hyp = _create_hypothesis(self.client, proj["id"]).json()

        # Create evidence (evidence_list now correctly sets project FK)
        ev_res = self.client.post(
            f"/api/core/projects/{proj['id']}/evidence/",
            {
                "summary": "Supporting observation",
                "source_type": "observation",
                "confidence": 0.85,
            },
            format="json",
        )
        evidence = ev_res.json()

        # Link with Bayesian update (LR > 1 = supports)
        res = self.client.post(
            f"/api/core/projects/{proj['id']}/hypotheses/{hyp['id']}/link-evidence/",
            {
                "evidence_id": evidence["id"],
                "likelihood_ratio": 3.0,
                "reasoning": "Strong observational support",
                "apply": True,
            },
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("update_result", data)

        # Posterior should be > prior (0.5) with LR=3.0
        posterior = data["update_result"]["posterior"]
        self.assertGreater(posterior, 0.5)

    def test_bayesian_update_via_evidence_from_code(self):
        """Test Bayesian update using the from-code endpoint (which handles project FK correctly)."""
        from core.models import Hypothesis

        proj = _create_project(self.client).json()
        hyp = _create_hypothesis(self.client, proj["id"]).json()

        # Use evidence_from_code which correctly links evidence to project
        res = self.client.post(
            "/api/core/evidence/from-code/",
            {
                "project_id": proj["id"],
                "hypothesis_ids": [hyp["id"]],
                "summary": "Strong supporting evidence",
                "source_type": "simulation",
                "confidence": 0.9,
                "likelihood_ratios": {str(hyp["id"]): 3.0},
            },
            format="json",
        )
        self.assertEqual(res.status_code, 201)

        # Check posterior > prior after supporting evidence
        hyp_obj = Hypothesis.objects.get(id=hyp["id"])
        self.assertGreater(float(hyp_obj.current_probability), 0.5)

    def test_project_list_filters(self):
        _create_project(self.client, "Active Project")
        _create_project(self.client, "Another Project")

        res = self.client.get("/api/core/projects/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()), 2)

    def test_project_isolation_between_users(self):
        """User A cannot see User B's projects."""
        _create_project(self.client, "User A Project")

        user_b = _make_user("userb@example.com", Tier.PRO)
        client_b = APIClient()
        client_b.force_authenticate(user_b)

        res = client_b.get("/api/core/projects/")
        self.assertEqual(len(res.json()), 0)

    def test_hypothesis_not_in_other_project(self):
        """Hypothesis from project A cannot be accessed via project B."""
        proj_a = _create_project(self.client, "Project A").json()
        proj_b = _create_project(self.client, "Project B").json()
        hyp = _create_hypothesis(self.client, proj_a["id"]).json()

        # Try to get hypothesis from wrong project
        res = self.client.get(f"/api/core/projects/{proj_b['id']}/hypotheses/{hyp['id']}/")
        self.assertEqual(res.status_code, 404)

    def test_project_advance_phase(self):
        proj = _create_project(self.client).json()
        res = self.client.post(
            f"/api/core/projects/{proj['id']}/advance-phase/",
            {"phase": "measure", "notes": "Starting measurement"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["current_phase"], "measure")

    def test_project_comment_changelog(self):
        proj = _create_project(self.client).json()
        res = self.client.post(
            f"/api/core/projects/{proj['id']}/comment/",
            {"text": "Initial assessment complete"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        changelog = res.json()["changelog"]
        self.assertTrue(len(changelog) > 0)


# =========================================================================
# Evidence from Code (Coder → Core Integration)
# =========================================================================


@SECURE_OFF
class EvidenceFromCodeTest(TestCase):
    """Test Coder → Evidence integration (create_evidence_from_code)."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("coder@example.com", Tier.PRO)
        self.client.force_authenticate(self.user)

    def test_create_evidence_from_code(self):
        proj = _create_project(self.client).json()
        hyp = _create_hypothesis(self.client, proj["id"]).json()

        res = self.client.post(
            "/api/core/evidence/from-code/",
            {
                "project_id": proj["id"],
                "hypothesis_ids": [hyp["id"]],
                "summary": "Simulation confirms X→Y relationship",
                "source_type": "simulation",
                "code": "import numpy as np\n# simulation code here",
                "p_value": 0.02,
                "effect_size": 0.8,
                "sample_size": 100,
                "confidence": 0.9,
                "likelihood_ratios": {str(hyp["id"]): 2.5},
            },
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertIn("evidence", data)
        self.assertIn("links", data)
        self.assertEqual(len(data["links"]), 1)
        # Bayesian update should have occurred
        self.assertIn("prior", data["links"][0])
        self.assertIn("posterior", data["links"][0])
        self.assertGreater(data["links"][0]["posterior"], data["links"][0]["prior"])

    def test_evidence_from_code_no_hypothesis(self):
        """Can create evidence without linking to any hypothesis."""
        proj = _create_project(self.client).json()

        res = self.client.post(
            "/api/core/evidence/from-code/",
            {
                "project_id": proj["id"],
                "summary": "General observation from code",
                "confidence": 0.7,
            },
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        self.assertEqual(len(res.json()["links"]), 0)

    def test_evidence_from_code_missing_project(self):
        res = self.client.post(
            "/api/core/evidence/from-code/",
            {
                "project_id": str(uuid.uuid4()),
                "summary": "Test",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 404)

    def test_evidence_from_code_skips_invalid_hypothesis(self):
        """Invalid hypothesis IDs are silently skipped."""
        proj = _create_project(self.client).json()

        res = self.client.post(
            "/api/core/evidence/from-code/",
            {
                "project_id": proj["id"],
                "hypothesis_ids": [str(uuid.uuid4())],
                "summary": "Test with bad hypothesis",
                "confidence": 0.8,
            },
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        self.assertEqual(len(res.json()["links"]), 0)


# =========================================================================
# Evidence from Analysis (DSW → Core Integration)
# =========================================================================


@SECURE_OFF
class EvidenceFromAnalysisTest(TestCase):
    """Test DSW Analysis → Evidence integration (create_evidence_from_analysis)."""

    def setUp(self):
        self.client = APIClient()
        # require_ml needs PRO+ tier
        self.user = _make_user("analyst@example.com", Tier.PRO)
        self.client.force_authenticate(self.user)

    def test_create_evidence_from_analysis(self):
        proj = _create_project(self.client).json()
        hyp = _create_hypothesis(self.client, proj["id"]).json()

        res = self.client.post(
            "/api/core/evidence/from-analysis/",
            {
                "project_id": proj["id"],
                "hypothesis_ids": [hyp["id"]],
                "summary": "Regression shows significant effect",
                "analysis_type": "regression",
                "results": {"r2": 0.92, "coefficients": [1.2, -0.5]},
                "metrics": {"p_value": 0.01, "effect_size": 0.85, "sample_size": 50},
                "confidence": 0.95,
                "likelihood_ratios": {str(hyp["id"]): 4.0},
            },
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(len(data["links"]), 1)
        self.assertGreater(data["links"][0]["posterior"], 0.5)

    def test_analysis_evidence_free_user_blocked(self):
        """Free user cannot use @require_ml endpoint."""
        free_user = _make_user("free_analyst@example.com", Tier.FREE)
        client = APIClient()
        client.force_authenticate(free_user)

        proj_res = _create_project(client)
        # Free users can still create projects
        proj = proj_res.json()

        res = client.post(
            "/api/core/evidence/from-analysis/",
            {
                "project_id": proj["id"],
                "summary": "Test",
                "analysis_type": "regression",
                "results": {"r2": 0.5},
            },
            format="json",
        )
        self.assertEqual(res.status_code, 403)


# =========================================================================
# Knowledge Graph Integration
# =========================================================================


@SECURE_OFF
class KnowledgeGraphTest(TestCase):
    """Test knowledge graph entity/relationship CRUD and consistency."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("graph@example.com", Tier.PRO)
        self.client.force_authenticate(self.user)

    def test_graph_auto_created(self):
        res = self.client.get("/api/core/graph/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("id", data)
        self.assertEqual(data["entity_count"], 0)

    def test_create_entity(self):
        res = self.client.post(
            "/api/core/graph/entities/",
            {
                "name": "Temperature",
                "entity_type": "variable",
                "description": "Process temperature in °C",
                "unit": "°C",
                "typical_min": 100,
                "typical_max": 200,
            },
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["name"], "Temperature")
        self.assertEqual(data["entity_type"], "variable")
        self.assertEqual(data["unit"], "°C")

    def test_create_relationship(self):
        # Create two entities first
        e1 = self.client.post(
            "/api/core/graph/entities/",
            {
                "name": "Temperature",
                "entity_type": "variable",
            },
            format="json",
        ).json()
        e2 = self.client.post(
            "/api/core/graph/entities/",
            {
                "name": "Yield",
                "entity_type": "variable",
            },
            format="json",
        ).json()

        # Create relationship
        res = self.client.post(
            "/api/core/graph/relationships/",
            {
                "source": e1["id"],
                "target": e2["id"],
                "relation_type": "causes",
                "strength": 0.8,
                "confidence": 0.9,
            },
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["relation_type"], "causes")

    def test_graph_isolation_between_users(self):
        """User B cannot see User A's entities."""
        self.client.post(
            "/api/core/graph/entities/",
            {
                "name": "Secret Variable",
                "entity_type": "variable",
            },
            format="json",
        )

        user_b = _make_user("graphb@example.com", Tier.PRO)
        client_b = APIClient()
        client_b.force_authenticate(user_b)

        res = client_b.get("/api/core/graph/entities/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()), 0)

    def test_consistency_check_empty_graph(self):
        """Empty graph should have no consistency issues."""
        res = self.client.post("/api/core/graph/check-consistency/")
        self.assertEqual(res.status_code, 200)
        # Empty graph should be consistent
        self.assertIn("issues", res.json())

    def test_entity_detail_update(self):
        e = self.client.post(
            "/api/core/graph/entities/",
            {
                "name": "Pressure",
                "entity_type": "variable",
                "unit": "psi",
            },
            format="json",
        ).json()

        res = self.client.put(
            f"/api/core/graph/entities/{e['id']}/",
            {
                "name": "Pressure Updated",
                "entity_type": "variable",
                "unit": "bar",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["name"], "Pressure Updated")
        self.assertEqual(res.json()["unit"], "bar")

    def test_entity_delete(self):
        e = self.client.post(
            "/api/core/graph/entities/",
            {
                "name": "Temp",
                "entity_type": "variable",
            },
            format="json",
        ).json()

        res = self.client.delete(f"/api/core/graph/entities/{e['id']}/")
        self.assertEqual(res.status_code, 204)

        # Verify deleted
        res = self.client.get(f"/api/core/graph/entities/{e['id']}/")
        self.assertEqual(res.status_code, 404)


# =========================================================================
# File Upload Integration
# =========================================================================


@SECURE_OFF
class FileUploadTest(TestCase):
    """Test file upload, quota, and security."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("files@example.com", Tier.PRO)
        self.client.force_authenticate(self.user)
        # Clear cached Fernet instance so override_settings takes effect
        import core.encryption

        core.encryption._fernet_instance = None

    def tearDown(self):
        import core.encryption

        core.encryption._fernet_instance = None

    def test_upload_csv(self):
        content = b"col1,col2\n1,2\n3,4"
        f = SimpleUploadedFile("test.csv", content, content_type="text/csv")
        res = self.client.post("/api/files/upload/", {"file": f}, format="multipart")
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["name"], "test.csv")
        self.assertEqual(data["mime_type"], "text/csv")

    def test_upload_json(self):
        content = b'{"data": [1, 2, 3]}'
        f = SimpleUploadedFile("data.json", content, content_type="application/json")
        res = self.client.post("/api/files/upload/", {"file": f}, format="multipart")
        self.assertEqual(res.status_code, 201)

    def test_upload_no_file_returns_400(self):
        res = self.client.post("/api/files/upload/", {}, format="multipart")
        self.assertEqual(res.status_code, 400)
        self.assertIn("No file", res.json()["error"])

    def test_dangerous_extension_blocked(self):
        """Executable files should be rejected."""
        for ext in [".exe", ".bat", ".sh", ".py", ".php"]:
            f = SimpleUploadedFile(f"test{ext}", b"content", content_type="application/octet-stream")
            res = self.client.post("/api/files/upload/", {"file": f}, format="multipart")
            self.assertEqual(res.status_code, 400, msg=f"{ext} should be blocked")

    def test_list_files(self):
        # Upload two files
        for name in ["a.csv", "b.csv"]:
            f = SimpleUploadedFile(name, b"data", content_type="text/csv")
            self.client.post("/api/files/upload/", {"file": f}, format="multipart")

        res = self.client.get("/api/files/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["total"], 2)

    def test_files_isolated_per_user(self):
        """User B cannot see User A's files."""
        f = SimpleUploadedFile("secret.csv", b"secret data", content_type="text/csv")
        self.client.post("/api/files/upload/", {"file": f}, format="multipart")

        user_b = _make_user("filesb@example.com", Tier.PRO)
        client_b = APIClient()
        client_b.force_authenticate(user_b)

        res = client_b.get("/api/files/")
        self.assertEqual(res.json()["total"], 0)

    def test_upload_unauthenticated(self):
        client = APIClient()
        f = SimpleUploadedFile("test.csv", b"data", content_type="text/csv")
        res = client.post("/api/files/upload/", {"file": f}, format="multipart")
        self.assertIn(res.status_code, [401, 403])


# =========================================================================
# FMEA → Hypothesis Integration
# =========================================================================


@SECURE_OFF
class FMEAHypothesisLinkTest(TestCase):
    """Test FMEA row → Hypothesis evidence linking."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("fmea@example.com", Tier.PRO)
        # FMEA uses plain Django views — need force_login
        self.client.force_login(self.user)

    def test_create_fmea(self):
        res = self.client.post(
            "/api/fmea/create/",
            {
                "title": "Process FMEA",
                "fmea_type": "process",
            },
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("id", data)

    def test_add_fmea_row(self):
        fmea = self.client.post(
            "/api/fmea/create/",
            {
                "title": "Test FMEA",
                "fmea_type": "process",
            },
            content_type="application/json",
        ).json()

        res = self.client.post(
            f"/api/fmea/{fmea['id']}/rows/",
            {
                "failure_mode": "Overheating",
                "effect": "Product degradation",
                "cause": "Cooling system failure",
                "severity": 8,
                "occurrence": 4,
                "detection": 6,
            },
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()["row"]
        self.assertEqual(data["failure_mode"], "Overheating")
        self.assertEqual(data["rpn"], 8 * 4 * 6)  # 192

    def test_link_fmea_to_hypothesis(self):
        """Link FMEA row to hypothesis and verify evidence creation."""
        from core.models import Evidence, EvidenceLink

        # Create project with hypothesis (via DRF endpoints)
        drf_client = APIClient()
        drf_client.force_authenticate(self.user)
        proj = _create_project(drf_client).json()
        hyp = _create_hypothesis(drf_client, proj["id"], "Cooling failure causes defects").json()

        # Create FMEA with project link
        fmea_res = self.client.post(
            "/api/fmea/create/",
            {
                "title": "FMEA for Project",
                "fmea_type": "process",
                "project_id": str(proj["id"]),
            },
            content_type="application/json",
        ).json()

        # Add row — response is {"success": true, "row": {...}}
        row_res = self.client.post(
            f"/api/fmea/{fmea_res['id']}/rows/",
            {
                "failure_mode": "Cooling failure",
                "effect": "Overheating",
                "cause": "Blocked coolant lines",
                "severity": 9,
                "occurrence": 5,
                "detection": 7,
            },
            content_type="application/json",
        ).json()
        row_id = row_res["row"]["id"]

        # Link to hypothesis
        res = self.client.post(
            f"/api/fmea/{fmea_res['id']}/rows/{row_id}/link/",
            {"hypothesis_id": str(hyp["id"])},
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertIn("evidence_id", data)
        self.assertIn("likelihood_ratio", data)
        self.assertIn("direction", data)

        # Verify Evidence was created
        evidence = Evidence.objects.get(id=data["evidence_id"])
        self.assertIn("FMEA", evidence.summary)
        self.assertIn("Cooling failure", evidence.summary)

        # Verify EvidenceLink was created
        link = EvidenceLink.objects.get(id=data["link_id"])
        self.assertEqual(str(link.hypothesis_id), hyp["id"])

    def test_fmea_rpn_summary(self):
        fmea = self.client.post(
            "/api/fmea/create/",
            {
                "title": "RPN Test",
                "fmea_type": "process",
            },
            content_type="application/json",
        ).json()

        # Add rows with different RPNs
        for s, o, d in [(9, 8, 7), (3, 2, 1), (5, 5, 5)]:
            self.client.post(
                f"/api/fmea/{fmea['id']}/rows/",
                {
                    "failure_mode": f"Failure S{s}",
                    "effect": "Effect",
                    "cause": "Cause",
                    "severity": s,
                    "occurrence": o,
                    "detection": d,
                },
                content_type="application/json",
            )

        res = self.client.get(f"/api/fmea/{fmea['id']}/summary/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("total_rows", data)
        self.assertEqual(data["total_rows"], 3)
        self.assertIn("max_rpn", data)
        self.assertEqual(data["max_rpn"], 9 * 8 * 7)  # 504


# =========================================================================
# Project Hub (Cross-Tool Aggregation)
# =========================================================================


@SECURE_OFF
class ProjectHubTest(TestCase):
    """Test project hub aggregates data from multiple modules."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("hub@example.com", Tier.PRO)
        self.client.force_authenticate(self.user)

    def test_project_hub_returns_all_sections(self):
        proj = _create_project(self.client).json()

        # Add a hypothesis
        _create_hypothesis(self.client, proj["id"])

        res = self.client.get(f"/api/core/projects/{proj['id']}/hub/")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        # Verify hub structure
        self.assertIn("project", data)
        self.assertIn("tools", data)
        self.assertIn("counts", data)
        self.assertEqual(data["project"]["title"], "Test Project")
        self.assertEqual(data["counts"]["hypotheses"], 1)

    def test_hub_includes_evidence_summary(self):
        proj = _create_project(self.client).json()
        hyp = _create_hypothesis(self.client, proj["id"]).json()

        # Add evidence
        self.client.post(
            f"/api/core/projects/{proj['id']}/evidence/",
            {
                "summary": "Test evidence",
                "source_type": "observation",
                "hypothesis_ids": [hyp["id"]],
            },
            format="json",
        )

        res = self.client.get(f"/api/core/projects/{proj['id']}/hub/")
        data = res.json()
        self.assertIn("evidence_summary", data)


# =========================================================================
# Tenant Project Isolation
# =========================================================================


@SECURE_OFF
class TenantProjectIsolationTest(TestCase):
    """Test that tenant projects are properly isolated and shared."""

    def setUp(self):
        from core.models import Membership, Tenant

        self.client = APIClient()
        self.owner = _make_user("owner@example.com", Tier.TEAM)
        self.member = _make_user("member@example.com", Tier.TEAM)
        self.outsider = _make_user("outsider@example.com", Tier.PRO)

        self.tenant = Tenant.objects.create(name="Test Org", slug="test-org", plan="team")
        Membership.objects.create(
            tenant=self.tenant,
            user=self.owner,
            role="owner",
            is_active=True,
            joined_at=timezone.now(),
        )
        Membership.objects.create(
            tenant=self.tenant,
            user=self.member,
            role="member",
            is_active=True,
            joined_at=timezone.now(),
        )

    def test_personal_project_not_shared(self):
        """Personal project is only visible to creator."""
        self.client.force_authenticate(self.owner)
        proj = _create_project(self.client, "Owner's Personal").json()

        # Member should not see personal project
        self.client.force_authenticate(self.member)
        res = self.client.get("/api/core/projects/")
        project_ids = [p["id"] for p in res.json()]
        self.assertNotIn(proj["id"], project_ids)

    def test_tenant_project_visible_to_members(self):
        """Tenant project should be visible to all tenant members."""
        from core.models import Project

        # Create project assigned to tenant (user must be null per check constraint)
        proj = Project.objects.create(
            title="Team Project",
            problem_statement="Shared problem",
            tenant=self.tenant,
        )

        # Member should see it
        self.client.force_authenticate(self.member)
        res = self.client.get("/api/core/projects/")
        project_ids = [p["id"] for p in res.json()]
        self.assertIn(str(proj.id), project_ids)

        # Outsider should not
        self.client.force_authenticate(self.outsider)
        res = self.client.get("/api/core/projects/")
        project_ids = [p["id"] for p in res.json()]
        self.assertNotIn(str(proj.id), project_ids)


# =========================================================================
# Dataset Integration
# =========================================================================


@SECURE_OFF
class DatasetTest(TestCase):
    """Test dataset CRUD within projects."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("dataset@example.com", Tier.PRO)
        self.client.force_authenticate(self.user)

    def test_create_dataset(self):
        proj = _create_project(self.client).json()

        res = self.client.post(
            f"/api/core/projects/{proj['id']}/datasets/",
            {
                "name": "Experiment Data",
                "description": "Measurements from run 1",
                "data": {"columns": ["x", "y"], "rows": [[1, 2], [3, 4]]},
            },
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["name"], "Experiment Data")

    def test_list_datasets(self):
        proj = _create_project(self.client).json()

        # Create two datasets
        for name in ["Dataset A", "Dataset B"]:
            self.client.post(
                f"/api/core/projects/{proj['id']}/datasets/",
                {
                    "name": name,
                    "data": {"rows": [[1]]},
                },
                format="json",
            )

        res = self.client.get(f"/api/core/projects/{proj['id']}/datasets/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()), 2)


# =========================================================================
# Experiment Design Integration
# =========================================================================


@SECURE_OFF
class ExperimentDesignTest(TestCase):
    """Test experiment design CRUD within projects."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("doe@example.com", Tier.PRO)
        self.client.force_authenticate(self.user)

    def test_create_experiment_design(self):
        proj = _create_project(self.client).json()
        hyp = _create_hypothesis(self.client, proj["id"]).json()

        res = self.client.post(
            f"/api/core/projects/{proj['id']}/designs/",
            {
                "name": "Full Factorial",
                "description": "2^3 design for temperature/pressure/speed",
                "design_type": "full_factorial",
                "hypothesis": hyp["id"],
                "factors": [
                    {"name": "Temperature", "levels": [100, 200]},
                    {"name": "Pressure", "levels": [1, 5]},
                    {"name": "Speed", "levels": [10, 20]},
                ],
            },
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["name"], "Full Factorial")
        self.assertEqual(data["design_type"], "full_factorial")
        self.assertEqual(data["hypothesis"], hyp["id"])

    def test_list_experiment_designs(self):
        proj = _create_project(self.client).json()

        self.client.post(
            f"/api/core/projects/{proj['id']}/designs/",
            {
                "name": "Design 1",
                "design_type": "full_factorial",
            },
            format="json",
        )

        res = self.client.get(f"/api/core/projects/{proj['id']}/designs/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()), 1)
