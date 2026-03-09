"""Cross-module QMS integration tests.

Exercises the closed-loop flows between ISO 9001 modules:
  - Audit finding → NCR auto-creation
  - NCR lifecycle → evidence on close
  - CAPA lifecycle → RCA backflow
  - FMEA row → hypothesis link → Bayesian update
  - Training → document linkage
  - Supplier state machine
  - QMS dashboard aggregation across modules

CR: 60855fc4
"""

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from accounts.constants import Tier
from core.models import Evidence, EvidenceLink, Hypothesis, Project

User = get_user_model()
SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)


def _user(email="qms@test.com", tier=Tier.TEAM):
    username = email.split("@")[0]
    u = User.objects.create_user(username=username, email=email, password="testpass123!")
    u.tier = tier
    u.save(update_fields=["tier"])
    return u


def _api(user):
    c = APIClient()
    c.force_login(user)
    return c


def _post(client, url, data):
    return client.post(url, data, format="json")


def _put(client, url, data):
    return client.put(url, data, format="json")


# ============================================================================
# 1. Audit → NCR auto-creation
# ============================================================================


@SECURE_OFF
class AuditToNCRIntegrationTest(TestCase):
    """Audit findings with NC type auto-create linked NCRs."""

    def setUp(self):
        self.user = _user("audit_ncr@test.com")
        self.c = _api(self.user)

    def test_nc_major_finding_creates_ncr(self):
        # Create audit
        resp = _post(
            self.c,
            "/api/iso/audits/",
            {
                "title": "Q1 Internal Audit",
                "audit_type": "internal",
                "scheduled_date": "2026-03-15",
            },
        )
        self.assertEqual(resp.status_code, 201)
        audit_id = resp.json()["id"]

        # Add NC major finding
        resp = _post(
            self.c,
            f"/api/iso/audits/{audit_id}/findings/",
            {
                "finding_type": "nc_major",
                "description": "Weld procedure not followed on line 3",
                "iso_clause": "8.5",
            },
        )
        self.assertEqual(resp.status_code, 201)
        finding = resp.json()

        # NCR should be auto-created
        self.assertIn("ncr_id", finding)
        self.assertIsNotNone(finding["ncr_id"])

        # Verify NCR exists and has correct attributes
        resp = self.c.get(f"/api/iso/ncrs/{finding['ncr_id']}/")
        self.assertEqual(resp.status_code, 200)
        ncr = resp.json()
        self.assertEqual(ncr["severity"], "critical")  # nc_major → critical
        self.assertEqual(ncr["source"], "internal_audit")
        self.assertEqual(ncr["status"], "open")

    def test_nc_minor_finding_creates_ncr_with_major_severity(self):
        resp = _post(
            self.c,
            "/api/iso/audits/",
            {
                "title": "Clause 7.2 Audit",
                "audit_type": "internal",
                "scheduled_date": "2026-03-20",
            },
        )
        audit_id = resp.json()["id"]

        resp = _post(
            self.c,
            f"/api/iso/audits/{audit_id}/findings/",
            {
                "finding_type": "nc_minor",
                "description": "Training record expired for operator",
                "iso_clause": "7.2",
            },
        )
        self.assertEqual(resp.status_code, 201)
        ncr_id = resp.json()["ncr_id"]
        self.assertIsNotNone(ncr_id)

        resp = self.c.get(f"/api/iso/ncrs/{ncr_id}/")
        self.assertEqual(resp.json()["severity"], "major")  # nc_minor → major

    def test_observation_finding_does_not_create_ncr(self):
        resp = _post(
            self.c,
            "/api/iso/audits/",
            {
                "title": "Observation Audit",
                "audit_type": "internal",
                "scheduled_date": "2026-04-01",
            },
        )
        audit_id = resp.json()["id"]

        resp = _post(
            self.c,
            f"/api/iso/audits/{audit_id}/findings/",
            {
                "finding_type": "observation",
                "description": "Good practice noted in area 5",
                "iso_clause": "9.1",
            },
        )
        self.assertEqual(resp.status_code, 201)
        self.assertIsNone(resp.json().get("ncr_id"))

    def test_ncr_has_auto_created_project(self):
        """NCR auto-creation also creates a core.Project for evidence linking."""
        resp = _post(
            self.c,
            "/api/iso/audits/",
            {
                "title": "Project Link Audit",
                "audit_type": "internal",
                "scheduled_date": "2026-03-22",
            },
        )
        audit_id = resp.json()["id"]

        resp = _post(
            self.c,
            f"/api/iso/audits/{audit_id}/findings/",
            {
                "finding_type": "nc_major",
                "description": "Calibration records missing for CMM",
                "iso_clause": "7.1.5",
            },
        )
        ncr_id = resp.json()["ncr_id"]

        resp = self.c.get(f"/api/iso/ncrs/{ncr_id}/")
        ncr = resp.json()
        self.assertIsNotNone(ncr.get("project_id"))
        self.assertTrue(Project.objects.filter(id=ncr["project_id"]).exists())


# ============================================================================
# 2. NCR lifecycle → evidence on close
# ============================================================================


@SECURE_OFF
class NCRLifecycleEvidenceTest(TestCase):
    """NCR transitions through states and creates evidence on close."""

    def setUp(self):
        self.user = _user("ncr_lifecycle@test.com")
        self.c = _api(self.user)

    def _create_ncr(self):
        resp = _post(
            self.c,
            "/api/iso/ncrs/",
            {
                "title": "Surface finish out of spec",
                "severity": "major",
                "description": "Ra value 3.2 vs spec 1.6 on part 4217",
                "iso_clause": "8.6",
            },
        )
        self.assertEqual(resp.status_code, 201)
        return resp.json()["id"]

    def _transition_ncr(self, ncr_id, data):
        """Transition NCR via PUT on detail endpoint."""
        return _put(self.c, f"/api/iso/ncrs/{ncr_id}/", data)

    def test_ncr_full_lifecycle(self):
        ncr_id = self._create_ncr()

        # open → investigation (requires assigned_to)
        resp = self._transition_ncr(
            ncr_id,
            {
                "status": "investigation",
                "assigned_to": str(self.user.id),
            },
        )
        self.assertEqual(resp.status_code, 200)

        # investigation → capa (requires root_cause)
        resp = self._transition_ncr(ncr_id, {"status": "capa", "root_cause": "Operator error"})
        self.assertEqual(resp.status_code, 200)

        # capa → verification
        resp = self._transition_ncr(ncr_id, {"status": "verification"})
        self.assertEqual(resp.status_code, 200)

        # verification → closed (requires approved_by)
        resp = self._transition_ncr(
            ncr_id,
            {
                "status": "closed",
                "approved_by": str(self.user.id),
            },
        )
        self.assertEqual(resp.status_code, 200)

        # Verify final state
        resp = self.c.get(f"/api/iso/ncrs/{ncr_id}/")
        self.assertEqual(resp.json()["status"], "closed")

    def test_ncr_close_creates_evidence(self):
        ncr_id = self._create_ncr()

        # Get the project_id
        resp = self.c.get(f"/api/iso/ncrs/{ncr_id}/")
        project_id = resp.json().get("project_id")
        if not project_id:
            self.skipTest("NCR has no auto-project, evidence test not applicable")

        # Transition through to closed
        self._transition_ncr(
            ncr_id,
            {
                "status": "investigation",
                "assigned_to": str(self.user.id),
            },
        )
        self._transition_ncr(ncr_id, {"status": "capa", "root_cause": "Material defect"})
        self._transition_ncr(ncr_id, {"status": "verification"})
        self._transition_ncr(
            ncr_id,
            {
                "status": "closed",
                "approved_by": str(self.user.id),
            },
        )

        # Evidence should exist on the project
        evidence = Evidence.objects.filter(
            project_id=project_id,
            source_description__contains=ncr_id,
        )
        self.assertTrue(evidence.exists(), "NCR close should create evidence on linked project")

    def test_invalid_transition_rejected(self):
        ncr_id = self._create_ncr()

        # Try to jump from open → closed (invalid)
        resp = self._transition_ncr(ncr_id, {"status": "closed"})
        self.assertEqual(resp.status_code, 400)

    def test_ncr_state_machine_matches_model(self):
        """Frontend and backend state machines must agree."""
        from agents_api.models import NonconformanceRecord

        expected = {
            "open": {"investigation"},
            "investigation": {"open", "capa"},
            "capa": {"investigation", "verification"},
            "verification": {"capa", "closed"},
            "closed": {"verification"},
        }
        self.assertEqual(NonconformanceRecord.TRANSITIONS, expected)


# ============================================================================
# 3. CAPA lifecycle → RCA backflow
# ============================================================================


@SECURE_OFF
class CAPARCABackflowTest(TestCase):
    """RCA root cause backflows to linked CAPA."""

    def setUp(self):
        self.user = _user("capa_rca@test.com")
        self.c = _api(self.user)
        self.project = Project.objects.create(user=self.user, title="CAPA-RCA Test", methodology="none")

    def test_rca_root_cause_backfills_capa(self):
        # Create RCA session
        resp = _post(
            self.c,
            "/api/rca/sessions/create/",
            {
                "title": "Seal failure RCA",
                "event": "Product leak at customer site",
                "project_id": str(self.project.id),
            },
        )
        self.assertIn(resp.status_code, (200, 201))
        rca_id = resp.json()["session"]["id"]

        # Create CAPA linked to RCA
        resp = _post(
            self.c,
            "/api/capa/",
            {
                "title": "CAPA for seal failure",
                "source_type": "rca",
                "source_id": str(rca_id),
                "rca_session_id": str(rca_id),
                "priority": "high",
                "project_id": str(self.project.id),
            },
        )
        self.assertEqual(resp.status_code, 201)
        capa_id = resp.json()["id"]

        # Set root cause on RCA
        resp = self.c.put(
            f"/api/rca/sessions/{rca_id}/update/",
            {"root_cause": "Seal material degrades above 80°C — operating temp 95°C"},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)

        # CAPA should now have the root cause backfilled
        resp = self.c.get(f"/api/capa/{capa_id}/")
        self.assertEqual(resp.status_code, 200)
        capa = resp.json()
        self.assertIn("80°C", capa.get("root_cause", ""), "RCA root cause should backflow to linked CAPA")


# ============================================================================
# 4. FMEA → hypothesis link → Bayesian update
# ============================================================================


@SECURE_OFF
class FMEAHypothesisLinkTest(TestCase):
    """FMEA row linked to hypothesis creates evidence and updates posterior."""

    def setUp(self):
        self.user = _user("fmea_hyp@test.com", tier=Tier.PRO)
        self.c = _api(self.user)
        self.project = Project.objects.create(user=self.user, title="FMEA Integration", methodology="none")
        self.hypothesis = Hypothesis.objects.create(
            project=self.project,
            statement="Material degradation is root cause",
            prior_probability=0.5,
            current_probability=0.5,
        )

    def test_link_row_creates_evidence_and_updates_posterior(self):
        # Create FMEA
        resp = _post(
            self.c,
            "/api/fmea/create/",
            {
                "title": "Process FMEA",
                "fmea_type": "process",
                "project_id": str(self.project.id),
            },
        )
        self.assertEqual(resp.status_code, 200)
        fmea_id = resp.json()["id"]

        # Add row
        resp = _post(
            self.c,
            f"/api/fmea/{fmea_id}/rows/",
            {
                "failure_mode": "Material degradation",
                "effect": "Seal failure",
                "cause": "Operating above rated temperature",
                "severity": 8,
                "occurrence": 5,
                "detection": 6,
            },
        )
        self.assertEqual(resp.status_code, 200)
        row_id = resp.json()["row"]["id"]

        # Link to hypothesis
        resp = _post(
            self.c,
            f"/api/fmea/{fmea_id}/rows/{row_id}/link/",
            {
                "hypothesis_id": str(self.hypothesis.id),
            },
        )
        self.assertEqual(resp.status_code, 200)

        # Evidence should exist
        evidence = Evidence.objects.filter(
            project=self.project,
            source_description__contains="fmea",
        )
        self.assertTrue(evidence.exists(), "FMEA link should create evidence")

        # EvidenceLink should exist
        links = EvidenceLink.objects.filter(hypothesis=self.hypothesis)
        self.assertTrue(links.exists(), "FMEA link should create EvidenceLink")

        # Posterior should have moved from prior
        self.hypothesis.refresh_from_db()
        self.assertNotEqual(
            self.hypothesis.current_probability, 0.5, "Bayesian update should change posterior from prior"
        )


# ============================================================================
# 5. Training → Document linkage
# ============================================================================


@SECURE_OFF
class TrainingDocumentLinkTest(TestCase):
    """Training requirements can link to controlled documents."""

    def setUp(self):
        self.user = _user("training_doc@test.com")
        self.c = _api(self.user)

    def test_training_linked_to_document(self):
        # Create document
        resp = _post(
            self.c,
            "/api/iso/documents/",
            {
                "title": "Weld Inspection SOP",
                "document_number": "SOP-042",
                "category": "procedure",
                "content": "Standard weld inspection procedure...",
            },
        )
        self.assertEqual(resp.status_code, 201)
        doc_id = resp.json()["id"]

        # Create training requirement linked to document
        resp = _post(
            self.c,
            "/api/iso/training/",
            {
                "title": "Weld Inspector Certification",
                "iso_clause": "7.2",
                "document_id": doc_id,
            },
        )
        self.assertEqual(resp.status_code, 201)
        req = resp.json()
        self.assertEqual(req["document_id"], doc_id)

    def test_training_record_creation(self):
        # Create requirement
        resp = _post(
            self.c,
            "/api/iso/training/",
            {
                "title": "GD&T Level II",
                "iso_clause": "7.2",
            },
        )
        req_id = resp.json()["id"]

        # Add record
        resp = _post(
            self.c,
            f"/api/iso/training/{req_id}/records/",
            {
                "employee_name": "A. Chen",
                "status": "complete",
                "completed_at": "2026-03-01",
            },
        )
        self.assertEqual(resp.status_code, 201)
        rec = resp.json()
        self.assertEqual(rec["status"], "complete")


# ============================================================================
# 6. Supplier state machine
# ============================================================================


@SECURE_OFF
class SupplierStateMachineTest(TestCase):
    """Supplier lifecycle transitions are enforced."""

    def setUp(self):
        self.user = _user("supplier@test.com")
        self.c = _api(self.user)

    def _create_supplier(self):
        resp = _post(
            self.c,
            "/api/iso/suppliers/",
            {
                "name": "Alcoa Fastening",
                "contact_email": "quality@alcoa.test",
            },
        )
        self.assertEqual(resp.status_code, 201)
        return resp.json()["id"]

    def test_supplier_starts_pending(self):
        sid = self._create_supplier()
        resp = self.c.get(f"/api/iso/suppliers/{sid}/")
        self.assertEqual(resp.json()["status"], "pending")

    def _transition(self, sid, data):
        return _put(self.c, f"/api/iso/suppliers/{sid}/", data)

    def test_supplier_approval_requires_quality_rating(self):
        sid = self._create_supplier()
        resp = self._transition(sid, {"status": "approved"})
        # Should fail without quality_rating
        self.assertEqual(resp.status_code, 400)

    def test_supplier_full_lifecycle(self):
        sid = self._create_supplier()

        # pending → approved (with rating)
        resp = self._transition(
            sid,
            {
                "status": "approved",
                "quality_rating": 4,
            },
        )
        self.assertEqual(resp.status_code, 200)

        # approved → suspended (requires notes)
        resp = self._transition(
            sid,
            {
                "status": "suspended",
                "notes": "Quality issue on lot 4217",
            },
        )
        self.assertEqual(resp.status_code, 200)

        # suspended → approved (re-qualify)
        resp = self._transition(
            sid,
            {
                "status": "approved",
                "quality_rating": 3,
            },
        )
        self.assertEqual(resp.status_code, 200)

    def test_invalid_transition_rejected(self):
        sid = self._create_supplier()
        # pending → preferred is not valid
        resp = self._transition(sid, {"status": "preferred"})
        self.assertEqual(resp.status_code, 400)


# ============================================================================
# 7. QMS dashboard aggregation
# ============================================================================


@SECURE_OFF
class QMSDashboardIntegrationTest(TestCase):
    """Dashboard pulls data from all QMS modules."""

    def setUp(self):
        self.user = _user("dashboard@test.com")
        self.c = _api(self.user)

    def test_dashboard_returns_all_sections(self):
        resp = self.c.get("/api/iso/dashboard/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        # Must contain all module sections
        self.assertIn("clause_coverage", data)
        self.assertIn("ncrs", data)
        self.assertIn("upcoming_audits", data)
        self.assertIn("training", data)
        self.assertIn("last_review", data)
        self.assertIn("capa_due_soon", data)
        self.assertIn("documents", data)
        self.assertIn("suppliers", data)

    def test_dashboard_ncrs_structure(self):
        resp = self.c.get("/api/iso/dashboard/")
        ncrs = resp.json()["ncrs"]
        self.assertIn("open", ncrs)
        self.assertIn("by_severity", ncrs)
        self.assertIn("overdue_capas", ncrs)
        self.assertIn("trend", ncrs)

    def test_dashboard_clause_coverage_has_all_top_level(self):
        resp = self.c.get("/api/iso/dashboard/")
        clauses = resp.json()["clause_coverage"]
        clause_nums = {c["clause"] for c in clauses}
        self.assertEqual(clause_nums, {"4", "5", "6", "7", "8", "9", "10"})

    def test_dashboard_reflects_created_ncr(self):
        """Creating an NCR should be reflected in dashboard counts."""
        # Baseline
        resp = self.c.get("/api/iso/dashboard/")
        baseline = resp.json()["ncrs"]["open"]

        # Create NCR
        _post(
            self.c,
            "/api/iso/ncrs/",
            {
                "title": "Dashboard test NCR",
                "severity": "minor",
                "description": "Test NCR for dashboard count verification",
            },
        )

        # Dashboard should show +1
        resp = self.c.get("/api/iso/dashboard/")
        self.assertEqual(resp.json()["ncrs"]["open"], baseline + 1)


# ============================================================================
# 8. Landing page accuracy
# ============================================================================


class LandingPageAccuracyTest(TestCase):
    """Marketing claims must match actual model state machines."""

    def test_ncr_has_five_states_not_six(self):
        """NCR model has 5 states. Landing page must not claim 6-state NCR workflow.

        The 6-state workflow (with containment) is the CAPA model, not NCR.
        If this test fails, either add containment to NCR or fix the landing page.
        """
        from agents_api.models import NonconformanceRecord

        states = [c[0] for c in NonconformanceRecord.Status.choices]
        self.assertEqual(len(states), 5)
        self.assertNotIn("containment", states)
        self.assertEqual(states, ["open", "investigation", "capa", "verification", "closed"])

    def test_capa_has_six_states_with_containment(self):
        """CAPA model has 6 states including containment."""
        from agents_api.models import CAPAReport

        states = [c[0] for c in CAPAReport.Status.choices]
        self.assertEqual(len(states), 6)
        self.assertIn("containment", states)
