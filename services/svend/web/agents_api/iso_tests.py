"""ISO 9001 QMS comprehensive tests — 28 endpoints, workflow enforcement, evidence hooks.

Covers:
- Tier gating (@require_team — FREE/PRO blocked, Team/Enterprise allowed)
- NCR CRUD + full workflow (open → investigation → capa → verification → closed)
- NCR evidence hooks (root_cause, corrective_action, verification_result)
- NCR auto-Study creation, launch-RCA, file attachment
- NCR stats endpoint
- Internal Audit CRUD + findings + auto-NCR cascade for NC findings
- Audit workflow enforcement (complete requires findings, report_issued requires closed findings)
- Audit Checklists CRUD
- Training Matrix (requirements, records, completion tracking, expiry)
- Management Review auto-snapshot (NCR, audit, training summaries)
- Document Control CRUD + version management + revision snapshots + workflow
- Supplier Management CRUD + workflow transitions + evaluation score auto-rating
- Study Actions (raise-capa, schedule-audit, request-doc-update, flag-training-gap, flag-fmea-update)
- Dashboard KPIs
- User isolation (owner-based filtering)
"""

import json
from datetime import date, timedelta

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.utils import timezone

from accounts.constants import Tier
from agents_api.models import (
    NonconformanceRecord,
    InternalAudit,
    AuditFinding,
    AuditChecklist,
    TrainingRequirement,
    TrainingRecord,
    ManagementReview,
    ControlledDocument,
    DocumentRevision,
    SupplierRecord,
    FMEA,
    Report,
)
from core.models import Project, Evidence, StudyAction

User = get_user_model()
SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)


def _post(client, url, data=None):
    return client.post(url, json.dumps(data or {}), content_type="application/json")


def _put(client, url, data=None):
    return client.put(url, json.dumps(data or {}), content_type="application/json")


def _delete(client, url, data=None):
    body = json.dumps(data) if data else "{}"
    return client.delete(url, body, content_type="application/json")


def _make_team_user(email, **kwargs):
    username = kwargs.pop("username", email.split("@")[0])
    user = User.objects.create_user(
        username=username, email=email, password="testpass123!", **kwargs
    )
    user.tier = Tier.TEAM
    user.save(update_fields=["tier"])
    return user


def _make_free_user(email, **kwargs):
    username = kwargs.pop("username", email.split("@")[0])
    user = User.objects.create_user(
        username=username, email=email, password="testpass123!", **kwargs
    )
    user.tier = Tier.FREE
    user.save(update_fields=["tier"])
    return user


# =============================================================================
# Tier Gating
# =============================================================================

@SECURE_OFF
class TierGatingTest(TestCase):
    """Verify @require_team blocks free/pro users from QMS endpoints."""

    def setUp(self):
        self.free_user = _make_free_user("free@test.com")
        self.team_user = _make_team_user("team@test.com")

    def test_free_user_blocked_from_dashboard(self):
        self.client.force_login(self.free_user)
        resp = self.client.get("/api/iso/dashboard/")
        self.assertEqual(resp.status_code, 403)
        self.assertIn("Team plan required", resp.json()["error"])

    def test_free_user_blocked_from_ncrs(self):
        self.client.force_login(self.free_user)
        resp = self.client.get("/api/iso/ncrs/")
        self.assertEqual(resp.status_code, 403)

    def test_free_user_blocked_from_audits(self):
        self.client.force_login(self.free_user)
        resp = self.client.get("/api/iso/audits/")
        self.assertEqual(resp.status_code, 403)

    def test_team_user_allowed(self):
        self.client.force_login(self.team_user)
        resp = self.client.get("/api/iso/dashboard/")
        self.assertEqual(resp.status_code, 200)

    def test_unauthenticated_blocked(self):
        resp = self.client.get("/api/iso/dashboard/")
        self.assertEqual(resp.status_code, 401)


# =============================================================================
# NCR CRUD
# =============================================================================

@SECURE_OFF
class NCRCrudTest(TestCase):
    """NCR create, read, update, delete."""

    def setUp(self):
        self.user = _make_team_user("ncrcrud@test.com")
        self.client.force_login(self.user)

    def test_create_ncr(self):
        resp = _post(self.client, "/api/iso/ncrs/", {
            "title": "Defective weld on frame",
            "description": "Crack found during inspection",
            "severity": "major",
            "source": "process",
            "iso_clause": "8.7",
        })
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data["title"], "Defective weld on frame")
        self.assertEqual(data["severity"], "major")
        self.assertEqual(data["status"], "open")
        self.assertIsNotNone(data["project_id"])  # auto-Study

    def test_list_ncrs(self):
        _post(self.client, "/api/iso/ncrs/", {"title": "NCR-001"})
        _post(self.client, "/api/iso/ncrs/", {"title": "NCR-002"})
        resp = self.client.get("/api/iso/ncrs/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.json()), 2)

    def test_list_ncrs_filter_by_status(self):
        _post(self.client, "/api/iso/ncrs/", {"title": "NCR-open"})
        resp = self.client.get("/api/iso/ncrs/?status=open")
        self.assertEqual(len(resp.json()), 1)
        resp = self.client.get("/api/iso/ncrs/?status=closed")
        self.assertEqual(len(resp.json()), 0)

    def test_list_ncrs_filter_by_severity(self):
        _post(self.client, "/api/iso/ncrs/", {"title": "Minor", "severity": "minor"})
        _post(self.client, "/api/iso/ncrs/", {"title": "Critical", "severity": "critical"})
        resp = self.client.get("/api/iso/ncrs/?severity=critical")
        items = resp.json()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["severity"], "critical")

    def test_get_ncr_detail(self):
        ncr = _post(self.client, "/api/iso/ncrs/", {"title": "Detail NCR"}).json()
        resp = self.client.get(f"/api/iso/ncrs/{ncr['id']}/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["title"], "Detail NCR")

    def test_update_ncr_fields(self):
        ncr = _post(self.client, "/api/iso/ncrs/", {"title": "Update NCR"}).json()
        resp = _put(self.client, f"/api/iso/ncrs/{ncr['id']}/", {
            "title": "Updated Title",
            "containment_action": "Isolated batch",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["title"], "Updated Title")
        self.assertEqual(resp.json()["containment_action"], "Isolated batch")

    def test_delete_ncr(self):
        ncr = _post(self.client, "/api/iso/ncrs/", {"title": "Delete NCR"}).json()
        resp = self.client.delete(f"/api/iso/ncrs/{ncr['id']}/")
        self.assertEqual(resp.status_code, 200)
        resp2 = self.client.get(f"/api/iso/ncrs/{ncr['id']}/")
        self.assertEqual(resp2.status_code, 404)

    def test_ncr_not_found_returns_404(self):
        import uuid
        resp = self.client.get(f"/api/iso/ncrs/{uuid.uuid4()}/")
        self.assertEqual(resp.status_code, 404)


# =============================================================================
# NCR Workflow
# =============================================================================

@SECURE_OFF
class NCRWorkflowTest(TestCase):
    """NCR status transitions with workflow enforcement."""

    def setUp(self):
        self.user = _make_team_user("ncrflow@test.com")
        self.client.force_login(self.user)
        self.ncr = _post(self.client, "/api/iso/ncrs/", {
            "title": "Workflow NCR",
            "severity": "major",
        }).json()

    def test_open_to_investigation_requires_assigned_to(self):
        """Cannot go to investigation without assigned_to."""
        resp = _put(self.client, f"/api/iso/ncrs/{self.ncr['id']}/", {
            "status": "investigation",
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("assigned_to", resp.json()["error"])

    def test_open_to_investigation_with_assigned_to(self):
        resp = _put(self.client, f"/api/iso/ncrs/{self.ncr['id']}/", {
            "status": "investigation",
            "assigned_to": self.user.id,
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "investigation")

    def test_full_workflow_open_to_closed(self):
        """Walk through the full NCR workflow: open → investigation → capa → verification → closed."""
        ncr_id = self.ncr["id"]

        # open → investigation
        _put(self.client, f"/api/iso/ncrs/{ncr_id}/", {
            "status": "investigation",
            "assigned_to": self.user.id,
        })
        # investigation → capa
        _put(self.client, f"/api/iso/ncrs/{ncr_id}/", {
            "status": "capa",
            "root_cause": "Improper fixturing",
        })
        # capa → verification
        _put(self.client, f"/api/iso/ncrs/{ncr_id}/", {
            "status": "verification",
            "corrective_action": "New fixture design installed",
        })
        # verification → closed
        resp = _put(self.client, f"/api/iso/ncrs/{ncr_id}/", {
            "status": "closed",
            "approved_by": self.user.id,
            "verification_result": "10 samples passed",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "closed")
        self.assertIsNotNone(resp.json()["closed_at"])

    def test_closed_requires_approved_by(self):
        """Cannot close without approved_by."""
        ncr_id = self.ncr["id"]
        _put(self.client, f"/api/iso/ncrs/{ncr_id}/", {
            "status": "investigation", "assigned_to": self.user.id,
        })
        _put(self.client, f"/api/iso/ncrs/{ncr_id}/", {"status": "capa"})
        _put(self.client, f"/api/iso/ncrs/{ncr_id}/", {"status": "verification"})
        resp = _put(self.client, f"/api/iso/ncrs/{ncr_id}/", {"status": "closed"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("approved_by", resp.json()["error"])

    def test_invalid_transition_blocked(self):
        """Cannot skip steps (open → capa)."""
        resp = _put(self.client, f"/api/iso/ncrs/{self.ncr['id']}/", {
            "status": "capa",
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Cannot transition", resp.json()["error"])

    def test_backward_transition_allowed(self):
        """investigation → open is valid (revert)."""
        ncr_id = self.ncr["id"]
        _put(self.client, f"/api/iso/ncrs/{ncr_id}/", {
            "status": "investigation", "assigned_to": self.user.id,
        })
        resp = _put(self.client, f"/api/iso/ncrs/{ncr_id}/", {"status": "open"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "open")

    def test_status_changes_recorded(self):
        """Status changes are tracked in status_changes list."""
        ncr_id = self.ncr["id"]
        _put(self.client, f"/api/iso/ncrs/{ncr_id}/", {
            "status": "investigation",
            "assigned_to": self.user.id,
            "status_note": "Starting investigation",
        })
        ncr = self.client.get(f"/api/iso/ncrs/{ncr_id}/").json()
        self.assertEqual(len(ncr["status_changes"]), 1)
        sc = ncr["status_changes"][0]
        self.assertEqual(sc["from_status"], "open")
        self.assertEqual(sc["to_status"], "investigation")
        self.assertEqual(sc["note"], "Starting investigation")


# =============================================================================
# NCR Evidence Hooks & Auto-Study
# =============================================================================

@SECURE_OFF
class NCREvidenceHooksTest(TestCase):
    """Verify NCR field updates create Evidence records via evidence bridge."""

    def setUp(self):
        self.user = _make_team_user("ncrevidence@test.com")
        self.client.force_login(self.user)
        self.ncr = _post(self.client, "/api/iso/ncrs/", {
            "title": "Evidence NCR",
            "severity": "critical",
        }).json()

    def test_auto_study_created(self):
        """NCR creation auto-creates a core.Project tagged ['auto-created', 'ncr']."""
        project_id = self.ncr["project_id"]
        self.assertIsNotNone(project_id)
        project = Project.objects.get(id=project_id)
        self.assertIn("auto-created", project.tags)
        self.assertIn("ncr", project.tags)

    def test_root_cause_creates_evidence(self):
        """Updating root_cause creates evidence in the auto-Study."""
        ncr_id = self.ncr["id"]
        _put(self.client, f"/api/iso/ncrs/{ncr_id}/", {
            "root_cause": "Operator training gap on torque specification",
        })
        project_id = self.ncr["project_id"]
        evidence = Evidence.objects.filter(project_id=project_id)
        rc_evidence = [e for e in evidence if "root_cause" in (e.source_description or "")]
        self.assertTrue(len(rc_evidence) > 0)

    def test_corrective_action_creates_evidence(self):
        ncr_id = self.ncr["id"]
        _put(self.client, f"/api/iso/ncrs/{ncr_id}/", {
            "corrective_action": "Installed poka-yoke torque tool",
        })
        project_id = self.ncr["project_id"]
        evidence = Evidence.objects.filter(project_id=project_id)
        ca_evidence = [e for e in evidence if "corrective_action" in (e.source_description or "")]
        self.assertTrue(len(ca_evidence) > 0)

    def test_verification_result_creates_evidence(self):
        ncr_id = self.ncr["id"]
        _put(self.client, f"/api/iso/ncrs/{ncr_id}/", {
            "verification_result": "50 consecutive samples within spec",
        })
        project_id = self.ncr["project_id"]
        evidence = Evidence.objects.filter(project_id=project_id)
        vr_evidence = [e for e in evidence if "verification_result" in (e.source_description or "")]
        self.assertTrue(len(vr_evidence) > 0)

    def test_close_creates_closure_evidence(self):
        """Closing NCR creates a closure evidence record."""
        ncr_id = self.ncr["id"]
        _put(self.client, f"/api/iso/ncrs/{ncr_id}/", {
            "status": "investigation", "assigned_to": self.user.id,
        })
        _put(self.client, f"/api/iso/ncrs/{ncr_id}/", {"status": "capa"})
        _put(self.client, f"/api/iso/ncrs/{ncr_id}/", {"status": "verification"})
        _put(self.client, f"/api/iso/ncrs/{ncr_id}/", {
            "status": "closed", "approved_by": self.user.id,
        })
        project_id = self.ncr["project_id"]
        evidence = Evidence.objects.filter(project_id=project_id)
        close_ev = [e for e in evidence if "status_closed" in (e.source_description or "")]
        self.assertTrue(len(close_ev) > 0)


# =============================================================================
# NCR Launch RCA
# =============================================================================

@SECURE_OFF
class NCRLaunchRCATest(TestCase):
    """Launch-RCA creates an RCA session linked to the NCR."""

    def setUp(self):
        self.user = _make_team_user("ncrrca@test.com")
        self.client.force_login(self.user)

    def test_launch_rca(self):
        ncr = _post(self.client, "/api/iso/ncrs/", {
            "title": "RCA Test NCR",
            "description": "Recurring dimensional nonconformance",
        }).json()
        resp = _post(self.client, f"/api/iso/ncrs/{ncr['id']}/launch-rca/")
        self.assertEqual(resp.status_code, 201)
        rca_id = resp.json()["rca_session_id"]
        self.assertIsNotNone(rca_id)
        # NCR now has rca_session_id
        ncr_detail = self.client.get(f"/api/iso/ncrs/{ncr['id']}/").json()
        self.assertEqual(ncr_detail["rca_session_id"], rca_id)

    def test_launch_rca_not_found(self):
        import uuid
        resp = _post(self.client, f"/api/iso/ncrs/{uuid.uuid4()}/launch-rca/")
        self.assertEqual(resp.status_code, 404)


# =============================================================================
# NCR Stats
# =============================================================================

@SECURE_OFF
class NCRStatsTest(TestCase):
    """NCR stats endpoint returns correct aggregates."""

    def setUp(self):
        self.user = _make_team_user("ncrstats@test.com")
        self.client.force_login(self.user)

    def test_stats_empty(self):
        resp = self.client.get("/api/iso/ncrs/stats/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total"], 0)
        self.assertEqual(data["open"], 0)
        self.assertEqual(data["overdue_capas"], 0)

    def test_stats_counts(self):
        _post(self.client, "/api/iso/ncrs/", {"title": "NCR-A"})
        _post(self.client, "/api/iso/ncrs/", {"title": "NCR-B"})
        resp = self.client.get("/api/iso/ncrs/stats/")
        data = resp.json()
        self.assertEqual(data["total"], 2)
        self.assertEqual(data["open"], 2)

    def test_overdue_capas(self):
        """NCR with past due CAPA shows as overdue."""
        _post(self.client, "/api/iso/ncrs/", {
            "title": "Overdue NCR",
            "capa_due_date": str(date.today() - timedelta(days=7)),
        })
        resp = self.client.get("/api/iso/ncrs/stats/")
        self.assertEqual(resp.json()["overdue_capas"], 1)


# =============================================================================
# Internal Audit CRUD
# =============================================================================

@SECURE_OFF
class AuditCrudTest(TestCase):
    """Internal audit create, read, update, delete."""

    def setUp(self):
        self.user = _make_team_user("auditcrud@test.com")
        self.client.force_login(self.user)

    def test_create_audit(self):
        resp = _post(self.client, "/api/iso/audits/", {
            "title": "Process Audit Q1",
            "scheduled_date": str(date.today() + timedelta(days=14)),
            "lead_auditor": "Jane Smith",
            "iso_clauses": ["7.1", "8.5"],
            "departments": ["Production", "QA"],
            "scope": "Review of production controls",
        })
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data["title"], "Process Audit Q1")
        self.assertEqual(data["status"], "planned")
        self.assertEqual(data["iso_clauses"], ["7.1", "8.5"])

    def test_list_audits(self):
        _post(self.client, "/api/iso/audits/", {"title": "Audit A"})
        _post(self.client, "/api/iso/audits/", {"title": "Audit B"})
        resp = self.client.get("/api/iso/audits/")
        self.assertEqual(len(resp.json()), 2)

    def test_list_audits_filter_by_status(self):
        _post(self.client, "/api/iso/audits/", {"title": "A"})
        resp = self.client.get("/api/iso/audits/?status=planned")
        self.assertEqual(len(resp.json()), 1)
        resp = self.client.get("/api/iso/audits/?status=complete")
        self.assertEqual(len(resp.json()), 0)

    def test_update_audit(self):
        audit = _post(self.client, "/api/iso/audits/", {"title": "X"}).json()
        resp = _put(self.client, f"/api/iso/audits/{audit['id']}/", {
            "title": "Updated Audit",
            "summary": "All clear",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["title"], "Updated Audit")
        self.assertEqual(resp.json()["summary"], "All clear")

    def test_delete_audit(self):
        audit = _post(self.client, "/api/iso/audits/", {"title": "Del"}).json()
        resp = self.client.delete(f"/api/iso/audits/{audit['id']}/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(self.client.get(f"/api/iso/audits/{audit['id']}/").status_code, 404)


# =============================================================================
# Audit Workflow & Findings
# =============================================================================

@SECURE_OFF
class AuditWorkflowTest(TestCase):
    """Audit workflow enforcement: complete needs findings, report needs closed findings."""

    def setUp(self):
        self.user = _make_team_user("auditflow@test.com")
        self.client.force_login(self.user)
        self.audit = _post(self.client, "/api/iso/audits/", {
            "title": "Workflow Audit",
        }).json()

    def test_cannot_complete_without_findings(self):
        resp = _put(self.client, f"/api/iso/audits/{self.audit['id']}/", {
            "status": "complete",
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("no findings", resp.json()["error"])

    def test_complete_with_findings(self):
        """Add a finding, then complete the audit."""
        _post(self.client, f"/api/iso/audits/{self.audit['id']}/findings/", {
            "finding_type": "observation",
            "description": "Good housekeeping practices observed",
        })
        resp = _put(self.client, f"/api/iso/audits/{self.audit['id']}/", {
            "status": "complete",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "complete")
        self.assertIsNotNone(resp.json()["completed_date"])

    def test_cannot_issue_report_with_open_findings(self):
        """report_issued blocked when open findings exist."""
        _post(self.client, f"/api/iso/audits/{self.audit['id']}/findings/", {
            "finding_type": "observation",
            "description": "Open finding",
            "status": "open",
        })
        _put(self.client, f"/api/iso/audits/{self.audit['id']}/", {
            "status": "complete",
        })
        resp = _put(self.client, f"/api/iso/audits/{self.audit['id']}/", {
            "status": "report_issued",
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("still open", resp.json()["error"])

    def test_report_issued_after_closing_findings(self):
        """Close all findings, then issue report."""
        finding = _post(self.client, f"/api/iso/audits/{self.audit['id']}/findings/", {
            "finding_type": "observation",
            "description": "Finding",
            "status": "closed",
        }).json()
        _put(self.client, f"/api/iso/audits/{self.audit['id']}/", {"status": "complete"})
        resp = _put(self.client, f"/api/iso/audits/{self.audit['id']}/", {
            "status": "report_issued",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "report_issued")


# =============================================================================
# Audit Finding Auto-NCR
# =============================================================================

@SECURE_OFF
class AuditFindingAutoNCRTest(TestCase):
    """NC findings auto-create NCRs with correct severity mapping."""

    def setUp(self):
        self.user = _make_team_user("findingncr@test.com")
        self.client.force_login(self.user)
        self.audit = _post(self.client, "/api/iso/audits/", {
            "title": "NCR Audit",
        }).json()

    def test_nc_major_creates_critical_ncr(self):
        resp = _post(self.client, f"/api/iso/audits/{self.audit['id']}/findings/", {
            "finding_type": "nc_major",
            "description": "Major nonconformity in process control",
            "iso_clause": "8.5.1",
        })
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertIn("ncr_id", data)
        ncr = NonconformanceRecord.objects.get(id=data["ncr_id"])
        self.assertEqual(ncr.severity, "critical")
        self.assertEqual(ncr.source, "internal_audit")
        self.assertEqual(ncr.iso_clause, "8.5.1")
        # Auto-Study should be created for the NCR
        self.assertIsNotNone(ncr.project_id)

    def test_nc_minor_creates_major_ncr(self):
        resp = _post(self.client, f"/api/iso/audits/{self.audit['id']}/findings/", {
            "finding_type": "nc_minor",
            "description": "Minor documentation gap",
        })
        data = resp.json()
        ncr = NonconformanceRecord.objects.get(id=data["ncr_id"])
        self.assertEqual(ncr.severity, "major")

    def test_observation_no_ncr(self):
        resp = _post(self.client, f"/api/iso/audits/{self.audit['id']}/findings/", {
            "finding_type": "observation",
            "description": "Good practice noted",
        })
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertIsNone(data.get("ncr_id"))

    def test_opportunity_no_ncr(self):
        resp = _post(self.client, f"/api/iso/audits/{self.audit['id']}/findings/", {
            "finding_type": "opportunity",
            "description": "Could improve labeling",
        })
        data = resp.json()
        self.assertIsNone(data.get("ncr_id"))

    def test_finding_linked_to_ncr_bidirectional(self):
        """Finding has ncr_id, and the NCR is retrievable."""
        resp = _post(self.client, f"/api/iso/audits/{self.audit['id']}/findings/", {
            "finding_type": "nc_major",
            "description": "Calibration overdue",
        })
        ncr_id = resp.json()["ncr_id"]
        ncr_detail = self.client.get(f"/api/iso/ncrs/{ncr_id}/").json()
        self.assertIn("Audit Finding", ncr_detail["title"])


# =============================================================================
# Audit Checklists
# =============================================================================

@SECURE_OFF
class AuditChecklistTest(TestCase):
    """Audit checklist CRUD."""

    def setUp(self):
        self.user = _make_team_user("checklist@test.com")
        self.client.force_login(self.user)

    def test_create_checklist(self):
        resp = _post(self.client, "/api/iso/checklists/", {
            "name": "Production Audit Checklist",
            "iso_clause": "8.5",
            "check_items": [
                {"question": "Are work instructions posted?", "guidance": "Check each station"},
                {"question": "Is calibration current?", "guidance": "Review cal stickers"},
            ],
        })
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data["name"], "Production Audit Checklist")
        self.assertEqual(len(data["check_items"]), 2)

    def test_list_checklists(self):
        _post(self.client, "/api/iso/checklists/", {"name": "CL-1"})
        _post(self.client, "/api/iso/checklists/", {"name": "CL-2"})
        resp = self.client.get("/api/iso/checklists/")
        self.assertEqual(len(resp.json()), 2)

    def test_update_checklist(self):
        cl = _post(self.client, "/api/iso/checklists/", {"name": "Original"}).json()
        resp = _put(self.client, f"/api/iso/checklists/{cl['id']}/", {
            "name": "Updated Checklist",
            "check_items": [{"question": "New item"}],
        })
        self.assertEqual(resp.json()["name"], "Updated Checklist")
        self.assertEqual(len(resp.json()["check_items"]), 1)

    def test_delete_checklist(self):
        cl = _post(self.client, "/api/iso/checklists/", {"name": "Del"}).json()
        resp = self.client.delete(f"/api/iso/checklists/{cl['id']}/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(self.client.get(f"/api/iso/checklists/{cl['id']}/").status_code, 404)


# =============================================================================
# Training Matrix
# =============================================================================

@SECURE_OFF
class TrainingMatrixTest(TestCase):
    """Training requirements and records CRUD + completion tracking."""

    def setUp(self):
        self.user = _make_team_user("training@test.com")
        self.client.force_login(self.user)

    def test_create_training_requirement(self):
        resp = _post(self.client, "/api/iso/training/", {
            "name": "Torque Wrench Operation",
            "description": "Proper use of calibrated torque tools",
            "iso_clause": "7.2",
            "frequency_months": 12,
            "is_mandatory": True,
        })
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data["name"], "Torque Wrench Operation")
        self.assertEqual(data["frequency_months"], 12)
        self.assertTrue(data["is_mandatory"])

    def test_list_training_requirements(self):
        _post(self.client, "/api/iso/training/", {"name": "TR-1"})
        _post(self.client, "/api/iso/training/", {"name": "TR-2"})
        resp = self.client.get("/api/iso/training/")
        self.assertEqual(len(resp.json()), 2)

    def test_update_training_requirement(self):
        req = _post(self.client, "/api/iso/training/", {"name": "Old"}).json()
        resp = _put(self.client, f"/api/iso/training/{req['id']}/", {
            "name": "Updated Training",
            "frequency_months": 6,
        })
        self.assertEqual(resp.json()["name"], "Updated Training")
        self.assertEqual(resp.json()["frequency_months"], 6)

    def test_delete_training_requirement(self):
        req = _post(self.client, "/api/iso/training/", {"name": "Del"}).json()
        resp = self.client.delete(f"/api/iso/training/{req['id']}/")
        self.assertEqual(resp.status_code, 200)

    def test_add_training_record(self):
        req = _post(self.client, "/api/iso/training/", {"name": "Record Test"}).json()
        resp = _post(self.client, f"/api/iso/training/{req['id']}/records/", {
            "employee_name": "John Doe",
            "employee_email": "john@company.com",
            "status": "not_started",
        })
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data["employee_name"], "John Doe")
        self.assertEqual(data["status"], "not_started")

    def test_complete_record_sets_timestamps(self):
        """Marking record complete sets completed_at and computes expires_at."""
        req = _post(self.client, "/api/iso/training/", {
            "name": "Expiry Test",
            "frequency_months": 6,
        }).json()
        record = _post(self.client, f"/api/iso/training/{req['id']}/records/", {
            "employee_name": "Jane",
            "status": "complete",
        }).json()
        self.assertIsNotNone(record["completed_at"])
        self.assertIsNotNone(record["expires_at"])  # 6 months * 30 days

    def test_one_time_training_no_expiry(self):
        """frequency_months=0 means no expiry date."""
        req = _post(self.client, "/api/iso/training/", {
            "name": "One-time",
            "frequency_months": 0,
        }).json()
        record = _post(self.client, f"/api/iso/training/{req['id']}/records/", {
            "employee_name": "Bob",
            "status": "complete",
        }).json()
        self.assertIsNotNone(record["completed_at"])
        self.assertIsNone(record["expires_at"])

    def test_update_record_to_complete(self):
        """Updating status to complete via PUT also sets timestamps."""
        req = _post(self.client, "/api/iso/training/", {
            "name": "Update Test",
            "frequency_months": 12,
        }).json()
        record = _post(self.client, f"/api/iso/training/{req['id']}/records/", {
            "employee_name": "Alice",
            "status": "in_progress",
        }).json()
        self.assertIsNone(record["completed_at"])

        resp = _put(self.client, f"/api/iso/training/records/{record['id']}/", {
            "status": "complete",
        })
        self.assertEqual(resp.status_code, 200)
        updated = resp.json()
        self.assertIsNotNone(updated["completed_at"])
        self.assertIsNotNone(updated["expires_at"])

    def test_delete_record(self):
        req = _post(self.client, "/api/iso/training/", {"name": "Del Rec"}).json()
        record = _post(self.client, f"/api/iso/training/{req['id']}/records/", {
            "employee_name": "Del",
        }).json()
        resp = self.client.delete(f"/api/iso/training/records/{record['id']}/")
        self.assertEqual(resp.status_code, 200)

    def test_completion_rate_in_requirement(self):
        """Requirement to_dict includes computed completion_rate."""
        req = _post(self.client, "/api/iso/training/", {"name": "Rate Test"}).json()
        _post(self.client, f"/api/iso/training/{req['id']}/records/", {
            "employee_name": "Alice", "status": "complete",
        })
        _post(self.client, f"/api/iso/training/{req['id']}/records/", {
            "employee_name": "Bob", "status": "not_started",
        })
        req_detail = self.client.get(f"/api/iso/training/{req['id']}/").json()
        self.assertEqual(req_detail["completion_rate"], 50)


# =============================================================================
# Management Review
# =============================================================================

@SECURE_OFF
class ManagementReviewTest(TestCase):
    """Management review with auto-populated data snapshot."""

    def setUp(self):
        self.user = _make_team_user("review@test.com")
        self.client.force_login(self.user)

    def test_create_review_auto_snapshot(self):
        """Creating a review auto-captures QMS metrics."""
        # Seed some data first
        _post(self.client, "/api/iso/ncrs/", {"title": "NCR-1", "severity": "minor"})
        _post(self.client, "/api/iso/ncrs/", {"title": "NCR-2", "severity": "major"})

        resp = _post(self.client, "/api/iso/reviews/", {
            "title": "Q1 Review",
        })
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data["status"], "scheduled")
        snap = data["data_snapshot"]
        self.assertIn("ncr_summary", snap)
        self.assertIn("audit_summary", snap)
        self.assertIn("training_summary", snap)
        self.assertEqual(snap["ncr_summary"]["open"], 2)
        self.assertEqual(snap["ncr_summary"]["by_severity"]["minor"], 1)
        self.assertEqual(snap["ncr_summary"]["by_severity"]["major"], 1)

    def test_review_snapshot_training_compliance(self):
        """Snapshot correctly calculates training compliance rate."""
        req = _post(self.client, "/api/iso/training/", {"name": "T1"}).json()
        _post(self.client, f"/api/iso/training/{req['id']}/records/", {
            "employee_name": "A", "status": "complete",
        })
        _post(self.client, f"/api/iso/training/{req['id']}/records/", {
            "employee_name": "B", "status": "complete",
        })
        _post(self.client, f"/api/iso/training/{req['id']}/records/", {
            "employee_name": "C", "status": "not_started",
        })
        review = _post(self.client, "/api/iso/reviews/", {}).json()
        self.assertEqual(review["data_snapshot"]["training_summary"]["compliance_rate"], 67)

    def test_review_snapshot_prior_actions(self):
        """Second review captures outputs of the first completed review."""
        rev1 = _post(self.client, "/api/iso/reviews/", {}).json()
        _put(self.client, f"/api/iso/reviews/{rev1['id']}/", {
            "status": "complete",
            "outputs": {"action_1": "Increase audit frequency"},
        })
        rev2 = _post(self.client, "/api/iso/reviews/", {}).json()
        snap = rev2["data_snapshot"]
        self.assertEqual(snap["prior_actions"]["action_1"], "Increase audit frequency")

    def test_list_reviews(self):
        _post(self.client, "/api/iso/reviews/", {})
        _post(self.client, "/api/iso/reviews/", {})
        resp = self.client.get("/api/iso/reviews/")
        self.assertEqual(len(resp.json()), 2)

    def test_update_review(self):
        rev = _post(self.client, "/api/iso/reviews/", {}).json()
        resp = _put(self.client, f"/api/iso/reviews/{rev['id']}/", {
            "status": "in_progress",
            "attendees": ["Alice", "Bob"],
            "minutes": "Discussed NCR trends.",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "in_progress")
        self.assertEqual(resp.json()["attendees"], ["Alice", "Bob"])

    def test_delete_review(self):
        rev = _post(self.client, "/api/iso/reviews/", {}).json()
        resp = self.client.delete(f"/api/iso/reviews/{rev['id']}/")
        self.assertEqual(resp.status_code, 200)


# =============================================================================
# Document Control
# =============================================================================

@SECURE_OFF
class DocumentControlTest(TestCase):
    """Document CRUD, workflow transitions, version management."""

    def setUp(self):
        self.user = _make_team_user("docctrl@test.com")
        self.client.force_login(self.user)

    def test_create_document(self):
        resp = _post(self.client, "/api/iso/documents/", {
            "title": "SOP-001 Receiving Inspection",
            "document_number": "SOP-001",
            "category": "SOP",
            "iso_clause": "8.4",
            "content": "1. Verify PO number...",
            "retention_years": 10,
        })
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data["title"], "SOP-001 Receiving Inspection")
        self.assertEqual(data["status"], "draft")
        self.assertEqual(data["current_version"], "1.0")

    def test_list_documents(self):
        _post(self.client, "/api/iso/documents/", {"title": "DOC-1"})
        _post(self.client, "/api/iso/documents/", {"title": "DOC-2"})
        resp = self.client.get("/api/iso/documents/")
        self.assertEqual(len(resp.json()), 2)

    def test_list_documents_filter_by_status(self):
        _post(self.client, "/api/iso/documents/", {"title": "D"})
        resp = self.client.get("/api/iso/documents/?status=draft")
        self.assertEqual(len(resp.json()), 1)
        resp = self.client.get("/api/iso/documents/?status=approved")
        self.assertEqual(len(resp.json()), 0)

    def test_list_documents_search(self):
        _post(self.client, "/api/iso/documents/", {"title": "Receiving Inspection", "document_number": "SOP-001"})
        _post(self.client, "/api/iso/documents/", {"title": "Shipping", "document_number": "SOP-002"})
        resp = self.client.get("/api/iso/documents/?search=Receiving")
        self.assertEqual(len(resp.json()), 1)
        resp = self.client.get("/api/iso/documents/?search=SOP-002")
        self.assertEqual(len(resp.json()), 1)

    def test_update_document(self):
        doc = _post(self.client, "/api/iso/documents/", {"title": "X"}).json()
        resp = _put(self.client, f"/api/iso/documents/{doc['id']}/", {
            "title": "Updated SOP",
            "content": "New content here",
        })
        self.assertEqual(resp.json()["title"], "Updated SOP")
        self.assertEqual(resp.json()["content"], "New content here")

    def test_delete_document(self):
        doc = _post(self.client, "/api/iso/documents/", {"title": "Del"}).json()
        resp = self.client.delete(f"/api/iso/documents/{doc['id']}/")
        self.assertEqual(resp.status_code, 200)


# =============================================================================
# Document Workflow
# =============================================================================

@SECURE_OFF
class DocumentWorkflowTest(TestCase):
    """Document status transitions + version bumping + revision snapshots."""

    def setUp(self):
        self.user = _make_team_user("docflow@test.com")
        self.client.force_login(self.user)
        self.doc = _post(self.client, "/api/iso/documents/", {
            "title": "Workflow Doc",
            "content": "Initial content v1",
        }).json()

    def test_draft_to_review_requires_content(self):
        """Cannot go to review if content is empty."""
        empty_doc = _post(self.client, "/api/iso/documents/", {
            "title": "Empty",
            "content": "",
        }).json()
        resp = _put(self.client, f"/api/iso/documents/{empty_doc['id']}/", {
            "status": "review",
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("content", resp.json()["error"])

    def test_draft_to_review(self):
        resp = _put(self.client, f"/api/iso/documents/{self.doc['id']}/", {
            "status": "review",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "review")

    def test_review_to_approved_requires_approved_by_user(self):
        _put(self.client, f"/api/iso/documents/{self.doc['id']}/", {"status": "review"})
        resp = _put(self.client, f"/api/iso/documents/{self.doc['id']}/", {
            "status": "approved",
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("approved_by_user", resp.json()["error"])

    def test_review_to_approved_with_approver(self):
        _put(self.client, f"/api/iso/documents/{self.doc['id']}/", {"status": "review"})
        resp = _put(self.client, f"/api/iso/documents/{self.doc['id']}/", {
            "status": "approved",
            "approved_by_user": self.user.id,
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "approved")
        self.assertIsNotNone(resp.json()["approved_at"])

    def test_approved_to_review_bumps_version(self):
        """approved → review creates revision snapshot and bumps 1.0 → 1.1."""
        doc_id = self.doc["id"]
        _put(self.client, f"/api/iso/documents/{doc_id}/", {"status": "review"})
        _put(self.client, f"/api/iso/documents/{doc_id}/", {
            "status": "approved", "approved_by_user": self.user.id,
        })
        # Now trigger revision cycle: approved → review
        resp = _put(self.client, f"/api/iso/documents/{doc_id}/", {
            "status": "review",
            "revision_note": "Updated per audit finding",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "review")
        self.assertEqual(data["current_version"], "1.1")
        # Revision snapshot should exist
        self.assertTrue(len(data["revisions"]) >= 1)
        rev = data["revisions"][0]
        self.assertEqual(rev["version"], "1.0")
        self.assertEqual(rev["change_summary"], "Updated per audit finding")

    def test_multiple_revision_cycles(self):
        """Multiple revision cycles bump version correctly: 1.0 → 1.1 → 1.2."""
        doc_id = self.doc["id"]
        # Cycle 1: draft → review → approved → review (bump to 1.1)
        _put(self.client, f"/api/iso/documents/{doc_id}/", {"status": "review"})
        _put(self.client, f"/api/iso/documents/{doc_id}/", {
            "status": "approved", "approved_by_user": self.user.id,
        })
        _put(self.client, f"/api/iso/documents/{doc_id}/", {"status": "review"})
        # Cycle 2: review → approved → review (bump to 1.2)
        _put(self.client, f"/api/iso/documents/{doc_id}/", {
            "status": "approved", "approved_by_user": self.user.id,
        })
        resp = _put(self.client, f"/api/iso/documents/{doc_id}/", {"status": "review"})
        self.assertEqual(resp.json()["current_version"], "1.2")
        self.assertEqual(len(resp.json()["revisions"]), 2)

    def test_approved_to_obsolete(self):
        doc_id = self.doc["id"]
        _put(self.client, f"/api/iso/documents/{doc_id}/", {"status": "review"})
        _put(self.client, f"/api/iso/documents/{doc_id}/", {
            "status": "approved", "approved_by_user": self.user.id,
        })
        resp = _put(self.client, f"/api/iso/documents/{doc_id}/", {"status": "obsolete"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "obsolete")

    def test_obsolete_is_terminal(self):
        """Cannot transition from obsolete."""
        doc_id = self.doc["id"]
        _put(self.client, f"/api/iso/documents/{doc_id}/", {"status": "review"})
        _put(self.client, f"/api/iso/documents/{doc_id}/", {
            "status": "approved", "approved_by_user": self.user.id,
        })
        _put(self.client, f"/api/iso/documents/{doc_id}/", {"status": "obsolete"})
        resp = _put(self.client, f"/api/iso/documents/{doc_id}/", {"status": "review"})
        self.assertEqual(resp.status_code, 400)

    def test_invalid_transition_blocked(self):
        """draft → approved is not valid."""
        resp = _put(self.client, f"/api/iso/documents/{self.doc['id']}/", {
            "status": "approved",
        })
        self.assertEqual(resp.status_code, 400)

    def test_status_changes_recorded(self):
        doc_id = self.doc["id"]
        _put(self.client, f"/api/iso/documents/{doc_id}/", {
            "status": "review",
            "status_note": "Ready for review",
        })
        doc = self.client.get(f"/api/iso/documents/{doc_id}/").json()
        self.assertEqual(len(doc["status_changes"]), 1)
        self.assertEqual(doc["status_changes"][0]["from_status"], "draft")
        self.assertEqual(doc["status_changes"][0]["to_status"], "review")
        self.assertEqual(doc["status_changes"][0]["note"], "Ready for review")


# =============================================================================
# Supplier Management
# =============================================================================

@SECURE_OFF
class SupplierManagementTest(TestCase):
    """Supplier CRUD + workflow transitions + evaluation scoring."""

    def setUp(self):
        self.user = _make_team_user("supplier@test.com")
        self.client.force_login(self.user)

    def test_create_supplier(self):
        resp = _post(self.client, "/api/iso/suppliers/", {
            "name": "Acme Fasteners",
            "supplier_type": "component",
            "contact_name": "Bob Jones",
            "contact_email": "bob@acme.com",
            "products_services": "M6 bolts, M8 washers",
        })
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data["name"], "Acme Fasteners")
        self.assertEqual(data["status"], "pending")
        self.assertEqual(data["supplier_type"], "component")

    def test_list_suppliers(self):
        _post(self.client, "/api/iso/suppliers/", {"name": "S1"})
        _post(self.client, "/api/iso/suppliers/", {"name": "S2"})
        resp = self.client.get("/api/iso/suppliers/")
        self.assertEqual(len(resp.json()), 2)

    def test_list_suppliers_filter(self):
        _post(self.client, "/api/iso/suppliers/", {"name": "S1", "supplier_type": "raw_material"})
        _post(self.client, "/api/iso/suppliers/", {"name": "S2", "supplier_type": "service"})
        resp = self.client.get("/api/iso/suppliers/?supplier_type=raw_material")
        self.assertEqual(len(resp.json()), 1)

    def test_list_suppliers_search(self):
        _post(self.client, "/api/iso/suppliers/", {"name": "Acme Corp", "contact_name": "Alice"})
        _post(self.client, "/api/iso/suppliers/", {"name": "Beta Inc", "contact_name": "Bob"})
        resp = self.client.get("/api/iso/suppliers/?search=Acme")
        self.assertEqual(len(resp.json()), 1)
        resp = self.client.get("/api/iso/suppliers/?search=Bob")
        self.assertEqual(len(resp.json()), 1)

    def test_update_supplier(self):
        s = _post(self.client, "/api/iso/suppliers/", {"name": "X"}).json()
        resp = _put(self.client, f"/api/iso/suppliers/{s['id']}/", {
            "name": "Updated Supplier",
            "contact_phone": "+1234567890",
        })
        self.assertEqual(resp.json()["name"], "Updated Supplier")
        self.assertEqual(resp.json()["contact_phone"], "+1234567890")

    def test_delete_supplier(self):
        s = _post(self.client, "/api/iso/suppliers/", {"name": "Del"}).json()
        resp = self.client.delete(f"/api/iso/suppliers/{s['id']}/")
        self.assertEqual(resp.status_code, 200)


# =============================================================================
# Supplier Workflow
# =============================================================================

@SECURE_OFF
class SupplierWorkflowTest(TestCase):
    """Supplier status transitions with required fields."""

    def setUp(self):
        self.user = _make_team_user("suppflow@test.com")
        self.client.force_login(self.user)
        self.supplier = _post(self.client, "/api/iso/suppliers/", {
            "name": "Workflow Supplier",
        }).json()

    def test_pending_to_approved_requires_quality_rating(self):
        resp = _put(self.client, f"/api/iso/suppliers/{self.supplier['id']}/", {
            "status": "approved",
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("quality_rating", resp.json()["error"])

    def test_pending_to_approved_with_rating(self):
        resp = _put(self.client, f"/api/iso/suppliers/{self.supplier['id']}/", {
            "status": "approved",
            "quality_rating": 4,
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "approved")
        self.assertEqual(resp.json()["quality_rating"], 4)

    def test_pending_to_conditional_requires_notes(self):
        resp = _put(self.client, f"/api/iso/suppliers/{self.supplier['id']}/", {
            "status": "conditional",
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("notes", resp.json()["error"])

    def test_pending_to_conditional_with_notes(self):
        resp = _put(self.client, f"/api/iso/suppliers/{self.supplier['id']}/", {
            "status": "conditional",
            "notes": "Pending on-site audit",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "conditional")

    def test_pending_to_disqualified_requires_reason(self):
        resp = _put(self.client, f"/api/iso/suppliers/{self.supplier['id']}/", {
            "status": "disqualified",
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("disqualification_reason", resp.json()["error"])

    def test_pending_to_disqualified_with_reason(self):
        resp = _put(self.client, f"/api/iso/suppliers/{self.supplier['id']}/", {
            "status": "disqualified",
            "disqualification_reason": "Failed incoming quality audit",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "disqualified")

    def test_disqualified_is_terminal(self):
        """Cannot transition from disqualified."""
        sid = self.supplier["id"]
        _put(self.client, f"/api/iso/suppliers/{sid}/", {
            "status": "disqualified",
            "disqualification_reason": "Critical failure",
        })
        resp = _put(self.client, f"/api/iso/suppliers/{sid}/", {
            "status": "approved", "quality_rating": 3,
        })
        self.assertEqual(resp.status_code, 400)

    def test_approved_to_suspended(self):
        sid = self.supplier["id"]
        _put(self.client, f"/api/iso/suppliers/{sid}/", {
            "status": "approved", "quality_rating": 4,
        })
        resp = _put(self.client, f"/api/iso/suppliers/{sid}/", {
            "status": "suspended",
            "notes": "Late delivery 3x",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "suspended")

    def test_suspended_to_approved(self):
        """Supplier can be reinstated from suspension."""
        sid = self.supplier["id"]
        _put(self.client, f"/api/iso/suppliers/{sid}/", {
            "status": "approved", "quality_rating": 4,
        })
        _put(self.client, f"/api/iso/suppliers/{sid}/", {
            "status": "suspended", "notes": "Issue",
        })
        resp = _put(self.client, f"/api/iso/suppliers/{sid}/", {
            "status": "approved", "quality_rating": 3,
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "approved")

    def test_status_changes_recorded(self):
        sid = self.supplier["id"]
        _put(self.client, f"/api/iso/suppliers/{sid}/", {
            "status": "approved",
            "quality_rating": 5,
            "status_note": "Initial approval",
        })
        s = self.client.get(f"/api/iso/suppliers/{sid}/").json()
        self.assertEqual(len(s["status_changes"]), 1)
        self.assertEqual(s["status_changes"][0]["from_status"], "pending")
        self.assertEqual(s["status_changes"][0]["to_status"], "approved")

    def test_evaluation_scores_auto_rating(self):
        """Setting evaluation_scores auto-computes quality_rating."""
        sid = self.supplier["id"]
        resp = _put(self.client, f"/api/iso/suppliers/{sid}/", {
            "evaluation_scores": {"quality": 5, "delivery": 3, "price": 4, "communication": 4},
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["quality_rating"], 4)  # avg(5,3,4,4)=4.0

    def test_invalid_transition_blocked(self):
        """pending → suspended is not valid."""
        resp = _put(self.client, f"/api/iso/suppliers/{self.supplier['id']}/", {
            "status": "suspended", "notes": "X",
        })
        self.assertEqual(resp.status_code, 400)


# =============================================================================
# Study Actions — raise CAPA
# =============================================================================

@SECURE_OFF
class StudyRaiseCAPATest(TestCase):
    """study_raise_capa creates a CAPA report and StudyAction."""

    def setUp(self):
        self.user = _make_team_user("studycapa@test.com")
        self.client.force_login(self.user)
        self.project = Project.objects.create(
            user=self.user,
            title="Investigation Study",
            methodology="none",
        )

    def test_raise_capa(self):
        resp = _post(self.client, "/api/iso/study-actions/raise-capa/", {
            "project_id": str(self.project.id),
        })
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["action"], "raise_capa")
        self.assertIn("report", data)
        self.assertEqual(data["report"]["report_type"], "capa")
        # StudyAction created
        actions = StudyAction.objects.filter(project=self.project, action_type="raise_capa")
        self.assertEqual(actions.count(), 1)
        self.assertEqual(str(actions.first().target_id), data["report"]["id"])

    def test_capa_pre_fills_problem(self):
        """CAPA report pre-fills problem_description from project title."""
        resp = _post(self.client, "/api/iso/study-actions/raise-capa/", {
            "project_id": str(self.project.id),
        })
        report = resp.json()["report"]
        sections = report["sections"]
        self.assertEqual(sections["problem_description"], self.project.title)

    def test_capa_pre_fills_root_cause_from_evidence(self):
        """If Study has root cause evidence, CAPA pre-fills root_cause_analysis."""
        Evidence.objects.create(
            project=self.project,
            summary="Misaligned fixture caused dimensional drift",
            source_description="ncr:root_cause",
            source_type="analysis",
        )
        resp = _post(self.client, "/api/iso/study-actions/raise-capa/", {
            "project_id": str(self.project.id),
        })
        sections = resp.json()["report"]["sections"]
        self.assertIn("fixture", sections["root_cause_analysis"])

    def test_missing_project_id_rejected(self):
        resp = _post(self.client, "/api/iso/study-actions/raise-capa/", {})
        self.assertEqual(resp.status_code, 400)

    def test_wrong_project_rejected(self):
        import uuid
        resp = _post(self.client, "/api/iso/study-actions/raise-capa/", {
            "project_id": str(uuid.uuid4()),
        })
        self.assertEqual(resp.status_code, 404)


# =============================================================================
# Study Actions — schedule audit
# =============================================================================

@SECURE_OFF
class StudyScheduleAuditTest(TestCase):
    """study_schedule_audit creates a verification audit and StudyAction."""

    def setUp(self):
        self.user = _make_team_user("studyaudit@test.com")
        self.client.force_login(self.user)
        self.project = Project.objects.create(
            user=self.user,
            title="Audit Study",
            methodology="none",
        )

    def test_schedule_audit(self):
        resp = _post(self.client, "/api/iso/study-actions/schedule-audit/", {
            "project_id": str(self.project.id),
            "scheduled_date": str(date.today() + timedelta(days=30)),
        })
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["action"], "schedule_audit")
        self.assertIn("audit", data)
        # StudyAction created
        actions = StudyAction.objects.filter(project=self.project, action_type="schedule_audit")
        self.assertEqual(actions.count(), 1)

    def test_default_scheduled_date(self):
        """If no date provided, defaults to 30 days from now."""
        resp = _post(self.client, "/api/iso/study-actions/schedule-audit/", {
            "project_id": str(self.project.id),
        })
        audit = resp.json()["audit"]
        expected = str(date.today() + timedelta(days=30))
        self.assertEqual(audit["scheduled_date"], expected)

    def test_scope_from_study(self):
        """Scope auto-populated from study title."""
        resp = _post(self.client, "/api/iso/study-actions/schedule-audit/", {
            "project_id": str(self.project.id),
        })
        audit = resp.json()["audit"]
        self.assertIn("Audit Study", audit["scope"])


# =============================================================================
# Study Actions — request doc update
# =============================================================================

@SECURE_OFF
class StudyRequestDocUpdateTest(TestCase):
    """study_request_doc_update creates a document and StudyAction."""

    def setUp(self):
        self.user = _make_team_user("studydoc@test.com")
        self.client.force_login(self.user)
        self.project = Project.objects.create(
            user=self.user,
            title="Doc Update Study",
            methodology="none",
        )

    def test_request_doc_update(self):
        resp = _post(self.client, "/api/iso/study-actions/request-doc-update/", {
            "project_id": str(self.project.id),
            "title": "Update SOP-042",
        })
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["action"], "request_doc_update")
        doc = data["document"]
        self.assertEqual(doc["status"], "draft")
        self.assertEqual(doc["source_study_id"], str(self.project.id))
        # StudyAction created
        actions = StudyAction.objects.filter(project=self.project, action_type="request_doc_update")
        self.assertEqual(actions.count(), 1)

    def test_auto_justification(self):
        """Content auto-filled from study context when not provided."""
        resp = _post(self.client, "/api/iso/study-actions/request-doc-update/", {
            "project_id": str(self.project.id),
        })
        doc = resp.json()["document"]
        self.assertIn("Doc Update Study", doc["content"])


# =============================================================================
# Study Actions — flag training gap
# =============================================================================

@SECURE_OFF
class StudyFlagTrainingGapTest(TestCase):
    """study_flag_training_gap creates a training requirement and StudyAction."""

    def setUp(self):
        self.user = _make_team_user("studytrain@test.com")
        self.client.force_login(self.user)
        self.project = Project.objects.create(
            user=self.user,
            title="Training Gap Study",
            methodology="none",
        )

    def test_flag_training_gap(self):
        resp = _post(self.client, "/api/iso/study-actions/flag-training-gap/", {
            "project_id": str(self.project.id),
            "name": "GD&T for Inspectors",
            "description": "Gap identified in dimensional inspection competency",
            "iso_clause": "7.2",
            "frequency_months": 12,
        })
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["action"], "flag_training_gap")
        req = data["training"]
        self.assertEqual(req["name"], "GD&T for Inspectors")
        self.assertTrue(req["is_mandatory"])
        # StudyAction created
        actions = StudyAction.objects.filter(project=self.project, action_type="flag_training_gap")
        self.assertEqual(actions.count(), 1)

    def test_name_required(self):
        resp = _post(self.client, "/api/iso/study-actions/flag-training-gap/", {
            "project_id": str(self.project.id),
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("name", resp.json()["error"])


# =============================================================================
# Study Actions — flag FMEA update
# =============================================================================

@SECURE_OFF
class StudyFlagFMEAUpdateTest(TestCase):
    """study_flag_fmea_update marks FMEA for review and creates StudyAction."""

    def setUp(self):
        self.user = _make_team_user("studyfmea@test.com")
        self.client.force_login(self.user)
        self.project = Project.objects.create(
            user=self.user,
            title="FMEA Review Study",
            methodology="none",
        )
        self.fmea = FMEA.objects.create(
            owner=self.user,
            project=self.project,
            title="Process FMEA — Assembly",
            status="active",
        )

    def test_flag_fmea_update(self):
        resp = _post(self.client, "/api/iso/study-actions/flag-fmea-update/", {
            "project_id": str(self.project.id),
            "fmea_id": str(self.fmea.id),
            "notes": "New failure mode discovered",
        })
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["action"], "flag_fmea_update")
        # FMEA status changed to review
        self.fmea.refresh_from_db()
        self.assertEqual(self.fmea.status, "review")
        # StudyAction created
        actions = StudyAction.objects.filter(project=self.project, action_type="flag_fmea_update")
        self.assertEqual(actions.count(), 1)
        self.assertEqual(actions.first().notes, "New failure mode discovered")

    def test_fmea_id_required(self):
        resp = _post(self.client, "/api/iso/study-actions/flag-fmea-update/", {
            "project_id": str(self.project.id),
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("fmea_id", resp.json()["error"])

    def test_wrong_fmea_rejected(self):
        import uuid
        resp = _post(self.client, "/api/iso/study-actions/flag-fmea-update/", {
            "project_id": str(self.project.id),
            "fmea_id": str(uuid.uuid4()),
        })
        self.assertEqual(resp.status_code, 404)

    def test_already_review_idempotent(self):
        """Flagging an FMEA already in review is idempotent."""
        self.fmea.status = "review"
        self.fmea.save()
        resp = _post(self.client, "/api/iso/study-actions/flag-fmea-update/", {
            "project_id": str(self.project.id),
            "fmea_id": str(self.fmea.id),
        })
        self.assertEqual(resp.status_code, 201)
        self.fmea.refresh_from_db()
        self.assertEqual(self.fmea.status, "review")


# =============================================================================
# Dashboard KPIs
# =============================================================================

@SECURE_OFF
class DashboardKPITest(TestCase):
    """ISO dashboard returns comprehensive KPIs."""

    def setUp(self):
        self.user = _make_team_user("dashboard@test.com")
        self.client.force_login(self.user)

    def test_empty_dashboard(self):
        resp = self.client.get("/api/iso/dashboard/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("clause_coverage", data)
        self.assertIn("ncrs", data)
        self.assertIn("upcoming_audits", data)
        self.assertIn("training", data)
        self.assertIn("documents", data)
        self.assertIn("suppliers", data)
        # 7 ISO clauses always present
        self.assertEqual(len(data["clause_coverage"]), 7)

    def test_dashboard_ncr_kpis(self):
        _post(self.client, "/api/iso/ncrs/", {"title": "N1", "severity": "minor"})
        _post(self.client, "/api/iso/ncrs/", {"title": "N2", "severity": "critical"})
        resp = self.client.get("/api/iso/dashboard/")
        ncrs = resp.json()["ncrs"]
        self.assertEqual(ncrs["open"], 2)
        self.assertEqual(ncrs["by_severity"]["minor"], 1)
        self.assertEqual(ncrs["by_severity"]["critical"], 1)

    def test_dashboard_overdue_capas(self):
        _post(self.client, "/api/iso/ncrs/", {
            "title": "Overdue",
            "capa_due_date": str(date.today() - timedelta(days=5)),
        })
        resp = self.client.get("/api/iso/dashboard/")
        self.assertEqual(resp.json()["ncrs"]["overdue_capas"], 1)

    def test_dashboard_capa_due_soon(self):
        _post(self.client, "/api/iso/ncrs/", {
            "title": "Due Soon",
            "capa_due_date": str(date.today() + timedelta(days=7)),
        })
        resp = self.client.get("/api/iso/dashboard/")
        self.assertEqual(len(resp.json()["capa_due_soon"]), 1)
        self.assertEqual(resp.json()["capa_due_soon"][0]["title"], "Due Soon")

    def test_dashboard_training_compliance(self):
        req = _post(self.client, "/api/iso/training/", {"name": "T"}).json()
        _post(self.client, f"/api/iso/training/{req['id']}/records/", {
            "employee_name": "A", "status": "complete",
        })
        _post(self.client, f"/api/iso/training/{req['id']}/records/", {
            "employee_name": "B", "status": "not_started",
        })
        resp = self.client.get("/api/iso/dashboard/")
        self.assertEqual(resp.json()["training"]["compliance_rate"], 50)
        self.assertEqual(resp.json()["training"]["gaps_count"], 1)

    def test_dashboard_clause_coverage(self):
        """NCR with iso_clause activates that clause in coverage."""
        _post(self.client, "/api/iso/ncrs/", {"title": "N", "iso_clause": "8.7"})
        resp = self.client.get("/api/iso/dashboard/")
        clause_8 = [c for c in resp.json()["clause_coverage"] if c["clause"] == "8"][0]
        self.assertEqual(clause_8["status"], "active")

    def test_dashboard_document_kpis(self):
        _post(self.client, "/api/iso/documents/", {"title": "D1"})
        _post(self.client, "/api/iso/documents/", {"title": "D2"})
        resp = self.client.get("/api/iso/dashboard/")
        self.assertEqual(resp.json()["documents"]["total"], 2)

    def test_dashboard_supplier_kpis(self):
        _post(self.client, "/api/iso/suppliers/", {"name": "S1"})
        resp = self.client.get("/api/iso/dashboard/")
        self.assertEqual(resp.json()["suppliers"]["total"], 1)

    def test_dashboard_last_review(self):
        rev = _post(self.client, "/api/iso/reviews/", {}).json()
        _put(self.client, f"/api/iso/reviews/{rev['id']}/", {"status": "complete"})
        resp = self.client.get("/api/iso/dashboard/")
        self.assertIsNotNone(resp.json()["last_review"])
        self.assertIn("days_ago", resp.json()["last_review"])


# =============================================================================
# User Isolation
# =============================================================================

@SECURE_OFF
class UserIsolationTest(TestCase):
    """Owner-based filtering: users cannot see each other's QMS data."""

    def setUp(self):
        self.user_a = _make_team_user("iso_a@test.com")
        self.user_b = _make_team_user("iso_b@test.com")

    def test_ncr_isolation(self):
        self.client.force_login(self.user_a)
        _post(self.client, "/api/iso/ncrs/", {"title": "A's NCR"})
        self.client.force_login(self.user_b)
        resp = self.client.get("/api/iso/ncrs/")
        self.assertEqual(len(resp.json()), 0)

    def test_ncr_detail_isolation(self):
        self.client.force_login(self.user_a)
        ncr = _post(self.client, "/api/iso/ncrs/", {"title": "A's NCR"}).json()
        self.client.force_login(self.user_b)
        resp = self.client.get(f"/api/iso/ncrs/{ncr['id']}/")
        self.assertEqual(resp.status_code, 404)

    def test_audit_isolation(self):
        self.client.force_login(self.user_a)
        _post(self.client, "/api/iso/audits/", {"title": "A's Audit"})
        self.client.force_login(self.user_b)
        resp = self.client.get("/api/iso/audits/")
        self.assertEqual(len(resp.json()), 0)

    def test_training_isolation(self):
        self.client.force_login(self.user_a)
        _post(self.client, "/api/iso/training/", {"name": "A's Training"})
        self.client.force_login(self.user_b)
        resp = self.client.get("/api/iso/training/")
        self.assertEqual(len(resp.json()), 0)

    def test_document_isolation(self):
        self.client.force_login(self.user_a)
        _post(self.client, "/api/iso/documents/", {"title": "A's Doc"})
        self.client.force_login(self.user_b)
        resp = self.client.get("/api/iso/documents/")
        self.assertEqual(len(resp.json()), 0)

    def test_supplier_isolation(self):
        self.client.force_login(self.user_a)
        _post(self.client, "/api/iso/suppliers/", {"name": "A's Supplier"})
        self.client.force_login(self.user_b)
        resp = self.client.get("/api/iso/suppliers/")
        self.assertEqual(len(resp.json()), 0)

    def test_review_isolation(self):
        self.client.force_login(self.user_a)
        _post(self.client, "/api/iso/reviews/", {})
        self.client.force_login(self.user_b)
        resp = self.client.get("/api/iso/reviews/")
        self.assertEqual(len(resp.json()), 0)

    def test_checklist_isolation(self):
        self.client.force_login(self.user_a)
        _post(self.client, "/api/iso/checklists/", {"name": "A's CL"})
        self.client.force_login(self.user_b)
        resp = self.client.get("/api/iso/checklists/")
        self.assertEqual(len(resp.json()), 0)

    def test_dashboard_isolation(self):
        """User B's dashboard doesn't include A's NCRs."""
        self.client.force_login(self.user_a)
        _post(self.client, "/api/iso/ncrs/", {"title": "A's NCR"})
        self.client.force_login(self.user_b)
        resp = self.client.get("/api/iso/dashboard/")
        self.assertEqual(resp.json()["ncrs"]["open"], 0)


# =============================================================================
# End-to-End: NCR → RCA → Evidence → CAPA → Audit (ISO loop closure)
# =============================================================================

@SECURE_OFF
class ISOLoopClosureTest(TestCase):
    """Full ISO corrective action loop: NCR → investigation → CAPA → verification audit."""

    def setUp(self):
        self.user = _make_team_user("isoloop@test.com")
        self.client.force_login(self.user)

    def test_full_iso_loop(self):
        # 1. Raise NCR from audit finding
        audit = _post(self.client, "/api/iso/audits/", {
            "title": "Process Audit — Assembly",
        }).json()
        finding = _post(self.client, f"/api/iso/audits/{audit['id']}/findings/", {
            "finding_type": "nc_major",
            "description": "Torque not verified on safety-critical fasteners",
            "iso_clause": "8.5.1",
        }).json()
        ncr_id = finding["ncr_id"]
        self.assertIsNotNone(ncr_id)

        # 2. Launch RCA from NCR
        rca_resp = _post(self.client, f"/api/iso/ncrs/{ncr_id}/launch-rca/")
        self.assertEqual(rca_resp.status_code, 201)

        # 3. Update NCR through workflow
        _put(self.client, f"/api/iso/ncrs/{ncr_id}/", {
            "status": "investigation",
            "assigned_to": self.user.id,
        })
        _put(self.client, f"/api/iso/ncrs/{ncr_id}/", {
            "root_cause": "No torque verification step in work instruction",
        })
        _put(self.client, f"/api/iso/ncrs/{ncr_id}/", {
            "status": "capa",
            "corrective_action": "Added mandatory torque verification step to SOP-042",
        })

        # 4. Raise CAPA from the auto-Study
        ncr = self.client.get(f"/api/iso/ncrs/{ncr_id}/").json()
        project_id = ncr["project_id"]
        capa_resp = _post(self.client, "/api/iso/study-actions/raise-capa/", {
            "project_id": project_id,
        })
        self.assertEqual(capa_resp.status_code, 201)

        # 5. Schedule verification audit
        audit_resp = _post(self.client, "/api/iso/study-actions/schedule-audit/", {
            "project_id": project_id,
        })
        self.assertEqual(audit_resp.status_code, 201)

        # 6. Request document update
        doc_resp = _post(self.client, "/api/iso/study-actions/request-doc-update/", {
            "project_id": project_id,
        })
        self.assertEqual(doc_resp.status_code, 201)

        # 7. Verify traceability — all StudyActions linked to same project
        actions = StudyAction.objects.filter(project_id=project_id)
        action_types = set(actions.values_list("action_type", flat=True))
        self.assertIn("raise_capa", action_types)
        self.assertIn("schedule_audit", action_types)
        self.assertIn("request_doc_update", action_types)

        # 8. Verify evidence chain — root_cause, corrective_action evidence created
        evidence = Evidence.objects.filter(project_id=project_id)
        self.assertTrue(evidence.count() >= 2)  # At minimum root_cause + corrective_action

        # 9. Close NCR
        _put(self.client, f"/api/iso/ncrs/{ncr_id}/", {"status": "verification"})
        _put(self.client, f"/api/iso/ncrs/{ncr_id}/", {
            "status": "closed",
            "approved_by": self.user.id,
            "verification_result": "20 consecutive builds verified",
        })
        ncr_final = self.client.get(f"/api/iso/ncrs/{ncr_id}/").json()
        self.assertEqual(ncr_final["status"], "closed")

        # 10. Dashboard should reflect the resolution
        dash = self.client.get("/api/iso/dashboard/").json()
        self.assertEqual(dash["ncrs"]["open"], 0)
