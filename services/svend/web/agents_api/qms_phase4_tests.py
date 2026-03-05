"""Phase 4A tests — Training Competency (Harvey Balls + Certification Status).

Standard: TRN-001 v1.0
CR: Phase 4A — Training Matrix Harvey Ball Competency + Certification Status
"""

import json
from datetime import timedelta

from django.test import TestCase, override_settings
from django.utils import timezone

from accounts.models import User
from agents_api.models import (
    ControlledDocument,
    TrainingRecord,
    TrainingRecordChange,
    TrainingRequirement,
)

SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)


def _make_team_user(username, email):
    """Create a user with team tier for @require_team gating."""
    user = User.objects.create_user(username=username, email=email, password="test1234")
    user.tier = "team"
    user.save(update_fields=["tier"])
    return user


class CompetencyLevelModelTest(TestCase):
    """TRN-001 §2.1 — TWI 4-level competency field on TrainingRecord."""

    @classmethod
    def setUpTestData(cls):
        cls.user = User.objects.create_user(
            username="trn_model",
            email="trn_model@test.com",
            password="test1234",
        )
        cls.req = TrainingRequirement.objects.create(
            owner=cls.user,
            name="Forklift Operation",
        )

    def test_competency_level_field_exists(self):
        """TrainingRecord has competency_level IntegerField."""
        record = TrainingRecord.objects.create(
            requirement=self.req,
            employee_name="Alice",
        )
        self.assertTrue(hasattr(record, "competency_level"))

    def test_competency_level_default_zero(self):
        """Default competency_level is 0 (No Exposure)."""
        record = TrainingRecord.objects.create(
            requirement=self.req,
            employee_name="Bob",
        )
        self.assertEqual(record.competency_level, 0)

    def test_competency_level_accepts_valid_range(self):
        """competency_level accepts values 0–4."""
        for level in range(5):
            record = TrainingRecord.objects.create(
                requirement=self.req,
                employee_name=f"Worker_{level}",
                competency_level=level,
            )
            record.refresh_from_db()
            self.assertEqual(record.competency_level, level)


class CertificationStatusPropertyTest(TestCase):
    """TRN-001 §3.1 — Computed certification_status property."""

    @classmethod
    def setUpTestData(cls):
        cls.user = User.objects.create_user(
            username="trn_cert",
            email="trn_cert@test.com",
            password="test1234",
        )
        cls.req = TrainingRequirement.objects.create(
            owner=cls.user,
            name="GMP Training",
            frequency_months=12,
        )

    def test_current_status(self):
        """Complete with expires_at > 30 days out → 'current'."""
        record = TrainingRecord.objects.create(
            requirement=self.req,
            employee_name="Alice",
            status="complete",
            completed_at=timezone.now(),
            expires_at=timezone.now() + timedelta(days=365),
        )
        self.assertEqual(record.certification_status, "current")

    def test_expiring_status(self):
        """Complete with expires_at within 30 days → 'expiring'."""
        record = TrainingRecord.objects.create(
            requirement=self.req,
            employee_name="Bob",
            status="complete",
            completed_at=timezone.now() - timedelta(days=335),
            expires_at=timezone.now() + timedelta(days=15),
        )
        self.assertEqual(record.certification_status, "expiring")

    def test_expired_status(self):
        """Status 'expired' → 'expired'. Also complete with past expires_at → 'expired'."""
        record1 = TrainingRecord.objects.create(
            requirement=self.req,
            employee_name="Carol",
            status="expired",
        )
        self.assertEqual(record1.certification_status, "expired")

        record2 = TrainingRecord.objects.create(
            requirement=self.req,
            employee_name="Dave",
            status="complete",
            completed_at=timezone.now() - timedelta(days=400),
            expires_at=timezone.now() - timedelta(days=35),
        )
        self.assertEqual(record2.certification_status, "expired")

    def test_incomplete_status(self):
        """not_started and in_progress → 'incomplete'."""
        for status in ("not_started", "in_progress"):
            record = TrainingRecord.objects.create(
                requirement=self.req,
                employee_name=f"Worker_{status}",
                status=status,
            )
            self.assertEqual(record.certification_status, "incomplete")

    def test_current_no_expiry(self):
        """Complete with no expires_at (one-time training) → 'current'."""
        record = TrainingRecord.objects.create(
            requirement=self.req,
            employee_name="Eve",
            status="complete",
            completed_at=timezone.now(),
        )
        self.assertEqual(record.certification_status, "current")


class TrainingRecordSerializationTest(TestCase):
    """TRN-001 §5.1 — to_dict() includes competency + certification."""

    @classmethod
    def setUpTestData(cls):
        cls.user = User.objects.create_user(
            username="trn_serial",
            email="trn_serial@test.com",
            password="test1234",
        )
        cls.req = TrainingRequirement.objects.create(
            owner=cls.user,
            name="SPC Basics",
        )

    def test_to_dict_includes_competency_level(self):
        """to_dict() output contains competency_level."""
        record = TrainingRecord.objects.create(
            requirement=self.req,
            employee_name="Alice",
            competency_level=3,
        )
        d = record.to_dict()
        self.assertIn("competency_level", d)
        self.assertEqual(d["competency_level"], 3)

    def test_to_dict_includes_certification_status(self):
        """to_dict() output contains certification_status."""
        record = TrainingRecord.objects.create(
            requirement=self.req,
            employee_name="Bob",
            status="complete",
            completed_at=timezone.now(),
        )
        d = record.to_dict()
        self.assertIn("certification_status", d)
        self.assertEqual(d["certification_status"], "current")


@SECURE_OFF
class CompetencyLevelAPITest(TestCase):
    """TRN-001 §2 + §6 — API accepts competency_level in create/update with audit logging."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _make_team_user("trn_api", "trn_api@test.com")
        cls.req = TrainingRequirement.objects.create(
            owner=cls.user,
            name="FMEA Training",
        )

    def test_create_record_with_competency_level(self):
        """POST /training/<req_id>/records/ accepts competency_level."""
        self.client.force_login(self.user)
        resp = self.client.post(
            f"/api/iso/training/{self.req.id}/records/",
            data=json.dumps(
                {
                    "employee_name": "Alice",
                    "competency_level": 2,
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data["competency_level"], 2)

    def test_update_competency_level(self):
        """PUT /training/records/<id>/ updates competency_level."""
        record = TrainingRecord.objects.create(
            requirement=self.req,
            employee_name="Bob",
            competency_level=1,
        )
        self.client.force_login(self.user)
        resp = self.client.put(
            f"/api/iso/training/records/{record.id}/",
            data=json.dumps({"competency_level": 3}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["competency_level"], 3)

    def test_update_competency_level_logged(self):
        """Competency level change creates TrainingRecordChange entry."""
        record = TrainingRecord.objects.create(
            requirement=self.req,
            employee_name="Carol",
            competency_level=0,
        )
        self.client.force_login(self.user)
        self.client.put(
            f"/api/iso/training/records/{record.id}/",
            data=json.dumps({"competency_level": 4}),
            content_type="application/json",
        )
        change = TrainingRecordChange.objects.filter(
            record=record,
            field_name="competency_level",
        ).first()
        self.assertIsNotNone(change)
        # log_change uses str(old_val or ""), and 0 is falsy → stored as ""
        self.assertIn(change.old_value, ("0", ""))
        self.assertEqual(change.new_value, "4")

    def test_competency_level_clamped(self):
        """Values outside 0-4 are clamped."""
        self.client.force_login(self.user)
        resp = self.client.post(
            f"/api/iso/training/{self.req.id}/records/",
            data=json.dumps(
                {
                    "employee_name": "Dave",
                    "competency_level": 99,
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(resp.json()["competency_level"], 4)


@SECURE_OFF
class TrainingGridRenderingTest(TestCase):
    """TRN-001 §4 — Grid data includes competency + certification for rendering."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _make_team_user("trn_grid", "trn_grid@test.com")
        cls.req = TrainingRequirement.objects.create(
            owner=cls.user,
            name="VSM Training",
        )
        TrainingRecord.objects.create(
            requirement=cls.req,
            employee_name="Alice",
            status="complete",
            completed_at=timezone.now(),
            competency_level=3,
        )
        TrainingRecord.objects.create(
            requirement=cls.req,
            employee_name="Bob",
            status="not_started",
            competency_level=0,
        )

    def test_grid_returns_competency_and_cert_data(self):
        """GET /training/ returns records with competency_level and certification_status."""
        self.client.force_login(self.user)
        resp = self.client.get("/api/iso/training/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        records = data[0]["records"]
        for record in records:
            self.assertIn("competency_level", record)
            self.assertIn("certification_status", record)

    def test_grid_returns_harvey_ball_mapping(self):
        """Records map competency levels to correct values."""
        self.client.force_login(self.user)
        resp = self.client.get("/api/iso/training/")
        data = resp.json()
        records = {r["employee_name"]: r for r in data[0]["records"]}
        self.assertEqual(records["Alice"]["competency_level"], 3)
        self.assertEqual(records["Alice"]["certification_status"], "current")
        self.assertEqual(records["Bob"]["competency_level"], 0)
        self.assertEqual(records["Bob"]["certification_status"], "incomplete")

    def test_certification_status_computed_not_stored(self):
        """certification_status is a property, not a DB column."""
        field_names = [f.name for f in TrainingRecord._meta.get_fields()]
        self.assertNotIn("certification_status", field_names)


# =========================================================================
# Phase 4D — Document Linkage + Artifact Uploads (TRN-001 §7)
# =========================================================================


@SECURE_OFF
class DocumentLinkTest(TestCase):
    """TRN-001 §7.1 — Nullable FK from TrainingRequirement to ControlledDocument."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _make_team_user("doc_link", "doc_link@test.com")
        cls.doc = ControlledDocument.objects.create(
            owner=cls.user,
            title="SOP-101 Assembly",
            document_number="SOP-101",
            current_version="2.0",
            status="approved",
        )

    def test_create_requirement_with_document(self):
        """POST /training/ with document_id links requirement to document."""
        self.client.force_login(self.user)
        resp = self.client.post(
            "/api/iso/training/",
            data=json.dumps(
                {
                    "name": "Assembly Training",
                    "document_id": str(self.doc.id),
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data["document_id"], str(self.doc.id))
        self.assertEqual(data["document_version"], "2.0")

    def test_update_requirement_clear_document(self):
        """PUT /training/<id>/ with document_id=null clears the link."""
        req = TrainingRequirement.objects.create(
            owner=self.user,
            name="Linked Req",
            document=self.doc,
            document_version="2.0",
        )
        self.client.force_login(self.user)
        resp = self.client.put(
            f"/api/iso/training/{req.id}/",
            data=json.dumps({"document_id": None}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIsNone(data["document_id"])

    def test_to_dict_includes_document_info(self):
        """TrainingRequirement.to_dict() includes document fields."""
        req = TrainingRequirement.objects.create(
            owner=self.user,
            name="Doc-linked",
            document=self.doc,
            document_version="2.0",
        )
        d = req.to_dict()
        self.assertEqual(d["document_id"], str(self.doc.id))
        self.assertIn("SOP-101", d["document_title"])
        self.assertEqual(d["document_version"], "2.0")


@SECURE_OFF
class ArtifactUploadTest(TestCase):
    """TRN-001 §7.2 — M2M artifact uploads on TrainingRecord."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _make_team_user("art_up", "art_up@test.com")
        cls.req = TrainingRequirement.objects.create(
            owner=cls.user,
            name="Welding Cert",
        )
        cls.record = TrainingRecord.objects.create(
            requirement=cls.req,
            employee_name="Welder A",
        )
        from files.models import UserFile

        cls.file = UserFile.objects.create(
            user=cls.user,
            original_name="cert.pdf",
            file_type="pdf",
            mime_type="application/pdf",
        )

    def test_attach_artifact(self):
        """POST /training/records/<id>/files/ attaches a file."""
        self.client.force_login(self.user)
        resp = self.client.post(
            f"/api/iso/training/records/{self.record.id}/files/",
            data=json.dumps({"file_id": str(self.file.id)}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn(str(self.file.id), data["artifact_ids"])

    def test_detach_artifact(self):
        """DELETE /training/records/<id>/files/ removes a file."""
        self.record.artifacts.add(self.file)
        self.client.force_login(self.user)
        resp = self.client.delete(
            f"/api/iso/training/records/{self.record.id}/files/",
            data=json.dumps({"file_id": str(self.file.id)}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["artifact_ids"], [])

    def test_to_dict_includes_artifact_ids(self):
        """TrainingRecord.to_dict() includes artifact_ids."""
        self.record.artifacts.add(self.file)
        d = self.record.to_dict()
        self.assertIn(str(self.file.id), d["artifact_ids"])


@SECURE_OFF
class RevisionRetrainingTest(TestCase):
    """TRN-001 §7.3 — Document revision flags linked training records for retraining."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _make_team_user("retrain", "retrain@test.com")
        cls.doc = ControlledDocument.objects.create(
            owner=cls.user,
            title="WI-200 Inspection",
            document_number="WI-200",
            current_version="1.0",
            status="approved",
            content="Step 1: Inspect surface finish",
        )
        cls.req_linked = TrainingRequirement.objects.create(
            owner=cls.user,
            name="Inspection Training",
            document=cls.doc,
            document_version="1.0",
        )
        cls.req_unlinked = TrainingRequirement.objects.create(
            owner=cls.user,
            name="Safety Orientation",
        )

    def test_revision_flags_linked_records(self):
        """Transitioning document approved→review flags linked complete records as expired."""
        record = TrainingRecord.objects.create(
            requirement=self.req_linked,
            employee_name="Inspector",
            status="complete",
            completed_at=timezone.now(),
        )
        unlinked_record = TrainingRecord.objects.create(
            requirement=self.req_unlinked,
            employee_name="Inspector",
            status="complete",
            completed_at=timezone.now(),
        )
        self.client.force_login(self.user)
        resp = self.client.put(
            f"/api/iso/documents/{self.doc.id}/",
            data=json.dumps({"status": "review"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        record.refresh_from_db()
        unlinked_record.refresh_from_db()
        self.assertEqual(record.status, "expired")
        self.assertEqual(unlinked_record.status, "complete")  # Unaffected

    def test_revision_creates_audit_trail(self):
        """Retraining flag creates TrainingRecordChange entries."""
        record = TrainingRecord.objects.create(
            requirement=self.req_linked,
            employee_name="Audited",
            status="complete",
            completed_at=timezone.now(),
        )
        self.client.force_login(self.user)
        self.client.put(
            f"/api/iso/documents/{self.doc.id}/",
            data=json.dumps({"status": "review"}),
            content_type="application/json",
        )
        change = TrainingRecordChange.objects.filter(
            record=record,
            field_name="status",
        ).first()
        self.assertIsNotNone(change)
        self.assertEqual(change.old_value, "complete")
        self.assertEqual(change.new_value, "expired")
