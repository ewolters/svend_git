"""Electronic Signature tests — 21 CFR Part 11 compliance.

Covers:
- Re-authentication enforcement (password re-entry)
- CRUD: sign documents with 5 meanings
- Uniqueness constraint (same signer + doc + meaning)
- Immutability (cannot update/delete, hash chain integrity)
- List/query by document
- Verification endpoints
- All 7 signable document types
"""

import json
import uuid

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from accounts.constants import Tier
from agents_api.models import (
    FMEA,
    CAPAReport,
    ControlledDocument,
    ElectronicSignature,
    InternalAudit,
    ManagementReview,
    NonconformanceRecord,
    TrainingRecord,
    TrainingRequirement,
)

User = get_user_model()
SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)
PASSWORD = "testpass123!"


def _post(client, url, data=None):
    return client.post(url, json.dumps(data or {}), content_type="application/json")


def _make_team_user(email, **kwargs):
    username = kwargs.pop("username", email.split("@")[0])
    user = User.objects.create_user(
        username=username, email=email, password=PASSWORD, **kwargs
    )
    user.tier = Tier.TEAM
    user.save(update_fields=["tier"])
    return user


def _make_free_user(email, **kwargs):
    username = kwargs.pop("username", email.split("@")[0])
    user = User.objects.create_user(
        username=username, email=email, password=PASSWORD, **kwargs
    )
    user.tier = Tier.FREE
    user.save(update_fields=["tier"])
    return user


def _create_ncr(user):
    return NonconformanceRecord.objects.create(
        owner=user,
        title="Test NCR",
        description="Test desc",
        severity="minor",
    )


def _sign(client, doc_type, doc_id, meaning="approved", password=PASSWORD, reason=""):
    return _post(
        client,
        "/api/iso/signatures/",
        {
            "document_type": doc_type,
            "document_id": str(doc_id),
            "meaning": meaning,
            "password": password,
            "reason": reason,
        },
    )


def _err_msg(resp):
    body = resp.json()
    err = body.get("error", body.get("message", ""))
    if isinstance(err, dict):
        return err.get("message", str(err))
    return str(err)


# =============================================================================
# Re-authentication (21 CFR Part 11 §11.100)
# =============================================================================


@SECURE_OFF
class ESignatureAuthTest(TestCase):
    """Re-authentication enforcement."""

    def setUp(self):
        self.user = _make_team_user("esigauth@test.com")
        self.client.force_login(self.user)
        self.ncr = _create_ncr(self.user)

    def test_sign_requires_password(self):
        resp = _post(
            self.client,
            "/api/iso/signatures/",
            {
                "document_type": "ncr",
                "document_id": str(self.ncr.id),
                "meaning": "approved",
            },
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("password", _err_msg(resp))

    def test_sign_wrong_password_rejected(self):
        resp = _sign(self.client, "ncr", self.ncr.id, password="wrongpass")
        self.assertEqual(resp.status_code, 403)
        self.assertIn("Password", _err_msg(resp))

    def test_sign_correct_password_succeeds(self):
        resp = _sign(self.client, "ncr", self.ncr.id)
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data["meaning"], "approved")
        self.assertEqual(data["document_type"], "ncr")

    def test_unauthenticated_blocked(self):
        self.client.logout()
        resp = _sign(self.client, "ncr", self.ncr.id)
        self.assertIn(resp.status_code, [401, 403])

    def test_free_tier_blocked(self):
        free = _make_free_user("esigfree@test.com")
        self.client.force_login(free)
        ncr = _create_ncr(free)
        resp = _sign(self.client, "ncr", ncr.id)
        self.assertEqual(resp.status_code, 403)


# =============================================================================
# CRUD — Signing
# =============================================================================


@SECURE_OFF
class ESignatureCrudTest(TestCase):
    """Basic signing operations."""

    def setUp(self):
        self.user = _make_team_user("esigcrud@test.com")
        self.client.force_login(self.user)
        self.ncr = _create_ncr(self.user)

    def test_approve_ncr(self):
        resp = _sign(self.client, "ncr", self.ncr.id, "approved")
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data["meaning"], "approved")
        self.assertIn("entry_hash", data)
        self.assertIn("signer", data)
        self.assertEqual(data["signer"]["id"], self.user.id)

    def test_reject_requires_reason(self):
        resp = _sign(self.client, "ncr", self.ncr.id, "rejected")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("reason", _err_msg(resp).lower())

    def test_reject_with_reason_succeeds(self):
        resp = _sign(
            self.client, "ncr", self.ncr.id, "rejected", reason="Insufficient evidence"
        )
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(resp.json()["reason"], "Insufficient evidence")

    def test_all_meanings_accepted(self):
        for meaning in ["approved", "rejected", "reviewed", "authored", "witnessed"]:
            ncr = _create_ncr(self.user)
            kwargs = {"reason": "Test rejection"} if meaning == "rejected" else {}
            resp = _sign(self.client, "ncr", ncr.id, meaning, **kwargs)
            self.assertEqual(resp.status_code, 201, f"Failed for meaning: {meaning}")

    def test_invalid_meaning_rejected(self):
        resp = _sign(self.client, "ncr", self.ncr.id, "invalid_meaning")
        self.assertEqual(resp.status_code, 400)

    def test_invalid_document_type_rejected(self):
        resp = _sign(self.client, "unknown_type", self.ncr.id, "approved")
        self.assertEqual(resp.status_code, 400)

    def test_nonexistent_document_rejected(self):
        resp = _sign(self.client, "ncr", uuid.uuid4(), "approved")
        self.assertEqual(resp.status_code, 404)


# =============================================================================
# Uniqueness
# =============================================================================


@SECURE_OFF
class ESignatureUniquenessTest(TestCase):
    """Duplicate prevention."""

    def setUp(self):
        self.user = _make_team_user("esiguniq@test.com")
        self.client.force_login(self.user)
        self.ncr = _create_ncr(self.user)

    def test_cannot_sign_same_meaning_twice(self):
        resp1 = _sign(self.client, "ncr", self.ncr.id, "approved")
        self.assertEqual(resp1.status_code, 201)
        resp2 = _sign(self.client, "ncr", self.ncr.id, "approved")
        self.assertEqual(resp2.status_code, 409)

    def test_different_meanings_allowed(self):
        resp1 = _sign(self.client, "ncr", self.ncr.id, "approved")
        self.assertEqual(resp1.status_code, 201)
        resp2 = _sign(self.client, "ncr", self.ncr.id, "reviewed")
        self.assertEqual(resp2.status_code, 201)

    def test_different_documents_allowed(self):
        ncr2 = _create_ncr(self.user)
        resp1 = _sign(self.client, "ncr", self.ncr.id, "approved")
        self.assertEqual(resp1.status_code, 201)
        resp2 = _sign(self.client, "ncr", ncr2.id, "approved")
        self.assertEqual(resp2.status_code, 201)


# =============================================================================
# Immutability (CFR Part 11)
# =============================================================================


@SECURE_OFF
class ESignatureImmutabilityTest(TestCase):
    """Immutability and hash chain integrity."""

    def setUp(self):
        self.user = _make_team_user("esigimm@test.com")
        self.client.force_login(self.user)

    def test_signature_cannot_be_updated(self):
        ncr = _create_ncr(self.user)
        resp = _sign(self.client, "ncr", ncr.id, "approved")
        sig = ElectronicSignature.objects.get(id=resp.json()["id"])
        sig.reason = "Tampered"
        with self.assertRaises(ValueError):
            sig.save()

    def test_signature_cannot_be_deleted(self):
        ncr = _create_ncr(self.user)
        resp = _sign(self.client, "ncr", ncr.id, "approved")
        sig = ElectronicSignature.objects.get(id=resp.json()["id"])
        with self.assertRaises(PermissionError):
            sig.delete()

    def test_hash_chain_integrity(self):
        for i in range(3):
            ncr = _create_ncr(self.user)
            _sign(self.client, "ncr", ncr.id, "approved")
        result = ElectronicSignature.verify_chain(tenant_id=None)
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["entries_checked"], 3)

    def test_single_signature_integrity(self):
        ncr = _create_ncr(self.user)
        resp = _sign(self.client, "ncr", ncr.id, "approved")
        sig = ElectronicSignature.objects.get(id=resp.json()["id"])
        self.assertTrue(sig.verify_integrity())

    def test_document_snapshot_captured(self):
        ncr = _create_ncr(self.user)
        resp = _sign(self.client, "ncr", ncr.id, "approved")
        sig = ElectronicSignature.objects.get(id=resp.json()["id"])
        self.assertIsInstance(sig.after_snapshot, dict)
        self.assertEqual(sig.after_snapshot.get("title"), "Test NCR")


# =============================================================================
# List / Query
# =============================================================================


@SECURE_OFF
class ESignatureListTest(TestCase):
    """Query signatures."""

    def setUp(self):
        self.user = _make_team_user("esiglist@test.com")
        self.client.force_login(self.user)

    def test_list_by_document(self):
        ncr = _create_ncr(self.user)
        _sign(self.client, "ncr", ncr.id, "approved")
        _sign(self.client, "ncr", ncr.id, "reviewed")
        resp = self.client.get(
            f"/api/iso/signatures/?document_type=ncr&document_id={ncr.id}"
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.json()), 2)

    def test_list_empty(self):
        resp = self.client.get("/api/iso/signatures/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.json()), 0)

    def test_user_isolation(self):
        user_a = self.user
        user_b = _make_team_user("esiglist_b@test.com")
        ncr_a = _create_ncr(user_a)
        _sign(self.client, "ncr", ncr_a.id, "approved")

        self.client.force_login(user_b)
        resp = self.client.get("/api/iso/signatures/")
        self.assertEqual(len(resp.json()), 0)


# =============================================================================
# Verify Endpoints
# =============================================================================


@SECURE_OFF
class ESignatureVerifyTest(TestCase):
    """Verification endpoints."""

    def setUp(self):
        self.user = _make_team_user("esigverify@test.com")
        self.client.force_login(self.user)

    def test_verify_valid_signature(self):
        ncr = _create_ncr(self.user)
        sig_resp = _sign(self.client, "ncr", ncr.id, "approved")
        sig_id = sig_resp.json()["id"]
        resp = self.client.get(f"/api/iso/signatures/{sig_id}/verify/")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["is_valid"])

    def test_verify_chain(self):
        ncr = _create_ncr(self.user)
        _sign(self.client, "ncr", ncr.id, "approved")
        resp = self.client.get("/api/iso/signatures/verify-chain/")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["is_valid"])


# =============================================================================
# All 7 Signable Document Types
# =============================================================================


@SECURE_OFF
class ESignatureDocumentTypesTest(TestCase):
    """Verify signing works for all 7 QMS document types."""

    def setUp(self):
        self.user = _make_team_user("esigdocs@test.com")
        self.client.force_login(self.user)

    def test_sign_ncr(self):
        ncr = _create_ncr(self.user)
        resp = _sign(self.client, "ncr", ncr.id, "approved")
        self.assertEqual(resp.status_code, 201)

    def test_sign_capa(self):
        capa = CAPAReport.objects.create(
            owner=self.user,
            title="Test CAPA",
            priority="medium",
        )
        resp = _sign(self.client, "capa", capa.id, "approved")
        self.assertEqual(resp.status_code, 201)

    def test_sign_document(self):
        doc = ControlledDocument.objects.create(
            owner=self.user,
            title="SOP-001",
            content="Standard operating procedure",
        )
        resp = _sign(self.client, "document", doc.id, "approved")
        self.assertEqual(resp.status_code, 201)

    def test_sign_review(self):
        review = ManagementReview.objects.create(
            owner=self.user,
            title="Q1 Review",
            meeting_date="2026-03-01",
        )
        resp = _sign(self.client, "review", review.id, "witnessed")
        self.assertEqual(resp.status_code, 201)

    def test_sign_audit(self):
        audit = InternalAudit.objects.create(
            owner=self.user,
            title="Audit-001",
            scheduled_date="2026-03-15",
        )
        resp = _sign(self.client, "audit", audit.id, "approved")
        self.assertEqual(resp.status_code, 201)

    def test_sign_training(self):
        req = TrainingRequirement.objects.create(
            owner=self.user,
            name="GMP Basics",
        )
        record = TrainingRecord.objects.create(
            requirement=req,
            employee_name="John Doe",
            employee_email="john@test.com",
        )
        resp = _sign(self.client, "training", record.id, "approved")
        self.assertEqual(resp.status_code, 201)

    def test_sign_fmea(self):
        fmea = FMEA.objects.create(
            owner=self.user,
            title="Process FMEA-001",
        )
        resp = _sign(self.client, "fmea", fmea.id, "reviewed")
        self.assertEqual(resp.status_code, 201)
