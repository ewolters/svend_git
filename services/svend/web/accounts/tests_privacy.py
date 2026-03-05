"""Tests for self-service data export (PRIV-001, SOC 2 P1.8)."""

import json
import os
import tempfile
from datetime import timedelta

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.utils import timezone

from accounts.models import DataExportRequest

User = get_user_model()
SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)


def _post(client, url, data=None):
    return client.post(url, json.dumps(data or {}), content_type="application/json")


def _get(client, url):
    return client.get(url)


def _delete(client, url):
    return client.delete(url)


def _cleanup_export(export_req):
    """Remove generated export file if it exists."""
    export_req.refresh_from_db()
    if export_req.file_path and os.path.exists(export_req.file_path):
        os.remove(export_req.file_path)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


@SECURE_OFF
class DataExportRequestModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="exportuser", email="export@test.com", password="testpass123!")

    def test_create_request(self):
        req = DataExportRequest.objects.create(user=self.user)
        self.assertEqual(req.status, DataExportRequest.Status.PENDING)
        self.assertEqual(req.export_format, DataExportRequest.ExportFormat.JSON)
        self.assertIsNotNone(req.id)
        self.assertIsNotNone(req.created_at)

    def test_status_transitions(self):
        req = DataExportRequest.objects.create(user=self.user)
        req.status = DataExportRequest.Status.PROCESSING
        req.processing_started_at = timezone.now()
        req.save()
        req.refresh_from_db()
        self.assertEqual(req.status, "processing")

        req.status = DataExportRequest.Status.COMPLETED
        req.completed_at = timezone.now()
        req.expires_at = timezone.now() + timedelta(days=7)
        req.save()
        req.refresh_from_db()
        self.assertEqual(req.status, "completed")

    def test_expires_at_set_on_completion(self):
        req = DataExportRequest.objects.create(user=self.user)
        now = timezone.now()
        req.status = DataExportRequest.Status.COMPLETED
        req.completed_at = now
        req.expires_at = now + timedelta(days=7)
        req.save()
        req.refresh_from_db()
        delta = req.expires_at - req.completed_at
        self.assertAlmostEqual(delta.days, 7, delta=1)

    def test_user_cascade_delete(self):
        DataExportRequest.objects.create(user=self.user)
        self.assertEqual(DataExportRequest.objects.count(), 1)
        self.user.delete()
        self.assertEqual(DataExportRequest.objects.count(), 0)


# ---------------------------------------------------------------------------
# View tests (export runs synchronously inline)
# ---------------------------------------------------------------------------


@SECURE_OFF
class DataExportViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="viewuser", email="view@test.com", password="testpass123!")
        self.other = User.objects.create_user(username="otheruser", email="other@test.com", password="testpass123!")
        self.client.login(username="viewuser", password="testpass123!")

    def test_create_export(self):
        res = _post(self.client, "/api/privacy/exports/")
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertIn(data["status"], ("pending", "completed"))
        self.assertIn("id", data)
        # Cleanup generated file
        export_req = DataExportRequest.objects.get(id=data["id"])
        _cleanup_export(export_req)

    def test_create_export_unauthenticated(self):
        self.client.logout()
        res = _post(self.client, "/api/privacy/exports/")
        self.assertIn(res.status_code, (401, 403))

    def test_rate_limit_24h(self):
        res1 = _post(self.client, "/api/privacy/exports/")
        self.assertEqual(res1.status_code, 201)
        export_req = DataExportRequest.objects.get(id=res1.json()["id"])
        res2 = _post(self.client, "/api/privacy/exports/")
        self.assertEqual(res2.status_code, 429)
        _cleanup_export(export_req)

    def test_rate_limit_cancelled_doesnt_count(self):
        res = _post(self.client, "/api/privacy/exports/")
        export_id = res.json()["id"]
        export_req = DataExportRequest.objects.get(id=export_id)
        _cleanup_export(export_req)
        # Cancel it
        _delete(self.client, f"/api/privacy/exports/{export_id}/")
        # Should be able to create another
        res2 = _post(self.client, "/api/privacy/exports/")
        self.assertEqual(res2.status_code, 201)
        export_req2 = DataExportRequest.objects.get(id=res2.json()["id"])
        _cleanup_export(export_req2)

    def test_list_exports(self):
        res = _post(self.client, "/api/privacy/exports/")
        export_req = DataExportRequest.objects.get(id=res.json()["id"])
        res = _get(self.client, "/api/privacy/exports/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()), 1)
        _cleanup_export(export_req)

    def test_list_excludes_other_users(self):
        DataExportRequest.objects.create(user=self.other)
        res = _get(self.client, "/api/privacy/exports/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()), 0)

    def test_export_detail_own(self):
        res = _post(self.client, "/api/privacy/exports/")
        export_id = res.json()["id"]
        res = _get(self.client, f"/api/privacy/exports/{export_id}/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["id"], export_id)
        export_req = DataExportRequest.objects.get(id=export_id)
        _cleanup_export(export_req)

    def test_export_detail_other_user(self):
        export_req = DataExportRequest.objects.create(user=self.other)
        res = _get(self.client, f"/api/privacy/exports/{export_req.id}/")
        self.assertEqual(res.status_code, 404)

    def test_download_pending_export(self):
        export_req = DataExportRequest.objects.create(user=self.user)
        res = _get(self.client, f"/api/privacy/exports/{export_req.id}/?download=true")
        self.assertEqual(res.status_code, 400)

    def test_download_expired_export(self):
        export_req = DataExportRequest.objects.create(user=self.user, status=DataExportRequest.Status.EXPIRED)
        res = _get(self.client, f"/api/privacy/exports/{export_req.id}/?download=true")
        self.assertEqual(res.status_code, 410)

    def test_download_completed_export(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"test": True}, f)
            temp_path = f.name

        try:
            export_req = DataExportRequest.objects.create(
                user=self.user,
                status=DataExportRequest.Status.COMPLETED,
                file_path=temp_path,
                completed_at=timezone.now(),
                expires_at=timezone.now() + timedelta(days=7),
            )
            res = _get(self.client, f"/api/privacy/exports/{export_req.id}/?download=true")
            self.assertEqual(res.status_code, 200)
            self.assertIn("attachment", res.get("Content-Disposition", ""))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


# ---------------------------------------------------------------------------
# Cancel tests
# ---------------------------------------------------------------------------


@SECURE_OFF
class CancelExportTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="canceluser", email="cancel@test.com", password="testpass123!")
        self.other = User.objects.create_user(
            username="cancelother", email="cancelother@test.com", password="testpass123!"
        )
        self.client.login(username="canceluser", password="testpass123!")

    def test_cancel_pending(self):
        export_req = DataExportRequest.objects.create(user=self.user)
        res = _delete(self.client, f"/api/privacy/exports/{export_req.id}/")
        self.assertEqual(res.status_code, 200)
        export_req.refresh_from_db()
        self.assertEqual(export_req.status, DataExportRequest.Status.CANCELLED)

    def test_cancel_completed_deletes_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"test": True}, f)
            temp_path = f.name

        export_req = DataExportRequest.objects.create(
            user=self.user,
            status=DataExportRequest.Status.COMPLETED,
            file_path=temp_path,
            completed_at=timezone.now(),
            expires_at=timezone.now() + timedelta(days=7),
        )
        res = _delete(self.client, f"/api/privacy/exports/{export_req.id}/")
        self.assertEqual(res.status_code, 200)
        self.assertFalse(os.path.exists(temp_path))
        export_req.refresh_from_db()
        self.assertEqual(export_req.status, DataExportRequest.Status.EXPIRED)

    def test_cancel_other_user(self):
        export_req = DataExportRequest.objects.create(user=self.other)
        res = _delete(self.client, f"/api/privacy/exports/{export_req.id}/")
        self.assertEqual(res.status_code, 404)


# ---------------------------------------------------------------------------
# Export generation task tests
# ---------------------------------------------------------------------------


@SECURE_OFF
class GenerateExportTaskTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="taskuser", email="task@test.com", password="testpass123!")
        self.other = User.objects.create_user(username="taskother", email="taskother@test.com", password="testpass123!")

    def _run_export(self, user=None):
        user = user or self.user
        export_req = DataExportRequest.objects.create(user=user)
        from accounts.privacy_tasks import generate_export

        result = generate_export({"export_id": str(export_req.id)})
        export_req.refresh_from_db()
        return export_req, result

    def test_generates_valid_json(self):
        export_req, _ = self._run_export()
        self.assertEqual(export_req.status, DataExportRequest.Status.COMPLETED)
        self.assertTrue(os.path.exists(export_req.file_path))
        with open(export_req.file_path) as f:
            data = json.load(f)
        self.assertIn("export_metadata", data)
        os.remove(export_req.file_path)

    def test_export_contains_all_sections(self):
        export_req, _ = self._run_export()
        with open(export_req.file_path) as f:
            data = json.load(f)
        expected = [
            "export_metadata",
            "profile",
            "subscription",
            "conversations",
            "analysis_results",
            "triage_results",
            "saved_models",
            "usage_summary",
            "notifications",
        ]
        for section in expected:
            self.assertIn(section, data, f"Missing section: {section}")
        os.remove(export_req.file_path)

    def test_profile_section_complete(self):
        export_req, _ = self._run_export()
        with open(export_req.file_path) as f:
            data = json.load(f)
        profile = data["profile"]
        self.assertEqual(profile["email"], "task@test.com")
        self.assertEqual(profile["username"], "taskuser")
        os.remove(export_req.file_path)

    def test_excludes_other_users_data(self):
        """CRITICAL: no cross-user data leakage."""
        from chat.models import Conversation

        Conversation.objects.create(user=self.other, title="Other's private convo")
        export_req, _ = self._run_export(self.user)
        with open(export_req.file_path) as f:
            data = json.load(f)
        self.assertEqual(len(data["conversations"]), 0)
        os.remove(export_req.file_path)

    def test_sets_completed_status(self):
        export_req, result = self._run_export()
        self.assertEqual(export_req.status, DataExportRequest.Status.COMPLETED)
        self.assertIsNotNone(export_req.completed_at)
        self.assertIsNotNone(export_req.expires_at)
        self.assertIsNotNone(export_req.file_size_bytes)
        self.assertGreater(export_req.file_size_bytes, 0)
        os.remove(export_req.file_path)

    def test_handles_user_with_no_data(self):
        """Fresh user exports empty sections, not errors."""
        export_req, _ = self._run_export()
        with open(export_req.file_path) as f:
            data = json.load(f)
        self.assertEqual(data["conversations"], [])
        self.assertEqual(data["analysis_results"], [])
        self.assertIsNone(data["subscription"])
        os.remove(export_req.file_path)


# ---------------------------------------------------------------------------
# Cleanup task tests
# ---------------------------------------------------------------------------


@SECURE_OFF
class ExportCleanupTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="cleanuser", email="clean@test.com", password="testpass123!")

    def test_expired_exports_cleaned(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({}, f)
            temp_path = f.name

        DataExportRequest.objects.create(
            user=self.user,
            status=DataExportRequest.Status.COMPLETED,
            file_path=temp_path,
            completed_at=timezone.now() - timedelta(days=10),
            expires_at=timezone.now() - timedelta(days=3),
        )

        from accounts.privacy_tasks import cleanup_expired_exports

        result = cleanup_expired_exports()
        self.assertEqual(result["expired_count"], 1)
        self.assertFalse(os.path.exists(temp_path))

    def test_active_exports_preserved(self):
        DataExportRequest.objects.create(
            user=self.user,
            status=DataExportRequest.Status.COMPLETED,
            completed_at=timezone.now(),
            expires_at=timezone.now() + timedelta(days=5),
        )

        from accounts.privacy_tasks import cleanup_expired_exports

        result = cleanup_expired_exports()
        self.assertEqual(result["expired_count"], 0)

    def test_missing_file_handled(self):
        DataExportRequest.objects.create(
            user=self.user,
            status=DataExportRequest.Status.COMPLETED,
            file_path="/nonexistent/path/export.json",
            completed_at=timezone.now() - timedelta(days=10),
            expires_at=timezone.now() - timedelta(days=3),
        )

        from accounts.privacy_tasks import cleanup_expired_exports

        result = cleanup_expired_exports()
        self.assertEqual(result["expired_count"], 1)


# ---------------------------------------------------------------------------
# PII inventory test
# ---------------------------------------------------------------------------


@SECURE_OFF
class PIIInventoryTest(TestCase):
    def test_all_user_fk_models_documented(self):
        """Verify key user-FK models appear in the export output."""
        user = User.objects.create_user(username="piiuser", email="pii@test.com", password="testpass123!")
        export_req = DataExportRequest.objects.create(user=user)

        from accounts.privacy_tasks import generate_export

        generate_export({"export_id": str(export_req.id)})
        export_req.refresh_from_db()

        with open(export_req.file_path) as f:
            data = json.load(f)

        required_sections = [
            "profile",
            "subscription",
            "conversations",
            "analysis_results",
            "triage_results",
            "saved_models",
            "usage_summary",
            "notifications",
        ]
        for section in required_sections:
            self.assertIn(section, data)

        os.remove(export_req.file_path)


# ---------------------------------------------------------------------------
# Data correction test (existing endpoint)
# ---------------------------------------------------------------------------


@SECURE_OFF
class DataCorrectionTest(TestCase):
    def test_profile_update(self):
        user = User.objects.create_user(username="corruser", email="corr@test.com", password="testpass123!")
        self.client.login(username="corruser", password="testpass123!")
        res = self.client.patch(
            "/api/auth/profile/",
            json.dumps({"display_name": "Updated Name", "bio": "New bio"}),
            content_type="application/json",
        )
        self.assertIn(res.status_code, (200, 204))
        user.refresh_from_db()
        self.assertEqual(user.display_name, "Updated Name")


# ---------------------------------------------------------------------------
# Compliance check test
# ---------------------------------------------------------------------------


@SECURE_OFF
class PrivacyComplianceCheckTest(TestCase):
    def test_check_passes(self):
        from syn.audit.compliance import check_privacy_data_export

        result = check_privacy_data_export()
        self.assertEqual(result["status"], "pass")
