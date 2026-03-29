"""Functional scenario tests for files/views.py.

Covers all 9 public view functions across 5 test classes.
Follows TST-001: Django TestCase + self.client, force_login for auth,
@override_settings(SECURE_SSL_REDIRECT=False).
"""

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from accounts.constants import Tier
from files.models import UserFile, UserQuota

User = get_user_model()

SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user(email, tier=Tier.TEAM, **kwargs):
    """Create a user with given tier."""
    username = email.split("@")[0]
    user = User.objects.create_user(username=username, email=email, password="testpass123", **kwargs)
    user.tier = tier
    user.save(update_fields=["tier"])
    return user


def _err_msg(resp):
    """Extract error message from ErrorEnvelopeMiddleware response."""
    data = resp.json()
    err = data.get("error", "")
    if isinstance(err, dict):
        return err.get("message", "")
    return err


def _upload_file(
    client,
    name="test.txt",
    content=b"hello world",
    content_type="text/plain",
    folder="",
    description="",
):
    """Upload a file via the API and return the response."""
    f = SimpleUploadedFile(name, content, content_type=content_type)
    data = {"file": f}
    if folder:
        data["folder"] = folder
    if description:
        data["description"] = description
    return client.post("/api/files/upload/", data, format="multipart")


# =========================================================================
# 1. FileUploadDownloadTest
# =========================================================================


@SECURE_OFF
class FileUploadDownloadTest(TestCase):
    """Scenario: upload files, list them, download, verify content."""

    def setUp(self):
        self.user = _make_user("uploader@example.com")
        self.client.force_login(self.user)

    def test_upload_list_download_scenario(self):
        """Upload a file, find it in the list, download it, verify content."""
        # Step 1: Upload
        resp = _upload_file(self.client, name="notes.txt", content=b"my notes")
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        file_id = data["id"]
        self.assertEqual(data["name"], "notes.txt")
        self.assertEqual(data["size"], 8)
        self.assertEqual(data["mime_type"], "text/plain")

        # Step 2: List files -- should contain the uploaded file
        resp = self.client.get("/api/files/")
        self.assertEqual(resp.status_code, 200)
        listing = resp.json()
        self.assertEqual(listing["total"], 1)
        self.assertEqual(listing["files"][0]["id"], file_id)
        self.assertEqual(listing["files"][0]["name"], "notes.txt")

        # Step 3: Download and verify content
        resp = self.client.get(f"/api/files/{file_id}/download/")
        self.assertEqual(resp.status_code, 200)
        content = b"".join(resp.streaming_content)
        self.assertEqual(content, b"my notes")

    def test_upload_multiple_and_filter_by_folder(self):
        """Upload files to different folders, filter list by folder."""
        _upload_file(self.client, name="a.txt", content=b"aaa", folder="docs")
        _upload_file(self.client, name="b.txt", content=b"bbb", folder="images")
        _upload_file(self.client, name="c.txt", content=b"ccc", folder="docs")

        # Filter by folder=docs
        resp = self.client.get("/api/files/", {"folder": "docs"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total"], 2)
        names = {f["name"] for f in data["files"]}
        self.assertEqual(names, {"a.txt", "c.txt"})

    def test_upload_and_search_by_name(self):
        """Upload files, search by name substring."""
        _upload_file(self.client, name="report_q1.txt", content=b"q1")
        _upload_file(self.client, name="report_q2.txt", content=b"q2")
        _upload_file(self.client, name="summary.txt", content=b"sum")

        resp = self.client.get("/api/files/", {"search": "report"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total"], 2)

    def test_upload_no_file_returns_400(self):
        """Upload with no file field returns 400."""
        resp = self.client.post("/api/files/upload/", {}, format="multipart")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("No file", _err_msg(resp))

    def test_upload_dangerous_extension_rejected(self):
        """Upload a .exe file is rejected."""
        resp = _upload_file(
            self.client,
            name="malware.exe",
            content=b"\x00" * 100,
            content_type="application/octet-stream",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("not allowed", _err_msg(resp))

    def test_upload_dangerous_mime_rejected(self):
        """Upload with dangerous MIME type is rejected."""
        resp = _upload_file(
            self.client,
            name="program.dat",
            content=b"\x00" * 100,
            content_type="application/x-executable",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("not allowed", _err_msg(resp))

    def test_list_pagination(self):
        """List files with limit and offset."""
        for i in range(5):
            _upload_file(self.client, name=f"file{i}.txt", content=b"x")

        resp = self.client.get("/api/files/", {"limit": 2, "offset": 0})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total"], 5)
        self.assertEqual(data["limit"], 2)
        self.assertEqual(len(data["files"]), 2)

        resp = self.client.get("/api/files/", {"limit": 2, "offset": 4})
        data = resp.json()
        self.assertEqual(len(data["files"]), 1)

    def test_download_nonexistent_file_returns_404(self):
        """Download a file that doesn't exist returns 404."""
        import uuid

        fake_id = uuid.uuid4()
        resp = self.client.get(f"/api/files/{fake_id}/download/")
        self.assertEqual(resp.status_code, 404)

    def test_unauthenticated_list_returns_401_or_403(self):
        """Unauthenticated request to list files is rejected."""
        self.client.logout()
        resp = self.client.get("/api/files/")
        self.assertIn(resp.status_code, (401, 403))


# =========================================================================
# 2. FileDetailTest
# =========================================================================


@SECURE_OFF
class FileDetailTest(TestCase):
    """Scenario: get detail, update metadata, delete file."""

    def setUp(self):
        self.user = _make_user("detail@example.com")
        self.client.force_login(self.user)

    def test_get_update_delete_scenario(self):
        """Upload -> get detail -> update metadata -> delete -> verify gone."""
        # Upload
        resp = _upload_file(
            self.client,
            name="data.csv",
            content=b"a,b,c\n1,2,3",
            content_type="text/csv",
            folder="raw",
            description="Raw data",
        )
        self.assertEqual(resp.status_code, 201)
        file_id = resp.json()["id"]

        # GET detail
        resp = self.client.get(f"/api/files/{file_id}/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["name"], "data.csv")
        self.assertEqual(data["folder"], "raw")
        self.assertEqual(data["description"], "Raw data")
        self.assertFalse(data["is_public"])
        self.assertIsNotNone(data["accessed_at"])

        # PATCH -- update metadata
        resp = self.client.patch(
            f"/api/files/{file_id}/",
            {
                "folder": "processed",
                "description": "Cleaned data",
                "tags": ["csv", "q1"],
            },
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["folder"], "processed")
        self.assertEqual(data["description"], "Cleaned data")
        self.assertEqual(data["tags"], ["csv", "q1"])

        # Verify DB state
        uf = UserFile.objects.get(id=file_id)
        self.assertEqual(uf.folder, "processed")
        self.assertEqual(uf.tags, ["csv", "q1"])

        # DELETE
        resp = self.client.delete(f"/api/files/{file_id}/")
        self.assertEqual(resp.status_code, 204)

        # Verify gone
        self.assertFalse(UserFile.objects.filter(id=file_id).exists())

    def test_detail_not_found(self):
        """GET detail for nonexistent file returns 404."""
        import uuid

        resp = self.client.get(f"/api/files/{uuid.uuid4()}/")
        self.assertEqual(resp.status_code, 404)

    def test_cannot_access_other_users_file(self):
        """User cannot access another user's file."""
        other = _make_user("other@example.com")
        self.client.force_login(other)
        resp = _upload_file(self.client, name="secret.txt", content=b"secret")
        file_id = resp.json()["id"]

        # Switch back to original user
        self.client.force_login(self.user)
        resp = self.client.get(f"/api/files/{file_id}/")
        self.assertEqual(resp.status_code, 404)

    def test_patch_is_public_ignored(self):
        """PATCH cannot set is_public — must use create_share_link endpoint (HIGH-10)."""
        resp = _upload_file(self.client, name="pub.txt", content=b"public")
        file_id = resp.json()["id"]

        resp = self.client.patch(
            f"/api/files/{file_id}/",
            {"is_public": True},
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        # is_public should NOT be changed via PATCH
        self.assertFalse(UserFile.objects.get(id=file_id).is_public)

    def test_delete_updates_quota(self):
        """Deleting a file should reduce quota used_bytes and file_count."""
        resp = _upload_file(self.client, name="big.txt", content=b"x" * 500)
        file_id = resp.json()["id"]

        quota = UserQuota.get_or_create_for_user(self.user)
        used_before = quota.used_bytes
        count_before = quota.file_count

        self.client.delete(f"/api/files/{file_id}/")

        quota.refresh_from_db()
        self.assertEqual(quota.used_bytes, used_before - 500)
        self.assertEqual(quota.file_count, count_before - 1)


# =========================================================================
# 3. FileSharingTest
# =========================================================================


@SECURE_OFF
class FileSharingTest(TestCase):
    """Scenario: create share link, access shared file, revoke, verify revoked."""

    def setUp(self):
        self.user = _make_user("sharer@example.com")
        self.client.force_login(self.user)

    def test_share_access_revoke_scenario(self):
        """Upload -> share -> access as anon -> revoke -> verify inaccessible."""
        # Upload
        resp = _upload_file(self.client, name="shared.txt", content=b"shared content")
        self.assertEqual(resp.status_code, 201)
        file_id = resp.json()["id"]

        # Create share link
        resp = self.client.post(f"/api/files/{file_id}/share/")
        self.assertEqual(resp.status_code, 200)
        share_data = resp.json()
        token = share_data["share_token"]
        self.assertTrue(len(token) > 10)
        self.assertIn("/api/files/shared/", share_data["url"])

        # Verify DB state
        uf = UserFile.objects.get(id=file_id)
        self.assertTrue(uf.is_public)
        self.assertEqual(uf.share_token, token)

        # Access shared file as anonymous (logout)
        self.client.logout()
        resp = self.client.get(f"/api/files/shared/{token}/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["name"], "shared.txt")
        self.assertEqual(data["size"], 14)

        # Download shared file
        resp = self.client.get(f"/api/files/shared/{token}/", {"download": "true"})
        self.assertEqual(resp.status_code, 200)
        content = b"".join(resp.streaming_content)
        self.assertEqual(content, b"shared content")

        # Re-authenticate and revoke
        self.client.force_login(self.user)
        resp = self.client.delete(f"/api/files/{file_id}/unshare/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "share_revoked")

        # Verify DB state after revoke
        uf.refresh_from_db()
        self.assertFalse(uf.is_public)
        self.assertEqual(uf.share_token, "")

        # Verify shared link no longer works (anonymous)
        self.client.logout()
        resp = self.client.get(f"/api/files/shared/{token}/")
        self.assertEqual(resp.status_code, 404)

    def test_share_nonexistent_file_returns_404(self):
        """Create share link for nonexistent file returns 404."""
        import uuid

        resp = self.client.post(f"/api/files/{uuid.uuid4()}/share/")
        self.assertEqual(resp.status_code, 404)

    def test_revoke_nonexistent_file_returns_404(self):
        """Revoke share for nonexistent file returns 404."""
        import uuid

        resp = self.client.delete(f"/api/files/{uuid.uuid4()}/unshare/")
        self.assertEqual(resp.status_code, 404)

    def test_shared_file_invalid_token_returns_404(self):
        """Access shared file with invalid token returns 404."""
        self.client.logout()
        resp = self.client.get("/api/files/shared/nonexistent_token_xyz/")
        self.assertEqual(resp.status_code, 404)

    def test_share_detail_shows_token_when_public(self):
        """GET detail for a shared file includes share_token."""
        resp = _upload_file(self.client, name="show_token.txt", content=b"tok")
        file_id = resp.json()["id"]

        # Share
        resp = self.client.post(f"/api/files/{file_id}/share/")
        token = resp.json()["share_token"]

        # GET detail should include share_token
        resp = self.client.get(f"/api/files/{file_id}/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["share_token"], token)

    def test_cannot_share_other_users_file(self):
        """User cannot create share link for another user's file."""
        other = _make_user("victim@example.com")
        self.client.force_login(other)
        resp = _upload_file(self.client, name="private.txt", content=b"mine")
        file_id = resp.json()["id"]

        # Switch to attacker
        self.client.force_login(self.user)
        resp = self.client.post(f"/api/files/{file_id}/share/")
        self.assertEqual(resp.status_code, 404)


# =========================================================================
# 4. StorageQuotaTest
# =========================================================================


@SECURE_OFF
class StorageQuotaTest(TestCase):
    """Scenario: check quota, upload affects quota, quota exceeded."""

    def setUp(self):
        self.user = _make_user("quota@example.com")
        self.client.force_login(self.user)

    def test_quota_initial_state(self):
        """Fresh user should have quota with zero usage."""
        resp = self.client.get("/api/files/quota/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["used_bytes"], 0)
        self.assertEqual(data["file_count"], 0)
        self.assertGreater(data["quota_bytes"], 0)
        self.assertEqual(data["usage_percent"], 0.0)
        self.assertEqual(data["tier"], self.user.tier)

    def test_upload_increases_quota(self):
        """Uploading a file increases used_bytes and file_count."""
        # Check initial
        resp = self.client.get("/api/files/quota/")
        initial_used = resp.json()["used_bytes"]
        initial_count = resp.json()["file_count"]

        # Upload
        content = b"x" * 1024
        _upload_file(self.client, name="measure.txt", content=content)

        # Check updated
        resp = self.client.get("/api/files/quota/")
        data = resp.json()
        self.assertEqual(data["used_bytes"], initial_used + 1024)
        self.assertEqual(data["file_count"], initial_count + 1)

    def test_quota_exceeded_rejects_upload(self):
        """When quota is full, upload is rejected with 413."""
        # Set quota to very small
        quota = UserQuota.get_or_create_for_user(self.user)
        quota.quota_bytes = 100
        quota.used_bytes = 90
        quota.save()

        # Try uploading a file larger than remaining
        resp = _upload_file(self.client, name="toobig.txt", content=b"x" * 50)
        self.assertEqual(resp.status_code, 413)
        self.assertIn("quota", _err_msg(resp).lower())

    def test_max_file_size_exceeded(self):
        """File larger than max_file_size_bytes is rejected."""
        quota = UserQuota.get_or_create_for_user(self.user)
        quota.max_file_size_bytes = 100
        quota.save()

        resp = _upload_file(self.client, name="huge.txt", content=b"x" * 200)
        self.assertEqual(resp.status_code, 413)
        self.assertIn("too large", _err_msg(resp).lower())

    def test_max_files_exceeded(self):
        """When file_count hits max_files, upload is rejected."""
        quota = UserQuota.get_or_create_for_user(self.user)
        quota.max_files = 2
        quota.file_count = 2
        quota.save()

        resp = _upload_file(self.client, name="extra.txt", content=b"nope")
        self.assertEqual(resp.status_code, 413)
        self.assertIn("limit", _err_msg(resp).lower())

    def test_unauthenticated_quota_rejected(self):
        """Unauthenticated request to quota endpoint is rejected."""
        self.client.logout()
        resp = self.client.get("/api/files/quota/")
        self.assertIn(resp.status_code, (401, 403))


# =========================================================================
# 5. FolderTest
# =========================================================================


@SECURE_OFF
class FolderTest(TestCase):
    """Scenario: list folders, folder filtering."""

    def setUp(self):
        self.user = _make_user("folders@example.com")
        self.client.force_login(self.user)

    def test_list_folders_empty(self):
        """No files means no folders."""
        resp = self.client.get("/api/files/folders/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["folders"], [])

    def test_list_folders_with_files(self):
        """Folders are derived from uploaded files, sorted, deduplicated."""
        _upload_file(self.client, name="a.txt", content=b"a", folder="beta")
        _upload_file(self.client, name="b.txt", content=b"b", folder="alpha")
        _upload_file(self.client, name="c.txt", content=b"c", folder="beta")
        _upload_file(self.client, name="d.txt", content=b"d")  # no folder

        resp = self.client.get("/api/files/folders/")
        self.assertEqual(resp.status_code, 200)
        folders = resp.json()["folders"]
        self.assertEqual(folders, ["alpha", "beta"])

    def test_folders_isolated_per_user(self):
        """User A's folders should not include user B's folders."""
        _upload_file(self.client, name="a.txt", content=b"a", folder="my_folder")

        other = _make_user("other_folder@example.com")
        self.client.force_login(other)
        _upload_file(self.client, name="b.txt", content=b"b", folder="their_folder")

        # Other user should only see their folder
        resp = self.client.get("/api/files/folders/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["folders"], ["their_folder"])

        # Original user should only see their folder
        self.client.force_login(self.user)
        resp = self.client.get("/api/files/folders/")
        self.assertEqual(resp.json()["folders"], ["my_folder"])

    def test_list_files_filter_by_type(self):
        """List files filtered by file_type."""
        _upload_file(self.client, name="pic.png", content=b"\x89PNG", content_type="image/png")
        _upload_file(self.client, name="doc.txt", content=b"text", content_type="text/plain")

        resp = self.client.get("/api/files/", {"type": "image"})
        self.assertEqual(resp.status_code, 200)
        # image/png should be detected as image type
        for f in resp.json()["files"]:
            self.assertEqual(f["type"], "image")
