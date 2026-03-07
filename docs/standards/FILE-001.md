**FILE-001: FILE STORAGE STANDARD**

**Version:** 1.0
**Status:** APPROVED
**Date:** 2026-03-07
**Author:** Eric + Claude (Systems Architect)
**Compliance:**
- DOC-001 >= 1.0 (Documentation Structure)
- SEC-001 >= 1.0 (Security Architecture -- file encryption, access control)
- DAT-001 >= 1.0 (Data Model -- UUID PKs, field patterns)
- SOC 2 CC6.1 (Logical Access Controls -- per-user file isolation)
- SOC 2 CC6.7 (Restriction of Access to System Configurations)

---

## **1. SCOPE AND PURPOSE**

FILE-001 defines the file storage, sharing, and quota management patterns for Svend's file upload system.

## **2. MODELS**

### **2.1 UserFile**

<!-- assert: UserFile model stores uploaded files with per-user isolation and encryption support | check=file-model -->
<!-- impl: files/models.py:UserFile -->
<!-- impl: files/models.py:UserQuota -->
<!-- impl: files/models.py:user_file_path -->
<!-- test: files.tests_views.FileUploadDownloadTest.test_upload_list_download_scenario -->
<!-- test: files.tests_views.StorageQuotaTest.test_quota_initial_state -->
<!-- test: files.tests_views.StorageQuotaTest.test_upload_increases_quota -->

UserFile tracks uploaded files with metadata (name, size, MIME type, folder, encryption status). UserQuota enforces per-tier storage limits.

## **3. FILE OPERATIONS**

### **3.1 Upload & Download**

<!-- assert: File upload validates extension/MIME, enforces quota, stores securely; download serves with correct content-disposition | check=file-upload-download -->
<!-- impl: files/views.py:upload_file -->
<!-- impl: files/views.py:list_files -->
<!-- impl: files/views.py:download_file -->
<!-- impl: files/views.py:file_detail -->
<!-- test: files.tests_views.FileUploadDownloadTest.test_upload_list_download_scenario -->
<!-- test: files.tests_views.FileUploadDownloadTest.test_upload_multiple_and_filter_by_folder -->
<!-- test: files.tests_views.FileUploadDownloadTest.test_upload_and_search_by_name -->
<!-- test: files.tests_views.FileUploadDownloadTest.test_upload_no_file_returns_400 -->
<!-- test: files.tests_views.FileUploadDownloadTest.test_upload_dangerous_extension_rejected -->
<!-- test: files.tests_views.FileUploadDownloadTest.test_upload_dangerous_mime_rejected -->
<!-- test: files.tests_views.FileUploadDownloadTest.test_list_pagination -->
<!-- test: files.tests_views.FileDetailTest.test_get_update_delete_scenario -->
<!-- test: files.tests_views.FileDetailTest.test_cannot_access_other_users_file -->
<!-- test: files.tests_views.FileDetailTest.test_delete_updates_quota -->

### **3.2 Sharing**

<!-- assert: File sharing creates unique share tokens; shared files accessible without auth; revocation removes access | check=file-sharing -->
<!-- impl: files/views.py:create_share_link -->
<!-- impl: files/views.py:shared_file -->
<!-- impl: files/views.py:revoke_share_link -->
<!-- test: files.tests_views.FileSharingTest.test_share_access_revoke_scenario -->
<!-- test: files.tests_views.FileSharingTest.test_shared_file_invalid_token_returns_404 -->
<!-- test: files.tests_views.FileSharingTest.test_cannot_share_other_users_file -->

### **3.3 Quota Management**

<!-- assert: Storage quota enforced per-tier with per-file and total limits | check=file-quota -->
<!-- impl: files/views.py:storage_quota -->
<!-- impl: files/views.py:list_folders -->
<!-- test: files.tests_views.StorageQuotaTest.test_quota_initial_state -->
<!-- test: files.tests_views.StorageQuotaTest.test_upload_increases_quota -->
<!-- test: files.tests_views.StorageQuotaTest.test_quota_exceeded_rejects_upload -->
<!-- test: files.tests_views.StorageQuotaTest.test_max_file_size_exceeded -->
<!-- test: files.tests_views.FolderTest.test_list_folders_with_files -->
<!-- test: files.tests_views.FolderTest.test_folders_isolated_per_user -->

## **4. SECURITY**

- Files stored under per-user directory structure (`user_file_path`)
- Dangerous extensions (.exe, .bat, .cmd, .scr, .pif, .com) blocked at upload
- Dangerous MIME types blocked at upload
- Per-user isolation enforced on all CRUD operations
- Share tokens are unique random strings; revocation deletes the token

## **5. ACCEPTANCE CRITERIA**

| # | Assertion | Check ID |
|---|-----------|----------|
| 1 | UserFile model with per-user isolation | file-model |
| 2 | Upload validates and enforces quota | file-upload-download |
| 3 | Sharing with token-based access | file-sharing |
| 4 | Quota management per-tier | file-quota |

---

## **REVISION HISTORY**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-07 | Claude | Initial release -- file storage, sharing, quota management |
