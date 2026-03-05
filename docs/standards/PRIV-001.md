**PRIV-001: PRIVACY & DATA PROTECTION STANDARD**

**Version:** 1.0
**Status:** APPROVED
**Date:** 2026-03-05
**Author:** Eric + Claude (Systems Architect)
**Compliance:**
- DOC-001 >= 1.2 (Documentation Structure)
- SEC-001 >= 1.1 (Security Architecture)
- SOC 2 Trust Services Criteria P1.1-P1.8 (Privacy)
- GDPR Articles 15, 17, 20 (informational alignment)
**Related Standards:**
- AUD-001 >= 1.0 (Audit Trail)
- API-001 >= 1.0 (API Design)
- DAT-001 >= 1.0 (Data Model)
- SCH-001 >= 1.0 (Cognitive Scheduler)

---

## **1. SCOPE AND PURPOSE**

### **1.1 Purpose**

PRIV-001 defines PII inventory, data subject rights, self-service data export specification, retention policies, and SOC 2 Privacy Criteria compliance mapping for the Svend platform. This standard ensures users can access, correct, and export their personal data in a documented, auditable, and machine-readable manner.

**Core Principle:**

> Users own their data. The platform is a custodian, not a proprietor. Access, correction, and portability are non-negotiable rights.

### **1.2 Scope**

**Applies to:**
- All Django models with a `user` foreign key (direct PII linkage)
- All encrypted fields storing user-generated content
- Data export, correction, and deletion workflows
- Audit logging of privacy-relevant operations

**Does NOT apply to:**
- System audit logs (`SysLogEntry`) — retained per AUD-001
- ML training pipeline data (`TrainingCandidate`) — internal operational data
- Pipeline diagnostic traces (`TraceLog`) — internal debugging data
- Aggregate analytics without user-level attribution

---

## **2. NORMATIVE REFERENCES**

### **2.1 Kjerne Standards**

| Standard | Section | Requirement |
|----------|---------|-------------|
| **SEC-001** | §4-7 | Auth patterns, field encryption, tenant isolation |
| **API-001** | §4 | Endpoint naming conventions |
| **DAT-001** | §3 | UUID primary keys, model patterns |
| **AUD-001** | §3-4 | Immutable audit logging |
| **SCH-001** | §5 | Task registration and scheduling |

### **2.2 External Standards**

| Standard | Clause | Requirement |
|----------|--------|-------------|
| **SOC 2** | P1.1-P1.8 | Privacy Trust Services Criteria |
| **GDPR** | Art. 15 | Right of access (informational) |
| **GDPR** | Art. 17 | Right to erasure (informational) |
| **GDPR** | Art. 20 | Right to data portability (informational) |

---

## **3. PII INVENTORY**

### **3.1 Data Classification**

| Classification | Description | Examples |
|----------------|-------------|----------|
| **PII-Direct** | Directly identifies an individual | email, display_name |
| **PII-Financial** | Payment or billing information | stripe_customer_id |
| **PII-Content** | User-generated content linked to identity | messages, analysis results |
| **PII-Behavioral** | Usage patterns linked to identity | page views, query counts |

### **3.2 PII Inventory Table**

<!-- assert: PII inventory covers all user-FK models | check=priv-pii-inventory -->
<!-- impl: accounts/tests_privacy.py -->
<!-- test: accounts.tests_privacy.PIIInventoryTest.test_all_user_fk_models_documented -->

| Model | Fields | Classification | Encrypted | Retention |
|-------|--------|---------------|-----------|-----------|
| `accounts.User` | email, display_name, bio | PII-Direct | No | Account lifetime |
| `accounts.User` | industry, role, experience_level, organization_size | PII-Direct | No | Account lifetime |
| `accounts.User` | preferences, avatar_url, current_theme | PII-Direct | No | Account lifetime |
| `accounts.User` | stripe_customer_id | PII-Financial | Yes (Fernet) | Account lifetime |
| `accounts.User` | total_queries, total_tokens_used, last_active_at | PII-Behavioral | No | Account lifetime |
| `accounts.Subscription` | status, billing period dates | PII-Financial | No | Account lifetime |
| `chat.Conversation` | title | PII-Content | No | Account lifetime |
| `chat.Message` | content, reasoning_trace, tool_calls | PII-Content | Yes (Fernet) | Account lifetime |
| `chat.UsageLog` | request_count, tokens, domain_counts | PII-Behavioral | No | 90 days |
| `chat.EventLog` | event_type, category, action, page | PII-Behavioral | No | 90 days |
| `agents_api.DSWResult` | data (analysis results) | PII-Content | Yes (Fernet) | Account lifetime |
| `agents_api.TriageResult` | cleaned_csv, report_markdown, summary_stats | PII-Content | Yes (Fernet) | Account lifetime |
| `agents_api.SavedModel` | name, description, training_config, data_lineage | PII-Content | No | Account lifetime |
| `agents_api.AgentLog` | agent_name, action, error | PII-Behavioral | No | 90 days |
| `notifications.Notification` | title, message | PII-Content | No | Account lifetime |
| `notifications.NotificationPreference` | email_mode, muted_types | PII-Direct | No | Account lifetime |

### **3.3 Encryption Status**

All sensitive content fields use `EncryptedTextField` or `EncryptedCharField` from `core.encryption` (Fernet: AES-128-CBC + HMAC-SHA256). See SEC-001 §7 for envelope encryption architecture.

---

## **4. DATA SUBJECT RIGHTS**

### **4.1 Right of Access (SOC 2 P1.8)**

Users can access their data through:

1. **Profile view:** `GET /api/auth/me/` — returns profile fields
2. **Data export:** `POST /api/privacy/exports/` — generates machine-readable JSON of all user data

<!-- assert: Self-service data export endpoint exists | check=priv-data-export -->
<!-- impl: accounts/privacy_views.py -->
<!-- test: accounts.tests_privacy.DataExportViewTests.test_create_export -->

### **4.2 Right of Correction (SOC 2 P1.8)**

Users can correct their profile data through:

1. **Profile update:** `PUT /api/auth/profile/` — updates display_name, bio, industry, role, experience_level, organization_size, avatar_url, preferences, current_theme

<!-- assert: Profile update endpoint allows correction of PII fields | check=priv-correction -->
<!-- impl: api/views.py:update_profile -->
<!-- test: accounts.tests_privacy.DataCorrectionTest.test_profile_update -->

Correction of conversation content and analysis results is not supported — these are append-only user-generated records. Users can delete individual conversations via `DELETE /api/conversations/<id>/`.

### **4.3 Right of Portability**

Export format is machine-readable JSON with a documented, versioned schema (§5). This satisfies SOC 2 P1.8 and aligns with GDPR Art. 20.

### **4.4 Right of Deletion (Future)**

Full account deletion with verified data purge across all tables is planned as a separate feature. Current capability: users can delete individual conversations. Account-level deletion requires manual request per privacy policy §6.

---

## **5. DATA EXPORT SPECIFICATION**

### **5.1 Export Model**

<!-- assert: DataExportRequest model tracks export lifecycle | check=priv-export-model -->
<!-- impl: accounts/models.py:DataExportRequest -->
<!-- test: accounts.tests_privacy.DataExportRequestModelTests.test_create_request -->

```
DataExportRequest
├── id (UUID PK)
├── user (FK → User, CASCADE)
├── status (pending → processing → completed | failed | expired | cancelled)
├── export_format ("json")
├── file_path, file_size_bytes
├── created_at, processing_started_at, completed_at, expires_at, downloaded_at
```

### **5.2 Export Scope**

| Section | Source Model(s) | Fields Included |
|---------|----------------|-----------------|
| `export_metadata` | DataExportRequest | export_id, user_id, generated_at, format_version |
| `profile` | User | email, username, display_name, bio, demographics, tier, preferences, dates |
| `subscription` | Subscription | status, billing period, created_at |
| `conversations` | Conversation + Message | titles, message content (decrypted), timestamps |
| `analysis_results` | DSWResult | result data (decrypted), metadata |
| `triage_results` | TriageResult | report_markdown (decrypted), summary_stats, metadata |
| `saved_models` | SavedModel | name, description, configs |
| `usage_summary` | UsageLog | aggregated request counts, token usage |
| `notifications` | Notification | titles, messages, read status, timestamps |

### **5.3 Excluded Data**

| Data | Reason |
|------|--------|
| `syn.audit.SysLogEntry` | System audit trail — retained per AUD-001, not user content |
| `chat.TraceLog` | Internal pipeline diagnostics |
| `chat.TrainingCandidate` | Internal ML pipeline data |
| `chat.EventLog` (detail) | High-volume telemetry — summary count included |
| `agents_api.AgentLog` | Internal operational logging |
| `stripe_customer_id` | Opaque Stripe internal identifier |
| `stripe_subscription_id` | Opaque Stripe internal identifier |

### **5.4 Delivery and Security**

- Export files stored at `media/exports/export-{user_id}-{export_id}.json`
- Download requires authentication — same user only (SEC-001 §4)
- Files auto-expire 7 days after generation
- Cleanup task runs weekly to delete expired files

<!-- assert: Export files auto-expire after 7 days | check=priv-export-expiry -->
<!-- impl: accounts/privacy_tasks.py:cleanup_expired_exports -->
<!-- test: accounts.tests_privacy.ExportCleanupTests.test_expired_exports_cleaned -->

### **5.5 Rate Limiting**

One export request per user per 24-hour window. Cancelled exports do not count toward the limit.

<!-- assert: Export rate limited to 1 per 24h per user | check=priv-rate-limit -->
<!-- impl: accounts/privacy_views.py -->
<!-- test: accounts.tests_privacy.DataExportViewTests.test_rate_limit_24h -->

### **5.6 Async Generation**

Export generation runs as an async task via syn.sched (SCH-001):

| Task | Queue | Priority | Timeout | Retries |
|------|-------|----------|---------|---------|
| `privacy.generate_export` | BATCH | NORMAL | 300s | 2 |
| `privacy.cleanup_expired_exports` | BATCH | LOW | 120s | 1 |

Schedule: `privacy.cleanup_expired_exports` runs weekly (Sunday 03:00 UTC).

---

## **6. RETENTION POLICIES**

### **6.1 Retention Schedule**

| Data Category | Retention | Deletion Trigger | Enforcement |
|---------------|-----------|------------------|-------------|
| Account profile | Account lifetime | Account deletion | Manual (future: automated) |
| Conversations + messages | Account lifetime | User delete or account deletion | `DELETE /api/conversations/<id>/` |
| Analysis results | Account lifetime | Account deletion | Manual |
| Usage logs | 90 days rolling | Automated cleanup | data_retention check (existing) |
| Event logs | 90 days rolling | Automated cleanup | data_retention check (existing) |
| Agent logs | 90 days rolling | Automated cleanup | data_retention check (existing) |
| Export files | 7 days after generation | `privacy.cleanup_expired_exports` | Weekly cron |
| Audit logs | 1 year minimum | SOC 2 requirement | AUD-001 retention policy |

### **6.2 Automated Enforcement**

- `privacy.cleanup_expired_exports` — weekly, deletes expired export files
- `audit.cleanup_violations` — weekly, cleans resolved violations >90 days (existing)
- `data_retention` compliance check — verifies retention policy adherence (existing)

---

## **7. SOC 2 COMPLIANCE MAPPING**

| Control | Requirement | Implementation | Status |
|---------|-------------|----------------|--------|
| **P1.1** | Privacy notice | `/privacy/` page (templates/privacy.html) | Met |
| **P1.2** | Choice and consent | Registration flow, email opt-out | Met |
| **P1.3** | Collection limitation | Privacy policy §1 documents collection | Met |
| **P1.4** | Use limitation | Tier-based feature gating (BILL-001) | Met |
| **P1.5** | Retention and disposal | Retention schedule (§6), cleanup tasks | Met |
| **P1.6** | Disposal | Cleanup tasks, manual account deletion | Partial |
| **P1.7** | Quality | Profile edit endpoint | Met |
| **P1.8** | Access and correction | Data export + profile edit | **Met** |

---

## **8. AUDIT REQUIREMENTS**

All privacy-relevant operations MUST be logged via `generate_entry()` (AUD-001):

<!-- assert: Privacy export operations are audit-logged | check=priv-audit-logging -->
<!-- impl: accounts/privacy_tasks.py:generate_export -->
<!-- test: accounts.tests_privacy.GenerateExportTaskTests.test_audit_log_created -->

| Event | Trigger |
|-------|---------|
| `privacy.export.requested` | User initiates export |
| `privacy.export.completed` | Export file generated successfully |
| `privacy.export.downloaded` | User downloads export file |
| `privacy.export.expired` | Cleanup task expires old export |
| `privacy.export.cancelled` | User cancels export request |
| `privacy.export.failed` | Export generation fails |

---

## **9. ACCEPTANCE CRITERIA**

Summary of all assertions in this standard:

| # | Assertion | Check ID |
|---|-----------|----------|
| 1 | PII inventory covers all user-FK models | priv-pii-inventory |
| 2 | Self-service data export endpoint exists | priv-data-export |
| 3 | Profile update endpoint allows correction | priv-correction |
| 4 | DataExportRequest model tracks lifecycle | priv-export-model |
| 5 | Export files auto-expire after 7 days | priv-export-expiry |
| 6 | Export rate limited to 1 per 24h per user | priv-rate-limit |
| 7 | Privacy export operations are audit-logged | priv-audit-logging |

---

## **10. REVISION HISTORY**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-05 | Eric + Claude | Initial release — PII inventory, data export, SOC 2 P1.8 closure |
