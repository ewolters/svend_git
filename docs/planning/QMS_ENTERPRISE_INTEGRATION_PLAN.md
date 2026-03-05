# Enterprise QMS Integration — Implementation Plan

**Version:** 1.0
**Date:** 2026-03-04
**Authors:** Eric + Claude (Systems Architect)
**Roadmap Item:** QMS Enterprise Integration (Major Update) — `9490b2d5`
**Parent Roadmap:** `docs/roadmaps/2026-03-03_QMS_roadmap_.md` (Phases 2-4)
**Standards:** QMS-001, QMS-002, SEC-001, DAT-001, API-001, CHG-001

---

## 1. What We're Building

Five capabilities that turn Svend's QMS from a tool collection into an enterprise compliance platform:

1. **In-app notification system** (bell icon, real-time)
2. **Email notification with secure response links** (non-user participation)
3. **21 CFR Part 11 compliant signature chain** (NCRs, CAPAs, Supplier Reviews)
4. **Management Review templates** (customizable, clause 9.3)
5. **Artifact uploads** (attached to NCRs, CAPAs, and other QMS records)

---

## 2. Current State

| Capability | Status | What Exists |
|-----------|--------|-------------|
| Notifications (bell) | **Not implemented** | No Notification model, no bell UI |
| Email notifications | **Partial** | Drip email system (`api/tasks.py`), `send_mail()` infra, SMTP configured. No QMS-specific emails. No response links for non-users. |
| CFR signatures | **Not implemented** | Audit trail + hash chain (`SynaraImmutableLog`) provides the immutability backbone. 21 CFR Part 11 §11.10(e) referenced in code comments. No actual e-signature model or workflow. |
| Management Review | **Implemented (basic)** | `ManagementReview` model exists in `iso_views.py`. CRUD endpoints. No customizable templates. No automated input aggregation. |
| Artifact uploads | **Partial** | `files/` app exists with upload, download, sharing, quotas. Not wired to NCR/CAPA/QMS records. |
| NCR | **Implemented** | `NonconformanceRecord` model in `agents_api/models.py` |
| CAPA | **Embedded** | CAPA is a status within NCR lifecycle, not a standalone model |
| Supplier Reviews | **Implemented** | `SupplierRecord` model exists |
| Employee/ResourceCommitment/ActionToken | **Implemented** | Phase 4B complete (QMS-002) |

**Key foundation already in place:**
- `ActionToken` model (QMS-002 §4.3) — cryptographic, time-limited, single-use, action-scoped tokens. This is the backbone for email response links.
- `SynaraImmutableLog` — hash-chained audit trail. This + a new Signature model gives us CFR compliance.
- `files/` app — upload/download/quota infrastructure. Just needs FK wiring to QMS records.

---

## 3. Architecture Decisions

### 3.1 Notification Model

New model: `Notification` in `agents_api/models.py`

```
Notification (uuid, SynaraEntity)
  ├── user            FK → User (recipient)
  ├── tenant          FK → Tenant (nullable — personal notifications have no tenant)
  ├── category        CharField (qms / system / billing / collaboration)
  ├── severity        CharField (info / warning / action_required / critical)
  ├── title           CharField(255)
  ├── message         TextField
  ├── link            CharField(512) — relative URL to relevant page
  ├── source_type     CharField — ContentType-style reference (e.g., "ncr", "capa", "review")
  ├── source_id       UUIDField (nullable) — FK to source object
  ├── is_read         BooleanField (default False)
  ├── read_at         DateTimeField (nullable)
  ├── is_email_sent   BooleanField (default False) — tracks if email notification also sent
  └── expires_at      DateTimeField (nullable) — auto-dismiss after expiry
```

**Why not Django's ContentType framework?** It adds complexity for minimal benefit. A string `source_type` + UUID `source_id` is sufficient for linking notifications to QMS records and keeps queries simple.

**Delivery:** SSE (Server-Sent Events) for real-time push to the bell icon. No WebSocket dependency — SSE works through Cloudflare, is simpler, and matches our single-server architecture.

### 3.2 Email Response System

Reuses the existing `ActionToken` infrastructure from QMS-002. New action types:

| Action Type | Trigger | Email Content |
|------------|---------|---------------|
| `approve_signature` | Signature request on NCR/CAPA | "Review and sign [document]. [Approve] [Reject with Comment]" |
| `acknowledge_ncr` | NCR assigned to responsible party | "[NCR title] requires your attention. [Acknowledge] [Request Info]" |
| `review_capa` | CAPA submitted for review | "CAPA [title] is ready for review. [Approve] [Return for Revision]" |
| `supplier_response` | Supplier CAPA issued | "A corrective action has been requested. [View Details] [Respond]" |

Extends ActionToken.action_type choices. Response pages are minimal, scoped, no-auth forms served at `/action/<token>/`.

### 3.3 CFR Part 11 Signature Chain

New model: `ElectronicSignature` in `agents_api/models.py`

```
ElectronicSignature (uuid, SynaraImmutableLog)
  ├── signer          FK → User (nullable — non-users sign via ActionToken)
  ├── signer_name     CharField(255) — display name at time of signing
  ├── signer_email    EmailField
  ├── employee        FK → Employee (nullable — linked if non-user)
  ├── document_type   CharField (ncr / capa / supplier_review / management_review / deviation)
  ├── document_id     UUIDField — FK to signed document
  ├── signature_type  CharField (approval / rejection / acknowledgment / review)
  ├── meaning         CharField(255) — "I approve this NCR closure" (21 CFR §11.50: meaning of signature)
  ├── comment         TextField (nullable — required for rejections)
  ├── ip_address      GenericIPAddressField
  ├── user_agent      CharField(512)
  ├── action_token    FK → ActionToken (nullable — set when signed via email link)
  └── password_confirmed BooleanField — True if signer re-entered password (§11.10(a))
```

**Why inherit from SynaraImmutableLog?** CFR Part 11 §11.10(e) requires that signed records cannot be modified or deleted. `SynaraImmutableLog.save()` raises `ValidationError` on re-save, and `delete()` raises `ValueError`. The hash chain provides tamper detection. This is the exact guarantee CFR demands.

**Signature workflow:**
1. User initiates signature request on a document (NCR, CAPA, etc.)
2. System creates `Notification` + sends email with `ActionToken` (action_type=`approve_signature`)
3. Signer clicks link → sees document summary + meaning statement
4. Signer re-enters password (for authenticated users) or confirms via token (for non-users)
5. `ElectronicSignature` is created (immutable, hash-chained)
6. If all required signatures collected → document transitions to next state
7. `SysLogEntry` records the event for audit trail

**Signature requirements per document type:**

| Document | Required Signatures | Sequence |
|----------|-------------------|----------|
| NCR closure | Originator + Quality Manager | Parallel |
| CAPA approval | CAPA Owner + Quality Manager + Management Representative | Sequential |
| Supplier CAPA | Internal Quality + Supplier Contact | Sequential |
| Management Review | Management Representative + (optional) additional reviewers | Parallel |
| Deviation/Waiver | Originator + Quality + Engineering (if applicable) | Sequential |

### 3.4 Management Review Templates

Extend existing `ManagementReview` model with:

```
ManagementReviewTemplate (uuid, SynaraEntity)
  ├── tenant          FK → Tenant
  ├── name            CharField(255)
  ├── description     TextField
  ├── sections        JSONField (list of section definitions)
  ├── is_default      BooleanField
  └── iso_clause      CharField — "9.3" by default

ManagementReview (existing, extend with):
  ├── template        FK → ManagementReviewTemplate (nullable — backward compat)
  ├── signatures      reverse FK from ElectronicSignature
  ├── auto_inputs     JSONField — aggregated data from QMS modules
```

**Template sections** (ISO 9001 §9.3.2 inputs):

| Section | Auto-populated From | Manual Input |
|---------|-------------------|--------------|
| Actions from previous reviews | Previous ManagementReview | Updates on status |
| Customer satisfaction | Feedback model | Trends, analysis |
| Quality objectives progress | HoshinProject KPIs | Commentary |
| Process performance | SPC data, VSM metrics | Analysis |
| Nonconformities & CAPAs | NCR/CAPA counts, closure rates | Root cause trends |
| Audit results | ComplianceReport | Findings discussion |
| Supplier performance | SupplierRecord scores | Trends |
| Resource adequacy | Employee/ResourceCommitment | Gaps, needs |
| Risks & opportunities | FMEA high-RPN items | Risk register updates |
| Improvement opportunities | A3 pipeline, Hoshin projects | Prioritization |

The auto-population endpoint aggregates data from across the QMS system into `auto_inputs` JSON. The template defines which sections are included and in what order. Users can customize templates per tenant.

### 3.5 Artifact Uploads

Wire existing `files/` app to QMS records via a junction model:

```
QMSAttachment (uuid, SynaraEntity)
  ├── file            FK → files.UploadedFile
  ├── document_type   CharField (ncr / capa / supplier_review / management_review / fmea / rca / a3)
  ├── document_id     UUIDField
  ├── category        CharField (evidence / photo / report / supporting_doc / corrective_action)
  ├── description     CharField(255)
  ├── uploaded_by     FK → User
  └── is_controlled   BooleanField — True if subject to document control (CFR)
```

**Why a junction model instead of FK on UploadedFile?** A single file might be attached to multiple documents (e.g., a photo of a defect attached to both the NCR and the FMEA). The junction model handles M2M cleanly without modifying the `files` app.

---

## 4. Implementation Phases

### Phase 1 — Standard & Schema (DoS) [March 2026]

**Goal:** Write the standard (NTF-001: Notification & Signature Standard), create the CR, define all models/endpoints/assertions.

**Tasks:**
1. Draft NTF-001 standard with machine-readable assertion hooks
   - Notification model assertions
   - ElectronicSignature model + CFR compliance assertions
   - Email response system assertions (extends QMS-002 ActionToken)
   - ManagementReviewTemplate assertions
   - QMSAttachment assertions
2. Define API surface for all new endpoints
3. Create CR linking to NTF-001 assertions
4. Risk assessment (4-agent: security analyst focus on CFR + token auth surface)

**Outputs:**
- `docs/standards/NTF-001.md` (v0.1)
- CR in `submitted` state with risk assessment
- Updated QMS-001 to reference NTF-001

**Estimated scope:** ~2 days (documentation + CR process)

### Phase 2 — Infrastructure & Schema Build [March 2026]

**Goal:** Models, migrations, base API endpoints. All tested against the standard.

**Tasks:**
1. Create models:
   - `Notification` (with indexes on user+is_read, tenant+created_at)
   - `ElectronicSignature` (inherits SynaraImmutableLog, with hash chain)
   - `ManagementReviewTemplate`
   - `QMSAttachment`
   - Extend `ActionToken.action_type` choices
2. Run migrations
3. Create API endpoints (CRUD + filtered list views)
4. Write compliance tests per NTF-001 assertions
5. Verify `run_compliance --standards` passes for NTF-001

**Outputs:**
- Migration file(s)
- API endpoints (JSON in/out per API-001)
- Test file: `syn/audit/tests/test_notification.py`
- Test file: `agents_api/ntf_tests.py`
- All NTF-001 assertions pass

**Estimated scope:** ~3-4 days

### Phase 3 — Notification System [March 2026]

**Goal:** Full in-app notification system with bell icon and real-time delivery.

**Tasks:**
1. SSE endpoint: `GET /api/notifications/stream/` — long-lived connection, pushes new notifications
2. Notification API:
   - `GET /api/notifications/` — paginated list (filter: is_read, category, severity)
   - `POST /api/notifications/<id>/read/` — mark as read
   - `POST /api/notifications/read-all/` — mark all as read
   - `GET /api/notifications/unread-count/` — badge count for bell
3. Bell icon UI component in `base_app.html`:
   - Dropdown panel showing recent notifications
   - Unread count badge
   - "Mark all read" action
   - Click notification → navigate to `link` URL
4. Notification triggers (emit from QMS lifecycle events):
   - NCR created/assigned/status-changed
   - CAPA status transitions
   - Signature requested/completed
   - Management Review scheduled/due
   - ActionItem overdue
   - High-RPN FMEA alert (>200, no linked RCA)
5. Secure API response surface for non-users:
   - `/action/<token>/` pages for notification responses
   - Scoped, no-auth, minimal UI

**Outputs:**
- SSE streaming endpoint
- Bell icon in all app pages
- Notification triggers wired to QMS events
- ActionToken response pages for non-users

**Estimated scope:** ~4-5 days

### Phase 4 — Email Notification & Response [March 2026]

**Goal:** Email delivery for all notification categories, with secure response links for non-users.

**Tasks:**
1. Email templates (HTML + plaintext):
   - Signature request (approve/reject via ActionToken link)
   - NCR acknowledgment
   - CAPA review request
   - Supplier CAPA notification
   - Management Review reminder
   - Weekly QMS digest (open NCRs, pending signatures, overdue actions)
2. Email sending infrastructure:
   - `syn.sched` task: `send_notification_email` — dequeues pending notifications where email is warranted
   - Rate limiting: max 50 emails/hour per tenant (prevent spam on bulk operations)
   - Unsubscribe: per-category opt-out stored on User model
3. Response pages (extend ActionToken):
   - Signature approval page: shows document summary, meaning statement, confirm button
   - Signature rejection page: requires comment
   - NCR acknowledgment page: simple confirm
   - CAPA review page: approve/return-for-revision with comment
4. Non-user response security:
   - Token validation (expired? used? scope match?)
   - CSRF on POST (even for token pages)
   - Rate limit on token page renders (prevent brute-force)
   - Log all token actions to SysLogEntry

**Outputs:**
- Email templates (6+)
- `syn.sched` email task
- Response pages for 4 action types
- Full audit trail on all email-triggered actions

**Estimated scope:** ~4-5 days

### Phase 5 — Frontend Integration [April 2026]

**Goal:** Wire everything into the existing templates. Management Review template builder. Artifact upload UI.

**Tasks:**
1. Signature workflow UI:
   - "Request Signatures" button on NCR/CAPA detail pages
   - Signature status panel (who signed, who pending, timestamps)
   - Signature history on document detail page
   - PDF generation with signature block
2. Management Review template UI:
   - Template builder: drag-and-drop section ordering, enable/disable auto-populate per section
   - Review creation from template: auto-populate + manual edit
   - Review approval workflow with electronic signatures
3. Artifact upload UI:
   - Drag-and-drop upload zone on NCR/CAPA/FMEA/RCA detail pages
   - File list with category tags, preview (images), download
   - Controlled document indicator for CFR-relevant files
4. QMS Dashboard enhancements:
   - Notification center (full page view)
   - Signature audit log (for quality managers)
   - Open items summary (NCRs pending signature, overdue CAPAs, upcoming reviews)

**Outputs:**
- Signature workflow integrated into NCR, CAPA, Supplier Review, Management Review pages
- Template builder for Management Reviews
- Artifact upload on all QMS record types
- Enhanced QMS dashboard

**Estimated scope:** ~5-7 days

---

## 5. Standards Impact

| Standard | Change |
|----------|--------|
| **NTF-001** (NEW) | Notification & Signature Standard — defines all new models, endpoints, CFR compliance requirements |
| **QMS-001** | Add §11: Notification Triggers, §12: Electronic Signatures. Update cross-module integration matrix. |
| **QMS-002** | Extend ActionToken action_type choices. Add email template specs for QMS notifications. |
| **SEC-001** | Add SSE endpoint to rate limiting rules. Document CFR signature auth requirements. |
| **DAT-001** | Document new models (Notification, ElectronicSignature, ManagementReviewTemplate, QMSAttachment). |
| **API-001** | Document SSE streaming pattern. Document notification API endpoints. |
| **CHG-001** | All phases create CRs per standard process. Phase 2 = migration type. Others = feature type. |

---

## 6. Security Considerations

### 6.1 CFR Part 11 Compliance

| Requirement | §11 Reference | Implementation |
|------------|---------------|----------------|
| Closed system controls | §11.10 | SynaraImmutableLog hash chain + audit trail |
| Signature manifestation | §11.50 | `meaning` field on ElectronicSignature |
| Signature linking | §11.70 | `document_type` + `document_id` on ElectronicSignature |
| Unique identification | §11.100 | User PK (authenticated) or Employee PK (non-user via token) |
| Electronic records integrity | §11.10(e) | Hash chain prevents tampering. `delete()` blocked. |
| Audit trail | §11.10(e) | SysLogEntry for every signature event |
| Authority checks | §11.10(g) | Signature requirements defined per document type |
| Re-authentication | §11.10(a) | `password_confirmed` field — user re-enters password before signing |

### 6.2 Token Security

- All ActionToken security requirements from QMS-002 §4.3 apply
- New action types follow same scoping rules
- Signature tokens have shorter expiry (24h vs 72h default) given the sensitivity
- Failed password confirmation on signature page → immediate token invalidation

### 6.3 SSE Endpoint

- Authenticated only (session cookie required)
- Per-user connection limit (max 3 concurrent SSE connections)
- Heartbeat every 30s to detect stale connections
- Auto-reconnect on client side (EventSource handles this natively)

---

## 7. Migration Strategy

| Phase | Migration Type | Risk |
|-------|---------------|------|
| Phase 2 | `CreateModel` × 4, `AlterField` × 1 (ActionToken choices) | Low — new tables only, no existing data modified |
| Phase 5 | None — frontend only | None |

**Rollback:** All new models are additive. Dropping tables + removing migration is clean rollback. No existing models modified.

---

## 8. Dependencies

```
Phase 1 (Standard) ─── no dependencies
Phase 2 (Schema)   ─── requires Phase 1 (standard defines what to build)
Phase 3 (Notif)    ─── requires Phase 2 (Notification model must exist)
Phase 4 (Email)    ─── requires Phase 2 + Phase 3 (email extends notifications)
Phase 5 (Frontend) ─── requires Phases 2-4 (UI wires to backend)
```

Phases 3 and 4 can partially overlap — email templates can be designed while notification system is built.

---

## 9. What This Unlocks

After all 5 phases:

1. **NCR → signature → closure** — quality manager opens NCR, assigns responsible party, responsible party acknowledges via email, corrective action uploaded as artifact, quality manager signs closure, management representative co-signs. Full audit trail.
2. **CAPA lifecycle** — NCR escalates to CAPA, CAPA owner assigned + notified, root cause analysis linked (RCA module), corrective action tracked (ActionItem), effectiveness review scheduled, electronic signatures at each gate. UUID chain from NCR → CAPA → RCA → Evidence → Hypothesis.
3. **Supplier CAPA** — internal quality issues supplier CAPA, supplier receives email with response link, supplier uploads corrective action plan as artifact, internal quality reviews and signs. No Svend account needed for supplier.
4. **Management Review** — quarterly review auto-populated with QMS metrics (NCR counts, CAPA closure rates, SPC trends, Hoshin progress, audit results). Template customizable per tenant. Electronic signatures on review approval. ISO 9001 §9.3 fully addressed.
5. **Non-user participation** — plant floor operators, suppliers, and external auditors interact with the QMS via secure email links. No account required. Every action logged and auditable.

This is the gap between Svend and Arena QMS / ETQ Reliance. They have document control and CAPA but charge $15-25K/yr. We'll have the same + integrated statistical engine + AI + Bayesian evidence at $299/mo.

---

*Plan authored 2026-03-04. Implementation begins after testing lockdown sprint completes.*
