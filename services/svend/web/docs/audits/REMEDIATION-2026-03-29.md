# Remediation Plan — QMS Platform Audit 2026-03-29

**Based on:** [AUDIT-2026-03-29.md](AUDIT-2026-03-29.md)
**Approach:** Phased by severity and blast radius. Each phase is a deployable increment.

---

## Phase 0 — STOP THE BLEEDING (Critical security + data integrity)
**Target: Immediate — before next deploy**
**Effort: ~4 hours**

| # | Finding | Action | Files | Est |
|---|---------|--------|-------|-----|
| 1 | CRIT-04: Unauthenticated `/app/*` | Add `login_required` wrapper to all `app/` TemplateViews in `svend/urls.py`. Consider a blanket middleware for `/app/` prefix. | `svend/urls.py` | 30m |
| 2 | CRIT-05: Shared conversation data leak | Remove `reasoning_trace` and `tool_calls` from `shared_conversation` response. Add `expires_at` field to `SharedConversation` model, check on access. | `chat/views.py`, `chat/models.py` | 45m |
| 3 | HIGH-01: Webhook cross-tenant dispatch | Add `tenant` FK to `WebhookEndpoint` (if missing), pass `tenant_id` into `dispatch_event()`, filter endpoints by tenant. | `notifications/webhook_delivery.py`, `notifications/models.py` | 45m |
| 4 | HIGH-02: Safety FMEA cross-tenant injection | Add tenant filter to FMEA lookup in `process_card`: `get_object_or_404(FMEA, id=fmea_id, owner__memberships__tenant=tenant)` | `safety/views.py:444` | 15m |
| 5 | CRIT-01/02/03: Race conditions | Convert `UsageLog.log_request`, `UserQuota.add_file/remove_file`, and `Board.save` to use `F()` expressions and/or `select_for_update()`. | `chat/models.py`, `files/models.py`, `agents_api/models.py` | 90m |

**Verification:** Run full test suite. Manual smoke test of shared conversations (verify no traces), webhook dispatch (verify tenant filtering), file quota (verify atomic increment).

---

## Phase 1 — AUTH & AUTHORIZATION HARDENING
**Target: Within 1 week**
**Effort: ~8 hours**

| # | Finding | Action | Files | Est |
|---|---------|--------|-------|-----|
| 6 | HIGH-05: Tier mapping never matches | Update `get_or_create_for_user` tier keys to match `Tier` enum (`"FREE"`, `"FOUNDER"`, `"PRO"`, `"TEAM"`, `"ENTERPRISE"`). | `files/models.py:248-272` | 30m |
| 7 | HIGH-06: `subscription_tier` attribute miss | Change `getattr(user, "subscription_tier", "FREE")` to `getattr(user, "tier", "FREE")` | `agents_api/models.py:311` | 10m |
| 8 | HIGH-09: CSRF exempt on session endpoints | Audit all 30+ `@csrf_exempt` decorators. Remove from endpoints that accept session auth (safety, loop, notifications). Keep only for true API-key-only endpoints. | `safety/views.py`, `loop/views.py`, `notifications/views.py` | 2h |
| 9 | HIGH-10: File `is_public` bypass | Remove `is_public` from the generic PATCH handler. Only settable through `create_share_link`. | `files/views.py:234-244` | 20m |
| 10 | HIGH-07: Forge API key brute-force | Add IP-based throttle on failed Forge API key lookups. Consider consolidating with `accounts.APIKey`. | `forge/views.py:33-74` | 1h |
| 11 | MED-10: Webhook secret plaintext | Migrate `WebhookEndpoint.secret` to `EncryptedCharField`. | `notifications/models.py:175` | 45m |
| 12 | MED-18: Exception details in API | Replace `str(e)[:200]` with generic error message. Log original exception server-side. | `agents_api/views.py:68` | 15m |
| 13 | MED-21: Auditor token exposure | Return full token only on POST (creation). GET list returns prefix only. | `loop/views.py:2118-2138` | 30m |
| 14 | LOW-16: Case-sensitive email uniqueness | Use `email__iexact` in registration check. Normalize to lowercase on save. | `api/views.py:970` | 15m |

**Verification:** Test each auth change with both authenticated and unauthenticated requests. Verify Forge rate limiting triggers. Confirm tier-gated quotas now apply correctly.

---

## Phase 2 — DATA INTEGRITY & MODEL CLEANUP
**Target: Within 2 weeks**
**Effort: ~6 hours**

| # | Finding | Action | Files | Est |
|---|---------|--------|-------|-----|
| 15 | HIGH-08: Duplicate graph edges | Add `UniqueConstraint(fields=["graph", "source", "target", "relation_type"])` to `ProcessEdge`. Migration + data cleanup. | `graph/models.py` | 45m |
| 16 | HIGH-12: EdgeEvidence CASCADE | Change `on_delete=CASCADE` to `on_delete=PROTECT`. Add edge soft-delete mechanism. | `graph/models.py:385-389` | 1h |
| 17 | MED-01: CAPA audit trail stale values | Move `_log_field_changes()` before the `setattr` loop in status transition. | `agents_api/capa_views.py:230-290` | 30m |
| 18 | MED-02: CAPA hard delete | Add `is_deleted` BooleanField + filter in querysets. Remove or repurpose DELETE endpoint. | `agents_api/capa_views.py`, `agents_api/models.py` | 1h |
| 19 | MED-09: ModelVersion activate race | Wrap in `transaction.atomic()` with `select_for_update()`. | `chat/models.py:61-74` | 30m |
| 20 | MED-11: Forge Job ownerless | Add `CheckConstraint(check=Q(api_key__isnull=False) | Q(user__isnull=False))` | `forge/models.py:62-70` | 20m |
| 21 | MED-12: CharField PKs | Migrate `DSWResult.id` and `TriageResult.id` to UUIDField. | `agents_api/models.py` | 45m |
| 22 | MED-13: TextField for JSON | Migrate `Workflow.steps`, `SavedModel.metrics`, `SavedModel.feature_names` to JSONField. | `agents_api/models.py` | 30m |
| 23 | MED-14: Cross-graph edge validation | Add `clean()` to `ProcessEdge` validating source/target graph membership. | `graph/models.py:244-255` | 20m |
| 24 | LOW-14: Naive datetimes in workbench | Replace `datetime.now()` with `timezone.now()` in workbench models. | `workbench/models.py` | 15m |

**Verification:** Run migrations on staging. Verify graph constraints with duplicate edge test. Confirm CAPA soft-delete preserves audit trail.

---

## Phase 3 — TENANT ISOLATION FOR ENTERPRISE
**Target: Within 3 weeks (before any enterprise onboarding)**
**Effort: ~12 hours**

| # | Finding | Action | Files | Est |
|---|---------|--------|-------|-----|
| 25 | HIGH-03/04: Tenant FK gaps | Add `tenant` FK to: `Workbench`, `KnowledgeGraph`, `EpistemicLog`, `Report`, `IshikawaDiagram`, `CEMatrix`, `ValueStreamMap`, `PlantSimulation`, `ManagementReview`, `SupplierRecord`. Add tenant-scoped managers. | Multiple models.py files | 4h |
| 26 | MED-05: Enable tenant isolation | Set `TENANT_ISOLATION_ENABLED = True`. Test all views for proper tenant filtering. | `svend/settings.py`, all views | 4h |
| 27 | MED-06: Tenant isolation test suite | Write cross-tenant access tests for every app: create data as Tenant A, verify Tenant B cannot read/write/delete. | New test files | 4h |

**Verification:** Full test suite with `TENANT_ISOLATION_ENABLED = True`. Cross-tenant penetration testing on staging.

---

## Phase 4 — TEST COVERAGE RECOVERY
**Target: Within 4-6 weeks**
**Effort: ~40 hours**

### P0 — Core QMS (loop app)
| # | Action | Target Coverage | Est |
|---|--------|----------------|-----|
| 28 | Write tests for loop models (Signal, Commitment, ModeTransition, ProcessConfirmation, ForcedFailureTest) | 80%+ | 6h |
| 29 | Write tests for loop views (all CRUD endpoints, status transitions, policy management) | 70%+ | 8h |
| 30 | Write tests for Auditor Portal token auth (creation, validation, expiry, access control) | 90%+ | 3h |
| 31 | Write tests for loop/services.py `fulfill_commitment()` cross-app workflow | 80%+ | 4h |

### P1 — Fix broken tests
| # | Action | Est |
|---|--------|-----|
| 32 | Rewrite `agents_api/tests.py` — replace 7 `skipTest()` escape hatches with proper test setup | 3h |
| 33 | Rewrite `forge/tests.py` — replace 3 `skipTest()` calls, test auth integration | 2h |
| 34 | Write `files/tests.py` — replace 3-line stub with encryption, quota, upload tests | 3h |

### P2 — Coverage expansion
| # | Action | Est |
|---|--------|-----|
| 35 | Hoshin views (2,094L, ~13% -> 60%) | 4h |
| 36 | Learn views (2,450L, ~24% -> 60%) | 4h |
| 37 | Migrate 39 `_make_user` copies to use `conftest.make_user()` | 3h |

---

## Phase 5 — INFRASTRUCTURE & POLISH
**Target: Within 6-8 weeks**
**Effort: ~8 hours**

| # | Finding | Action | Est |
|---|---------|--------|-----|
| 38 | MED-04: Readiness recurrence score | Default to 0 instead of 8. Surface as "not yet measured" in UI. | 30m |
| 39 | MED-08: LLM cost controls | Add daily token budget per user. Track cumulative input+output tokens. Reject over budget. | 3h |
| 40 | MED-07: CSP drift | Remove CSP header from Caddy, let Django middleware handle it exclusively. | 30m |
| 41 | MED-06: CSP unsafe-eval | Audit JS deps for eval() usage. Remove if possible. | 1h |
| 42 | LOW-02: CONN_MAX_AGE | Add `DATABASES["default"]["CONN_MAX_AGE"] = 60` | 10m |
| 43 | LOW-03: Stripe idempotency | Add `idempotency_key` to `Customer.create()` and `Session.create()` | 20m |
| 44 | LOW-07: Gunicorn workers | Evaluate switch to `gthread` worker class for I/O-bound LLM calls | 1h |
| 45 | LOW-17: Per-account login throttle | Integrate `LoginAttempt` model into login view for per-account lockout | 1h |
| 46 | MED-17: LoginAttempt cleanup | Add Tempora periodic task to purge records older than 90 days | 30m |

---

## Tracking

| Phase | Findings Addressed | Effort | Deadline |
|-------|-------------------|--------|----------|
| **Phase 0** | 5 CRITICAL, 2 HIGH | ~4h | Before next deploy |
| **Phase 1** | 7 HIGH, 4 MEDIUM | ~8h | 2026-04-05 |
| **Phase 2** | 2 HIGH, 8 MEDIUM, 1 LOW | ~6h | 2026-04-12 |
| **Phase 3** | 2 HIGH, 1 MEDIUM | ~12h | 2026-04-19 |
| **Phase 4** | 1 HIGH, 1 MEDIUM (+ coverage) | ~40h | 2026-05-10 |
| **Phase 5** | 4 MEDIUM, 5 LOW | ~8h | 2026-05-24 |
| **TOTAL** | **59 findings** | **~78h** | 8 weeks |

*6 LOW findings (LOW-01, LOW-04, LOW-08, LOW-09, LOW-11, LOW-15) deferred as non-impactful. Address opportunistically.*

---

*Plan generated from AUDIT-2026-03-29.md. Review with engineering before executing Phase 0.*
