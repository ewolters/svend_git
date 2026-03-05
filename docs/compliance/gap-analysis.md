# Gap Analysis — SOC 2 Readiness

**Last Updated:** 2026-03-05 (CHG-001 v1.6 lockdown + SOC 2 automation + security quick wins)
**Assessment Method:** Codebase audit + configuration review + automated compliance checks
**Assessed By:** Eric + Claude

---

## Summary

| Category | Met | Partial | Gap | Total |
|---|---|---|---|---|
| Security (CC1-CC9) | 13 | 11 | 1 | 25 |
| Availability (A1) | 1 | 2 | 0 | 3 |
| Processing Integrity (PI1) | 4 | 1 | 0 | 5 |
| Confidentiality (C1) | 1 | 2 | 0 | 3 |
| Privacy (P1) | 3 | 4 | 1 | 8 |
| **Total** | **22** | **20** | **2** | **44** |

**Bottom line:** Strong encryption, access control, and change management foundation. CHG-001 v1.6 flipped 3 controls to Met (CC1.5, CC3.4, CC6.7). Security quick wins (Argon2, account lockout, token expiry) implemented. Remaining gaps: MFA, audit logging, automated monitoring, vulnerability scanning pipeline. SOC 2 compliance score live on dashboard via `soc2_control_coverage()`.

---

## What's Working Well

### Encryption (Strong)
- Fernet field-level encryption for PII (Stripe IDs, chat messages)
- AES-256-CBC encrypted database backups with 30-day rotation
- HTTPS enforced with HSTS (2 years, preload)
- Session cookies: HTTPOnly, Secure, SameSite=Lax
- Encrypted file storage for user uploads
- Encryption key stored outside codebase (`~/.svend_encryption_key`)

### Access Control (Solid Foundation)
- Decorator-based RBAC: `@require_auth`, `@require_paid`, `@require_team`, `@require_enterprise`, `@require_org_admin`
- Multi-tenancy with org roles (owner/admin/member/viewer)
- Per-resource ownership validation on API endpoints — **IDOR audit completed during Synara migration (2026-03)**
- 7-day invite expiry on org invitations
- Guest access scoped via tokenized links

### Infrastructure Hardening (Good)
- Cloudflare Tunnel (no direct port exposure)
- fail2ban on SSH (3 retries, 24hr ban)
- UFW firewall
- systemd resource limits (4GB memory, 200% CPU)
- Gunicorn worker recycling (1000 requests + jitter)
- Caddy security headers (HSTS, X-Frame-Options: DENY, nosniff, Referrer-Policy)

### Data Handling (Adequate)
- PII minimized to auth + billing essentials
- Analytics uses hashed IPs (SHA-256)
- Password validation (length, complexity, common password check)
- CSRF protection on all views (fixed 2026-02-22)

### Change Management (CHG-001 v1.6 — Strong)
- ChangeRequest model with full lifecycle (draft → completed) and escalating field gates
- 3-layer enforcement: model `clean()`, API `validate_for_transition()`, daily compliance check
- Multi-agent risk assessment (4 roles, 5 dimensions) for features/migrations
- Immutable ChangeLog chain with UUID traceability to commits, log.md, compliance checks
- 14 automated checks covering field completeness, risk assessment gates, commit linkage
- Emergency change detection with 24h retroactive RA enforcement
- SOC 2 compliance automation: `soc2_control_coverage()` maps all 52 controls to checks, live on dashboard

### Security Quick Wins (2026-03-05)
- Argon2id password hashing (primary, PBKDF2 fallback for existing hashes)
- Email verification token 24-hour expiry
- Account lockout: 5 failed attempts → 15 minute lockout (per-username, complements IP throttle)
- Explicit X_FRAME_OPTIONS = DENY

### Synara Migration Security Gains (2026-03)
- **IDOR fixes:** Per-resource ownership validated across all Synara endpoints (belief graphs, hypotheses, evidence)
- **Persistence safety:** Transaction-scoped saves with proper FK cascade; no orphaned data on partial failures
- **Cache bounds:** Synara graph cache bounded (max 1000 entries, 30-min TTL); prevents unbounded memory growth
- **Prompt injection hardening:** LLM interface sandboxes user content with structured system/user message separation; DSL inputs never passed raw to LLM
- **DSL safety:** Hypothesis DSL parser validates syntax before evaluation; rejects malformed/injection-like inputs; no `eval()` or `exec()`

---

## Critical Gaps (Must Fix Before Audit)

### GAP-01: No Multi-Factor Authentication
- **TSC:** CC6.2
- **Current:** Email/password only
- **Required:** MFA for privileged accounts at minimum, ideally all users
- **Impact:** Auditor will flag as material weakness
- **Effort:** Medium — TOTP (pyotp) integration, ~2 days
- **DEBT ref:** Not yet tracked

### GAP-02: No Application Audit Trail
- **TSC:** CC2.1, CC4.1, CC7.3
- **Current:** File-based logs (svend.log, access.log) with rotation. No structured audit trail of who-did-what-when.
- **Required:** Immutable audit log of security-relevant events: login/logout, permission changes, data access, admin actions
- **Impact:** Auditor will ask "show me who accessed customer X's data on date Y" — we can't answer that
- **Effort:** Medium — AuditLog model + middleware, ~2-3 days
- **DEBT ref:** Not yet tracked

### GAP-03: No Automated Vulnerability Scanning
- **TSC:** CC7.1
- **Current:** Manual security review; DEBT.md tracking
- **Required:** Automated dependency scanning (Dependabot/Snyk), ideally SAST
- **Impact:** Auditor expects evidence of regular scanning
- **Effort:** Low — GitHub Dependabot alerts are free, ~1 hour to enable
- **DEBT ref:** Not yet tracked

### GAP-04: No Monitoring/Alerting System
- **TSC:** CC4.1, CC7.2, A1.1
- **Current:** Logs exist but nobody watches them in real-time
- **Required:** Alerting on: service down, error rate spike, failed login surge, resource exhaustion
- **Impact:** "How do you know when something breaks?" — currently: we don't, proactively
- **Effort:** Medium — UptimeRobot/Healthchecks.io for uptime; simple log alerting script, ~1-2 days
- **DEBT ref:** Not yet tracked

### GAP-05: No Self-Service Data Export
- **TSC:** P1.8
- **Current:** Users can view their data in-app but cannot export all personal data
- **Required:** GDPR-style data portability (also good for SOC 2 Privacy criteria)
- **Effort:** Low-Medium — JSON export endpoint for user data, ~1 day
- **DEBT ref:** Not yet tracked

---

## Partial Controls (Need Strengthening)

### PAR-01: Session Management — **FIXED**
- **TSC:** CC6.2, CC6.6
- **Status:** SESSION_COOKIE_AGE=28800 (8hr) already in `settings.py`. Session invalidation on password change: still pending (REM-14).

### PAR-02: Password Hashing — **FIXED**
- **TSC:** CC6.2
- **Status:** Argon2id primary hasher in `settings.py`. Existing PBKDF2 hashes auto-upgrade on next login.

### PAR-03: Account Lockout — **FIXED**
- **TSC:** CC6.2
- **Status:** `LoginAttempt` model in `accounts/models.py`. 5 failed attempts → 15 min lockout. Per-username + per-IP throttle (DRF).

### PAR-04: Email Token Expiry — **FIXED**
- **TSC:** CC6.2
- **Status:** 24-hour expiry implemented in `accounts/models.py` via `email_verification_token_sent_at` field.

### PAR-05: Guest Token Security
- **TSC:** CC6.2
- **Current:** Guest tokens stored plaintext
- **Fix:** Hash guest tokens like verification tokens (SHA-256)
- **Effort:** Low
- **DEBT ref:** P1 M18

### PAR-06: Silent Decryption Failure
- **TSC:** C1.2
- **Current:** Failed decryption returns ciphertext as plaintext (fallback behavior)
- **Fix:** Raise exception on decryption failure; never return raw ciphertext
- **Effort:** Trivial — change fallback behavior in `encryption.py`
- **DEBT ref:** P1 H6

### PAR-07: Stripe Webhook Idempotency
- **TSC:** PI1.3
- **Current:** Stripe events can be replayed; no deduplication
- **Fix:** Store processed event IDs; skip duplicates
- **Effort:** Low — event ID check before processing
- **DEBT ref:** P1 H7

### PAR-08: Content Security Policy
- **TSC:** CC5.2
- **Current:** CSP allows `unsafe-inline` and `unsafe-eval` (no XSS protection)
- **Fix:** Move inline scripts to files; use nonces; remove `unsafe-eval`
- **Effort:** High — major frontend refactor (inline JS/CSS is core architecture)
- **DEBT ref:** P2 L14

### PAR-09: Off-Site Backups
- **TSC:** A1.3
- **Current:** Encrypted backups on same machine only
- **Fix:** Push encrypted backups to Backblaze B2 or S3
- **Effort:** Low — `b2` CLI sync in backup script, ~2 hours
- **DEBT ref:** P3 (tracked)

### PAR-10: Change Management Formalization
- **TSC:** CC8.1
- **Current:** Manual git + log.md process; no CI/CD; no staging; no automated testing
- **Fix (phased):**
  1. Pre-commit hooks (linting, basic checks) — Low effort
  2. GitHub Actions CI (test suite) — Medium effort
  3. Staging environment — Medium effort
- **DEBT ref:** Not formally tracked

### PAR-11: Formal Risk Register
- **TSC:** CC3.2
- **Current:** DEBT.md tracks security items but isn't a formal risk register
- **Fix:** Create risk register with likelihood/impact scoring in `policies/risk-assessment.md`
- **Effort:** Low — documentation task
- **DEBT ref:** N/A

### PAR-12: Vendor Assessment Process
- **TSC:** CC9.1, CC9.2
- **Current:** Vendors identified but no formal assessment or review cadence
- **Fix:** Document vendor SOC 2/PCI status; establish annual review
- **Effort:** Low — documentation task
- **DEBT ref:** N/A

### PAR-13: Data Retention Policy
- **TSC:** C1.3, P1.5, P1.6
- **Current:** Backup rotation (30 days) and purge script exist but no formal retention schedule
- **Fix:** Define retention periods by data type; document disposal procedures
- **Effort:** Low — documentation + minor code for complete account deletion
- **DEBT ref:** Not tracked

### PAR-14: Privacy Policy Alignment
- **TSC:** P1.1, P1.2
- **Current:** Privacy policy exists on landing page but not reviewed for SOC 2 Privacy criteria
- **Fix:** Review and update privacy policy; add granular consent options
- **Effort:** Low — legal review + template update
- **DEBT ref:** Not tracked

---

## Prioritized Remediation Order

**Phase 1 — Quick Wins (1-2 days)**
1. PAR-02: Argon2 password hashing
2. PAR-04: Email token expiry
3. PAR-06: Fix silent decryption failure
4. PAR-07: Webhook idempotency
5. PAR-01: Session timeout settings
6. PAR-03: Account lockout
7. PAR-05: Hash guest tokens
8. GAP-03: Enable Dependabot

**Phase 2 — Core Infrastructure (1-2 weeks)**
9. GAP-01: MFA (TOTP)
10. GAP-02: Application audit trail
11. GAP-04: Monitoring + alerting
12. PAR-09: Off-site backups
13. PAR-10: Pre-commit hooks + basic CI

**Phase 3 — Formalization (1 week)**
14. PAR-11: Formal risk register
15. PAR-12: Vendor assessments
16. PAR-13: Data retention policy
17. PAR-14: Privacy policy update
18. GAP-05: Self-service data export

**Phase 4 — Hardening (ongoing)**
19. PAR-08: CSP hardening (major effort, can be phased)
20. PAR-10: Staging environment + full CI/CD
21. Tabletop incident response exercise
22. Penetration test (pre-audit)
