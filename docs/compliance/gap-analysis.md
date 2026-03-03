# Gap Analysis — SOC 2 Readiness

**Last Updated:** 2026-03-03
**Assessment Method:** Codebase audit + configuration review
**Assessed By:** Eric + Claude

---

## Summary

| Category | Met | Partial | Gap | Total |
|---|---|---|---|---|
| Security (CC1-CC9) | 10 | 14 | 1 | 25 |
| Availability (A1) | 1 | 2 | 0 | 3 |
| Processing Integrity (PI1) | 4 | 1 | 0 | 5 |
| Confidentiality (C1) | 1 | 2 | 0 | 3 |
| Privacy (P1) | 3 | 4 | 1 | 8 |
| **Total** | **19** | **23** | **2** | **44** |

**Bottom line:** Strong encryption and access control foundation. Critical gaps in MFA, audit logging, automated monitoring, and vulnerability scanning. Most "Partial" items need procedural formalization rather than new technology.

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
- Per-resource ownership validation on API endpoints
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

### PAR-01: Session Management
- **TSC:** CC6.2, CC6.6
- **Current:** Django default 14-day session; no idle timeout; no invalidation on password change
- **Fix:** Set SESSION_COOKIE_AGE to 8 hours; implement idle timeout; invalidate sessions on password change
- **Effort:** Low — settings change + small code addition
- **DEBT ref:** P1 M1, P1 M3

### PAR-02: Password Hashing
- **TSC:** CC6.2
- **Current:** PBKDF2-SHA256 (Django default)
- **Fix:** Migrate to Argon2id (GPU-resistant)
- **Effort:** Low — `pip install argon2-cffi`, update PASSWORD_HASHERS
- **DEBT ref:** P1 M2

### PAR-03: Account Lockout
- **TSC:** CC6.2
- **Current:** No lockout after failed login attempts; unlimited brute-force possible
- **Fix:** Rate-limit login attempts per IP/account; lock after N failures
- **Effort:** Low — django-axes or custom middleware, ~half day
- **DEBT ref:** P1 H3

### PAR-04: Email Token Expiry
- **TSC:** CC6.2
- **Current:** Email verification tokens never expire
- **Fix:** Add 48-hour expiry check
- **Effort:** Trivial — timestamp comparison in verify function
- **DEBT ref:** P1 M7

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
