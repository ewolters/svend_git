# Remediation Tracker

**Last Updated:** 2026-03-05 (SOC 2 quick wins + CHG-001 lockdown)
**Purpose:** Track every gap and partial control from gap-analysis.md through to resolution

Status: **Open** | **In Progress** | **Done** | **Accepted** (risk accepted, won't fix)

---

## Phase 1 — Quick Wins (Target: 1-2 days)

| ID | Gap | Action | Owner | DEBT Ref | Target Date | Status |
|---|---|---|---|---|---|---|
| REM-01 | PAR-02: PBKDF2 password hashing | ~~Install argon2-cffi~~ | Eric | P1 M2 | 2026-03-05 | **Done** |
| REM-02 | PAR-04: Email tokens never expire | ~~Add 24h expiry check~~ | Eric | P1 M7 | 2026-03-05 | **Done** |
| REM-03 | PAR-06: Silent decryption failure | Change encryption.py fallback to raise exception instead of returning ciphertext | Eric | P1 H6 | TBD | Open |
| REM-04 | PAR-07: Webhook replay | Store processed Stripe event IDs; skip duplicates in webhook handler | Eric | P1 H7 | TBD | Open |
| REM-05 | PAR-01: Session timeout | ~~SESSION_COOKIE_AGE=28800 already set~~ | Eric | P1 M1 | 2026-03-05 | **Done** |
| REM-06 | PAR-03: Account lockout | ~~LoginAttempt model + login view integration~~ | Eric | P1 H3 | 2026-03-05 | **Done** |
| REM-07 | PAR-05: Guest token hashing | Hash guest tokens with SHA-256 (like verification tokens) | Eric | P1 M18 | TBD | Open |
| REM-08 | GAP-03: Dependency scanning | Enable GitHub Dependabot alerts on kjerne repo | Eric | -- | TBD | Open |

## Phase 2 — Core Infrastructure (Target: 1-2 weeks)

| ID | Gap | Action | Owner | DEBT Ref | Target Date | Status |
|---|---|---|---|---|---|---|
| REM-09 | GAP-01: No MFA | Implement TOTP via pyotp; mandatory for enterprise, optional for others; add backup codes | Eric | -- | TBD | Open |
| REM-10 | GAP-02: No audit trail | Create AuditLog model; add middleware for login/logout/data access/admin actions; immutable append-only | Eric | -- | TBD | Open |
| REM-11 | GAP-04: No monitoring | Set up UptimeRobot or Healthchecks.io for uptime; add error rate alerting; failed login surge detection | Eric | -- | TBD | Open |
| REM-12 | PAR-09: Backups same machine | Add Backblaze B2 sync to backup_db.sh; verify restore from B2 | Eric | P3 | TBD | Open |
| REM-13 | PAR-10: No CI (phase 1) | Add pre-commit hooks (ruff linting, type checks); basic GitHub Actions workflow | Eric | -- | TBD | Open |
| REM-14 | PAR-01 (ext): Session invalidation | Invalidate all sessions on password change | Eric | P1 M3 | TBD | Open |

## Phase 3 — Formalization (Target: 1 week)

| ID | Gap | Action | Owner | DEBT Ref | Target Date | Status |
|---|---|---|---|---|---|---|
| REM-15 | PAR-11: No formal risk register | Risk register created in risk-assessment.md — conduct first quarterly review | Eric | -- | TBD | Open |
| REM-16 | PAR-12: No vendor assessments | Obtain Anthropic SOC 2 report; complete first formal vendor review for all critical vendors | Eric | -- | TBD | Open |
| REM-17 | PAR-13: No retention schedule | Define retention periods by data type; implement account deletion with full data purge | Eric | -- | TBD | Open |
| REM-18 | PAR-14: Privacy policy misaligned | Review and update privacy policy against SOC 2 Privacy criteria; add granular consent | Eric | -- | TBD | Open |
| REM-19 | GAP-05: No data export | Build JSON export endpoint for user's own data (email, profile, projects, analyses, files) | Eric | -- | TBD | Open |

## Phase 4 — Hardening and Testing (Target: ongoing)

| ID | Gap | Action | Owner | DEBT Ref | Target Date | Status |
|---|---|---|---|---|---|---|
| REM-20 | PAR-08: CSP gaps | Phase 1: Audit inline JS/CSS usage. Phase 2: Extract to files. Phase 3: Add nonces, remove unsafe-* | Eric | P2 L14 | TBD | Open |
| REM-21 | PAR-10: No staging (phase 2) | Set up staging environment (separate DB, separate systemd service, same server or VPS) | Eric | -- | TBD | Open |
| REM-22 | Pre-audit 6.4: No tabletop | Conduct tabletop incident response exercise; document results | Eric | -- | TBD | Open |
| REM-23 | Pre-audit 7.3: Backup restore untested | Perform backup restoration test; document procedure and results | Eric | -- | TBD | Open |
| REM-24 | Pre-audit 7.6: No DR drill | Conduct annual disaster recovery drill; document results | Eric | -- | TBD | Open |
| REM-25 | Encryption key rotation untested | Perform key rotation test on non-production data; document procedure | Eric | -- | TBD | Open |
| REM-26 | Penetration test | Engage for pre-audit pen test (or self-conduct structured security test) | Eric | -- | TBD | Open |

---

## Completed Remediations

| ID | Gap | Action Taken | Completed | Evidence |
|---|---|---|---|---|
| REM-01 | PAR-02: Argon2 password hashing | Installed argon2-cffi; `PASSWORD_HASHERS` set to Argon2id primary with PBKDF2 fallback in `settings.py` | 2026-03-05 | `svend/settings.py` PASSWORD_HASHERS |
| REM-02 | PAR-04: Email token expiry | Added `email_verification_token_sent_at` field; 24h expiry check in `verify_email()` | 2026-03-05 | `accounts/models.py` verify_email() |
| REM-05 | PAR-01: Session timeout | SESSION_COOKIE_AGE=28800 (8hr) already in place; confirmed working | 2026-03-05 | `svend/settings.py` line 149 |
| REM-06 | PAR-03: Account lockout | `LoginAttempt` model with `is_locked_out()`, `record()`, `clear_on_success()`; integrated into login view | 2026-03-05 | `accounts/models.py`, `api/views.py` login() |
| CHG-01 | CC1.5/CC3.4/CC6.7/CC8.1: Change management gaps | CHG-001 v1.6 lockdown: ChangeRequest model validation, API transition gates, 14-check compliance check, risk assessment enforcement for all code types, 76 CRs backfilled | 2026-03-05 | `syn/audit/models.py`, `syn/audit/compliance.py`, `api/internal_views.py` |
| SOC-01 | SOC 2 compliance automation | `SOC2_CONTROL_MATRIX` (52 controls), `soc2_control_coverage()` function, dashboard with Pareto charts and TSC breakdowns | 2026-03-05 | `syn/audit/compliance.py`, `templates/internal_dashboard.html` |
| SYN-01 | CC6.4/CC6.8: IDOR vulnerabilities | Synara migration: per-resource ownership validation on all belief graph, hypothesis, and evidence endpoints | 2026-03-03 | `synara_views.py` — all queries filter by `user=request.user` or tenant membership |
| SYN-02 | PI1.3: Persistence safety | Synara migration: transaction-scoped saves with proper FK cascade; no orphaned data on partial failures | 2026-03-03 | `synara_views.py`, `synara/synara.py` — atomic operations on graph mutations |
| SYN-03 | A1.1: Unbounded cache growth | Synara migration: graph cache bounded to max 1000 entries with 30-min TTL | 2026-03-03 | `synara/` — LRU cache with explicit bounds |
| SYN-04 | CC3.3/CC6.8: Prompt injection | Synara migration: LLM interface sandboxes user content with structured system/user message separation; user input never concatenated into system prompts | 2026-03-03 | `synara/llm_interface.py` — structured message protocol |
| SYN-05 | PI1.1/CC6.8: DSL injection | Synara migration: hypothesis DSL parser validates syntax before evaluation; rejects malformed inputs; no `eval()` or `exec()` | 2026-03-03 | `synara/dsl.py` — whitelist-based parser |

---

## Cross-Reference: Pre-Audit Failures → Remediation Items

| Pre-Audit Item | Failure | Remediation |
|---|---|---|
| 2.2 MFA | Not implemented | REM-09 |
| 2.3 Session timeout | 14-day default | REM-05 |
| 2.4 Account lockout | Not implemented | REM-06 |
| 2.8 Offboarding/data purge | No complete purge | REM-17, REM-19 |
| 2.9 Access reviews | No formal process | REM-15 |
| 3.6 Key rotation tested | Never performed | REM-25 |
| 4.4 Audit trail | Not implemented | REM-10 |
| 4.6 Monitoring/alerting | Not implemented | REM-11 |
| 4.8 Admin audit trail | Not implemented | REM-10 |
| 5.6 Automated testing/CI | No CI/CD | REM-13 |
| 5.7 Staging environment | Production only | REM-21 |
| 5.8 Dependency scanning | Not enabled | REM-08 |
| 6.4 Tabletop exercise | Not conducted | REM-22 |
| 7.3 Backup restore test | Not tested | REM-23 |
| 7.4 Off-site backup | Same machine | REM-12 |
| 7.6 DR drill | Not conducted | REM-24 |
| 8.3 Vendor assessments | Not performed | REM-16 |
| 9.2 Privacy policy aligned | Needs review | REM-18 |
| 9.3 Retention schedule | Not defined | REM-17 |
| 9.4 Data export | Not implemented | REM-19 |
