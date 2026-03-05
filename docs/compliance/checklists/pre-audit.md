# Pre-Audit Readiness Checklist

**Last Updated:** 2026-03-05
**Purpose:** Verify all SOC 2 audit requirements are met before engaging an auditor

Status key: PASS / FAIL / N/A

---

## 1. Governance and Policies

| # | Item | Status | Evidence | Notes |
|---|---|---|---|---|
| 1.1 | Information Security Policy documented and approved | PASS | `policies/information-security.md` | v1.0 effective 2026-03-03 |
| 1.2 | Access Control Policy documented | PASS | `policies/access-control.md` | |
| 1.3 | Change Management Policy documented | PASS | `policies/change-management.md` | |
| 1.4 | Incident Response Plan documented | PASS | `policies/incident-response.md` | |
| 1.5 | Risk Assessment Policy with risk register | PASS | `policies/risk-assessment.md` | 15 active risks scored |
| 1.6 | Vendor Management Policy with inventory | PASS | `policies/vendor-management.md` | 5 vendors inventoried |
| 1.7 | Data Classification Policy documented | PASS | `policies/data-classification.md` | 4 levels defined |
| 1.8 | BCDR Plan documented | PASS | `policies/bcdr.md` | RTO/RPO defined |
| 1.9 | Acceptable Use Policy with acknowledgment | PASS | `policies/acceptable-use.md` | Signed by founder |
| 1.10 | Encryption Policy documented | PASS | `policies/encryption.md` | |
| 1.11 | All policies reviewed within last 12 months | PASS | Policy headers | All created 2026-03-03 |
| 1.12 | Policy exceptions documented with justification | PASS | Each policy has exceptions section | |

## 2. Access Control

| # | Item | Status | Evidence | Notes |
|---|---|---|---|---|
| 2.1 | Password policy enforced (length, complexity) | PASS | `settings.py` AUTH_PASSWORD_VALIDATORS | 4 validators active |
| 2.2 | MFA enabled for privileged accounts | **FAIL** | -- | Not implemented (GAP-01) |
| 2.3 | Session timeout configured | PASS | `settings.py` SESSION_COOKIE_AGE=28800 | 8 hours (SOC 2 CC6.1/CC6.6) |
| 2.4 | Account lockout after failed attempts | PASS | `accounts/models.py` LoginAttempt | 5 failures → 15 min lockout |
| 2.5 | RBAC enforced across all endpoints | PASS | `permissions.py` decorators | Tier + role-based |
| 2.6 | Least privilege for infrastructure access | PASS | SSH key-only; localhost DB | Single admin |
| 2.7 | User onboarding requires email verification | PASS | `accounts/models.py` | SHA-256 hashed tokens |
| 2.8 | User offboarding/deprovisioning process | **FAIL** | -- | No complete data purge (GAP-05) |
| 2.9 | Access reviews performed quarterly | **FAIL** | -- | No formal review process |
| 2.10 | Third-party access documented | PASS | `policies/vendor-management.md` | No standing vendor access |

## 3. Encryption

| # | Item | Status | Evidence | Notes |
|---|---|---|---|---|
| 3.1 | TLS 1.2+ enforced for all external connections | PASS | Caddy + Cloudflare | HSTS preload enabled |
| 3.2 | PII encrypted at rest | PASS | `core/encryption.py` | Fernet fields |
| 3.3 | Backups encrypted | PASS | `backup_db.sh` | AES-256-CBC |
| 3.4 | Encryption keys stored securely | PASS | `~/.svend_encryption_key` | Owner-read only |
| 3.5 | Key rotation procedure documented | PASS | `policies/encryption.md` sec 4.2 | Not yet tested |
| 3.6 | Key rotation tested | **FAIL** | -- | Never performed |
| 3.7 | No secrets in source code | PASS | `.gitignore`, env vars | Verified in codebase audit |
| 3.8 | Password hashing uses strong algorithm | PASS | Argon2id primary | `settings.py` PASSWORD_HASHERS; PBKDF2 fallback |

## 4. Logging and Monitoring

| # | Item | Status | Evidence | Notes |
|---|---|---|---|---|
| 4.1 | Application logs configured with rotation | PASS | `settings.py` logging config | 10MB, 5-10 backups |
| 4.2 | Access logs (HTTP) configured | PASS | Gunicorn + Caddy | `/var/log/svend/access.log` |
| 4.3 | Security events logged separately | PASS | `logs/security.log` | Django security logger |
| 4.4 | Application audit trail (who-did-what) | **FAIL** | -- | Not implemented (GAP-02) |
| 4.5 | Log retention meets policy requirements | PASS | Rotation config | ~50MB per log type |
| 4.6 | Real-time monitoring/alerting | **FAIL** | -- | Not implemented (GAP-04) |
| 4.7 | Failed login attempts logged | PASS | Django security log | Not aggregated or alerted |
| 4.8 | Admin actions logged | **FAIL** | -- | No admin audit trail |

## 5. Change Management

| # | Item | Status | Evidence | Notes |
|---|---|---|---|---|
| 5.1 | Version control in use | PASS | Git | Full history available |
| 5.2 | Change log maintained | PASS | `log.md` | Timestamped entries |
| 5.3 | Changes reviewed before deployment | PASS | Founder + AI review; multi-agent risk assessment | CHG-001 v1.6 |
| 5.4 | Rollback procedure documented | PASS | `policies/change-management.md` | |
| 5.5 | Pre-deployment backup performed | PASS | `backup_db.sh` | Daily automated |
| 5.6 | Automated testing in CI/CD | **FAIL** | -- | No CI/CD pipeline (PAR-10) |
| 5.7 | Staging environment available | **FAIL** | -- | Production only (PAR-10) |
| 5.8 | Dependency vulnerability scanning | **FAIL** | -- | Not enabled (GAP-03) |

## 6. Incident Response

| # | Item | Status | Evidence | Notes |
|---|---|---|---|---|
| 6.1 | Incident response plan documented | PASS | `policies/incident-response.md` | Severity levels + procedures |
| 6.2 | Incident severity classification defined | PASS | IRP-001 section 3 | SEV-1 through SEV-4 |
| 6.3 | Communication plan documented | PASS | IRP-001 section 6 | Internal + external templates |
| 6.4 | Tabletop exercise conducted | **FAIL** | -- | Not yet performed |
| 6.5 | Post-incident review process defined | PASS | IRP-001 section 5 | |

## 7. Business Continuity

| # | Item | Status | Evidence | Notes |
|---|---|---|---|---|
| 7.1 | RTO/RPO defined | PASS | `policies/bcdr.md` | RTO < 4hr, RPO < 24hr |
| 7.2 | Backup schedule documented | PASS | systemd timer | Daily encrypted |
| 7.3 | Backup restoration tested | **FAIL** | -- | Never tested |
| 7.4 | Off-site backup in place | **FAIL** | -- | Same machine only (PAR-09) |
| 7.5 | Disaster recovery scenarios documented | PASS | `policies/bcdr.md` sec 4 | 6 scenarios |
| 7.6 | DR drill conducted annually | **FAIL** | -- | Never conducted |

## 8. Vendor Management

| # | Item | Status | Evidence | Notes |
|---|---|---|---|---|
| 8.1 | Vendor inventory maintained | PASS | `policies/vendor-management.md` | 5 vendors |
| 8.2 | Critical vendor SOC 2 reports obtained | PASS | Stripe + Cloudflare verified | Anthropic: to be obtained |
| 8.3 | Vendor risk assessments performed | **FAIL** | -- | No formal assessments (PAR-12) |
| 8.4 | Data processing agreements in place | PASS | Stripe, Cloudflare, Anthropic ToS | |
| 8.5 | Vendor offboarding procedure documented | PASS | VMP-001 section 7 | |

## 9. Privacy

| # | Item | Status | Evidence | Notes |
|---|---|---|---|---|
| 9.1 | Privacy policy published | PASS | svend.ai | |
| 9.2 | Privacy policy aligned with SOC 2 criteria | **FAIL** | -- | Needs review (PAR-14) |
| 9.3 | Data retention schedule defined | **FAIL** | -- | Informal only (PAR-13) |
| 9.4 | Data subject access/export capability | **FAIL** | -- | Not implemented (GAP-05) |
| 9.5 | PII inventory documented | PASS | `policies/data-classification.md` | |

---

## Summary

| Category | Pass | Fail | Total | Pass Rate |
|---|---|---|---|---|
| Governance | 12 | 0 | 12 | 100% |
| Access Control | 7 | 3 | 10 | 70% |
| Encryption | 7 | 1 | 8 | 88% |
| Logging/Monitoring | 5 | 3 | 8 | 63% |
| Change Management | 5 | 3 | 8 | 63% |
| Incident Response | 4 | 1 | 5 | 80% |
| Business Continuity | 3 | 3 | 6 | 50% |
| Vendor Management | 4 | 1 | 5 | 80% |
| Privacy | 2 | 3 | 5 | 40% |
| **Total** | **49** | **18** | **67** | **73%** |

**Audit readiness: Improving.** 18 items remain. Session timeout, account lockout, and Argon2 hashing fixed. CHG-001 v1.6 strengthened change management. See [remediation.md](remediation.md) for fix tracker.
