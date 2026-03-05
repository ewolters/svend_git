# Risk Assessment Policy

**Policy ID:** RAP-001
**Version:** 1.0
**Effective Date:** 2026-03-03
**Last Updated:** 2026-03-05
**Owner:** Eric (Founder)
**Review Cycle:** Annual
**Parent Policy:** [Information Security Policy](information-security.md)

---

## 1. Purpose

Establish the methodology for identifying, assessing, treating, and monitoring risks to the Svend platform's security, availability, and data integrity.

## 2. Scope

- Application security risks
- Infrastructure risks
- Third-party/vendor risks
- Data risks (loss, exposure, corruption)
- Operational risks (availability, performance)
- Compliance risks

## 3. Risk Assessment Methodology

### 3.1 Likelihood Scale

| Rating | Label | Definition |
|---|---|---|
| 1 | Rare | Less than once per year; requires unlikely combination of factors |
| 2 | Unlikely | Once per year; possible but not expected |
| 3 | Possible | 2-4 times per year; has happened in similar systems |
| 4 | Likely | Monthly; expected to occur |
| 5 | Almost Certain | Weekly or more; actively occurring or near-certain |

### 3.2 Impact Scale

| Rating | Label | Definition |
|---|---|---|
| 1 | Negligible | No customer impact; no data exposure; < 5 min disruption |
| 2 | Minor | Limited customer impact; no sensitive data; < 1 hour disruption |
| 3 | Moderate | Multiple customers affected; internal data exposed; < 4 hour disruption |
| 4 | Major | Significant customer impact; PII exposed; < 24 hour disruption; regulatory notice |
| 5 | Severe | All customers affected; mass data breach; extended outage; legal action |

### 3.3 Risk Score

**Risk Score = Likelihood x Impact**

| Score | Risk Level | Treatment |
|---|---|---|
| 1-4 | **Low** | Accept and monitor |
| 5-9 | **Medium** | Mitigate within 90 days |
| 10-15 | **High** | Mitigate within 30 days |
| 16-25 | **Critical** | Mitigate immediately |

## 4. Risk Register

### Active Risks

| ID | Risk | L | I | Score | Level | Treatment | Status | DEBT Ref |
|---|---|---|---|---|---|---|---|---|
| R-002 | Session hijacking (8-hour session, no MFA) | 2 | 3 | 6 | Medium | Session reduced to 8h ✓; MFA still needed (REM-09) | Open (partial) | P1 M1 |
| R-003 | Credential stuffing (no MFA) | 3 | 4 | 12 | High | Implement MFA | Open | -- |
| R-004 | Encrypted backup on same machine (single point of failure) | 2 | 5 | 10 | High | Off-site backup to B2/S3 | Open | P3 |
| R-005 | Supply chain vulnerability (no dependency scanning) | 3 | 3 | 9 | Medium | Enable Dependabot | Open | -- |
| R-006 | Silent decryption failure leaks ciphertext | 2 | 4 | 8 | Medium | Fix fallback behavior | Open | P1 H6 |
| R-007 | Stripe webhook replay (no idempotency) | 2 | 3 | 6 | Medium | Deduplicate by event ID | Open | P1 H7 |
| R-008 | Server compromise (single user runs everything) | 1 | 5 | 5 | Medium | Separate service accounts | Accepted | P2 M11 |
| R-009 | XSS via CSP gaps (unsafe-inline/unsafe-eval) | 2 | 3 | 6 | Medium | CSP hardening (phased) | Open | P2 L14 |
| R-010 | No audit trail (can't investigate incidents) | 3 | 4 | 12 | High | Application audit logging | Open | -- |
| R-011 | Service outage undetected (no monitoring) | 3 | 3 | 9 | Medium | Uptime monitoring + alerting | Open | -- |
| R-013 | Data loss from disk failure | 1 | 5 | 5 | Medium | Off-site backups (same as R-004) | Open | P3 |
| R-014 | Insider threat (single operator) | 1 | 5 | 5 | Medium | Accept -- inherent to small org; git history provides accountability | Accepted | -- |
| R-015 | Third-party data breach (Stripe, Anthropic) | 1 | 4 | 4 | Low | Monitor vendor security posture; review SOC 2 reports | Monitor | -- |

### Closed/Mitigated Risks

| ID | Risk | Mitigated | Mitigation |
|---|---|---|---|
| R-C01 | CSRF bypass on all views | 2026-02-22 | Removed @csrf_exempt from 278 views |
| R-C02 | IDOR on DSW/SPC/FMEA endpoints | 2026-02-20 | Added user= filter to all data-accessing views |
| R-C03 | Prompt injection in LLM features | 2026-02-22 | XML-wrapped user inputs; boundary instructions; 2000-char limits |
| R-C04 | SSRF via PDF generation | 2026-02-22 | --disable-local-file-access; HTML sanitization |
| R-C05 | X-Forwarded-For spoofing | 2026-02-22 | NUM_PROXIES=1 in DRF settings |
| R-C06 | Brute-force account compromise (no lockout) | 2026-03-05 | LoginAttempt model: 5 failures → 15-min lockout per username + DRF IP throttle |
| R-C07 | Email token used after compromise (no expiry) | 2026-03-05 | 24-hour token expiry via `email_verification_token_sent_at` field |

## 5. Risk Treatment Options

| Option | When to Use |
|---|---|
| **Mitigate** | Implement controls to reduce likelihood or impact |
| **Accept** | Risk is low or mitigation cost exceeds potential loss |
| **Transfer** | Shift risk to third party (insurance, vendor SLA) |
| **Avoid** | Eliminate the activity that creates the risk |

## 6. Assessment Schedule

| Activity | Frequency |
|---|---|
| Full risk assessment | Annual |
| Risk register review | Quarterly |
| New feature risk review | Per significant feature launch |
| Post-incident risk update | After every SEV-1 or SEV-2 incident |
| Vendor risk review | Annual per vendor |

<!-- policy-watches: accounts/models.py:LoginAttempt, settings.py:SESSION_COOKIE_AGE, accounts/models.py:verify_email -->

## 7. Integration with DEBT Tracker

Risks that require code changes are cross-referenced to `.kjerne/DEBT.md`. DEBT priority levels map to risk treatment urgency:

| DEBT Priority | Risk Level Equivalent |
|---|---|
| P0 | Critical (immediate) |
| P1 | High (30 days) |
| P2 | Medium (90 days) |
| P3 | Low (accept/monitor) |
