# Evidence Collection Guide

**Last Updated:** 2026-03-03
**Purpose:** Map every auditor evidence request to its source location in the Svend platform

---

## How to Use

Auditors will request evidence for each control. This document tells you exactly where to find it. Evidence should be collected fresh before the audit (not stale snapshots) and placed in `evidence/` with dated filenames.

## Collection Format

- **Configuration files:** Copy with sensitive values redacted (replace with `[REDACTED]`)
- **Screenshots:** PNG with timestamps visible
- **Log samples:** 7-day extract, PII redacted
- **Reports:** Generated from system, dated

---

## 1. Governance Evidence

| Evidence Needed | Source | How to Collect |
|---|---|---|
| Information Security Policy (signed/approved) | `docs/compliance/policies/information-security.md` | Copy from repo; show git commit date as approval |
| All sub-policies (10 documents) | `docs/compliance/policies/` | Full directory listing with file dates |
| Policy review records | Git log for policy files | `git log --oneline docs/compliance/policies/` |
| Risk register | `docs/compliance/policies/risk-assessment.md` sec 4 | Copy risk register table |
| Organizational chart / roles | CLAUDE.md; `policies/information-security.md` sec 6 | Screenshot of roles table |
| Acceptable use acknowledgment | `policies/acceptable-use.md` sec 9 | Copy acknowledgment table |

## 2. Access Control Evidence

| Evidence Needed | Source | How to Collect |
|---|---|---|
| Password policy configuration | `settings.py` lines ~100-110 (AUTH_PASSWORD_VALIDATORS) | Copy section (no secrets) |
| RBAC implementation | `accounts/permissions.py` | Copy decorator definitions |
| User role assignments | Database: `core_membership` table | `SELECT user_id, role, tenant_id FROM core_membership;` |
| Active user list with tiers | Database: `accounts_user` + `accounts_subscription` | Query with email, tier, last_active, is_active |
| Session configuration | `settings.py` session settings | Copy relevant lines |
| SSH access configuration | `/etc/ssh/sshd_config` | Copy (redact key paths) |
| fail2ban configuration | `fail2ban-svend.conf` | Full file copy |
| Firewall rules | `sudo ufw status verbose` | Command output |
| MFA configuration | N/A | **Gap** — document as planned |
| Email verification flow | `accounts/models.py` verification methods | Code excerpt |

## 3. Encryption Evidence

| Evidence Needed | Source | How to Collect |
|---|---|---|
| TLS configuration | `Caddyfile` + Cloudflare SSL settings | Copy Caddyfile; screenshot Cloudflare SSL/TLS page |
| HSTS configuration | `settings.py` SECURE_* settings | Copy relevant lines |
| Field encryption implementation | `core/encryption.py` | Code excerpt showing Fernet usage |
| Encrypted field usage | Model definitions using EncryptedCharField etc. | `grep -rn "Encrypted" models.py` output |
| Backup encryption | `backup_db.sh` | Copy script (redact key reference) |
| Key storage permissions | `ls -la ~/.svend_encryption_key` | Command output showing owner-only read |
| SSL certificate validity | `openssl s_client -connect svend.ai:443` | Certificate chain output |

## 4. Logging Evidence

| Evidence Needed | Source | How to Collect |
|---|---|---|
| Logging configuration | `settings.py` LOGGING dict | Copy configuration block |
| Sample application logs (7 days) | `logs/svend.log` | Extract with PII redacted |
| Sample security logs (7 days) | `logs/security.log` | Extract with PII redacted |
| Sample access logs (7 days) | `/var/log/svend/access.log` | Extract with IPs redacted |
| Log rotation configuration | `settings.py` handler configs | Show maxBytes and backupCount |
| Gunicorn logging config | `gunicorn.conf.py` | Copy logging section |
| Caddy logging config | `Caddyfile` log block | Copy log configuration |
| Audit trail records | N/A | **Gap** — document as planned |
| Monitoring dashboard | N/A | **Gap** — document as planned |

## 5. Change Management Evidence

| Evidence Needed | Source | How to Collect |
|---|---|---|
| Version control evidence | `git log --oneline -50` | Recent commit history |
| Change log entries (3 months) | `log.md` | Extract last 3 months of entries |
| Debt tracker | `.kjerne/DEBT.md` | Full file copy |
| Deployment process documentation | `policies/change-management.md` sec 5 | Policy reference |
| Sample change with full lifecycle | `log.md` + `git show <hash>` | Pick a representative change showing proposal-implement-verify-log |
| Pre-deployment backup evidence | systemd timer + backup files | `systemctl status svend-backup.timer` + `ls -la /home/eric/backups/svend/ \| head` |
| Rollback evidence (if any) | `log.md` entries tagged rollback | Search log for rollback events |

## 6. Incident Response Evidence

| Evidence Needed | Source | How to Collect |
|---|---|---|
| Incident response plan | `policies/incident-response.md` | Full document |
| Incident log (if any incidents occurred) | Incident files in `evidence/incidents/` | Copy incident records |
| Tabletop exercise results | N/A | **Gap** — schedule and conduct |
| Communication templates | `policies/incident-response.md` sec 6 | Copy templates |

## 7. Business Continuity Evidence

| Evidence Needed | Source | How to Collect |
|---|---|---|
| BCDR plan | `policies/bcdr.md` | Full document |
| Backup schedule proof | `systemctl list-timers svend-backup.timer` | Command output |
| Backup existence proof | `ls -la /home/eric/backups/svend/ \| tail -5` | Directory listing showing recent backups |
| Backup restoration test results | N/A | **Gap** — perform and document test |
| DR drill results | N/A | **Gap** — schedule and conduct |
| RTO/RPO targets | `policies/bcdr.md` sec 2 | Copy targets table |

## 8. Vendor Evidence

| Evidence Needed | Source | How to Collect |
|---|---|---|
| Vendor inventory | `policies/vendor-management.md` sec 3 | Copy inventory tables |
| Stripe SOC 2 / PCI DSS report | Stripe Trust Center | Download and save to `evidence/vendor/` |
| Cloudflare SOC 2 report | Cloudflare Trust Hub | Download and save |
| Anthropic SOC 2 report | Anthropic (request) | Request and save |
| Data processing agreements | Service agreements | Screenshot or PDF of relevant terms |
| Vendor review records | N/A | **Gap** — conduct first review |

## 9. Privacy Evidence

| Evidence Needed | Source | How to Collect |
|---|---|---|
| Published privacy policy | svend.ai | Screenshot with URL and date |
| PII inventory | `policies/data-classification.md` sec 4 | Copy Confidential data table |
| Consent mechanism | Registration flow | Screenshot of email verification |
| Data retention schedule | N/A | **Gap** — define and document |
| Data export capability | N/A | **Gap** — implement |
| Analytics anonymization | `api/models.py` SiteVisit | Code showing SHA-256 IP hashing |

---

## Evidence File Naming Convention

```
evidence/
├── governance/
│   └── 2026-03-XX_policy_git_log.txt
├── access-control/
│   └── 2026-03-XX_password_validators.txt
├── encryption/
│   └── 2026-03-XX_ssl_certificate.txt
├── logging/
│   └── 2026-03-XX_7day_access_log_sample.txt
├── change-management/
│   └── 2026-03-XX_recent_commits.txt
├── incidents/
│   └── (incident records if any)
├── bcdr/
│   └── 2026-03-XX_backup_listing.txt
├── vendor/
│   └── stripe_soc2_2025.pdf
└── privacy/
    └── 2026-03-XX_privacy_policy_screenshot.png
```

Date format: `YYYY-MM-DD` prefix on all evidence files.
