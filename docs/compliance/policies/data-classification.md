# Data Classification Policy

**Policy ID:** DCP-001
**Version:** 1.0
**Effective Date:** 2026-03-03
**Last Updated:** 2026-03-05
**Owner:** Eric (Founder)
**Review Cycle:** Annual
**Parent Policy:** [Information Security Policy](information-security.md)

---

## 1. Purpose

Define classification levels for data processed by Svend, and prescribe handling, storage, transmission, and disposal requirements for each level.

## 2. Scope

All data created, collected, processed, stored, or transmitted by the Svend platform, including customer data, operational data, and platform metadata.

## 3. Classification Levels

| Level | Label | Color | Definition |
|---|---|---|---|
| 4 | **Restricted** | Red | Compromise causes severe harm. Encryption keys, API secrets, database credentials. |
| 3 | **Confidential** | Orange | Customer PII, financial data, analysis results, project hypotheses. Breach requires notification. |
| 2 | **Internal** | Yellow | Operational data not meant for public. Logs, metrics, architecture docs, source code. |
| 1 | **Public** | Green | Intentionally public. Marketing site, published tools, blog posts, open-source components. |

## 4. Data Inventory by Classification

### Restricted (Level 4)

| Data | Storage Location | Protection |
|---|---|---|
| Database credentials | Environment variable | Not in source code; loaded at runtime |
| `FIELD_ENCRYPTION_KEY` | `~/.svend_encryption_key` | File permissions (owner-only); not in git |
| `SECRET_KEY` | Environment variable | Not in source code |
| Stripe API keys | Environment variable | Rotatable via Stripe dashboard |
| Anthropic API key | Environment variable | Rotatable via Anthropic console |
| SMTP credentials | Environment variable | Rotatable |
| SSH private keys | `~/.ssh/` | File permissions; passphrase protected |
| Backup encryption passphrase | Derived from `FIELD_ENCRYPTION_KEY` | Same protection as encryption key |

### Confidential (Level 3)

| Data | Storage | Protection |
|---|---|---|
| User email addresses | PostgreSQL `accounts_user.email` | HTTPS in transit; database-level access control |
| User names | PostgreSQL `accounts_user.first_name`, `last_name` | Same |
| Stripe customer IDs | PostgreSQL (encrypted via `EncryptedCharField`) | Fernet encryption at rest |
| Chat messages | PostgreSQL (encrypted via `EncryptedTextField`) | Fernet encryption at rest |
| User analysis data | PostgreSQL (project, hypothesis, evidence tables) | Per-user ownership isolation |
| Uploaded files | Filesystem (`media/`) | `EncryptedFileSystemStorage` |
| Database backups | `/home/eric/backups/svend/` | AES-256-CBC encrypted |

### Internal (Level 2)

| Data | Storage | Protection |
|---|---|---|
| Application logs | `logs/`, `/var/log/svend/` | File permissions; rotated |
| Source code | Git repository | SSH-authenticated access; GitHub private repo |
| Configuration files | Repository (non-secret parts) | Git access control |
| Deployment scripts | Repository | Git access control |
| DEBT tracker | `.kjerne/DEBT.md` | Git access control |
| Architecture docs | `reference_docs/` | Git access control |
| Analytics (hashed) | PostgreSQL `SiteVisit` | SHA-256 hashed IPs |

### Public (Level 1)

| Data | Location |
|---|---|
| Marketing site content | `site/index.html` |
| Public calculator tools | `templates/tools/` |
| Playbook pages | `templates/playbooks/` |
| Blog posts | Published via admin interface |
| Terms of service | Landing page |
| Privacy policy | Landing page |

## 5. Handling Rules

| Action | Restricted | Confidential | Internal | Public |
|---|---|---|---|---|
| **Storage** | Environment vars or encrypted files only; never in source code | Encrypted at rest (Fernet/AES-256) | Standard storage; access-controlled | No restrictions |
| **Transmission** | Never transmitted in plaintext; avoid transmission when possible | HTTPS only; encrypted API calls | HTTPS preferred | No restrictions |
| **Access** | System owner only | Authenticated users (own data only); system owner for admin | Authorized personnel | Anyone |
| **Logging** | Never log values; log access events only | Redact PII from logs; log access events | Standard logging | No restrictions |
| **Sharing** | Never share externally | Share only with data subject or authorized tenant members | Share within development context | Freely shareable |
| **Disposal** | Secure wipe; key destruction | Secure deletion; backup rotation (30 days) | Standard deletion | No restrictions |
| **Backup** | Backed up via encrypted database backup | Backed up via encrypted database backup | Backed up via git | Not backed up |

## 6. Labeling

- **Source code:** Comments indicate when handling Restricted/Confidential data (e.g., `# CONFIDENTIAL: user PII`)
- **Database fields:** Encrypted fields use `EncryptedCharField`/`EncryptedTextField` — the field type itself is the label
- **Files:** Environment files (`.env*`) treated as Restricted regardless of content
- **Logs:** Security logs (`security.log`) treated as Internal; must not contain Confidential data

## 7. Data in Transit

| Path | Protection |
|---|---|
| Browser → Cloudflare | TLS 1.2+ (Cloudflare edge) |
| Cloudflare → Caddy | Cloudflare Tunnel (encrypted) |
| Caddy → Gunicorn | Localhost (127.0.0.1); no network exposure |
| Application → PostgreSQL | Localhost connection |
| Application → Stripe API | HTTPS |
| Application → Anthropic API | HTTPS |
| Application → SMTP | TLS |
| Backup script → Filesystem | Local; encrypted output |

## 8. Data Disposal

| Data Type | Retention | Disposal Method |
|---|---|---|
| Database backups | 30 days | Automated rotation in `backup_db.sh` |
| Application logs | 5 rotations x 10MB | Log rotation (overwrite) |
| User accounts (deleted) | TBD (gap — needs formal deletion) | Planned: complete data purge script |
| Email verification tokens | 24 hours or until verified (whichever first) | Expired tokens rejected; cleared on verification |
| Org invitations | 7-day expiry | Marked inactive; purged by `run_purge.sh` |
| Session data | 8 hours (`SESSION_COOKIE_AGE=28800`) | Django session cleanup |

## 9. Compliance Notes

- PCI DSS: No cardholder data ever touches Svend — Stripe.js handles all card input client-side
- GDPR: Data subject access requests serviceable via application (gap: no bulk export yet)
- SOC 2 Confidentiality: Encrypted storage + ownership isolation + access control = framework in place

<!-- policy-watches: settings.py:SESSION_COOKIE_AGE, accounts/models.py:verify_email -->
