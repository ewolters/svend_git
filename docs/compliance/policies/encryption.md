# Encryption Policy

**Policy ID:** ENC-001
**Version:** 1.0
**Effective Date:** 2026-03-03
**Last Updated:** 2026-03-05
**Owner:** Eric (Founder)
**Review Cycle:** Annual
**Parent Policy:** [Information Security Policy](information-security.md)

---

## 1. Purpose

Define cryptographic standards for protecting data in transit, at rest, and in backup for the Svend platform.

## 2. Scope

- All data transmission between clients, servers, and third parties
- All stored data classified as Confidential or Restricted (see [data-classification.md](data-classification.md))
- All backup and archival data
- Key management practices

## 3. Encryption Standards

### 3.1 Data in Transit

| Path | Protocol | Minimum Standard | Implementation |
|---|---|---|---|
| Client → Cloudflare | TLS | TLS 1.2+ | Cloudflare edge termination |
| Cloudflare → Server | Cloudflare Tunnel | Encrypted tunnel | `cloudflared` service |
| Application → Stripe | HTTPS | TLS 1.2+ | Stripe Python SDK |
| Application → Anthropic | HTTPS | TLS 1.2+ | Anthropic Python SDK |
| Application → SMTP | STARTTLS | TLS 1.2+ | Django EMAIL_USE_TLS=True |

**HSTS Configuration:**
- Max-age: 63,072,000 seconds (2 years)
- Include subdomains: Yes
- Preload: Yes
- HTTPS redirect: Enforced (`SECURE_SSL_REDIRECT = True`)

### 3.2 Data at Rest — Application Layer

| Mechanism | Algorithm | Key Size | Implementation |
|---|---|---|---|
| Field-level encryption | Fernet (AES-128-CBC + HMAC-SHA256) | 128-bit AES | `core/encryption.py` |
| File encryption | Fernet | 128-bit AES | `core/encrypted_storage.py` |
| Password hashing | Argon2id (primary); PBKDF2-SHA256 (fallback) | N/A | `settings.py` PASSWORD_HASHERS |
| Token hashing | SHA-256 | 256-bit | Email verification tokens |
| IP anonymization | SHA-256 | 256-bit | SiteVisit analytics |

**Encrypted Fields:**
- `User.stripe_customer_id` — `EncryptedCharField`
- `Chat.Message` content — `EncryptedTextField`
- User uploaded files — `EncryptedFileSystemStorage`

**Custom Field Types** (`core/encryption.py`):
- `EncryptedCharField` — VARCHAR with Fernet envelope (max 500 chars for ~1.4x expansion)
- `EncryptedTextField` — TEXT with Fernet envelope
- `EncryptedJSONField` — JSON serialized, Fernet encrypted, stored as TEXT

### 3.3 Data at Rest — Backups

| Parameter | Value |
|---|---|
| Algorithm | AES-256-CBC |
| Key derivation | PBKDF2 |
| Tool | OpenSSL (`openssl enc -aes-256-cbc -pbkdf2`) |
| Key source | Derived from `FIELD_ENCRYPTION_KEY` |
| Storage | `/home/eric/backups/svend/` (local; off-site planned) |

### 3.4 Data at Rest — Database

| Parameter | Value |
|---|---|
| PostgreSQL | Standard file-level storage (no TDE) |
| Protection | Filesystem permissions; localhost-only access |
| Sensitive fields | Application-layer Fernet encryption |

## 4. Key Management

### 4.1 Encryption Key

| Attribute | Value |
|---|---|
| Key ID | `FIELD_ENCRYPTION_KEY` |
| Storage | `~/.svend_encryption_key` (file, owner-read only) |
| Format | Fernet key (URL-safe base64-encoded 32-byte key) |
| Loaded via | `settings.py` at application startup |
| Backup | Manual secure copy (gap: needs formal offsite backup) |
| Rotation | Not yet rotated; rotation procedure not implemented |

### 4.2 Key Rotation Procedure (Planned)

1. Generate new Fernet key
2. Run re-encryption migration for all encrypted model fields
3. Re-encrypt stored files via `encrypt_existing_files` management command
4. Update `~/.svend_encryption_key` with new key
5. Restart application
6. Generate fresh encrypted backups
7. Verify decryption of sample records
8. Securely destroy old key after confirming all data re-encrypted

### 4.3 Other Secrets

| Secret | Storage | Rotation |
|---|---|---|
| `SECRET_KEY` (Django) | Environment variable | Rotate on suspected compromise |
| Stripe API keys | Environment variable | Rotatable via Stripe dashboard |
| Anthropic API key | Environment variable | Rotatable via Anthropic console |
| SMTP credentials | Environment variable | Rotate per provider policy |
| SSH keys | `~/.ssh/` (passphrase protected) | Rotate annually or on compromise |

## 5. Prohibited Practices

- Storing encryption keys or API secrets in source code or version control
- Using symmetric encryption with keys shorter than 128 bits
- Using TLS versions below 1.2
- Using MD5 or SHA-1 for any security purpose
- Transmitting Restricted data without encryption
- Logging decrypted Confidential or Restricted data

## 6. Known Issues and Remediation

| Issue | Risk | Remediation | Status |
|---|---|---|---|
| Silent decryption failure returns ciphertext | Ciphertext exposed as plaintext on key mismatch | Change fallback to raise exception | Open (P1 H6) |
| No key rotation procedure tested | Key compromise requires manual re-encryption | Implement and test rotation script | Open |
| Fernet uses AES-128, not AES-256 | Adequate but not maximum strength | Accept — Fernet's HMAC-SHA256 adds integrity; AES-128 is sufficient for SOC 2 | Accepted |
| TEMPORA_CLUSTER_SECRET derived from SECRET_KEY | Chain of compromise | Separate secret | Open (P2 L7) |

### Resolved Issues

| Issue | Resolution | Date |
|---|---|---|
| PBKDF2 for passwords (not Argon2) | Migrated to Argon2id primary hasher; PBKDF2 fallback for legacy hashes (auto-upgrade on next login) | 2026-03-05 |

## 7. Compliance Mapping

| Requirement | SOC 2 Criteria | Status |
|---|---|---|
| Encryption in transit | CC6.1, C1.2 | Met (TLS 1.2+, HSTS, Cloudflare Tunnel) |
| Encryption at rest | C1.2 | Met (Fernet for PII, AES-256 for backups) |
| Key management | CC6.3 | Partial (keys secured, rotation not tested) |
| Password hashing | CC6.2 | Met (Argon2id primary, PBKDF2 fallback) |

<!-- policy-watches: settings.py:PASSWORD_HASHERS -->
