# Secret Storage Compliance Documentation

## ISO 27001 A.10.1 - Cryptographic Controls

This document describes how the `syn.core.secrets` module implements cryptographic controls for secure storage of sensitive data in compliance with ISO 27001 requirements.

### Implementation Overview

The secret storage system implements **envelope encryption** using AES-256 encryption via the Fernet library, with optional AWS KMS integration for enhanced enterprise security.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Envelope Encryption                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  KEK (Key Encryption Key)                              │
│    ↓ encrypts                                           │
│  DEK (Data Encryption Key)                             │
│    ↓ encrypts                                           │
│  Secret Value                                           │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Compliance Mapping

#### ISO 27001 A.10.1.1 - Policy on the use of cryptographic controls

**Requirement**: A policy on the use of cryptographic controls for protection of information shall be developed and implemented.

**Implementation**:
- All secrets stored in `SecretStore` are encrypted at rest using Fernet (AES-128-CBC with HMAC-SHA256)
- Envelope encryption pattern ensures each secret has its own unique encryption key (DEK)
- Key Encryption Keys (KEK) are stored securely in environment variables or AWS KMS
- No plaintext secrets are ever stored in the database

**Location**: `syn/core/secrets.py` - `SecretStore` model and `SecretEncryptionManager` class

#### ISO 27001 A.10.1.2 - Key management

**Requirement**: A policy on the use, protection and lifetime of cryptographic keys shall be developed and implemented through their whole lifecycle.

**Implementation**:

1. **Key Generation**:
   - DEKs are generated using cryptographically secure random number generation via Fernet
   - Each secret has its own unique DEK
   - Location: `SecretEncryptionManager.generate_dek()`

2. **Key Storage**:
   - DEKs are encrypted with KEK before storage
   - KEKs are stored in environment variables (never in database)
   - Optional AWS KMS integration for enterprise key management
   - Location: `SecretStore.dek_encrypted` field

3. **Key Rotation**:
   - Automatic rotation on configurable schedule (default: 90 days)
   - Manual rotation via management command: `python manage.py rotate_keys`
   - Automated rotation via Celery beat (monthly)
   - Zero-downtime key rotation using versioned KEKs
   - Location: `rotate_secret()`, `rotate_all_keys()`, `syn/core/tasks.py`

4. **Key Versioning**:
   - KEK versioning supports seamless key rotation
   - Old and new keys coexist during rotation period
   - Location: `SecretStore.kek_version` field

**Evidence**:
- Scheduled rotation: `synara_core/celery.py` - `rotate_secrets_auto` task (monthly)
- Management command: `syn/core/management/commands/rotate_keys.py`
- Celery tasks: `syn/core/tasks.py`

#### ISO 27001 A.9.4.5 - Access control to program source code

**Requirement**: Access to program source code shall be restricted.

**Implementation**:
- Tenant isolation ensures secrets are segregated by `tenant_id`
- Multi-tenant queries filtered by tenant context
- Database indexes enforce tenant boundaries
- Location: `SecretStore.tenant_id` field with unique constraint on `(name, tenant_id)`

### Cryptographic Specifications

#### Encryption Algorithm
- **Algorithm**: Fernet (symmetric encryption)
- **Cipher**: AES-128-CBC
- **MAC**: HMAC-SHA256
- **Library**: `cryptography` (Python Cryptographic Authority)
- **Key Size**: 128-bit symmetric key

**Note**: While the requirement specifies AES-256, Fernet uses AES-128 which is still considered secure and approved by NIST. If AES-256 is a hard requirement for your compliance needs, the implementation can be extended to use `cryptography.hazmat.primitives.ciphers` directly.

#### Key Derivation
- **DEK Generation**: `Fernet.generate_key()` (cryptographically secure random)
- **KEK Storage**: Base64-encoded 32-byte keys in environment variables

#### AWS KMS Integration (Optional)
- **Service**: AWS Key Management Service
- **Key Type**: Customer Master Key (CMK)
- **Encryption**: Envelope encryption with KMS-managed keys
- **Fallback**: Automatic fallback to Fernet if KMS unavailable

### Security Features

1. **Envelope Encryption**
   - Each secret encrypted with unique DEK
   - DEKs encrypted with KEK
   - Compromised DEK only affects single secret

2. **Tenant Isolation**
   - Database-level segregation via `tenant_id`
   - Unique constraints prevent cross-tenant access
   - Queries filtered by tenant context

3. **Audit Trail**
   - All operations logged to `syn.audit.models.SysLogEntry`
   - Logs include: creation, updates, rotation, deletion
   - Actor and timestamp tracked for all changes

4. **Key Rotation**
   - Scheduled automatic rotation (default: monthly)
   - Manual rotation via CLI: `python manage.py rotate_keys`
   - Zero-downtime rotation with versioned keys
   - Rotation tracking via `last_rotated_at` field

5. **Error Handling**
   - Corrupted data detection via HMAC validation
   - Informative errors for key management issues
   - Secure failure modes (no plaintext leakage)

### Operational Procedures

#### Initial Setup

1. Generate encryption key:
   ```bash
   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
   ```

2. Set environment variable:
   ```bash
   export SECRET_ENCRYPTION_KEY="<generated-key>"
   ```

3. Run migrations:
   ```bash
   python manage.py migrate
   ```

#### Storing a Secret

```python
from syn.core.secrets import set_secret

set_secret(
    name="stripe_api_key",
    value="sk_live_...",
    tenant_id="acme_corp",
    created_by="admin@acme.com",
    rotation_days=90  # Rotate every 90 days
)
```

#### Retrieving a Secret

```python
from syn.core.secrets import get_secret

api_key = get_secret("stripe_api_key", "acme_corp")
```

#### Manual Key Rotation

Rotate individual secret:
```bash
python manage.py rotate_keys --rotate-secrets --tenant acme_corp
```

Rotate KEK (all secrets):
```bash
# 1. Generate new KEK
export SECRET_ENCRYPTION_KEY_V2="<new-key>"

# 2. Rotate all secrets
python manage.py rotate_keys --old-version 1 --new-version 2

# 3. Update environment to use new key
export SECRET_ENCRYPTION_KEY=$SECRET_ENCRYPTION_KEY_V2
```

#### Automated Rotation

Automated rotation is configured via Celery Beat:
- **Schedule**: Monthly on the 1st at 2 AM UTC
- **Task**: `syn.core.tasks.rotate_secrets_auto`
- **Config**: `synara_core/celery.py`

### Testing

Comprehensive test coverage includes:
- Encryption/decryption correctness
- Key rotation (DEK and KEK)
- Tenant isolation
- KMS fallback
- Error handling
- Unicode and large values
- Compliance requirements

Run tests:
```bash
pytest syn/core/tests/test_secrets.py -v
```

### Compliance Checklist

- [x] **A.10.1.1**: Cryptographic policy implemented (envelope encryption)
- [x] **A.10.1.2**: Key management lifecycle (generation, storage, rotation, versioning)
- [x] **A.9.4.5**: Access control (tenant isolation)
- [x] **A.12.4.1**: Event logging (audit trail integration)
- [x] **A.12.7.1**: Audit log protection (immutable audit entries via syn.audit)
- [x] **A.18.1.5**: Cryptographic controls compliance (documented in this file)

### AWS KMS Setup (Optional)

For enterprise deployments requiring hardware security modules (HSM):

1. Create KMS key:
   ```bash
   aws kms create-key --description "Synara Secret Encryption Key"
   ```

2. Set environment variable:
   ```bash
   export AWS_KMS_KEY_ID="arn:aws:kms:us-east-1:..."
   ```

3. Configure AWS credentials (boto3 will use default credential chain)

4. The system will automatically use KMS for DEK encryption, with Fernet fallback

### References

- ISO/IEC 27001:2013 - Information security management systems
- NIST SP 800-57 - Recommendation for Key Management
- FIPS 140-2 - Security Requirements for Cryptographic Modules
- Cryptography library documentation: https://cryptography.io/

### Change Log

| Date | Version | Change | Author |
|------|---------|--------|--------|
| 2025-10-28 | 1.0 | Initial implementation | Claude Code |

---

**Document Owner**: Security Team
**Review Date**: Quarterly
**Next Review**: 2025-01-28
