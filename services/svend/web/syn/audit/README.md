# Audit System - Tamper-Proof Logging

Blockchain-style tamper-proof audit logging system with hash chaining for immutable forensic records.

## Overview

The audit system provides cryptographically secure, immutable audit logs where each entry contains a hash that includes the previous entry's hash, creating a blockchain-style chain that makes tampering detectable.

## Features

- **Immutability**: Entries cannot be modified after creation
- **Hash Chain**: Each entry links to previous via cryptographic hash
- **Tamper Detection**: Any modification breaks the hash chain
- **Tenant Isolation**: Separate chains per tenant
- **Integrity Verification**: Automated validation of hash chains
- **Violation Alerting**: Automatic alerts on detected tampering
- **Forensic Trail**: Complete audit trail for compliance

## Models

### SysLogEntry

Immutable system log entry with hash chain integrity.

```python
SysLogEntry(
    id=1,                                  # Sequential ID
    timestamp='2025-10-27T16:30:00Z',     # When created
    actor='user@example.com',              # Who performed action
    event_name='user.login',               # What happened
    payload={'ip': '192.168.1.1'},        # Event data
    payload_hash='abc123...',              # SHA-256 of payload
    correlation_id='corr-123',             # Distributed tracing
    tenant_id='tenant-123',                # Tenant isolation
    previous_hash='def456...',             # Links to previous entry
    current_hash='ghi789...',              # Hash of this entry
    is_genesis=False                       # First entry flag
)
```

### IntegrityViolation

Record of detected integrity violations.

```python
IntegrityViolation(
    id=uuid,
    detected_at='2025-10-27T17:00:00Z',
    tenant_id='tenant-123',
    violation_type='hash_mismatch',
    entry_id=123,
    details={'message': 'Hash verification failed'},
    resolved=False
)
```

## Usage

### Creating Audit Entries

```python
from syn.audit.utils import generate_entry

# Create audit entry
entry = generate_entry(
    tenant_id="tenant-123",
    actor="user@example.com",
    event_name="user.login",
    payload={"ip": "192.168.1.1", "method": "oauth"},
    correlation_id="corr-abc123"
)

# Entry is automatically added to hash chain
print(f"Entry created: {entry.current_hash[:8]}...")
```

### Verifying Chain Integrity

```python
from syn.audit.utils import verify_chain_integrity

# Verify entire chain for tenant
result = verify_chain_integrity("tenant-123")

if result['is_valid']:
    print(f"✓ Chain intact: {result['total_entries']} entries")
else:
    print(f"✗ Violations: {len(result['violations'])}")
    for violation in result['violations']:
        print(f"  - {violation['type']}: {violation['message']}")
```

### Getting Audit Trail

```python
from syn.audit.utils import get_audit_trail

# Get recent audit entries
trail = get_audit_trail(
    tenant_id="tenant-123",
    event_name="user.login",  # Optional filter
    actor="user@example.com",  # Optional filter
    limit=100
)

for entry in trail:
    print(f"{entry.timestamp}: {entry.actor} - {entry.event_name}")
```

## Hash Chain Mechanics

### Hash Computation

Each entry's hash includes:
1. Timestamp
2. Actor
3. Event name
4. Payload hash (SHA-256 of payload JSON)
5. Correlation ID
6. Tenant ID
7. **Previous entry's hash** (creates chain linkage)

```python
hash_data = {
    'timestamp': '2025-10-27T16:30:00Z',
    'actor': 'user@example.com',
    'event_name': 'user.login',
    'payload_hash': 'abc123...',
    'correlation_id': 'corr-123',
    'tenant_id': 'tenant-123',
    'previous_hash': 'def456...'  # Chain link
}

current_hash = SHA256(json.dumps(hash_data, sort_keys=True))
```

### Chain Structure

```
Genesis Entry (ID=1)
├─ previous_hash: "000...000" (64 zeros)
├─ current_hash:  "abc123..."
└─ is_genesis:    True

Entry 2
├─ previous_hash: "abc123..." ◄── Links to Entry 1
├─ current_hash:  "def456..."
└─ is_genesis:    False

Entry 3
├─ previous_hash: "def456..." ◄── Links to Entry 2
├─ current_hash:  "ghi789..."
└─ is_genesis:    False
```

### Tamper Detection

If any entry is modified:
1. Its `current_hash` no longer matches computed hash
2. Next entry's `previous_hash` no longer matches
3. Chain is broken and detectable

## Management Commands

### Verify Integrity

```bash
# Verify all tenants
python manage.py verify_syslog_integrity

# Verify specific tenant
python manage.py verify_syslog_integrity --tenant=tenant-123

# Record violations to database
python manage.py verify_syslog_integrity --record-violations

# Exit with error if violations found
python manage.py verify_syslog_integrity --fail-on-violation
```

Example output:
```
Verifying tenant: tenant-123
  ✓ Chain intact: 150 entries verified

Verifying tenant: tenant-456
  ✗ Chain compromised: 2 violations detected
    - hash_mismatch: Hash mismatch for entry 45
    - chain_break: Chain break at entry 46

VERIFICATION SUMMARY
======================================================================
Tenants verified: 2
Total entries: 200
Valid chains: 1
Invalid chains: 1
Total violations: 2
```

## Celery Tasks

### Periodic Verification

```python
from syn.audit.tasks import verify_audit_integrity

# Run verification manually
result = verify_audit_integrity.delay()

# Schedule periodic verification in Celery beat
CELERY_BEAT_SCHEDULE = {
    'verify-audit-integrity': {
        'task': 'audit.tasks.verify_audit_integrity',
        'schedule': crontab(minute=0, hour='*/6'),  # Every 6 hours
    },
}
```

### Cleanup Old Violations

```python
from syn.audit.tasks import cleanup_old_violations

# Clean up resolved violations older than 90 days
deleted = cleanup_old_violations.delay(days=90)
```

## Immutability Enforcement

### Cannot Update Entries

```python
entry = SysLogEntry.objects.get(id=123)

# This will raise ValidationError
entry.actor = "attacker@example.com"
entry.save()  # ✗ ValidationError: Entries are immutable
```

### Cannot Bulk Update

```python
# This will fail
SysLogEntry.objects.filter(
    tenant_id="tenant-123"
).update(actor="attacker")  # ✗ Exception
```

### Cannot Delete (via Admin)

Admin interface is read-only. Deletion requires superuser privileges and should only be done for legal/compliance reasons.

## Tenant Isolation

Each tenant has an independent hash chain:

```python
# Tenant A's chain
A1 → A2 → A3 → A4

# Tenant B's chain (independent)
B1 → B2 → B3 → B4
```

Tampering with one tenant's chain doesn't affect others.

## Integrity Violations

When tampering is detected, an `IntegrityViolation` record is created and a `governance.audit_integrity_violation` event is emitted.

```python
from syn.audit.utils import record_integrity_violation

violation = record_integrity_violation(
    tenant_id="tenant-123",
    violation_type='hash_mismatch',
    entry_id=45,
    details={
        'expected_hash': 'abc123...',
        'actual_hash': 'tampered...',
        'severity': 'critical'
    }
)

# Governance event emitted automatically
# Alert ops team via event subscription
```

## Event Types

Common event names to log:

**Authentication**:
- `auth.login_success`
- `auth.login_failure`
- `auth.logout`
- `auth.password_change`

**Authorization**:
- `authz.permission_granted`
- `authz.permission_denied`
- `authz.role_assigned`

**Data Access**:
- `data.read`
- `data.create`
- `data.update`
- `data.delete`

**System**:
- `system.config_change`
- `system.backup_created`
- `system.restore_initiated`

**Security**:
- `security.suspicious_activity`
- `security.rate_limit_exceeded`
- `security.vulnerability_detected`

## Examples

### Example 1: Log User Login

```python
from syn.audit.utils import generate_entry

def handle_user_login(user, request):
    # Perform authentication
    authenticate_user(user)

    # Log to audit trail
    generate_entry(
        tenant_id=user.tenant_id,
        actor=user.email,
        event_name="auth.login_success",
        payload={
            "ip": request.META.get('REMOTE_ADDR'),
            "user_agent": request.META.get('HTTP_USER_AGENT'),
            "method": "password",
        },
        correlation_id=request.correlation_id
    )
```

### Example 2: Log Data Deletion

```python
from syn.audit.utils import generate_entry

def delete_sensitive_data(data_id, actor):
    # Get data before deletion
    data = Data.objects.get(id=data_id)

    # Log deletion
    generate_entry(
        tenant_id=data.tenant_id,
        actor=actor.email,
        event_name="data.delete",
        payload={
            "data_id": str(data_id),
            "data_type": data.type,
            "reason": "user_request",
            "snapshot": data.to_dict()  # Preserve record
        }
    )

    # Perform deletion
    data.delete()
```

### Example 3: Scheduled Integrity Check

```python
from celery import shared_task
from syn.audit.utils import verify_chain_integrity, record_integrity_violation

@shared_task
def nightly_audit_check():
    """Run nightly integrity verification."""
    from syn.synara.models import Tenant

    violations_found = []

    for tenant in Tenant.objects.filter(is_active=True):
        result = verify_chain_integrity(str(tenant.id))

        if not result['is_valid']:
            # Record violations
            for violation in result['violations']:
                v = record_integrity_violation(
                    tenant_id=str(tenant.id),
                    violation_type=violation['type'],
                    entry_id=violation.get('entry_id'),
                    details=violation
                )
                violations_found.append(v)

    # Send summary email if violations found
    if violations_found:
        send_security_alert(
            subject=f"Audit Integrity Violations: {len(violations_found)}",
            violations=violations_found
        )

    return {
        'violations_found': len(violations_found)
    }
```

## Testing

Run tests:

```bash
python manage.py test syn.audit.tests.test_syslog
```

### Test Coverage

- ✅ Entry creation and hash computation
- ✅ Genesis entry properties
- ✅ Hash chain linkage
- ✅ Immutability enforcement
- ✅ Update prevention
- ✅ Hash chain validation
- ✅ Tamper detection
- ✅ Chain break detection
- ✅ Violation recording
- ✅ Governance event emission
- ✅ Tenant isolation
- ✅ Compliance requirements

## Compliance

### SOC 2 CC7.2
**System Monitoring - Audit Logging**
- Complete audit trail of system activities
- Immutable records prevent evidence tampering
- Automated integrity verification
- Violation detection and alerting

### ISO 27001 A.12.7
**Audit Log Protection**
- Cryptographic protection against tampering
- Evidence of unauthorized modifications
- Secure log storage and retention
- Regular integrity verification

## Performance

- **Write Performance**: O(1) per entry (append-only)
- **Verification**: O(n) where n = entries per tenant
- **Storage**: ~500 bytes per entry (varies with payload)
- **Indexes**: Optimized for tenant, timestamp, event_name queries

## Best Practices

1. **Log Critical Actions**: Authentication, authorization, data access
2. **Include Context**: IP address, user agent, correlation IDs
3. **Regular Verification**: Schedule periodic integrity checks
4. **Monitor Violations**: Set up alerts for integrity violations
5. **Investigate Immediately**: Any violation requires investigation
6. **Preserve Evidence**: Never delete entries without legal approval
7. **Tenant Isolation**: Ensure tenant_id is always set correctly

## Limitations

1. **No Time Travel**: Cannot insert entries with past timestamps
2. **No Editing**: Entries are permanent (by design)
3. **Sequential IDs**: IDs must be sequential (database constraint)
4. **Single Chain**: Each tenant has one chain (no branches)

## Troubleshooting

### Verification Fails

1. Check for database corruption
2. Verify no manual database edits
3. Check for race conditions during entry creation
4. Review application logs for errors

### High Storage Usage

1. Implement retention policy for old entries
2. Archive old entries to cold storage
3. Compress large payloads
4. Review event logging frequency

### Performance Issues

1. Ensure database indexes are present
2. Partition large tables by tenant_id
3. Use read replicas for verification
4. Cache verification results

## Security Considerations

- **Hash Algorithm**: SHA-256 (cryptographically secure)
- **Serialization**: JSON with sorted keys (deterministic)
- **Database Security**: Restrict write access to application only
- **Admin Access**: Read-only for most users, delete only for superuser
- **Backup**: Regular backups of audit log
- **Monitoring**: Alert on unusual entry patterns

## License

Copyright © 2025 Synara Core Team. All rights reserved.
