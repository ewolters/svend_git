"""
Comprehensive tests for tamper-proof audit logging system.

Tests:
- Immutability enforcement
- Hash chain validation
- Integrity failure detection
- Tenant isolation

Compliance: SOC 2 CC7.2 / ISO 27001 A.12.7
"""

import uuid

from django.core.exceptions import ValidationError
from django.test import TestCase, TransactionTestCase

from syn.audit.models import IntegrityViolation, SysLogEntry
from syn.audit.utils import (
    generate_entry,
    get_audit_trail,
    record_integrity_violation,
    verify_chain_integrity,
)


class TestSysLogEntryModel(TestCase):
    """Test SysLogEntry model functionality."""

    def setUp(self):
        """Set up test data."""
        self.tenant_id = uuid.uuid4()
        self.actor = "user@example.com"
        self.event_name = "user.login"

    def test_create_log_entry(self):
        """Test creating a basic log entry."""
        entry = SysLogEntry(
            tenant_id=self.tenant_id,
            actor=self.actor,
            event_name=self.event_name,
            payload={"ip": "192.168.1.1"},
        )
        entry.save()

        self.assertIsNotNone(entry.id)
        self.assertIsNotNone(entry.timestamp)
        self.assertIsNotNone(entry.current_hash)
        self.assertIsNotNone(entry.payload_hash)

    def test_genesis_entry_creation(self):
        """Test genesis (first) entry has special properties."""
        entry = SysLogEntry.objects.create(
            tenant_id=self.tenant_id,
            actor=self.actor,
            event_name=self.event_name,
            payload={},
        )

        # Genesis entry properties
        self.assertTrue(entry.is_genesis)
        self.assertEqual(entry.previous_hash, "0" * 64)
        self.assertEqual(len(entry.current_hash), 64)

    def test_hash_chain_linkage(self):
        """Test subsequent entries link to previous entry."""
        # Create first entry
        entry1 = SysLogEntry.objects.create(
            tenant_id=self.tenant_id,
            actor=self.actor,
            event_name="event.one",
            payload={"data": "first"},
        )

        # Create second entry
        entry2 = SysLogEntry.objects.create(
            tenant_id=self.tenant_id,
            actor=self.actor,
            event_name="event.two",
            payload={"data": "second"},
        )

        # Second entry should link to first
        self.assertFalse(entry2.is_genesis)
        self.assertEqual(entry2.previous_hash, entry1.current_hash)
        self.assertNotEqual(entry2.current_hash, entry1.current_hash)

    def test_hash_includes_all_fields(self):
        """Test hash computation includes all critical fields."""
        entry = SysLogEntry.objects.create(
            tenant_id=self.tenant_id,
            actor=self.actor,
            event_name=self.event_name,
            payload={"test": "data"},
            correlation_id=uuid.uuid4(),
        )

        # Changing any field should produce different hash
        original_hash = entry.current_hash

        # Can't actually change entry (immutable), but verify hash computation
        test_entry = SysLogEntry(
            tenant_id=self.tenant_id,
            actor="different@example.com",  # Changed actor
            event_name=self.event_name,
            payload={"test": "data"},
            timestamp=entry.timestamp,
            previous_hash=entry.previous_hash,
        )
        test_entry.payload_hash = test_entry._compute_payload_hash()
        test_hash = test_entry._compute_current_hash()

        self.assertNotEqual(original_hash, test_hash)


class TestImmutability(TestCase):
    """Test immutability enforcement."""

    def setUp(self):
        """Set up test data."""
        self.tenant_id = uuid.uuid4()

    def test_cannot_update_entry(self):
        """Test entries cannot be updated after creation."""
        entry = SysLogEntry.objects.create(
            tenant_id=self.tenant_id,
            actor="user@example.com",
            event_name="test.event",
            payload={"data": "original"},
        )

        # Attempt to modify and save
        entry.payload = {"data": "modified"}

        with self.assertRaises(ValidationError) as cm:
            entry.save()

        self.assertIn("immutable", str(cm.exception).lower())

    def test_cannot_update_via_queryset(self):
        """Test bulk updates are prevented."""
        # Create entries
        for i in range(3):
            SysLogEntry.objects.create(
                tenant_id=self.tenant_id,
                actor="user@example.com",
                event_name=f"event.{i}",
                payload={"index": i},
            )

        # Attempt bulk update should fail
        with self.assertRaises(Exception):
            SysLogEntry.objects.filter(tenant_id=self.tenant_id).update(
                actor="hacker@example.com"
            )

    def test_can_create_new_entries(self):
        """Test new entries can still be created."""
        # Create first entry
        entry1 = SysLogEntry.objects.create(
            tenant_id=self.tenant_id,
            actor="user@example.com",
            event_name="event.one",
            payload={},
        )

        # Create second entry should work
        entry2 = SysLogEntry.objects.create(
            tenant_id=self.tenant_id,
            actor="user@example.com",
            event_name="event.two",
            payload={},
        )

        self.assertIsNotNone(entry2.id)
        self.assertNotEqual(entry1.id, entry2.id)


class TestHashChainValidation(TestCase):
    """Test hash chain integrity validation."""

    def setUp(self):
        """Set up test data."""
        self.tenant_id = uuid.uuid4()

    def test_valid_chain(self):
        """Test validation passes for intact chain."""
        # Create a chain of entries
        for i in range(5):
            SysLogEntry.objects.create(
                tenant_id=self.tenant_id,
                actor="user@example.com",
                event_name=f"event.{i}",
                payload={"index": i},
            )

        # Verify chain
        result = verify_chain_integrity(self.tenant_id)

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["total_entries"], 5)
        self.assertEqual(len(result["violations"]), 0)
        self.assertTrue(result["genesis_valid"])

    def test_verify_individual_entry_hash(self):
        """Test individual entry hash verification."""
        entry = SysLogEntry.objects.create(
            tenant_id=self.tenant_id,
            actor="user@example.com",
            event_name="test.event",
            payload={"data": "test"},
        )

        # Hash should be valid
        self.assertTrue(entry.verify_hash())

    def test_verify_chain_link(self):
        """Test chain link verification."""
        # Create two entries
        entry1 = SysLogEntry.objects.create(
            tenant_id=self.tenant_id,
            actor="user@example.com",
            event_name="event.one",
            payload={},
        )

        entry2 = SysLogEntry.objects.create(
            tenant_id=self.tenant_id,
            actor="user@example.com",
            event_name="event.two",
            payload={},
        )

        # Chain links should be valid
        self.assertTrue(entry1.verify_chain_link())  # Genesis
        self.assertTrue(entry2.verify_chain_link())  # Links to entry1

    def test_detect_hash_tampering(self):
        """Test detection of hash tampering."""
        # Create entry
        entry = SysLogEntry.objects.create(
            tenant_id=self.tenant_id,
            actor="user@example.com",
            event_name="test.event",
            payload={"data": "original"},
        )

        # Simulate tampering by changing hash directly in DB
        SysLogEntry.objects.filter(id=entry.id).update(
            current_hash="tampered_hash_" + "0" * 48
        )

        # Reload entry
        entry.refresh_from_db()

        # Verification should fail
        self.assertFalse(entry.verify_hash())

    def test_detect_chain_break(self):
        """Test detection of broken chain links."""
        # Create entries
        SysLogEntry.objects.create(
            tenant_id=self.tenant_id,
            actor="user@example.com",
            event_name="event.one",
            payload={},
        )

        entry2 = SysLogEntry.objects.create(
            tenant_id=self.tenant_id,
            actor="user@example.com",
            event_name="event.two",
            payload={},
        )

        # Break the chain by changing previous_hash
        SysLogEntry.objects.filter(id=entry2.id).update(
            previous_hash="broken_hash_" + "0" * 51
        )

        # Reload and verify
        entry2.refresh_from_db()
        self.assertFalse(entry2.verify_chain_link())


class TestIntegrityFailureDetection(TestCase):
    """Test integrity failure detection and alerting."""

    def setUp(self):
        """Set up test data."""
        self.tenant_id = uuid.uuid4()

    def test_record_integrity_violation(self):
        """Test recording of integrity violations."""
        violation = record_integrity_violation(
            tenant_id=self.tenant_id,
            violation_type="hash_mismatch",
            entry_id=123,
            details={"message": "Hash does not match"},
        )

        # Violation should be recorded
        self.assertIsNotNone(violation.id)
        self.assertEqual(violation.tenant_id, self.tenant_id)
        self.assertEqual(violation.violation_type, "hash_mismatch")
        self.assertFalse(violation.is_resolved)

    def test_detect_and_record_violations(self):
        """Test automatic detection and recording of violations."""
        # Create valid chain
        for i in range(3):
            SysLogEntry.objects.create(
                tenant_id=self.tenant_id,
                actor="user@example.com",
                event_name=f"event.{i}",
                payload={},
            )

        # Tamper with middle entry
        entries = SysLogEntry.objects.filter(tenant_id=self.tenant_id).order_by("id")
        middle_entry = entries[1]
        SysLogEntry.objects.filter(id=middle_entry.id).update(
            current_hash="tampered_" + "0" * 56
        )

        # Verify chain (should detect violation)
        result = verify_chain_integrity(self.tenant_id)

        self.assertFalse(result["is_valid"])
        self.assertGreater(len(result["violations"]), 0)

        # Record violations
        for violation in result["violations"]:
            record_integrity_violation(
                tenant_id=self.tenant_id,
                violation_type=violation["type"],
                entry_id=violation.get("entry_id"),
                details=violation,
            )

        # Check violations were recorded
        violations = IntegrityViolation.objects.filter(tenant_id=self.tenant_id)
        self.assertGreater(violations.count(), 0)


class TestTenantIsolation(TestCase):
    """Test tenant isolation in audit logs."""

    def test_separate_chains_per_tenant(self):
        """Test each tenant has independent audit chain."""
        tenant1 = uuid.uuid4()
        tenant2 = uuid.uuid4()

        # Create entries for tenant 1
        for i in range(3):
            SysLogEntry.objects.create(
                tenant_id=tenant1,
                actor="user1@example.com",
                event_name=f"event.{i}",
                payload={},
            )

        # Create entries for tenant 2
        for i in range(3):
            SysLogEntry.objects.create(
                tenant_id=tenant2,
                actor="user2@example.com",
                event_name=f"event.{i}",
                payload={},
            )

        # Each tenant should have separate chains
        tenant1_entries = list(
            SysLogEntry.objects.filter(tenant_id=tenant1).order_by("id")
        )
        tenant2_entries = list(
            SysLogEntry.objects.filter(tenant_id=tenant2).order_by("id")
        )

        self.assertEqual(len(tenant1_entries), 3)
        self.assertEqual(len(tenant2_entries), 3)

        # Each should have own genesis
        self.assertTrue(tenant1_entries[0].is_genesis)
        self.assertTrue(tenant2_entries[0].is_genesis)

        # Chains should be independent
        self.assertNotEqual(
            tenant1_entries[1].previous_hash, tenant2_entries[1].previous_hash
        )

    def test_chain_validation_per_tenant(self):
        """Test chain validation is tenant-specific."""
        tenant1 = uuid.uuid4()
        tenant2 = uuid.uuid4()

        # Create entries for both tenants
        for tenant in [tenant1, tenant2]:
            for i in range(3):
                SysLogEntry.objects.create(
                    tenant_id=tenant,
                    actor=f"user@{tenant}.com",
                    event_name=f"event.{i}",
                    payload={},
                )

        # Tamper with tenant1's chain
        tenant1_entry = SysLogEntry.objects.filter(tenant_id=tenant1).first()
        SysLogEntry.objects.filter(id=tenant1_entry.id).update(
            current_hash="tampered_" + "0" * 56
        )

        # Verify both chains
        result1 = verify_chain_integrity(tenant1)
        result2 = verify_chain_integrity(tenant2)

        # Tenant1 should be invalid, tenant2 should be valid
        self.assertFalse(result1["is_valid"])
        self.assertTrue(result2["is_valid"])

    def test_get_audit_trail_tenant_filtered(self):
        """Test audit trail retrieval is tenant-filtered."""
        tenant1 = uuid.uuid4()
        tenant2 = uuid.uuid4()

        # Create entries for both tenants
        SysLogEntry.objects.create(
            tenant_id=tenant1,
            actor="user1@example.com",
            event_name="event.one",
            payload={},
        )

        SysLogEntry.objects.create(
            tenant_id=tenant2,
            actor="user2@example.com",
            event_name="event.two",
            payload={},
        )

        # Get trail for tenant1
        trail1 = get_audit_trail(tenant1)

        # Should only include tenant1's entries
        self.assertEqual(len(trail1), 1)
        self.assertEqual(trail1[0].tenant_id, tenant1)


class TestGenerateEntry(TestCase):
    """Test generate_entry utility function."""

    def test_generate_entry_basic(self):
        """Test generating a basic audit entry."""
        tenant_id = uuid.uuid4()
        entry = generate_entry(
            tenant_id=tenant_id,
            actor="user@example.com",
            event_name="user.login",
            payload={"ip": "192.168.1.1"},
        )

        self.assertIsNotNone(entry.id)
        self.assertEqual(entry.tenant_id, tenant_id)
        self.assertEqual(entry.actor, "user@example.com")
        self.assertEqual(entry.event_name, "user.login")
        self.assertEqual(entry.payload["ip"], "192.168.1.1")

    def test_generate_entry_with_correlation(self):
        """Test generating entry with correlation ID."""
        correlation_id = uuid.uuid4()
        entry = generate_entry(
            tenant_id=uuid.uuid4(),
            actor="system",
            event_name="batch.process",
            correlation_id=correlation_id,
        )

        self.assertEqual(entry.correlation_id, correlation_id)

    def test_generate_entry_creates_chain(self):
        """Test multiple generated entries form a chain."""
        tenant_id = uuid.uuid4()

        # Generate multiple entries
        entry1 = generate_entry(
            tenant_id=tenant_id, actor="user@example.com", event_name="event.one"
        )

        entry2 = generate_entry(
            tenant_id=tenant_id, actor="user@example.com", event_name="event.two"
        )

        # Should form a chain
        self.assertTrue(entry1.is_genesis)
        self.assertFalse(entry2.is_genesis)
        self.assertEqual(entry2.previous_hash, entry1.current_hash)


class TestGetChainMethods(TestCase):
    """Test chain retrieval methods."""

    def setUp(self):
        """Set up test data."""
        self.tenant_id = uuid.uuid4()

        # Create chain
        for i in range(5):
            SysLogEntry.objects.create(
                tenant_id=self.tenant_id,
                actor="user@example.com",
                event_name=f"event.{i}",
                payload={"index": i},
            )

    def test_get_chain_head(self):
        """Test getting most recent entry."""
        head = SysLogEntry.get_chain_head(self.tenant_id)

        self.assertIsNotNone(head)
        self.assertEqual(head.payload["index"], 4)  # Last entry

    def test_get_genesis(self):
        """Test getting first entry."""
        genesis = SysLogEntry.get_genesis(self.tenant_id)

        self.assertIsNotNone(genesis)
        self.assertTrue(genesis.is_genesis)
        self.assertEqual(genesis.payload["index"], 0)  # First entry


class TestComplianceRequirements(TestCase):
    """Test compliance with SOC 2 CC7.2 / ISO 27001 A.12.7."""

    def test_audit_trail_completeness(self):
        """Test audit trail captures all required fields (SOC 2 CC7.2)."""
        entry = generate_entry(
            tenant_id=uuid.uuid4(),
            actor="user@example.com",
            event_name="sensitive.action",
            payload={"resource": "data", "action": "delete"},
        )

        # Required audit fields
        self.assertIsNotNone(entry.timestamp)
        self.assertIsNotNone(entry.actor)
        self.assertIsNotNone(entry.event_name)
        self.assertIsNotNone(entry.payload)
        self.assertIsNotNone(entry.tenant_id)

    def test_tamper_evidence(self):
        """Test tampering leaves detectable evidence (ISO 27001 A.12.7)."""
        # Create entry
        entry = SysLogEntry.objects.create(
            tenant_id=uuid.uuid4(),
            actor="user@example.com",
            event_name="test.event",
            payload={"data": "original"},
        )

        # Simulate tampering
        SysLogEntry.objects.filter(id=entry.id).update(
            current_hash="tampered_" + "0" * 56
        )

        # Reload and verify - tampering should be detected
        entry.refresh_from_db()
        self.assertFalse(entry.verify_hash())

    def test_immutability_enforcement(self):
        """Test entries cannot be altered (SOC 2 CC7.2)."""
        entry = SysLogEntry.objects.create(
            tenant_id=uuid.uuid4(),
            actor="user@example.com",
            event_name="test.event",
            payload={},
        )

        # Attempt modification
        entry.actor = "attacker@example.com"

        with self.assertRaises(ValidationError):
            entry.save()

    def test_violation_alerting(self):
        """Test violations are recorded and trigger events (SOC 2 CC7.2)."""
        violation = record_integrity_violation(
            tenant_id=uuid.uuid4(),
            violation_type="chain_break",
            entry_id=123,
            details={"severity": "critical"},
        )

        # Violation should be recorded in DB
        self.assertIsNotNone(violation.id)
        self.assertEqual(violation.violation_type, "chain_break")


class TestConcurrentChainWrites(TransactionTestCase):
    """Test advisory lock prevents chain breaks under concurrent writes."""

    def test_concurrent_writes_maintain_chain(self):
        """Concurrent writes to the same tenant serialize via advisory lock.

        Reproduces the race condition that caused the entry 928/929 chain break:
        multiple threads writing simultaneously must not produce duplicate
        previous_hash values.
        """
        import concurrent.futures

        from django.db import close_old_connections

        tenant = uuid.uuid4()

        # Seed the chain
        SysLogEntry.objects.create(
            tenant_id=tenant, actor="setup", event_name="seed", payload={}
        )

        def write_entry(i):
            close_old_connections()
            try:
                SysLogEntry.objects.create(
                    tenant_id=tenant,
                    actor=f"writer-{i}",
                    event_name=f"concurrent.{i}",
                    payload={"i": i},
                )
            finally:
                close_old_connections()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(write_entry, i) for i in range(8)]
            for f in futures:
                f.result()  # raise if any failed

        # Verify the entire chain is intact
        result = verify_chain_integrity(tenant)
        self.assertTrue(
            result["is_valid"],
            f"Chain broken after concurrent writes: {result['violations']}",
        )
        self.assertEqual(result["total_entries"], 9)  # 1 seed + 8 concurrent

        # Verify no duplicate previous_hash values (the exact bug we're preventing)
        entries = list(SysLogEntry.objects.filter(tenant_id=tenant).order_by("id"))
        prev_hashes = [e.previous_hash for e in entries if not e.is_genesis]
        self.assertEqual(
            len(prev_hashes),
            len(set(prev_hashes)),
            "Duplicate previous_hash found — race condition",
        )
