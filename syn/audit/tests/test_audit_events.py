"""
Audit event catalog, redaction, and payload builder functional tests.

Locks event catalog schema, PII/sensitive field redaction, and
payload builder output against SOC 2 controls.

Compliance: SOC 2 CC7.2, CC4.1, CC6.1
"""

import re
import uuid

from django.test import SimpleTestCase

from syn.audit.events import (
    AUDIT_EVENTS,
    build_chain_verified_payload,
    build_drift_sla_breached_payload,
    build_drift_violation_payload,
    build_entry_created_payload,
    build_genesis_created_payload,
    build_governance_integrity_violation_payload,
    build_integrity_violation_payload,
    build_violation_resolved_payload,
    redact_event_payload,
)

# =========================================================================
# Event Catalog (SOC 2 CC7.2, CC4.1)
# =========================================================================

_DOTTED_RE = re.compile(r"^[a-z]+\.[a-z_]+(\.[a-z_]+)?$")


class EventCatalogTest(SimpleTestCase):
    """Lock event catalog schema."""

    def test_all_events_have_required_keys(self):
        """Every event: description, category, audit, priority, siem_forward, payload_schema (CC7.2)."""
        required = {
            "description",
            "category",
            "audit",
            "priority",
            "siem_forward",
            "payload_schema",
        }
        for name, event in AUDIT_EVENTS.items():
            with self.subTest(event=name):
                missing = required - set(event.keys())
                self.assertEqual(missing, set(), f"Event '{name}' missing keys: {missing}")

    def test_critical_events_siem_forwarded(self):
        """priority=1 → siem_forward=True (CC4.1)."""
        for name, event in AUDIT_EVENTS.items():
            if event["priority"] == 1:
                with self.subTest(event=name):
                    self.assertTrue(
                        event["siem_forward"],
                        f"Priority-1 event '{name}' not SIEM forwarded",
                    )

    def test_event_names_dotted_convention(self):
        """Names match [a-z]+.[a-z_]+.[a-z_]+ (CC7.2)."""
        for name in AUDIT_EVENTS:
            with self.subTest(event=name):
                self.assertRegex(name, _DOTTED_RE, f"Event name '{name}' violates convention")

    def test_payload_schemas_have_required_fields(self):
        """Each schema has 'required' array (CC7.2)."""
        for name, event in AUDIT_EVENTS.items():
            schema = event["payload_schema"]
            with self.subTest(event=name):
                self.assertIn("required", schema, f"Event '{name}' schema missing 'required'")
                self.assertIsInstance(schema["required"], list)
                self.assertGreater(
                    len(schema["required"]),
                    0,
                    f"Event '{name}' has empty required list",
                )

    def test_no_recursive_audit(self):
        """audit.entry.created has audit=False (CC7.2)."""
        self.assertFalse(AUDIT_EVENTS["audit.entry.created"]["audit"])


# =========================================================================
# Redaction (SOC 2 CC6.1)
# =========================================================================


class RedactEventPayloadTest(SimpleTestCase):
    """Lock redact_event_payload() behavior."""

    def test_redacts_password_field(self):
        """password → ***REDACTED*** (CC6.1)."""
        payload = {"user": "alice", "password": "s3cret"}
        result, count, tags = redact_event_payload(payload)
        self.assertEqual(result["password"], "***REDACTED***")
        self.assertEqual(result["user"], "alice")

    def test_redacts_api_key_field(self):
        """api_key → ***REDACTED*** (CC6.1)."""
        payload = {"api_key": "sk-abc123"}
        result, count, tags = redact_event_payload(payload)
        self.assertEqual(result["api_key"], "***REDACTED***")

    def test_masks_email_pii(self):
        """user@example.com → us***@example.com (CC6.1)."""
        payload = {"email": "user@example.com"}
        result, count, tags = redact_event_payload(payload)
        self.assertEqual(result["email"], "us***@example.com")

    def test_masks_short_pii(self):
        """Short PII string → *** (CC6.1)."""
        payload = {"name": "Jo"}
        result, count, tags = redact_event_payload(payload)
        self.assertEqual(result["name"], "***")

    def test_nested_dict_redaction(self):
        """Nested sensitive fields redacted (CC6.1)."""
        payload = {
            "outer": "safe",
            "nested": {"password": "hidden", "data": "visible"},
        }
        result, count, tags = redact_event_payload(payload)
        self.assertEqual(result["nested"]["password"], "***REDACTED***")
        self.assertEqual(result["nested"]["data"], "visible")

    def test_list_of_dicts_redaction(self):
        """List items redacted (CC6.1)."""
        payload = {
            "items": [
                {"token": "abc123", "id": 1},
                {"token": "def456", "id": 2},
            ]
        }
        result, count, tags = redact_event_payload(payload)
        for item in result["items"]:
            self.assertEqual(item["token"], "***REDACTED***")
            self.assertIsInstance(item["id"], int)

    def test_redaction_count_accurate(self):
        """Correct count returned (CC6.1)."""
        payload = {"password": "a", "secret": "b", "api_key": "c", "safe": "d"}
        result, count, tags = redact_event_payload(payload)
        self.assertEqual(count, 3)

    def test_sensitivity_tags_populated(self):
        """SENSITIVE/PII tags added (CC6.1)."""
        payload = {"password": "secret", "email": "user@example.com"}
        result, count, tags = redact_event_payload(payload)
        self.assertIn("SENSITIVE", tags)
        self.assertIn("PII", tags)


# =========================================================================
# Payload Builders (SOC 2 CC7.2)
# =========================================================================


class _MockEntry:
    """Lightweight mock for SysLogEntry."""

    def __init__(self, **kwargs):
        self.id = kwargs.get("id", 1)
        self.correlation_id = kwargs.get("correlation_id", uuid.uuid4())
        self.tenant_id = kwargs.get("tenant_id", str(uuid.uuid4()))
        self.event_name = kwargs.get("event_name", "test.event")
        self.actor = kwargs.get("actor", "test_actor")
        self.current_hash = kwargs.get("current_hash", "a" * 64)
        self.previous_hash = kwargs.get("previous_hash", "0" * 64)
        self.is_genesis = kwargs.get("is_genesis", True)


class _MockViolation:
    """Lightweight mock for IntegrityViolation."""

    def __init__(self, **kwargs):
        self.id = kwargs.get("id", uuid.uuid4())
        self.tenant_id = kwargs.get("tenant_id", str(uuid.uuid4()))
        self.violation_type = kwargs.get("violation_type", "hash_mismatch")
        self.entry_id = kwargs.get("entry_id", 42)
        self.details = kwargs.get("details", {})
        self.resolution_notes = kwargs.get("resolution_notes", "")


class _MockDrift:
    """Lightweight mock for DriftViolation."""

    def __init__(self, **kwargs):
        self.id = kwargs.get("id", uuid.uuid4())
        self.drift_signature = kwargs.get("drift_signature", "sig123")
        self.correlation_id = kwargs.get("correlation_id", uuid.uuid4())
        self.tenant_id = kwargs.get("tenant_id", uuid.uuid4())
        self.enforcement_check = kwargs.get("enforcement_check", "ENC-001")
        self.severity = kwargs.get("severity", "HIGH")
        self.file_path = kwargs.get("file_path", "test.py")
        self.line_number = kwargs.get("line_number", 10)
        self.function_name = kwargs.get("function_name", "test_func")
        self.violation_message = kwargs.get("violation_message", "Test violation")
        self.git_commit_sha = kwargs.get("git_commit_sha", "abc123")
        self.detected_by = kwargs.get("detected_by", "runner")
        self.is_auto_fix_safe = kwargs.get("is_auto_fix_safe", False)
        self.auto_fix_safe = kwargs.get("auto_fix_safe", False)
        self.remediation_script = kwargs.get("remediation_script", None)
        self.remediation_due_at = kwargs.get("remediation_due_at", None)
        self.resolved_by = kwargs.get("resolved_by", None)
        self.resolution_notes = kwargs.get("resolution_notes", "")
        self.governance_rule_id = kwargs.get("governance_rule_id", None)


class PayloadBuilderTest(SimpleTestCase):
    """Lock payload builder output schemas."""

    def test_entry_created_payload_schema(self):
        """Has entry_id, tenant_id, event_name, actor (CC7.2)."""
        entry = _MockEntry()
        payload = build_entry_created_payload(entry)
        for key in ("entry_id", "tenant_id", "event_name", "actor"):
            self.assertIn(key, payload, f"Missing key: {key}")

    def test_genesis_created_payload_schema(self):
        """Has entry_id, tenant_id, current_hash (CC7.2)."""
        entry = _MockEntry()
        payload = build_genesis_created_payload(entry)
        for key in ("entry_id", "tenant_id", "current_hash"):
            self.assertIn(key, payload, f"Missing key: {key}")

    def test_integrity_violation_payload_schema(self):
        """Has violation_id, tenant_id, violation_type (CC7.2)."""
        v = _MockViolation()
        payload = build_integrity_violation_payload(v)
        for key in ("violation_id", "tenant_id", "violation_type"):
            self.assertIn(key, payload, f"Missing key: {key}")

    def test_drift_violation_payload_schema(self):
        """Has drift_id, enforcement_check, severity (CC7.2)."""
        d = _MockDrift()
        payload = build_drift_violation_payload(d)
        for key in ("drift_id", "enforcement_check", "severity"):
            self.assertIn(key, payload, f"Missing key: {key}")

    def test_chain_verified_payload_schema(self):
        """Has tenant_id, total_entries, is_valid (CC7.2)."""
        payload = build_chain_verified_payload(
            tenant_id="t1",
            total_entries=10,
            is_valid=True,
        )
        for key in ("tenant_id", "total_entries", "is_valid"):
            self.assertIn(key, payload, f"Missing key: {key}")

    def test_governance_payload_requires_investigation(self):
        """requires_investigation defaults True (CC7.2)."""
        v = _MockViolation()
        payload = build_governance_integrity_violation_payload(v)
        self.assertTrue(payload["requires_investigation"])

    def test_drift_sla_breached_hours_overdue(self):
        """Has hours_overdue field (CC7.2)."""
        d = _MockDrift()
        payload = build_drift_sla_breached_payload(d, hours_overdue=12.5)
        self.assertIn("hours_overdue", payload)
        self.assertEqual(payload["hours_overdue"], 12.5)

    def test_violation_resolved_payload_schema(self):
        """Has violation_id, resolved_by (CC7.2)."""
        v = _MockViolation()
        payload = build_violation_resolved_payload(v, resolved_by="admin")
        for key in ("violation_id", "resolved_by"):
            self.assertIn(key, payload, f"Missing key: {key}")
        self.assertEqual(payload["resolved_by"], "admin")
