"""
Audit infrastructure functional tests.

Locks IntegrityViolation, DriftViolation, RiskAssessment models,
verify_chain_integrity(), record_integrity_violation(), and get_audit_trail()
with functional assertions mapped to SOC 2 controls.

Compliance: SOC 2 CC7.2, CC7.3, CC3.4
"""

import uuid
from datetime import timedelta
from unittest.mock import patch

from django.db import IntegrityError
from django.test import TestCase
from django.utils import timezone

from syn.audit.models import (
    AgentVote,
    ChangeRequest,
    DriftViolation,
    IntegrityViolation,
    RiskAssessment,
    SysLogEntry,
)
from syn.audit.utils import (
    generate_entry,
    get_audit_trail,
    record_integrity_violation,
    verify_chain_integrity,
)

# =========================================================================
# Helpers
# =========================================================================

_TENANT = str(uuid.uuid4())
_TENANT_B = str(uuid.uuid4())


def _make_drift(**overrides):
    """Create a DriftViolation with sane defaults."""
    defaults = {
        "drift_signature": uuid.uuid4().hex,
        "severity": "HIGH",
        "enforcement_check": "ENC-001",
        "file_path": "syn/audit/models.py",
        "line_number": 42,
        "detected_by": "compliance_runner",
        "violation_message": "Test violation",
        "tenant_id": _TENANT,
    }
    defaults.update(overrides)
    return DriftViolation.objects.create(**defaults)


def _make_cr():
    """Create a minimal ChangeRequest for FK refs."""
    return ChangeRequest.objects.create(
        title="Test change request for risk assessment",
        description="Minimal CR for test scaffolding purposes.",
        change_type="infrastructure",
        author="test",
    )


# =========================================================================
# IntegrityViolation Model (SOC 2 CC7.2)
# =========================================================================


class IntegrityViolationModelTest(TestCase):
    """Lock IntegrityViolation model behavior."""

    def test_create_violation_records_fields(self):
        """Create violation, verify all fields populated (CC7.2)."""
        v = IntegrityViolation.objects.create(
            tenant_id=_TENANT,
            violation_type="hash_mismatch",
            entry_id=1,
            details={"expected": "abc", "actual": "def"},
        )
        v.refresh_from_db()
        self.assertEqual(v.violation_type, "hash_mismatch")
        self.assertEqual(v.tenant_id, uuid.UUID(_TENANT))
        self.assertEqual(v.entry_id, 1)
        self.assertEqual(v.details["expected"], "abc")
        self.assertIsNotNone(v.detected_at)

    def test_violation_types_all_defined(self):
        """4 violation types: hash_mismatch, chain_break, missing_entry, duplicate_hash (CC7.2)."""
        choices = dict(IntegrityViolation._meta.get_field("violation_type").choices)
        expected = {"hash_mismatch", "chain_break", "missing_entry", "duplicate_hash"}
        self.assertEqual(set(choices.keys()), expected)

    def test_default_unresolved(self):
        """New violation: is_resolved=False, resolved_at=None (CC7.2)."""
        v = IntegrityViolation.objects.create(
            tenant_id=_TENANT,
            violation_type="chain_break",
        )
        self.assertFalse(v.is_resolved)
        self.assertIsNone(v.resolved_at)

    def test_resolve_violation(self):
        """Set is_resolved=True + resolved_at + resolution_notes, verify persist (CC7.2)."""
        v = IntegrityViolation.objects.create(
            tenant_id=_TENANT,
            violation_type="hash_mismatch",
        )
        now = timezone.now()
        # IntegrityViolation uses SynaraImmutableLog — update via queryset
        IntegrityViolation.objects.filter(pk=v.pk).update(
            is_resolved=True,
            resolved_at=now,
            resolution_notes="Investigated, false positive.",
        )
        v.refresh_from_db()
        self.assertTrue(v.is_resolved)
        self.assertIsNotNone(v.resolved_at)
        self.assertEqual(v.resolution_notes, "Investigated, false positive.")

    def test_ordering_most_recent_first(self):
        """Default ordering is -detected_at (CC7.2)."""
        ordering = IntegrityViolation._meta.ordering
        self.assertEqual(ordering, ["-detected_at"])

    def test_uuid_primary_key(self):
        """PK is UUID (CC7.2)."""
        v = IntegrityViolation.objects.create(
            tenant_id=_TENANT,
            violation_type="missing_entry",
        )
        self.assertIsInstance(v.pk, uuid.UUID)

    def test_tenant_filter_isolation(self):
        """2 tenants, filter returns only matching (CC7.2)."""
        IntegrityViolation.objects.create(tenant_id=_TENANT, violation_type="hash_mismatch")
        IntegrityViolation.objects.create(tenant_id=_TENANT_B, violation_type="chain_break")
        qs_a = IntegrityViolation.objects.filter(tenant_id=_TENANT)
        qs_b = IntegrityViolation.objects.filter(tenant_id=_TENANT_B)
        self.assertEqual(qs_a.count(), 1)
        self.assertEqual(qs_b.count(), 1)
        self.assertEqual(qs_a.first().violation_type, "hash_mismatch")


# =========================================================================
# DriftViolation Model (SOC 2 CC7.2, CC7.3)
# =========================================================================


class DriftViolationModelTest(TestCase):
    """Lock DriftViolation model behavior."""

    def test_create_drift_violation(self):
        """Create with required fields, verify saved (CC7.2)."""
        dv = _make_drift()
        dv.refresh_from_db()
        self.assertEqual(dv.severity, "HIGH")
        self.assertEqual(dv.enforcement_check, "ENC-001")
        self.assertIsNotNone(dv.detected_at)

    def test_severity_choices(self):
        """4 severity levels: CRITICAL, HIGH, MEDIUM, LOW (CC7.2)."""
        choices = dict(DriftViolation.SEVERITY_CHOICES)
        self.assertEqual(set(choices.keys()), {"CRITICAL", "HIGH", "MEDIUM", "LOW"})

    def test_enforcement_check_choices(self):
        """All 13 check IDs exist (CC7.2)."""
        choices = dict(DriftViolation.ENFORCEMENT_CHECK_CHOICES)
        expected_prefixes = {
            "ENC-001",
            "ENC-002",
            "ENC-003",
            "ENC-004",
            "ENC-005",
            "ENC-006",
            "ENC-007",
            "ENC-008",
            "ENC-009",
            "ENC-010",
            "ENC-011",
            "STD",
            "CAL",
        }
        self.assertEqual(set(choices.keys()), expected_prefixes)

    def test_immutability_blocks_arbitrary_updates(self):
        """save() without update_fields raises ValueError (CC7.3)."""
        dv = _make_drift()
        dv.violation_message = "Changed!"
        with self.assertRaises(ValueError):
            dv.save()

    def test_immutability_allows_resolution_fields(self):
        """save(update_fields=["resolved_at"]) works (CC7.3)."""
        dv = _make_drift()
        dv.resolved_at = timezone.now()
        dv.save(update_fields=["resolved_at"])
        dv.refresh_from_db()
        self.assertIsNotNone(dv.resolved_at)

    def test_delete_blocked(self):
        """delete() raises ValueError (CC7.3)."""
        dv = _make_drift()
        with self.assertRaises(ValueError):
            dv.delete()

    def test_sla_due_date_auto_computed(self):
        """Set remediation_sla_hours=24, verify remediation_due_at computed (CC7.2)."""
        dv = _make_drift(remediation_sla_hours=24)
        dv.refresh_from_db()
        self.assertIsNotNone(dv.remediation_due_at)
        # Should be approximately detected_at + 24h
        delta = dv.remediation_due_at - dv.detected_at
        self.assertAlmostEqual(delta.total_seconds(), 24 * 3600, delta=60)

    def test_is_overdue_true_when_past_due(self):
        """remediation_due_at in past → True (CC7.2)."""
        dv = _make_drift()
        # Force remediation_due_at into the past
        DriftViolation.objects.filter(pk=dv.pk).update(remediation_due_at=timezone.now() - timedelta(hours=1))
        dv.refresh_from_db()
        self.assertTrue(dv.is_overdue())

    def test_is_overdue_false_when_resolved(self):
        """resolved_at set → is_overdue False (CC7.2)."""
        dv = _make_drift()
        dv.resolved_at = timezone.now()
        dv.save(update_fields=["resolved_at"])
        dv.refresh_from_db()
        self.assertFalse(dv.is_overdue())

    def test_drift_signature_unique(self):
        """Duplicate drift_signature raises IntegrityError (CC7.3)."""
        sig = uuid.uuid4().hex
        _make_drift(drift_signature=sig)
        with self.assertRaises(IntegrityError):
            _make_drift(drift_signature=sig)


# =========================================================================
# RiskAssessment compute_aggregate (SOC 2 CC3.4)
# =========================================================================


class RiskAssessmentComputeTest(TestCase):
    """Lock RiskAssessment.compute_aggregate() behavior."""

    def setUp(self):
        self.cr = _make_cr()
        self.ra = RiskAssessment.objects.create(
            change_request=self.cr,
            assessment_type="multi_agent",
        )

    def _vote(self, role, recommendation, scores, conditions=None):
        return AgentVote.objects.create(
            risk_assessment=self.ra,
            agent_role=role,
            recommendation=recommendation,
            risk_scores=scores,
            rationale=f"Test rationale for {role}",
            conditions=conditions or [],
        )

    def test_compute_aggregate_averages_scores(self):
        """2 votes with known scores → averages correct (CC3.4)."""
        self._vote(
            "security_analyst",
            "approve",
            {"security": 2, "availability": 4, "integrity": 3, "confidentiality": 2, "privacy": 1},
        )
        self._vote(
            "architect",
            "approve",
            {"security": 4, "availability": 2, "integrity": 1, "confidentiality": 2, "privacy": 3},
        )
        self.ra.compute_aggregate()
        self.ra.refresh_from_db()
        self.assertAlmostEqual(self.ra.security_score, 3.0)
        self.assertAlmostEqual(self.ra.availability_score, 3.0)
        self.assertAlmostEqual(self.ra.integrity_score, 2.0)
        self.assertAlmostEqual(self.ra.confidentiality_score, 2.0)
        self.assertAlmostEqual(self.ra.privacy_score, 2.0)

    def test_overall_score_is_max_dimension(self):
        """overall = max of 5 dimension averages (CC3.4)."""
        self._vote(
            "security_analyst",
            "approve",
            {"security": 5, "availability": 1, "integrity": 1, "confidentiality": 1, "privacy": 1},
        )
        self.ra.compute_aggregate()
        self.ra.refresh_from_db()
        self.assertAlmostEqual(self.ra.overall_score, 5.0)

    def test_security_analyst_veto(self):
        """security_analyst "reject" → overall = "reject" (CC3.4)."""
        self._vote(
            "security_analyst",
            "reject",
            {"security": 5, "availability": 1, "integrity": 1, "confidentiality": 1, "privacy": 1},
        )
        self._vote(
            "architect",
            "approve",
            {"security": 1, "availability": 1, "integrity": 1, "confidentiality": 1, "privacy": 1},
        )
        self._vote(
            "operations",
            "approve",
            {"security": 1, "availability": 1, "integrity": 1, "confidentiality": 1, "privacy": 1},
        )
        self._vote(
            "quality", "approve", {"security": 1, "availability": 1, "integrity": 1, "confidentiality": 1, "privacy": 1}
        )
        self.ra.compute_aggregate()
        self.ra.refresh_from_db()
        self.assertEqual(self.ra.overall_recommendation, "reject")

    def test_majority_reject(self):
        """3/4 reject → overall = "reject" (CC3.4)."""
        self._vote(
            "architect",
            "reject",
            {"security": 3, "availability": 3, "integrity": 3, "confidentiality": 3, "privacy": 3},
        )
        self._vote(
            "operations",
            "reject",
            {"security": 3, "availability": 3, "integrity": 3, "confidentiality": 3, "privacy": 3},
        )
        self._vote(
            "quality", "reject", {"security": 3, "availability": 3, "integrity": 3, "confidentiality": 3, "privacy": 3}
        )
        self._vote(
            "security_analyst",
            "approve",
            {"security": 1, "availability": 1, "integrity": 1, "confidentiality": 1, "privacy": 1},
        )
        self.ra.compute_aggregate()
        self.ra.refresh_from_db()
        self.assertEqual(self.ra.overall_recommendation, "reject")

    def test_approve_with_conditions_merges(self):
        """Conditions merged from all conditional votes (CC3.4)."""
        self._vote(
            "security_analyst",
            "approve_with_conditions",
            {"security": 2, "availability": 1, "integrity": 1, "confidentiality": 1, "privacy": 1},
            conditions=["Add WAF rule"],
        )
        self._vote(
            "architect",
            "approve_with_conditions",
            {"security": 1, "availability": 2, "integrity": 1, "confidentiality": 1, "privacy": 1},
            conditions=["Update docs"],
        )
        self.ra.compute_aggregate()
        self.ra.refresh_from_db()
        self.assertEqual(self.ra.overall_recommendation, "approve_with_conditions")
        self.assertIn("Add WAF rule", self.ra.conditions)
        self.assertIn("Update docs", self.ra.conditions)

    def test_unanimous_approve(self):
        """All "approve" → overall = "approve" (CC3.4)."""
        for role in ("security_analyst", "architect", "operations", "quality"):
            self._vote(
                role, "approve", {"security": 1, "availability": 1, "integrity": 1, "confidentiality": 1, "privacy": 1}
            )
        self.ra.compute_aggregate()
        self.ra.refresh_from_db()
        self.assertEqual(self.ra.overall_recommendation, "approve")

    def test_no_votes_returns_early(self):
        """No votes doesn't crash or change scores (CC3.4)."""
        self.ra.compute_aggregate()
        self.ra.refresh_from_db()
        # Scores remain at defaults
        self.assertEqual(self.ra.overall_score, 0)


# =========================================================================
# verify_chain_integrity (SOC 2 CC7.2)
# =========================================================================


class VerifyChainIntegrityTest(TestCase):
    """Lock verify_chain_integrity() behavior."""

    def setUp(self):
        self.tenant = str(uuid.uuid4())

    def test_empty_chain_valid(self):
        """No entries → is_valid=True (CC7.2)."""
        result = verify_chain_integrity(self.tenant)
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["total_entries"], 0)

    def test_single_entry_valid(self):
        """One genesis entry validates (CC7.2)."""
        generate_entry(self.tenant, "test", "test.event", {"key": "val"})
        result = verify_chain_integrity(self.tenant)
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["total_entries"], 1)
        self.assertTrue(result["genesis_valid"])

    def test_tampered_hash_detected(self):
        """Alter hash → hash_mismatch detected (CC7.2)."""
        entry = generate_entry(self.tenant, "test", "test.event", {"key": "val"})
        # Tamper with the hash directly in DB
        SysLogEntry.objects.filter(pk=entry.pk).update(current_hash="0" * 64)
        result = verify_chain_integrity(self.tenant)
        self.assertFalse(result["is_valid"])
        violation_types = [v["type"] for v in result["violations"]]
        self.assertIn("hash_mismatch", violation_types)

    def test_broken_chain_detected(self):
        """Break previous_hash → chain_break detected (CC7.2)."""
        generate_entry(self.tenant, "test", "event1", {})
        entry2 = generate_entry(self.tenant, "test", "event2", {})
        # Break the chain link
        SysLogEntry.objects.filter(pk=entry2.pk).update(previous_hash="f" * 64)
        result = verify_chain_integrity(self.tenant)
        self.assertFalse(result["is_valid"])
        violation_types = [v["type"] for v in result["violations"]]
        self.assertIn("chain_break", violation_types)

    def test_valid_multi_entry_chain(self):
        """5 sequential entries → valid (CC7.2)."""
        for i in range(5):
            generate_entry(self.tenant, "test", f"event.{i}", {"seq": i})
        result = verify_chain_integrity(self.tenant)
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["total_entries"], 5)

    def test_return_schema(self):
        """Result has: is_valid, total_entries, violations, genesis_valid, message (CC7.2)."""
        result = verify_chain_integrity(self.tenant)
        for key in ("is_valid", "total_entries", "violations", "genesis_valid", "message"):
            self.assertIn(key, result, f"Missing key: {key}")


# =========================================================================
# record_integrity_violation (SOC 2 CC7.2)
# =========================================================================


class RecordIntegrityViolationTest(TestCase):
    """Lock record_integrity_violation() behavior."""

    @patch("syn.audit.events.emit_audit_event", return_value=True)
    def test_creates_violation_record(self, mock_emit):
        """record_integrity_violation() creates record (CC7.2)."""
        v = record_integrity_violation(_TENANT, "hash_mismatch", entry_id=99)
        self.assertIsNotNone(v.pk)
        self.assertEqual(v.violation_type, "hash_mismatch")

    @patch("syn.audit.events.emit_audit_event", return_value=True)
    def test_details_persisted(self, mock_emit):
        """details dict stored and retrievable (CC7.2)."""
        details = {"expected": "abc123", "actual": "def456"}
        v = record_integrity_violation(_TENANT, "chain_break", details=details)
        v.refresh_from_db()
        self.assertEqual(v.details["expected"], "abc123")
        self.assertEqual(v.details["actual"], "def456")

    @patch("syn.audit.events.emit_audit_event", return_value=True)
    def test_violation_type_persisted(self, mock_emit):
        """type matches input (CC7.2)."""
        for vtype in ("hash_mismatch", "chain_break", "missing_entry", "duplicate_hash"):
            v = record_integrity_violation(_TENANT, vtype)
            self.assertEqual(v.violation_type, vtype)


# =========================================================================
# get_audit_trail (SOC 2 CC7.2)
# =========================================================================


class GetAuditTrailTest(TestCase):
    """Lock get_audit_trail() behavior."""

    def setUp(self):
        self.tenant = str(uuid.uuid4())
        self.corr = str(uuid.uuid4())
        generate_entry(self.tenant, "alice", "user.login", {"ip": "1.2.3.4"}, self.corr)
        generate_entry(self.tenant, "bob", "user.logout", {"reason": "timeout"})
        generate_entry(self.tenant, "alice", "data.export", {"format": "csv"})

    def test_filter_by_event_name(self):
        """event_name filter works (CC7.2)."""
        result = get_audit_trail(self.tenant, event_name="user.login")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].event_name, "user.login")

    def test_filter_by_actor(self):
        """actor filter works (CC7.2)."""
        result = get_audit_trail(self.tenant, actor="alice")
        self.assertEqual(len(result), 2)
        for entry in result:
            self.assertEqual(entry.actor, "alice")

    def test_filter_by_correlation_id(self):
        """correlation filter works (CC7.2)."""
        result = get_audit_trail(self.tenant, correlation_id=self.corr)
        self.assertEqual(len(result), 1)
        self.assertEqual(str(result[0].correlation_id), self.corr)

    def test_limit_respected(self):
        """limit=1 returns ≤1 (CC7.2)."""
        result = get_audit_trail(self.tenant, limit=1)
        self.assertLessEqual(len(result), 1)
