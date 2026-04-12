"""
INC-001 compliance tests: Incident Response Management Standard.

Tests verify Incident model structure, IncidentLog immutability, lifecycle
timestamps, SLA breach properties, compliance check integration, notification
types, and standard existence.

Standard: INC-001
"""

import os
import re
from pathlib import Path

from django.test import SimpleTestCase

WEB_ROOT = Path(os.path.dirname(__file__)).parent.parent.parent
STANDARDS_DIR = WEB_ROOT.parent.parent.parent / "docs" / "standards"


def _read(path):
    try:
        return Path(path).read_text(errors="ignore")
    except Exception:
        return ""


# ── §1: Standard Exists ─────────────────────────────────────────────────


class INC001StandardTest(SimpleTestCase):
    """INC-001 §1: Standard document exists and has required sections."""

    def setUp(self):
        self.inc_standard = _read(STANDARDS_DIR / "INC-001.md")
        self.assertGreater(len(self.inc_standard), 0, "INC-001.md not found")

    def test_standard_exists(self):
        """INC-001.md exists in docs/standards/."""
        self.assertIn("INC-001", self.inc_standard)

    def test_has_scope_section(self):
        """INC-001 contains scope section."""
        self.assertIn("SCOPE", self.inc_standard.upper())

    def test_has_classification_section(self):
        """INC-001 contains classification section."""
        self.assertIn("CLASSIFICATION", self.inc_standard.upper())

    def test_has_lifecycle_section(self):
        """INC-001 contains lifecycle section."""
        self.assertIn("LIFECYCLE", self.inc_standard.upper())

    def test_has_sla_targets_section(self):
        """INC-001 contains SLA targets section."""
        self.assertIn("SLA TARGET", self.inc_standard.upper())

    def test_has_post_mortem_section(self):
        """INC-001 contains post-mortem section."""
        self.assertIn("POST-MORTEM", self.inc_standard.upper())

    def test_has_compliance_mapping(self):
        """INC-001 maps to SOC 2 CC7.1 and CC7.4."""
        self.assertIn("CC7.1", self.inc_standard)
        self.assertIn("CC7.4", self.inc_standard)

    def test_has_acceptance_criteria(self):
        """INC-001 contains acceptance criteria section."""
        self.assertIn("ACCEPTANCE CRITERIA", self.inc_standard.upper())

    def test_has_machine_readable_hooks(self):
        """INC-001 contains <!-- assert: --> hooks per DOC-001 §7."""
        assert_count = self.inc_standard.count("<!-- assert:")
        self.assertGreaterEqual(assert_count, 5, f"Only {assert_count} assertion hooks found")

    def test_has_test_hooks(self):
        """INC-001 contains <!-- test: --> hooks linking to test_incidents."""
        self.assertIn("test_incidents", self.inc_standard)

    def test_cross_references_sla_001(self):
        """INC-001 cross-references SLA-001."""
        self.assertIn("SLA-001", self.inc_standard)

    def test_cross_references_chg_001(self):
        """INC-001 cross-references CHG-001."""
        self.assertIn("CHG-001", self.inc_standard)


# ── §3: Incident Model ──────────────────────────────────────────────────


class IncidentModelTest(SimpleTestCase):
    """INC-001 §3-§6: Incident model has required fields and properties."""

    def setUp(self):
        self.models_src = _read(WEB_ROOT / "syn" / "audit" / "models.py")
        self.assertGreater(len(self.models_src), 0, "models.py not found")

    def test_model_exists(self):
        """Incident model class is defined in syn/audit/models.py."""
        self.assertIn("class Incident(models.Model)", self.models_src)

    def test_uuid_primary_key(self):
        """Incident uses UUID primary key per DAT-001."""
        # Find Incident class body
        m = re.search(
            r"class Incident\(models\.Model\).*?(?=\nclass |\Z)",
            self.models_src,
            re.DOTALL,
        )
        self.assertIsNotNone(m)
        body = m.group()
        self.assertIn("UUIDField(primary_key=True", body)

    def test_severity_choices(self):
        """Incident model has severity field with critical/high/medium/low."""
        m = re.search(
            r"class Incident\(models\.Model\).*?(?=\nclass |\Z)",
            self.models_src,
            re.DOTALL,
        )
        body = m.group()
        self.assertIn("SEVERITY_CHOICES", body)
        for level in ["critical", "high", "medium", "low"]:
            self.assertIn(f'"{level}"', body)

    def test_status_choices(self):
        """Incident model has status field with lifecycle states."""
        m = re.search(
            r"class Incident\(models\.Model\).*?(?=\nclass |\Z)",
            self.models_src,
            re.DOTALL,
        )
        body = m.group()
        self.assertIn("STATUS_CHOICES", body)
        for state in [
            "detected",
            "acknowledged",
            "investigating",
            "mitigating",
            "resolved",
            "post_mortem",
            "closed",
        ]:
            self.assertIn(f'"{state}"', body)

    def test_category_choices(self):
        """Incident model has category field for classification."""
        m = re.search(
            r"class Incident\(models\.Model\).*?(?=\nclass |\Z)",
            self.models_src,
            re.DOTALL,
        )
        body = m.group()
        self.assertIn("CATEGORY_CHOICES", body)
        for cat in ["outage", "degradation", "security", "data", "dependency", "other"]:
            self.assertIn(f'"{cat}"', body)

    def test_lifecycle_timestamps(self):
        """Incident model has lifecycle timestamp fields for SLA measurement."""
        m = re.search(
            r"class Incident\(models\.Model\).*?(?=\nclass |\Z)",
            self.models_src,
            re.DOTALL,
        )
        body = m.group()
        for ts_field in [
            "detected_at",
            "acknowledged_at",
            "investigating_at",
            "mitigating_at",
            "resolved_at",
            "closed_at",
        ]:
            self.assertIn(ts_field, body, f"Missing lifecycle timestamp: {ts_field}")

    def test_actor_fields(self):
        """Incident model has reported_by and assigned_to fields."""
        m = re.search(
            r"class Incident\(models\.Model\).*?(?=\nclass |\Z)",
            self.models_src,
            re.DOTALL,
        )
        body = m.group()
        self.assertIn("reported_by", body)
        self.assertIn("assigned_to", body)

    def test_change_request_fk(self):
        """Incident links to ChangeRequest for remediation (INC-001 §8.3)."""
        m = re.search(
            r"class Incident\(models\.Model\).*?(?=\nclass |\Z)",
            self.models_src,
            re.DOTALL,
        )
        body = m.group()
        self.assertIn("change_request", body)
        self.assertIn("ChangeRequest", body)

    def test_resolution_fields(self):
        """Incident has root_cause, resolution_summary, and post_mortem_notes."""
        m = re.search(
            r"class Incident\(models\.Model\).*?(?=\nclass |\Z)",
            self.models_src,
            re.DOTALL,
        )
        body = m.group()
        for field in ["root_cause", "resolution_summary", "post_mortem_notes"]:
            self.assertIn(field, body, f"Missing resolution field: {field}")

    def test_sla_properties(self):
        """Incident has SLA breach detection properties."""
        m = re.search(
            r"class Incident\(models\.Model\).*?(?=\nclass |\Z)",
            self.models_src,
            re.DOTALL,
        )
        body = m.group()
        for prop in [
            "ack_elapsed_hours",
            "resolution_elapsed_hours",
            "is_ack_sla_breached",
            "is_resolution_sla_breached",
        ]:
            self.assertIn(prop, body, f"Missing SLA property: {prop}")

    def test_db_table_name(self):
        """Incident uses syn_audit_incident table name."""
        m = re.search(
            r"class Incident\(models\.Model\).*?(?=\nclass |\Z)",
            self.models_src,
            re.DOTALL,
        )
        body = m.group()
        self.assertIn('db_table = "syn_audit_incident"', body)

    def test_correlation_id(self):
        """Incident has correlation_id for audit trail linkage."""
        m = re.search(
            r"class Incident\(models\.Model\).*?(?=\nclass |\Z)",
            self.models_src,
            re.DOTALL,
        )
        body = m.group()
        self.assertIn("correlation_id", body)


# ── §5.2: IncidentLog ───────────────────────────────────────────────────


class IncidentLogTest(SimpleTestCase):
    """INC-001 §5.2: IncidentLog model is immutable and records transitions."""

    def setUp(self):
        self.models_src = _read(WEB_ROOT / "syn" / "audit" / "models.py")

    def test_model_exists(self):
        """IncidentLog model class is defined."""
        self.assertIn("class IncidentLog(models.Model)", self.models_src)

    def test_action_choices(self):
        """IncidentLog has ACTION_CHOICES with state transitions and events."""
        m = re.search(
            r"class IncidentLog\(models\.Model\).*?(?=\nclass |\Z)",
            self.models_src,
            re.DOTALL,
        )
        body = m.group()
        self.assertIn("ACTION_CHOICES", body)
        for action in [
            "detected",
            "acknowledged",
            "escalated",
            "comment",
            "severity_changed",
        ]:
            self.assertIn(f'"{action}"', body)

    def test_incident_fk(self):
        """IncidentLog has FK to Incident."""
        m = re.search(
            r"class IncidentLog\(models\.Model\).*?(?=\nclass |\Z)",
            self.models_src,
            re.DOTALL,
        )
        body = m.group()
        self.assertIn("ForeignKey", body)
        self.assertIn("Incident", body)

    def test_state_transition_fields(self):
        """IncidentLog records from_state and to_state."""
        m = re.search(
            r"class IncidentLog\(models\.Model\).*?(?=\nclass |\Z)",
            self.models_src,
            re.DOTALL,
        )
        body = m.group()
        self.assertIn("from_state", body)
        self.assertIn("to_state", body)

    def test_actor_field(self):
        """IncidentLog records who made the change."""
        m = re.search(
            r"class IncidentLog\(models\.Model\).*?(?=\nclass |\Z)",
            self.models_src,
            re.DOTALL,
        )
        body = m.group()
        self.assertIn("actor", body)

    def test_immutability(self):
        """IncidentLog save() raises error on update, delete() raises error."""
        m = re.search(
            r"class IncidentLog\(models\.Model\).*?(?=\nclass |\Z)",
            self.models_src,
            re.DOTALL,
        )
        body = m.group()
        # save() blocks updates
        self.assertIn("ValidationError", body)
        # delete() is blocked
        self.assertIn("ValueError", body)
        self.assertIn("cannot be deleted", body.lower())

    def test_db_table_name(self):
        """IncidentLog uses syn_audit_incident_log table name."""
        m = re.search(
            r"class IncidentLog\(models\.Model\).*?(?=\nclass |\Z)",
            self.models_src,
            re.DOTALL,
        )
        body = m.group()
        self.assertIn('db_table = "syn_audit_incident_log"', body)

    def test_default_permissions(self):
        """IncidentLog restricts permissions to add and view only."""
        m = re.search(
            r"class IncidentLog\(models\.Model\).*?(?=\nclass |\Z)",
            self.models_src,
            re.DOTALL,
        )
        body = m.group()
        self.assertIn("default_permissions", body)


# ── §6: SLA Measurement ─────────────────────────────────────────────────


class IncidentSLAMeasurementTest(SimpleTestCase):
    """INC-001 §6: _measure_incident_response dispatches correctly."""

    def setUp(self):
        self.compliance_src = _read(WEB_ROOT / "syn" / "audit" / "compliance.py")
        self.assertGreater(len(self.compliance_src), 0, "compliance.py not found")

    def test_measure_incident_response_not_stub(self):
        """_measure_incident_response is a real implementation, not a stub."""
        fn_match = re.search(
            r"def _measure_incident_response\(.*?(?=\ndef |\Z)",
            self.compliance_src,
            re.DOTALL,
        )
        self.assertIsNotNone(fn_match)
        fn_body = fn_match.group()
        # Should not just return unmeasurable
        self.assertNotIn(
            'return {"status": "unmeasurable"',
            fn_body.split("\n")[1] if len(fn_body.split("\n")) > 1 else "",
        )
        # Should reference Incident model
        self.assertIn("Incident", fn_body)

    def test_measure_incident_response_queries_model(self):
        """_measure_incident_response queries the Incident model."""
        fn_match = re.search(
            r"def _measure_incident_response\(.*?(?=\ndef |\Z)",
            self.compliance_src,
            re.DOTALL,
        )
        fn_body = fn_match.group()
        self.assertIn("Incident.objects.filter", fn_body)

    def test_measure_dispatches_ack_vs_resolution(self):
        """_measure_incident_response differentiates ack vs resolution SLAs."""
        fn_match = re.search(
            r"def _measure_incident_response\(.*?(?=\ndef |\Z)",
            self.compliance_src,
            re.DOTALL,
        )
        fn_body = fn_match.group()
        self.assertIn("ack", fn_body)
        self.assertIn("is_ack_sla_breached", fn_body)
        self.assertIn("is_resolution_sla_breached", fn_body)

    def test_measure_calculates_compliance_percentage(self):
        """_measure_incident_response calculates compliance percentage."""
        fn_match = re.search(
            r"def _measure_incident_response\(.*?(?=\ndef |\Z)",
            self.compliance_src,
            re.DOTALL,
        )
        fn_body = fn_match.group()
        self.assertIn("compliance_pct", fn_body)

    def test_no_incidents_returns_met(self):
        """_measure_incident_response returns 'met' when no incidents exist."""
        fn_match = re.search(
            r"def _measure_incident_response\(.*?(?=\ndef |\Z)",
            self.compliance_src,
            re.DOTALL,
        )
        fn_body = fn_match.group()
        self.assertIn("No incidents", fn_body)


# ── Compliance Check: incident_readiness ─────────────────────────────────


class IncidentReadinessTest(SimpleTestCase):
    """INC-001: incident_readiness check references INC-001 and Incident model."""

    def setUp(self):
        self.compliance_src = _read(WEB_ROOT / "syn" / "audit" / "compliance.py")

    def test_incident_readiness_registered(self):
        """incident_readiness check is registered in compliance.py."""
        self.assertIn('"incident_readiness"', self.compliance_src)

    def test_checks_inc_001_standard(self):
        """incident_readiness verifies INC-001 standard exists."""
        fn_match = re.search(
            r"def check_incident_readiness\(.*?(?=\ndef |\Z)",
            self.compliance_src,
            re.DOTALL,
        )
        self.assertIsNotNone(fn_match)
        fn_body = fn_match.group()
        self.assertIn("INC-001", fn_body)

    def test_checks_incident_model(self):
        """incident_readiness verifies Incident model is accessible."""
        fn_match = re.search(
            r"def check_incident_readiness\(.*?(?=\ndef |\Z)",
            self.compliance_src,
            re.DOTALL,
        )
        fn_body = fn_match.group()
        self.assertIn("Incident", fn_body)

    def test_soc2_cc74_attached(self):
        """incident_readiness maps to SOC 2 CC7.4."""
        # Find the register decorator
        m = re.search(r"@register\([^)]*incident_readiness[^)]*\)", self.compliance_src)
        self.assertIsNotNone(m)
        self.assertIn("CC7.4", m.group())


# ── Notification Types ───────────────────────────────────────────────────


class IncidentNotificationTest(SimpleTestCase):
    """INC-001 §7: Notification types for incident events."""

    def setUp(self):
        self.notif_src = _read(WEB_ROOT / "notifications" / "models.py")
        self.assertGreater(len(self.notif_src), 0, "notifications/models.py not found")

    def test_incident_created_type(self):
        """incident_created notification type exists."""
        self.assertIn("INCIDENT_CREATED", self.notif_src)
        self.assertIn('"incident_created"', self.notif_src)

    def test_incident_escalated_type(self):
        """incident_escalated notification type exists."""
        self.assertIn("INCIDENT_ESCALATED", self.notif_src)
        self.assertIn('"incident_escalated"', self.notif_src)

    def test_incident_resolved_type(self):
        """incident_resolved notification type exists."""
        self.assertIn("INCIDENT_RESOLVED", self.notif_src)
        self.assertIn('"incident_resolved"', self.notif_src)


# ── API Endpoints ────────────────────────────────────────────────────────


class IncidentAPITest(SimpleTestCase):
    """INC-001: API endpoints for incident management."""

    def setUp(self):
        self.views_src = _read(WEB_ROOT / "api" / "internal_views.py")
        self.urls_src = _read(WEB_ROOT / "api" / "urls.py")
        self.assertGreater(len(self.views_src), 0, "internal_views.py not found")
        self.assertGreater(len(self.urls_src), 0, "urls.py not found")

    def test_incident_list_endpoint(self):
        """api_incident_list function exists."""
        self.assertIn("def api_incident_list(", self.views_src)

    def test_incident_create_endpoint(self):
        """api_incident_create function exists."""
        self.assertIn("def api_incident_create(", self.views_src)

    def test_incident_detail_endpoint(self):
        """api_incident_detail function exists."""
        self.assertIn("def api_incident_detail(", self.views_src)

    def test_incident_transition_endpoint(self):
        """api_incident_transition function exists."""
        self.assertIn("def api_incident_transition(", self.views_src)

    def test_url_routes_registered(self):
        """Incident URL routes are registered in urls.py."""
        self.assertIn("internal/incidents/", self.urls_src)
        self.assertIn("internal/incidents/create/", self.urls_src)
        self.assertIn("internal_incident_create", self.urls_src)
        self.assertIn("internal_incident_transition", self.urls_src)

    def test_transition_creates_log(self):
        """api_incident_transition creates IncidentLog entries."""
        fn_match = re.search(
            r"def api_incident_transition\(.*?(?=\ndef |\Z)",
            self.views_src,
            re.DOTALL,
        )
        self.assertIsNotNone(fn_match)
        fn_body = fn_match.group()
        self.assertIn("IncidentLog", fn_body)


# ── SLA-001 Integration ─────────────────────────────────────────────────


class SLA001IntegrationTest(SimpleTestCase):
    """INC-001: SLA-001 incident SLAs are automated."""

    def setUp(self):
        self.sla_standard = _read(STANDARDS_DIR / "SLA-001.md")
        self.assertGreater(len(self.sla_standard), 0, "SLA-001.md not found")

    def test_incident_slas_automated(self):
        """All incident response SLAs in SLA-001 are measurement=automated."""
        # Find all incident_response SLA tags
        incident_tags = re.findall(
            r"<!--\s*sla:.*?metric=incident_response.*?-->",
            self.sla_standard,
        )
        self.assertGreater(len(incident_tags), 0, "No incident_response SLA tags found")
        for tag in incident_tags:
            self.assertIn(
                "measurement=automated",
                tag,
                f"Incident SLA not automated: {tag[:80]}",
            )

    def test_no_manual_incident_measurement(self):
        """No incident SLA tags have measurement=manual."""
        incident_tags = re.findall(
            r"<!--\s*sla:.*?metric=incident_response.*?-->",
            self.sla_standard,
        )
        for tag in incident_tags:
            self.assertNotIn(
                "measurement=manual",
                tag,
                f"Incident SLA still manual: {tag[:80]}",
            )
