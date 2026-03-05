"""
CHG-001 compliance tests: Change Management Standard.

Tests ChangeRequest, ChangeLog, RiskAssessment, and AgentVote models
for field definitions, lifecycle states, and UUID linking.

Standard: CHG-001
"""

import unittest

from django.test import SimpleTestCase

from syn.audit.models import AgentVote, ChangeLog, ChangeRequest, RiskAssessment


class ChangeTypesTest(SimpleTestCase):
    """CHG-001 §4.1: ChangeRequest has change_type field with categories."""

    REQUIRED_TYPES = [
        "feature",
        "enhancement",
        "bugfix",
        "hotfix",
        "security",
        "infrastructure",
        "migration",
        "documentation",
        "plan",
        "debt",
    ]

    def test_all_change_types_defined(self):
        type_values = [c[0] for c in ChangeRequest.CHANGE_TYPE_CHOICES]
        for ct in self.REQUIRED_TYPES:
            self.assertIn(ct, type_values, f"Change type '{ct}' missing")

    def test_change_type_field_exists(self):
        field = ChangeRequest._meta.get_field("change_type")
        self.assertIsNotNone(field)
        self.assertEqual(field.max_length, 20)


class RiskLevelsTest(SimpleTestCase):
    """CHG-001 §4.2: ChangeRequest has risk_level field."""

    REQUIRED_LEVELS = ["critical", "high", "medium", "low"]

    def test_all_risk_levels_defined(self):
        level_values = [c[0] for c in ChangeRequest.RISK_LEVEL_CHOICES]
        for level in self.REQUIRED_LEVELS:
            self.assertIn(level, level_values, f"Risk level '{level}' missing")

    def test_default_risk_level(self):
        field = ChangeRequest._meta.get_field("risk_level")
        self.assertEqual(field.default, "medium")


class LifecycleStatesTest(SimpleTestCase):
    """CHG-001 §5.1: ChangeRequest has status field with lifecycle states."""

    REQUIRED_STATUSES = [
        "draft",
        "submitted",
        "risk_assessed",
        "approved",
        "rejected",
        "in_progress",
        "testing",
        "completed",
        "failed",
        "rolled_back",
        "cancelled",
    ]

    def test_all_statuses_defined(self):
        status_values = [c[0] for c in ChangeRequest.STATUS_CHOICES]
        for status in self.REQUIRED_STATUSES:
            self.assertIn(status, status_values, f"Status '{status}' missing")

    def test_default_status_is_draft(self):
        field = ChangeRequest._meta.get_field("status")
        self.assertEqual(field.default, "draft")


class ChangeRequestFieldsTest(SimpleTestCase):
    """CHG-001 §4: ChangeRequest model field completeness."""

    def test_uuid_primary_key(self):
        field = ChangeRequest._meta.get_field("id")
        self.assertTrue(field.primary_key)

    def test_implementation_plan_field(self):
        field = ChangeRequest._meta.get_field("implementation_plan")
        self.assertIsNotNone(field)

    def test_rollback_plan_field(self):
        field = ChangeRequest._meta.get_field("rollback_plan")
        self.assertIsNotNone(field)

    def test_correlation_id_field(self):
        field = ChangeRequest._meta.get_field("correlation_id")
        self.assertIsNotNone(field)
        self.assertTrue(field.db_index)

    def test_is_emergency_field(self):
        field = ChangeRequest._meta.get_field("is_emergency")
        self.assertIsNotNone(field)
        self.assertFalse(field.default)


class UUIDLinkingTest(SimpleTestCase):
    """CHG-001 §6: ChangeRequest contains UUID-linked cross-reference fields."""

    UUID_LINK_FIELDS = [
        "compliance_check_ids",
        "drift_violation_ids",
        "audit_entry_ids",
        "related_change_ids",
        "commit_shas",
    ]

    def test_uuid_link_fields_exist(self):
        for field_name in self.UUID_LINK_FIELDS:
            field = ChangeRequest._meta.get_field(field_name)
            self.assertIsNotNone(field, f"Field '{field_name}' missing")

    def test_parent_change_field(self):
        field = ChangeRequest._meta.get_field("parent_change_id")
        self.assertIsNotNone(field)
        self.assertTrue(field.null)


class ChangeLogTest(SimpleTestCase):
    """CHG-001 §5: ChangeLog records lifecycle events."""

    def test_model_exists(self):
        self.assertTrue(hasattr(ChangeLog, "_meta"))

    def test_has_action_field(self):
        field = ChangeLog._meta.get_field("action")
        self.assertIsNotNone(field)

    def test_has_change_request_fk(self):
        field = ChangeLog._meta.get_field("change_request")
        self.assertIsNotNone(field)

    def test_has_details_field(self):
        field = ChangeLog._meta.get_field("details")
        self.assertIsNotNone(field)

    def test_has_timestamp(self):
        field = ChangeLog._meta.get_field("timestamp")
        self.assertIsNotNone(field)


class RiskAssessmentTest(SimpleTestCase):
    """CHG-001 §7: RiskAssessment captures multi-dimensional risk analysis."""

    def test_model_exists(self):
        self.assertTrue(hasattr(RiskAssessment, "_meta"))

    def test_has_change_request_fk(self):
        field = RiskAssessment._meta.get_field("change_request")
        self.assertIsNotNone(field)

    def test_has_overall_score(self):
        field = RiskAssessment._meta.get_field("overall_score")
        self.assertIsNotNone(field)

    def test_has_dimension_scores(self):
        for dim in [
            "security_score",
            "availability_score",
            "integrity_score",
            "confidentiality_score",
            "privacy_score",
        ]:
            field = RiskAssessment._meta.get_field(dim)
            self.assertIsNotNone(field, f"Dimension field '{dim}' missing")


class AgentVoteTest(SimpleTestCase):
    """CHG-001 §7: AgentVote records agent assessments."""

    def test_model_exists(self):
        self.assertTrue(hasattr(AgentVote, "_meta"))

    def test_has_agent_role_field(self):
        field = AgentVote._meta.get_field("agent_role")
        self.assertIsNotNone(field)

    def test_has_recommendation_field(self):
        field = AgentVote._meta.get_field("recommendation")
        self.assertIsNotNone(field)

    def test_has_rationale_field(self):
        field = AgentVote._meta.get_field("rationale")
        self.assertIsNotNone(field)

    def test_has_risk_assessment_fk(self):
        field = AgentVote._meta.get_field("risk_assessment")
        self.assertIsNotNone(field)

    def test_agent_role_choices_include_required_perspectives(self):
        role_values = [c[0] for c in AgentVote.AGENT_ROLE_CHOICES]
        for role in ["security_analyst", "architect", "operations", "quality"]:
            self.assertIn(role, role_values, f"Agent role '{role}' missing")


if __name__ == "__main__":
    unittest.main()
