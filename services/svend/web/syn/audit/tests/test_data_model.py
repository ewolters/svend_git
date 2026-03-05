"""
DAT-001 compliance tests: Data Model Standard.

Tests model structure, immutability enforcement, ownership constraints,
field patterns, indexes, and domain model schemas.

Standard: DAT-001
"""

from django.core.exceptions import ValidationError
from django.db import IntegrityError, models
from django.test import SimpleTestCase, TestCase

from syn.core.base_models import SynaraEntity, SynaraImmutableLog


class SysLogBigAutoTest(SimpleTestCase):
    """DAT-001 §4.2: SysLogEntry uses BigAutoField for sequential ordering."""

    def test_syslog_pk_is_bigauto(self):
        from syn.audit.models import SysLogEntry

        pk = SysLogEntry._meta.pk
        self.assertIsInstance(pk, models.BigAutoField)

    def test_syslog_pk_not_uuid(self):
        from syn.audit.models import SysLogEntry

        pk = SysLogEntry._meta.pk
        self.assertNotIsInstance(pk, models.UUIDField)


class ImmutableLogSaveTest(TestCase):
    """DAT-001 §5.3: SynaraImmutableLog.save() prevents updates on existing records."""

    def test_create_succeeds(self):
        from syn.audit.models import SysLogEntry

        entry = SysLogEntry(event_name="test.immutable_create", actor="test")
        entry.save()
        self.assertIsNotNone(entry.pk)

    def test_resave_raises_validation_error(self):
        from syn.audit.models import SysLogEntry

        entry = SysLogEntry(event_name="test.immutable_resave", actor="test")
        entry.save()
        with self.assertRaises(ValidationError):
            entry.save()

    def test_error_message_references_immutability(self):
        from syn.audit.models import SysLogEntry

        entry = SysLogEntry(event_name="test.immutable_msg", actor="test")
        entry.save()
        with self.assertRaises(ValidationError) as ctx:
            entry.save()
        self.assertIn("immutable", str(ctx.exception).lower())


class ImmutableLogDeleteTest(TestCase):
    """DAT-001 §5.3: SynaraImmutableLog.delete() raises ValueError to protect hash chain."""

    def test_delete_raises_value_error(self):
        from syn.audit.models import SysLogEntry

        entry = SysLogEntry(event_name="test.immutable_delete", actor="test")
        entry.save()
        with self.assertRaises(ValueError):
            entry.delete()

    def test_delete_error_references_integrity(self):
        from syn.audit.models import SysLogEntry

        entry = SysLogEntry(event_name="test.immutable_aud", actor="test")
        entry.save()
        with self.assertRaises(ValueError) as ctx:
            entry.delete()
        self.assertIn("cannot be deleted", str(ctx.exception))


class SynaraMetaTest(SimpleTestCase):
    """DAT-001 §5.4: SynaraEntity subclasses configure behavior via SynaraMeta."""

    def test_default_synara_meta_attributes(self):
        meta = SynaraEntity.SynaraMeta
        self.assertEqual(meta.event_domain, "")
        self.assertEqual(meta.emit_events, [])
        self.assertEqual(meta.lifecycle_states, [])
        self.assertEqual(meta.terminal_states, [])
        self.assertFalse(meta.require_governance)

    def test_synara_meta_is_overridable(self):
        """Subclasses can override SynaraMeta attributes."""

        class TestEntity(SynaraEntity):
            class Meta:
                app_label = "test"

            class SynaraMeta:
                event_domain = "test.entity"
                emit_events = ["created", "updated"]
                lifecycle_states = ["draft", "active"]
                terminal_states = ["archived"]
                require_governance = True

        meta = TestEntity.SynaraMeta
        self.assertEqual(meta.event_domain, "test.entity")
        self.assertEqual(meta.emit_events, ["created", "updated"])
        self.assertTrue(meta.require_governance)


class UserXorTenantTest(TestCase):
    """DAT-001 §6.1: Project and KnowledgeGraph enforce User XOR Tenant via CheckConstraint."""

    def test_project_constraint_exists(self):
        from core.models.project import Project

        names = [c.name for c in Project._meta.constraints]
        self.assertIn("project_has_single_owner", names)

    def test_knowledge_graph_constraint_exists(self):
        from core.models.graph import KnowledgeGraph

        names = [c.name for c in KnowledgeGraph._meta.constraints]
        self.assertIn("graph_has_single_owner", names)

    def test_project_both_owners_rejected(self):
        """Setting both user and tenant violates the constraint."""
        from accounts.models import User
        from core.models.project import Project
        from core.models.tenant import Tenant

        user = User.objects.create_user(username="xor_test", password="testpass123!")
        tenant = Tenant.objects.create(name="XOR Test Org", slug="xor-test-org")
        with self.assertRaises(IntegrityError):
            Project.objects.create(title="Both owners", user=user, tenant=tenant)

    def test_project_no_owner_rejected(self):
        """Setting neither user nor tenant violates the constraint."""
        from core.models.project import Project

        with self.assertRaises(IntegrityError):
            Project.objects.create(title="No owner", user=None, tenant=None)


class JSONFieldDefaultsTest(SimpleTestCase):
    """DAT-001 §7.3: JSONField uses typed defaults (list or dict), not mutable literals."""

    def test_project_jsonfields_use_callable_defaults(self):
        from core.models.project import Project

        for field in Project._meta.get_fields():
            if isinstance(field, models.JSONField) and hasattr(field, "default"):
                if field.default is models.fields.NOT_PROVIDED:
                    continue  # nullable optional fields (synara_state, interview_state)
                self.assertTrue(
                    callable(field.default),
                    f"Project.{field.name} default is not callable: {field.default}",
                )

    def test_hypothesis_jsonfields_use_callable_defaults(self):
        from core.models.hypothesis import Hypothesis

        for field in Hypothesis._meta.get_fields():
            if isinstance(field, models.JSONField) and hasattr(field, "default"):
                if field.default is models.fields.NOT_PROVIDED:
                    continue
                self.assertTrue(
                    callable(field.default),
                    f"Hypothesis.{field.name} default is not callable: {field.default}",
                )


class ForeignKeyPatternsTest(SimpleTestCase):
    """DAT-001 §7.4: FK relationships follow documented patterns."""

    def test_hypothesis_project_fk_cascades(self):
        from core.models.hypothesis import Hypothesis

        fk = Hypothesis._meta.get_field("project")
        self.assertEqual(fk.remote_field.on_delete.__name__, "CASCADE")

    def test_hypothesis_project_fk_has_related_name(self):
        from core.models.hypothesis import Hypothesis

        fk = Hypothesis._meta.get_field("project")
        self.assertEqual(fk.remote_field.related_name, "hypotheses")

    def test_hypothesis_merged_into_set_null(self):
        from core.models.hypothesis import Hypothesis

        fk = Hypothesis._meta.get_field("merged_into")
        self.assertEqual(fk.remote_field.on_delete.__name__, "SET_NULL")

    def test_evidence_link_hypothesis_fk_cascades(self):
        from core.models.hypothesis import EvidenceLink

        fk = EvidenceLink._meta.get_field("hypothesis")
        self.assertEqual(fk.remote_field.on_delete.__name__, "CASCADE")


class IndexPatternsTest(SimpleTestCase):
    """DAT-001 §8.1: Indexes follow ownership+filter, recency, correlation, and ranking patterns."""

    def _index_field_names(self, model):
        """Extract list of index field-name tuples from a model."""
        return [tuple(f for f in idx.fields) for idx in model._meta.indexes]

    def test_project_ownership_filter_indexes(self):
        from core.models.project import Project

        indexes = self._index_field_names(Project)
        self.assertIn(("user", "status"), indexes)
        self.assertIn(("tenant", "status"), indexes)

    def test_project_recency_index(self):
        from core.models.project import Project

        indexes = self._index_field_names(Project)
        self.assertIn(("-updated_at",), indexes)

    def test_hypothesis_ranking_index(self):
        from core.models.hypothesis import Hypothesis

        indexes = self._index_field_names(Hypothesis)
        self.assertIn(("project", "-current_probability"), indexes)

    def test_hypothesis_status_filter_index(self):
        from core.models.hypothesis import Hypothesis

        indexes = self._index_field_names(Hypothesis)
        self.assertIn(("project", "status"), indexes)

    def test_entity_graph_indexes(self):
        from core.models.graph import Entity

        indexes = self._index_field_names(Entity)
        self.assertIn(("graph", "entity_type"), indexes)
        self.assertIn(("graph", "name"), indexes)

    def test_relationship_traversal_indexes(self):
        from core.models.graph import Relationship

        indexes = self._index_field_names(Relationship)
        self.assertIn(("graph", "relation_type"), indexes)
        self.assertIn(("source", "relation_type"), indexes)
        self.assertIn(("target", "relation_type"), indexes)


class CheckConstraintTest(SimpleTestCase):
    """DAT-001 §8.2: CheckConstraints enforce domain invariants at the database level."""

    def test_project_has_check_constraint(self):
        from core.models.project import Project

        constraint_types = [type(c).__name__ for c in Project._meta.constraints]
        self.assertIn("CheckConstraint", constraint_types)

    def test_knowledge_graph_has_check_constraint(self):
        from core.models.graph import KnowledgeGraph

        constraint_types = [type(c).__name__ for c in KnowledgeGraph._meta.constraints]
        self.assertIn("CheckConstraint", constraint_types)

    def test_evidence_link_unique_together(self):
        from core.models.hypothesis import EvidenceLink

        self.assertIn(("hypothesis", "evidence"), EvidenceLink._meta.unique_together)

    def test_membership_unique_together(self):
        from core.models.tenant import Membership

        self.assertIn(("tenant", "user"), Membership._meta.unique_together)


class ProjectModelFieldsTest(SimpleTestCase):
    """DAT-001 §9.1: Project stores 5W2H problem definition, SMART goals, methodology, phases, team, and changelog."""

    REQUIRED_FIELDS = [
        # 5W2H problem definition
        "problem_whats", "problem_wheres", "problem_whens",
        "problem_magnitude", "problem_statement",
        # SMART goals
        "goal_statement", "goal_metric", "goal_baseline", "goal_target",
        "goal_unit", "goal_deadline",
        # Scope
        "scope_in", "scope_out", "constraints", "assumptions",
        # Team
        "champion_name", "leader_name", "team_members",
        # Methodology and phases
        "methodology", "current_phase", "phase_history", "milestones",
        # Resolution
        "resolution_summary", "resolution_actions",
        # Changelog
        "changelog",
    ]

    def test_all_required_fields_exist(self):
        from core.models.project import Project

        field_names = {f.name for f in Project._meta.get_fields() if hasattr(f, "name")}
        for field_name in self.REQUIRED_FIELDS:
            self.assertIn(field_name, field_names, f"Project missing field: {field_name}")

    def test_project_has_uuid_pk(self):
        from core.models.project import Project

        self.assertIsInstance(Project._meta.pk, models.UUIDField)


class HypothesisModelFieldsTest(SimpleTestCase):
    """DAT-001 §9.2: Hypothesis uses If/Then/Because structure with Bayesian probability tracking."""

    REQUIRED_FIELDS = [
        "if_clause", "then_clause", "because_clause",
        "prior_probability", "current_probability", "probability_history",
        "confirmation_threshold", "rejection_threshold",
        "status", "is_testable",
    ]

    def test_all_required_fields_exist(self):
        from core.models.hypothesis import Hypothesis

        field_names = {f.name for f in Hypothesis._meta.get_fields() if hasattr(f, "name")}
        for field_name in self.REQUIRED_FIELDS:
            self.assertIn(field_name, field_names, f"Hypothesis missing field: {field_name}")

    def test_hypothesis_has_uuid_pk(self):
        from core.models.hypothesis import Hypothesis

        self.assertIsInstance(Hypothesis._meta.pk, models.UUIDField)

    def test_hypothesis_fk_to_project(self):
        from core.models.hypothesis import Hypothesis

        fk = Hypothesis._meta.get_field("project")
        self.assertTrue(fk.is_relation)


class EvidenceLinkTest(SimpleTestCase):
    """DAT-001 §9.4: EvidenceLink is M2M junction with unique_together and likelihood_ratio."""

    def test_unique_together_on_hypothesis_evidence(self):
        from core.models.hypothesis import EvidenceLink

        self.assertIn(("hypothesis", "evidence"), EvidenceLink._meta.unique_together)

    def test_has_likelihood_ratio(self):
        from core.models.hypothesis import EvidenceLink

        field = EvidenceLink._meta.get_field("likelihood_ratio")
        self.assertIsNotNone(field)

    def test_has_direction(self):
        from core.models.hypothesis import EvidenceLink

        field = EvidenceLink._meta.get_field("direction")
        self.assertIsNotNone(field)


class KnowledgeGraphOwnershipTest(SimpleTestCase):
    """DAT-001 §9.5: KnowledgeGraph uses User XOR Tenant with Entity and Relationship subgraph."""

    def test_has_user_xor_tenant_constraint(self):
        from core.models.graph import KnowledgeGraph

        names = [c.name for c in KnowledgeGraph._meta.constraints]
        self.assertIn("graph_has_single_owner", names)

    def test_entity_belongs_to_graph(self):
        from core.models.graph import Entity

        fk = Entity._meta.get_field("graph")
        self.assertTrue(fk.is_relation)

    def test_relationship_has_source_and_target(self):
        from core.models.graph import Relationship

        for field_name in ("source", "target"):
            fk = Relationship._meta.get_field(field_name)
            self.assertTrue(fk.is_relation, f"Relationship.{field_name} is not a relation")
