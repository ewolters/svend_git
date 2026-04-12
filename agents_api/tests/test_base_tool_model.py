"""
ToolModel abstract base class behavioral tests.

Validates that ToolModel provides correct field infrastructure, status
transition validation, serialization, and SynaraEntity inheritance for
all QMS tool modules.

Standard:     DAT-001 §11 (Tool Model Base Class)
Compliance:   TST-001 §4, ORG-001 §2.2, SEC-001 §5.2

<!-- test: agents_api.tests.test_base_tool_model.ToolModelFieldTests -->
<!-- test: agents_api.tests.test_base_tool_model.ToolModelTransitionTests -->
<!-- test: agents_api.tests.test_base_tool_model.ToolModelSerializationTests -->
<!-- test: agents_api.tests.test_base_tool_model.ToolModelSynaraEntityTests -->
<!-- test: agents_api.tests.test_base_tool_model.ToolModelOwnershipTests -->

CR: 0d4ef46d
"""

import uuid

from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.db import models
from django.test import TestCase, override_settings

from accounts.constants import Tier
from agents_api.base_tool_model import ToolModel
from core.models import Project
from syn.core.base_models import SynaraEntity

User = get_user_model()
SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)


# =============================================================================
# Concrete test model — exercises ToolModel in a real DB context
# =============================================================================


class ConcreteToolNoTransitions(ToolModel):
    """Minimal concrete subclass with no state machine."""

    custom_field = models.TextField(blank=True, default="")

    class Meta(ToolModel.Meta):
        app_label = "agents_api"
        db_table = "test_concrete_tool_no_transitions"

    class SynaraMeta:
        event_domain = "test.tool_no_transitions"
        emit_events = ["created", "updated", "deleted"]


class ConcreteToolWithTransitions(ToolModel):
    """Concrete subclass with full state machine."""

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        ACTIVE = "active", "Active"
        REVIEW = "review", "Under Review"
        COMPLETE = "complete", "Complete"
        ARCHIVED = "archived", "Archived"

    VALID_TRANSITIONS = {
        "draft": ["active"],
        "active": ["review", "draft"],
        "review": ["complete", "active"],
        "complete": ["archived"],
        "archived": [],
    }

    TRANSITION_REQUIREMENTS = {
        "complete": ["findings"],
        "archived": [],
    }

    findings = models.TextField(blank=True, default="")

    class Meta(ToolModel.Meta):
        app_label = "agents_api"
        db_table = "test_concrete_tool_with_transitions"

    class SynaraMeta:
        event_domain = "test.tool_with_transitions"
        emit_events = ["created", "updated", "deleted"]


# =============================================================================
# Shared table setup — both concrete models must exist for FK cascade safety
# =============================================================================

_TEST_MODELS = [ConcreteToolNoTransitions, ConcreteToolWithTransitions]


def _create_test_tables():
    from django.db import connection

    with connection.schema_editor() as editor:
        for model in _TEST_MODELS:
            try:
                editor.create_model(model)
            except Exception:
                pass


def _drop_test_tables():
    from django.db import connection

    with connection.schema_editor() as editor:
        for model in reversed(_TEST_MODELS):
            try:
                editor.delete_model(model)
            except Exception:
                pass


# =============================================================================
# Helpers
# =============================================================================


def _user(email="tool_test@test.com", tier=Tier.PRO):
    username = email.split("@")[0]
    u = User.objects.create_user(username=username, email=email, password="testpass123!")
    u.tier = tier
    u.save(update_fields=["tier"])
    return u


def _project(user, title="Test Project"):
    return Project.objects.create(
        title=title,
        user=user,
        problem_statement="Test problem",
    )


# =============================================================================
# 1. Field Infrastructure Tests
# =============================================================================


@SECURE_OFF
class ToolModelFieldTests(TestCase):
    """DAT-001 §11: ToolModel provides correct fields with proper types."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _create_test_tables()

    @classmethod
    def tearDownClass(cls):
        _drop_test_tables()
        super().tearDownClass()

    def test_title_field_exists_as_charfield_max_255(self):
        """ToolModel.title is CharField(max_length=255)."""
        field = ConcreteToolNoTransitions._meta.get_field("title")
        self.assertIsInstance(field, models.CharField)
        self.assertEqual(field.max_length, 255)

    def test_description_field_exists_as_textfield_blank(self):
        """ToolModel.description is TextField(blank=True, default='')."""
        field = ConcreteToolNoTransitions._meta.get_field("description")
        self.assertIsInstance(field, models.TextField)
        self.assertTrue(field.blank)
        self.assertEqual(field.default, "")

    def test_status_field_exists_as_charfield_indexed(self):
        """ToolModel.status is CharField(max_length=25, db_index=True, default='draft')."""
        field = ConcreteToolNoTransitions._meta.get_field("status")
        self.assertIsInstance(field, models.CharField)
        self.assertEqual(field.max_length, 25)
        self.assertTrue(field.db_index)
        self.assertEqual(field.default, "draft")

    def test_owner_fk_to_user_nullable_set_null(self):
        """ToolModel.owner is FK→User, nullable, on_delete=SET_NULL per ORG-001 §2.2."""
        field = ConcreteToolNoTransitions._meta.get_field("owner")
        self.assertIsInstance(field, models.ForeignKey)
        self.assertTrue(field.null)
        self.assertTrue(field.blank)
        self.assertEqual(field.remote_field.on_delete.__name__, models.SET_NULL.__name__)

    def test_project_fk_nullable_set_null(self):
        """ToolModel.project is FK→Project, nullable, on_delete=SET_NULL."""
        field = ConcreteToolNoTransitions._meta.get_field("project")
        self.assertIsInstance(field, models.ForeignKey)
        self.assertTrue(field.null)
        self.assertTrue(field.blank)
        self.assertEqual(field.remote_field.on_delete.__name__, models.SET_NULL.__name__)

    def test_site_fk_nullable_set_null(self):
        """ToolModel.site is FK→Site, nullable, on_delete=SET_NULL."""
        field = ConcreteToolNoTransitions._meta.get_field("site")
        self.assertIsInstance(field, models.ForeignKey)
        self.assertTrue(field.null)
        self.assertTrue(field.blank)
        self.assertEqual(field.remote_field.on_delete.__name__, models.SET_NULL.__name__)

    def test_abstract_meta(self):
        """ToolModel itself is abstract — cannot be instantiated directly."""
        self.assertTrue(ToolModel._meta.abstract)

    def test_concrete_subclass_creates_with_defaults(self):
        """Concrete subclass can be saved with only required fields (title)."""
        user = _user()
        record = ConcreteToolNoTransitions.objects.create(
            title="Test Record",
            owner=user,
            created_by=user.email,
        )
        self.assertEqual(record.title, "Test Record")
        self.assertEqual(record.status, "draft")
        self.assertEqual(record.description, "")
        self.assertIsNone(record.project)
        self.assertIsNone(record.site)
        self.assertIsNotNone(record.id)
        self.assertIsNotNone(record.created_at)

    def test_owner_can_be_null_for_site_scoped(self):
        """Owner can be NULL when record is site-scoped per ORG-001 §2.2."""
        record = ConcreteToolNoTransitions.objects.create(
            title="Site-Scoped Record",
            owner=None,
            created_by="system",
        )
        self.assertIsNone(record.owner)
        self.assertIsNone(record.owner_id)

    def test_project_link_optional(self):
        """Project FK is optional — records can exist without project context."""
        user = _user("no_project@test.com")
        record = ConcreteToolNoTransitions.objects.create(title="No Project", owner=user)
        self.assertIsNone(record.project_id)

    def test_project_link_persists(self):
        """Project FK persists and can be queried."""
        user = _user("with_project@test.com")
        project = _project(user)
        record = ConcreteToolNoTransitions.objects.create(title="With Project", owner=user, project=project)
        record.refresh_from_db()
        self.assertEqual(record.project_id, project.id)

    def test_related_name_uses_class_name(self):
        """Related names use %(class)s pattern for unique reverse relations."""
        field = ConcreteToolNoTransitions._meta.get_field("owner")
        self.assertIn("concretetoolnotransitions", field.remote_field.related_name)


# =============================================================================
# 2. Status Transition Tests
# =============================================================================


@SECURE_OFF
class ToolModelTransitionTests(TestCase):
    """DAT-001 §11.3: Status transition validation enforces state machine."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _create_test_tables()

    @classmethod
    def tearDownClass(cls):
        _drop_test_tables()
        super().tearDownClass()

    def setUp(self):
        self.user = _user("transition@test.com")

    def test_valid_transition_draft_to_active(self):
        """draft → active is allowed when defined in VALID_TRANSITIONS."""
        record = ConcreteToolWithTransitions(title="Test", status="draft", owner=self.user)
        is_valid, error = record.validate_transition("active")
        self.assertTrue(is_valid)
        self.assertEqual(error, "")

    def test_invalid_transition_draft_to_complete(self):
        """draft → complete is not in VALID_TRANSITIONS and is rejected."""
        record = ConcreteToolWithTransitions(title="Test", status="draft", owner=self.user)
        is_valid, error = record.validate_transition("complete")
        self.assertFalse(is_valid)
        self.assertIn("Cannot transition", error)
        self.assertIn("draft", error)
        self.assertIn("complete", error)

    def test_invalid_transition_archived_to_anything(self):
        """archived is a terminal state — no transitions allowed."""
        record = ConcreteToolWithTransitions(title="Test", status="archived", owner=self.user)
        is_valid, error = record.validate_transition("draft")
        self.assertFalse(is_valid)
        self.assertIn("Cannot transition", error)

    def test_transition_requirement_blocks_without_field(self):
        """complete requires 'findings' field — blocked when empty."""
        record = ConcreteToolWithTransitions(title="Test", status="review", findings="", owner=self.user)
        is_valid, error = record.validate_transition("complete")
        self.assertFalse(is_valid)
        self.assertIn("findings", error)

    def test_transition_requirement_passes_with_field(self):
        """complete requires 'findings' — allowed when populated."""
        record = ConcreteToolWithTransitions(
            title="Test",
            status="review",
            findings="Root cause: corrosion in valve seat",
            owner=self.user,
        )
        is_valid, error = record.validate_transition("complete")
        self.assertTrue(is_valid)
        self.assertEqual(error, "")

    def test_transition_requirement_whitespace_only_treated_as_empty(self):
        """Whitespace-only field values are treated as empty for requirements."""
        record = ConcreteToolWithTransitions(title="Test", status="review", findings="   \n\t  ", owner=self.user)
        is_valid, error = record.validate_transition("complete")
        self.assertFalse(is_valid)
        self.assertIn("findings", error)

    def test_transition_to_sets_status_on_valid(self):
        """transition_to() sets the status field on valid transition."""
        record = ConcreteToolWithTransitions(title="Test", status="draft", owner=self.user)
        record.transition_to("active")
        self.assertEqual(record.status, "active")

    def test_transition_to_raises_on_invalid(self):
        """transition_to() raises ValidationError on invalid transition."""
        record = ConcreteToolWithTransitions(title="Test", status="draft", owner=self.user)
        with self.assertRaises(ValidationError) as ctx:
            record.transition_to("complete")
        self.assertIn("Cannot transition", str(ctx.exception))

    def test_transition_to_does_not_save(self):
        """transition_to() changes in-memory status but does not persist."""
        record = ConcreteToolWithTransitions.objects.create(title="Persist Test", status="draft", owner=self.user)
        record.transition_to("active")
        self.assertEqual(record.status, "active")
        record.refresh_from_db()
        self.assertEqual(record.status, "draft")  # not saved

    def test_no_transitions_map_allows_any_status(self):
        """Model without VALID_TRANSITIONS allows any status change."""
        record = ConcreteToolNoTransitions(title="Test", status="draft", owner=self.user)
        is_valid, error = record.validate_transition("anything_at_all")
        self.assertTrue(is_valid)
        self.assertEqual(error, "")

    def test_full_lifecycle_draft_to_archived(self):
        """Walk the full lifecycle: draft → active → review → complete → archived."""
        record = ConcreteToolWithTransitions.objects.create(
            title="Full Lifecycle",
            status="draft",
            owner=self.user,
        )

        # draft → active
        record.transition_to("active")
        self.assertEqual(record.status, "active")

        # active → review
        record.transition_to("review")
        self.assertEqual(record.status, "review")

        # review → complete (requires findings)
        record.findings = "Investigation complete. Root cause confirmed."
        record.transition_to("complete")
        self.assertEqual(record.status, "complete")

        # complete → archived
        record.transition_to("archived")
        self.assertEqual(record.status, "archived")

        # archived → nothing (terminal)
        with self.assertRaises(ValidationError):
            record.transition_to("draft")

    def test_backward_transition_active_to_draft(self):
        """active → draft is valid (re-open / rework pattern)."""
        record = ConcreteToolWithTransitions(title="Test", status="active", owner=self.user)
        is_valid, _ = record.validate_transition("draft")
        self.assertTrue(is_valid)

    def test_multiple_allowed_targets_from_active(self):
        """active can transition to either review or draft."""
        record = ConcreteToolWithTransitions(title="Test", status="active", owner=self.user)
        valid_review, _ = record.validate_transition("review")
        valid_draft, _ = record.validate_transition("draft")
        valid_complete, _ = record.validate_transition("complete")
        self.assertTrue(valid_review)
        self.assertTrue(valid_draft)
        self.assertFalse(valid_complete)


# =============================================================================
# 3. Serialization Tests
# =============================================================================


@SECURE_OFF
class ToolModelSerializationTests(TestCase):
    """DAT-001 §11.4: to_dict() provides consistent API-ready output."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _create_test_tables()

    @classmethod
    def tearDownClass(cls):
        _drop_test_tables()
        super().tearDownClass()

    def setUp(self):
        self.user = _user("serial@test.com")

    def test_to_dict_includes_all_base_fields(self):
        """to_dict() includes id, title, description, status, ownership, timestamps."""
        record = ConcreteToolNoTransitions.objects.create(
            title="Serialize Me",
            description="A test record",
            status="draft",
            owner=self.user,
            created_by=self.user.email,
        )
        d = record.to_dict()
        expected_keys = {
            "id",
            "title",
            "description",
            "status",
            "owner_id",
            "project_id",
            "site_id",
            "created_by",
            "updated_by",
            "created_at",
            "updated_at",
            "correlation_id",
        }
        self.assertEqual(set(d.keys()), expected_keys)

    def test_to_dict_id_and_correlation_are_valid_uuid_strings(self):
        """id and correlation_id serialize as valid UUID strings."""
        record = ConcreteToolNoTransitions.objects.create(title="UUID Test", owner=self.user)
        d = record.to_dict()
        self.assertIsInstance(d["id"], str)
        self.assertIsInstance(d["correlation_id"], str)
        uuid.UUID(d["id"])
        uuid.UUID(d["correlation_id"])

    def test_to_dict_owner_id_is_string(self):
        """owner_id serializes as string (User PK type may vary)."""
        record = ConcreteToolNoTransitions.objects.create(title="Owner Test", owner=self.user)
        d = record.to_dict()
        self.assertIsInstance(d["owner_id"], str)
        self.assertEqual(d["owner_id"], str(self.user.pk))

    def test_to_dict_null_fks_are_none(self):
        """Null FK fields serialize as None, not 'None' string."""
        record = ConcreteToolNoTransitions.objects.create(title="Null FK Test", owner=self.user)
        d = record.to_dict()
        self.assertIsNone(d["project_id"])
        self.assertIsNone(d["site_id"])

    def test_to_dict_timestamps_are_iso_format(self):
        """Timestamps serialize as ISO 8601 strings."""
        record = ConcreteToolNoTransitions.objects.create(title="Timestamp Test", owner=self.user)
        d = record.to_dict()
        self.assertIn("T", d["created_at"])  # ISO format has T separator
        self.assertIn("T", d["updated_at"])

    def test_to_dict_with_project_includes_project_id(self):
        """to_dict() includes project_id as string when project is linked."""
        project = _project(self.user)
        record = ConcreteToolNoTransitions.objects.create(title="Project Test", owner=self.user, project=project)
        d = record.to_dict()
        self.assertEqual(d["project_id"], str(project.id))

    def test_to_dict_is_extensible_by_subclass(self):
        """Subclass can extend to_dict() with super() call."""

        class ExtendedTool(ConcreteToolWithTransitions):
            class Meta:
                proxy = True

            def to_dict(self):
                d = super().to_dict()
                d["findings"] = self.findings
                return d

        record = ExtendedTool.objects.create(
            title="Extended",
            findings="Found something",
            owner=self.user,
        )
        d = record.to_dict()
        self.assertIn("findings", d)
        self.assertEqual(d["findings"], "Found something")
        self.assertIn("id", d)  # base fields still present
        self.assertIn("title", d)


# =============================================================================
# 4. SynaraEntity Inheritance Tests
# =============================================================================


@SECURE_OFF
class ToolModelSynaraEntityTests(TestCase):
    """DAT-001 §11.1: ToolModel inherits SynaraEntity infrastructure."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _create_test_tables()

    @classmethod
    def tearDownClass(cls):
        _drop_test_tables()
        super().tearDownClass()

    def test_inherits_from_synara_entity(self):
        """ToolModel is a subclass of SynaraEntity."""
        self.assertTrue(issubclass(ToolModel, SynaraEntity))

    def test_uuid_primary_key(self):
        """Records get UUID v4 primary keys from SynaraEntity."""
        user = _user("uuid_pk@test.com")
        record = ConcreteToolNoTransitions.objects.create(title="UUID PK Test", owner=user)
        self.assertIsInstance(record.id, uuid.UUID)
        self.assertEqual(record.id.version, 4)

    def test_correlation_id_auto_generated_unique(self):
        """Each record gets a unique correlation_id from SynaraEntity."""
        user = _user("corr@test.com")
        r1 = ConcreteToolNoTransitions.objects.create(title="R1", owner=user)
        r2 = ConcreteToolNoTransitions.objects.create(title="R2", owner=user)
        self.assertIsInstance(r1.correlation_id, uuid.UUID)
        self.assertIsInstance(r2.correlation_id, uuid.UUID)
        self.assertNotEqual(r1.correlation_id, r2.correlation_id)

    def test_soft_delete_hides_from_default_queryset(self):
        """Soft-deleted records are excluded from objects manager."""
        user = _user("soft_del@test.com")
        record = ConcreteToolNoTransitions.objects.create(title="Delete Me", owner=user)
        record_id = record.id

        record.delete(deleted_by=user.email)

        # Default manager excludes deleted
        self.assertFalse(ConcreteToolNoTransitions.objects.filter(id=record_id).exists())
        # all_objects still finds it
        deleted = ConcreteToolNoTransitions.all_objects.get(id=record_id)
        self.assertTrue(deleted.is_deleted)
        self.assertEqual(deleted.deleted_by, user.email)
        self.assertIsNotNone(deleted.deleted_at)

    def test_soft_delete_preserves_data(self):
        """Soft delete preserves all record data for audit trail."""
        user = _user("preserve@test.com")
        record = ConcreteToolNoTransitions.objects.create(
            title="Preserve Me",
            description="Important data",
            owner=user,
            created_by=user.email,
        )
        record.delete(deleted_by=user.email)

        preserved = ConcreteToolNoTransitions.all_objects.get(id=record.id)
        self.assertEqual(preserved.title, "Preserve Me")
        self.assertEqual(preserved.description, "Important data")
        self.assertEqual(preserved.created_by, user.email)

    def test_restore_after_soft_delete(self):
        """Restored record reappears in default queryset."""
        user = _user("restore@test.com")
        record = ConcreteToolNoTransitions.objects.create(title="Restore Me", owner=user)
        record.delete(deleted_by=user.email)

        # Restore
        deleted = ConcreteToolNoTransitions.all_objects.get(id=record.id)
        deleted.restore(restored_by=user.email)

        # Back in default queryset
        restored = ConcreteToolNoTransitions.objects.get(id=record.id)
        self.assertFalse(restored.is_deleted)
        self.assertIsNone(restored.deleted_at)
        self.assertEqual(restored.deleted_by, "")

    def test_metadata_json_field_default_empty_dict(self):
        """metadata field defaults to empty dict and accepts arbitrary JSON."""
        user = _user("meta@test.com")
        record = ConcreteToolNoTransitions.objects.create(title="Metadata Test", owner=user)
        self.assertEqual(record.metadata, {})

        record.metadata = {"custom_key": "custom_value", "count": 42}
        record.save(update_fields=["metadata"])
        record.refresh_from_db()
        self.assertEqual(record.metadata["custom_key"], "custom_value")
        self.assertEqual(record.metadata["count"], 42)

    def test_tenant_id_field_exists_nullable(self):
        """tenant_id is available for multi-tenant isolation (SEC-001 §5.2)."""
        user = _user("tenant@test.com")
        tenant_uuid = uuid.uuid4()
        record = ConcreteToolNoTransitions.objects.create(title="Tenant Test", owner=user, tenant_id=tenant_uuid)
        record.refresh_from_db()
        self.assertEqual(record.tenant_id, tenant_uuid)

    def test_created_by_is_charfield_not_fk(self):
        """created_by is CharField (audit string), not FK — per SynaraEntity convention."""
        field = ConcreteToolNoTransitions._meta.get_field("created_by")
        self.assertIsInstance(field, models.CharField)
        self.assertEqual(field.max_length, 255)

    def test_ordering_by_created_at_descending(self):
        """Default ordering is -created_at from SynaraEntity.Meta."""
        ordering = ConcreteToolNoTransitions._meta.ordering
        self.assertIn("-created_at", ordering)


# =============================================================================
# 5. Ownership & Permission Compatibility Tests
# =============================================================================


@SECURE_OFF
class ToolModelOwnershipTests(TestCase):
    """ORG-001 §2.2: Ownership patterns compatible with qms_* helpers."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _create_test_tables()

    @classmethod
    def tearDownClass(cls):
        _drop_test_tables()
        super().tearDownClass()

    def test_owner_query_by_user(self):
        """Records can be filtered by owner FK for individual access."""
        u1 = _user("owner1@test.com")
        u2 = _user("owner2@test.com")
        ConcreteToolNoTransitions.objects.create(title="User 1 Record", owner=u1)
        ConcreteToolNoTransitions.objects.create(title="User 2 Record", owner=u2)

        u1_records = ConcreteToolNoTransitions.objects.filter(owner=u1)
        self.assertEqual(u1_records.count(), 1)
        self.assertEqual(u1_records.first().title, "User 1 Record")

    def test_site_scoped_record_has_null_owner(self):
        """Site-scoped records have owner=None, site=<Site> per ORG-001 §2.2."""
        record = ConcreteToolNoTransitions.objects.create(
            title="Site Scoped",
            owner=None,
            created_by="admin@factory.com",
        )
        self.assertIsNone(record.owner)
        # Can still query by created_by string
        found = ConcreteToolNoTransitions.objects.filter(created_by="admin@factory.com")
        self.assertEqual(found.count(), 1)

    def test_str_representation(self):
        """__str__ returns '{ClassName}: {title} ({status})'."""
        user = _user("str_test@test.com")
        record = ConcreteToolNoTransitions(title="My Record", status="draft", owner=user)
        expected = "ConcreteToolNoTransitions: My Record (draft)"
        self.assertEqual(str(record), expected)

    def test_owner_deletion_sets_null(self):
        """Deleting the owner user sets owner to NULL (SET_NULL)."""
        user = _user("delete_me@test.com")
        record = ConcreteToolNoTransitions.objects.create(title="Orphan Test", owner=user, created_by=user.email)
        user_id = user.id

        # Delete the user
        User.objects.filter(id=user_id).delete()

        record.refresh_from_db()
        self.assertIsNone(record.owner)
        # Audit string preserved
        self.assertEqual(record.created_by, "delete_me@test.com")

    def test_project_deletion_sets_null(self):
        """Deleting linked project sets project to NULL (SET_NULL)."""
        user = _user("proj_del@test.com")
        project = _project(user)
        record = ConcreteToolNoTransitions.objects.create(title="Project Link", owner=user, project=project)
        project_id = project.id

        Project.objects.filter(id=project_id).delete()

        record.refresh_from_db()
        self.assertIsNone(record.project)
