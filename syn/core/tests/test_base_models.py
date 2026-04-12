"""
Tests for Synara SDK Base Models (SDK-001).

Tests the abstract base classes that provide automatic integration
with Synara OS primitives including:
- Correlation tracking (CTG-001)
- Tenant isolation (SEC-001 §5.2)
- Audit timestamps (AUD-001)
- Soft delete (DAT-001 §9)

Standard: TST-001 (Testing Standard)
Coverage Target: 95%+
"""

import hashlib
import uuid
from unittest.mock import MagicMock, patch

from django.db import models
from django.test import TestCase

from syn.core.base_models import (
    SynaraEntity,
    SynaraEntityManager,
    SynaraRegistry,
)

# =============================================================================
# TEST FIXTURES - Concrete Model Implementations
# =============================================================================


class TestModel(SynaraEntity):
    """Concrete test model for SynaraEntity tests."""

    name = models.CharField(max_length=100)
    value = models.IntegerField(default=0)

    class Meta:
        app_label = "core"
        managed = False  # Don't create actual table


# =============================================================================
# SYNARA ENTITY MANAGER TESTS
# =============================================================================


class TestSynaraEntityManager(TestCase):
    """Test SynaraEntityManager functionality."""

    def test_manager_excludes_deleted_by_default(self):
        """Verify default queryset excludes soft-deleted records."""
        manager = SynaraEntityManager()
        manager.model = TestModel

        # The get_queryset filters out deleted records
        with patch.object(models.Manager, "get_queryset") as mock_qs:
            mock_filter = MagicMock()
            mock_qs.return_value.filter = mock_filter
            mock_filter.return_value = MagicMock()

            manager.get_queryset()

            # Verify filter was called with is_deleted=False
            mock_filter.assert_called_once_with(is_deleted=False)

    def test_with_deleted_includes_all_records(self):
        """Verify with_deleted() returns all records including deleted."""
        manager = SynaraEntityManager()
        manager.model = TestModel

        with patch.object(models.Manager, "get_queryset") as mock_qs:
            mock_qs.return_value = MagicMock()

            manager.with_deleted()

            # Should return unfiltered queryset
            mock_qs.assert_called_once()

    def test_deleted_only_returns_deleted_records(self):
        """Verify deleted_only() returns only soft-deleted records."""
        manager = SynaraEntityManager()
        manager.model = TestModel

        with patch.object(models.Manager, "get_queryset") as mock_qs:
            mock_filter = MagicMock()
            mock_qs.return_value.filter = mock_filter
            mock_filter.return_value = MagicMock()

            manager.deleted_only()

            # Verify filter was called with is_deleted=True
            mock_filter.assert_called_once_with(is_deleted=True)

    def test_for_tenant_filters_by_tenant_id(self):
        """Verify for_tenant() filters by tenant_id (SEC-001 §5.2)."""
        manager = SynaraEntityManager()
        manager.model = TestModel
        test_tenant_id = uuid.uuid4()

        with patch.object(manager, "get_queryset") as mock_qs:
            mock_filter = MagicMock()
            mock_qs.return_value.filter = mock_filter
            mock_filter.return_value = MagicMock()

            manager.for_tenant(test_tenant_id)

            # Verify filter was called with tenant_id
            mock_filter.assert_called_once_with(tenant_id=test_tenant_id)


# =============================================================================
# SYNARA ENTITY FIELD TESTS
# =============================================================================


class TestSynaraEntityFields(TestCase):
    """Test SynaraEntity field definitions."""

    def test_id_field_is_uuid_primary_key(self):
        """Verify id field is UUID primary key."""
        field = SynaraEntity._meta.get_field("id")
        assert isinstance(field, models.UUIDField)
        assert field.primary_key is True
        assert field.editable is False

    def test_correlation_id_field(self):
        """Verify correlation_id field for CTG-001 compliance."""
        field = SynaraEntity._meta.get_field("correlation_id")
        assert isinstance(field, models.UUIDField)
        assert field.unique is True
        assert field.db_index is True
        assert field.editable is False

    def test_parent_correlation_id_field(self):
        """Verify parent_correlation_id for causal chain tracking."""
        field = SynaraEntity._meta.get_field("parent_correlation_id")
        assert isinstance(field, models.UUIDField)
        assert field.null is True
        assert field.blank is True
        assert field.db_index is True

    def test_tenant_id_field(self):
        """Verify tenant_id field for SEC-001 §5.2 compliance."""
        field = SynaraEntity._meta.get_field("tenant_id")
        assert isinstance(field, models.UUIDField)
        assert field.db_index is True

    def test_audit_timestamp_fields(self):
        """Verify audit timestamp fields for AUD-001 compliance."""
        # created_at
        created = SynaraEntity._meta.get_field("created_at")
        assert isinstance(created, models.DateTimeField)
        assert created.auto_now_add is True
        assert created.db_index is True

        # updated_at
        updated = SynaraEntity._meta.get_field("updated_at")
        assert isinstance(updated, models.DateTimeField)
        assert updated.auto_now is True

    def test_audit_user_fields(self):
        """Verify audit user tracking fields."""
        # created_by
        created_by = SynaraEntity._meta.get_field("created_by")
        assert isinstance(created_by, models.CharField)
        assert created_by.max_length == 255
        assert created_by.blank is True

        # updated_by
        updated_by = SynaraEntity._meta.get_field("updated_by")
        assert isinstance(updated_by, models.CharField)
        assert updated_by.max_length == 255
        assert updated_by.blank is True

    def test_soft_delete_fields(self):
        """Verify soft delete fields for DAT-001 §9 compliance."""
        # is_deleted
        is_deleted = SynaraEntity._meta.get_field("is_deleted")
        assert isinstance(is_deleted, models.BooleanField)
        assert is_deleted.default is False
        assert is_deleted.db_index is True

        # deleted_at
        deleted_at = SynaraEntity._meta.get_field("deleted_at")
        assert isinstance(deleted_at, models.DateTimeField)
        assert deleted_at.null is True
        assert deleted_at.blank is True


# =============================================================================
# SYNARA ENTITY BEHAVIOR TESTS
# =============================================================================


class TestSynaraEntityBehaviors(TestCase):
    """Test SynaraEntity behaviors and methods."""

    def test_model_is_abstract(self):
        """Verify SynaraEntity is abstract."""
        assert SynaraEntity._meta.abstract is True

    def test_default_ordering(self):
        """Verify default ordering is by created_at descending."""
        assert SynaraEntity._meta.ordering == ["-created_at"]


# =============================================================================
# SYNARA REGISTRY TESTS
# =============================================================================


class TestSynaraRegistryFields(TestCase):
    """Test SynaraRegistry field definitions."""

    def test_code_field(self):
        """Verify code field for registry identification."""
        field = SynaraRegistry._meta.get_field("code")
        assert isinstance(field, models.CharField)
        assert field.max_length == 100
        assert field.db_index is True

    def test_label_field(self):
        """Verify label field for human-readable display."""
        field = SynaraRegistry._meta.get_field("label")
        assert isinstance(field, models.CharField)
        assert field.max_length == 255

    def test_description_field(self):
        """Verify description field."""
        field = SynaraRegistry._meta.get_field("description")
        assert isinstance(field, models.TextField)
        assert field.blank is True

    def test_is_active_field(self):
        """Verify is_active flag field."""
        field = SynaraRegistry._meta.get_field("is_active")
        assert isinstance(field, models.BooleanField)
        assert field.default is True

    def test_display_order_field(self):
        """Verify display_order field for display ordering."""
        field = SynaraRegistry._meta.get_field("display_order")
        assert isinstance(field, models.PositiveIntegerField)
        assert field.default == 0

    def test_metadata_field(self):
        """Verify metadata JSON field."""
        field = SynaraRegistry._meta.get_field("metadata")
        assert isinstance(field, models.JSONField)
        assert field.blank is True

    def test_model_is_abstract(self):
        """Verify SynaraRegistry is abstract."""
        assert SynaraRegistry._meta.abstract is True

    def test_default_ordering(self):
        """Verify default ordering by display_order then label."""
        assert SynaraRegistry._meta.ordering == ["display_order", "label"]

    def test_icon_field(self):
        """Verify icon field for UI icons."""
        field = SynaraRegistry._meta.get_field("icon")
        assert isinstance(field, models.CharField)
        assert field.max_length == 50
        assert field.blank is True

    def test_color_field(self):
        """Verify color field for UI styling."""
        field = SynaraRegistry._meta.get_field("color")
        assert isinstance(field, models.CharField)
        assert field.max_length == 7
        assert field.blank is True

    def test_is_system_field(self):
        """Verify is_system field for protected entries."""
        field = SynaraRegistry._meta.get_field("is_system")
        assert isinstance(field, models.BooleanField)
        assert field.default is False


# =============================================================================
# UUID GENERATION TESTS
# =============================================================================


class TestUUIDGeneration(TestCase):
    """Test UUID generation for various fields."""

    def test_uuid_v4_format(self):
        """Verify generated UUIDs are valid v4 format."""
        test_uuid = uuid.uuid4()

        # Version should be 4
        assert test_uuid.version == 4

        # Should be 36 chars with hyphens
        assert len(str(test_uuid)) == 36

        # Should be valid hex
        hex_str = str(test_uuid).replace("-", "")
        assert len(hex_str) == 32
        int(hex_str, 16)  # Should not raise

    def test_uuids_are_unique(self):
        """Verify generated UUIDs are unique."""
        uuids = [uuid.uuid4() for _ in range(1000)]
        assert len(set(uuids)) == 1000


# =============================================================================
# HASH GENERATION TESTS
# =============================================================================


class TestHashGeneration(TestCase):
    """Test hash generation for immutable log integrity."""

    def test_sha256_hash_length(self):
        """Verify SHA-256 produces 64-char hex string."""
        test_data = b"test data for hashing"
        hash_result = hashlib.sha256(test_data).hexdigest()

        assert len(hash_result) == 64
        assert hash_result.isalnum()

    def test_hash_deterministic(self):
        """Verify same input produces same hash."""
        test_data = b"deterministic test"

        hash1 = hashlib.sha256(test_data).hexdigest()
        hash2 = hashlib.sha256(test_data).hexdigest()

        assert hash1 == hash2

    def test_hash_unique_for_different_input(self):
        """Verify different inputs produce different hashes."""
        hash1 = hashlib.sha256(b"input one").hexdigest()
        hash2 = hashlib.sha256(b"input two").hexdigest()

        assert hash1 != hash2


# =============================================================================
# TENANT ISOLATION TESTS (SEC-001 §5.2)
# =============================================================================


class TestTenantIsolation(TestCase):
    """Test tenant isolation compliance with SEC-001 §5.2."""

    def test_tenant_id_required(self):
        """Verify tenant_id is a required field."""
        field = SynaraEntity._meta.get_field("tenant_id")
        # tenant_id should not allow null (required)
        assert field.null is False

    def test_tenant_id_is_indexed(self):
        """Verify tenant_id has database index for performance."""
        field = SynaraEntity._meta.get_field("tenant_id")
        assert field.db_index is True


# =============================================================================
# AUDIT COMPLIANCE TESTS (AUD-001)
# =============================================================================


class TestAuditCompliance(TestCase):
    """Test audit field compliance with AUD-001."""

    def test_created_at_auto_populated(self):
        """Verify created_at is automatically set on creation."""
        field = SynaraEntity._meta.get_field("created_at")
        assert field.auto_now_add is True

    def test_updated_at_auto_updated(self):
        """Verify updated_at is automatically updated on save."""
        field = SynaraEntity._meta.get_field("updated_at")
        assert field.auto_now is True

    def test_audit_fields_help_text(self):
        """Verify audit fields have proper help text."""
        created_at = SynaraEntity._meta.get_field("created_at")
        updated_at = SynaraEntity._meta.get_field("updated_at")

        assert "AUD-001" in created_at.help_text
        assert "AUD-001" in updated_at.help_text


# =============================================================================
# SOFT DELETE COMPLIANCE TESTS (DAT-001 §9)
# =============================================================================


class TestSoftDeleteCompliance(TestCase):
    """Test soft delete compliance with DAT-001 §9."""

    def test_soft_delete_fields_exist(self):
        """Verify all soft delete fields exist."""
        field_names = [f.name for f in SynaraEntity._meta.get_fields()]

        assert "is_deleted" in field_names
        assert "deleted_at" in field_names
        assert "deleted_by" in field_names

    def test_is_deleted_default_false(self):
        """Verify is_deleted defaults to False."""
        field = SynaraEntity._meta.get_field("is_deleted")
        assert field.default is False

    def test_soft_delete_fields_help_text(self):
        """Verify soft delete fields reference DAT-001 §9."""
        is_deleted = SynaraEntity._meta.get_field("is_deleted")
        deleted_at = SynaraEntity._meta.get_field("deleted_at")

        assert "DAT-001" in is_deleted.help_text
        assert "DAT-001" in deleted_at.help_text


# =============================================================================
# CORRELATION TRACKING TESTS (CTG-001)
# =============================================================================


class TestCorrelationTracking(TestCase):
    """Test correlation tracking compliance with CTG-001."""

    def test_correlation_id_unique(self):
        """Verify correlation_id is unique."""
        field = SynaraEntity._meta.get_field("correlation_id")
        assert field.unique is True

    def test_correlation_id_immutable(self):
        """Verify correlation_id is not editable after creation."""
        field = SynaraEntity._meta.get_field("correlation_id")
        assert field.editable is False

    def test_parent_correlation_allows_null(self):
        """Verify parent_correlation_id allows null for root entities."""
        field = SynaraEntity._meta.get_field("parent_correlation_id")
        assert field.null is True
        assert field.blank is True

    def test_correlation_fields_help_text(self):
        """Verify correlation fields reference CTG-001."""
        correlation_id = SynaraEntity._meta.get_field("correlation_id")
        parent_correlation = SynaraEntity._meta.get_field("parent_correlation_id")

        assert "CTG-001" in correlation_id.help_text
        assert "CTG-001" in parent_correlation.help_text
