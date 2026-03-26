"""
Database behavior tests for Synara abstract base models.

Tests ACTUAL DATABASE BEHAVIOR (real round-trips) of:
- SynaraEntity: UUID PKs, correlation IDs, soft delete, managers, tenant filtering
- SynaraRegistry: code normalization, lookup, activation, choices
- SynaraImmutableLog: hash chain, immutability, integrity verification

Uses concrete subclasses with schema_editor() to create/drop tables since the
base models are abstract.

Standard: SDK-001, DAT-001 §9, AUD-001
"""

import uuid

from django.db import connection, models
from django.test import TestCase

from syn.core.base_models import SynaraEntity, SynaraImmutableLog, SynaraRegistry

# =============================================================================
# CONCRETE TEST SUBCLASSES
# =============================================================================


class ConcreteEntity(SynaraEntity):
    name = models.CharField(max_length=100, default="")

    class Meta(SynaraEntity.Meta):
        app_label = "syn_core"
        db_table = "test_concrete_entity"


class ConcreteRegistry(SynaraRegistry):
    class Meta(SynaraRegistry.Meta):
        app_label = "syn_core"
        db_table = "test_concrete_registry"


class ConcreteImmutableLog(SynaraImmutableLog):
    class Meta(SynaraImmutableLog.Meta):
        app_label = "syn_core"
        db_table = "test_concrete_immutable_log"


# =============================================================================
# SYNARA ENTITY DB TESTS
# =============================================================================


class SynaraEntityDBTest(TestCase):
    """Database behavior tests for SynaraEntity base class."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        with connection.schema_editor() as editor:
            editor.create_model(ConcreteEntity)

    @classmethod
    def tearDownClass(cls):
        with connection.schema_editor() as editor:
            editor.delete_model(ConcreteEntity)
        super().tearDownClass()

    def test_save_assigns_uuid_and_created_at(self):
        """save() assigns UUID id and created_at timestamp."""
        entity = ConcreteEntity.objects.create(name="test")
        self.assertIsNotNone(entity.id)
        self.assertIsInstance(entity.id, uuid.UUID)
        self.assertIsNotNone(entity.created_at)

    def test_save_assigns_unique_correlation_id(self):
        """save() assigns a unique correlation_id."""
        e1 = ConcreteEntity.objects.create(name="first")
        e2 = ConcreteEntity.objects.create(name="second")
        self.assertIsNotNone(e1.correlation_id)
        self.assertIsInstance(e1.correlation_id, uuid.UUID)
        self.assertNotEqual(e1.correlation_id, e2.correlation_id)

    def test_soft_delete_sets_flags(self):
        """delete() sets is_deleted, deleted_at, deleted_by."""
        entity = ConcreteEntity.objects.create(name="to_delete")
        entity.delete(deleted_by="user-42")

        entity.refresh_from_db()
        self.assertTrue(entity.is_deleted)
        self.assertIsNotNone(entity.deleted_at)
        self.assertEqual(entity.deleted_by, "user-42")

    def test_hard_delete_removes_from_db(self):
        """delete(hard_delete=True) permanently removes the record."""
        entity = ConcreteEntity.objects.create(name="hard_delete_me")
        pk = entity.pk
        entity.delete(hard_delete=True)

        self.assertFalse(ConcreteEntity.all_objects.filter(pk=pk).exists())

    def test_restore_clears_delete_markers(self):
        """restore() clears is_deleted, deleted_at, deleted_by."""
        entity = ConcreteEntity.objects.create(name="restorable")
        entity.delete(deleted_by="user-42")
        entity.restore(restored_by="admin")

        entity.refresh_from_db()
        self.assertFalse(entity.is_deleted)
        self.assertIsNone(entity.deleted_at)
        self.assertEqual(entity.deleted_by, "")
        self.assertEqual(entity.updated_by, "admin")

    def test_restore_on_non_deleted_is_noop(self):
        """restore() on a non-deleted entity is a no-op."""
        entity = ConcreteEntity.objects.create(name="alive")
        original_updated_at = entity.updated_at
        entity.restore()

        entity.refresh_from_db()
        self.assertFalse(entity.is_deleted)
        # updated_at should not have changed since restore() returned early
        self.assertEqual(entity.updated_at, original_updated_at)

    def test_objects_manager_excludes_soft_deleted(self):
        """Default objects manager excludes soft-deleted records."""
        alive = ConcreteEntity.objects.create(name="alive")
        deleted = ConcreteEntity.objects.create(name="deleted")
        deleted.delete()

        qs = ConcreteEntity.objects.all()
        self.assertIn(alive.pk, [e.pk for e in qs])
        self.assertNotIn(deleted.pk, [e.pk for e in qs])

    def test_all_objects_manager_includes_all(self):
        """all_objects manager includes soft-deleted records."""
        alive = ConcreteEntity.objects.create(name="alive2")
        deleted = ConcreteEntity.objects.create(name="deleted2")
        deleted.delete()

        pks = list(ConcreteEntity.all_objects.values_list("pk", flat=True))
        self.assertIn(alive.pk, pks)
        self.assertIn(deleted.pk, pks)

    def test_objects_with_deleted_includes_all(self):
        """objects.with_deleted() includes soft-deleted records."""
        alive = ConcreteEntity.objects.create(name="wd_alive")
        deleted = ConcreteEntity.objects.create(name="wd_deleted")
        deleted.delete()

        pks = list(ConcreteEntity.objects.with_deleted().values_list("pk", flat=True))
        self.assertIn(alive.pk, pks)
        self.assertIn(deleted.pk, pks)

    def test_objects_deleted_only(self):
        """objects.deleted_only() returns only soft-deleted records."""
        alive = ConcreteEntity.objects.create(name="do_alive")
        deleted = ConcreteEntity.objects.create(name="do_deleted")
        deleted.delete()

        pks = list(ConcreteEntity.objects.deleted_only().values_list("pk", flat=True))
        self.assertNotIn(alive.pk, pks)
        self.assertIn(deleted.pk, pks)

    def test_for_tenant_filters_by_tenant_id(self):
        """for_tenant() filters records by tenant_id."""
        t1 = uuid.uuid4()
        t2 = uuid.uuid4()
        e1 = ConcreteEntity.objects.create(name="tenant1", tenant_id=t1)
        e2 = ConcreteEntity.objects.create(name="tenant2", tenant_id=t2)

        pks = list(ConcreteEntity.objects.for_tenant(t1).values_list("pk", flat=True))
        self.assertIn(e1.pk, pks)
        self.assertNotIn(e2.pk, pks)

    def test_to_dict_serialization(self):
        """to_dict() includes all expected keys."""
        entity = ConcreteEntity.objects.create(name="serial", created_by="user-1")
        d = entity.to_dict()

        expected_keys = {
            "id",
            "correlation_id",
            "parent_correlation_id",
            "tenant_id",
            "created_at",
            "updated_at",
            "created_by",
            "updated_by",
            "is_deleted",
            "metadata",
        }
        self.assertEqual(set(d.keys()), expected_keys)
        self.assertEqual(d["id"], str(entity.id))
        self.assertEqual(d["created_by"], "user-1")
        self.assertFalse(d["is_deleted"])


# =============================================================================
# SYNARA REGISTRY DB TESTS
# =============================================================================


class SynaraRegistryDBTest(TestCase):
    """Database behavior tests for SynaraRegistry base class."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        with connection.schema_editor() as editor:
            editor.create_model(ConcreteRegistry)

    @classmethod
    def tearDownClass(cls):
        with connection.schema_editor() as editor:
            editor.delete_model(ConcreteRegistry)
        super().tearDownClass()

    def test_save_lowercases_code(self):
        """save() lowercases the code field."""
        entry = ConcreteRegistry.objects.create(code="SUBMITTED", label="Submitted")
        entry.refresh_from_db()
        self.assertEqual(entry.code, "submitted")

    def test_get_by_code_finds_entry(self):
        """get_by_code() finds an entry by code."""
        ConcreteRegistry.objects.create(code="approved", label="Approved")
        entry = ConcreteRegistry.get_by_code("approved")
        self.assertEqual(entry.label, "Approved")

    def test_get_by_code_tenant_specific_first(self):
        """get_by_code() with tenant_id returns tenant-specific entry first."""
        tenant = uuid.uuid4()
        ConcreteRegistry.objects.create(
            code="status", label="Global Status", tenant_id=None
        )
        ConcreteRegistry.objects.create(
            code="status", label="Tenant Status", tenant_id=tenant
        )
        entry = ConcreteRegistry.get_by_code("status", tenant_id=tenant)
        self.assertEqual(entry.label, "Tenant Status")

    def test_get_by_code_falls_back_to_global(self):
        """get_by_code() with tenant_id falls back to global if no tenant match."""
        tenant = uuid.uuid4()
        ConcreteRegistry.objects.create(
            code="fallback", label="Global Fallback", tenant_id=None
        )
        entry = ConcreteRegistry.get_by_code("fallback", tenant_id=tenant)
        self.assertEqual(entry.label, "Global Fallback")

    def test_get_by_code_raises_does_not_exist(self):
        """get_by_code() raises DoesNotExist for missing code."""
        with self.assertRaises(ConcreteRegistry.DoesNotExist):
            ConcreteRegistry.get_by_code("nonexistent_code_xyz")

    def test_deactivate(self):
        """deactivate() sets is_active=False."""
        entry = ConcreteRegistry.objects.create(code="active_one", label="Active")
        entry.deactivate()
        entry.refresh_from_db()
        self.assertFalse(entry.is_active)

    def test_activate(self):
        """activate() sets is_active=True."""
        entry = ConcreteRegistry.objects.create(code="inactive_one", label="Inactive")
        entry.deactivate()
        entry.activate()
        entry.refresh_from_db()
        self.assertTrue(entry.is_active)

    def test_get_choices_returns_tuples(self):
        """get_choices() returns list of (code, label) tuples."""
        ConcreteRegistry.objects.create(
            code="choice_a", label="Choice A", display_order=1
        )
        ConcreteRegistry.objects.create(
            code="choice_b", label="Choice B", display_order=2
        )
        choices = ConcreteRegistry.get_choices()
        self.assertIsInstance(choices, list)
        # Verify our entries are in the list
        codes = [c[0] for c in choices]
        self.assertIn("choice_a", codes)
        self.assertIn("choice_b", codes)

    def test_to_dict_serialization(self):
        """to_dict() includes all expected keys."""
        entry = ConcreteRegistry.objects.create(
            code="serialized",
            label="Serialized",
            description="desc",
            display_order=5,
            icon="check",
            color="#4CAF50",
        )
        d = entry.to_dict()
        expected_keys = {
            "id",
            "code",
            "label",
            "description",
            "display_order",
            "icon",
            "color",
            "is_active",
            "is_system",
            "metadata",
        }
        self.assertEqual(set(d.keys()), expected_keys)
        self.assertEqual(d["code"], "serialized")
        self.assertEqual(d["display_order"], 5)


# =============================================================================
# SYNARA IMMUTABLE LOG DB TESTS
# =============================================================================


class SynaraImmutableLogDBTest(TestCase):
    """Database behavior tests for SynaraImmutableLog base class."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        with connection.schema_editor() as editor:
            editor.create_model(ConcreteImmutableLog)

    @classmethod
    def tearDownClass(cls):
        with connection.schema_editor() as editor:
            editor.delete_model(ConcreteImmutableLog)
        super().tearDownClass()

    def _make_entry(self, tenant_id=None, event_name="test.event", **kwargs):
        """Helper to create an immutable log entry."""
        defaults = {
            "tenant_id": tenant_id,
            "event_name": event_name,
            "actor": "test-user",
        }
        defaults.update(kwargs)
        return ConcreteImmutableLog.objects.create(**defaults)

    def test_save_computes_entry_hash(self):
        """save() computes a non-empty 64-char hex entry_hash."""
        entry = self._make_entry()
        self.assertIsNotNone(entry.entry_hash)
        self.assertEqual(len(entry.entry_hash), 64)
        # Verify it's valid hex
        int(entry.entry_hash, 16)

    def test_first_entry_previous_hash_empty(self):
        """First entry in a chain has empty previous_hash."""
        tenant = uuid.uuid4()
        entry = self._make_entry(tenant_id=tenant)
        self.assertEqual(entry.previous_hash, "")

    def test_second_entry_chains_to_first(self):
        """Second entry's previous_hash equals first entry's entry_hash."""
        tenant = uuid.uuid4()
        first = self._make_entry(tenant_id=tenant, event_name="chain.first")
        second = self._make_entry(tenant_id=tenant, event_name="chain.second")
        self.assertEqual(second.previous_hash, first.entry_hash)

    def test_save_existing_raises_value_error(self):
        """Second save() on an existing entry raises ValueError (immutability)."""
        entry = self._make_entry()
        with self.assertRaises(ValueError) as ctx:
            entry.save()
        self.assertIn("immutable", str(ctx.exception).lower())

    def test_delete_raises_permission_error(self):
        """delete() raises PermissionError."""
        entry = self._make_entry()
        with self.assertRaises(PermissionError) as ctx:
            entry.delete()
        self.assertIn("immutable", str(ctx.exception).lower())

    def test_verify_integrity_true_for_untampered(self):
        """verify_integrity() returns True for an untampered entry."""
        entry = self._make_entry()
        self.assertTrue(entry.verify_integrity())

    def test_verify_integrity_false_if_hash_tampered(self):
        """verify_integrity() returns False if entry_hash was tampered."""
        entry = self._make_entry()
        # Tamper the hash via raw SQL to bypass immutability
        ConcreteImmutableLog.objects.filter(pk=entry.pk).update(entry_hash="0" * 64)
        entry.refresh_from_db()
        self.assertFalse(entry.verify_integrity())

    def test_verify_chain_valid_for_intact_chain(self):
        """verify_chain() returns is_valid=True for an intact chain."""
        tenant = uuid.uuid4()
        self._make_entry(tenant_id=tenant, event_name="v.first")
        self._make_entry(tenant_id=tenant, event_name="v.second")
        self._make_entry(tenant_id=tenant, event_name="v.third")

        result = ConcreteImmutableLog.verify_chain(tenant_id=tenant)
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["entries_checked"], 3)
        self.assertEqual(result["failures"], [])

    def test_verify_chain_invalid_for_tampered_hash(self):
        """verify_chain() returns is_valid=False when entry_hash is tampered."""
        tenant = uuid.uuid4()
        first = self._make_entry(tenant_id=tenant, event_name="t.first")
        self._make_entry(tenant_id=tenant, event_name="t.second")

        # Tamper the first entry's hash via raw update
        ConcreteImmutableLog.objects.filter(pk=first.pk).update(
            entry_hash="dead" + "0" * 60
        )

        result = ConcreteImmutableLog.verify_chain(tenant_id=tenant)
        self.assertFalse(result["is_valid"])
        self.assertGreater(len(result["failures"]), 0)

    def test_hash_is_deterministic(self):
        """Same content produces the same hash (deterministic)."""
        fixed_corr = uuid.uuid4()
        tenant = uuid.uuid4()

        entry = ConcreteImmutableLog(
            correlation_id=fixed_corr,
            tenant_id=tenant,
            event_name="deterministic.test",
            actor="user-1",
            before_snapshot={},
            after_snapshot={"key": "value"},
            changes={"key": ["", "value"]},
            reason="testing",
            previous_hash="",
        )
        hash1 = entry._compute_entry_hash()
        hash2 = entry._compute_entry_hash()
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)
