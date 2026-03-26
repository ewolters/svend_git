"""
Tests for Synara Core Infrastructure (SDK-001 §6).

Covers:
- Mixins (CorrelationMixin, TenantMixin, AuditMixin, SoftDeleteMixin,
          EventEmitterMixin, LifecycleMixin, MetadataMixin, VersioningMixin)
- Configuration (DeploymentProfile, FeatureFlagState, SecretClassification,
                 FeatureFlags, AuthenticationSettings, SynaraSettings, anti-patterns)
- Versioning (SemanticVersion, breaking change detection, deprecation lifecycle,
              checksum verification, VersionManager, SynaraVersionedMixin)

Standard: TST-001
SOC 2: CC6.1 (Tenant Isolation), CC7.2 (Integrity), CC8.1 (Change Management)
"""

import uuid
from datetime import timedelta
from unittest.mock import MagicMock, patch

from django.db import models
from django.test import SimpleTestCase
from django.utils import timezone

from syn.core.config import (
    AuthenticationSettings,
    DeploymentProfile,
    FeatureFlags,
    FeatureFlagState,
    LogFormat,
    LogLevel,
    ObservabilitySettings,
    SecretClassification,
    SecuritySettings,
)
from syn.core.mixins import (
    AuditMixin,
    CorrelationMixin,
    EventEmitterMixin,
    LifecycleMixin,
    MetadataMixin,
    SoftDeleteManager,
    SoftDeleteMixin,
    TenantMixin,
    VersioningMixin,
)
from syn.core.versioning import (
    DeprecationInfo,
    SchemaDiff,
    SemanticVersion,
    VersionChangeType,
    VersionCompatibility,
    VersionLifecycle,
    VersionManager,
    compute_version_checksum,
    detect_breaking_changes,
    verify_version_checksum,
)

# =============================================================================
# MIXIN FIELD TESTS (SimpleTestCase — no DB)
# =============================================================================


class CorrelationMixinFieldTest(SimpleTestCase):
    """Test CorrelationMixin field definitions (CTG-001 §5)."""

    def test_correlation_id_field_exists(self):
        field = CorrelationMixin._meta.get_field("correlation_id")
        self.assertIsInstance(field, models.UUIDField)
        self.assertTrue(field.unique)
        self.assertTrue(field.db_index)
        self.assertFalse(field.editable)

    def test_parent_correlation_id_field_exists(self):
        field = CorrelationMixin._meta.get_field("parent_correlation_id")
        self.assertIsInstance(field, models.UUIDField)
        self.assertTrue(field.null)
        self.assertTrue(field.blank)
        self.assertTrue(field.db_index)

    def test_is_abstract(self):
        self.assertTrue(CorrelationMixin._meta.abstract)


class TenantMixinFieldTest(SimpleTestCase):
    """Test TenantMixin field definitions (SEC-001 §5.2)."""

    def test_tenant_id_field_exists(self):
        field = TenantMixin._meta.get_field("tenant_id")
        self.assertIsInstance(field, models.UUIDField)
        self.assertTrue(field.db_index)
        self.assertFalse(field.null)

    def test_is_abstract(self):
        self.assertTrue(TenantMixin._meta.abstract)


class AuditMixinFieldTest(SimpleTestCase):
    """Test AuditMixin field definitions (AUD-001)."""

    def test_created_at_auto_now_add(self):
        field = AuditMixin._meta.get_field("created_at")
        self.assertIsInstance(field, models.DateTimeField)
        self.assertTrue(field.auto_now_add)
        self.assertTrue(field.db_index)

    def test_updated_at_auto_now(self):
        field = AuditMixin._meta.get_field("updated_at")
        self.assertIsInstance(field, models.DateTimeField)
        self.assertTrue(field.auto_now)

    def test_created_by_field(self):
        field = AuditMixin._meta.get_field("created_by")
        self.assertIsInstance(field, models.CharField)
        self.assertEqual(field.max_length, 255)
        self.assertTrue(field.blank)
        self.assertEqual(field.default, "")

    def test_updated_by_field(self):
        field = AuditMixin._meta.get_field("updated_by")
        self.assertIsInstance(field, models.CharField)
        self.assertEqual(field.max_length, 255)
        self.assertTrue(field.blank)

    def test_is_abstract(self):
        self.assertTrue(AuditMixin._meta.abstract)


class SoftDeleteMixinFieldTest(SimpleTestCase):
    """Test SoftDeleteMixin field definitions (DAT-001 §9)."""

    def test_is_deleted_field(self):
        field = SoftDeleteMixin._meta.get_field("is_deleted")
        self.assertIsInstance(field, models.BooleanField)
        self.assertFalse(field.default)
        self.assertTrue(field.db_index)

    def test_deleted_at_field(self):
        field = SoftDeleteMixin._meta.get_field("deleted_at")
        self.assertIsInstance(field, models.DateTimeField)
        self.assertTrue(field.null)
        self.assertTrue(field.blank)

    def test_deleted_by_field(self):
        field = SoftDeleteMixin._meta.get_field("deleted_by")
        self.assertIsInstance(field, models.CharField)
        self.assertEqual(field.max_length, 255)

    def test_has_soft_delete_manager(self):
        # Django wraps managers in ManagerDescriptor for abstract models
        # Check the manager's creation_counter and auto_created properties
        descriptor = SoftDeleteMixin.__dict__["objects"]
        self.assertEqual(descriptor.manager.auto_created, False)
        self.assertIsInstance(descriptor.manager, SoftDeleteManager)

    def test_has_all_objects_manager(self):
        descriptor = SoftDeleteMixin.__dict__["all_objects"]
        self.assertIsInstance(descriptor.manager, models.Manager)

    def test_is_abstract(self):
        self.assertTrue(SoftDeleteMixin._meta.abstract)


class EventEmitterMixinFieldTest(SimpleTestCase):
    """Test EventEmitterMixin class attributes (EVT-001)."""

    def test_default_event_domain_empty(self):
        self.assertEqual(EventEmitterMixin.event_domain, "")

    def test_default_emit_on_save_false(self):
        self.assertFalse(EventEmitterMixin.emit_on_save)

    def test_is_abstract(self):
        self.assertTrue(EventEmitterMixin._meta.abstract)


class LifecycleMixinFieldTest(SimpleTestCase):
    """Test LifecycleMixin class attributes (GOV-001)."""

    def test_default_lifecycle_states_empty(self):
        self.assertEqual(LifecycleMixin.lifecycle_states, [])

    def test_default_terminal_states_empty(self):
        self.assertEqual(LifecycleMixin.terminal_states, [])

    def test_default_initial_state_empty(self):
        self.assertEqual(LifecycleMixin.initial_state, "")

    def test_default_transitions_empty(self):
        self.assertEqual(LifecycleMixin.transitions, {})

    def test_is_abstract(self):
        self.assertTrue(LifecycleMixin._meta.abstract)


class MetadataMixinFieldTest(SimpleTestCase):
    """Test MetadataMixin field definitions (MOD-001 §10)."""

    def test_metadata_field(self):
        field = MetadataMixin._meta.get_field("metadata")
        self.assertIsInstance(field, models.JSONField)
        self.assertTrue(field.blank)

    def test_is_abstract(self):
        self.assertTrue(MetadataMixin._meta.abstract)


class VersioningMixinFieldTest(SimpleTestCase):
    """Test VersioningMixin field definitions (SDK-001 §6.8)."""

    def test_version_field(self):
        field = VersioningMixin._meta.get_field("version")
        self.assertIsInstance(field, models.CharField)
        self.assertEqual(field.max_length, 20)
        self.assertEqual(field.default, "1.0.0")

    def test_version_number_field(self):
        field = VersioningMixin._meta.get_field("version_number")
        self.assertIsInstance(field, models.PositiveIntegerField)
        self.assertEqual(field.default, 1)

    def test_is_abstract(self):
        self.assertTrue(VersioningMixin._meta.abstract)


# =============================================================================
# MIXIN BEHAVIOR TESTS (SimpleTestCase — mock-based)
# =============================================================================


class AuditMixinBehaviorTest(SimpleTestCase):
    """Test AuditMixin helper methods."""

    def _make_instance(self):
        inst = MagicMock(spec=AuditMixin)
        inst.created_by = ""
        inst.updated_by = ""
        return inst

    def test_set_created_by_sets_both_fields(self):
        inst = self._make_instance()
        AuditMixin.set_created_by(inst, "user-123")
        self.assertEqual(inst.created_by, "user-123")
        self.assertEqual(inst.updated_by, "user-123")

    def test_set_updated_by_only_sets_updated(self):
        inst = self._make_instance()
        inst.created_by = "user-123"
        AuditMixin.set_updated_by(inst, "user-456")
        self.assertEqual(inst.updated_by, "user-456")
        self.assertEqual(inst.created_by, "user-123")


class EventEmitterMixinBehaviorTest(SimpleTestCase):
    """Test EventEmitterMixin emit_event logic."""

    def test_emit_event_no_domain_logs_warning(self):
        inst = MagicMock(spec=EventEmitterMixin)
        inst.event_domain = ""
        inst.__class__.__name__ = "TestModel"
        with patch("syn.core.mixins.logger") as mock_logger:
            EventEmitterMixin.emit_event(inst, "created")
            mock_logger.warning.assert_called_once()

    def test_emit_event_builds_payload(self):
        inst = MagicMock(spec=EventEmitterMixin)
        inst.event_domain = "test.model"
        inst.pk = uuid.uuid4()
        inst.__class__.__name__ = "TestModel"
        inst.correlation_id = uuid.uuid4()
        inst.tenant_id = uuid.uuid4()

        with patch("syn.core.mixins.logger"):
            EventEmitterMixin.emit_event(inst, "created", {"extra": "data"})

    def test_emit_event_handles_import_error(self):
        inst = MagicMock(spec=EventEmitterMixin)
        inst.event_domain = "test.model"
        inst.pk = uuid.uuid4()
        inst.__class__.__name__ = "TestModel"
        del inst.correlation_id
        del inst.tenant_id

        with patch("syn.core.mixins.logger") as mock_logger:
            EventEmitterMixin.emit_event(inst, "deleted")
            mock_logger.info.assert_called_once()


class LifecycleMixinBehaviorTest(SimpleTestCase):
    """Test LifecycleMixin state machine logic."""

    def _make_instance(self, status="draft"):
        inst = MagicMock()
        inst.status = status
        inst.lifecycle_states = ["draft", "submitted", "approved", "rejected"]
        inst.terminal_states = ["approved", "rejected"]
        inst.transitions = {
            "draft": ["submitted"],
            "submitted": ["approved", "rejected"],
        }
        # Wire real methods
        inst.get_status_value = lambda: LifecycleMixin.get_status_value(inst)
        inst.is_terminal = lambda: LifecycleMixin.is_terminal(inst)
        inst.can_transition_to = lambda s: LifecycleMixin.can_transition_to(inst, s)
        return inst

    def test_get_status_value_returns_string(self):
        inst = self._make_instance("draft")
        self.assertEqual(inst.get_status_value(), "draft")

    def test_is_terminal_true_for_approved(self):
        inst = self._make_instance("approved")
        self.assertTrue(inst.is_terminal())

    def test_is_terminal_false_for_draft(self):
        inst = self._make_instance("draft")
        self.assertFalse(inst.is_terminal())

    def test_can_transition_valid(self):
        inst = self._make_instance("draft")
        self.assertTrue(inst.can_transition_to("submitted"))

    def test_can_transition_invalid(self):
        inst = self._make_instance("draft")
        self.assertFalse(inst.can_transition_to("approved"))

    def test_can_transition_blocked_from_terminal(self):
        inst = self._make_instance("approved")
        self.assertFalse(inst.can_transition_to("draft"))

    def test_transition_to_raises_on_invalid(self):
        inst = self._make_instance("draft")
        inst.emit_event = MagicMock()
        with self.assertRaises(ValueError):
            LifecycleMixin.transition_to(inst, "approved")

    def test_transition_to_force_bypasses_validation(self):
        inst = self._make_instance("draft")
        inst.emit_event = MagicMock()
        inst._set_status = MagicMock()
        result = LifecycleMixin.transition_to(inst, "approved", force=True)
        self.assertTrue(result)
        inst._set_status.assert_called_once_with("approved")

    def test_transition_to_emits_event(self):
        inst = self._make_instance("draft")
        inst.emit_event = MagicMock()
        inst._set_status = MagicMock()
        LifecycleMixin.transition_to(inst, "submitted", actor="user-1", reason="ready")
        inst.emit_event.assert_called_once()
        call_args = inst.emit_event.call_args
        self.assertEqual(call_args[0][0], "status_changed")
        payload = call_args[0][1]
        self.assertEqual(payload["from_status"], "draft")
        self.assertEqual(payload["to_status"], "submitted")
        self.assertEqual(payload["actor"], "user-1")

    def test_can_transition_no_transitions_defined(self):
        inst = self._make_instance("draft")
        inst.transitions = {}
        self.assertTrue(inst.can_transition_to("submitted"))
        self.assertTrue(inst.can_transition_to("approved"))


class MetadataMixinBehaviorTest(SimpleTestCase):
    """Test MetadataMixin get/set/remove helpers."""

    def _make_instance(self, metadata=None):
        inst = MagicMock()
        inst.metadata = metadata or {}
        inst.save = MagicMock()
        return inst

    def test_get_metadata_simple_key(self):
        inst = self._make_instance({"color": "blue"})
        result = MetadataMixin.get_metadata(inst, "color")
        self.assertEqual(result, "blue")

    def test_get_metadata_dot_notation(self):
        inst = self._make_instance({"nested": {"deep": "value"}})
        result = MetadataMixin.get_metadata(inst, "nested.deep")
        self.assertEqual(result, "value")

    def test_get_metadata_default(self):
        inst = self._make_instance({})
        result = MetadataMixin.get_metadata(inst, "missing", "default_val")
        self.assertEqual(result, "default_val")

    def test_get_metadata_non_dict_path_returns_default(self):
        inst = self._make_instance({"key": "string_not_dict"})
        result = MetadataMixin.get_metadata(inst, "key.sub", "fallback")
        self.assertEqual(result, "fallback")

    def test_set_metadata_simple(self):
        inst = self._make_instance({})
        MetadataMixin.set_metadata(inst, "key", "val")
        self.assertEqual(inst.metadata["key"], "val")
        inst.save.assert_called_once_with(update_fields=["metadata"])

    def test_set_metadata_dot_notation_creates_nested(self):
        inst = self._make_instance({})
        MetadataMixin.set_metadata(inst, "a.b.c", 42)
        self.assertEqual(inst.metadata["a"]["b"]["c"], 42)

    def test_remove_metadata_existing_key(self):
        inst = self._make_instance({"key": "val", "other": "keep"})
        MetadataMixin.remove_metadata(inst, "key")
        self.assertNotIn("key", inst.metadata)
        self.assertIn("other", inst.metadata)
        inst.save.assert_called_once()

    def test_remove_metadata_missing_key_no_error(self):
        inst = self._make_instance({})
        MetadataMixin.remove_metadata(inst, "nonexistent")
        inst.save.assert_not_called()

    def test_remove_metadata_dot_notation(self):
        inst = self._make_instance({"a": {"b": "val", "c": "keep"}})
        MetadataMixin.remove_metadata(inst, "a.b")
        self.assertNotIn("b", inst.metadata["a"])
        self.assertIn("c", inst.metadata["a"])


class VersioningMixinBehaviorTest(SimpleTestCase):
    """Test VersioningMixin version parsing and incrementing."""

    def _make_instance(self, version="1.0.0", version_number=1):
        class FakeVersioned:
            parse_version = VersioningMixin.parse_version
            increment_patch = VersioningMixin.increment_patch
            increment_minor = VersioningMixin.increment_minor
            increment_major = VersioningMixin.increment_major

        inst = FakeVersioned()
        inst.version = version
        inst.version_number = version_number
        return inst

    def test_parse_version_standard(self):
        inst = self._make_instance("2.3.4")
        result = inst.parse_version()
        self.assertEqual(result, (2, 3, 4))

    def test_parse_version_invalid_returns_default(self):
        inst = self._make_instance("invalid")
        result = inst.parse_version()
        self.assertEqual(result, (1, 0, 0))

    def test_increment_patch(self):
        inst = self._make_instance("1.2.3", 5)
        inst.increment_patch()
        self.assertEqual(inst.version, "1.2.4")
        self.assertEqual(inst.version_number, 6)

    def test_increment_minor(self):
        inst = self._make_instance("1.2.3", 5)
        inst.increment_minor()
        self.assertEqual(inst.version, "1.3.0")
        self.assertEqual(inst.version_number, 6)

    def test_increment_major(self):
        inst = self._make_instance("1.2.3", 5)
        inst.increment_major()
        self.assertEqual(inst.version, "2.0.0")
        self.assertEqual(inst.version_number, 6)


# =============================================================================
# CONFIG ENUM TESTS (SimpleTestCase)
# =============================================================================


class DeploymentProfileEnumTest(SimpleTestCase):
    """Test DeploymentProfile enum values (CONFIG-001 §7)."""

    def test_all_profiles_defined(self):
        expected = {"local", "docker", "staging", "production", "render"}
        actual = {p.value for p in DeploymentProfile}
        self.assertEqual(actual, expected)

    def test_is_str_enum(self):
        self.assertIsInstance(DeploymentProfile.LOCAL, str)
        self.assertEqual(DeploymentProfile.LOCAL, "local")


class FeatureFlagStateEnumTest(SimpleTestCase):
    """Test FeatureFlagState enum values (CONFIG-001 §5.2)."""

    def test_all_states_defined(self):
        expected = {"disabled", "beta", "ga", "deprecated", "sunset"}
        actual = {s.value for s in FeatureFlagState}
        self.assertEqual(actual, expected)

    def test_lifecycle_order(self):
        states = [s.value for s in FeatureFlagState]
        self.assertEqual(states[0], "disabled")
        self.assertEqual(states[-1], "sunset")


class SecretClassificationEnumTest(SimpleTestCase):
    """Test SecretClassification enum values (CONFIG-001 §9.1)."""

    def test_all_levels_defined(self):
        expected = {"critical", "sensitive", "internal", "public"}
        actual = {c.value for c in SecretClassification}
        self.assertEqual(actual, expected)

    def test_critical_is_highest(self):
        self.assertEqual(SecretClassification.CRITICAL.value, "critical")

    def test_public_is_lowest(self):
        self.assertEqual(SecretClassification.PUBLIC.value, "public")


class LogLevelEnumTest(SimpleTestCase):
    """Test LogLevel enum values."""

    def test_all_levels_defined(self):
        expected = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        actual = {ll.value for ll in LogLevel}
        self.assertEqual(actual, expected)


class LogFormatEnumTest(SimpleTestCase):
    """Test LogFormat enum values."""

    def test_formats_defined(self):
        expected = {"json", "text"}
        actual = {lf.value for lf in LogFormat}
        self.assertEqual(actual, expected)


# =============================================================================
# CONFIG SETTINGS TESTS (SimpleTestCase)
# =============================================================================


class FeatureFlagsTest(SimpleTestCase):
    """Test FeatureFlags Pydantic settings (CONFIG-001 §5)."""

    def test_default_soft_delete_cascade_enabled(self):
        flags = FeatureFlags()
        self.assertTrue(flags.enable_soft_delete_cascade)

    def test_default_oidc_disabled(self):
        flags = FeatureFlags()
        self.assertFalse(flags.enable_oidc)

    def test_default_api_keys_disabled(self):
        flags = FeatureFlags()
        self.assertFalse(flags.enable_api_keys)

    def test_default_vault_disabled(self):
        flags = FeatureFlags()
        self.assertFalse(flags.enable_vault_secrets)

    def test_default_cognition_engine_enabled(self):
        flags = FeatureFlags()
        self.assertTrue(flags.enable_cognition_engine)

    def test_env_prefix_is_feature(self):
        self.assertEqual(FeatureFlags.model_config["env_prefix"], "FEATURE_")

    def test_extra_ignored(self):
        self.assertEqual(FeatureFlags.model_config["extra"], "ignore")


class AuthenticationSettingsTest(SimpleTestCase):
    """Test AuthenticationSettings validation (CONFIG-001 §4.3)."""

    def test_default_algorithm_hs256(self):
        settings = AuthenticationSettings(JWT_SECRET="test-secret-key-123")
        self.assertEqual(settings.jwt_algorithm, "HS256")

    def test_default_expiration_3600(self):
        settings = AuthenticationSettings(JWT_SECRET="test-secret-key-123")
        self.assertEqual(settings.jwt_expiration_seconds, 3600)

    def test_default_require_auth_true(self):
        settings = AuthenticationSettings(JWT_SECRET="test-secret-key-123")
        self.assertTrue(settings.require_auth)

    def test_invalid_algorithm_rejected(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            AuthenticationSettings(
                JWT_SECRET="test-secret-key-123", JWT_ALGORITHM="INVALID"
            )

    def test_valid_algorithms_accepted(self):
        for algo in ["HS256", "HS384", "HS512", "RS256", "RS384", "RS512"]:
            settings = AuthenticationSettings(
                JWT_SECRET="test-secret-key-123", JWT_ALGORITHM=algo
            )
            self.assertEqual(settings.jwt_algorithm, algo)


class ObservabilitySettingsTest(SimpleTestCase):
    """Test ObservabilitySettings defaults (CONFIG-001 §4.2)."""

    def test_default_log_level_info(self):
        settings = ObservabilitySettings()
        self.assertEqual(settings.log_level, LogLevel.INFO)

    def test_default_log_format_json(self):
        settings = ObservabilitySettings()
        self.assertEqual(settings.log_format, LogFormat.JSON)

    def test_default_otel_endpoint_none(self):
        settings = ObservabilitySettings()
        self.assertIsNone(settings.otel_endpoint)

    def test_default_service_name(self):
        settings = ObservabilitySettings()
        self.assertEqual(settings.otel_service_name, "synara-core")


class SecuritySettingsTest(SimpleTestCase):
    """Test SecuritySettings defaults (CONFIG-001 §4.3)."""

    def test_default_csrf_cookie_not_secure(self):
        settings = SecuritySettings()
        self.assertFalse(settings.csrf_cookie_secure)

    def test_default_x_frame_sameorigin(self):
        settings = SecuritySettings()
        self.assertEqual(settings.x_frame_options, "SAMEORIGIN")

    def test_default_content_type_nosniff_true(self):
        settings = SecuritySettings()
        self.assertTrue(settings.secure_content_type_nosniff)


# =============================================================================
# SEMANTIC VERSION TESTS (SimpleTestCase)
# =============================================================================


class SemanticVersionParseTest(SimpleTestCase):
    """Test SemanticVersion parsing (SCHEMA-001 §6)."""

    def test_parse_standard_version(self):
        v = SemanticVersion.parse("1.2.3")
        self.assertEqual(v.major, 1)
        self.assertEqual(v.minor, 2)
        self.assertEqual(v.patch, 3)

    def test_parse_with_prerelease(self):
        v = SemanticVersion.parse("2.0.0-beta.1")
        self.assertEqual(v.major, 2)
        self.assertEqual(v.prerelease, "beta.1")

    def test_parse_with_build(self):
        v = SemanticVersion.parse("1.0.0+build.456")
        self.assertEqual(v.build, "build.456")

    def test_parse_single_digit(self):
        v = SemanticVersion.parse("3")
        self.assertEqual(v, SemanticVersion(3, 0, 0))

    def test_parse_two_digits(self):
        v = SemanticVersion.parse("2.5")
        self.assertEqual(v, SemanticVersion(2, 5, 0))

    def test_parse_empty_raises(self):
        with self.assertRaises(ValueError):
            SemanticVersion.parse("")

    def test_parse_invalid_raises(self):
        with self.assertRaises(ValueError):
            SemanticVersion.parse("not.a.version.string")

    def test_from_int(self):
        v = SemanticVersion.from_int(5)
        self.assertEqual(v, SemanticVersion(1, 4, 0))

    def test_from_int_zero(self):
        v = SemanticVersion.from_int(0)
        self.assertEqual(v, SemanticVersion(1, 0, 0))


class SemanticVersionComparisonTest(SimpleTestCase):
    """Test SemanticVersion comparison operators."""

    def test_equal_versions(self):
        self.assertEqual(SemanticVersion(1, 0, 0), SemanticVersion(1, 0, 0))

    def test_less_than_major(self):
        self.assertLess(SemanticVersion(1, 0, 0), SemanticVersion(2, 0, 0))

    def test_less_than_minor(self):
        self.assertLess(SemanticVersion(1, 0, 0), SemanticVersion(1, 1, 0))

    def test_less_than_patch(self):
        self.assertLess(SemanticVersion(1, 0, 0), SemanticVersion(1, 0, 1))

    def test_greater_than(self):
        self.assertGreater(SemanticVersion(2, 0, 0), SemanticVersion(1, 9, 9))

    def test_less_equal(self):
        self.assertLessEqual(SemanticVersion(1, 0, 0), SemanticVersion(1, 0, 0))
        self.assertLessEqual(SemanticVersion(1, 0, 0), SemanticVersion(1, 0, 1))

    def test_greater_equal(self):
        self.assertGreaterEqual(SemanticVersion(1, 0, 1), SemanticVersion(1, 0, 0))
        self.assertGreaterEqual(SemanticVersion(1, 0, 0), SemanticVersion(1, 0, 0))

    def test_prerelease_lower_than_release(self):
        self.assertLess(
            SemanticVersion(1, 0, 0, prerelease="beta"),
            SemanticVersion(1, 0, 0),
        )

    def test_equality_ignores_build(self):
        v1 = SemanticVersion(1, 0, 0, build="100")
        v2 = SemanticVersion(1, 0, 0, build="200")
        self.assertEqual(v1, v2)

    def test_hash_consistent(self):
        v1 = SemanticVersion(1, 2, 3)
        v2 = SemanticVersion(1, 2, 3)
        self.assertEqual(hash(v1), hash(v2))

    def test_not_equal_to_non_version(self):
        self.assertNotEqual(SemanticVersion(1, 0, 0), "1.0.0")


class SemanticVersionBumpTest(SimpleTestCase):
    """Test SemanticVersion bump methods."""

    def test_bump_major(self):
        v = SemanticVersion(1, 2, 3).bump_major()
        self.assertEqual(v, SemanticVersion(2, 0, 0))

    def test_bump_minor(self):
        v = SemanticVersion(1, 2, 3).bump_minor()
        self.assertEqual(v, SemanticVersion(1, 3, 0))

    def test_bump_patch(self):
        v = SemanticVersion(1, 2, 3).bump_patch()
        self.assertEqual(v, SemanticVersion(1, 2, 4))


class SemanticVersionStringTest(SimpleTestCase):
    """Test SemanticVersion string representation."""

    def test_str_simple(self):
        self.assertEqual(str(SemanticVersion(1, 2, 3)), "1.2.3")

    def test_str_with_prerelease(self):
        self.assertEqual(str(SemanticVersion(2, 0, 0, prerelease="rc.1")), "2.0.0-rc.1")

    def test_str_with_build(self):
        self.assertEqual(str(SemanticVersion(1, 0, 0, build="456")), "1.0.0+456")

    def test_str_full(self):
        self.assertEqual(
            str(SemanticVersion(1, 0, 0, prerelease="beta", build="42")),
            "1.0.0-beta+42",
        )


class SemanticVersionCompatibilityTest(SimpleTestCase):
    """Test SemanticVersion compatibility checks (SCHEMA-001 §6.2)."""

    def test_same_major_compatible(self):
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(1, 5, 3)
        self.assertTrue(v1.is_compatible_with(v2))

    def test_different_major_incompatible(self):
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(2, 0, 0)
        self.assertFalse(v1.is_compatible_with(v2))

    def test_get_compatibility_breaking(self):
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(2, 0, 0)
        self.assertEqual(v1.get_compatibility(v2), VersionCompatibility.BREAKING)

    def test_get_compatibility_forward(self):
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(1, 1, 0)
        self.assertEqual(
            v1.get_compatibility(v2), VersionCompatibility.FORWARD_COMPATIBLE
        )

    def test_get_compatibility_transparent(self):
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(1, 0, 1)
        self.assertEqual(v1.get_compatibility(v2), VersionCompatibility.TRANSPARENT)


# =============================================================================
# VERSIONING ENUM TESTS (SimpleTestCase)
# =============================================================================


class VersionChangeTypeEnumTest(SimpleTestCase):
    """Test VersionChangeType enum."""

    def test_all_types_defined(self):
        expected = {"major", "minor", "patch", "none"}
        actual = {t.value for t in VersionChangeType}
        self.assertEqual(actual, expected)


class VersionLifecycleEnumTest(SimpleTestCase):
    """Test VersionLifecycle enum (SCHEMA-001 §6.3)."""

    def test_all_states_defined(self):
        expected = {"draft", "active", "deprecated", "end_of_life", "retired"}
        actual = {s.value for s in VersionLifecycle}
        self.assertEqual(actual, expected)


class VersionCompatibilityEnumTest(SimpleTestCase):
    """Test VersionCompatibility enum (SCHEMA-001 §6.2)."""

    def test_all_levels_defined(self):
        expected = {"forward_compatible", "breaking", "transparent"}
        actual = {c.value for c in VersionCompatibility}
        self.assertEqual(actual, expected)


# =============================================================================
# BREAKING CHANGE DETECTION TESTS (SimpleTestCase)
# =============================================================================


class DetectBreakingChangesTest(SimpleTestCase):
    """Test detect_breaking_changes function (SCHEMA-001 §6.1)."""

    def test_no_changes_returns_patch(self):
        schema = {"properties": {"name": {"type": "string"}}, "required": ["name"]}
        diff = detect_breaking_changes(schema, schema)
        self.assertEqual(diff.version_change, VersionChangeType.PATCH)
        self.assertFalse(diff.migration_required)

    def test_field_removal_is_major(self):
        old = {"properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
        new = {"properties": {"name": {"type": "string"}}}
        diff = detect_breaking_changes(old, new)
        self.assertEqual(diff.version_change, VersionChangeType.MAJOR)
        self.assertIn("age", diff.removed_fields)
        self.assertTrue(diff.migration_required)

    def test_optional_field_addition_is_minor(self):
        old = {"properties": {"name": {"type": "string"}}}
        new = {"properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
        diff = detect_breaking_changes(old, new)
        self.assertEqual(diff.version_change, VersionChangeType.MINOR)
        self.assertIn("age", diff.added_fields)
        self.assertFalse(diff.migration_required)

    def test_required_field_addition_is_major(self):
        old = {"properties": {"name": {"type": "string"}}, "required": ["name"]}
        new = {
            "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
            "required": ["name", "email"],
        }
        diff = detect_breaking_changes(old, new)
        self.assertEqual(diff.version_change, VersionChangeType.MAJOR)
        self.assertTrue(diff.migration_required)

    def test_type_change_is_major(self):
        old = {"properties": {"age": {"type": "string"}}}
        new = {"properties": {"age": {"type": "integer"}}}
        diff = detect_breaking_changes(old, new)
        self.assertEqual(diff.version_change, VersionChangeType.MAJOR)
        self.assertEqual(len(diff.modified_fields), 1)
        self.assertEqual(diff.modified_fields[0].change, "type_changed")

    def test_optional_to_required_is_major(self):
        old = {"properties": {"name": {"type": "string"}}, "required": []}
        new = {"properties": {"name": {"type": "string"}}, "required": ["name"]}
        diff = detect_breaking_changes(old, new)
        self.assertEqual(diff.version_change, VersionChangeType.MAJOR)

    def test_minimum_constraint_tightened_is_major(self):
        old = {"properties": {"age": {"type": "integer", "minimum": 0}}}
        new = {"properties": {"age": {"type": "integer", "minimum": 18}}}
        diff = detect_breaking_changes(old, new)
        self.assertEqual(diff.version_change, VersionChangeType.MAJOR)
        self.assertTrue(diff.migration_required)

    def test_maximum_constraint_tightened_is_major(self):
        old = {"properties": {"score": {"type": "integer", "maximum": 100}}}
        new = {"properties": {"score": {"type": "integer", "maximum": 50}}}
        diff = detect_breaking_changes(old, new)
        self.assertEqual(diff.version_change, VersionChangeType.MAJOR)

    def test_enum_value_removed_is_major(self):
        old = {"properties": {"status": {"type": "string", "enum": ["a", "b", "c"]}}}
        new = {"properties": {"status": {"type": "string", "enum": ["a", "b"]}}}
        diff = detect_breaking_changes(old, new)
        self.assertEqual(diff.version_change, VersionChangeType.MAJOR)

    def test_enum_value_added_is_minor(self):
        old = {"properties": {"status": {"type": "string", "enum": ["a", "b"]}}}
        new = {"properties": {"status": {"type": "string", "enum": ["a", "b", "c"]}}}
        diff = detect_breaking_changes(old, new)
        self.assertEqual(diff.version_change, VersionChangeType.MINOR)

    def test_pattern_change_is_major(self):
        old = {"properties": {"code": {"type": "string", "pattern": "^[A-Z]+$"}}}
        new = {"properties": {"code": {"type": "string", "pattern": "^[a-z]+$"}}}
        diff = detect_breaking_changes(old, new)
        self.assertEqual(diff.version_change, VersionChangeType.MAJOR)

    def test_compatibility_set_correctly_breaking(self):
        old = {"properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
        new = {"properties": {"name": {"type": "string"}}}
        diff = detect_breaking_changes(old, new)
        self.assertEqual(diff.compatibility, VersionCompatibility.BREAKING)

    def test_compatibility_set_correctly_forward(self):
        old = {"properties": {"name": {"type": "string"}}}
        new = {"properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
        diff = detect_breaking_changes(old, new)
        self.assertEqual(diff.compatibility, VersionCompatibility.FORWARD_COMPATIBLE)

    def test_migration_complexity_high_for_many_removals(self):
        old = {
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"},
                "c": {"type": "string"},
            }
        }
        new = {"properties": {}}
        diff = detect_breaking_changes(old, new)
        self.assertEqual(diff.migration_complexity, "HIGH")


class SchemaDiffToDictTest(SimpleTestCase):
    """Test SchemaDiff serialization."""

    def test_to_dict_contains_all_keys(self):
        diff = SchemaDiff(version_change=VersionChangeType.PATCH)
        result = diff.to_dict()
        expected_keys = {
            "version_change",
            "added_fields",
            "removed_fields",
            "modified_fields",
            "compatibility",
            "migration_required",
            "estimated_impact",
            "migration_complexity",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_to_dict_values_serialized(self):
        diff = SchemaDiff(
            version_change=VersionChangeType.MAJOR,
            added_fields=["new_field"],
            compatibility=VersionCompatibility.BREAKING,
        )
        result = diff.to_dict()
        self.assertEqual(result["version_change"], "major")
        self.assertEqual(result["compatibility"], "breaking")
        self.assertEqual(result["added_fields"], ["new_field"])


# =============================================================================
# DEPRECATION LIFECYCLE TESTS (SimpleTestCase)
# =============================================================================


class DeprecationInfoTest(SimpleTestCase):
    """Test DeprecationInfo lifecycle management (SCHEMA-001 §6.3)."""

    def test_minimum_sunset_days_constant(self):
        self.assertEqual(DeprecationInfo.MINIMUM_SUNSET_DAYS, 180)

    def test_grace_period_days_constant(self):
        self.assertEqual(DeprecationInfo.GRACE_PERIOD_DAYS, 90)

    def test_validate_sunset_date_valid(self):
        now = timezone.now()
        info = DeprecationInfo(
            deprecated_at=now,
            sunset_date=now + timedelta(days=200),
        )
        self.assertTrue(info.validate_sunset_date())

    def test_validate_sunset_date_too_early(self):
        now = timezone.now()
        info = DeprecationInfo(
            deprecated_at=now,
            sunset_date=now + timedelta(days=30),
        )
        self.assertFalse(info.validate_sunset_date())

    def test_validate_sunset_date_no_dates_is_valid(self):
        info = DeprecationInfo()
        self.assertTrue(info.validate_sunset_date())

    def test_is_past_sunset_true(self):
        info = DeprecationInfo(sunset_date=timezone.now() - timedelta(days=1))
        self.assertTrue(info.is_past_sunset())

    def test_is_past_sunset_false(self):
        info = DeprecationInfo(sunset_date=timezone.now() + timedelta(days=30))
        self.assertFalse(info.is_past_sunset())

    def test_is_past_sunset_no_date(self):
        info = DeprecationInfo()
        self.assertFalse(info.is_past_sunset())

    def test_is_past_grace_period_true(self):
        info = DeprecationInfo(sunset_date=timezone.now() - timedelta(days=100))
        self.assertTrue(info.is_past_grace_period())

    def test_is_past_grace_period_false(self):
        info = DeprecationInfo(sunset_date=timezone.now() - timedelta(days=10))
        self.assertFalse(info.is_past_grace_period())

    def test_lifecycle_state_active(self):
        info = DeprecationInfo()
        self.assertEqual(info.get_lifecycle_state(), VersionLifecycle.ACTIVE)

    def test_lifecycle_state_deprecated(self):
        info = DeprecationInfo(
            deprecated_at=timezone.now(),
            sunset_date=timezone.now() + timedelta(days=180),
        )
        self.assertEqual(info.get_lifecycle_state(), VersionLifecycle.DEPRECATED)

    def test_lifecycle_state_end_of_life(self):
        info = DeprecationInfo(
            deprecated_at=timezone.now() - timedelta(days=200),
            sunset_date=timezone.now() - timedelta(days=10),
        )
        self.assertEqual(info.get_lifecycle_state(), VersionLifecycle.END_OF_LIFE)

    def test_lifecycle_state_retired(self):
        info = DeprecationInfo(
            deprecated_at=timezone.now() - timedelta(days=400),
            sunset_date=timezone.now() - timedelta(days=100),
        )
        self.assertEqual(info.get_lifecycle_state(), VersionLifecycle.RETIRED)

    def test_to_dict_contains_all_keys(self):
        info = DeprecationInfo(
            deprecated_at=timezone.now(),
            deprecated_by="admin",
            reason="Replaced",
        )
        result = info.to_dict()
        expected_keys = {
            "deprecated_at",
            "deprecated_by",
            "sunset_date",
            "migration_path",
            "successor_version",
            "reason",
            "lifecycle_state",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_to_dict_none_dates(self):
        info = DeprecationInfo()
        result = info.to_dict()
        self.assertIsNone(result["deprecated_at"])
        self.assertIsNone(result["sunset_date"])


# =============================================================================
# CHECKSUM TESTS (SimpleTestCase)
# =============================================================================


class ChecksumTest(SimpleTestCase):
    """Test version checksum functions (SCHEMA-001 §6.4)."""

    def test_compute_produces_64_char_hex(self):
        result = compute_version_checksum({"key": "value"})
        self.assertEqual(len(result), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in result))

    def test_compute_deterministic(self):
        definition = {"name": "test", "version": 1}
        self.assertEqual(
            compute_version_checksum(definition),
            compute_version_checksum(definition),
        )

    def test_compute_different_for_different_input(self):
        self.assertNotEqual(
            compute_version_checksum({"a": 1}),
            compute_version_checksum({"a": 2}),
        )

    def test_compute_key_order_independent(self):
        self.assertEqual(
            compute_version_checksum({"b": 2, "a": 1}),
            compute_version_checksum({"a": 1, "b": 2}),
        )

    def test_verify_checksum_valid(self):
        definition = {"test": "data"}
        checksum = compute_version_checksum(definition)
        self.assertTrue(verify_version_checksum(definition, checksum))

    def test_verify_checksum_invalid(self):
        definition = {"test": "data"}
        self.assertFalse(verify_version_checksum(definition, "invalid_checksum"))

    def test_verify_checksum_tampered(self):
        definition = {"test": "data"}
        checksum = compute_version_checksum(definition)
        tampered = {"test": "tampered"}
        self.assertFalse(verify_version_checksum(tampered, checksum))


# =============================================================================
# VERSION MANAGER TESTS (SimpleTestCase — mock-based)
# =============================================================================


class VersionManagerTest(SimpleTestCase):
    """Test VersionManager orchestration."""

    def test_compute_next_version_major(self):
        vm = VersionManager(MagicMock(), "tenant-1")
        current = SemanticVersion(1, 2, 3)
        result = vm.compute_next_version(current, VersionChangeType.MAJOR)
        self.assertEqual(result, SemanticVersion(2, 0, 0))

    def test_compute_next_version_minor(self):
        vm = VersionManager(MagicMock(), "tenant-1")
        current = SemanticVersion(1, 2, 3)
        result = vm.compute_next_version(current, VersionChangeType.MINOR)
        self.assertEqual(result, SemanticVersion(1, 3, 0))

    def test_compute_next_version_patch(self):
        vm = VersionManager(MagicMock(), "tenant-1")
        current = SemanticVersion(1, 2, 3)
        result = vm.compute_next_version(current, VersionChangeType.PATCH)
        self.assertEqual(result, SemanticVersion(1, 2, 4))

    def test_validate_rollback_valid(self):
        vm = VersionManager(MagicMock(), "tenant-1")
        is_valid, errors = vm.validate_rollback("2.0.0", "1.0.0")
        self.assertTrue(is_valid)
        self.assertEqual(errors, [])

    def test_validate_rollback_newer_target_invalid(self):
        vm = VersionManager(MagicMock(), "tenant-1")
        is_valid, errors = vm.validate_rollback("1.0.0", "2.0.0")
        self.assertFalse(is_valid)
        self.assertTrue(len(errors) > 0)

    def test_validate_rollback_same_version_invalid(self):
        vm = VersionManager(MagicMock(), "tenant-1")
        is_valid, errors = vm.validate_rollback("1.0.0", "1.0.0")
        self.assertFalse(is_valid)

    def test_validate_rollback_invalid_format(self):
        vm = VersionManager(MagicMock(), "tenant-1")
        is_valid, errors = vm.validate_rollback("invalid", "1.0.0")
        self.assertFalse(is_valid)
        self.assertIn("Invalid version format", errors[0])

    def test_get_current_version_not_found(self):
        class FakeModel:
            class DoesNotExist(Exception):
                pass

            class Objects:
                @staticmethod
                def get(**kwargs):
                    raise FakeModel.DoesNotExist()

            objects = Objects()

        vm = VersionManager(FakeModel, "tenant-1")
        result = vm.get_current_version("entity-1")
        self.assertIsNone(result)


# =============================================================================
# MIXIN EXPORTS TEST (SimpleTestCase)
# =============================================================================


class MixinExportsTest(SimpleTestCase):
    """Test that all mixins are properly exported."""

    def test_all_mixins_in_module_all(self):
        from syn.core import mixins

        expected = {
            "CorrelationMixin",
            "TenantMixin",
            "AuditMixin",
            "SoftDeleteMixin",
            "SoftDeleteManager",
            "EventEmitterMixin",
            "LifecycleMixin",
            "MetadataMixin",
            "VersioningMixin",
        }
        self.assertEqual(set(mixins.__all__), expected)


class VersioningExportsTest(SimpleTestCase):
    """Test that all versioning symbols are properly exported."""

    def test_all_versioning_in_module_all(self):
        from syn.core import versioning

        expected = {
            "SemanticVersion",
            "VersionChangeType",
            "VersionLifecycle",
            "VersionCompatibility",
            "SchemaChange",
            "SchemaDiff",
            "detect_breaking_changes",
            "DeprecationInfo",
            "compute_version_checksum",
            "verify_version_checksum",
            "VersionManager",
            "SynaraVersionedMixin",
        }
        self.assertEqual(set(versioning.__all__), expected)
