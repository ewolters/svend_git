"""
Behavioral tests for Synara versioning and configuration modules.

Tests:
- syn/core/versioning.py: detect_breaking_changes(), DeprecationInfo,
  compute_version_checksum, verify_version_checksum, SemanticVersion
- syn/core/config.py: SynaraSettings.validate_production_settings,
  check_anti_patterns

Standard: TST-001
"""

from datetime import timedelta
from unittest.mock import patch

from django.test import SimpleTestCase
from django.utils import timezone

from syn.core.versioning import (
    DeprecationInfo,
    SemanticVersion,
    VersionChangeType,
    VersionCompatibility,
    VersionLifecycle,
    compute_version_checksum,
    detect_breaking_changes,
    verify_version_checksum,
)

# =============================================================================
# DETECT BREAKING CHANGES (SCHEMA-001 §6.1)
# =============================================================================


class DetectBreakingChangesTest(SimpleTestCase):
    """Behavioral tests for detect_breaking_changes()."""

    def _schema(self, properties=None, required=None):
        """Helper to build a JSON-schema-like dict."""
        s = {}
        if properties is not None:
            s["properties"] = properties
        if required is not None:
            s["required"] = required
        return s

    # -- field removal ---------------------------------------------------------

    def test_field_removal_is_major_breaking(self):
        old = self._schema(
            properties={"name": {"type": "string"}, "age": {"type": "integer"}},
        )
        new = self._schema(
            properties={"name": {"type": "string"}},
        )
        diff = detect_breaking_changes(old, new)

        self.assertEqual(diff.version_change, VersionChangeType.MAJOR)
        self.assertIn("age", diff.removed_fields)
        self.assertTrue(diff.migration_required)
        self.assertEqual(diff.compatibility, VersionCompatibility.BREAKING)

    # -- optional field addition -----------------------------------------------

    def test_optional_field_addition_is_minor(self):
        old = self._schema(properties={"name": {"type": "string"}})
        new = self._schema(
            properties={"name": {"type": "string"}, "bio": {"type": "string"}},
        )
        diff = detect_breaking_changes(old, new)

        self.assertEqual(diff.version_change, VersionChangeType.MINOR)
        self.assertIn("bio", diff.added_fields)
        self.assertFalse(diff.migration_required)

    # -- required field addition -----------------------------------------------

    def test_required_field_addition_is_major(self):
        old = self._schema(properties={"name": {"type": "string"}})
        new = self._schema(
            properties={"name": {"type": "string"}, "email": {"type": "string"}},
            required=["email"],
        )
        diff = detect_breaking_changes(old, new)

        self.assertEqual(diff.version_change, VersionChangeType.MAJOR)
        self.assertIn("email", diff.added_fields)
        self.assertTrue(diff.migration_required)

    # -- type change -----------------------------------------------------------

    def test_type_change_is_major(self):
        old = self._schema(properties={"score": {"type": "integer"}})
        new = self._schema(properties={"score": {"type": "string"}})
        diff = detect_breaking_changes(old, new)

        self.assertEqual(diff.version_change, VersionChangeType.MAJOR)
        self.assertTrue(diff.migration_required)
        type_changes = [m for m in diff.modified_fields if m.change == "type_changed"]
        self.assertEqual(len(type_changes), 1)
        self.assertEqual(type_changes[0].old_value, "integer")
        self.assertEqual(type_changes[0].new_value, "string")

    # -- optional to required --------------------------------------------------

    def test_optional_to_required_is_major(self):
        old = self._schema(
            properties={"name": {"type": "string"}},
            required=[],
        )
        new = self._schema(
            properties={"name": {"type": "string"}},
            required=["name"],
        )
        diff = detect_breaking_changes(old, new)

        self.assertEqual(diff.version_change, VersionChangeType.MAJOR)
        req_changes = [
            m for m in diff.modified_fields if m.change == "required_changed"
        ]
        self.assertEqual(len(req_changes), 1)
        self.assertEqual(req_changes[0].old_value, False)
        self.assertEqual(req_changes[0].new_value, True)

    # -- required to optional --------------------------------------------------

    def test_required_to_optional_is_not_breaking(self):
        """Relaxing a requirement is not a breaking change."""
        old = self._schema(
            properties={"name": {"type": "string"}},
            required=["name"],
        )
        new = self._schema(
            properties={"name": {"type": "string"}},
            required=[],
        )
        diff = detect_breaking_changes(old, new)

        self.assertNotEqual(diff.version_change, VersionChangeType.MAJOR)

    # -- constraint: min tightened ---------------------------------------------

    def test_constraint_tightened_min_is_major(self):
        old = self._schema(properties={"age": {"type": "integer", "minimum": 0}})
        new = self._schema(properties={"age": {"type": "integer", "minimum": 18}})
        diff = detect_breaking_changes(old, new)

        self.assertEqual(diff.version_change, VersionChangeType.MAJOR)
        constraint_changes = [
            m for m in diff.modified_fields if m.change == "constraint_changed"
        ]
        self.assertTrue(any("minimum" in str(c.new_value) for c in constraint_changes))

    # -- constraint: max tightened ---------------------------------------------

    def test_constraint_tightened_max_is_major(self):
        old = self._schema(properties={"score": {"type": "integer", "maximum": 100}})
        new = self._schema(properties={"score": {"type": "integer", "maximum": 50}})
        diff = detect_breaking_changes(old, new)

        self.assertEqual(diff.version_change, VersionChangeType.MAJOR)
        constraint_changes = [
            m for m in diff.modified_fields if m.change == "constraint_changed"
        ]
        self.assertTrue(any("maximum" in str(c.new_value) for c in constraint_changes))

    # -- enum removal ----------------------------------------------------------

    def test_enum_value_removal_is_major(self):
        old = self._schema(
            properties={
                "status": {"type": "string", "enum": ["open", "closed", "pending"]}
            },
        )
        new = self._schema(
            properties={"status": {"type": "string", "enum": ["open", "closed"]}},
        )
        diff = detect_breaking_changes(old, new)

        self.assertEqual(diff.version_change, VersionChangeType.MAJOR)
        enum_changes = [
            m for m in diff.modified_fields if m.change == "enum_values_removed"
        ]
        self.assertEqual(len(enum_changes), 1)

    # -- enum addition ---------------------------------------------------------

    def test_enum_value_addition_is_minor(self):
        old = self._schema(
            properties={"status": {"type": "string", "enum": ["open", "closed"]}},
        )
        new = self._schema(
            properties={
                "status": {"type": "string", "enum": ["open", "closed", "pending"]}
            },
        )
        diff = detect_breaking_changes(old, new)

        self.assertEqual(diff.version_change, VersionChangeType.MINOR)

    # -- no changes ------------------------------------------------------------

    def test_no_changes_is_patch(self):
        schema = self._schema(
            properties={"name": {"type": "string"}},
            required=["name"],
        )
        diff = detect_breaking_changes(schema, schema)

        self.assertEqual(diff.version_change, VersionChangeType.PATCH)
        self.assertFalse(diff.migration_required)
        self.assertEqual(diff.compatibility, VersionCompatibility.TRANSPARENT)

    # -- multiple breaking changes -> highest ----------------------------------

    def test_multiple_breaking_changes_returns_highest(self):
        """When both removal and type change occur, result is still MAJOR."""
        old = self._schema(
            properties={
                "name": {"type": "string"},
                "score": {"type": "integer"},
            },
        )
        new = self._schema(
            properties={
                "score": {"type": "string"},  # type changed
                # 'name' removed
            },
        )
        diff = detect_breaking_changes(old, new)

        self.assertEqual(diff.version_change, VersionChangeType.MAJOR)
        self.assertIn("name", diff.removed_fields)
        self.assertTrue(diff.migration_required)
        self.assertEqual(diff.compatibility, VersionCompatibility.BREAKING)


# =============================================================================
# DEPRECATION INFO (SCHEMA-001 §6.3)
# =============================================================================


class DeprecationInfoTest(SimpleTestCase):
    """Behavioral tests for DeprecationInfo lifecycle logic."""

    def test_lifecycle_active_before_deprecation(self):
        """No deprecated_at means ACTIVE."""
        info = DeprecationInfo()
        self.assertEqual(info.get_lifecycle_state(), VersionLifecycle.ACTIVE)

    def test_lifecycle_deprecated_between_sunset_and_now(self):
        """After deprecation but before sunset -> DEPRECATED."""
        now = timezone.now()
        info = DeprecationInfo(
            deprecated_at=now - timedelta(days=10),
            sunset_date=now + timedelta(days=200),
        )
        self.assertEqual(info.get_lifecycle_state(), VersionLifecycle.DEPRECATED)

    def test_lifecycle_end_of_life_past_sunset(self):
        """Past sunset but within grace period -> END_OF_LIFE."""
        now = timezone.now()
        info = DeprecationInfo(
            deprecated_at=now - timedelta(days=300),
            sunset_date=now - timedelta(days=10),
        )
        self.assertEqual(info.get_lifecycle_state(), VersionLifecycle.END_OF_LIFE)

    def test_lifecycle_retired_past_grace_period(self):
        """Past sunset + grace period -> RETIRED."""
        now = timezone.now()
        info = DeprecationInfo(
            deprecated_at=now - timedelta(days=400),
            sunset_date=now - timedelta(days=100),  # 100 > GRACE_PERIOD_DAYS (90)
        )
        self.assertEqual(info.get_lifecycle_state(), VersionLifecycle.RETIRED)

    def test_validate_sunset_date_rejects_under_180_days(self):
        """Sunset date must be >= 180 days after deprecation."""
        now = timezone.now()
        info = DeprecationInfo(
            deprecated_at=now,
            sunset_date=now + timedelta(days=90),  # only 90, need 180
        )
        self.assertFalse(info.validate_sunset_date())

        # Exactly 180 days should pass
        info_ok = DeprecationInfo(
            deprecated_at=now,
            sunset_date=now + timedelta(days=180),
        )
        self.assertTrue(info_ok.validate_sunset_date())

    def test_to_dict_serialization(self):
        """to_dict produces expected keys and serialized values."""
        now = timezone.now()
        info = DeprecationInfo(
            deprecated_at=now,
            deprecated_by="admin",
            sunset_date=now + timedelta(days=200),
            migration_path="/api/v2/",
            successor_version="2.0.0",
            reason="Replaced by v2",
        )
        d = info.to_dict()

        self.assertEqual(d["deprecated_at"], now.isoformat())
        self.assertEqual(d["deprecated_by"], "admin")
        self.assertEqual(d["sunset_date"], (now + timedelta(days=200)).isoformat())
        self.assertEqual(d["migration_path"], "/api/v2/")
        self.assertEqual(d["successor_version"], "2.0.0")
        self.assertEqual(d["reason"], "Replaced by v2")
        self.assertEqual(d["lifecycle_state"], VersionLifecycle.DEPRECATED.value)


# =============================================================================
# VERSION COMPATIBILITY (SCHEMA-001 §6.2)
# =============================================================================


class VersionCompatibilityTest(SimpleTestCase):
    """Behavioral tests for SemanticVersion compatibility methods."""

    def test_same_major_is_compatible(self):
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(1, 5, 3)
        self.assertTrue(v1.is_compatible_with(v2))
        self.assertTrue(v2.is_compatible_with(v1))

    def test_different_major_is_incompatible(self):
        v1 = SemanticVersion(1, 9, 9)
        v2 = SemanticVersion(2, 0, 0)
        self.assertFalse(v1.is_compatible_with(v2))
        self.assertFalse(v2.is_compatible_with(v1))

    def test_get_compatibility_major_bump(self):
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(2, 0, 0)
        self.assertEqual(v1.get_compatibility(v2), VersionCompatibility.BREAKING)

    def test_get_compatibility_minor_bump(self):
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(1, 1, 0)
        self.assertEqual(
            v1.get_compatibility(v2), VersionCompatibility.FORWARD_COMPATIBLE
        )

    def test_get_compatibility_patch_bump(self):
        v1 = SemanticVersion(1, 2, 0)
        v2 = SemanticVersion(1, 2, 5)
        self.assertEqual(v1.get_compatibility(v2), VersionCompatibility.TRANSPARENT)


# =============================================================================
# CHECKSUM (SCHEMA-001 §6.4)
# =============================================================================


class ChecksumTest(SimpleTestCase):
    """Behavioral tests for checksum computation and verification."""

    def test_compute_checksum_deterministic(self):
        definition = {"type": "object", "properties": {"a": 1, "b": 2}}
        c1 = compute_version_checksum(definition)
        c2 = compute_version_checksum(definition)
        self.assertEqual(c1, c2)
        self.assertEqual(len(c1), 64)  # SHA-256 hex digest length

    def test_verify_checksum_match(self):
        definition = {"name": "test", "version": "1.0.0"}
        checksum = compute_version_checksum(definition)
        self.assertTrue(verify_version_checksum(definition, checksum))

    def test_verify_checksum_mismatch(self):
        definition = {"name": "test"}
        self.assertFalse(verify_version_checksum(definition, "0" * 64))

    def test_different_inputs_different_checksums(self):
        d1 = {"a": 1}
        d2 = {"a": 2}
        self.assertNotEqual(
            compute_version_checksum(d1),
            compute_version_checksum(d2),
        )


# =============================================================================
# CONFIG VALIDATION (CONFIG-001 §10)
# =============================================================================


class ConfigValidationTest(SimpleTestCase):
    """Behavioral tests for SynaraSettings validation and anti-pattern checks."""

    def test_production_rejects_debug_true(self):
        """validate_production_settings raises when DEBUG=True + PRODUCTION."""
        from pydantic import ValidationError as PydanticValidationError

        from syn.core.config import SynaraSettings

        with self.assertRaises(PydanticValidationError) as ctx:
            SynaraSettings(
                DJANGO_SECRET_KEY="test-secret-key-not-real",
                DJANGO_DEBUG=True,
                DEPLOYMENT_PROFILE="production",
                DJANGO_ALLOWED_HOSTS="svend.ai",
            )
        self.assertIn("DEBUG must be False in production", str(ctx.exception))

    def test_production_rejects_wildcard_allowed_hosts(self):
        """validate_production_settings raises when ALLOWED_HOSTS=* + PRODUCTION."""
        from pydantic import ValidationError as PydanticValidationError

        from syn.core.config import SynaraSettings

        with self.assertRaises(PydanticValidationError) as ctx:
            SynaraSettings(
                DJANGO_SECRET_KEY="test-secret-key-not-real",
                DJANGO_DEBUG=False,
                DEPLOYMENT_PROFILE="production",
                DJANGO_ALLOWED_HOSTS="*",
            )
        self.assertIn("ALLOWED_HOSTS cannot be * in production", str(ctx.exception))

    @patch("syn.core.config.get_settings")
    def test_check_anti_patterns_flags_debug_in_production(self, mock_get_settings):
        """check_anti_patterns returns warning for DEBUG=True in production."""
        from syn.core.config import SecuritySettings, check_anti_patterns

        mock_obj = mock_get_settings.return_value
        mock_obj.is_production = True
        mock_obj.django_debug = True
        mock_obj.allowed_hosts_list = ["svend.ai"]
        mock_obj.get_security.return_value = SecuritySettings(
            CSRF_COOKIE_SECURE=False,
            SESSION_COOKIE_SECURE=False,
            SECURE_SSL_REDIRECT=False,
        )

        warnings = check_anti_patterns()

        self.assertTrue(
            any("DEBUG=True in production" in w for w in warnings),
            f"Expected DEBUG warning in {warnings}",
        )

    @patch("syn.core.config.get_settings")
    def test_check_anti_patterns_clean_for_local(self, mock_get_settings):
        """check_anti_patterns returns no warnings for local profile."""
        mock_get_settings.return_value.is_production = False

        from syn.core.config import check_anti_patterns

        warnings = check_anti_patterns()

        self.assertEqual(warnings, [])
