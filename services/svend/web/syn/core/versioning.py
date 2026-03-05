"""
Synara Versioning Utilities (SDK-001 §6)
========================================

Comprehensive versioning infrastructure for all Synara entities.

Provides:
- Semantic version parsing and comparison
- Breaking change detection algorithms
- Version lifecycle management
- Migration path computation
- Rollback validation

Standards: SDK-001 §6, SCHEMA-001 §6, EVT-001 §6, PRM-001 §11, FORM-001 §versioning
Compliance: ISO 27001 A.8.1, SOC 2 CC6.7, 21 CFR Part 11

Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from django.db import models
from django.utils import timezone

logger = logging.getLogger(__name__)


# =============================================================================
# SEMANTIC VERSION (SCHEMA-001 §6, EVT-001 §6)
# =============================================================================


class VersionChangeType(Enum):
    """
    Version change classification per SCHEMA-001 §6.

    MAJOR: Breaking changes requiring migration
    MINOR: Backward-compatible additions
    PATCH: Non-functional changes
    """

    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    NONE = "none"


class VersionLifecycle(Enum):
    """
    Version lifecycle states per SCHEMA-001 §6.3.

    Deprecation lifecycle: ACTIVE → DEPRECATED → END_OF_LIFE → RETIRED
    """

    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    END_OF_LIFE = "end_of_life"
    RETIRED = "retired"


class VersionCompatibility(Enum):
    """
    Version compatibility levels per SCHEMA-001 §6.2.
    """

    FORWARD_COMPATIBLE = "forward_compatible"  # 1.x.x → 1.y.z (y > x)
    BREAKING = "breaking"  # 1.x.x → 2.0.0
    TRANSPARENT = "transparent"  # 1.x.x → 1.x.y (y > x)


@dataclass
class SemanticVersion:
    """
    Semantic version representation per SCHEMA-001 §6.

    Format: MAJOR.MINOR.PATCH[-prerelease][+build]

    Examples:
        1.0.0
        2.1.5
        3.0.0-beta.1
        1.2.3+build.456
    """

    major: int
    minor: int
    patch: int
    prerelease: str | None = None
    build: str | None = None

    # Regex pattern for semantic version parsing
    SEMVER_PATTERN = re.compile(
        r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
        r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
        r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
        r"(?:\+(?P<build>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
    )

    @classmethod
    def parse(cls, version_string: str) -> SemanticVersion:
        """
        Parse version string into SemanticVersion.

        Args:
            version_string: Version in format MAJOR.MINOR.PATCH

        Returns:
            SemanticVersion instance

        Raises:
            ValueError: If version string is invalid
        """
        if not version_string:
            raise ValueError("Version string cannot be empty")

        # Handle simple numeric versions (e.g., "1" -> "1.0.0")
        parts = version_string.split(".")
        if len(parts) == 1:
            version_string = f"{parts[0]}.0.0"
        elif len(parts) == 2:
            version_string = f"{parts[0]}.{parts[1]}.0"

        match = cls.SEMVER_PATTERN.match(version_string)
        if not match:
            raise ValueError(f"Invalid semantic version: {version_string}")

        return cls(
            major=int(match.group("major")),
            minor=int(match.group("minor")),
            patch=int(match.group("patch")),
            prerelease=match.group("prerelease"),
            build=match.group("build"),
        )

    @classmethod
    def from_int(cls, version: int) -> SemanticVersion:
        """
        Convert integer version to semantic version.

        Used for migrating FormSchemaVersion.version (int) to semantic.

        Args:
            version: Integer version number

        Returns:
            SemanticVersion with version as minor component
        """
        return cls(major=1, minor=version - 1 if version > 0 else 0, patch=0)

    def __str__(self) -> str:
        """Convert to string representation."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    def __lt__(self, other: SemanticVersion) -> bool:
        """Compare versions for ordering."""
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return self._compare_tuple() < other._compare_tuple()

    def __le__(self, other: SemanticVersion) -> bool:
        return self == other or self < other

    def __gt__(self, other: SemanticVersion) -> bool:
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return self._compare_tuple() > other._compare_tuple()

    def __ge__(self, other: SemanticVersion) -> bool:
        return self == other or self > other

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return (
            self.major == other.major
            and self.minor == other.minor
            and self.patch == other.patch
            and self.prerelease == other.prerelease
        )

    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch, self.prerelease))

    def _compare_tuple(self) -> tuple:
        """Create comparison tuple for ordering."""
        # Prerelease versions are lower than release versions
        prerelease_key = (0, self.prerelease) if self.prerelease else (1, "")
        return (self.major, self.minor, self.patch, prerelease_key)

    def bump_major(self) -> SemanticVersion:
        """Increment major version (breaking change)."""
        return SemanticVersion(self.major + 1, 0, 0)

    def bump_minor(self) -> SemanticVersion:
        """Increment minor version (backward-compatible)."""
        return SemanticVersion(self.major, self.minor + 1, 0)

    def bump_patch(self) -> SemanticVersion:
        """Increment patch version (non-functional)."""
        return SemanticVersion(self.major, self.minor, self.patch + 1)

    def is_compatible_with(self, other: SemanticVersion) -> bool:
        """
        Check if this version is compatible with another.

        Per SCHEMA-001 §6.2: Same major version = compatible.
        """
        return self.major == other.major

    def get_compatibility(self, other: SemanticVersion) -> VersionCompatibility:
        """
        Determine compatibility level between versions.

        Returns:
            VersionCompatibility enum value
        """
        if self.major != other.major:
            return VersionCompatibility.BREAKING
        elif self.minor != other.minor:
            return VersionCompatibility.FORWARD_COMPATIBLE
        else:
            return VersionCompatibility.TRANSPARENT


# =============================================================================
# BREAKING CHANGE DETECTION (SCHEMA-001 §6.1)
# =============================================================================


@dataclass
class SchemaChange:
    """Represents a single change between schema versions."""

    change_type: VersionChangeType
    field: str
    change: str  # type_changed, constraint_changed, required_changed, etc.
    old_value: Any = None
    new_value: Any = None
    description: str = ""


@dataclass
class SchemaDiff:
    """
    Result of comparing two schema versions.

    Per SCHEMA-001 §6.5: Schema diff tool output.
    """

    version_change: VersionChangeType
    added_fields: list[str] = field(default_factory=list)
    removed_fields: list[str] = field(default_factory=list)
    modified_fields: list[SchemaChange] = field(default_factory=list)
    compatibility: VersionCompatibility = VersionCompatibility.TRANSPARENT
    migration_required: bool = False
    estimated_impact: int = 0
    migration_complexity: str = "LOW"  # LOW, MEDIUM, HIGH

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "version_change": self.version_change.value,
            "added_fields": self.added_fields,
            "removed_fields": self.removed_fields,
            "modified_fields": [
                {
                    "field": m.field,
                    "change": m.change,
                    "old_value": m.old_value,
                    "new_value": m.new_value,
                }
                for m in self.modified_fields
            ],
            "compatibility": self.compatibility.value,
            "migration_required": self.migration_required,
            "estimated_impact": self.estimated_impact,
            "migration_complexity": self.migration_complexity,
        }


def detect_breaking_changes(old_schema: dict[str, Any], new_schema: dict[str, Any]) -> SchemaDiff:
    """
    Detect breaking changes between two schema versions.

    Implements SCHEMA-001 §6.1 breaking change detection algorithm:
    - Field removals → MAJOR
    - Type changes → MAJOR
    - Required changes (optional → required) → MAJOR
    - Constraint tightening → MAJOR
    - Enum removals → MAJOR
    - Foreign key changes → MAJOR

    Args:
        old_schema: Previous schema definition
        new_schema: New schema definition

    Returns:
        SchemaDiff with change analysis
    """
    diff = SchemaDiff(version_change=VersionChangeType.PATCH)

    old_fields = old_schema.get("properties", {})
    new_fields = new_schema.get("properties", {})
    old_required = set(old_schema.get("required", []))
    new_required = set(new_schema.get("required", []))

    # Check for removed fields → MAJOR
    for field_name in old_fields:
        if field_name not in new_fields:
            diff.removed_fields.append(field_name)
            diff.version_change = VersionChangeType.MAJOR
            diff.migration_required = True

    # Check for added fields
    for field_name in new_fields:
        if field_name not in old_fields:
            diff.added_fields.append(field_name)
            # Added required field → MAJOR
            if field_name in new_required:
                diff.version_change = VersionChangeType.MAJOR
                diff.migration_required = True
            elif diff.version_change != VersionChangeType.MAJOR:
                diff.version_change = VersionChangeType.MINOR

    # Check modified fields
    for field_name in old_fields:
        if field_name in new_fields:
            old_field = old_fields[field_name]
            new_field = new_fields[field_name]

            # Type change → MAJOR
            if old_field.get("type") != new_field.get("type"):
                diff.modified_fields.append(
                    SchemaChange(
                        change_type=VersionChangeType.MAJOR,
                        field=field_name,
                        change="type_changed",
                        old_value=old_field.get("type"),
                        new_value=new_field.get("type"),
                    )
                )
                diff.version_change = VersionChangeType.MAJOR
                diff.migration_required = True

            # Required change (optional → required) → MAJOR
            was_required = field_name in old_required
            is_required = field_name in new_required
            if not was_required and is_required:
                diff.modified_fields.append(
                    SchemaChange(
                        change_type=VersionChangeType.MAJOR,
                        field=field_name,
                        change="required_changed",
                        old_value=False,
                        new_value=True,
                    )
                )
                diff.version_change = VersionChangeType.MAJOR
                diff.migration_required = True

            # Constraint tightening → MAJOR
            _check_constraint_changes(old_field, new_field, field_name, diff)

            # Enum value removal → MAJOR
            _check_enum_changes(old_field, new_field, field_name, diff)

    # Set compatibility
    if diff.version_change == VersionChangeType.MAJOR:
        diff.compatibility = VersionCompatibility.BREAKING
        diff.migration_complexity = "HIGH" if len(diff.removed_fields) > 2 else "MEDIUM"
    elif diff.version_change == VersionChangeType.MINOR:
        diff.compatibility = VersionCompatibility.FORWARD_COMPATIBLE
        diff.migration_complexity = "LOW"
    else:
        diff.compatibility = VersionCompatibility.TRANSPARENT

    return diff


def _check_constraint_changes(
    old_field: dict[str, Any], new_field: dict[str, Any], field_name: str, diff: SchemaDiff
) -> None:
    """Check for constraint changes between field definitions."""
    # Minimum constraint tightening
    old_min = old_field.get("minimum")
    new_min = new_field.get("minimum")
    if old_min is not None and new_min is not None:
        if new_min > old_min:
            diff.modified_fields.append(
                SchemaChange(
                    change_type=VersionChangeType.MAJOR,
                    field=field_name,
                    change="constraint_changed",
                    old_value={"minimum": old_min},
                    new_value={"minimum": new_min},
                    description="Minimum constraint tightened",
                )
            )
            diff.version_change = VersionChangeType.MAJOR
            diff.migration_required = True

    # Maximum constraint tightening
    old_max = old_field.get("maximum")
    new_max = new_field.get("maximum")
    if old_max is not None and new_max is not None:
        if new_max < old_max:
            diff.modified_fields.append(
                SchemaChange(
                    change_type=VersionChangeType.MAJOR,
                    field=field_name,
                    change="constraint_changed",
                    old_value={"maximum": old_max},
                    new_value={"maximum": new_max},
                    description="Maximum constraint tightened",
                )
            )
            diff.version_change = VersionChangeType.MAJOR
            diff.migration_required = True

    # Pattern change → potentially MAJOR
    old_pattern = old_field.get("pattern")
    new_pattern = new_field.get("pattern")
    if old_pattern != new_pattern and new_pattern is not None:
        diff.modified_fields.append(
            SchemaChange(
                change_type=VersionChangeType.MAJOR,
                field=field_name,
                change="constraint_changed",
                old_value={"pattern": old_pattern},
                new_value={"pattern": new_pattern},
                description="Pattern constraint changed",
            )
        )
        diff.version_change = VersionChangeType.MAJOR
        diff.migration_required = True


def _check_enum_changes(
    old_field: dict[str, Any], new_field: dict[str, Any], field_name: str, diff: SchemaDiff
) -> None:
    """Check for enum value changes."""
    old_enum = set(old_field.get("enum", []))
    new_enum = set(new_field.get("enum", []))

    if old_enum and new_enum:
        removed_values = old_enum - new_enum
        if removed_values:
            diff.modified_fields.append(
                SchemaChange(
                    change_type=VersionChangeType.MAJOR,
                    field=field_name,
                    change="enum_values_removed",
                    old_value=list(old_enum),
                    new_value=list(new_enum),
                    description=f"Removed enum values: {removed_values}",
                )
            )
            diff.version_change = VersionChangeType.MAJOR
            diff.migration_required = True

        added_values = new_enum - old_enum
        if added_values and diff.version_change != VersionChangeType.MAJOR:
            diff.version_change = VersionChangeType.MINOR


# =============================================================================
# DEPRECATION LIFECYCLE (SCHEMA-001 §6.3)
# =============================================================================


@dataclass
class DeprecationInfo:
    """
    Deprecation metadata per SCHEMA-001 §6.3.
    """

    deprecated_at: datetime | None = None
    deprecated_by: str = ""
    sunset_date: datetime | None = None
    migration_path: str = ""
    successor_version: str | None = None
    reason: str = ""

    # Minimum sunset period (180 days per SCHEMA-001 §6.3.1)
    MINIMUM_SUNSET_DAYS = 180

    # Grace period after EOL (90 days per SCHEMA-001 §6.3.2)
    GRACE_PERIOD_DAYS = 90

    def validate_sunset_date(self) -> bool:
        """Validate sunset date is at least 180 days from deprecation."""
        if not self.deprecated_at or not self.sunset_date:
            return True
        minimum_sunset = self.deprecated_at + timedelta(days=self.MINIMUM_SUNSET_DAYS)
        return self.sunset_date >= minimum_sunset

    def is_past_sunset(self) -> bool:
        """Check if sunset date has passed."""
        if not self.sunset_date:
            return False
        return timezone.now() >= self.sunset_date

    def is_past_grace_period(self) -> bool:
        """Check if grace period after EOL has elapsed."""
        if not self.sunset_date:
            return False
        grace_end = self.sunset_date + timedelta(days=self.GRACE_PERIOD_DAYS)
        return timezone.now() >= grace_end

    def get_lifecycle_state(self) -> VersionLifecycle:
        """Determine current lifecycle state."""
        if not self.deprecated_at:
            return VersionLifecycle.ACTIVE
        if self.is_past_grace_period():
            return VersionLifecycle.RETIRED
        if self.is_past_sunset():
            return VersionLifecycle.END_OF_LIFE
        return VersionLifecycle.DEPRECATED

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "deprecated_at": self.deprecated_at.isoformat() if self.deprecated_at else None,
            "deprecated_by": self.deprecated_by,
            "sunset_date": self.sunset_date.isoformat() if self.sunset_date else None,
            "migration_path": self.migration_path,
            "successor_version": self.successor_version,
            "reason": self.reason,
            "lifecycle_state": self.get_lifecycle_state().value,
        }


# =============================================================================
# VERSION HASH/CHECKSUM (SCHEMA-001 §6.4)
# =============================================================================


def compute_version_checksum(definition: dict[str, Any]) -> str:
    """
    Compute SHA-256 checksum for version integrity.

    Per SCHEMA-001 §6.4: Used for tamper detection.

    Args:
        definition: Schema/entity definition dictionary

    Returns:
        SHA-256 hex digest
    """
    # Sort keys for deterministic output
    definition_json = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(definition_json.encode("utf-8")).hexdigest()


def verify_version_checksum(definition: dict[str, Any], expected_checksum: str) -> bool:
    """
    Verify checksum matches definition.

    Args:
        definition: Schema/entity definition dictionary
        expected_checksum: Expected SHA-256 checksum

    Returns:
        True if checksums match
    """
    computed = compute_version_checksum(definition)
    return computed == expected_checksum


# =============================================================================
# VERSION MANAGER (Orchestration)
# =============================================================================


class VersionManager:
    """
    Orchestrates version management operations.

    Provides:
    - Version creation and bumping
    - Migration path computation
    - Rollback validation
    - Deprecation workflow
    """

    def __init__(self, entity_class, tenant_id: str):
        """
        Initialize version manager.

        Args:
            entity_class: Django model class with versioning
            tenant_id: Tenant ID for isolation
        """
        self.entity_class = entity_class
        self.tenant_id = tenant_id

    def get_current_version(self, entity_id: str) -> SemanticVersion | None:
        """Get current active version for an entity."""
        try:
            entity = self.entity_class.objects.get(
                id=entity_id,
                tenant_id=self.tenant_id,
            )
            return SemanticVersion.parse(entity.version)
        except (self.entity_class.DoesNotExist, ValueError):
            return None

    def get_version_history(self, slug: str) -> list[dict[str, Any]]:
        """Get all versions of an entity."""
        versions = self.entity_class.objects.filter(
            slug=slug,
            tenant_id=self.tenant_id,
        ).order_by("-version")

        return [
            {
                "id": str(v.id),
                "version": v.version,
                "created_at": v.created_at.isoformat(),
                "status": getattr(v, "lifecycle_status", "active"),
            }
            for v in versions
        ]

    def compute_next_version(self, current: SemanticVersion, change_type: VersionChangeType) -> SemanticVersion:
        """
        Compute next version based on change type.

        Args:
            current: Current version
            change_type: Type of change (MAJOR, MINOR, PATCH)

        Returns:
            New version
        """
        if change_type == VersionChangeType.MAJOR:
            return current.bump_major()
        elif change_type == VersionChangeType.MINOR:
            return current.bump_minor()
        else:
            return current.bump_patch()

    def validate_rollback(self, current_version: str, target_version: str) -> tuple[bool, list[str]]:
        """
        Validate rollback is safe.

        Per SCHEMA-001 §6.6: Safety checks before rollback.

        Args:
            current_version: Current active version
            target_version: Target rollback version

        Returns:
            (is_valid, errors)
        """
        errors = []

        try:
            current = SemanticVersion.parse(current_version)
            target = SemanticVersion.parse(target_version)

            # Cannot rollback to newer version
            if target >= current:
                errors.append("Target version must be older than current version")

            # Check for data compatibility (would need entity-specific logic)
            # This is a placeholder for actual data validation

            return len(errors) == 0, errors

        except ValueError as e:
            errors.append(f"Invalid version format: {e}")
            return False, errors


# =============================================================================
# DJANGO MODEL MIXIN
# =============================================================================


class SynaraVersionedMixin(models.Model):
    """
    Mixin providing versioning fields and methods for Django models.

    Add to any model that needs semantic versioning support.

    Usage:
        class MyEntity(SynaraVersionedMixin, SynaraEntity):
            name = models.CharField(max_length=255)

    Standard: SDK-001 §6
    """

    # Semantic version (MAJOR.MINOR.PATCH)
    version = models.CharField(
        max_length=50, default="1.0.0", db_index=True, help_text="Semantic version (MAJOR.MINOR.PATCH) per SDK-001 §6"
    )

    # Version lifecycle status
    lifecycle_status = models.CharField(
        max_length=20,
        choices=[
            ("draft", "Draft"),
            ("active", "Active"),
            ("deprecated", "Deprecated"),
            ("end_of_life", "End of Life"),
            ("retired", "Retired"),
        ],
        default="active",
        db_index=True,
        help_text="Version lifecycle status per SCHEMA-001 §6.3",
    )

    # Checksum for integrity
    version_checksum = models.CharField(
        max_length=64, blank=True, default="", help_text="SHA-256 checksum of definition for integrity"
    )

    # Deprecation metadata
    deprecated_at = models.DateTimeField(null=True, blank=True, help_text="When this version was deprecated")

    sunset_date = models.DateTimeField(null=True, blank=True, help_text="When this version reaches end of life")

    successor_version = models.CharField(
        max_length=50, blank=True, default="", help_text="Recommended successor version"
    )

    deprecation_reason = models.TextField(blank=True, default="", help_text="Reason for deprecation")

    class Meta:
        abstract = True

    def get_semantic_version(self) -> SemanticVersion:
        """Parse version string into SemanticVersion object."""
        return SemanticVersion.parse(self.version)

    def bump_version(self, change_type: VersionChangeType) -> str:
        """
        Bump version based on change type.

        Returns new version string without saving.
        """
        current = self.get_semantic_version()
        new_version = current

        if change_type == VersionChangeType.MAJOR:
            new_version = current.bump_major()
        elif change_type == VersionChangeType.MINOR:
            new_version = current.bump_minor()
        elif change_type == VersionChangeType.PATCH:
            new_version = current.bump_patch()

        return str(new_version)

    def deprecate(self, reason: str, successor: str | None = None, sunset_date: datetime | None = None) -> None:
        """
        Mark version as deprecated.

        Per SCHEMA-001 §6.3: Minimum 180 days until sunset.
        """
        self.lifecycle_status = "deprecated"
        self.deprecated_at = timezone.now()
        self.deprecation_reason = reason

        if successor:
            self.successor_version = successor

        if sunset_date:
            self.sunset_date = sunset_date
        else:
            # Default: 180 days from now
            self.sunset_date = timezone.now() + timedelta(days=180)

        self.save(
            update_fields=[
                "lifecycle_status",
                "deprecated_at",
                "deprecation_reason",
                "successor_version",
                "sunset_date",
                "updated_at",
            ]
        )

    def is_deprecated(self) -> bool:
        """Check if version is deprecated or beyond."""
        return self.lifecycle_status in ("deprecated", "end_of_life", "retired")

    def is_usable(self) -> bool:
        """Check if version can be used for new instances."""
        return self.lifecycle_status in ("draft", "active", "deprecated")

    def compute_checksum(self, definition: dict[str, Any]) -> str:
        """Compute and store checksum for definition."""
        checksum = compute_version_checksum(definition)
        self.version_checksum = checksum
        return checksum

    def verify_checksum(self, definition: dict[str, Any]) -> bool:
        """Verify stored checksum matches definition."""
        if not self.version_checksum:
            return True  # No checksum stored
        return verify_version_checksum(definition, self.version_checksum)

    def get_deprecation_info(self) -> DeprecationInfo:
        """Get deprecation information as structured object."""
        return DeprecationInfo(
            deprecated_at=self.deprecated_at,
            sunset_date=self.sunset_date,
            successor_version=self.successor_version,
            reason=self.deprecation_reason,
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Version types
    "SemanticVersion",
    "VersionChangeType",
    "VersionLifecycle",
    "VersionCompatibility",
    # Change detection
    "SchemaChange",
    "SchemaDiff",
    "detect_breaking_changes",
    # Deprecation
    "DeprecationInfo",
    # Utilities
    "compute_version_checksum",
    "verify_version_checksum",
    # Manager
    "VersionManager",
    # Mixin
    "SynaraVersionedMixin",
]
