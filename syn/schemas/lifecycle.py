"""
Schema Lifecycle (SCONF-001 §5)
===============================

Schema lifecycle management for version transitions and deprecation.

Standard:     SCONF-001 §5.3 (Schema Lifecycle)
Compliance:   CGS-1001 §6 (Versioning Governance)
Location:     syn/schemas/lifecycle.py
Version:      1.0.0
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from syn.schemas.types import ChangeType, SchemaStatus

# =============================================================================
# LIFECYCLE TRANSITIONS (SCONF-001 §5.3)
# =============================================================================


# Valid state transitions
VALID_TRANSITIONS: Dict[SchemaStatus, List[SchemaStatus]] = {
    SchemaStatus.DRAFT: [SchemaStatus.ACTIVE, SchemaStatus.RETIRED],
    SchemaStatus.ACTIVE: [SchemaStatus.DEPRECATED],
    SchemaStatus.DEPRECATED: [SchemaStatus.END_OF_LIFE, SchemaStatus.ACTIVE],  # Can reactivate
    SchemaStatus.END_OF_LIFE: [SchemaStatus.RETIRED],
    SchemaStatus.RETIRED: [],  # Terminal state
}


@dataclass
class LifecycleTransition:
    """
    Record of a schema lifecycle transition.

    Standard: SCONF-001 §5.3
    """

    from_status: SchemaStatus
    to_status: SchemaStatus
    transitioned_at: datetime
    transitioned_by: str  # User ID or system
    reason: Optional[str] = None
    notes: Optional[str] = None


class SchemaLifecycle:
    """
    Schema lifecycle manager per SCONF-001 §5.3.

    Manages:
    - Version transitions
    - Deprecation scheduling
    - End-of-life management
    - Retention policies
    """

    # Configuration per SCONF-001 §5.3
    DEPRECATION_NOTICE_DAYS: int = 30
    EOL_GRACE_DAYS: int = 90
    MAX_VERSIONS_RETAINED: int = 10

    def __init__(
        self,
        schema_uri: str,
        current_status: SchemaStatus = SchemaStatus.DRAFT,
        version: str = "1.0.0",
    ):
        """
        Initialize lifecycle manager.

        Args:
            schema_uri: Schema URI (e.g., "sconf://qms/capa/1.0.0")
            current_status: Current lifecycle status
            version: Current version string
        """
        self.schema_uri = schema_uri
        self.current_status = current_status
        self.version = version
        self.transitions: List[LifecycleTransition] = []
        self.deprecation_date: Optional[datetime] = None
        self.eol_date: Optional[datetime] = None

    def can_transition_to(self, new_status: SchemaStatus) -> bool:
        """
        Check if transition to new status is valid.

        Args:
            new_status: Target status

        Returns:
            True if transition is allowed
        """
        return new_status in VALID_TRANSITIONS.get(self.current_status, [])

    def transition(
        self,
        to_status: SchemaStatus,
        by: str,
        reason: Optional[str] = None,
    ) -> LifecycleTransition:
        """
        Transition schema to new status.

        Args:
            to_status: Target status
            by: User or system performing transition
            reason: Reason for transition

        Returns:
            LifecycleTransition record

        Raises:
            ValueError: If transition is not valid
        """
        if not self.can_transition_to(to_status):
            raise ValueError(f"Invalid transition from {self.current_status.value} to {to_status.value}")

        transition = LifecycleTransition(
            from_status=self.current_status,
            to_status=to_status,
            transitioned_at=datetime.utcnow(),
            transitioned_by=by,
            reason=reason,
        )

        self.transitions.append(transition)
        self.current_status = to_status

        # Set deprecation/EOL dates
        if to_status == SchemaStatus.DEPRECATED:
            self.deprecation_date = datetime.utcnow()
            self.eol_date = datetime.utcnow() + timedelta(days=self.EOL_GRACE_DAYS)

        return transition

    def deprecate(
        self,
        by: str,
        reason: str,
        eol_days: Optional[int] = None,
    ) -> LifecycleTransition:
        """
        Deprecate the schema.

        Args:
            by: User or system performing deprecation
            reason: Reason for deprecation
            eol_days: Days until end-of-life (default: EOL_GRACE_DAYS)

        Returns:
            LifecycleTransition record
        """
        transition = self.transition(SchemaStatus.DEPRECATED, by, reason)

        grace_days = eol_days or self.EOL_GRACE_DAYS
        self.eol_date = datetime.utcnow() + timedelta(days=grace_days)

        return transition

    def is_deprecated(self) -> bool:
        """Check if schema is deprecated."""
        return self.current_status in [
            SchemaStatus.DEPRECATED,
            SchemaStatus.END_OF_LIFE,
        ]

    def is_retired(self) -> bool:
        """Check if schema is retired (terminal state)."""
        return self.current_status == SchemaStatus.RETIRED

    def days_until_eol(self) -> Optional[int]:
        """Get days until end-of-life, if scheduled."""
        if not self.eol_date:
            return None
        delta = self.eol_date - datetime.utcnow()
        return max(0, delta.days)

    def needs_deprecation_notice(self) -> bool:
        """Check if deprecation notice should be sent."""
        if self.current_status != SchemaStatus.DEPRECATED:
            return False
        days = self.days_until_eol()
        return days is not None and days <= self.DEPRECATION_NOTICE_DAYS


# =============================================================================
# VERSION MANAGEMENT (SCONF-001 §5.1)
# =============================================================================


@dataclass
class SemanticVersion:
    """
    Semantic version per SCONF-001 §5.1.

    Format: MAJOR.MINOR.PATCH
    """

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, version_str: str) -> "SemanticVersion":
        """Parse version string."""
        parts = version_str.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {version_str}")
        return cls(
            major=int(parts[0]),
            minor=int(parts[1]),
            patch=int(parts[2]),
        )

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def bump(self, change_type: ChangeType) -> "SemanticVersion":
        """
        Bump version based on change type.

        Args:
            change_type: Type of change

        Returns:
            New SemanticVersion
        """
        if change_type == ChangeType.MAJOR:
            return SemanticVersion(self.major + 1, 0, 0)
        elif change_type == ChangeType.MINOR:
            return SemanticVersion(self.major, self.minor + 1, 0)
        else:  # PATCH
            return SemanticVersion(self.major, self.minor, self.patch + 1)

    def is_compatible_with(self, other: "SemanticVersion") -> bool:
        """
        Check backward compatibility.

        Returns:
            True if other version is backward compatible
        """
        return self.major == other.major


def determine_change_type(
    added_fields: List[str],
    removed_fields: List[str],
    modified_fields: List[str],
) -> ChangeType:
    """
    Determine change type based on schema diff.

    Standard: SCONF-001 §5.1

    Args:
        added_fields: New optional fields
        removed_fields: Removed fields
        modified_fields: Fields with type/constraint changes

    Returns:
        ChangeType for versioning
    """
    # Breaking changes
    if removed_fields or modified_fields:
        return ChangeType.MAJOR

    # Compatible additions
    if added_fields:
        return ChangeType.MINOR

    # Non-functional
    return ChangeType.PATCH
