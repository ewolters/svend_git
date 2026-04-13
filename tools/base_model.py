"""
Abstract base model for pluggable QMS tool modules.

Provides shared fields and behavior for all QMS tools (A3, FMEA, RCA,
CAPA, Ishikawa, C&E Matrix, etc.) on top of SynaraEntity infrastructure.

Standard:     DAT-001 §11 (Tool Model Base Class)
Compliance:   ORG-001 §2.2 (ownership), SEC-001 §5.2 (tenant isolation)

NOTE: Site FK currently points to agents_api.Site. This will become
qms_core.Site at Phase 4 cutover.
"""

from __future__ import annotations

import logging
from typing import Any

from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models

from syn.core.base_models import SynaraEntity

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL MODEL BASE CLASS (DAT-001 §11)
# =============================================================================


class ToolModel(SynaraEntity):
    """Abstract base for all QMS tool models.

    Inherits from SynaraEntity:
        id (UUID PK), correlation_id, parent_correlation_id, tenant_id,
        created_at, updated_at, created_by (CharField), updated_by (CharField),
        is_deleted, deleted_at, deleted_by, metadata (JSONField),
        objects (excludes soft-deleted), all_objects (includes deleted).

    Adds QMS-specific fields:
        title, description, status, owner (FK→User), project (FK→Project),
        site (FK→Site).

    Subclasses MUST define:
        class Status(models.TextChoices):
            DRAFT = "draft", "Draft"
            ...

    Subclasses MAY define:
        VALID_TRANSITIONS: dict[str, list[str]]
            Maps current status → allowed next statuses.
        TRANSITION_REQUIREMENTS: dict[str, list[str]]
            Maps target status → required non-empty fields.
    """

    # =========================================================================
    # COMMON QMS FIELDS
    # =========================================================================

    title = models.CharField(
        max_length=255,
        help_text="Tool record title",
    )

    description = models.TextField(
        blank=True,
        default="",
        help_text="Extended description or context",
    )

    status = models.CharField(
        max_length=25,
        default="draft",
        db_index=True,
        help_text="Lifecycle status (subclass defines choices)",
    )

    # =========================================================================
    # OWNERSHIP (ORG-001 §2.2)
    # Nullable per ORG-001 §2.2 — NULL when site-scoped in enterprise context.
    # =========================================================================

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="%(class)s_owned",
        help_text="Record owner (NULL when site-scoped per ORG-001 §2.2)",
    )

    # =========================================================================
    # PROJECT LINK (CANON-001)
    # Optional — most tools link to a Project for evidence integration.
    # =========================================================================

    project = models.ForeignKey(
        "core.Project",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="%(class)s_records",
        help_text="Linked project for evidence integration (CANON-001)",
    )

    # =========================================================================
    # SITE CONTEXT (QMS-001)
    # Optional — enterprise QMS tools are scoped to a manufacturing site.
    # NOTE: Points to agents_api.Site until Phase 4 (qms_core.Site).
    # =========================================================================

    site = models.ForeignKey(
        "agents_api.Site",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="%(class)s_records",
        help_text="Manufacturing site (enterprise QMS context)",
    )

    # =========================================================================
    # TRANSITION MAPS (override in subclasses)
    # =========================================================================

    VALID_TRANSITIONS: dict[str, list[str]] = {}
    TRANSITION_REQUIREMENTS: dict[str, list[str]] = {}

    class Meta(SynaraEntity.Meta):
        abstract = True

    def __str__(self):
        label = self.__class__.__name__
        return f"{label}: {self.title} ({self.status})"

    # =========================================================================
    # STATUS TRANSITION VALIDATION
    # =========================================================================

    def validate_transition(self, new_status: str) -> tuple[bool, str]:
        """Validate a status transition against the state machine.

        If VALID_TRANSITIONS is empty (no state machine defined), any
        transition is allowed — the subclass manages status externally.

        Returns:
            (is_valid, error_message) tuple. error_message is empty on success.
        """
        if not self.VALID_TRANSITIONS:
            return True, ""

        allowed = self.VALID_TRANSITIONS.get(self.status, [])
        if new_status not in allowed:
            return (
                False,
                f"Cannot transition from '{self.status}' to '{new_status}'. Allowed: {allowed}",
            )

        missing = self._check_transition_requirements(new_status)
        if missing:
            return (
                False,
                f"Fields required for '{new_status}': {missing}",
            )

        return True, ""

    def _check_transition_requirements(self, new_status: str) -> list[str]:
        """Check that required fields are non-empty for a target status.

        Returns list of missing field names (empty if all satisfied).
        """
        required = self.TRANSITION_REQUIREMENTS.get(new_status, [])
        missing = []
        for field_name in required:
            value = getattr(self, field_name, None)
            if value is None:
                missing.append(field_name)
            elif isinstance(value, str) and not value.strip():
                missing.append(field_name)
        return missing

    def transition_to(self, new_status: str) -> None:
        """Transition status with validation.

        Raises ValidationError if the transition is not allowed.
        Does NOT call save() — caller is responsible for persistence.
        """
        is_valid, error = self.validate_transition(new_status)
        if not is_valid:
            raise ValidationError(error)
        self.status = new_status

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """Base serialization for API responses.

        Subclasses should call super().to_dict() and extend the result
        with tool-specific fields.
        """
        return {
            "id": str(self.id),
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "owner_id": str(self.owner_id) if self.owner_id else None,
            "project_id": str(self.project_id) if self.project_id else None,
            "site_id": str(self.site_id) if self.site_id else None,
            "created_by": self.created_by,
            "updated_by": self.updated_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "correlation_id": str(self.correlation_id),
        }
