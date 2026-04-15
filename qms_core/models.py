"""QMS Core models — shared infrastructure for cross-module integration.

ArtifactReference: universal reference registry for the pull contract.
When a consumer (A3, RCA, investigation) pulls from a source (workbench
analysis, FMEA row, etc.), a row is created here tracking the link.

Architecture: docs/planning/object_271/qms_architecture.md §2.4
"""

import uuid

from django.conf import settings
from django.db import models


class ArtifactReference(models.Model):
    """Universal reference registry — who pulls from whom.

    When an RCA session pulls a chart from a workbench analysis, a row is created
    here recording that link. Used for: delete friction (warn before deleting
    a source that has consumers), tombstone rendering (consumers detect orphaned
    refs), and dependency tracing (audit trail).
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Source side
    source_app = models.CharField(max_length=50, db_index=True)
    source_type = models.CharField(max_length=100)
    source_id = models.UUIDField()
    artifact_key = models.CharField(max_length=200, blank=True, default="")

    # Consumer side
    consumer_app = models.CharField(max_length=50, db_index=True)
    consumer_type = models.CharField(max_length=100)
    consumer_id = models.UUIDField()

    # Lifecycle
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="artifact_references",
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="artifact_references",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    source_deleted_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "artifact_references"
        indexes = [
            models.Index(fields=["source_app", "source_type", "source_id"]),
            models.Index(fields=["consumer_app", "consumer_type", "consumer_id"]),
            models.Index(fields=["tenant", "source_app"]),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=[
                    "source_app",
                    "source_id",
                    "artifact_key",
                    "consumer_app",
                    "consumer_type",
                    "consumer_id",
                ],
                name="unique_artifact_reference",
            ),
        ]

    def __str__(self):
        key = f"/{self.artifact_key}" if self.artifact_key else ""
        return f"{self.source_app}:{self.source_type}/{self.source_id}{key} -> {self.consumer_app}:{self.consumer_type}/{self.consumer_id}"
