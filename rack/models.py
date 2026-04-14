# rack models — extracted from agents_api.models
# db_table overrides point at existing tables, no migration needed

import uuid

from django.conf import settings
from django.db import models


class RackSession(models.Model):
    """Persisted rack configuration — units, cables, mainframe state."""

    class SessionType(models.TextChoices):
        SANDBOX = "sandbox", "Sandbox"
        ARTIFACT = "artifact", "Artifact"
        WORKFLOW = "workflow", "Workflow"
        DASHBOARD = "dashboard", "Dashboard"

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        ACTIVE = "active", "Active"
        ARCHIVED = "archived", "Archived"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255, default="Untitled Rack")
    description = models.TextField(blank=True)
    session_type = models.CharField(
        max_length=20,
        choices=SessionType.choices,
        default=SessionType.SANDBOX,
    )
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.DRAFT,
    )

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="rack_sessions_ext",
    )
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="rack_sessions_ext",
    )
    project = models.ForeignKey(
        "core.Project",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="rack_sessions_ext",
    )

    state = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "rack_sessions"
        managed = False
        ordering = ["-updated_at"]

    def __str__(self):
        return f"{self.title} ({self.session_type})"

    def to_dict(self):
        return {
            "id": str(self.id),
            "title": self.title,
            "description": self.description,
            "session_type": self.session_type,
            "status": self.status,
            "state": self.state,
            "project_id": str(self.project_id) if self.project_id else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
