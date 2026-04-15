# ishikawa models — extracted from agents_api.models
# db_table overrides point at existing tables, no migration needed

import uuid

from django.conf import settings
from django.db import models

DEFAULT_6M_BRANCHES = [
    {"category": "Man", "causes": []},
    {"category": "Machine", "causes": []},
    {"category": "Method", "causes": []},
    {"category": "Material", "causes": []},
    {"category": "Measurement", "causes": []},
    {"category": "Mother Nature", "causes": []},
]


class IshikawaDiagram(models.Model):
    """Ishikawa (Fishbone) diagram for common cause analysis.

    For COMMON CAUSE problems — systemic issues mapped across 6M categories.
    Unlike RCA (special cause, causal chain), Ishikawa maps all contributing
    factors to a process-level effect and feeds Kaizen improvement.
    """

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        ANALYZING = "analyzing", "Analyzing"
        COMPLETE = "complete", "Complete"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="ishikawa_diagrams",
    )
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="ik_diagrams",
    )
    project = models.ForeignKey(
        "core.Project",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="ik_diagrams",
    )
    title = models.CharField(max_length=255, blank=True)
    effect = models.TextField(
        blank=True,
        help_text="The process-level effect being analyzed (the fish head)",
    )
    branches = models.JSONField(
        default=list,
        help_text="6M category branches with recursive causes",
    )
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.DRAFT,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "ishikawa_diagrams"

        ordering = ["-updated_at"]
        verbose_name = "Ishikawa Diagram"
        verbose_name_plural = "Ishikawa Diagrams"

    def __str__(self):
        return f"Ishikawa: {self.title or self.effect[:50]} ({self.status})"

    def to_dict(self):
        return {
            "id": str(self.id),
            "project_id": str(self.project_id) if self.project_id else None,
            "title": self.title,
            "effect": self.effect,
            "branches": self.branches,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
