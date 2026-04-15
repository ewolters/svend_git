# reports models — extracted from agents_api.models
# db_table overrides point at existing tables, no migration needed

import uuid

from django.conf import settings
from django.db import models


class Report(models.Model):
    """Flexible report model for CAPA, 8D, and future report types."""

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        IN_PROGRESS = "in_progress", "In Progress"
        REVIEW = "review", "Under Review"
        COMPLETE = "complete", "Complete"
        ARCHIVED = "archived", "Archived"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="reports_reports",
    )

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="reports_ext",
    )

    project = models.ForeignKey(
        "core.Project",
        on_delete=models.CASCADE,
        related_name="reports_ext",
    )

    report_type = models.CharField(max_length=30)
    title = models.CharField(max_length=255)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.DRAFT)

    sections = models.JSONField(default=dict, blank=True)
    imported_from = models.JSONField(default=dict, blank=True)
    embedded_diagrams = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "reports"

        ordering = ["-updated_at"]
        verbose_name = "Report"
        verbose_name_plural = "Reports"

    def __str__(self):
        return f"{self.get_type_name()}: {self.title} ({self.status})"

    def get_type_name(self):
        from agents_api.report_types import REPORT_TYPES

        rt = REPORT_TYPES.get(self.report_type, {})
        return rt.get("name", self.report_type.upper())

    def to_dict(self):
        return {
            "id": str(self.id),
            "project_id": str(self.project_id),
            "report_type": self.report_type,
            "type_name": self.get_type_name(),
            "title": self.title,
            "status": self.status,
            "sections": self.sections,
            "imported_from": self.imported_from,
            "embedded_diagrams": self.embedded_diagrams,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
