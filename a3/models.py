# a3 models — extracted from agents_api.models
# db_table overrides point at existing tables, no migration needed

import uuid

from django.conf import settings
from django.db import models


class A3Report(models.Model):
    """Toyota-style A3 problem-solving report."""

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        IN_PROGRESS = "in_progress", "In Progress"
        REVIEW = "review", "Under Review"
        COMPLETE = "complete", "Complete"
        ARCHIVED = "archived", "Archived"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="a3_reports_ext",
    )
    site = models.ForeignKey(
        "agents_api.Site",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="a3_records_ext",
    )
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="a3_reports",
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="a3_reports_created_ext",
    )

    project = models.ForeignKey(
        "core.Project",
        on_delete=models.CASCADE,
        related_name="a3_reports_ext",
    )

    notebook = models.ForeignKey(
        "core.Notebook",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="a3_reports_ext",
    )

    title = models.CharField(max_length=255)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.DRAFT)

    background = models.TextField(blank=True)
    current_condition = models.TextField(blank=True)
    goal = models.TextField(blank=True)
    root_cause = models.TextField(blank=True)

    countermeasures = models.TextField(blank=True)
    implementation_plan = models.TextField(blank=True)
    follow_up = models.TextField(blank=True)

    imported_from = models.JSONField(default=dict, blank=True)
    embedded_diagrams = models.JSONField(default=dict, blank=True)
    last_critique = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "a3_reports"
        managed = False
        ordering = ["-updated_at"]
        verbose_name = "A3 Report"
        verbose_name_plural = "A3 Reports"

    def __str__(self):
        return f"A3: {self.title} ({self.status})"

    def to_dict(self):
        return {
            "id": str(self.id),
            "project_id": str(self.project_id),
            "notebook_id": str(self.notebook_id) if self.notebook_id else None,
            "title": self.title,
            "status": self.status,
            "site_id": str(self.site_id) if self.site_id else None,
            "created_by_id": str(self.created_by_id) if self.created_by_id else None,
            "background": self.background,
            "current_condition": self.current_condition,
            "goal": self.goal,
            "root_cause": self.root_cause,
            "countermeasures": self.countermeasures,
            "implementation_plan": self.implementation_plan,
            "follow_up": self.follow_up,
            "imported_from": self.imported_from,
            "embedded_diagrams": self.embedded_diagrams,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
