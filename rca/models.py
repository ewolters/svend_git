# rca models — extracted from agents_api.models
# db_table overrides point at existing tables, no migration needed

import uuid

from django.conf import settings
from django.db import models


class RCASession(models.Model):
    """Root Cause Analysis session for special cause investigation."""

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        INVESTIGATING = "investigating", "Investigating"
        ROOT_CAUSE_IDENTIFIED = "root_cause_identified", "Root Cause Identified"
        VERIFIED = "verified", "Verified"
        CLOSED = "closed", "Closed"

    VALID_TRANSITIONS = {
        "draft": ["investigating"],
        "investigating": ["root_cause_identified", "draft"],
        "root_cause_identified": ["verified", "investigating"],
        "verified": ["closed", "investigating"],
        "closed": ["investigating"],
    }

    TRANSITION_REQUIREMENTS = {
        "root_cause_identified": ["root_cause"],
        "verified": ["countermeasure"],
        "closed": ["evaluation"],
    }

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="rca_sessions_ext",
    )
    site = models.ForeignKey(
        "agents_api.Site",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="rca_records_ext",
    )
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="rca_sessions",
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="rca_sessions_created_ext",
    )

    project = models.ForeignKey(
        "core.Project",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="rca_sessions_ext",
    )

    a3_report = models.ForeignKey(
        "agents_api.A3Report",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="rca_sessions_ext",
    )

    title = models.CharField(max_length=255, blank=True)
    event = models.TextField(help_text="Description of the incident")

    chain = models.JSONField(default=list, help_text="Causal chain steps with critiques")

    root_cause = models.TextField(blank=True)
    countermeasure = models.TextField(blank=True)
    evaluation = models.TextField(blank=True)
    reopen_reason = models.TextField(blank=True)

    status = models.CharField(max_length=25, choices=Status.choices, default=Status.DRAFT)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    embedding = models.BinaryField(null=True, blank=True)

    class Meta:
        db_table = "rca_sessions"
        managed = False
        ordering = ["-updated_at"]
        verbose_name = "RCA Session"
        verbose_name_plural = "RCA Sessions"

    def __str__(self):
        return f"RCA: {self.title or self.event[:50]} ({self.status})"

    def validate_transition(self, new_status, reopen_reason=""):
        allowed = self.VALID_TRANSITIONS.get(self.status, [])
        if new_status not in allowed:
            return (False, f"Cannot transition from '{self.status}' to '{new_status}'. Allowed: {allowed}")
        required_fields = self.TRANSITION_REQUIREMENTS.get(new_status, [])
        for field in required_fields:
            if not getattr(self, field, "").strip():
                return (False, f"Field '{field}' is required before transitioning to '{new_status}'")
        if self.status == "closed" and new_status == "investigating":
            if not reopen_reason.strip():
                return False, "Reopening a closed session requires a reopen_reason"
        return True, ""

    def generate_embedding(self):
        from agents_api.embeddings import generate_rca_embedding

        embedding = generate_rca_embedding(event=self.event, chain=self.chain or [], root_cause=self.root_cause)
        if embedding is not None:
            self.embedding = embedding.tobytes()
            return True
        return False

    def get_embedding(self):
        import numpy as np

        if self.embedding is None:
            return None
        return np.frombuffer(self.embedding, dtype=np.float32)

    def to_dict(self):
        return {
            "id": str(self.id),
            "project_id": str(self.project_id) if self.project_id else None,
            "a3_report_id": str(self.a3_report_id) if self.a3_report_id else None,
            "title": self.title,
            "site_id": str(self.site_id) if self.site_id else None,
            "created_by_id": str(self.created_by_id) if self.created_by_id else None,
            "event": self.event,
            "chain": self.chain,
            "root_cause": self.root_cause,
            "countermeasure": self.countermeasure,
            "evaluation": self.evaluation,
            "reopen_reason": self.reopen_reason,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
