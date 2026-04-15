# learn models — extracted from agents_api.models
# db_table overrides point at existing tables, no migration needed

import uuid

from django.conf import settings
from django.db import models


class SectionProgress(models.Model):
    """Tracks a user's completion of individual course sections."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="learn_section_progress",
    )
    module_id = models.CharField(max_length=64)
    section_id = models.CharField(max_length=64)
    is_completed = models.BooleanField(default=False, db_column="completed")
    completed_at = models.DateTimeField(null=True, blank=True)
    time_spent_seconds = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "learn_section_progress"

        ordering = ["-updated_at"]

    def __str__(self):
        status = "done" if self.is_completed else "in progress"
        return f"{self.user} — {self.module_id}/{self.section_id} ({status})"


class AssessmentAttempt(models.Model):
    """Records a single certification assessment attempt."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="learn_assessment_attempts",
    )
    questions = models.JSONField(default=list)
    answers = models.JSONField(default=dict)
    score = models.FloatField(null=True, blank=True)
    is_passed = models.BooleanField(default=False, db_column="passed")
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "learn_assessment_attempt"

        ordering = ["-started_at"]

    def __str__(self):
        score_str = f"{self.score:.0%}" if self.score is not None else "pending"
        return f"{self.user} — assessment {score_str}"


class LearnSession(models.Model):
    """Active learning session for a tool-integrated tutorial section."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="learn_sessions_ext",
    )
    module_id = models.CharField(max_length=64)
    section_id = models.CharField(max_length=64)
    project = models.ForeignKey(
        "core.Project",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="learn_sessions_ext",
        help_text="Sandbox project for this learning session",
    )
    state = models.JSONField(default=dict)
    steps_completed = models.JSONField(default=list)
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "learn_session"

        ordering = ["-started_at"]

    def __str__(self):
        n = len(self.steps_completed) if self.steps_completed else 0
        status = "done" if self.completed_at else f"{n} steps"
        return f"{self.user} — {self.module_id}/{self.section_id} ({status})"
