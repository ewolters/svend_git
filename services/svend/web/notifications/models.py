"""Notification system models — NTF-001."""

import uuid

from django.conf import settings
from django.db import models

from .tokens import NotificationToken  # noqa: F401 — registered for migrations


class NotificationType(models.TextChoices):
    """NTF-001 §4.2 — enumerated notification categories."""

    CAPA_STATUS = "capa_status", "CAPA Status Change"
    NCR_OVERDUE = "ncr_overdue", "NCR Overdue"
    NCR_ASSIGNED = "ncr_assigned", "NCR Assigned"
    APPROVAL_REQUEST = "approval_request", "Approval Request"
    ESIG_REQUEST = "esig_request", "E-Signature Request"
    SPC_ALARM = "spc_alarm", "SPC Alarm"
    REVIEW_DUE = "review_due", "Management Review Due"
    DOC_REVIEW = "doc_review", "Document Review Reminder"
    TRAINING_DUE = "training_due", "Training Due"
    TRAINING_EXPIRED = "training_expired", "Training Expired"
    AUDIT_SCHEDULED = "audit_scheduled", "Audit Scheduled"
    ACTION_DUE = "action_due", "Action Item Due"
    ASSIGNMENT = "assignment", "Assignment Change"
    SYSTEM = "system", "System Notification"
    INCIDENT_CREATED = "incident_created", "Incident Created"
    INCIDENT_ESCALATED = "incident_escalated", "Incident Escalated"
    INCIDENT_RESOLVED = "incident_resolved", "Incident Resolved"
    # Harada Method
    ROUTINE_REMINDER = "routine_reminder", "Routine Reminder"
    ROUTINE_STREAK = "routine_streak", "Routine Streak"
    ROUTINE_MISSED = "routine_missed", "Routine Missed"
    DIARY_REMINDER = "diary_reminder", "Daily Diary Reminder"
    GOAL_DUE = "goal_due", "Goal Approaching Deadline"
    HANSEI_DUE = "hansei_due", "Hansei Kai Reflection Due"


# Fields that are immutable after creation (NTF-001 §4.4)
_IMMUTABLE_FIELDS = frozenset(["recipient_id", "notification_type", "title", "message", "entity_type", "entity_id"])


class Notification(models.Model):
    """NTF-001 §4.1 — core notification record.

    Write-once except for is_read toggle. Immutability enforced in save().
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    recipient = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="notifications",
        db_index=True,
    )
    notification_type = models.CharField(max_length=30, choices=NotificationType.choices, db_index=True)
    title = models.CharField(max_length=300)
    message = models.TextField(blank=True, default="")
    entity_type = models.CharField(max_length=30, blank=True, default="")
    entity_id = models.UUIDField(null=True, blank=True)
    is_read = models.BooleanField(default=False, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        db_table = "notifications"
        ordering = ["-created_at"]
        indexes = [
            models.Index(
                fields=["recipient", "is_read", "-created_at"],
                name="ntf_recipient_unread",
            ),
        ]

    def __str__(self):
        return f"[{self.notification_type}] {self.title}"

    def save(self, **kwargs):
        # NTF-001 §4.4 — immutability enforcement
        if self.pk and not self._state.adding:
            try:
                existing = Notification.objects.only(*_IMMUTABLE_FIELDS).get(pk=self.pk)
            except Notification.DoesNotExist:
                pass
            else:
                for field in _IMMUTABLE_FIELDS:
                    if getattr(self, field) != getattr(existing, field):
                        raise ValueError(f"Notification field '{field}' is immutable after creation")
        super().save(**kwargs)

    def to_dict(self):
        return {
            "id": str(self.id),
            "notification_type": self.notification_type,
            "title": self.title,
            "message": self.message,
            "entity_type": self.entity_type,
            "entity_id": str(self.entity_id) if self.entity_id else None,
            "is_read": self.is_read,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
