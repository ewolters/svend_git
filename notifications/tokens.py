"""Notification action tokens — NTF-001 §5.2.1.

Single-use, time-limited, action-scoped tokens for email one-click actions.
Separate from agents_api.ActionToken (QMS-002) by design.
"""

import secrets
import uuid
from datetime import timedelta

from django.conf import settings
from django.db import models
from django.utils import timezone


class NotificationToken(models.Model):
    """Secure token for one-click email actions on notifications.

    Tokens are cryptographically random (secrets.token_urlsafe >= 32 bytes),
    expire after 72 hours or first use, and are action-scoped.

    Standard: NTF-001 §5.2.1
    """

    ACTION_CHOICES = [
        ("acknowledge", "Acknowledge"),
        ("view", "View"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="notification_tokens",
    )
    notification = models.ForeignKey(
        "notifications.Notification",
        on_delete=models.CASCADE,
        related_name="tokens",
    )
    action_type = models.CharField(max_length=20, choices=ACTION_CHOICES)
    token = models.CharField(max_length=64, unique=True, db_index=True)
    expires_at = models.DateTimeField()
    used_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    # Delivery tracking
    email_sent_at = models.DateTimeField(null=True, blank=True)
    email_failed_at = models.DateTimeField(null=True, blank=True)
    email_opened_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "notification_tokens"
        indexes = [
            models.Index(fields=["user", "-created_at"], name="ntf_token_user_recent"),
        ]

    def save(self, *args, **kwargs):
        if not self.token:
            self.token = secrets.token_urlsafe(32)
        if not self.expires_at:
            self.expires_at = timezone.now() + timedelta(hours=72)
        super().save(*args, **kwargs)

    @property
    def is_valid(self):
        return self.used_at is None and self.expires_at > timezone.now()

    def use(self):
        self.used_at = timezone.now()
        self.save(update_fields=["used_at"])

    def __str__(self):
        return f"NTFToken({self.action_type}) for {self.notification_id}"

    def to_dict(self):
        return {
            "id": str(self.id),
            "notification_id": str(self.notification_id),
            "action_type": self.action_type,
            "is_valid": self.is_valid,
            "expires_at": self.expires_at.isoformat(),
        }
