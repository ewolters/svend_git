"""Notification system models — NTF-001."""

import fnmatch
import hashlib
import hmac
import json
import secrets
import uuid

from django.conf import settings
from django.db import models

from core.encryption import EncryptedCharField

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
    # Resource Commitments (QMS-002)
    COMMITMENT_REQUESTED = "commitment_requested", "Commitment Requested"
    COMMITMENT_CONFIRMED = "commitment_confirmed", "Commitment Confirmed"
    COMMITMENT_DECLINED = "commitment_declined", "Commitment Declined"
    # Harada Method
    ROUTINE_REMINDER = "routine_reminder", "Routine Reminder"
    ROUTINE_STREAK = "routine_streak", "Routine Streak"
    ROUTINE_MISSED = "routine_missed", "Routine Missed"
    DIARY_REMINDER = "diary_reminder", "Daily Diary Reminder"
    GOAL_DUE = "goal_due", "Goal Approaching Deadline"
    HANSEI_DUE = "hansei_due", "Hansei Kai Reflection Due"
    # Phase C-E QMS features
    AFE_APPROVAL = "afe_approval", "AFE Approval"
    COMPLAINT_ASSIGNED = "complaint_assigned", "Complaint Assigned"
    COMPLAINT_STATUS = "complaint_status", "Complaint Status Change"
    AUDIT_FINDING = "audit_finding", "Audit Finding"
    CHECKLIST_COMPLETE = "checklist_complete", "Checklist Completed"


# Fields that are immutable after creation (NTF-001 §4.4)
_IMMUTABLE_FIELDS = frozenset(
    [
        "recipient_id",
        "notification_type",
        "title",
        "message",
        "entity_type",
        "entity_id",
    ]
)


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


# ---------------------------------------------------------------------------
# Webhook Event Delivery — NTF-001 §5.4
# ---------------------------------------------------------------------------

# Tier limits for webhook endpoints
WEBHOOK_ENDPOINT_LIMITS = {
    "team": 3,
    "enterprise": 10,
}

WEBHOOK_DAILY_EVENT_LIMITS = {
    "team": 1000,
    "enterprise": 10000,
}

# Circuit breaker threshold
WEBHOOK_CIRCUIT_BREAKER_THRESHOLD = 10

# Retry schedule (seconds after first attempt)
WEBHOOK_RETRY_DELAYS = [60, 300, 1800]  # 1min, 5min, 30min


class WebhookEndpoint(models.Model):
    """Customer-registered HTTP endpoint for event delivery.

    ⚠ SECURITY: Outbound only — no inbound attack surface. HMAC-signed
    payloads prevent spoofing. HTTPS required for endpoint URLs.

    Standard: NTF-001 §5.4
    Compliance: SOC 2 CC6.1 (Logical Access Security)
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="webhook_endpoints",
    )
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        related_name="webhook_endpoints",
        null=True,
        blank=True,
    )

    url = models.URLField(max_length=500)
    secret = EncryptedCharField()  # HMAC signing secret — encrypted at rest
    event_patterns = models.JSONField(default=list)  # ["fmea.*", "capa.status_changed"]
    description = models.CharField(max_length=200, blank=True, default="")

    is_active = models.BooleanField(default=True)
    failure_count = models.IntegerField(default=0)
    disabled_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "webhook_endpoints"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "is_active"], name="webhook_user_active"),
            models.Index(fields=["tenant", "is_active"], name="webhook_tenant_active"),
        ]

    def __str__(self):
        return f"{self.url} ({self.description or 'unnamed'})"

    def matches_event(self, event_name):
        """Check if this endpoint subscribes to the given event name."""
        for pattern in self.event_patterns:
            if pattern == "*":
                return True
            if fnmatch.fnmatch(event_name, pattern):
                return True
        return False

    def record_success(self):
        """Reset failure count on successful delivery."""
        if self.failure_count > 0:
            type(self).objects.filter(pk=self.pk).update(failure_count=0)
            self.failure_count = 0

    def record_failure(self):
        """Increment failure count. Disable if circuit breaker threshold hit."""
        from django.utils import timezone

        new_count = models.F("failure_count") + 1
        type(self).objects.filter(pk=self.pk).update(failure_count=new_count)
        self.failure_count += 1

        if self.failure_count >= WEBHOOK_CIRCUIT_BREAKER_THRESHOLD:
            type(self).objects.filter(pk=self.pk).update(is_active=False, disabled_at=timezone.now())
            self.is_active = False

    def sign_payload(self, payload_dict):
        """Compute HMAC-SHA256 signature for a payload dict.

        Returns: "sha256=<hex digest>"
        """
        payload_bytes = json.dumps(payload_dict, sort_keys=True).encode("utf-8")
        digest = hmac.new(self.secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
        return f"sha256={digest}"

    @staticmethod
    def generate_secret():
        """Generate a random HMAC signing secret."""
        return secrets.token_hex(32)

    @classmethod
    def create_for_user(cls, user, url, event_patterns, description="", tenant=None):
        """Create a new webhook endpoint.

        Returns (secret_plaintext, endpoint_instance). Secret shown once.
        """
        if not url.startswith("https://"):
            raise ValueError("Webhook URLs must use HTTPS")

        secret = cls.generate_secret()
        endpoint = cls.objects.create(
            user=user,
            tenant=tenant,
            url=url,
            secret=secret,
            event_patterns=event_patterns,
            description=description,
        )
        return secret, endpoint


class WebhookDelivery(models.Model):
    """Tracks a single webhook delivery attempt (or retry chain).

    Standard: NTF-001 §5.4
    """

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        DELIVERED = "delivered", "Delivered"
        FAILED = "failed", "Failed"
        EXHAUSTED = "exhausted", "Exhausted"  # All retries used

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    endpoint = models.ForeignKey(
        WebhookEndpoint,
        on_delete=models.CASCADE,
        related_name="deliveries",
    )
    event_name = models.CharField(max_length=100, db_index=True)
    payload = models.JSONField(default=dict)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    response_code = models.IntegerField(null=True, blank=True)
    response_body = models.TextField(blank=True, default="")  # First 500 chars
    attempt_count = models.IntegerField(default=0)
    next_retry_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    delivered_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "webhook_deliveries"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["endpoint", "-created_at"], name="whd_endpoint_created"),
            models.Index(fields=["status", "next_retry_at"], name="whd_status_retry"),
        ]

    def __str__(self):
        return f"{self.event_name} → {self.endpoint.url} ({self.status})"
