"""Forge models for synthetic data generation."""

import uuid
from django.conf import settings
from django.db import models
from django.utils import timezone
from accounts.models import User
from accounts.constants import Tier  # Use unified Tier


class DataType(models.TextChoices):
    TABULAR = "tabular", "Tabular"
    TEXT = "text", "Text"


class QualityLevel(models.TextChoices):
    STANDARD = "standard", "Standard"
    PREMIUM = "premium", "Premium"


class JobStatus(models.TextChoices):
    QUEUED = "queued", "Queued"
    PROCESSING = "processing", "Processing"
    COMPLETED = "completed", "Completed"
    FAILED = "failed", "Failed"
    CANCELLED = "cancelled", "Cancelled"


# Removed duplicate Tier class - now using accounts.constants.Tier


class APIKey(models.Model):
    """API key for Forge access."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="forge_api_keys")
    name = models.CharField(max_length=100)
    key_hash = models.CharField(max_length=64, unique=True, db_index=True)
    key_prefix = models.CharField(max_length=8)  # First 8 chars for display
    tier = models.CharField(max_length=20, choices=Tier.choices, default=Tier.FREE)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_used_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        verbose_name = "API Key"
        verbose_name_plural = "API Keys"

    def __str__(self):
        return f"{self.name} ({self.key_prefix}...)"


class Job(models.Model):
    """Synthetic data generation job."""

    job_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    api_key = models.ForeignKey(APIKey, on_delete=models.CASCADE, related_name="jobs", null=True, blank=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="forge_jobs", null=True, blank=True)

    # Job configuration
    data_type = models.CharField(max_length=20, choices=DataType.choices)
    domain = models.CharField(max_length=100, blank=True, default="")
    record_count = models.PositiveIntegerField()
    schema_def = models.JSONField(default=dict)
    quality_level = models.CharField(max_length=20, choices=QualityLevel.choices, default=QualityLevel.STANDARD)
    output_format = models.CharField(max_length=20, default="jsonl")

    # Status
    status = models.CharField(max_length=20, choices=JobStatus.choices, default=JobStatus.QUEUED)
    progress = models.PositiveSmallIntegerField(default=0)  # 0-100
    error_message = models.TextField(blank=True, default="")

    # Results
    records_generated = models.PositiveIntegerField(null=True, blank=True)
    result_path = models.CharField(max_length=500, blank=True, default="")
    result_size_bytes = models.PositiveIntegerField(null=True, blank=True)
    quality_score = models.FloatField(null=True, blank=True)
    quality_report = models.JSONField(null=True, blank=True)

    # Billing
    cost_cents = models.PositiveIntegerField(default=0)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    # Tempora task tracking
    task_id = models.UUIDField(null=True, blank=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["status", "created_at"]),
            models.Index(fields=["api_key", "created_at"]),
        ]

    def __str__(self):
        return f"Job {self.job_id} ({self.data_type}, {self.status})"

    def mark_processing(self):
        self.status = JobStatus.PROCESSING
        self.started_at = timezone.now()
        self.save(update_fields=["status", "started_at"])

    def mark_completed(self, result_path: str, records: int, size_bytes: int):
        self.status = JobStatus.COMPLETED
        self.completed_at = timezone.now()
        self.result_path = result_path
        self.records_generated = records
        self.result_size_bytes = size_bytes
        self.progress = 100
        self.save(update_fields=[
            "status", "completed_at", "result_path",
            "records_generated", "result_size_bytes", "progress"
        ])

    def mark_failed(self, error: str):
        self.status = JobStatus.FAILED
        self.error_message = error[:1000]
        self.completed_at = timezone.now()
        self.save(update_fields=["status", "error_message", "completed_at"])


class UsageLog(models.Model):
    """Usage tracking for billing."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    api_key = models.ForeignKey(APIKey, on_delete=models.CASCADE, related_name="usage_logs")
    job = models.ForeignKey(Job, on_delete=models.CASCADE, related_name="usage_logs")

    data_type = models.CharField(max_length=20, choices=DataType.choices)
    quality_level = models.CharField(max_length=20, choices=QualityLevel.choices)
    record_count = models.PositiveIntegerField()
    cost_cents = models.PositiveIntegerField()

    period_start = models.DateTimeField()
    period_end = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["api_key", "period_start", "period_end"]),
        ]


class SchemaTemplate(models.Model):
    """Reusable schema templates."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    domain = models.CharField(max_length=100)
    data_type = models.CharField(max_length=20, choices=DataType.choices, default=DataType.TABULAR)
    schema_def = models.JSONField()
    is_builtin = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["domain", "name"]

    def __str__(self):
        return f"{self.domain}/{self.name}"
