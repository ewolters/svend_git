# dsw models — extracted from agents_api.models
# db_table overrides point at existing tables, no migration needed

import uuid

from django.conf import settings
from django.db import models

from core.encryption import EncryptedTextField


class DSWResult(models.Model):
    """Stored result from DSW pipeline run."""

    id = models.CharField(max_length=50, primary_key=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="dsw_results_ext",
    )
    result_type = models.CharField(max_length=50)
    data = EncryptedTextField()
    created_at = models.DateTimeField(auto_now_add=True)

    project = models.ForeignKey(
        "core.Project",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="dsw_results_ext",
    )

    title = models.CharField(max_length=255, blank=True)

    class Meta:
        db_table = "agents_api_dswresult"

        ordering = ["-created_at"]

    def get_summary(self):
        import json

        try:
            data = json.loads(self.data)
            summary_parts = []
            if self.title:
                summary_parts.append(self.title)
            if "analysis" in data:
                summary_parts.append(f"Analysis: {data['analysis']}")
            if "summary" in data:
                return data["summary"]
            if "findings" in data and isinstance(data["findings"], list):
                summary_parts.extend(data["findings"][:3])
            return " | ".join(summary_parts) if summary_parts else self.result_type
        except (json.JSONDecodeError, KeyError):
            return self.title or self.result_type


class SavedModel(models.Model):
    """User's saved ML models from DSW pipeline."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="dsw_saved_models",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="dsw_saved_models",
    )
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    model_type = models.CharField(max_length=100)
    model_path = models.CharField(max_length=500)
    dsw_result_id = models.CharField(max_length=50, blank=True)
    metrics = models.TextField(blank=True)
    feature_names = models.TextField(blank=True)
    target_name = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    project = models.ForeignKey(
        "core.Project",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="dsw_saved_models",
    )

    training_config = models.JSONField(default=dict, blank=True)
    data_lineage = models.JSONField(default=dict, blank=True)

    version = models.IntegerField(default=1)
    parent_model = models.ForeignKey(
        "self",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="dsw_retrained_versions",
    )

    class Meta:
        db_table = "agents_api_savedmodel"

        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} ({self.model_type})"
