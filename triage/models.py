# triage models — extracted from agents_api.models
# db_table overrides point at existing tables, no migration needed

import uuid

from django.conf import settings
from django.db import models

from core.encryption import EncryptedTextField


class TriageResult(models.Model):
    """Stored result from Triage (data cleaning) pipeline."""

    id = models.CharField(max_length=50, primary_key=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="triage_triage_results",
    )
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="triage_results",
    )
    original_filename = models.CharField(max_length=255)
    cleaned_csv = EncryptedTextField()
    report_markdown = EncryptedTextField()
    summary_json = EncryptedTextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "agents_api_triageresult"
        managed = False
        ordering = ["-created_at"]


class AgentLog(models.Model):
    """Operational log for agent invocations."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="triage_agent_logs",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="triage_agent_logs",
    )
    agent = models.CharField(max_length=50)
    action = models.CharField(max_length=50)
    latency_ms = models.IntegerField(null=True)
    is_success = models.BooleanField(default=True, db_column="success")
    error_message = models.TextField(blank=True)
    metadata = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "agents_api_agentlog"
        managed = False
        ordering = ["-created_at"]
