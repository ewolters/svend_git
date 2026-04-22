"""Configurable workflow system — replaces hardcoded MODE_MAP and frozen transitions.

WorkflowTemplate defines phases and transitions for a QMS flow.
SignalTypeRegistry replaces frozen Signal.SourceType TextChoices enum.
"""

import uuid

from django.db import models


class WorkflowTemplate(models.Model):
    """Defines the phases and transitions for a QMS flow."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="workflow_templates",
        help_text="null = system workflow",
    )
    name = models.CharField(max_length=200)
    is_system = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "qms_workflow_templates"
        ordering = ["name"]

    def __str__(self):
        return self.name


class WorkflowPhase(models.Model):
    """One phase in a workflow."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workflow = models.ForeignKey(
        WorkflowTemplate,
        on_delete=models.CASCADE,
        related_name="phases",
    )
    key = models.CharField(max_length=50)
    label = models.CharField(max_length=100)
    sort_order = models.IntegerField()
    color = models.CharField(max_length=20, blank=True)
    available_templates = models.ManyToManyField(
        "qms.ToolTemplate",
        blank=True,
        related_name="workflow_phases",
    )

    class Meta:
        db_table = "qms_workflow_phases"
        unique_together = [("workflow", "key")]
        ordering = ["sort_order"]

    def __str__(self):
        return f"{self.workflow.name} / {self.label}"


class WorkflowTransition(models.Model):
    """Allowed transition between phases."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workflow = models.ForeignKey(
        WorkflowTemplate,
        on_delete=models.CASCADE,
        related_name="transitions",
    )
    from_phase = models.ForeignKey(
        WorkflowPhase,
        on_delete=models.CASCADE,
        related_name="outgoing",
    )
    to_phase = models.ForeignKey(
        WorkflowPhase,
        on_delete=models.CASCADE,
        related_name="incoming",
    )
    label = models.CharField(max_length=200)
    gate_conditions = models.JSONField(
        default=dict,
        blank=True,
        help_text='Optional: {"requires_sections": ["root_cause"]}',
    )

    class Meta:
        db_table = "qms_workflow_transitions"
        unique_together = [("workflow", "from_phase", "to_phase")]

    def __str__(self):
        return f"{self.from_phase.label} -> {self.to_phase.label}: {self.label}"


class SignalTypeRegistry(models.Model):
    """Replaces frozen Signal.SourceType TextChoices enum.

    System types correspond to existing enum values.
    Tenants can add custom signal types.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="signal_types",
        help_text="null = system type",
    )
    key = models.CharField(max_length=50)
    label = models.CharField(max_length=200)
    default_severity = models.CharField(max_length=20, default="warning")
    is_system = models.BooleanField(default=False)
    icon = models.CharField(max_length=50, blank=True)
    auto_phase = models.ForeignKey(
        WorkflowPhase,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="auto_signal_types",
        help_text="Optional: auto-enter this phase on signal creation",
    )

    class Meta:
        db_table = "qms_signal_type_registry"
        constraints = [
            models.UniqueConstraint(
                fields=["tenant", "key"],
                name="unique_signal_type_per_tenant",
            ),
        ]
        ordering = ["key"]

    def __str__(self):
        return f"{self.key}: {self.label}"
