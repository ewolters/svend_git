"""Flowchart models — connections, templates, instances.

Devices are plugins (registered in syn.plugins.registry).
Connections are typed edges between device ports.
Templates are pre-wired flowcharts (JSON blob).
Instances are user's working copies of templates.
"""

from django.conf import settings
from django.db import models

from syn.core.base_models import SynaraEntity


class FlowchartTemplate(SynaraEntity):
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, default="")
    definition = models.JSONField(default=dict)
    devices_used = models.JSONField(default=list, help_text="List of plugin names for query index.")
    is_shared = models.BooleanField(default=False)
    parent_template = models.ForeignKey(
        "self", null=True, blank=True, on_delete=models.SET_NULL, related_name="children"
    )

    class Meta:
        db_table = "flowchart_template"
        ordering = ["-created_at"]

    class SynaraMeta:
        event_domain = "flowchart"
        emit_events = ["created", "updated"]

    def __str__(self):
        return f"Template: {self.name}"

    def clone(self, new_name, actor_tenant_id=None):
        return FlowchartTemplate(
            name=new_name,
            description=self.description,
            definition=self.definition.copy(),
            devices_used=self.devices_used.copy(),
            parent_template=self,
            tenant_id=actor_tenant_id or self.tenant_id,
        )


class FlowchartInstance(SynaraEntity):
    name = models.CharField(max_length=255)
    template = models.ForeignKey(
        FlowchartTemplate, null=True, blank=True, on_delete=models.SET_NULL, related_name="instances"
    )
    definition = models.JSONField(default=dict)
    is_scratch = models.BooleanField(default=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="flowchart_instances",
        null=True,
        blank=True,
    )

    class Meta:
        db_table = "flowchart_instance"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "-created_at"]),
        ]

    class SynaraMeta:
        event_domain = "flowchart"
        emit_events = ["created", "updated"]

    def __str__(self):
        return f"Flowchart: {self.name}"
