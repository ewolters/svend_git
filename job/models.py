"""Job — Canvas run records.

A Job is a silent, immutable record of computation.
Every canvas run creates one Job with N JobOutputs.

Job extends SynaraEntity (UUID, tenant, audit, soft delete).
JobOutput extends SynaraImmutableLog (hash-chained, write-once).
"""

from django.db import models

from syn.core.base_models import SynaraEntity, SynaraImmutableLog


class Job(SynaraEntity):
    """A record of a canvas run — silent, invisible, audit-complete."""

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("running", "Running"),
        ("completed", "Completed"),
        ("failed", "Failed"),
        ("cancelled", "Cancelled"),
    ]

    canvas_id = models.UUIDField(
        null=True,
        blank=True,
        db_index=True,
        help_text="Which canvas layout was used. Null for API/CLI runs.",
    )
    plugin_name = models.CharField(
        max_length=100,
        null=True,
        blank=True,
        db_index=True,
        help_text="Which plugin produced this job. Null for legacy/API runs.",
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default="pending",
        db_index=True,
    )
    inputs = models.JSONField(
        default=dict,
        help_text="Frozen snapshot of everything that went in.",
    )
    outputs_summary = models.JSONField(
        default=dict,
        help_text="Summary of what came out, for listing/search.",
    )
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    duration_ms = models.IntegerField(null=True, blank=True)
    actor = models.CharField(max_length=255, db_index=True)
    is_scratch = models.BooleanField(
        default=False,
        help_text="Scratch work stays scratch until explicitly promoted.",
    )

    class Meta:
        db_table = "job_job"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["actor", "-created_at"]),
            models.Index(fields=["canvas_id", "-created_at"]),
        ]

    class SynaraMeta:
        event_domain = "job"
        emit_events = ["created", "updated"]

    def __str__(self):
        return f"Job {self.id} [{self.status}]"

    def to_dict(self):
        return {
            "id": str(self.id),
            "tenant_id": str(self.tenant_id) if self.tenant_id else None,
            "canvas_id": str(self.canvas_id) if self.canvas_id else None,
            "plugin_name": self.plugin_name,
            "status": self.status,
            "inputs": self.inputs,
            "outputs_summary": self.outputs_summary,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "actor": self.actor,
            "is_scratch": self.is_scratch,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


PROVENANCE_TYPES = [
    ("observed", "Observed"),
    ("calculated", "Calculated"),
    ("simulated", "Simulated"),
    ("projected", "Projected"),
]

OUTPUT_TYPES = [
    ("metric", "Metric"),
    ("chart", "Chart"),
    ("table", "Table"),
    ("text", "Text"),
    ("dataset", "Dataset"),
]


class JobOutput(SynaraImmutableLog):
    """An individual addressable result from a Job run.

    Immutable, hash-chained (21 CFR Part 11).
    UUID-addressable — any canvas can reference a specific output.
    """

    job = models.ForeignKey(
        Job,
        on_delete=models.PROTECT,
        related_name="outputs",
    )
    output_key = models.CharField(
        max_length=100,
        help_text='Named result, e.g. "cpk", "histogram", "violations_list".',
    )
    output_type = models.CharField(max_length=20, choices=OUTPUT_TYPES)
    value_numeric = models.FloatField(
        null=True,
        blank=True,
        help_text="For PCL-writable metric values.",
    )
    value_json = models.JSONField(
        default=dict,
        help_text="Full output payload.",
    )
    provenance = models.CharField(
        max_length=20,
        choices=PROVENANCE_TYPES,
        default="calculated",
    )
    measure_slug = models.SlugField(
        max_length=100,
        null=True,
        blank=True,
        help_text="If this output wrote to PCL, which measure slug.",
    )

    class Meta:
        db_table = "job_output"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["job", "output_key"]),
        ]

    def __str__(self):
        return f"{self.output_key} ({self.output_type}) from Job {self.job_id}"

    def to_dict(self):
        return {
            "id": str(self.id),
            "job_id": str(self.job_id),
            "output_key": self.output_key,
            "output_type": self.output_type,
            "value_numeric": self.value_numeric,
            "value_json": self.value_json,
            "provenance": self.provenance,
            "measure_slug": self.measure_slug,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
