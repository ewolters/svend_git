"""Register PCL + Job event schemas in EventSchemaRegistry."""

import uuid

from django.core.management.base import BaseCommand

# System-level sentinel tenant UUID for global/system schemas.
SYSTEM_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")

EVENT_SCHEMAS = [
    {
        "event_name": "pcl.datapoint.created",
        "version": "1",
        "schema": {
            "type": "object",
            "required": ["measure_id", "measure_slug", "value", "provenance", "source_type"],
            "properties": {
                "measure_id": {"type": "string", "format": "uuid"},
                "measure_slug": {"type": "string"},
                "value": {"type": "number"},
                "provenance": {"type": "string", "enum": ["observed", "calculated", "simulated", "projected"]},
                "source_type": {"type": "string"},
                "source_job_id": {"type": ["string", "null"], "format": "uuid"},
            },
        },
    },
    {
        "event_name": "pcl.measure.threshold_crossed",
        "version": "1",
        "schema": {
            "type": "object",
            "required": ["measure_id", "measure_slug", "value", "threshold_type", "threshold_value"],
            "properties": {
                "measure_id": {"type": "string", "format": "uuid"},
                "measure_slug": {"type": "string"},
                "value": {"type": "number"},
                "threshold_type": {"type": "string", "enum": ["below_min", "above_max"]},
                "threshold_value": {"type": "number"},
            },
        },
    },
    {
        "event_name": "job.execution.completed",
        "version": "1",
        "schema": {
            "type": "object",
            "required": ["job_id", "status", "output_count"],
            "properties": {
                "job_id": {"type": "string", "format": "uuid"},
                "canvas_id": {"type": ["string", "null"], "format": "uuid"},
                "status": {"type": "string"},
                "duration_ms": {"type": ["integer", "null"]},
                "output_count": {"type": "integer"},
            },
        },
    },
    {
        "event_name": "job.execution.failed",
        "version": "1",
        "schema": {
            "type": "object",
            "required": ["job_id", "error_summary"],
            "properties": {
                "job_id": {"type": "string", "format": "uuid"},
                "canvas_id": {"type": ["string", "null"], "format": "uuid"},
                "error_summary": {"type": "string"},
            },
        },
    },
]


class Command(BaseCommand):
    help = "Register PCL and Job event schemas in EventSchemaRegistry."

    def handle(self, *args, **options):
        from syn.events.models import EventSchemaRegistry

        for spec in EVENT_SCHEMAS:
            obj, created = EventSchemaRegistry.objects.update_or_create(
                event_name=spec["event_name"],
                defaults={
                    "version": spec["version"],
                    "schema": spec["schema"],
                    "is_active": True,
                    "tenant_id": SYSTEM_TENANT_ID,
                },
            )
            action = "Created" if created else "Updated"
            self.stdout.write(f"  {action}: {spec['event_name']} v{spec['version']}")

        self.stdout.write(self.style.SUCCESS(f"\nRegistered {len(EVENT_SCHEMAS)} event schemas."))
