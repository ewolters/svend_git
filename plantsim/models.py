# plantsim models — extracted from agents_api.models
# db_table overrides point at existing tables, no migration needed

import uuid

from django.conf import settings
from django.db import models


class PlantSimulation(models.Model):
    """Plant/factory layout for discrete-event simulation."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="plantsim_simulations",
    )
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="plantsim_simulations",
    )
    project = models.ForeignKey(
        "core.Project",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="plantsim_simulations",
    )
    name = models.CharField(max_length=255, default="Untitled Plant")
    description = models.TextField(blank=True)

    stations = models.JSONField(default=list)
    connections = models.JSONField(default=list)
    sources = models.JSONField(default=list)
    sinks = models.JSONField(default=list)
    work_centers = models.JSONField(default=list)

    simulation_config = models.JSONField(default=dict)
    simulation_results = models.JSONField(default=list)
    metric_snapshots = models.JSONField(default=list)

    zoom = models.FloatField(default=1.0)
    pan_x = models.FloatField(default=0.0)
    pan_y = models.FloatField(default=0.0)

    source_vsm = models.ForeignKey(
        "agents_api.ValueStreamMap",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="plantsim_derived",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "plant_simulation"

        ordering = ["-updated_at"]

    def __str__(self):
        return f"Plant: {self.name} ({self.owner.username})"

    def to_dict(self):
        return {
            "id": str(self.id),
            "project_id": str(self.project_id) if self.project_id else None,
            "name": self.name,
            "description": self.description,
            "stations": self.stations,
            "connections": self.connections,
            "sources": self.sources,
            "sinks": self.sinks,
            "work_centers": self.work_centers,
            "simulation_config": self.simulation_config,
            "simulation_results": self.simulation_results,
            "metric_snapshots": self.metric_snapshots,
            "source_vsm_id": str(self.source_vsm_id) if self.source_vsm_id else None,
            "zoom": self.zoom,
            "pan_x": self.pan_x,
            "pan_y": self.pan_y,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def to_manifest(self):
        return {
            "container_id": str(self.id),
            "container_type": "PlantSimulation",
            "title": self.name,
            "artifacts": [
                {
                    "id": str(self.id),
                    "type": "PlantSimulation",
                    "label": self.name,
                    "available_keys": [
                        "stations",
                        "connections",
                        "sources",
                        "sinks",
                        "work_centers",
                        "simulation_config",
                        "simulation_results",
                        "metric_snapshots",
                    ],
                    "summary": {
                        "station_count": len(self.stations or []),
                        "connection_count": len(self.connections or []),
                        "result_count": len(self.simulation_results or []),
                    },
                }
            ],
            "updated_at": self.updated_at.isoformat(),
        }
