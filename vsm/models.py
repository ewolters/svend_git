# vsm models — extracted from agents_api.models

import uuid
from datetime import datetime, timezone

from django.conf import settings
from django.db import models


class ValueStreamMap(models.Model):
    """Value Stream Map for lean process analysis."""

    class Status(models.TextChoices):
        CURRENT = "current", "Current State"
        FUTURE = "future", "Future State"
        ARCHIVED = "archived", "Archived"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="vsm_maps",
    )

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="vsm_maps",
        null=True,
        blank=True,
    )

    project = models.ForeignKey(
        "core.Project",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="vsm_maps",
    )

    name = models.CharField(max_length=255, default="Untitled VSM")
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.CURRENT)
    fiscal_year = models.CharField(max_length=10, blank=True, default="")
    paired_with = models.OneToOneField(
        "self",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="paired_map",
    )
    product_family = models.CharField(max_length=255, blank=True)

    # Customer/supplier info
    customer_name = models.CharField(max_length=255, blank=True, default="Customer")
    customer_demand = models.CharField(max_length=100, blank=True)
    takt_time = models.FloatField(null=True, blank=True)
    supplier_name = models.CharField(max_length=255, blank=True, default="Supplier")
    supply_frequency = models.CharField(max_length=100, blank=True)
    customers = models.JSONField(default=list)
    suppliers = models.JSONField(default=list)

    # Process data (JSON)
    process_steps = models.JSONField(default=list)
    inventory = models.JSONField(default=list)
    information_flow = models.JSONField(default=list)
    material_flow = models.JSONField(default=list)

    # Calculated metrics
    total_lead_time = models.FloatField(null=True, blank=True)
    total_process_time = models.FloatField(null=True, blank=True)
    pce = models.FloatField(null=True, blank=True)

    # Improvement opportunities
    kaizen_bursts = models.JSONField(default=list)
    work_centers = models.JSONField(default=list)
    metric_snapshots = models.JSONField(default=list, blank=True)

    # Canvas state
    zoom = models.FloatField(default=1.0)
    pan_x = models.FloatField(default=0.0)
    pan_y = models.FloatField(default=0.0)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "value_stream_maps"
        managed = False
        ordering = ["-updated_at"]
        verbose_name = "Value Stream Map"
        verbose_name_plural = "Value Stream Maps"

    def __str__(self):
        return f"VSM: {self.name} ({self.status})"

    def calculate_metrics(self):
        """Calculate total lead time, process time, and PCE."""
        total_ct = 0.0
        total_changeover = 0.0
        total_wait = 0.0

        wc_steps = {}
        standalone_cts = []

        for step in self.process_steps:
            ct = step.get("cycle_time", 0) or 0
            co = step.get("changeover_time", 0) or 0
            total_changeover += co
            wc_id = step.get("work_center_id")
            if wc_id:
                wc_steps.setdefault(wc_id, []).append(ct)
            else:
                standalone_cts.append(ct)

        total_ct += sum(standalone_cts)

        for wc_id, cts in wc_steps.items():
            rate_sum = sum(1.0 / ct for ct in cts if ct > 0)
            if rate_sum > 0:
                total_ct += 1.0 / rate_sum

        for inv in self.inventory:
            days = inv.get("days_of_supply", 0) or 0
            total_wait += days

        self.total_process_time = total_ct
        total_active_seconds = total_ct + total_changeover
        self.total_lead_time = total_wait + (total_active_seconds / 86400)

        if self.total_lead_time > 0:
            self.pce = round((total_ct / 86400 / self.total_lead_time) * 100, 4)
        else:
            self.pce = 0

        snap = {
            "lead_time": round(self.total_lead_time or 0, 4),
            "process_time": round(self.total_process_time or 0, 1),
            "pce": round(self.pce or 0, 2),
            "takt_time": self.takt_time,
            "step_count": len(self.process_steps or []),
            "inventory_count": len(self.inventory or []),
        }
        snapshots = self.metric_snapshots or []
        last = snapshots[-1] if snapshots else None
        changed = (
            not last
            or last.get("lead_time") != snap["lead_time"]
            or last.get("process_time") != snap["process_time"]
            or last.get("pce") != snap["pce"]
            or last.get("takt_time") != snap["takt_time"]
        )
        if changed and (snap["step_count"] > 0 or snap["inventory_count"] > 0):
            snap["timestamp"] = datetime.now(timezone.utc).isoformat()
            snapshots.append(snap)
            if len(snapshots) > 100:
                snapshots = snapshots[-100:]
            self.metric_snapshots = snapshots

    def to_dict(self):
        return {
            "id": str(self.id),
            "project_id": str(self.project_id) if self.project_id else None,
            "name": self.name,
            "status": self.status,
            "fiscal_year": self.fiscal_year,
            "paired_with_id": str(self.paired_with_id) if self.paired_with_id else None,
            "product_family": self.product_family,
            "customer_name": self.customer_name,
            "customer_demand": self.customer_demand,
            "takt_time": self.takt_time,
            "supplier_name": self.supplier_name,
            "supply_frequency": self.supply_frequency,
            "customers": self.customers,
            "suppliers": self.suppliers,
            "process_steps": self.process_steps,
            "inventory": self.inventory,
            "information_flow": self.information_flow,
            "material_flow": self.material_flow,
            "total_lead_time": self.total_lead_time,
            "total_process_time": self.total_process_time,
            "pce": self.pce,
            "kaizen_bursts": self.kaizen_bursts,
            "work_centers": self.work_centers,
            "zoom": self.zoom,
            "pan_x": self.pan_x,
            "pan_y": self.pan_y,
            "metric_snapshots": self.metric_snapshots or [],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
