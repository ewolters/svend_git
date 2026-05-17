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
        """Calculate total lead time, process time, and PCE.

        When batch_size is present on a step, changeover is amortized:
        effective_co = changeover_time / batch_size (per-unit changeover cost).
        This gives a more realistic lead time for batch production.
        """
        total_ct = 0.0
        total_changeover = 0.0
        total_wait = 0.0

        wc_steps = {}
        standalone_cts = []

        for step in self.process_steps:
            ct = step.get("cycle_time", 0) or 0
            co = step.get("changeover_time", 0) or 0
            batch = step.get("batch_size") or 0
            # Amortize changeover across batch if batch_size is set
            if batch > 0 and co > 0:
                total_changeover += co / batch  # per-unit changeover cost
            else:
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

        # Auto-compute WIP days for inventory triangles between steps
        self._compute_inventory_wip()

        for inv in self.inventory:
            days = inv.get("days_of_supply", 0) or inv.get("computed_days", 0) or 0
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

        # Write metrics to PCL
        self._write_metrics_to_pcl()

    def _write_metrics_to_pcl(self):
        """Write VSM metrics to PCL — map-level and per-step.

        Slug convention:
          vsm/{vsm_id}/lead-time
          vsm/{vsm_id}/process-time
          vsm/{vsm_id}/pce
          vsm/{vsm_id}/takt-time
          vsm/{vsm_id}/step/{step_id}/cycle-time
          vsm/{vsm_id}/step/{step_id}/changeover-time
          etc.

        Non-fatal — logs errors, never raises.
        """
        try:
            from pcl.service import ensure_and_write
        except ImportError:
            return

        vsm_id = str(self.id)[:8]  # Short ID for readable slugs
        tenant_id = self.tenant_id
        actor = "vsm"

        # Map-level metrics
        map_metrics = {
            "lead-time": (self.total_lead_time, "days"),
            "process-time": (self.total_process_time, "sec"),
            "pce": (self.pce, "%"),
        }
        if self.takt_time:
            map_metrics["takt-time"] = (self.takt_time, "sec")

        for key, (value, unit) in map_metrics.items():
            if value is None:
                continue
            try:
                ensure_and_write(
                    slug=f"vsm/{vsm_id}/{key}",
                    value=float(value),
                    source_type="vsm",
                    actor=actor,
                    tenant_id=tenant_id,
                    unit=unit,
                    measure_type="process",
                    provenance="calculated",
                    notes=f"VSM: {self.name}",
                )
            except Exception:
                pass

        # Per-step metrics
        for step in self.process_steps or []:
            step_id = step.get("id", "")
            if not step_id:
                continue

            step_metrics = {}
            ct = step.get("cycle_time")
            if ct and ct > 0:
                step_metrics["cycle-time"] = (ct, "sec")
            co = step.get("changeover_time")
            if co and co > 0:
                step_metrics["changeover-time"] = (co, "sec")
            ut = step.get("uptime")
            if ut is not None:
                step_metrics["uptime"] = (ut, "%")
            ops = step.get("operators")
            if ops is not None:
                step_metrics["operators"] = (ops, "count")
            batch = step.get("batch_size")
            if batch and batch > 0:
                step_metrics["batch-size"] = (batch, "units")

            for key, (value, unit) in step_metrics.items():
                try:
                    ensure_and_write(
                        slug=f"vsm/{vsm_id}/step/{step_id}/{key}",
                        value=float(value),
                        source_type="vsm",
                        actor=actor,
                        tenant_id=tenant_id,
                        unit=unit,
                        measure_type="process",
                        provenance="observed",
                        notes=f"VSM step: {step.get('name', step_id)}",
                    )
                except Exception:
                    pass

    def _compute_inventory_wip(self):
        """Auto-compute days of supply for inventory triangles.

        For each inventory triangle, find the upstream and downstream steps
        by x-position. If the downstream step has a batch process or larger
        lot size, compute the expected WIP buildup.

        Only sets `computed_days` — never overwrites user-entered `days_of_supply`.
        No circular logic: reads step parameters only, never other inventory values.
        """
        import re

        steps = sorted(self.process_steps or [], key=lambda s: s.get("x", 0))
        if not steps:
            return

        # Parse VSM-level demand as fallback
        demand_per_day = 0
        demand_str = self.customer_demand or ""
        nums = re.findall(r"[\d.]+", demand_str)
        if nums:
            demand_per_day = float(nums[0])
            low = demand_str.lower()
            if "/month" in low or "month" in low:
                demand_per_day /= 22
            elif "/week" in low or "week" in low:
                demand_per_day /= 5
            elif "/year" in low or "annual" in low:
                demand_per_day /= 260

        for inv in self.inventory or []:
            # Skip if user has manually set days_of_supply
            if inv.get("days_of_supply"):
                continue

            inv_x = inv.get("x", 0)

            # Find upstream step (closest step to the left)
            upstream = None
            for s in reversed(steps):
                if s.get("x", 0) < inv_x:
                    upstream = s
                    break

            # Find downstream step (closest step to the right)
            downstream = None
            for s in steps:
                if s.get("x", 0) > inv_x:
                    downstream = s
                    break

            if not downstream:
                continue

            # Get demand at the downstream step
            step_demand = downstream.get("demand_rate")
            step_demand_unit = (downstream.get("demand_unit") or "").lower()
            if step_demand:
                d = float(step_demand)
                if "month" in step_demand_unit:
                    d /= 22
                elif "week" in step_demand_unit:
                    d /= 5
                elif "year" in step_demand_unit or "annual" in step_demand_unit:
                    d /= 260
            else:
                d = demand_per_day

            if d <= 0:
                continue

            # Compute WIP buildup from lot size mismatch
            ds_batch = downstream.get("batch_size") or 0
            ds_batch_process = downstream.get("batch_process", False)
            us_batch = (upstream.get("batch_size") or 1) if upstream else 1

            if ds_batch_process and ds_batch > 0:
                # Batch process downstream: WIP = average inventory = batch/2
                avg_wip = ds_batch / 2
                computed_days = round(avg_wip / d, 1)
                inv["computed_days"] = computed_days
                inv["computed_reason"] = (
                    f"Batch process downstream ({downstream.get('name', '?')}): "
                    f"avg WIP = {ds_batch}/2 = {avg_wip:.0f} units = {computed_days} days"
                )
            elif ds_batch > us_batch and ds_batch > 1:
                # Discretionary batch mismatch
                avg_wip = ds_batch / 2
                computed_days = round(avg_wip / d, 1)
                inv["computed_days"] = computed_days
                inv["computed_reason"] = (
                    f"Lot size mismatch: upstream lot={us_batch}, "
                    f"downstream lot={ds_batch}. "
                    f"Avg WIP = {avg_wip:.0f} units = {computed_days} days"
                )

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

    def to_manifest(self):
        steps = self.process_steps or []
        bursts = self.kaizen_bursts or []
        return {
            "container_id": str(self.id),
            "container_type": "ValueStreamMap",
            "title": self.name,
            "status": self.status,
            "artifacts": [
                {
                    "id": str(self.id),
                    "type": "ValueStreamMap",
                    "label": self.name,
                    "available_keys": [
                        "process_steps",
                        "inventory",
                        "information_flow",
                        "material_flow",
                        "kaizen_bursts",
                        "work_centers",
                        "total_lead_time",
                        "total_process_time",
                        "pce",
                        "metric_snapshots",
                    ],
                    "summary": {
                        "step_count": len(steps),
                        "burst_count": len(bursts),
                        "pce": self.pce,
                    },
                }
            ],
            "updated_at": self.updated_at.isoformat(),
        }
