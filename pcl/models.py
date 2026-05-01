"""PCL — Process Characteristics Library models.

Three core models:
- Measure: a named, typed characteristic (extends SynaraEntity)
- Datapoint: an immutable observation (extends SynaraImmutableLog)
- MeasureTarget: aspirational value for a measure (extends SynaraEntity)
"""

from django.db import models

from syn.core.base_models import SynaraEntity, SynaraImmutableLog


class Measure(SynaraEntity):
    """A named, typed characteristic of a process, material, product, or resource.

    Raw measures (formula=None) store Datapoints.
    Calculated measures (formula set) resolve on read via [slug] references.
    """

    MEASURE_TYPES = [
        ("process", "Process"),
        ("material", "Material"),
        ("product", "Product"),
        ("resource", "Resource"),
    ]

    VALUE_TYPES = [
        ("continuous", "Continuous"),
        ("discrete", "Discrete"),
        ("proportion", "Proportion"),
        ("integer", "Integer"),
    ]

    name = models.CharField(max_length=255)
    slug = models.SlugField(max_length=100, db_index=True)
    definition = models.TextField(blank=True, default="")
    unit = models.CharField(max_length=50, blank=True, default="")
    measure_type = models.CharField(max_length=20, choices=MEASURE_TYPES)
    value_type = models.CharField(max_length=20, choices=VALUE_TYPES)

    range_min = models.FloatField(null=True, blank=True)
    range_max = models.FloatField(null=True, blank=True)

    # Calculated measures: formula references other slugs via [slug] syntax
    formula = models.TextField(null=True, blank=True)

    # Polymorphic parent (what this measure belongs to — VSM step, equipment, etc.)
    parent_type = models.CharField(max_length=100, blank=True, default="")
    parent_id = models.UUIDField(null=True, blank=True)

    # Cached aggregate (updated incrementally on each pcl.write())
    cached_value = models.FloatField(null=True, blank=True)
    cached_variance = models.FloatField(null=True, blank=True)
    cached_confidence = models.FloatField(null=True, blank=True)
    cached_n = models.IntegerField(default=0)
    cached_effective_n = models.FloatField(default=0.0)
    cached_at = models.DateTimeField(null=True, blank=True)

    # Recency decay (optional — for drifting processes)
    decay_enabled = models.BooleanField(default=False)
    decay_halflife_days = models.FloatField(null=True, blank=True)

    class Meta:
        db_table = "pcl_measure"
        ordering = ["name"]
        constraints = [
            models.UniqueConstraint(
                fields=["tenant_id", "slug"],
                condition=models.Q(is_deleted=False),
                name="unique_active_measure_slug_per_tenant",
            ),
        ]

    class SynaraMeta:
        event_domain = "pcl.measure"
        emit_events = ["created", "updated", "deleted"]

    def __str__(self):
        return f"{self.name} [{self.slug}] ({self.unit})"

    @property
    def is_calculated(self):
        return bool(self.formula)

    def to_dict(self):
        return {
            "id": str(self.id),
            "tenant_id": str(self.tenant_id) if self.tenant_id else None,
            "name": self.name,
            "slug": self.slug,
            "definition": self.definition,
            "unit": self.unit,
            "measure_type": self.measure_type,
            "value_type": self.value_type,
            "range_min": self.range_min,
            "range_max": self.range_max,
            "formula": self.formula,
            "is_calculated": self.is_calculated,
            "cached_value": self.cached_value,
            "cached_confidence": self.cached_confidence,
            "cached_n": self.cached_n,
            "decay_enabled": self.decay_enabled,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Datapoint(SynaraImmutableLog):
    """An immutable, timestamped observation of a raw measure.

    Write-once with hash-chain integrity (21 CFR Part 11).
    Confidence auto-computed from source_type x observation_count.
    """

    measure = models.ForeignKey(
        Measure,
        on_delete=models.CASCADE,
        related_name="datapoints",
    )
    value = models.FloatField()
    timestamp = models.DateTimeField(
        help_text="When the observation was made (may differ from created_at)",
    )
    source_type = models.CharField(max_length=30, db_index=True)
    source_ref_type = models.CharField(max_length=100, blank=True, default="")
    source_ref_id = models.UUIDField(null=True, blank=True)
    observation_count = models.IntegerField(default=1)
    notes = models.TextField(blank=True, default="")
    confidence = models.FloatField(default=0.0)

    class Meta:
        db_table = "pcl_datapoint"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["measure", "-created_at"]),
            models.Index(fields=["measure", "source_type"]),
        ]

    def save(self, *args, **kwargs):
        # Auto-compute confidence before the immutable save
        from pcl.confidence import compute_confidence

        if self._state.adding:
            self.confidence = compute_confidence(self.source_type, self.observation_count)
            if not self.timestamp:
                from django.utils import timezone

                self.timestamp = timezone.now()
            # Set event_name for the hash chain
            if not self.event_name:
                self.event_name = f"pcl.datapoint.created:{self.measure_id}"
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.measure.slug}={self.value} ({self.source_type}, conf={self.confidence:.2f})"

    def to_dict(self):
        return {
            "id": str(self.id),
            "measure_id": str(self.measure_id),
            "value": self.value,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "source_type": self.source_type,
            "source_ref_type": self.source_ref_type,
            "source_ref_id": str(self.source_ref_id) if self.source_ref_id else None,
            "observation_count": self.observation_count,
            "notes": self.notes,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class MeasureTarget(SynaraEntity):
    """Aspirational value for a measure — the working/future state layer."""

    measure = models.ForeignKey(
        Measure,
        on_delete=models.CASCADE,
        related_name="targets",
    )
    target_value = models.FloatField()
    target_date = models.DateField(null=True, blank=True)
    source = models.CharField(max_length=50, default="manual")
    source_ref_type = models.CharField(max_length=100, blank=True, default="")
    source_ref_id = models.UUIDField(null=True, blank=True)

    class Meta:
        db_table = "pcl_measure_target"
        ordering = ["-created_at"]

    class SynaraMeta:
        event_domain = "pcl.target"
        emit_events = ["created", "updated"]

    def __str__(self):
        return f"{self.measure.slug} target={self.target_value} ({self.source})"

    def to_dict(self):
        return {
            "id": str(self.id),
            "tenant_id": str(self.tenant_id) if self.tenant_id else None,
            "measure_id": str(self.measure_id),
            "target_value": self.target_value,
            "target_date": self.target_date.isoformat() if self.target_date else None,
            "source": self.source,
            "source_ref_type": self.source_ref_type,
            "source_ref_id": str(self.source_ref_id) if self.source_ref_id else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
