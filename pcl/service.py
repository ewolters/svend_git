"""PCL service layer — the public API.

pcl.read(slug)              → best-available value (aggregate or formula)
pcl.read(slug, at=datetime) → historical value
pcl.read_with_meta(slug)    → value + confidence + variance + staleness
pcl.write(slug, ...)        → create Datapoint, update aggregate
pcl.set_target(slug, ...)   → create/update MeasureTarget
"""

import logging
from typing import Optional

from django.utils import timezone

logger = logging.getLogger("pcl.service")


def read(slug: str, tenant_id=None, at=None) -> Optional[float]:
    """Read the current value of a measure.

    For raw measures: returns the cached aggregate value.
    For calculated measures: evaluates formula by resolving component slugs.
    Returns None if no data available.
    """
    from pcl.models import Measure

    try:
        m = Measure.objects.get(slug=slug, tenant_id=tenant_id)
    except Measure.DoesNotExist:
        return None

    if m.is_calculated:
        return _resolve_formula(m, tenant_id, at)

    if at is not None:
        return _read_historical(m, at)

    return m.cached_value


def read_with_meta(slug: str, tenant_id=None) -> dict:
    """Read measure value with full metadata.

    Returns dict with: value, confidence, variance, n, effective_n,
    latest_timestamp, staleness_days, shift_detected, unit, is_calculated.
    """
    from pcl.models import Measure

    try:
        m = Measure.objects.get(slug=slug, tenant_id=tenant_id)
    except Measure.DoesNotExist:
        return {"value": None, "error": "measure_not_found"}

    if m.is_calculated:
        value = _resolve_formula(m, tenant_id)
        # Calculated measure metadata: weakest link confidence
        from pcl.formulas import extract_slugs

        component_meta = []
        for dep_slug in extract_slugs(m.formula):
            dep_meta = read_with_meta(dep_slug, tenant_id)
            component_meta.append(dep_meta)

        confidences = [cm.get("confidence", 0) for cm in component_meta if cm.get("value") is not None]
        timestamps = [cm.get("latest_timestamp") for cm in component_meta if cm.get("latest_timestamp")]

        return {
            "value": value,
            "confidence": min(confidences) if confidences else 0.0,
            "variance": None,
            "n": None,
            "effective_n": None,
            "latest_timestamp": max(timestamps) if timestamps else None,
            "staleness_days": None,
            "shift_detected": False,
            "unit": m.unit,
            "is_calculated": True,
            "components": component_meta,
        }

    now = timezone.now()
    staleness = (now - m.cached_at).days if m.cached_at else None

    return {
        "value": m.cached_value,
        "confidence": m.cached_confidence or 0.0,
        "variance": m.cached_variance,
        "n": m.cached_n,
        "effective_n": m.cached_effective_n,
        "latest_timestamp": m.cached_at.isoformat() if m.cached_at else None,
        "staleness_days": staleness,
        "shift_detected": False,  # TODO: store in cache after full recompute
        "unit": m.unit,
        "is_calculated": False,
    }


def write(
    measure_slug: str,
    value: float,
    source_type: str,
    actor: str,
    tenant_id=None,
    observation_count: int = 1,
    timestamp=None,
    source_ref_type: str = "",
    source_ref_id=None,
    notes: str = "",
    provenance: str = "observed",
    source_job_id=None,
) -> dict:
    """Write a new datapoint and update the measure's cached aggregate.

    Returns dict with the created datapoint info.
    """
    from pcl.aggregation import update_aggregate_incremental
    from pcl.models import Datapoint, Measure

    try:
        m = Measure.objects.get(slug=measure_slug, tenant_id=tenant_id)
    except Measure.DoesNotExist:
        raise ValueError(f"Measure not found: {measure_slug}")

    if m.is_calculated:
        raise ValueError(f"Cannot write to calculated measure: {measure_slug}")

    # Range alarm check
    alarm = None
    if m.range_min is not None and value < m.range_min:
        alarm = {"type": "below_min", "value": value, "range_min": m.range_min}
    elif m.range_max is not None and value > m.range_max:
        alarm = {"type": "above_max", "value": value, "range_max": m.range_max}

    ts = timestamp or timezone.now()

    dp = Datapoint.objects.create(
        measure=m,
        value=value,
        timestamp=ts,
        source_type=source_type,
        source_ref_type=source_ref_type,
        source_ref_id=source_ref_id,
        observation_count=observation_count,
        notes=notes,
        actor=actor,
        tenant_id=tenant_id,
        provenance=provenance,
        source_job_id=source_job_id,
    )

    # Only observed/calculated update the operational cache.
    # Simulated/projected are stored but don't contaminate aggregate.
    if provenance in ("observed", "calculated"):
        agg = update_aggregate_incremental(
            cached_value=m.cached_value,
            cached_variance=m.cached_variance,
            cached_confidence=m.cached_confidence,
            cached_n=m.cached_n,
            cached_effective_n=m.cached_effective_n,
            new_value=value,
            new_confidence=dp.confidence,
            decay_enabled=m.decay_enabled,
            decay_halflife_days=m.decay_halflife_days,
        )

        m.cached_value = agg["value"]
        m.cached_variance = agg["variance"]
        m.cached_confidence = agg["confidence"]
        m.cached_n = agg["n"]
        m.cached_effective_n = agg["effective_n"]
        m.cached_at = ts
        m.save(
            update_fields=[
                "cached_value",
                "cached_variance",
                "cached_confidence",
                "cached_n",
                "cached_effective_n",
                "cached_at",
                "updated_at",
            ]
        )

    result = dp.to_dict()
    if alarm:
        result["alarm"] = alarm
    return result


def ensure_and_write(
    slug: str,
    value: float,
    source_type: str,
    actor: str,
    tenant_id=None,
    name: str = "",
    unit: str = "",
    measure_type: str = "process",
    value_type: str = "continuous",
    provenance: str = "calculated",
    source_ref_type: str = "",
    source_ref_id=None,
    notes: str = "",
) -> dict:
    """Write to a measure, auto-creating it if it doesn't exist.

    This is the entry point for tools that produce metrics (workbench,
    VSM, FMEA, etc.) — they shouldn't need to pre-create measures.
    """
    from pcl.models import Measure

    m, created = Measure.objects.get_or_create(
        slug=slug,
        tenant_id=tenant_id,
        defaults={
            "name": name or slug.replace("-", " ").replace("/", " — ").title(),
            "unit": unit,
            "measure_type": measure_type,
            "value_type": value_type,
            "created_by": actor,
        },
    )
    if created:
        logger.info("PCL auto-created measure: %s (tenant=%s)", slug, tenant_id)

    return write(
        measure_slug=slug,
        value=value,
        source_type=source_type,
        actor=actor,
        tenant_id=tenant_id,
        provenance=provenance,
        source_ref_type=source_ref_type,
        source_ref_id=source_ref_id,
        notes=notes,
    )


def set_target(
    measure_slug: str,
    target_value: float,
    source: str = "manual",
    actor: str = "",
    tenant_id=None,
    target_date=None,
    source_ref_type: str = "",
    source_ref_id=None,
) -> dict:
    """Set or update a target for a measure."""
    from pcl.models import Measure, MeasureTarget

    try:
        m = Measure.objects.get(slug=measure_slug, tenant_id=tenant_id)
    except Measure.DoesNotExist:
        raise ValueError(f"Measure not found: {measure_slug}")

    target, created = MeasureTarget.objects.update_or_create(
        measure=m,
        tenant_id=tenant_id,
        source=source,
        defaults={
            "target_value": target_value,
            "target_date": target_date,
            "source_ref_type": source_ref_type,
            "source_ref_id": source_ref_id,
            "created_by": actor,
        },
    )

    return target.to_dict()


def _resolve_formula(measure, tenant_id, at=None) -> Optional[float]:
    """Resolve a calculated measure by evaluating its formula."""
    from pcl.formulas import evaluate_pcl_formula, extract_slugs

    slugs = extract_slugs(measure.formula)
    variables = {}
    for slug in slugs:
        val = read(slug, tenant_id=tenant_id, at=at)
        if val is None:
            return None  # can't resolve if any component is missing
        variables[slug] = val

    try:
        return evaluate_pcl_formula(measure.formula, variables)
    except (ValueError, ZeroDivisionError) as e:
        logger.warning("Formula evaluation failed for %s: %s", measure.slug, e)
        return None


def _read_historical(measure, at) -> Optional[float]:
    """Compute aggregate from datapoints at or before the given time."""
    from pcl.aggregation import compute_aggregate

    dps = measure.datapoints.filter(timestamp__lte=at).order_by("timestamp")
    if not dps.exists():
        return None

    now = at
    values = []
    confidences = []
    ages = []
    for dp in dps:
        values.append(dp.value)
        confidences.append(dp.confidence)
        age = (now - dp.timestamp).total_seconds() / 86400.0
        ages.append(age)

    result = compute_aggregate(
        values=values,
        confidences=confidences,
        timestamps_age_days=ages,
        decay_enabled=measure.decay_enabled,
        decay_halflife_days=measure.decay_halflife_days,
    )
    return result["value"]
