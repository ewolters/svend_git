"""PCL — Process Characteristics Library API views."""

import json
import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from accounts.permissions import require_auth
from pcl import service
from pcl.models import Measure
from qms_core.permissions import require_tenant

logger = logging.getLogger("pcl.views")


def _json_body(request):
    try:
        return json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return {}


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------


@csrf_exempt
@require_http_methods(["GET"])
@require_auth
def measure_list(request):
    """List all measures for the user's tenant."""
    tenant, err = require_tenant(request.user)
    if err:
        return err

    qs = Measure.objects.filter(tenant_id=tenant.id)

    # Optional filters
    measure_type = request.GET.get("measure_type")
    if measure_type:
        qs = qs.filter(measure_type=measure_type)

    value_type = request.GET.get("value_type")
    if value_type:
        qs = qs.filter(value_type=value_type)

    calculated = request.GET.get("calculated")
    if calculated == "true":
        qs = qs.exclude(formula__isnull=True).exclude(formula="")
    elif calculated == "false":
        qs = qs.filter(formula__isnull=True) | qs.filter(formula="")

    q = request.GET.get("q")
    if q:
        qs = qs.filter(name__icontains=q)

    return JsonResponse({"measures": [m.to_dict() for m in qs], "count": qs.count()})


@csrf_exempt
@require_http_methods(["GET"])
@require_auth
def measure_detail(request, measure_id):
    """Get a single measure with full metadata."""
    tenant, err = require_tenant(request.user)
    if err:
        return err

    try:
        m = Measure.objects.get(id=measure_id, tenant_id=tenant.id)
    except Measure.DoesNotExist:
        return JsonResponse({"error": "Measure not found"}, status=404)

    meta = service.read_with_meta(m.slug, tenant_id=tenant.id)
    result = m.to_dict()
    result["meta"] = meta

    # Include recent datapoints
    recent_dps = m.datapoints.order_by("-created_at")[:20]
    result["recent_datapoints"] = [dp.to_dict() for dp in recent_dps]

    # Include targets
    targets = m.targets.all()
    result["targets"] = [t.to_dict() for t in targets]

    return JsonResponse(result)


@csrf_exempt
@require_http_methods(["POST"])
@require_auth
def measure_create(request):
    """Create a new measure."""
    tenant, err = require_tenant(request.user)
    if err:
        return err

    data = _json_body(request)
    name = data.get("name", "").strip()
    slug = data.get("slug", "").strip()

    if not name or not slug:
        return JsonResponse({"error": "name and slug are required"}, status=400)

    if Measure.objects.filter(slug=slug, tenant_id=tenant.id).exists():
        return JsonResponse({"error": f"Measure with slug '{slug}' already exists"}, status=409)

    m = Measure.objects.create(
        tenant_id=tenant.id,
        name=name,
        slug=slug,
        definition=data.get("definition", ""),
        unit=data.get("unit", ""),
        measure_type=data.get("measure_type", "process"),
        value_type=data.get("value_type", "continuous"),
        range_min=data.get("range_min"),
        range_max=data.get("range_max"),
        formula=data.get("formula") or None,
        parent_type=data.get("parent_type", ""),
        parent_id=data.get("parent_id"),
        decay_enabled=data.get("decay_enabled", False),
        decay_halflife_days=data.get("decay_halflife_days"),
        created_by=request.user.email,
    )

    return JsonResponse(m.to_dict(), status=201)


# ---------------------------------------------------------------------------
# Datapoints (write)
# ---------------------------------------------------------------------------


@csrf_exempt
@require_http_methods(["POST"])
@require_auth
def datapoint_write(request):
    """Write a new datapoint via pcl.write()."""
    tenant, err = require_tenant(request.user)
    if err:
        return err

    data = _json_body(request)
    slug = data.get("measure_slug", "").strip()
    if not slug:
        return JsonResponse({"error": "measure_slug is required"}, status=400)

    value = data.get("value")
    if value is None:
        return JsonResponse({"error": "value is required"}, status=400)

    try:
        value = float(value)
    except (TypeError, ValueError):
        return JsonResponse({"error": "value must be a number"}, status=400)

    # Manual entry confirmation: if source_type is manual, require confirm_value
    source_type = data.get("source_type", "manual")
    if source_type == "manual":
        confirm = data.get("confirm_value")
        if confirm is None or float(confirm) != value:
            return JsonResponse(
                {"error": "Manual entry requires confirm_value matching value"},
                status=400,
            )

    try:
        result = service.write(
            measure_slug=slug,
            value=value,
            source_type=source_type,
            actor=request.user.email,
            tenant_id=tenant.id,
            observation_count=data.get("observation_count", 1),
            source_ref_type=data.get("source_ref_type", ""),
            source_ref_id=data.get("source_ref_id"),
            notes=data.get("notes", ""),
        )
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse(result, status=201)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


@csrf_exempt
@require_http_methods(["GET"])
@require_auth
def measure_read(request, slug):
    """Read current value of a measure (aggregate or formula)."""
    tenant, err = require_tenant(request.user)
    if err:
        return err

    meta = request.GET.get("meta", "").lower() == "true"

    if meta:
        result = service.read_with_meta(slug, tenant_id=tenant.id)
    else:
        value = service.read(slug, tenant_id=tenant.id)
        result = {"slug": slug, "value": value}

    return JsonResponse(result)


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


@csrf_exempt
@require_http_methods(["POST"])
@require_auth
def target_set(request):
    """Set or update a target for a measure."""
    tenant, err = require_tenant(request.user)
    if err:
        return err

    data = _json_body(request)
    slug = data.get("measure_slug", "").strip()
    if not slug:
        return JsonResponse({"error": "measure_slug is required"}, status=400)

    target_value = data.get("target_value")
    if target_value is None:
        return JsonResponse({"error": "target_value is required"}, status=400)

    try:
        result = service.set_target(
            measure_slug=slug,
            target_value=float(target_value),
            source=data.get("source", "manual"),
            actor=request.user.email,
            tenant_id=tenant.id,
            target_date=data.get("target_date"),
            source_ref_type=data.get("source_ref_type", ""),
            source_ref_id=data.get("source_ref_id"),
        )
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse(result, status=201)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@csrf_exempt
@require_http_methods(["GET"])
@require_auth
def measure_search(request):
    """Search measures by name or slug."""
    tenant, err = require_tenant(request.user)
    if err:
        return err

    q = request.GET.get("q", "").strip()
    if not q or len(q) < 2:
        return JsonResponse({"error": "q parameter must be at least 2 chars"}, status=400)

    qs = Measure.objects.filter(tenant_id=tenant.id)
    qs = qs.filter(name__icontains=q) | qs.filter(slug__icontains=q)
    qs = qs[:20]

    return JsonResponse({"results": [m.to_dict() for m in qs], "count": len(qs)})
