"""ForgeRack session CRUD — save/load rack configurations."""

import json
import statistics

from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_http_methods

from accounts.permissions import require_auth
from agents_api.models import RackSession

FREE_SESSION_LIMIT = 3


def _rack_queryset(user):
    """Get rack sessions accessible to the user."""
    from django.db.models import Q

    qs = RackSession.objects.filter(Q(user=user))
    if hasattr(user, "tenant") and user.tenant:
        qs = RackSession.objects.filter(Q(user=user) | Q(tenant=user.tenant))
    return qs


def _can_edit(user, session):
    if session.user_id == user.id:
        return True
    if session.tenant and hasattr(user, "tenant") and session.tenant == user.tenant:
        return True
    return False


def _check_limit(user):
    """Free users limited to FREE_SESSION_LIMIT sessions."""
    sub = getattr(user, "subscription", None)
    if sub and getattr(sub, "tier", "free") != "free":
        return True  # Paid — no limit
    count = RackSession.objects.filter(user=user).count()
    return count < FREE_SESSION_LIMIT


@require_auth
@require_http_methods(["GET"])
def list_rack_sessions(request):
    sessions = _rack_queryset(request.user).values("id", "title", "session_type", "status", "updated_at")
    return JsonResponse(
        [
            {
                "id": str(s["id"]),
                "title": s["title"],
                "session_type": s["session_type"],
                "status": s["status"],
                "updated_at": s["updated_at"].isoformat() if s["updated_at"] else None,
            }
            for s in sessions
        ],
        safe=False,
    )


@require_auth
@require_http_methods(["POST"])
def create_rack_session(request):
    if not _check_limit(request.user):
        return JsonResponse(
            {"error": f"Free tier limited to {FREE_SESSION_LIMIT} saved sessions"},
            status=403,
        )

    data = json.loads(request.body)
    session = RackSession.objects.create(
        title=data.get("title", "Untitled Rack"),
        description=data.get("description", ""),
        session_type=data.get("session_type", RackSession.SessionType.SANDBOX),
        state=data.get("state", {}),
        user=request.user,
        tenant=getattr(request.user, "tenant", None),
    )
    return JsonResponse(session.to_dict(), status=201)


@require_auth
@require_http_methods(["GET"])
def get_rack_session(request, session_id):
    session = get_object_or_404(RackSession, id=session_id)
    if not _can_edit(request.user, session):
        return JsonResponse({"error": "Not authorized"}, status=403)
    return JsonResponse(session.to_dict())


@require_auth
@require_http_methods(["POST", "PUT"])
def update_rack_session(request, session_id):
    session = get_object_or_404(RackSession, id=session_id)
    if not _can_edit(request.user, session):
        return JsonResponse({"error": "Not authorized"}, status=403)

    data = json.loads(request.body)
    if "title" in data:
        session.title = data["title"]
    if "description" in data:
        session.description = data["description"]
    if "state" in data:
        session.state = data["state"]
    if "status" in data:
        session.status = data["status"]
    if "session_type" in data:
        session.session_type = data["session_type"]
    session.save()
    return JsonResponse(session.to_dict())


@require_auth
@require_http_methods(["DELETE"])
def delete_rack_session(request, session_id):
    session = get_object_or_404(RackSession, id=session_id)
    if not _can_edit(request.user, session):
        return JsonResponse({"error": "Not authorized"}, status=403)
    session.delete()
    return JsonResponse({"status": "deleted"})


# ═══════════════════════════════════════════════════════════════
# Rack Compute — forge package bridge for client-side units
# ═══════════════════════════════════════════════════════════════

_RACK_OPS = {}


def _rack_op(name):
    """Decorator to register a rack compute operation."""

    def decorator(fn):
        _RACK_OPS[name] = fn
        return fn

    return decorator


@_rack_op("mean")
def _op_mean(d):
    return {"value": statistics.mean(d["values"])}


@_rack_op("median")
def _op_median(d):
    return {"value": statistics.median(d["values"])}


@_rack_op("stdev")
def _op_stdev(d):
    return {"value": statistics.stdev(d["values"])}


@_rack_op("descriptive")
def _op_descriptive(d):
    v = d["values"]
    return {
        "mean": statistics.mean(v),
        "median": statistics.median(v),
        "stdev": statistics.stdev(v) if len(v) > 1 else 0,
        "min": min(v),
        "max": max(v),
        "n": len(v),
        "range": max(v) - min(v),
        "sum": sum(v),
    }


@_rack_op("ttest_2sample")
def _op_ttest(d):
    from forgestat.parametric.ttest import two_sample

    result = two_sample(d["a"], d["b"])
    alpha = d.get("alpha", 0.05)
    return {
        "t": result.statistic,
        "p": result.p_value,
        "df": result.df,
        "ci_lower": getattr(result, "ci_lower", None),
        "ci_upper": getattr(result, "ci_upper", None),
        "effect_size": getattr(result, "effect_size", None),
        "significant": result.p_value < alpha,
        "mean_a": statistics.mean(d["a"]),
        "mean_b": statistics.mean(d["b"]),
        "n_a": len(d["a"]),
        "n_b": len(d["b"]),
    }


@_rack_op("pearson")
def _op_pearson(d):
    from forgestat.parametric.correlation import correlation

    result = correlation({"x": d["x"], "y": d["y"]}, method="pearson")
    pair = result.pairs[0]
    return {
        "r": pair.r,
        "p": pair.p_value,
        "r_squared": pair.r_squared,
        "n": pair.n,
    }


@_rack_op("spearman")
def _op_spearman(d):
    from forgestat.parametric.correlation import correlation

    result = correlation({"x": d["x"], "y": d["y"]}, method="spearman")
    pair = result.pairs[0]
    return {"rho": pair.r, "p": pair.p_value, "n": pair.n}


@_rack_op("regression")
def _op_regression(d):
    x, y = d["x"], d["y"]
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    den = sum((x[i] - mx) ** 2 for i in range(n))
    slope = num / den if den else 0
    intercept = my - slope * mx
    return {"slope": slope, "intercept": intercept, "n": n}


@require_auth
@require_http_methods(["POST"])
def rack_compute(request):
    """Dispatch statistical operations to forge packages.

    POST /api/rack/compute/
    Body: {"op": "ttest_2sample", "data": {"a": [...], "b": [...]}}
    Returns: {"result": {...}} or {"error": "..."}
    """
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    op = body.get("op")
    data = body.get("data")

    if not op or not data:
        return JsonResponse({"error": "Missing op or data"}, status=400)

    if op not in _RACK_OPS:
        return JsonResponse(
            {"error": f"Unknown op: {op}", "available": list(_RACK_OPS.keys())},
            status=400,
        )

    # Coerce string values to floats in any list fields
    import logging

    logger = logging.getLogger("forgerack.compute")

    for key, val in data.items():
        if isinstance(val, list):
            coerced = []
            for v in val:
                try:
                    coerced.append(float(v))
                except (TypeError, ValueError):
                    pass  # skip non-numeric
            data[key] = coerced

    try:
        result = _RACK_OPS[op](data)
        # Sanitize NaN/Inf → null for valid JSON
        import math

        def _clean(v):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                logger.warning("NaN/Inf in %s result for op=%s", v, op)
                return 0.0
            return v

        result = {k: _clean(v) for k, v in result.items()}
        return JsonResponse({"result": result})
    except Exception as e:
        logger.exception("rack compute error: op=%s", op)
        return JsonResponse({"error": str(e)}, status=422)
