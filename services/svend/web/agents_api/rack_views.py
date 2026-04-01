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


def _has_variance(vals):
    """Check if a list has non-zero variance (not all identical)."""
    if len(vals) < 2:
        return False
    return any(v != vals[0] for v in vals[1:])


@_rack_op("pearson")
def _op_pearson(d):
    x, y = d["x"], d["y"]
    if not _has_variance(x) or not _has_variance(y):
        return {"r": 0.0, "p": 1.0, "r_squared": 0.0, "n": len(x), "warning": "zero variance"}
    from forgestat.parametric.correlation import correlation

    result = correlation({"x": x, "y": y}, method="pearson")
    pair = result.pairs[0]
    return {"r": pair.r, "p": pair.p_value, "r_squared": pair.r_squared, "n": pair.n}


@_rack_op("spearman")
def _op_spearman(d):
    x, y = d["x"], d["y"]
    if not _has_variance(x) or not _has_variance(y):
        return {"rho": 0.0, "p": 1.0, "n": len(x), "warning": "zero variance"}
    from forgestat.parametric.correlation import correlation

    result = correlation({"x": x, "y": y}, method="spearman")
    pair = result.pairs[0]
    return {"rho": pair.r, "p": pair.p_value, "n": pair.n}


@_rack_op("control_chart")
def _op_control_chart(d):
    from forgespc.capability import calculate_capability
    from forgespc.charts import individuals_moving_range_chart

    vals = d["values"]
    lsl = d.get("lsl")
    usl = d.get("usl")

    # Run chart
    r = individuals_moving_range_chart(vals)
    result = {
        "mean": r.limits.cl,
        "ucl": r.limits.ucl,
        "lcl": r.limits.lcl,
        "in_control": r.in_control,
        "out_of_control": [
            {"index": p.index, "value": p.value, "rule": getattr(p, "rule", 1)} for p in r.out_of_control
        ],
        "violations": [
            {
                "index": v.index,
                "value": getattr(v, "value", None),
                "rule": getattr(v, "rule_number", 0),
                "desc": getattr(v, "description", ""),
            }
            for v in r.run_violations
        ],
        "n": len(vals),
    }

    # Capability if specs provided
    if lsl is not None or usl is not None:
        cap = calculate_capability(vals, usl=usl, lsl=lsl)
        result["cpk"] = cap.cpk
        result["cp"] = cap.cp
        result["dpmo"] = cap.dpmo
        result["sigma_level"] = cap.sigma_level
        result["ppm"] = round(cap.dpmo) if cap.dpmo is not None else None

    return result


@_rack_op("histogram")
def _op_histogram(d):
    vals = d["values"]
    n_bins = d.get("bins", 15)
    n = len(vals)

    if n_bins == 0 or n_bins is None:
        # Sturges' rule
        import math as _math

        n_bins = max(1, int(_math.ceil(_math.log2(n) + 1)))

    mn, mx = min(vals), max(vals)
    rng = mx - mn or 1
    bin_width = rng / n_bins
    bins = [0] * n_bins
    edges = [mn + i * bin_width for i in range(n_bins + 1)]

    for v in vals:
        idx = int((v - mn) / bin_width)
        if idx >= n_bins:
            idx = n_bins - 1
        bins[idx] += 1

    mean = sum(vals) / n
    variance = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0
    std = variance**0.5

    # Skewness and kurtosis
    if n > 2 and std > 0:
        skew = (n / ((n - 1) * (n - 2))) * sum(((v - mean) / std) ** 3 for v in vals)
    else:
        skew = 0.0
    if n > 3 and std > 0:
        k4 = sum(((v - mean) / std) ** 4 for v in vals)
        kurt = ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * k4 - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    else:
        kurt = 0.0

    return {
        "bins": bins,
        "edges": edges,
        "n_bins": n_bins,
        "bin_width": bin_width,
        "mean": mean,
        "std": std,
        "skewness": skew,
        "kurtosis": kurt,
        "n": n,
        "min": mn,
        "max": mx,
    }


@_rack_op("gage_rr")
def _op_gage_rr(d):
    from forgespc.gage import gage_rr_crossed

    parts = d["parts"]
    operators = d["operators"]
    measurements = d["measurements"]

    # Validate balanced design before calling forgespc
    combos = set()
    for p, o in zip(parts, operators):
        combos.add((str(p), str(o)))
    unique_parts = set(str(p) for p in parts)
    unique_ops = set(str(o) for o in operators)
    expected = len(unique_parts) * len(unique_ops)
    actual_combos = len(set((str(p), str(o)) for p, o in zip(parts, operators)))
    if actual_combos < expected:
        missing = expected - actual_combos
        return {
            "error_type": "unbalanced_design",
            "message": (
                f"Unbalanced design: {len(unique_parts)} parts x {len(unique_ops)} operators = "
                f"{expected} combinations expected, but only {actual_combos} found ({missing} missing). "
                f"Each part must be measured by every operator for crossed Gage R&R."
            ),
            "n_parts": len(unique_parts),
            "n_operators": len(unique_ops),
            "parts": sorted(unique_parts),
            "operators": sorted(unique_ops),
        }

    result = gage_rr_crossed(
        parts=parts,
        operators=operators,
        measurements=measurements,
        tolerance=d.get("tolerance"),
    )
    return {
        "repeatability": result.var_repeatability,
        "reproducibility": result.var_reproducibility,
        "grr": result.var_grr,
        "part_variation": result.var_part,
        "total_variation": result.var_total,
        "grr_percent": result.grr_percent,
        "ndc": result.ndc,
        "assessment": result.assessment,
        "n_parts": result.n_parts,
        "n_operators": result.n_operators,
        "n_replicates": result.n_replicates,
        "n_total": result.n_total,
        "pct_contribution": result.pct_contribution,
        "pct_study_var": result.pct_study_var,
    }


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

    # Coerce lists that look numeric (first non-null element is a number)
    for key, val in data.items():
        if isinstance(val, list) and len(val) > 0:
            # Check if the first non-null value is numeric
            first = next((v for v in val if v is not None and str(v).strip()), None)
            try:
                float(first)
                is_numeric = True
            except (TypeError, ValueError):
                is_numeric = False
            if is_numeric:
                coerced = []
                for v in val:
                    try:
                        coerced.append(float(v))
                    except (TypeError, ValueError):
                        pass
                data[key] = coerced

    logger.info(
        "rack compute op=%s keys=%s lens=%s",
        op,
        list(data.keys()),
        {k: len(v) if isinstance(v, list) else type(v).__name__ for k, v in data.items()},
    )
    if op in ("pearson", "spearman") and "x" in data and "y" in data:
        logger.info("  x[:5]=%s y[:5]=%s", data["x"][:5], data["y"][:5])

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
