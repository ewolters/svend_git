"""Hoshin Kanri savings calculation engine.

Eight calculation methods for CI project savings tracking.
All follow the pattern: Savings = (Baseline - Actual) * Volume * CostPerUnit

Ported from Neptune-Hoshin (github.com/ewolters/neptune-hoshin) and adapted
for the Svend platform's VSM integration.
"""

import ast
import math
import operator


# ---------------------------------------------------------------------------
# Core calculation dispatch
# ---------------------------------------------------------------------------

CALCULATION_METHODS = {
    "waste_pct": {
        "name": "Waste Percentage",
        "category": "material",
        "description": "Savings from reducing waste/scrap percentage",
        "formula": "(Baseline% - Actual%) × Volume × CostPerUnit",
        "variables": ["baseline_pct", "actual_pct", "volume", "cost_per_unit"],
    },
    "time_reduction": {
        "name": "Time Reduction",
        "category": "labor",
        "description": "Savings from reducing cycle time or changeover",
        "formula": "(BaselineTime - ActualTime) / 3600 × Volume × LaborRate",
        "variables": ["baseline_seconds", "actual_seconds", "volume", "labor_rate_per_hour"],
    },
    "headcount": {
        "name": "Headcount Reduction",
        "category": "labor",
        "description": "Savings from labor efficiency / FTE reduction",
        "formula": "(BaselineHC - ActualHC) × CostPerEmployee",
        "variables": ["baseline_headcount", "actual_headcount", "cost_per_employee"],
    },
    "claims": {
        "name": "Claims / Quality Returns",
        "category": "quality",
        "description": "Savings from reducing warranty or quality claims",
        "formula": "(Baseline% - Actual%) × Sales",
        "variables": ["baseline_pct", "actual_pct", "sales_dollars"],
    },
    "layout": {
        "name": "Layout / Width Optimization",
        "category": "material",
        "description": "Savings from layout or width/footprint reduction",
        "formula": "WidthReduction × Volume × CostPerUnit",
        "variables": ["width_reduction", "volume", "cost_per_unit"],
    },
    "freight": {
        "name": "Freight / Logistics",
        "category": "other",
        "description": "Savings from logistics or shipping optimization",
        "formula": "(BaselineCost - ActualCost) per shipment × Shipments",
        "variables": ["baseline_cost", "actual_cost", "shipment_count"],
    },
    "energy": {
        "name": "Energy Reduction",
        "category": "other",
        "description": "Savings from energy or utility consumption reduction",
        "formula": "(BaselineUsage - ActualUsage) × CostPerUnit",
        "variables": ["baseline_usage", "actual_usage", "cost_per_unit"],
    },
    "direct": {
        "name": "Direct Cost Comparison",
        "category": "other",
        "description": "Direct before/after cost comparison",
        "formula": "BaselineCost - ActualCost",
        "variables": ["baseline_cost", "actual_cost"],
    },
    "custom": {
        "name": "Custom Formula",
        "category": "custom",
        "description": "User-defined formula using baseline, actual, volume, sales, rate, variance",
        "formula": "(user-defined)",
        "variables": ["baseline", "actual", "volume", "sales", "rate", "variance"],
    },
}


def calculate_savings(method, baseline, actual, volume=1.0, cost_per_unit=1.0, **kwargs):
    """Calculate savings using the specified method.

    Args:
        method: One of the CALCULATION_METHODS keys
        baseline: Baseline metric value (prior year / before improvement)
        actual: Actual metric value (current / after improvement)
        volume: Production volume or activity count
        cost_per_unit: Cost conversion factor

    Returns:
        dict with keys: method, savings, improvement_pct, details
    """
    calculators = {
        "waste_pct": _waste_pct,
        "time_reduction": _time_reduction,
        "headcount": _headcount,
        "claims": _claims,
        "layout": _layout,
        "freight": _freight,
        "energy": _energy,
        "direct": _direct,
        "custom": _custom,
    }
    calc_fn = calculators.get(method, _direct)
    return calc_fn(baseline, actual, volume, cost_per_unit, **kwargs)


def _safe_pct(delta, baseline):
    """Safe percentage calculation avoiding division by zero."""
    if not baseline:
        return 0.0
    return round(delta / baseline * 100, 1)


def _waste_pct(baseline, actual, volume, cost_per_unit, **kw):
    delta = baseline - actual
    savings = (delta / 100.0) * volume * cost_per_unit
    return {
        "method": "waste_pct",
        "savings": round(savings, 2),
        "improvement_pct": _safe_pct(delta, baseline),
        "details": {"waste_reduction_pct": round(delta, 3)},
    }


def _time_reduction(baseline, actual, volume, cost_per_unit, **kw):
    delta = baseline - actual  # seconds saved per unit
    savings = (delta / 3600.0) * volume * cost_per_unit  # hours * labor rate
    return {
        "method": "time_reduction",
        "savings": round(savings, 2),
        "improvement_pct": _safe_pct(delta, baseline),
        "details": {"time_saved_seconds": round(delta, 2)},
    }


def _headcount(baseline, actual, volume, cost_per_unit, **kw):
    delta = baseline - actual
    savings = delta * cost_per_unit  # cost_per_unit = cost per employee
    return {
        "method": "headcount",
        "savings": round(savings, 2),
        "improvement_pct": _safe_pct(delta, baseline),
        "details": {"headcount_reduction": delta},
    }


def _claims(baseline, actual, volume, cost_per_unit, **kw):
    delta = baseline - actual  # percentage points
    sales = kw.get("sales", volume * cost_per_unit)
    savings = (delta / 100.0) * sales
    return {
        "method": "claims",
        "savings": round(savings, 2),
        "improvement_pct": _safe_pct(delta, baseline),
        "details": {"claims_reduction_pct": round(delta, 3)},
    }


def _layout(baseline, actual, volume, cost_per_unit, **kw):
    delta = baseline - actual  # width or area reduction
    savings = delta * volume * cost_per_unit
    return {
        "method": "layout",
        "savings": round(savings, 2),
        "improvement_pct": _safe_pct(delta, baseline),
        "details": {"dimension_reduction": delta},
    }


def _freight(baseline, actual, volume, cost_per_unit, **kw):
    delta = baseline - actual  # cost per shipment reduction
    savings = delta * volume  # volume = shipment count
    return {
        "method": "freight",
        "savings": round(savings, 2),
        "improvement_pct": _safe_pct(delta, baseline),
        "details": {"cost_per_shipment_reduction": round(delta, 2)},
    }


def _energy(baseline, actual, volume, cost_per_unit, **kw):
    delta = baseline - actual  # usage units
    savings = delta * cost_per_unit
    return {
        "method": "energy",
        "savings": round(savings, 2),
        "improvement_pct": _safe_pct(delta, baseline),
        "details": {"usage_reduction": round(delta, 2)},
    }


def _direct(baseline, actual, volume=1.0, cost_per_unit=1.0, **kw):
    savings = baseline - actual
    return {
        "method": "direct",
        "savings": round(savings, 2),
        "improvement_pct": _safe_pct(savings, baseline),
        "details": {},
    }


def _custom(baseline, actual, volume=1.0, cost_per_unit=1.0, **kw):
    formula = kw.get("formula", "")
    sales = kw.get("sales", 0)
    if not formula:
        return _direct(baseline, actual, volume, cost_per_unit)

    variables = {
        "baseline": float(baseline),
        "actual": float(actual),
        "volume": float(volume),
        "rate": float(cost_per_unit),
        "sales": float(sales),
        "variance": float(baseline) - float(actual),
    }
    try:
        savings = evaluate_custom_formula(formula, variables)
    except (ValueError, TypeError, ZeroDivisionError) as e:
        return {
            "method": "custom",
            "savings": 0,
            "improvement_pct": 0,
            "details": {"formula": formula, "error": str(e)},
        }

    return {
        "method": "custom",
        "savings": round(float(savings), 2),
        "improvement_pct": _safe_pct(float(baseline) - float(actual), float(baseline)),
        "details": {"formula": formula, "variables_used": variables},
    }


# ---------------------------------------------------------------------------
# Safe custom formula evaluation (restricted AST)
# ---------------------------------------------------------------------------

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCS = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sqrt": math.sqrt,
    "pow": pow,
}


def evaluate_custom_formula(formula, variables):
    """Safely evaluate a formula string with restricted AST walking.

    Only allows: arithmetic operators, numeric literals, variable names
    from the variables dict, and a whitelist of safe functions.

    Args:
        formula: e.g. "(baseline - actual) * volume * rate"
        variables: dict of {name: float_value}

    Returns:
        float result

    Raises:
        ValueError: if formula contains unsafe operations
    """
    if len(formula) > 500:
        raise ValueError("Formula too long (max 500 chars)")

    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid formula syntax: {e}")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Non-numeric constant: {node.value!r}")
        elif isinstance(node, ast.Name):
            if node.id in variables:
                return float(variables[node.id])
            raise ValueError(f"Unknown variable: {node.id}")
        elif isinstance(node, ast.BinOp):
            op_fn = _SAFE_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Pow) and right > 10:
                raise ValueError("Exponent too large (max 10)")
            return op_fn(left, right)
        elif isinstance(node, ast.UnaryOp):
            op_fn = _SAFE_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            return op_fn(_eval(node.operand))
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls allowed")
            fn = _SAFE_FUNCS.get(node.func.id)
            if fn is None:
                raise ValueError(f"Function not allowed: {node.func.id}")
            args = [_eval(a) for a in node.args]
            return fn(*args)
        else:
            raise ValueError(f"Unsupported expression: {type(node).__name__}")

    return _eval(tree)


# ---------------------------------------------------------------------------
# TTM (Trailing Twelve-Month) baseline calculation
# ---------------------------------------------------------------------------

def calculate_ttm_baseline(baseline_data):
    """Calculate trailing 12-month average from baseline data.

    Args:
        baseline_data: list of {month, metric_value, volume, cost_per_unit}

    Returns:
        dict with ttm_metric, ttm_volume, month_count
    """
    values = [b["metric_value"] for b in baseline_data if b.get("metric_value") is not None]
    volumes = [b["volume"] for b in baseline_data if b.get("volume") is not None]

    return {
        "ttm_metric": sum(values) / len(values) if values else 0,
        "ttm_volume": sum(volumes) / len(volumes) if volumes else 0,
        "month_count": len(values),
    }


# ---------------------------------------------------------------------------
# Monthly savings aggregation
# ---------------------------------------------------------------------------

def aggregate_monthly_savings(monthly_actuals):
    """Aggregate monthly savings into YTD and trend data.

    Args:
        monthly_actuals: list of {month, baseline, actual, volume, cost_per_unit, savings}

    Returns:
        dict with ytd_savings, monthly_trend, months_reported
    """
    ytd = 0.0
    trend = []
    for m in sorted(monthly_actuals, key=lambda x: x.get("month", 0)):
        s = m.get("savings", 0) or 0
        ytd += s
        trend.append({
            "month": m.get("month"),
            "savings": round(s, 2),
            "cumulative": round(ytd, 2),
        })

    return {
        "ytd_savings": round(ytd, 2),
        "monthly_trend": trend,
        "months_reported": len([m for m in monthly_actuals if m.get("savings") is not None]),
    }


# ---------------------------------------------------------------------------
# VSM delta estimation (for auto-proposals)
# ---------------------------------------------------------------------------

def estimate_savings_from_vsm_delta(
    current_step,
    future_step,
    method="time_reduction",
    annual_volume=1.0,
    cost_per_unit=1.0,
):
    """Estimate savings from a VSM current vs future state step comparison.

    Used by the auto-proposal feature to generate savings estimates
    from kaizen burst improvements on future-state VSMs.

    Args:
        current_step: dict with cycle_time, changeover_time, uptime, operators, batch_size
        future_step: dict with same fields (improved values)
        method: calculation method to use
        annual_volume: annual production volume
        cost_per_unit: cost per unit or labor rate

    Returns:
        dict with deltas, estimated_savings, and suggested_method
    """
    ct_current = float(current_step.get("cycle_time") or 0)
    ct_future = float(future_step.get("cycle_time") or 0)
    co_current = float(current_step.get("changeover_time") or 0)
    co_future = float(future_step.get("changeover_time") or 0)
    uptime_current = float(current_step.get("uptime") or 100)
    uptime_future = float(future_step.get("uptime") or 100)
    ops_current = float(current_step.get("operators") or 0)
    ops_future = float(future_step.get("operators") or 0)

    ct_delta = ct_current - ct_future
    co_delta = co_current - co_future
    uptime_delta = uptime_future - uptime_current
    ops_delta = ops_current - ops_future

    # Auto-detect best method from deltas
    suggested = method
    if ops_delta > 0 and ct_delta <= 0:
        suggested = "headcount"
    elif ct_delta > 0 or co_delta > 0:
        suggested = "time_reduction"

    # Estimate savings using the dominant delta
    if suggested == "headcount":
        result = calculate_savings("headcount", ops_current, ops_future, 1, cost_per_unit)
    elif suggested == "time_reduction":
        total_time_current = ct_current + (co_current / max(1, float(current_step.get("batch_size") or 1)))
        total_time_future = ct_future + (co_future / max(1, float(future_step.get("batch_size") or 1)))
        result = calculate_savings("time_reduction", total_time_current, total_time_future, annual_volume, cost_per_unit)
    else:
        result = calculate_savings(method, ct_current, ct_future, annual_volume, cost_per_unit)

    return {
        "cycle_time_delta": round(ct_delta, 2),
        "changeover_delta": round(co_delta, 2),
        "uptime_delta": round(uptime_delta, 1),
        "operators_delta": round(ops_delta, 1),
        "estimated_annual_savings": result["savings"],
        "suggested_method": suggested,
        "improvement_pct": result["improvement_pct"],
    }
