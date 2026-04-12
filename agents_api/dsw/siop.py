"""
SIOP — Sales, Inventory & Operations Planning analyses for the DSW.

Promotes tactical standalone calculators into full DSW analyses with:
- Statistical rigor (confidence intervals, hypothesis integration)
- Bayesian evidence bridge (investigation support)
- Structured narratives + what-if explorers
- Plotly visualizations

Analyses
--------
1. abc_analysis         — ABC/XYZ Pareto classification + demand variability matrix
2. eoq                  — Economic Order Quantity with sensitivity
3. safety_stock         — Statistical safety stock & reorder point
4. inventory_turns      — Turns analysis with benchmarking
5. service_level        — Fill rate / cycle service level trade-off curves
6. demand_profile       — Demand variability profiling (CoV, trend, seasonality)
7. kanban_sizing        — Pull system card calculation with statistical justification
8. epei                 — Every Part Every Interval production scheduling
9. rop_simulation       — Monte Carlo reorder-point / (s,Q) policy simulation
10. mrp_netting         — Gross-to-net requirements explosion

CR: 7fa03dc8
"""

import math

import numpy as np

from .common import (
    COLOR_BAD,
    COLOR_GOLD,
    COLOR_GOOD,
    COLOR_INFO,
    COLOR_NEUTRAL,
    COLOR_REFERENCE,
    COLOR_WARNING,
    _narrative,
)

__all__ = ["run_siop"]


# ── Plotly helpers ────────────────────────────────────────────────────────


def _bar_trace(x, y, name="", color=COLOR_INFO, text=None):
    return {
        "type": "bar",
        "x": list(x),
        "y": list(y),
        "name": name,
        "marker": {"color": color},
        "text": list(text) if text is not None else None,
        "textposition": "outside" if text is not None else None,
    }


def _scatter_trace(x, y, name="", color=COLOR_INFO, mode="lines", dash=None):
    line = {"color": color}
    if dash:
        line["dash"] = dash
    return {
        "type": "scatter",
        "x": list(x),
        "y": list(y),
        "name": name,
        "mode": mode,
        "line": line,
    }


def _layout(title="", xaxis="", yaxis="", height=420):
    return {
        "title": {"text": title, "font": {"size": 14}},
        "xaxis": {"title": xaxis},
        "yaxis": {"title": yaxis},
        "height": height,
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "margin": {"l": 60, "r": 30, "t": 50, "b": 60},
    }


# ═══════════════════════════════════════════════════════════════════════════
# DISPATCHER
# ═══════════════════════════════════════════════════════════════════════════


def run_siop(df, analysis_id, config):
    """Dispatch SIOP analysis by analysis_id."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    dispatch = {
        "abc_analysis": _run_abc,
        "eoq": _run_eoq,
        "safety_stock": _run_safety_stock,
        "inventory_turns": _run_inventory_turns,
        "service_level": _run_service_level,
        "demand_profile": _run_demand_profile,
        "kanban_sizing": _run_kanban_sizing,
        "epei": _run_epei,
        "rop_simulation": _run_rop_simulation,
        "mrp_netting": _run_mrp_netting,
        "inventory_policy_wizard": _run_inventory_policy_wizard,
    }

    handler = dispatch.get(analysis_id)
    if handler:
        return handler(df, config)

    result["summary"] = f"Error: Unknown SIOP analysis: {analysis_id}"
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 1. ABC / XYZ ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════


def _run_abc(df, config):
    """
    ABC/XYZ Pareto classification.

    ABC: by annual value (unit_cost × annual_demand) or single value column.
    XYZ: by demand coefficient of variation.

    Config
    ------
    item_col : str       — SKU / part identifier column
    value_col : str      — annual value (or revenue) column (for ABC)
    demand_cols : list    — period demand columns (for XYZ CoV) [optional]
    unit_cost_col : str   — unit cost column (multiplied by demand if value_col absent)
    demand_col : str      — annual demand column (used with unit_cost_col)
    abc_thresholds : list — cumulative % breakpoints [default: [80, 95]]
    xyz_thresholds : list — CoV breakpoints [default: [0.5, 1.0]]
    """
    result = {"plots": [], "summary": "", "guide_observation": "", "statistics": {}}

    item_col = config.get("item_col", "") or config.get("var1", "")
    value_col = config.get("value_col", "") or config.get("var2", "")
    demand_cols = config.get("demand_cols", [])
    unit_cost_col = config.get("unit_cost_col", "")
    demand_col = config.get("demand_col", "")
    abc_thresh = config.get(
        "abc_thresholds",
        [
            _float(config, "a_threshold", 80),
            _float(config, "b_threshold", 95),
        ],
    )
    xyz_thresh = config.get("xyz_thresholds", [0.5, 1.0])

    if not item_col or item_col not in df.columns:
        result["summary"] = "Error: Select an item/SKU identifier column."
        return result

    work = df[[c for c in df.columns if c in df.columns]].copy()
    work = work.dropna(subset=[item_col])

    # ── Compute annual value ──
    if value_col and value_col in df.columns:
        work["_value"] = work[value_col].astype(float)
    elif unit_cost_col and demand_col and unit_cost_col in df.columns and demand_col in df.columns:
        work["_value"] = work[unit_cost_col].astype(float) * work[demand_col].astype(float)
    else:
        # Try to auto-detect: look for columns with "value", "revenue", "cost"
        candidates = [c for c in df.columns if any(k in c.lower() for k in ("value", "revenue", "sales", "spend"))]
        if candidates:
            work["_value"] = work[candidates[0]].astype(float)
            value_col = candidates[0]
        else:
            result["summary"] = "Error: Provide a value column, or unit_cost + demand columns for ABC classification."
            return result

    total_value = work["_value"].sum()
    if total_value <= 0:
        result["summary"] = "Error: Total value is zero or negative — check your data."
        return result

    # ── ABC Classification ──
    work = work.sort_values("_value", ascending=False).reset_index(drop=True)
    work["_cum_value"] = work["_value"].cumsum()
    work["_cum_pct"] = (work["_cum_value"] / total_value) * 100

    a_thresh = float(abc_thresh[0]) if len(abc_thresh) > 0 else 80
    b_thresh = float(abc_thresh[1]) if len(abc_thresh) > 1 else 95

    def _abc_class(cum_pct):
        if cum_pct <= a_thresh:
            return "A"
        elif cum_pct <= b_thresh:
            return "B"
        return "C"

    work["abc_class"] = work["_cum_pct"].apply(_abc_class)

    abc_counts = work["abc_class"].value_counts().to_dict()
    n_total = len(work)
    abc_value = work.groupby("abc_class")["_value"].sum().to_dict()

    # ── XYZ Classification (if demand columns provided) ──
    has_xyz = False
    if demand_cols and all(c in df.columns for c in demand_cols):
        has_xyz = True
        x_thresh = float(xyz_thresh[0]) if len(xyz_thresh) > 0 else 0.5
        y_thresh = float(xyz_thresh[1]) if len(xyz_thresh) > 1 else 1.0

        demand_data = work[demand_cols].astype(float)
        means = demand_data.mean(axis=1)
        stds = demand_data.std(axis=1, ddof=1)
        work["_cov"] = np.where(means > 0, stds / means, np.inf)

        def _xyz_class(cov):
            if cov <= x_thresh:
                return "X"
            elif cov <= y_thresh:
                return "Y"
            return "Z"

        work["xyz_class"] = work["_cov"].apply(_xyz_class)
        work["abc_xyz"] = work["abc_class"] + work["xyz_class"]

    # ── Statistics ──
    stats = {
        "total_items": n_total,
        "total_value": round(total_value, 2),
        "abc_counts": {k: int(v) for k, v in abc_counts.items()},
        "abc_pct_items": {k: round(int(v) / n_total * 100, 1) for k, v in abc_counts.items()},
        "abc_pct_value": {k: round(v / total_value * 100, 1) for k, v in abc_value.items()},
        "a_threshold": a_thresh,
        "b_threshold": b_thresh,
    }

    if has_xyz:
        xyz_counts = work["xyz_class"].value_counts().to_dict()
        stats["xyz_counts"] = {k: int(v) for k, v in xyz_counts.items()}
        matrix = work.groupby("abc_xyz").size().to_dict()
        stats["abc_xyz_matrix"] = {k: int(v) for k, v in matrix.items()}
        stats["mean_cov"] = round(float(work["_cov"].replace([np.inf], np.nan).mean()), 3)

    result["statistics"] = stats

    # ── Pareto Chart (ABC) ──
    items = work[item_col].astype(str).tolist()
    values = work["_value"].tolist()
    cum_pct = work["_cum_pct"].tolist()
    colors = [COLOR_GOOD if c == "A" else COLOR_WARNING if c == "B" else COLOR_BAD for c in work["abc_class"]]

    pareto_bar = {
        "type": "bar",
        "x": items[:50],
        "y": values[:50],
        "name": "Value",
        "marker": {"color": colors[:50]},
        "yaxis": "y",
    }
    pareto_line = {
        "type": "scatter",
        "x": items[:50],
        "y": cum_pct[:50],
        "name": "Cumulative %",
        "mode": "lines+markers",
        "line": {"color": COLOR_REFERENCE, "width": 2},
        "yaxis": "y2",
    }
    # Threshold lines
    shapes = [
        {
            "type": "line",
            "y0": a_thresh,
            "y1": a_thresh,
            "x0": 0,
            "x1": 1,
            "xref": "paper",
            "yref": "y2",
            "line": {"color": COLOR_GOOD, "dash": "dash", "width": 1},
        },
        {
            "type": "line",
            "y0": b_thresh,
            "y1": b_thresh,
            "x0": 0,
            "x1": 1,
            "xref": "paper",
            "yref": "y2",
            "line": {"color": COLOR_WARNING, "dash": "dash", "width": 1},
        },
    ]
    pareto_layout = _layout("ABC Pareto Analysis", "Item", "Value ($)")
    pareto_layout["yaxis2"] = {
        "title": "Cumulative %",
        "overlaying": "y",
        "side": "right",
        "range": [0, 105],
    }
    pareto_layout["shapes"] = shapes
    pareto_layout["showlegend"] = True
    result["plots"].append({"data": [pareto_bar, pareto_line], "layout": pareto_layout})

    # ── ABC Pie Chart ──
    abc_labels = sorted(abc_counts.keys())
    abc_colors = {"A": COLOR_GOOD, "B": COLOR_WARNING, "C": COLOR_BAD}
    pie_items = {
        "type": "pie",
        "labels": [f"Class {k} ({abc_counts.get(k, 0)} items)" for k in abc_labels],
        "values": [abc_counts.get(k, 0) for k in abc_labels],
        "marker": {"colors": [abc_colors.get(k, COLOR_NEUTRAL) for k in abc_labels]},
        "name": "Items",
        "domain": {"x": [0, 0.48]},
        "title": {"text": "Items by Class"},
    }
    pie_value = {
        "type": "pie",
        "labels": [f"Class {k} (${abc_value.get(k, 0):,.0f})" for k in abc_labels],
        "values": [abc_value.get(k, 0) for k in abc_labels],
        "marker": {"colors": [abc_colors.get(k, COLOR_NEUTRAL) for k in abc_labels]},
        "name": "Value",
        "domain": {"x": [0.52, 1]},
        "title": {"text": "Value by Class"},
    }
    pie_layout = _layout("ABC Distribution", height=350)
    pie_layout["showlegend"] = False
    result["plots"].append({"data": [pie_items, pie_value], "layout": pie_layout})

    # ── XYZ Matrix Heatmap (if applicable) ──
    if has_xyz:
        abc_cats = ["A", "B", "C"]
        xyz_cats = ["X", "Y", "Z"]
        z_matrix = []
        text_matrix = []
        for abc in abc_cats:
            row = []
            trow = []
            for xyz in xyz_cats:
                key = abc + xyz
                count = stats["abc_xyz_matrix"].get(key, 0)
                row.append(count)
                trow.append(f"{key}: {count}")
            z_matrix.append(row)
            text_matrix.append(trow)

        heatmap = {
            "type": "heatmap",
            "z": z_matrix,
            "x": xyz_cats,
            "y": abc_cats,
            "text": text_matrix,
            "texttemplate": "%{text}",
            "colorscale": [[0, "#1a1a2e"], [0.5, COLOR_INFO], [1, COLOR_GOOD]],
            "showscale": True,
            "colorbar": {"title": "Count"},
        }
        heat_layout = _layout("ABC-XYZ Matrix", "Demand Variability (XYZ)", "Value Classification (ABC)")
        result["plots"].append({"data": [heatmap], "layout": heat_layout})

    # ── Summary ──
    a_items = abc_counts.get("A", 0)
    a_value_pct = stats["abc_pct_value"].get("A", 0)
    c_items = abc_counts.get("C", 0)
    c_value_pct = stats["abc_pct_value"].get("C", 0)

    lines = [
        f"<<COLOR:accent>>ABC Analysis — {n_total} items, ${total_value:,.0f} total value<</COLOR>>",
        "",
        f"<<COLOR:good>>Class A: {a_items} items ({round(a_items / n_total * 100, 1)}%) → {a_value_pct}% of value<</COLOR>>",
        f"<<COLOR:warn>>Class B: {abc_counts.get('B', 0)} items → {stats['abc_pct_value'].get('B', 0)}% of value<</COLOR>>",
        f"<<COLOR:muted>>Class C: {c_items} items ({round(c_items / n_total * 100, 1)}%) → {c_value_pct}% of value<</COLOR>>",
    ]
    if has_xyz:
        ax = stats["abc_xyz_matrix"].get("AX", 0)
        az = stats["abc_xyz_matrix"].get("AZ", 0)
        lines.append("")
        lines.append(f"<<COLOR:accent>>XYZ overlay: mean CoV = {stats['mean_cov']:.3f}<</COLOR>>")
        lines.append(f"  AX (high value, stable demand): {ax} items — ideal for JIT/kanban")
        lines.append(f"  AZ (high value, volatile demand): {az} items — need safety stock review")

    result["summary"] = "\n".join(lines)
    result["guide_observation"] = (
        f"ABC analysis of {n_total} SKUs: Class A = {a_items} items covering {a_value_pct}% of value. "
        f"Class C = {c_items} items covering {c_value_pct}% of value. "
        + (
            f"ABC-XYZ matrix shows {stats['abc_xyz_matrix'].get('AZ', 0)} high-value volatile items needing attention. "
            if has_xyz
            else ""
        )
        + "Consider differentiated inventory policies by class."
    )
    result["narrative"] = _narrative(
        verdict=f"Pareto principle holds: {round(a_items / n_total * 100)}% of items drive {a_value_pct}% of value",
        body=(
            f"Class A items ({a_items}) should receive the most inventory management attention — "
            f"frequent reviews, tight safety stock, supplier partnerships. "
            f"Class C items ({c_items}) can use simplified replenishment (e.g., 2-bin, min-max). "
            + (
                f"The XYZ overlay reveals {stats['abc_xyz_matrix'].get('AZ', 0)} AZ items (high value + volatile demand) "
                f"that are prime candidates for demand sensing or buffer optimization."
                if has_xyz
                else ""
            )
        ),
        next_steps=[
            "Run Safety Stock analysis on Class A items with appropriate service levels",
            "Run EOQ analysis to optimize order quantities by class",
            (
                "Review AZ items for demand sensing or flexible supply contracts"
                if has_xyz
                else "Add period demand columns for XYZ classification"
            ),
            "Consider kanban for AX items with stable demand patterns",
        ],
    )

    # ── What-if explorer ──
    result["what_if"] = {
        "parameters": [
            {
                "name": "a_threshold",
                "label": "A-class threshold (%)",
                "default": a_thresh,
                "min": 50,
                "max": 95,
                "step": 5,
            },
            {
                "name": "b_threshold",
                "label": "B-class threshold (%)",
                "default": b_thresh,
                "min": 80,
                "max": 99,
                "step": 1,
            },
        ],
        "description": "Adjust Pareto thresholds to see how ABC classification changes.",
    }

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 2. EOQ — Economic Order Quantity
# ═══════════════════════════════════════════════════════════════════════════


def _run_eoq(df, config):
    """
    EOQ with total cost curve, sensitivity analysis, and quantity discount support.

    Config
    ------
    demand : float      — annual demand (D)
    order_cost : float  — cost per order (S)
    unit_cost : float   — unit purchase cost (C)
    holding_pct : float — holding cost as % of unit cost (default 25%)
    holding_cost : float — holding cost per unit/year (overrides pct if given)
    discounts : list    — [{min_qty, price}] quantity discount brackets [optional]
    demand_col : str    — column with demand data (alternative to scalar)
    """
    result = {"plots": [], "summary": "", "guide_observation": "", "statistics": {}}

    # Pull parameters — support both scalar config and data column
    D = _float(config, "demand")
    S = _float(config, "order_cost")
    C = _float(config, "unit_cost")
    h_pct = _float(config, "holding_pct", 25) / 100
    H_direct = _float(config, "holding_cost", None)

    # If demand comes from data column, compute annual total
    demand_col = config.get("demand_col", "")
    if demand_col and demand_col in df.columns and D is None:
        periods = len(df)
        D = float(df[demand_col].sum())
        if periods > 0:
            # Assume data represents periods — annualize if needed
            periods_per_year = _float(config, "periods_per_year", 12)
            if periods < periods_per_year:
                D = D * (periods_per_year / periods)

    if D is None or D <= 0:
        result["summary"] = "Error: Annual demand (D) must be a positive number."
        return result
    if S is None or S <= 0:
        result["summary"] = "Error: Order cost (S) must be a positive number."
        return result
    if C is None or C <= 0:
        result["summary"] = "Error: Unit cost (C) must be a positive number."
        return result

    H = H_direct if H_direct is not None else C * h_pct

    # ── EOQ calculation ──
    eoq = math.sqrt(2 * D * S / H)
    orders_per_year = D / eoq
    annual_order_cost = orders_per_year * S
    annual_holding_cost = (eoq / 2) * H
    total_cost = annual_order_cost + annual_holding_cost
    cycle_time_days = 365 / orders_per_year

    stats = {
        "eoq": round(eoq, 1),
        "orders_per_year": round(orders_per_year, 2),
        "annual_order_cost": round(annual_order_cost, 2),
        "annual_holding_cost": round(annual_holding_cost, 2),
        "total_inventory_cost": round(total_cost, 2),
        "cycle_time_days": round(cycle_time_days, 1),
        "annual_demand": round(D, 0),
        "order_cost": round(S, 2),
        "unit_cost": round(C, 2),
        "holding_cost_per_unit": round(H, 2),
    }

    # ── Quantity discount evaluation ──
    discounts = config.get("discounts", [])
    discount_analysis = []
    if discounts:
        for bracket in sorted(discounts, key=lambda b: b.get("min_qty", 0)):
            q = float(bracket.get("min_qty", 0))
            p = float(bracket.get("price", C))
            h_disc = p * h_pct
            eoq_disc = math.sqrt(2 * D * S / h_disc)
            # Order qty is max(EOQ at this price, min_qty for bracket)
            order_q = max(eoq_disc, q) if q > 0 else eoq_disc
            oc = (D / order_q) * S
            hc = (order_q / 2) * h_disc
            pc = D * p
            tc = oc + hc + pc
            discount_analysis.append(
                {
                    "min_qty": q,
                    "price": p,
                    "order_qty": round(order_q, 1),
                    "total_cost": round(tc, 2),
                }
            )
        stats["discount_analysis"] = discount_analysis

    result["statistics"] = stats

    # ── Cost curve plot ──
    q_range = np.linspace(max(1, eoq * 0.1), eoq * 3, 200)
    oc_curve = (D / q_range) * S
    hc_curve = (q_range / 2) * H
    tc_curve = oc_curve + hc_curve

    result["plots"].append(
        {
            "data": [
                _scatter_trace(q_range, oc_curve, "Ordering Cost", COLOR_WARNING, dash="dash"),
                _scatter_trace(q_range, hc_curve, "Holding Cost", COLOR_INFO, dash="dash"),
                _scatter_trace(q_range, tc_curve, "Total Cost", COLOR_GOOD),
                {
                    "type": "scatter",
                    "x": [eoq],
                    "y": [total_cost],
                    "mode": "markers+text",
                    "name": f"EOQ = {eoq:.0f}",
                    "marker": {"color": COLOR_GOOD, "size": 12, "symbol": "diamond"},
                    "text": [f"EOQ = {eoq:.0f}"],
                    "textposition": "top center",
                },
            ],
            "layout": _layout("EOQ Cost Curve", "Order Quantity (Q)", "Annual Cost ($)"),
        }
    )

    # ── Sensitivity analysis ──
    sens_params = ["demand", "order_cost", "holding_cost"]
    sens_range = np.linspace(0.5, 2.0, 11)  # ±50%
    sens_data = []
    for param in sens_params:
        eoqs = []
        for mult in sens_range:
            d, s, h = D, S, H
            if param == "demand":
                d *= mult
            elif param == "order_cost":
                s *= mult
            else:
                h *= mult
            eoqs.append(math.sqrt(2 * d * s / h))
        sens_data.append(
            _scatter_trace(
                (sens_range * 100).tolist(),
                eoqs,
                param.replace("_", " ").title(),
                [COLOR_GOOD, COLOR_WARNING, COLOR_INFO][len(sens_data)],
            )
        )

    sens_layout = _layout("EOQ Sensitivity", "Parameter % of Baseline", "EOQ")
    sens_layout["shapes"] = [
        {
            "type": "line",
            "x0": 100,
            "x1": 100,
            "y0": 0,
            "y1": 1,
            "yref": "paper",
            "line": {"color": COLOR_NEUTRAL, "dash": "dot"},
        },
    ]
    result["plots"].append({"data": sens_data, "layout": sens_layout})

    # ── Summary ──
    result["summary"] = "\n".join(
        [
            "<<COLOR:accent>>EOQ Analysis — Optimal Order Quantity<</COLOR>>",
            "",
            f"<<COLOR:good>>EOQ = {eoq:,.0f} units<</COLOR>>",
            f"  Orders per year: {orders_per_year:.1f}  |  Cycle time: {cycle_time_days:.0f} days",
            "",
            f"  Annual ordering cost:  ${annual_order_cost:,.2f}",
            f"  Annual holding cost:   ${annual_holding_cost:,.2f}",
            f"  <<COLOR:accent>>Total inventory cost:   ${total_cost:,.2f}<</COLOR>>",
            "",
            f"  Demand = {D:,.0f}/yr  |  Order cost = ${S:,.2f}  |  Holding = ${H:,.2f}/unit/yr",
        ]
    )

    if discount_analysis:
        best = min(discount_analysis, key=lambda d: d["total_cost"])
        result["summary"] += (
            f"\n\n<<COLOR:warn>>Quantity discount: best total cost at Q = {best['order_qty']:,.0f} (${best['total_cost']:,.2f} incl. purchase)<</COLOR>>"
        )

    result["guide_observation"] = (
        f"EOQ = {eoq:,.0f} units. At {orders_per_year:.1f} orders/year, "
        f"total inventory cost = ${total_cost:,.2f}. "
        f"EOQ is robust to ±20% parameter changes (square root dampening)."
    )
    result["narrative"] = _narrative(
        verdict=f"Optimal order quantity is {eoq:,.0f} units ({orders_per_year:.1f} orders/year)",
        body=(
            f"The classic EOQ balances ordering cost (${S:,.2f}/order) against holding cost "
            f"(${H:,.2f}/unit/year = {h_pct * 100:.0f}% of ${C:,.2f}). "
            f"The square-root formula makes EOQ robust — a 50% error in demand only shifts EOQ by ~22%. "
            f"Cycle time of {cycle_time_days:.0f} days between orders."
        ),
        next_steps=[
            "Run Safety Stock analysis to set reorder point",
            "Run ABC analysis to prioritize which items get EOQ vs. simpler policies",
            "Consider quantity discounts if supplier offers price breaks",
            "Run ROP Simulation to validate (s,Q) policy under demand uncertainty",
        ],
    )

    result["what_if"] = {
        "parameters": [
            {
                "name": "demand",
                "label": "Annual Demand",
                "default": D,
                "min": 1,
                "max": D * 5,
                "step": max(1, D // 100),
            },
            {
                "name": "order_cost",
                "label": "Order Cost ($)",
                "default": S,
                "min": 1,
                "max": S * 10,
                "step": 1,
            },
            {
                "name": "holding_pct",
                "label": "Holding Cost (%)",
                "default": h_pct * 100,
                "min": 5,
                "max": 50,
                "step": 1,
            },
        ],
        "description": "Explore how EOQ changes with demand, ordering cost, and holding cost.",
    }

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 3. SAFETY STOCK
# ═══════════════════════════════════════════════════════════════════════════


def _run_safety_stock(df, config):
    """
    Statistical safety stock and reorder point.

    Supports both scalar inputs and data-driven computation.

    Config
    ------
    demand_mean : float       — mean demand per period
    demand_std : float        — demand standard deviation per period
    lead_time : float         — mean lead time (periods)
    lead_time_std : float     — lead time standard deviation [default: 0]
    service_level : float     — target service level % [default: 95]
    demand_col : str          — column with demand observations (overrides scalars)
    lead_time_col : str       — column with LT observations (overrides scalars)
    """
    from scipy.stats import norm

    result = {"plots": [], "summary": "", "guide_observation": "", "statistics": {}}

    # Data-driven or scalar
    demand_col = config.get("demand_col", "")
    lt_col = config.get("lead_time_col", "")

    if demand_col and demand_col in df.columns:
        d_data = df[demand_col].dropna().astype(float)
        d_mean = float(d_data.mean())
        d_std = float(d_data.std(ddof=1))
    else:
        d_mean = _float(config, "demand_mean")
        d_std = _float(config, "demand_std")

    if lt_col and lt_col in df.columns:
        lt_data = df[lt_col].dropna().astype(float)
        lt_mean = float(lt_data.mean())
        lt_std = float(lt_data.std(ddof=1))
    else:
        lt_mean = _float(config, "lead_time")
        lt_std = _float(config, "lead_time_std", 0)

    sl = _float(config, "service_level", 95) / 100

    if d_mean is None or d_mean <= 0:
        result["summary"] = "Error: Mean demand must be positive."
        return result
    if d_std is None or d_std < 0:
        result["summary"] = "Error: Demand standard deviation must be non-negative."
        return result
    if lt_mean is None or lt_mean <= 0:
        result["summary"] = "Error: Lead time must be positive."
        return result

    z = norm.ppf(sl)

    # Combined demand during lead time variability
    sigma_dlt = math.sqrt(lt_mean * d_std**2 + d_mean**2 * lt_std**2)
    ss = z * sigma_dlt
    rop = d_mean * lt_mean + ss
    avg_demand_lt = d_mean * lt_mean

    stats = {
        "safety_stock": round(ss, 1),
        "reorder_point": round(rop, 1),
        "avg_demand_during_lt": round(avg_demand_lt, 1),
        "sigma_demand_lt": round(sigma_dlt, 2),
        "z_score": round(z, 3),
        "service_level": round(sl * 100, 1),
        "demand_mean": round(d_mean, 2),
        "demand_std": round(d_std, 3),
        "lead_time_mean": round(lt_mean, 2),
        "lead_time_std": round(lt_std, 3),
    }
    result["statistics"] = stats

    # ── Demand distribution plot with ROP ──
    x_range = np.linspace(max(0, avg_demand_lt - 4 * sigma_dlt), avg_demand_lt + 4 * sigma_dlt, 300)
    pdf_vals = norm.pdf(x_range, avg_demand_lt, sigma_dlt) if sigma_dlt > 0 else np.zeros_like(x_range)

    fill_x = x_range[x_range <= rop]
    fill_y = norm.pdf(fill_x, avg_demand_lt, sigma_dlt) if sigma_dlt > 0 else np.zeros_like(fill_x)

    result["plots"].append(
        {
            "data": [
                _scatter_trace(x_range, pdf_vals, "Demand During LT", COLOR_INFO),
                {
                    "type": "scatter",
                    "x": list(fill_x),
                    "y": list(fill_y),
                    "fill": "tozeroy",
                    "fillcolor": "rgba(74, 159, 110, 0.3)",
                    "line": {"color": "rgba(0,0,0,0)"},
                    "name": f"Covered ({sl * 100:.0f}%)",
                    "showlegend": True,
                },
            ],
            "layout": {
                **_layout("Demand During Lead Time", "Demand (units)", "Probability Density"),
                "shapes": [
                    {
                        "type": "line",
                        "x0": rop,
                        "x1": rop,
                        "y0": 0,
                        "y1": 1,
                        "yref": "paper",
                        "line": {"color": COLOR_GOOD, "width": 2, "dash": "dash"},
                    },
                    {
                        "type": "line",
                        "x0": avg_demand_lt,
                        "x1": avg_demand_lt,
                        "y0": 0,
                        "y1": 1,
                        "yref": "paper",
                        "line": {"color": COLOR_NEUTRAL, "width": 1, "dash": "dot"},
                    },
                ],
                "annotations": [
                    {
                        "x": rop,
                        "y": 1,
                        "yref": "paper",
                        "text": f"ROP = {rop:.0f}",
                        "showarrow": False,
                        "yanchor": "bottom",
                    },
                ],
            },
        }
    )

    # ── Service level trade-off curve ──
    sl_range = np.linspace(0.80, 0.999, 50)
    z_range = norm.ppf(sl_range)
    ss_range = z_range * sigma_dlt

    result["plots"].append(
        {
            "data": [
                _scatter_trace(sl_range * 100, ss_range, "Safety Stock", COLOR_INFO),
                {
                    "type": "scatter",
                    "x": [sl * 100],
                    "y": [ss],
                    "mode": "markers",
                    "marker": {"color": COLOR_GOOD, "size": 10, "symbol": "diamond"},
                    "name": f"Current ({sl * 100:.0f}%)",
                },
            ],
            "layout": _layout(
                "Service Level vs Safety Stock",
                "Service Level (%)",
                "Safety Stock (units)",
            ),
        }
    )

    result["summary"] = "\n".join(
        [
            f"<<COLOR:accent>>Safety Stock Analysis — {sl * 100:.0f}% Service Level<</COLOR>>",
            "",
            f"<<COLOR:good>>Safety Stock = {ss:,.0f} units<</COLOR>>",
            f"<<COLOR:good>>Reorder Point = {rop:,.0f} units<</COLOR>>",
            "",
            f"  Avg demand during LT: {avg_demand_lt:,.0f} units",
            f"  σ (demand during LT): {sigma_dlt:,.1f} units",
            f"  Z-score ({sl * 100:.0f}%): {z:.3f}",
            "",
            f"  Demand: {d_mean:,.1f} ± {d_std:,.2f} per period",
            f"  Lead time: {lt_mean:,.1f} ± {lt_std:,.2f} periods",
        ]
    )

    result["guide_observation"] = (
        f"Safety stock = {ss:,.0f} units at {sl * 100:.0f}% service level. "
        f"Reorder point = {rop:,.0f} units. "
        f"Lead time variability {'is a major driver' if lt_std > 0.1 * lt_mean else 'is minimal'} of safety stock."
    )
    result["narrative"] = _narrative(
        verdict=f"Safety stock of {ss:,.0f} units provides {sl * 100:.0f}% service level",
        body=(
            f"The reorder point ({rop:,.0f} units) = expected demand during lead time ({avg_demand_lt:,.0f}) + "
            f"safety stock ({ss:,.0f}). "
            f"{'Lead time variability (σ_LT = ' + f'{lt_std:.2f}) is a significant contributor — reducing it would lower SS substantially.' if lt_std > 0 else 'Lead time is deterministic — all safety stock covers demand variability.'}"
        ),
        next_steps=[
            "Run ROP Simulation to validate stockout probability under real demand patterns",
            "Run EOQ to pair optimal order quantity with this reorder point",
            "Run Service Level analysis to explore cost-service trade-offs",
            (
                f"Consider reducing lead time variability (current σ = {lt_std:.2f}) for biggest SS reduction"
                if lt_std > 0
                else "Consider adding lead time variability data for more realistic SS"
            ),
        ],
    )

    result["what_if"] = {
        "parameters": [
            {
                "name": "service_level",
                "label": "Service Level (%)",
                "default": sl * 100,
                "min": 80,
                "max": 99.9,
                "step": 0.5,
            },
            {
                "name": "lead_time",
                "label": "Lead Time (periods)",
                "default": lt_mean,
                "min": 0.1,
                "max": lt_mean * 5,
                "step": 0.5,
            },
            {
                "name": "demand_std",
                "label": "Demand Std Dev",
                "default": d_std,
                "min": 0,
                "max": d_std * 5,
                "step": max(0.1, d_std / 20),
            },
        ],
        "description": "Explore how safety stock changes with service level, lead time, and demand variability.",
    }

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 4. INVENTORY TURNS
# ═══════════════════════════════════════════════════════════════════════════


def _run_inventory_turns(df, config):
    """
    Inventory turns analysis with benchmarking and trend.

    Config
    ------
    cogs : float            — annual COGS ($)
    avg_inventory : float   — average inventory value ($)
    cogs_col : str          — column with periodic COGS data
    inventory_col : str     — column with periodic inventory values
    industry : str          — industry for benchmark comparison [optional]
    """
    result = {"plots": [], "summary": "", "guide_observation": "", "statistics": {}}

    # Benchmarks by industry
    benchmarks = {
        "manufacturing": {
            "typical": 6,
            "good": 10,
            "world_class": 20,
            "label": "Manufacturing",
        },
        "retail": {"typical": 8, "good": 14, "world_class": 25, "label": "Retail"},
        "food": {
            "typical": 12,
            "good": 20,
            "world_class": 40,
            "label": "Food & Beverage",
        },
        "automotive": {
            "typical": 8,
            "good": 15,
            "world_class": 30,
            "label": "Automotive",
        },
        "pharma": {
            "typical": 3,
            "good": 6,
            "world_class": 12,
            "label": "Pharmaceutical",
        },
        "electronics": {
            "typical": 6,
            "good": 12,
            "world_class": 25,
            "label": "Electronics",
        },
    }

    cogs_col = config.get("cogs_col", "")
    inv_col = config.get("inventory_col", "")
    industry = config.get("industry", "manufacturing")

    # Data-driven trend
    has_trend = False
    if cogs_col and inv_col and cogs_col in df.columns and inv_col in df.columns:
        has_trend = True
        cogs_series = df[cogs_col].astype(float)
        inv_series = df[inv_col].astype(float)
        periods = len(df)
        periods_per_year = _float(config, "periods_per_year", 12)
        annual_cogs = (
            float(cogs_series.sum()) * (periods_per_year / periods)
            if periods < periods_per_year
            else float(cogs_series.sum())
        )
        avg_inv = float(inv_series.mean())

        # Per-period turns
        period_turns = (cogs_series * periods_per_year / inv_series).replace([np.inf, -np.inf], np.nan)
        turns_trend = period_turns.dropna().tolist()
    else:
        annual_cogs = _float(config, "cogs")
        avg_inv = _float(config, "avg_inventory")

    if annual_cogs is None or annual_cogs <= 0:
        result["summary"] = "Error: COGS must be a positive number."
        return result
    if avg_inv is None or avg_inv <= 0:
        result["summary"] = "Error: Average inventory must be a positive number."
        return result

    turns = annual_cogs / avg_inv
    doh = 365 / turns
    woh = 52 / turns

    bench = benchmarks.get(industry, benchmarks["manufacturing"])
    if turns >= bench["world_class"]:
        rating = "World Class"
        rating_color = "good"
    elif turns >= bench["good"]:
        rating = "Good"
        rating_color = "accent"
    elif turns >= bench["typical"]:
        rating = "Typical"
        rating_color = "warn"
    else:
        rating = "Below Average"
        rating_color = "bad"

    stats = {
        "inventory_turns": round(turns, 2),
        "days_on_hand": round(doh, 1),
        "weeks_on_hand": round(woh, 2),
        "annual_cogs": round(annual_cogs, 2),
        "avg_inventory": round(avg_inv, 2),
        "rating": rating,
        "industry": bench["label"],
        "benchmark_typical": bench["typical"],
        "benchmark_good": bench["good"],
        "benchmark_world_class": bench["world_class"],
    }
    result["statistics"] = stats

    # ── Gauge chart ──
    max_gauge = bench["world_class"] * 1.5
    result["plots"].append(
        {
            "data": [
                {
                    "type": "indicator",
                    "mode": "gauge+number+delta",
                    "value": turns,
                    "title": {"text": "Inventory Turns"},
                    "gauge": {
                        "axis": {"range": [0, max_gauge]},
                        "bar": {"color": COLOR_GOOD},
                        "steps": [
                            {
                                "range": [0, bench["typical"]],
                                "color": "rgba(208, 96, 96, 0.3)",
                            },
                            {
                                "range": [bench["typical"], bench["good"]],
                                "color": "rgba(232, 149, 71, 0.3)",
                            },
                            {
                                "range": [bench["good"], bench["world_class"]],
                                "color": "rgba(74, 159, 110, 0.3)",
                            },
                            {
                                "range": [bench["world_class"], max_gauge],
                                "color": "rgba(74, 159, 175, 0.3)",
                            },
                        ],
                        "threshold": {
                            "line": {"color": COLOR_REFERENCE, "width": 3},
                            "value": bench["good"],
                        },
                    },
                }
            ],
            "layout": _layout(f"Inventory Turns — {bench['label']}", height=350),
        }
    )

    # ── Trend chart (if data-driven) ──
    if has_trend and len(turns_trend) > 1:
        result["plots"].append(
            {
                "data": [
                    _scatter_trace(
                        list(range(1, len(turns_trend) + 1)),
                        turns_trend,
                        "Period Turns",
                        COLOR_INFO,
                    ),
                    _scatter_trace(
                        [1, len(turns_trend)],
                        [bench["good"]] * 2,
                        f"Good ({bench['good']})",
                        COLOR_GOOD,
                        dash="dash",
                    ),
                    _scatter_trace(
                        [1, len(turns_trend)],
                        [bench["typical"]] * 2,
                        f"Typical ({bench['typical']})",
                        COLOR_WARNING,
                        dash="dot",
                    ),
                ],
                "layout": _layout("Turns Trend", "Period", "Annualized Turns"),
            }
        )

    result["summary"] = "\n".join(
        [
            f"<<COLOR:accent>>Inventory Turns Analysis — {bench['label']}<</COLOR>>",
            "",
            f"<<COLOR:{rating_color}>>{turns:.1f} turns/year — {rating}<</COLOR>>",
            f"  Days on Hand: {doh:.0f}  |  Weeks on Hand: {woh:.1f}",
            "",
            f"  COGS: ${annual_cogs:,.0f}  |  Avg Inventory: ${avg_inv:,.0f}",
            "",
            f"  Benchmarks ({bench['label']}): Typical {bench['typical']} | Good {bench['good']} | World Class {bench['world_class']}",
        ]
    )
    result["guide_observation"] = (
        f"Inventory turns = {turns:.1f} ({rating} for {bench['label']}). "
        f"Days on hand = {doh:.0f}. "
        f"{'Opportunity to improve — reducing DOH by 10 days would free ~$' + f'{annual_cogs * 10 / 365:,.0f} in working capital.' if rating in ('Below Average', 'Typical') else 'Strong inventory velocity.'}"
    )
    result["narrative"] = _narrative(
        verdict=f"{turns:.1f} turns/year = {doh:.0f} days on hand ({rating} for {bench['label']})",
        body=(
            f"Each dollar of inventory generates ${turns:.2f} in COGS annually. "
            f"At current levels, ${avg_inv:,.0f} is tied up in inventory. "
            f"Improving to {bench['good']} turns would reduce average inventory to ${annual_cogs / bench['good']:,.0f}, "
            f"freeing ${avg_inv - annual_cogs / bench['good']:,.0f} in working capital."
        ),
        next_steps=[
            "Run ABC analysis to identify which items drive the most inventory value",
            "Run EOQ analysis to optimize order quantities",
            "Review Class C items for potential min-max or consignment policies",
            f"Target: reduce DOH from {doh:.0f} to {365 / bench['good']:.0f} days ({bench['good']} turns)",
        ],
    )

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 5. SERVICE LEVEL TRADE-OFF
# ═══════════════════════════════════════════════════════════════════════════


def _run_service_level(df, config):
    """
    Service level vs inventory cost trade-off curves.

    Config
    ------
    demand_mean, demand_std, lead_time, lead_time_std : float
    unit_cost : float         — for $ conversion
    holding_pct : float       — holding cost % [default: 25]
    stockout_cost : float     — cost per unit stockout [optional, for ETSC]
    demand_col, lead_time_col : str — data columns [optional]
    """
    from scipy.stats import norm

    result = {"plots": [], "summary": "", "guide_observation": "", "statistics": {}}

    demand_col = config.get("demand_col", "")
    if demand_col and demand_col in df.columns:
        d_data = df[demand_col].dropna().astype(float)
        d_mean = float(d_data.mean())
        d_std = float(d_data.std(ddof=1))
    else:
        d_mean = _float(config, "demand_mean")
        d_std = _float(config, "demand_std")

    lt_mean = _float(config, "lead_time")
    lt_std = _float(config, "lead_time_std", 0)
    C = _float(config, "unit_cost", 1)
    h_pct = _float(config, "holding_pct", 25) / 100
    stockout_cost = _float(config, "stockout_cost", None)

    if d_mean is None or d_std is None or lt_mean is None:
        result["summary"] = "Error: Provide demand_mean, demand_std, and lead_time."
        return result

    sigma_dlt = math.sqrt(lt_mean * d_std**2 + d_mean**2 * lt_std**2)
    H = C * h_pct

    # ── Build trade-off curves ──
    sl_range = np.linspace(0.80, 0.999, 100)
    z_range = norm.ppf(sl_range)
    ss_range = z_range * sigma_dlt
    ss_cost = ss_range * H  # annual holding cost of safety stock

    # Expected shortage per cycle (for fill rate / Type II)
    # E[shortage] = sigma * L(z) where L(z) = phi(z) - z*(1-Phi(z))
    L_z = norm.pdf(z_range) - z_range * (1 - sl_range)
    expected_shortage = sigma_dlt * L_z
    fill_rate = 1 - (expected_shortage / (d_mean * lt_mean))
    fill_rate = np.clip(fill_rate, 0, 1)

    # Total expected cost (if stockout cost given)
    periods_per_year = _float(config, "periods_per_year", 12)
    orders_per_year = periods_per_year  # rough: one replenishment per period
    if stockout_cost is not None:
        annual_stockout_cost = expected_shortage * stockout_cost * orders_per_year
        total_cost = ss_cost + annual_stockout_cost
    else:
        annual_stockout_cost = None
        total_cost = None

    # Find optimal service level (minimum total cost)
    optimal_sl = None
    if total_cost is not None:
        opt_idx = np.argmin(total_cost)
        optimal_sl = float(sl_range[opt_idx]) * 100
        optimal_ss = float(ss_range[opt_idx])
        optimal_cost = float(total_cost[opt_idx])

    stats = {
        "sigma_demand_lt": round(sigma_dlt, 2),
        "service_levels": {
            "90%": {
                "ss": round(float(norm.ppf(0.90) * sigma_dlt), 0),
                "cost": round(float(norm.ppf(0.90) * sigma_dlt * H), 2),
            },
            "95%": {
                "ss": round(float(norm.ppf(0.95) * sigma_dlt), 0),
                "cost": round(float(norm.ppf(0.95) * sigma_dlt * H), 2),
            },
            "99%": {
                "ss": round(float(norm.ppf(0.99) * sigma_dlt), 0),
                "cost": round(float(norm.ppf(0.99) * sigma_dlt * H), 2),
            },
        },
    }
    if optimal_sl is not None:
        stats["optimal_service_level"] = round(optimal_sl, 1)
        stats["optimal_safety_stock"] = round(optimal_ss, 0)
        stats["optimal_total_cost"] = round(optimal_cost, 2)
    result["statistics"] = stats

    # ── Safety stock cost curve ──
    traces = [_scatter_trace(sl_range * 100, ss_cost, "SS Holding Cost", COLOR_INFO)]
    if annual_stockout_cost is not None:
        traces.append(
            _scatter_trace(
                sl_range * 100,
                annual_stockout_cost,
                "Stockout Cost",
                COLOR_BAD,
                dash="dash",
            )
        )
        traces.append(_scatter_trace(sl_range * 100, total_cost, "Total Cost", COLOR_GOOD))
        if optimal_sl is not None:
            traces.append(
                {
                    "type": "scatter",
                    "x": [optimal_sl],
                    "y": [optimal_cost],
                    "mode": "markers+text",
                    "name": f"Optimum ({optimal_sl:.1f}%)",
                    "marker": {"color": COLOR_GOOD, "size": 12, "symbol": "diamond"},
                    "text": [f"{optimal_sl:.1f}%"],
                    "textposition": "top center",
                }
            )

    result["plots"].append(
        {
            "data": traces,
            "layout": _layout("Service Level vs Cost", "Cycle Service Level (%)", "Annual Cost ($)"),
        }
    )

    # ── Fill rate vs cycle service level ──
    result["plots"].append(
        {
            "data": [
                _scatter_trace(sl_range * 100, fill_rate * 100, "Fill Rate", COLOR_GOOD),
                _scatter_trace([80, 100], [80, 100], "1:1 Line", COLOR_NEUTRAL, dash="dot"),
            ],
            "layout": _layout(
                "Fill Rate vs Cycle Service Level",
                "Cycle Service Level (%)",
                "Fill Rate (%)",
            ),
        }
    )

    ss95 = stats["service_levels"]["95%"]
    result["summary"] = "\n".join(
        [
            "<<COLOR:accent>>Service Level Trade-Off Analysis<</COLOR>>",
            "",
            f"  90% SL → SS = {stats['service_levels']['90%']['ss']:,.0f} units (${stats['service_levels']['90%']['cost']:,.2f}/yr)",
            f"  <<COLOR:good>>95% SL → SS = {ss95['ss']:,.0f} units (${ss95['cost']:,.2f}/yr)<</COLOR>>",
            f"  99% SL → SS = {stats['service_levels']['99%']['ss']:,.0f} units (${stats['service_levels']['99%']['cost']:,.2f}/yr)",
            "",
            f"  Going from 95% → 99% costs ${stats['service_levels']['99%']['cost'] - ss95['cost']:,.2f}/yr extra",
        ]
    )
    if optimal_sl is not None:
        result["summary"] += (
            f"\n\n<<COLOR:accent>>Cost-optimal service level: {optimal_sl:.1f}% (total cost ${optimal_cost:,.2f}/yr)<</COLOR>>"
        )

    result["guide_observation"] = (
        f"Service level trade-off: 95% SL requires {ss95['ss']:,.0f} units SS costing ${ss95['cost']:,.2f}/yr. "
        f"Going to 99% doubles the holding cost. " + (f"Cost-optimal SL is {optimal_sl:.1f}%. " if optimal_sl else "")
    )
    result["narrative"] = _narrative(
        verdict="Marginal cost of safety stock increases exponentially above 95%",
        body=(
            "The last 4% of service level (95% → 99%) typically costs as much as the first 95%. "
            "Fill rate (actual demand satisfied) is typically higher than cycle service level — "
            "a 95% CSL often delivers 98%+ fill rate due to partial cycle coverage."
        ),
        next_steps=[
            "Differentiate service levels by ABC class (99% for A, 95% for B, 90% for C)",
            "Run Safety Stock analysis at the chosen service level",
            "Consider stockout cost data for economically optimal SL",
        ],
    )

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 6. DEMAND PROFILE
# ═══════════════════════════════════════════════════════════════════════════


def _run_demand_profile(df, config):
    """
    Demand variability profiling: CoV, trend detection, seasonality, intermittency.

    Config
    ------
    demand_col : str    — column with demand data
    item_col : str      — item identifier for multi-SKU [optional]
    period_col : str    — time period column [optional]
    """
    result = {"plots": [], "summary": "", "guide_observation": "", "statistics": {}}

    demand_col = config.get("demand_col", "") or config.get("var1", "")
    if not demand_col or demand_col not in df.columns:
        result["summary"] = "Error: Select a demand data column."
        return result

    demand = df[demand_col].astype(float)

    n = len(demand)
    mean_d = float(demand.mean())
    std_d = float(demand.std(ddof=1)) if n > 1 else 0
    cov = std_d / mean_d if mean_d > 0 else float("inf")
    zeros = int((demand == 0).sum())
    zero_pct = zeros / n * 100 if n > 0 else 0

    # ADI (Average Demand Interval) for intermittent demand
    nonzero_idx = demand[demand > 0].index.tolist()
    if len(nonzero_idx) > 1:
        intervals = [nonzero_idx[i + 1] - nonzero_idx[i] for i in range(len(nonzero_idx) - 1)]
        adi = float(np.mean(intervals))
    else:
        adi = float(n) if zeros > 0 else 1.0

    # Classify demand pattern (Syntetos-Boylan)
    cv2 = cov**2
    if adi < 1.32 and cv2 < 0.49:
        pattern = "Smooth"
        method = "Moving average or exponential smoothing"
    elif adi < 1.32 and cv2 >= 0.49:
        pattern = "Erratic"
        method = "Exponential smoothing with wider safety stock"
    elif adi >= 1.32 and cv2 < 0.49:
        pattern = "Intermittent"
        method = "Croston's method"
    else:
        pattern = "Lumpy"
        method = "Croston's or Syntetos-Boylan Approximation (SBA)"

    # Trend detection (simple linear regression slope significance)
    from scipy.stats import linregress

    x = np.arange(n)
    if n > 2:
        slope, intercept, r_value, p_value, _ = linregress(x, demand.values)
        has_trend = p_value < 0.05
        trend_dir = "increasing" if slope > 0 else "decreasing"
        trend_pct = abs(slope * n / mean_d) * 100 if mean_d > 0 else 0
    else:
        slope, p_value, has_trend, trend_dir, trend_pct, r_value = (
            0,
            1,
            False,
            "flat",
            0,
            0,
        )

    stats = {
        "n_periods": n,
        "mean": round(mean_d, 2),
        "std": round(std_d, 2),
        "cov": round(cov, 3),
        "zero_periods": zeros,
        "zero_pct": round(zero_pct, 1),
        "adi": round(adi, 2),
        "pattern": pattern,
        "recommended_method": method,
        "trend_slope": round(slope, 4),
        "trend_p_value": round(p_value, 4),
        "trend_significant": has_trend,
        "trend_r_squared": round(r_value**2, 3),
    }
    result["statistics"] = stats

    # ── Demand time series ──
    periods = list(range(1, n + 1))
    traces = [_bar_trace(periods, demand.tolist(), "Demand", COLOR_INFO)]
    if has_trend:
        trend_line = intercept + slope * x
        traces.append(
            _scatter_trace(
                periods,
                trend_line.tolist(),
                f"Trend ({trend_dir})",
                COLOR_WARNING,
                dash="dash",
            )
        )
    traces.append(_scatter_trace(periods, [mean_d] * n, f"Mean ({mean_d:.1f})", COLOR_NEUTRAL, dash="dot"))

    result["plots"].append({"data": traces, "layout": _layout("Demand History", "Period", "Demand")})

    # ── Syntetos-Boylan quadrant ──
    result["plots"].append(
        {
            "data": [
                {
                    "type": "scatter",
                    "x": [adi],
                    "y": [cv2],
                    "mode": "markers+text",
                    "marker": {"color": COLOR_GOOD, "size": 14, "symbol": "diamond"},
                    "text": [pattern],
                    "textposition": "top center",
                    "name": "This item",
                },
                # Quadrant boundaries
                _scatter_trace([1.32, 1.32], [0, 2], "ADI = 1.32", COLOR_NEUTRAL, dash="dash"),
                _scatter_trace([0, 3], [0.49, 0.49], "CV² = 0.49", COLOR_NEUTRAL, dash="dash"),
            ],
            "layout": {
                **_layout(
                    "Syntetos-Boylan Classification",
                    "ADI (Avg Demand Interval)",
                    "CV² (Demand Variability)",
                ),
                "annotations": [
                    {
                        "x": 0.66,
                        "y": 0.24,
                        "text": "Smooth",
                        "showarrow": False,
                        "font": {"color": COLOR_GOOD},
                    },
                    {
                        "x": 0.66,
                        "y": 1.2,
                        "text": "Erratic",
                        "showarrow": False,
                        "font": {"color": COLOR_WARNING},
                    },
                    {
                        "x": 2.0,
                        "y": 0.24,
                        "text": "Intermittent",
                        "showarrow": False,
                        "font": {"color": COLOR_INFO},
                    },
                    {
                        "x": 2.0,
                        "y": 1.2,
                        "text": "Lumpy",
                        "showarrow": False,
                        "font": {"color": COLOR_BAD},
                    },
                ],
            },
        }
    )

    result["summary"] = "\n".join(
        [
            f"<<COLOR:accent>>Demand Profile — {pattern} Pattern<</COLOR>>",
            "",
            f"  Mean: {mean_d:,.1f}  |  Std: {std_d:,.1f}  |  CoV: {cov:.3f}",
            f"  Zero periods: {zeros}/{n} ({zero_pct:.0f}%)  |  ADI: {adi:.2f}",
            f"  {'<<COLOR:warn>>Trend: ' + trend_dir + f' ({trend_pct:.1f}% over horizon, p={p_value:.3f})<</COLOR>>' if has_trend else '  No significant trend detected'}",
            "",
            f"<<COLOR:good>>Recommended forecasting: {method}<</COLOR>>",
        ]
    )
    result["guide_observation"] = (
        f"Demand pattern: {pattern} (CoV={cov:.3f}, ADI={adi:.2f}). "
        f"{'Significant ' + trend_dir + ' trend detected. ' if has_trend else ''}"
        f"Recommended method: {method}."
    )
    result["narrative"] = _narrative(
        verdict=f"{pattern} demand pattern — {method} recommended",
        body=(
            f"Syntetos-Boylan classification places this item in the {pattern.lower()} quadrant "
            f"(ADI = {adi:.2f}, CV² = {cv2:.2f}). "
            + (
                f"A {trend_dir} trend of {trend_pct:.1f}% is present (p = {p_value:.3f}), "
                f"which should be accounted for in forecasting. "
                if has_trend
                else ""
            )
            + f"{'High intermittency (' + str(zeros) + ' zero periods) — standard methods will overstock.' if zero_pct > 20 else ''}"
        ),
        next_steps=[
            f"Apply {method} for this demand pattern",
            "Run Safety Stock analysis using appropriate demand distribution",
            "Run ABC/XYZ analysis across all SKUs for portfolio-level classification",
            (
                "Consider demand sensing for erratic/lumpy items"
                if pattern in ("Erratic", "Lumpy")
                else "Monitor for trend emergence with control charts"
            ),
        ],
    )

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 7. KANBAN SIZING
# ═══════════════════════════════════════════════════════════════════════════


def _run_kanban_sizing(df, config):
    """
    Statistical kanban card calculation.

    Config
    ------
    demand : float          — demand per period
    lead_time : float       — replenishment lead time (same unit as demand period)
    safety_pct : float      — safety factor % [default: 15]
    container_size : float  — units per container [default: 1]
    demand_col : str        — demand data column [optional]
    """
    from scipy.stats import norm

    result = {"plots": [], "summary": "", "guide_observation": "", "statistics": {}}

    demand_col = config.get("demand_col", "")
    if demand_col and demand_col in df.columns:
        d_data = df[demand_col].dropna().astype(float)
        D = float(d_data.mean())
        D_std = float(d_data.std(ddof=1))
    else:
        D = _float(config, "demand")
        D_std = _float(config, "demand_std", 0)

    LT = _float(config, "lead_time")
    safety_pct = _float(config, "safety_pct", 15) / 100
    container = _float(config, "container_size", 1)

    if D is None or D <= 0:
        result["summary"] = "Error: Demand must be positive."
        return result
    if LT is None or LT <= 0:
        result["summary"] = "Error: Lead time must be positive."
        return result
    if container <= 0:
        container = 1

    base = D * LT
    safety = base * safety_pct
    total = base + safety
    cards = math.ceil(total / container)
    total_inventory = cards * container

    # Statistical approach (if std available)
    if D_std and D_std > 0:
        sigma_dlt = D_std * math.sqrt(LT)
        z95 = norm.ppf(0.95)
        stat_safety = z95 * sigma_dlt
        stat_total = base + stat_safety
        stat_cards = math.ceil(stat_total / container)
    else:
        stat_safety = None
        stat_cards = None

    stats = {
        "demand_per_period": round(D, 2),
        "lead_time": round(LT, 2),
        "base_stock": round(base, 1),
        "safety_stock": round(safety, 1),
        "total_stock": round(total, 1),
        "container_size": round(container, 1),
        "kanban_cards": cards,
        "total_inventory": round(total_inventory, 1),
    }
    if stat_cards is not None:
        stats["statistical_safety_stock"] = round(stat_safety, 1)
        stats["statistical_kanban_cards"] = stat_cards
    result["statistics"] = stats

    # ── Pipeline visualization ──
    stages = ["Base Stock\n(D × LT)", "Safety Stock", "Total Cards"]
    values = [base, safety, total_inventory]
    colors = [COLOR_INFO, COLOR_WARNING, COLOR_GOOD]
    result["plots"].append(
        {
            "data": [_bar_trace(stages, values, "Inventory", colors)],
            "layout": _layout("Kanban Stock Breakdown", "", "Units"),
        }
    )

    result["summary"] = "\n".join(
        [
            "<<COLOR:accent>>Kanban Sizing<</COLOR>>",
            "",
            f"<<COLOR:good>>{cards} kanban cards × {container:.0f} units = {total_inventory:.0f} units total<</COLOR>>",
            "",
            f"  Base stock (D × LT): {base:.0f} units",
            f"  Safety stock ({safety_pct * 100:.0f}%): {safety:.0f} units",
            f"  Total: {total:.0f} units → {cards} cards",
        ]
    )
    if stat_cards is not None:
        result["summary"] += (
            f"\n\n  <<COLOR:accent>>Statistical alternative (95% SL): {stat_cards} cards ({stat_safety:.0f} units SS)<</COLOR>>"
        )

    result["guide_observation"] = (
        f"Kanban: {cards} cards at {container:.0f} units each = {total_inventory:.0f} total inventory. "
        f"Base stock = {base:.0f}, safety = {safety:.0f}."
    )
    result["narrative"] = _narrative(
        verdict=f"{cards} kanban cards needed ({total_inventory:.0f} units in system)",
        body=(
            f"Standard kanban formula: cards = ⌈(D × LT × (1 + safety%)) / container⌉. "
            f"With demand = {D:.1f}/period, LT = {LT:.1f} periods, safety = {safety_pct * 100:.0f}%, "
            f"container = {container:.0f} units."
            + (
                f" Statistical method (95% SL using demand σ = {D_std:.2f}) suggests {stat_cards} cards."
                if stat_cards
                else ""
            )
        ),
        next_steps=[
            "Run Demand Profile to verify demand stability (kanban works best with smooth demand)",
            "Run Safety Stock analysis for precise statistical safety factor",
            "Monitor card circulation — stuck cards indicate process problems",
        ],
    )

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 8. EPEI — Every Part Every Interval
# ═══════════════════════════════════════════════════════════════════════════


def _run_epei(df, config):
    """
    EPEI calculation for production scheduling frequency.

    Config
    ------
    available_hours : float    — available hours per day
    num_parts : int            — number of unique parts/SKUs
    changeover_time : float    — average changeover time (minutes)
    target_pct : float         — % of available time budgeted for changeovers [default: 10]
    parts_col : str            — column listing parts [optional, for count]
    changeover_col : str       — column with changeover times per part [optional]
    """
    result = {"plots": [], "summary": "", "guide_observation": "", "statistics": {}}

    avail = _float(config, "available_hours")
    co_time = _float(config, "changeover_time")  # minutes per changeover
    target = _float(config, "target_pct", 10) / 100

    # From data
    parts_col = config.get("parts_col", "")
    co_col = config.get("changeover_col", "")

    if parts_col and parts_col in df.columns:
        num_parts = int(df[parts_col].nunique())
    else:
        num_parts = int(_float(config, "num_parts", 0))

    if co_col and co_col in df.columns:
        co_data = df[co_col].dropna().astype(float)
        co_time = float(co_data.mean())

    if not avail or avail <= 0:
        result["summary"] = "Error: Available hours must be positive."
        return result
    if num_parts <= 0:
        result["summary"] = "Error: Number of parts must be positive."
        return result
    if co_time is None or co_time <= 0:
        result["summary"] = "Error: Changeover time must be positive."
        return result

    avail_min = avail * 60
    budget_min = avail_min * target
    total_co = num_parts * co_time
    epei = total_co / budget_min if budget_min > 0 else float("inf")
    co_pct_actual = (total_co / avail_min) * 100

    stats = {
        "epei_days": round(epei, 2),
        "num_parts": num_parts,
        "changeover_time_min": round(co_time, 1),
        "total_changeover_min": round(total_co, 1),
        "available_min": round(avail_min, 0),
        "budget_min": round(budget_min, 0),
        "changeover_pct_actual": round(co_pct_actual, 1),
        "target_pct": round(target * 100, 1),
    }
    result["statistics"] = stats

    # ── EPEI sensitivity to changeover reduction ──
    co_range = np.linspace(max(1, co_time * 0.1), co_time * 2, 50)
    epei_range = (num_parts * co_range) / budget_min

    result["plots"].append(
        {
            "data": [
                _scatter_trace(co_range, epei_range, "EPEI", COLOR_INFO),
                {
                    "type": "scatter",
                    "x": [co_time],
                    "y": [epei],
                    "mode": "markers",
                    "marker": {"color": COLOR_GOOD, "size": 10, "symbol": "diamond"},
                    "name": f"Current ({co_time:.0f} min)",
                },
            ],
            "layout": _layout("EPEI vs Changeover Time", "Avg Changeover (min)", "EPEI (days)"),
        }
    )

    result["summary"] = "\n".join(
        [
            "<<COLOR:accent>>EPEI — Every Part Every Interval<</COLOR>>",
            "",
            f"<<COLOR:good>>EPEI = {epei:.1f} days<</COLOR>>",
            f"  (Produce every SKU every {epei:.1f} days)",
            "",
            f"  Parts: {num_parts}  |  Avg C/O: {co_time:.0f} min  |  Total C/O: {total_co:.0f} min/cycle",
            f"  C/O budget: {budget_min:.0f} min ({target * 100:.0f}% of {avail_min:.0f} min)",
            f"  Actual C/O load: {co_pct_actual:.1f}%",
        ]
    )
    if co_pct_actual > target * 100:
        result["summary"] += (
            f"\n\n<<COLOR:bad>>Warning: actual C/O load ({co_pct_actual:.1f}%) exceeds budget ({target * 100:.0f}%)<</COLOR>>"
        )

    result["guide_observation"] = (
        f"EPEI = {epei:.1f} days for {num_parts} parts at {co_time:.0f} min avg changeover. "
        f"Changeover budget utilization: {co_pct_actual:.1f}%."
    )
    result["narrative"] = _narrative(
        verdict=f"Every part every {epei:.1f} days",
        body=(
            f"With {num_parts} SKUs and {co_time:.0f}-minute average changeover, "
            f"one full cycle takes {total_co:.0f} minutes of changeover time. "
            f"At a {target * 100:.0f}% C/O budget ({budget_min:.0f} min/day), the cycle takes {epei:.1f} days. "
            f"{'This exceeds the changeover budget — consider SMED to reduce changeover times.' if co_pct_actual > target * 100 else ''}"
        ),
        next_steps=[
            "Apply SMED to reduce changeover times (target 50% reduction)",
            f"Halving C/O time would reduce EPEI to {epei / 2:.1f} days",
            "Run ABC analysis to prioritize which parts run most frequently",
            "Consider dedicated lines for highest-volume A items",
        ],
    )

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 9. ROP SIMULATION — Monte Carlo (s,Q) Policy
# ═══════════════════════════════════════════════════════════════════════════


def _run_rop_simulation(df, config):
    """
    Monte Carlo simulation of (s, Q) reorder-point policy.

    Config
    ------
    demand_mean, demand_std : float
    lead_time, lead_time_std : float
    reorder_point : float       — ROP (s)
    order_quantity : float      — order quantity (Q)
    holding_cost : float        — $/unit/period
    stockout_cost : float       — $/unit short
    order_cost : float          — $/order
    periods : int               — simulation horizon [default: 365]
    runs : int                  — Monte Carlo runs [default: 1000]
    demand_col : str            — demand data for empirical distribution [optional]
    """
    result = {"plots": [], "summary": "", "guide_observation": "", "statistics": {}}

    demand_col = config.get("demand_col", "")
    if demand_col and demand_col in df.columns:
        d_data = df[demand_col].dropna().astype(float).values
        d_mean = float(d_data.mean())
        d_std = float(d_data.std(ddof=1))
        use_empirical = True
    else:
        d_mean = _float(config, "demand_mean")
        d_std = _float(config, "demand_std")
        d_data = None
        use_empirical = False

    lt_mean = _float(config, "lead_time")
    lt_std = _float(config, "lead_time_std", 0)
    rop = _float(config, "reorder_point")
    Q = _float(config, "order_quantity")
    h_cost = _float(config, "holding_cost", 0)
    s_cost = _float(config, "stockout_cost", 0)
    o_cost = _float(config, "order_cost", 0)
    n_periods = int(_float(config, "periods", 365))
    n_runs = min(int(_float(config, "runs", 1000)), 5000)

    if d_mean is None or lt_mean is None or rop is None or Q is None:
        result["summary"] = "Error: Provide demand_mean, lead_time, reorder_point, and order_quantity."
        return result

    rng = np.random.default_rng(42)

    # ── Simulate ──
    all_fill_rates = []
    all_stockouts = []
    all_costs = []
    sample_inventory = None

    for run in range(n_runs):
        inventory = rop + Q  # start full
        backorder = 0
        orders_in_transit = []  # (arrival_period, qty)
        total_demand = 0
        total_served = 0
        n_stockout_days = 0
        cost = 0
        inv_trace = [] if run == 0 else None

        for t in range(n_periods):
            # Receive arrivals
            new_transit = []
            for arrival, qty in orders_in_transit:
                if arrival <= t:
                    inventory += qty
                else:
                    new_transit.append((arrival, qty))
            orders_in_transit = new_transit

            # Fill backorders
            if backorder > 0 and inventory > 0:
                filled = min(backorder, inventory)
                backorder -= filled
                inventory -= filled
                total_served += filled

            # Generate demand
            if use_empirical and d_data is not None:
                demand = max(0, float(rng.choice(d_data)))
            else:
                demand = max(0, rng.normal(d_mean, d_std))
            total_demand += demand

            # Fill demand
            if inventory >= demand:
                inventory -= demand
                total_served += demand
            else:
                served = max(0, inventory)
                total_served += served
                shortage = demand - served
                backorder += shortage
                inventory = 0
                n_stockout_days += 1

            # Costs
            cost += max(0, inventory) * h_cost
            cost += backorder * s_cost

            # Reorder check
            if inventory <= rop and not any(True for _ in orders_in_transit):
                lt = max(1, round(rng.normal(lt_mean, lt_std) if lt_std > 0 else lt_mean))
                orders_in_transit.append((t + lt, Q))
                cost += o_cost

            if inv_trace is not None:
                inv_trace.append(inventory)

        fill_rate = total_served / total_demand if total_demand > 0 else 1.0
        all_fill_rates.append(fill_rate)
        all_stockouts.append(n_stockout_days)
        all_costs.append(cost)

        if run == 0:
            sample_inventory = inv_trace

    fill_rates = np.array(all_fill_rates)
    stockouts = np.array(all_stockouts)
    costs = np.array(all_costs)

    stats = {
        "mean_fill_rate": round(float(fill_rates.mean()) * 100, 2),
        "p5_fill_rate": round(float(np.percentile(fill_rates, 5)) * 100, 2),
        "mean_stockout_days": round(float(stockouts.mean()), 1),
        "stockout_probability": round(float((stockouts > 0).mean()) * 100, 1),
        "mean_total_cost": round(float(costs.mean()), 2),
        "p95_total_cost": round(float(np.percentile(costs, 95)), 2),
        "reorder_point": round(rop, 1),
        "order_quantity": round(Q, 1),
        "simulation_runs": n_runs,
        "simulation_periods": n_periods,
    }
    result["statistics"] = stats

    # ── Sample inventory trace ──
    if sample_inventory:
        periods = list(range(1, len(sample_inventory) + 1))
        result["plots"].append(
            {
                "data": [
                    _scatter_trace(periods, sample_inventory, "Inventory", COLOR_INFO),
                    _scatter_trace(
                        [1, n_periods],
                        [rop, rop],
                        f"ROP = {rop:.0f}",
                        COLOR_WARNING,
                        dash="dash",
                    ),
                    _scatter_trace([1, n_periods], [0, 0], "Stockout", COLOR_BAD, dash="dot"),
                ],
                "layout": _layout("Sample Inventory Trajectory", "Day", "Inventory (units)"),
            }
        )

    # ── Fill rate histogram ──
    result["plots"].append(
        {
            "data": [
                {
                    "type": "histogram",
                    "x": (fill_rates * 100).tolist(),
                    "nbinsx": 40,
                    "marker": {"color": COLOR_INFO},
                    "name": "Fill Rate",
                }
            ],
            "layout": _layout(f"Fill Rate Distribution ({n_runs} runs)", "Fill Rate (%)", "Frequency"),
        }
    )

    mean_fr = stats["mean_fill_rate"]
    result["summary"] = "\n".join(
        [
            f"<<COLOR:accent>>ROP Simulation — (s={rop:.0f}, Q={Q:.0f}) Policy<</COLOR>>",
            "",
            f"<<COLOR:good>>Mean fill rate: {mean_fr:.1f}%<</COLOR>>  (5th percentile: {stats['p5_fill_rate']:.1f}%)",
            f"  Stockout probability: {stats['stockout_probability']:.1f}%  |  Avg stockout days: {stats['mean_stockout_days']:.1f}/{n_periods}",
            f"  Mean total cost: ${stats['mean_total_cost']:,.2f}  (95th pctile: ${stats['p95_total_cost']:,.2f})",
            "",
            f"  ({n_runs} Monte Carlo runs × {n_periods} periods)",
        ]
    )
    result["guide_observation"] = (
        f"(s={rop:.0f}, Q={Q:.0f}) policy achieves {mean_fr:.1f}% mean fill rate. "
        f"Stockout risk: {stats['stockout_probability']:.1f}% of runs had at least one stockout."
    )
    result["narrative"] = _narrative(
        verdict=f"{mean_fr:.1f}% mean fill rate with (s={rop:.0f}, Q={Q:.0f}) policy",
        body=(
            f"Monte Carlo simulation ({n_runs} runs × {n_periods} periods) using "
            f"{'empirical demand distribution' if use_empirical else f'N({d_mean:.1f}, {d_std:.1f}²) demand'} "
            f"and {'variable' if lt_std > 0 else 'fixed'} lead time of {lt_mean:.1f} periods. "
            f"5th percentile fill rate is {stats['p5_fill_rate']:.1f}% — this is the worst-case performance level."
        ),
        next_steps=[
            "Adjust ROP up to improve fill rate (currently at 5th pctile: {:.1f}%)".format(stats["p5_fill_rate"]),
            "Run Safety Stock analysis to analytically set ROP",
            "Run Service Level analysis to find cost-optimal service level",
        ],
    )

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 10. MRP NETTING
# ═══════════════════════════════════════════════════════════════════════════


def _run_mrp_netting(df, config):
    """
    Gross-to-net requirements explosion.

    Expects DataFrame with columns for time-phased gross requirements,
    scheduled receipts, and parameters for lot sizing.

    Config
    ------
    gross_col : str         — column with gross requirements by period
    receipts_col : str      — column with scheduled receipts [optional]
    on_hand : float         — beginning on-hand inventory
    safety_stock : float    — safety stock level [default: 0]
    lead_time : int         — lead time in periods
    lot_size : str          — "lot_for_lot" | "eoq" | "fixed" [default: lot_for_lot]
    fixed_qty : float       — fixed order quantity (if lot_size="fixed")
    eoq_qty : float         — EOQ quantity (if lot_size="eoq")
    """
    result = {"plots": [], "summary": "", "guide_observation": "", "statistics": {}}

    gross_col = config.get("gross_col", "") or config.get("var1", "")
    receipts_col = config.get("receipts_col", "")
    on_hand = _float(config, "on_hand", 0)
    ss = _float(config, "safety_stock", 0)
    lt = int(_float(config, "lead_time", 1))
    lot_rule = config.get("lot_size", "lot_for_lot")
    fixed_qty = _float(config, "fixed_qty", 100)
    eoq_qty = _float(config, "eoq_qty", 100)

    if not gross_col or gross_col not in df.columns:
        result["summary"] = "Error: Select a gross requirements column."
        return result

    gross = df[gross_col].fillna(0).astype(float).tolist()
    n = len(gross)
    receipts = (
        df[receipts_col].fillna(0).astype(float).tolist() if receipts_col and receipts_col in df.columns else [0] * n
    )

    # ── MRP netting logic ──
    projected = []
    net_req = []
    planned_receipts = [0] * n
    planned_orders = [0] * n
    inv = on_hand

    for t in range(n):
        available = inv + receipts[t] + planned_receipts[t]
        net = gross[t] + ss - available
        if net > 0:
            # Lot sizing
            if lot_rule == "lot_for_lot":
                order_qty = net
            elif lot_rule == "fixed":
                order_qty = math.ceil(net / fixed_qty) * fixed_qty
            elif lot_rule == "eoq":
                order_qty = math.ceil(net / eoq_qty) * eoq_qty
            else:
                order_qty = net

            # Place order LT periods back (or now if within horizon)
            release_period = t - lt
            if release_period >= 0:
                planned_orders[release_period] = order_qty
            planned_receipts[t] += order_qty
            available += order_qty
            net_req.append(net)
        else:
            net_req.append(0)

        inv = available - gross[t]
        projected.append(inv)

    total_orders = sum(1 for o in planned_orders if o > 0)
    total_ordered = sum(planned_orders)
    total_gross = sum(gross)
    min_proj = min(projected)

    stats = {
        "periods": n,
        "total_gross_requirements": round(total_gross, 0),
        "total_planned_orders": round(total_ordered, 0),
        "number_of_orders": total_orders,
        "beginning_on_hand": round(on_hand, 0),
        "ending_inventory": round(projected[-1] if projected else 0, 0),
        "min_projected_inventory": round(min_proj, 0),
        "lot_sizing_rule": lot_rule,
        "lead_time": lt,
        "safety_stock": round(ss, 0),
    }
    result["statistics"] = stats

    # ── MRP grid visualization ──
    periods = list(range(1, n + 1))
    result["plots"].append(
        {
            "data": [
                _bar_trace(periods, gross, "Gross Requirements", COLOR_BAD),
                _bar_trace(
                    periods,
                    [planned_receipts[t] for t in range(n)],
                    "Planned Receipts",
                    COLOR_GOOD,
                ),
                _scatter_trace(
                    periods,
                    projected,
                    "Projected Inventory",
                    COLOR_INFO,
                    mode="lines+markers",
                ),
                _scatter_trace(
                    [1, n],
                    [ss, ss],
                    f"Safety Stock ({ss:.0f})",
                    COLOR_WARNING,
                    dash="dash",
                ),
            ],
            "layout": _layout("MRP Netting — Requirements Plan", "Period", "Units"),
        }
    )

    # ── Planned order releases ──
    if any(o > 0 for o in planned_orders):
        result["plots"].append(
            {
                "data": [_bar_trace(periods, planned_orders, "Planned Order Releases", COLOR_GOLD)],
                "layout": _layout("Planned Order Releases", "Period", "Order Qty"),
            }
        )

    result["summary"] = "\n".join(
        [
            f"<<COLOR:accent>>MRP Netting — {n} Period Plan<</COLOR>>",
            "",
            f"  Total gross requirements: {total_gross:,.0f}",
            f"  Planned orders: {total_ordered:,.0f} across {total_orders} releases",
            f"  Lot sizing: {lot_rule.replace('_', ' ').title()}  |  Lead time: {lt} periods",
            "",
            f"  Beginning OH: {on_hand:,.0f}  |  Ending: {projected[-1] if projected else 0:,.0f}",
            f"  Min projected inventory: {min_proj:,.0f}"
            + (" <<COLOR:bad>>(SHORTAGE)<</COLOR>>" if min_proj < 0 else ""),
        ]
    )
    if min_proj < 0:
        result["summary"] += (
            f"\n\n<<COLOR:bad>>Warning: Projected shortage of {abs(min_proj):,.0f} units — review safety stock or expedite.<</COLOR>>"
        )

    result["guide_observation"] = (
        f"MRP plan: {total_ordered:,.0f} units across {total_orders} orders to cover {total_gross:,.0f} gross requirements. "
        f"{'Shortage detected — action required.' if min_proj < 0 else 'No shortages projected.'}"
    )
    result["narrative"] = _narrative(
        verdict=f"{'Feasible' if min_proj >= 0 else 'SHORTAGE'} — {total_orders} planned orders, {total_ordered:,.0f} units",
        body=(
            f"Gross-to-net explosion with {lot_rule.replace('_', ' ')} lot sizing and {lt}-period lead time. "
            f"Beginning on-hand of {on_hand:,.0f} covers initial demand. "
            + (
                f"Minimum projected inventory is {min_proj:,.0f} (below zero = shortage)."
                if min_proj < 0
                else f"Minimum projected inventory is {min_proj:,.0f} units — above safety stock."
            )
        ),
        next_steps=[
            "Review planned order releases with suppliers/production",
            "Run EOQ or Safety Stock to refine lot sizing parameters",
            "Validate gross requirements against latest demand forecast",
            "Check capacity availability for planned production orders",
        ],
    )

    return result


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def _float(config, key, default=None):
    """Safely extract a float from config."""
    val = config.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ═══════════════════════════════════════════════════════════════════════════
# 11. INVENTORY POLICY WIZARD — ABC → differentiated SS + EOQ + Kanban
# ═══════════════════════════════════════════════════════════════════════════


def _sf(val, default=0.0):
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _run_inventory_policy_wizard(df, config):
    """Chain ABC classification → per-class safety stock + EOQ + kanban sizing.

    Applies differentiated service levels by ABC class:
    - Class A (high value): 99% service level
    - Class B (medium value): 95% service level
    - Class C (low value): 90% service level

    Config:
        value_col: str — column for item value/revenue
        demand_col: str — column for demand per period
        demand_std_col: str — column for demand std dev (optional)
        lead_time: float — replenishment lead time in periods
        container_size: float — units per kanban container (default 1)
        holding_cost_pct: float — annual holding cost as % of unit cost (default 0.25)
        order_cost: float — cost per order (default 100)
        a_service: float — Class A target SL (default 0.99)
        b_service: float — Class B target SL (default 0.95)
        c_service: float — Class C target SL (default 0.90)
    """
    from scipy import stats as sp_stats

    value_col = config.get("value_col", "")
    demand_col = config.get("demand_col", "")
    demand_std_col = config.get("demand_std_col", "")
    lead_time = _sf(config.get("lead_time"), 14)
    container_size = _sf(config.get("container_size"), 1)
    holding_pct = _sf(config.get("holding_cost_pct"), 0.25)
    order_cost = _sf(config.get("order_cost"), 100)

    service_levels = {
        "A": _sf(config.get("a_service"), 0.99),
        "B": _sf(config.get("b_service"), 0.95),
        "C": _sf(config.get("c_service"), 0.90),
    }

    # Validate columns
    if not value_col or value_col not in df.columns:
        return {"plots": [], "summary": "Error: value_col required", "statistics": {}}
    if not demand_col or demand_col not in df.columns:
        return {"plots": [], "summary": "Error: demand_col required", "statistics": {}}

    work = df.copy()
    if len(work) == 0:
        return {
            "plots": [],
            "summary": "Error: no data rows",
            "statistics": {"total_items": 0},
        }
    work["_value"] = work[value_col].astype(float)
    work["_demand"] = work[demand_col].astype(float)
    if demand_std_col and demand_std_col in df.columns:
        work["_demand_std"] = work[demand_std_col].astype(float)
    else:
        work["_demand_std"] = work["_demand"] * 0.2  # default 20% CV

    # ABC classification
    work = work.sort_values("_value", ascending=False)
    total_value = work["_value"].sum()
    if total_value == 0:
        return {
            "plots": [],
            "summary": "Error: total value is zero — check value_col",
            "statistics": {"total_items": len(work)},
        }
    work["_cum_pct"] = work["_value"].cumsum() / total_value * 100
    work["abc_class"] = "C"
    work.loc[work["_cum_pct"] <= 80, "abc_class"] = "A"
    work.loc[(work["_cum_pct"] > 80) & (work["_cum_pct"] <= 95), "abc_class"] = "B"

    # Per-item policy calculation
    policies = []
    for _, row in work.iterrows():
        cls = row["abc_class"]
        sl = service_levels[cls]
        z = sp_stats.norm.ppf(sl)
        demand = row["_demand"]
        demand_std = row["_demand_std"]
        unit_cost = row["_value"] / max(demand, 1) if demand > 0 else row["_value"]

        # Safety stock
        ss = z * demand_std * math.sqrt(lead_time)
        rop = demand * lead_time + ss

        # EOQ
        annual_demand = demand * 12  # assume monthly demand
        holding_cost = unit_cost * holding_pct
        eoq = math.sqrt(2 * annual_demand * order_cost / max(holding_cost, 0.01))

        # Kanban cards
        kanban_cards = math.ceil((demand * lead_time * (1 + ss / max(demand * lead_time, 1))) / max(container_size, 1))

        policies.append(
            {
                "abc_class": cls,
                "service_level": sl,
                "demand": round(demand, 1),
                "safety_stock": round(ss, 1),
                "reorder_point": round(rop, 1),
                "eoq": round(eoq, 1),
                "kanban_cards": kanban_cards,
                "z_score": round(z, 2),
            }
        )

    work["_policy"] = policies

    # Aggregate by class
    class_summary = {}
    for cls in ["A", "B", "C"]:
        cls_items = [p for p in policies if p["abc_class"] == cls]
        if cls_items:
            class_summary[cls] = {
                "count": len(cls_items),
                "service_level": service_levels[cls],
                "avg_safety_stock": round(np.mean([p["safety_stock"] for p in cls_items]), 1),
                "avg_eoq": round(np.mean([p["eoq"] for p in cls_items]), 1),
                "total_kanban_cards": sum(p["kanban_cards"] for p in cls_items),
                "avg_reorder_point": round(np.mean([p["reorder_point"] for p in cls_items]), 1),
            }

    total_cards = sum(p["kanban_cards"] for p in policies)

    # Build summary
    summary_parts = [
        "<<COLOR:accent>>Inventory Policy Wizard<</COLOR>>",
        f"<<COLOR:title>>Differentiated policy across {len(policies)} SKUs<</COLOR>>",
    ]
    for cls in ["A", "B", "C"]:
        if cls in class_summary:
            s = class_summary[cls]
            summary_parts.append(
                f"<<COLOR:highlight>>Class {cls}<</COLOR>> ({s['count']} items, {int(s['service_level'] * 100)}% SL): "
                f"avg SS={s['avg_safety_stock']}, avg EOQ={s['avg_eoq']}, "
                f"total kanban cards={s['total_kanban_cards']}"
            )
    summary_parts.append(f"\n<<COLOR:accent>>Total kanban cards: {total_cards}<</COLOR>>")

    return {
        "plots": [],  # Could add class comparison charts
        "summary": "\n".join(summary_parts),
        "guide_observation": f"Policy wizard: {len(policies)} items classified, {total_cards} kanban cards needed",
        "statistics": {
            "total_items": len(policies),
            "class_summary": class_summary,
            "total_kanban_cards": total_cards,
            "service_levels": service_levels,
            "lead_time": lead_time,
            "container_size": container_size,
            "policies": policies[:100],  # Cap for response size
        },
        "narrative": {
            "verdict": f"Inventory policy for {len(policies)} SKUs: {total_cards} kanban cards across 3 service tiers",
            "body": (
                f"Applied differentiated service levels by ABC class. "
                f"Class A ({service_levels['A']:.0%} SL) gets tighter safety stock; "
                f"Class C ({service_levels['C']:.0%} SL) allows more stockout risk. "
                f"Total system inventory: {total_cards} kanban containers."
            ),
            "next_steps": (
                "1. Review Class A items individually — high value warrants item-level tuning\n"
                "2. Run Demand Profile on volatile items to validate safety stock assumptions\n"
                "3. Generate kanban cards for all items → print and deploy\n"
                "4. Set up SPC monitoring on Class A consumption to detect demand shifts"
            ),
        },
    }
