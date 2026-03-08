"""
SPC (Statistical Process Control) API Views

Endpoints for:
- Control chart creation and analysis
- Process capability studies
- Statistical summaries
- File upload and field mapping
"""

import json
import logging
import math
import os
import tempfile

from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated, require_auth

from . import spc
from .dsw.common import sanitize_for_json
from .dsw.standardize import standardize_output
from .models import Problem

logger = logging.getLogger(__name__)

# Cache for parsed datasets (in production, use Redis or similar)
# Key: user_id:filename -> ParsedDataset
# Bounded to prevent unbounded memory growth
_CACHE_MAX_SIZE = 256
_parsed_data_cache: dict[str, spc.ParsedDataset] = {}


def _cache_put(key: str, value: spc.ParsedDataset) -> None:
    """Add to cache with LRU eviction when full."""
    if len(_parsed_data_cache) >= _CACHE_MAX_SIZE:
        # Evict oldest entry
        _parsed_data_cache.pop(next(iter(_parsed_data_cache)), None)
    _parsed_data_cache[key] = value


# =============================================================================
# Control Charts
# =============================================================================


@require_http_methods(["POST"])
@gated
def control_chart(request):
    """
    Create a control chart from data.

    Request body:
    {
        "chart_type": "I-MR" | "X-bar R" | "p" | "c",
        "data": [1.2, 1.3, ...] or [[1.2, 1.3], [1.1, 1.4], ...] for subgroups,
        "sample_sizes": [50, 50, ...],  // for p-chart
        "usl": 10.0,  // optional
        "lsl": 0.0,   // optional
        "problem_id": "uuid"  // optional, to save as evidence
    }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    chart_type = body.get("chart_type", "I-MR")
    data = body.get("data", [])
    usl = body.get("usl")
    lsl = body.get("lsl")
    problem_id = body.get("problem_id")
    investigation_id = body.get("investigation_id")

    if not data:
        return JsonResponse({"error": "Data is required"}, status=400)

    # Phase 3: optional FMEA and Evidence hooks
    fmea_row_id = body.get("fmea_row_id")
    project_id = body.get("project_id")

    try:
        if chart_type == "I-MR":
            # Individuals and Moving Range
            if not isinstance(data[0], (int, float)):
                return JsonResponse({"error": "I-MR chart requires flat data array"}, status=400)
            result = spc.individuals_moving_range_chart(data, usl=usl, lsl=lsl)

        elif chart_type == "X-bar R":
            # X-bar and R (subgroups)
            if not isinstance(data[0], list):
                return JsonResponse({"error": "X-bar R chart requires subgroup data (array of arrays)"}, status=400)
            result = spc.xbar_r_chart(data, usl=usl, lsl=lsl)

        elif chart_type == "p":
            # p-chart for proportion defective
            sample_sizes = body.get("sample_sizes", [])
            if not sample_sizes:
                return JsonResponse({"error": "p-chart requires sample_sizes"}, status=400)
            if not isinstance(data[0], (int, float)):
                return JsonResponse({"error": "p-chart requires defective counts as flat array"}, status=400)
            result = spc.p_chart([int(d) for d in data], [int(s) for s in sample_sizes])

        elif chart_type == "c":
            # c-chart for defect counts
            if not isinstance(data[0], (int, float)):
                return JsonResponse({"error": "c-chart requires defect counts as flat array"}, status=400)
            result = spc.c_chart([int(d) for d in data])

        elif chart_type == "T-squared":
            # Hotelling's T² for multivariate data
            if not isinstance(data[0], list):
                return JsonResponse(
                    {"error": "T-squared requires multivariate data (array of arrays, each row = [var1, var2, ...])"},
                    status=400,
                )
            result = spc.hotelling_t_squared_chart(data, usl=usl, lsl=lsl)

        else:
            return JsonResponse({"error": f"Unknown chart type: {chart_type}"}, status=400)

        # Optionally save as evidence to a problem
        if problem_id:
            try:
                problem = Problem.objects.get(id=problem_id, user=request.user)
                evidence_summary = (
                    f"Control Chart Analysis ({chart_type}): "
                    f"{'IN CONTROL' if result.in_control else 'OUT OF CONTROL'}. "
                    f"{len(result.out_of_control)} points outside limits."
                )
                from .problem_views import write_context_file

                problem.add_evidence(
                    summary=evidence_summary,
                    evidence_type="data_analysis",
                    source="SPC Control Chart",
                )
                write_context_file(problem)
            except Problem.DoesNotExist:
                pass  # Ignore if problem not found

        # Investigation bridge (CANON-002 §12)
        if investigation_id:
            evidence_summary = (
                f"Control Chart Analysis ({chart_type}): "
                f"{'IN CONTROL' if result.in_control else 'OUT OF CONTROL'}. "
                f"{len(result.out_of_control)} points outside limits."
            )
            _spc_connect_investigation(
                request,
                investigation_id,
                evidence_summary,
                tool_type="spc_control_chart",
                sample_size=len(data),
            )

        # Phase 3: SPC → FMEA auto-trigger (QMS-001 §11.4)
        fmea_update = None
        if fmea_row_id and not result.in_control:
            fmea_update = _spc_fmea_hook(
                request.user,
                fmea_row_id,
                ooc_count=len(result.out_of_control),
                total_points=len(getattr(result, "data_points", data)),
            )

        # Phase 3: SPC → Evidence bridge (QMS-001 §11.4)
        evidence_created = None
        if project_id and not result.in_control:
            evidence_created = _spc_evidence_hook(
                request.user,
                project_id,
                chart_type,
                ooc_count=len(result.out_of_control),
                total_points=len(getattr(result, "data_points", data)),
                cl=getattr(result, "center_line", None),
                ucl=getattr(result, "ucl", None),
                lcl=getattr(result, "lcl", None),
            )

        resp = sanitize_for_json(
            {
                "success": True,
                "chart": result.to_dict(),
            }
        )
        if fmea_update:
            resp["fmea_update"] = fmea_update
        if evidence_created:
            resp["evidence_created"] = evidence_created

        return JsonResponse(resp)

    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)
    except Exception as e:
        logger.exception("Control chart error")
        return JsonResponse({"error": str(e)}, status=500)


@require_http_methods(["POST"])
@gated
def recommend_chart(request):
    """
    Recommend appropriate control chart type.

    Request body:
    {
        "data_type": "continuous" | "attribute",
        "subgroup_size": 1,
        "attribute_type": "defectives" | "defects"  // for attribute data
    }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    data_type = body.get("data_type", "continuous")
    subgroup_size = body.get("subgroup_size", 1)
    attribute_type = body.get("attribute_type")
    problem_id = body.get("problem_id")
    investigation_id = body.get("investigation_id")

    recommendation = spc.recommend_chart_type(data_type, subgroup_size, attribute_type)

    explanations = {
        "I-MR": "Individuals and Moving Range chart - for continuous data with sample size of 1",
        "X-bar R": "X-bar and Range chart - for continuous data with subgroups of 2-10",
        "X-bar S": "X-bar and Standard Deviation chart - for continuous data with larger subgroups",
        "p": "p-chart - for proportion defective (varying sample sizes OK)",
        "np": "np-chart - for number defective (constant sample size)",
        "c": "c-chart - for count of defects per unit (constant opportunity)",
        "u": "u-chart - for defects per unit (varying opportunity)",
    }

    response_data = {
        "recommended": recommendation,
        "explanation": explanations.get(recommendation, ""),
        "all_options": explanations,
    }

    # Optionally save recommendation as evidence
    if problem_id:
        try:
            problem = Problem.objects.get(id=problem_id, user=request.user)
            evidence_summary = (
                f"SPC chart recommendation: {recommendation} — "
                f"{explanations.get(recommendation, '')} "
                f"(data_type={data_type}, subgroup_size={subgroup_size})"
            )
            from .problem_views import write_context_file

            evidence = problem.add_evidence(
                summary=evidence_summary,
                evidence_type="observation",
                source="SPC Chart Recommender",
            )
            write_context_file(problem)
            response_data["problem_updated"] = True
            response_data["evidence_id"] = evidence["id"]
        except Problem.DoesNotExist:
            pass

    # Investigation bridge (CANON-002 §12)
    if investigation_id:
        evidence_summary = (
            f"SPC chart recommendation: {recommendation} — "
            f"{explanations.get(recommendation, '')} "
            f"(data_type={data_type}, subgroup_size={subgroup_size})"
        )
        _spc_connect_investigation(
            request,
            investigation_id,
            evidence_summary,
            tool_type="spc_recommend",
        )

    return JsonResponse(sanitize_for_json(response_data))


# =============================================================================
# Process Capability
# =============================================================================


@require_http_methods(["POST"])
@gated
def capability_study(request):
    """
    Perform process capability study.

    Request body:
    {
        "data": [1.2, 1.3, ...],
        "usl": 10.0,
        "lsl": 0.0,
        "target": 5.0,  // optional
        "subgroup_size": 1,  // optional
        "problem_id": "uuid"  // optional
    }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    data = body.get("data", [])
    usl = body.get("usl")
    lsl = body.get("lsl")
    target = body.get("target")
    subgroup_size = body.get("subgroup_size", 1)
    problem_id = body.get("problem_id")
    investigation_id = body.get("investigation_id")

    if not data:
        return JsonResponse({"error": "Data is required"}, status=400)
    if usl is None or lsl is None:
        return JsonResponse({"error": "USL and LSL are required"}, status=400)

    try:
        result = spc.calculate_capability(
            data=data,
            usl=usl,
            lsl=lsl,
            target=target,
            subgroup_size=subgroup_size,
        )

        # Optionally save as evidence to a problem
        if problem_id:
            try:
                problem = Problem.objects.get(id=problem_id, user=request.user)
                evidence_summary = (
                    f"Process Capability Study: Cpk={result.cpk:.2f}, Ppk={result.ppk:.2f}, "
                    f"Sigma Level={result.sigma_level:.1f}, Yield={result.yield_percent:.2f}%. "
                    f"{result.interpretation}"
                )
                from .problem_views import write_context_file

                problem.add_evidence(
                    summary=evidence_summary,
                    evidence_type="data_analysis",
                    source="SPC Capability Study",
                )
                write_context_file(problem)
            except Problem.DoesNotExist:
                pass

        # Investigation bridge (CANON-002 §12)
        if investigation_id:
            evidence_summary = (
                f"Process Capability Study: Cpk={result.cpk:.2f}, Ppk={result.ppk:.2f}, "
                f"Sigma Level={result.sigma_level:.1f}, Yield={result.yield_percent:.2f}%. "
                f"{result.interpretation}"
            )
            _spc_connect_investigation(
                request,
                investigation_id,
                evidence_summary,
                tool_type="spc_capability",
                sample_size=len(data),
            )

        resp = _build_capability_response(data, result)
        resp["success"] = True
        return JsonResponse(sanitize_for_json(resp))

    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)
    except Exception as e:
        logger.exception("Capability study error")
        return JsonResponse({"error": str(e)}, status=500)


def _build_capability_response(data, cap):
    """Build a full analysis response (summary + plots) for capability study.

    Returns a dict in the same format as renderStatsOutput expects:
    {summary, plots, guide_observation, what_if_data}
    """
    import numpy as np
    from scipy import stats as sp_stats

    mean = cap.mean
    sigma = cap.sigma_overall
    sigma_w = cap.sigma_within
    lsl, usl, target = cap.lsl, cap.usl, cap.target
    n = len(data)

    # ── Summary text ────────────────────────────────────────────
    summary = f"<<COLOR:title>>CAPABILITY ANALYSIS<</COLOR>>  (n = {n})\n\n"
    summary += f"  Mean:           {mean:.4f}\n"
    summary += f"  Within σ:       {sigma_w:.4f}\n"
    summary += f"  Overall σ:      {sigma:.4f}\n"
    summary += f"  Min:            {min(data):.4f}\n"
    summary += f"  Max:            {max(data):.4f}\n"

    summary += "\n<<COLOR:accent>>Capability Indices:<</COLOR>>\n"
    summary += f"  Cp:   {cap.cp:.3f}     Pp:   {cap.pp:.3f}\n"
    summary += f"  Cpk:  {cap.cpk:.3f}     Ppk:  {cap.ppk:.3f}\n"

    # Cpm (Taguchi) — only meaningful with a target that differs from midpoint
    if target is not None and sigma > 0:
        cpm = (usl - lsl) / (6 * math.sqrt(sigma**2 + (mean - target) ** 2))
        summary += f"  Cpm:  {cpm:.3f}   (Taguchi, target = {target})\n"

    summary += "\n<<COLOR:accent>>Expected Performance:<</COLOR>>\n"
    z_lower = (mean - lsl) / sigma if sigma > 0 else float("inf")
    z_upper = (usl - mean) / sigma if sigma > 0 else float("inf")
    ppm_below = float(sp_stats.norm.cdf(-z_lower) * 1e6)
    ppm_above = float(sp_stats.norm.cdf(-z_upper) * 1e6)
    ppm_total = ppm_below + ppm_above
    summary += f"  PPM < LSL:  {ppm_below:,.0f}\n"
    summary += f"  PPM > USL:  {ppm_above:,.0f}\n"
    summary += f"  PPM Total:  {ppm_total:,.0f}\n"
    summary += f"  Yield:      {cap.yield_percent:.4f}%\n"
    summary += f"  Sigma:      {cap.sigma_level:.2f}\n"

    if cap.cpk >= 1.33:
        summary += f"\n<<COLOR:success>>Process is capable (Cpk = {cap.cpk:.3f} ≥ 1.33)<</COLOR>>"
    elif cap.cpk >= 1.0:
        summary += f"\n<<COLOR:warning>>Process is marginally capable (1.0 ≤ Cpk = {cap.cpk:.3f} < 1.33)<</COLOR>>"
    else:
        summary += f"\n<<COLOR:danger>>Process is NOT capable (Cpk = {cap.cpk:.3f} < 1.0)<</COLOR>>"

    # ── Plot 1: Histogram with normal curve ─────────────────────
    data_arr = np.array(data, dtype=float)
    std = float(sigma)
    x_range = np.linspace(float(np.min(data_arr)) - 2 * std, float(np.max(data_arr)) + 2 * std, 300)
    pdf_vals = sp_stats.norm.pdf(x_range, mean, std)

    hist_traces = [
        {
            "type": "histogram",
            "x": [float(v) for v in data],
            "histnorm": "probability density",
            "name": "Observed",
            "marker": {"color": "rgba(74, 159, 110, 0.35)", "line": {"color": "#4a9f6e", "width": 1}},
        },
        {
            "type": "scatter",
            "x": x_range.tolist(),
            "y": pdf_vals.tolist(),
            "mode": "lines",
            "name": "Normal Fit",
            "line": {"color": "#4a90d9", "width": 2.5},
        },
    ]

    shapes_h = []
    annotations_h = []

    # Mean line
    shapes_h.append(
        {
            "type": "line",
            "x0": mean,
            "x1": mean,
            "y0": 0,
            "y1": 1,
            "yref": "paper",
            "line": {"color": "#00b894", "width": 2},
        }
    )
    annotations_h.append(
        {
            "x": mean,
            "y": 1.06,
            "yref": "paper",
            "text": "Mean",
            "showarrow": False,
            "font": {"color": "#00b894", "size": 10},
        }
    )

    # Target line
    if target is not None:
        shapes_h.append(
            {
                "type": "line",
                "x0": target,
                "x1": target,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": "#e8c547", "dash": "dashdot", "width": 1.5},
            }
        )
        annotations_h.append(
            {
                "x": target,
                "y": 1.06,
                "yref": "paper",
                "text": "Target",
                "showarrow": False,
                "font": {"color": "#e8c547", "size": 10},
            }
        )

    # LSL / USL lines
    shapes_h.append(
        {
            "type": "line",
            "x0": lsl,
            "x1": lsl,
            "y0": 0,
            "y1": 1,
            "yref": "paper",
            "line": {"color": "#e85747", "dash": "dash", "width": 2},
        }
    )
    annotations_h.append(
        {
            "x": lsl,
            "y": 1.06,
            "yref": "paper",
            "text": "LSL",
            "showarrow": False,
            "font": {"color": "#e85747", "size": 11},
        }
    )
    shapes_h.append(
        {
            "type": "line",
            "x0": usl,
            "x1": usl,
            "y0": 0,
            "y1": 1,
            "yref": "paper",
            "line": {"color": "#e85747", "dash": "dash", "width": 2},
        }
    )
    annotations_h.append(
        {
            "x": usl,
            "y": 1.06,
            "yref": "paper",
            "text": "USL",
            "showarrow": False,
            "font": {"color": "#e85747", "size": 11},
        }
    )

    plots = []
    plots.append(
        {
            "title": "Capability Histogram",
            "data": hist_traces,
            "layout": {
                "template": "plotly_dark",
                "height": 320,
                "shapes": shapes_h,
                "annotations": annotations_h,
                "showlegend": True,
                "legend": {"x": 1, "xanchor": "right", "y": 1, "bgcolor": "rgba(0,0,0,0)"},
                "margin": {"t": 40, "r": 20},
                "xaxis": {"title": "Measurement"},
                "yaxis": {"title": "Density"},
            },
        }
    )

    # ── Plot 2: Process spread vs specs ─────────────────────────
    spread_lo = mean - 3 * std
    spread_hi = mean + 3 * std
    pad = (usl - lsl) * 0.15

    spread_traces = [
        {
            "type": "bar",
            "y": [""],
            "x": [usl - lsl],
            "base": [lsl],
            "orientation": "h",
            "name": "Spec Range",
            "marker": {"color": "rgba(232, 87, 71, 0.15)", "line": {"color": "#e85747", "width": 1.5}},
            "width": [0.5],
        },
        {
            "type": "bar",
            "y": [""],
            "x": [spread_hi - spread_lo],
            "base": [spread_lo],
            "orientation": "h",
            "name": "Process \u00b13\u03c3",
            "marker": {"color": "rgba(74, 159, 110, 0.25)", "line": {"color": "#4a9f6e", "width": 1.5}},
            "width": [0.3],
        },
    ]

    spread_shapes = []
    spread_annot = []
    spread_shapes.append(
        {
            "type": "line",
            "x0": mean,
            "x1": mean,
            "y0": -0.3,
            "y1": 0.3,
            "line": {"color": "#00b894", "width": 2.5},
        }
    )
    spread_annot.append(
        {
            "x": mean,
            "y": 0.35,
            "text": f"\u03bc={mean:.2f}",
            "showarrow": False,
            "font": {"color": "#00b894", "size": 10},
        }
    )
    spread_annot.append(
        {
            "x": lsl,
            "y": -0.35,
            "text": f"LSL={lsl}",
            "showarrow": False,
            "font": {"color": "#e85747", "size": 10},
        }
    )
    spread_annot.append(
        {
            "x": usl,
            "y": -0.35,
            "text": f"USL={usl}",
            "showarrow": False,
            "font": {"color": "#e85747", "size": 10},
        }
    )
    spread_annot.append(
        {
            "x": spread_lo,
            "y": 0.35,
            "text": "-3\u03c3",
            "showarrow": False,
            "font": {"color": "#4a9f6e", "size": 9},
        }
    )
    spread_annot.append(
        {
            "x": spread_hi,
            "y": 0.35,
            "text": "+3\u03c3",
            "showarrow": False,
            "font": {"color": "#4a9f6e", "size": 9},
        }
    )
    if target is not None:
        spread_shapes.append(
            {
                "type": "line",
                "x0": target,
                "x1": target,
                "y0": -0.3,
                "y1": 0.3,
                "line": {"color": "#e8c547", "dash": "dashdot", "width": 1.5},
            }
        )
        spread_annot.append(
            {
                "x": target,
                "y": -0.35,
                "text": f"T={target}",
                "showarrow": False,
                "font": {"color": "#e8c547", "size": 10},
            }
        )

    plots.append(
        {
            "title": "Process Spread vs Specification",
            "data": spread_traces,
            "layout": {
                "template": "plotly_dark",
                "height": 180,
                "barmode": "overlay",
                "shapes": spread_shapes,
                "annotations": spread_annot,
                "showlegend": True,
                "legend": {"x": 1, "xanchor": "right", "y": 1, "bgcolor": "rgba(0,0,0,0)"},
                "xaxis": {"range": [min(lsl, spread_lo) - pad, max(usl, spread_hi) + pad], "title": "Measurement"},
                "yaxis": {"visible": False, "range": [-0.5, 0.5]},
                "margin": {"t": 35, "b": 45, "l": 20, "r": 20},
            },
        }
    )

    # ── Plot 3: Normal probability plot (Q-Q) ──────────────────
    sorted_data = np.sort(data_arr)
    n_pts = len(sorted_data)
    probs = (np.arange(1, n_pts + 1) - 0.5) / n_pts
    theoretical_q = sp_stats.norm.ppf(probs, mean, std)

    sw_data = data[:5000] if n > 5000 else data
    sw_stat, sw_p = sp_stats.shapiro(sw_data)
    normality_note = f"Shapiro-Wilk p = {sw_p:.4f}" + (" (normal)" if sw_p >= 0.05 else " (non-normal)")

    plots.append(
        {
            "title": f"Normal Probability Plot  ({normality_note})",
            "data": [
                {
                    "type": "scatter",
                    "x": theoretical_q.tolist(),
                    "y": sorted_data.tolist(),
                    "mode": "markers",
                    "name": "Data",
                    "marker": {"color": "#4a9f6e", "size": 5, "opacity": 0.7},
                },
                {
                    "type": "scatter",
                    "x": [float(theoretical_q.min()), float(theoretical_q.max())],
                    "y": [float(theoretical_q.min()), float(theoretical_q.max())],
                    "mode": "lines",
                    "name": "Reference",
                    "line": {"color": "#e85747", "dash": "dash", "width": 1.5},
                },
            ],
            "layout": {
                "template": "plotly_dark",
                "height": 300,
                "xaxis": {"title": "Theoretical Quantiles"},
                "yaxis": {"title": "Observed"},
                "showlegend": False,
                "margin": {"t": 35},
            },
        }
    )

    guide_obs = f"Process capability Cpk = {cap.cpk:.2f}. " + ("Capable." if cap.cpk >= 1.33 else "Needs improvement.")

    what_if = {
        "type": "capability",
        "mean": float(mean),
        "std": float(std),
        "n": int(n),
        "current_lsl": float(lsl),
        "current_usl": float(usl),
        "data_values": [float(v) for v in data[:5000]],
    }

    return {
        "summary": summary,
        "plots": plots,
        "guide_observation": guide_obs,
        "what_if_data": what_if,
    }


# =============================================================================
# Statistical Summary
# =============================================================================


@require_http_methods(["POST"])
@gated
def statistical_summary(request):
    """
    Calculate statistical summary of data.

    Request body:
    {
        "data": [1.2, 1.3, ...],
        "problem_id": "uuid"  // optional
    }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    data = body.get("data", [])
    problem_id = body.get("problem_id")
    investigation_id = body.get("investigation_id")

    if not data or len(data) < 2:
        return JsonResponse({"error": "Need at least 2 data points"}, status=400)

    try:
        result = spc.calculate_summary(data)
        response_data = {
            "success": True,
            "summary": result.to_dict(),
        }

        # Optionally save as evidence
        if problem_id:
            try:
                problem = Problem.objects.get(id=problem_id, user=request.user)
                summary_dict = result.to_dict()
                evidence_summary = (
                    f"Statistical Summary (n={summary_dict.get('n', len(data))}): "
                    f"mean={summary_dict.get('mean', 0):.4f}, "
                    f"std={summary_dict.get('std', 0):.4f}, "
                    f"median={summary_dict.get('median', 0):.4f}"
                )
                from .problem_views import write_context_file

                evidence = problem.add_evidence(
                    summary=evidence_summary,
                    evidence_type="data_analysis",
                    source="SPC Statistical Summary",
                )
                write_context_file(problem)
                response_data["problem_updated"] = True
                response_data["evidence_id"] = evidence["id"]
            except Problem.DoesNotExist:
                pass

        # Investigation bridge (CANON-002 §12)
        if investigation_id:
            summary_dict = result.to_dict()
            evidence_summary = (
                f"Statistical Summary (n={summary_dict.get('n', len(data))}): "
                f"mean={summary_dict.get('mean', 0):.4f}, "
                f"std={summary_dict.get('std', 0):.4f}, "
                f"median={summary_dict.get('median', 0):.4f}"
            )
            _spc_connect_investigation(
                request,
                investigation_id,
                evidence_summary,
                tool_type="spc_summary",
                sample_size=len(data),
            )

        return JsonResponse(sanitize_for_json(response_data))

    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)
    except Exception as e:
        logger.exception("Statistical summary error")
        return JsonResponse({"error": str(e)}, status=500)


# =============================================================================
# Chart Types Info
# =============================================================================

# =============================================================================
# File Upload and Field Mapping
# =============================================================================


@require_http_methods(["POST"])
@require_auth
def upload_data(request):
    """
    Upload XLSX/CSV file for SPC analysis.

    Returns column information for field mapping.
    """
    if "file" not in request.FILES:
        return JsonResponse({"error": "No file provided"}, status=400)

    uploaded = request.FILES["file"]
    filename = uploaded.name

    # Validate file type
    ext = filename.lower().split(".")[-1]
    if ext not in ["xlsx", "xls", "csv"]:
        return JsonResponse({"error": "Unsupported file type. Please upload .xlsx, .xls, or .csv"}, status=400)

    try:
        # Save to temp file for parsing
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            for chunk in uploaded.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        # Parse the file
        parsed = spc.parse_uploaded_file(tmp_path, filename)

        # Clean up temp file
        os.unlink(tmp_path)

        if parsed.errors:
            return JsonResponse(
                {
                    "error": parsed.errors[0],
                    "errors": parsed.errors,
                },
                status=400,
            )

        # Cache the parsed data for subsequent analysis
        cache_key = f"{request.user.id}:{filename}"
        _cache_put(cache_key, parsed)

        # Return column info for field mapping
        return JsonResponse(
            {
                "success": True,
                "filename": filename,
                "row_count": parsed.row_count,
                "columns": [c.to_dict() for c in parsed.columns],
                "preview": {col: vals for col, vals in parsed.data.items()},
                "cache_key": cache_key,
            }
        )

    except Exception:
        logger.exception("File upload error")
        return JsonResponse({"error": "File upload failed. Please check file size and format."}, status=500)


@require_http_methods(["POST"])
@gated
def analyze_uploaded(request):
    """
    Run SPC analysis on uploaded data with field mapping.

    Request body:
    {
        "cache_key": "user_id:filename",
        "analysis_type": "control_chart" | "capability" | "summary",
        "value_column": "measurement",
        "subgroup_column": "batch",  // optional
        "chart_type": "I-MR",  // for control_chart
        "usl": 10.0,  // for capability
        "lsl": 0.0,   // for capability
        "problem_id": "uuid"  // optional, to save as evidence
    }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    cache_key = body.get("cache_key")
    # Validate cache key belongs to requesting user (format: user_id:filename)
    if cache_key and not cache_key.startswith(f"{request.user.id}:"):
        return JsonResponse({"error": "Access denied"}, status=403)
    if not cache_key or cache_key not in _parsed_data_cache:
        return JsonResponse({"error": "Data not found. Please upload the file again."}, status=400)

    parsed = _parsed_data_cache[cache_key]

    analysis_type = body.get("analysis_type", "control_chart")
    value_column = body.get("value_column")
    subgroup_column = body.get("subgroup_column")
    problem_id = body.get("problem_id")
    investigation_id = body.get("investigation_id")

    if not value_column:
        return JsonResponse({"error": "value_column is required"}, status=400)

    try:
        # Extract data with field mapping
        extracted = spc.extract_spc_data(
            parsed,
            value_column=value_column,
            subgroup_column=subgroup_column,
        )

        result = None
        evidence_summary = None

        if analysis_type == "control_chart":
            chart_type = body.get("chart_type", "I-MR")
            usl = body.get("usl")
            lsl = body.get("lsl")

            if chart_type == "T-squared" and extracted["type"] == "subgroups":
                # Multivariate: treat subgroups as rows of multi-variable observations
                result = spc.hotelling_t_squared_chart(extracted["data"], usl=usl, lsl=lsl)
            elif extracted["type"] == "subgroups":
                if chart_type in ["X-bar R", "xbar_r"]:
                    result = spc.xbar_r_chart(extracted["data"], usl=usl, lsl=lsl)
                else:
                    # Flatten for I-MR
                    flat_data = [v for sg in extracted["data"] for v in sg]
                    result = spc.individuals_moving_range_chart(flat_data, usl=usl, lsl=lsl)
            else:
                if chart_type == "I-MR":
                    result = spc.individuals_moving_range_chart(extracted["data"], usl=usl, lsl=lsl)
                elif chart_type == "c":
                    result = spc.c_chart([int(x) for x in extracted["data"]])
                else:
                    result = spc.individuals_moving_range_chart(extracted["data"], usl=usl, lsl=lsl)

            evidence_summary = (
                f"Control Chart ({result.chart_type}): "
                f"{'IN CONTROL' if result.in_control else 'OUT OF CONTROL'}. "
                f"Mean={result.limits.cl:.4f}, UCL={result.limits.ucl:.4f}, LCL={result.limits.lcl:.4f}. "
                f"{len(result.out_of_control)} out-of-control points."
            )

            response_data = {
                "success": True,
                "analysis_type": "control_chart",
                "chart": result.to_dict(),
                "data_source": {
                    "filename": parsed.filename,
                    "value_column": value_column,
                    "n_points": extracted.get("n_points") or sum(len(sg) for sg in extracted.get("data", [])),
                },
            }

        elif analysis_type == "capability":
            usl = body.get("usl")
            lsl = body.get("lsl")
            target = body.get("target")

            if usl is None or lsl is None:
                return JsonResponse({"error": "USL and LSL required for capability"}, status=400)

            if extracted["type"] == "subgroups":
                flat_data = [v for sg in extracted["data"] for v in sg]
                subgroup_size = extracted.get("subgroup_size", 1)
            else:
                flat_data = extracted["data"]
                subgroup_size = 1

            result = spc.calculate_capability(
                flat_data,
                usl=usl,
                lsl=lsl,
                target=target,
                subgroup_size=subgroup_size,
            )

            evidence_summary = (
                f"Process Capability: Cpk={result.cpk:.3f}, Ppk={result.ppk:.3f}, "
                f"Sigma Level={result.sigma_level:.2f}, Yield={result.yield_percent:.2f}%. "
                f"{result.interpretation}"
            )

            resp = _build_capability_response(flat_data, result)

            response_data = {
                "success": True,
                "analysis_type": "capability",
                **resp,
                "data_source": {
                    "filename": parsed.filename,
                    "value_column": value_column,
                    "n_samples": result.n_samples,
                },
            }

            # Apply DSW post-processing (education, narrative, evidence grade, charts)
            standardize_output(response_data, "spc", "capability")

        elif analysis_type == "summary":
            if extracted["type"] == "subgroups":
                flat_data = [v for sg in extracted["data"] for v in sg]
            else:
                flat_data = extracted["data"]

            result = spc.calculate_summary(flat_data)

            response_data = {
                "success": True,
                "analysis_type": "summary",
                "summary": result.to_dict(),
                "data_source": {
                    "filename": parsed.filename,
                    "value_column": value_column,
                },
            }

        else:
            return JsonResponse({"error": f"Unknown analysis_type: {analysis_type}"}, status=400)

        # Save evidence to problem if requested
        if problem_id and evidence_summary:
            try:
                problem = Problem.objects.get(id=problem_id, user=request.user)
                problem.add_evidence(
                    summary=evidence_summary,
                    evidence_type="analysis",
                    source=f"SPC {analysis_type.replace('_', ' ').title()} - {parsed.filename}",
                )
                problem.save()
                response_data["evidence_added"] = True
            except Problem.DoesNotExist:
                pass

        # Investigation bridge (CANON-002 §12)
        if investigation_id and evidence_summary:
            _spc_connect_investigation(
                request,
                investigation_id,
                evidence_summary,
                tool_type=f"spc_{analysis_type}",
                sample_size=parsed.row_count,
            )

        return JsonResponse(sanitize_for_json(response_data))

    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)
    except Exception as e:
        logger.exception("Analysis error")
        return JsonResponse({"error": str(e)}, status=500)


@require_http_methods(["POST"])
@gated
def gage_rr(request):
    """
    Perform Gage R&R (Crossed) study.

    Request body (inline data):
    {
        "parts": ["P1", "P1", "P2", ...],
        "operators": ["Op1", "Op1", "Op2", ...],
        "measurements": [1.23, 1.25, ...],
        "tolerance": 0.5,  // optional (USL - LSL)
        "problem_id": "uuid"  // optional
    }

    Or (from file upload):
    {
        "cache_key": "user_id:filename",
        "part_column": "Part",
        "operator_column": "Operator",
        "measurement_column": "Measurement",
        "tolerance": 0.5,
        "problem_id": "uuid"
    }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    tolerance = body.get("tolerance")
    problem_id = body.get("problem_id")
    investigation_id = body.get("investigation_id")

    try:
        # Get data from inline arrays or from cached file upload
        cache_key = body.get("cache_key")
        if cache_key:
            if cache_key not in _parsed_data_cache:
                return JsonResponse({"error": "Data not found. Please upload the file again."}, status=400)

            parsed = _parsed_data_cache[cache_key]
            part_col = body.get("part_column")
            op_col = body.get("operator_column")
            meas_col = body.get("measurement_column")

            if not all([part_col, op_col, meas_col]):
                return JsonResponse(
                    {"error": "part_column, operator_column, and measurement_column are required"}, status=400
                )

            for col in [part_col, op_col, meas_col]:
                if col not in parsed.data:
                    return JsonResponse({"error": f"Column '{col}' not found in uploaded data"}, status=400)

            # Build parallel arrays, skipping rows with missing values
            parts_list = []
            operators_list = []
            measurements_list = []
            for i in range(parsed.row_count):
                p_val = parsed.data[part_col][i]
                o_val = parsed.data[op_col][i]
                m_val = parsed.data[meas_col][i]
                if p_val is not None and o_val is not None and m_val is not None:
                    parts_list.append(str(p_val))
                    operators_list.append(str(o_val))
                    measurements_list.append(float(m_val))
        else:
            parts_list = body.get("parts", [])
            operators_list = body.get("operators", [])
            measurements_list = body.get("measurements", [])

        if not parts_list or not operators_list or not measurements_list:
            return JsonResponse({"error": "parts, operators, and measurements are required"}, status=400)

        result = spc.gage_rr_crossed(
            parts=parts_list,
            operators=operators_list,
            measurements=measurements_list,
            tolerance=tolerance,
        )

        # Optionally save as evidence to a problem
        if problem_id:
            try:
                problem = Problem.objects.get(id=problem_id, user=request.user)
                evidence_summary = (
                    f"Gage R&R Study: %GRR={result.grr_percent:.1f}% ({result.assessment}). "
                    f"NDC={result.ndc}. {result.n_parts} parts, {result.n_operators} operators, "
                    f"{result.n_replicates} replicates."
                )
                from .problem_views import write_context_file

                problem.add_evidence(
                    summary=evidence_summary,
                    evidence_type="data_analysis",
                    source="SPC Gage R&R",
                )
                write_context_file(problem)
            except Problem.DoesNotExist:
                pass

        # Investigation bridge (CANON-002 §12)
        if investigation_id:
            evidence_summary = (
                f"Gage R&R Study: %GRR={result.grr_percent:.1f}% ({result.assessment}). "
                f"NDC={result.ndc}. {result.n_parts} parts, {result.n_operators} operators, "
                f"{result.n_replicates} replicates."
            )
            n_measurements = result.n_parts * result.n_operators * result.n_replicates
            _spc_connect_investigation(
                request,
                investigation_id,
                evidence_summary,
                tool_type="spc_gage_rr",
                sample_size=n_measurements,
            )

        return JsonResponse(
            sanitize_for_json(
                {
                    "success": True,
                    "gage_rr": result.to_dict(),
                }
            )
        )

    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)
    except Exception as e:
        logger.exception("Gage R&R error")
        return JsonResponse({"error": str(e)}, status=500)


@require_http_methods(["GET"])
@require_auth
def chart_types(request):
    """Get information about available control chart types."""
    return JsonResponse(
        {
            "chart_types": [
                {
                    "id": "I-MR",
                    "name": "Individuals & Moving Range",
                    "description": "For continuous data with subgroup size of 1",
                    "data_type": "continuous",
                    "subgroup_size": "1",
                    "use_when": "Each measurement is independent (e.g., batch process, daily readings)",
                },
                {
                    "id": "X-bar R",
                    "name": "X-bar and Range",
                    "description": "For continuous data with rational subgroups of 2-10",
                    "data_type": "continuous",
                    "subgroup_size": "2-10",
                    "use_when": "Multiple measurements per sample (e.g., 5 parts per hour)",
                },
                {
                    "id": "p",
                    "name": "p-Chart (Proportion)",
                    "description": "For proportion defective with varying sample sizes",
                    "data_type": "attribute",
                    "subgroup_size": "varies",
                    "use_when": "Counting defective items from samples of different sizes",
                },
                {
                    "id": "c",
                    "name": "c-Chart (Count)",
                    "description": "For count of defects in same-sized units",
                    "data_type": "attribute",
                    "subgroup_size": "constant",
                    "use_when": "Counting defects per unit (e.g., scratches per panel)",
                },
                {
                    "id": "T-squared",
                    "name": "Hotelling's T² (Multivariate)",
                    "description": "For multivariate continuous data (2+ characteristics per observation)",
                    "data_type": "continuous_multivariate",
                    "subgroup_size": "2+ variables",
                    "use_when": "Multiple interrelated measurements per sample (e.g., length & width, temp & pressure)",
                },
            ],
            "capability_indices": [
                {
                    "id": "cp",
                    "name": "Cp (Capability Index)",
                    "description": "Process potential - spec width / process width",
                    "interpretation": "Ignores centering. Cp=1 means 3σ = half spec width.",
                },
                {
                    "id": "cpk",
                    "name": "Cpk (Capability Index - Centered)",
                    "description": "Process capability accounting for centering",
                    "interpretation": "The key metric. Cpk ≥ 1.33 is typically required.",
                },
                {
                    "id": "pp",
                    "name": "Pp (Performance Index)",
                    "description": "Like Cp but uses overall (long-term) variation",
                    "interpretation": "Includes all sources of variation.",
                },
                {
                    "id": "ppk",
                    "name": "Ppk (Performance Index - Centered)",
                    "description": "Like Cpk but uses overall variation",
                    "interpretation": "Real-world performance including drift.",
                },
            ],
            "sigma_levels": [
                {"sigma": 2, "dpmo": 308537, "yield": "69.1%"},
                {"sigma": 3, "dpmo": 66807, "yield": "93.3%"},
                {"sigma": 4, "dpmo": 6210, "yield": "99.38%"},
                {"sigma": 5, "dpmo": 233, "yield": "99.977%"},
                {"sigma": 6, "dpmo": 3.4, "yield": "99.99966%"},
            ],
        }
    )


# =============================================================================
# Phase 3: SPC Closed-Loop Automation Hooks (QMS-001 §11.4)
# =============================================================================


def _spc_fmea_hook(user, fmea_row_id, ooc_count, total_points):
    """Auto-update FMEA occurrence score when SPC detects OOC.

    Same mapping logic as fmea_views.spc_update_occurrence, but called
    programmatically from SPC control_chart endpoint.
    """
    from .models import FMEARow

    try:
        row = FMEARow.objects.select_related("fmea").get(id=fmea_row_id, fmea__owner=user)
    except FMEARow.DoesNotExist:
        logger.warning("SPC FMEA hook: row %s not found for user", fmea_row_id)
        return None

    ooc_rate = ooc_count / max(total_points, 1)

    # AIAG OOC-to-occurrence mapping (same as fmea_views.py)
    if ooc_rate == 0:
        new_occ = 1
    elif ooc_rate < 0.01:
        new_occ = 2
    elif ooc_rate < 0.02:
        new_occ = 3
    elif ooc_rate < 0.05:
        new_occ = 4
    elif ooc_rate < 0.10:
        new_occ = 5
    elif ooc_rate < 0.15:
        new_occ = 6
    elif ooc_rate < 0.20:
        new_occ = 7
    elif ooc_rate < 0.30:
        new_occ = 8
    elif ooc_rate < 0.50:
        new_occ = 9
    else:
        new_occ = 10

    old_occ = row.occurrence
    old_rpn = row.rpn
    row.occurrence = new_occ
    row.save()

    logger.info("SPC FMEA hook: row %s occurrence %d→%d, RPN %d→%d", fmea_row_id, old_occ, new_occ, old_rpn, row.rpn)

    return {
        "row_id": str(row.id),
        "old_occurrence": old_occ,
        "new_occurrence": new_occ,
        "old_rpn": old_rpn,
        "new_rpn": row.rpn,
        "ooc_rate": round(ooc_rate, 4),
    }


def _spc_connect_investigation(
    request, investigation_id, event_description, tool_type="spc", sample_size=None, measurement_system_id=None
):
    """Connect SPC output to an investigation via the bridge (CANON-002 §12).

    Called alongside problem.add_evidence() for dual-write during migration.
    Returns bridge result dict or None on error.
    """
    from core.models import MeasurementSystem

    from .investigation_bridge import InferenceSpec, connect_tool

    try:
        # Use a generic MeasurementSystem for the tool output link
        tool_output, _ = MeasurementSystem.objects.get_or_create(
            name=f"SPC {tool_type}",
            owner=request.user,
            defaults={"system_type": "variable"},
        )
        spec = InferenceSpec(
            event_description=event_description,
            sample_size=sample_size,
            measurement_system_id=measurement_system_id,
        )
        result = connect_tool(
            investigation_id=investigation_id,
            tool_output=tool_output,
            tool_type=tool_type,
            user=request.user,
            spec=spec,
        )
        return result
    except Exception:
        logger.exception("SPC investigation bridge error")
        return None


def _spc_evidence_hook(user, project_id, chart_type, ooc_count, total_points, cl=None, ucl=None, lcl=None):
    """Create Evidence entry when SPC detects OOC or low capability.

    Uses evidence_bridge for deduplication.
    """
    from core.models import Project

    from .evidence_bridge import create_tool_evidence

    try:
        project = Project.objects.get(id=project_id, user=user)
    except Project.DoesNotExist:
        logger.warning("SPC evidence hook: project %s not found", project_id)
        return None

    summary = f"SPC OOC: {chart_type} detected {ooc_count} out-of-control points out of {total_points} total."
    details = f"Chart type: {chart_type}\nOOC count: {ooc_count}\nTotal points: {total_points}"
    if cl is not None:
        details += f"\nCenter line: {cl}"
    if ucl is not None:
        details += f"\nUCL: {ucl}"
    if lcl is not None:
        details += f"\nLCL: {lcl}"

    evidence, _ = create_tool_evidence(
        project=project,
        user=user,
        summary=summary,
        source_tool="spc",
        source_id=f"chart_{chart_type}",
        source_field="control_chart",
        details=details,
        source_type="analysis",
        confidence=0.7,
    )

    if evidence:
        logger.info("SPC evidence hook: created evidence %s for project %s", evidence.id, project_id)
        return {"evidence_id": str(evidence.id), "summary": summary}

    return None
