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
import tempfile
import os

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated, require_auth
from . import spc
from .models import Problem

logger = logging.getLogger(__name__)

# Cache for parsed datasets (in production, use Redis or similar)
# Key: user_id:filename -> ParsedDataset
_parsed_data_cache: dict[str, spc.ParsedDataset] = {}


# =============================================================================
# Control Charts
# =============================================================================

@csrf_exempt
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

    if not data:
        return JsonResponse({"error": "Data is required"}, status=400)

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
                return JsonResponse({"error": "T-squared requires multivariate data (array of arrays, each row = [var1, var2, ...])"}, status=400)
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

        return JsonResponse({
            "success": True,
            "chart": result.to_dict(),
        })

    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)
    except Exception as e:
        logger.exception("Control chart error")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
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

    return JsonResponse(response_data)


# =============================================================================
# Process Capability
# =============================================================================

@csrf_exempt
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

        return JsonResponse({
            "success": True,
            "capability": result.to_dict(),
        })

    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)
    except Exception as e:
        logger.exception("Capability study error")
        return JsonResponse({"error": str(e)}, status=500)


# =============================================================================
# Statistical Summary
# =============================================================================

@csrf_exempt
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

        return JsonResponse(response_data)

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

@csrf_exempt
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
    ext = filename.lower().split('.')[-1]
    if ext not in ['xlsx', 'xls', 'csv']:
        return JsonResponse({
            "error": "Unsupported file type. Please upload .xlsx, .xls, or .csv"
        }, status=400)

    try:
        # Save to temp file for parsing
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as tmp:
            for chunk in uploaded.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        # Parse the file
        parsed = spc.parse_uploaded_file(tmp_path, filename)

        # Clean up temp file
        os.unlink(tmp_path)

        if parsed.errors:
            return JsonResponse({
                "error": parsed.errors[0],
                "errors": parsed.errors,
            }, status=400)

        # Cache the parsed data for subsequent analysis
        cache_key = f"{request.user.id}:{filename}"
        _parsed_data_cache[cache_key] = parsed

        # Return column info for field mapping
        return JsonResponse({
            "success": True,
            "filename": filename,
            "row_count": parsed.row_count,
            "columns": [c.to_dict() for c in parsed.columns],
            "preview": {col: vals for col, vals in parsed.data.items()},
            "cache_key": cache_key,
        })

    except Exception as e:
        logger.exception("File upload error")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
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
    if not cache_key or cache_key not in _parsed_data_cache:
        return JsonResponse({
            "error": "Data not found. Please upload the file again."
        }, status=400)

    parsed = _parsed_data_cache[cache_key]

    analysis_type = body.get("analysis_type", "control_chart")
    value_column = body.get("value_column")
    subgroup_column = body.get("subgroup_column")
    problem_id = body.get("problem_id")

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

            response_data = {
                "success": True,
                "analysis_type": "capability",
                "capability": result.to_dict(),
                "data_source": {
                    "filename": parsed.filename,
                    "value_column": value_column,
                    "n_samples": result.n_samples,
                },
            }

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

        return JsonResponse(response_data)

    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)
    except Exception as e:
        logger.exception("Analysis error")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
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
                return JsonResponse({"error": "part_column, operator_column, and measurement_column are required"}, status=400)

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

        return JsonResponse({
            "success": True,
            "gage_rr": result.to_dict(),
        })

    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)
    except Exception as e:
        logger.exception("Gage R&R error")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
@require_auth
def chart_types(request):
    """Get information about available control chart types."""
    return JsonResponse({
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
    })
