"""Forge-only analysis views for the new workbench.

This module calls forge handlers DIRECTLY. No legacy dispatch, no shims,
no fallback. The demo workbench at /app/demo/analysis/ uses these endpoints.

CR: 9999588b-2da0-47ff-8c47-1967f8ba7f53
"""

import io
import json
import logging
import tempfile
import time
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated

logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADING (self-contained, no legacy imports)
# =============================================================================


def _read_csv_safe(file_or_path):
    """Read CSV with encoding fallback: UTF-8 then latin-1."""
    if hasattr(file_or_path, "read"):
        raw = file_or_path.read()
        try:
            return pd.read_csv(io.BytesIO(raw), encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(raw), encoding="latin-1")
    try:
        return pd.read_csv(file_or_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(file_or_path, encoding="latin-1")


def _resolve_data(request, body):
    """Load DataFrame from request. Returns (df, error_response)."""
    analysis_type = body.get("type", "")

    # Inline data
    inline_data = body.get("data")
    if inline_data and isinstance(inline_data, dict):
        try:
            df = pd.DataFrame(inline_data)
            if len(df) > 10000:
                return None, JsonResponse({"error": "Inline data limited to 10,000 rows"}, status=400)
            return df, None
        except Exception as e:
            return None, JsonResponse({"error": f"Invalid inline data: {e}"}, status=400)

    data_id = body.get("data_id")

    # Uploaded CSV (data_xxx format)
    if data_id and data_id.startswith("data_"):
        for base in [
            Path(settings.MEDIA_ROOT) / "analysis_data" / str(request.user.id),
            Path(tempfile.gettempdir()) / "svend_analysis",
        ]:
            path = base / f"{data_id}.csv"
            if path.exists():
                try:
                    return _read_csv_safe(path), None
                except Exception:
                    pass

    # Triage cleaned dataset
    if data_id:
        try:
            from agents_api.models import TriageResult

            triage = TriageResult.objects.get(id=data_id, user=request.user)
            return pd.read_csv(io.StringIO(triage.cleaned_csv)), None
        except Exception:
            pass

    # Analyses that don't require data
    if analysis_type in ("simulation", "bayesian", "siop"):
        return pd.DataFrame(), None

    return None, JsonResponse({"error": "No data loaded. Please load a dataset first."}, status=400)


# =============================================================================
# FORGE DISPATCH (direct calls, no legacy, no fallback)
# =============================================================================

_FORGE_DISPATCH = {
    "stats": "agents_api.analysis.forge_stats.run_forge_stats",
    "spc": "agents_api.analysis.forge_spc.run_forge_spc",
    "bayesian": "agents_api.analysis.forge_bayesian.run_forge_bayesian",
    "ml": "agents_api.analysis.forge_ml.run_forge_ml",
    "d_type": "agents_api.analysis.forge_misc.run_forge_d_type",
    "simulation": "agents_api.analysis.forge_misc.run_forge_simulation",
    "causal": "agents_api.analysis.forge_misc.run_forge_causal",
    "drift": "agents_api.analysis.forge_misc.run_forge_drift",
    "anytime": "agents_api.analysis.forge_misc.run_forge_anytime",
    "quality_econ": "agents_api.analysis.forge_misc.run_forge_quality_econ",
    "pbs": "agents_api.analysis.forge_misc.run_forge_pbs",
    "ishap": "agents_api.analysis.forge_misc.run_forge_ishap",
    "bayes_msa": "agents_api.analysis.forge_misc.run_forge_bayes_msa",
    "reliability": "agents_api.analysis.forge_misc.run_forge_reliability",
}


def _run_forge(analysis_type, analysis_id, df, config):
    """Call the forge handler for this analysis type. Returns result dict or None."""
    import importlib

    dotted = _FORGE_DISPATCH.get(analysis_type)
    if not dotted:
        return None

    module_path, func_name = dotted.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    func = getattr(mod, func_name)
    return func(analysis_id, df, config)


# =============================================================================
# ENDPOINTS
# =============================================================================


@require_http_methods(["POST"])
@gated
def run_analysis(request):
    """Forge-only analysis endpoint for the new workbench."""
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    analysis_type = body.get("type")
    analysis_id = body.get("analysis")
    config = body.get("config", {})

    if not analysis_type or not analysis_id:
        return JsonResponse({"error": "Missing type or analysis"}, status=400)

    start = time.time()

    # Load data
    df, err = _resolve_data(request, body)
    if err:
        return err

    # Dispatch to forge
    try:
        result = _run_forge(analysis_type, analysis_id, df, config)
    except Exception as e:
        logger.exception(f"Forge analysis error: {analysis_type}/{analysis_id}")
        return JsonResponse({"error": str(e)}, status=500)

    if result is None:
        return JsonResponse(
            {"error": f"No forge handler for {analysis_type}/{analysis_id}"},
            status=400,
        )

    # Standardize output
    try:
        from agents_api.analysis.standardize import standardize_output

        result = standardize_output(result, analysis_type, analysis_id)
    except Exception:
        logger.warning("Standardize failed, returning raw forge result", exc_info=True)

    latency = int((time.time() - start) * 1000)
    logger.info(f"Forge analysis: {analysis_type}/{analysis_id} in {latency}ms")

    return JsonResponse(result, safe=False)


@require_http_methods(["POST"])
@gated
def upload_data(request):
    """Handle CSV upload for the new workbench. Saves to MEDIA_ROOT."""
    import uuid

    if not request.FILES.get("file"):
        return JsonResponse({"error": "No file uploaded"}, status=400)

    f = request.FILES["file"]
    data_id = f"data_{uuid.uuid4().hex[:8]}"

    # Save to user's analysis_data directory
    data_dir = Path(settings.MEDIA_ROOT) / "analysis_data" / str(request.user.id)
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / f"{data_id}.csv"

    with open(dest, "wb") as out:
        for chunk in f.chunks():
            out.write(chunk)

    # Read back to get column info
    try:
        df = _read_csv_safe(dest)
        columns = list(df.columns)
        rows = len(df)
        preview = df.head(10).to_dict(orient="records")
    except Exception as e:
        return JsonResponse({"error": f"Could not parse CSV: {e}"}, status=400)

    return JsonResponse(
        {
            "data_id": data_id,
            "columns": columns,
            "rows": rows,
            "preview": preview,
        }
    )
