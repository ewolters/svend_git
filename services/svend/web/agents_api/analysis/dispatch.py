"""Analysis dispatcher — routes analysis requests to the clean analysis/ package.

Replaces dsw/dispatch.py as the single entry point for all analysis requests.
Imports from analysis.* (migrated modules) and agents_api.* (not yet migrated).

DSW-001 §4.1: Stateless dispatch — every request is self-contained.
DSW-001 §4.3: All routing through dispatch.py, never direct calls.

CR: 3c0d0e53
"""

import json
import logging
import math
import tempfile
import time
import uuid
from pathlib import Path

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated

from ..models import DSWResult
from .common import get_cached_model, log_agent_action, safe_json_response

logger = logging.getLogger(__name__)


# =============================================================================
# AUDIT HELPERS
# =============================================================================


def _log_rejection(request, reason, analysis_type=None, analysis_id=None):
    """Log analysis validation rejection to audit trail (FEAT-093, QUAL-001)."""
    try:
        from syn.audit.utils import generate_entry

        tenant_id = getattr(request.user, "tenant_id", None)
        actor = getattr(request.user, "email", "anonymous")
        generate_entry(
            tenant_id=tenant_id,
            actor=actor,
            event_name="quality.analysis_rejected",
            payload={
                "reason": reason,
                "analysis_type": analysis_type,
                "analysis_id": analysis_id,
            },
        )
    except Exception:
        logger.warning("Failed to log quality rejection", exc_info=True)


# =============================================================================
# DATA SOURCE RESOLUTION (DSW-001 §4.2)
# =============================================================================


def _read_csv_safe(file_or_path):
    """Read CSV with encoding fallback: UTF-8 → latin-1."""
    import io

    import pandas as pd

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


def _resolve_data(request, body, analysis_type, analysis_id):
    """Resolve DataFrame from request body per DSW-001 §4.2 fallback order.

    Returns (df, error_response). If df is None, error_response is set.
    """
    from io import StringIO

    import pandas as pd

    df = None

    # Source 0: Inline data (from learning module or API callers)
    inline_data = body.get("data")
    if inline_data and isinstance(inline_data, dict):
        try:
            df = pd.DataFrame(inline_data)
            if len(df) > 10000:
                _log_rejection(request, "inline_data_too_large", analysis_type, analysis_id)
                return None, JsonResponse({"error": "Inline data limited to 10,000 rows"}, status=400)
        except Exception as e:
            _log_rejection(request, f"invalid_inline_data: {e}", analysis_type, analysis_id)
            return None, JsonResponse({"error": f"Invalid inline data: {e}"}, status=400)

    data_id = body.get("data_id")

    # Source 1: Uploaded via upload_data endpoint (data_xxx format)
    if df is None and data_id and data_id.startswith("data_"):
        try:
            data_dir = Path(settings.MEDIA_ROOT) / "analysis_data" / str(request.user.id)
            data_path = data_dir / f"{data_id}.csv"
            if data_path.exists():
                df = _read_csv_safe(data_path)
        except Exception:
            pass

        # Fallback to temp directory
        if df is None:
            try:
                data_dir = Path(tempfile.gettempdir()) / "svend_analysis"
                data_path = data_dir / f"{data_id}.csv"
                if data_path.exists():
                    df = _read_csv_safe(data_path)
            except Exception:
                pass

    # Source 2: Triage cleaned dataset
    if df is None and data_id:
        try:
            from ..models import TriageResult

            triage_result = TriageResult.objects.get(id=data_id, user=request.user)
            df = pd.read_csv(StringIO(triage_result.cleaned_csv))
        except Exception:
            pass

    # Source 3: Empty DataFrame for analyses that don't require data
    if df is None and analysis_type in ("simulation", "bayesian", "siop"):
        df = pd.DataFrame()
    elif df is None:
        _log_rejection(request, "no_data_loaded", analysis_type, analysis_id)
        return None, JsonResponse({"error": "No data loaded. Please load a dataset first."}, status=400)

    return df, None


# =============================================================================
# ANALYSIS ROUTING (DSW-001 §4.1, §4.3)
# =============================================================================

# Maps analysis_type → (import_path, function_name, extra_args)
# extra_args: 'user' = pass request.user, 'model' = pass cached model
_ROUTE_TABLE = {
    "stats": ("agents_api.analysis.stats", "run_statistical_analysis", None),
    "ml": ("agents_api.analysis.ml", "run_ml_analysis", "user"),
    "spc": ("agents_api.analysis.spc", "run_spc_analysis", None),
    "viz": ("agents_api.analysis.viz", "run_visualization", None),
    "bayesian": ("agents_api.analysis.bayesian", "run_bayesian_analysis", None),
    "reliability": ("agents_api.analysis.reliability", "run_reliability_analysis", None),
    "simulation": ("agents_api.analysis.simulation", "run_simulation", "user"),
    "d_type": ("agents_api.analysis.d_type", "run_d_type", None),
    "siop": ("agents_api.analysis.siop", "run_siop", None),
    # Not yet migrated to analysis/ — import from agents_api top-level
    "causal": ("agents_api.causal_discovery", "run_causal_discovery", None),
    "drift": ("agents_api.drift_detection", "run_drift_detection", None),
    "anytime": ("agents_api.anytime_valid", "run_anytime_valid", None),
    "bayes_msa": ("agents_api.msa_bayes", "run_bayes_msa", None),
    "quality_econ": ("agents_api.quality_economics", "run_quality_econ", None),
    "pbs": ("agents_api.pbs_engine", "run_pbs", None),
    "ishap": ("agents_api.interventional_shap", "run_interventional_shap", "model"),
}


def _dispatch_analysis(analysis_type, df, analysis_id, config, request):
    """Route to the appropriate analysis handler.

    Returns result dict or raises on unknown type.
    """
    import importlib

    route = _ROUTE_TABLE.get(analysis_type)
    if route is None:
        return None

    module_path, func_name, extra = route
    mod = importlib.import_module(module_path)
    func = getattr(mod, func_name)

    if extra == "user":
        return func(df, analysis_id, config, request.user)
    elif extra == "model":
        model_key = config.get("model_key", "")
        model_obj, model_feats = None, []
        if model_key:
            cached = get_cached_model(request.user.id if request.user else 0, model_key)
            if cached:
                model_obj = cached.get("model")
                model_feats = cached.get("meta", {}).get("features", [])
        return func(df, analysis_id, config, model=model_obj, model_features=model_feats)
    else:
        return func(df, analysis_id, config)


# =============================================================================
# RESULT PERSISTENCE (DSW-001 §6.2)
# =============================================================================


def _persist_result(request, result, analysis_type, analysis_id, config, project_id, result_title):
    """Save DSWResult for A3/method import if requested."""
    try:
        result_id = f"dsw_{uuid.uuid4().hex[:8]}"
        from core.models import Project

        project = None
        if project_id:
            try:
                project = Project.objects.get(id=project_id, user=request.user)
            except Project.DoesNotExist:
                pass

        DSWResult.objects.create(
            id=result_id,
            user=request.user,
            result_type=f"{analysis_type}_{analysis_id}",
            data=json.dumps(
                {
                    "analysis_type": analysis_type,
                    "analysis_id": analysis_id,
                    "config": config,
                    "summary": result.get("summary", ""),
                    "guide_observation": result.get("guide_observation", ""),
                    "plots": result.get("plots", []),
                    "plots_count": len(result.get("plots", [])),
                }
            ),
            project=project,
            title=result_title or f"{analysis_id.replace('_', ' ').title()} Analysis",
        )
        result["result_id"] = result_id
    except Exception as e:
        logger.warning(f"Could not save DSW result: {e}")


# =============================================================================
# INVESTIGATION BRIDGE (CANON-002 §12)
# =============================================================================


def _connect_investigation(request, investigation_id, result, analysis_type, analysis_id, config):
    """Connect analysis output to an investigation via the bridge.

    Extracts statistical metrics from the standardized result, builds an
    InferenceSpec, and calls connect_tool() with tool_type="dsw".
    """
    from .standardize import _classify_effect, _extract_p_value

    try:
        from ..investigation_bridge import InferenceSpec, connect_tool

        p_value = _extract_p_value(result)
        stats = result.get("statistics", {})
        if not isinstance(stats, dict):
            stats = {}
        sample_size = None
        for _key in ("n", "n_obs", "n_total", "sample_size"):
            _val = stats.get(_key)
            if _val is not None:
                try:
                    sample_size = int(float(_val))
                except (TypeError, ValueError):
                    pass
                break
        effect_mag = _classify_effect(result)

        label = analysis_id.replace("_", " ").title()
        parts = [f"DSW {analysis_type}/{label}"]
        if p_value is not None and math.isfinite(p_value):
            parts.append(f"p={p_value:.4f}")
        if effect_mag:
            parts.append(f"effect={effect_mag}")
        evidence_grade = result.get("evidence_grade")
        if evidence_grade:
            parts.append(f"grade={evidence_grade}")
        event_description = " — ".join(parts)

        study_quality_factors = {}
        if config.get("blinding"):
            study_quality_factors["blinding"] = config["blinding"]
        if config.get("pre_registration"):
            study_quality_factors["pre_registration"] = config["pre_registration"]

        from core.models import MeasurementSystem

        tool_output, _ = MeasurementSystem.objects.get_or_create(
            name=f"DSW {analysis_type}",
            owner=request.user,
            defaults={"system_type": "variable"},
        )

        spec = InferenceSpec(
            event_description=event_description,
            context={
                "analysis_type": analysis_type,
                "analysis_id": analysis_id,
                "p_value": p_value,
                "effect_magnitude": effect_mag,
                "evidence_grade": evidence_grade,
            },
            raw_output={
                "summary": result.get("summary", "")[:500],
                "statistics": stats,
            },
            sample_size=sample_size,
            study_quality_factors=study_quality_factors or None,
        )

        return connect_tool(
            investigation_id=investigation_id,
            tool_output=tool_output,
            tool_type="dsw",
            user=request.user,
            spec=spec,
        )

    except Exception:
        logger.exception("Analysis investigation bridge error")
        return None


# =============================================================================
# MAIN ENDPOINT
# =============================================================================


@require_http_methods(["POST"])
@gated
def run_analysis(request):
    """Run a statistical/ML analysis via the clean analysis/ package.

    Request body:
    {
        "type": "stats" | "ml" | "spc" | "viz" | ...,
        "analysis": "anova" | "regression" | etc,
        "config": {...},
        "data_id": "...",
        "project_id": "..." (optional),
        "title": "..." (optional),
        "save_result": true (optional)
    }

    DSW-001 §4.1: Stateless dispatch — data in, result out.
    DSW-001 §4.3: Single entry point through dispatch.py.
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        _log_rejection(request, "invalid_json")
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    analysis_type = body.get("type")
    analysis_id = body.get("analysis")
    config = body.get("config", {})
    project_id = body.get("project_id")
    investigation_id = body.get("investigation_id")
    result_title = body.get("title", "")
    save_result = body.get("save_result", False)

    start_time = time.time()

    try:
        # Resolve data (DSW-001 §4.2)
        df, error_response = _resolve_data(request, body, analysis_type, analysis_id)
        if error_response:
            return error_response

        # Dispatch to analysis handler (DSW-001 §4.3)
        result = _dispatch_analysis(analysis_type, df, analysis_id, config, request)
        if result is None:
            _log_rejection(request, f"unknown_analysis_type: {analysis_type}", analysis_type, analysis_id)
            return JsonResponse({"error": f"Unknown analysis type: {analysis_type}"}, status=400)

        latency = int((time.time() - start_time) * 1000)
        log_agent_action(request.user, "analysis", analysis_id, latency_ms=latency)

        logger.info(
            f"Analysis complete: {analysis_id}, plots: {len(result.get('plots', []))}, "
            f"summary length: {len(result.get('summary', ''))}"
        )

        # Persist result for A3 import (DSW-001 §6.2)
        if save_result or project_id:
            _persist_result(request, result, analysis_type, analysis_id, config, project_id, result_title)

        # Post-process: enforce canonical output schema (INIT-009 / E9-002)
        from .standardize import standardize_output

        result = standardize_output(result, analysis_type, analysis_id)

        # Investigation bridge (CANON-002 §12)
        if investigation_id:
            bridge_result = _connect_investigation(
                request=request,
                investigation_id=investigation_id,
                result=result,
                analysis_type=analysis_type,
                analysis_id=analysis_id,
                config=config,
            )
            if bridge_result:
                result["investigation_bridge"] = bridge_result

        return safe_json_response(result)

    except Exception as e:
        logger.exception(f"Analysis error: {e}")
        log_agent_action(request.user, "analysis", analysis_id, success=False, error_message=str(e))
        return JsonResponse({"error": str(e)}, status=500)
