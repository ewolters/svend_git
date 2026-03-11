"""DSW Analysis dispatcher -- routes analysis requests to category-specific handlers."""

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


def _log_rejection(request, reason, analysis_type=None, analysis_id=None):
    """Log DSW validation rejection to audit trail (FEAT-093, QUAL-001)."""
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


# =============================================================================
# ANALYSIS WORKBENCH ENDPOINTS
# =============================================================================


@require_http_methods(["POST"])
@gated
def run_analysis(request):
    """
    Run a statistical/ML analysis.

    Request body:
    {
        "type": "stats" | "ml" | "spc" | "viz",
        "analysis": "anova" | "regression" | etc,
        "config": {...},
        "data_id": "...",
        "project_id": "..." (optional - links result to a Project for A3 import),
        "title": "..." (optional - human-readable title for the result),
        "save_result": true (optional - persist result for later import)
    }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        _log_rejection(request, "invalid_json")
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    analysis_type = body.get("type")
    analysis_id = body.get("analysis")
    config = body.get("config", {})
    data_id = body.get("data_id")
    project_id = body.get("project_id")
    investigation_id = body.get("investigation_id")
    result_title = body.get("title", "")
    save_result = body.get("save_result", False)

    start_time = time.time()

    try:
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
                    return JsonResponse({"error": "Inline data limited to 10,000 rows"}, status=400)
            except Exception as e:
                _log_rejection(request, f"invalid_inline_data: {e}", analysis_type, analysis_id)
                return JsonResponse({"error": f"Invalid inline data: {e}"}, status=400)

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

        if df is None and analysis_type in ("simulation", "bayesian", "siop"):
            df = pd.DataFrame()  # Simulation can run without data (user-defined distributions)
        elif df is None:
            _log_rejection(request, "no_data_loaded", analysis_type, analysis_id)
            return JsonResponse({"error": "No data loaded. Please load a dataset first."}, status=400)

        result = {"plots": [], "summary": "", "guide_observation": ""}

        # Route to appropriate analysis
        if analysis_type == "stats":
            from .stats import run_statistical_analysis

            result = run_statistical_analysis(df, analysis_id, config)
        elif analysis_type == "ml":
            from .ml import run_ml_analysis

            result = run_ml_analysis(df, analysis_id, config, request.user)
        elif analysis_type == "spc":
            from .spc_pkg import run_spc_analysis

            result = run_spc_analysis(df, analysis_id, config)
        elif analysis_type == "viz":
            from .viz import run_visualization

            result = run_visualization(df, analysis_id, config)
        elif analysis_type == "bayesian":
            from .bayesian import run_bayesian_analysis

            result = run_bayesian_analysis(df, analysis_id, config)
        elif analysis_type == "reliability":
            from .reliability import run_reliability_analysis

            result = run_reliability_analysis(df, analysis_id, config)
        elif analysis_type == "simulation":
            from .simulation import run_simulation

            result = run_simulation(df, analysis_id, config, request.user)
        elif analysis_type == "causal":
            from ..causal_discovery import run_causal_discovery

            result = run_causal_discovery(df, analysis_id, config)
        elif analysis_type == "drift":
            from ..drift_detection import run_drift_detection

            result = run_drift_detection(df, analysis_id, config)
        elif analysis_type == "anytime":
            from ..anytime_valid import run_anytime_valid

            result = run_anytime_valid(df, analysis_id, config)
        elif analysis_type == "bayes_msa":
            from ..msa_bayes import run_bayes_msa

            result = run_bayes_msa(df, analysis_id, config)
        elif analysis_type == "quality_econ":
            from ..quality_economics import run_quality_econ

            result = run_quality_econ(df, analysis_id, config)
        elif analysis_type == "pbs":
            from ..pbs_engine import run_pbs

            result = run_pbs(df, analysis_id, config)
        elif analysis_type == "d_type":
            from .d_type import run_d_type

            result = run_d_type(df, analysis_id, config)
        elif analysis_type == "ishap":
            from ..interventional_shap import run_interventional_shap

            model_key = config.get("model_key", "")
            model_obj, model_feats = None, []
            if model_key:
                cached = get_cached_model(request.user.id if request.user else 0, model_key)
                if cached:
                    model_obj = cached.get("model")
                    model_feats = cached.get("meta", {}).get("features", [])
            result = run_interventional_shap(df, analysis_id, config, model=model_obj, model_features=model_feats)
        elif analysis_type == "siop":
            from .siop import run_siop

            result = run_siop(df, analysis_id, config)
        else:
            _log_rejection(request, f"unknown_analysis_type: {analysis_type}", analysis_type, analysis_id)
            return JsonResponse({"error": f"Unknown analysis type: {analysis_type}"}, status=400)

        latency = int((time.time() - start_time) * 1000)
        log_agent_action(request.user, "analysis", analysis_id, latency_ms=latency)

        logger.info(
            f"Analysis complete: {analysis_id}, plots: {len(result.get('plots', []))}, summary length: {len(result.get('summary', ''))}"
        )

        # Save result to database for A3/method import if requested
        if save_result or project_id:
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

        # Post-process: enforce canonical output schema (INIT-009 / E9-002)
        from .standardize import standardize_output

        result = standardize_output(result, analysis_type, analysis_id)

        # Investigation bridge (CANON-002 §12) — link DSW evidence to investigation
        if investigation_id:
            bridge_result = _dsw_connect_investigation(
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


def _dsw_connect_investigation(request, investigation_id, result, analysis_type, analysis_id, config):
    """Connect DSW analysis output to an investigation via the bridge (CANON-002 §12).

    Extracts statistical metrics from the standardized result, builds an
    InferenceSpec, and calls connect_tool() with tool_type="dsw".

    Returns bridge result dict or None on error.
    """
    from .standardize import _classify_effect, _extract_p_value

    try:
        from ..investigation_bridge import InferenceSpec, connect_tool

        # Extract statistical metrics from the standardized result
        p_value = _extract_p_value(result)
        stats = result.get("statistics", {})
        if not isinstance(stats, dict):
            stats = {}
        for _key in ("n", "n_obs", "n_total", "sample_size"):
            _val = stats.get(_key)
            if _val is not None:
                try:
                    sample_size = int(float(_val))
                except (TypeError, ValueError):
                    sample_size = None
                break
        else:
            sample_size = None
        effect_mag = _classify_effect(result)

        # Build human-readable event description
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

        # Study quality factors from config (optional)
        study_quality_factors = {}
        if config.get("blinding"):
            study_quality_factors["blinding"] = config["blinding"]
        if config.get("pre_registration"):
            study_quality_factors["pre_registration"] = config["pre_registration"]

        # Use MeasurementSystem as tool_output (UUID PK required by InvestigationToolLink)
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

        bridge_result = connect_tool(
            investigation_id=investigation_id,
            tool_output=tool_output,
            tool_type="dsw",
            user=request.user,
            spec=spec,
        )
        return bridge_result

    except Exception:
        logger.exception("DSW investigation bridge error")
        return None
