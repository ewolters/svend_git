"""DSW Analysis dispatcher -- routes analysis requests to category-specific handlers."""

import json
import logging
import re
import time
import uuid
import tempfile
from pathlib import Path

import numpy as np

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings

from accounts.permissions import gated
from ..models import DSWResult
from .common import log_agent_action, get_cached_model


logger = logging.getLogger(__name__)


# =============================================================================
# ANALYSIS WORKBENCH ENDPOINTS
# =============================================================================

@csrf_exempt
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
        "problem_id": "..." (optional - links results as evidence to a Problem),
        "project_id": "..." (optional - links result to a Project for A3 import),
        "title": "..." (optional - human-readable title for the result),
        "save_result": true (optional - persist result for later import)
    }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    analysis_type = body.get("type")
    analysis_id = body.get("analysis")
    config = body.get("config", {})
    data_id = body.get("data_id")
    project_id = body.get("project_id")
    result_title = body.get("title", "")
    save_result = body.get("save_result", False)

    start_time = time.time()

    try:
        import pandas as pd
        import numpy as np
        from io import StringIO

        df = None

        # Source 0: Inline data (from learning module or API callers)
        inline_data = body.get("data")
        if inline_data and isinstance(inline_data, dict):
            try:
                df = pd.DataFrame(inline_data)
                if len(df) > 10000:
                    return JsonResponse({"error": "Inline data limited to 10,000 rows"}, status=400)
            except Exception as e:
                return JsonResponse({"error": f"Invalid inline data: {e}"}, status=400)

        # Source 1: Uploaded via upload_data endpoint (data_xxx format)
        if df is None and data_id and data_id.startswith("data_"):
            try:
                data_dir = Path(settings.MEDIA_ROOT) / "analysis_data" / str(request.user.id)
                data_path = data_dir / f"{data_id}.csv"
                if data_path.exists():
                    df = pd.read_csv(data_path)
            except Exception:
                pass

            # Fallback to temp directory
            if df is None:
                try:
                    data_dir = Path(tempfile.gettempdir()) / "svend_analysis"
                    data_path = data_dir / f"{data_id}.csv"
                    if data_path.exists():
                        df = pd.read_csv(data_path)
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

        if df is None and analysis_type == "simulation":
            df = pd.DataFrame()  # Simulation can run without data (user-defined distributions)
        elif df is None:
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
            from .spc import run_spc_analysis
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
        elif analysis_type == "ishap":
            from ..interventional_shap import run_interventional_shap
            model_key = config.get("model_key", "")
            model_obj, model_feats = None, []
            if model_key:
                cached = get_cached_model(request.user.id if request.user else 0, model_key)
                if cached:
                    model_obj = cached.get("model")
                    model_feats = cached.get("meta", {}).get("features", [])
            result = run_interventional_shap(df, analysis_id, config,
                                            model=model_obj, model_features=model_feats)
        else:
            return JsonResponse({"error": f"Unknown analysis type: {analysis_type}"}, status=400)

        latency = int((time.time() - start_time) * 1000)
        log_agent_action(request.user, "analysis", analysis_id, latency_ms=latency)

        logger.info(f"Analysis complete: {analysis_id}, plots: {len(result.get('plots', []))}, summary length: {len(result.get('summary', ''))}")

        # Link results as evidence to a Problem (if problem_id provided)
        problem_id = body.get("problem_id")
        if problem_id:
            try:
                from ..views import add_finding_to_problem

                # Use guide_observation if set, otherwise build from analysis_id + summary
                observation = result.get("guide_observation", "")
                if not observation:
                    summary_text = result.get("summary", "")
                    # Strip color tags and truncate for evidence summary
                    clean = re.sub(r"<<COLOR:\w+>>|<</COLOR>>", "", summary_text)
                    observation = f"DSW {analysis_id}: {clean[:200]}" if clean else f"DSW analysis: {analysis_id}"

                evidence_type = {
                    "stats": "data_analysis",
                    "ml": "data_analysis",
                    "bayesian": "data_analysis",
                    "spc": "data_analysis",
                    "viz": "observation",
                }.get(analysis_type, "data_analysis")

                evidence = add_finding_to_problem(
                    user=request.user,
                    problem_id=problem_id,
                    summary=observation,
                    evidence_type=evidence_type,
                    source=f"DSW ({analysis_id})",
                )
                if evidence:
                    result["problem_updated"] = True
                    result["evidence_id"] = evidence["id"]
            except Exception as e:
                logger.warning(f"Could not link DSW result to problem {problem_id}: {e}")

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
                    data=json.dumps({
                        "analysis_type": analysis_type,
                        "analysis_id": analysis_id,
                        "config": config,
                        "summary": result.get("summary", ""),
                        "guide_observation": result.get("guide_observation", ""),
                        "plots_count": len(result.get("plots", [])),
                    }),
                    project=project,
                    title=result_title or f"{analysis_id.replace('_', ' ').title()} Analysis",
                )
                result["result_id"] = result_id
            except Exception as e:
                logger.warning(f"Could not save DSW result: {e}")

        return JsonResponse(result)

    except Exception as e:
        logger.exception(f"Analysis error: {e}")
        log_agent_action(request.user, "analysis", analysis_id, success=False, error_message=str(e))
        return JsonResponse({"error": str(e)}, status=500)

