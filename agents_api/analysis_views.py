"""Forge-only analysis views for the new workbench.

This module calls forge handlers DIRECTLY. No legacy dispatch, no shims,
no fallback. Every analysis creates an Artifact on a Workbench. Every
upload creates a dataset Artifact. Nothing is stateless.

CR: 9999588b, c0d36833
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
# WORKBENCH INTEGRATION
# =============================================================================

# Map (analysis_type, analysis_id pattern) → ArtifactType
_ARTIFACT_TYPE_MAP = {
    # SPC
    "spc": "spc_chart",
    # Specific overrides
    ("spc", "capability"): "capability_study",
    ("spc", "nonnormal_capability"): "capability_study",
    ("spc", "degradation_capability"): "capability_study",
    ("spc", "gage_rr"): "descriptive_stats",  # MSA
    # Stats
    ("stats", "anova"): "anova",
    ("stats", "anova_twoway"): "anova",
    ("stats", "repeated_measures"): "anova",
    ("stats", "regression"): "regression",
    ("stats", "multiple_regression"): "regression",
    ("stats", "logistic"): "regression",
    ("stats", "robust_regression"): "regression",
    ("stats", "stepwise"): "regression",
    ("stats", "best_subsets"): "regression",
    ("stats", "glm"): "regression",
    ("stats", "poisson_regression"): "regression",
    ("stats", "nonlinear"): "regression",
    ("stats", "correlation"): "correlation",
    ("stats", "partial_correlation"): "correlation",
    ("stats", "spearman"): "correlation",
    ("stats", "descriptive"): "descriptive_stats",
    ("stats", "graphical_summary"): "descriptive_stats",
    ("stats", "data_profile"): "descriptive_stats",
    # ML
    "ml": "ml_model",
    # Bayesian
    "bayesian": "hypothesis_test",
    # Default for stats
    "stats": "hypothesis_test",
    # Reliability, forecast, etc.
    "reliability": "hypothesis_test",
    "simulation": "hypothesis_test",
    "causal": "hypothesis_test",
    "drift": "hypothesis_test",
    "anytime": "hypothesis_test",
    "quality_econ": "hypothesis_test",
    "pbs": "hypothesis_test",
    "ishap": "hypothesis_test",
    "bayes_msa": "descriptive_stats",
    "d_type": "hypothesis_test",
}


def _resolve_artifact_type(analysis_type, analysis_id):
    """Map analysis type/id to the best-matching ArtifactType value."""
    # Try specific (type, id) first
    key = (analysis_type, analysis_id)
    if key in _ARTIFACT_TYPE_MAP:
        return _ARTIFACT_TYPE_MAP[key]
    # Fall back to type-level default
    if analysis_type in _ARTIFACT_TYPE_MAP:
        return _ARTIFACT_TYPE_MAP[analysis_type]
    return "hypothesis_test"


def _get_or_create_workbench(request, body):
    """Get workbench from request, or create an ephemeral one.

    Returns (workbench, created_bool). The workbench_id can come from:
    - body["workbench_id"] — explicit
    - GET param ?workbench=<id> — from URL
    - Auto-create if neither provided
    """
    from workbench.models import Workbench

    wb_id = body.get("workbench_id") or request.GET.get("workbench")

    if wb_id:
        try:
            wb = Workbench.objects.get(id=wb_id, user=request.user)
            return wb, False
        except Workbench.DoesNotExist:
            pass

    # Auto-create a blank workbench for this session
    wb = Workbench.objects.create(
        user=request.user,
        title="Analysis Session",
        template="blank",
    )
    return wb, True


def _create_artifact(workbench, artifact_type, title, content, source="forge", source_artifact_id=None, tags=None):
    """Create an Artifact on a Workbench and return it."""
    from workbench.models import Artifact

    artifact = Artifact.objects.create(
        workbench=workbench,
        artifact_type=artifact_type,
        title=title,
        content=content,
        source=source,
        source_artifact_id=source_artifact_id,
        tags=tags or [],
    )
    return artifact


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
    if analysis_type in ("simulation", "bayesian"):
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
    """Forge-only analysis endpoint. Creates an Artifact on the Workbench."""
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

    # ── Create Artifact on Workbench ──
    artifact_id = None
    workbench_id = None
    try:
        wb, _created = _get_or_create_workbench(request, body)
        workbench_id = str(wb.id)

        artifact_type = _resolve_artifact_type(analysis_type, analysis_id)
        title = f"{analysis_id.replace('_', ' ').title()}"
        if body.get("data_id"):
            title += f" — {body['data_id']}"

        # Find source dataset artifact (if data_id matches one on this workbench)
        source_artifact_id = None
        data_id = body.get("data_id")
        if data_id:
            from workbench.models import Artifact as WBArtifact

            source = (
                WBArtifact.objects.filter(
                    workbench=wb,
                    artifact_type__in=["dataset", "cleaned_dataset"],
                    content__data_id=data_id,
                )
                .order_by("-created_at")
                .first()
            )
            if source:
                source_artifact_id = source.id

        artifact = _create_artifact(
            workbench=wb,
            artifact_type=artifact_type,
            title=title,
            content=result,
            source="forge",
            source_artifact_id=source_artifact_id,
            tags=[analysis_type, analysis_id],
        )
        artifact_id = str(artifact.id)
    except Exception:
        logger.warning("Artifact creation failed, returning result anyway", exc_info=True)

    # Include artifact/workbench IDs in response
    result["_artifact_id"] = artifact_id
    result["_workbench_id"] = workbench_id

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

    # ── Create dataset Artifact on Workbench ──
    artifact_id = None
    workbench_id = None
    try:
        # Parse workbench_id from form data or query param
        wb_id = request.POST.get("workbench_id") or request.GET.get("workbench")
        wb_body = {"workbench_id": wb_id} if wb_id else {}
        wb, _created = _get_or_create_workbench(request, wb_body)
        workbench_id = str(wb.id)

        # Column type info for the dataset record
        variables = []
        for col in columns:
            dtype = str(df[col].dtype)
            variables.append({"name": col, "dtype": dtype})

        dataset_content = {
            "data_id": data_id,
            "file_name": f.name,
            "rows": rows,
            "columns": columns,
            "variables": variables,
            "preview": preview,
        }

        artifact = _create_artifact(
            workbench=wb,
            artifact_type="dataset",
            title=f.name or data_id,
            content=dataset_content,
            source="upload",
        )
        artifact_id = str(artifact.id)

        # Also update workbench.datasets list
        wb.datasets.append(
            {
                "name": f.name or data_id,
                "data_id": data_id,
                "artifact_id": artifact_id,
                "rows": rows,
                "columns": columns,
                "variables": variables,
            }
        )
        wb.save(update_fields=["datasets", "updated_at"])

    except Exception:
        logger.warning("Dataset artifact creation failed", exc_info=True)

    return JsonResponse(
        {
            "data_id": data_id,
            "columns": columns,
            "rows": rows,
            "preview": preview,
            "_artifact_id": artifact_id,
            "_workbench_id": workbench_id,
        }
    )


# =============================================================================
# FORGEPAD — command-driven analysis
# =============================================================================

# One session per user (in-memory, resets on server restart).
_forgepad_sessions: dict = {}


@require_http_methods(["POST"])
@gated
def forgepad_run(request):
    """Execute a ForgePad command against the user's session."""
    from forgepad.executor import execute
    from forgepad.parser import parse
    from forgepad.session import Session

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    command = body.get("command", "").strip()
    if not command:
        return JsonResponse({"error": "Empty command"}, status=400)

    data_id = body.get("data_id")

    # Get or create session for this user
    uid = request.user.id
    session = _forgepad_sessions.get(uid)
    if session is None:
        session = Session()
        _forgepad_sessions[uid] = session

    # If data_id changed or session has no data, reload
    if data_id and session.metadata.get("data_id") != data_id:
        df, err = _resolve_data(request, {"data_id": data_id, "type": ""})
        if err:
            return err
        if df is not None and not df.empty:
            session.load_data(df.to_dict(orient="list"))
            session.metadata["data_id"] = data_id

    # Parse and execute
    parsed = parse(command)
    result = execute(session, parsed)

    # Store in session history
    session.history.append(result)
    if result.result_name:
        session.named_results[result.result_name] = result

    # Serialize charts (ChartSpec → dict for ForgeViz)
    charts = []
    for chart in result.charts:
        if hasattr(chart, "to_dict"):
            charts.append(chart.to_dict())
        elif isinstance(chart, dict):
            charts.append(chart)

    # ── Create Artifact on Workbench (if command produced data) ──
    artifact_id = None
    workbench_id = None
    if result.success and (result.data or charts):
        try:
            wb, _created = _get_or_create_workbench(request, body)
            workbench_id = str(wb.id)

            content = {
                "command": command,
                "summary": result.summary,
                "statistics": result.data,
                "plots": charts,
            }
            artifact = _create_artifact(
                workbench=wb,
                artifact_type="hypothesis_test",
                title=f"ForgePad: {parsed.verb}",
                content=content,
                source="forgepad",
                tags=["forgepad", parsed.verb],
            )
            artifact_id = str(artifact.id)
        except Exception:
            logger.warning("ForgePad artifact creation failed", exc_info=True)

    return JsonResponse(
        {
            "success": result.success,
            "summary": result.summary,
            "error": result.error,
            "charts": charts,
            "statistics": result.data,
            "_artifact_id": artifact_id,
            "_workbench_id": workbench_id,
        }
    )
