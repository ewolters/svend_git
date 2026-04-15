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
# SESSION INTEGRATION (new workbench models)
# =============================================================================


def _get_session(request, body):
    """Get session from request body or query param. Returns session or None."""
    from workbench.models import AnalysisSession

    session_id = body.get("session_id") or request.GET.get("session")
    if session_id:
        try:
            return AnalysisSession.objects.get(id=session_id, user=request.user)
        except AnalysisSession.DoesNotExist:
            pass
    return None


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

    # ── Persist to session (if session_id provided) ──
    analysis_record_id = None
    session = _get_session(request, body)
    if session:
        try:
            from workbench.models import SessionAnalysis, SessionDataset

            dataset_id = body.get("dataset_id")
            dataset = None
            if dataset_id:
                dataset = SessionDataset.objects.filter(id=dataset_id, session=session).first()

            record = SessionAnalysis.objects.create(
                session=session,
                dataset=dataset,
                analysis_type=analysis_type,
                analysis_id=analysis_id,
                columns_used=list(config.values()) if isinstance(config, dict) else [],
                config=config,
                statistics=result.get("statistics", {}),
                narrative=result.get("narrative", {}),
                summary=result.get("summary", ""),
                charts=result.get("plots", result.get("charts", [])),
                diagnostics=result.get("diagnostics", []),
                assumptions=result.get("assumptions", {}),
                education=result.get("education"),
                bayesian_shadow=result.get("bayesian_shadow"),
                evidence_grade=result.get("evidence_grade", ""),
                guide_observation=result.get("guide_observation", ""),
            )
            analysis_record_id = str(record.id)
            session.save()  # touch updated_at
        except Exception:
            logger.warning("Session analysis persistence failed", exc_info=True)

    result["_analysis_record_id"] = analysis_record_id

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

    # ── Persist dataset to session (if session_id provided) ──
    dataset_record_id = None
    session_id = request.POST.get("session_id") or request.GET.get("session")
    if session_id:
        try:
            from workbench.models import AnalysisSession, SessionDataset

            session = AnalysisSession.objects.get(id=session_id, user=request.user)

            col_meta = []
            for col in columns:
                dtype = str(df[col].dtype)
                is_num = dtype.startswith(("int", "float"))
                col_meta.append(
                    {
                        "name": col,
                        "dtype": "numeric" if is_num else "categorical",
                    }
                )

            ds = SessionDataset.objects.create(
                session=session,
                name=f.name or data_id,
                source="upload",
                data=df.to_dict(orient="list"),
                columns_meta=col_meta,
                row_count=rows,
            )
            dataset_record_id = str(ds.id)
            session.save()
        except Exception:
            logger.warning("Session dataset persistence failed", exc_info=True)

    return JsonResponse(
        {
            "data_id": data_id,
            "columns": columns,
            "rows": rows,
            "preview": preview,
            "_dataset_record_id": dataset_record_id,
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

    # ── Persist to session (if session_id provided) ──
    analysis_record_id = None
    wb_session = _get_session(request, body)
    if wb_session and result.success and (result.data or charts):
        try:
            from workbench.models import SessionAnalysis

            record = SessionAnalysis.objects.create(
                session=wb_session,
                analysis_type="forgepad",
                analysis_id=parsed.verb,
                config={"command": command},
                statistics=result.data or {},
                summary=result.summary or "",
                charts=charts,
            )
            analysis_record_id = str(record.id)
            wb_session.save()
        except Exception:
            logger.warning("ForgePad session persistence failed", exc_info=True)

    return JsonResponse(
        {
            "success": result.success,
            "summary": result.summary,
            "error": result.error,
            "charts": charts,
            "statistics": result.data,
            "_analysis_record_id": analysis_record_id,
        }
    )
