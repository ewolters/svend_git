"""DSW (Decision Science Workbench) API views.

Thin routing layer — all analysis engines live in dsw/ submodules.
Endpoints here delegate to dsw/dispatch.py, dsw/endpoints_data.py,
and dsw/endpoints_ml.py.
"""

import json
import logging
import tempfile
from pathlib import Path

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated, gated_paid, require_auth, require_enterprise

logger = logging.getLogger(__name__)


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
    """Route analysis requests through dsw/dispatch — Phase 1 of monolith split."""
    from .dsw.dispatch import run_analysis as _dispatch_analysis

    return _dispatch_analysis(request)


def _load_dataset(user, data_id):
    """Load a dataset by data_id for the given user. Returns DataFrame or None."""
    from io import StringIO

    import pandas as pd

    if not data_id:
        return None

    df = None
    # Source 1: Uploaded via upload_data endpoint
    if data_id.startswith("data_"):
        try:
            data_dir = Path(settings.MEDIA_ROOT) / "analysis_data" / str(user.id)
            data_path = data_dir / f"{data_id}.csv"
            if data_path.exists():
                df = _read_csv_safe(data_path)
        except Exception:
            pass
        if df is None:
            try:
                data_dir = Path(tempfile.gettempdir()) / "svend_analysis"
                data_path = data_dir / f"{data_id}.csv"
                if data_path.exists():
                    df = _read_csv_safe(data_path)
            except Exception:
                pass

    # Source 2: Triage cleaned dataset
    if df is None:
        try:
            from .models import TriageResult

            triage_result = TriageResult.objects.get(id=data_id, user=user)
            df = pd.read_csv(StringIO(triage_result.cleaned_csv))
        except Exception:
            pass

    return df


@require_http_methods(["POST"])
@gated_paid
def explain_selection(request):
    """Explain what selected data points have in common using LLM.

    Request body:
    {
        "data_id": "data_xxx",
        "indices": [3, 7, 12, 15],
        "analysis_context": "regression diagnostics"
    }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    data_id = body.get("data_id")
    indices = body.get("indices", [])
    context = body.get("analysis_context", "")

    if not data_id or not indices:
        return JsonResponse({"error": "Missing data_id or indices"}, status=400)
    if len(indices) > 100:
        return JsonResponse({"error": "Too many points selected (max 100)"}, status=400)

    df = _load_dataset(request.user, data_id)
    if df is None:
        return JsonResponse({"error": "Dataset not found"}, status=404)

    valid_idx = [i for i in indices if 0 <= i < len(df)]
    if not valid_idx:
        return JsonResponse({"error": "No valid indices"}, status=400)

    selected = df.iloc[valid_idx]
    remaining = df.drop(df.index[valid_idx])

    sel_desc = selected.describe(include="all").to_string()
    rem_desc = remaining.describe(include="all").to_string() if len(remaining) > 0 else "N/A"
    sel_sample = selected.head(20).to_string()

    prompt = (
        f"The user selected {len(valid_idx)} data points out of {len(df)} total "
        f"in a {context or 'statistical analysis'}.\n\n"
        f"Selected rows (sample, up to 20):\n{sel_sample}\n\n"
        f"Selected summary statistics:\n{sel_desc}\n\n"
        f"Remaining data summary statistics:\n{rem_desc}\n\n"
        "What distinguishes the selected points from the rest? Look for:\n"
        "- Common categorical values (same machine, shift, batch, operator, etc.)\n"
        "- Numeric patterns (extreme values, specific ranges)\n"
        "- Temporal patterns (same time period, sequential)\n\n"
        "Be specific and cite column names and values. 2-3 sentences max."
    )

    try:
        from .llm_manager import LLMManager

        response = LLMManager.chat(
            user=request.user,
            messages=[{"role": "user", "content": prompt}],
            system="You are a data analyst explaining patterns in selected data points. Be concise, specific, and cite evidence from the data.",
            max_tokens=300,
            temperature=0.3,
        )
        return JsonResponse({"explanation": response})
    except Exception as e:
        logger.warning(f"Explain selection LLM call failed: {e}")
        return JsonResponse({"error": "Explanation service unavailable"}, status=503)


@require_http_methods(["GET"])
@require_auth
def hypothesis_timeline(request):
    """Return probability history for a hypothesis, enriched with evidence summaries.

    Query params: project_id, hypothesis_id
    """
    from core.models import Evidence, Hypothesis

    project_id = request.GET.get("project_id")
    hypothesis_id = request.GET.get("hypothesis_id")
    if not project_id or not hypothesis_id:
        return JsonResponse({"error": "Missing project_id or hypothesis_id"}, status=400)

    hypothesis = Hypothesis.objects.filter(id=hypothesis_id, project_id=project_id, project__user=request.user).first()
    if not hypothesis:
        return JsonResponse({"error": "Hypothesis not found"}, status=404)

    history = hypothesis.probability_history or []
    evidence_ids = [h.get("evidence_id") for h in history if h.get("evidence_id")]
    evidence_map = {}
    if evidence_ids:
        evidence_map = {
            str(e.id): {"summary": e.summary, "source_type": e.source_type, "p_value": e.p_value}
            for e in Evidence.objects.filter(id__in=evidence_ids)
        }

    timeline = []
    for entry in history:
        point = {
            "probability": entry.get("probability"),
            "previous": entry.get("previous"),
            "timestamp": entry.get("timestamp"),
            "strength": entry.get("strength"),
            "likelihood_ratio": entry.get("likelihood_ratio"),
        }
        eid = entry.get("evidence_id")
        if eid and eid in evidence_map:
            point["evidence"] = evidence_map[eid]
        timeline.append(point)

    return JsonResponse(
        {
            "hypothesis": {
                "id": str(hypothesis.id),
                "statement": hypothesis.statement
                if hasattr(hypothesis, "statement")
                else str(hypothesis.description or ""),
                "prior": float(hypothesis.prior_probability),
                "current": float(hypothesis.current_probability),
                "status": hypothesis.status,
            },
            "timeline": timeline,
            "confirmation_threshold": float(getattr(hypothesis, "confirmation_threshold", 0.95)),
            "rejection_threshold": float(getattr(hypothesis, "rejection_threshold", 0.05)),
        }
    )


# =============================================================================
# ROUTING STUBS — delegate to dsw/endpoints_data.py
# =============================================================================


@require_http_methods(["POST"])
@require_auth
def upload_data(request):
    """Route to dsw/endpoints_data — Phase 5 of monolith split."""
    from .dsw.endpoints_data import upload_data as _ep_upload_data

    return _ep_upload_data(request)


@require_http_methods(["POST"])
@gated
def execute_code(request):
    """Route to dsw/endpoints_data — Phase 5 of monolith split."""
    from .dsw.endpoints_data import execute_code as _ep_execute_code

    return _ep_execute_code(request)


@require_http_methods(["POST"])
@gated
def generate_code(request):
    """Route to dsw/endpoints_data — Phase 5 of monolith split."""
    from .dsw.endpoints_data import generate_code as _ep_generate_code

    return _ep_generate_code(request)


@require_http_methods(["POST"])
@require_enterprise
def analyst_assistant(request):
    """Route to dsw/endpoints_data — Phase 5 of monolith split."""
    from .dsw.endpoints_data import analyst_assistant as _ep_analyst_assistant

    return _ep_analyst_assistant(request)


# ============================================================================
# DATA TRANSFORMATION TOOLS
# ============================================================================


@require_http_methods(["POST"])
@gated
def transform_data(request):
    """Route to dsw/endpoints_data — Phase 5 of monolith split."""
    from .dsw.endpoints_data import transform_data as _ep_transform_data

    return _ep_transform_data(request)


@require_http_methods(["POST"])
@require_auth
def download_data(request):
    """Route to dsw/endpoints_data — Phase 5 of monolith split."""
    from .dsw.endpoints_data import download_data as _ep_download_data

    return _ep_download_data(request)


@require_http_methods(["POST"])
@require_auth
def retrieve_data(request):
    """Route to dsw/endpoints_data — retrieve saved dataset by data_id."""
    from .dsw.endpoints_data import retrieve_data as _ep_retrieve_data

    return _ep_retrieve_data(request)


@require_http_methods(["POST"])
@gated
def triage_data(request):
    """Route to dsw/endpoints_data — Phase 5 of monolith split."""
    from .dsw.endpoints_data import triage_data as _ep_triage_data

    return _ep_triage_data(request)


@require_http_methods(["POST"])
@gated
def triage_scan(request):
    """Route to dsw/endpoints_data — Phase 5 of monolith split."""
    from .dsw.endpoints_data import triage_scan as _ep_triage_scan

    return _ep_triage_scan(request)
