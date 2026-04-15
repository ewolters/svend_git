"""Analysis Workbench views — session-based persistence.

CRUD for sessions, datasets, and analyses. Pull contract endpoints
for cross-tool integration (manifest + sub-artifact access).
"""

import json
import logging

from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated, require_auth
from qms_core import pull_views

from .models import AnalysisSession, SessionAnalysis, SessionDataset

logger = logging.getLogger(__name__)


# =============================================================================
# Pull contract helpers
# =============================================================================


def _wb_get_session(request, pk):
    """Get a session filtered by user (+ tenant)."""
    try:
        return AnalysisSession.objects.get(id=pk, user=request.user)
    except AnalysisSession.DoesNotExist:
        return None


def _wb_get_analysis(request, artifact_id):
    """Get an analysis owned by this user."""
    try:
        return SessionAnalysis.objects.select_related("session").get(
            id=artifact_id,
            session__user=request.user,
        )
    except SessionAnalysis.DoesNotExist:
        return None


def _wb_analysis_to_dict(request, artifact_id):
    """Get analysis as dict for pull_artifact_detail."""
    obj = _wb_get_analysis(request, artifact_id)
    if obj is None:
        return None
    return {
        "id": str(obj.id),
        "analysis_type": obj.analysis_type,
        "analysis_id": obj.analysis_id,
        "dataset_id": str(obj.dataset_id) if obj.dataset_id else None,
        "columns_used": obj.columns_used,
        "config": obj.config,
        "statistics": obj.statistics,
        "narrative": obj.narrative,
        "summary": obj.summary,
        "charts": obj.charts,
        "diagnostics": obj.diagnostics,
        "assumptions": obj.assumptions,
        "education": obj.education,
        "bayesian_shadow": obj.bayesian_shadow,
        "evidence_grade": obj.evidence_grade,
        "guide_observation": obj.guide_observation,
        "created_at": obj.created_at.isoformat(),
    }


def _wb_sub_artifact(obj, key_path):
    """Delegate to model's get_sub_artifact."""
    return obj.get_sub_artifact(key_path)


def _wb_delete_session(obj):
    """Delete a session."""
    obj.delete()


# =============================================================================
# Sessions
# =============================================================================


@require_http_methods(["GET", "POST"])
@require_auth
def session_list_create(request):
    """List user's sessions or create a new one."""
    if request.method == "GET":
        sessions = AnalysisSession.objects.filter(user=request.user).order_by("-updated_at")[:50]
        return JsonResponse(
            {
                "sessions": [
                    {
                        "id": str(s.id),
                        "title": s.title,
                        "dataset_count": s.datasets.count(),
                        "analysis_count": s.analyses.count(),
                        "updated_at": s.updated_at.isoformat(),
                        "created_at": s.created_at.isoformat(),
                    }
                    for s in sessions
                ]
            }
        )

    data = json.loads(request.body)
    session = AnalysisSession.objects.create(
        user=request.user,
        title=data.get("title", ""),
        description=data.get("description", ""),
    )
    return JsonResponse({"id": str(session.id), "title": session.title}, status=201)


@require_http_methods(["GET", "PATCH", "DELETE"])
@require_auth
def session_detail(request, session_id):
    """Get, update, or delete a session."""
    session = get_object_or_404(AnalysisSession, id=session_id, user=request.user)

    if request.method == "GET":
        return JsonResponse(
            {
                "id": str(session.id),
                "title": session.title,
                "description": session.description,
                "datasets": [
                    {
                        "id": str(d.id),
                        "name": d.name,
                        "source": d.source,
                        "row_count": d.row_count,
                        "columns_meta": d.columns_meta,
                        "created_at": d.created_at.isoformat(),
                    }
                    for d in session.datasets.all()
                ],
                "analyses": [
                    {
                        "id": str(a.id),
                        "analysis_type": a.analysis_type,
                        "analysis_id": a.analysis_id,
                        "dataset_id": str(a.dataset_id) if a.dataset_id else None,
                        "columns_used": a.columns_used,
                        "evidence_grade": a.evidence_grade,
                        "summary": a.summary[:200],
                        "created_at": a.created_at.isoformat(),
                    }
                    for a in session.analyses.all()
                ],
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
            }
        )

    if request.method == "PATCH":
        data = json.loads(request.body)
        if "title" in data:
            session.title = data["title"]
        if "description" in data:
            session.description = data["description"]
        session.save()
        return JsonResponse({"id": str(session.id), "title": session.title})

    session.delete()
    return JsonResponse({"deleted": True})


# =============================================================================
# Datasets
# =============================================================================


@require_http_methods(["GET", "POST"])
@require_auth
def dataset_list_create(request, session_id):
    """List datasets in a session or add a new one."""
    session = get_object_or_404(AnalysisSession, id=session_id, user=request.user)

    if request.method == "GET":
        return JsonResponse(
            {
                "datasets": [
                    {
                        "id": str(d.id),
                        "name": d.name,
                        "source": d.source,
                        "row_count": d.row_count,
                        "columns_meta": d.columns_meta,
                    }
                    for d in session.datasets.all()
                ]
            }
        )

    data = json.loads(request.body)
    dataset = SessionDataset.objects.create(
        session=session,
        name=data.get("name", "Untitled"),
        source=data.get("source", SessionDataset.Source.UPLOAD),
        data=data.get("data", {}),
        columns_meta=data.get("columns_meta", []),
        row_count=data.get("row_count", 0),
    )

    parent_ids = data.get("parent_dataset_ids", [])
    if parent_ids:
        parents = SessionDataset.objects.filter(id__in=parent_ids, session=session)
        dataset.parent_datasets.set(parents)

    session.save()
    return JsonResponse(
        {
            "id": str(dataset.id),
            "name": dataset.name,
            "row_count": dataset.row_count,
        },
        status=201,
    )


@require_http_methods(["GET", "DELETE"])
@require_auth
def dataset_detail(request, session_id, dataset_id):
    """Get full dataset (with data) or delete it."""
    session = get_object_or_404(AnalysisSession, id=session_id, user=request.user)
    dataset = get_object_or_404(SessionDataset, id=dataset_id, session=session)

    if request.method == "GET":
        return JsonResponse(
            {
                "id": str(dataset.id),
                "name": dataset.name,
                "source": dataset.source,
                "data": dataset.data,
                "columns_meta": dataset.columns_meta,
                "row_count": dataset.row_count,
                "parent_datasets": [str(p.id) for p in dataset.parent_datasets.all()],
            }
        )

    dataset.delete()
    return JsonResponse({"deleted": True})


# =============================================================================
# Analyses
# =============================================================================


@require_http_methods(["GET", "POST"])
@gated
def analysis_list_create(request, session_id):
    """List analyses in a session or record a new result."""
    session = get_object_or_404(AnalysisSession, id=session_id, user=request.user)

    if request.method == "GET":
        return JsonResponse(
            {
                "analyses": [
                    {
                        "id": str(a.id),
                        "analysis_type": a.analysis_type,
                        "analysis_id": a.analysis_id,
                        "dataset_id": str(a.dataset_id) if a.dataset_id else None,
                        "columns_used": a.columns_used,
                        "statistics": a.statistics,
                        "narrative": a.narrative,
                        "summary": a.summary,
                        "charts": a.charts,
                        "diagnostics": a.diagnostics,
                        "assumptions": a.assumptions,
                        "education": a.education,
                        "bayesian_shadow": a.bayesian_shadow,
                        "evidence_grade": a.evidence_grade,
                        "guide_observation": a.guide_observation,
                        "created_at": a.created_at.isoformat(),
                    }
                    for a in session.analyses.all()
                ]
            }
        )

    data = json.loads(request.body)
    dataset_id = data.get("dataset_id")
    dataset = None
    if dataset_id:
        dataset = get_object_or_404(SessionDataset, id=dataset_id, session=session)

    analysis = SessionAnalysis.objects.create(
        session=session,
        dataset=dataset,
        analysis_type=data.get("analysis_type", ""),
        analysis_id=data.get("analysis_id", ""),
        columns_used=data.get("columns_used", []),
        config=data.get("config", {}),
        statistics=data.get("statistics", {}),
        narrative=data.get("narrative", {}),
        summary=data.get("summary", ""),
        charts=data.get("charts", []),
        diagnostics=data.get("diagnostics", []),
        assumptions=data.get("assumptions", {}),
        education=data.get("education"),
        bayesian_shadow=data.get("bayesian_shadow"),
        evidence_grade=data.get("evidence_grade", ""),
        guide_observation=data.get("guide_observation", ""),
    )

    session.save()
    return JsonResponse({"id": str(analysis.id)}, status=201)


@require_http_methods(["GET", "DELETE"])
@require_auth
def analysis_detail(request, session_id, analysis_id):
    """Get full analysis result or delete it."""
    session = get_object_or_404(AnalysisSession, id=session_id, user=request.user)
    analysis = get_object_or_404(SessionAnalysis, id=analysis_id, session=session)

    if request.method == "GET":
        return JsonResponse(
            {
                "id": str(analysis.id),
                "analysis_type": analysis.analysis_type,
                "analysis_id": analysis.analysis_id,
                "dataset_id": str(analysis.dataset_id) if analysis.dataset_id else None,
                "columns_used": analysis.columns_used,
                "config": analysis.config,
                "statistics": analysis.statistics,
                "narrative": analysis.narrative,
                "summary": analysis.summary,
                "charts": analysis.charts,
                "diagnostics": analysis.diagnostics,
                "assumptions": analysis.assumptions,
                "education": analysis.education,
                "bayesian_shadow": analysis.bayesian_shadow,
                "evidence_grade": analysis.evidence_grade,
                "guide_observation": analysis.guide_observation,
                "created_at": analysis.created_at.isoformat(),
            }
        )

    analysis.delete()
    return JsonResponse({"deleted": True})


# =============================================================================
# Pull contract — manifest + sub-artifact access
# =============================================================================


@require_http_methods(["GET"])
@require_auth
def session_manifest(request, session_id):
    """Return the pullable manifest for this session.

    Other tools browse this to decide what to reference.
    """
    session = get_object_or_404(AnalysisSession, id=session_id, user=request.user)
    return JsonResponse(session.get_manifest())


@require_http_methods(["GET"])
@require_auth
def analysis_sub_artifact(request, session_id, analysis_id, key_path):
    """Access a specific sub-artifact by key path.

    Examples:
        GET /sessions/<id>/analyses/<id>/statistics/p_value/
        GET /sessions/<id>/analyses/<id>/charts/0/
        GET /sessions/<id>/analyses/<id>/narrative/verdict/
    """
    session = get_object_or_404(AnalysisSession, id=session_id, user=request.user)
    analysis = get_object_or_404(SessionAnalysis, id=analysis_id, session=session)

    value = analysis.get_sub_artifact(key_path)
    if value is None:
        return JsonResponse({"error": f"Sub-artifact not found: {key_path}"}, status=404)

    return JsonResponse({"key": key_path, "value": value})


# =============================================================================
# Pull contract — reference registration + delete with friction
# =============================================================================


def session_references(request, session_id):
    """List references pointing to this session's artifacts."""
    return pull_views.pull_list_references(
        request,
        session_id,
        source_app="workbench",
        source_type="AnalysisSession",
    )


def analysis_register_reference(request, analysis_id):
    """Register a reference to a specific analysis."""
    return pull_views.pull_register_reference(
        request,
        analysis_id,
        source_app="workbench",
        source_type="SessionAnalysis",
    )


def analysis_pull_detail(request, analysis_id):
    """Pull contract: full analysis artifact detail."""
    return pull_views.pull_artifact_detail(
        request,
        analysis_id,
        get_artifact_fn=_wb_analysis_to_dict,
    )


def analysis_pull_sub_artifact(request, analysis_id, key_path):
    """Pull contract: sub-artifact by key path."""
    return pull_views.pull_sub_artifact(
        request,
        analysis_id,
        key_path,
        get_artifact_fn=_wb_get_analysis,
        sub_artifact_fn=_wb_sub_artifact,
    )


def session_delete_with_friction(request, session_id):
    """Delete session with friction — warns if active references exist."""
    return pull_views.pull_delete_with_friction(
        request,
        session_id,
        source_app="workbench",
        source_type="AnalysisSession",
        get_obj_fn=_wb_get_session,
        delete_fn=_wb_delete_session,
    )
