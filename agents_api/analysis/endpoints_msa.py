"""Measurement System Analysis (MSA) endpoints — Gage study persistence.

Extracted from spc_views.py. These endpoints manage MeasurementSystem and
GageStudy records. No SPC computation — pure model CRUD.

CR: ce6a4ceb (Phase 0, from spc_views.py)
"""

import json

from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from accounts.permissions import require_auth


@require_http_methods(["POST"])
@require_auth
def save_gage_study(request):
    """Save Gage R&R results to a MeasurementSystem + GageStudy.

    Creates or retrieves a MeasurementSystem by name for the user,
    then creates a GageStudy with the provided results.
    Auto-quarantine logic in GageStudy.save() handles %GRR > 30%.
    """
    from django.utils import timezone

    from core.models.measurement import GageStudy, MeasurementSystem

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    name = (data.get("name") or "").strip()
    if not name:
        return JsonResponse({"error": "name is required"}, status=400)

    grr_percent = data.get("grr_percent")
    ndc = data.get("ndc")
    study_type_str = data.get("study_type", "GRR_CROSSED")

    study_type_map = {
        "GRR_CROSSED": GageStudy.StudyType.GRR_CROSSED,
        "GRR_NESTED": GageStudy.StudyType.GRR_NESTED,
        "GRR_BAYESIAN": GageStudy.StudyType.GRR_CROSSED,
        "ATTRIBUTE_AGREEMENT": GageStudy.StudyType.ATTRIBUTE_AGREEMENT,
    }
    study_type = study_type_map.get(study_type_str, GageStudy.StudyType.GRR_CROSSED)

    ms, _created = MeasurementSystem.objects.get_or_create(
        name=name,
        owner=request.user,
        defaults={
            "system_type": MeasurementSystem.SystemType.VARIABLE,
            "status": MeasurementSystem.Status.ACTIVE,
        },
    )

    study = GageStudy.objects.create(
        measurement_system=ms,
        study_type=study_type,
        completed_at=timezone.now(),
        grr_percent=float(grr_percent) if grr_percent is not None else None,
        ndc=int(ndc) if ndc is not None else None,
    )

    return JsonResponse(
        {
            "id": str(study.id),
            "system_id": str(ms.id),
            "system_name": ms.name,
            "system_status": ms.status,
            "validity": study.measurement_validity,
            "quarantined": ms.status == MeasurementSystem.Status.QUARANTINED,
        },
        status=201,
    )


@require_http_methods(["GET"])
@require_auth
def recent_gage_studies(request):
    """Return recent Gage R&R studies for the user.

    Shows %GRR and assessment for each measurement system.
    """
    from core.models.measurement import MeasurementSystem

    systems = MeasurementSystem.objects.filter(owner=request.user).order_by("-updated_at")[:20]

    results = []
    for ms in systems:
        latest = ms.gage_studies.order_by("-completed_at").first()
        if latest and latest.grr_percent is not None:
            results.append(
                {
                    "system_id": str(ms.id),
                    "system_name": ms.name,
                    "status": ms.status,
                    "grr_percent": round(latest.grr_percent, 2),
                    "ndc": latest.ndc,
                    "assessment": (
                        "Acceptable"
                        if latest.grr_percent < 10
                        else "Marginal"
                        if latest.grr_percent < 30
                        else "Unacceptable"
                    ),
                    "completed_at": latest.completed_at.isoformat() if latest.completed_at else None,
                }
            )

    return JsonResponse({"measurement_systems": results})
