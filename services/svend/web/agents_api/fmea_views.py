"""FMEA API views — CRUD for FMEAs and rows, evidence linking, RPN summary.

Persistent FMEA with S/O/D scoring. Rows can optionally link to Hypothesis
objects and generate Evidence + EvidenceLink records for Bayesian updates.
"""

import json
import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.shortcuts import get_object_or_404

from accounts.permissions import gated_paid
from .models import ActionItem, FMEA, FMEARow
from core.models import Project, Hypothesis, Evidence, EvidenceLink

logger = logging.getLogger(__name__)


# =============================================================================
# FMEA CRUD
# =============================================================================

@csrf_exempt
@gated_paid
@require_http_methods(["GET"])
def list_fmeas(request):
    """List user's FMEAs.

    Query params:
    - project_id: filter by project
    - status: filter by status
    - fmea_type: filter by type (process/design/system)
    """
    fmeas = FMEA.objects.filter(owner=request.user).select_related("project")

    project_id = request.GET.get("project_id")
    if project_id:
        fmeas = fmeas.filter(project_id=project_id)

    status = request.GET.get("status")
    if status:
        fmeas = fmeas.filter(status=status)

    fmea_type = request.GET.get("fmea_type")
    if fmea_type:
        fmeas = fmeas.filter(fmea_type=fmea_type)

    # Return lightweight list (no rows)
    results = []
    for f in fmeas[:50]:
        d = {
            "id": str(f.id),
            "project_id": str(f.project_id) if f.project_id else None,
            "project_title": f.project.title if f.project else None,
            "title": f.title,
            "description": f.description,
            "status": f.status,
            "fmea_type": f.fmea_type,
            "row_count": f.rows.count(),
            "max_rpn": max((r.rpn for r in f.rows.all()), default=0),
            "created_at": f.created_at.isoformat(),
            "updated_at": f.updated_at.isoformat(),
        }
        results.append(d)

    return JsonResponse({"fmeas": results})


@csrf_exempt
@gated_paid
@require_http_methods(["POST"])
def create_fmea(request):
    """Create a new FMEA.

    Request body:
    {
        "title": "Process FMEA — Widget Assembly",
        "description": "optional",
        "fmea_type": "process" | "design" | "system",
        "project_id": "uuid (optional)"
    }
    """
    try:
        data = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    title = data.get("title", "").strip()
    if not title:
        return JsonResponse({"error": "title required"}, status=400)

    fmea_type = data.get("fmea_type", "process")
    if fmea_type not in ("process", "design", "system"):
        return JsonResponse({"error": "fmea_type must be process, design, or system"}, status=400)

    project = None
    project_id = data.get("project_id")
    if project_id:
        try:
            project = Project.objects.get(id=project_id, user=request.user)
        except Project.DoesNotExist:
            return JsonResponse({"error": "Study not found"}, status=404)

    fmea = FMEA.objects.create(
        owner=request.user,
        project=project,
        title=title,
        description=data.get("description", ""),
        fmea_type=fmea_type,
    )

    return JsonResponse({
        "id": str(fmea.id),
        "fmea": fmea.to_dict(),
    })


@csrf_exempt
@gated_paid
@require_http_methods(["GET"])
def get_fmea(request, fmea_id):
    """Get a single FMEA with all rows and available hypotheses."""
    fmea = get_object_or_404(FMEA, id=fmea_id, owner=request.user)

    # Available hypotheses for linking
    hypotheses = []
    if fmea.project:
        hypotheses = list(
            Hypothesis.objects.filter(project=fmea.project)
            .values("id", "statement", "current_probability", "status")[:20]
        )

    # Action items linked to any row in this FMEA
    row_ids = list(fmea.rows.values_list("id", flat=True))
    action_items = ActionItem.objects.filter(source_type="fmea", source_id__in=row_ids) if row_ids else []

    return JsonResponse({
        "fmea": fmea.to_dict(),
        "action_items": [i.to_dict() for i in action_items],
        "project": {
            "id": str(fmea.project.id),
            "title": fmea.project.title,
        } if fmea.project else None,
        "available_hypotheses": [
            {
                "id": str(h["id"]),
                "statement": h["statement"],
                "probability": h["current_probability"],
                "status": h["status"],
            }
            for h in hypotheses
        ],
    })


@csrf_exempt
@gated_paid
@require_http_methods(["PUT", "PATCH"])
def update_fmea(request, fmea_id):
    """Update FMEA metadata (title, description, status, type, project)."""
    fmea = get_object_or_404(FMEA, id=fmea_id, owner=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if "title" in data:
        fmea.title = data["title"]
    if "description" in data:
        fmea.description = data["description"]
    if "status" in data:
        fmea.status = data["status"]
    if "fmea_type" in data:
        fmea.fmea_type = data["fmea_type"]
    if "project_id" in data:
        if data["project_id"]:
            try:
                fmea.project = Project.objects.get(id=data["project_id"], user=request.user)
            except Project.DoesNotExist:
                return JsonResponse({"error": "Study not found"}, status=404)
        else:
            fmea.project = None

    fmea.save()

    return JsonResponse({
        "success": True,
        "fmea": fmea.to_dict(),
    })


@csrf_exempt
@gated_paid
@require_http_methods(["DELETE"])
def delete_fmea(request, fmea_id):
    """Delete an FMEA and all its rows."""
    fmea = get_object_or_404(FMEA, id=fmea_id, owner=request.user)
    fmea.delete()
    return JsonResponse({"success": True})


# =============================================================================
# FMEA Row CRUD
# =============================================================================

@csrf_exempt
@gated_paid
@require_http_methods(["POST"])
def add_row(request, fmea_id):
    """Add a new failure mode row to an FMEA.

    Request body:
    {
        "process_step": "Assembly",
        "failure_mode": "Bolt not torqued",
        "effect": "Loose joint, safety risk",
        "severity": 8,
        "cause": "Operator fatigue",
        "occurrence": 4,
        "current_controls": "Visual inspection",
        "detection": 6,
        "recommended_action": "Add torque wrench with limit",
        "action_owner": "Line Lead",
        "hypothesis_id": "uuid (optional)"
    }
    """
    fmea = get_object_or_404(FMEA, id=fmea_id, owner=request.user)

    try:
        data = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    failure_mode = data.get("failure_mode", "").strip()
    if not failure_mode:
        return JsonResponse({"error": "failure_mode required"}, status=400)

    # Auto-assign sort_order
    max_order = fmea.rows.order_by("-sort_order").values_list("sort_order", flat=True).first()
    sort_order = (max_order or 0) + 1

    # Validate S/O/D in 1-10
    severity = _clamp_score(data.get("severity", 1))
    occurrence = _clamp_score(data.get("occurrence", 1))
    detection = _clamp_score(data.get("detection", 1))

    # Optional hypothesis link
    hypothesis = None
    hypothesis_id = data.get("hypothesis_id")
    if hypothesis_id and fmea.project:
        try:
            hypothesis = Hypothesis.objects.get(id=hypothesis_id, project=fmea.project)
        except Hypothesis.DoesNotExist:
            pass

    row = FMEARow.objects.create(
        fmea=fmea,
        sort_order=sort_order,
        process_step=data.get("process_step", ""),
        failure_mode=failure_mode,
        effect=data.get("effect", ""),
        severity=severity,
        cause=data.get("cause", ""),
        occurrence=occurrence,
        current_controls=data.get("current_controls", ""),
        detection=detection,
        recommended_action=data.get("recommended_action", ""),
        action_owner=data.get("action_owner", ""),
        hypothesis_link=hypothesis,
    )

    return JsonResponse({
        "success": True,
        "row": row.to_dict(),
    })


@csrf_exempt
@gated_paid
@require_http_methods(["PUT", "PATCH"])
def update_row(request, fmea_id, row_id):
    """Update a failure mode row."""
    fmea = get_object_or_404(FMEA, id=fmea_id, owner=request.user)
    row = get_object_or_404(FMEARow, id=row_id, fmea=fmea)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    # Text fields
    for field in ("process_step", "failure_mode", "effect", "cause",
                  "current_controls", "recommended_action", "action_owner"):
        if field in data:
            setattr(row, field, data[field])

    if "action_status" in data:
        row.action_status = data["action_status"]

    # S/O/D scores
    if "severity" in data:
        row.severity = _clamp_score(data["severity"])
    if "occurrence" in data:
        row.occurrence = _clamp_score(data["occurrence"])
    if "detection" in data:
        row.detection = _clamp_score(data["detection"])

    # Revised scores
    if "revised_severity" in data:
        row.revised_severity = _clamp_score(data["revised_severity"]) if data["revised_severity"] else None
    if "revised_occurrence" in data:
        row.revised_occurrence = _clamp_score(data["revised_occurrence"]) if data["revised_occurrence"] else None
    if "revised_detection" in data:
        row.revised_detection = _clamp_score(data["revised_detection"]) if data["revised_detection"] else None

    row.save()  # auto-computes rpn and revised_rpn

    return JsonResponse({
        "success": True,
        "row": row.to_dict(),
    })


@csrf_exempt
@gated_paid
@require_http_methods(["DELETE"])
def delete_row(request, fmea_id, row_id):
    """Delete a failure mode row."""
    fmea = get_object_or_404(FMEA, id=fmea_id, owner=request.user)
    row = get_object_or_404(FMEARow, id=row_id, fmea=fmea)
    row.delete()
    return JsonResponse({"success": True})


@csrf_exempt
@gated_paid
@require_http_methods(["POST"])
def reorder_rows(request, fmea_id):
    """Reorder rows by supplying a list of row IDs in desired order.

    Request body:
    {
        "row_ids": ["uuid1", "uuid2", "uuid3"]
    }
    """
    fmea = get_object_or_404(FMEA, id=fmea_id, owner=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    row_ids = data.get("row_ids", [])
    if not row_ids:
        return JsonResponse({"error": "row_ids required"}, status=400)

    for i, rid in enumerate(row_ids):
        FMEARow.objects.filter(id=rid, fmea=fmea).update(sort_order=i)

    return JsonResponse({
        "success": True,
        "fmea": fmea.to_dict(),
    })


# =============================================================================
# Hypothesis Linking & Evidence Generation
# =============================================================================

@csrf_exempt
@gated_paid
@require_http_methods(["POST"])
def link_to_hypothesis(request, fmea_id, row_id):
    """Link an FMEA row to a hypothesis, creating evidence.

    Request body:
    {
        "hypothesis_id": "uuid"
    }

    Creates an Evidence record from the failure mode details and an
    EvidenceLink connecting it to the hypothesis. The likelihood ratio
    is derived from S/O/D scores:
    - High severity + high occurrence → strong evidence supporting hypothesis
    - Low scores → weaker evidence
    """
    fmea = get_object_or_404(FMEA, id=fmea_id, owner=request.user)
    row = get_object_or_404(FMEARow, id=row_id, fmea=fmea)

    if not fmea.project:
        return JsonResponse({"error": "FMEA must be linked to a study"}, status=400)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    hypothesis_id = data.get("hypothesis_id")
    if not hypothesis_id:
        return JsonResponse({"error": "hypothesis_id required"}, status=400)

    try:
        hypothesis = Hypothesis.objects.get(id=hypothesis_id, project=fmea.project)
    except Hypothesis.DoesNotExist:
        return JsonResponse({"error": "Hypothesis not found"}, status=404)

    # Link the row to the hypothesis
    row.hypothesis_link = hypothesis
    row.save()

    # Create evidence from the failure mode
    summary = (
        f"FMEA: {row.failure_mode} — "
        f"S={row.severity}, O={row.occurrence}, D={row.detection}, RPN={row.rpn}. "
        f"Effect: {row.effect[:200]}"
    )
    if row.cause:
        summary += f" | Cause: {row.cause[:200]}"

    evidence = Evidence.objects.create(
        project=fmea.project,
        summary=summary,
        details=f"Process step: {row.process_step}\n"
                f"Failure mode: {row.failure_mode}\n"
                f"Effect: {row.effect}\n"
                f"Cause: {row.cause}\n"
                f"Current controls: {row.current_controls}\n"
                f"RPN: {row.rpn} (S={row.severity} × O={row.occurrence} × D={row.detection})",
        source_type="analysis",
        source_description=f"FMEA: {fmea.title}",
        result_type="qualitative",
        confidence=_rpn_to_confidence(row.rpn),
        created_by=request.user,
    )

    # Compute likelihood ratio from severity × occurrence
    # High S×O means the failure mode is real and frequent → supports hypothesis
    lr = _compute_likelihood_ratio(row.severity, row.occurrence)

    link = EvidenceLink.objects.create(
        hypothesis=hypothesis,
        evidence=evidence,
        likelihood_ratio=lr,
        reasoning=f"FMEA failure mode '{row.failure_mode}' with RPN={row.rpn} "
                  f"(S={row.severity}, O={row.occurrence}, D={row.detection})",
        is_manual=False,
    )

    # Apply evidence to update hypothesis probability
    new_prob = hypothesis.apply_evidence(link)

    return JsonResponse({
        "success": True,
        "row": row.to_dict(),
        "evidence_id": str(evidence.id),
        "link_id": str(link.id),
        "likelihood_ratio": lr,
        "direction": link.direction,
        "hypothesis_probability": new_prob,
    })


@csrf_exempt
@gated_paid
@require_http_methods(["POST"])
def record_revision(request, fmea_id, row_id):
    """Record revised S/O/D scores after corrective action.

    Request body:
    {
        "revised_severity": 3,
        "revised_occurrence": 2,
        "revised_detection": 4
    }

    If the row is linked to a hypothesis, generates new evidence showing
    the RPN reduction and updates the hypothesis probability.
    """
    fmea = get_object_or_404(FMEA, id=fmea_id, owner=request.user)
    row = get_object_or_404(FMEARow, id=row_id, fmea=fmea)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    row.revised_severity = _clamp_score(data.get("revised_severity", row.severity))
    row.revised_occurrence = _clamp_score(data.get("revised_occurrence", row.occurrence))
    row.revised_detection = _clamp_score(data.get("revised_detection", row.detection))
    row.save()  # auto-computes revised_rpn

    result = {
        "success": True,
        "row": row.to_dict(),
        "rpn_reduction": row.rpn - (row.revised_rpn or row.rpn),
    }

    # If linked to hypothesis and there's an actual improvement, generate evidence
    if row.hypothesis_link and fmea.project and row.revised_rpn and row.revised_rpn < row.rpn:
        hypothesis = row.hypothesis_link

        action_desc = row.recommended_action[:200] if row.recommended_action else "corrective action"
        summary = (
            f"FMEA: {row.failure_mode} — Occurrence reduced from {row.occurrence} "
            f"to {row.revised_occurrence} after {action_desc}. "
            f"RPN: {row.rpn} → {row.revised_rpn}."
        )

        evidence = Evidence.objects.create(
            project=fmea.project,
            summary=summary,
            details=f"Original: S={row.severity}, O={row.occurrence}, D={row.detection}, RPN={row.rpn}\n"
                    f"Revised: S={row.revised_severity}, O={row.revised_occurrence}, "
                    f"D={row.revised_detection}, RPN={row.revised_rpn}\n"
                    f"Improvement: {row.rpn - row.revised_rpn} RPN reduction ({(1 - row.revised_rpn/row.rpn)*100:.0f}%)",
            source_type="analysis",
            source_description=f"FMEA revision: {fmea.title}",
            result_type="quantitative",
            confidence=0.85,
            measured_value=float(row.revised_rpn),
            expected_value=float(row.rpn),
            created_by=request.user,
        )

        # RPN reduction opposes the hypothesis (problem addressed)
        # Larger reduction → stronger opposition
        reduction_ratio = row.revised_rpn / row.rpn  # 0..1, lower = better
        lr = 0.3 + 0.7 * reduction_ratio  # maps to 0.3..1.0 (opposes)

        link = EvidenceLink.objects.create(
            hypothesis=hypothesis,
            evidence=evidence,
            likelihood_ratio=lr,
            reasoning=f"FMEA corrective action reduced RPN from {row.rpn} to {row.revised_rpn} "
                      f"({(1 - reduction_ratio)*100:.0f}% reduction). Problem is being addressed.",
            is_manual=False,
        )

        new_prob = hypothesis.apply_evidence(link)

        result["evidence_id"] = str(evidence.id)
        result["link_id"] = str(link.id)
        result["likelihood_ratio"] = lr
        result["hypothesis_probability"] = new_prob

    return JsonResponse(result)


# =============================================================================
# RPN Summary / Pareto
# =============================================================================

@csrf_exempt
@gated_paid
@require_http_methods(["GET"])
def rpn_summary(request, fmea_id):
    """Get RPN summary with Pareto data and before/after comparison."""
    fmea = get_object_or_404(FMEA, id=fmea_id, owner=request.user)

    rows = list(fmea.rows.order_by("-rpn"))

    if not rows:
        return JsonResponse({
            "total_rows": 0,
            "pareto": [],
            "summary": {},
        })

    rpns = [r.rpn for r in rows]
    total_rpn = sum(rpns)

    # Pareto: sorted by RPN descending with cumulative %
    pareto = []
    cumulative = 0
    for r in rows:
        cumulative += r.rpn
        pareto.append({
            "id": str(r.id),
            "failure_mode": r.failure_mode,
            "process_step": r.process_step,
            "rpn": r.rpn,
            "severity": r.severity,
            "occurrence": r.occurrence,
            "detection": r.detection,
            "cumulative_pct": (cumulative / total_rpn * 100) if total_rpn else 0,
            "revised_rpn": r.revised_rpn,
            "action_status": r.action_status,
            "hypothesis_id": str(r.hypothesis_link_id) if r.hypothesis_link_id else None,
        })

    # Before/after comparison (only rows with revised scores)
    revised_rows = [r for r in rows if r.revised_rpn is not None]
    before_total = sum(r.rpn for r in revised_rows)
    after_total = sum(r.revised_rpn for r in revised_rows)

    # Risk buckets
    critical = sum(1 for r in rows if r.rpn > 200)
    high = sum(1 for r in rows if 100 < r.rpn <= 200)
    medium = sum(1 for r in rows if 50 < r.rpn <= 100)
    low = sum(1 for r in rows if r.rpn <= 50)

    return JsonResponse({
        "total_rows": len(rows),
        "total_rpn": total_rpn,
        "avg_rpn": total_rpn / len(rows),
        "max_rpn": max(rpns),
        "pareto": pareto,
        "risk_buckets": {
            "critical": critical,
            "high": high,
            "medium": medium,
            "low": low,
        },
        "revision_summary": {
            "revised_count": len(revised_rows),
            "before_total_rpn": before_total,
            "after_total_rpn": after_total,
            "total_reduction": before_total - after_total,
            "reduction_pct": ((before_total - after_total) / before_total * 100) if before_total else 0,
        },
    })


# =============================================================================
# Helpers
# =============================================================================

def _clamp_score(value):
    """Clamp S/O/D score to 1-10."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return 1
    return max(1, min(10, v))


def _rpn_to_confidence(rpn):
    """Map RPN to evidence confidence (0.5-0.95).

    Higher RPN → higher confidence that the failure mode is real.
    """
    # RPN ranges from 1 to 1000
    # Map to 0.5-0.95 via log scale
    import math
    normalized = math.log(max(rpn, 1)) / math.log(1000)  # 0..1
    return 0.5 + 0.45 * normalized


def _compute_likelihood_ratio(severity, occurrence):
    """Compute likelihood ratio from severity and occurrence.

    S×O product (1-100) maps to LR:
    - High S×O (>50): strong support (LR 3-8)
    - Medium S×O (20-50): moderate support (LR 1.5-3)
    - Low S×O (<20): weak support (LR 1.1-1.5)
    """
    so = severity * occurrence
    if so >= 50:
        # Map 50-100 → 3.0-8.0
        lr = 3.0 + 5.0 * (so - 50) / 50
    elif so >= 20:
        # Map 20-50 → 1.5-3.0
        lr = 1.5 + 1.5 * (so - 20) / 30
    else:
        # Map 1-20 → 1.1-1.5
        lr = 1.1 + 0.4 * (so - 1) / 19
    return round(lr, 2)


# ── Action Items ──────────────────────────────────────────────────────

@csrf_exempt
@gated_paid
@require_http_methods(["GET"])
def list_fmea_actions(request, fmea_id):
    """List all action items linked to any row in this FMEA."""
    fmea = get_object_or_404(FMEA, id=fmea_id, user=request.user)
    row_ids = list(fmea.rows.values_list("id", flat=True))
    items = ActionItem.objects.filter(source_type="fmea", source_id__in=row_ids)
    return JsonResponse({"action_items": [i.to_dict() for i in items]})


@csrf_exempt
@gated_paid
@require_http_methods(["POST"])
def promote_fmea_action(request, fmea_id, row_id):
    """Promote an FMEA row's recommended action to a tracked ActionItem."""
    fmea = get_object_or_404(FMEA, id=fmea_id, user=request.user)
    row = get_object_or_404(FMEARow, id=row_id, fmea=fmea)

    if not fmea.project:
        return JsonResponse({"error": "FMEA must be linked to a project first"}, status=400)

    # Check if already promoted
    existing = ActionItem.objects.filter(source_type="fmea", source_id=row.id).first()
    if existing:
        return JsonResponse({"action_item": existing.to_dict()})

    data = {}
    try:
        data = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        pass

    title = data.get("title", "").strip() or row.recommended_action[:255] or f"FMEA action: {row.failure_mode}"

    item = ActionItem.objects.create(
        project=fmea.project,
        title=title,
        description=data.get("description", f"From FMEA row: {row.process_step} — {row.failure_mode}\nRPN: {row.rpn}"),
        owner_name=data.get("owner_name", row.action_owner),
        status="not_started",
        due_date=data.get("due_date"),
        source_type="fmea",
        source_id=row.id,
    )
    return JsonResponse({"success": True, "action_item": item.to_dict()}, status=201)
