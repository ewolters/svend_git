"""QMS Health Dashboard — cross-module intelligence endpoint.

Aggregates metrics across all 5 QMS modules (FMEA, RCA, A3, VSM, Hoshin)
into a single unified health view. Pure computation — no LLM required.
"""

import logging

from django.db.models import Count, Q
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated_paid

from .models import (
    FMEA,
    A3Report,
    CAPAReport,
    FMEARow,
    HoshinProject,
    RCASession,
    ValueStreamMap,
)

logger = logging.getLogger(__name__)


@gated_paid
@require_http_methods(["GET"])
def qms_dashboard(request):
    """Aggregate QMS health metrics across all 5 modules.

    Returns risk levels, completion rates, and an overall health score.
    """
    user = request.user

    # --- FMEA ---
    fmeas = FMEA.objects.filter(owner=user)
    fmea_count = fmeas.count()
    active_fmeas = fmeas.filter(status__in=["active", "draft"]).count()

    rows = FMEARow.objects.filter(fmea__owner=user)
    critical_rows = rows.filter(rpn__gte=200).count()
    high_rows = rows.filter(rpn__gte=100, rpn__lt=200).count()
    medium_rows = rows.filter(rpn__gte=50, rpn__lt=100).count()
    low_rows = rows.filter(rpn__lt=50).count()
    rows_without_actions = rows.filter(
        rpn__gte=100,
        recommended_action="",
    ).count()

    # --- RCA ---
    rca_sessions = RCASession.objects.filter(owner=user)
    rca_total = rca_sessions.count()
    rca_by_status = dict(
        rca_sessions.values_list("status")
        .annotate(c=Count("id"))
        .values_list("status", "c")
    )
    # Average chain depth
    rca_active = rca_sessions.exclude(status="draft")
    avg_chain_depth = 0
    if rca_active.exists():
        depths = []
        for s in rca_active[:100]:
            chain = s.chain or []
            depths.append(len(chain))
        avg_chain_depth = round(sum(depths) / len(depths), 1) if depths else 0

    # --- A3 ---
    a3_reports = A3Report.objects.filter(owner=user)
    a3_total = a3_reports.count()
    a3_sections = [
        "background",
        "current_condition",
        "goal",
        "root_cause",
        "countermeasures",
        "implementation_plan",
        "follow_up",
    ]
    avg_completion = 0
    if a3_total > 0:
        completions = []
        for report in a3_reports[:50]:
            filled = sum(1 for s in a3_sections if getattr(report, s, ""))
            completions.append(filled / 7)
        avg_completion = round(sum(completions) / len(completions), 2)

    # --- VSM ---
    vsms = ValueStreamMap.objects.filter(owner=user)
    vsm_total = vsms.count()
    with_future = vsms.filter(status="future").count()
    avg_pce = 0
    bottleneck_count = 0
    if vsm_total > 0:
        pces = [v.pce for v in vsms if v.pce and v.pce > 0]
        avg_pce = round(sum(pces) / len(pces), 1) if pces else 0
        for v in vsms:
            steps = v.process_steps or []
            bottleneck_count += sum(1 for s in steps if s.get("is_bottleneck"))

    # --- Hoshin (enterprise — may be empty for non-enterprise users) ---
    hoshin_data = {
        "total_projects": 0,
        "on_track_pct": 0,
        "ytd_savings": 0,
        "target_savings": 0,
        "delayed": 0,
    }
    try:
        from core.models import Membership

        membership = Membership.objects.filter(user=user).first()
        if membership:
            hoshin_projects = HoshinProject.objects.filter(
                site__tenant=membership.tenant
            )
            hoshin_data["total_projects"] = hoshin_projects.count()
            active_hp = hoshin_projects.filter(hoshin_status="active")
            delayed = hoshin_projects.filter(hoshin_status="delayed").count()
            hoshin_data["delayed"] = delayed
            if active_hp.exists():
                on_track = active_hp.filter(
                    Q(hoshin_status="active"),
                ).count()
                hoshin_data["on_track_pct"] = round(
                    on_track / max(active_hp.count() + delayed, 1) * 100
                )
            # Savings aggregation
            for hp in hoshin_projects:
                hoshin_data["target_savings"] += float(hp.annual_savings_target or 0)
                hoshin_data["ytd_savings"] += float(hp.ytd_savings)
    except Exception:
        pass  # Hoshin not available for this user

    # --- Overall health score ---
    # Simple heuristic: penalize for critical risks, reward for completions
    # --- CAPA ---
    capas = CAPAReport.objects.filter(owner=user)
    capa_total = capas.count()
    capa_open = capas.exclude(status="closed").count()
    capa_overdue = (
        capas.exclude(status="closed")
        .filter(due_date__lt=__import__("datetime").date.today())
        .count()
    )
    capa_critical = capas.filter(priority="critical").exclude(status="closed").count()
    capa_by_status = {}
    for row in capas.values("status").annotate(c=Count("id")):
        capa_by_status[row["status"]] = row["c"]

    score_factors = []
    if fmea_count > 0:
        risk_ratio = 1 - (critical_rows / max(rows.count(), 1))
        score_factors.append(risk_ratio)
    if rca_total > 0:
        closed_ratio = rca_by_status.get("closed", 0) / max(rca_total, 1)
        score_factors.append(closed_ratio)
    if a3_total > 0:
        score_factors.append(avg_completion)
    if vsm_total > 0 and avg_pce > 0:
        pce_score = min(avg_pce / 25, 1.0)  # 25% PCE = perfect score
        score_factors.append(pce_score)

    overall = (
        round(sum(score_factors) / max(len(score_factors), 1), 2)
        if score_factors
        else 0
    )

    if overall >= 0.7:
        health = "good"
    elif overall >= 0.4:
        health = "needs_attention"
    else:
        health = "at_risk"

    return JsonResponse(
        {
            "fmea": {
                "total": fmea_count,
                "active_fmeas": active_fmeas,
                "critical_rows": critical_rows,
                "high_rows": high_rows,
                "medium_rows": medium_rows,
                "low_rows": low_rows,
                "rows_without_actions": rows_without_actions,
            },
            "rca": {
                "total": rca_total,
                "avg_chain_depth": avg_chain_depth,
                "by_status": rca_by_status,
            },
            "a3": {
                "total": a3_total,
                "avg_completion": avg_completion,
            },
            "vsm": {
                "total": vsm_total,
                "with_future_state": with_future,
                "avg_pce": avg_pce,
                "bottleneck_count": bottleneck_count,
            },
            "hoshin": hoshin_data,
            "capa": {
                "total": capa_total,
                "open": capa_open,
                "overdue": capa_overdue,
                "critical_open": capa_critical,
                "by_status": capa_by_status,
            },
            "overall_score": overall,
            "overall_health": health,
        }
    )
