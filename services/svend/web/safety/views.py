"""HIRARC Safety views — Frontier Card, audit scheduling, 5S Pareto, KPIs.

All endpoints require Enterprise tier via @require_feature("safety").
Tenant isolation via agents_api.permissions.
"""

import json
import logging
from datetime import date, timedelta

from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from accounts.permissions import require_feature
from agents_api.permissions import get_accessible_sites, get_tenant

from .models import (
    AuditAssignment,
    AuditSchedule,
    FrontierCard,
    FrontierZone,
    aggregate_five_s_pareto,
    process_card_to_fmea,
)

logger = logging.getLogger("svend.safety")


def _require_tenant(user):
    tenant = get_tenant(user)
    if not tenant:
        return None, JsonResponse({"error": "Enterprise account required"}, status=403)
    return tenant, None


# =============================================================================
# FRONTIER ZONES
# =============================================================================


@csrf_exempt
@require_feature("safety")
@require_http_methods(["GET", "POST"])
def zone_list_create(request):
    """GET: list zones. POST: create zone."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    if request.method == "GET":
        sites, is_admin = get_accessible_sites(request.user, tenant)
        qs = FrontierZone.objects.filter(site__in=sites, is_active=True)
        site_id = request.GET.get("site")
        if site_id:
            qs = qs.filter(site_id=site_id)
        return JsonResponse(
            [
                {
                    "id": str(z.id),
                    "site_id": str(z.site_id),
                    "name": z.name,
                    "description": z.description,
                    "zone_type": z.zone_type,
                }
                for z in qs
            ],
            safe=False,
        )

    data = json.loads(request.body)
    from agents_api.models import Site

    site = get_object_or_404(Site, id=data["site_id"], tenant=tenant)
    zone = FrontierZone.objects.create(
        site=site,
        name=data["name"],
        description=data.get("description", ""),
        zone_type=data.get("zone_type", "general"),
    )
    return JsonResponse({"id": str(zone.id), "name": zone.name}, status=201)


@csrf_exempt
@require_feature("safety")
@require_http_methods(["GET", "PUT", "DELETE"])
def zone_detail(request, zone_id):
    """GET/PUT/DELETE a frontier zone."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    zone = get_object_or_404(FrontierZone, id=zone_id, site__tenant=tenant)

    if request.method == "GET":
        return JsonResponse(
            {
                "id": str(zone.id),
                "site_id": str(zone.site_id),
                "name": zone.name,
                "description": zone.description,
                "zone_type": zone.zone_type,
                "is_active": zone.is_active,
            }
        )

    if request.method == "DELETE":
        zone.is_active = False
        zone.save(update_fields=["is_active"])
        return JsonResponse({"ok": True})

    data = json.loads(request.body)
    for field in ("name", "description", "zone_type"):
        if field in data:
            setattr(zone, field, data[field])
    zone.save()
    return JsonResponse({"id": str(zone.id), "name": zone.name})


# =============================================================================
# AUDIT SCHEDULING
# =============================================================================


@csrf_exempt
@require_feature("safety")
@require_http_methods(["GET", "POST"])
def schedule_list_create(request):
    """GET: list schedules. POST: create weekly schedule."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    if request.method == "GET":
        sites, _ = get_accessible_sites(request.user, tenant)
        qs = AuditSchedule.objects.filter(site__in=sites).select_related("site")
        site_id = request.GET.get("site")
        if site_id:
            qs = qs.filter(site_id=site_id)
        return JsonResponse(
            [
                {
                    "id": str(s.id),
                    "site_id": str(s.site_id),
                    "site_name": s.site.name,
                    "week_start": s.week_start.isoformat(),
                    "completion_rate": s.completion_rate,
                    "assignment_count": s.assignments.count(),
                }
                for s in qs[:20]
            ],
            safe=False,
        )

    data = json.loads(request.body)
    from agents_api.models import Site

    site = get_object_or_404(Site, id=data["site_id"], tenant=tenant)
    week_start = date.fromisoformat(data["week_start"])

    schedule, created = AuditSchedule.objects.get_or_create(
        site=site,
        week_start=week_start,
        defaults={"published_by": request.user},
    )

    # Bulk create assignments if provided
    assignments = data.get("assignments", [])
    from agents_api.models import Employee

    created_assignments = []
    for a in assignments:
        emp = get_object_or_404(Employee, id=a["auditor_id"], tenant=tenant)
        zone = get_object_or_404(FrontierZone, id=a["zone_id"], site=site)
        assignment = AuditAssignment.objects.create(
            schedule=schedule,
            auditor=emp,
            zone=zone,
            target_date=date.fromisoformat(a.get("target_date", week_start.isoformat())),
        )
        created_assignments.append(str(assignment.id))

    return JsonResponse(
        {
            "id": str(schedule.id),
            "week_start": schedule.week_start.isoformat(),
            "assignments_created": len(created_assignments),
            "created": created,
        },
        status=201,
    )


@csrf_exempt
@require_feature("safety")
@require_http_methods(["GET"])
def schedule_detail(request, schedule_id):
    """Get schedule with all assignments."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    schedule = get_object_or_404(AuditSchedule, id=schedule_id, site__tenant=tenant)
    assignments = schedule.assignments.select_related("auditor", "zone").all()

    return JsonResponse(
        {
            "id": str(schedule.id),
            "site_id": str(schedule.site_id),
            "week_start": schedule.week_start.isoformat(),
            "completion_rate": schedule.completion_rate,
            "assignments": [
                {
                    "id": str(a.id),
                    "auditor_id": str(a.auditor_id),
                    "auditor_name": a.auditor.name,
                    "zone_id": str(a.zone_id),
                    "zone_name": a.zone.name,
                    "target_date": a.target_date.isoformat(),
                    "status": a.status,
                    "completed_at": a.completed_at.isoformat() if a.completed_at else None,
                }
                for a in assignments
            ],
        }
    )


@csrf_exempt
@require_feature("safety")
@require_http_methods(["PUT"])
def assignment_update(request, assignment_id):
    """Update assignment status."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    assignment = get_object_or_404(AuditAssignment, id=assignment_id, schedule__site__tenant=tenant)
    data = json.loads(request.body)

    if "status" in data:
        assignment.status = data["status"]
        if data["status"] == "completed":
            assignment.completed_at = timezone.now()
    assignment.save()
    return JsonResponse({"id": str(assignment.id), "status": assignment.status})


# =============================================================================
# FRONTIER CARDS
# =============================================================================


@csrf_exempt
@require_feature("safety")
@require_http_methods(["GET", "POST"])
def card_list_create(request):
    """GET: list cards. POST: create card (submit audit)."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    if request.method == "GET":
        sites, _ = get_accessible_sites(request.user, tenant)
        qs = FrontierCard.objects.filter(site__in=sites).select_related("auditor", "zone", "site")
        site_id = request.GET.get("site")
        if site_id:
            qs = qs.filter(site_id=site_id)
        unprocessed = request.GET.get("unprocessed")
        if unprocessed and unprocessed.lower() == "true":
            qs = qs.filter(is_processed=False)

        limit = min(int(request.GET.get("limit", 50)), 200)
        return JsonResponse([c.to_dict() for c in qs[:limit]], safe=False)

    # POST — submit a new Frontier Card
    data = json.loads(request.body)
    from agents_api.models import Employee, Site

    site = get_object_or_404(Site, id=data["site_id"], tenant=tenant)
    auditor = get_object_or_404(Employee, id=data["auditor_id"], tenant=tenant)
    zone = get_object_or_404(FrontierZone, id=data["zone_id"], site=site)

    card = FrontierCard.objects.create(
        auditor=auditor,
        zone=zone,
        site=site,
        audit_date=date.fromisoformat(data.get("audit_date", date.today().isoformat())),
        shift=data.get("shift", ""),
        safety_observations=data.get("safety_observations", []),
        five_s_tallies=data.get("five_s_tallies", {}),
        operator_name=data.get("operator_name", ""),
        operator_concern=data.get("operator_concern", ""),
        operator_improvement=data.get("operator_improvement", ""),
        operator_near_miss=data.get("operator_near_miss", ""),
        has_safety_crossfeed=data.get("has_safety_crossfeed", False),
        crossfeed_notes=data.get("crossfeed_notes", ""),
    )

    # Link to assignment if provided
    assignment_id = data.get("assignment_id")
    if assignment_id:
        try:
            assignment = AuditAssignment.objects.get(id=assignment_id)
            card.assignment = assignment
            card.save(update_fields=["assignment"])
            assignment.status = "completed"
            assignment.completed_at = timezone.now()
            assignment.save(update_fields=["status", "completed_at"])
        except AuditAssignment.DoesNotExist:
            pass

    # Notify on critical/high findings
    _notify_high_severity(card)

    return JsonResponse(card.to_dict(), status=201)


@csrf_exempt
@require_feature("safety")
@require_http_methods(["GET"])
def card_detail(request, card_id):
    """Get single card with full details."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    card = get_object_or_404(FrontierCard, id=card_id, site__tenant=tenant)
    return JsonResponse(card.to_dict())


# =============================================================================
# CARD-TO-FMEA PROCESSING
# =============================================================================


@csrf_exempt
@require_feature("safety")
@require_http_methods(["POST"])
def process_card(request, card_id):
    """Process a Frontier Card into FMEA rows.

    Expects: {"fmea_id": "uuid"} — the FMEA to add rows to.
    """
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    card = get_object_or_404(FrontierCard, id=card_id, site__tenant=tenant)

    if card.is_processed:
        return JsonResponse(
            {"error": "Card already processed", "fmea_rows": card.fmea_rows_created},
            status=400,
        )

    data = json.loads(request.body)
    fmea_id = data.get("fmea_id")
    if not fmea_id:
        return JsonResponse({"error": "fmea_id required"}, status=400)

    from agents_api.models import FMEA

    fmea = get_object_or_404(FMEA, id=fmea_id)
    created_ids = process_card_to_fmea(card, fmea, request.user)

    return JsonResponse(
        {
            "processed": True,
            "card_id": str(card.id),
            "fmea_id": str(fmea.id),
            "rows_created": len(created_ids),
            "row_ids": created_ids,
        }
    )


# =============================================================================
# 5S PARETO
# =============================================================================


@csrf_exempt
@require_feature("safety")
@require_http_methods(["GET"])
def five_s_pareto(request):
    """Get 5S Pareto data for a site."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    site_id = request.GET.get("site")
    if not site_id:
        return JsonResponse({"error": "site query param required"}, status=400)

    from agents_api.models import Site

    site = get_object_or_404(Site, id=site_id, tenant=tenant)
    min_cards = int(request.GET.get("min_cards", 10))

    result = aggregate_five_s_pareto(site, min_cards=min_cards)
    if result is None:
        return JsonResponse(
            {
                "error": f"Insufficient data — need at least {min_cards} cards with 5S tallies",
                "card_count": FrontierCard.objects.filter(site=site).exclude(five_s_tallies={}).count(),
            },
            status=200,
        )

    return JsonResponse(result)


# =============================================================================
# SAFETY KPI DASHBOARD
# =============================================================================


@csrf_exempt
@require_feature("safety")
@require_http_methods(["GET"])
def safety_dashboard(request):
    """Safety KPI dashboard — leading and lagging indicators."""
    tenant, err = _require_tenant(request.user)
    if err:
        return err

    sites, _ = get_accessible_sites(request.user, tenant)
    site_id = request.GET.get("site")
    if site_id:
        from agents_api.models import Site

        sites = Site.objects.filter(id=site_id, tenant=tenant)

    # Date ranges
    now = timezone.now()
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)

    # Cards
    all_cards = FrontierCard.objects.filter(site__in=sites)
    month_cards = all_cards.filter(created_at__gte=month_ago)
    week_cards = all_cards.filter(created_at__gte=week_ago)

    # Schedules this week
    today = date.today()
    week_start = today - timedelta(days=today.weekday())
    current_schedules = AuditSchedule.objects.filter(site__in=sites, week_start=week_start)
    total_assignments = 0
    completed_assignments = 0
    for sched in current_schedules:
        assigns = sched.assignments.all()
        total_assignments += assigns.count()
        completed_assignments += assigns.filter(status="completed").count()

    completion_rate = round(completed_assignments / total_assignments * 100, 1) if total_assignments > 0 else None

    # Hazard counts
    month_at_risk = sum(c.at_risk_count for c in month_cards)
    week_at_risk = sum(c.at_risk_count for c in week_cards)

    # Processing time (avg hours from card creation to processing)
    processed_cards = all_cards.filter(is_processed=True, processed_at__isnull=False).order_by("-processed_at")[:50]
    processing_times = []
    for c in processed_cards:
        delta = (c.processed_at - c.created_at).total_seconds() / 3600
        processing_times.append(delta)
    avg_processing_hours = round(sum(processing_times) / len(processing_times), 1) if processing_times else None

    # Severity distribution this month
    severity_counts = {"C": 0, "H": 0, "M": 0, "L": 0}
    for card in month_cards:
        for obs in card.safety_observations or []:
            if obs.get("rating") in ("AR", "U"):
                sev = obs.get("severity", "L")
                severity_counts[sev] = severity_counts.get(sev, 0) + 1

    # Operator interaction rate
    month_count = month_cards.count()
    interaction_count = month_cards.exclude(operator_name="").count()
    interaction_rate = round(interaction_count / month_count * 100, 1) if month_count > 0 else None

    return JsonResponse(
        {
            "period": {"from": month_ago.date().isoformat(), "to": today.isoformat()},
            "leading": {
                "audit_completion_rate": completion_rate,
                "audit_completion_target": 95.0,
                "hazards_this_month": month_at_risk,
                "hazards_this_week": week_at_risk,
                "operator_interaction_rate": interaction_rate,
                "avg_processing_hours": avg_processing_hours,
                "cards_this_month": month_count,
                "unprocessed_cards": all_cards.filter(is_processed=False).count(),
            },
            "severity_distribution": severity_counts,
            "totals": {
                "total_cards": all_cards.count(),
                "total_zones": FrontierZone.objects.filter(site__in=sites, is_active=True).count(),
                "total_processed": all_cards.filter(is_processed=True).count(),
            },
        }
    )


# =============================================================================
# NOTIFICATION HELPER
# =============================================================================


def _notify_high_severity(card):
    """Notify site admins on Critical or High findings."""
    try:
        highest = card.highest_severity
        if highest not in ("C", "H"):
            return

        from agents_api.models import SiteAccess
        from notifications.helpers import notify

        admins = SiteAccess.objects.filter(site=card.site, role__in=("admin", "member")).select_related("user")

        title = f"{'CRITICAL' if highest == 'C' else 'HIGH'} safety finding: {card.zone.name}"
        message = (
            f"Frontier Card audit by {card.auditor.name} on {card.audit_date} "
            f"identified {card.at_risk_count} at-risk condition(s) in {card.zone.name}."
        )

        for access in admins:
            notify(
                recipient=access.user,
                notification_type="system",
                title=title,
                message=message,
                entity_type="frontier_card",
                entity_id=card.id,
            )
    except Exception:
        logger.exception("Failed to notify on high-severity card %s", card.id)
