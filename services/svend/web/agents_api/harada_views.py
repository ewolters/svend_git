"""
Harada Method API — questionnaire, goals, 64-window, routines, diary.

All endpoints require authentication via @gated_paid.
Questionnaire scenarios are randomized at render time per instrument design.

<!-- impl: agents_api/harada_views.py:get_questionnaire -->
<!-- impl: agents_api/harada_views.py:submit_responses -->
<!-- impl: agents_api/harada_views.py:get_response_history -->
<!-- impl: agents_api/harada_views.py:list_create_goals -->
<!-- impl: agents_api/harada_views.py:goal_detail -->
<!-- impl: agents_api/harada_views.py:list_create_window -->
<!-- impl: agents_api/harada_views.py:update_window_cell -->
<!-- impl: agents_api/harada_views.py:check_routine -->
<!-- impl: agents_api/harada_views.py:routine_history -->
<!-- impl: agents_api/harada_views.py:list_create_diary -->
<!-- impl: agents_api/harada_views.py:diary_detail -->
"""

import json
import logging
import random
import uuid as _uuid
from datetime import date, timedelta

from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated_paid
from core.models import (
    DailyDiary,
    HaradaGoal,
    QuestionDimension,
    QuestionnaireResponse,
    RoutineCheck,
    Scenario,
    Window64,
)

logger = logging.getLogger("svend.harada")


# ---------------------------------------------------------------------------
# Questionnaire
# ---------------------------------------------------------------------------


@require_http_methods(["GET"])
@gated_paid
def get_questionnaire(request):
    """
    Present the CI Readiness questionnaire with randomized scenario selection
    and option order.

    GET /api/harada/questionnaire/?instrument=ci_readiness

    For forced-choice: selects one scenario per dimension (excluding previously
    seen in last session), randomizes ABCD option order.
    For Likert: returns question text with 1-5 scale.
    """
    instrument = request.GET.get("instrument", "ci_readiness")
    dimensions = QuestionDimension.objects.filter(instrument=instrument).order_by("dimension_number")

    if not dimensions.exists():
        return JsonResponse({"error": f"No dimensions for instrument: {instrument}"}, status=404)

    # Find previously used scenario IDs (from last session) to avoid on retake
    last_session = (
        QuestionnaireResponse.objects.filter(user=request.user, dimension__instrument=instrument)
        .order_by("-timestamp")
        .values_list("session_id", flat=True)
        .first()
    )
    used_scenario_ids = set()
    if last_session:
        used_scenario_ids = set(
            QuestionnaireResponse.objects.filter(user=request.user, session_id=last_session)
            .exclude(scenario=None)
            .values_list("scenario_id", flat=True)
        )

    session_id = str(_uuid.uuid4())
    items = []

    # Experience gate for Q11 (Measurement System Trust)
    # Default to profile, fallback to asking
    Q11_EXPERIENCED = (
        "I have delayed or revised a conclusion after discovering a problem with how the data was collected."
    )
    Q11_EARLY_CAREER = "Before drawing conclusions from data, I investigate whether the measurement system itself could explain the result."

    user = request.user
    experience_known = bool(getattr(user, "experience_level", None)) or getattr(user, "role", "") == "student"
    is_early_career = None
    if experience_known:
        is_early_career = getattr(user, "experience_level", "") == "beginner" or getattr(user, "role", "") == "student"

    # If experience is unknown, we'll flag it so the UI can ask
    needs_experience_question = not experience_known and instrument == "ci_readiness"

    for dim in dimensions:
        item = {
            "dimension_id": str(dim.id),
            "dimension_number": dim.dimension_number,
            "name": dim.name,
            "description": dim.description,
            "response_type": dim.response_type,
            "category": dim.category,
        }

        if dim.response_type == "likert":
            question_text = dim.question_text

            # Q11 experience gate
            if instrument == "ci_readiness" and dim.dimension_number == 11:
                if is_early_career is True:
                    question_text = Q11_EARLY_CAREER
                elif is_early_career is False:
                    question_text = Q11_EXPERIENCED
                # If unknown, default to experienced — UI will ask experience first
                item["experience_gated"] = True

            item["question_text"] = question_text
            item["scale_min"] = 1
            item["scale_max"] = 5
            item["scale_labels"] = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
        else:
            # Forced-choice: select scenario, randomize options
            scenarios = list(dim.scenarios.filter(is_active=True))
            if not scenarios:
                continue

            # Prefer unseen scenarios on retake
            unseen = [s for s in scenarios if s.id not in used_scenario_ids]
            selected = random.choice(unseen) if unseen else random.choice(scenarios)

            # Build options in random order
            options = [
                {"label": selected.option_a_label, "text": selected.option_a},
                {"label": selected.option_b_label, "text": selected.option_b},
                {"label": selected.option_c_label, "text": selected.option_c},
                {"label": selected.option_d_label, "text": selected.option_d},
            ]
            random.shuffle(options)

            item["scenario_id"] = str(selected.id)
            item["scenario_key"] = selected.scenario_key
            item["situation"] = selected.situation
            item["options"] = options

        items.append(item)

    result = {
        "instrument": instrument,
        "session_id": session_id,
        "dimensions": items,
        "retake_number": QuestionnaireResponse.objects.filter(user=request.user, dimension__instrument=instrument)
        .values("session_id")
        .distinct()
        .count()
        + 1,
    }

    if needs_experience_question:
        result["experience_question"] = {
            "prompt": "How many years of continuous improvement or quality experience do you have?",
            "options": [
                {"value": "early", "label": "Less than 2 years"},
                {"value": "experienced", "label": "2 or more years"},
            ],
            "purpose": "q11_variant",
        }

    return JsonResponse(result)


@require_http_methods(["POST"])
@gated_paid
def submit_responses(request):
    """
    Submit questionnaire responses for a session.

    POST /api/harada/questionnaire/submit/
    {
        "instrument": "ci_readiness",
        "session_id": "uuid",
        "responses": [
            {"dimension_id": "uuid", "score": 4},                    // Likert
            {"dimension_id": "uuid", "scenario_id": "uuid",
             "option_chosen": "system_thinker"},                     // Forced-choice
        ]
    }
    """
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    instrument = data.get("instrument", "ci_readiness")
    session_id = data.get("session_id")
    responses = data.get("responses", [])

    # Handle experience gate answer — store on profile so we don't ask again
    experience_answer = data.get("experience_answer")
    if experience_answer:
        user = request.user
        if experience_answer == "early":
            user.experience_level = "beginner"
        else:
            user.experience_level = "intermediate"
        user.save(update_fields=["experience_level"])

    if not session_id or not responses:
        return JsonResponse({"error": "session_id and responses required"}, status=400)

    # Determine instrument version (retake count)
    version = (
        QuestionnaireResponse.objects.filter(user=request.user, dimension__instrument=instrument)
        .values("session_id")
        .distinct()
        .count()
        + 1
    )

    created = []
    for resp in responses:
        dim_id = resp.get("dimension_id")
        if not dim_id:
            continue

        try:
            dim = QuestionDimension.objects.get(id=dim_id, instrument=instrument)
        except QuestionDimension.DoesNotExist:
            continue

        scenario = None
        scenario_id = resp.get("scenario_id")
        if scenario_id:
            try:
                scenario = Scenario.objects.get(id=scenario_id)
            except Scenario.DoesNotExist:
                pass

        qr = QuestionnaireResponse.objects.create(
            user=request.user,
            dimension=dim,
            instrument_version=version,
            session_id=session_id,
            score=resp.get("score"),
            scenario=scenario,
            option_chosen=resp.get("option_chosen", ""),
        )
        created.append(str(qr.id))

    logger.info("Harada %s v%d: %d responses from %s", instrument, version, len(created), request.user.email)

    return JsonResponse(
        {
            "instrument": instrument,
            "version": version,
            "session_id": session_id,
            "responses_saved": len(created),
        }
    )


@require_http_methods(["GET"])
@gated_paid
def get_response_history(request):
    """
    Get questionnaire response history for the current user.

    GET /api/harada/questionnaire/history/?instrument=ci_readiness
    """
    instrument = request.GET.get("instrument", "ci_readiness")

    sessions = (
        QuestionnaireResponse.objects.filter(user=request.user, dimension__instrument=instrument)
        .values("session_id", "instrument_version")
        .distinct()
        .order_by("-instrument_version")
    )

    history = []
    for sess in sessions:
        responses = QuestionnaireResponse.objects.filter(
            user=request.user, session_id=sess["session_id"]
        ).select_related("dimension", "scenario")

        items = []
        for r in responses:
            item = {
                "dimension_number": r.dimension.dimension_number,
                "dimension_name": r.dimension.name,
                "response_type": r.dimension.response_type,
                "timestamp": r.timestamp.isoformat(),
            }
            if r.score is not None:
                item["score"] = r.score
            if r.option_chosen:
                item["option_chosen"] = r.option_chosen
                if r.scenario:
                    item["scenario_key"] = r.scenario.scenario_key
            items.append(item)

        history.append(
            {
                "session_id": str(sess["session_id"]),
                "version": sess["instrument_version"],
                "timestamp": responses.first().timestamp.isoformat() if responses.exists() else None,
                "responses": items,
            }
        )

    return JsonResponse({"instrument": instrument, "history": history})


# ---------------------------------------------------------------------------
# Archetype
# ---------------------------------------------------------------------------


@require_http_methods(["GET"])
@gated_paid
def get_archetype(request):
    """
    Get the user's current archetype assignment from CI Readiness clustering.

    GET /api/harada/archetype/
    """
    from core.models import ArchetypeAssignment

    assignment = ArchetypeAssignment.objects.filter(user=request.user).order_by("-created_at").first()

    if not assignment:
        return JsonResponse(
            {"archetype": None, "message": "No archetype assigned yet. Complete the CI Readiness questionnaire."}
        )

    # Get all assignments for trajectory
    all_assignments = ArchetypeAssignment.objects.filter(user=request.user).order_by("created_at")

    return JsonResponse(
        {
            "archetype": {
                "cluster_id": assignment.cluster_id,
                "cluster_label": assignment.cluster_label or f"Archetype {assignment.cluster_id + 1}",
                "feature_vector": assignment.feature_vector,
                "cluster_distances": assignment.cluster_distances,
                "silhouette_score": assignment.silhouette_score,
                "version": assignment.instrument_version,
                "assigned_at": assignment.created_at.isoformat(),
            },
            "trajectory": [
                {
                    "cluster_id": a.cluster_id,
                    "cluster_label": a.cluster_label or f"Archetype {a.cluster_id + 1}",
                    "version": a.instrument_version,
                    "assigned_at": a.created_at.isoformat(),
                }
                for a in all_assignments
            ],
        }
    )


# ---------------------------------------------------------------------------
# Goals
# ---------------------------------------------------------------------------


def _serialize_goal(g):
    return {
        "id": str(g.id),
        "parent_id": str(g.parent_id) if g.parent_id else None,
        "horizon": g.horizon,
        "title": g.title,
        "description": g.description,
        "status": g.status,
        "service_at_home": g.service_at_home,
        "service_at_work": g.service_at_work,
        "perspectives": g.perspectives,
        "target_date": str(g.target_date) if g.target_date else None,
        "achieved_at": g.achieved_at.isoformat() if g.achieved_at else None,
        "created_at": g.created_at.isoformat(),
        "children": [_serialize_goal(c) for c in g.children.all()],
    }


@require_http_methods(["GET", "POST"])
@gated_paid
def list_create_goals(request):
    """
    GET  — List user's Harada goals (tree structure from roots).
    POST — Create a new goal.
    """
    if request.method == "GET":
        roots = HaradaGoal.objects.filter(user=request.user, parent=None).prefetch_related(
            "children__children__children"
        )
        return JsonResponse({"goals": [_serialize_goal(g) for g in roots]})

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    title = data.get("title", "").strip()
    if not title:
        return JsonResponse({"error": "title required"}, status=400)

    horizon = data.get("horizon", "long_term")
    if horizon not in [c[0] for c in HaradaGoal.Horizon.choices]:
        return JsonResponse({"error": f"Invalid horizon: {horizon}"}, status=400)

    parent = None
    parent_id = data.get("parent_id")
    if parent_id:
        try:
            parent = HaradaGoal.objects.get(id=parent_id, user=request.user)
        except HaradaGoal.DoesNotExist:
            return JsonResponse({"error": "Parent goal not found"}, status=404)

    goal = HaradaGoal.objects.create(
        user=request.user,
        parent=parent,
        horizon=horizon,
        title=title,
        description=data.get("description", ""),
        service_at_home=data.get("service_at_home", ""),
        service_at_work=data.get("service_at_work", ""),
        perspectives=data.get("perspectives", {}),
        target_date=data.get("target_date"),
    )

    logger.info("Harada goal created: %s [%s]", goal.title, goal.horizon)
    return JsonResponse(_serialize_goal(goal), status=201)


@require_http_methods(["GET", "PATCH", "DELETE"])
@gated_paid
def goal_detail(request, goal_id):
    """GET/PATCH/DELETE a Harada goal."""
    try:
        goal = HaradaGoal.objects.get(id=goal_id, user=request.user)
    except HaradaGoal.DoesNotExist:
        return JsonResponse({"error": "Goal not found"}, status=404)

    if request.method == "GET":
        return JsonResponse(_serialize_goal(goal))

    if request.method == "DELETE":
        goal.delete()
        return JsonResponse({"ok": True})

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    for field in [
        "title",
        "description",
        "status",
        "horizon",
        "service_at_home",
        "service_at_work",
        "perspectives",
        "target_date",
    ]:
        if field in data:
            setattr(goal, field, data[field])

    if data.get("status") == "achieved" and not goal.achieved_at:
        goal.achieved_at = timezone.now()

    goal.save()
    return JsonResponse(_serialize_goal(goal))


# ---------------------------------------------------------------------------
# 64-Window
# ---------------------------------------------------------------------------


@require_http_methods(["GET", "POST"])
@gated_paid
def list_create_window(request):
    """
    GET  — Get user's 64-window grid.
    POST — Create or update a cell.
    """
    if request.method == "GET":
        cells = Window64.objects.filter(user=request.user)
        grid = {}
        for cell in cells:
            if cell.goal_number not in grid:
                grid[cell.goal_number] = []
            grid[cell.goal_number].append(
                {
                    "id": str(cell.id),
                    "goal_number": cell.goal_number,
                    "position": cell.position,
                    "cell_type": cell.cell_type,
                    "text": cell.text,
                    "is_completed": cell.is_completed,
                    "completed_at": cell.completed_at.isoformat() if cell.completed_at else None,
                    "harada_goal_id": str(cell.harada_goal_id) if cell.harada_goal_id else None,
                }
            )
        return JsonResponse({"window": grid})

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    goal_number = data.get("goal_number")
    position = data.get("position")
    text = data.get("text", "").strip()

    if goal_number is None or position is None or not text:
        return JsonResponse({"error": "goal_number, position, and text required"}, status=400)

    if not (1 <= goal_number <= 8) or not (0 <= position <= 8):
        return JsonResponse({"error": "goal_number 1-8, position 0-8"}, status=400)

    cell_type = "goal" if position == 0 else data.get("cell_type", "task")

    cell, created = Window64.objects.update_or_create(
        user=request.user,
        goal_number=goal_number,
        position=position,
        defaults={
            "cell_type": cell_type,
            "text": text,
            "harada_goal_id": data.get("harada_goal_id"),
        },
    )

    return JsonResponse(
        {
            "id": str(cell.id),
            "goal_number": cell.goal_number,
            "position": cell.position,
            "cell_type": cell.cell_type,
            "text": cell.text,
            "is_completed": cell.is_completed,
            "created": created,
        },
        status=201 if created else 200,
    )


@require_http_methods(["PATCH"])
@gated_paid
def update_window_cell(request, cell_id):
    """Mark a cell as completed or update text."""
    try:
        cell = Window64.objects.get(id=cell_id, user=request.user)
    except Window64.DoesNotExist:
        return JsonResponse({"error": "Cell not found"}, status=404)

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if "text" in data:
        cell.text = data["text"]
    if "is_completed" in data:
        cell.is_completed = data["is_completed"]
        if cell.is_completed and not cell.completed_at:
            cell.completed_at = timezone.now()
    if "cell_type" in data:
        cell.cell_type = data["cell_type"]

    cell.save()
    return JsonResponse(
        {
            "id": str(cell.id),
            "text": cell.text,
            "is_completed": cell.is_completed,
            "cell_type": cell.cell_type,
        }
    )


# ---------------------------------------------------------------------------
# Routine Tracker
# ---------------------------------------------------------------------------


@require_http_methods(["GET", "POST"])
@gated_paid
def check_routine(request):
    """
    GET  — Get today's routine checklist.
    POST — Check/uncheck a routine for a date.

    POST body: {"window_cell_id": uuid, "date": "2026-03-17", "is_completed": true}
    """
    if request.method == "GET":
        target_date = request.GET.get("date", str(date.today()))
        routines = Window64.objects.filter(user=request.user, cell_type="routine")
        checks = {c.window_cell_id: c for c in RoutineCheck.objects.filter(user=request.user, date=target_date)}

        items = []
        for r in routines:
            check = checks.get(r.id)
            items.append(
                {
                    "window_cell_id": str(r.id),
                    "goal_number": r.goal_number,
                    "text": r.text,
                    "is_completed": check.is_completed if check else False,
                    "notes": check.notes if check else "",
                }
            )

        return JsonResponse({"date": target_date, "routines": items})

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    cell_id = data.get("window_cell_id")
    check_date = data.get("date", str(date.today()))
    is_completed = data.get("is_completed", True)

    if not cell_id:
        return JsonResponse({"error": "window_cell_id required"}, status=400)

    try:
        cell = Window64.objects.get(id=cell_id, user=request.user, cell_type="routine")
    except Window64.DoesNotExist:
        return JsonResponse({"error": "Routine not found"}, status=404)

    check, created = RoutineCheck.objects.update_or_create(
        user=request.user,
        window_cell=cell,
        date=check_date,
        defaults={"is_completed": is_completed, "notes": data.get("notes", "")},
    )

    return JsonResponse(
        {
            "window_cell_id": str(cell.id),
            "date": str(check.date),
            "is_completed": check.is_completed,
            "created": created,
        }
    )


@require_http_methods(["GET"])
@gated_paid
def routine_history(request):
    """
    Get routine completion history for streak tracking.

    GET /api/harada/routines/history/?days=30
    """
    days = int(request.GET.get("days", 30))
    start_date = date.today() - timedelta(days=days)

    checks = (
        RoutineCheck.objects.filter(user=request.user, date__gte=start_date)
        .select_related("window_cell")
        .order_by("date")
    )

    # Build daily summary
    by_date = {}
    for c in checks:
        d = str(c.date)
        if d not in by_date:
            by_date[d] = {"date": d, "completed": 0, "total": 0}
        by_date[d]["total"] += 1
        if c.is_completed:
            by_date[d]["completed"] += 1

    # Calculate streak
    streak = 0
    today = date.today()
    routines_count = Window64.objects.filter(user=request.user, cell_type="routine").count()
    if routines_count > 0:
        check_date = today
        while True:
            day_checks = RoutineCheck.objects.filter(user=request.user, date=check_date, is_completed=True).count()
            if day_checks >= routines_count:
                streak += 1
                check_date -= timedelta(days=1)
            else:
                break

    return JsonResponse(
        {
            "days": sorted(by_date.values(), key=lambda x: x["date"]),
            "streak": streak,
            "routines_count": routines_count,
        }
    )


# ---------------------------------------------------------------------------
# Daily Diary
# ---------------------------------------------------------------------------


def _serialize_diary(d):
    return {
        "id": str(d.id),
        "date": str(d.date),
        "daily_phrase": d.daily_phrase,
        "time_blocks": d.time_blocks,
        "top_tasks": d.top_tasks,
        "scores": {
            "overall": d.score_overall,
            "mental": d.score_mental,
            "body": d.score_body,
            "work": d.score_work,
            "relations": d.score_relations,
            "life": d.score_life,
            "learning": d.score_learning,
            "routines": d.score_routines,
        },
        "score_total": d.total_score,
        "score_comments": d.score_comments,
        "challenges": d.challenges,
        "what_differently": d.what_differently,
        "notes": d.notes,
        "tasks_completed": d.tasks_completed,
        "created_at": d.created_at.isoformat(),
    }


@require_http_methods(["GET", "POST"])
@gated_paid
def list_create_diary(request):
    """
    GET  — List diary entries (last 30 days by default).
    POST — Create or update today's diary entry.
    """
    if request.method == "GET":
        days = int(request.GET.get("days", 30))
        start = date.today() - timedelta(days=days)
        entries = DailyDiary.objects.filter(user=request.user, date__gte=start)
        return JsonResponse({"entries": [_serialize_diary(d) for d in entries]})

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    diary_date = data.get("date", str(date.today()))

    scores = data.get("scores", {})
    diary, created = DailyDiary.objects.update_or_create(
        user=request.user,
        date=diary_date,
        defaults={
            "daily_phrase": data.get("daily_phrase", ""),
            "time_blocks": data.get("time_blocks", []),
            "top_tasks": data.get("top_tasks", []),
            "score_overall": scores.get("overall"),
            "score_mental": scores.get("mental"),
            "score_body": scores.get("body"),
            "score_work": scores.get("work"),
            "score_relations": scores.get("relations"),
            "score_life": scores.get("life"),
            "score_learning": scores.get("learning"),
            "score_routines": scores.get("routines"),
            "score_comments": data.get("score_comments", {}),
            "challenges": data.get("challenges", ""),
            "what_differently": data.get("what_differently", ""),
            "notes": data.get("notes", ""),
        },
    )

    logger.info("Diary %s for %s: total=%s", "created" if created else "updated", diary_date, diary.total_score)
    return JsonResponse(_serialize_diary(diary), status=201 if created else 200)


@require_http_methods(["GET", "PATCH", "DELETE"])
@gated_paid
def diary_detail(request, diary_date):
    """GET/PATCH/DELETE a specific diary entry by date."""
    try:
        diary = DailyDiary.objects.get(user=request.user, date=diary_date)
    except DailyDiary.DoesNotExist:
        return JsonResponse({"error": "No diary entry for this date"}, status=404)

    if request.method == "GET":
        return JsonResponse(_serialize_diary(diary))

    if request.method == "DELETE":
        diary.delete()
        return JsonResponse({"ok": True})

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    for field in [
        "daily_phrase",
        "time_blocks",
        "top_tasks",
        "challenges",
        "what_differently",
        "notes",
        "score_comments",
    ]:
        if field in data:
            setattr(diary, field, data[field])

    scores = data.get("scores", {})
    for dim in ["overall", "mental", "body", "work", "relations", "life", "learning", "routines"]:
        if dim in scores:
            setattr(diary, f"score_{dim}", scores[dim])

    diary.save()
    return JsonResponse(_serialize_diary(diary))
