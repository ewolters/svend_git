"""Harada Method background tasks — daily reminders via Tempora.

One cron task handles routine reminders, missed routine alerts,
diary reminders, and goal deadline warnings.
"""

import logging
from datetime import date, timedelta

from django.contrib.auth import get_user_model

logger = logging.getLogger("svend.harada")
User = get_user_model()

STREAK_MILESTONES = {7, 14, 30, 60, 90}


def harada_daily_reminders(payload, context=None):
    """Daily cron task: scan users for missed routines, diary, and goal deadlines.

    Runs at ~8AM UTC. Single pass per user — efficient.
    """
    from core.models import DailyDiary, HaradaGoal, RoutineCheck, Window64
    from notifications.helpers import notify

    today = date.today()
    yesterday = today - timedelta(days=1)

    stats = {
        "routine_missed": 0,
        "routine_reminder": 0,
        "diary_reminder": 0,
        "goal_due": 0,
    }

    # Get all users with active routines
    users_with_routines = (
        Window64.objects.filter(cell_type="routine")
        .values_list("user_id", flat=True)
        .distinct()
    )

    for user_id in users_with_routines:
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            continue

        routines = Window64.objects.filter(user=user, cell_type="routine")
        routine_count = routines.count()

        # --- Missed yesterday ---
        yesterday_checks = RoutineCheck.objects.filter(
            user=user, date=yesterday, is_completed=True
        ).count()

        if yesterday_checks < routine_count:
            missed = routine_count - yesterday_checks
            notify(
                recipient=user,
                notification_type="routine_missed",
                title=f"Missed {missed} routine{'s' if missed != 1 else ''} yesterday",
                message=f"You completed {yesterday_checks}/{routine_count} routines yesterday.",
                entity_type="routine",
            )
            stats["routine_missed"] += 1

        # --- Today's reminder (no checks yet) ---
        today_checks = RoutineCheck.objects.filter(user=user, date=today).count()
        if today_checks == 0 and routine_count > 0:
            notify(
                recipient=user,
                notification_type="routine_reminder",
                title=f"{routine_count} routine{'s' if routine_count != 1 else ''} to complete today",
                entity_type="routine",
            )
            stats["routine_reminder"] += 1

    # --- Diary reminder (users who had a diary yesterday but not today) ---
    users_with_diary = DailyDiary.objects.filter(date=yesterday).values_list(
        "user_id", flat=True
    )
    for user_id in users_with_diary:
        has_today = DailyDiary.objects.filter(user_id=user_id, date=today).exists()
        if not has_today:
            try:
                user = User.objects.get(id=user_id)
                notify(
                    recipient=user,
                    notification_type="diary_reminder",
                    title="Daily diary reminder",
                    message="Take a moment to plan your day and reflect.",
                    entity_type="diary",
                )
                stats["diary_reminder"] += 1
            except User.DoesNotExist:
                continue

    # --- Goal deadline warnings (within 7 days) ---
    deadline_window = today + timedelta(days=7)
    approaching_goals = HaradaGoal.objects.filter(
        status="active",
        target_date__lte=deadline_window,
        target_date__gte=today,
    ).select_related("user")

    for goal in approaching_goals:
        days_left = (goal.target_date - today).days
        notify(
            recipient=goal.user,
            notification_type="goal_due",
            title=f"Goal due in {days_left} day{'s' if days_left != 1 else ''}: {goal.title[:50]}",
            entity_type="harada_goal",
            entity_id=str(goal.id),
        )
        stats["goal_due"] += 1

    logger.info("Harada daily reminders: %s", stats)
    return stats


def check_streak_and_notify(user, routine_cell):
    """Called after a routine check-in. Notifies on streak milestones."""
    from core.models import RoutineCheck
    from notifications.helpers import notify

    # Count consecutive days (including today)
    streak = 0
    check_date = date.today()
    while True:
        exists = RoutineCheck.objects.filter(
            user=user, window_cell=routine_cell, date=check_date, is_completed=True
        ).exists()
        if exists:
            streak += 1
            check_date -= timedelta(days=1)
        else:
            break

    if streak in STREAK_MILESTONES:
        notify(
            recipient=user,
            notification_type="routine_streak",
            title=f"{streak}-day streak: {routine_cell.text[:50]}",
            message=f"You've maintained this routine for {streak} consecutive days.",
            entity_type="routine",
            entity_id=str(routine_cell.id),
        )
        logger.info(
            "Streak milestone %d for %s on '%s'", streak, user.email, routine_cell.text
        )

    return streak


def notify_hansei_due(user, goal):
    """Called when a goal is marked achieved. Prompts for Hansei Kai reflection."""
    from notifications.helpers import notify

    notify(
        recipient=user,
        notification_type="hansei_due",
        title=f"Reflection due: {goal.title[:50]}",
        message="You achieved your goal. Take time to reflect — what went well, what didn't, what will you do differently?",
        entity_type="harada_goal",
        entity_id=str(goal.id),
    )
