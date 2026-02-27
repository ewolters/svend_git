"""Reap zombie tasks — mark stale RUNNING tasks as FAILURE.

Tasks stuck in RUNNING state beyond a timeout are likely from crashed workers.
Run periodically via cron or Tempora schedule:

    python manage.py reap_zombie_tasks
    python manage.py reap_zombie_tasks --timeout-minutes 60
"""

import logging

from django.core.management.base import BaseCommand
from django.utils import timezone

from tempora.models import CognitiveTask
from tempora.types import TaskState

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_MINUTES = 30


class Command(BaseCommand):
    help = "Mark stale RUNNING tasks as FAILURE after timeout"

    def add_arguments(self, parser):
        parser.add_argument(
            "--timeout-minutes",
            type=int,
            default=DEFAULT_TIMEOUT_MINUTES,
            help=f"Minutes after which a RUNNING task is considered stale (default: {DEFAULT_TIMEOUT_MINUTES})",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be reaped without making changes",
        )

    def handle(self, *args, **options):
        timeout_minutes = options["timeout_minutes"]
        dry_run = options["dry_run"]
        cutoff = timezone.now() - timezone.timedelta(minutes=timeout_minutes)

        zombies = CognitiveTask.objects.filter(
            state=TaskState.RUNNING.value,
            started_at__lt=cutoff,
        )

        count = zombies.count()
        if count == 0:
            self.stdout.write("No zombie tasks found.")
            return

        if dry_run:
            self.stdout.write(f"[DRY RUN] Would reap {count} zombie task(s):")
            for task in zombies[:20]:
                age = (timezone.now() - task.started_at).total_seconds() / 60
                self.stdout.write(f"  {task.id} — {task.task_name} — running {age:.0f}m")
            return

        reaped = zombies.update(
            state=TaskState.FAILURE.value,
            completed_at=timezone.now(),
        )

        logger.warning(f"Reaped {reaped} zombie task(s) older than {timeout_minutes} minutes")
        self.stdout.write(self.style.SUCCESS(f"Reaped {reaped} zombie task(s)."))
