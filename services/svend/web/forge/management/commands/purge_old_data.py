"""Purge old data according to retention policy.

Retention schedule:
  - Forge jobs: 30 days
  - Triage results: 30 days
  - DSW results: 30 days
  - TraceLog: 30 days
  - AgentLog: 30 days
  - TrainingCandidate (exported/rejected): 30 / 7 days
  - EventLog: 90 days
  - RequestMetric: 90 days
  - SharedConversation (expired): delete on expiry
  - BlogView (api.models): 180 days

Usage:
    python manage.py purge_old_data --dry-run
    python manage.py purge_old_data
"""

import os
from datetime import timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone

from agents_api.models import AgentLog, DSWResult, TriageResult
from chat.models import EventLog, SharedConversation, TraceLog, TrainingCandidate
from forge.models import Job


class Command(BaseCommand):
    help = "Purge old data according to retention policy"

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be deleted without actually deleting",
        )

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        now = timezone.now()
        total_deleted = 0

        if dry_run:
            self.stdout.write(self.style.WARNING("DRY RUN - no data will be deleted\n"))

        # ── 30-day retention ──────────────────────────────────────────
        cutoff_30 = now - timedelta(days=30)

        # Forge jobs (with file cleanup)
        old_jobs = Job.objects.filter(created_at__lt=cutoff_30)
        job_count = old_jobs.count()
        if job_count:
            if not dry_run:
                for job in old_jobs:
                    if job.result_path and os.path.exists(job.result_path):
                        try:
                            os.remove(job.result_path)
                        except OSError:
                            pass
                old_jobs.delete()
            self.stdout.write(f"Forge jobs: {job_count}")
            total_deleted += job_count

        # Triage results
        count = TriageResult.objects.filter(created_at__lt=cutoff_30).count()
        if count:
            if not dry_run:
                TriageResult.objects.filter(created_at__lt=cutoff_30).delete()
            self.stdout.write(f"Triage results: {count}")
            total_deleted += count

        # DSW results
        count = DSWResult.objects.filter(created_at__lt=cutoff_30).count()
        if count:
            if not dry_run:
                DSWResult.objects.filter(created_at__lt=cutoff_30).delete()
            self.stdout.write(f"DSW results: {count}")
            total_deleted += count

        # Trace logs
        count = TraceLog.objects.filter(created_at__lt=cutoff_30).count()
        if count:
            if not dry_run:
                TraceLog.objects.filter(created_at__lt=cutoff_30).delete()
            self.stdout.write(f"Trace logs: {count}")
            total_deleted += count

        # Agent logs
        count = AgentLog.objects.filter(created_at__lt=cutoff_30).count()
        if count:
            if not dry_run:
                AgentLog.objects.filter(created_at__lt=cutoff_30).delete()
            self.stdout.write(f"Agent logs: {count}")
            total_deleted += count

        # Training candidates — exported: 30 days, rejected: 7 days
        count_exported = TrainingCandidate.objects.filter(
            status="exported", created_at__lt=cutoff_30
        ).count()
        cutoff_7 = now - timedelta(days=7)
        count_rejected = TrainingCandidate.objects.filter(
            status="rejected", created_at__lt=cutoff_7
        ).count()
        tc_total = count_exported + count_rejected
        if tc_total:
            if not dry_run:
                TrainingCandidate.objects.filter(
                    status="exported", created_at__lt=cutoff_30
                ).delete()
                TrainingCandidate.objects.filter(
                    status="rejected", created_at__lt=cutoff_7
                ).delete()
            self.stdout.write(
                f"Training candidates: {tc_total} (exported: {count_exported}, rejected: {count_rejected})"
            )
            total_deleted += tc_total

        # ── 90-day retention ──────────────────────────────────────────
        cutoff_90 = now - timedelta(days=90)

        # Event logs
        count = EventLog.objects.filter(created_at__lt=cutoff_90).count()
        if count:
            if not dry_run:
                EventLog.objects.filter(created_at__lt=cutoff_90).delete()
            self.stdout.write(f"Event logs: {count}")
            total_deleted += count

        # Request metrics (HTTP telemetry)
        from syn.log.models import RequestMetric

        count = RequestMetric.objects.filter(created_at__lt=cutoff_90).count()
        if count:
            if not dry_run:
                RequestMetric.objects.filter(created_at__lt=cutoff_90).delete()
            self.stdout.write(f"Request metrics: {count}")
            total_deleted += count

        # ── Expired shared conversations ──────────────────────────────
        count = SharedConversation.objects.filter(
            expires_at__isnull=False, expires_at__lt=now
        ).count()
        if count:
            if not dry_run:
                SharedConversation.objects.filter(
                    expires_at__isnull=False, expires_at__lt=now
                ).delete()
            self.stdout.write(f"Expired shared conversations: {count}")
            total_deleted += count

        # ── 180-day retention ─────────────────────────────────────────
        cutoff_180 = now - timedelta(days=180)

        # Blog views (api.models.BlogView if it exists)
        try:
            from api.models import BlogView

            count = BlogView.objects.filter(created_at__lt=cutoff_180).count()
            if count:
                if not dry_run:
                    BlogView.objects.filter(created_at__lt=cutoff_180).delete()
                self.stdout.write(f"Blog views: {count}")
                total_deleted += count
        except (ImportError, Exception):
            pass

        # ── Summary ───────────────────────────────────────────────────
        if total_deleted == 0:
            self.stdout.write(self.style.SUCCESS("Nothing to purge"))
        else:
            prefix = "Would delete" if dry_run else "Deleted"
            self.stdout.write(
                self.style.SUCCESS(f"\n{prefix} {total_deleted} total records")
            )
