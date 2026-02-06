"""Purge Forge and Triage data older than 30 days."""

import os
from datetime import timedelta
from django.core.management.base import BaseCommand
from django.utils import timezone

from forge.models import Job
from agents_api.models import TriageResult, DSWResult


class Command(BaseCommand):
    help = "Purge Forge jobs and data older than 30 days"

    def add_arguments(self, parser):
        parser.add_argument(
            "--days",
            type=int,
            default=30,
            help="Delete data older than this many days (default: 30)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be deleted without actually deleting",
        )

    def handle(self, *args, **options):
        days = options["days"]
        dry_run = options["dry_run"]
        cutoff = timezone.now() - timedelta(days=days)

        self.stdout.write(f"Purging Forge data older than {days} days (before {cutoff})")
        if dry_run:
            self.stdout.write(self.style.WARNING("DRY RUN - no data will be deleted"))

        # Find old jobs
        old_jobs = Job.objects.filter(created_at__lt=cutoff)
        job_count = old_jobs.count()

        if job_count == 0:
            self.stdout.write(self.style.SUCCESS("No old jobs found"))
            return

        self.stdout.write(f"Found {job_count} jobs to delete")

        # Delete result files
        files_deleted = 0
        bytes_freed = 0

        for job in old_jobs:
            if job.result_path and os.path.exists(job.result_path):
                if dry_run:
                    self.stdout.write(f"  Would delete: {job.result_path}")
                else:
                    try:
                        size = os.path.getsize(job.result_path)
                        os.remove(job.result_path)
                        files_deleted += 1
                        bytes_freed += size
                    except OSError as e:
                        self.stdout.write(
                            self.style.WARNING(f"  Failed to delete {job.result_path}: {e}")
                        )

        # Delete job records
        if not dry_run:
            old_jobs.delete()

        # Purge old Triage results
        old_triage = TriageResult.objects.filter(created_at__lt=cutoff)
        triage_count = old_triage.count()
        if triage_count > 0:
            self.stdout.write(f"Found {triage_count} Triage results to delete")
            if not dry_run:
                old_triage.delete()

        # Purge old DSW results
        old_dsw = DSWResult.objects.filter(created_at__lt=cutoff)
        dsw_count = old_dsw.count()
        if dsw_count > 0:
            self.stdout.write(f"Found {dsw_count} DSW results to delete")
            if not dry_run:
                old_dsw.delete()

        # Summary
        if dry_run:
            self.stdout.write(self.style.WARNING(
                f"Would delete: {job_count} Forge jobs, {triage_count} Triage results, {dsw_count} DSW results"
            ))
        else:
            self.stdout.write(self.style.SUCCESS(
                f"Deleted {job_count} Forge jobs, {files_deleted} files, "
                f"freed {bytes_freed / 1024 / 1024:.1f} MB\n"
                f"Deleted {triage_count} Triage results, {dsw_count} DSW results"
            ))
