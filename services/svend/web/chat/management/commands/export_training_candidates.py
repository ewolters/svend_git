"""Export training candidates to JSONL format for model training."""

import json
from datetime import datetime
from pathlib import Path

from django.core.management.base import BaseCommand

from chat.models import TrainingCandidate


class Command(BaseCommand):
    help = "Export training candidates to JSONL format"

    def add_arguments(self, parser):
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Output file path (default: training_candidates_YYYYMMDD.jsonl)",
        )
        parser.add_argument(
            "--status",
            type=str,
            choices=["pending", "approved", "all"],
            default="approved",
            help="Which candidates to export (default: approved)",
        )
        parser.add_argument(
            "--type",
            type=str,
            choices=["low_confidence", "verification_failed", "error", "user_flagged", "random_sample", "all"],
            default="all",
            help="Which candidate type to export (default: all)",
        )
        parser.add_argument(
            "--mark-exported",
            action="store_true",
            help="Mark exported candidates as 'exported' status",
        )
        parser.add_argument(
            "--with-corrections-only",
            action="store_true",
            help="Only export candidates that have corrected responses",
        )

    def handle(self, *args, **options):
        # Build query
        queryset = TrainingCandidate.objects.all()

        # Filter by status
        if options["status"] != "all":
            queryset = queryset.filter(status=options["status"])

        # Filter by type
        if options["type"] != "all":
            queryset = queryset.filter(candidate_type=options["type"])

        # Filter by corrections
        if options["with_corrections_only"]:
            queryset = queryset.exclude(corrected_response="")

        count = queryset.count()
        self.stdout.write(f"Found {count} candidates to export")

        if count == 0:
            self.stdout.write(self.style.WARNING("No candidates to export"))
            return

        # Output path
        if options["output"]:
            output_path = Path(options["output"])
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"training_candidates_{timestamp}.jsonl")

        # Export
        exported = 0
        with open(output_path, "w") as f:
            for candidate in queryset.iterator():
                training_data = candidate.to_training_format()
                f.write(json.dumps(training_data) + "\n")
                exported += 1

        self.stdout.write(self.style.SUCCESS(f"Exported {exported} candidates to {output_path}"))

        # Mark as exported if requested
        if options["mark_exported"]:
            updated = queryset.update(status=TrainingCandidate.Status.EXPORTED)
            self.stdout.write(f"Marked {updated} candidates as exported")

        # Print summary
        self.stdout.write("\n--- Summary by Type ---")
        for ctype in TrainingCandidate.CandidateType.choices:
            type_count = queryset.filter(candidate_type=ctype[0]).count()
            if type_count > 0:
                self.stdout.write(f"  {ctype[1]}: {type_count}")
