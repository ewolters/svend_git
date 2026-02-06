"""Generate invite codes for alpha access."""

from django.core.management.base import BaseCommand

from accounts.models import InviteCode


class Command(BaseCommand):
    help = "Generate invite codes for alpha access"

    def add_arguments(self, parser):
        parser.add_argument(
            "count",
            type=int,
            nargs="?",
            default=1,
            help="Number of codes to generate (default: 1)",
        )
        parser.add_argument(
            "--uses",
            type=int,
            default=1,
            help="Max uses per code (default: 1)",
        )
        parser.add_argument(
            "--note",
            type=str,
            default="",
            help="Note for these codes (e.g., 'For Mom')",
        )
        parser.add_argument(
            "--list",
            action="store_true",
            help="List all existing codes instead of generating",
        )

    def handle(self, *args, **options):
        if options["list"]:
            self.list_codes()
            return

        codes = InviteCode.generate(
            count=options["count"],
            max_uses=options["uses"],
            note=options["note"],
        )

        self.stdout.write(self.style.SUCCESS(f"\nGenerated {len(codes)} invite code(s):\n"))
        for code in codes:
            self.stdout.write(f"  {code.code}")
            if code.note:
                self.stdout.write(f"  (note: {code.note})")
            self.stdout.write("")

        self.stdout.write(self.style.NOTICE(
            f"\nEach code can be used {options['uses']} time(s)."
        ))
        self.stdout.write(self.style.NOTICE(
            "Share these with your alpha testers!"
        ))

    def list_codes(self):
        """List all existing invite codes."""
        codes = InviteCode.objects.all().order_by("-created_at")

        if not codes:
            self.stdout.write(self.style.WARNING("No invite codes found."))
            return

        self.stdout.write(self.style.SUCCESS(f"\n{codes.count()} invite code(s):\n"))
        self.stdout.write(f"{'Code':<15} {'Uses':<10} {'Status':<10} {'Note'}")
        self.stdout.write("-" * 60)

        for code in codes:
            status = "valid" if code.is_valid else "exhausted"
            if not code.is_active:
                status = "deactivated"
            self.stdout.write(
                f"{code.code:<15} {code.times_used}/{code.max_uses:<7} {status:<10} {code.note}"
            )
