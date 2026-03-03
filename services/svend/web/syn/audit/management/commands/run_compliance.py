"""
Management command to run compliance checks.

Usage:
    python manage.py run_compliance              # run today's rotating checks
    python manage.py run_compliance --all        # run all 10 checks
    python manage.py run_compliance --check=ssl_tls  # run a specific check
    python manage.py run_compliance --report     # generate monthly report
"""

from django.core.management.base import BaseCommand

from syn.audit.compliance import ALL_CHECKS, generate_monthly_report, run_check, run_daily_checks


class Command(BaseCommand):
    help = "Run automated compliance checks"

    def add_arguments(self, parser):
        parser.add_argument("--all", action="store_true", help="Run all checks")
        parser.add_argument("--check", type=str, help="Run a specific check by name")
        parser.add_argument("--report", action="store_true", help="Generate monthly report")

    def handle(self, *args, **options):
        if options["report"]:
            report = generate_monthly_report()
            self.stdout.write(self.style.SUCCESS(
                f"Report generated: {report.period_start} - {report.period_end} "
                f"({report.pass_rate:.1f}% pass rate, {report.total_checks} checks)"
            ))
            return

        if options["check"]:
            name = options["check"]
            if name not in ALL_CHECKS:
                self.stderr.write(self.style.ERROR(
                    f"Unknown check: {name}. Available: {', '.join(ALL_CHECKS.keys())}"
                ))
                return
            result = run_check(name)
            self._print_result(result)
            return

        if options["all"]:
            results = [run_check(name) for name in ALL_CHECKS]
        else:
            results = run_daily_checks()

        passed = sum(1 for r in results if r.status == "pass")
        failed = sum(1 for r in results if r.status == "fail")
        warnings = sum(1 for r in results if r.status == "warning")
        errors = sum(1 for r in results if r.status == "error")

        for r in results:
            self._print_result(r)

        self.stdout.write("")
        self.stdout.write(
            f"Summary: {passed} passed, {failed} failed, "
            f"{warnings} warnings, {errors} errors"
        )

    def _print_result(self, check):
        style = {
            "pass": self.style.SUCCESS,
            "fail": self.style.ERROR,
            "warning": self.style.WARNING,
            "error": self.style.ERROR,
        }.get(check.status, self.style.NOTICE)

        self.stdout.write(style(
            f"  [{check.status.upper():7s}] {check.check_name} "
            f"({check.duration_ms:.0f}ms) — {check.category}"
        ))
