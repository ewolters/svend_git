"""
Management command to run compliance checks.

Usage:
    python manage.py run_compliance              # run today's rotating checks
    python manage.py run_compliance --all        # run all 11 checks
    python manage.py run_compliance --check=ssl_tls  # run a specific check
    python manage.py run_compliance --report     # generate monthly report
    python manage.py run_compliance --standards  # verbose per-assertion standards output
"""

from django.core.management.base import BaseCommand

from syn.audit.compliance import ALL_CHECKS, generate_monthly_report, run_check, run_daily_checks


class Command(BaseCommand):
    help = "Run automated compliance checks"

    def add_arguments(self, parser):
        parser.add_argument("--all", action="store_true", help="Run all checks")
        parser.add_argument("--check", type=str, help="Run a specific check by name")
        parser.add_argument("--report", action="store_true", help="Generate monthly report")
        parser.add_argument("--standards", action="store_true", help="Run standards checks with verbose per-assertion output")

    def handle(self, *args, **options):
        if options["report"]:
            report = generate_monthly_report()
            self.stdout.write(self.style.SUCCESS(
                f"Report generated: {report.period_start} - {report.period_end} "
                f"({report.pass_rate:.1f}% pass rate, {report.total_checks} checks)"
            ))
            return

        if options["standards"]:
            self._run_standards_verbose()
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

    def _run_standards_verbose(self):
        """Run standards checks with per-assertion detail."""
        from syn.audit.standards import parse_all_standards, verify_assertion

        assertions = parse_all_standards()
        if not assertions:
            self.stderr.write(self.style.WARNING("No standards assertions found."))
            return

        current_std = ""
        passed = failed = warnings = 0

        for a in assertions:
            if a.standard != current_std:
                current_std = a.standard
                self.stdout.write(f"\n  {current_std}")
                self.stdout.write("  " + "=" * 40)

            result = verify_assertion(a)
            status = result["status"]

            if status == "pass":
                passed += 1
                style = self.style.SUCCESS
            elif status == "fail":
                failed += 1
                style = self.style.ERROR
            else:
                warnings += 1
                style = self.style.WARNING

            self.stdout.write(style(
                f"    [{status.upper():7s}] {a.check_id}"
            ))
            self.stdout.write(f"             {a.text[:90]}")

            # Show impl results
            for ic in result.get("impl_checks", []):
                mark = "ok" if ic["ok"] else "FAIL"
                self.stdout.write(f"             impl: [{mark}] {ic['impl']}")

            # Show code check results
            for cc in result.get("code_checks", []):
                mark = "ok" if cc["ok"] else "FAIL"
                self.stdout.write(f"             code:{cc['type']}: [{mark}] {cc['message'][:70]}")

        self.stdout.write(f"\n  Total: {len(assertions)} assertions — "
                          f"{passed} passed, {failed} failed, {warnings} warnings")

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
