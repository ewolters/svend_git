"""
Management command to run compliance checks.

Usage:
    python manage.py run_compliance              # run today's rotating checks
    python manage.py run_compliance --all        # run all 25 checks
    python manage.py run_compliance --check=ssl_tls  # run a specific check
    python manage.py run_compliance --report     # generate monthly report
    python manage.py run_compliance --standards  # verbose per-assertion standards output
    python manage.py run_compliance --check=code_style --json  # JSON output (for CI artifacts)
"""

import json
import sys

from django.core.management.base import BaseCommand

from syn.audit.compliance import ALL_CHECKS, generate_monthly_report, run_check, run_daily_checks


class Command(BaseCommand):
    help = "Run automated compliance checks"

    def add_arguments(self, parser):
        parser.add_argument("--all", action="store_true", help="Run all checks")
        parser.add_argument("--check", type=str, help="Run a specific check by name")
        parser.add_argument("--report", action="store_true", help="Generate monthly report")
        parser.add_argument(
            "--standards", action="store_true", help="Run standards checks with verbose per-assertion output"
        )
        parser.add_argument("--run-tests", action="store_true", help="Execute linked tests (use with --standards)")
        parser.add_argument("--json", action="store_true", help="Output results as JSON (for CI artifacts)")
        parser.add_argument("--exit-code", action="store_true", help="Exit with code 1 if any check fails")

    def handle(self, *args, **options):
        if options["report"]:
            report = generate_monthly_report()
            self.stdout.write(
                self.style.SUCCESS(
                    f"Report generated: {report.period_start} - {report.period_end} "
                    f"({report.pass_rate:.1f}% pass rate, {report.total_checks} checks)"
                )
            )
            return

        if options["standards"]:
            self._run_standards_verbose(run_tests=options.get("run_tests", False))
            return

        use_json = options["json"]
        use_exit_code = options["exit_code"]

        if options["check"]:
            name = options["check"]
            if name not in ALL_CHECKS:
                self.stderr.write(self.style.ERROR(f"Unknown check: {name}. Available: {', '.join(ALL_CHECKS.keys())}"))
                return
            result = run_check(name)
            if use_json:
                self.stdout.write(json.dumps(self._check_to_dict(result), indent=2))
            else:
                self._print_result(result)
            if use_exit_code and result.status in ("fail", "error"):
                sys.exit(1)
            return

        if options["all"]:
            results = [run_check(name) for name in ALL_CHECKS]
        else:
            results = run_daily_checks()

        if use_json:
            self.stdout.write(
                json.dumps(
                    {
                        "summary": {
                            "passed": sum(1 for r in results if r.status == "pass"),
                            "failed": sum(1 for r in results if r.status == "fail"),
                            "warnings": sum(1 for r in results if r.status == "warning"),
                            "errors": sum(1 for r in results if r.status == "error"),
                            "total": len(results),
                        },
                        "checks": [self._check_to_dict(r) for r in results],
                    },
                    indent=2,
                )
            )
        else:
            passed = sum(1 for r in results if r.status == "pass")
            failed = sum(1 for r in results if r.status == "fail")
            warnings = sum(1 for r in results if r.status == "warning")
            errors = sum(1 for r in results if r.status == "error")

            for r in results:
                self._print_result(r)

            self.stdout.write("")
            self.stdout.write(f"Summary: {passed} passed, {failed} failed, {warnings} warnings, {errors} errors")

        if use_exit_code and any(r.status in ("fail", "error") for r in results):
            sys.exit(1)

    @staticmethod
    def _check_to_dict(check):
        """Serialize a ComplianceCheck model instance to a JSON-safe dict."""
        return {
            "check": check.check_name,
            "status": check.status,
            "category": check.category,
            "duration_ms": round(check.duration_ms, 1),
            "soc2_controls": check.soc2_controls or [],
            "details": check.details or {},
        }

    def _run_standards_verbose(self, run_tests=False):
        """Run standards checks with per-assertion detail."""
        from syn.audit.standards import parse_all_standards, verify_assertion

        assertions = parse_all_standards()
        if not assertions:
            self.stderr.write(self.style.WARNING("No standards assertions found."))
            return

        current_std = ""
        passed = failed = warnings = 0
        tests_linked = tests_passed = tests_failed = tests_skipped = tests_missing = 0

        for a in assertions:
            if a.standard != current_std:
                current_std = a.standard
                self.stdout.write(f"\n  {current_std}")
                self.stdout.write("  " + "=" * 40)

            result = verify_assertion(a, run_tests=run_tests)
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

            self.stdout.write(style(f"    [{status.upper():7s}] {a.check_id}"))
            self.stdout.write(f"             {a.text[:90]}")

            # Show impl results
            for ic in result.get("impl_checks", []):
                mark = "ok" if ic["ok"] else "FAIL"
                self.stdout.write(f"             impl: [{mark}] {ic['impl']}")

            # Show code check results
            for cc in result.get("code_checks", []):
                mark = "ok" if cc["ok"] else "FAIL"
                self.stdout.write(f"             code:{cc['type']}: [{mark}] {cc['message'][:70]}")

            # Show test results
            for tc in result.get("test_checks", []):
                tests_linked += 1
                if tc.get("ran"):
                    status = tc.get("status", "")
                    if status == "pass" or tc.get("passed"):
                        mark = "PASS"
                        tests_passed += 1
                    elif status == "skip":
                        mark = "SKIP"
                        tests_skipped += 1
                    else:
                        mark = "FAIL"
                        tests_failed += 1
                elif tc.get("status") == "skip":
                    mark = "SKIP"
                    tests_skipped += 1
                elif tc.get("exists"):
                    mark = "exists"
                else:
                    mark = "MISSING"
                    tests_missing += 1
                self.stdout.write(f"             test: [{mark}] {tc['test']}")

        summary = f"\n  Total: {len(assertions)} assertions — {passed} passed, {failed} failed, {warnings} warnings"
        if tests_linked > 0:
            summary += f"\n  Tests: {tests_linked} linked"
            if run_tests or tests_passed > 0:
                summary += f", {tests_passed} passed, {tests_failed} failed"
                if tests_skipped > 0:
                    summary += f", {tests_skipped} skipped"
            if tests_missing > 0:
                summary += f", {tests_missing} missing"
        self.stdout.write(summary)

    def _print_result(self, check):
        style = {
            "pass": self.style.SUCCESS,
            "fail": self.style.ERROR,
            "warning": self.style.WARNING,
            "error": self.style.ERROR,
        }.get(check.status, self.style.NOTICE)

        self.stdout.write(
            style(f"  [{check.status.upper():7s}] {check.check_name} ({check.duration_ms:.0f}ms) — {check.category}")
        )
