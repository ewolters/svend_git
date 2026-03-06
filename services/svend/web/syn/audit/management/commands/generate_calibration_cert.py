"""
Management command to generate a monthly calibration certificate.

Usage:
    python manage.py generate_calibration_cert

Standard: CAL-001 §10.3, §11
Compliance: SOC 2 CC4.1, ISO/IEC 17025:2017
"""

import json
from datetime import date
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Generate a monthly calibration certificate (CAL-001 §10.3)"

    def handle(self, *args, **options):
        from syn.audit.models import CalibrationReport

        base = Path(settings.BASE_DIR)
        findings = []
        status_overall = "pass"

        # 1. Run all calibration cases
        cal_data = {"cases_run": 0, "cases_passed": 0, "pass_rate": 0, "results": []}
        try:
            from agents_api.calibration import run_calibration

            cal_data = run_calibration(n=None)  # Run ALL cases
        except Exception as e:
            findings.append(f"Calibration run error: {e}")
            status_overall = "fail"

        # 2. Read coverage data if available
        overall_coverage = None
        coverage_json = base / "coverage.json"
        if coverage_json.exists():
            try:
                data = json.loads(coverage_json.read_text())
                overall_coverage = data.get("totals", {}).get("percent_covered", 0)
            except Exception as e:
                findings.append(f"Coverage read error: {e}")
        else:
            findings.append("coverage.json not found — run coverage before generating cert")

        # 3. Count golden files
        golden_dir = base / "agents_api" / "tests" / "golden"
        golden_count = len(list(golden_dir.glob("*.json"))) if golden_dir.exists() else 0

        # 4. Count complexity violations
        complexity_violations = 0
        for py_file in base.rglob("*.py"):
            rel = str(py_file.relative_to(base))
            if "/migrations/" in rel or "/test" in rel:
                continue
            try:
                lines = sum(1 for _ in py_file.open())
                if lines > 3000:
                    complexity_violations += 1
            except Exception:
                pass

        # 5. Ratchet check
        last_report = CalibrationReport.objects.order_by("-date").first()
        ratchet_baseline = last_report.ratchet_baseline if last_report else 0.0
        if overall_coverage is not None and overall_coverage < ratchet_baseline:
            findings.append(f"Coverage regression: {overall_coverage:.1f}% < ratchet {ratchet_baseline:.1f}%")
            status_overall = "fail"

        # Determine certificate pass/fail
        cal_pass_rate = cal_data.get("pass_rate", 0)
        if cal_pass_rate < 100:
            status_overall = "fail" if cal_pass_rate < 80 else "warning"
            findings.append(f"Calibration pass rate: {cal_pass_rate}%")

        new_ratchet = max(overall_coverage or 0, ratchet_baseline)

        # Create certificate report
        report = CalibrationReport.objects.create(
            date=date.today(),
            overall_coverage=overall_coverage,
            calibration_pass_rate=cal_pass_rate,
            calibration_cases_run=cal_data.get("cases_run", 0),
            calibration_cases_passed=cal_data.get("cases_passed", 0),
            golden_file_count=golden_count,
            complexity_violations=complexity_violations,
            ratchet_baseline=new_ratchet,
            is_certificate=True,
            details={
                "status": status_overall,
                "findings": findings,
                "calibration_results": cal_data.get("results", []),
                "complexity_violations_count": complexity_violations,
            },
        )

        self.stdout.write("")
        self.stdout.write("=" * 60)
        self.stdout.write("  CALIBRATION CERTIFICATE")
        self.stdout.write(f"  Date: {report.date}")
        self.stdout.write(f"  ID: {report.id}")
        self.stdout.write("=" * 60)
        self.stdout.write("")

        if overall_coverage is not None:
            self.stdout.write(f"  Coverage: {overall_coverage:.1f}%")
        self.stdout.write(
            f"  Calibration: {cal_data.get('cases_passed', 0)}/{cal_data.get('cases_run', 0)} cases passed ({cal_pass_rate}%)"
        )
        self.stdout.write(f"  Golden files: {golden_count}")
        self.stdout.write(f"  Complexity violations: {complexity_violations}")
        self.stdout.write(f"  Ratchet baseline: {new_ratchet:.1f}%")
        self.stdout.write("")

        if findings:
            self.stdout.write("  Findings:")
            for f in findings:
                self.stdout.write(f"    - {f}")
            self.stdout.write("")

        if status_overall == "pass":
            self.stdout.write(self.style.SUCCESS("  RESULT: PASS"))
        elif status_overall == "warning":
            self.stdout.write(self.style.WARNING("  RESULT: PASS WITH WARNINGS"))
        else:
            self.stdout.write(self.style.ERROR("  RESULT: FAIL"))

        self.stdout.write("=" * 60)
