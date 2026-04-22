"""
Management command to measure code coverage and store a CalibrationReport.

Usage:
    coverage run manage.py test --keepdb && coverage json
    python manage.py measure_coverage

Standard: CAL-001 §5, §10.2
Compliance: SOC 2 CC4.1 (Monitoring Activities)
"""

import json
from datetime import date
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

# Module tier classification per CAL-001 §4
TIER1_MODULES = [
    "agents_api/analysis/stats/__init__.py",
    "agents_api/analysis/stats/parametric.py",
    "agents_api/analysis/stats/nonparametric.py",
    "agents_api/analysis/stats/regression.py",
    "agents_api/analysis/stats/posthoc.py",
    "agents_api/analysis/stats/quality.py",
    "agents_api/analysis/stats/advanced.py",
    "agents_api/analysis/exploratory/__init__.py",
    "agents_api/analysis/spc/__init__.py",
    "agents_api/analysis/bayesian/__init__.py",
    "agents_api/analysis/reliability/__init__.py",
    "agents_api/analysis/dispatch.py",
    "agents_api/analysis/standardize.py",
    "agents_api/analysis/common.py",
    "agents_api/calibration.py",
]

TIER2_MODULES = [
    "agents_api/analysis/ml/__init__.py",
    "dsw/analysis_views.py",
    "dsw/views.py",
    "dsw/experimenter_views.py",
    "agents_api/synara_views.py",
    "accounts/permissions.py",
    "accounts/models.py",
    "core/views.py",
    "agents_api/analysis/viz/__init__.py",
    "agents_api/analysis/simulation/__init__.py",
]

TIER3_MODULES = [
    "fmea/views.py",
    "rca/views.py",
    "a3/views.py",
    "hoshin/views.py",
    "vsm/views.py",
    "whiteboard/views.py",
    "reports/views.py",
    "learn/views.py",
    "triage/views.py",
    "dsw/forecast_views.py",
    "guide/views.py",
    "capa/views.py",
    "dsw/autopilot_views.py",
    "hoshin/xmatrix_views.py",
    "plantsim/views.py",
    "workbench/views.py",
    "forge/views.py",
    "files/views.py",
    "chat/views.py",
    "notifications/views.py",
]

TIER4_MODULES = [
    "syn/audit/compliance.py",
    "syn/sched/core.py",
    "api/internal_views.py",
    "api/views.py",
    "syn/audit/models.py",
    "syn/audit/standards.py",
    "agents_api/models.py",
    "syn/sched/models.py",
    "workbench/models.py",
]


def _tier_coverage(coverage_data, tier_modules):
    """Compute average coverage for a set of modules."""
    total_stmts = 0
    total_covered = 0
    files = coverage_data.get("files", {})
    for mod_path in tier_modules:
        # coverage.json uses absolute paths or relative — try both
        for key, data in files.items():
            if key.endswith(mod_path):
                summary = data.get("summary", {})
                total_stmts += summary.get("num_statements", 0)
                total_covered += summary.get("covered_lines", 0)
                break
    if total_stmts == 0:
        return None
    return round(total_covered / total_stmts * 100, 1)


class Command(BaseCommand):
    help = "Measure code coverage and store a CalibrationReport (CAL-001 §5)"

    def handle(self, *args, **options):
        from syn.audit.models import CalibrationReport

        base = Path(settings.BASE_DIR)
        coverage_json = base / "coverage.json"

        if not coverage_json.exists():
            self.stderr.write(
                self.style.ERROR(
                    "coverage.json not found. Run:\n  coverage run manage.py test --keepdb && coverage json"
                )
            )
            return

        data = json.loads(coverage_json.read_text())
        totals = data.get("totals", {})
        overall = totals.get("percent_covered", 0)

        # Per-tier coverage
        t1 = _tier_coverage(data, TIER1_MODULES)
        t2 = _tier_coverage(data, TIER2_MODULES)
        t3 = _tier_coverage(data, TIER3_MODULES)
        t4 = _tier_coverage(data, TIER4_MODULES)

        # Golden file count
        golden_dir = base / "agents_api" / "tests" / "golden"
        golden_count = len(list(golden_dir.glob("*.json"))) if golden_dir.exists() else 0

        # Complexity violations (files > 3000 lines)
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

        # Ratchet check
        last_report = CalibrationReport.objects.order_by("-date").first()
        ratchet_baseline = last_report.ratchet_baseline if last_report else 0.0
        ratchet_pass = overall >= ratchet_baseline

        # New ratchet is max of current and previous
        new_ratchet = max(overall, ratchet_baseline)

        report = CalibrationReport.objects.create(
            date=date.today(),
            overall_coverage=overall,
            tier1_coverage=t1,
            tier2_coverage=t2,
            tier3_coverage=t3,
            tier4_coverage=t4,
            golden_file_count=golden_count,
            complexity_violations=complexity_violations,
            ratchet_baseline=new_ratchet,
            details={
                "total_statements": totals.get("num_statements", 0),
                "covered_lines": totals.get("covered_lines", 0),
                "missing_lines": totals.get("missing_lines", 0),
            },
        )

        self.stdout.write(self.style.SUCCESS(f"CalibrationReport created: {report.id}"))
        self.stdout.write(f"  Date: {report.date}")
        self.stdout.write(f"  Overall coverage: {overall:.1f}%")
        self.stdout.write(f"  Tier 1: {t1}%  Tier 2: {t2}%  Tier 3: {t3}%  Tier 4: {t4}%")
        self.stdout.write(f"  Golden files: {golden_count}")
        self.stdout.write(f"  Complexity violations: {complexity_violations}")
        self.stdout.write(f"  Ratchet baseline: {ratchet_baseline:.1f}% → {new_ratchet:.1f}%")

        if ratchet_pass:
            self.stdout.write(self.style.SUCCESS("  Ratchet: PASS"))
        else:
            self.stdout.write(self.style.ERROR(f"  Ratchet: FAIL ({overall:.1f}% < {ratchet_baseline:.1f}%)"))
