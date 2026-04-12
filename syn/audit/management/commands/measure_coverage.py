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
    "agents_api/dsw/stats.py",
    "agents_api/dsw/stats_parametric.py",
    "agents_api/dsw/stats_nonparametric.py",
    "agents_api/dsw/stats_regression.py",
    "agents_api/dsw/stats_posthoc.py",
    "agents_api/dsw/stats_quality.py",
    "agents_api/dsw/stats_advanced.py",
    "agents_api/dsw/stats_exploratory.py",
    "agents_api/dsw/spc.py",
    "agents_api/dsw/bayesian.py",
    "agents_api/dsw/reliability.py",
    "agents_api/dsw/dispatch.py",
    "agents_api/dsw/standardize.py",
    "agents_api/dsw/common.py",
    "agents_api/calibration.py",
]

TIER2_MODULES = [
    "agents_api/dsw/ml.py",
    "agents_api/dsw_views.py",
    "agents_api/spc_views.py",
    "agents_api/experimenter_views.py",
    "agents_api/synara_views.py",
    "accounts/permissions.py",
    "accounts/models.py",
    "core/views.py",
    "agents_api/dsw/viz.py",
    "agents_api/dsw/simulation.py",
]

TIER3_MODULES = [
    "agents_api/fmea_views.py",
    "agents_api/rca_views.py",
    "agents_api/a3_views.py",
    "agents_api/hoshin_views.py",
    "agents_api/vsm_views.py",
    "agents_api/whiteboard_views.py",
    "agents_api/report_views.py",
    "agents_api/iso_views.py",
    "agents_api/learn_views.py",
    "agents_api/triage_views.py",
    "agents_api/forecast_views.py",
    "agents_api/guide_views.py",
    "agents_api/capa_views.py",
    "agents_api/autopilot_views.py",
    "agents_api/xmatrix_views.py",
    "agents_api/workflow_views.py",
    "agents_api/plantsim_views.py",
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
