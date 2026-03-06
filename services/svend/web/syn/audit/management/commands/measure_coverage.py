"""
Management command to measure statistical symbol coverage and store a CalibrationReport.

Usage:
    python manage.py measure_coverage

Standard: CAL-001 §5, §10.2
Compliance: SOC 2 CC4.1 (Monitoring Activities)

Coverage metric: symbol governance scoped to statistical engine files —
% of symbols (functions/classes) in DSW, SPC, PBS, and calibration modules
with both <!-- impl: file:Symbol --> and <!-- test: --> hooks in standards.
"""

from datetime import date

from django.core.management.base import BaseCommand

# Statistical engine file prefixes — the code that calibration verifies
STATISTICAL_PREFIXES = (
    "agents_api/dsw/",
    "agents_api/spc.py",
    "agents_api/spc_views.py",
    "agents_api/pbs_engine.py",
    "agents_api/calibration.py",
)


def _is_statistical_file(rel_path):
    """Return True if rel_path is a statistical engine file."""
    return any(rel_path.startswith(p) for p in STATISTICAL_PREFIXES)


class Command(BaseCommand):
    help = "Measure statistical symbol coverage and store a CalibrationReport (CAL-001 §5)"

    def handle(self, *args, **options):
        from pathlib import Path

        from django.conf import settings

        from syn.audit.compliance import check_symbol_coverage
        from syn.audit.models import CalibrationReport

        base = Path(settings.BASE_DIR)

        # Primary metric: symbol governance scoped to statistical engines
        cov_result = check_symbol_coverage(file_filter=_is_statistical_file)
        details = cov_result.get("details", {})
        overall = details.get("covered_pct", 0)

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
            golden_file_count=golden_count,
            complexity_violations=complexity_violations,
            ratchet_baseline=new_ratchet,
            details={
                "metric": "statistical_symbol_governance",
                "scope": "statistical_engines",
                "file_prefixes": list(STATISTICAL_PREFIXES),
                "total_symbols": details.get("total_symbols", 0),
                "covered_symbols": details.get("covered_symbols", 0),
                "specified_untested": details.get("specified_untested", 0),
                "ungoverned": details.get("ungoverned", 0),
            },
        )

        self.stdout.write(self.style.SUCCESS(f"CalibrationReport created: {report.id}"))
        self.stdout.write(f"  Date: {report.date}")
        self.stdout.write(f"  Statistical symbol coverage: {overall:.1f}%")
        self.stdout.write(f"  Symbols: {details.get('covered_symbols', 0)} / {details.get('total_symbols', 0)}")
        self.stdout.write(f"  Scope: {', '.join(STATISTICAL_PREFIXES)}")
        self.stdout.write(f"  Golden files: {golden_count}")
        self.stdout.write(f"  Complexity violations: {complexity_violations}")
        self.stdout.write(f"  Ratchet baseline: {ratchet_baseline:.1f}% → {new_ratchet:.1f}%")

        if ratchet_pass:
            self.stdout.write(self.style.SUCCESS("  Ratchet: PASS"))
        else:
            self.stdout.write(self.style.ERROR(f"  Ratchet: FAIL ({overall:.1f}% < {ratchet_baseline:.1f}%)"))
