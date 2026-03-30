# Phase 0: Compliance System Prep & Cleanup

**Date:** 2026-03-30
**Standard:** CMP-001 v1.6
**File:** `syn/audit/compliance.py` (5,965 lines, 37 checks)

---

## The Problem

Two compliance systems now exist:
1. **SVEND** (`syn/audit/compliance.py`) — 37 checks, runs via `manage.py run_compliance`
2. **ForgeGov** (`~/forgegov/`) — 6-stage pipeline, runs via `forgegov run`

They overlap on 11 checks. After the migration, some SVEND checks become forgegov's responsibility (package-level concerns), some stay in SVEND (Django/server concerns), and one new bridge check connects them.

---

## Check Classification

### STAYS IN SVEND (26 checks) — needs Django, server, or database

| # | Check | Why SVEND |
|---|-------|-----------|
| 1 | `check_audit_integrity` | Reads AuditLog Django models |
| 2 | `check_security_config` | Reads Django settings, server config |
| 3 | `check_encryption_status` | Tests field-level encryption on Django models |
| 4 | `check_access_logging` | Reads SysLogEntry Django models |
| 5 | `check_backup_freshness` | Checks server backup files |
| 6 | `check_password_policy` | Reads Django AUTH_PASSWORD_VALIDATORS |
| 7 | `check_data_retention` | Queries database record ages |
| 8 | `check_privacy_data_export` | Tests privacy export Django view |
| 9 | `check_ssl_tls` | Checks server TLS config |
| 10 | `check_change_management` | Reads ChangeRequest Django models |
| 11 | `check_session_security` | Reads Django session settings |
| 12 | `check_error_handling` | Tests ErrorEnvelopeMiddleware |
| 13 | `check_rate_limiting` | Tests DRF throttle config |
| 14 | `check_secret_management` | Checks environment variables on server |
| 15 | `check_log_completeness` | Reads logging config |
| 16 | `check_security_headers` | Tests CSP/HSTS settings |
| 17 | `check_incident_readiness` | Checks incident response config |
| 18 | `check_sla_compliance` | Reads SLA config + database |
| 19 | `check_tenant_isolation_lint` | AST-scans Django views for tenant filtering |
| 20 | `check_caching` | Checks Django cache config |
| 21 | `check_roadmap` | Reads planning system models |
| 22 | `check_output_quality` | Tests DSW output format |
| 23 | `check_policy_review` | Reads QMSPolicy Django models |
| 24 | `check_complexity_governance` | Reads code complexity metrics |
| 25 | `check_endpoint_coverage` | Tests URL route coverage |
| 26 | `check_risk_registry` | Reads risk registry models |

### MOVES TO FORGEGOV (7 checks) — pure code/package concerns

| # | Check | Why ForgeGov | ForgeGov Stage |
|---|-------|-------------|----------------|
| 1 | `check_test_execution` | Runs pytest — forgegov test stage does this | test |
| 2 | `check_test_coverage` | Reads coverage.json — forgegov test stage produces this | test |
| 3 | `check_code_style` | Runs ruff — forgegov lint stage does this | lint |
| 4 | `check_standards_compliance` | Parses standards docs — could be forgegov contract | contract |
| 5 | `check_symbol_coverage` | AST-scans for impl/test hooks — forgegov contract | contract |
| 6 | `check_statistical_calibration` | Runs golden references — forgecal does this | calibrate |
| 7 | `check_calibration_coverage` | Checks calibration completeness — forgecal drift | calibrate |

### NEW (1 check) — bridge

| # | Check | What It Does |
|---|-------|-------------|
| 1 | `check_forge_ecosystem` | Reads `~/.forge/reports/forgegov_latest.json`, checks freshness + passed |

### QUESTIONABLE (3 checks) — could go either way

| # | Check | Current | Recommendation |
|---|-------|---------|---------------|
| 1 | `check_architecture_map` | Reads docs/standards/ for MAP-001 compliance | **Stay in SVEND** — references Django apps and URL structure |
| 2 | `check_architecture` | Checks layer boundaries | **Stay in SVEND** — needs Django app structure |
| 3 | `check_permission_coverage` | Checks decorator usage on views | **Stay in SVEND** — AST scans Django views |

---

## Cleanup Actions

### Action 1: Add `check_forge_ecosystem` to SVEND

```python
def check_forge_ecosystem():
    """CMP-001 §X: Forge ecosystem health via forgegov bridge.

    Reads the forgegov pipeline report from ~/.forge/reports/forgegov_latest.json.
    Passes if: file exists, is fresh (< 24h), and report.passed == true.
    Replaces 7 separate checks that forgegov now owns.
    """
    report_path = Path.home() / ".forge" / "reports" / "forgegov_latest.json"

    if not report_path.exists():
        return fail("No forgegov report found. Run: cd ~/forgegov && forgegov run")

    report = json.loads(report_path.read_text())

    # Freshness check
    timestamp = datetime.fromisoformat(report["timestamp"])
    age_hours = (datetime.now(timezone.utc) - timestamp).total_seconds() / 3600
    if age_hours > 24:
        return fail(f"ForgeGov report is {age_hours:.0f}h old (max 24h). Run: forgegov run")

    # Pass check
    if not report.get("passed"):
        failed_stages = [s["stage"] for s in report.get("stages", []) if not s.get("passed")]
        return fail(f"ForgeGov pipeline failed stages: {', '.join(failed_stages)}")

    return pass_(f"ForgeGov: {report.get('total_duration_s', 0):.1f}s, all stages passed")
```

### Action 2: Remove 7 checks from SVEND

Delete these functions from `compliance.py`:
- `check_test_execution` (~130 lines)
- `check_test_coverage` (~774 lines)
- `check_code_style` (~436 lines)
- `check_standards_compliance` (~158 lines)
- `check_symbol_coverage` (~237 lines)
- `check_statistical_calibration` (~74 lines)
- `check_calibration_coverage` (~99 lines)

**Total removal: ~1,908 lines** (32% of the file)

### Action 3: Update check registry

The `CHECKS` dict at the top of `compliance.py` needs to:
- Remove the 7 deleted checks
- Add `check_forge_ecosystem`
- Update schedule (forge_ecosystem runs daily as critical)

### Action 4: Update CMP-001 standard

Add new section for forge ecosystem bridge. Reference forgegov governance model. Document the handoff: what SVEND owns vs what forgegov owns.

### Action 5: Verify no regressions

After cleanup:
```bash
# Forge side
cd ~/forgegov && forgegov run

# SVEND side
set -a && source /etc/svend/env && set +a
cd ~/kjerne/services/svend/web
python3 manage.py run_compliance --all
```

Both must pass. The 7 removed checks should now be covered by `check_forge_ecosystem` reading forgegov's report.

---

## What This Achieves

**Before:** 37 SVEND checks, some duplicating what forgegov does.
**After:** 27 SVEND checks + 1 bridge check = 28 total. ForgeGov handles the rest.

**Lines removed from compliance.py:** ~1,908 (5,965 → ~4,057)

**Cleaner ownership:**
- SVEND checks what only SVEND can check (Django, server, database)
- ForgeGov checks what it owns (packages, tests, lint, calibration)
- One bridge connects them
- CMP-001 standard updated to reflect the split

---

## File Changes

| File | Change |
|------|--------|
| `syn/audit/compliance.py` | Delete 7 checks (~1,908 lines), add `check_forge_ecosystem` (~30 lines) |
| `docs/standards/CMP-001.md` | Add forge ecosystem bridge section |
| `syn/audit/management/commands/run_compliance.py` | No changes needed (reads from CHECKS dict) |

---

## Owner

**Backend** does the code changes (compliance.py is a Python file).
**Quality (forgegov session)** validates the handoff works.
**Frontend** not involved.
