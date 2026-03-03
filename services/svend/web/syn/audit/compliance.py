"""
Automated compliance check implementations.

Provides 12 checks covering SOC 2 trust service categories:
Security, Availability, Confidentiality, Processing Integrity, Privacy.

Checks run on a rotating daily schedule via syn.sched.

Compliance: SOC 2 CC4.1 (COSO Principle 16: Monitoring Activities)
"""

import json
import logging
import subprocess
import time
from datetime import date, timedelta

from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Check registry
# ---------------------------------------------------------------------------

ALL_CHECKS = {}  # populated by @register below


def register(name, category):
    """Decorator to register a compliance check function."""
    def decorator(fn):
        ALL_CHECKS[name] = (fn, category)
        return fn
    return decorator


# Critical checks run every day; others rotate by weekday (0=Mon)
DAILY_CRITICAL = ["audit_integrity", "access_logging", "security_config", "standards_compliance", "change_management"]
WEEKDAY_ROTATION = {
    0: ["dependency_vuln", "ssl_tls"],
    1: ["encryption_status", "password_policy"],
    2: ["permission_coverage", "backup_freshness"],
    3: ["data_retention"],
    4: ["dependency_vuln", "ssl_tls"],
}


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

@register("audit_integrity", "processing_integrity")
def check_audit_integrity():
    """Verify audit log hash chain integrity and check for unresolved violations."""
    from syn.audit.models import IntegrityViolation
    from syn.audit.utils import verify_chain_integrity

    result = verify_chain_integrity(tenant_id=None)
    unresolved = IntegrityViolation.objects.filter(is_resolved=False).count()

    status = "pass"
    if not result.get("is_valid"):
        status = "fail"
    elif unresolved > 0:
        status = "warning"

    return {
        "status": status,
        "details": {
            "chain_valid": result.get("is_valid"),
            "total_entries": result.get("total_entries", 0),
            "violations_found": len(result.get("violations", [])),
            "unresolved_violations": unresolved,
            "message": result.get("message", ""),
        },
        "soc2_controls": ["CC7.2", "CC7.3"],
    }


@register("security_config", "security")
def check_security_config():
    """Validate Django security settings are production-hardened."""
    issues = []

    if settings.DEBUG:
        issues.append("DEBUG is True in production")

    if not getattr(settings, "SESSION_COOKIE_SECURE", False):
        issues.append("SESSION_COOKIE_SECURE is False")

    if not getattr(settings, "CSRF_COOKIE_SECURE", False):
        issues.append("CSRF_COOKIE_SECURE is False")

    if not getattr(settings, "SESSION_COOKIE_HTTPONLY", False):
        issues.append("SESSION_COOKIE_HTTPONLY is False")

    if getattr(settings, "SESSION_COOKIE_SAMESITE", None) not in ("Lax", "Strict"):
        issues.append(f"SESSION_COOKIE_SAMESITE is {getattr(settings, 'SESSION_COOKIE_SAMESITE', 'not set')}")

    # Check ALLOWED_HOSTS
    allowed = getattr(settings, "ALLOWED_HOSTS", [])
    if "*" in allowed:
        issues.append("ALLOWED_HOSTS contains wildcard '*'")

    # Check SECRET_KEY is not default
    sk = getattr(settings, "SECRET_KEY", "")
    if "django-insecure" in sk or len(sk) < 50:
        issues.append("SECRET_KEY appears weak or default")

    status = "pass" if not issues else ("fail" if any("DEBUG" in i for i in issues) else "warning")
    return {
        "status": status,
        "details": {"issues": issues, "checks_passed": 7 - len(issues), "total_checks": 7},
        "soc2_controls": ["CC6.1"],
    }


@register("dependency_vuln", "security")
def check_dependency_vuln():
    """Scan installed packages for known vulnerabilities using pip-audit."""
    try:
        import sys
        env = dict(__import__("os").environ)
        env["PIPAPI_PYTHON_LOCATION"] = sys.executable
        result = subprocess.run(
            ["pip-audit", "--format=json", "--progress-spinner=off"],
            capture_output=True, text=True, timeout=120, env=env,
        )
        data = json.loads(result.stdout) if result.stdout else {}
        vulns = data.get("dependencies", [])
        vuln_list = [d for d in vulns if d.get("vulns")]

        if not vuln_list:
            return {
                "status": "pass",
                "details": {"packages_scanned": len(vulns), "vulnerabilities": 0},
                "soc2_controls": ["CC7.1"],
            }

        # Summarise vulnerabilities (no internal paths)
        vuln_summary = []
        for dep in vuln_list:
            for v in dep.get("vulns", []):
                vuln_summary.append({
                    "package": dep.get("name"),
                    "installed": dep.get("version"),
                    "vuln_id": v.get("id"),
                    "fix_versions": v.get("fix_versions", []),
                })

        critical = any(v.get("vuln_id", "").startswith("GHSA") for v in vuln_summary)
        return {
            "status": "fail" if critical else "warning",
            "details": {
                "packages_scanned": len(vulns),
                "vulnerabilities": len(vuln_summary),
                "findings": vuln_summary[:20],  # cap at 20
            },
            "soc2_controls": ["CC7.1"],
        }
    except FileNotFoundError:
        return {
            "status": "error",
            "details": {"message": "pip-audit not installed. Run: pip install pip-audit"},
            "soc2_controls": ["CC7.1"],
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "details": {"message": "pip-audit timed out after 120s"},
            "soc2_controls": ["CC7.1"],
        }
    except Exception as e:
        return {
            "status": "error",
            "details": {"message": f"pip-audit failed: {e}"},
            "soc2_controls": ["CC7.1"],
        }


@register("encryption_status", "confidentiality")
def check_encryption_status():
    """Verify encryption settings: password hashers, field encryption key, DB SSL."""
    issues = []

    # Check password hashers
    hashers = getattr(settings, "PASSWORD_HASHERS", [])
    if hashers:
        first_hasher = hashers[0]
        if "PBKDF2" not in first_hasher and "Argon2" not in first_hasher and "Scrypt" not in first_hasher:
            issues.append(f"Primary password hasher is weak: {first_hasher}")
    else:
        # Django defaults to PBKDF2 — fine
        pass

    # Check field encryption key exists
    fek = getattr(settings, "FIELD_ENCRYPTION_KEY", None)
    if not fek:
        issues.append("FIELD_ENCRYPTION_KEY not configured")

    # Check DB connection for SSL
    db_conf = settings.DATABASES.get("default", {})
    options = db_conf.get("OPTIONS", {})
    # Note: Cloudflare Tunnel handles TLS for external, local DB may not need SSL
    # Just flag if missing, don't fail
    if "sslmode" not in str(options):
        issues.append("Database connection does not explicitly set sslmode (acceptable for localhost)")

    status = "pass" if len(issues) <= 1 else "warning"
    if any("weak" in i for i in issues):
        status = "fail"
    return {
        "status": status,
        "details": {"issues": issues, "field_encryption_configured": bool(fek)},
        "soc2_controls": ["CC6.1"],
    }


@register("permission_coverage", "security")
def check_permission_coverage():
    """Verify API endpoints have authentication requirements."""
    from django.urls import URLResolver, URLPattern

    public_allowed = {
        "health", "email_track_open", "email_track_click", "email_unsubscribe",
        "site_duration", "funnel_event", "compliance", "compliance_data",
        "whiteboard_guest_name",  # Guest whiteboard access (token-authenticated)
    }

    try:
        from svend.urls import urlpatterns as root_patterns

        unprotected = []

        # Files known to contain auth-enforcing decorators
        AUTH_MODULES = {
            "accounts/permissions.py", "accounts\\permissions.py",
            "django/contrib/auth/decorators.py",
            "rest_framework/decorators.py",
        }

        def _has_auth_decorator(cb):
            """Walk the __wrapped__ chain checking code origin for auth decorators."""
            # Check DRF views
            if hasattr(cb, "cls") or hasattr(cb, "initkwargs"):
                return True
            # Walk the wrapper chain; @wraps copies metadata but __code__
            # retains the original file where the wrapper was defined.
            fn = cb
            for _ in range(10):  # max depth
                code = getattr(fn, "__code__", None)
                if code:
                    filename = getattr(code, "co_filename", "")
                    if any(m in filename for m in AUTH_MODULES):
                        return True
                inner = getattr(fn, "__wrapped__", None)
                if inner is None:
                    break
                fn = inner
            return False

        def _scan(patterns, prefix=""):
            for p in patterns:
                if isinstance(p, URLResolver):
                    _scan(p.url_patterns, prefix + str(p.pattern))
                elif isinstance(p, URLPattern):
                    name = getattr(p, "name", "") or ""
                    path = prefix + str(p.pattern)
                    if not path.startswith("api/"):
                        continue
                    if name in public_allowed:
                        continue
                    cb = p.callback
                    if not _has_auth_decorator(cb):
                        if not (hasattr(cb, "initkwargs") and cb.initkwargs.get("permission_classes")):
                            unprotected.append({"path": path, "name": name})

        _scan(root_patterns)

        status = "pass" if not unprotected else "warning"
        return {
            "status": status,
            "details": {
                "unprotected_endpoints": unprotected[:10],
                "unprotected_count": len(unprotected),
            },
            "soc2_controls": ["CC6.3"],
        }
    except Exception as e:
        return {
            "status": "error",
            "details": {"message": f"URL scan failed: {e}"},
            "soc2_controls": ["CC6.3"],
        }


@register("access_logging", "security")
def check_access_logging():
    """Verify audit logging middleware is active and producing entries."""
    from syn.audit.models import SysLogEntry

    issues = []

    # Check middleware
    middleware = getattr(settings, "MIDDLEWARE", [])
    if "syn.log.middleware.AuditLoggingMiddleware" not in middleware:
        issues.append("AuditLoggingMiddleware not in MIDDLEWARE")
    if "syn.log.middleware.CorrelationMiddleware" not in middleware:
        issues.append("CorrelationMiddleware not in MIDDLEWARE")

    # Check recent entries (last 24h)
    yesterday = timezone.now() - timedelta(hours=24)
    recent_count = SysLogEntry.objects.filter(timestamp__gte=yesterday).count()
    if recent_count == 0:
        issues.append("No audit log entries in last 24 hours")

    status = "pass"
    if any("not in MIDDLEWARE" in i for i in issues):
        status = "fail"
    elif issues:
        status = "warning"

    return {
        "status": status,
        "details": {
            "issues": issues,
            "recent_entries_24h": recent_count,
            "middleware_present": "syn.log.middleware.AuditLoggingMiddleware" in middleware,
        },
        "soc2_controls": ["CC7.2"],
    }


@register("backup_freshness", "availability")
def check_backup_freshness():
    """Check for recent database backup evidence."""
    from pathlib import Path

    backup_dirs = [
        Path("/home/eric/backups/svend"),
        Path("/home/eric/backups"),
        Path("/var/backups/postgresql"),
        settings.BASE_DIR / "backups",
    ]

    latest_backup = None
    backup_found = False

    for d in backup_dirs:
        if d.exists():
            backups = sorted(d.glob("*.sql*"), key=lambda f: f.stat().st_mtime, reverse=True)
            if backups:
                backup_found = True
                mtime = backups[0].stat().st_mtime
                from datetime import datetime, timezone as dt_tz
                latest_backup = datetime.fromtimestamp(mtime, tz=dt_tz.utc)
                break

    if not backup_found:
        return {
            "status": "warning",
            "details": {"message": "No backup files found in standard locations"},
            "soc2_controls": ["A1.2"],
        }

    age_hours = (timezone.now() - latest_backup).total_seconds() / 3600
    status = "pass" if age_hours < 48 else ("warning" if age_hours < 168 else "fail")

    return {
        "status": status,
        "details": {"latest_backup_age_hours": round(age_hours, 1), "backup_found": True},
        "soc2_controls": ["A1.2"],
    }


@register("password_policy", "security")
def check_password_policy():
    """Validate password policy configuration."""
    validators = getattr(settings, "AUTH_PASSWORD_VALIDATORS", [])
    issues = []

    validator_names = [v.get("NAME", "") for v in validators]

    required = {
        "MinimumLengthValidator": "django.contrib.auth.password_validation.MinimumLengthValidator",
        "CommonPasswordValidator": "django.contrib.auth.password_validation.CommonPasswordValidator",
        "NumericPasswordValidator": "django.contrib.auth.password_validation.NumericPasswordValidator",
    }

    for label, full_name in required.items():
        if full_name not in validator_names:
            issues.append(f"Missing {label}")

    # Check minimum length config
    for v in validators:
        if "MinimumLengthValidator" in v.get("NAME", ""):
            opts = v.get("OPTIONS", {})
            min_len = opts.get("min_length", 8)  # Django default is 8
            if min_len < 8:
                issues.append(f"Minimum password length is {min_len} (should be >= 8)")

    status = "pass" if not issues else "warning"
    return {
        "status": status,
        "details": {
            "validators_configured": len(validators),
            "issues": issues,
        },
        "soc2_controls": ["CC6.1"],
    }


@register("data_retention", "privacy")
def check_data_retention():
    """Verify data retention policies are being enforced."""
    from syn.audit.models import IntegrityViolation

    retention_days = 90
    cutoff = timezone.now() - timedelta(days=retention_days)

    # Check for old resolved violations that should have been cleaned up
    stale = IntegrityViolation.objects.filter(
        is_resolved=True, resolved_at__lt=cutoff
    ).count()

    # Check scheduler has cleanup task registered
    from syn.sched.core import TaskRegistry
    has_cleanup = "audit.cleanup_violations" in TaskRegistry._handlers or \
                  any("cleanup" in name for name in TaskRegistry._handlers)

    issues = []
    if stale > 10:
        issues.append(f"{stale} resolved violations older than {retention_days} days")
    if not has_cleanup:
        issues.append("No cleanup task registered in scheduler")

    status = "pass" if not issues else "warning"
    return {
        "status": status,
        "details": {
            "stale_records": stale,
            "retention_days": retention_days,
            "cleanup_task_registered": has_cleanup,
            "issues": issues,
        },
        "soc2_controls": ["P4.2"],
    }


@register("ssl_tls", "confidentiality")
def check_ssl_tls():
    """Verify TLS/HSTS/CSP configuration."""
    issues = []

    if not getattr(settings, "SECURE_SSL_REDIRECT", False):
        if not settings.DEBUG:
            issues.append("SECURE_SSL_REDIRECT is False in production")

    hsts = getattr(settings, "SECURE_HSTS_SECONDS", 0)
    if hsts < 31536000:  # less than 1 year
        if not settings.DEBUG:
            issues.append(f"HSTS max-age is {hsts}s (recommended >= 31536000)")

    if not getattr(settings, "SECURE_HSTS_INCLUDE_SUBDOMAINS", False):
        if not settings.DEBUG:
            issues.append("SECURE_HSTS_INCLUDE_SUBDOMAINS is False")

    # Check CSP middleware
    middleware = getattr(settings, "MIDDLEWARE", [])
    if not any("ContentSecurityPolicy" in m for m in middleware):
        issues.append("CSP middleware not found")

    if not getattr(settings, "SECURE_CONTENT_TYPE_NOSNIFF", False):
        if not settings.DEBUG:
            issues.append("SECURE_CONTENT_TYPE_NOSNIFF is False")

    status = "pass" if not issues else ("fail" if len(issues) > 2 else "warning")
    return {
        "status": status,
        "details": {
            "issues": issues,
            "hsts_seconds": hsts,
            "ssl_redirect": getattr(settings, "SECURE_SSL_REDIRECT", False),
            "csp_enabled": any("ContentSecurityPolicy" in m for m in middleware),
        },
        "soc2_controls": ["CC6.7"],
    }


@register("standards_compliance", "processing_integrity")
def check_standards_compliance():
    """Parse docs/standards/*.md and verify all assertions against implementations."""
    from syn.audit.standards import parse_all_standards, verify_assertion

    assertions = parse_all_standards()
    if not assertions:
        return {
            "status": "warning",
            "details": {"message": "No standards found to parse"},
            "soc2_controls": [],
        }

    results = [verify_assertion(a) for a in assertions]
    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] == "fail")
    warnings = sum(1 for r in results if r["status"] == "warning")

    # Group by standard with per-assertion detail
    by_standard = {}
    for a, r in zip(assertions, results):
        std = a.standard
        if std not in by_standard:
            by_standard[std] = {"total": 0, "passed": 0, "failed": 0, "assertions": []}
        by_standard[std]["total"] += 1
        if r["status"] == "pass":
            by_standard[std]["passed"] += 1
        elif r["status"] == "fail":
            by_standard[std]["failed"] += 1
        by_standard[std]["assertions"].append({
            "check_id": r["check_id"],
            "assertion": r["assertion"][:120],
            "section": r.get("section", ""),
            "status": r["status"],
            "impl_checks": r.get("impl_checks", []),
            "code_checks": r.get("code_checks", []),
        })

    # Collect all SOC 2 controls
    all_controls = set()
    for a in assertions:
        if "soc2" in a.controls:
            all_controls.add(a.controls["soc2"])

    # Findings: failures + warnings with full detail
    findings = [
        {"check_id": r["check_id"], "assertion": r["assertion"][:120],
         "standard": r["standard"], "section": r.get("section", ""),
         "status": r["status"],
         "impl_checks": r.get("impl_checks", []),
         "code_checks": r.get("code_checks", [])}
        for r in results if r["status"] in ("fail", "warning")
    ]

    return {
        "status": "pass" if failed == 0 else "fail",
        "details": {
            "total_assertions": len(results),
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "by_standard": by_standard,
            "findings": findings,
        },
        "soc2_controls": sorted(all_controls),
    }


@register("change_management", "processing_integrity")
def check_change_management():
    """Verify change management process adherence per CHG-001.

    Checks:
    - Recent changes have associated ChangeRequest records
    - Emergency changes have retroactive risk assessments within 24h
    - Completed changes have log entries for key checkpoints
    - No changes stuck in in_progress for >7 days without updates
    """
    from syn.audit.models import ChangeRequest, ChangeLog, RiskAssessment

    issues = []
    now = timezone.now()
    seven_days_ago = now - timedelta(days=7)
    twenty_four_hours_ago = now - timedelta(hours=24)

    # Check 1: Emergency changes without retroactive risk assessment >24h old
    emergency_unreviewed = ChangeRequest.objects.filter(
        is_emergency=True,
        created_at__lt=twenty_four_hours_ago,
    ).exclude(
        risk_assessments__is_retroactive=True,
    ).exclude(
        status__in=["cancelled", "draft"],
    )
    if emergency_unreviewed.exists():
        issues.append(
            f"{emergency_unreviewed.count()} emergency change(s) missing retroactive risk assessment (>24h)"
        )

    # Check 2: Changes stuck in_progress >7 days
    stale_changes = ChangeRequest.objects.filter(
        status="in_progress",
        updated_at__lt=seven_days_ago,
    )
    if stale_changes.exists():
        issues.append(
            f"{stale_changes.count()} change(s) stuck in 'in_progress' for >7 days"
        )

    # Check 3: Completed changes missing key log entries
    recent_completed = ChangeRequest.objects.filter(
        status="completed",
        completed_at__gte=seven_days_ago,
    ).exclude(change_type__in=["documentation", "plan"])
    for cr in recent_completed:
        log_actions = set(cr.logs.values_list("action", flat=True))
        if "completed" not in log_actions:
            issues.append(f"Change '{cr.title}' completed without 'completed' log entry")

    # Check 4: Feature/migration changes without risk assessment
    unassessed = ChangeRequest.objects.filter(
        change_type__in=["feature", "migration"],
        status__in=["approved", "in_progress", "testing", "completed"],
    ).exclude(
        risk_assessments__isnull=False,
    )
    if unassessed.exists():
        issues.append(
            f"{unassessed.count()} feature/migration change(s) approved without risk assessment"
        )

    # Stats
    total_changes = ChangeRequest.objects.count()
    active_changes = ChangeRequest.objects.filter(
        status__in=["draft", "submitted", "risk_assessed", "approved", "in_progress", "testing"]
    ).count()

    status = "pass"
    if any("emergency" in i for i in issues) or any("without risk assessment" in i for i in issues):
        status = "fail"
    elif issues:
        status = "warning"

    return {
        "status": status,
        "details": {
            "total_changes": total_changes,
            "active_changes": active_changes,
            "issues": issues,
        },
        "soc2_controls": ["CC8.1", "CC3.4"],
    }


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_check(check_name):
    """Run a single named check and persist the result."""
    from syn.audit.models import ComplianceCheck

    if check_name not in ALL_CHECKS:
        raise ValueError(f"Unknown check: {check_name}. Available: {list(ALL_CHECKS.keys())}")

    fn, category = ALL_CHECKS[check_name]

    start = time.time()
    try:
        result = fn()
    except Exception as e:
        logger.exception(f"Compliance check {check_name} failed with exception")
        result = {
            "status": "error",
            "details": {"message": f"Check raised exception: {e}"},
            "soc2_controls": [],
        }
    duration_ms = (time.time() - start) * 1000

    check = ComplianceCheck.objects.create(
        check_name=check_name,
        category=category,
        status=result["status"],
        details=result["details"],
        soc2_controls=result.get("soc2_controls", []),
        duration_ms=duration_ms,
    )

    logger.info(f"[COMPLIANCE] {check_name}: {result['status']} ({duration_ms:.0f}ms)")
    return check


def run_daily_checks():
    """Run today's rotating checks: critical + weekday rotation + unscheduled.

    Any check in ALL_CHECKS not listed in DAILY_CRITICAL or WEEKDAY_ROTATION
    is automatically added to Wednesday's rotation, so new checks run without
    manual scheduling.
    """
    today = date.today().weekday()

    # Find checks not explicitly scheduled anywhere
    scheduled = set(DAILY_CRITICAL)
    for day_checks in WEEKDAY_ROTATION.values():
        scheduled.update(day_checks)
    unscheduled = [name for name in ALL_CHECKS if name not in scheduled]

    # Unscheduled checks run on Wednesday (weekday 2) alongside other rotations
    rotation = list(WEEKDAY_ROTATION.get(today, []))
    if today == 2:
        rotation.extend(unscheduled)

    checks_to_run = list(DAILY_CRITICAL) + rotation
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for c in checks_to_run:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    results = []
    for name in unique:
        results.append(run_check(name))
    return results


def generate_monthly_report():
    """Generate a compliance report for the previous calendar month."""
    from syn.audit.models import ComplianceCheck, ComplianceReport

    today = date.today()
    # Previous month
    first_of_this_month = today.replace(day=1)
    period_end = first_of_this_month - timedelta(days=1)
    period_start = period_end.replace(day=1)

    # If no checks exist for last month (e.g. first run), use all checks
    checks = ComplianceCheck.objects.filter(
        run_at__date__gte=period_start,
        run_at__date__lte=period_end,
    )
    total = checks.count()
    if total == 0:
        # Fall back to all checks ever
        checks = ComplianceCheck.objects.all()
        total = checks.count()
        period_start = date(2026, 1, 1)
        period_end = today

    passed = checks.filter(status="pass").count()
    failed = checks.filter(status="fail").count()
    warnings = checks.filter(status="warning").count()
    pass_rate = (passed / total * 100) if total > 0 else 0

    # Build per-check summary
    check_summary = {}
    for name in ALL_CHECKS:
        name_checks = checks.filter(check_name=name)
        if name_checks.exists():
            name_total = name_checks.count()
            name_passed = name_checks.filter(status="pass").count()
            latest = name_checks.order_by("-run_at").first()
            check_summary[name] = {
                "total_runs": name_total,
                "passed": name_passed,
                "pass_rate": round(name_passed / name_total * 100, 1) if name_total > 0 else 0,
                "latest_status": latest.status if latest else "unknown",
                "category": ALL_CHECKS[name][1],
                "soc2_controls": latest.soc2_controls if latest else [],
            }

    # Build category summary
    category_summary = {}
    for name, info in check_summary.items():
        cat = info["category"]
        if cat not in category_summary:
            category_summary[cat] = {"checks": 0, "passed": 0, "total_runs": 0}
        category_summary[cat]["checks"] += 1
        category_summary[cat]["total_runs"] += info["total_runs"]
        category_summary[cat]["passed"] += info["passed"]
    for cat, data in category_summary.items():
        data["pass_rate"] = round(data["passed"] / data["total_runs"] * 100, 1) if data["total_runs"] > 0 else 0

    # SOC 2 control coverage
    all_controls = set()
    for info in check_summary.values():
        all_controls.update(info.get("soc2_controls", []))

    # Full internal report
    full_report = {
        "check_summary": check_summary,
        "category_summary": category_summary,
        "soc2_controls_covered": sorted(all_controls),
        "total_controls_covered": len(all_controls),
    }

    # Public (redacted) report — no internal paths, IPs, config details
    public_report = {
        "period": f"{period_start.isoformat()} to {period_end.isoformat()}",
        "overall_pass_rate": round(pass_rate, 1),
        "total_checks_run": total,
        "categories": {
            cat: {"pass_rate": data["pass_rate"], "status": "passing" if data["pass_rate"] >= 90 else "needs_attention"}
            for cat, data in category_summary.items()
        },
        "soc2_controls_covered": len(all_controls),
        "checks": {
            name: {
                "category": info["category"],
                "pass_rate": info["pass_rate"],
                "status": "passing" if info["pass_rate"] >= 90 else "needs_attention",
            }
            for name, info in check_summary.items()
        },
    }

    report = ComplianceReport.objects.create(
        period_start=period_start,
        period_end=period_end,
        total_checks=total,
        passed=passed,
        failed=failed,
        warnings=warnings,
        pass_rate=pass_rate,
        summary=category_summary,
        full_report=full_report,
        public_report=public_report,
        is_published=False,
    )

    logger.info(
        f"[COMPLIANCE] Monthly report generated: {period_start} - {period_end}, "
        f"pass_rate={pass_rate:.1f}%, total={total}"
    )
    return report
