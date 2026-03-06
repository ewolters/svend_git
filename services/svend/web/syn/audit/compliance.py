"""
Automated compliance check implementations.

Provides 29 checks covering SOC 2 trust service categories:
Security, Availability, Confidentiality, Processing Integrity, Privacy.

Checks run on a rotating daily schedule via syn.sched.

Compliance: SOC 2 CC4.1 (COSO Principle 16: Monitoring Activities)
"""

import ast
import json
import logging
import os
import re as _re
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Check registry
# ---------------------------------------------------------------------------

ALL_CHECKS = {}  # populated by @register below


def register(name, category, soc2_controls=None):
    """Decorator to register a compliance check function.

    Args:
        name: Unique check identifier.
        category: SOC 2 trust service category.
        soc2_controls: Declared SOC 2 control IDs (auto-discovered by views).
    """

    def decorator(fn):
        fn.soc2_controls = soc2_controls or []
        ALL_CHECKS[name] = (fn, category)
        return fn

    return decorator


def get_check_soc2_controls(name):
    """Return declared SOC 2 controls for a specific check."""
    if name in ALL_CHECKS:
        fn, _cat = ALL_CHECKS[name]
        return getattr(fn, "soc2_controls", [])
    return []


def get_all_soc2_controls():
    """Return all SOC 2 controls declared across registered checks."""
    controls = set()
    for fn, _cat in ALL_CHECKS.values():
        controls.update(getattr(fn, "soc2_controls", []))
    return sorted(controls)


# ---------------------------------------------------------------------------
# SOC 2 Control Matrix — all 44 controls mapped to compliance checks
# ---------------------------------------------------------------------------

SOC2_CONTROL_MATRIX = {
    # CC1 — Control Environment
    "CC1.1": {
        "category": "security",
        "tsc": "CC1",
        "name": "Integrity and ethical values",
        "checks": [],
        "manual_status": "partial",
        "manual_reason": "Policy drafted, no signed acknowledgment",
    },
    "CC1.2": {
        "category": "security",
        "tsc": "CC1",
        "name": "Board oversight",
        "checks": ["change_management"],
        "manual_status": "met",
    },
    "CC1.3": {
        "category": "security",
        "tsc": "CC1",
        "name": "Structure and authority",
        "checks": ["permission_coverage"],
        "manual_status": "met",
    },
    "CC1.4": {
        "category": "security",
        "tsc": "CC1",
        "name": "Commitment to competence",
        "checks": [],
        "manual_status": "met",
    },
    "CC1.5": {
        "category": "security",
        "tsc": "CC1",
        "name": "Enforces accountability",
        "checks": ["change_management"],
        "manual_status": "met",
    },
    # CC2 — Communication and Information
    "CC2.1": {
        "category": "security",
        "tsc": "CC2",
        "name": "Uses relevant quality information",
        "checks": ["log_completeness", "access_logging"],
        "manual_status": "partial",
        "manual_reason": "Logs exist, no centralized monitoring/alerting",
    },
    "CC2.2": {
        "category": "security",
        "tsc": "CC2",
        "name": "Communicates internally",
        "checks": ["change_management"],
        "manual_status": "met",
    },
    "CC2.3": {
        "category": "security",
        "tsc": "CC2",
        "name": "Communicates externally",
        "checks": [],
        "manual_status": "partial",
        "manual_reason": "ToS/privacy need SOC 2 alignment review",
    },
    # CC3 — Risk Assessment
    "CC3.1": {
        "category": "security",
        "tsc": "CC3",
        "name": "Specifies suitable objectives",
        "checks": ["architecture_map", "roadmap"],
        "manual_status": "met",
    },
    "CC3.2": {
        "category": "security",
        "tsc": "CC3",
        "name": "Identifies and analyzes risk",
        "checks": ["change_management"],
        "manual_status": "partial",
        "manual_reason": "No formal risk register with likelihood/impact scoring",
    },
    "CC3.3": {
        "category": "security",
        "tsc": "CC3",
        "name": "Considers potential for fraud",
        "checks": ["rate_limiting"],
        "manual_status": "partial",
        "manual_reason": "No formal fraud risk assessment",
    },
    "CC3.4": {
        "category": "security",
        "tsc": "CC3",
        "name": "Identifies and assesses changes",
        "checks": ["change_management"],
        "manual_status": "met",
    },
    # CC4 — Monitoring Activities
    "CC4.1": {
        "category": "security",
        "tsc": "CC4",
        "name": "Monitoring activities",
        "checks": ["standards_compliance", "sla_compliance", "statistical_calibration", "output_quality"],
        "manual_status": "partial",
        "manual_reason": "No active monitoring/alerting system",
    },
    "CC4.2": {
        "category": "security",
        "tsc": "CC4",
        "name": "Evaluates deficiencies",
        "checks": [],
        "manual_status": "met",
    },
    # CC5 — Control Activities
    "CC5.1": {
        "category": "security",
        "tsc": "CC5",
        "name": "Control activities",
        "checks": ["security_config", "permission_coverage"],
        "manual_status": "met",
    },
    "CC5.2": {
        "category": "security",
        "tsc": "CC5",
        "name": "Technology controls",
        "checks": ["security_headers"],
        "manual_status": "met",
    },
    "CC5.3": {
        "category": "security",
        "tsc": "CC5",
        "name": "Policies and procedures",
        "checks": [],
        "manual_status": "met",
        "manual_reason": "Automated deployment via ops/deploy.sh (OPS-001 §4.4), CI via GitHub Actions (INIT-010).",
    },
    # CC6 — Logical and Physical Access
    "CC6.1": {
        "category": "security",
        "tsc": "CC6",
        "name": "Logical access security",
        "checks": [
            "security_config",
            "encryption_status",
            "password_policy",
            "session_security",
            "secret_management",
            "security_headers",
            "caching",
        ],
        "manual_status": "met",
    },
    "CC6.2": {
        "category": "security",
        "tsc": "CC6",
        "name": "User authentication",
        "checks": ["rate_limiting", "permission_coverage"],
        "manual_status": "partial",
        "manual_reason": "No MFA",
    },
    "CC6.3": {
        "category": "security",
        "tsc": "CC6",
        "name": "Infrastructure access",
        "checks": ["permission_coverage"],
        "manual_status": "met",
    },
    "CC6.4": {
        "category": "security",
        "tsc": "CC6",
        "name": "Software access restriction",
        "checks": ["permission_coverage"],
        "manual_status": "met",
    },
    "CC6.5": {"category": "security", "tsc": "CC6", "name": "Physical access", "checks": [], "manual_status": "met"},
    "CC6.6": {
        "category": "security",
        "tsc": "CC6",
        "name": "System lifecycle access",
        "checks": ["session_security"],
        "manual_status": "partial",
        "manual_reason": "No formal offboarding/deprovisioning workflow",
    },
    "CC6.7": {
        "category": "security",
        "tsc": "CC6",
        "name": "Infrastructure changes",
        "checks": ["change_management", "ssl_tls"],
        "manual_status": "met",
    },
    "CC6.8": {
        "category": "security",
        "tsc": "CC6",
        "name": "Security vulnerabilities",
        "checks": ["dependency_vuln"],
        "manual_status": "partial",
        "manual_reason": "No automated vulnerability scanning pipeline",
    },
    # CC7 — System Operations
    "CC7.1": {
        "category": "security",
        "tsc": "CC7",
        "name": "Vulnerability detection",
        "checks": ["dependency_vuln"],
        "manual_status": "partial",
        "manual_reason": "pip-audit check exists but no continuous scanning pipeline",
    },
    "CC7.2": {
        "category": "security",
        "tsc": "CC7",
        "name": "Anomaly monitoring",
        "checks": [
            "audit_integrity",
            "access_logging",
            "log_completeness",
            "error_handling",
            "architecture_map",
            "architecture",
            "caching",
            "output_quality",
        ],
        "manual_status": "partial",
        "manual_reason": "No application-level anomaly detection",
    },
    "CC7.3": {
        "category": "security",
        "tsc": "CC7",
        "name": "Security event evaluation",
        "checks": ["audit_integrity"],
        "manual_status": "partial",
        "manual_reason": "No SIEM or automated triage",
    },
    "CC7.4": {
        "category": "security",
        "tsc": "CC7",
        "name": "Incident response",
        "checks": ["incident_readiness"],
        "manual_status": "partial",
        "manual_reason": "Policy drafted, not tested via tabletop exercise",
    },
    "CC7.5": {
        "category": "security",
        "tsc": "CC7",
        "name": "System fault handling",
        "checks": ["error_handling"],
        "manual_status": "met",
    },
    # CC8 — Change Management
    "CC8.1": {
        "category": "security",
        "tsc": "CC8",
        "name": "Change management",
        "checks": ["change_management", "code_style", "architecture", "symbol_coverage"],
        "manual_status": "met",
        "manual_reason": "CI/CD via GitHub Actions + ops/deploy.sh (INIT-010). No staging — single-server, deploy script has rollback.",
    },
    # CC9 — Risk Mitigation
    "CC9.1": {
        "category": "security",
        "tsc": "CC9",
        "name": "Vendor risk assessment",
        "checks": ["standards_compliance", "roadmap"],
        "manual_status": "partial",
        "manual_reason": "No formal vendor assessment process",
    },
    "CC9.2": {
        "category": "security",
        "tsc": "CC9",
        "name": "Vendor relationships",
        "checks": ["backup_freshness", "sla_compliance"],
        "manual_status": "partial",
        "manual_reason": "No regular vendor review cadence",
    },
    # A1 — Availability
    "A1.1": {
        "category": "availability",
        "tsc": "A1",
        "name": "Capacity management",
        "checks": ["sla_compliance"],
        "manual_status": "partial",
        "manual_reason": "No auto-scaling or capacity monitoring/alerting",
    },
    "A1.2": {
        "category": "availability",
        "tsc": "A1",
        "name": "Environmental threats",
        "checks": ["backup_freshness"],
        "manual_status": "met",
    },
    "A1.3": {
        "category": "availability",
        "tsc": "A1",
        "name": "Recovery operations",
        "checks": ["backup_freshness"],
        "manual_status": "partial",
        "manual_reason": "Backups on same machine, no off-site replication",
    },
    # PI1 — Processing Integrity
    "PI1.1": {
        "category": "processing_integrity",
        "tsc": "PI1",
        "name": "Quality information",
        "checks": ["output_quality"],
        "manual_status": "met",
    },
    "PI1.2": {
        "category": "processing_integrity",
        "tsc": "PI1",
        "name": "Accurate processing",
        "checks": ["statistical_calibration", "output_quality"],
        "manual_status": "met",
    },
    "PI1.3": {
        "category": "processing_integrity",
        "tsc": "PI1",
        "name": "Complete processing",
        "checks": ["audit_integrity"],
        "manual_status": "partial",
        "manual_reason": "No end-to-end data integrity checksums",
    },
    "PI1.4": {
        "category": "processing_integrity",
        "tsc": "PI1",
        "name": "Accurate outputs",
        "checks": ["output_quality", "statistical_calibration"],
        "manual_status": "met",
    },
    "PI1.5": {
        "category": "processing_integrity",
        "tsc": "PI1",
        "name": "Error handling",
        "checks": ["error_handling"],
        "manual_status": "met",
    },
    # C1 — Confidentiality
    "C1.1": {
        "category": "confidentiality",
        "tsc": "C1",
        "name": "Identifies confidential info",
        "checks": ["encryption_status"],
        "manual_status": "partial",
        "manual_reason": "Classification policy drafted, not enforced systematically",
    },
    "C1.2": {
        "category": "confidentiality",
        "tsc": "C1",
        "name": "Protects confidential info",
        "checks": ["encryption_status", "ssl_tls", "session_security"],
        "manual_status": "met",
    },
    "C1.3": {
        "category": "confidentiality",
        "tsc": "C1",
        "name": "Disposes confidential info",
        "checks": ["data_retention"],
        "manual_status": "partial",
        "manual_reason": "No formal data disposal/retention SLA",
    },
    # P1 — Privacy
    "P1.1": {
        "category": "privacy",
        "tsc": "P1",
        "name": "Privacy notice",
        "checks": [],
        "manual_status": "partial",
        "manual_reason": "Needs SOC 2 alignment review",
    },
    "P1.2": {
        "category": "privacy",
        "tsc": "P1",
        "name": "Choice and consent",
        "checks": [],
        "manual_status": "partial",
        "manual_reason": "No granular consent management",
    },
    "P1.3": {
        "category": "privacy",
        "tsc": "P1",
        "name": "PII collection purpose",
        "checks": [],
        "manual_status": "met",
    },
    "P1.4": {"category": "privacy", "tsc": "P1", "name": "PII usage limitation", "checks": [], "manual_status": "met"},
    "P1.5": {
        "category": "privacy",
        "tsc": "P1",
        "name": "PII retention",
        "checks": ["data_retention"],
        "manual_status": "partial",
        "manual_reason": "No formal retention schedule",
    },
    "P1.6": {
        "category": "privacy",
        "tsc": "P1",
        "name": "PII disposal",
        "checks": ["data_retention"],
        "manual_status": "partial",
        "manual_reason": "No verified complete data deletion",
    },
    "P1.7": {"category": "privacy", "tsc": "P1", "name": "PII quality", "checks": [], "manual_status": "met"},
    "P1.8": {
        "category": "privacy",
        "tsc": "P1",
        "name": "PII access and correction",
        "checks": ["privacy_data_export"],
        "manual_status": "met",
        "manual_reason": "Self-service data export via PRIV-001",
    },
}


def _min_status(a, b):
    """Return the worse of two statuses (gap < partial < met)."""
    order = {"gap": 0, "partial": 1, "met": 2}
    return a if order.get(a, 0) <= order.get(b, 0) else b


def soc2_control_coverage():
    """Evaluate all 44 SOC 2 controls against latest compliance check results.

    Returns dict with:
    - controls: list of {id, name, category, tsc, status, checks, check_results, reason}
    - by_tsc: {tsc: {met, partial, gap}} counts per TSC category
    - overall: {met, partial, gap} total counts
    - score: percentage of met controls
    """
    from syn.audit.models import ComplianceCheck

    # Get latest result per check
    latest_results = {}
    for check_name in ALL_CHECKS:
        latest = ComplianceCheck.objects.filter(check_name=check_name).order_by("-run_at").first()
        if latest:
            latest_results[check_name] = latest.status

    controls = []
    tsc_summary = {}

    for ctrl_id in sorted(SOC2_CONTROL_MATRIX.keys()):
        ctrl = SOC2_CONTROL_MATRIX[ctrl_id]
        tsc = ctrl["tsc"]
        if tsc not in tsc_summary:
            tsc_summary[tsc] = {"met": 0, "partial": 0, "gap": 0}

        # Start with manual status
        effective_status = ctrl["manual_status"]
        manual_reason = ctrl.get("manual_reason", "")

        # If checks are mapped, let automated results influence status
        mapped_checks = ctrl.get("checks", [])
        if mapped_checks:
            check_statuses = []
            for cn in mapped_checks:
                cs = latest_results.get(cn)
                if cs:
                    check_statuses.append((cn, cs))

            if check_statuses:
                if any(s == "fail" for _, s in check_statuses):
                    effective_status = _min_status(effective_status, "partial")
                elif any(s == "error" for _, s in check_statuses):
                    effective_status = _min_status(effective_status, "partial")
                elif all(s == "pass" for _, s in check_statuses):
                    # All checks pass — upgrade to met (manual_reason is informational, not a gate)
                    if effective_status == "partial":
                        effective_status = "met"

        controls.append(
            {
                "id": ctrl_id,
                "name": ctrl["name"],
                "category": ctrl["category"],
                "tsc": tsc,
                "status": effective_status,
                "checks": mapped_checks,
                "check_results": {cn: latest_results.get(cn, "pending") for cn in mapped_checks},
                "reason": manual_reason if effective_status != "met" else "",
                "policy_evidence": ctrl.get("policy_evidence", ""),
            }
        )

        tsc_summary[tsc][effective_status] += 1

    total = {"met": 0, "partial": 0, "gap": 0}
    for tsc_counts in tsc_summary.values():
        for k in total:
            total[k] += tsc_counts[k]

    score = round(total["met"] / max(sum(total.values()), 1) * 100, 1)

    return {
        "controls": controls,
        "by_tsc": tsc_summary,
        "overall": total,
        "score": score,
    }


# Critical checks run every day; others rotate by weekday (0=Mon)
DAILY_CRITICAL = [
    "audit_integrity",
    "access_logging",
    "security_config",
    "standards_compliance",
    "change_management",
    "sla_compliance",
]
WEEKDAY_ROTATION = {
    0: ["dependency_vuln", "ssl_tls", "log_completeness", "security_headers"],
    1: ["encryption_status", "password_policy", "session_security", "secret_management"],
    2: [
        "permission_coverage",
        "backup_freshness",
        "error_handling",
        "incident_readiness",
        "symbol_coverage",
        "output_quality",
        "calibration_coverage",
        "complexity_governance",
        "endpoint_coverage",
    ],
    3: ["data_retention", "rate_limiting", "code_style", "roadmap", "statistical_calibration"],
    4: ["dependency_vuln", "ssl_tls"],
}


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


@register("audit_integrity", "processing_integrity", soc2_controls=["CC7.2", "CC7.3"])
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


@register("security_config", "security", soc2_controls=["CC6.1"])
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


@register("dependency_vuln", "security", soc2_controls=["CC7.1"])
def check_dependency_vuln():
    """Scan installed packages for known vulnerabilities using pip-audit."""
    try:
        import sys

        env = dict(__import__("os").environ)
        env["PIPAPI_PYTHON_LOCATION"] = sys.executable
        result = subprocess.run(
            ["pip-audit", "--format=json", "--progress-spinner=off"],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
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
                vuln_summary.append(
                    {
                        "package": dep.get("name"),
                        "installed": dep.get("version"),
                        "vuln_id": v.get("id"),
                        "fix_versions": v.get("fix_versions", []),
                    }
                )

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


@register("encryption_status", "confidentiality", soc2_controls=["CC6.1"])
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


@register("permission_coverage", "security", soc2_controls=["CC6.3"])
def check_permission_coverage():
    """Verify API endpoints have authentication requirements."""
    from django.urls import URLPattern, URLResolver

    public_allowed = {
        "health",
        "email_track_open",
        "email_track_click",
        "email_unsubscribe",
        "site_duration",
        "funnel_event",
        "compliance",
        "compliance_data",
        "whiteboard_guest_name",  # Guest whiteboard access (token-authenticated)
        "notification_type_unsubscribe",  # Signed-token unsubscribe (no session auth)
    }

    try:
        from svend.urls import urlpatterns as root_patterns

        unprotected = []

        # Files known to contain auth-enforcing decorators
        AUTH_MODULES = {
            "accounts/permissions.py",
            "accounts\\permissions.py",
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


@register("access_logging", "security", soc2_controls=["CC7.2"])
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


@register("backup_freshness", "availability", soc2_controls=["A1.2"])
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
                from datetime import datetime
                from datetime import timezone as dt_tz

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


@register("password_policy", "security", soc2_controls=["CC6.1"])
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


@register("data_retention", "privacy", soc2_controls=["P4.2"])
def check_data_retention():
    """Verify data retention policies are being enforced."""
    from syn.audit.models import IntegrityViolation

    retention_days = 90
    cutoff = timezone.now() - timedelta(days=retention_days)

    # Check for old resolved violations that should have been cleaned up
    stale = IntegrityViolation.objects.filter(is_resolved=True, resolved_at__lt=cutoff).count()

    # Check scheduler has cleanup task registered
    from syn.sched.core import TaskRegistry

    has_cleanup = "audit.cleanup_violations" in TaskRegistry._handlers or any(
        "cleanup" in name for name in TaskRegistry._handlers
    )

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


@register("privacy_data_export", "privacy", soc2_controls=["P1.8"])
def check_privacy_data_export():
    """Verify self-service data export capability exists (PRIV-001)."""
    issues = []

    # Check 1: Model exists and is queryable
    try:
        from accounts.models import DataExportRequest

        DataExportRequest.objects.count()
    except Exception as e:
        issues.append(f"DataExportRequest model inaccessible: {e}")

    # Check 2: Export endpoint is registered
    try:
        from django.urls import reverse

        reverse("privacy:exports")
    except Exception:
        issues.append("Privacy export endpoint not found in URL config")

    # Check 3: Export task is registered
    try:
        from syn.sched.core import TaskRegistry

        if "privacy.generate_export" not in TaskRegistry._handlers:
            issues.append("privacy.generate_export task not registered")
    except Exception:
        issues.append("Cannot verify task registration")

    status = "pass" if not issues else "fail"
    return {
        "status": status,
        "details": {"issues": issues},
        "soc2_controls": ["P1.8"],
    }


@register("ssl_tls", "confidentiality", soc2_controls=["CC6.7"])
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


# ---------------------------------------------------------------------------
# Drift violation helpers (CMP-001 §5.6)
# ---------------------------------------------------------------------------

DRIFT_SEVERITY_MAP = {
    "fail": "HIGH",
    "warning": "MEDIUM",
}

DRIFT_SLA_HOURS = {
    "CRITICAL": 4,
    "HIGH": 24,
    "MEDIUM": 72,
    "LOW": 168,
}


def _compute_drift_signature(standard, check_id, section):
    """Deterministic hash for deduplication. Same assertion = same signature."""
    import hashlib

    payload = f"{standard}:{check_id}:{section}"
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _sync_drift_violations(assertions, results):
    """Create/resolve DriftViolation records based on assertion results.

    Runs in its own transaction.atomic() so drift records persist even if
    something downstream fails or rolls back. CMP-001 §5.6.
    """
    from django.db import transaction
    from django.utils import timezone as tz

    from syn.audit.models import DriftViolation

    # Collect all signatures from current run (for auto-resolution)
    current_signatures = {}
    for a, r in zip(assertions, results):
        sig = _compute_drift_signature(a.standard, a.check_id, a.section)
        current_signatures[sig] = (a, r)

    with transaction.atomic():
        # 1. Create violations for failures/warnings
        for sig, (a, r) in current_signatures.items():
            if r["status"] not in DRIFT_SEVERITY_MAP:
                continue  # pass — no violation needed
            # Check if unresolved violation already exists
            if DriftViolation.objects.filter(drift_signature=sig, resolved_at__isnull=True).exists():
                continue  # idempotent — already tracked
            severity = DRIFT_SEVERITY_MAP[r["status"]]
            sla_hours = DRIFT_SLA_HOURS.get(severity)
            DriftViolation.objects.create(
                drift_signature=sig,
                severity=severity,
                enforcement_check="STD",
                file_path=f"docs/standards/{a.standard}.md",
                violation_message=f"[{a.check_id}] {a.text[:200]}",
                detected_by="compliance_runner",
                remediation_sla_hours=sla_hours,
            )

        # 2. Auto-resolve violations for assertions that now pass
        passing_sigs = {sig for sig, (a, r) in current_signatures.items() if r["status"] == "pass"}
        if passing_sigs:
            now = tz.now()
            open_violations = DriftViolation.objects.filter(
                drift_signature__in=passing_sigs,
                resolved_at__isnull=True,
                enforcement_check="STD",
            )
            for dv in open_violations:
                dv.resolved_at = now
                dv.resolved_by = "compliance_runner"
                dv.resolution_notes = "Auto-resolved: assertion now passes"
                dv.save(update_fields=["resolved_at", "resolved_by", "resolution_notes"])


@register("standards_compliance", "processing_integrity", soc2_controls=["CC4.1", "CC9.1"])
def check_standards_compliance():
    """Parse docs/standards/*.md and verify assertions (impls, code, test existence — no execution)."""
    from syn.audit.standards import parse_all_standards, verify_assertion

    assertions = parse_all_standards()
    if not assertions:
        return {
            "status": "warning",
            "details": {"message": "No standards found to parse"},
            "soc2_controls": [],
        }

    results = [verify_assertion(a, run_tests=False) for a in assertions]
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
        by_standard[std]["assertions"].append(
            {
                "check_id": r["check_id"],
                "assertion": r["assertion"][:120],
                "section": r.get("section", ""),
                "status": r["status"],
                "impl_checks": r.get("impl_checks", []),
                "code_checks": r.get("code_checks", []),
                "test_checks": r.get("test_checks", []),
            }
        )

    # Collect all SOC 2 controls
    all_controls = set()
    for a in assertions:
        if "soc2" in a.controls:
            all_controls.add(a.controls["soc2"])

    # Findings: failures + warnings with full detail
    findings = [
        {
            "check_id": r["check_id"],
            "assertion": r["assertion"][:120],
            "standard": r["standard"],
            "section": r.get("section", ""),
            "status": r["status"],
            "impl_checks": r.get("impl_checks", []),
            "code_checks": r.get("code_checks", []),
            "test_checks": r.get("test_checks", []),
        }
        for r in results
        if r["status"] in ("fail", "warning")
    ]

    # Aggregate test stats
    tests_linked = tests_exist = tests_missing = 0
    tests_passed = tests_failed = tests_skipped = 0
    seen_tests = set()
    for r in results:
        for tc in r.get("test_checks", []):
            tests_linked += 1
            test_path = tc.get("test", "")
            seen_tests.add(test_path)
            if tc.get("exists"):
                tests_exist += 1
            else:
                tests_missing += 1
            if tc.get("ran"):
                status = tc.get("status", "")
                if status == "pass" or tc.get("passed"):
                    tests_passed += 1
                elif status == "skip":
                    tests_skipped += 1
                else:
                    tests_failed += 1
    tests_unique = len(seen_tests)

    # Sync drift violations (CMP-001 §5.6) — own transaction
    try:
        _sync_drift_violations(assertions, results)
    except Exception as e:
        logger.warning("Drift violation sync failed: %s", e)

    return {
        "status": "pass" if failed == 0 else "fail",
        "details": {
            "total_assertions": len(results),
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "tests_linked": tests_linked,
            "tests_unique": tests_unique,
            "tests_exist": tests_exist,
            "tests_missing": tests_missing,
            "tests_passed": tests_passed,
            "tests_failed": tests_failed,
            "tests_skipped": tests_skipped,
            "by_standard": by_standard,
            "findings": findings,
        },
        "soc2_controls": sorted(all_controls),
    }


def run_standards_tests_for(standard_name):
    """Run linked tests for a single standard (~15s). Called per-standard by the dashboard."""
    from syn.audit.standards import parse_all_standards, verify_assertion

    assertions = [a for a in parse_all_standards() if a.standard == standard_name]
    if not assertions:
        return {"standard": standard_name, "status": "warning", "message": "No assertions found"}

    results = [verify_assertion(a, run_tests=True) for a in assertions]

    tests_passed = tests_failed = tests_skipped = 0
    for r in results:
        for tc in r.get("test_checks", []):
            if tc.get("ran"):
                status = tc.get("status", "")
                if status == "pass" or tc.get("passed"):
                    tests_passed += 1
                elif status == "skip":
                    tests_skipped += 1
                else:
                    tests_failed += 1

    return {
        "standard": standard_name,
        "status": "pass" if tests_failed == 0 else "fail",
        "assertions": len(assertions),
        "tests_passed": tests_passed,
        "tests_failed": tests_failed,
        "tests_skipped": tests_skipped,
    }


@register("change_management", "processing_integrity", soc2_controls=["CC8.1", "CC3.4"])
def check_change_management():
    """Verify change management process adherence per CHG-001 v1.6.

    15 checks with FAIL/WARNING severity classification.
    CHG-001 §7.1.1 enforcement — field requirements, risk assessment gates,
    commit traceability, log chain completeness.

    FAIL-level (SOC 2 violations):
      1. Emergency changes without retroactive RA >24h
      2. Feature/migration past approved without RA
      3. CRs with empty description

    Note: commit_shas and log_md_ref are WARNING for historical CRs
    (genuinely unrecoverable) but FAIL-gated for new CRs via
    validate_for_transition() at the API layer.

    WARNING-level (process gaps):
      6. Changes stuck in_progress >7 days
      7. Completed changes missing 'completed' log entry
      8. Submitted+ CRs missing justification
      9. Approved+ CRs missing implementation_plan
     10. Approved+ CRs missing rollback_plan (rollback-required types)
     11. In_progress+ CRs missing testing_plan
     12. Enhancement/bugfix/security/infrastructure/debt past approved without RA
     13. ChangeLog 'completed' entries missing commit_sha in details
     14. Completed CRs with compliance language but empty compliance_check_ids
     15. Active code CRs without planning linkage (feature_id/task_id)
    """
    from syn.audit.models import ChangeLog, ChangeRequest

    fail_issues = []
    warn_issues = []
    now = timezone.now()
    seven_days_ago = now - timedelta(days=7)
    twenty_four_hours_ago = now - timedelta(hours=24)

    # Type classifications
    CODE_TYPES = [
        "feature",
        "enhancement",
        "bugfix",
        "hotfix",
        "security",
        "infrastructure",
        "migration",
        "debt",
    ]
    EXEMPT_TYPES = ["documentation", "plan"]
    ROLLBACK_TYPES = ["feature", "migration", "infrastructure", "security"]
    MULTI_AGENT_TYPES = ["feature", "migration"]
    SINGLE_AGENT_TYPES = [
        "enhancement",
        "bugfix",
        "security",
        "infrastructure",
        "debt",
    ]
    PAST_APPROVED = ["approved", "in_progress", "testing", "completed"]
    SUBMITTED_PLUS = [
        "submitted",
        "risk_assessed",
        "approved",
        "in_progress",
        "testing",
        "completed",
    ]
    IN_PROGRESS_PLUS = ["in_progress", "testing", "completed"]

    # ── FAIL checks ──

    # 1. Emergency changes without retroactive risk assessment >24h
    emergency_unreviewed = (
        ChangeRequest.objects.filter(
            is_emergency=True,
            created_at__lt=twenty_four_hours_ago,
        )
        .exclude(
            risk_assessments__is_retroactive=True,
        )
        .exclude(
            status__in=["cancelled", "draft"],
        )
    )
    if emergency_unreviewed.exists():
        fail_issues.append(
            f"{emergency_unreviewed.count()} emergency change(s) missing retroactive risk assessment (>24h)"
        )

    # 2. Feature/migration past approved without risk assessment
    unassessed_critical = ChangeRequest.objects.filter(
        change_type__in=MULTI_AGENT_TYPES,
        status__in=PAST_APPROVED,
    ).exclude(
        risk_assessments__isnull=False,
    )
    if unassessed_critical.exists():
        fail_issues.append(
            f"{unassessed_critical.count()} feature/migration change(s) approved without risk assessment"
        )

    # 3. Completed code CRs missing commit_shas
    #    Pre-enforcement CRs have genuinely unrecoverable commit_shas.
    #    New CRs are FAIL-gated via validate_for_transition().
    #    Historical gaps are WARNING; going forward they are blocked at API.
    completed_code = ChangeRequest.objects.filter(
        status="completed",
        change_type__in=CODE_TYPES,
    )
    missing_commits = completed_code.filter(commit_shas=[])
    if missing_commits.exists():
        warn_issues.append(
            f"{missing_commits.count()} completed code CR(s) missing commit_shas — CR→git traceability gap"
        )

    # 4. Completed code CRs missing log_md_ref (same — historical gap)
    missing_log_ref = completed_code.filter(log_md_ref="")
    if missing_log_ref.exists():
        warn_issues.append(
            f"{missing_log_ref.count()} completed code CR(s) missing log_md_ref — CR→log.md traceability gap"
        )

    # 5. CRs with empty description
    empty_desc = ChangeRequest.objects.filter(description="").exclude(
        status__in=["cancelled", "draft"],
    )
    if empty_desc.exists():
        fail_issues.append(f"{empty_desc.count()} CR(s) with empty description")

    # ── WARNING checks ──

    # 6. Changes stuck in_progress >7 days
    stale_changes = ChangeRequest.objects.filter(
        status="in_progress",
        updated_at__lt=seven_days_ago,
    )
    if stale_changes.exists():
        warn_issues.append(f"{stale_changes.count()} change(s) stuck in 'in_progress' for >7 days")

    # 7. Completed changes missing 'completed' log entry
    recent_completed = ChangeRequest.objects.filter(
        status="completed",
    ).exclude(change_type__in=EXEMPT_TYPES)
    for cr in recent_completed:
        log_actions = set(cr.logs.values_list("action", flat=True))
        if "completed" not in log_actions:
            warn_issues.append(f"Change '{cr.title[:50]}' completed without 'completed' log entry")

    # 8. Submitted+ CRs missing justification (non-exempt)
    missing_justification = ChangeRequest.objects.filter(
        status__in=SUBMITTED_PLUS,
        justification="",
    ).exclude(change_type__in=EXEMPT_TYPES)
    if missing_justification.exists():
        warn_issues.append(f"{missing_justification.count()} submitted+ CR(s) missing justification")

    # 9. Approved+ CRs missing implementation_plan (non-exempt)
    missing_impl_plan = ChangeRequest.objects.filter(
        status__in=PAST_APPROVED,
        implementation_plan={},
    ).exclude(change_type__in=EXEMPT_TYPES)
    if missing_impl_plan.exists():
        warn_issues.append(f"{missing_impl_plan.count()} approved+ CR(s) missing implementation_plan")

    # 10. Approved+ CRs missing rollback_plan (rollback-required types)
    missing_rollback = ChangeRequest.objects.filter(
        status__in=PAST_APPROVED,
        change_type__in=ROLLBACK_TYPES,
        rollback_plan={},
    )
    if missing_rollback.exists():
        warn_issues.append(
            f"{missing_rollback.count()} approved+ CR(s) missing "
            f"rollback_plan (feature/migration/infrastructure/security)"
        )

    # 11. In_progress+ CRs missing testing_plan (non-exempt)
    missing_testing = ChangeRequest.objects.filter(
        status__in=IN_PROGRESS_PLUS,
        testing_plan={},
    ).exclude(change_type__in=EXEMPT_TYPES)
    if missing_testing.exists():
        warn_issues.append(f"{missing_testing.count()} in_progress+ CR(s) missing testing_plan")

    # 12. Enhancement/bugfix/security/infrastructure/debt past approved
    #     without risk assessment (single-agent required)
    unassessed_single = ChangeRequest.objects.filter(
        change_type__in=SINGLE_AGENT_TYPES,
        status__in=PAST_APPROVED,
    ).exclude(
        risk_assessments__isnull=False,
    )
    if unassessed_single.exists():
        warn_issues.append(
            f"{unassessed_single.count()} enhancement/bugfix/security/"
            f"infrastructure/debt CR(s) approved without risk assessment"
        )

    # 13. Completed CRs where NO 'completed' log entry has commit_sha
    # (Per-CR check: any completed entry with commit_sha satisfies the requirement,
    #  since log entries are immutable and supplementary entries may backfill.)
    completed_code_crs_ids = (
        ChangeRequest.objects.filter(status="completed")
        .exclude(change_type__in=EXEMPT_TYPES)
        .values_list("id", flat=True)
    )
    missing_sha_cr_count = 0
    for cr_id in completed_code_crs_ids:
        cr_completed_logs = ChangeLog.objects.filter(change_request_id=cr_id, action="completed")
        has_sha = False
        for log in cr_completed_logs:
            details = log.details if isinstance(log.details, dict) else {}
            if details.get("commit_sha") or details.get("commit_shas"):
                has_sha = True
                break
        if not has_sha and cr_completed_logs.exists():
            missing_sha_cr_count += 1
    if missing_sha_cr_count:
        warn_issues.append(f"{missing_sha_cr_count} 'completed' log entry(s) missing commit_sha in details")

    # 14. Completed CRs referencing compliance remediation but empty compliance_check_ids
    # Only flag CRs that explicitly remediate findings — not general compliance infrastructure
    remediation_keywords = ["remediates finding", "fixes compliance fail", "resolves drift violation", "clears finding"]
    completed_all = ChangeRequest.objects.filter(
        status="completed",
        compliance_check_ids=[],
    )
    unlinked_compliance = 0
    for cr in completed_all:
        desc_lower = (cr.description or "").lower()
        if any(kw in desc_lower for kw in remediation_keywords):
            unlinked_compliance += 1
    if unlinked_compliance:
        warn_issues.append(
            f"{unlinked_compliance} completed CR(s) reference compliance but have empty compliance_check_ids"
        )

    # 15. Code CRs without planning linkage (WARNING)
    # Exempt: documentation, plan, hotfix, debt — these legitimately may not map to features
    planning_exempt = {"documentation", "plan", "hotfix", "debt"}
    unlinked_planning = (
        ChangeRequest.objects.exclude(
            status__in=["completed", "cancelled"],
        )
        .exclude(
            change_type__in=planning_exempt,
        )
        .filter(
            feature_id__isnull=True,
            task_id__isnull=True,
        )
        .count()
    )
    if unlinked_planning:
        warn_issues.append(f"{unlinked_planning} active code CR(s) without planning link (feature_id/task_id)")

    # ── Stats and field completeness ──

    total_changes = ChangeRequest.objects.count()
    active_changes = ChangeRequest.objects.filter(
        status__in=[
            "draft",
            "submitted",
            "risk_assessed",
            "approved",
            "in_progress",
            "testing",
        ]
    ).count()

    all_code_crs = ChangeRequest.objects.filter(change_type__in=CODE_TYPES)
    code_total = all_code_crs.count() or 1  # avoid /0

    field_completeness = {
        "commit_shas": f"{all_code_crs.exclude(commit_shas=[]).count()}/{code_total}",
        "log_md_ref": f"{all_code_crs.exclude(log_md_ref='').count()}/{code_total}",
        "rollback_plan": f"{all_code_crs.exclude(rollback_plan={}).count()}/{code_total}",
        "testing_plan": f"{all_code_crs.exclude(testing_plan={}).count()}/{code_total}",
        "implementation_plan": f"{all_code_crs.exclude(implementation_plan={}).count()}/{code_total}",
        "justification": f"{all_code_crs.exclude(justification='').count()}/{code_total}",
        "risk_assessment": f"{sum(1 for cr in all_code_crs if cr.risk_assessments.exists())}/{code_total}",
    }

    # ── Determine status ──

    all_issues = [f"[FAIL] {i}" for i in fail_issues] + [f"[WARN] {i}" for i in warn_issues]

    if fail_issues:
        status = "fail"
    elif warn_issues:
        status = "warning"
    else:
        status = "pass"

    return {
        "status": status,
        "details": {
            "total_changes": total_changes,
            "active_changes": active_changes,
            "issues": all_issues,
            "fail_count": len(fail_issues),
            "warn_count": len(warn_issues),
            "field_completeness": field_completeness,
        },
        "soc2_controls": ["CC8.1", "CC3.4"],
    }


@register("session_security", "security", soc2_controls=["CC6.1", "CC6.6"])
def check_session_security():
    """Validate session lifecycle controls (SOC 2 CC6.1/CC6.6, ISO A.9.4, NIST AC-12)."""
    issues = []

    # Session timeout — Django default is 2 weeks (1209600s), should be ≤8h for production
    age = getattr(settings, "SESSION_COOKIE_AGE", 1209600)
    if age > 28800:
        issues.append(f"SESSION_COOKIE_AGE is {age}s ({age // 3600}h) — recommended ≤ 28800 (8h)")

    # Browser close should end session
    if not getattr(settings, "SESSION_EXPIRE_AT_BROWSER_CLOSE", False):
        issues.append("SESSION_EXPIRE_AT_BROWSER_CLOSE is False — sessions persist after browser close")

    # Session cookie flags (cross-check with security_config)
    if not getattr(settings, "SESSION_COOKIE_SECURE", False) and not settings.DEBUG:
        issues.append("SESSION_COOKIE_SECURE is False in production")
    if not getattr(settings, "SESSION_COOKIE_HTTPONLY", False):
        issues.append("SESSION_COOKIE_HTTPONLY is False")

    # Session engine — database-backed is preferred for server-side invalidation
    engine = getattr(settings, "SESSION_ENGINE", "django.contrib.sessions.backends.db")
    if "signed_cookies" in engine:
        issues.append("Session engine uses signed_cookies — no server-side invalidation possible")

    status = "pass" if not issues else "warning"
    return {
        "status": status,
        "details": {
            "issues": issues,
            "session_cookie_age": age,
            "session_engine": engine,
            "checks_passed": 5 - len(issues),
            "total_checks": 5,
        },
        "soc2_controls": ["CC6.1", "CC6.6"],
    }


@register("error_handling", "processing_integrity", soc2_controls=["CC7.2", "CC7.5"])
def check_error_handling():
    """Validate error envelope and info leak prevention (SOC 2 CC7.2/CC7.5, ISO A.12.4, NIST SI-11)."""
    issues = []

    middleware = getattr(settings, "MIDDLEWARE", [])

    # ErrorEnvelopeMiddleware must be active
    has_error_envelope = any("ErrorEnvelope" in m for m in middleware)
    if not has_error_envelope:
        issues.append("ErrorEnvelopeMiddleware not found in MIDDLEWARE — unhandled exceptions may leak stack traces")

    # APIHeadersMiddleware must be active (API-001 §8.2)
    has_api_headers = any("APIHeaders" in m for m in middleware)
    if not has_api_headers:
        issues.append("APIHeadersMiddleware not found in MIDDLEWARE — no Accept validation or X-Response-Time headers")

    # IdempotencyMiddleware must be active (API-001 §9)
    has_idempotency = any("Idempotency" in m for m in middleware)
    if not has_idempotency:
        issues.append("IdempotencyMiddleware not found in MIDDLEWARE — no POST replay protection")

    # DEBUG must be False in production
    if settings.DEBUG:
        issues.append("DEBUG is True — stack traces visible to users")

    # DEBUG_PROPAGATE_EXCEPTIONS should be False
    if getattr(settings, "DEBUG_PROPAGATE_EXCEPTIONS", False):
        issues.append("DEBUG_PROPAGATE_EXCEPTIONS is True — exceptions bypass error envelope")

    # Check for custom error templates (warning only)
    from pathlib import Path

    template_dir = Path(settings.BASE_DIR) / "templates"
    missing_templates = []
    for code in ["400", "403", "404", "500"]:
        if not (template_dir / f"{code}.html").exists():
            missing_templates.append(f"{code}.html")
    if missing_templates:
        issues.append(f"Missing custom error templates: {', '.join(missing_templates)} — Django defaults may leak info")

    status = (
        "pass"
        if not issues
        else ("fail" if any("DEBUG is True" in i or "Middleware" in i for i in issues) else "warning")
    )
    return {
        "status": status,
        "details": {
            "issues": issues,
            "error_envelope_active": has_error_envelope,
            "api_headers_active": has_api_headers,
            "idempotency_active": has_idempotency,
            "debug_mode": settings.DEBUG,
            "missing_error_templates": missing_templates,
        },
        "soc2_controls": ["CC7.2", "CC7.5"],
    }


@register("rate_limiting", "security", soc2_controls=["CC6.2", "CC3.3"])
def check_rate_limiting():
    """Validate brute force protection (SOC 2 CC6.2/CC3.3, ISO A.9.4, NIST AC-7)."""
    issues = []

    # Check DRF throttle configuration
    drf = getattr(settings, "REST_FRAMEWORK", {})
    throttle_classes = drf.get("DEFAULT_THROTTLE_CLASSES", [])
    throttle_rates = drf.get("DEFAULT_THROTTLE_RATES", {})

    has_throttle = len(throttle_classes) > 0
    if not has_throttle:
        issues.append("No DEFAULT_THROTTLE_CLASSES configured in REST_FRAMEWORK")

    has_rates = len(throttle_rates) > 0
    if not has_rates:
        issues.append("No DEFAULT_THROTTLE_RATES configured in REST_FRAMEWORK")

    # Check NUM_PROXIES is set (needed for correct IP detection behind proxy)
    num_proxies = drf.get("NUM_PROXIES", 0)
    if num_proxies == 0:
        issues.append("REST_FRAMEWORK NUM_PROXIES is 0 — rate limiting may use spoofable X-Forwarded-For")

    # Check middleware for any rate limiting
    middleware = getattr(settings, "MIDDLEWARE", [])
    has_rate_middleware = any("ratelimit" in m.lower() or "throttle" in m.lower() for m in middleware)

    status = "pass" if not issues else ("fail" if not has_throttle else "warning")
    return {
        "status": status,
        "details": {
            "issues": issues,
            "throttle_classes": [c.split(".")[-1] for c in throttle_classes],
            "throttle_rates": throttle_rates,
            "num_proxies": num_proxies,
            "rate_middleware": has_rate_middleware,
        },
        "soc2_controls": ["CC6.2", "CC3.3"],
    }


@register("secret_management", "confidentiality", soc2_controls=["CC6.1"])
def check_secret_management():
    """Validate credential hygiene (SOC 2 CC6.1, ISO A.10.1, NIST SC-12)."""
    issues = []

    # Check that settings.py uses config object (not hardcoded)
    settings_path = Path(settings.BASE_DIR) / "svend" / "settings.py"
    hardcoded_secrets = False
    if settings_path.exists():
        content = settings_path.read_text()
        # SECRET_KEY should reference config, not be a literal string
        import re

        if re.search(r'SECRET_KEY\s*=\s*["\']', content):
            issues.append("SECRET_KEY appears hardcoded in settings.py (should use config/env)")
            hardcoded_secrets = True
        # Check for hardcoded API keys
        for pattern in [r'ANTHROPIC_API_KEY\s*=\s*["\']sk-', r'STRIPE_SECRET\s*=\s*["\']sk_']:
            if re.search(pattern, content):
                issues.append("API key appears hardcoded in settings.py")
                hardcoded_secrets = True

    # Field encryption key must be configured
    fek = getattr(settings, "FIELD_ENCRYPTION_KEY", None)
    if not fek:
        issues.append("FIELD_ENCRYPTION_KEY is not configured")

    # Check for .env file in project root (ok if covered by .gitignore)
    env_file = Path(settings.BASE_DIR) / ".env"
    env_in_gitignore = False
    gitignore_path = Path(settings.BASE_DIR) / ".gitignore"
    if gitignore_path.exists():
        env_in_gitignore = ".env" in gitignore_path.read_text()
    if env_file.exists() and not env_in_gitignore:
        issues.append(".env file found in project root and NOT in .gitignore — risk of committing secrets")

    # SECRET_KEY strength
    sk = getattr(settings, "SECRET_KEY", "")
    if len(sk) < 50:
        issues.append(f"SECRET_KEY length is {len(sk)} (recommended ≥50)")

    status = "pass" if not issues else ("fail" if hardcoded_secrets else "warning")
    return {
        "status": status,
        "details": {
            "issues": issues,
            "field_encryption_configured": bool(fek),
            "secret_key_length": len(sk),
            "env_file_present": env_file.exists(),
            "hardcoded_secrets_found": hardcoded_secrets,
        },
        "soc2_controls": ["CC6.1"],
    }


@register("log_completeness", "security", soc2_controls=["CC7.2"])
def check_log_completeness():
    """Validate logging pipeline health (SOC 2 CC7.2, ISO A.12.4.2, NIST AU-9/AU-12)."""
    issues = []

    logging_config = getattr(settings, "LOGGING", {})

    # Check handlers exist
    handlers = logging_config.get("handlers", {})
    if not handlers:
        issues.append("No logging handlers configured")

    # Check for file handler (persistence)
    has_file_handler = any(
        "FileHandler" in h.get("class", "") or "RotatingFileHandler" in h.get("class", "") for h in handlers.values()
    )
    if not has_file_handler:
        issues.append("No file-based log handler — logs not persisted to disk")

    # Check for security logger
    loggers = logging_config.get("loggers", {})
    has_audit_logger = "syn.audit" in loggers
    if not has_audit_logger:
        issues.append("No syn.audit logger configured — audit events may not be captured")

    # Check root logger level (should not be DEBUG in production)
    root = logging_config.get("root", {})
    root_level = root.get("level", "WARNING")
    if root_level == "DEBUG" and not settings.DEBUG:
        issues.append("Root logger level is DEBUG in production — excessive logging, potential info leak")

    # Check log files are writable
    unwritable = []
    for name, handler in handlers.items():
        filename = handler.get("filename")
        if filename:
            log_path = Path(filename)
            if log_path.parent.exists() and not os.access(str(log_path.parent), os.W_OK):
                unwritable.append(str(filename))
    if unwritable:
        issues.append(f"Log directories not writable: {', '.join(unwritable)}")

    # Check CorrelationMiddleware position (should be before AuditLoggingMiddleware)
    middleware = getattr(settings, "MIDDLEWARE", [])
    corr_idx = next((i for i, m in enumerate(middleware) if "CorrelationMiddleware" in m), -1)
    audit_idx = next((i for i, m in enumerate(middleware) if "AuditLoggingMiddleware" in m), -1)
    if corr_idx >= 0 and audit_idx >= 0 and corr_idx > audit_idx:
        issues.append("CorrelationMiddleware should be before AuditLoggingMiddleware in MIDDLEWARE")

    status = "pass" if not issues else ("fail" if not handlers else "warning")
    return {
        "status": status,
        "details": {
            "issues": issues,
            "handler_count": len(handlers),
            "logger_count": len(loggers),
            "has_file_handler": has_file_handler,
            "has_audit_logger": has_audit_logger,
            "root_level": root_level,
        },
        "soc2_controls": ["CC7.2"],
    }


@register("security_headers", "security", soc2_controls=["CC6.1"])
def check_security_headers():
    """Validate HTTP security headers beyond TLS (SOC 2 CC6.1, ISO A.13.1, NIST SC-8)."""
    issues = []

    # X-Frame-Options (clickjacking protection)
    xfo = getattr(settings, "X_FRAME_OPTIONS", "SAMEORIGIN")
    if xfo not in ("DENY", "SAMEORIGIN"):
        issues.append(f"X_FRAME_OPTIONS is '{xfo}' — should be DENY or SAMEORIGIN")

    # Referrer-Policy
    rp = getattr(settings, "SECURE_REFERRER_POLICY", None)
    if not rp and not settings.DEBUG:
        issues.append("SECURE_REFERRER_POLICY is not set — browser sends full Referer header")

    # Cross-Origin-Opener-Policy
    coop = getattr(settings, "SECURE_CROSS_ORIGIN_OPENER_POLICY", None)
    if not coop and not settings.DEBUG:
        issues.append("SECURE_CROSS_ORIGIN_OPENER_POLICY is not set")

    # CSRF trusted origins — should not contain wildcards
    csrf_origins = getattr(settings, "CSRF_TRUSTED_ORIGINS", [])
    for origin in csrf_origins:
        if "*" in origin:
            issues.append(f"CSRF_TRUSTED_ORIGINS contains wildcard: {origin}")

    # CORS origins — should not allow all
    cors_all = getattr(settings, "CORS_ALLOW_ALL_ORIGINS", False)
    if cors_all:
        issues.append("CORS_ALLOW_ALL_ORIGINS is True — any origin can make credentialed requests")

    # CSP policy presence (check for key directives)
    csp = getattr(settings, "CONTENT_SECURITY_POLICY", {})
    if csp:
        if "'none'" not in csp.get("object-src", []):
            issues.append("CSP object-src does not include 'none' — plugin execution possible")
        if "'self'" not in csp.get("base-uri", []):
            issues.append("CSP base-uri does not include 'self' — base tag injection possible")
    else:
        if not settings.DEBUG:
            issues.append("CONTENT_SECURITY_POLICY not configured")

    status = "pass" if not issues else ("fail" if cors_all else "warning")
    return {
        "status": status,
        "details": {
            "issues": issues,
            "x_frame_options": xfo,
            "referrer_policy": rp,
            "cross_origin_opener_policy": coop,
            "cors_allow_all": cors_all,
            "csp_configured": bool(csp),
        },
        "soc2_controls": ["CC6.1"],
    }


@register("incident_readiness", "processing_integrity", soc2_controls=["CC7.4"])
def check_incident_readiness():
    """Validate incident response preparedness (SOC 2 CC7.4, ISO A.16.1, NIST IR-1/IR-8)."""
    issues = []

    # Check INC-001 standard exists
    standards_dir = Path(settings.BASE_DIR).parent.parent.parent / "docs"
    inc_standard = standards_dir / "standards" / "INC-001.md"
    inc_found = inc_standard.exists()
    if not inc_found:
        issues.append("INC-001 incident response standard not found in docs/standards/")

    # Check incident response policy exists
    ir_paths = [
        standards_dir / "compliance" / "policies" / "incident-response.md",
        standards_dir / "compliance" / "incident-response.md",
    ]
    ir_found = any(p.exists() for p in ir_paths) or inc_found  # INC-001 serves as policy
    if not ir_found:
        issues.append("Incident response policy not found")

    # Check Incident model exists and is accessible
    try:
        from syn.audit.models import Incident  # noqa: F401

        incident_model_ok = True
    except ImportError:
        incident_model_ok = False
        issues.append("Incident model not found in syn.audit.models")

    # Check emergency change process documented (CHG-001 §9)
    chg_path = standards_dir / "standards" / "CHG-001.md"
    chg_has_emergency = False
    if chg_path.exists():
        content = chg_path.read_text()
        chg_has_emergency = "emergency" in content.lower() and "retroactive" in content.lower()
    if not chg_has_emergency:
        issues.append("CHG-001 does not document emergency change process with retroactive review")

    # Check BCDR documentation exists
    bcdr_paths = [
        standards_dir / "compliance" / "policies" / "bcdr.md",
        standards_dir / "compliance" / "bcdr.md",
    ]
    bcdr_found = any(p.exists() for p in bcdr_paths)
    if not bcdr_found:
        issues.append("Business continuity / disaster recovery policy not found")

    # Check recent emergency changes have retroactive review
    try:
        from datetime import timedelta

        from django.utils import timezone

        from syn.audit.models import ChangeRequest

        cutoff = timezone.now() - timedelta(days=30)
        emergencies = ChangeRequest.objects.filter(
            is_emergency=True,
            created_at__gte=cutoff,
        ).exclude(status__in=["cancelled", "draft"])

        unreviewed = 0
        for e in emergencies:
            has_retro = e.risk_assessments.filter(is_retroactive=True).exists()
            if not has_retro:
                unreviewed += 1
        if unreviewed > 0:
            issues.append(f"{unreviewed} emergency change(s) in last 30d without retroactive risk assessment")
    except Exception:
        pass  # Models may not be available in all contexts

    status = "pass" if not issues else ("fail" if not inc_found else "warning")
    return {
        "status": status,
        "details": {
            "issues": issues,
            "inc_001_standard": inc_found,
            "incident_response_doc": ir_found,
            "incident_model": incident_model_ok if "incident_model_ok" in dir() else False,
            "emergency_process_documented": chg_has_emergency,
            "bcdr_doc": bcdr_found,
        },
        "soc2_controls": ["CC7.4"],
    }


@register("sla_compliance", "availability", soc2_controls=["CC9.2", "CC4.1"])
def check_sla_compliance():
    """Measure all SLA definitions against operational data (SLA-001, SOC 2 CC9.2/CC4.1)."""
    from syn.audit.standards import parse_all_sla_definitions

    sla_defs = parse_all_sla_definitions()
    if not sla_defs:
        return {
            "status": "warning",
            "details": {"error": "No SLA definitions found in standards"},
            "soc2_controls": ["CC9.2", "CC4.1"],
        }

    results = []
    for sla in sla_defs:
        measurement = _measure_sla(sla)
        results.append(measurement)

    # Determine overall status: fail if any critical SLA breached, warning if non-critical breached
    critical_breach = any(r["status"] == "breach" and r["severity"] == "critical" for r in results)
    any_breach = any(r["status"] == "breach" for r in results)

    total = len(results)
    met = sum(1 for r in results if r["status"] == "met")
    breached = sum(1 for r in results if r["status"] == "breach")
    unmeasurable = sum(1 for r in results if r["status"] == "unmeasurable")

    if critical_breach:
        status = "fail"
    elif any_breach:
        status = "warning"
    else:
        status = "pass"

    return {
        "status": status,
        "details": {
            "total_slas": total,
            "met": met,
            "breached": breached,
            "unmeasurable": unmeasurable,
            "sla_results": results,
        },
        "soc2_controls": ["CC9.2", "CC4.1"],
    }


def _measure_sla(sla):
    """Dispatch measurement by metric type. Returns dict with status and current_value."""
    base = {
        "sla_id": sla.sla_id,
        "description": sla.description,
        "metric": sla.metric,
        "target": sla.target,
        "window": sla.window,
        "severity": sla.severity,
        "measurement": sla.measurement,
        "standard": sla.standard,
        "section": sla.section,
    }

    if sla.measurement == "manual":
        base["status"] = "unmeasurable"
        base["current_value"] = None
        base["reason"] = "Manual measurement — requires operational verification"
        return base

    try:
        dispatch = {
            "availability": _measure_availability,
            "durability": _measure_durability,
            "compliance": _measure_compliance_rate,
            "change_velocity": _measure_change_velocity,
            "response_time": _measure_response_time,
            "incident_response": _measure_incident_response,
        }
        fn = dispatch.get(sla.metric)
        if fn:
            result = fn(sla)
            base.update(result)
        else:
            base["status"] = "unmeasurable"
            base["current_value"] = None
            base["reason"] = f"Unknown metric type: {sla.metric}"
    except Exception as e:
        base["status"] = "unmeasurable"
        base["current_value"] = None
        base["reason"] = f"Measurement error: {str(e)[:200]}"

    return base


def _parse_target(target_str):
    """Parse target string like '99.9%', '2000ms', '24h', '168h' into (value, unit)."""
    import re as _re

    m = _re.match(r"([\d.]+)\s*(%|ms|h|d|s)", target_str)
    if not m:
        return None, None
    return float(m.group(1)), m.group(2)


def _measure_availability(sla):
    """Measure availability via health endpoint ping success rate (SLA-001 §5.1)."""
    from syn.audit.models import HealthPing

    now = timezone.now()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    pings = HealthPing.objects.filter(timestamp__gte=month_start)
    total = pings.count()
    if total == 0:
        return {
            "status": "unmeasurable",
            "current_value": None,
            "reason": "No health pings recorded this month. Health ping task may not be running.",
        }

    healthy = pings.filter(is_healthy=True).count()
    availability_pct = (healthy / total) * 100

    target_val, _ = _parse_target(sla.target)
    if target_val is None:
        return {"status": "unmeasurable", "current_value": None, "reason": f"Cannot parse target: {sla.target}"}

    return {
        "status": "met" if availability_pct >= target_val else "breach",
        "current_value": f"{availability_pct:.2f}%",
    }


def _measure_durability(sla):
    """Measure durability SLAs — delegates to backup_freshness for RPO."""
    if "rpo" in sla.sla_id or "recovery-point" in sla.sla_id:
        # Check backup freshness: is the latest backup within target hours?
        from syn.audit.models import ComplianceCheck

        latest_backup = (
            ComplianceCheck.objects.filter(check_name="backup_freshness", status="pass").order_by("-run_at").first()
        )
        if not latest_backup:
            return {"status": "unmeasurable", "current_value": None, "reason": "No backup_freshness check results"}

        target_val, unit = _parse_target(sla.target)
        if unit == "h" and target_val:
            hours_since = (timezone.now() - latest_backup.run_at).total_seconds() / 3600
            return {
                "status": "met" if hours_since <= target_val else "breach",
                "current_value": f"{hours_since:.1f}h",
            }

    # RTO and other durability SLAs are manual
    return {"status": "unmeasurable", "current_value": None, "reason": "Requires operational verification"}


def _measure_compliance_rate(sla):
    """Measure compliance check pass rate for the current month."""
    from syn.audit.models import ComplianceCheck

    now = timezone.now()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    checks = ComplianceCheck.objects.filter(run_at__gte=month_start)
    total = checks.count()
    if total == 0:
        return {"status": "unmeasurable", "current_value": None, "reason": "No checks this month"}

    passed = checks.filter(status="pass").count()
    rate = (passed / total) * 100

    target_val, _ = _parse_target(sla.target)
    if target_val is None:
        return {"status": "unmeasurable", "current_value": None, "reason": f"Cannot parse target: {sla.target}"}

    return {
        "status": "met" if rate >= target_val else "breach",
        "current_value": f"{rate:.1f}%",
    }


def _measure_change_velocity(sla):
    """Measure change management SLAs: emergency retro review, post-incident review, stale changes."""
    from datetime import timedelta

    from syn.audit.models import ChangeRequest

    now = timezone.now()
    target_val, unit = _parse_target(sla.target)
    if target_val is None or unit != "h":
        return {"status": "unmeasurable", "current_value": None, "reason": f"Cannot parse target: {sla.target}"}

    target_hours = target_val

    if "retro" in sla.sla_id or "emergency" in sla.sla_id:
        # Emergency changes without retroactive risk assessment within target hours
        cutoff = now - timedelta(days=30)
        emergencies = ChangeRequest.objects.filter(
            is_emergency=True,
            created_at__gte=cutoff,
        ).exclude(status__in=["cancelled", "draft"])

        violations = 0
        for cr in emergencies:
            has_retro = cr.risk_assessments.filter(is_retroactive=True).exists()
            if not has_retro:
                age_hours = (now - cr.created_at).total_seconds() / 3600
                if age_hours > target_hours:
                    violations += 1

        return {
            "status": "met" if violations == 0 else "breach",
            "current_value": f"{violations} violation(s)",
        }

    elif "post-incident" in sla.sla_id:
        # Post-incident review within target hours
        cutoff = now - timedelta(days=30)
        emergencies = ChangeRequest.objects.filter(
            is_emergency=True,
            created_at__gte=cutoff,
            status="completed",
        )

        violations = 0
        for cr in emergencies:
            # Check if a review log entry exists within target hours of completion
            completion_log = cr.logs.filter(action="completed").order_by("created_at").first()
            if completion_log:
                review_log = cr.logs.filter(
                    action__in=["review", "post_incident_review"],
                    created_at__lte=completion_log.created_at + timedelta(hours=target_hours),
                ).exists()
                if not review_log:
                    violations += 1

        return {
            "status": "met" if violations == 0 else "breach",
            "current_value": f"{violations} violation(s)",
        }

    elif "stale" in sla.sla_id:
        # Changes stuck in in_progress beyond target hours
        stale_cutoff = now - timedelta(hours=target_hours)
        stale = ChangeRequest.objects.filter(
            status="in_progress",
            updated_at__lt=stale_cutoff,
        ).count()

        return {
            "status": "met" if stale == 0 else "breach",
            "current_value": f"{stale} stale CR(s)",
        }

    return {"status": "unmeasurable", "current_value": None, "reason": "Unrecognized change_velocity SLA"}


def _measure_response_time(sla):
    """Response time SLAs — measured from RequestMetric bucketed telemetry."""
    from syn.log.models import RequestMetric

    now = timezone.now()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    buckets = RequestMetric.objects.filter(
        bucket_start__gte=month_start,
        bucket_start__lt=now,
    )
    if not buckets.exists():
        return {
            "status": "unmeasurable",
            "current_value": None,
            "reason": "No request metrics recorded yet this month",
        }

    # Merge all reservoir samples for the month
    all_samples = []
    for b in buckets.only("duration_samples"):
        all_samples.extend(b.duration_samples or [])

    if not all_samples:
        return {
            "status": "unmeasurable",
            "current_value": None,
            "reason": "No duration samples available",
        }

    target_value, target_unit = _parse_target(sla.target)

    # Determine percentile from sla_id (p99 vs p95)
    pct = 99 if "p99" in sla.sla_id else 95

    sorted_s = sorted(all_samples)
    n = len(sorted_s)
    k = (n - 1) * (pct / 100)
    f_idx = int(k)
    c_idx = min(f_idx + 1, n - 1)
    current_ms = sorted_s[f_idx] + (k - f_idx) * (sorted_s[c_idx] - sorted_s[f_idx])

    met = current_ms <= target_value
    return {
        "status": "met" if met else "breach",
        "current_value": f"{current_ms:.0f}ms",
        "target": sla.target,
        "percentile": f"p{pct}",
        "sample_count": len(all_samples),
    }


def _measure_incident_response(sla):
    """Measure incident response SLAs using Incident model (INC-001 §6)."""
    from syn.audit.models import Incident

    now = timezone.now()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    incidents = Incident.objects.filter(detected_at__gte=month_start)
    if not incidents.exists():
        return {"status": "met", "current_value": "No incidents", "reason": "No incidents recorded this month"}

    # Check based on SLA sla_id: ack SLAs vs resolution SLAs
    if "ack" in sla.sla_id.lower():
        breached = sum(1 for i in incidents if i.is_ack_sla_breached)
    else:
        resolved = incidents.exclude(status__in=["detected", "acknowledged", "investigating", "mitigating"])
        if not resolved.exists():
            return {
                "status": "met",
                "current_value": "No resolved incidents",
                "reason": "No resolved incidents to measure resolution SLA against",
            }
        breached = sum(1 for i in resolved if i.is_resolution_sla_breached)
        incidents = resolved

    total = incidents.count()
    compliance_pct = ((total - breached) / total) * 100 if total else 100

    target_val, _ = _parse_target(sla.target)
    if target_val is None:
        return {"status": "unmeasurable", "current_value": None, "reason": f"Cannot parse target: {sla.target}"}

    return {
        "status": "met" if compliance_pct >= target_val else "breach",
        "current_value": f"{compliance_pct:.1f}% ({breached}/{total} breached)",
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


# ---------------------------------------------------------------------------
# Code Style Check (STY-001)
# ---------------------------------------------------------------------------

# Directories to skip when scanning for style violations
_STYLE_SKIP_DIRS = {"__pycache__", "migrations", "staticfiles", "media", "node_modules"}

# snake_case pattern for files and functions
_SNAKE_RE = _re.compile(r"^[a-z_][a-z0-9_]*$")

# PascalCase pattern for classes (allow _ prefix for private classes)
_PASCAL_RE = _re.compile(r"^_?[A-Z][a-zA-Z0-9]*$")

# File name pattern (lowercase_snake.py, allow _ prefix for private modules)
_FILE_SNAKE_RE = _re.compile(r"^_?[a-z][a-z0-9_]*\.py$")


def _should_skip_path(path):
    """Check if a path should be skipped for style scanning."""
    return any(skip in path.parts for skip in _STYLE_SKIP_DIRS)


def _scan_file_names(web_root):
    """Check all .py files match lowercase_snake.py pattern.

    STY-001 §4.1: All Python files use lowercase_snake naming.
    Skips migrations/, __pycache__/, staticfiles/, media/.
    """
    violations = []
    for py_file in sorted(web_root.rglob("*.py")):
        if _should_skip_path(py_file):
            continue
        name = py_file.name
        if name == "__init__.py":
            continue
        # Skip config files with dots (e.g., gunicorn.conf.py)
        if "." in name.replace(".py", ""):
            continue
        if not _FILE_SNAKE_RE.match(name):
            violations.append(
                {
                    "file": str(py_file.relative_to(web_root)),
                    "violation": f"File '{name}' does not match lowercase_snake.py",
                }
            )
    return violations


def _scan_class_names(web_root):
    """AST-based: all class names must be PascalCase.

    STY-001 §4.2: Classes use PascalCase naming.
    """
    violations = []
    for py_file in sorted(web_root.rglob("*.py")):
        if _should_skip_path(py_file):
            continue
        try:
            source = py_file.read_text(errors="ignore")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if not _PASCAL_RE.match(node.name):
                    violations.append(
                        {
                            "file": str(py_file.relative_to(web_root)),
                            "class": node.name,
                            "line": node.lineno,
                        }
                    )
    return violations


def _scan_function_names(web_root):
    """AST-based: all function names must be lowercase_snake.

    STY-001 §4.3: Functions and methods use lowercase_snake naming.
    Excludes dunder methods (__init__, __str__, etc.).
    """
    violations = []
    for py_file in sorted(web_root.rglob("*.py")):
        if _should_skip_path(py_file):
            continue
        try:
            source = py_file.read_text(errors="ignore")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name
                # Skip dunder methods
                if name.startswith("__") and name.endswith("__"):
                    continue
                # Skip standard unittest/Django lifecycle methods (camelCase by design)
                if name in ("setUp", "tearDown", "setUpClass", "tearDownClass", "setUpTestData", "addCleanup"):
                    continue
                if not _SNAKE_RE.match(name):
                    violations.append(
                        {
                            "file": str(py_file.relative_to(web_root)),
                            "function": name,
                            "line": node.lineno,
                        }
                    )
    return violations


def _check_module_docstrings(web_root):
    """AST-based: every .py file (except empty __init__.py) has a module docstring.

    STY-001 §6.1: Module-level docstring as first statement.
    """
    missing = []
    for py_file in sorted(web_root.rglob("*.py")):
        if _should_skip_path(py_file):
            continue
        try:
            source = py_file.read_text(errors="ignore")
        except Exception:
            continue
        # Skip empty __init__.py files
        if py_file.name == "__init__.py" and not source.strip():
            continue
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue
        docstring = ast.get_docstring(tree)
        if not docstring:
            missing.append(
                {
                    "file": str(py_file.relative_to(web_root)),
                }
            )
    return missing


def _check_import_order(web_root):
    """Check imports follow stdlib -> third-party -> local ordering.

    STY-001 §5: Three-group import convention.
    Scans ALL .py files in the project.
    """
    # Get stdlib module names
    if hasattr(sys, "stdlib_module_names"):
        stdlib_names = sys.stdlib_module_names
    else:
        # Python < 3.10 fallback
        stdlib_names = set(sys.builtin_module_names)

    violations = []
    for fpath in sorted(web_root.rglob("*.py")):
        if _should_skip_path(fpath):
            continue
        try:
            source = fpath.read_text(errors="ignore")
            tree = ast.parse(source, filename=str(fpath))
        except (SyntaxError, UnicodeDecodeError):
            continue

        # Extract top-level import nodes
        imports = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_mod = alias.name.split(".")[0]
                    imports.append((node.lineno, root_mod, "import"))
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_mod = node.module.split(".")[0]
                else:
                    root_mod = ""
                level = node.level or 0
                if level > 0:
                    imports.append((node.lineno, f".{root_mod}", "relative"))
                else:
                    imports.append((node.lineno, root_mod, "from"))

        if not imports:
            continue

        # Classify each import
        prev_group = 0
        for lineno, mod, kind in imports:
            if kind == "relative" or mod.startswith("."):
                group = 3  # local/relative
            elif mod in stdlib_names:
                group = 1  # stdlib
            elif mod in (
                "syn",
                "core",
                "agents_api",
                "api",
                "accounts",
                "chat",
                "workbench",
                "forge",
                "files",
                "tempora",
                "agents",
                "inference",
                "svend_config",
                "notifications",
                "privacy",
            ):
                group = 3  # local
            else:
                group = 2  # third-party

            if group < prev_group:
                violations.append(
                    {
                        "file": str(fpath.relative_to(web_root)),
                        "line": lineno,
                        "violation": f"Import '{mod}' (group {group}) after group {prev_group}",
                    }
                )
                break  # One violation per file is enough
            prev_group = group

    return violations


def _check_wildcard_imports(web_root):
    """Check for wildcard imports (from X import *).

    STY-001 §5.2: No wildcard imports.
    """
    violations = []
    for py_file in sorted(web_root.rglob("*.py")):
        if _should_skip_path(py_file):
            continue
        try:
            source = py_file.read_text(errors="ignore")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        violations.append(
                            {
                                "file": str(py_file.relative_to(web_root)),
                                "line": node.lineno,
                                "module": node.module or "",
                            }
                        )
    return violations


def _check_model_field_naming(web_root):
    """Check Django model fields follow DAT-001 §7 naming conventions.

    Rules:
    - BooleanField must have is_ prefix (tracked as debt — migration required)
    - ForeignKey must NOT have _fk suffix
    - JSONField must NOT use mutable defaults ([] or {})
    """
    _BOOL_RE = _re.compile(r"^(is|has|can)_[a-z]")
    violations = {"boolean_prefix": [], "fk_suffix": [], "mutable_default": []}

    model_dirs = [
        web_root / "agents_api",
        web_root / "core",
        web_root / "accounts",
        web_root / "api",
        web_root / "chat",
        web_root / "files",
        web_root / "forge",
        web_root / "workbench",
        web_root / "syn",
    ]

    for model_dir in model_dirs:
        for py_file in sorted(model_dir.rglob("*.py")):
            if _should_skip_path(py_file):
                continue
            # Only check model files (models.py or files inside models/ dirs)
            if py_file.stem != "models" and "models" not in py_file.parts:
                continue
            try:
                source = py_file.read_text(errors="ignore")
                tree = ast.parse(source, filename=str(py_file))
            except (SyntaxError, UnicodeDecodeError):
                continue

            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                for item in node.body:
                    if not isinstance(item, ast.Assign):
                        continue
                    if not item.targets:
                        continue
                    target = item.targets[0]
                    if not isinstance(target, ast.Name):
                        continue
                    field_name = target.id
                    call = item.value
                    if not isinstance(call, ast.Call):
                        continue

                    # Get field type name
                    func = call.func
                    if isinstance(func, ast.Attribute):
                        field_type = func.attr
                    elif isinstance(func, ast.Name):
                        field_type = func.id
                    else:
                        continue

                    rel_path = str(py_file.relative_to(web_root))

                    # BooleanField / NullBooleanField must have is_ prefix
                    if field_type in ("BooleanField", "NullBooleanField"):
                        if not _BOOL_RE.match(field_name):
                            violations["boolean_prefix"].append(
                                {
                                    "file": rel_path,
                                    "line": item.lineno,
                                    "field": field_name,
                                    "class": node.name,
                                }
                            )

                    # ForeignKey must NOT have _fk suffix
                    if field_type == "ForeignKey" and field_name.endswith("_fk"):
                        violations["fk_suffix"].append(
                            {
                                "file": rel_path,
                                "line": item.lineno,
                                "field": field_name,
                                "class": node.name,
                            }
                        )

                    # JSONField must NOT use mutable defaults
                    if field_type == "JSONField":
                        for kw in call.keywords:
                            if kw.arg == "default":
                                if isinstance(kw.value, (ast.List, ast.Dict)):
                                    violations["mutable_default"].append(
                                        {
                                            "file": rel_path,
                                            "line": item.lineno,
                                            "field": field_name,
                                            "class": node.name,
                                        }
                                    )

    return violations


_UPPER_SNAKE_RE = _re.compile(r"^_?[A-Z][A-Z0-9_]*$")

# Key infrastructure files for module layout checking
_KEY_INFRA_FILES = [
    "syn/audit/models.py",
    "syn/audit/compliance.py",
    "syn/audit/utils.py",
    "syn/audit/events.py",
    "syn/audit/signals.py",
    "syn/core/base_models.py",
    "syn/err/exceptions.py",
    "syn/log/middleware.py",
]


def _scan_constant_names(web_root):
    """Scan module-level assignments that look like constants.

    Verify all-uppercase target names match ^[A-Z][A-Z0-9_]*$.
    Skip dunder names (__all__, __version__).

    STY-001 §4.4: Constants use UPPER_SNAKE_CASE.
    """
    violations = []
    for py_file in sorted(web_root.rglob("*.py")):
        if _should_skip_path(py_file):
            continue
        try:
            source = py_file.read_text(errors="ignore")
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            continue

        rel_path = str(py_file.relative_to(web_root))
        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                name = target.id
                # Skip dunder names
                if name.startswith("__") and name.endswith("__"):
                    continue
                # Only check names that look like constants (all-uppercase)
                if name != name.upper():
                    continue
                if not _UPPER_SNAKE_RE.match(name):
                    violations.append(
                        {
                            "file": rel_path,
                            "line": node.lineno,
                            "name": name,
                            "violation": f"Constant '{name}' does not match UPPER_SNAKE_CASE",
                        }
                    )
    return violations


def _check_module_layout_order(web_root):
    """Check module layout order for key infrastructure files.

    Expected order: (a) module docstring, (b) imports, (c) constants/logger,
    (d) private helpers (_-prefixed), (e) public code.

    STY-001 §5: Module layout conventions.
    """
    violations = []
    for rel in _KEY_INFRA_FILES:
        fpath = web_root / rel
        if not fpath.exists():
            continue
        try:
            source = fpath.read_text(errors="ignore")
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            continue

        # Classify top-level nodes by category and track first line
        first_import = None
        first_constant = None
        first_func_or_class = None

        for node in ast.iter_child_nodes(tree):
            lineno = getattr(node, "lineno", 0)
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if first_import is None:
                    first_import = lineno
            elif isinstance(node, ast.Assign):
                # Module-level assignment (constant/logger)
                if first_constant is None:
                    first_constant = lineno
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if first_func_or_class is None:
                    first_func_or_class = lineno

        # Check ordering: imports before constants, constants before functions
        if first_import and first_constant and first_import > first_constant:
            violations.append(
                {
                    "file": rel,
                    "violation": f"Imports (line {first_import}) appear after constants (line {first_constant})",
                }
            )
        if first_constant and first_func_or_class and first_constant > first_func_or_class:
            violations.append(
                {
                    "file": rel,
                    "violation": f"Constants (line {first_constant}) appear after functions/classes (line {first_func_or_class})",
                }
            )
        if first_import and first_func_or_class and first_import > first_func_or_class:
            violations.append(
                {
                    "file": rel,
                    "violation": f"Imports (line {first_import}) appear after functions/classes (line {first_func_or_class})",
                }
            )

    return violations


@register("code_style", "processing_integrity", soc2_controls=["CC8.1"])
def check_code_style():
    """STY-001: Code style & naming conventions enforcement.

    Scans .py files for naming violations (files, classes, functions),
    missing docstrings, import order issues, wildcard imports, URL kebab-case,
    timestamp field naming, and class docstrings.

    Status: fail on file/class naming violations, warning on function/docstring/import issues.

    Compliance: STY-001 (Code Style), SOC 2 CC8.1
    """
    web_root = Path(settings.BASE_DIR)

    file_violations = _scan_file_names(web_root)
    class_violations = _scan_class_names(web_root)
    function_violations = _scan_function_names(web_root)
    missing_docstrings = _check_module_docstrings(web_root)
    import_violations = _check_import_order(web_root)
    wildcard_violations = _check_wildcard_imports(web_root)
    field_violations = _check_model_field_naming(web_root)
    url_violations = _check_arch_url_kebab_case(web_root)
    timestamp_violations = _check_arch_timestamp_naming(web_root)
    class_docstring_violations = _scan_class_docstrings(web_root)
    constant_violations = _scan_constant_names(web_root)
    layout_violations = _check_module_layout_order(web_root)

    total = (
        len(file_violations)
        + len(class_violations)
        + len(function_violations)
        + len(missing_docstrings)
        + len(import_violations)
        + len(wildcard_violations)
        + len(field_violations["fk_suffix"])
        + len(field_violations["mutable_default"])
        + len(url_violations)
        + len(timestamp_violations)
        + len(constant_violations)
        + len(layout_violations)
    )

    # Count scanned files
    files_scanned = sum(1 for _ in web_root.rglob("*.py") if not _should_skip_path(_))

    # Determine status: fail on hard violations, warning on debt items
    hard_fail = (
        file_violations
        or class_violations
        or wildcard_violations
        or field_violations["fk_suffix"]
        or field_violations["mutable_default"]
        or field_violations["boolean_prefix"]
        or url_violations
        or constant_violations
    )
    soft_warn = (
        function_violations
        or missing_docstrings
        or import_violations
        or timestamp_violations
        or class_docstring_violations
        or layout_violations
    )

    if hard_fail:
        status = "fail"
    elif soft_warn:
        status = "warning"
    else:
        status = "pass"

    return {
        "status": status,
        "details": {
            "files_scanned": files_scanned,
            "file_naming_violations": file_violations[:20],
            "class_naming_violations": class_violations[:20],
            "function_naming_violations": function_violations[:20],
            "missing_docstrings": missing_docstrings[:20],
            "import_order_violations": import_violations[:20],
            "wildcard_import_violations": wildcard_violations[:20],
            "boolean_field_violations": field_violations["boolean_prefix"][:20],
            "fk_suffix_violations": field_violations["fk_suffix"][:20],
            "mutable_default_violations": field_violations["mutable_default"][:20],
            "url_kebab_violations": url_violations[:20],
            "timestamp_naming_violations": timestamp_violations[:20],
            "class_docstring_violations": class_docstring_violations[:20],
            "constant_violations": constant_violations[:20],
            "layout_violations": layout_violations[:20],
            "total_violations": total,
        },
        "soc2_controls": ["CC8.1"],
    }


# ---------------------------------------------------------------------------
# Architecture Map Check (MAP-001)
# ---------------------------------------------------------------------------

_GIT_ROOT = Path(settings.BASE_DIR).parent.parent.parent  # /home/eric/kjerne
_MAP_STANDARDS_DIR = _GIT_ROOT / "docs" / "standards"

# Non-standard codes that match the standard ID regex but aren't standards
_NON_STANDARD_CODES = {
    # Crypto/encoding algorithms
    "SHA-256",
    "SHA-384",
    "SHA-512",
    "AES-256",
    "AES-128",
    "TLS-128",
    "RSA-256",
    "UTF-008",
    # HTTP status codes
    "HTTP-200",
    "HTTP-400",
    "HTTP-401",
    "HTTP-403",
    "HTTP-404",
    "HTTP-500",
    # Process/anti-pattern IDs (not standards)
    "DEBT-001",
    "DOC-002",
    "AP-003",  # Anti-pattern sub-number within CONFIG-001-AP-003
    "BOOT-001",
    "BOOT-005",  # Dev/prod bootstrap process IDs
    # Error/boundary sub-codes (SRX-CFG-001, SRX-DB-001)
    "CFG-001",
    "DB-001",
    # Enforcement check type choices (DriftViolation.enforcement_check)
    "ENC-002",
    "ENC-003",
    "ENC-004",
    "ENC-005",
    "ENC-006",
    "ENC-007",
    "ENC-008",
    "ENC-009",
    "ENC-010",
    "ENC-011",
    # Scheduler primitive config IDs (PCONF-SCH-101 etc.)
    "SCH-101",
    "SCH-103",
    "SCH-201",
    "SCH-202",
    "SCH-501",
    # System/invariant control codes (SYS-200 INV-008)
    "SYS-200",
    "INV-001",
    "INV-008",
    "INV-011",
    # Placeholder
    "XXX-001",
    # Planning system IDs (INIT-xxx, FEAT-xxx, TASK-xxx) — not standards
    "INIT-003",
    "INIT-009",
    # Calibration case IDs (CAL-INF-001 etc.)
    "INF-001",
}


def _parse_map_table(map_path, table_marker):
    """Parse a GFM pipe table following a <!-- table: marker --> tag.

    Returns list of dicts with lowercase, stripped header keys.
    """
    try:
        lines = map_path.read_text().split("\n")
    except Exception:
        return []

    # Find the marker line
    marker_line = f"<!-- table: {table_marker} -->"
    start = None
    for i, line in enumerate(lines):
        if marker_line in line:
            start = i + 1
            break
    if start is None:
        return []

    # Find header row (first pipe-delimited row after marker)
    header_idx = None
    for i in range(start, min(start + 5, len(lines))):
        if i < len(lines) and "|" in lines[i] and not lines[i].strip().startswith("<!--"):
            header_idx = i
            break
    if header_idx is None:
        return []

    headers = [h.strip().strip("*").lower() for h in lines[header_idx].split("|") if h.strip()]

    # Skip separator row (|---|---|...)
    data_start = header_idx + 2

    results = []
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith("#") or line.startswith("---"):
            break
        if "|" not in line:
            break
        cells = [c.strip() for c in line.split("|") if c.strip() != ""]
        if len(cells) >= len(headers):
            row = {}
            for j, key in enumerate(headers):
                row[key] = cells[j] if j < len(cells) else ""
            results.append(row)
    return results


def _git_head_sha():
    """Get current HEAD commit SHA, or None on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(_GIT_ROOT),
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def _git_head_author():
    """Get HEAD commit author, or None on failure."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%an"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(_GIT_ROOT),
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def _git_changed_files(since_hours=24):
    """Get files changed in last N hours via git log."""
    try:
        result = subprocess.run(
            ["git", "log", f"--since={since_hours} hours ago", "--name-only", "--pretty=format:"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(_GIT_ROOT),
        )
        if result.returncode == 0:
            return list(set(f for f in result.stdout.strip().split("\n") if f))
        return []
    except Exception:
        return []


_STANDARD_ID_RE = _re.compile(r"\b([A-Z]{2,6}-\d{3,4})\b")


def _scan_phantom_references(registry_ids, scan_root):
    """Find standard ID references in .py files not in registry.

    Returns list of {"id", "file"} for unregistered references.
    """
    unregistered = []
    seen = set()

    for py_file in scan_root.rglob("*.py"):
        # Skip noise directories
        parts = py_file.parts
        if "__pycache__" in parts or "migrations" in parts:
            continue

        try:
            content = py_file.read_text(errors="ignore")
        except Exception:
            continue

        for match in _STANDARD_ID_RE.finditer(content):
            code = match.group(1)
            if code in registry_ids or code in _NON_STANDARD_CODES:
                continue
            key = (code, str(py_file))
            if key not in seen:
                seen.add(key)
                rel_path = str(py_file.relative_to(scan_root))
                unregistered.append({"id": code, "file": rel_path})

    return unregistered


def _find_unmapped_modules(module_paths, web_root):
    """Find directories with .py files not covered by module map.

    Returns list of unmapped directory paths (relative to web_root).
    """
    unmapped = []

    # Check syn/ subdirectories
    syn_dir = web_root / "syn"
    if syn_dir.exists():
        for subdir in sorted(syn_dir.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith("__"):
                continue
            has_py = any(subdir.rglob("*.py"))
            if not has_py:
                continue
            rel = f"syn/{subdir.name}/"
            if not any(mp.startswith(rel) or mp == rel for mp in module_paths):
                unmapped.append(rel)

    # Check top-level app directories (those with __init__.py or apps.py)
    for subdir in sorted(web_root.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith((".", "__")):
            continue
        if subdir.name in (
            "syn",
            "static",
            "staticfiles",
            "media",
            "logs",
            "forge_results",
            "templates",
            "__pycache__",
        ):
            continue
        has_init = (subdir / "__init__.py").exists() or (subdir / "apps.py").exists()
        if not has_init:
            continue
        rel = f"{subdir.name}/"
        if not any(mp.startswith(rel) or mp == rel for mp in module_paths):
            unmapped.append(rel)

    return unmapped


def _check_mapped_paths_exist(module_map, web_root):
    """Verify all module map paths exist on disk.

    Returns list of {"module_path"} for missing paths.
    """
    missing = []
    for entry in module_map:
        mod_path = entry.get("module", "").rstrip("/")
        if not mod_path or mod_path == "—":
            continue
        full = web_root / mod_path
        if not full.exists():
            missing.append({"module_path": mod_path})
    return missing


def _map_files_to_standards(changed_files, module_map):
    """Map changed file paths to affected standards via module map prefix matching.

    changed_files: paths relative to git root (e.g., "services/svend/web/syn/audit/compliance.py")
    module_map: parsed table with "module" and "standards" keys

    Returns dict: {standard_id: [file1, file2, ...]}.
    """
    web_prefix = "services/svend/web/"
    affected = {}

    for fpath in changed_files:
        # Normalize to web-relative path
        if fpath.startswith(web_prefix):
            rel = fpath[len(web_prefix) :]
        else:
            continue

        for entry in module_map:
            mod_path = entry.get("module", "").rstrip("/")
            if not mod_path or mod_path == "—":
                continue
            if rel.startswith(mod_path):
                standards_str = entry.get("standards", "")
                for std_id in [s.strip() for s in standards_str.split(",")]:
                    if std_id and std_id != "—":
                        affected.setdefault(std_id, []).append(fpath)
                break

    return affected


def _sync_map_drift_violations(findings, commit_sha, author):
    """Create DriftViolation records for architecture map findings.

    Uses own transaction.atomic() so drift records persist regardless
    of what happens downstream.
    """
    import hashlib

    from django.db import transaction

    from syn.audit.models import DriftViolation

    severity_map = {
        "registry_drift": "HIGH",
        "phantom_drift": "MEDIUM",
        "coverage_drift": "LOW",
        "file_drift": "MEDIUM",
    }

    with transaction.atomic():
        for f in findings:
            payload = f"{f['type']}:{f.get('id', '')}:{f.get('module', '')}:{f.get('file', '')}"
            sig = hashlib.sha256(payload.encode()).hexdigest()[:32]

            if DriftViolation.objects.filter(drift_signature=sig, resolved_at__isnull=True).exists():
                continue

            severity = severity_map.get(f["type"], "MEDIUM")
            sla_hours = DRIFT_SLA_HOURS.get(severity)

            DriftViolation.objects.create(
                drift_signature=sig,
                severity=severity,
                enforcement_check="STD",
                file_path=f.get("file", f.get("module", "MAP-001")),
                violation_message=f["message"],
                detected_by="architecture_map_check",
                git_commit_sha=commit_sha or "",
                git_author=author or "",
                remediation_sla_hours=sla_hours,
            )


@register("architecture_map", "processing_integrity", soc2_controls=["CC7.2", "CC3.1"])
def check_architecture_map():
    """MAP-001: Architecture map enforcement.

    Parses the standards registry and module map tables from MAP-001.md,
    then verifies:
    1. All APPROVED standards have .md files
    2. No unregistered phantom standard references in codebase
    3. All module map paths exist on disk
    4. All syn/ modules are mapped
    5. Git change tracking: identifies affected standards

    SOC 2: CC7.2 (System Activity Monitoring)
    ISO 27001: A.8.1 (Inventory of Assets)
    """
    map_path = _MAP_STANDARDS_DIR / "MAP-001.md"
    if not map_path.exists():
        return {
            "status": "fail",
            "details": {"message": "MAP-001.md not found in docs/standards/"},
            "soc2_controls": ["CC7.2"],
        }

    web_root = Path(settings.BASE_DIR)
    findings = []

    # 1. Parse tables
    registry = _parse_map_table(map_path, "standards-registry")
    module_map = _parse_map_table(map_path, "module-map")

    if not registry:
        return {
            "status": "fail",
            "details": {"message": "Could not parse standards-registry table from MAP-001.md"},
            "soc2_controls": ["CC7.2"],
        }

    # 2. Verify APPROVED standards have files
    missing_files = []
    for entry in registry:
        if entry.get("status") == "APPROVED" and entry.get("file", "—") != "—":
            file_path = _GIT_ROOT / entry["file"]
            if not file_path.exists():
                missing_files.append(entry["id"])
                findings.append(
                    {
                        "type": "registry_drift",
                        "id": entry["id"],
                        "file": entry["file"],
                        "message": f"APPROVED standard {entry['id']} file not found: {entry['file']}",
                    }
                )

    # 3. Scan for unregistered phantom references
    registry_ids = {e.get("id", "") for e in registry}
    phantoms = _scan_phantom_references(registry_ids, web_root / "syn")
    for p in phantoms:
        findings.append(
            {
                "type": "phantom_drift",
                "id": p["id"],
                "file": p["file"],
                "message": f"Unregistered standard reference {p['id']} in syn/{p['file']}",
            }
        )

    # Deduplicate phantom IDs for summary
    phantom_ids = sorted(set(p["id"] for p in phantoms))

    # 4. Check module map paths exist
    missing_paths = _check_mapped_paths_exist(module_map, web_root)
    for mp in missing_paths:
        findings.append(
            {
                "type": "file_drift",
                "module": mp["module_path"],
                "message": f"Mapped module path no longer exists: {mp['module_path']}",
            }
        )

    # 5. Find unmapped modules
    module_paths = [e.get("module", "") for e in module_map]
    unmapped = _find_unmapped_modules(module_paths, web_root)
    for u in unmapped:
        findings.append(
            {
                "type": "coverage_drift",
                "module": u,
                "message": f"Unmapped module directory: {u}",
            }
        )

    # 6. Git change tracking
    commit_sha = _git_head_sha()
    commit_author = _git_head_author()
    changed_files = _git_changed_files()
    affected_standards = _map_files_to_standards(changed_files, module_map)

    # 7. Create DriftViolation records
    try:
        _sync_map_drift_violations(findings, commit_sha, commit_author)
    except Exception as e:
        logger.warning("Architecture map drift sync failed: %s", e)

    # Determine status
    has_registry_drift = any(f["type"] == "registry_drift" for f in findings)

    if has_registry_drift:
        status = "fail"
    elif findings:
        status = "warning"
    else:
        status = "pass"

    approved_count = sum(1 for e in registry if e.get("status") == "APPROVED")
    phantom_count = sum(1 for e in registry if e.get("status") == "PHANTOM")

    return {
        "status": status,
        "details": {
            "registry_entries": len(registry),
            "approved_count": approved_count,
            "phantom_count": phantom_count,
            "module_map_entries": len(module_map),
            "missing_standard_files": missing_files,
            "unregistered_phantoms": phantom_ids,
            "unmapped_modules": unmapped,
            "missing_mapped_paths": [mp["module_path"] for mp in missing_paths],
            "git_commit": commit_sha,
            "git_author": commit_author,
            "changed_files_count": len(changed_files),
            "affected_standards": {k: v for k, v in affected_standards.items()},
            "findings": findings,
        },
        "soc2_controls": ["CC7.2", "CC3.1"],
    }


# ── ARCH-001: Architecture & Structure ──────────────────────────────────

_ARCH_REQUIRED_DIRS = [
    "svend",
    "syn",
    "syn/core",
    "syn/audit",
    "syn/log",
    "syn/api",
    "syn/err",
    "syn/sched",
    "core",
    "accounts",
    "agents_api",
    "api",
    "chat",
    "workbench",
    "forge",
    "files",
    "svend_config",
    "templates",
    "static",
    "ops",
]

_ARCH_PROHIBITED_DIRS = ["tempora", "forge_results"]

_ARCH_FEATURE_APPS = {"agents_api", "chat", "workbench", "forge", "files", "inference"}

_ARCH_ALLOWED_ROOT_FILES = {
    "manage.py",
    "pyproject.toml",
    "gunicorn.conf.py",
    ".gitignore",
    ".env",
    ".env.example",
    ".env.production",
    "Caddyfile",
    "coverage.json",
}

_ARCH_EMPTY_DIR_SKIP = {"migrations", "__pycache__", "media", "logs", "staticfiles", "node_modules"}

_ARCH_FILE_SIZE_WARN = 2000
_ARCH_FILE_SIZE_FAIL = 3000
# Known large files tracked in .kjerne/DEBT.md — exempt from hard fail
_ARCH_KNOWN_LARGE_FILES = {
    "agents_api/dsw_views.py",
    "agents_api/models.py",
    "agents_api/learn_content.py",
    "agents_api/pbs_engine.py",
    "agents_api/dsw/stats.py",
    "agents_api/iso_tests.py",
    "agents_api/dsw/common.py",
    "api/internal_views.py",
    "syn/audit/compliance.py",
    "agents_api/dsw/ml.py",
    "agents_api/dsw/spc.py",
    "agents_api/dsw/bayesian.py",
    "agents_api/dsw/stats_exploratory.py",
    "agents_api/dsw/stats_advanced.py",
    "api/internal_views.py",
    "syn/audit/compliance.py",
}

_ARCH_FILES_PER_APP_WARN = 20
_ARCH_FILES_PER_APP_FAIL = 30
# Known apps exceeding file count limits — tracked in DEBT.md
_ARCH_KNOWN_LARGE_APPS = {"agents_api"}

# Directory naming pattern
_DIR_SNAKE_RE = _re.compile(r"^[a-z][a-z0-9_]*$")

# Known cross-app import exemptions (documented in ARCH-001 §5.2)
_ARCH_CROSS_IMPORT_EXEMPT_FILES = {
    "management",  # Management commands may cross boundaries
    "tests",  # Test files may import from any layer
}

# Known core/ layer boundary exemptions
_ARCH_CORE_LAYER_EXEMPT = {
    "seed_nlp_demo.py",  # Management command — crosses boundaries by design
    "views.py",  # core/views.py project detail aggregates agents_api models (existing debt)
}

# Known timestamp fields that don't follow _at convention (tracked as debt)
_ARCH_KNOWN_TIMESTAMP_EXCEPTIONS = {
    "action_time",  # Django LogEntry (built-in)
    "expire_date",  # Django Session (built-in)
    "date_joined",  # Django User (built-in)
    "last_login",  # Django User (built-in)
    "current_period_start",  # Billing interval boundary
    "current_period_end",  # Billing interval boundary
    "period_start",  # Usage interval boundary
    "period_end",  # Usage interval boundary
    "scheduled_for",  # Onboarding scheduling
    "last_accessed",  # Cache access timestamp
    "last_seen",  # Presence tracking
    "last_run",  # Scheduler last execution
    "bucket_start",  # Metrics time bucket
    "timestamp",  # Audit log (SysLogEntry, ChangeLog, LogEntry legacy)
    "deadline",  # Task deadline
}


def _check_arch_dir_naming(web_root):
    """ARCH-001/STY-001 §4.1: Directories use lowercase_snake naming."""
    violations = []
    for d in sorted(web_root.rglob("*")):
        if not d.is_dir():
            continue
        if any(s in d.parts for s in _STYLE_SKIP_DIRS):
            continue
        # Skip dotdirs and all their children
        if d.name.startswith(".") or any(p.startswith(".") for p in d.relative_to(web_root).parts):
            continue
        if not _DIR_SNAKE_RE.match(d.name):
            violations.append(str(d.relative_to(web_root)))
    return violations


def _check_arch_nested_duplicates(web_root):
    """ARCH-001 §10: No nested duplicate directories (e.g., agents/agents/)."""
    duplicates = []
    for d in sorted(web_root.rglob("*")):
        if not d.is_dir():
            continue
        if any(s in d.parts for s in _STYLE_SKIP_DIRS):
            continue
        if d.name == d.parent.name and d.parent != web_root:
            duplicates.append(str(d.relative_to(web_root)))
    return duplicates


def _check_arch_files_per_app(web_root):
    """ARCH-001 §7: Files per app growth boundaries."""
    oversized = []
    app_dirs = [
        "accounts",
        "agents_api",
        "api",
        "chat",
        "core",
        "files",
        "forge",
        "inference",
        "workbench",
    ]
    for app in app_dirs:
        app_dir = web_root / app
        if not app_dir.is_dir():
            continue
        count = sum(1 for f in app_dir.rglob("*.py") if "__pycache__" not in f.parts and "migrations" not in f.parts)
        if count > _ARCH_FILES_PER_APP_WARN:
            severity = "fail" if count > _ARCH_FILES_PER_APP_FAIL else "warning"
            if app in _ARCH_KNOWN_LARGE_APPS:
                severity = "warning"  # Known debt
            oversized.append({"app": app, "files": count, "severity": severity})
    return oversized


def _check_arch_core_layer(web_root):
    """ARCH-001 §5: core/ must not import from feature apps."""
    violations = []
    core_dir = web_root / "core"
    if not core_dir.is_dir():
        return violations
    for py_file in core_dir.rglob("*.py"):
        if "__pycache__" in py_file.parts or "migrations" in py_file.parts:
            continue
        if "tests" in py_file.parts or py_file.name.startswith("test_"):
            continue
        if py_file.name in _ARCH_CORE_LAYER_EXEMPT:
            continue
        try:
            source = py_file.read_text(errors="ignore")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in _ARCH_FEATURE_APPS:
                        violations.append(
                            {
                                "file": str(py_file.relative_to(web_root)),
                                "line": node.lineno,
                                "import": alias.name,
                            }
                        )
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in _ARCH_FEATURE_APPS:
                    violations.append(
                        {
                            "file": str(py_file.relative_to(web_root)),
                            "line": node.lineno,
                            "import": node.module,
                        }
                    )
    return violations


def _check_arch_cross_imports(web_root):
    """ARCH-001 §5: Feature apps should not import from each other."""
    violations = []
    for app in _ARCH_FEATURE_APPS:
        app_dir = web_root / app
        if not app_dir.is_dir():
            continue
        other_apps = _ARCH_FEATURE_APPS - {app}
        for py_file in app_dir.rglob("*.py"):
            if "__pycache__" in py_file.parts or "migrations" in py_file.parts:
                continue
            if "tests" in py_file.parts or py_file.name.startswith("test_"):
                continue
            if "management" in py_file.parts:
                continue
            try:
                source = py_file.read_text(errors="ignore")
                tree = ast.parse(source, filename=str(py_file))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        target = alias.name.split(".")[0]
                        if target in other_apps:
                            violations.append(
                                {
                                    "file": str(py_file.relative_to(web_root)),
                                    "line": node.lineno,
                                    "import": alias.name,
                                    "from_app": app,
                                    "to_app": target,
                                }
                            )
                elif isinstance(node, ast.ImportFrom) and node.module:
                    target = node.module.split(".")[0]
                    if target in other_apps:
                        violations.append(
                            {
                                "file": str(py_file.relative_to(web_root)),
                                "line": node.lineno,
                                "import": node.module,
                                "from_app": app,
                                "to_app": target,
                            }
                        )
    return violations


# Django built-in URL segments that use underscores (can't be changed)
_URL_KEBAB_EXEMPT_SEGMENTS = {
    "password_reset",  # Django auth views
    "password_change",  # Django auth views
}


def _check_arch_url_kebab_case(web_root):
    """STY-001 §4.1: URL path segments use kebab-case (no underscores)."""
    violations = []
    for urls_file in web_root.rglob("urls.py"):
        if "__pycache__" in urls_file.parts or "migrations" in urls_file.parts:
            continue
        try:
            source = urls_file.read_text(errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(source.splitlines(), 1):
            # Match path("segment_with_underscore/", ...) patterns
            match = _re.search(r'path\(["\']([^"\']+)["\']', line)
            if not match:
                continue
            url_path = match.group(1)
            # Strip path parameters
            cleaned = _re.sub(r"<[^>]+>", "", url_path)
            segments = [s for s in cleaned.split("/") if s]
            for seg in segments:
                if "_" in seg and seg not in _URL_KEBAB_EXEMPT_SEGMENTS:
                    violations.append(
                        {
                            "file": str(urls_file.relative_to(web_root)),
                            "line": i,
                            "url": url_path,
                            "segment": seg,
                        }
                    )
                    break
    return violations


def _check_arch_timestamp_naming(web_root):
    """STY-001 §4.4: DateTimeField names should end in _at suffix."""
    violations = []
    for py_file in web_root.rglob("*.py"):
        if _should_skip_path(py_file):
            continue
        if py_file.name != "models.py" and "models" not in py_file.parent.name:
            continue
        try:
            source = py_file.read_text(errors="ignore")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in ast.walk(node):
                if not isinstance(item, ast.Assign):
                    continue
                for target in item.targets:
                    if not isinstance(target, ast.Name):
                        continue
                    if not isinstance(item.value, ast.Call):
                        continue
                    func = item.value.func
                    func_name = ""
                    if isinstance(func, ast.Attribute):
                        func_name = func.attr
                    elif isinstance(func, ast.Name):
                        func_name = func.id
                    if func_name == "DateTimeField":
                        field_name = target.id
                        if field_name in _ARCH_KNOWN_TIMESTAMP_EXCEPTIONS:
                            continue
                        if not field_name.endswith("_at"):
                            violations.append(
                                {
                                    "file": str(py_file.relative_to(web_root)),
                                    "class": node.name,
                                    "field": field_name,
                                }
                            )
    return violations


def _scan_class_docstrings(web_root):
    """STY-001 §6.2: All classes should have docstrings."""
    violations = []
    for py_file in web_root.rglob("*.py"):
        if _should_skip_path(py_file):
            continue
        name = py_file.name
        # Skip test files, admin, management commands — docstrings not required
        if (
            name.startswith("test_")
            or name.startswith("tests")
            or name.endswith("_tests.py")
            or "tests" in py_file.parts
        ):
            continue
        if name == "admin.py" or name == "apps.py" or "management" in py_file.parts:
            continue
        try:
            source = py_file.read_text(errors="ignore")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                docstring = ast.get_docstring(node)
                if not docstring:
                    # Skip @dataclass classes — fields are self-documenting
                    if any(
                        (isinstance(d, ast.Name) and d.id == "dataclass")
                        or (isinstance(d, ast.Attribute) and d.attr == "dataclass")
                        for d in node.decorator_list
                    ):
                        continue
                    # Skip serializer subclasses — Meta class or parent is self-documenting
                    if any(
                        (isinstance(b, ast.Attribute) and "Serializer" in getattr(b, "attr", ""))
                        or (isinstance(b, ast.Name) and "Serializer" in b.id)
                        for b in node.bases
                    ):
                        continue
                    # Skip Django TextChoices/IntegerChoices enums — choices are self-documenting
                    if any(
                        (isinstance(b, ast.Attribute) and b.attr in ("TextChoices", "IntegerChoices"))
                        or (isinstance(b, ast.Name) and b.id in ("TextChoices", "IntegerChoices"))
                        for b in node.bases
                    ):
                        continue
                    # Skip Sitemap subclasses — attrs are self-documenting
                    if any((isinstance(b, ast.Name) and b.id == "Sitemap") for b in node.bases):
                        continue
                    violations.append(
                        {
                            "file": str(py_file.relative_to(web_root)),
                            "class": node.name,
                            "line": node.lineno,
                        }
                    )
    return violations


# Test file detection patterns
_TEST_FILE_RE = _re.compile(r"^test_[a-z][a-z0-9_]*\.py$")
_TEST_FILE_LEGACY_RE = _re.compile(r"^[a-z][a-z0-9_]*_tests\.py$")


def _check_arch_test_placement(web_root):
    """ARCH-001 §6: Test files must live in tests/ directories."""
    violations = []
    for py_file in web_root.rglob("*.py"):
        if "__pycache__" in py_file.parts or "migrations" in py_file.parts:
            continue
        if "staticfiles" in py_file.parts or "media" in py_file.parts:
            continue
        name = py_file.name
        # Is this a test file?
        is_test = _TEST_FILE_RE.match(name) or _TEST_FILE_LEGACY_RE.match(name) or name == "tests.py"
        if not is_test:
            continue
        # Must be inside a tests/ directory
        if "tests" in py_file.relative_to(web_root).parts:
            continue
        violations.append(str(py_file.relative_to(web_root)))
    return violations


def _check_arch_test_init(web_root):
    """ARCH-001 §6: tests/ directories must have __init__.py for discovery."""
    missing = []
    for tests_dir in web_root.rglob("tests"):
        if not tests_dir.is_dir():
            continue
        if "__pycache__" in tests_dir.parts or "migrations" in tests_dir.parts:
            continue
        if "staticfiles" in tests_dir.parts or "media" in tests_dir.parts:
            continue
        # Must have at least one .py file to count as a test package
        py_files = list(tests_dir.glob("*.py"))
        if not py_files:
            continue
        if not (tests_dir / "__init__.py").exists():
            missing.append(str(tests_dir.relative_to(web_root)))
    return missing


def _check_arch_testcase_in_prod(web_root):
    """ARCH-001 §6: TestCase subclasses must not appear in non-test files."""
    violations = []
    for py_file in web_root.rglob("*.py"):
        if _should_skip_path(py_file):
            continue
        name = py_file.name
        # Skip actual test files
        if name.startswith("test_") or name.startswith("tests_") or name.endswith("_tests.py") or name == "tests.py":
            continue
        if "tests" in py_file.parts:
            continue
        try:
            source = py_file.read_text(errors="ignore")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for base in node.bases:
                base_name = ""
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if base_name in ("TestCase", "SimpleTestCase", "TransactionTestCase", "LiveServerTestCase"):
                    violations.append(
                        {
                            "file": str(py_file.relative_to(web_root)),
                            "class": node.name,
                            "line": node.lineno,
                            "base": base_name,
                        }
                    )
    return violations


def _check_arch_required_dirs(web_root):
    """ARCH-001 §4: Verify required directories exist."""
    missing = []
    for d in _ARCH_REQUIRED_DIRS:
        if not (web_root / d).is_dir():
            missing.append(d)
    return missing


def _check_arch_prohibited_dirs(web_root):
    """ARCH-001 §8: No prohibited directories."""
    found = []
    for d in _ARCH_PROHIBITED_DIRS:
        if (web_root / d).is_dir():
            found.append(d)
    return found


def _check_arch_empty_dirs(web_root):
    """ARCH-001 §8: No empty directories."""
    empty = []
    for dirpath in sorted(web_root.rglob("*")):
        if not dirpath.is_dir():
            continue
        if any(skip in dirpath.parts for skip in _ARCH_EMPTY_DIR_SKIP):
            continue
        if dirpath.name.startswith("__"):
            continue
        children = list(dirpath.iterdir())
        if not children:
            empty.append(str(dirpath.relative_to(web_root)))
    return empty


def _check_arch_layer_boundaries(web_root):
    """ARCH-001 §5: syn/ must not import from feature apps."""
    violations = []
    syn_dir = web_root / "syn"
    for py_file in syn_dir.rglob("*.py"):
        if "__pycache__" in py_file.parts or "migrations" in py_file.parts:
            continue
        # Exempt compliance.py, test files, and svend_tasks.py (they introspect all layers)
        if py_file.name == "compliance.py" and "audit" in py_file.parts:
            continue
        if py_file.name == "svend_tasks.py" and "sched" in py_file.parts:
            continue
        if "tests" in py_file.parts or py_file.name.startswith("test_"):
            continue
        if "management" in py_file.parts:
            continue
        try:
            source = py_file.read_text(errors="ignore")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod_root = alias.name.split(".")[0]
                    if mod_root in _ARCH_FEATURE_APPS:
                        violations.append(
                            {
                                "file": str(py_file.relative_to(web_root)),
                                "line": node.lineno,
                                "import": alias.name,
                            }
                        )
            elif isinstance(node, ast.ImportFrom) and node.module:
                mod_root = node.module.split(".")[0]
                if mod_root in _ARCH_FEATURE_APPS:
                    violations.append(
                        {
                            "file": str(py_file.relative_to(web_root)),
                            "line": node.lineno,
                            "import": node.module,
                        }
                    )
    return violations


def _check_arch_file_sizes(web_root):
    """ARCH-001 §7: File size growth boundaries."""
    oversized = []
    for py_file in web_root.rglob("*.py"):
        if "__pycache__" in py_file.parts or "migrations" in py_file.parts:
            continue
        if "staticfiles" in py_file.parts:
            continue
        try:
            line_count = sum(1 for _ in py_file.open(errors="ignore"))
        except OSError:
            continue
        if line_count > _ARCH_FILE_SIZE_WARN:
            rel_path = str(py_file.relative_to(web_root))
            if line_count > _ARCH_FILE_SIZE_FAIL and rel_path in _ARCH_KNOWN_LARGE_FILES:
                severity = "warning"  # Known debt — tracked in DEBT.md
            elif line_count > _ARCH_FILE_SIZE_FAIL:
                severity = "fail"
            else:
                severity = "warning"
            oversized.append(
                {
                    "file": rel_path,
                    "lines": line_count,
                    "severity": severity,
                }
            )
    return oversized


def _check_arch_root_files(web_root):
    """ARCH-001 §10: No unexpected files at web root."""
    unexpected = []
    for item in sorted(web_root.iterdir()):
        if item.is_dir():
            continue
        if item.name in _ARCH_ALLOWED_ROOT_FILES:
            continue
        if item.name.startswith("."):
            continue
        unexpected.append(item.name)
    return unexpected


@register("architecture", "processing_integrity", soc2_controls=["CC7.2", "CC8.1"])
def check_architecture():
    """ARCH-001: Architecture & structure enforcement.

    Validates canonical directory structure, prohibited directories,
    layer boundaries, file size limits, root file hygiene,
    directory naming, nested duplicates, files per app, and cross-imports.

    SOC 2: CC7.2 (System Component Inventory)
    ISO 27001: A.8.1 (Asset Inventory)
    """
    web_root = Path(settings.BASE_DIR)

    missing_dirs = _check_arch_required_dirs(web_root)
    prohibited_dirs = _check_arch_prohibited_dirs(web_root)
    empty_dirs = _check_arch_empty_dirs(web_root)
    layer_violations = _check_arch_layer_boundaries(web_root)
    oversized_files = _check_arch_file_sizes(web_root)
    unexpected_root = _check_arch_root_files(web_root)
    dir_naming = _check_arch_dir_naming(web_root)
    nested_dupes = _check_arch_nested_duplicates(web_root)
    files_per_app = _check_arch_files_per_app(web_root)
    core_layer = _check_arch_core_layer(web_root)
    cross_imports = _check_arch_cross_imports(web_root)
    test_placement = _check_arch_test_placement(web_root)
    test_init = _check_arch_test_init(web_root)
    testcase_in_prod = _check_arch_testcase_in_prod(web_root)

    hard_fail = (
        missing_dirs
        or prohibited_dirs
        or layer_violations
        or dir_naming
        or nested_dupes
        or core_layer
        or test_init
        or testcase_in_prod
        or any(f["severity"] == "fail" for f in oversized_files)
        or any(f["severity"] == "fail" for f in files_per_app)
    )
    soft_warn = (
        empty_dirs
        or unexpected_root
        or cross_imports
        or test_placement
        or any(f["severity"] == "warning" for f in oversized_files)
        or any(f["severity"] == "warning" for f in files_per_app)
    )

    if hard_fail:
        status = "fail"
    elif soft_warn:
        status = "warning"
    else:
        status = "pass"

    return {
        "status": status,
        "details": {
            "missing_required_dirs": missing_dirs,
            "prohibited_dirs_found": prohibited_dirs,
            "empty_dirs": empty_dirs[:20],
            "layer_boundary_violations": layer_violations[:20],
            "oversized_files": oversized_files[:20],
            "unexpected_root_files": unexpected_root,
            "dir_naming_violations": dir_naming[:20],
            "nested_duplicate_dirs": nested_dupes[:20],
            "files_per_app": files_per_app,
            "core_layer_violations": core_layer[:20],
            "cross_import_violations": cross_imports[:20],
            "test_placement_violations": test_placement[:20],
            "test_init_missing": test_init,
            "testcase_in_prod_code": testcase_in_prod[:20],
        },
        "soc2_controls": ["CC7.2", "CC8.1"],
    }


# ---------------------------------------------------------------------------
# Caching Check (CACHE-001)
# ---------------------------------------------------------------------------

_CACHE_REQUIRED_MIDDLEWARE = "accounts.middleware.NoCacheDynamicMiddleware"
_CACHE_WHITENOISE_MIDDLEWARE = "whitenoise.middleware.WhiteNoiseMiddleware"
_CACHE_WHITENOISE_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

# In-memory caches that must have max-size constants
_CACHE_MEMORY_BOUNDS = {
    "agents_api/dsw/common.py": "MODEL_CACHE_MAX_SIZE",
    "agents_api/problem_views.py": "_INTERVIEW_CACHE_MAX",
    "agents_api/spc_views.py": "_CACHE_MAX_SIZE",
    "agents_api/synara_views.py": "_SYNARA_CACHE_MAX",
}

# CDN version pin pattern: @major.minor or /major.minor.patch/ (cdnjs format)
_CDN_VERSION_RE = _re.compile(r"@\d+\.\d+|/\d+\.\d+\.\d+/")


def _check_cache_middleware(web_root):
    """CACHE-001 §4: NoCacheDynamicMiddleware in MIDDLEWARE."""
    middleware = getattr(settings, "MIDDLEWARE", [])
    issues = []
    if _CACHE_REQUIRED_MIDDLEWARE not in middleware:
        issues.append("NoCacheDynamicMiddleware not in MIDDLEWARE")
    if _CACHE_WHITENOISE_MIDDLEWARE not in middleware:
        issues.append("WhiteNoiseMiddleware not in MIDDLEWARE")
    else:
        # Check ordering: WhiteNoise must come before NoCacheDynamic
        wn_idx = middleware.index(_CACHE_WHITENOISE_MIDDLEWARE)
        if _CACHE_REQUIRED_MIDDLEWARE in middleware:
            nc_idx = middleware.index(_CACHE_REQUIRED_MIDDLEWARE)
            if wn_idx > nc_idx:
                issues.append("WhiteNoiseMiddleware must be before NoCacheDynamicMiddleware")
    return issues


def _check_cache_whitenoise_storage():
    """CACHE-001 §4.2: WhiteNoise uses CompressedManifestStaticFilesStorage."""
    storages = getattr(settings, "STORAGES", {})
    backend = storages.get("staticfiles", {}).get("BACKEND", "")
    if backend != _CACHE_WHITENOISE_STORAGE:
        return [f"staticfiles backend is '{backend}', expected '{_CACHE_WHITENOISE_STORAGE}'"]
    return []


def _check_cache_memory_bounds(web_root):
    """CACHE-001 §6: In-memory caches have max size bounds."""
    missing = []
    for rel_path, constant_name in _CACHE_MEMORY_BOUNDS.items():
        full_path = web_root / rel_path
        if not full_path.exists():
            continue
        source = full_path.read_text(errors="ignore")
        if constant_name not in source:
            missing.append({"file": rel_path, "expected_constant": constant_name})
    return missing


def _check_cache_cdn_versions(web_root):
    """CACHE-001 §8: CDN script/link tags have version pins."""
    violations = []
    templates_dir = web_root / "templates"
    if not templates_dir.is_dir():
        return violations
    cdn_pattern = _re.compile(
        r'(?:src|href)=["\']https?://(?:cdn\.jsdelivr\.net|unpkg\.com|cdnjs\.cloudflare\.com)/([^"\']+)["\']'
    )
    for html_file in templates_dir.rglob("*.html"):
        source = html_file.read_text(errors="ignore")
        for i, line in enumerate(source.splitlines(), 1):
            for match in cdn_pattern.finditer(line):
                url_path = match.group(1)
                if not _CDN_VERSION_RE.search(url_path):
                    violations.append(
                        {
                            "file": str(html_file.relative_to(web_root)),
                            "line": i,
                            "url": match.group(0),
                        }
                    )
    return violations


def _check_cache_idempotency_ttl():
    """CACHE-001 §5: Idempotency cache has bounded TTL."""
    try:
        from syn.api.middleware import IDEMPOTENCY_TTL_HOURS

        if IDEMPOTENCY_TTL_HOURS > 48:
            return [f"IDEMPOTENCY_TTL_HOURS={IDEMPOTENCY_TTL_HOURS} exceeds 48h maximum"]
    except ImportError:
        return ["Cannot import IDEMPOTENCY_TTL_HOURS from syn.api.middleware"]
    return []


@register("caching", "security", soc2_controls=["CC6.1", "CC7.2"])
def check_caching():
    """CACHE-001: Caching patterns & HTTP cache control.

    Validates middleware presence, static file storage backend,
    in-memory cache bounds, CDN version pins, and idempotency TTL.

    SOC 2: CC6.1 (Logical Access Security)
    OWASP: Cache Poisoning Prevention
    """
    web_root = Path(settings.BASE_DIR)

    middleware_issues = _check_cache_middleware(web_root)
    storage_issues = _check_cache_whitenoise_storage()
    memory_issues = _check_cache_memory_bounds(web_root)
    cdn_issues = _check_cache_cdn_versions(web_root)
    idempotency_issues = _check_cache_idempotency_ttl()

    hard_fail = middleware_issues or storage_issues or memory_issues
    soft_warn = cdn_issues or idempotency_issues

    if hard_fail:
        status = "fail"
    elif soft_warn:
        status = "warning"
    else:
        status = "pass"

    return {
        "status": status,
        "details": {
            "middleware_issues": middleware_issues,
            "storage_issues": storage_issues,
            "memory_bound_issues": memory_issues,
            "cdn_version_issues": cdn_issues[:20],
            "idempotency_issues": idempotency_issues,
        },
        "soc2_controls": ["CC6.1", "CC7.2"],
    }


@register("roadmap", "processing_integrity", soc2_controls=["CC9.1"])
def check_roadmap():
    """Verify roadmap hygiene per RDM-001."""
    from api.models import RoadmapItem

    issues = []
    now = timezone.now()
    month = now.month
    year = now.year
    q_num = (month - 1) // 3 + 1
    current_q = f"Q{q_num}-{year}"

    if q_num == 4:
        next_q = f"Q1-{year + 1}"
    else:
        next_q = f"Q{q_num + 1}-{year}"

    def _quarter_to_tuple(q_str):
        try:
            parts = q_str.split("-")
            return (int(parts[1]), int(parts[0][1]))
        except (IndexError, ValueError):
            return (9999, 9)

    current_tuple = _quarter_to_tuple(current_q)

    # Check 1: Stale items — past-quarter items still "planned"
    stale = []
    for item in RoadmapItem.objects.filter(status="planned"):
        if _quarter_to_tuple(item.quarter) < current_tuple:
            stale.append(f"{item.quarter}: {item.title}")
    if stale:
        issues.append(f"{len(stale)} item(s) in past quarters still 'planned': {', '.join(stale[:3])}")

    # Check 2: Shipped items missing shipped_at
    missing_shipped = RoadmapItem.objects.filter(status="shipped", shipped_at__isnull=True).count()
    if missing_shipped:
        issues.append(f"{missing_shipped} shipped item(s) missing shipped_at timestamp")

    # Check 3: Empty upcoming quarter
    upcoming_count = RoadmapItem.objects.filter(quarter=next_q).count()
    if upcoming_count == 0:
        issues.append(f"No roadmap items for upcoming quarter {next_q}")

    total = RoadmapItem.objects.count()
    public = RoadmapItem.objects.filter(is_public=True).count()

    status = "pass"
    if stale:
        status = "fail"
    elif issues:
        status = "warning"

    return {
        "status": status,
        "details": {
            "total_items": total,
            "public_items": public,
            "current_quarter": current_q,
            "next_quarter": next_q,
            "issues": issues,
        },
        "soc2_controls": ["CC9.1"],
    }


@register("symbol_coverage", "processing_integrity", soc2_controls=["CC8.1", "CC4.1"])
def check_symbol_coverage():
    """Non-gameable symbol-level governance metric.

    AST-walks all production .py files to inventory every public top-level
    function and class.  Only symbol-level <!-- impl: file.py:Symbol --> hooks
    count — file-level hooks are explicitly skipped.  Each symbol is classified:

      covered           — impl hook on assertion that also has test hooks
      specified_untested — impl hook but assertion lacks test hooks
      ungoverned        — no impl hook references this symbol

    Risk score per file = ungoverned_loc * 1.0 + specified_untested_loc * 0.5
    """
    import ast
    import os

    from syn.audit.standards import parse_all_standards

    issues = []
    web_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # ------------------------------------------------------------------
    # Step 1: Build symbol inventory from production .py files via AST
    # ------------------------------------------------------------------
    skip_dirs = {"migrations", "__pycache__", ".git", "node_modules", ".venv"}
    all_symbols = {}  # "rel_path:Name" -> {kind, loc, file, name}

    def _loc_range(filepath, start, end):
        """Count non-blank, non-comment lines within a line range."""
        try:
            with open(filepath) as fh:
                lines = fh.readlines()
            count = 0
            for i in range(start - 1, min(end, len(lines))):
                s = lines[i].strip()
                if s and not s.startswith("#"):
                    count += 1
            return count
        except Exception:
            return 0

    for root, dirs, files in os.walk(web_root):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for f in files:
            if not f.endswith(".py"):
                continue
            if f.startswith("__init__"):
                continue
            if "test" in f.lower():
                continue
            full_path = os.path.join(root, f)
            rel = os.path.relpath(full_path, web_root)

            try:
                with open(full_path) as fh:
                    source = fh.read()
                tree = ast.parse(source)
            except (SyntaxError, UnicodeDecodeError, OSError):
                continue

            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    if node.name.startswith("_"):
                        continue
                    loc = _loc_range(full_path, node.lineno, node.end_lineno)
                    if loc > 0:
                        all_symbols[f"{rel}:{node.name}"] = {
                            "kind": "class",
                            "loc": loc,
                            "file": rel,
                            "name": node.name,
                        }
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("_"):
                        continue
                    loc = _loc_range(full_path, node.lineno, node.end_lineno)
                    if loc > 0:
                        all_symbols[f"{rel}:{node.name}"] = {
                            "kind": "function",
                            "loc": loc,
                            "file": rel,
                            "name": node.name,
                        }

    # ------------------------------------------------------------------
    # Step 2: Map symbol-level impl hooks to assertions
    # ------------------------------------------------------------------
    assertions = parse_all_standards()
    governed = set()  # keys with impl hook (may lack tests)
    covered_set = set()  # keys where assertion also has test hooks

    for a in assertions:
        has_tests = len(a.tests) > 0
        for impl_ref in a.impls:
            impl_ref = impl_ref.strip()
            if ":" not in impl_ref:
                continue  # file-level hook — skip entirely
            file_part, symbol_part = impl_ref.split(":", 1)
            # Class.method hooks resolve to the class
            top_symbol = symbol_part.split(".")[0]
            key = f"{file_part}:{top_symbol}"

            governed.add(key)
            if has_tests:
                covered_set.add(key)

    # ------------------------------------------------------------------
    # Step 3: Classify every symbol into buckets
    # ------------------------------------------------------------------
    covered = {}
    specified = {}
    ungoverned = {}

    for key, info in all_symbols.items():
        if key in covered_set:
            covered[key] = info
        elif key in governed:
            specified[key] = info
        else:
            ungoverned[key] = info

    # ------------------------------------------------------------------
    # Step 4: Aggregate by file and module
    # ------------------------------------------------------------------
    total_symbols = len(all_symbols)
    total_loc = sum(s["loc"] for s in all_symbols.values())
    covered_loc = sum(s["loc"] for s in covered.values())
    specified_loc = sum(s["loc"] for s in specified.values())
    ungoverned_loc = sum(s["loc"] for s in ungoverned.values())

    covered_pct = round(len(covered) / total_symbols * 100, 1) if total_symbols else 0
    covered_loc_pct = round(covered_loc / total_loc * 100, 1) if total_loc else 0

    # Per-file risk
    file_risk = {}
    for key, info in all_symbols.items():
        f = info["file"]
        if f not in file_risk:
            file_risk[f] = {
                "ungoverned_loc": 0,
                "specified_untested_loc": 0,
                "covered_loc": 0,
                "total_loc": 0,
                "total_symbols": 0,
                "covered_symbols": 0,
                "specified_symbols": 0,
                "ungoverned_symbols": 0,
            }
        fr = file_risk[f]
        fr["total_loc"] += info["loc"]
        fr["total_symbols"] += 1
        if key in covered:
            fr["covered_loc"] += info["loc"]
            fr["covered_symbols"] += 1
        elif key in specified:
            fr["specified_untested_loc"] += info["loc"]
            fr["specified_symbols"] += 1
        else:
            fr["ungoverned_loc"] += info["loc"]
            fr["ungoverned_symbols"] += 1

    for fr in file_risk.values():
        fr["risk_score"] = round(fr["ungoverned_loc"] * 1.0 + fr["specified_untested_loc"] * 0.5, 1)

    top_risk = sorted(
        [{"file": f, **fr} for f, fr in file_risk.items()],
        key=lambda x: -x["risk_score"],
    )[:20]

    # Per-module breakdown
    by_module = {}
    for f, fr in file_risk.items():
        module = f.split("/")[0] if "/" in f else "root"
        if module not in by_module:
            by_module[module] = {
                "total_loc": 0,
                "covered_loc": 0,
                "specified_untested_loc": 0,
                "ungoverned_loc": 0,
                "total_symbols": 0,
                "covered_symbols": 0,
                "specified_symbols": 0,
                "ungoverned_symbols": 0,
            }
        bm = by_module[module]
        for k in list(bm.keys()):
            bm[k] += fr.get(k, 0)

    module_list = []
    for mod in sorted(by_module.keys()):
        d = by_module[mod]
        d["module"] = mod
        d["covered_pct"] = round(d["covered_symbols"] / d["total_symbols"] * 100, 1) if d["total_symbols"] else 0
        module_list.append(d)

    total_risk = round(ungoverned_loc * 1.0 + specified_loc * 0.5, 1)

    # ------------------------------------------------------------------
    # Step 5: Status (percentage thresholds)
    # ------------------------------------------------------------------
    if covered_pct < 30:
        issues.append(f"Symbol coverage {covered_pct}% below 30% threshold")
        status = "fail"
    elif covered_pct < 50:
        issues.append(f"Symbol coverage {covered_pct}% below 50% target")
        status = "warning"
    else:
        status = "pass"

    return {
        "status": status,
        "details": {
            "total_symbols": total_symbols,
            "covered_symbols": len(covered),
            "specified_untested_symbols": len(specified),
            "ungoverned_symbols": len(ungoverned),
            "total_loc": total_loc,
            "covered_loc": covered_loc,
            "specified_untested_loc": specified_loc,
            "ungoverned_loc": ungoverned_loc,
            "covered_pct": covered_pct,
            "covered_loc_pct": covered_loc_pct,
            "risk_score": total_risk,
            "by_module": module_list,
            "top_risk": top_risk,
            "issues": issues,
        },
        "soc2_controls": ["CC8.1", "CC4.1"],
    }


# ---------------------------------------------------------------------------
# Statistical Calibration (STAT-001 §15)
# ---------------------------------------------------------------------------


@register("statistical_calibration", "processing_integrity", soc2_controls=["CC4.1"])
def check_statistical_calibration():
    """Run statistical calibration — feed known reference data through analysis
    functions and verify outputs within tolerance.

    Standard: STAT-001 §15
    Compliance: SOC 2 CC4.1, ISO 9001:2015 §8.5.1
    """
    issues = []
    try:
        from agents_api.calibration import run_calibration

        cal = run_calibration()
    except Exception as e:
        return {
            "status": "error",
            "details": {"error": str(e), "cases_run": 0, "pass_rate": 0},
            "soc2_controls": ["CC4.1"],
        }

    cases_run = cal["cases_run"]
    pass_rate = cal["pass_rate"]
    drift_cases = cal["drift_cases"]

    # Create DriftViolation for each failed case
    if drift_cases:
        try:
            from syn.audit.models import DriftViolation

            for case_id in drift_cases:
                # Find the case result for detail
                case_detail = next((r for r in cal["results"] if r["case_id"] == case_id), {})
                severity = "HIGH" if len(drift_cases) > 3 else "MEDIUM"
                DriftViolation.objects.create(
                    enforcement_check="CAL",
                    severity=severity,
                    file_path="agents_api/calibration.py",
                    description=f"Calibration case {case_id} failed: {case_detail.get('description', '')}",
                    expected_pattern=f"All calibration expectations pass for {case_id}",
                    actual_pattern=f"Failed checks: {[c for c in case_detail.get('checks', []) if not c.get('passed')]}",
                )
        except Exception as e:
            issues.append(f"Could not create DriftViolation: {e}")

    # Determine status
    if pass_rate >= 100:
        status = "pass"
    elif pass_rate >= 80:
        status = "warning"
        issues.append(f"{len(drift_cases)} calibration case(s) failed: {drift_cases}")
    else:
        status = "fail"
        issues.append(f"Calibration pass rate {pass_rate}% below 80% threshold")

    return {
        "status": status,
        "details": {
            "cases_run": cases_run,
            "cases_passed": cal["cases_passed"],
            "pass_rate": pass_rate,
            "seed": cal["seed"],
            "drift_cases": drift_cases,
            "results": cal["results"],
            "issues": issues,
        },
        "soc2_controls": ["CC4.1"],
    }


# ---------------------------------------------------------------------------
# Output Quality (QUAL-001)
# ---------------------------------------------------------------------------


@register("output_quality", "processing_integrity", soc2_controls=["CC4.1", "CC7.2"])
def check_output_quality():
    """Verify output quality infrastructure per QUAL-001.

    Validates that:
    - _validate_statistics_bounds exists in standardize.py
    - standardize_output is called in dispatch.py
    - REQUIRED_FIELDS has expected keys
    - Calibration pool has ≥15 cases across ≥5 categories
    - Bounded metrics list matches QUAL-001 §6.2 spec

    Standard: QUAL-001 §6, §10
    Compliance: SOC 2 CC4.1, CC7.2
    """
    import inspect

    issues = []
    status = "pass"

    # 1. Verify _validate_statistics_bounds exists in standardize.py
    try:
        from agents_api.dsw.standardize import _validate_statistics_bounds

        sig = inspect.signature(_validate_statistics_bounds)
        params = list(sig.parameters.keys())
        if "result" not in params:
            issues.append("_validate_statistics_bounds missing 'result' parameter")
    except ImportError:
        issues.append("Cannot import _validate_statistics_bounds from standardize.py")
        status = "fail"

    # 2. Verify standardize_output is called in dispatch.py
    try:
        import agents_api.dsw.dispatch as dispatch_mod

        source = inspect.getsource(dispatch_mod)
        if "standardize_output" not in source:
            issues.append("standardize_output not called in dispatch.py")
            status = "fail"
    except Exception as e:
        issues.append(f"Cannot inspect dispatch.py: {e}")

    # 3. Verify REQUIRED_FIELDS has expected keys
    try:
        from agents_api.dsw.standardize import REQUIRED_FIELDS

        expected_keys = {
            "summary",
            "plots",
            "narrative",
            "education",
            "diagnostics",
            "guide_observation",
            "evidence_grade",
            "bayesian_shadow",
            "what_if",
        }
        missing = expected_keys - set(REQUIRED_FIELDS.keys())
        if missing:
            issues.append(f"REQUIRED_FIELDS missing keys: {sorted(missing)}")
            status = "fail"
    except ImportError:
        issues.append("Cannot import REQUIRED_FIELDS from standardize.py")
        status = "fail"

    # 4. Verify bounded metrics match QUAL-001 §6.2
    try:
        from agents_api.dsw.standardize import (
            _BOUNDED_METRICS,
            _FINITE_METRICS,
            _POSITIVE_METRICS,
        )

        # p_value must be bounded [0, 1]
        p_keys = [keys for keys, lo, hi in _BOUNDED_METRICS if "p_value" in keys]
        if not p_keys:
            issues.append("_BOUNDED_METRICS missing p_value entry")
            status = "fail"
        # bf10 must be positive
        if "bf10" not in _POSITIVE_METRICS:
            issues.append("_POSITIVE_METRICS missing bf10")
            status = "fail"
        # cpk must be finite
        if "cpk" not in _FINITE_METRICS:
            issues.append("_FINITE_METRICS missing cpk")
            status = "fail"
    except ImportError:
        issues.append("Cannot import bounded metrics from standardize.py")
        status = "fail"

    # 5. Verify calibration pool size and category coverage
    try:
        from agents_api.calibration import REFERENCE_POOL

        pool_size = len(REFERENCE_POOL)
        categories = {c.category for c in REFERENCE_POOL}
        if pool_size < 15:
            issues.append(f"Calibration pool has {pool_size} cases, need ≥15")
            if status == "pass":
                status = "warning"
        if len(categories) < 5:
            issues.append(f"Calibration pool covers {len(categories)} categories ({sorted(categories)}), need ≥5")
            if status == "pass":
                status = "warning"
    except ImportError:
        issues.append("Cannot import REFERENCE_POOL from calibration.py")
        if status == "pass":
            status = "warning"

    return {
        "status": status,
        "details": {
            "issues": issues,
            "checks_performed": 5,
            "bounds_validation": status != "fail",
        },
        "soc2_controls": ["CC4.1", "CC7.2"],
    }


# ---------------------------------------------------------------------------
# Policy staleness detection
# ---------------------------------------------------------------------------


def _extract_policy_date(content):
    """Extract the most recent date from policy header (Last Updated or Effective Date)."""
    # Prefer Last Updated over Effective Date
    for pattern in [r"\*\*Last Updated:\*\*\s*(\d{4}-\d{2}-\d{2})", r"\*\*Effective Date:\*\*\s*(\d{4}-\d{2}-\d{2})"]:
        m = _re.search(pattern, content)
        if m:
            try:
                return date.fromisoformat(m.group(1))
            except ValueError:
                continue
    return None


def _extract_policy_watches(content):
    """Extract watched files from <!-- policy-watches: ... --> tags."""
    watches = []
    for m in _re.finditer(r"<!--\s*policy-watches:\s*(.+?)\s*-->", content):
        for entry in m.group(1).split(","):
            entry = entry.strip()
            if ":" in entry:
                fpath, symbol = entry.split(":", 1)
                watches.append({"file": fpath.strip(), "symbol": symbol.strip()})
            elif entry:
                watches.append({"file": entry, "symbol": ""})
    return watches


@register("policy_review", "processing_integrity", soc2_controls=["CC1.5", "CC5.3"])
def check_policy_review():
    """Detect policies that may be stale relative to code changes they depend on."""
    policy_dir = _GIT_ROOT / "docs" / "compliance" / "policies"
    web_dir = _GIT_ROOT / "services" / "svend" / "web"
    findings = []
    policies_checked = 0
    policies_with_watches = 0

    if not policy_dir.exists():
        return {
            "status": "warning",
            "details": {"message": "Policy directory not found", "policies_checked": 0},
        }

    for md in sorted(policy_dir.glob("*.md")):
        content = md.read_text()
        policies_checked += 1

        last_updated = _extract_policy_date(content)
        watches = _extract_policy_watches(content)

        if not watches:
            continue
        policies_with_watches += 1

        stale_watches = []
        for watch in watches:
            # Resolve watched file relative to web/ (code files) or repo root (docs)
            watched_path = web_dir / watch["file"]
            if not watched_path.exists():
                watched_path = _GIT_ROOT / watch["file"]
            if watched_path.exists():
                mtime = datetime.fromtimestamp(watched_path.stat().st_mtime).date()
                if last_updated and mtime > last_updated:
                    stale_watches.append(
                        {
                            "file": watch["file"],
                            "symbol": watch.get("symbol", ""),
                            "modified": mtime.isoformat(),
                        }
                    )

        if stale_watches:
            findings.append(
                {
                    "policy": md.name,
                    "last_updated": last_updated.isoformat() if last_updated else "unknown",
                    "stale_watches": stale_watches,
                    "message": f"{md.name}: {len(stale_watches)} watched file(s) changed since last update",
                }
            )

    status = "warning" if findings else "pass"
    return {
        "status": status,
        "details": {
            "policies_checked": policies_checked,
            "policies_with_watches": policies_with_watches,
            "stale_policies": len(findings),
            "findings": findings,
        },
    }


# ---------------------------------------------------------------------------
# Calibration & Verification (CAL-001)
# ---------------------------------------------------------------------------

# File size exemptions — files documented in DEBT.md as known large modules.
# Maps relative path suffix → DEBT.md priority for exemption.
_COMPLEXITY_EXEMPTIONS = {
    # Files >3000 lines — tracked in .kjerne/DEBT.md and CAL-001 §8.1
    "agents_api/dsw/stats_advanced.py": "P3",
    "agents_api/dsw/stats_exploratory.py": "P3",
    "agents_api/dsw/spc.py": "P3",
    "agents_api/dsw/bayesian.py": "P3",
    "agents_api/dsw/ml.py": "P3",
    "agents_api/dsw/common.py": "P3",
    "agents_api/learn_content.py": "P3",
    "agents_api/models.py": "P3",
    "agents_api/pbs_engine.py": "P3",
    "api/internal_views.py": "P3",
    "syn/audit/compliance.py": "P3",
}


@register("calibration_coverage", "processing_integrity", soc2_controls=["CC4.1", "CC7.2"])
def check_calibration_coverage():
    """CAL-001 §5/§6: Coverage measurement, golden files, ratchet.

    Validates that:
    - .coveragerc exists and is configured
    - coverage.json exists and is reasonably fresh
    - Coverage hasn't regressed below ratchet baseline
    - Golden file directory exists

    Standard: CAL-001 §5, §6
    Compliance: SOC 2 CC4.1, ISO/IEC 17025:2017
    """
    issues = []
    base = Path(settings.BASE_DIR)

    # 1. Check .coveragerc exists
    coveragerc = base / ".coveragerc"
    if not coveragerc.exists():
        issues.append(".coveragerc not found — create per CAL-001 §5.1")

    # 2. Check coverage.json freshness
    coverage_json = base / "coverage.json"
    has_coverage = False
    current_coverage = 0
    if not coverage_json.exists():
        issues.append("coverage.json not found — run: coverage run manage.py test --keepdb && coverage json")
    else:
        age_days = (time.time() - coverage_json.stat().st_mtime) / 86400
        if age_days > 14:
            issues.append(f"coverage.json is {age_days:.0f} days old (>14 days)")
        try:
            data = json.loads(coverage_json.read_text())
            current_coverage = data.get("totals", {}).get("percent_covered", 0)
            has_coverage = True
        except Exception as e:
            issues.append(f"coverage.json parse error: {e}")

    # 3. Ratchet baseline check
    try:
        from syn.audit.models import CalibrationReport

        last_report = CalibrationReport.objects.order_by("-date").first()
        if last_report and has_coverage:
            if current_coverage < last_report.ratchet_baseline:
                issues.append(
                    f"Coverage regression: {current_coverage:.1f}% < ratchet baseline {last_report.ratchet_baseline:.1f}%"
                )
    except Exception:
        pass  # Model may not exist yet during initial setup

    # 4. Golden file count
    golden_dir = base / "agents_api" / "tests" / "golden"
    golden_count = len(list(golden_dir.glob("*.json"))) if golden_dir.exists() else 0

    # Determine status
    has_ratchet_fail = any("regression" in i.lower() for i in issues)
    has_missing_rc = any(".coveragerc" in i for i in issues)

    if has_ratchet_fail:
        status = "fail"
    elif has_missing_rc:
        status = "fail"
    elif issues:
        status = "warning"
    else:
        status = "pass"

    return {
        "status": status,
        "details": {
            "coveragerc_exists": coveragerc.exists(),
            "coverage_measured": has_coverage,
            "current_coverage": current_coverage if has_coverage else None,
            "golden_file_count": golden_count,
            "issues": issues,
        },
        "soc2_controls": ["CC4.1", "CC7.2"],
    }


@register("complexity_governance", "processing_integrity", soc2_controls=["CC4.1"])
def check_complexity_governance():
    """CAL-001 §8: File size within budget.

    Checks all Python source files against the 3000-line limit from ARCH-001 §7.
    Files documented in DEBT.md are exempt.

    Standard: CAL-001 §8, ARCH-001 §7
    Compliance: SOC 2 CC4.1
    """
    base = Path(settings.BASE_DIR)
    violations = []
    warnings = []
    files_checked = 0

    for py_file in sorted(base.rglob("*.py")):
        rel = str(py_file.relative_to(base))
        # Skip non-source files
        if any(skip in rel for skip in ["/migrations/", "/test", "/__pycache__/", "/staticfiles/", "_tests.py"]):
            continue
        files_checked += 1

        try:
            line_count = sum(1 for _ in py_file.open())
        except Exception:
            continue

        # Check exemption
        is_exempt = any(rel.endswith(exempt_path) for exempt_path in _COMPLEXITY_EXEMPTIONS)

        if line_count > 3000 and not is_exempt:
            violations.append({"file": rel, "lines": line_count, "limit": 3000})
        elif line_count > 2000:
            warnings.append({"file": rel, "lines": line_count, "threshold": 2000})

    if violations:
        status = "fail"
    elif warnings:
        status = "warning"
    else:
        status = "pass"

    return {
        "status": status,
        "details": {
            "files_checked": files_checked,
            "violations": violations,
            "warnings": warnings,
            "exemptions": list(_COMPLEXITY_EXEMPTIONS.keys()),
        },
        "soc2_controls": ["CC4.1"],
    }


@register("endpoint_coverage", "processing_integrity", soc2_controls=["CC4.1", "CC7.2"])
def check_endpoint_coverage():
    """CAL-001 §7: Every view function has at least one test.

    Scans all *_views.py and views.py files for public view functions,
    then checks for corresponding test coverage.

    Standard: CAL-001 §7
    Compliance: SOC 2 CC4.1, CC7.2
    """
    base = Path(settings.BASE_DIR)
    view_files = sorted(set(list(base.rglob("*_views.py")) + list(base.rglob("views.py"))))

    total_endpoints = 0
    tested_endpoints = 0
    module_stats = []

    # Collect all test files for cross-reference
    test_files = list(base.rglob("test_*.py")) + list(base.rglob("*_tests.py"))
    test_content_cache = {}
    for tf in test_files:
        try:
            test_content_cache[str(tf)] = tf.read_text()
        except Exception:
            pass
    all_test_content = "\n".join(test_content_cache.values())

    for vf in view_files:
        rel = str(vf.relative_to(base))
        # Skip test files, migrations, staticfiles
        if any(skip in rel for skip in ["/test", "/migrations/", "/__pycache__/", "/staticfiles/"]):
            continue

        try:
            source = vf.read_text()
        except Exception:
            continue

        # Extract view function names — functions decorated with @require_auth, @gated, etc.
        # or functions that take 'request' as first parameter
        tree = ast.parse(source)
        view_funcs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                # Check if it has 'request' as first arg
                args = node.args
                if args.args and args.args[0].arg == "request":
                    view_funcs.append(node.name)

        endpoint_count = len(view_funcs)
        total_endpoints += endpoint_count

        # Check which view functions have tests
        tested = 0
        for fn_name in view_funcs:
            # Look for test methods that reference this function name
            if fn_name in all_test_content:
                tested += 1
        tested_endpoints += tested

        if endpoint_count > 0:
            module_stats.append(
                {
                    "file": rel,
                    "endpoints": endpoint_count,
                    "tested": tested,
                    "gap": endpoint_count - tested,
                }
            )

    coverage_pct = (tested_endpoints / total_endpoints * 100) if total_endpoints else 0

    return {
        "status": "pass",  # Informational in Phase 1 — no enforcement yet
        "details": {
            "total_endpoints": total_endpoints,
            "tested_endpoints": tested_endpoints,
            "coverage_pct": round(coverage_pct, 1),
            "modules": module_stats,
        },
        "soc2_controls": ["CC4.1", "CC7.2"],
    }
