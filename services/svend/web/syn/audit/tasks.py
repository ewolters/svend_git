"""
Celery tasks for periodic audit log verification.

Standard:     CEL-001 §3-9 (Celery Task Patterns)
Compliance:   SOC 2 CC7.2 / ISO 27001 A.12.7
"""

import logging

from syn.synara.celery_compat import shared_task  # SCH-001: Celery removed

logger = logging.getLogger(__name__)


@shared_task(
    name="syn.audit.tasks.verify_audit_integrity",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=60,
    retry_backoff_max=900,
    max_retries=3,
    soft_time_limit=600,
    time_limit=660,
)
def verify_audit_integrity(self):
    """
    Periodic task to verify audit log integrity for all tenants.

    Checks the hash chain for each tenant and alerts on violations.
    Should be scheduled to run regularly (e.g., hourly or daily).

    Returns:
        Dictionary with verification results

    Schedule in Celery beat:
        CELERY_BEAT_SCHEDULE = {
            'verify-audit-integrity': {
                'task': 'audit.tasks.verify_audit_integrity',
                'schedule': crontab(minute=0, hour='*/6'),  # Every 6 hours
            },
        }

    Compliance:
    - SOC 2 CC7.2: Continuous audit log monitoring
    - ISO 27001 A.12.7: Automated integrity verification
    """
    from syn.audit.models import SysLogEntry
    from syn.audit.utils import record_integrity_violation, verify_chain_integrity

    try:
        # Get all unique tenant IDs
        tenant_ids = SysLogEntry.objects.values_list("tenant_id", flat=True).distinct()

        logger.info(f"Starting audit integrity verification for {len(tenant_ids)} tenants")

        violations_detected = 0
        tenants_verified = 0
        total_entries_checked = 0

        for tenant_id in tenant_ids:
            try:
                result = verify_chain_integrity(tenant_id)

                tenants_verified += 1
                total_entries_checked += result["total_entries"]

                if not result["is_valid"]:
                    # Record violations
                    for violation in result["violations"]:
                        try:
                            record_integrity_violation(
                                tenant_id=tenant_id,
                                violation_type=violation["type"],
                                entry_id=violation.get("entry_id"),
                                details=violation,
                            )
                            violations_detected += 1

                            logger.critical(
                                f"Audit integrity violation for tenant {tenant_id}: " f"{violation['type']}"
                            )

                        except Exception as e:
                            logger.error(f"Failed to record violation for tenant {tenant_id}: {e}")

            except Exception as e:
                logger.error(f"Failed to verify chain for tenant {tenant_id}: {e}", exc_info=True)

        # Log summary
        if violations_detected == 0:
            logger.info(
                f"Audit integrity verification complete: "
                f"{tenants_verified} tenants, {total_entries_checked} entries verified. "
                f"All chains intact."
            )
        else:
            logger.critical(
                f"Audit integrity verification found {violations_detected} violations "
                f"across {tenants_verified} tenants!"
            )

        return {
            "success": True,
            "tenants_verified": tenants_verified,
            "total_entries": total_entries_checked,
            "violations_detected": violations_detected,
        }

    except Exception as e:
        logger.error(f"Audit integrity verification failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }


@shared_task(
    name="syn.audit.tasks.cleanup_old_violations",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=30,
    retry_backoff_max=300,
    max_retries=3,
    soft_time_limit=180,
    time_limit=210,
    ignore_result=True,
)
def cleanup_old_violations(self, days: int = 90):
    """
    Clean up old resolved integrity violations.

    Removes resolved violations older than specified days.
    Keeps unresolved violations indefinitely.

    Args:
        days: Age threshold in days (default: 90)

    Returns:
        Number of violations deleted

    Compliance: SOC 2 CC7.2 - Data retention policy
    """
    from datetime import timedelta

    from django.utils import timezone

    from syn.audit.models import IntegrityViolation

    try:
        cutoff = timezone.now() - timedelta(days=days)

        # Delete only resolved violations
        deleted_count, _ = IntegrityViolation.objects.filter(is_resolved=True, resolved_at__lt=cutoff).delete()

        logger.info(f"Cleaned up {deleted_count} resolved integrity violations " f"older than {days} days")

        return deleted_count

    except Exception as e:
        logger.error(f"Violation cleanup failed: {e}", exc_info=True)
        return 0
