"""
Celery tasks for syn.core app.

Includes automated secret rotation and key management tasks.

Standard:     CEL-001 §3-9 (Celery Task Patterns)
Compliance:   ISO 27001 A.10.1.2 (Key Management)
"""

import logging

from django.utils import timezone

try:
    from syn.synara.celery_compat import shared_task
except ImportError:
    # Celery compat layer not available in Svend — provide no-op decorator
    def shared_task(*args, **kwargs):
        def decorator(func):
            func.delay = lambda *a, **kw: None
            func.apply_async = lambda *a, **kw: None
            return func
        if args and callable(args[0]):
            return decorator(args[0])
        return decorator

from syn.core.secrets import SecretStore, rotate_secret

logger = logging.getLogger(__name__)


# CEL-001 §4.2: Task naming - domain.entity.action
@shared_task(
    name="core.secrets.rotate",  # CEL-001 §4.2: domain.entity.action naming
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=60,
    retry_backoff_max=3600,
    max_retries=3,
    soft_time_limit=300,
    time_limit=360,
    ignore_result=True,
)
def rotate_secrets_auto(self):
    """
    Automatically rotate secrets that are due for rotation.

    This task runs on a schedule (default: monthly) and rotates
    secrets based on their rotation_schedule_days setting.

    Returns:
        dict: Summary of rotation results
    """
    logger.info("Starting automatic secret rotation")

    rotated = 0
    failed = 0
    skipped = 0

    # Find all secrets that need rotation
    secrets = SecretStore.objects.all()

    for secret in secrets:
        try:
            if secret.needs_rotation():
                logger.info(
                    f"Rotating secret: {secret.name}",
                    extra={
                        "secret_name": secret.name,
                        "tenant_id": secret.tenant_id,
                        "last_rotated": secret.last_rotated_at,
                    },
                )

                # Rotate the secret (re-encrypts with new DEK)
                rotate_secret(secret.name, secret.tenant_id)
                rotated += 1

            else:
                skipped += 1

        except Exception as e:
            failed += 1
            logger.error(
                f"Failed to rotate secret: {secret.name}",
                extra={"secret_name": secret.name, "tenant_id": secret.tenant_id, "error": str(e)},
                exc_info=True,
            )

    result = {
        "rotated": rotated,
        "failed": failed,
        "skipped": skipped,
        "total": secrets.count(),
        "timestamp": timezone.now().isoformat(),
    }

    logger.info(f"Secret rotation complete: {rotated} rotated, {failed} failed, {skipped} skipped")

    return result


# CEL-001 §4.2: Task naming - domain.entity.action
@shared_task(
    name="core.secrets.check_expiration",  # CEL-001 §4.2: domain.entity.action naming
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=30,
    retry_backoff_max=300,
    max_retries=3,
    soft_time_limit=120,
    time_limit=150,
    ignore_result=True,
)
def check_secret_expiration(self):
    """
    Check for secrets that will expire soon and log warnings.

    This task helps administrators stay aware of upcoming
    rotation requirements.

    Returns:
        dict: Summary of expiring secrets
    """
    logger.info("Checking for expiring secrets")

    from datetime import timedelta

    # Find secrets expiring in the next 7 days
    warning_threshold = timezone.now() + timedelta(days=7)
    expiring_soon = []

    secrets = SecretStore.objects.all()

    for secret in secrets:
        if secret.last_rotated_at:
            next_rotation = secret.last_rotated_at + timedelta(days=secret.rotation_schedule_days)
        else:
            next_rotation = secret.created_at + timedelta(days=secret.rotation_schedule_days)

        if next_rotation <= warning_threshold:
            days_until = (next_rotation - timezone.now()).days
            expiring_soon.append(
                {"name": secret.name, "tenant_id": secret.tenant_id, "days_until_rotation": days_until}
            )

            if days_until <= 0:
                logger.warning(
                    f"Secret overdue for rotation: {secret.name}",
                    extra={"secret_name": secret.name, "tenant_id": secret.tenant_id, "days_overdue": abs(days_until)},
                )
            else:
                logger.info(
                    f"Secret expiring soon: {secret.name} ({days_until} days)",
                    extra={
                        "secret_name": secret.name,
                        "tenant_id": secret.tenant_id,
                        "days_until_rotation": days_until,
                    },
                )

    result = {"expiring_soon": expiring_soon, "count": len(expiring_soon), "timestamp": timezone.now().isoformat()}

    return result
