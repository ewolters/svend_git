"""Webhook delivery engine — NTF-001 §5.4.

Handles HTTP delivery, HMAC signing, retry scheduling, and circuit breaker.
All delivery runs async via syn.sched — never blocks request processing.

Compliance: SOC 2 CC6.1 (Logical Access Security)
"""

import logging
from datetime import timedelta

import requests
from django.utils import timezone

from .models import (
    WEBHOOK_RETRY_DELAYS,
    WebhookDelivery,
    WebhookEndpoint,
)

logger = logging.getLogger(__name__)

# Outbound request timeout (seconds)
_DELIVERY_TIMEOUT = 10

# User-Agent for webhook requests
_USER_AGENT = "Svend-Webhooks/1.0"


def dispatch_event(event_name, payload, *, tenant_id=None):
    """Find all matching endpoints and schedule delivery for each.

    Called from signal handlers or Cortex.publish() — this is the main
    entry point for the webhook system.

    Args:
        event_name: Dot-separated event name (e.g., "fmea.created")
        payload: Dict with event data
        tenant_id: UUID of the originating tenant. When provided, only
            endpoints belonging to that tenant (or with no tenant) are matched.
            This prevents cross-tenant event leakage (SOC 2 CC6.3).
    """
    qs = WebhookEndpoint.objects.filter(is_active=True)
    if tenant_id is not None:
        from django.db.models import Q

        qs = qs.filter(Q(tenant_id=tenant_id) | Q(tenant__isnull=True, user__isnull=False))
    endpoints = qs

    for endpoint in endpoints:
        if endpoint.matches_event(event_name):
            delivery = WebhookDelivery.objects.create(
                endpoint=endpoint,
                event_name=event_name,
                payload={
                    "event": event_name,
                    "timestamp": timezone.now().isoformat(),
                    "webhook_id": str(endpoint.id),
                    "data": payload,
                },
            )
            # Attempt immediate delivery
            deliver(delivery.id)


def deliver(delivery_id):
    """Attempt to deliver a single webhook event.

    Called immediately on first attempt, and by syn.sched for retries.
    """
    try:
        delivery = WebhookDelivery.objects.select_related("endpoint").get(id=delivery_id)
    except WebhookDelivery.DoesNotExist:
        logger.error("WebhookDelivery %s not found", delivery_id)
        return

    endpoint = delivery.endpoint

    if not endpoint.is_active:
        delivery.status = WebhookDelivery.Status.EXHAUSTED
        delivery.save(update_fields=["status"])
        return

    # Sign the payload
    signature = endpoint.sign_payload(delivery.payload)

    # Deliver
    delivery.attempt_count += 1
    try:
        resp = requests.post(
            endpoint.url,
            json=delivery.payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": _USER_AGENT,
                "X-Svend-Signature": signature,
                "X-Svend-Event": delivery.event_name,
                "X-Svend-Delivery": str(delivery.id),
            },
            timeout=_DELIVERY_TIMEOUT,
        )
        delivery.response_code = resp.status_code
        delivery.response_body = resp.text[:500]

        if 200 <= resp.status_code < 300:
            # Success
            delivery.status = WebhookDelivery.Status.DELIVERED
            delivery.delivered_at = timezone.now()
            delivery.next_retry_at = None
            endpoint.record_success()
            logger.info(
                "Webhook delivered: %s → %s (%d)",
                delivery.event_name,
                endpoint.url,
                resp.status_code,
            )
        else:
            _handle_failure(delivery, endpoint, f"HTTP {resp.status_code}")

    except requests.Timeout:
        delivery.response_code = None
        delivery.response_body = "Timeout after 10s"
        _handle_failure(delivery, endpoint, "timeout")

    except requests.RequestException as e:
        delivery.response_code = None
        delivery.response_body = str(e)[:500]
        _handle_failure(delivery, endpoint, str(e)[:100])

    delivery.save(
        update_fields=[
            "status",
            "response_code",
            "response_body",
            "attempt_count",
            "next_retry_at",
            "delivered_at",
        ]
    )


def _handle_failure(delivery, endpoint, reason):
    """Handle a failed delivery attempt — schedule retry or exhaust."""
    endpoint.record_failure()

    retry_index = delivery.attempt_count - 1  # 0-based
    if retry_index < len(WEBHOOK_RETRY_DELAYS):
        delay_seconds = WEBHOOK_RETRY_DELAYS[retry_index]
        delivery.status = WebhookDelivery.Status.FAILED
        delivery.next_retry_at = timezone.now() + timedelta(seconds=delay_seconds)
        logger.warning(
            "Webhook delivery failed (%s), retry in %ds: %s → %s",
            reason,
            delay_seconds,
            delivery.event_name,
            endpoint.url,
        )
    else:
        delivery.status = WebhookDelivery.Status.EXHAUSTED
        delivery.next_retry_at = None
        logger.error(
            "Webhook delivery exhausted after %d attempts: %s → %s",
            delivery.attempt_count,
            delivery.event_name,
            endpoint.url,
        )


def process_retries():
    """Process all pending webhook retries. Called by syn.sched.

    Finds deliveries where next_retry_at <= now and status == failed,
    then re-attempts delivery for each.
    """
    now = timezone.now()
    pending = WebhookDelivery.objects.filter(
        status=WebhookDelivery.Status.FAILED,
        next_retry_at__lte=now,
        endpoint__is_active=True,
    ).values_list("id", flat=True)[:100]  # Batch limit

    for delivery_id in pending:
        deliver(delivery_id)

    return len(pending)
