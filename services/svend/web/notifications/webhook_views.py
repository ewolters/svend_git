"""Webhook management endpoints — NTF-001 §5.4.

POST   /api/webhooks/                 — Create endpoint (returns secret once)
GET    /api/webhooks/                 — List user's endpoints
PUT    /api/webhooks/<id>/            — Update endpoint
DELETE /api/webhooks/<id>/            — Delete endpoint
POST   /api/webhooks/<id>/test/       — Send test event
GET    /api/webhooks/<id>/deliveries/ — Delivery log

Requires TEAM+ tier for all operations.
"""

import json
import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from accounts.permissions import require_team

from .models import WEBHOOK_ENDPOINT_LIMITS, WebhookDelivery, WebhookEndpoint

logger = logging.getLogger(__name__)


def _get_tier_limit(user):
    """Get webhook endpoint limit for user's tier."""
    return WEBHOOK_ENDPOINT_LIMITS.get(user.tier, 0)


def _get_user_endpoints(user):
    """Get endpoints owned by user (personal or via tenant)."""
    from agents_api.permissions import get_tenant

    tenant = get_tenant(user)
    if tenant:
        return WebhookEndpoint.objects.filter(tenant=tenant)
    return WebhookEndpoint.objects.filter(user=user, tenant__isnull=True)


@csrf_exempt
@require_team
@require_http_methods(["GET", "POST"])
def endpoint_list_create(request):
    """List or create webhook endpoints."""
    if request.method == "GET":
        return _list_endpoints(request)
    return _create_endpoint(request)


def _list_endpoints(request):
    endpoints = _get_user_endpoints(request.user).order_by("-created_at")
    return JsonResponse(
        {
            "endpoints": [
                {
                    "id": str(e.id),
                    "url": e.url,
                    "description": e.description,
                    "event_patterns": e.event_patterns,
                    "is_active": e.is_active,
                    "failure_count": e.failure_count,
                    "disabled_at": e.disabled_at.isoformat() if e.disabled_at else None,
                    "created_at": e.created_at.isoformat(),
                }
                for e in endpoints
            ],
            "limit": _get_tier_limit(request.user),
            "active_count": endpoints.filter(is_active=True).count(),
        }
    )


def _create_endpoint(request):
    from agents_api.permissions import get_tenant

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    url = data.get("url", "").strip()
    if not url:
        return JsonResponse({"error": "url is required"}, status=400)

    event_patterns = data.get("event_patterns", [])
    if not isinstance(event_patterns, list) or not event_patterns:
        return JsonResponse(
            {"error": "event_patterns must be a non-empty list of event patterns"},
            status=400,
        )

    # Check tier limit
    tenant = get_tenant(request.user)
    limit = _get_tier_limit(request.user)
    existing = _get_user_endpoints(request.user).filter(is_active=True).count()
    if existing >= limit:
        return JsonResponse(
            {"error": f"Webhook endpoint limit reached ({limit} for {request.user.tier} tier)"},
            status=403,
        )

    try:
        secret, endpoint = WebhookEndpoint.create_for_user(
            user=request.user,
            url=url,
            event_patterns=event_patterns,
            description=data.get("description", "")[:200],
            tenant=tenant,
        )
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)

    # Audit log
    try:
        from syn.audit.utils import generate_entry

        generate_entry(
            tenant_id=str(tenant.id) if tenant else "",
            actor=request.user.email,
            event_name="webhook.endpoint_created",
            payload={
                "endpoint_id": str(endpoint.id),
                "url": url,
                "event_patterns": event_patterns,
            },
        )
    except Exception:
        pass

    logger.info("Webhook endpoint created: %s for %s", url, request.user.email)

    return JsonResponse(
        {
            "id": str(endpoint.id),
            "url": endpoint.url,
            "secret": secret,  # Shown exactly once
            "event_patterns": endpoint.event_patterns,
            "description": endpoint.description,
            "created_at": endpoint.created_at.isoformat(),
            "warning": "Save this secret now. It will not be shown again.",
        },
        status=201,
    )


@csrf_exempt
@require_team
@require_http_methods(["PUT", "DELETE"])
def endpoint_detail(request, endpoint_id):
    """Update or delete a webhook endpoint."""
    try:
        endpoint = _get_user_endpoints(request.user).get(id=endpoint_id)
    except WebhookEndpoint.DoesNotExist:
        return JsonResponse({"error": "Endpoint not found"}, status=404)

    if request.method == "DELETE":
        endpoint.delete()
        logger.info("Webhook endpoint deleted: %s", endpoint.url)
        return JsonResponse({"status": "deleted"})

    # PUT — update
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if "url" in data:
        url = data["url"].strip()
        if not url.startswith("https://"):
            return JsonResponse({"error": "Webhook URLs must use HTTPS"}, status=400)
        endpoint.url = url

    if "event_patterns" in data:
        patterns = data["event_patterns"]
        if not isinstance(patterns, list) or not patterns:
            return JsonResponse({"error": "event_patterns must be a non-empty list"}, status=400)
        endpoint.event_patterns = patterns

    if "description" in data:
        endpoint.description = str(data["description"])[:200]

    if "is_active" in data:
        endpoint.is_active = bool(data["is_active"])
        if endpoint.is_active:
            # Re-enabling clears circuit breaker
            endpoint.failure_count = 0
            endpoint.disabled_at = None

    endpoint.save()
    return JsonResponse(
        {
            "id": str(endpoint.id),
            "url": endpoint.url,
            "event_patterns": endpoint.event_patterns,
            "description": endpoint.description,
            "is_active": endpoint.is_active,
        }
    )


@csrf_exempt
@require_team
@require_http_methods(["POST"])
def endpoint_test(request, endpoint_id):
    """Send a test event to verify endpoint connectivity."""
    try:
        endpoint = _get_user_endpoints(request.user).get(id=endpoint_id)
    except WebhookEndpoint.DoesNotExist:
        return JsonResponse({"error": "Endpoint not found"}, status=404)

    from .webhook_delivery import deliver

    delivery = WebhookDelivery.objects.create(
        endpoint=endpoint,
        event_name="webhook.test",
        payload={
            "event": "webhook.test",
            "timestamp": __import__("django").utils.timezone.now().isoformat(),
            "webhook_id": str(endpoint.id),
            "data": {
                "message": "This is a test event from Svend.",
                "endpoint_url": endpoint.url,
            },
        },
    )

    deliver(delivery.id)
    delivery.refresh_from_db()

    return JsonResponse(
        {
            "delivery_id": str(delivery.id),
            "status": delivery.status,
            "response_code": delivery.response_code,
            "response_body": delivery.response_body[:200],
        }
    )


@csrf_exempt
@require_team
@require_http_methods(["GET"])
def endpoint_deliveries(request, endpoint_id):
    """List recent deliveries for an endpoint."""
    try:
        endpoint = _get_user_endpoints(request.user).get(id=endpoint_id)
    except WebhookEndpoint.DoesNotExist:
        return JsonResponse({"error": "Endpoint not found"}, status=404)

    deliveries = endpoint.deliveries.order_by("-created_at")[:100]

    return JsonResponse(
        {
            "deliveries": [
                {
                    "id": str(d.id),
                    "event_name": d.event_name,
                    "status": d.status,
                    "response_code": d.response_code,
                    "attempt_count": d.attempt_count,
                    "created_at": d.created_at.isoformat(),
                    "delivered_at": d.delivered_at.isoformat() if d.delivered_at else None,
                }
                for d in deliveries
            ]
        }
    )
