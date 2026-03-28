"""API key management endpoints — SEC-001 §4.5.

POST /api/auth/keys/     — Create a new key (session auth only)
GET  /api/auth/keys/     — List user's keys (session or API key)
DELETE /api/auth/keys/<id>/  — Revoke a key (session auth only)

⚠ SECURITY: Create and revoke require session auth — a stolen API key
cannot create new keys or revoke others (prevents persistence attacks).
"""

import json
import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .models import API_KEY_LIMITS, APIKey
from .permissions import require_auth

logger = logging.getLogger(__name__)


def _require_session_auth(request):
    """Return error response if request is using API key auth instead of session."""
    if getattr(request, "api_key", None) is not None:
        return JsonResponse(
            {
                "error": "This endpoint requires session authentication. "
                "API key management cannot be done via API key.",
                "code": "session_auth_required",
            },
            status=403,
        )
    return None


@csrf_exempt
@require_auth
@require_http_methods(["GET", "POST"])
def key_list_create(request):
    """List or create API keys."""
    if request.method == "GET":
        return _list_keys(request)
    return _create_key(request)


def _list_keys(request):
    """List user's active API keys (metadata only, never hashes)."""
    keys = APIKey.objects.filter(user=request.user, revoked_at__isnull=True).order_by("-created_at")

    return JsonResponse(
        {
            "keys": [
                {
                    "id": str(k.id),
                    "name": k.name,
                    "key_prefix": k.key_prefix,
                    "created_at": k.created_at.isoformat(),
                    "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
                    "expires_at": k.expires_at.isoformat() if k.expires_at else None,
                    "is_active": k.is_active,
                }
                for k in keys
            ],
            "limit": API_KEY_LIMITS.get(request.user.tier, 0),
            "active_count": keys.filter(is_active=True).count(),
        }
    )


def _create_key(request):
    """Create a new API key. Returns the raw key exactly once."""
    # Session auth only — prevent stolen key from creating persistence
    err = _require_session_auth(request)
    if err:
        return err

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        data = {}

    name = str(data.get("name", ""))[:100]
    expires_in_days = data.get("expires_in_days")

    expires_at = None
    if expires_in_days:
        try:
            from datetime import timedelta

            from django.utils import timezone

            days = int(expires_in_days)
            if days < 1 or days > 365:
                return JsonResponse({"error": "expires_in_days must be 1-365"}, status=400)
            expires_at = timezone.now() + timedelta(days=days)
        except (ValueError, TypeError):
            return JsonResponse({"error": "expires_in_days must be an integer"}, status=400)

    try:
        plaintext, api_key = APIKey.create_for_user(
            user=request.user,
            name=name,
            expires_at=expires_at,
        )
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=403)

    # Audit log
    try:
        from syn.audit.utils import generate_entry

        generate_entry(
            tenant_id=str(getattr(request, "tenant_id", None) or ""),
            actor=request.user.email,
            event_name="api_key.created",
            payload={
                "key_id": str(api_key.id),
                "key_prefix": api_key.key_prefix,
                "name": name,
                "expires_at": expires_at.isoformat() if expires_at else None,
            },
        )
    except Exception:
        pass  # Don't fail key creation because audit logging failed

    logger.info(
        "API key created: %s for user %s (%s)",
        api_key.key_prefix,
        request.user.email,
        name or "unnamed",
    )

    return JsonResponse(
        {
            "id": str(api_key.id),
            "key": plaintext,  # ⚠ Shown exactly once — never stored or logged
            "key_prefix": api_key.key_prefix,
            "name": api_key.name,
            "created_at": api_key.created_at.isoformat(),
            "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
            "warning": "Save this key now. It will not be shown again.",
        },
        status=201,
    )


@csrf_exempt
@require_auth
@require_http_methods(["DELETE"])
def key_revoke(request, key_id):
    """Revoke an API key by ID."""
    # Session auth only
    err = _require_session_auth(request)
    if err:
        return err

    try:
        api_key = APIKey.objects.get(id=key_id, user=request.user)
    except APIKey.DoesNotExist:
        return JsonResponse({"error": "Key not found"}, status=404)

    if not api_key.is_active:
        return JsonResponse({"error": "Key already revoked"}, status=400)

    api_key.revoke()

    # Audit log
    try:
        from syn.audit.utils import generate_entry

        generate_entry(
            tenant_id=str(getattr(request, "tenant_id", None) or ""),
            actor=request.user.email,
            event_name="api_key.revoked",
            payload={
                "key_id": str(api_key.id),
                "key_prefix": api_key.key_prefix,
                "name": api_key.name,
            },
        )
    except Exception:
        pass

    logger.info(
        "API key revoked: %s for user %s",
        api_key.key_prefix,
        request.user.email,
    )

    return JsonResponse({"status": "revoked", "id": str(api_key.id)})
