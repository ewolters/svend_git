"""Credential resolver middleware for API key authentication.

SEC-001 §4.5 — resolves Bearer tokens to request.user so all existing
decorators (@require_auth, @rate_limited, @gated_paid) work transparently.

Middleware position: immediately after Django's AuthenticationMiddleware.
If a Bearer sv_... header is present, this middleware overrides request.user
with the key's owner. If absent, session auth is unchanged.

Design: the resolver pattern accepts pluggable credential types. When OIDC/JWT
is added later, a second resolver plugs into the same middleware — no
rip-and-replace needed.

Compliance: SOC 2 CC6.1 (Logical Access Security)
"""

import logging

from django.http import JsonResponse

logger = logging.getLogger(__name__)

# Paths that should never require API key auth (public, health, static)
_EXEMPT_PREFIXES = (
    "/static/",
    "/favicon.ico",
    "/api/health/",
    "/admin/",
)


class APIKeyAuthMiddleware:
    """Resolve Authorization: Bearer sv_... to request.user.

    ⚠ SECURITY-CRITICAL: This middleware is a credential resolver. It sets
    request.user — the identity that all downstream middleware and decorators
    trust. Bugs here are privilege escalation vulnerabilities.

    Behavior:
    - Bearer sv_... present + valid → override request.user, set request.api_key
    - Bearer sv_... present + invalid → 401 immediately (fail-closed)
    - Bearer <other> present → ignore (reserved for future JWT/OIDC)
    - No Authorization header → fall through to session auth (no change)
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Skip exempt paths
        if request.path.startswith(_EXEMPT_PREFIXES):
            request.api_key = None
            return self.get_response(request)

        auth_header = request.headers.get("Authorization", "")

        if not auth_header.startswith("Bearer sv_"):
            # No API key header — fall through to session auth
            request.api_key = None
            return self.get_response(request)

        # Extract the raw key
        raw_key = auth_header[7:].strip()  # Strip "Bearer "

        # Lazy import to avoid circular imports at module load
        from .models import APIKey

        api_key = APIKey.resolve(raw_key)

        if api_key is None:
            logger.warning(
                "API key auth failed: invalid, revoked, or expired key (prefix: %s)",
                raw_key[:11] if len(raw_key) >= 11 else "short",
            )
            return JsonResponse(
                {
                    "error": "Invalid, revoked, or expired API key",
                    "code": "invalid_api_key",
                },
                status=401,
            )

        # Override request.user with the key's owner
        request.user = api_key.user
        request.api_key = api_key

        return self.get_response(request)
