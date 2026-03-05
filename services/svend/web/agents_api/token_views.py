"""ActionToken views — unauthenticated, token-is-credential.

Standard: QMS-002 §2.3, SEC-001
"""

import logging

from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .models import ActionToken

logger = logging.getLogger(__name__)


@csrf_exempt
@require_http_methods(["GET", "POST"])
def action_token_view(request, token):
    """Render (GET) or execute (POST) an action token.

    No auth required — the token IS the credential.
    Tokens are single-use, time-limited, and action-scoped.
    """
    tok = get_object_or_404(ActionToken, token=token)

    if not tok.is_valid:
        return JsonResponse(
            {"error": "Token expired or already used"},
            status=410,
        )

    if request.method == "GET":
        return JsonResponse(tok.to_dict())

    # POST — execute the scoped action and mark used
    tok.use()
    return JsonResponse({"ok": True, "action": tok.action_type})
