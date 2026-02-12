"""Subscription and access control middleware."""

from datetime import timedelta

from django.conf import settings
from django.http import JsonResponse
from django.utils import timezone

from .constants import is_paid_tier


class SubscriptionMiddleware:
    """Middleware to check subscription status and enforce limits.

    Adds request.subscription_active and request.can_query to requests.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Skip for non-authenticated users
        if not hasattr(request, "user") or not request.user.is_authenticated:
            request.subscription_active = False
            request.can_query = False
            return self.get_response(request)

        user = request.user

        # Check subscription status
        request.subscription_active = (
            hasattr(user, "subscription") and user.subscription.is_active
        ) or is_paid_tier(user.tier)

        # Check query limits
        request.can_query = user.can_query()

        # Update last active (throttled to avoid DB write on every request)
        now = timezone.now()
        if user.last_active_at is None or (now - user.last_active_at) > timedelta(minutes=5):
            user.last_active_at = now
            user.save(update_fields=["last_active_at"])

        return self.get_response(request)


class QueryLimitMiddleware:
    """Middleware to enforce query rate limits on API endpoints.

    Returns 429 if user has exceeded their daily limit.
    """

    PROTECTED_PATHS = [
        "/api/chat/",
        "/api/query/",
        "/api/inference/",
        "/api/agents/",
    ]

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Only check POST requests to protected paths
        if request.method != "POST":
            return self.get_response(request)

        is_protected = any(
            request.path.startswith(path) for path in self.PROTECTED_PATHS
        )

        if not is_protected:
            return self.get_response(request)

        # Skip for unauthenticated
        if not hasattr(request, "user") or not request.user.is_authenticated:
            return self.get_response(request)

        user = request.user

        # Check limit
        if not user.can_query():
            return JsonResponse(
                {
                    "error": "Daily query limit reached",
                    "tier": user.tier,
                    "limit": user.daily_limit,
                    "used": user.queries_today,
                    "upgrade_url": "/billing/checkout/",
                },
                status=429,
            )

        return self.get_response(request)


class InviteRequiredMiddleware:
    """Middleware to enforce invite-only registration.

    Only active when REQUIRE_INVITE is True.
    """

    REGISTRATION_PATHS = [
        "/accounts/signup/",
        "/accounts/register/",
        "/api/auth/register/",
    ]

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if not settings.REQUIRE_INVITE:
            return self.get_response(request)

        # Only check registration endpoints
        is_registration = any(
            request.path.startswith(path) for path in self.REGISTRATION_PATHS
        )

        if not is_registration:
            return self.get_response(request)

        # Check for invite code and plan in request (normalize to uppercase)
        invite_code = (request.POST.get("invite_code") or request.GET.get("invite") or "").strip().upper()
        plan = (request.POST.get("plan") or request.GET.get("plan") or "").strip().lower()

        # Also check JSON body for API requests
        import json
        if request.content_type and request.content_type.startswith("application/json"):
            try:
                raw_body = request.body.decode('utf-8') if isinstance(request.body, bytes) else request.body
                body = json.loads(raw_body)
                if not invite_code and body.get("invite_code"):
                    invite_code = body["invite_code"].strip().upper()
                if not plan and body.get("plan"):
                    plan = body["plan"].strip().lower()
            except (json.JSONDecodeError, AttributeError, UnicodeDecodeError):
                pass

        # Paid plans bypass invite requirement (they're paying customers)
        paid_plans = ["founder", "pro", "team", "enterprise"]
        if plan in paid_plans:
            return self.get_response(request)

        if not invite_code:
            return JsonResponse(
                {"error": "Invite code required", "message": "Registration is invite-only during alpha."},
                status=403,
            )

        # Validate invite code
        from .models import InviteCode

        try:
            invite = InviteCode.objects.get(code=invite_code)
            if not invite.is_valid:
                return JsonResponse(
                    {"error": "Invalid invite code", "message": "This invite code has expired or been used."},
                    status=403,
                )
            # Store for use in registration view
            request.invite_code = invite
        except InviteCode.DoesNotExist:
            return JsonResponse(
                {"error": "Invalid invite code", "message": "This invite code does not exist."},
                status=403,
            )

        return self.get_response(request)
