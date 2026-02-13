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


