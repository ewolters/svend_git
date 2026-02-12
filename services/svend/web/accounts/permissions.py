"""Tier-based access control decorators.

All access control lives here. Use these decorators on views:
- @rate_limited - Standard auth + rate limiting (use for most endpoints)
- @require_paid - Requires any paid tier
- @require_team - Requires Team or Enterprise
- @require_enterprise - Requires Enterprise (for Anthropic access)
"""

from functools import wraps
from django.http import JsonResponse

from .constants import Tier, has_feature, is_paid_tier, can_use_ml, can_use_anthropic


def require_auth(view_func):
    """Require authenticated user."""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({"error": "Authentication required"}, status=401)
        return view_func(request, *args, **kwargs)
    return wrapper


def rate_limited(view_func):
    """Single source of truth for rate limiting.

    Combines:
    - Authentication check
    - Email verification check (optional, controlled by settings)
    - Daily query limit enforcement based on tier
    - Automatic query count increment on success

    This is the standard decorator for most API endpoints.

    Tier limits (from constants.py):
    - FREE: 5 queries/day
    - FOUNDER: 50 queries/day ($19/month, first 100 users)
    - PRO: 50 queries/day ($29/month)
    - TEAM: 200 queries/day ($79/month)
    - ENTERPRISE: 1000 queries/day ($199/month)
    """
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        # Auth check
        if not request.user.is_authenticated:
            return JsonResponse({"error": "Authentication required"}, status=401)

        user = request.user

        # Rate limit check
        if not user.can_query():
            return JsonResponse({
                "error": "Daily query limit reached",
                "limit": user.daily_limit,
                "used": user.queries_today,
                "tier": user.tier,
                "upgrade_url": "/billing/checkout/",
                "message": f"You've used all {user.daily_limit} queries for today. "
                          f"Upgrade your plan for more queries.",
            }, status=429)

        # Execute the view
        response = view_func(request, *args, **kwargs)

        # Increment query count on successful response
        if response.status_code < 400:
            user.increment_queries()

        return response
    return wrapper


def require_paid(view_func):
    """Require any paid tier (Founder, Pro, Team, Enterprise)."""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({"error": "Authentication required"}, status=401)

        if not is_paid_tier(request.user.tier):
            return JsonResponse({
                "error": "Paid subscription required",
                "tier": request.user.tier,
                "upgrade_url": "/billing/checkout/",
                "message": "This feature requires a paid subscription.",
            }, status=403)

        return view_func(request, *args, **kwargs)
    return wrapper


def require_team(view_func):
    """Require Team or Enterprise tier for collaboration features."""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({"error": "Authentication required"}, status=401)

        if not has_feature(request.user.tier, "collaboration"):
            return JsonResponse({
                "error": "Team plan required for collaboration features",
                "tier": request.user.tier,
                "upgrade_url": "/billing/checkout/?plan=team",
                "message": "Collaboration features require a Team or Enterprise plan.",
            }, status=403)

        return view_func(request, *args, **kwargs)
    return wrapper


def require_enterprise(view_func):
    """Require Enterprise tier for AI assistant (Anthropic models)."""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({"error": "Authentication required"}, status=401)

        if not has_feature(request.user.tier, "ai_assistant"):
            return JsonResponse({
                "error": "Enterprise plan required for AI assistant",
                "tier": request.user.tier,
                "upgrade_url": "/billing/checkout/?plan=enterprise",
                "message": "Access to Anthropic models requires an Enterprise plan.",
            }, status=403)

        return view_func(request, *args, **kwargs)
    return wrapper


def require_feature(feature: str):
    """Generic feature gate decorator.

    Usage:
        @require_feature("forge_api")
        def my_view(request):
            ...
    """
    def decorator(view_func):
        @wraps(view_func)
        def wrapper(request, *args, **kwargs):
            if not request.user.is_authenticated:
                return JsonResponse({"error": "Authentication required"}, status=401)

            if not has_feature(request.user.tier, feature):
                return JsonResponse({
                    "error": f"Feature '{feature}' not available on your plan",
                    "tier": request.user.tier,
                    "upgrade_url": "/billing/checkout/",
                }, status=403)

            return view_func(request, *args, **kwargs)
        return wrapper
    return decorator


def require_ml(view_func):
    """Require ML access (PRO, FOUNDER, TEAM, or ENTERPRISE)."""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({"error": "Authentication required"}, status=401)

        if not can_use_ml(request.user.tier):
            return JsonResponse({
                "error": "ML features require a paid subscription",
                "tier": request.user.tier,
                "upgrade_url": "/billing/checkout/",
                "message": "Upgrade to PRO ($29/mo) to access ML model training.",
            }, status=403)

        return view_func(request, *args, **kwargs)
    return wrapper


def gated_paid(view_func):
    """Auth + paid tier (full_tools) + rate limiting.

    Use for premium analysis tools: DOE, Synara, Whiteboard, A3, VSM,
    RCA, Forecast, Forge, Guide.

    Free users get 403 with upgrade prompt.
    Paid users get rate-limited per tier.
    """
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({"error": "Authentication required"}, status=401)

        user = request.user

        if not has_feature(user.tier, "full_tools"):
            return JsonResponse({
                "error": "Upgrade required",
                "feature": "full_tools",
                "tier": user.tier,
                "upgrade_url": "/billing/checkout/",
                "message": "This tool requires a paid subscription. Upgrade to unlock all analysis tools.",
            }, status=403)

        if not user.can_query():
            return JsonResponse({
                "error": "Daily query limit reached",
                "limit": user.daily_limit,
                "used": user.queries_today,
                "tier": user.tier,
                "upgrade_url": "/billing/checkout/",
            }, status=429)

        response = view_func(request, *args, **kwargs)

        if response.status_code < 400:
            user.increment_queries()

        return response
    return wrapper


def require_org_admin(view_func):
    """Require that the user is an owner or admin of their org.

    Checks Membership.can_admin (owner or admin role), NOT Django is_staff.
    The tenant is resolved from the request via _get_tenant() pattern
    or from the user's active membership.
    """
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({"error": "Authentication required"}, status=401)

        from core.models import Membership
        membership = Membership.objects.filter(
            user=request.user, is_active=True
        ).select_related("tenant").first()

        if not membership:
            return JsonResponse({
                "error": "You are not a member of any organization",
            }, status=403)

        if not membership.can_admin:
            return JsonResponse({
                "error": "Organization admin access required",
                "role": membership.role,
                "message": "Only owners and admins can manage organization settings.",
            }, status=403)

        # Attach tenant and membership to request for downstream use
        request.org_tenant = membership.tenant
        request.org_membership = membership
        return view_func(request, *args, **kwargs)
    return wrapper


# Legacy aliases for backwards compatibility
gated = rate_limited  # Old name -> new name
require_auth = require_auth
require_query_limit = rate_limited  # Merged into rate_limited
