"""Subscription and access control middleware."""

import hashlib
import re
from datetime import timedelta
from urllib.parse import urlparse

from django.http import JsonResponse
from django.utils import timezone

from .constants import is_paid_tier

BOT_PATTERN = re.compile(
    r"bot|crawl|spider|slurp|bingpreview|facebookexternalhit|Googlebot|"
    r"Baiduspider|YandexBot|DuckDuckBot|Twitterbot|LinkedInBot|"
    r"HeadlessChrome|PhantomJS|Selenium|Puppeteer|"
    r"Claude-User|pageburst|Google-|^Google$|"
    r"Python/|aiohttp|httpx|Go-http-client|Java/|wget|curl/|"
    r"axios/|Palo Alto|Cortex|Xpanse",
    re.IGNORECASE,
)

# Spoofed mobile UAs with ancient OS versions — almost certainly headless crawlers.
# Real users update; 24+ unique IPs claiming iOS 13 or Android 6 in 2026 = bots.
_STALE_MOBILE_RE = re.compile(
    r"(?:iPhone OS|CPU OS) (?:[5-9]|1[0-5])_"  # iOS/iPadOS < 16 (released Sep 2022)
    r"|Android [1-9]\.| Android [1-9] ",  # Android < 10
)

SKIP_PREFIXES = ("/static/", "/media/", "/api/", "/admin/", "/billing/")
SKIP_PATHS = {"/favicon.ico", "/robots.txt", "/sitemap.xml", "/health/"}


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
        request.subscription_active = (hasattr(user, "subscription") and user.subscription.is_active) or is_paid_tier(
            user.tier
        )

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

        is_protected = any(request.path.startswith(path) for path in self.PROTECTED_PATHS)

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


class NoCacheDynamicMiddleware:
    """Prevent Cloudflare from caching dynamic responses.

    WhiteNoise handles static files before this middleware runs,
    so this only affects Django-rendered pages and API responses.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        if "Cache-Control" not in response:
            response["Cache-Control"] = "no-cache, no-store, must-revalidate, private"
        return response


class SafetySubdomainMiddleware:
    """Route safety.svend.ai to the safety app.

    Redirects root (/) and generic app path (/app/) to /app/safety/.
    Login, register, API, static, and other paths pass through normally.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        host = request.get_host().split(":")[0]
        if host == "safety.svend.ai" and request.path in ("/", "/app/"):
            from django.shortcuts import redirect

            return redirect("/app/safety/")
        return self.get_response(request)


class SiteVisitMiddleware:
    """Track anonymous page visits for site-wide marketing analytics.

    Records GET requests to user-facing pages. Skips static files, API calls,
    admin, and common non-page paths. Fire-and-forget: never breaks the request.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)

        # Only track GET requests that returned 200
        if request.method != "GET" or response.status_code != 200:
            return response

        path = request.path

        # Skip non-page paths
        if path in SKIP_PATHS:
            return response
        if any(path.startswith(p) for p in SKIP_PREFIXES):
            return response

        # Skip staff users — don't pollute analytics with our own traffic
        if hasattr(request, "user") and request.user.is_authenticated and request.user.is_staff:
            return response

        try:
            from api.models import SiteVisit

            ua = request.META.get("HTTP_USER_AGENT", "")
            referrer = request.META.get("HTTP_REFERER", "")
            ip = request.META.get("HTTP_CF_CONNECTING_IP", "") or request.META.get("REMOTE_ADDR", "")
            ip_hash = hashlib.sha256(ip.encode()).hexdigest() if ip else ""

            ref_domain = ""
            if referrer:
                try:
                    ref_domain = urlparse(referrer).netloc
                except Exception:
                    pass

            is_bot = bool(not ua.strip() or BOT_PATTERN.search(ua) or _STALE_MOBILE_RE.search(ua))

            # Cloudflare adds CF-IPCountry header (2-letter ISO code)
            country = request.META.get("HTTP_CF_IPCOUNTRY", "")
            if country in ("XX", "T1"):  # XX=unknown, T1=Tor
                country = ""

            SiteVisit.objects.create(
                path=path[:300],
                referrer=referrer[:500],
                referrer_domain=ref_domain[:200],
                ip_hash=ip_hash,
                user_agent=ua[:500],
                country=country[:2],
                is_bot=is_bot,
                method=request.method[:10],
            )
        except Exception:
            pass  # Never let analytics break the request

        return response
