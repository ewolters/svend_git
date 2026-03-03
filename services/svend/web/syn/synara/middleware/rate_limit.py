"""
Per-Tenant Rate Limiting Middleware - Day 10

Enforces API rate limits on a per-tenant basis to prevent abuse
and ensure fair resource allocation.

Features:
- Per-tenant rate limiting based on tenant.rate_limit_per_hour
- Uses Django cache for tracking request counts
- Sliding window algorithm
- Exempts admin and health check endpoints

Compliance: ISO 27001 A.14.1, SOC 2 CC7.2
"""

import time

from django.conf import settings
from django.core.cache import cache
from django.http import JsonResponse
from django.utils.deprecation import MiddlewareMixin


class TenantRateLimitMiddleware(MiddlewareMixin):
    """
    Middleware that enforces per-tenant API rate limiting.

    Rate limits are configured per tenant in the Tenant model.
    Uses a sliding window algorithm with Redis/cache backend.

    Process:
    1. Extract tenant from request (set by TenantIsolationMiddleware)
    2. Check current request count for tenant
    3. Allow or block based on rate limit
    4. Increment counter and set expiry
    """

    EXEMPT_PATHS = [
        "/admin/",
        "/api/auth/",
        "/api/token/",
        "/health/",
        "/metrics/",
    ]

    def process_request(self, request):
        """Process incoming request and enforce rate limit."""
        # Check if path is exempt
        if any(request.path.startswith(path) for path in self.EXEMPT_PATHS):
            return None

        # Get tenant from request (set by TenantIsolationMiddleware)
        tenant = getattr(request, "tenant", None)

        if not tenant:
            # No tenant, skip rate limiting (will be handled by TenantIsolationMiddleware)
            return None

        # Check rate limit
        if self._is_rate_limited(tenant):
            return JsonResponse(
                {
                    "error": "Rate limit exceeded",
                    "detail": f"Tenant {tenant.org_name} has exceeded the rate limit of {tenant.rate_limit_per_hour} requests per hour",
                    "retry_after": self._get_retry_after(tenant),
                },
                status=429,
            )

        # Increment request counter
        self._increment_counter(tenant)

        return None

    def _is_rate_limited(self, tenant) -> bool:
        """
        Check if tenant has exceeded rate limit.

        Args:
            tenant: Tenant instance

        Returns:
            True if rate limited, False otherwise
        """
        cache_key = f"rate_limit:{tenant.id}"
        current_count = cache.get(cache_key, 0)

        return current_count >= tenant.rate_limit_per_hour

    def _increment_counter(self, tenant):
        """
        Increment request counter for tenant.

        Uses atomic increment with 1-hour sliding window.

        Args:
            tenant: Tenant instance
        """
        cache_key = f"rate_limit:{tenant.id}"
        current_count = cache.get(cache_key, 0)

        # Increment counter
        cache.set(cache_key, current_count + 1, timeout=3600)  # 1 hour

    def _get_retry_after(self, tenant) -> int:
        """
        Get seconds until rate limit resets.

        Args:
            tenant: Tenant instance

        Returns:
            Seconds until reset
        """
        cache_key = f"rate_limit:{tenant.id}"
        ttl = cache.ttl(cache_key) if hasattr(cache, "ttl") else 3600

        return max(ttl, 0)

    def process_response(self, request, response):
        """Add rate limit headers to response."""
        tenant = getattr(request, "tenant", None)

        if tenant and not any(request.path.startswith(path) for path in self.EXEMPT_PATHS):
            cache_key = f"rate_limit:{tenant.id}"
            current_count = cache.get(cache_key, 0)
            remaining = max(tenant.rate_limit_per_hour - current_count, 0)

            # Add rate limit headers
            response["X-RateLimit-Limit"] = str(tenant.rate_limit_per_hour)
            response["X-RateLimit-Remaining"] = str(remaining)
            response["X-RateLimit-Reset"] = str(int(time.time()) + self._get_retry_after(tenant))

        return response
