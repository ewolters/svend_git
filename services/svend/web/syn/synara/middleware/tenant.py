"""
Tenant Isolation Middleware

Scopes database queries to the authenticated user's tenant.
Adapted for Svend: uses Django session auth + core.Membership lookup.
Individual users without tenants pass through — no fail-closed.

Compliance: ISO 27001 A.9.4, SOC 2 CC6.1
"""

import logging
import threading

from django.utils.deprecation import MiddlewareMixin

logger = logging.getLogger(__name__)

# Thread-local storage for tenant context
_thread_locals = threading.local()


def get_current_tenant():
    """Get the current tenant from thread-local storage."""
    return getattr(_thread_locals, "tenant", None)


def set_current_tenant(tenant):
    """Set the current tenant in thread-local storage."""
    _thread_locals.tenant = tenant


def clear_current_tenant():
    """Clear the current tenant from thread-local storage."""
    if hasattr(_thread_locals, "tenant"):
        delattr(_thread_locals, "tenant")


class TenantIsolationMiddleware(MiddlewareMixin):
    """
    Middleware that sets tenant context from authenticated user's membership.

    Svend uses session auth (not JWT). Tenants are optional — individual
    users operate without a tenant. Enterprise users have a Membership
    linking them to a Tenant.

    Process:
    1. Check if user is authenticated
    2. Look up active Membership → Tenant
    3. Set tenant in thread-local context (or None for individual users)
    4. Clear tenant context after request
    """

    EXEMPT_PATHS = [
        "/admin/",
        "/login/",
        "/logout/",
        "/register/",
        "/static/",
        "/health/",
        "/favicon.ico",
        "/billing/",
    ]

    def process_request(self, request):
        """Set tenant context from authenticated user."""
        # Skip exempt paths
        if any(request.path.startswith(p) for p in self.EXEMPT_PATHS):
            return None

        # Unauthenticated → no tenant, pass through
        if not hasattr(request, "user") or not request.user.is_authenticated:
            request.tenant = None
            request.tenant_id = None
            set_current_tenant(None)
            return None

        # Staff/superusers bypass tenant isolation
        if request.user.is_staff or request.user.is_superuser:
            request.tenant = None
            request.tenant_id = None
            set_current_tenant(None)
            return None

        # Look up tenant via Membership
        tenant = self._get_user_tenant(request.user)
        set_current_tenant(tenant)
        request.tenant = tenant
        request.tenant_id = tenant.id if tenant else None

        return None

    def process_response(self, request, response):
        """Clear tenant context after request."""
        clear_current_tenant()
        return response

    def process_exception(self, request, exception):
        """Clear tenant context on exception."""
        clear_current_tenant()
        return None

    def _get_user_tenant(self, user):
        """
        Get the active tenant for a user via Membership.

        Returns Tenant instance or None (individual user, no team).
        """
        try:
            from core.models import Membership

            membership = Membership.objects.filter(
                user=user,
                is_active=True,
            ).select_related("tenant").first()

            if membership and membership.tenant.is_active:
                return membership.tenant
        except Exception as e:
            logger.warning(f"Tenant lookup failed for user {user.pk}: {e}")

        return None


class TenantQuerySetMixin:
    """
    Mixin for model managers to automatically scope queries to current tenant.

    Usage:
        class MyManager(TenantQuerySetMixin, models.Manager):
            pass

        class MyModel(models.Model):
            objects = MyManager()
    """

    def get_queryset(self):
        """Get queryset scoped to current tenant."""
        qs = super().get_queryset()
        tenant = get_current_tenant()

        if tenant and hasattr(self.model, "tenant"):
            qs = qs.filter(tenant=tenant)

        return qs
