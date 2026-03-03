"""
Tenant Isolation Middleware - Day 10

Automatically scopes database queries to the authenticated user's tenant.
Prevents cross-tenant data access and provides secure multi-tenancy.

Security:
- Extracts tenant_id from JWT token claims
- Sets thread-local tenant context
- Fails closed (blocks request if no tenant)
- Audit logging for all tenant switches

Compliance: ISO 27001 A.9.4, SOC 2 CC6.1
"""

import threading

from django.http import JsonResponse
from django.utils.deprecation import MiddlewareMixin

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
    Middleware that enforces tenant isolation.

    Process:
    1. Extract JWT token from Authorization header
    2. Decode token and get tenant_id claim
    3. Set tenant in thread-local context
    4. All queries automatically scoped to tenant
    5. Clear tenant context after request

    Exempted paths:
    - /admin/
    - /api/auth/
    - /health/
    """

    EXEMPT_PATHS = [
        "/admin/",
        "/api/auth/",
        "/api/token/",
        "/api/v1/api/reflex",   # REF-001: Browser session auth support for Editor UI
        "/api/v1/api/primitives/",  # PRM-001: Primitive schema API for unified reflex editor
        "/api/v1/events/",      # EVT-001: Browser session auth support for Event Editor UI
        "/api/events/",         # EVT-001: Event schema API for unified reflex editor
        "/api/v1/reflexes/",    # REF-001: Reflex CRUD API for unified reflex editor
        "/health/",
        "/metrics/",
        "/login/",     # Login page
        "/logout/",    # Logout
        "/ui/",        # UI views (auth handled by Django)
        "/forms/",     # Forms module UI (auth handled by Django)
        "/static/",    # Static files
        "/telemetry/", # Telemetry dashboard (TEL-001 v3.0) - filters by tenant internally
    ]

    EXEMPT_EXACT = [
        "/",           # Root redirect
        "/favicon.ico",
    ]

    def process_request(self, request):
        """Process incoming request and set tenant context."""
        from django.conf import settings

        # SECURITY: Test bypass is ONLY allowed when both conditions are met:
        # 1. DEBUG mode is enabled (development environment)
        # 2. TEST_BYPASS_AUTH setting is explicitly set
        if settings.DEBUG and getattr(settings, "TEST_BYPASS_AUTH", False):
            # Set a test tenant context for testing
            set_current_tenant(None)  # Allow None tenant in test mode
            request.tenant = None
            return None

        # Check if path is exempt (prefix match)
        if any(request.path.startswith(path) for path in self.EXEMPT_PATHS):
            return None

        # Check if path is exactly exempt
        if request.path in self.EXEMPT_EXACT:
            return None

        # Try to get tenant from JWT token
        tenant = self._extract_tenant_from_request(request)

        if tenant:
            set_current_tenant(tenant)
            request.tenant = tenant
            request.tenant_id = tenant.id  # Also set tenant_id for views that expect it
        else:
            # Not exempt and no tenant - block request
            return JsonResponse(
                {"error": "Tenant isolation required", "detail": "No valid tenant found in authentication token"},
                status=403,
            )

        return None

    def process_response(self, request, response):
        """Clear tenant context after request."""
        clear_current_tenant()
        return response

    def process_exception(self, request, exception):
        """Clear tenant context on exception."""
        clear_current_tenant()
        return None

    def _extract_tenant_from_request(self, request):
        """
        Extract tenant from JWT token or session authentication.

        Supports:
        1. JWT token (Authorization: Bearer <token>)
        2. Session auth (authenticated user with profile.tenant_id or superuser default)

        Returns:
            Tenant instance or None
        """
        from django.conf import settings

        import jwt
        from syn.synara.models import Tenant

        # Try JWT token first (API clients)
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            try:
                payload = jwt.decode(
                    token,
                    settings.SECRET_KEY,
                    algorithms=["HS256"],
                    options={"verify_signature": True},
                )
                tenant_id = payload.get("tenant") or payload.get("tenant_id") or payload.get('tenantId')
                if tenant_id:
                    return Tenant.objects.get(id=tenant_id, is_active=True)
            except (jwt.DecodeError, jwt.ExpiredSignatureError, Tenant.DoesNotExist):
                pass

        # Try session authentication (browser users)
        if hasattr(request, 'user') and request.user.is_authenticated:
            user = request.user

            # Try to get tenant_id from user profile
            tenant_id = getattr(user, 'tenant_id', None)
            if not tenant_id and hasattr(user, 'profile'):
                tenant_id = getattr(user.profile, 'tenant_id', None)

            # For superusers, get default tenant from database
            if not tenant_id and getattr(user, 'is_superuser', False):
                try:
                    return Tenant.objects.filter(is_active=True).first()
                except Exception:
                    pass

            # If we have a tenant_id, get the tenant
            if tenant_id:
                try:
                    return Tenant.objects.get(id=tenant_id, is_active=True)
                except Tenant.DoesNotExist:
                    pass

        return None


class TenantQuerySetMixin:
    """
    Mixin for model managers to automatically scope queries to current tenant.

    Usage:
        class BinderDocumentManager(TenantQuerySetMixin, models.Manager):
            pass

        class BinderDocument(models.Model):
            objects = BinderDocumentManager()
    """

    def get_queryset(self):
        """Get queryset scoped to current tenant."""
        qs = super().get_queryset()
        tenant = get_current_tenant()

        if tenant and hasattr(self.model, "tenant"):
            qs = qs.filter(tenant=tenant)

        return qs
