"""
Logging middleware for LOG-001/002 compliant request tracking.

Standard: LOG-001 §5.3, CTG-001 §5
Compliance: NIST SP 800-53 AU-2, AU-3 / ISO 27001 A.12.4.1

This module provides:
- CorrelationMiddleware: Propagates correlation_id through request lifecycle
- RequestLoggingMiddleware: Logs request/response details

Usage:
    Add to Django MIDDLEWARE:
        MIDDLEWARE = [
            'syn.log.middleware.CorrelationMiddleware',
            'syn.log.middleware.RequestLoggingMiddleware',
            ...
        ]
"""

import logging
import time
import uuid
from typing import Callable

from django.http import HttpRequest, HttpResponse

from syn.log.handlers import (
    set_correlation_id,
    set_tenant_id,
    set_actor_id,
    get_correlation_id,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Header Constants (API-002 §8.1)
# =============================================================================

HEADER_CORRELATION_ID = "X-Correlation-Id"
HEADER_SYN_REQUEST_ID = "Syn-Request-Id"
HEADER_TRACEPARENT = "traceparent"


# =============================================================================
# Correlation Middleware (LOG-001 §5.3, CTG-001 §5)
# =============================================================================


class CorrelationMiddleware:
    """
    Middleware to propagate correlation ID through request lifecycle.

    Standard: LOG-001 §5.3, CTG-001 §5
    Compliance: NIST SP 800-53 AU-2 (Audit Events)

    Features:
    - Extracts correlation_id from request headers
    - Generates new ID if not provided
    - Sets context variables for logging
    - Adds correlation_id to response headers
    - Extracts tenant_id from authenticated user

    Headers checked (in order):
    1. X-Correlation-Id
    2. Syn-Request-Id
    3. traceparent (W3C format)
    """

    def __init__(self, get_response: Callable):
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        # Extract or generate correlation ID
        correlation_id = self._get_correlation_id(request)

        # Set context variables for logging
        set_correlation_id(correlation_id)

        # Extract tenant ID if available
        tenant_id = self._get_tenant_id(request)
        if tenant_id:
            set_tenant_id(tenant_id)

        # Extract actor ID if available
        actor_id = self._get_actor_id(request)
        if actor_id:
            set_actor_id(actor_id)

        # Store on request for downstream use
        request.correlation_id = correlation_id
        request.syn_correlation_id = correlation_id  # Alias for compatibility

        # Process request
        response = self.get_response(request)

        # Add correlation ID to response headers
        response[HEADER_CORRELATION_ID] = correlation_id
        response[HEADER_SYN_REQUEST_ID] = correlation_id

        # Clear context after request
        set_correlation_id(None)
        set_tenant_id(None)
        set_actor_id(None)

        return response

    def _get_correlation_id(self, request: HttpRequest) -> str:
        """Extract correlation ID from headers or generate new one."""
        # Check standard header
        correlation_id = request.headers.get(HEADER_CORRELATION_ID)
        if correlation_id:
            return correlation_id

        # Check Synara header
        correlation_id = request.headers.get(HEADER_SYN_REQUEST_ID)
        if correlation_id:
            return correlation_id

        # Check W3C traceparent header
        traceparent = request.headers.get(HEADER_TRACEPARENT)
        if traceparent:
            # Extract trace-id from traceparent (format: version-trace_id-parent_id-flags)
            parts = traceparent.split("-")
            if len(parts) >= 2:
                return parts[1]

        # Check if already set on request (e.g., by SynRequestIdMiddleware)
        if hasattr(request, "syn_request_id"):
            return request.syn_request_id

        # Generate new UUID
        return str(uuid.uuid4())

    def _get_tenant_id(self, request: HttpRequest) -> str | None:
        """Extract tenant ID from request."""
        # Check request attribute (set by tenant middleware)
        if hasattr(request, "tenant_id"):
            return str(request.tenant_id)

        # Check authenticated user's tenant
        if hasattr(request, "user") and request.user.is_authenticated:
            if hasattr(request.user, "tenant_id"):
                return str(request.user.tenant_id)

        # Check header
        tenant_header = request.headers.get("X-Tenant-Id")
        if tenant_header:
            return tenant_header

        return None

    def _get_actor_id(self, request: HttpRequest) -> str | None:
        """Extract actor ID from request."""
        # Check authenticated user
        if hasattr(request, "user") and request.user.is_authenticated:
            # Prefer email or username
            if hasattr(request.user, "email") and request.user.email:
                return request.user.email
            if hasattr(request.user, "username"):
                return request.user.username
            return str(request.user.pk)

        return None


# =============================================================================
# Request Logging Middleware (LOG-001 §6)
# =============================================================================


class RequestLoggingMiddleware:
    """
    Middleware to log HTTP request/response details.

    Standard: LOG-001 §6
    Compliance: NIST SP 800-53 AU-3 (Content of Audit Records)

    Features:
    - Logs request start with method, path, user
    - Logs response with status code and duration
    - Respects log level settings
    - Excludes sensitive paths and headers

    Configuration (Django settings):
        LOG_REQUEST_PATHS_EXCLUDE = ['/health/', '/ready/', '/metrics/']
        LOG_REQUEST_LEVEL = 'INFO'
    """

    # Default paths to exclude from logging
    DEFAULT_EXCLUDE_PATHS = [
        "/health/",
        "/ready/",
        "/live/",
        "/metrics/",
        "/static/",
        "/media/",
        "/favicon.ico",
    ]

    # Sensitive headers to mask
    SENSITIVE_HEADERS = [
        "authorization",
        "cookie",
        "x-api-key",
        "x-auth-token",
    ]

    def __init__(self, get_response: Callable):
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        # Check if path should be excluded
        if self._should_exclude(request.path):
            return self.get_response(request)

        # Record start time
        start_time = time.time()

        # Log request start
        self._log_request_start(request)

        # Process request
        response = self.get_response(request)

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Log request completion
        self._log_request_complete(request, response, duration_ms)

        return response

    def _should_exclude(self, path: str) -> bool:
        """Check if path should be excluded from logging."""
        from django.conf import settings

        exclude_paths = getattr(
            settings, "LOG_REQUEST_PATHS_EXCLUDE", self.DEFAULT_EXCLUDE_PATHS
        )
        return any(path.startswith(excluded) for excluded in exclude_paths)

    def _log_request_start(self, request: HttpRequest) -> None:
        """Log request start."""
        correlation_id = get_correlation_id()
        user = self._get_user_display(request)

        logger.info(
            f"Request started: {request.method} {request.path}",
            extra={
                "correlation_id": correlation_id,
                "method": request.method,
                "path": request.path,
                "user": user,
                "user_agent": request.headers.get("User-Agent", ""),
                "remote_addr": self._get_client_ip(request),
            },
        )

    def _log_request_complete(
        self,
        request: HttpRequest,
        response: HttpResponse,
        duration_ms: float,
    ) -> None:
        """Log request completion."""
        correlation_id = get_correlation_id()
        user = self._get_user_display(request)

        # Determine log level based on status code
        if response.status_code >= 500:
            log_func = logger.error
        elif response.status_code >= 400:
            log_func = logger.warning
        else:
            log_func = logger.info

        log_func(
            f"Request completed: {request.method} {request.path} "
            f"-> {response.status_code} ({duration_ms:.2f}ms)",
            extra={
                "correlation_id": correlation_id,
                "method": request.method,
                "path": request.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
                "user": user,
                "content_length": len(response.content) if hasattr(response, "content") else 0,
            },
        )

    def _get_user_display(self, request: HttpRequest) -> str:
        """Get user display string for logging."""
        if hasattr(request, "user") and request.user.is_authenticated:
            if hasattr(request.user, "email") and request.user.email:
                return request.user.email
            return str(request.user)
        return "anonymous"

    def _get_client_ip(self, request: HttpRequest) -> str:
        """Get client IP address from request."""
        # Check X-Forwarded-For header (behind proxy)
        x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
        if x_forwarded_for:
            return x_forwarded_for.split(",")[0].strip()

        # Check X-Real-IP header
        x_real_ip = request.META.get("HTTP_X_REAL_IP")
        if x_real_ip:
            return x_real_ip

        # Fall back to REMOTE_ADDR
        return request.META.get("REMOTE_ADDR", "unknown")


# =============================================================================
# Audit Logging Middleware (AUD-001 §5)
# =============================================================================


class AuditLoggingMiddleware:
    """
    Middleware to create audit log entries for sensitive operations.

    Standard: AUD-001 §5
    Compliance: SOC 2 CC7.2, ISO 27001 A.12.7

    Features:
    - Creates SysLogEntry for auditable requests
    - Configurable audit paths and methods
    - Captures request/response summary
    - Links to correlation ID for tracing

    Configuration (Django settings):
        AUDIT_PATHS = ['/api/v1/users/', '/api/v1/documents/']
        AUDIT_METHODS = ['POST', 'PUT', 'PATCH', 'DELETE']
    """

    # Default paths that require audit logging
    DEFAULT_AUDIT_PATHS = [
        "/api/",
    ]

    # Methods that require audit logging
    DEFAULT_AUDIT_METHODS = ["POST", "PUT", "PATCH", "DELETE"]

    def __init__(self, get_response: Callable):
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        # Check if request should be audited
        should_audit = self._should_audit(request)

        # Process request
        response = self.get_response(request)

        # Create audit entry if needed
        if should_audit and response.status_code < 500:
            self._create_audit_entry(request, response)

        return response

    def _should_audit(self, request: HttpRequest) -> bool:
        """Check if request should be audited."""
        from django.conf import settings

        audit_paths = getattr(settings, "AUDIT_PATHS", self.DEFAULT_AUDIT_PATHS)
        audit_methods = getattr(settings, "AUDIT_METHODS", self.DEFAULT_AUDIT_METHODS)

        # Check method
        if request.method not in audit_methods:
            return False

        # Check path
        return any(request.path.startswith(path) for path in audit_paths)

    def _create_audit_entry(
        self,
        request: HttpRequest,
        response: HttpResponse,
    ) -> None:
        """Create audit log entry for request."""
        try:
            from syn.audit import generate_entry

            # Get actor
            actor = "anonymous"
            if hasattr(request, "user") and request.user.is_authenticated:
                if hasattr(request.user, "email") and request.user.email:
                    actor = request.user.email
                else:
                    actor = str(request.user)

            # Get tenant
            tenant_id = None
            if hasattr(request, "tenant_id") and request.tenant_id is not None:
                tenant_id = str(request.tenant_id)
            elif hasattr(request, "user") and hasattr(request.user, "tenant_id") and request.user.tenant_id is not None:
                tenant_id = str(request.user.tenant_id)

            # Build event name
            event_name = f"api.{request.method.lower()}.{self._path_to_event(request.path)}"

            # Build payload
            payload = {
                "method": request.method,
                "path": request.path,
                "status_code": response.status_code,
                "client_ip": self._get_client_ip(request),
            }

            # Add query params (sanitized)
            if request.GET:
                payload["query_params"] = {
                    k: v for k, v in request.GET.items() if k.lower() not in ["password", "token", "key"]
                }

            # Get correlation ID (must be valid UUID for SysLogEntry)
            correlation_id = getattr(request, "correlation_id", None)
            if correlation_id:
                # Validate UUID format - skip if not a valid UUID
                try:
                    import uuid as uuid_module
                    uuid_module.UUID(str(correlation_id))
                except (ValueError, AttributeError):
                    # Store non-UUID correlation in payload instead
                    payload["request_id"] = correlation_id
                    correlation_id = None

            # Create audit entry
            generate_entry(
                tenant_id=tenant_id,
                actor=actor,
                event_name=event_name,
                payload=payload,
                correlation_id=correlation_id,
            )

        except Exception as e:
            logger.warning(f"Failed to create audit entry: {e}")

    def _path_to_event(self, path: str) -> str:
        """Convert path to event name."""
        # Remove leading/trailing slashes and API version prefix
        path = path.strip("/")
        if path.startswith("api/v"):
            # Remove api/v1/ prefix
            parts = path.split("/", 2)
            if len(parts) > 2:
                path = parts[2]
            else:
                path = parts[-1]

        # Replace slashes with dots, remove UUIDs
        import re

        path = re.sub(r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}", "", path)
        path = re.sub(r"/+", ".", path)
        path = re.sub(r"\.+", ".", path)
        path = path.strip(".")

        return path or "request"

    def _get_client_ip(self, request: HttpRequest) -> str:
        """Get client IP address from request."""
        x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
        if x_forwarded_for:
            return x_forwarded_for.split(",")[0].strip()
        return request.META.get("REMOTE_ADDR", "unknown")
