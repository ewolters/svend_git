"""
Synara Error Exceptions (ERR-001)
=================================

SynaraError exception hierarchy for standardized error handling
across all Synara components.

Standard:     ERR-001 §5
Compliance:   ISO 27001 A.12.4, SOC 2 CC7.2
Version:      1.0.0

Exception Hierarchy
-------------------
    SynaraError (base)
    ├── ValidationError      - Input validation failures (400/422)
    ├── AuthenticationError  - Auth credential errors (401)
    ├── AuthorizationError   - Permission errors (403)
    ├── NotFoundError        - Resource not found (404)
    ├── ConflictError        - Resource state conflict (409)
    ├── RateLimitError       - Rate limit exceeded (429)
    ├── DependencyError      - External service failure (502/503)
    ├── DatabaseError        - Database operation failure (500)
    ├── TimeoutError         - Operation timeout (504)
    └── SystemError          - Internal system error (500)

All exceptions:
- Include correlation_id for distributed tracing
- Map to HTTP status codes per ERR-001 §3.1
- Support error context per ERR-001 §6
- Integrate with audit logging per AUD-001
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from .types import (
    CATEGORY_STATUS_CODES,
    ERROR_REGISTRY,
    RETRYABLE_CATEGORIES,
    ErrorCategory,
    ErrorContext,
    ErrorDetail,
    ErrorEnvelope,
    ErrorSeverity,
)

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("syn.audit")
error_logger = logging.getLogger("syn.errors")


# =============================================================================
# BASE EXCEPTION (ERR-001 §5.1)
# =============================================================================


class SynaraError(Exception):
    """
    Base exception for all Synara errors per ERR-001 §5.1.

    All Synara-specific exceptions MUST inherit from SynaraError.
    This ensures consistent error handling, logging, and API responses.

    Attributes:
        message: Human-readable error description
        code: Machine-readable error code (e.g., "validation_error")
        category: Error category from ErrorCategory enum
        severity: Error severity from ErrorSeverity enum
        correlation_id: Distributed tracing correlation ID
        context: Additional error context (ErrorContext)
        details: Field-level error details for validation errors
        retryable: Whether the operation can be retried
        http_status: HTTP status code for API responses
        doc_url: Documentation URL for the error

    Example:
        >>> raise SynaraError(
        ...     message="Invalid configuration",
        ...     code="config_error",
        ...     category=ErrorCategory.VALIDATION,
        ...     correlation_id=uuid4()
        ... )
    """

    # Default category for base class
    default_category: ErrorCategory = ErrorCategory.SYSTEM
    default_severity: ErrorSeverity = ErrorSeverity.ERROR
    default_code: str = "synara_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        category: ErrorCategory | None = None,
        severity: ErrorSeverity | None = None,
        correlation_id: UUID | None = None,
        context: ErrorContext | None = None,
        details: list[ErrorDetail] | None = None,
        cause: Exception | None = None,
        extra: dict[str, Any] | None = None,
    ):
        self.message = message
        self.code = code or self.default_code
        self.category = category or self.default_category
        self.severity = severity or self.default_severity
        self.correlation_id = correlation_id or uuid4()
        self.context = context
        self.details = details or []
        self.cause = cause
        self.extra = extra or {}
        self.timestamp = datetime.utcnow()

        # Derive retryable and HTTP status from category
        self.retryable = self.category in RETRYABLE_CATEGORIES
        self.http_status = CATEGORY_STATUS_CODES.get(self.category, 500)

        # Get doc URL from registry or generate default
        registry_entry = ERROR_REGISTRY.get(self.code)
        if registry_entry:
            self.doc_url = registry_entry.doc_url
        else:
            self.doc_url = f"https://docs.synara.io/errors#{self.code}"

        # Log to error logger
        self._log_error()

        super().__init__(message)

    def _log_error(self) -> None:
        """Log error to appropriate loggers."""
        log_data = {
            "error_code": self.code,
            "error_category": self.category.value,
            "error_severity": self.severity.value,
            "correlation_id": str(self.correlation_id),
            "error_message": self.message,  # Use error_message to avoid conflict with logging's reserved 'message' key
            "retryable": self.retryable,
            "http_status": self.http_status,
        }

        if self.context:
            # Handle both ErrorContext objects and plain dicts (from KernelException)
            if hasattr(self.context, "to_dict"):
                log_data["context"] = self.context.to_dict()
            elif isinstance(self.context, dict):
                log_data["context"] = self.context
            else:
                log_data["context"] = str(self.context)

        if self.details:
            log_data["details"] = [{"field": d.field, "code": d.code, "message": d.message} for d in self.details]

        if self.cause:
            log_data["cause_type"] = type(self.cause).__name__
            log_data["cause_message"] = str(self.cause)

        # Log at appropriate level based on severity
        if self.severity == ErrorSeverity.CRITICAL:
            error_logger.critical(f"SynaraError: {self.message}", extra=log_data)
            audit_logger.critical(
                f"CRITICAL ERROR: {self.code}",
                extra={"event_type": "error.critical", **log_data},
            )
        elif self.severity == ErrorSeverity.ERROR:
            error_logger.error(f"SynaraError: {self.message}", extra=log_data)
        elif self.severity == ErrorSeverity.WARNING:
            error_logger.warning(f"SynaraError: {self.message}", extra=log_data)
        else:
            error_logger.info(f"SynaraError: {self.message}", extra=log_data)

    def __str__(self) -> str:
        """Return string representation with correlation ID."""
        return f"[{self.code}] {self.message} (correlation_id={self.correlation_id})"

    def __repr__(self) -> str:
        """Return detailed representation."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"code={self.code!r}, "
            f"category={self.category.value!r}, "
            f"correlation_id={self.correlation_id!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for logging/serialization.

        Returns:
            Dictionary with all error details
        """
        result = {
            "error_type": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "correlation_id": str(self.correlation_id),
            "timestamp": self.timestamp.isoformat(),
            "retryable": self.retryable,
            "http_status": self.http_status,
            "doc_url": self.doc_url,
        }

        if self.context:
            # Handle both ErrorContext objects and plain dicts (from KernelException)
            if hasattr(self.context, "to_dict"):
                result["context"] = self.context.to_dict()
            elif isinstance(self.context, dict):
                result["context"] = self.context
            else:
                result["context"] = str(self.context)

        if self.details:
            result["details"] = [
                {
                    "field": d.field,
                    "code": d.code,
                    "message": d.message,
                    **({"value": d.value} if d.value is not None else {}),
                }
                for d in self.details
            ]

        if self.cause:
            result["cause"] = {
                "type": type(self.cause).__name__,
                "message": str(self.cause),
            }

        if self.extra:
            result["extra"] = self.extra

        return result

    def to_envelope(self, request_id: str | None = None) -> ErrorEnvelope:
        """
        Convert to ErrorEnvelope for API responses per ERR-002.

        Args:
            request_id: Syn-Request-Id header value

        Returns:
            ErrorEnvelope instance
        """
        return ErrorEnvelope(
            code=self.code,
            message=self.message,
            retryable=self.retryable,
            request_id=request_id or str(self.correlation_id),
            details=self.details,
            doc=self.doc_url,
            correlation=str(self.correlation_id),
        )


# =============================================================================
# VALIDATION ERROR (ERR-001 §5.2)
# =============================================================================


class ValidationError(SynaraError):
    """
    Input validation error per ERR-001 §5.2.

    Raised when request data fails validation (schema, format, business rules).

    HTTP Status: 400 (Bad Request) or 422 (Unprocessable Entity)

    Example:
        >>> raise ValidationError(
        ...     message="Invalid CAPA ID format",
        ...     field="capa_id",
        ...     expected="UUID v4",
        ...     received="not-a-uuid"
        ... )
    """

    default_category = ErrorCategory.VALIDATION
    default_severity = ErrorSeverity.WARNING
    default_code = "validation_error"

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        expected: str | None = None,
        received: Any | None = None,
        **kwargs,
    ):
        self.field = field
        self.expected = expected
        self.received = received

        # Build details from field info
        details = kwargs.pop("details", [])
        if field and not details:
            details = [
                ErrorDetail(
                    field=field,
                    code="invalid_value",
                    message=message,
                    value=received,
                )
            ]

        extra = kwargs.pop("extra", {})
        extra.update(
            {
                "field": field,
                "expected": expected,
                "received": str(received) if received is not None else None,
            }
        )

        super().__init__(message, details=details, extra=extra, **kwargs)


# =============================================================================
# AUTHENTICATION ERROR (ERR-001 §5.3)
# =============================================================================


class AuthenticationError(SynaraError):
    """
    Authentication error per ERR-001 §5.3.

    Raised when authentication fails (invalid/expired credentials).

    HTTP Status: 401 (Unauthorized)

    Example:
        >>> raise AuthenticationError(
        ...     message="JWT token expired",
        ...     auth_method="jwt"
        ... )
    """

    default_category = ErrorCategory.AUTHENTICATION
    default_severity = ErrorSeverity.WARNING
    default_code = "unauthorized"

    def __init__(
        self,
        message: str,
        *,
        auth_method: str | None = None,
        **kwargs,
    ):
        self.auth_method = auth_method

        extra = kwargs.pop("extra", {})
        extra["auth_method"] = auth_method

        super().__init__(message, extra=extra, **kwargs)


# =============================================================================
# AUTHORIZATION ERROR (ERR-001 §5.4)
# =============================================================================


class AuthorizationError(SynaraError):
    """
    Authorization error per ERR-001 §5.4.

    Raised when authenticated user lacks required permissions.

    HTTP Status: 403 (Forbidden)

    Example:
        >>> raise AuthorizationError(
        ...     message="User lacks admin permission",
        ...     required_permission="admin:write",
        ...     user_permissions=["user:read"]
        ... )
    """

    default_category = ErrorCategory.AUTHORIZATION
    default_severity = ErrorSeverity.WARNING
    default_code = "forbidden"

    def __init__(
        self,
        message: str,
        *,
        required_permission: str | None = None,
        user_permissions: list[str] | None = None,
        resource: str | None = None,
        **kwargs,
    ):
        self.required_permission = required_permission
        self.user_permissions = user_permissions or []
        self.resource = resource

        extra = kwargs.pop("extra", {})
        extra.update(
            {
                "required_permission": required_permission,
                "user_permissions": user_permissions,
                "resource": resource,
            }
        )

        # Log authorization failures to audit log
        # EVT-001: domain.entity.action naming pattern
        audit_logger.warning(
            f"Authorization denied: {message}",
            extra={
                "event_type": "security.authorization.denied",
                "required_permission": required_permission,
                "resource": resource,
            },
        )

        super().__init__(message, extra=extra, **kwargs)


# =============================================================================
# NOT FOUND ERROR (ERR-001 §5.5)
# =============================================================================


class NotFoundError(SynaraError):
    """
    Resource not found error per ERR-001 §5.5.

    Raised when a requested resource does not exist.

    HTTP Status: 404 (Not Found)

    Example:
        >>> raise NotFoundError(
        ...     message="CAPA not found",
        ...     resource_type="CAPA",
        ...     resource_id="550e8400-e29b-41d4-a716-446655440000"
        ... )
    """

    default_category = ErrorCategory.NOT_FOUND
    default_severity = ErrorSeverity.INFO
    default_code = "not_found"

    def __init__(
        self,
        message: str,
        *,
        resource_type: str | None = None,
        resource_id: str | None = None,
        **kwargs,
    ):
        self.resource_type = resource_type
        self.resource_id = resource_id

        extra = kwargs.pop("extra", {})
        extra.update(
            {
                "resource_type": resource_type,
                "resource_id": resource_id,
            }
        )

        super().__init__(message, extra=extra, **kwargs)


# =============================================================================
# CONFLICT ERROR (ERR-001 §5.6)
# =============================================================================


class ConflictError(SynaraError):
    """
    Resource conflict error per ERR-001 §5.6.

    Raised when operation conflicts with current resource state.

    HTTP Status: 409 (Conflict)

    Example:
        >>> raise ConflictError(
        ...     message="CAPA already in CLOSED state",
        ...     resource_type="CAPA",
        ...     resource_id="550e8400-e29b-41d4-a716-446655440000",
        ...     current_state="CLOSED",
        ...     requested_action="close"
        ... )
    """

    default_category = ErrorCategory.CONFLICT
    default_severity = ErrorSeverity.WARNING
    default_code = "conflict"

    def __init__(
        self,
        message: str,
        *,
        resource_type: str | None = None,
        resource_id: str | None = None,
        current_state: str | None = None,
        requested_action: str | None = None,
        **kwargs,
    ):
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.current_state = current_state
        self.requested_action = requested_action

        extra = kwargs.pop("extra", {})
        extra.update(
            {
                "resource_type": resource_type,
                "resource_id": resource_id,
                "current_state": current_state,
                "requested_action": requested_action,
            }
        )

        super().__init__(message, extra=extra, **kwargs)


# =============================================================================
# RATE LIMIT ERROR (ERR-001 §5.7)
# =============================================================================


class RateLimitError(SynaraError):
    """
    Rate limit exceeded error per ERR-001 §5.7.

    Raised when request rate exceeds configured limits.

    HTTP Status: 429 (Too Many Requests)

    Example:
        >>> raise RateLimitError(
        ...     message="API rate limit exceeded",
        ...     limit=100,
        ...     window_seconds=60,
        ...     retry_after_seconds=30
        ... )
    """

    default_category = ErrorCategory.RATE_LIMIT
    default_severity = ErrorSeverity.WARNING
    default_code = "rate_limit_exceeded"

    def __init__(
        self,
        message: str,
        *,
        limit: int | None = None,
        window_seconds: int | None = None,
        retry_after_seconds: int | None = None,
        **kwargs,
    ):
        self.limit = limit
        self.window_seconds = window_seconds
        self.retry_after_seconds = retry_after_seconds

        extra = kwargs.pop("extra", {})
        extra.update(
            {
                "limit": limit,
                "window_seconds": window_seconds,
                "retry_after_seconds": retry_after_seconds,
            }
        )

        super().__init__(message, extra=extra, **kwargs)


# =============================================================================
# DEPENDENCY ERROR (ERR-001 §5.8)
# =============================================================================


class DependencyError(SynaraError):
    """
    External dependency error per ERR-001 §5.8.

    Raised when an external service fails or is unavailable.

    HTTP Status: 502 (Bad Gateway) or 503 (Service Unavailable)

    Example:
        >>> raise DependencyError(
        ...     message="Document service unavailable",
        ...     service_name="document-service",
        ...     endpoint="/api/v1/documents"
        ... )
    """

    default_category = ErrorCategory.DEPENDENCY
    default_severity = ErrorSeverity.ERROR
    default_code = "dependency_error"

    def __init__(
        self,
        message: str,
        *,
        service_name: str | None = None,
        endpoint: str | None = None,
        status_code: int | None = None,
        **kwargs,
    ):
        self.service_name = service_name
        self.endpoint = endpoint
        self.service_status_code = status_code

        extra = kwargs.pop("extra", {})
        extra.update(
            {
                "service_name": service_name,
                "endpoint": endpoint,
                "service_status_code": status_code,
            }
        )

        super().__init__(message, extra=extra, **kwargs)


# =============================================================================
# DATABASE ERROR (ERR-001 §5.9)
# =============================================================================


class DatabaseError(SynaraError):
    """
    Database operation error per ERR-001 §5.9.

    Raised when a database operation fails.

    HTTP Status: 500 (Internal Server Error)

    Example:
        >>> raise DatabaseError(
        ...     message="Failed to create CAPA record",
        ...     operation="INSERT",
        ...     table="capa"
        ... )
    """

    default_category = ErrorCategory.DATABASE
    default_severity = ErrorSeverity.ERROR
    default_code = "database_error"

    def __init__(
        self,
        message: str,
        *,
        operation: str | None = None,
        table: str | None = None,
        constraint: str | None = None,
        **kwargs,
    ):
        self.operation = operation
        self.table = table
        self.constraint = constraint

        extra = kwargs.pop("extra", {})
        extra.update(
            {
                "db_operation": operation,
                "table": table,
                "constraint": constraint,
            }
        )

        super().__init__(message, extra=extra, **kwargs)


# =============================================================================
# TIMEOUT ERROR (ERR-001 §5.10)
# =============================================================================


class TimeoutError(SynaraError):
    """
    Operation timeout error per ERR-001 §5.10.

    Raised when an operation exceeds its time limit.

    HTTP Status: 504 (Gateway Timeout)

    Example:
        >>> raise TimeoutError(
        ...     message="RCA analysis timed out",
        ...     operation="analyze_rca",
        ...     timeout_ms=30000,
        ...     elapsed_ms=30500
        ... )
    """

    default_category = ErrorCategory.TIMEOUT
    default_severity = ErrorSeverity.ERROR
    default_code = "gateway_timeout"

    def __init__(
        self,
        message: str,
        *,
        operation: str | None = None,
        timeout_ms: int | None = None,
        elapsed_ms: int | None = None,
        **kwargs,
    ):
        self.operation = operation
        self.timeout_ms = timeout_ms
        self.elapsed_ms = elapsed_ms

        extra = kwargs.pop("extra", {})
        extra.update(
            {
                "operation": operation,
                "timeout_ms": timeout_ms,
                "elapsed_ms": elapsed_ms,
            }
        )

        super().__init__(message, extra=extra, **kwargs)


# =============================================================================
# SYSTEM ERROR (ERR-001 §5.11)
# =============================================================================


class SystemError(SynaraError):
    """
    Internal system error per ERR-001 §5.11.

    Raised for unexpected internal errors that don't fit other categories.

    HTTP Status: 500 (Internal Server Error)

    Example:
        >>> raise SystemError(
        ...     message="Unexpected state in cognition engine",
        ...     component="CognitionEngine",
        ...     severity=ErrorSeverity.CRITICAL
        ... )
    """

    default_category = ErrorCategory.SYSTEM
    default_severity = ErrorSeverity.ERROR
    default_code = "internal_error"

    def __init__(
        self,
        message: str,
        *,
        component: str | None = None,
        **kwargs,
    ):
        self.component = component

        extra = kwargs.pop("extra", {})
        extra["component"] = component

        super().__init__(message, extra=extra, **kwargs)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def wrap_exception(
    exc: Exception,
    *,
    message: str | None = None,
    correlation_id: UUID | None = None,
    context: ErrorContext | None = None,
) -> SynaraError:
    """
    Wrap a standard exception in a SynaraError.

    Preserves the original exception as the cause and maps common
    exception types to appropriate SynaraError subclasses.

    Args:
        exc: Original exception to wrap
        message: Override message (default: use original)
        correlation_id: Correlation ID for tracing
        context: Error context

    Returns:
        Appropriate SynaraError subclass
    """
    msg = message or str(exc)

    # Map common exception types
    if isinstance(exc, ValueError):
        return ValidationError(
            msg,
            cause=exc,
            correlation_id=correlation_id,
            context=context,
        )
    elif isinstance(exc, PermissionError):
        return AuthorizationError(
            msg,
            cause=exc,
            correlation_id=correlation_id,
            context=context,
        )
    elif isinstance(exc, FileNotFoundError):
        return NotFoundError(
            msg,
            cause=exc,
            correlation_id=correlation_id,
            context=context,
        )
    elif isinstance(exc, ConnectionError):
        return DependencyError(
            msg,
            cause=exc,
            correlation_id=correlation_id,
            context=context,
        )
    elif isinstance(exc, TimeoutError):
        return TimeoutError(
            msg,
            cause=exc,
            correlation_id=correlation_id,
            context=context,
        )
    else:
        return SystemError(
            msg,
            cause=exc,
            correlation_id=correlation_id,
            context=context,
        )


def create_error_from_code(
    code: str,
    *,
    message: str | None = None,
    correlation_id: UUID | None = None,
    **kwargs,
) -> SynaraError:
    """
    Create a SynaraError from a registered error code.

    Args:
        code: Registered error code (e.g., "validation_error")
        message: Override message (default: use template)
        correlation_id: Correlation ID
        **kwargs: Additional parameters for message template

    Returns:
        SynaraError instance

    Raises:
        ValueError: If error code is not registered
    """
    entry = ERROR_REGISTRY.get(code)
    if not entry:
        raise ValueError(f"Unknown error code: {code}")

    msg = message or entry.message_template.format(**kwargs)

    return SynaraError(
        message=msg,
        code=code,
        category=entry.category,
        severity=entry.severity,
        correlation_id=correlation_id,
        **kwargs,
    )
