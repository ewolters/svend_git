"""
Synara Error Types (ERR-001/002)
================================

Type definitions for error handling including severity levels,
categories, retry states, and circuit breaker states.

Standard:     ERR-001 §3-7, ERR-002 §3-5
Compliance:   ISO 27001 A.12.4, SOC 2 CC7.2
Version:      1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

# =============================================================================
# ERROR SEVERITY (ERR-001 §3.2)
# =============================================================================


class ErrorSeverity(str, Enum):
    """
    Error severity levels per ERR-001 §3.2.

    Aligned with AUD-001 logging severity levels.

    DEBUG: Diagnostic information for debugging
    INFO: Informational messages, no action required
    WARNING: Potential issue, may require attention
    ERROR: Error occurred, action required
    CRITICAL: Critical failure, immediate action required
    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# Severity to numeric level mapping
SEVERITY_LEVELS: dict[ErrorSeverity, int] = {
    ErrorSeverity.DEBUG: 10,
    ErrorSeverity.INFO: 20,
    ErrorSeverity.WARNING: 30,
    ErrorSeverity.ERROR: 40,
    ErrorSeverity.CRITICAL: 50,
}


# =============================================================================
# ERROR CATEGORY (ERR-001 §3.1)
# =============================================================================


class ErrorCategory(str, Enum):
    """
    Error categories per ERR-001 §3.1.

    Categories map to HTTP status codes and recovery strategies.

    VALIDATION: Input validation errors (400)
    AUTHENTICATION: Auth credential errors (401)
    AUTHORIZATION: Permission errors (403)
    NOT_FOUND: Resource not found (404)
    CONFLICT: Resource conflict (409)
    RATE_LIMIT: Rate limit exceeded (429)
    DEPENDENCY: External service errors (502, 503)
    DATABASE: Database errors (500)
    TIMEOUT: Operation timeout (504)
    SYSTEM: Internal system errors (500)
    """

    VALIDATION = "VALIDATION"
    AUTHENTICATION = "AUTHENTICATION"
    AUTHORIZATION = "AUTHORIZATION"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    RATE_LIMIT = "RATE_LIMIT"
    DEPENDENCY = "DEPENDENCY"
    DATABASE = "DATABASE"
    TIMEOUT = "TIMEOUT"
    SYSTEM = "SYSTEM"


# Category to HTTP status code mapping
CATEGORY_STATUS_CODES: dict[ErrorCategory, int] = {
    ErrorCategory.VALIDATION: 400,
    ErrorCategory.AUTHENTICATION: 401,
    ErrorCategory.AUTHORIZATION: 403,
    ErrorCategory.NOT_FOUND: 404,
    ErrorCategory.CONFLICT: 409,
    ErrorCategory.RATE_LIMIT: 429,
    ErrorCategory.DEPENDENCY: 503,
    ErrorCategory.DATABASE: 500,
    ErrorCategory.TIMEOUT: 504,
    ErrorCategory.SYSTEM: 500,
}


# Retryable categories per ERR-001 §7
RETRYABLE_CATEGORIES = {
    ErrorCategory.RATE_LIMIT,
    ErrorCategory.DEPENDENCY,
    ErrorCategory.TIMEOUT,
}


# =============================================================================
# CIRCUIT BREAKER STATES (ERR-001 §7.3)
# =============================================================================


class CircuitBreakerState(str, Enum):
    """
    Circuit breaker states per ERR-001 §7.3.

    State Machine:
        CLOSED → OPEN (on failure threshold)
        OPEN → HALF_OPEN (after recovery timeout)
        HALF_OPEN → CLOSED (on success) or OPEN (on failure)
    """

    CLOSED = "CLOSED"  # Normal operation, requests pass through
    OPEN = "OPEN"  # Circuit open, requests fail fast
    HALF_OPEN = "HALF_OPEN"  # Testing recovery, limited requests allowed


# =============================================================================
# RETRY STRATEGY (ERR-001 §7.1)
# =============================================================================


class RetryStrategy(str, Enum):
    """
    Retry strategies per ERR-001 §7.1.

    NONE: No retry, fail immediately
    IMMEDIATE: Retry immediately (not recommended)
    LINEAR: Linear backoff (delay * attempt)
    EXPONENTIAL: Exponential backoff with jitter (recommended)
    CONSTANT: Constant delay between retries
    """

    NONE = "NONE"
    IMMEDIATE = "IMMEDIATE"
    LINEAR = "LINEAR"
    EXPONENTIAL = "EXPONENTIAL"
    CONSTANT = "CONSTANT"


# =============================================================================
# RECOVERY MODE (ERR-001 §7.4)
# =============================================================================


class RecoveryMode(str, Enum):
    """
    Degraded mode recovery options per ERR-001 §7.4.

    FAIL_FAST: Immediate failure, no fallback
    FALLBACK: Use cached/default value
    QUEUE: Queue request for retry
    SKIP: Skip operation, continue processing
    """

    FAIL_FAST = "FAIL_FAST"
    FALLBACK = "FALLBACK"
    QUEUE = "QUEUE"
    SKIP = "SKIP"


# =============================================================================
# SYSTEM LAYER (ERR-001 §5, SBL-001)
# =============================================================================


class SystemLayer(str, Enum):
    """
    System layers per ERR-001 §5 and SBL-001.

    Layer identification for error context and routing.
    """

    API = "API"  # API layer (API-001/002)
    SERVICE = "SERVICE"  # Service/business logic layer
    REPOSITORY = "REPOSITORY"  # Data access layer
    INTEGRATION = "INTEGRATION"  # External integration layer
    KERNEL = "KERNEL"  # Synara kernel (SBL-001)
    CLI = "CLI"  # CLI layer (CLI-001)
    UI = "UI"  # User interface layer


# =============================================================================
# ERROR CONTEXT (ERR-001 §6)
# =============================================================================


@dataclass
class ErrorContext:
    """
    Error context per ERR-001 §6.

    Required context for all SynaraError instances.

    Attributes:
        correlation_id: Distributed tracing ID (required)
        timestamp: Error occurrence time (auto-generated)
        tenant_id: Tenant context (optional for system errors)
        user_id: User context (optional)
        operation: Operation that failed
        layer: System layer where error occurred
        request_id: API request ID (Syn-Request-Id)
        trace_id: OpenTelemetry trace ID
        span_id: OpenTelemetry span ID
    """

    correlation_id: UUID
    operation: str
    layer: SystemLayer
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tenant_id: UUID | None = None
    user_id: UUID | None = None
    request_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "correlation_id": str(self.correlation_id),
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": str(self.tenant_id) if self.tenant_id else None,
            "user_id": str(self.user_id) if self.user_id else None,
            "operation": self.operation,
            "layer": self.layer.value,
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "extra": self.extra,
        }


# =============================================================================
# RETRY CONFIGURATION (ERR-001 §7.1)
# =============================================================================


@dataclass
class RetryConfig:
    """
    Retry configuration per ERR-001 §7.1.

    Attributes:
        strategy: Retry strategy to use
        max_attempts: Maximum retry attempts (default: 3)
        base_delay_ms: Base delay in milliseconds (default: 1000)
        max_delay_ms: Maximum delay cap (default: 30000)
        jitter_factor: Jitter factor for exponential (default: 0.1)
        retryable_categories: Categories eligible for retry
    """

    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    max_attempts: int = 3
    base_delay_ms: int = 1000
    max_delay_ms: int = 30000
    jitter_factor: float = 0.1
    retryable_categories: frozenset = field(default_factory=lambda: frozenset(RETRYABLE_CATEGORIES))


# =============================================================================
# CIRCUIT BREAKER CONFIGURATION (ERR-001 §7.3)
# =============================================================================


@dataclass
class CircuitBreakerConfig:
    """
    Circuit breaker configuration per ERR-001 §7.3.

    Attributes:
        failure_threshold: Failures before opening circuit (default: 5)
        success_threshold: Successes in half-open before closing (default: 3)
        recovery_timeout_ms: Time before half-open test (default: 30000)
        half_open_max_requests: Max requests in half-open (default: 3)
        monitored_categories: Error categories that trigger breaker
    """

    failure_threshold: int = 5
    success_threshold: int = 3
    recovery_timeout_ms: int = 30000
    half_open_max_requests: int = 3
    monitored_categories: frozenset = field(
        default_factory=lambda: frozenset(
            {
                ErrorCategory.DEPENDENCY,
                ErrorCategory.DATABASE,
                ErrorCategory.TIMEOUT,
            }
        )
    )


# =============================================================================
# ERROR ENVELOPE (ERR-002 §3)
# =============================================================================


@dataclass
class ErrorDetail:
    """
    Error detail for validation errors per ERR-002 §3.3.

    Provides field-level error information.
    """

    field: str
    code: str
    message: str
    value: Any | None = None


@dataclass
class ErrorEnvelope:
    """
    Canonical error envelope per ERR-002 §3.

    Standard envelope for all API error responses.

    Required fields:
        code: Machine-readable error code (e.g., "validation_error")
        message: Human-readable message
        retryable: Whether the request can be retried
        request_id: Syn-Request-Id for tracing

    Optional fields:
        details: Field-level validation errors
        doc: Link to error documentation
        correlation: Correlation ID for distributed tracing
        locale: Locale for localized messages
    """

    code: str
    message: str
    retryable: bool
    request_id: str
    details: list[ErrorDetail] = field(default_factory=list)
    doc: str | None = None
    correlation: str | None = None
    locale: str = "en-US"

    def to_dict(self) -> dict[str, Any]:
        """Convert to API response format per ERR-002 §3."""
        envelope = {
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
                "request_id": self.request_id,
            }
        }

        if self.details:
            envelope["error"]["details"] = [
                {
                    "field": d.field,
                    "code": d.code,
                    "message": d.message,
                    **({"value": d.value} if d.value is not None else {}),
                }
                for d in self.details
            ]

        if self.doc:
            envelope["error"]["doc"] = self.doc

        if self.correlation:
            envelope["error"]["correlation"] = self.correlation

        if self.locale != "en-US":
            envelope["error"]["locale"] = self.locale

        return envelope


# =============================================================================
# ERROR REGISTRY ENTRY (ERR-002 §4)
# =============================================================================


@dataclass
class ErrorRegistryEntry:
    """
    Error registry entry per ERR-002 §4.

    Defines a registered error code with metadata.

    Attributes:
        code: Unique error code (e.g., "validation_error")
        category: Error category
        http_status: HTTP status code
        message_template: Default message template
        retryable: Whether error is retryable
        doc_url: Documentation URL
        severity: Default severity level
    """

    code: str
    category: ErrorCategory
    http_status: int
    message_template: str
    retryable: bool
    doc_url: str
    severity: ErrorSeverity = ErrorSeverity.ERROR


# =============================================================================
# DEFAULT CONFIGURATIONS
# =============================================================================


DEFAULT_RETRY_CONFIG = RetryConfig()

DEFAULT_CIRCUIT_BREAKER_CONFIG = CircuitBreakerConfig()


# =============================================================================
# ERROR REGISTRY (ERR-002 §4)
# =============================================================================


# Standard error codes registry
ERROR_REGISTRY: dict[str, ErrorRegistryEntry] = {
    "bad_request": ErrorRegistryEntry(
        code="bad_request",
        category=ErrorCategory.VALIDATION,
        http_status=400,
        message_template="The request was malformed or contained invalid data",
        retryable=False,
        doc_url="https://docs.synara.io/errors#bad_request",
        severity=ErrorSeverity.WARNING,
    ),
    "validation_error": ErrorRegistryEntry(
        code="validation_error",
        category=ErrorCategory.VALIDATION,
        http_status=422,
        message_template="Request validation failed: {details}",
        retryable=False,
        doc_url="https://docs.synara.io/errors#validation_error",
        severity=ErrorSeverity.WARNING,
    ),
    "unauthorized": ErrorRegistryEntry(
        code="unauthorized",
        category=ErrorCategory.AUTHENTICATION,
        http_status=401,
        message_template="Authentication required",
        retryable=False,
        doc_url="https://docs.synara.io/errors#unauthorized",
        severity=ErrorSeverity.WARNING,
    ),
    "forbidden": ErrorRegistryEntry(
        code="forbidden",
        category=ErrorCategory.AUTHORIZATION,
        http_status=403,
        message_template="You do not have permission to perform this action",
        retryable=False,
        doc_url="https://docs.synara.io/errors#forbidden",
        severity=ErrorSeverity.WARNING,
    ),
    "not_found": ErrorRegistryEntry(
        code="not_found",
        category=ErrorCategory.NOT_FOUND,
        http_status=404,
        message_template="The requested resource was not found",
        retryable=False,
        doc_url="https://docs.synara.io/errors#not_found",
        severity=ErrorSeverity.INFO,
    ),
    "conflict": ErrorRegistryEntry(
        code="conflict",
        category=ErrorCategory.CONFLICT,
        http_status=409,
        message_template="The request conflicts with the current state",
        retryable=False,
        doc_url="https://docs.synara.io/errors#conflict",
        severity=ErrorSeverity.WARNING,
    ),
    "rate_limit_exceeded": ErrorRegistryEntry(
        code="rate_limit_exceeded",
        category=ErrorCategory.RATE_LIMIT,
        http_status=429,
        message_template="Rate limit exceeded. Please retry after {retry_after} seconds",
        retryable=True,
        doc_url="https://docs.synara.io/errors#rate_limit_exceeded",
        severity=ErrorSeverity.WARNING,
    ),
    "internal_error": ErrorRegistryEntry(
        code="internal_error",
        category=ErrorCategory.SYSTEM,
        http_status=500,
        message_template="An unexpected error occurred",
        retryable=True,
        doc_url="https://docs.synara.io/errors#internal_error",
        severity=ErrorSeverity.ERROR,
    ),
    "service_unavailable": ErrorRegistryEntry(
        code="service_unavailable",
        category=ErrorCategory.DEPENDENCY,
        http_status=503,
        message_template="The service is temporarily unavailable",
        retryable=True,
        doc_url="https://docs.synara.io/errors#service_unavailable",
        severity=ErrorSeverity.ERROR,
    ),
    "gateway_timeout": ErrorRegistryEntry(
        code="gateway_timeout",
        category=ErrorCategory.TIMEOUT,
        http_status=504,
        message_template="The request timed out",
        retryable=True,
        doc_url="https://docs.synara.io/errors#gateway_timeout",
        severity=ErrorSeverity.ERROR,
    ),
    "database_error": ErrorRegistryEntry(
        code="database_error",
        category=ErrorCategory.DATABASE,
        http_status=500,
        message_template="A database error occurred",
        retryable=True,
        doc_url="https://docs.synara.io/errors#database_error",
        severity=ErrorSeverity.ERROR,
    ),
    "dependency_error": ErrorRegistryEntry(
        code="dependency_error",
        category=ErrorCategory.DEPENDENCY,
        http_status=502,
        message_template="An external service returned an error",
        retryable=True,
        doc_url="https://docs.synara.io/errors#dependency_error",
        severity=ErrorSeverity.ERROR,
    ),
}
