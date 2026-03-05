"""
Synara Error Module (ERR-001/002)
=================================

Centralized error handling for the Synara QMS platform.

Standard:     ERR-001 (Error Handling and Recovery)
              ERR-002 (Canonical Error Envelope and Registry)
Compliance:   ISO 27001 A.12.4, SOC 2 CC7.2
Version:      1.0.0

Overview
--------
This module provides:

1. **SynaraError Exception Hierarchy** (ERR-001 §5)
   - Base SynaraError class for all Synara exceptions
   - Specialized subclasses: ValidationError, AuthenticationError,
     AuthorizationError, NotFoundError, ConflictError, RateLimitError,
     DependencyError, DatabaseError, TimeoutError, SystemError

2. **Error Types and Enums** (ERR-001 §3)
   - ErrorSeverity: DEBUG, INFO, WARNING, ERROR, CRITICAL
   - ErrorCategory: VALIDATION, AUTHENTICATION, etc.
   - ErrorContext: Required context for errors

3. **Canonical Error Envelope** (ERR-002 §3)
   - ErrorEnvelope: Standard API error response format
   - ErrorDetail: Field-level validation error details
   - ErrorRegistryEntry: Registered error code metadata

4. **Retry Utilities** (ERR-001 §7)
   - ExponentialBackoff: Backoff calculator with jitter
   - retry decorator: Automatic retry with backoff
   - RetryConfig: Retry configuration

5. **Circuit Breaker** (ERR-001 §7.3)
   - CircuitBreaker: Circuit breaker pattern implementation
   - CircuitBreakerRegistry: Global circuit breaker registry
   - with_circuit_breaker decorator: Decorator for protection

6. **Bulkhead Pattern** (ERR-001 §7.2)
   - Bulkhead: Concurrency limiter
   - BulkheadFullError: Exception when bulkhead is full

Usage Examples
--------------

Raising Errors:
    >>> from syn.err import ValidationError, NotFoundError
    >>>
    >>> # Validation error with field details
    >>> raise ValidationError(
    ...     message="Invalid CAPA ID format",
    ...     field="capa_id",
    ...     expected="UUID v4",
    ...     received="not-a-uuid"
    ... )
    >>>
    >>> # Not found error
    >>> raise NotFoundError(
    ...     message="CAPA not found",
    ...     resource_type="CAPA",
    ...     resource_id="550e8400-e29b-41d4-a716-446655440000"
    ... )

Retry with Backoff:
    >>> from syn.err import retry, RetryConfig
    >>>
    >>> @retry(max_attempts=3, base_delay_ms=1000)
    ... def call_external_service():
    ...     return requests.get(url)

Circuit Breaker:
    >>> from syn.err import with_circuit_breaker, RecoveryMode
    >>>
    >>> @with_circuit_breaker("document-service")
    ... def get_document(doc_id: str):
    ...     return document_service.get(doc_id)
    >>>
    >>> # With fallback
    >>> @with_circuit_breaker(
    ...     "cache-service",
    ...     fallback=lambda key: None,
    ...     recovery_mode=RecoveryMode.FALLBACK
    ... )
    ... def get_cached_value(key: str):
    ...     return cache.get(key)

Error to API Response:
    >>> from syn.err import SynaraError
    >>>
    >>> try:
    ...     do_something()
    ... except SynaraError as e:
    ...     envelope = e.to_envelope(request_id=request.syn_request_id)
    ...     return JsonResponse(envelope.to_dict(), status=e.http_status)

References
----------
- ERR-001: Error Handling and Recovery Standard
- ERR-002: Canonical Error Envelope and Registry
- AUD-001: Audit Logging Standard (severity alignment)
- API-001: API Design Standard (error responses)
- SBL-001: Synara Bus Loop Standard (layer identification)
"""

from .events import (
    BULKHEAD_ACQUIRED,
    BULKHEAD_FULL,
    CIRCUIT_BREAKER_CLOSED,
    CIRCUIT_BREAKER_HALF_OPEN,
    CIRCUIT_BREAKER_OPENED,
    CIRCUIT_BREAKER_REJECTED,
    ERR_EVENTS_CATALOG,
    ERROR_CRITICAL,
    # Individual events for direct import
    ERROR_LOGGED,
    RATE_LIMIT_EXCEEDED,
    RECOVERY_FALLBACK_USED,
    RECOVERY_RETRY_ATTEMPTED,
    RECOVERY_RETRY_EXHAUSTED,
    RECOVERY_RETRY_SUCCEEDED,
    # Event definitions
    EventDefinition,
    get_event_definition,
    list_event_names,
)
from .exceptions import (
    AuthenticationError,
    AuthorizationError,
    ConflictError,
    DatabaseError,
    DependencyError,
    NotFoundError,
    RateLimitError,
    # Base exception
    SynaraError,
    SystemError,
    TimeoutError,
    # Exception subclasses
    ValidationError,
    create_error_from_code,
    # Utility functions
    wrap_exception,
)
from .retry import (
    # Bulkhead
    Bulkhead,
    BulkheadFullError,
    # Circuit breaker
    CircuitBreaker,
    CircuitBreakerMetrics,
    CircuitBreakerOpenError,
    CircuitBreakerRegistry,
    # Backoff
    ExponentialBackoff,
    circuit_breaker_registry,
    # Decorators
    retry,
    with_circuit_breaker,
)
from .types import (
    CATEGORY_STATUS_CODES,
    DEFAULT_CIRCUIT_BREAKER_CONFIG,
    DEFAULT_RETRY_CONFIG,
    ERROR_REGISTRY,
    RETRYABLE_CATEGORIES,
    # Constants
    SEVERITY_LEVELS,
    CircuitBreakerConfig,
    CircuitBreakerState,
    ErrorCategory,
    # Dataclasses
    ErrorContext,
    ErrorDetail,
    ErrorEnvelope,
    ErrorRegistryEntry,
    # Enums
    ErrorSeverity,
    RecoveryMode,
    RetryConfig,
    RetryStrategy,
    SystemLayer,
)

__all__ = [
    # ==========================================================================
    # Types and Enums
    # ==========================================================================
    "ErrorSeverity",
    "ErrorCategory",
    "CircuitBreakerState",
    "RetryStrategy",
    "RecoveryMode",
    "SystemLayer",
    # Dataclasses
    "ErrorContext",
    "RetryConfig",
    "CircuitBreakerConfig",
    "ErrorDetail",
    "ErrorEnvelope",
    "ErrorRegistryEntry",
    # Constants
    "SEVERITY_LEVELS",
    "CATEGORY_STATUS_CODES",
    "RETRYABLE_CATEGORIES",
    "ERROR_REGISTRY",
    "DEFAULT_RETRY_CONFIG",
    "DEFAULT_CIRCUIT_BREAKER_CONFIG",
    # ==========================================================================
    # Exceptions
    # ==========================================================================
    "SynaraError",
    "ValidationError",
    "AuthenticationError",
    "AuthorizationError",
    "NotFoundError",
    "ConflictError",
    "RateLimitError",
    "DependencyError",
    "DatabaseError",
    "TimeoutError",
    "SystemError",
    # Exception utilities
    "wrap_exception",
    "create_error_from_code",
    # ==========================================================================
    # Retry and Circuit Breaker
    # ==========================================================================
    "ExponentialBackoff",
    "CircuitBreaker",
    "CircuitBreakerMetrics",
    "CircuitBreakerOpenError",
    "CircuitBreakerRegistry",
    "circuit_breaker_registry",
    "Bulkhead",
    "BulkheadFullError",
    # Decorators
    "retry",
    "with_circuit_breaker",
    # ==========================================================================
    # Events
    # ==========================================================================
    "EventDefinition",
    "ERR_EVENTS_CATALOG",
    "get_event_definition",
    "list_event_names",
    "ERROR_LOGGED",
    "ERROR_CRITICAL",
    "RECOVERY_RETRY_ATTEMPTED",
    "RECOVERY_RETRY_SUCCEEDED",
    "RECOVERY_RETRY_EXHAUSTED",
    "RECOVERY_FALLBACK_USED",
    "CIRCUIT_BREAKER_OPENED",
    "CIRCUIT_BREAKER_HALF_OPEN",
    "CIRCUIT_BREAKER_CLOSED",
    "CIRCUIT_BREAKER_REJECTED",
    "BULKHEAD_FULL",
    "BULKHEAD_ACQUIRED",
    "RATE_LIMIT_EXCEEDED",
]


# Module version
__version__ = "1.0.0"

# Standard references
__standards__ = ["ERR-001", "ERR-002"]

# Compliance references
__compliance__ = ["ISO 27001 A.12.4", "SOC 2 CC7.2", "SOC 2 CC9.1"]
