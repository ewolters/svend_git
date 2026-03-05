"""
Synara Error Events Catalog (ERR-001/002)
=========================================

Event definitions for error handling, recovery, and circuit breaker events.

Standard:     ERR-001 §8, EVT-001 §4
Compliance:   ISO 27001 A.12.4, SOC 2 CC7.2
Version:      1.0.0

Event Categories
----------------
- error.*: Error occurrence and logging events
- recovery.*: Error recovery and retry events
- circuit_breaker.*: Circuit breaker state change events
- bulkhead.*: Bulkhead pattern events
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# =============================================================================
# EVENT DEFINITIONS (EVT-001 §4)
# =============================================================================


@dataclass
class EventDefinition:
    """Event definition per EVT-001 §4."""

    name: str
    description: str
    payload_schema: dict[str, Any]
    category: str
    severity: str = "INFO"
    compliance_refs: list[str] = field(default_factory=list)


# =============================================================================
# ERROR EVENTS (ERR-001 §8.1)
# =============================================================================


ERROR_LOGGED = EventDefinition(
    name="error.logged",
    description="An error has been logged to the error tracking system",
    category="error",
    severity="ERROR",
    payload_schema={
        "type": "object",
        "required": ["error_code", "error_category", "correlation_id", "message"],
        "properties": {
            "error_code": {
                "type": "string",
                "description": "Machine-readable error code",
            },
            "error_category": {
                "type": "string",
                "enum": [
                    "VALIDATION",
                    "AUTHENTICATION",
                    "AUTHORIZATION",
                    "NOT_FOUND",
                    "CONFLICT",
                    "RATE_LIMIT",
                    "DEPENDENCY",
                    "DATABASE",
                    "TIMEOUT",
                    "SYSTEM",
                ],
                "description": "Error category per ERR-001 §3.1",
            },
            "error_severity": {
                "type": "string",
                "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                "description": "Error severity per ERR-001 §3.2",
            },
            "correlation_id": {
                "type": "string",
                "format": "uuid",
                "description": "Correlation ID for distributed tracing",
            },
            "message": {
                "type": "string",
                "description": "Human-readable error message",
            },
            "http_status": {
                "type": "integer",
                "description": "HTTP status code",
            },
            "retryable": {
                "type": "boolean",
                "description": "Whether the operation can be retried",
            },
            "layer": {
                "type": "string",
                "enum": ["API", "SERVICE", "REPOSITORY", "INTEGRATION", "KERNEL", "CLI", "UI"],
                "description": "System layer where error occurred",
            },
            "operation": {
                "type": "string",
                "description": "Operation that failed",
            },
            "details": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string"},
                        "code": {"type": "string"},
                        "message": {"type": "string"},
                    },
                },
                "description": "Field-level error details",
            },
        },
    },
    compliance_refs=["ISO 27001 A.12.4.1", "SOC 2 CC7.2"],
)

ERROR_CRITICAL = EventDefinition(
    name="error.critical",
    description="A critical error requiring immediate attention has occurred",
    category="error",
    severity="CRITICAL",
    payload_schema={
        "type": "object",
        "required": ["error_code", "correlation_id", "message", "component"],
        "properties": {
            "error_code": {
                "type": "string",
                "description": "Machine-readable error code",
            },
            "correlation_id": {
                "type": "string",
                "format": "uuid",
                "description": "Correlation ID for distributed tracing",
            },
            "message": {
                "type": "string",
                "description": "Human-readable error message",
            },
            "component": {
                "type": "string",
                "description": "Component where critical error occurred",
            },
            "stack_trace": {
                "type": "string",
                "description": "Stack trace (internal use only)",
            },
            "impact": {
                "type": "string",
                "description": "Description of error impact",
            },
        },
    },
    compliance_refs=["ISO 27001 A.12.4.1", "SOC 2 CC7.2", "SOC 2 CC7.3"],
)


# =============================================================================
# RECOVERY EVENTS (ERR-001 §8.2)
# =============================================================================


RECOVERY_RETRY_ATTEMPTED = EventDefinition(
    name="recovery.retry.attempted",
    description="A retry attempt has been initiated for a failed operation",
    category="recovery",
    severity="INFO",
    payload_schema={
        "type": "object",
        "required": ["operation", "attempt", "max_attempts", "delay_ms"],
        "properties": {
            "operation": {
                "type": "string",
                "description": "Operation being retried",
            },
            "attempt": {
                "type": "integer",
                "minimum": 1,
                "description": "Current attempt number",
            },
            "max_attempts": {
                "type": "integer",
                "description": "Maximum allowed attempts",
            },
            "delay_ms": {
                "type": "integer",
                "description": "Delay before this retry in milliseconds",
            },
            "retry_strategy": {
                "type": "string",
                "enum": ["NONE", "IMMEDIATE", "LINEAR", "EXPONENTIAL", "CONSTANT"],
                "description": "Retry strategy being used",
            },
            "error_code": {
                "type": "string",
                "description": "Error code that triggered retry",
            },
            "correlation_id": {
                "type": "string",
                "format": "uuid",
                "description": "Correlation ID for tracing",
            },
        },
    },
    compliance_refs=["ISO 27001 A.17.1.1"],
)

RECOVERY_RETRY_SUCCEEDED = EventDefinition(
    name="recovery.retry.succeeded",
    description="A retry attempt succeeded",
    category="recovery",
    severity="INFO",
    payload_schema={
        "type": "object",
        "required": ["operation", "attempt", "total_duration_ms"],
        "properties": {
            "operation": {
                "type": "string",
                "description": "Operation that was retried",
            },
            "attempt": {
                "type": "integer",
                "description": "Successful attempt number",
            },
            "total_duration_ms": {
                "type": "integer",
                "description": "Total time including all retries",
            },
            "correlation_id": {
                "type": "string",
                "format": "uuid",
                "description": "Correlation ID for tracing",
            },
        },
    },
    compliance_refs=["ISO 27001 A.17.1.1"],
)

RECOVERY_RETRY_EXHAUSTED = EventDefinition(
    name="recovery.retry.exhausted",
    description="All retry attempts have been exhausted",
    category="recovery",
    severity="WARNING",
    payload_schema={
        "type": "object",
        "required": ["operation", "total_attempts", "final_error_code"],
        "properties": {
            "operation": {
                "type": "string",
                "description": "Operation that failed",
            },
            "total_attempts": {
                "type": "integer",
                "description": "Total attempts made",
            },
            "final_error_code": {
                "type": "string",
                "description": "Final error code",
            },
            "final_error_message": {
                "type": "string",
                "description": "Final error message",
            },
            "total_duration_ms": {
                "type": "integer",
                "description": "Total time spent retrying",
            },
            "correlation_id": {
                "type": "string",
                "format": "uuid",
                "description": "Correlation ID for tracing",
            },
        },
    },
    compliance_refs=["ISO 27001 A.17.1.1"],
)

RECOVERY_FALLBACK_USED = EventDefinition(
    name="recovery.fallback.used",
    description="A fallback mechanism was used after primary operation failed",
    category="recovery",
    severity="INFO",
    payload_schema={
        "type": "object",
        "required": ["operation", "fallback_type", "reason"],
        "properties": {
            "operation": {
                "type": "string",
                "description": "Primary operation that failed",
            },
            "fallback_type": {
                "type": "string",
                "enum": ["CACHE", "DEFAULT", "DEGRADED", "SKIP"],
                "description": "Type of fallback used",
            },
            "reason": {
                "type": "string",
                "description": "Reason for fallback",
            },
            "correlation_id": {
                "type": "string",
                "format": "uuid",
                "description": "Correlation ID for tracing",
            },
        },
    },
    compliance_refs=["ISO 27001 A.17.1.2"],
)


# =============================================================================
# CIRCUIT BREAKER EVENTS (ERR-001 §8.3)
# =============================================================================


CIRCUIT_BREAKER_OPENED = EventDefinition(
    name="circuit_breaker.opened",
    description="Circuit breaker has transitioned to OPEN state",
    category="circuit_breaker",
    severity="WARNING",
    payload_schema={
        "type": "object",
        "required": ["circuit_breaker_name", "consecutive_failures", "failure_threshold"],
        "properties": {
            "circuit_breaker_name": {
                "type": "string",
                "description": "Name of the circuit breaker",
            },
            "consecutive_failures": {
                "type": "integer",
                "description": "Number of consecutive failures",
            },
            "failure_threshold": {
                "type": "integer",
                "description": "Configured failure threshold",
            },
            "last_error_code": {
                "type": "string",
                "description": "Error code of last failure",
            },
            "recovery_timeout_ms": {
                "type": "integer",
                "description": "Time until half-open test",
            },
        },
    },
    compliance_refs=["ISO 27001 A.17.1.1", "SOC 2 CC9.1"],
)

CIRCUIT_BREAKER_HALF_OPEN = EventDefinition(
    name="circuit_breaker.half_open",
    description="Circuit breaker has transitioned to HALF_OPEN state for testing",
    category="circuit_breaker",
    severity="INFO",
    payload_schema={
        "type": "object",
        "required": ["circuit_breaker_name", "test_requests_allowed"],
        "properties": {
            "circuit_breaker_name": {
                "type": "string",
                "description": "Name of the circuit breaker",
            },
            "test_requests_allowed": {
                "type": "integer",
                "description": "Number of test requests allowed",
            },
            "time_in_open_ms": {
                "type": "integer",
                "description": "Time spent in OPEN state",
            },
        },
    },
    compliance_refs=["ISO 27001 A.17.1.1"],
)

CIRCUIT_BREAKER_CLOSED = EventDefinition(
    name="circuit_breaker.closed",
    description="Circuit breaker has recovered and transitioned to CLOSED state",
    category="circuit_breaker",
    severity="INFO",
    payload_schema={
        "type": "object",
        "required": ["circuit_breaker_name", "consecutive_successes"],
        "properties": {
            "circuit_breaker_name": {
                "type": "string",
                "description": "Name of the circuit breaker",
            },
            "consecutive_successes": {
                "type": "integer",
                "description": "Successful requests before closing",
            },
            "total_recovery_time_ms": {
                "type": "integer",
                "description": "Total time from OPEN to CLOSED",
            },
        },
    },
    compliance_refs=["ISO 27001 A.17.1.1"],
)

CIRCUIT_BREAKER_REJECTED = EventDefinition(
    name="circuit_breaker.rejected",
    description="Request rejected by open circuit breaker",
    category="circuit_breaker",
    severity="INFO",
    payload_schema={
        "type": "object",
        "required": ["circuit_breaker_name", "circuit_state"],
        "properties": {
            "circuit_breaker_name": {
                "type": "string",
                "description": "Name of the circuit breaker",
            },
            "circuit_state": {
                "type": "string",
                "enum": ["OPEN", "HALF_OPEN"],
                "description": "Current circuit state",
            },
            "rejected_count": {
                "type": "integer",
                "description": "Total rejected requests",
            },
            "correlation_id": {
                "type": "string",
                "format": "uuid",
                "description": "Correlation ID of rejected request",
            },
        },
    },
    compliance_refs=["ISO 27001 A.17.1.1"],
)


# =============================================================================
# BULKHEAD EVENTS (ERR-001 §8.4)
# =============================================================================


BULKHEAD_FULL = EventDefinition(
    name="bulkhead.full",
    description="Bulkhead is at capacity, requests being rejected",
    category="bulkhead",
    severity="WARNING",
    payload_schema={
        "type": "object",
        "required": ["bulkhead_name", "max_concurrent", "rejected_count"],
        "properties": {
            "bulkhead_name": {
                "type": "string",
                "description": "Name of the bulkhead",
            },
            "max_concurrent": {
                "type": "integer",
                "description": "Maximum concurrent requests",
            },
            "active_requests": {
                "type": "integer",
                "description": "Currently active requests",
            },
            "rejected_count": {
                "type": "integer",
                "description": "Total rejected requests",
            },
        },
    },
    compliance_refs=["ISO 27001 A.17.1.2"],
)

BULKHEAD_ACQUIRED = EventDefinition(
    name="bulkhead.acquired",
    description="Bulkhead slot acquired for request",
    category="bulkhead",
    severity="DEBUG",
    payload_schema={
        "type": "object",
        "required": ["bulkhead_name", "active_requests"],
        "properties": {
            "bulkhead_name": {
                "type": "string",
                "description": "Name of the bulkhead",
            },
            "active_requests": {
                "type": "integer",
                "description": "Currently active requests after acquisition",
            },
            "available_slots": {
                "type": "integer",
                "description": "Remaining available slots",
            },
            "wait_time_ms": {
                "type": "integer",
                "description": "Time spent waiting for slot",
            },
        },
    },
    compliance_refs=[],
)


# =============================================================================
# RATE LIMIT EVENTS (ERR-001 §8.5)
# =============================================================================


RATE_LIMIT_EXCEEDED = EventDefinition(
    name="rate_limit.exceeded",
    description="Request rate limit has been exceeded",
    category="rate_limit",
    severity="WARNING",
    payload_schema={
        "type": "object",
        "required": ["limit_name", "limit", "window_seconds", "retry_after_seconds"],
        "properties": {
            "limit_name": {
                "type": "string",
                "description": "Name of the rate limit",
            },
            "limit": {
                "type": "integer",
                "description": "Configured limit",
            },
            "window_seconds": {
                "type": "integer",
                "description": "Time window in seconds",
            },
            "current_count": {
                "type": "integer",
                "description": "Current request count",
            },
            "retry_after_seconds": {
                "type": "integer",
                "description": "Seconds until limit resets",
            },
            "tenant_id": {
                "type": "string",
                "format": "uuid",
                "description": "Tenant ID if tenant-specific limit",
            },
            "user_id": {
                "type": "string",
                "format": "uuid",
                "description": "User ID if user-specific limit",
            },
        },
    },
    compliance_refs=["ISO 27001 A.12.1.3"],
)


# =============================================================================
# EVENT CATALOG
# =============================================================================


ERR_EVENTS_CATALOG: dict[str, EventDefinition] = {
    # Error events
    "error.logged": ERROR_LOGGED,
    "error.critical": ERROR_CRITICAL,
    # Recovery events
    "recovery.retry.attempted": RECOVERY_RETRY_ATTEMPTED,
    "recovery.retry.succeeded": RECOVERY_RETRY_SUCCEEDED,
    "recovery.retry.exhausted": RECOVERY_RETRY_EXHAUSTED,
    "recovery.fallback.used": RECOVERY_FALLBACK_USED,
    # Circuit breaker events
    "circuit_breaker.opened": CIRCUIT_BREAKER_OPENED,
    "circuit_breaker.half_open": CIRCUIT_BREAKER_HALF_OPEN,
    "circuit_breaker.closed": CIRCUIT_BREAKER_CLOSED,
    "circuit_breaker.rejected": CIRCUIT_BREAKER_REJECTED,
    # Bulkhead events
    "bulkhead.full": BULKHEAD_FULL,
    "bulkhead.acquired": BULKHEAD_ACQUIRED,
    # Rate limit events
    "rate_limit.exceeded": RATE_LIMIT_EXCEEDED,
}


def get_event_definition(event_name: str) -> EventDefinition | None:
    """Get event definition by name."""
    return ERR_EVENTS_CATALOG.get(event_name)


def list_event_names() -> list[str]:
    """List all registered event names."""
    return list(ERR_EVENTS_CATALOG.keys())
