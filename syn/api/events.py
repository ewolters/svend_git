"""
API events catalog for API-001/002 compliant surface operations.

Standard: API-001 §17, API-002 §4, EVT-001 §5
Compliance: NIST SP 800-53 SC-8, AU-2 / ISO 27001 A.13.1, A.12.4

Events:
- Request lifecycle: received, validated, processed, completed
- Authentication: token_validated, token_expired, unauthorized
- Rate limiting: limit_checked, limit_exceeded, burst_exceeded
- Idempotency: key_validated, key_replayed, key_conflict
- Error handling: client_error, server_error, validation_error
- Deprecation: deprecated_endpoint_called, sunset_warning
- Runtime: health_check, ready_check, info_requested
"""

import uuid
from datetime import datetime
from typing import Any

# =============================================================================
# Event Catalog (EVT-001 §5, API-001 §17)
# =============================================================================

API_EVENTS = {
    # -------------------------------------------------------------------------
    # Request Lifecycle Events (API-002 §4)
    # -------------------------------------------------------------------------
    "api.request.received": {
        "description": "HTTP request received by API gateway",
        "category": "request",
        "severity": "info",
        "payload_schema": {
            "request_id": "str",
            "method": "str",
            "path": "str",
            "tenant_id": "UUID | None",
            "client_ip": "str",
            "user_agent": "str",
        },
    },
    "api.request.validated": {
        "description": "Request passed all validation checks",
        "category": "request",
        "severity": "info",
        "payload_schema": {
            "request_id": "str",
            "method": "str",
            "path": "str",
            "validation_duration_ms": "int",
        },
    },
    "api.request.processed": {
        "description": "Request processing completed",
        "category": "request",
        "severity": "info",
        "payload_schema": {
            "request_id": "str",
            "method": "str",
            "path": "str",
            "status_code": "int",
            "duration_ms": "int",
        },
    },
    "api.request.completed": {
        "description": "Response sent to client",
        "category": "request",
        "severity": "info",
        "payload_schema": {
            "request_id": "str",
            "status_code": "int",
            "response_size_bytes": "int",
            "total_duration_ms": "int",
        },
    },
    # -------------------------------------------------------------------------
    # Authentication Events (API-002 §19, SEC-001)
    # -------------------------------------------------------------------------
    "api.auth.token_validated": {
        "description": "Authentication token validated successfully",
        "category": "auth",
        "severity": "info",
        "payload_schema": {
            "request_id": "str",
            "user_id": "str",
            "tenant_id": "UUID",
            "token_type": "str",
        },
    },
    "api.auth.token_expired": {
        "description": "Authentication token has expired",
        "category": "auth",
        "severity": "warning",
        "payload_schema": {
            "request_id": "str",
            "token_type": "str",
            "expired_at": "datetime",
        },
    },
    "api.auth.unauthorized": {
        "description": "Authentication failed - 401 returned",
        "category": "auth",
        "severity": "warning",
        "audit": True,
        "payload_schema": {
            "request_id": "str",
            "path": "str",
            "reason": "str",
            "client_ip": "str",
        },
    },
    "api.auth.forbidden": {
        "description": "Authorization failed - 403 returned",
        "category": "auth",
        "severity": "warning",
        "audit": True,
        "payload_schema": {
            "request_id": "str",
            "path": "str",
            "user_id": "str",
            "required_permission": "str",
        },
    },
    # -------------------------------------------------------------------------
    # Rate Limiting Events (API-002 §19.2)
    # -------------------------------------------------------------------------
    "api.rate_limit.checked": {
        "description": "Rate limit checked for request",
        "category": "rate_limit",
        "severity": "info",
        "payload_schema": {
            "request_id": "str",
            "tenant_id": "UUID",
            "limit": "int",
            "remaining": "int",
            "reset_at": "datetime",
        },
    },
    "api.rate_limit.exceeded": {
        "description": "Rate limit exceeded - 429 returned",
        "category": "rate_limit",
        "severity": "warning",
        "payload_schema": {
            "request_id": "str",
            "tenant_id": "UUID",
            "limit": "int",
            "retry_after_seconds": "int",
        },
    },
    "api.rate_limit.burst_exceeded": {
        "description": "Burst rate limit exceeded",
        "category": "rate_limit",
        "severity": "warning",
        "payload_schema": {
            "request_id": "str",
            "tenant_id": "UUID",
            "burst_limit": "int",
            "current_burst": "int",
        },
    },
    # -------------------------------------------------------------------------
    # Idempotency Events (API-002 §9)
    # -------------------------------------------------------------------------
    "api.idempotency.key_validated": {
        "description": "Idempotency key validated and stored",
        "category": "idempotency",
        "severity": "info",
        "payload_schema": {
            "request_id": "str",
            "idempotency_key": "str",
            "tenant_id": "UUID",
            "expires_at": "datetime",
        },
    },
    "api.idempotency.key_replayed": {
        "description": "Idempotency key found - returning cached response",
        "category": "idempotency",
        "severity": "info",
        "payload_schema": {
            "request_id": "str",
            "idempotency_key": "str",
            "original_request_id": "str",
            "cached_status": "int",
        },
    },
    "api.idempotency.key_conflict": {
        "description": "Idempotency key used with different payload - 409 returned",
        "category": "idempotency",
        "severity": "warning",
        "payload_schema": {
            "request_id": "str",
            "idempotency_key": "str",
            "original_request_id": "str",
        },
    },
    # -------------------------------------------------------------------------
    # Pagination Events (API-002 §7)
    # -------------------------------------------------------------------------
    "api.pagination.cursor_created": {
        "description": "Cursor created for paginated response",
        "category": "pagination",
        "severity": "info",
        "payload_schema": {
            "request_id": "str",
            "cursor": "str",
            "page_size": "int",
            "total_estimate": "int",
        },
    },
    "api.pagination.cursor_used": {
        "description": "Cursor used to retrieve next page",
        "category": "pagination",
        "severity": "info",
        "payload_schema": {
            "request_id": "str",
            "cursor": "str",
            "items_returned": "int",
        },
    },
    "api.pagination.cursor_invalid": {
        "description": "Invalid or expired cursor provided",
        "category": "pagination",
        "severity": "warning",
        "payload_schema": {
            "request_id": "str",
            "cursor": "str",
            "reason": "str",
        },
    },
    # -------------------------------------------------------------------------
    # Error Handling Events (API-002 §10)
    # -------------------------------------------------------------------------
    "api.error.client_error": {
        "description": "Client error occurred (4xx)",
        "category": "error",
        "severity": "warning",
        "payload_schema": {
            "request_id": "str",
            "status_code": "int",
            "error_code": "str",
            "message": "str",
            "path": "str",
        },
    },
    "api.error.server_error": {
        "description": "Server error occurred (5xx)",
        "category": "error",
        "severity": "error",
        "payload_schema": {
            "request_id": "str",
            "status_code": "int",
            "error_code": "str",
            "message": "str",
            "path": "str",
            "exception_type": "str",
        },
    },
    "api.error.validation_error": {
        "description": "Request validation failed",
        "category": "error",
        "severity": "warning",
        "payload_schema": {
            "request_id": "str",
            "field_errors": "list[dict]",
            "path": "str",
        },
    },
    # -------------------------------------------------------------------------
    # Deprecation Events (API-002 §5.2)
    # -------------------------------------------------------------------------
    "api.deprecation.endpoint_called": {
        "description": "Deprecated endpoint was called",
        "category": "deprecation",
        "severity": "warning",
        "payload_schema": {
            "request_id": "str",
            "path": "str",
            "api_version": "str",
            "sunset_date": "datetime",
            "migration_url": "str",
        },
    },
    "api.deprecation.version_sunset": {
        "description": "API version approaching sunset date",
        "category": "deprecation",
        "severity": "warning",
        "payload_schema": {
            "api_version": "str",
            "sunset_date": "datetime",
            "days_remaining": "int",
            "active_clients": "int",
        },
    },
    # -------------------------------------------------------------------------
    # Runtime Operation Events (API-002 §17)
    # -------------------------------------------------------------------------
    "api.runtime.health_check": {
        "description": "Health check endpoint called",
        "category": "runtime",
        "severity": "info",
        "payload_schema": {
            "endpoint": "str",
            "status": "str",
            "duration_ms": "int",
            "components": "dict",
        },
    },
    "api.runtime.ready_check": {
        "description": "Readiness check endpoint called",
        "category": "runtime",
        "severity": "info",
        "payload_schema": {
            "status": "str",
            "duration_ms": "int",
            "checks": "dict",
        },
    },
    "api.runtime.info_requested": {
        "description": "Service info endpoint called",
        "category": "runtime",
        "severity": "info",
        "payload_schema": {
            "request_id": "str",
            "service": "str",
            "version": "str",
        },
    },
    # -------------------------------------------------------------------------
    # Circuit Breaker Events (API-002 §20)
    # -------------------------------------------------------------------------
    "api.circuit_breaker.opened": {
        "description": "Circuit breaker opened due to failures",
        "category": "resilience",
        "severity": "warning",
        "payload_schema": {
            "service": "str",
            "endpoint": "str",
            "failure_count": "int",
            "threshold": "int",
        },
    },
    "api.circuit_breaker.closed": {
        "description": "Circuit breaker closed after recovery",
        "category": "resilience",
        "severity": "info",
        "payload_schema": {
            "service": "str",
            "endpoint": "str",
            "recovery_duration_seconds": "int",
        },
    },
    "api.circuit_breaker.half_open": {
        "description": "Circuit breaker in half-open state testing recovery",
        "category": "resilience",
        "severity": "info",
        "payload_schema": {
            "service": "str",
            "endpoint": "str",
            "test_requests": "int",
        },
    },
    # -------------------------------------------------------------------------
    # Governance Events (GOV-001 §5)
    # -------------------------------------------------------------------------
    "api.governance.alert": {
        "description": "Governance alert for API subsystem",
        "category": "governance",
        "severity": "warning",
        "payload_schema": {
            "alert_type": "str",
            "message": "str",
            "severity": "str",
            "recommended_action": "str",
        },
    },
}


# =============================================================================
# Event Emission (EVT-001 §6)
# =============================================================================


def emit_api_event(
    event_name: str,
    payload: dict[str, Any],
    correlation_id: uuid.UUID | None = None,
    tenant_id: uuid.UUID | str | None = None,
) -> None:
    """
    Emit an API event to the Cortex event bus.

    Standard: EVT-001 §6
    Compliance: NIST SP 800-53 AU-2

    Args:
        event_name: Event name from API_EVENTS catalog
        payload: Event payload matching schema
        correlation_id: Correlation ID for tracing
        tenant_id: Tenant identifier for isolation

    Raises:
        ValueError: If event_name not in catalog
    """
    if event_name not in API_EVENTS:
        raise ValueError(f"Unknown API event: {event_name}. Must be one of: {list(API_EVENTS.keys())}")

    # Build full event payload
    event_payload = {
        "event_name": event_name,
        "correlation_id": str(correlation_id) if correlation_id else str(uuid.uuid4()),
        "tenant_id": str(tenant_id) if tenant_id else None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "source": "syn.api",
        "payload": payload,
    }

    # Import Cortex publisher if available
    try:
        from syn.cortex import publish

        publish(event_name, event_payload)
    except ImportError:
        # Cortex not available, log locally
        import logging

        logger = logging.getLogger("syn.api.events")
        logger.info(f"API event: {event_name}", extra={"payload": event_payload})


# =============================================================================
# Payload Builders (EVT-001 §6.2)
# =============================================================================


def build_request_received_payload(
    request_id: str,
    method: str,
    path: str,
    tenant_id: uuid.UUID | None,
    client_ip: str,
    user_agent: str,
) -> dict[str, Any]:
    """Build payload for api.request.received event."""
    return {
        "request_id": request_id,
        "method": method,
        "path": path,
        "tenant_id": str(tenant_id) if tenant_id else None,
        "client_ip": client_ip,
        "user_agent": user_agent,
    }


def build_request_processed_payload(
    request_id: str,
    method: str,
    path: str,
    status_code: int,
    duration_ms: int,
) -> dict[str, Any]:
    """Build payload for api.request.processed event."""
    return {
        "request_id": request_id,
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration_ms": duration_ms,
    }


def build_auth_unauthorized_payload(
    request_id: str,
    path: str,
    reason: str,
    client_ip: str,
) -> dict[str, Any]:
    """Build payload for api.auth.unauthorized event."""
    return {
        "request_id": request_id,
        "path": path,
        "reason": reason,
        "client_ip": client_ip,
    }


def build_rate_limit_exceeded_payload(
    request_id: str,
    tenant_id: uuid.UUID,
    limit: int,
    retry_after_seconds: int,
) -> dict[str, Any]:
    """Build payload for api.rate_limit.exceeded event."""
    return {
        "request_id": request_id,
        "tenant_id": str(tenant_id),
        "limit": limit,
        "retry_after_seconds": retry_after_seconds,
    }


def build_idempotency_key_replayed_payload(
    request_id: str,
    idempotency_key: str,
    original_request_id: str,
    cached_status: int,
) -> dict[str, Any]:
    """Build payload for api.idempotency.key_replayed event."""
    return {
        "request_id": request_id,
        "idempotency_key": idempotency_key,
        "original_request_id": original_request_id,
        "cached_status": cached_status,
    }


def build_client_error_payload(
    request_id: str,
    status_code: int,
    error_code: str,
    message: str,
    path: str,
) -> dict[str, Any]:
    """Build payload for api.error.client_error event."""
    return {
        "request_id": request_id,
        "status_code": status_code,
        "error_code": error_code,
        "message": message,
        "path": path,
    }


def build_server_error_payload(
    request_id: str,
    status_code: int,
    error_code: str,
    message: str,
    path: str,
    exception_type: str,
) -> dict[str, Any]:
    """Build payload for api.error.server_error event."""
    return {
        "request_id": request_id,
        "status_code": status_code,
        "error_code": error_code,
        "message": message,
        "path": path,
        "exception_type": exception_type,
    }


def build_deprecation_endpoint_called_payload(
    request_id: str,
    path: str,
    api_version: str,
    sunset_date: datetime,
    migration_url: str,
) -> dict[str, Any]:
    """Build payload for api.deprecation.endpoint_called event."""
    return {
        "request_id": request_id,
        "path": path,
        "api_version": api_version,
        "sunset_date": sunset_date.isoformat() + "Z",
        "migration_url": migration_url,
    }


def build_health_check_payload(
    endpoint: str,
    status: str,
    duration_ms: int,
    components: dict[str, Any],
) -> dict[str, Any]:
    """Build payload for api.runtime.health_check event."""
    return {
        "endpoint": endpoint,
        "status": status,
        "duration_ms": duration_ms,
        "components": components,
    }


def build_governance_api_alert_payload(
    alert_type: str,
    message: str,
    severity: str,
    recommended_action: str,
) -> dict[str, Any]:
    """Build payload for api.governance.alert event."""
    return {
        "alert_type": alert_type,
        "message": message,
        "severity": severity,
        "recommended_action": recommended_action,
    }
