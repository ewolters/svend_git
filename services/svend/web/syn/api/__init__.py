"""
Synara API Surface Module (API-001/002)
=======================================

API-001/002 compliant HTTP API surface with standardized headers,
cursor pagination, idempotency, and error handling.

Standard:     API-001 (Design), API-002 (Surface Conventions)
Compliance:   NIST SP 800-53 SC-8/SC-23, ISO 27001 A.13.1, SOC 2 CC6.1
Location:     syn/api/
Version:      1.0.0

Features:
---------
- SynRequestIdMiddleware: Request ID extraction/generation (API-002 §8.1)
- APIHeadersMiddleware: Required header enforcement (API-002 §8.2)
- IdempotencyMiddleware: POST idempotency (API-002 §9)
- ErrorEnvelopeMiddleware: Standard error responses (API-002 §10)
- SynaraCursorPagination: Cursor-based pagination (API-002 §7)
- Events: API lifecycle and error events (API-002 §4)

Middleware Order (recommended):
    MIDDLEWARE = [
        ...
        'syn.api.middleware.SynRequestIdMiddleware',
        'syn.api.middleware.APIHeadersMiddleware',
        'syn.api.middleware.IdempotencyMiddleware',
        'syn.api.middleware.ErrorEnvelopeMiddleware',
        ...
    ]

Pagination Configuration:
    REST_FRAMEWORK = {
        'DEFAULT_PAGINATION_CLASS': 'syn.api.pagination.SynaraCursorPagination',
        'PAGE_SIZE': 50,
    }

Usage:
------
    from syn.api import (
        SynaraCursorPagination,
        emit_api_event,
        API_EVENTS,
    )

    class MyViewSet(ModelViewSet):
        pagination_class = SynaraCursorPagination

        def list(self, request, *args, **kwargs):
            emit_api_event(
                "api.request.received",
                build_request_received_payload(...),
            )
            return super().list(request, *args, **kwargs)
"""

__version__ = "1.0.0"
__standard__ = "API-002"

# =============================================================================
# Middleware (API-002 §8-10)
# =============================================================================

from syn.api.middleware import (
    SynRequestIdMiddleware,
    APIHeadersMiddleware,
    IdempotencyMiddleware,
    ErrorEnvelopeMiddleware,
    HEADER_SYN_REQUEST_ID,
    HEADER_TRACEPARENT,
    HEADER_IDEMPOTENCY_KEY,
)

# =============================================================================
# Pagination (API-002 §7)
# =============================================================================

from syn.api.pagination import (
    SynaraCursorPagination,
    SynaraListPagination,
    create_cursor_response,
    encode_cursor,
    decode_cursor,
    DEFAULT_PAGE_SIZE,
    MIN_PAGE_SIZE,
    MAX_PAGE_SIZE,
)

# =============================================================================
# Events (API-002 §4, EVT-001 §5)
# =============================================================================

from syn.api.events import (
    API_EVENTS,
    emit_api_event,
    build_request_received_payload,
    build_request_processed_payload,
    build_auth_unauthorized_payload,
    build_rate_limit_exceeded_payload,
    build_idempotency_key_replayed_payload,
    build_client_error_payload,
    build_server_error_payload,
    build_deprecation_endpoint_called_payload,
    build_health_check_payload,
    build_governance_api_alert_payload,
)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Version info
    "__version__",
    "__standard__",
    # Middleware
    "SynRequestIdMiddleware",
    "APIHeadersMiddleware",
    "IdempotencyMiddleware",
    "ErrorEnvelopeMiddleware",
    "HEADER_SYN_REQUEST_ID",
    "HEADER_TRACEPARENT",
    "HEADER_IDEMPOTENCY_KEY",
    # Pagination
    "SynaraCursorPagination",
    "SynaraListPagination",
    "create_cursor_response",
    "encode_cursor",
    "decode_cursor",
    "DEFAULT_PAGE_SIZE",
    "MIN_PAGE_SIZE",
    "MAX_PAGE_SIZE",
    # Events
    "API_EVENTS",
    "emit_api_event",
    "build_request_received_payload",
    "build_request_processed_payload",
    "build_auth_unauthorized_payload",
    "build_rate_limit_exceeded_payload",
    "build_idempotency_key_replayed_payload",
    "build_client_error_payload",
    "build_server_error_payload",
    "build_deprecation_endpoint_called_payload",
    "build_health_check_payload",
    "build_governance_api_alert_payload",
]
