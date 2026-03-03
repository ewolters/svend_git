"""
API middleware for API-001/002 compliant HTTP handling.

Standard: API-002 §8, §9, ERR-001/002, POL-002 §8
Compliance: NIST SP 800-53 SC-8, SC-23 / ISO 27001 A.13.1

Middleware:
- SynRequestIdMiddleware: Extract/generate Syn-Request-Id header
- IdempotencyMiddleware: Validate and enforce Idempotency-Key
- APIHeadersMiddleware: Enforce required request/response headers
- ErrorEnvelopeMiddleware: Standardize error responses per ERR-002
"""

import hashlib
import json
import logging
import re
import time
import uuid
from typing import Callable, Set

from django.conf import settings
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.utils import timezone

from syn.err import SynaraError, ErrorEnvelope

logger = logging.getLogger(__name__)


# =============================================================================
# ERROR REDACTION (POL-002 §8)
# =============================================================================

# POL-002 §8: Patterns to redact from error messages and stack traces
SENSITIVE_ERROR_PATTERNS: Set[str] = {
    r"password[=:]\s*['\"]?[^'\"\s]+['\"]?",
    r"secret[=:]\s*['\"]?[^'\"\s]+['\"]?",
    r"api[_-]?key[=:]\s*['\"]?[^'\"\s]+['\"]?",
    r"token[=:]\s*['\"]?[^'\"\s]+['\"]?",
    r"auth[=:]\s*['\"]?[^'\"\s]+['\"]?",
    r"credential[s]?[=:]\s*['\"]?[^'\"\s]+['\"]?",
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
    r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
    r"\b\d{13,19}\b",  # Credit card
    r"Bearer\s+[A-Za-z0-9\-._~+/]+=*",  # Bearer token
}

# Environment variable patterns to never log
ENV_VAR_PATTERNS: Set[str] = {
    r"DATABASE_URL",
    r"SECRET_KEY",
    r"AWS_SECRET",
    r"REDIS_URL",
    r"POSTGRES_PASSWORD",
    r"DB_PASSWORD",
}


def redact_error_message(message: str) -> str:
    """
    Redact sensitive patterns from error messages.

    Standard: POL-002 §8
    - Error messages MUST NOT include sensitive field values in plain text
    - Error envelopes MUST apply redaction per ERR-002 scrubbing rules
    """
    redacted = message
    for pattern in SENSITIVE_ERROR_PATTERNS:
        redacted = re.sub(pattern, "***REDACTED***", redacted, flags=re.IGNORECASE)
    for pattern in ENV_VAR_PATTERNS:
        redacted = re.sub(f"{pattern}=[^\\s]+", f"{pattern}=***", redacted, flags=re.IGNORECASE)
    return redacted


def redact_exception_for_logging(exception: Exception) -> str:
    """
    Create a redacted version of exception for logging.

    Standard: POL-002 §8
    - Stacktraces MUST be scrubbed of sensitive values
    - Environment variables MUST NOT appear in logs
    """
    # Get exception type and message only (no full traceback with locals)
    exc_type = type(exception).__name__
    exc_message = redact_error_message(str(exception))
    return f"{exc_type}: {exc_message}"

# =============================================================================
# Constants (API-002 §8-9)
# =============================================================================

# Header names per API-002 §8.1
HEADER_SYN_REQUEST_ID = "Syn-Request-Id"
HEADER_TRACEPARENT = "traceparent"
HEADER_TRACESTATE = "tracestate"
HEADER_IDEMPOTENCY_KEY = "Idempotency-Key"
HEADER_CONTENT_TYPE = "Content-Type"
HEADER_ACCEPT = "Accept"

# Rate limit headers per API-002 §19.2
HEADER_RATELIMIT_LIMIT = "X-RateLimit-Limit"
HEADER_RATELIMIT_REMAINING = "X-RateLimit-Remaining"
HEADER_RATELIMIT_RESET = "X-RateLimit-Reset"
HEADER_RETRY_AFTER = "Retry-After"

# Idempotency settings per API-002 §9
IDEMPOTENCY_TTL_HOURS = 24
IDEMPOTENCY_KEY_PATTERN = r"^[0-9A-HJKMNP-TV-Z]{26}$"  # ULID pattern

# W3C traceparent pattern: version-traceid-spanid-traceflags
# version: 2 hex chars, traceid: 32 hex chars, spanid: 16 hex chars, flags: 2 hex chars
TRACEPARENT_PATTERN = r"^[0-9a-f]{2}-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$"

# Exempt paths (no header validation required)
EXEMPT_PATHS = [
    "/admin/",
    "/health/",
    "/ready",
    "/live",
    "/metrics",
    "/static/",
    "/media/",
]


def _is_exempt_path(path: str) -> bool:
    """Check if path is exempt from header validation."""
    return any(path.startswith(exempt) for exempt in EXEMPT_PATHS)


def _generate_request_id() -> str:
    """
    Generate a new request ID in ULID format.

    Returns:
        26-character ULID string
    """
    try:
        import ulid
        return str(ulid.new())
    except ImportError:
        # Fallback to UUID-based format if ulid not available
        return f"req_{uuid.uuid4().hex[:22]}"


# =============================================================================
# Syn-Request-Id Middleware (API-002 §8.1)
# =============================================================================

class SynRequestIdMiddleware:
    """
    Middleware to extract, validate, or generate Syn-Request-Id header.

    Standard: API-002 §8.1
    Compliance: NIST SP 800-53 SC-23 (Session Authenticity)

    Features:
    - Extracts Syn-Request-Id from request header
    - Generates new ID if not provided
    - Extracts and validates W3C traceparent header
    - Echoes ID in response header
    - Stores ID on request object for downstream use
    """

    def __init__(self, get_response: Callable):
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        import re

        # Extract or generate request ID
        request_id = request.headers.get(HEADER_SYN_REQUEST_ID)

        if not request_id:
            request_id = _generate_request_id()
            logger.debug(f"Generated request ID: {request_id}")

        # Store on request for downstream access
        request.syn_request_id = request_id

        # Extract W3C traceparent if provided (API-002 §8.1)
        traceparent = request.headers.get(HEADER_TRACEPARENT)
        if traceparent and re.match(TRACEPARENT_PATTERN, traceparent):
            # Parse: version-traceid-spanid-traceflags
            parts = traceparent.split("-")
            if len(parts) == 4:
                request.syn_trace_id = parts[1]
                request.syn_span_id = parts[2]
                request.syn_trace_flags = parts[3]
                logger.debug(f"Extracted trace context: trace_id={parts[1][:8]}...")
        else:
            request.syn_trace_id = None
            request.syn_span_id = None
            request.syn_trace_flags = None

        # Extract tracestate if provided
        request.syn_tracestate = request.headers.get(HEADER_TRACESTATE)

        # Also store start time for duration tracking
        request.syn_start_time = time.time()

        # Process request
        response = self.get_response(request)

        # Echo request ID in response
        response[HEADER_SYN_REQUEST_ID] = request_id

        # Echo traceparent if present
        if traceparent:
            response[HEADER_TRACEPARENT] = traceparent

        return response


# =============================================================================
# API Headers Middleware (API-002 §8.2)
# =============================================================================

class APIHeadersMiddleware:
    """
    Middleware to enforce required API headers.

    Standard: API-002 §8.1-8.2
    Compliance: ISO 27001 A.13.1 (Network Security)

    Features:
    - Validates Accept header on requests
    - Enforces Content-Type with charset on responses
    - Adds Vary header for caching
    - Tracks request duration
    """

    def __init__(self, get_response: Callable):
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        # Skip validation for exempt paths
        if _is_exempt_path(request.path):
            return self.get_response(request)

        # Validate Accept header for API paths
        if request.path.startswith("/api/"):
            accept = request.headers.get(HEADER_ACCEPT, "")
            if not accept or accept == "*/*":
                # Allow */* but prefer explicit JSON
                pass
            elif "application/json" not in accept and "application/vnd.synara" not in accept:
                return JsonResponse(
                    {
                        "error": {
                            "code": "unsupported_media_type",
                            "message": "Accept header must include application/json",
                            "retryable": False,
                            "request_id": getattr(request, "syn_request_id", None),
                        }
                    },
                    status=406,
                )

        # Process request
        response = self.get_response(request)

        # Ensure Content-Type includes charset
        content_type = response.get(HEADER_CONTENT_TYPE, "")
        if content_type and "charset" not in content_type.lower():
            if "application/json" in content_type:
                response[HEADER_CONTENT_TYPE] = "application/json; charset=utf-8"

        # Add Vary header for proper caching
        vary = response.get("Vary", "")
        if vary:
            response["Vary"] = f"{vary}, Accept, Accept-Encoding"
        else:
            response["Vary"] = "Accept, Accept-Encoding"

        # Add duration header if start time available
        start_time = getattr(request, "syn_start_time", None)
        if start_time:
            duration_ms = int((time.time() - start_time) * 1000)
            response["X-Response-Time"] = f"{duration_ms}ms"

        return response


# =============================================================================
# Idempotency Middleware (API-002 §9)
# =============================================================================

class IdempotencyMiddleware:
    """
    Middleware to enforce idempotency for POST requests.

    Standard: API-002 §9
    Compliance: NIST SP 800-53 SC-8

    Features:
    - Validates Idempotency-Key header on POST requests
    - Returns cached response for duplicate keys
    - Returns 409 Conflict for key reuse with different payload
    - 24-hour TTL for idempotency keys
    """

    def __init__(self, get_response: Callable):
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        # Only apply to POST requests on API paths
        if request.method != "POST" or not request.path.startswith("/api/"):
            return self.get_response(request)

        # Skip exempt paths
        if _is_exempt_path(request.path):
            return self.get_response(request)

        # Check for Idempotency-Key header
        idempotency_key = request.headers.get(HEADER_IDEMPOTENCY_KEY)

        if not idempotency_key:
            # Allow POST without idempotency key but log warning
            logger.warning(
                f"POST request without Idempotency-Key: {request.path}",
                extra={"request_id": getattr(request, "syn_request_id", None)},
            )
            return self.get_response(request)

        # Compute payload hash for conflict detection
        try:
            body = request.body
            payload_hash = hashlib.sha256(body).hexdigest()
        except Exception as e:
            logger.warning(
                f"[API-002] Failed to compute payload hash: {type(e).__name__}: {e}",
                extra={"request_id": getattr(request, "syn_request_id", None)},
            )
            payload_hash = ""

        # Check for existing idempotency key
        cached_response = self._get_cached_response(
            idempotency_key,
            getattr(request, "tenant_id", None),
        )

        if cached_response:
            # Check for payload conflict
            if cached_response.get("payload_hash") != payload_hash:
                return JsonResponse(
                    {
                        "error": {
                            "code": "idempotency_conflict",
                            "message": "Idempotency-Key already used with different request body",
                            "retryable": False,
                            "request_id": getattr(request, "syn_request_id", None),
                            "original_request_id": cached_response.get("request_id"),
                        }
                    },
                    status=409,
                )

            # Return cached response (replay)
            logger.info(
                f"Idempotency key replay: {idempotency_key}",
                extra={"request_id": getattr(request, "syn_request_id", None)},
            )
            return JsonResponse(
                cached_response.get("response_body", {}),
                status=cached_response.get("status_code", 200),
            )

        # Process request
        response = self.get_response(request)

        # Cache response for future replays (only on success)
        if 200 <= response.status_code < 300:
            self._cache_response(
                idempotency_key,
                getattr(request, "tenant_id", None),
                getattr(request, "syn_request_id", None),
                payload_hash,
                response,
            )

        return response

    def _get_cached_response(
        self,
        idempotency_key: str,
        tenant_id,
    ) -> dict | None:
        """
        Retrieve cached response for idempotency key.

        Uses Django cache or database depending on configuration.
        """
        try:
            from django.core.cache import cache
            cache_key = f"idempotency:{tenant_id}:{idempotency_key}"
            return cache.get(cache_key)
        except Exception as e:
            logger.error(f"Failed to retrieve idempotency cache: {e}")
            return None

    def _cache_response(
        self,
        idempotency_key: str,
        tenant_id,
        request_id: str,
        payload_hash: str,
        response: HttpResponse,
    ) -> None:
        """
        Cache response for idempotency key.
        """
        try:
            from django.core.cache import cache

            # Extract response body
            try:
                response_body = json.loads(response.content.decode("utf-8"))
            except Exception:
                response_body = {}

            cache_key = f"idempotency:{tenant_id}:{idempotency_key}"
            cache_data = {
                "request_id": request_id,
                "payload_hash": payload_hash,
                "status_code": response.status_code,
                "response_body": response_body,
                "cached_at": timezone.now().isoformat(),
            }

            # 24-hour TTL per API-002 §9
            cache.set(cache_key, cache_data, timeout=IDEMPOTENCY_TTL_HOURS * 3600)
        except Exception as e:
            logger.error(f"Failed to cache idempotency response: {e}")


# =============================================================================
# Error Envelope Middleware (API-002 §10)
# =============================================================================

class ErrorEnvelopeMiddleware:
    """
    Middleware to standardize error responses.

    Standard: API-002 §10, ERR-002
    Compliance: ISO 27001 A.12.4

    Features:
    - Wraps error responses in standard envelope
    - Adds request_id to all errors
    - Adds retryable flag based on status code
    - Adds doc link for error codes
    """

    # Status codes that are retryable
    RETRYABLE_STATUSES = {408, 429, 500, 502, 503, 504}

    # Error code mapping
    STATUS_TO_CODE = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        405: "method_not_allowed",
        406: "not_acceptable",
        408: "request_timeout",
        409: "conflict",
        410: "gone",
        413: "payload_too_large",
        415: "unsupported_media_type",
        422: "unprocessable_entity",
        429: "rate_limit_exceeded",
        500: "internal_error",
        502: "bad_gateway",
        503: "service_unavailable",
        504: "gateway_timeout",
    }

    def __init__(self, get_response: Callable):
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        response = self.get_response(request)

        # Only process error responses for API paths
        if not request.path.startswith("/api/"):
            return response

        # Only process error status codes
        if response.status_code < 400:
            return response

        # Skip if already has proper error envelope
        try:
            content = json.loads(response.content.decode("utf-8"))
            if "error" in content and "code" in content.get("error", {}):
                # Already has proper envelope, just add request_id if missing
                if "request_id" not in content["error"]:
                    content["error"]["request_id"] = getattr(
                        request, "syn_request_id", None
                    )
                    response.content = json.dumps(content).encode("utf-8")
                return response
        except (json.JSONDecodeError, UnicodeDecodeError):
            content = {}

        # Build standard error envelope
        request_id = getattr(request, "syn_request_id", None)
        status_code = response.status_code
        error_code = self.STATUS_TO_CODE.get(status_code, "error")
        retryable = status_code in self.RETRYABLE_STATUSES

        # Extract message from original response
        message = content.get("detail", content.get("error", content.get("message", "")))
        if not message:
            message = response.reason_phrase or f"HTTP {status_code}"

        # Build envelope
        error_envelope = {
            "error": {
                "code": error_code,
                "message": str(message),
                "retryable": retryable,
                "request_id": request_id,
                "doc": f"https://docs.synara.io/errors#{error_code}",
            }
        }

        # Add details if available
        if "details" in content:
            error_envelope["error"]["details"] = content["details"]
        elif "errors" in content:
            error_envelope["error"]["details"] = content["errors"]

        # Create new response with envelope
        new_response = JsonResponse(error_envelope, status=status_code)
        new_response[HEADER_SYN_REQUEST_ID] = request_id

        return new_response

    def process_exception(self, request: HttpRequest, exception: Exception) -> HttpResponse:
        """
        Handle exceptions and convert to error envelope per ERR-002.

        This method catches SynaraError exceptions and converts them to
        standard API error responses using the error's to_envelope() method.

        Standard: ERR-002 §3, API-002 §10, POL-002 §8
        """
        # Only process exceptions for API paths - let UI paths use Django's
        # default exception handling (which renders HTML error pages)
        if not request.path.startswith("/api/"):
            return None  # Let Django handle non-API exceptions

        request_id = getattr(request, "syn_request_id", None)

        # Handle SynaraError exceptions with their built-in envelope
        if isinstance(exception, SynaraError):
            envelope = exception.to_envelope(request_id=request_id)
            response = JsonResponse(envelope.to_dict(), status=exception.http_status)
            response[HEADER_SYN_REQUEST_ID] = request_id

            # POL-002 §8: Log error without sensitive data
            # Use redacted message instead of full exception
            logger.error(
                f"SynaraError: {exception.code}",
                extra={
                    "error_code": exception.code,
                    "correlation_id": str(exception.correlation_id),
                    "request_id": request_id,
                    "http_status": exception.http_status,
                    # POL-002 §8: Redact message for logging
                    "message_redacted": redact_error_message(str(exception.message)),
                },
            )

            return response

        # For non-SynaraError exceptions, return generic 500 response
        # POL-002 §8: Never expose internal exception details to client
        error_envelope = {
            "error": {
                "code": "internal_error",
                "message": "An unexpected error occurred",
                "retryable": True,
                "request_id": request_id,
                "doc": "https://docs.synara.io/errors#internal_error",
            }
        }

        # POL-002 §8: Log redacted exception instead of full traceback
        # ANTI-PATTERN FIX: AP-POL2-005 - Missing redaction for error messages
        # Using logger.error with redacted message instead of logger.exception
        # to prevent full stacktrace with potentially sensitive local variables
        redacted_exc = redact_exception_for_logging(exception)
        logger.error(
            f"Unhandled exception (redacted): {redacted_exc}",
            extra={
                "request_id": request_id,
                "exception_type": type(exception).__name__,
                # POL-002 §8: Mark as redacted for audit compliance
                "_redaction_applied": True,
            },
        )

        response = JsonResponse(error_envelope, status=500)
        response[HEADER_SYN_REQUEST_ID] = request_id
        return response
