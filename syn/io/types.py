"""
IO Types and Constants (IO-001 §3, IO-002 §3)
=============================================

Type definitions, enums, and constants for I/O contract enforcement.

Standard:     IO-001 §3 (Terminology), IO-002 §3 (Core Definitions)
Compliance:   NIST SP 800-53 SI-10 (Information Input Validation)
Location:     syn/io/types.py
Version:      1.0.0
"""

from enum import Enum
from typing import Dict, List

# =============================================================================
# FAILURE DOMAINS (IO-001 §5.4)
# =============================================================================


class FailureDomain(str, Enum):
    """
    Failure domain classification for I/O operations.

    Standard: IO-001 §5.4
    """

    FATAL = "fatal"
    """Cannot retry - permanent failure (e.g., validation error, not found)"""

    TRANSIENT = "transient"
    """Can retry - temporary failure (e.g., network timeout, rate limit)"""

    COMPENSABLE = "compensable"
    """Can undo - partial success requiring compensation"""


# =============================================================================
# IDEMPOTENCY STRATEGIES (IO-001 §7)
# =============================================================================


class IdempotencyStrategy(str, Enum):
    """
    Idempotency strategy for I/O operations.

    Standard: IO-001 §7.1
    """

    CORRELATION_ID = "correlation_id"
    """Use correlation_id as idempotency key (default)"""

    PAYLOAD_HASH = "payload_hash"
    """Hash request payload for deduplication"""

    CUSTOM = "custom"
    """Custom idempotency key from request"""

    NONE = "none"
    """No idempotency protection (use for safe/idempotent methods)"""


# =============================================================================
# VALIDATION LAYERS (IO-001 §6)
# =============================================================================


class ValidationLayer(int, Enum):
    """
    Validation layers for I/O gate.

    Standard: IO-001 §6.1
    """

    STRUCTURAL = 1
    """Layer 1: JSON/XML syntax, required fields (Cortex.publish)"""

    GOVERNANCE = 2
    """Layer 2: Policy evaluation (CGS-1001 Clause 04.4)"""

    SCHEMA = 7
    """Layer 7: Pydantic InputSchema/OutputSchema validation"""


# =============================================================================
# HTTP STATUS CODE CATEGORIES (IO-002 §11)
# =============================================================================


RETRYABLE_STATUS_CODES: List[int] = [408, 429, 500, 502, 503, 504]
"""HTTP status codes that are safe to retry per IO-002 §11"""

NON_RETRYABLE_STATUS_CODES: List[int] = [400, 401, 403, 404, 409, 410, 422]
"""HTTP status codes that should not be retried per IO-002 §11"""


# =============================================================================
# HEADER CONSTANTS (IO-002 §14)
# =============================================================================


class IOHeaders:
    """
    Standard I/O headers per IO-002 §14.

    Standard: IO-002 §14 (Header Requirements)
    """

    # Request headers
    SYN_REQUEST_ID = "Syn-Request-Id"
    """Client-generated correlation ID; server echoes (required)"""

    TRACEPARENT = "traceparent"
    """W3C Trace Context header for distributed tracing (required)"""

    TRACESTATE = "tracestate"
    """W3C Trace Context state header (optional)"""

    IDEMPOTENCY_KEY = "Idempotency-Key"
    """Idempotency key for POST requests (ULID format, required for POST with side effects)"""

    SYN_DEADLINE = "Syn-Deadline"
    """Request deadline in RFC3339 or epoch milliseconds"""

    # Response headers
    SYN_RATELIMIT_LIMIT = "Syn-RateLimit-Limit"
    """Maximum requests allowed per window"""

    SYN_RATELIMIT_REMAINING = "Syn-RateLimit-Remaining"
    """Remaining requests in current window"""

    SYN_RATELIMIT_RESET = "Syn-RateLimit-Reset"
    """Unix timestamp when window resets"""

    RETRY_AFTER = "Retry-After"
    """Seconds to wait before retrying (on 429/503)"""


# =============================================================================
# TIMEOUT DEFAULTS (IO-002 §7)
# =============================================================================


class TimeoutDefaults:
    """
    Default timeout values per IO-002 §7.

    Standard: IO-002 §7 (Timeouts and Deadlines)
    """

    # Client defaults
    CONNECT_MS = 1000
    """Connection timeout: 1 second"""

    TLS_HANDSHAKE_MS = 3000
    """TLS handshake timeout: 3 seconds"""

    REQUEST_MS = 10000
    """Request timeout: 10 seconds"""

    READ_MS = 10000
    """Read timeout: 10 seconds"""

    WRITE_MS = 10000
    """Write timeout: 10 seconds"""

    # Server defaults
    HEADER_MS = 5000
    """Header read timeout: 5 seconds"""

    IDLE_MS = 120000
    """Idle connection timeout: 120 seconds"""


# =============================================================================
# RETRY DEFAULTS (IO-002 §8)
# =============================================================================


class RetryDefaults:
    """
    Default retry configuration per IO-002 §8.

    Standard: IO-002 §8 (Retries and Backoff)
    """

    MAX_RETRIES = 4
    """Maximum retry attempts"""

    TOTAL_ATTEMPTS = 5
    """Total attempts including initial"""

    BASE_DELAY_MS = 100
    """Base delay for exponential backoff"""

    MAX_DELAY_MS = 30000
    """Maximum delay between retries"""

    MULTIPLIER = 2.0
    """Backoff multiplier"""

    JITTER_RATIO = 0.2
    """Jitter ratio for randomization"""

    RETRY_BUDGET_MS = 120000
    """Overall retry budget: 2 minutes"""

    RETRY_AFTER_CAP_MS = 60000
    """Maximum Retry-After honor: 60 seconds"""


# =============================================================================
# RATE LIMIT DEFAULTS (IO-002 §10)
# =============================================================================


class RateLimitDefaults:
    """
    Default rate limit configuration per IO-002 §10.

    Standard: IO-002 §10 (Rate Limiting and Quotas)
    """

    ANONYMOUS_PER_MINUTE = 60
    ANONYMOUS_BURST = 10

    AUTHENTICATED_PER_MINUTE = 1000
    AUTHENTICATED_BURST = 100

    SERVICE_ACCOUNT_PER_MINUTE = 10000
    SERVICE_ACCOUNT_BURST = 1000


# =============================================================================
# PAGINATION DEFAULTS (API-002 §7)
# =============================================================================


class PaginationDefaults:
    """
    Default pagination configuration per API-002 §7.

    Standard: API-002 §7 (Pagination, Filtering and Sorting)
    """

    DEFAULT_LIMIT = 50
    """Default page size"""

    MIN_LIMIT = 1
    """Minimum page size"""

    MAX_LIMIT = 200
    """Maximum page size"""

    STABLE_SORT_KEY = "created_at"
    """Default stable sort key for cursor pagination"""


# =============================================================================
# IDEMPOTENCY DEFAULTS (IO-002 §9, API-002 §9)
# =============================================================================


class IdempotencyDefaults:
    """
    Default idempotency configuration per IO-002 §9.

    Standard: IO-002 §9 (Idempotency), API-002 §9
    """

    TTL_HOURS = 24
    """Idempotency key retention: 24 hours"""

    KEY_FORMAT = "ulid"
    """Key format: ULID (26 characters)"""

    KEY_PATTERN = r"^[0-9A-HJKMNP-TV-Z]{26}$"
    """ULID validation pattern"""


# =============================================================================
# STREAMING DEFAULTS (IO-002 §12)
# =============================================================================


class StreamingDefaults:
    """
    Default streaming configuration per IO-002 §12.

    Standard: IO-002 §12 (Streaming and Chunked IO)
    """

    MAX_CHUNK_SIZE_BYTES = 1048576
    """Maximum chunk size: 1MB"""

    CHUNK_TIMEOUT_MS = 5000
    """Timeout between chunks: 5 seconds"""


# =============================================================================
# FAILURE DOMAIN MAPPING
# =============================================================================


STATUS_CODE_TO_FAILURE_DOMAIN: Dict[int, FailureDomain] = {
    # Fatal (non-retryable)
    400: FailureDomain.FATAL,
    401: FailureDomain.FATAL,
    403: FailureDomain.FATAL,
    404: FailureDomain.FATAL,
    409: FailureDomain.FATAL,
    410: FailureDomain.FATAL,
    422: FailureDomain.FATAL,
    # Transient (retryable)
    408: FailureDomain.TRANSIENT,
    429: FailureDomain.TRANSIENT,
    500: FailureDomain.TRANSIENT,
    502: FailureDomain.TRANSIENT,
    503: FailureDomain.TRANSIENT,
    504: FailureDomain.TRANSIENT,
}


def get_failure_domain(status_code: int) -> FailureDomain:
    """
    Get failure domain for HTTP status code.

    Standard: IO-001 §5.4, IO-002 §11
    """
    return STATUS_CODE_TO_FAILURE_DOMAIN.get(status_code, FailureDomain.FATAL)


def is_retryable(status_code: int) -> bool:
    """
    Check if HTTP status code is retryable.

    Standard: IO-002 §11
    """
    return status_code in RETRYABLE_STATUS_CODES
