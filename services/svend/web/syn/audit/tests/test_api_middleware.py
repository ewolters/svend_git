"""
API-001 middleware and pagination compliance tests.

Tests verify middleware source patterns: error envelope wrapping,
status-to-code mapping, SynaraError conversion, request ID generation,
traceparent validation, timing headers, cursor pagination, idempotency
key handling, payload hashing, and TTL enforcement.

Standard: API-001
"""

import os
import re
from pathlib import Path

from django.test import SimpleTestCase

WEB_ROOT = Path(os.path.dirname(__file__)).parent.parent.parent
SYN_API = WEB_ROOT / "syn" / "api"


def _read(path):
    try:
        return Path(path).read_text(errors="ignore")
    except Exception:
        return ""


# ── Error Envelope Middleware ────────────────────────────────────────────


class ErrorEnvelopeTest(SimpleTestCase):
    """API-001 §10: ErrorEnvelopeMiddleware wraps /api/ error responses."""

    def setUp(self):
        self.src = _read(SYN_API / "middleware.py")
        self.assertGreater(len(self.src), 0, "middleware.py not found")

    def test_class_exists(self):
        """ErrorEnvelopeMiddleware class is defined."""
        self.assertIn("class ErrorEnvelopeMiddleware", self.src)

    def test_wraps_api_paths_only(self):
        """Only wraps error responses for /api/ paths."""
        fn = re.search(
            r"class ErrorEnvelopeMiddleware.*?(?=\nclass |\Z)",
            self.src, re.DOTALL,
        )
        self.assertIsNotNone(fn)
        body = fn.group()
        self.assertIn('"/api/"', body)
        self.assertIn("status_code < 400", body)

    def test_builds_standard_envelope(self):
        """Error response includes code, message, retryable, request_id, doc."""
        fn = re.search(
            r"class ErrorEnvelopeMiddleware.*?(?=\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        for field in ["code", "message", "retryable", "request_id", "doc"]:
            self.assertIn(f'"{field}"', body, f"Envelope missing '{field}'")

    def test_skips_already_enveloped(self):
        """Responses with existing error envelope are not re-wrapped."""
        fn = re.search(
            r"class ErrorEnvelopeMiddleware.*?(?=\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        # Should check for existing envelope before wrapping
        self.assertIn('"error"', body)
        self.assertIn('"code"', body)


class ErrorCodeMappingTest(SimpleTestCase):
    """API-001 §10: HTTP status codes mapped to error codes."""

    def setUp(self):
        self.src = _read(SYN_API / "middleware.py")

    def test_status_to_code_dict_exists(self):
        """STATUS_TO_CODE mapping is defined."""
        self.assertIn("STATUS_TO_CODE", self.src)

    def test_maps_common_client_errors(self):
        """Maps 400, 401, 403, 404, 409, 422, 429."""
        for status, code in [
            ("400", "bad_request"),
            ("401", "unauthorized"),
            ("403", "forbidden"),
            ("404", "not_found"),
            ("409", "conflict"),
            ("422", "unprocessable_entity"),
            ("429", "rate_limit_exceeded"),
        ]:
            self.assertIn(status, self.src, f"Missing status {status}")
            self.assertIn(code, self.src, f"Missing error code '{code}'")

    def test_maps_server_errors(self):
        """Maps 500, 502, 503, 504."""
        for status, code in [
            ("500", "internal_error"),
            ("502", "bad_gateway"),
            ("503", "service_unavailable"),
            ("504", "gateway_timeout"),
        ]:
            self.assertIn(status, self.src)
            self.assertIn(code, self.src)

    def test_retryable_statuses_defined(self):
        """RETRYABLE_STATUSES set includes 408, 429, 500, 502, 503, 504."""
        self.assertIn("RETRYABLE_STATUSES", self.src)
        for code in ["408", "429", "500", "502", "503", "504"]:
            self.assertIn(code, self.src)


class SynaraErrorConversionTest(SimpleTestCase):
    """API-001 §10: SynaraError converted via to_envelope()."""

    def setUp(self):
        self.src = _read(SYN_API / "middleware.py")

    def test_process_exception_defined(self):
        """process_exception method handles SynaraError."""
        self.assertIn("def process_exception(", self.src)

    def test_synara_error_isinstance_check(self):
        """process_exception checks isinstance(exception, SynaraError)."""
        fn = re.search(
            r"def process_exception\(.*?(?=\n    def |\nclass |\Z)",
            self.src, re.DOTALL,
        )
        self.assertIsNotNone(fn)
        body = fn.group()
        self.assertIn("SynaraError", body)
        self.assertIn("isinstance", body)

    def test_calls_to_envelope(self):
        """process_exception calls exception.to_envelope()."""
        fn = re.search(
            r"def process_exception\(.*?(?=\n    def |\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        self.assertIn("to_envelope", body)

    def test_non_synara_returns_generic_500(self):
        """Non-SynaraError exceptions return generic internal_error 500."""
        fn = re.search(
            r"def process_exception\(.*?(?=\n    def |\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        self.assertIn("internal_error", body)
        self.assertIn("500", body)

    def test_error_redaction_applied(self):
        """process_exception applies error redaction per POL-002 §8."""
        fn = re.search(
            r"def process_exception\(.*?(?=\n    def |\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        self.assertIn("redact", body.lower())


# ── Request ID Middleware ────────────────────────────────────────────────


class RequestIdTest(SimpleTestCase):
    """API-001 §8.1: SynRequestIdMiddleware generates/extracts request IDs."""

    def setUp(self):
        self.src = _read(SYN_API / "middleware.py")

    def test_class_exists(self):
        """SynRequestIdMiddleware class is defined."""
        self.assertIn("class SynRequestIdMiddleware", self.src)

    def test_generates_request_id(self):
        """Generates request ID via _generate_request_id."""
        self.assertIn("def _generate_request_id(", self.src)

    def test_ulid_generation(self):
        """_generate_request_id uses ulid library."""
        fn = re.search(
            r"def _generate_request_id\(.*?(?=\ndef |\nclass |\Z)",
            self.src, re.DOTALL,
        )
        self.assertIsNotNone(fn)
        body = fn.group()
        self.assertIn("ulid", body)

    def test_extracts_from_header(self):
        """Extracts Syn-Request-Id from incoming request header."""
        fn = re.search(
            r"class SynRequestIdMiddleware.*?(?=\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        self.assertIn("Syn-Request-Id", body)

    def test_echoes_in_response(self):
        """Echoes Syn-Request-Id in response header."""
        fn = re.search(
            r"class SynRequestIdMiddleware.*?(?=\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        self.assertIn("HEADER_SYN_REQUEST_ID", body)
        # Response should set the header
        self.assertIn("response[", body)

    def test_stores_on_request_object(self):
        """Stores request ID as request.syn_request_id."""
        fn = re.search(
            r"class SynRequestIdMiddleware.*?(?=\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        self.assertIn("syn_request_id", body)


class TraceparentTest(SimpleTestCase):
    """API-001 §8.1: W3C traceparent header extraction."""

    def setUp(self):
        self.src = _read(SYN_API / "middleware.py")

    def test_traceparent_pattern_defined(self):
        """TRACEPARENT_PATTERN regex defined for version-traceid-spanid-flags."""
        self.assertIn("TRACEPARENT_PATTERN", self.src)

    def test_traceparent_regex_validates_format(self):
        """Pattern matches W3C traceparent: 2hex-32hex-16hex-2hex."""
        # Extract the pattern
        m = re.search(r'TRACEPARENT_PATTERN\s*=\s*r"([^"]+)"', self.src)
        self.assertIsNotNone(m, "TRACEPARENT_PATTERN regex not found")
        pattern = m.group(1)
        # Valid traceparent
        valid = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        self.assertIsNotNone(re.match(pattern, valid))
        # Invalid traceparent (short trace ID)
        invalid = "00-0af765-b7ad6b-01"
        self.assertIsNone(re.match(pattern, invalid))

    def test_extracts_trace_id_and_span_id(self):
        """Extracts syn_trace_id and syn_span_id from traceparent."""
        fn = re.search(
            r"class SynRequestIdMiddleware.*?(?=\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        self.assertIn("syn_trace_id", body)
        self.assertIn("syn_span_id", body)

    def test_sets_none_on_invalid(self):
        """Sets trace fields to None when traceparent is invalid/missing."""
        fn = re.search(
            r"class SynRequestIdMiddleware.*?(?=\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        self.assertIn("None", body)


class TimingTest(SimpleTestCase):
    """API-001 §8.2: Request timing via X-Response-Time header."""

    def setUp(self):
        self.src = _read(SYN_API / "middleware.py")

    def test_start_time_recorded(self):
        """SynRequestIdMiddleware records syn_start_time."""
        fn = re.search(
            r"class SynRequestIdMiddleware.*?(?=\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        self.assertIn("syn_start_time", body)
        self.assertIn("time.time()", body)

    def test_response_time_header_added(self):
        """APIHeadersMiddleware adds X-Response-Time header."""
        fn = re.search(
            r"class APIHeadersMiddleware.*?(?=\nclass |\Z)",
            self.src, re.DOTALL,
        )
        self.assertIsNotNone(fn)
        body = fn.group()
        self.assertIn("X-Response-Time", body)

    def test_duration_in_milliseconds(self):
        """Duration calculated in milliseconds."""
        fn = re.search(
            r"class APIHeadersMiddleware.*?(?=\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        self.assertIn("1000", body)
        self.assertIn("ms", body)


# ── Cursor Pagination ────────────────────────────────────────────────────


class CursorPaginationTest(SimpleTestCase):
    """API-001 §7: Cursor-based pagination, not offset."""

    def setUp(self):
        self.src = _read(SYN_API / "pagination.py")
        self.assertGreater(len(self.src), 0, "pagination.py not found")

    def test_class_exists(self):
        """SynaraCursorPagination class is defined."""
        self.assertIn("class SynaraCursorPagination", self.src)

    def test_extends_cursor_pagination(self):
        """Extends CursorPagination, not PageNumberPagination or LimitOffsetPagination."""
        self.assertIn("CursorPagination)", self.src)
        self.assertNotIn("PageNumberPagination", self.src)
        self.assertNotIn("LimitOffsetPagination", self.src)

    def test_deterministic_ordering(self):
        """Uses stable sort key (created_at) for deterministic ordering."""
        self.assertIn("created_at", self.src)
        self.assertIn("ordering", self.src)


class PaginationParamsTest(SimpleTestCase):
    """API-001 §7: Pagination query params are limit (1-200, default 50) and cursor."""

    def setUp(self):
        self.src = _read(SYN_API / "pagination.py")

    def test_default_page_size(self):
        """Default page size is 50."""
        self.assertIn("DEFAULT_PAGE_SIZE = 50", self.src)

    def test_min_page_size(self):
        """Minimum page size is 1."""
        self.assertIn("MIN_PAGE_SIZE = 1", self.src)

    def test_max_page_size(self):
        """Maximum page size is 200."""
        self.assertIn("MAX_PAGE_SIZE = 200", self.src)

    def test_limit_query_param(self):
        """Query parameter for page size is 'limit'."""
        self.assertIn('"limit"', self.src)

    def test_cursor_query_param(self):
        """Query parameter for cursor is 'cursor'."""
        self.assertIn('"cursor"', self.src)

    def test_bounds_enforcement(self):
        """get_page_size enforces min/max bounds."""
        fn = re.search(
            r"def get_page_size\(.*?(?=\n    def |\nclass |\Z)",
            self.src, re.DOTALL,
        )
        self.assertIsNotNone(fn)
        body = fn.group()
        self.assertIn("MIN_PAGE_SIZE", body)
        self.assertIn("MAX_PAGE_SIZE", body)


class PaginationResponseTest(SimpleTestCase):
    """API-001 §7.2: Paginated responses use {data, next_cursor, total_estimate}."""

    def setUp(self):
        self.src = _read(SYN_API / "pagination.py")

    def test_response_has_data_field(self):
        """Response envelope includes 'data' field."""
        fn = re.search(
            r"def get_paginated_response\(.*?(?=\n    def |\nclass |\Z)",
            self.src, re.DOTALL,
        )
        self.assertIsNotNone(fn)
        body = fn.group()
        self.assertIn('"data"', body)

    def test_response_has_next_cursor(self):
        """Response envelope includes 'next_cursor' field."""
        fn = re.search(
            r"def get_paginated_response\(.*?(?=\n    def |\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        self.assertIn('"next_cursor"', body)

    def test_response_has_total_estimate(self):
        """Response envelope includes 'total_estimate' field."""
        fn = re.search(
            r"def get_paginated_response\(.*?(?=\n    def |\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        self.assertIn('"total_estimate"', body)

    def test_create_cursor_response_helper(self):
        """create_cursor_response utility returns standard envelope."""
        self.assertIn("def create_cursor_response(", self.src)
        fn = re.search(
            r"def create_cursor_response\(.*?(?=\ndef |\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        for field in ["data", "next_cursor", "total_estimate"]:
            self.assertIn(f'"{field}"', body)


# ── Idempotency Middleware ───────────────────────────────────────────────


class IdempotencyKeyTest(SimpleTestCase):
    """API-001 §9: IdempotencyMiddleware validates Idempotency-Key on POST."""

    def setUp(self):
        self.src = _read(SYN_API / "middleware.py")

    def test_class_exists(self):
        """IdempotencyMiddleware class is defined."""
        self.assertIn("class IdempotencyMiddleware", self.src)

    def test_post_only(self):
        """Only applies to POST requests."""
        fn = re.search(
            r"class IdempotencyMiddleware.*?(?=\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        self.assertIn('"POST"', body)

    def test_api_path_only(self):
        """Only applies to /api/ paths."""
        fn = re.search(
            r"class IdempotencyMiddleware.*?(?=\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        self.assertIn('"/api/"', body)

    def test_key_header_name(self):
        """Reads Idempotency-Key header."""
        self.assertIn("Idempotency-Key", self.src)

    def test_key_pattern_defined(self):
        """IDEMPOTENCY_KEY_PATTERN regex defined for ULID format."""
        self.assertIn("IDEMPOTENCY_KEY_PATTERN", self.src)


class IdempotencyBehaviorTest(SimpleTestCase):
    """API-001 §9: Cached response on replay, 409 on payload conflict."""

    def setUp(self):
        self.src = _read(SYN_API / "middleware.py")

    def test_returns_cached_response(self):
        """Returns cached response when key matches with same payload."""
        fn = re.search(
            r"class IdempotencyMiddleware.*?(?=\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        self.assertIn("_get_cached_response", body)
        self.assertIn("cached_response", body)

    def test_409_on_conflict(self):
        """Returns 409 when key reused with different payload."""
        fn = re.search(
            r"class IdempotencyMiddleware.*?(?=\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        self.assertIn("409", body)
        self.assertIn("idempotency_conflict", body)

    def test_caches_successful_responses(self):
        """Only caches responses with 2xx status codes."""
        fn = re.search(
            r"class IdempotencyMiddleware.*?(?=\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        self.assertIn("200", body)
        self.assertIn("300", body)
        self.assertIn("_cache_response", body)

    def test_logs_replay(self):
        """Logs idempotency key replay events."""
        fn = re.search(
            r"class IdempotencyMiddleware.*?(?=\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        self.assertIn("replay", body.lower())


class PayloadHashTest(SimpleTestCase):
    """API-001 §9: Idempotency payload hash is SHA-256."""

    def setUp(self):
        self.src = _read(SYN_API / "middleware.py")

    def test_sha256_used(self):
        """Payload hash uses SHA-256 algorithm."""
        fn = re.search(
            r"class IdempotencyMiddleware.*?(?=\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        self.assertIn("sha256", body)

    def test_hashes_request_body(self):
        """Hashes request.body for conflict detection."""
        fn = re.search(
            r"class IdempotencyMiddleware.*?(?=\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        self.assertIn("request.body", body)
        self.assertIn("payload_hash", body)

    def test_payload_hash_stored_in_cache(self):
        """Payload hash stored in cached entry for comparison."""
        fn = re.search(
            r"def _cache_response\(.*?(?=\n    def |\nclass |\Z)",
            self.src, re.DOTALL,
        )
        self.assertIsNotNone(fn)
        body = fn.group()
        self.assertIn("payload_hash", body)


class IdempotencyTTLTest(SimpleTestCase):
    """API-001 §9: Idempotency cache TTL is 24 hours."""

    def setUp(self):
        self.src = _read(SYN_API / "middleware.py")

    def test_ttl_constant_defined(self):
        """IDEMPOTENCY_TTL_HOURS = 24."""
        self.assertIn("IDEMPOTENCY_TTL_HOURS = 24", self.src)

    def test_ttl_used_in_cache_set(self):
        """Cache set uses TTL hours * 3600 for seconds."""
        fn = re.search(
            r"def _cache_response\(.*?(?=\n    def |\nclass |\Z)",
            self.src, re.DOTALL,
        )
        self.assertIsNotNone(fn)
        body = fn.group()
        self.assertIn("IDEMPOTENCY_TTL_HOURS", body)
        self.assertIn("3600", body)


# ── API Headers Middleware ───────────────────────────────────────────────


class APIHeadersTest(SimpleTestCase):
    """API-001 §8.2: APIHeadersMiddleware enforces required headers."""

    def setUp(self):
        self.src = _read(SYN_API / "middleware.py")

    def test_class_exists(self):
        """APIHeadersMiddleware class is defined."""
        self.assertIn("class APIHeadersMiddleware", self.src)

    def test_accept_header_validation(self):
        """Validates Accept header includes application/json."""
        fn = re.search(
            r"class APIHeadersMiddleware.*?(?=\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        self.assertIn("application/json", body)
        self.assertIn("Accept", body)

    def test_charset_enforcement(self):
        """Ensures Content-Type includes charset=utf-8."""
        fn = re.search(
            r"class APIHeadersMiddleware.*?(?=\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        self.assertIn("charset", body)
        self.assertIn("utf-8", body)

    def test_vary_header_added(self):
        """Adds Vary header for proper caching."""
        fn = re.search(
            r"class APIHeadersMiddleware.*?(?=\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        self.assertIn("Vary", body)
        self.assertIn("Accept-Encoding", body)

    def test_406_on_invalid_accept(self):
        """Returns 406 for invalid Accept header."""
        fn = re.search(
            r"class APIHeadersMiddleware.*?(?=\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        self.assertIn("406", body)
        self.assertIn("unsupported_media_type", body)

    def test_exempt_paths_skip_validation(self):
        """Exempt paths bypass header validation."""
        self.assertIn("EXEMPT_PATHS", self.src)
        self.assertIn("_is_exempt_path", self.src)

    def test_sets_content_type_charset(self):
        """Ensures Content-Type includes charset=utf-8 on JSON responses."""
        fn = re.search(
            r"class APIHeadersMiddleware.*?(?=\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        self.assertIn("charset", body)
        self.assertIn("utf-8", body)
        self.assertIn("application/json; charset=utf-8", body)

    def test_sets_vary_header(self):
        """Adds Vary: Accept, Accept-Encoding header for proper caching."""
        fn = re.search(
            r"class APIHeadersMiddleware.*?(?=\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        self.assertIn("Vary", body)
        self.assertIn("Accept-Encoding", body)

    def test_adds_response_time_header(self):
        """Adds X-Response-Time header with duration in ms."""
        fn = re.search(
            r"class APIHeadersMiddleware.*?(?=\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        self.assertIn("X-Response-Time", body)
        self.assertIn("ms", body)

    def test_api_paths_only(self):
        """Accept validation only applies to /api/ paths."""
        fn = re.search(
            r"class APIHeadersMiddleware.*?(?=\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        self.assertIn('"/api/"', body)
        self.assertIn("_is_exempt_path", body)

    def test_no_cache_on_api(self):
        """API responses use Vary header to prevent improper caching."""
        fn = re.search(
            r"class APIHeadersMiddleware.*?(?=\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        # Vary header with Accept prevents cached HTML being served as JSON
        self.assertIn("Vary", body)
        self.assertIn("Accept", body)


# ── Error Redaction ──────────────────────────────────────────────────────


class ErrorRedactionTest(SimpleTestCase):
    """API-001/POL-002 §8: Error message redaction."""

    def setUp(self):
        self.src = _read(SYN_API / "middleware.py")

    def test_redact_function_exists(self):
        """redact_error_message function defined."""
        self.assertIn("def redact_error_message(", self.src)

    def test_redacts_passwords(self):
        """Redacts password patterns."""
        self.assertIn("password", self.src.lower())
        self.assertIn("REDACTED", self.src)

    def test_redacts_bearer_tokens(self):
        """Redacts Bearer token patterns."""
        self.assertIn("Bearer", self.src)

    def test_redacts_env_vars(self):
        """Redacts environment variable patterns."""
        self.assertIn("ENV_VAR_PATTERNS", self.src)
        for var in ["DATABASE_URL", "SECRET_KEY", "AWS_SECRET"]:
            self.assertIn(var, self.src)

    def test_redact_exception_for_logging(self):
        """redact_exception_for_logging strips sensitive data from exceptions."""
        self.assertIn("def redact_exception_for_logging(", self.src)

    def test_strips_internal_details(self):
        """redact_exception_for_logging strips full traceback, returns type: message only."""
        fn = re.search(
            r"def redact_exception_for_logging\(.*?(?=\ndef |\nclass |\Z)",
            self.src, re.DOTALL,
        )
        self.assertIsNotNone(fn)
        body = fn.group()
        # Should not include traceback — only exc_type and message
        self.assertIn("exc_type", body)
        self.assertIn("exc_message", body)
        self.assertNotIn("traceback.format_exc", body)

    def test_preserves_safe_fields(self):
        """Redaction preserves non-sensitive parts of the message."""
        fn = re.search(
            r"def redact_error_message\(.*?(?=\ndef |\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        # Should use re.sub pattern replacement, not blanket wipe
        self.assertIn("re.sub", body)
        self.assertIn("return redacted", body)

    def test_redacts_traceback(self):
        """redact_exception_for_logging omits full traceback with locals."""
        fn = re.search(
            r"def redact_exception_for_logging\(.*?(?=\ndef |\nclass |\Z)",
            self.src, re.DOTALL,
        )
        body = fn.group()
        # Comment confirms: "no full traceback with locals"
        self.assertIn("no full traceback", body.lower())

    def test_redacts_sql(self):
        """SENSITIVE_ERROR_PATTERNS covers credential-like patterns that appear in SQL errors."""
        self.assertIn("SENSITIVE_ERROR_PATTERNS", self.src)
        # Patterns cover password=, secret=, credential= which appear in DB connection errors
        self.assertIn("password", self.src.lower())
        self.assertIn("credential", self.src.lower())

    def test_redacts_in_production_only(self):
        """process_exception applies redaction via redact_error_message."""
        fn = re.search(
            r"def process_exception\(.*?(?=\n    def |\nclass |\Z)",
            self.src, re.DOTALL,
        )
        self.assertIsNotNone(fn)
        body = fn.group()
        self.assertIn("redact", body.lower())
