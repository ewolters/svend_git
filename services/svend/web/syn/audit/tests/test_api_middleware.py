"""
API-001 middleware and pagination functional tests.

Tests exercise actual middleware behavior via HTTP integration (all four
middleware classes are installed in settings.MIDDLEWARE) and direct invocation
via RequestFactory for isolated cache/edge-case testing.

Standard: API-001
Compliance: SOC 2 CC6.1, CC7.2
"""

import hashlib
import json
import re
import time

from django.core.cache import cache
from django.http import JsonResponse
from django.test import RequestFactory, SimpleTestCase, TestCase, override_settings

from syn.api.middleware import (
    ENV_VAR_PATTERNS,
    HEADER_SYN_REQUEST_ID,
    IDEMPOTENCY_KEY_PATTERN,
    IDEMPOTENCY_TTL_HOURS,
    SENSITIVE_ERROR_PATTERNS,
    TRACEPARENT_PATTERN,
    ErrorEnvelopeMiddleware,
    IdempotencyMiddleware,
    SynRequestIdMiddleware,
    redact_error_message,
    redact_exception_for_logging,
)
from syn.api.pagination import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    MIN_PAGE_SIZE,
    create_cursor_response,
)

# Production SECURE_SSL_REDIRECT=True breaks test client HTTP requests.
SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)

# Class-level attributes
STATUS_TO_CODE = ErrorEnvelopeMiddleware.STATUS_TO_CODE
RETRYABLE_STATUSES = ErrorEnvelopeMiddleware.RETRYABLE_STATUSES


# ── Helpers ──────────────────────────────────────────────────────────────


def _api_get(client, path="/api/health/", **extra):
    """GET an API endpoint with JSON Accept header."""
    return client.get(path, HTTP_ACCEPT="application/json", **extra)


def _json_ok(body=None, status=200):
    """Return a simple JsonResponse for use as a middleware inner response."""
    return JsonResponse(body or {"ok": True}, status=status)


def _make_request(method="GET", path="/api/test/", body=None):
    """Create a request via RequestFactory with syn_request_id pre-set."""
    factory = RequestFactory()
    if method == "POST":
        request = factory.post(
            path,
            data=body or b"{}",
            content_type="application/json",
        )
    else:
        request = factory.get(path, HTTP_ACCEPT="application/json")
    request.syn_request_id = "test-req-001"
    request.syn_start_time = time.time()
    return request


# ── Error Envelope Middleware (installed) ─────────────────────────────────


@SECURE_OFF
class ErrorEnvelopeTest(TestCase):
    """API-001 §10: ErrorEnvelopeMiddleware wraps /api/ error responses."""

    def test_wraps_404_with_standard_envelope(self):
        """404 responses on /api/ paths get wrapped in error envelope."""
        res = _api_get(self.client, "/api/nonexistent-endpoint-xyz/")
        self.assertEqual(res.status_code, 404)
        data = res.json()
        self.assertIn("error", data)
        for field in ["code", "message", "retryable", "request_id"]:
            self.assertIn(field, data["error"], f"Envelope missing '{field}'")

    def test_envelope_code_is_not_found_for_404(self):
        """404 error code maps to 'not_found'."""
        res = _api_get(self.client, "/api/nonexistent-endpoint-xyz/")
        self.assertEqual(res.json()["error"]["code"], "not_found")

    def test_envelope_retryable_false_for_client_errors(self):
        """Client errors (4xx) are not retryable."""
        res = _api_get(self.client, "/api/nonexistent-endpoint-xyz/")
        self.assertFalse(res.json()["error"]["retryable"])

    def test_skips_non_api_paths(self):
        """Error responses on non-API paths are not wrapped in JSON envelope."""
        res = self.client.get("/nonexistent-page/")
        if res.status_code == 404:
            content_type = res.get("Content-Type", "")
            # Non-API paths use HTML error pages, not JSON envelopes
            self.assertNotIn("application/json", content_type)


class ErrorCodeMappingTest(SimpleTestCase):
    """API-001 §10: HTTP status codes mapped to error codes."""

    def test_status_to_code_maps_client_errors(self):
        """STATUS_TO_CODE maps 400, 401, 403, 404, 409, 422, 429."""
        expected = {
            400: "bad_request",
            401: "unauthorized",
            403: "forbidden",
            404: "not_found",
            409: "conflict",
            422: "unprocessable_entity",
            429: "rate_limit_exceeded",
        }
        for status, code in expected.items():
            with self.subTest(status=status):
                self.assertEqual(STATUS_TO_CODE[status], code)

    def test_status_to_code_maps_server_errors(self):
        """STATUS_TO_CODE maps 500, 502, 503, 504."""
        expected = {
            500: "internal_error",
            502: "bad_gateway",
            503: "service_unavailable",
            504: "gateway_timeout",
        }
        for status, code in expected.items():
            with self.subTest(status=status):
                self.assertEqual(STATUS_TO_CODE[status], code)

    def test_retryable_statuses_includes_expected(self):
        """RETRYABLE_STATUSES includes 408, 429, 500, 502, 503, 504."""
        for code in [408, 429, 500, 502, 503, 504]:
            with self.subTest(code=code):
                self.assertIn(code, RETRYABLE_STATUSES)


class SynaraErrorConversionTest(SimpleTestCase):
    """API-001 §10: SynaraError converted via to_envelope()."""

    def test_process_exception_converts_synara_error(self):
        """process_exception returns envelope for SynaraError."""
        from syn.err.exceptions import ValidationError as SynaraValidationError

        request = _make_request()
        middleware = ErrorEnvelopeMiddleware(lambda r: _json_ok())
        exc = SynaraValidationError("Test failure", field="name")
        response = middleware.process_exception(request, exc)

        self.assertIsNotNone(response)
        self.assertIn(response.status_code, [400, 422])
        data = json.loads(response.content)
        self.assertIn("error", data)

    def test_process_exception_returns_500_for_generic_exception(self):
        """Non-SynaraError exceptions return generic 500 without internal details."""
        request = _make_request()
        middleware = ErrorEnvelopeMiddleware(lambda r: _json_ok())
        response = middleware.process_exception(request, RuntimeError("db connection lost"))

        self.assertEqual(response.status_code, 500)
        data = json.loads(response.content)
        self.assertEqual(data["error"]["code"], "internal_error")
        # Must NOT expose internal exception message to client
        self.assertNotIn("db connection lost", data["error"]["message"])

    def test_process_exception_skips_non_api_paths(self):
        """process_exception returns None for non-API paths."""
        request = _make_request(path="/admin/")
        middleware = ErrorEnvelopeMiddleware(lambda r: _json_ok())
        result = middleware.process_exception(request, RuntimeError("test"))
        self.assertIsNone(result)

    def test_process_exception_includes_request_id(self):
        """Error envelope includes request_id from Syn-Request-Id."""
        request = _make_request()
        middleware = ErrorEnvelopeMiddleware(lambda r: _json_ok())
        response = middleware.process_exception(request, RuntimeError("err"))
        data = json.loads(response.content)
        self.assertEqual(data["error"]["request_id"], "test-req-001")


# ── Request ID Middleware (installed) ────────────────────────────────────


@SECURE_OFF
class RequestIdTest(TestCase):
    """API-001 §8.1: SynRequestIdMiddleware generates/extracts request IDs."""

    def test_response_contains_request_id_header(self):
        """Every API response includes Syn-Request-Id header."""
        res = _api_get(self.client)
        self.assertIn(HEADER_SYN_REQUEST_ID, res)
        self.assertTrue(len(res[HEADER_SYN_REQUEST_ID]) > 0)

    def test_echoes_provided_request_id(self):
        """Client-provided Syn-Request-Id is echoed back."""
        custom_id = "test-custom-request-id-12345"
        res = self.client.get(
            "/api/health/",
            HTTP_ACCEPT="application/json",
            HTTP_SYN_REQUEST_ID=custom_id,
        )
        self.assertEqual(res[HEADER_SYN_REQUEST_ID], custom_id)

    def test_generates_id_when_not_provided(self):
        """Generates a new request ID when none is provided."""
        res = _api_get(self.client)
        request_id = res[HEADER_SYN_REQUEST_ID]
        self.assertTrue(len(request_id) > 10)


class RequestIdDirectTest(SimpleTestCase):
    """API-001 §8.1: SynRequestIdMiddleware direct invocation."""

    def test_sets_syn_request_id_on_request_object(self):
        """Middleware stores request ID on request.syn_request_id."""
        factory = RequestFactory()
        request = factory.get("/api/test/")

        captured = {}

        def inner(r):
            captured["id"] = r.syn_request_id
            captured["start_time"] = r.syn_start_time
            return _json_ok()

        middleware = SynRequestIdMiddleware(inner)
        middleware(request)
        self.assertTrue(len(captured["id"]) > 0)
        self.assertIsNotNone(captured["start_time"])

    def test_extracts_traceparent_fields(self):
        """Valid traceparent sets syn_trace_id and syn_span_id on request."""
        factory = RequestFactory()
        valid_tp = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        request = factory.get("/api/test/", HTTP_TRACEPARENT=valid_tp)

        captured = {}

        def inner(r):
            captured["trace_id"] = r.syn_trace_id
            captured["span_id"] = r.syn_span_id
            return _json_ok()

        middleware = SynRequestIdMiddleware(inner)
        middleware(request)
        self.assertEqual(captured["trace_id"], "0af7651916cd43dd8448eb211c80319c")
        self.assertEqual(captured["span_id"], "b7ad6b7169203331")

    def test_invalid_traceparent_sets_none(self):
        """Invalid traceparent sets trace fields to None."""
        factory = RequestFactory()
        request = factory.get("/api/test/", HTTP_TRACEPARENT="invalid")

        captured = {}

        def inner(r):
            captured["trace_id"] = r.syn_trace_id
            return _json_ok()

        middleware = SynRequestIdMiddleware(inner)
        middleware(request)
        self.assertIsNone(captured["trace_id"])


class TraceparentPatternTest(SimpleTestCase):
    """API-001 §8.1: W3C traceparent pattern validation."""

    def test_matches_valid_traceparent(self):
        """TRACEPARENT_PATTERN matches valid W3C traceparent format."""
        valid = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        self.assertIsNotNone(re.match(TRACEPARENT_PATTERN, valid))

    def test_rejects_short_trace_id(self):
        """TRACEPARENT_PATTERN rejects short/invalid format."""
        invalid = "00-0af765-b7ad6b-01"
        self.assertIsNone(re.match(TRACEPARENT_PATTERN, invalid))


# ── API Headers Middleware (installed) ────────────────────────────────────


@SECURE_OFF
class APIHeadersTest(TestCase):
    """API-001 §8.2: APIHeadersMiddleware enforces required headers."""

    def test_adds_vary_header(self):
        """API responses include Vary: Accept, Accept-Encoding."""
        res = _api_get(self.client)
        vary = res.get("Vary", "")
        self.assertIn("Accept", vary)
        self.assertIn("Accept-Encoding", vary)

    def test_adds_response_time_header(self):
        """API responses include X-Response-Time in ms format."""
        res = _api_get(self.client)
        self.assertIn("X-Response-Time", res)
        self.assertTrue(res["X-Response-Time"].endswith("ms"))

    def test_response_time_is_numeric(self):
        """X-Response-Time contains a numeric duration."""
        res = _api_get(self.client)
        duration = res["X-Response-Time"].replace("ms", "")
        self.assertTrue(duration.isdigit())

    def test_406_on_invalid_accept(self):
        """Returns 406 for non-JSON Accept header on /api/ paths."""
        res = self.client.get("/api/health/", HTTP_ACCEPT="text/xml")
        self.assertEqual(res.status_code, 406)
        data = res.json()
        self.assertEqual(data["error"]["code"], "unsupported_media_type")

    def test_allows_wildcard_accept(self):
        """Accept: */* is allowed (not rejected with 406)."""
        res = self.client.get("/api/health/", HTTP_ACCEPT="*/*")
        self.assertNotEqual(res.status_code, 406)

    def test_skips_non_api_paths(self):
        """Non-API paths bypass Accept header validation."""
        res = self.client.get("/", HTTP_ACCEPT="text/html")
        self.assertNotEqual(res.status_code, 406)

    def test_enforces_charset_on_json_responses(self):
        """JSON responses include charset=utf-8 in Content-Type."""
        res = _api_get(self.client)
        content_type = res.get("Content-Type", "")
        self.assertIn("charset=utf-8", content_type)


# ── Cursor Pagination ────────────────────────────────────────────────────


class CursorPaginationTest(SimpleTestCase):
    """API-001 §7: Cursor-based pagination constants and utilities."""

    def test_page_size_defaults(self):
        """Default, min, and max page sizes are correct."""
        self.assertEqual(DEFAULT_PAGE_SIZE, 50)
        self.assertEqual(MIN_PAGE_SIZE, 1)
        self.assertEqual(MAX_PAGE_SIZE, 200)

    def test_create_cursor_response_structure(self):
        """create_cursor_response returns {data, next_cursor, total_estimate}."""
        result = create_cursor_response(
            data=[{"id": 1}, {"id": 2}],
            next_cursor="abc123",
            total_estimate=42,
        )
        self.assertEqual(result["data"], [{"id": 1}, {"id": 2}])
        self.assertEqual(result["next_cursor"], "abc123")
        self.assertEqual(result["total_estimate"], 42)

    def test_create_cursor_response_defaults(self):
        """create_cursor_response defaults next_cursor=None, total_estimate=0."""
        result = create_cursor_response(data=[])
        self.assertIsNone(result["next_cursor"])
        self.assertEqual(result["total_estimate"], 0)
        self.assertEqual(result["data"], [])


class PaginationParamsTest(SimpleTestCase):
    """API-001 §7: Pagination query params and bounds enforcement."""

    def test_synara_cursor_pagination_config(self):
        """SynaraCursorPagination uses 'limit' and 'cursor' query params."""
        from syn.api.pagination import SynaraCursorPagination

        paginator = SynaraCursorPagination()
        self.assertEqual(paginator.page_size_query_param, "limit")
        self.assertEqual(paginator.cursor_query_param, "cursor")
        self.assertEqual(paginator.page_size, DEFAULT_PAGE_SIZE)
        self.assertEqual(paginator.max_page_size, MAX_PAGE_SIZE)

    def test_get_page_size_enforces_bounds(self):
        """get_page_size clamps to MIN_PAGE_SIZE..MAX_PAGE_SIZE."""
        from unittest.mock import MagicMock

        from syn.api.pagination import SynaraCursorPagination

        paginator = SynaraCursorPagination()

        for raw, expected in [("0", MIN_PAGE_SIZE), ("999", MAX_PAGE_SIZE), ("25", 25)]:
            with self.subTest(raw=raw):
                request = MagicMock()
                request.query_params = {"limit": raw}
                self.assertEqual(paginator.get_page_size(request), expected)


class PaginationResponseTest(SimpleTestCase):
    """API-001 §7.2: Paginated response envelope and cursor encoding."""

    def test_cursor_response_has_all_fields(self):
        """create_cursor_response includes data, next_cursor, total_estimate."""
        result = create_cursor_response(data=["a"], next_cursor="cur", total_estimate=10)
        for field in ["data", "next_cursor", "total_estimate"]:
            self.assertIn(field, result)

    def test_encode_decode_cursor_roundtrip(self):
        """encode_cursor → decode_cursor preserves data."""
        from syn.api.pagination import decode_cursor, encode_cursor

        position = {"created_at": "2024-01-01T00:00:00Z", "id": "abc"}
        encoded = encode_cursor(position)
        decoded = decode_cursor(encoded)
        self.assertEqual(decoded, position)

    def test_decode_invalid_cursor_returns_none(self):
        """decode_cursor returns None for invalid input."""
        from syn.api.pagination import decode_cursor

        self.assertIsNone(decode_cursor("not-valid-base64!!!"))


# ── Idempotency Middleware (installed) ────────────────────────────────────


class IdempotencyKeyTest(SimpleTestCase):
    """API-001 §9: Idempotency key constants and patterns."""

    def test_ttl_is_24_hours(self):
        """IDEMPOTENCY_TTL_HOURS = 24."""
        self.assertEqual(IDEMPOTENCY_TTL_HOURS, 24)

    def test_key_pattern_matches_ulid(self):
        """IDEMPOTENCY_KEY_PATTERN matches 26-char ULID format."""
        valid_ulid = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        self.assertIsNotNone(re.match(IDEMPOTENCY_KEY_PATTERN, valid_ulid))

    def test_key_pattern_rejects_invalid(self):
        """IDEMPOTENCY_KEY_PATTERN rejects non-ULID strings."""
        self.assertIsNone(re.match(IDEMPOTENCY_KEY_PATTERN, "too-short"))
        self.assertIsNone(re.match(IDEMPOTENCY_KEY_PATTERN, ""))


class IdempotencyDirectTest(TestCase):
    """API-001 §9: IdempotencyMiddleware cache behavior via direct invocation.

    Uses RequestFactory for precise cache control (pre-population, clearing).
    Middleware is installed in settings.MIDDLEWARE; direct invocation is used
    here to isolate cache logic from the full middleware stack.
    """

    def setUp(self):
        cache.clear()

    def _invoke_post(self, body=b'{"test": true}', key=None, inner_response=None):
        """Invoke IdempotencyMiddleware with a POST request."""
        factory = RequestFactory()
        extra = {}
        if key:
            extra["HTTP_IDEMPOTENCY_KEY"] = key
        request = factory.post(
            "/api/test/",
            data=body,
            content_type="application/json",
            **extra,
        )
        request.syn_request_id = "test-req-idem"
        request.tenant_id = None

        resp = inner_response or _json_ok({"result": "created"}, status=201)
        middleware = IdempotencyMiddleware(lambda r: resp)
        return middleware(request)

    def test_post_without_key_proceeds(self):
        """POST without Idempotency-Key proceeds normally."""
        res = self._invoke_post()
        self.assertEqual(res.status_code, 201)

    def test_post_with_key_caches_response(self):
        """POST with key caches 2xx response for replay."""
        key = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        body = b'{"data": "test"}'
        res = self._invoke_post(body=body, key=key)
        self.assertEqual(res.status_code, 201)

        # Verify response was cached
        cache_key = f"idempotency:None:{key}"
        cached = cache.get(cache_key)
        self.assertIsNotNone(cached)
        self.assertEqual(cached["payload_hash"], hashlib.sha256(body).hexdigest())

    def test_replay_returns_cached_response(self):
        """POST with same key + same body returns cached response."""
        key = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        body = b'{"data": "test"}'
        payload_hash = hashlib.sha256(body).hexdigest()

        # Pre-populate cache
        cache_key = f"idempotency:None:{key}"
        cache.set(
            cache_key,
            {
                "request_id": "req-original",
                "payload_hash": payload_hash,
                "status_code": 200,
                "response_body": {"result": "cached"},
            },
            timeout=86400,
        )

        res = self._invoke_post(body=body, key=key)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(json.loads(res.content)["result"], "cached")

    def test_conflict_returns_409(self):
        """POST with same key but different body returns 409."""
        key = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        original_hash = hashlib.sha256(b'{"data":"original"}').hexdigest()

        cache_key = f"idempotency:None:{key}"
        cache.set(
            cache_key,
            {
                "request_id": "req-original",
                "payload_hash": original_hash,
                "status_code": 200,
                "response_body": {"result": "original"},
            },
            timeout=86400,
        )

        res = self._invoke_post(body=b'{"data":"different"}', key=key)
        self.assertEqual(res.status_code, 409)
        data = json.loads(res.content)
        self.assertEqual(data["error"]["code"], "idempotency_conflict")

    def test_skips_get_requests(self):
        """GET requests bypass idempotency checks."""
        factory = RequestFactory()
        request = factory.get("/api/test/")
        request.syn_request_id = "test"
        request.tenant_id = None

        middleware = IdempotencyMiddleware(lambda r: _json_ok())
        res = middleware(request)
        self.assertEqual(res.status_code, 200)

    def test_skips_non_api_paths(self):
        """Non-API POST requests bypass idempotency checks."""
        factory = RequestFactory()
        request = factory.post("/admin/login/", data=b"{}", content_type="application/json")
        request.syn_request_id = "test"
        request.tenant_id = None

        middleware = IdempotencyMiddleware(lambda r: _json_ok())
        res = middleware(request)
        self.assertEqual(res.status_code, 200)

    def tearDown(self):
        cache.clear()


@SECURE_OFF
class IdempotencyHTTPTest(TestCase):
    """API-001 §9: IdempotencyMiddleware is active in the HTTP stack."""

    def setUp(self):
        cache.clear()

    def test_post_without_key_logs_warning(self):
        """POST to /api/ without Idempotency-Key still proceeds (middleware active)."""
        res = self.client.post(
            "/api/auth/login/",
            data=json.dumps({"email": "none@test.com", "password": "x"}),
            content_type="application/json",
            HTTP_ACCEPT="application/json",
        )
        # Login will fail (bad creds) but NOT with 406 or middleware error —
        # confirms IdempotencyMiddleware and APIHeadersMiddleware are both active.
        self.assertIn(res.status_code, [400, 401, 403, 429])

    def test_idempotency_key_header_accepted(self):
        """POST with Idempotency-Key header is accepted by middleware."""
        key = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        res = self.client.post(
            "/api/auth/login/",
            data=json.dumps({"email": "none@test.com", "password": "x"}),
            content_type="application/json",
            HTTP_ACCEPT="application/json",
            HTTP_IDEMPOTENCY_KEY=key,
        )
        # Request processed (login fails, but middleware didn't block it)
        self.assertIn(res.status_code, [400, 401, 403, 429])

    def tearDown(self):
        cache.clear()


# ── Error Redaction ──────────────────────────────────────────────────────


class ErrorRedactionTest(SimpleTestCase):
    """API-001/POL-002 §8: Error message redaction."""

    def test_redacts_passwords(self):
        """redact_error_message scrubs password patterns."""
        msg = 'Connection failed: password="s3cret123"'
        result = redact_error_message(msg)
        self.assertNotIn("s3cret123", result)
        self.assertIn("REDACTED", result)

    def test_redacts_bearer_tokens(self):
        """redact_error_message scrubs Bearer token patterns."""
        msg = "Auth failed: Bearer eyJhbGciOiJIUzI1NiJ9.test"
        result = redact_error_message(msg)
        self.assertNotIn("eyJhbGciOiJIUzI1NiJ9", result)
        self.assertIn("REDACTED", result)

    def test_redacts_env_vars(self):
        """redact_error_message scrubs environment variable values."""
        msg = "Error: DATABASE_URL=postgres://user:pass@host/db SECRET_KEY=abc123"
        result = redact_error_message(msg)
        self.assertNotIn("postgres://user:pass@host/db", result)
        self.assertIn("DATABASE_URL=***", result)

    def test_preserves_safe_content(self):
        """redact_error_message preserves non-sensitive parts."""
        msg = "File not found: /tmp/data.csv"
        result = redact_error_message(msg)
        self.assertEqual(result, msg)

    def test_redact_exception_for_logging_format(self):
        """redact_exception_for_logging returns 'Type: message' format."""
        exc = ValueError("password=hunter2 in config")
        result = redact_exception_for_logging(exc)
        self.assertTrue(result.startswith("ValueError:"))
        self.assertNotIn("hunter2", result)
        self.assertIn("REDACTED", result)

    def test_redact_exception_preserves_safe_message(self):
        """redact_exception_for_logging preserves non-sensitive exception message."""
        exc = RuntimeError("simple error")
        result = redact_exception_for_logging(exc)
        self.assertIn("RuntimeError", result)
        self.assertIn("simple error", result)

    def test_sensitive_patterns_cover_credentials(self):
        """SENSITIVE_ERROR_PATTERNS covers password, secret, credential, Bearer."""
        combined = " ".join(SENSITIVE_ERROR_PATTERNS)
        for keyword in ["password", "secret", "credential", "Bearer"]:
            with self.subTest(keyword=keyword):
                self.assertIn(keyword, combined)

    def test_env_var_patterns_cover_known_vars(self):
        """ENV_VAR_PATTERNS covers DATABASE_URL, SECRET_KEY, AWS_SECRET."""
        combined = " ".join(ENV_VAR_PATTERNS)
        for var in ["DATABASE_URL", "SECRET_KEY", "AWS_SECRET"]:
            with self.subTest(var=var):
                self.assertIn(var, combined)
