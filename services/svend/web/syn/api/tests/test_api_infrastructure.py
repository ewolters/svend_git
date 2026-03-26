"""
API-001/002 infrastructure tests: middleware, pagination, error envelope, redaction.

Tests verify error redaction, SynRequestIdMiddleware, APIHeadersMiddleware,
IdempotencyMiddleware, ErrorEnvelopeMiddleware, cursor pagination, and
cursor encoding. Each test class maps to SOC 2 controls.

Compliance: API-001/002, SOC 2 CC6.1, CC7.2, NIST SC-8/SC-23
CR: e2a78665-538a-43e6-bb84-49c46e1e375d
FEAT-088
"""

import json
import uuid
from unittest import mock

from django.http import HttpResponse, JsonResponse
from django.test import RequestFactory, SimpleTestCase

from syn.api.middleware import (
    EXEMPT_PATHS,
    HEADER_IDEMPOTENCY_KEY,
    HEADER_SYN_REQUEST_ID,
    HEADER_TRACEPARENT,
    APIHeadersMiddleware,
    ErrorEnvelopeMiddleware,
    IdempotencyMiddleware,
    SynRequestIdMiddleware,
    _is_exempt_path,
    redact_error_message,
    redact_exception_for_logging,
)
from syn.api.pagination import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    MIN_PAGE_SIZE,
    create_cursor_response,
    decode_cursor,
    encode_cursor,
)

# =============================================================================
# ERROR REDACTION TESTS (POL-002 §8) — SOC 2 CC6.1
# =============================================================================


class RedactErrorMessageTest(SimpleTestCase):
    """POL-002 §8: redact_error_message scrubs sensitive data."""

    def test_redacts_password(self):
        """Password values redacted from error messages."""
        msg = "Connection failed: password=mysecretpass123"
        result = redact_error_message(msg)
        self.assertNotIn("mysecretpass123", result)
        self.assertIn("***REDACTED***", result)

    def test_redacts_email(self):
        """Email addresses redacted from error messages."""
        msg = "User not found: admin@svend.ai"
        result = redact_error_message(msg)
        self.assertNotIn("admin@svend.ai", result)

    def test_redacts_ssn(self):
        """SSN patterns redacted from error messages."""
        msg = "Invalid SSN: 123-45-6789"
        result = redact_error_message(msg)
        self.assertNotIn("123-45-6789", result)

    def test_redacts_bearer_token(self):
        """Bearer tokens redacted from error messages."""
        msg = "Auth failed: Bearer eyJhbGciOiJIUzI1NiJ9.test"
        result = redact_error_message(msg)
        self.assertNotIn("eyJhbGciOiJIUzI1NiJ9", result)

    def test_redacts_env_vars(self):
        """Environment variable values redacted."""
        msg = "Config error: DATABASE_URL=postgres://user:pass@host/db"
        result = redact_error_message(msg)
        self.assertNotIn("postgres://user:pass@host/db", result)

    def test_clean_message_unchanged(self):
        """Clean messages pass through without modification."""
        msg = "Request processed successfully"
        result = redact_error_message(msg)
        self.assertEqual(result, msg)

    def test_redacts_api_key(self):
        """API key values redacted."""
        msg = "Invalid api_key=sk_live_abc123def456"
        result = redact_error_message(msg)
        self.assertNotIn("sk_live_abc123def456", result)


class RedactExceptionForLoggingTest(SimpleTestCase):
    """POL-002 §8: redact_exception_for_logging formats safely."""

    def test_formats_type_and_message(self):
        """Output includes exception type and message."""
        exc = ValueError("bad input")
        result = redact_exception_for_logging(exc)
        self.assertIn("ValueError", result)
        self.assertIn("bad input", result)

    def test_redacts_sensitive_in_exception(self):
        """Sensitive data in exception message is redacted."""
        exc = RuntimeError("Failed with password=secret123")
        result = redact_exception_for_logging(exc)
        self.assertNotIn("secret123", result)
        self.assertIn("***REDACTED***", result)


# =============================================================================
# EXEMPT PATH TESTS (API-002 §8) — SOC 2 CC6.1
# =============================================================================


class ExemptPathTest(SimpleTestCase):
    """API-002 §8: Exempt paths skip header validation."""

    def test_exempt_paths_recognized(self):
        """Known exempt paths return True."""
        for path in ["/admin/", "/health/", "/static/foo.js"]:
            with self.subTest(path=path):
                self.assertTrue(_is_exempt_path(path))

    def test_api_paths_not_exempt(self):
        """API paths return False."""
        self.assertFalse(_is_exempt_path("/api/dsw/"))
        self.assertFalse(_is_exempt_path("/api/internal/"))

    def test_exempt_paths_defined(self):
        """EXEMPT_PATHS has expected entries."""
        paths = set(EXEMPT_PATHS)
        self.assertIn("/health/", paths)
        self.assertIn("/admin/", paths)
        self.assertIn("/static/", paths)


# =============================================================================
# SYN REQUEST ID MIDDLEWARE TESTS (API-002 §8.1) — SOC 2 CC7.2
# =============================================================================


class SynRequestIdMiddlewareTest(SimpleTestCase):
    """API-002 §8.1: SynRequestIdMiddleware generates/extracts request IDs."""

    def setUp(self):
        self.factory = RequestFactory()

    def _make_middleware(self, response_status=200):
        def get_response(request):
            return HttpResponse(status=response_status)

        return SynRequestIdMiddleware(get_response)

    def test_generates_id_when_absent(self):
        """Generates request ID when no header provided."""
        middleware = self._make_middleware()
        request = self.factory.get("/api/test/")
        response = middleware(request)
        self.assertIn(HEADER_SYN_REQUEST_ID, response)
        self.assertTrue(len(response[HEADER_SYN_REQUEST_ID]) > 0)

    def test_extracts_from_header(self):
        """Extracts request ID from Syn-Request-Id header."""
        test_id = "req_test123"
        middleware = self._make_middleware()
        request = self.factory.get("/api/test/", HTTP_SYN_REQUEST_ID=test_id)
        response = middleware(request)
        self.assertEqual(response[HEADER_SYN_REQUEST_ID], test_id)

    def test_parses_traceparent(self):
        """Extracts trace context from W3C traceparent header."""
        trace_id = "abcdef1234567890abcdef1234567890"
        span_id = "0123456789abcdef"
        traceparent = f"00-{trace_id}-{span_id}-01"
        middleware = self._make_middleware()

        stored = {}

        def capture(request):
            stored["trace_id"] = request.syn_trace_id
            stored["span_id"] = request.syn_span_id
            return HttpResponse()

        middleware = SynRequestIdMiddleware(capture)
        request = self.factory.get("/api/test/", HTTP_TRACEPARENT=traceparent)
        middleware(request)

        self.assertEqual(stored["trace_id"], trace_id)
        self.assertEqual(stored["span_id"], span_id)

    def test_stores_on_request(self):
        """Stores syn_request_id on request object."""
        stored = {}

        def capture(request):
            stored["id"] = request.syn_request_id
            return HttpResponse()

        middleware = SynRequestIdMiddleware(capture)
        request = self.factory.get("/api/test/")
        middleware(request)

        self.assertIn("id", stored)
        self.assertTrue(len(stored["id"]) > 0)

    def test_echoes_traceparent(self):
        """Echoes traceparent in response when present."""
        traceparent = "00-abcdef1234567890abcdef1234567890-0123456789abcdef-01"
        middleware = self._make_middleware()
        request = self.factory.get("/api/test/", HTTP_TRACEPARENT=traceparent)
        response = middleware(request)
        self.assertEqual(response[HEADER_TRACEPARENT], traceparent)


# =============================================================================
# API HEADERS MIDDLEWARE TESTS (API-002 §8.2) — SOC 2 CC6.1
# =============================================================================


class APIHeadersMiddlewareTest(SimpleTestCase):
    """API-002 §8.2: APIHeadersMiddleware enforces headers."""

    def setUp(self):
        self.factory = RequestFactory()

    def _make_middleware(self, response_status=200, content_type="application/json"):
        def get_response(request):
            resp = HttpResponse(status=response_status)
            resp["Content-Type"] = content_type
            return resp

        return APIHeadersMiddleware(get_response)

    def test_skips_exempt_paths(self):
        """Exempt paths pass through without validation."""
        middleware = self._make_middleware()
        request = self.factory.get("/health/")
        response = middleware(request)
        self.assertEqual(response.status_code, 200)

    def test_rejects_non_json_accept(self):
        """Rejects API requests with non-JSON Accept header."""
        middleware = self._make_middleware()
        request = self.factory.get("/api/test/", HTTP_ACCEPT="text/html")
        response = middleware(request)
        self.assertEqual(response.status_code, 406)

    def test_allows_json_accept(self):
        """Allows API requests with JSON Accept header."""
        middleware = self._make_middleware()
        request = self.factory.get("/api/test/", HTTP_ACCEPT="application/json")
        response = middleware(request)
        self.assertEqual(response.status_code, 200)

    def test_allows_wildcard_accept(self):
        """Allows API requests with */* Accept header."""
        middleware = self._make_middleware()
        request = self.factory.get("/api/test/", HTTP_ACCEPT="*/*")
        response = middleware(request)
        self.assertEqual(response.status_code, 200)

    def test_browser_facing_api_bypasses_accept_check(self):
        """Browser-facing email endpoints bypass Accept header validation."""
        middleware = self._make_middleware()
        for path in [
            "/api/email/click/00000000-0000-0000-0000-000000000000/",
            "/api/email/open/00000000-0000-0000-0000-000000000000/",
            "/api/email/unsubscribe/",
            "/api/notifications/unsubscribe/",
        ]:
            request = self.factory.get(path, HTTP_ACCEPT="text/html")
            response = middleware(request)
            self.assertNotEqual(response.status_code, 406, f"{path} returned 406")

    def test_export_download_paths_bypass_accept_check(self):
        """PDF/file export and download paths bypass Accept header validation."""
        middleware = self._make_middleware()
        for path in [
            "/api/a3/00000000-0000-0000-0000-000000000000/export/pdf/",
            "/api/reports/00000000-0000-0000-0000-000000000000/export/pdf/",
            "/api/iso-docs/00000000-0000-0000-0000-000000000000/export/docx/",
            "/api/triage/abc123/download/",
            "/api/forge/download/00000000-0000-0000-0000-000000000000",
            "/api/dsw/models/00000000-0000-0000-0000-000000000000/",
        ]:
            request = self.factory.get(path, HTTP_ACCEPT="text/html")
            response = middleware(request)
            self.assertNotEqual(response.status_code, 406, f"{path} returned 406")

    def test_accept_header_parsed_by_media_type(self):
        """Accept header is parsed by media type, not substring."""
        middleware = self._make_middleware()
        # application/jsonl should NOT match application/json
        request = self.factory.get("/api/test/", HTTP_ACCEPT="application/jsonl")
        response = middleware(request)
        self.assertEqual(response.status_code, 406)

    def test_accept_with_quality_params(self):
        """Accept header with quality parameters is parsed correctly."""
        middleware = self._make_middleware()
        request = self.factory.get(
            "/api/test/", HTTP_ACCEPT="text/html,application/json;q=0.9"
        )
        response = middleware(request)
        self.assertEqual(response.status_code, 200)

    def test_adds_charset_to_json(self):
        """Adds charset=utf-8 to JSON Content-Type."""
        middleware = self._make_middleware()
        request = self.factory.get("/api/test/", HTTP_ACCEPT="application/json")
        response = middleware(request)
        self.assertIn("charset=utf-8", response["Content-Type"])

    def test_adds_vary_header(self):
        """Adds Vary header for caching."""
        middleware = self._make_middleware()
        request = self.factory.get("/api/test/", HTTP_ACCEPT="application/json")
        response = middleware(request)
        self.assertIn("Accept", response.get("Vary", ""))

    def test_adds_response_time(self):
        """Adds X-Response-Time header when start time available."""
        middleware = self._make_middleware()
        request = self.factory.get("/api/test/", HTTP_ACCEPT="application/json")
        request.syn_start_time = __import__("time").time() - 0.05
        response = middleware(request)
        self.assertIn("X-Response-Time", response)


# =============================================================================
# IDEMPOTENCY MIDDLEWARE TESTS (API-002 §9) — SOC 2 CC7.2
# =============================================================================


class IdempotencyMiddlewareTest(SimpleTestCase):
    """API-002 §9: IdempotencyMiddleware enforces POST idempotency."""

    def setUp(self):
        self.factory = RequestFactory()

    def _make_middleware(self, response_status=200, response_body=None):
        def get_response(request):
            body = response_body or {"result": "ok"}
            return JsonResponse(body, status=response_status)

        return IdempotencyMiddleware(get_response)

    def test_skips_get_requests(self):
        """GET requests pass through without idempotency check."""
        middleware = self._make_middleware()
        request = self.factory.get("/api/test/")
        response = middleware(request)
        self.assertEqual(response.status_code, 200)

    def test_allows_post_without_key(self):
        """POST without Idempotency-Key is allowed (with warning)."""
        middleware = self._make_middleware()
        request = self.factory.post(
            "/api/test/", data=b'{"a":1}', content_type="application/json"
        )
        response = middleware(request)
        self.assertEqual(response.status_code, 200)

    def test_skips_non_api_post(self):
        """POST to non-API path is not checked."""
        middleware = self._make_middleware()
        request = self.factory.post("/login/")
        response = middleware(request)
        self.assertEqual(response.status_code, 200)

    def test_rejects_invalid_idempotency_key_format(self):
        """Rejects Idempotency-Key that is not a valid ULID."""
        middleware = self._make_middleware()
        request = self.factory.post(
            "/api/test/",
            data=b'{"a":1}',
            content_type="application/json",
            HTTP_IDEMPOTENCY_KEY="not-a-ulid",
        )
        response = middleware(request)
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.content)
        self.assertEqual(data["error"]["code"], "invalid_idempotency_key")

    def test_conflict_on_different_payload(self):
        """Returns 409 when idempotency key reused with different payload."""
        middleware = self._make_middleware()

        # Mock cache to return cached response with different hash
        cached = {
            "payload_hash": "different_hash",
            "request_id": "req-original",
            "status_code": 200,
            "response_body": {"result": "original"},
        }

        with mock.patch.object(middleware, "_get_cached_response", return_value=cached):
            request = self.factory.post(
                "/api/test/",
                data=b'{"new": "data"}',
                content_type="application/json",
                HTTP_IDEMPOTENCY_KEY="01HX0000000000000000000000",
            )
            response = middleware(request)

        self.assertEqual(response.status_code, 409)
        data = json.loads(response.content)
        self.assertEqual(data["error"]["code"], "idempotency_conflict")

    def test_replays_cached_response(self):
        """Returns cached response for matching idempotency key + payload."""
        middleware = self._make_middleware()

        # Compute the hash of the test payload
        import hashlib

        payload = b'{"same": "data"}'
        payload_hash = hashlib.sha256(payload).hexdigest()

        cached = {
            "payload_hash": payload_hash,
            "request_id": "req-original",
            "status_code": 201,
            "response_body": {"id": "created-item"},
        }

        with mock.patch.object(middleware, "_get_cached_response", return_value=cached):
            request = self.factory.post(
                "/api/test/",
                data=payload,
                content_type="application/json",
                HTTP_IDEMPOTENCY_KEY="01HX0000000000000000000000",
            )
            response = middleware(request)

        self.assertEqual(response.status_code, 201)
        data = json.loads(response.content)
        self.assertEqual(data["id"], "created-item")


# =============================================================================
# ERROR ENVELOPE MIDDLEWARE TESTS (API-002 §10) — SOC 2 CC7.2
# =============================================================================


class ErrorEnvelopeMiddlewareTest(SimpleTestCase):
    """API-002 §10: ErrorEnvelopeMiddleware wraps errors."""

    def setUp(self):
        self.factory = RequestFactory()

    def _make_middleware(
        self, response_status=200, response_body=None, content_type="application/json"
    ):
        def get_response(request):
            if response_body is not None:
                resp = JsonResponse(response_body, status=response_status)
            else:
                resp = HttpResponse(status=response_status)
                resp["Content-Type"] = content_type
            return resp

        return ErrorEnvelopeMiddleware(get_response)

    def test_skips_non_api_paths(self):
        """Non-API paths pass through unchanged."""
        middleware = self._make_middleware(response_status=404)
        request = self.factory.get("/login/")
        response = middleware(request)
        self.assertEqual(response.status_code, 404)

    def test_skips_success_responses(self):
        """2xx responses pass through unchanged."""
        middleware = self._make_middleware(response_status=200)
        request = self.factory.get("/api/test/")
        response = middleware(request)
        self.assertEqual(response.status_code, 200)

    def test_wraps_400_error(self):
        """400 responses wrapped in error envelope."""
        middleware = self._make_middleware(
            response_status=400,
            response_body={"detail": "Invalid input"},
        )
        request = self.factory.get("/api/test/")
        request.syn_request_id = "req-test"
        response = middleware(request)

        data = json.loads(response.content)
        self.assertIn("error", data)
        self.assertEqual(data["error"]["code"], "bad_request")
        self.assertFalse(data["error"]["retryable"])
        self.assertEqual(data["error"]["request_id"], "req-test")

    def test_wraps_500_error(self):
        """500 responses wrapped with retryable=True."""
        middleware = self._make_middleware(
            response_status=500,
            response_body={"detail": "Server error"},
        )
        request = self.factory.get("/api/test/")
        request.syn_request_id = "req-500"
        response = middleware(request)

        data = json.loads(response.content)
        self.assertEqual(data["error"]["code"], "internal_error")
        self.assertTrue(data["error"]["retryable"])

    def test_preserves_existing_envelope(self):
        """Responses with existing error envelope are not re-wrapped."""
        existing_envelope = {
            "error": {
                "code": "custom_error",
                "message": "Custom message",
            }
        }
        middleware = self._make_middleware(
            response_status=400,
            response_body=existing_envelope,
        )
        request = self.factory.get("/api/test/")
        request.syn_request_id = "req-existing"
        response = middleware(request)

        data = json.loads(response.content)
        self.assertEqual(data["error"]["code"], "custom_error")

    def test_retryable_status_codes(self):
        """Correct status codes marked as retryable."""
        retryable = ErrorEnvelopeMiddleware.RETRYABLE_STATUSES
        self.assertIn(429, retryable)
        self.assertIn(503, retryable)
        self.assertIn(500, retryable)
        self.assertNotIn(400, retryable)
        self.assertNotIn(401, retryable)
        self.assertNotIn(404, retryable)

    def test_status_to_code_mapping(self):
        """STATUS_TO_CODE maps expected codes."""
        mapping = ErrorEnvelopeMiddleware.STATUS_TO_CODE
        self.assertEqual(mapping[400], "bad_request")
        self.assertEqual(mapping[401], "unauthorized")
        self.assertEqual(mapping[403], "forbidden")
        self.assertEqual(mapping[404], "not_found")
        self.assertEqual(mapping[429], "rate_limit_exceeded")
        self.assertEqual(mapping[500], "internal_error")

    def test_adds_doc_link(self):
        """Error envelope includes doc link."""
        middleware = self._make_middleware(
            response_status=404,
            response_body={"detail": "Not found"},
        )
        request = self.factory.get("/api/test/")
        request.syn_request_id = "req-doc"
        response = middleware(request)

        data = json.loads(response.content)
        self.assertIn("doc", data["error"])
        self.assertIn("not_found", data["error"]["doc"])

    def test_process_exception_synara_error(self):
        """process_exception handles SynaraError properly."""
        from syn.err.exceptions import NotFoundError

        middleware = self._make_middleware()
        request = self.factory.get("/api/test/")
        request.syn_request_id = "req-synara"

        exc = NotFoundError(message="Resource not found")
        response = middleware.process_exception(request, exc)

        self.assertIsNotNone(response)
        self.assertEqual(response.status_code, 404)
        data = json.loads(response.content)
        self.assertIn("error", data)

    def test_process_exception_generic(self):
        """process_exception wraps generic exceptions as 500."""
        middleware = self._make_middleware()
        request = self.factory.get("/api/test/")
        request.syn_request_id = "req-generic"

        exc = RuntimeError("unexpected failure")
        response = middleware.process_exception(request, exc)

        self.assertIsNotNone(response)
        self.assertEqual(response.status_code, 500)
        data = json.loads(response.content)
        self.assertEqual(data["error"]["code"], "internal_error")
        # Must NOT expose internal error details
        self.assertNotIn("unexpected failure", data["error"]["message"])

    def test_process_exception_skips_non_api(self):
        """process_exception returns None for non-API paths."""
        middleware = self._make_middleware()
        request = self.factory.get("/login/")
        request.syn_request_id = "req-non-api"

        exc = RuntimeError("error")
        response = middleware.process_exception(request, exc)

        self.assertIsNone(response)


# =============================================================================
# PAGINATION CONSTANTS TESTS (API-002 §7) — SOC 2 CC6.1
# =============================================================================


class PaginationConstantsTest(SimpleTestCase):
    """API-002 §7: Pagination constants and bounds."""

    def test_default_page_size(self):
        """Default page size is 50."""
        self.assertEqual(DEFAULT_PAGE_SIZE, 50)

    def test_min_page_size(self):
        """Minimum page size is 1."""
        self.assertEqual(MIN_PAGE_SIZE, 1)

    def test_max_page_size(self):
        """Maximum page size is 200."""
        self.assertEqual(MAX_PAGE_SIZE, 200)

    def test_min_less_than_max(self):
        """MIN_PAGE_SIZE < MAX_PAGE_SIZE."""
        self.assertLess(MIN_PAGE_SIZE, MAX_PAGE_SIZE)


# =============================================================================
# CURSOR ENCODING TESTS (API-002 §7) — SOC 2 CC6.1
# =============================================================================


class CursorEncodingTest(SimpleTestCase):
    """API-002 §7: Cursor encoding/decoding for pagination."""

    def test_encode_decode_roundtrip(self):
        """encode_cursor and decode_cursor are inverse operations."""
        position = {"created_at": "2026-01-01T00:00:00Z", "id": str(uuid.uuid4())}
        encoded = encode_cursor(position)
        decoded = decode_cursor(encoded)
        self.assertEqual(decoded, position)

    def test_decode_invalid_returns_none(self):
        """decode_cursor returns None for invalid input."""
        result = decode_cursor("not-valid-base64!!!")
        self.assertIsNone(result)

    def test_create_cursor_response_structure(self):
        """create_cursor_response has data, next_cursor, total_estimate."""
        response = create_cursor_response(
            data=[{"id": 1}, {"id": 2}],
            next_cursor="abc123",
            total_estimate=100,
        )
        self.assertEqual(len(response["data"]), 2)
        self.assertEqual(response["next_cursor"], "abc123")
        self.assertEqual(response["total_estimate"], 100)

    def test_create_cursor_response_defaults(self):
        """create_cursor_response defaults: next_cursor=None, total_estimate=0."""
        response = create_cursor_response(data=[])
        self.assertIsNone(response["next_cursor"])
        self.assertEqual(response["total_estimate"], 0)


# =============================================================================
# MIDDLEWARE CONSTANTS TESTS (API-002 §8-9) — SOC 2 CC6.1
# =============================================================================


class MiddlewareConstantsTest(SimpleTestCase):
    """API-002 §8-9: Middleware constants properly defined."""

    def test_header_names(self):
        """Header constants have expected values."""
        self.assertEqual(HEADER_SYN_REQUEST_ID, "Syn-Request-Id")
        self.assertEqual(HEADER_TRACEPARENT, "traceparent")
        self.assertEqual(HEADER_IDEMPOTENCY_KEY, "Idempotency-Key")

    def test_exempt_paths_minimum(self):
        """At least 5 exempt paths defined."""
        self.assertGreaterEqual(len(EXEMPT_PATHS), 5)
