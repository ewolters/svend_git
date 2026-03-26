"""
LOG-001/002 infrastructure tests: structured logging, handlers, middleware, events.

Tests verify context variable propagation, log filters, SynaraLogHandler,
JsonFormatter, all 4 middleware classes, 5 models, event catalog, and
payload builders. Each test class maps to SOC 2 controls.

Compliance: LOG-001/002, SOC 2 CC7.1 (Monitoring), CC4.1, NIST AU-2/AU-3
CR: 29e31197-a7fc-436a-a5f3-8c0dc6187ed2
FEAT-087
"""

import json
import logging
import uuid
from datetime import timedelta
from unittest import mock

from django.test import RequestFactory, SimpleTestCase, TestCase
from django.utils import timezone

from syn.log.events import (
    LOG_EVENTS,
    build_alert_configured_payload,
    build_alert_triggered_payload,
    build_entry_created_payload,
    build_governance_log_alert_payload,
    build_metrics_computed_payload,
    build_siem_forward_started_payload,
    build_stream_created_payload,
    emit_log_event,
)
from syn.log.handlers import (
    ActorFilter,
    CorrelationFilter,
    JsonFormatter,
    LogContext,
    SynaraLogHandler,
    TenantFilter,
    actor_id_var,
    configure_django_logging,
    correlation_id_var,
    get_actor_id,
    get_correlation_id,
    get_synara_logger,
    get_tenant_id,
    set_actor_id,
    set_correlation_id,
    set_tenant_id,
    tenant_id_var,
)
from syn.log.middleware import (
    HEADER_CORRELATION_ID,
    HEADER_SYN_REQUEST_ID,
    AuditLoggingMiddleware,
    CorrelationMiddleware,
    PerformanceMiddleware,
    RequestLoggingMiddleware,
)
from syn.log.models import (
    LOG_LEVEL_CHOICES,
    LOG_LEVEL_NUMERIC,
    LOG_SOURCE_CHOICES,
    LogAlert,
    LogEntry,
    LogMetric,
    LogStream,
    RequestMetric,
)

# =============================================================================
# CONTEXT VARIABLE TESTS (LOG-001 §5.3) — SOC 2 CC7.1
# =============================================================================


class ContextVariableTest(SimpleTestCase):
    """LOG-001 §5.3: Context variable get/set for correlation tracking."""

    def setUp(self):
        # Reset context vars before each test
        correlation_id_var.set(None)
        tenant_id_var.set(None)
        actor_id_var.set(None)

    def test_set_get_correlation_id(self):
        """set_correlation_id stores and get_correlation_id retrieves."""
        test_id = str(uuid.uuid4())
        set_correlation_id(test_id)
        self.assertEqual(get_correlation_id(), test_id)

    def test_set_correlation_id_uuid_object(self):
        """set_correlation_id accepts UUID objects and stores as string."""
        test_uuid = uuid.uuid4()
        set_correlation_id(test_uuid)
        self.assertEqual(get_correlation_id(), str(test_uuid))

    def test_set_correlation_id_none_clears(self):
        """set_correlation_id(None) clears the value."""
        set_correlation_id("abc")
        set_correlation_id(None)
        self.assertIsNone(get_correlation_id())

    def test_set_get_tenant_id(self):
        """set_tenant_id stores and get_tenant_id retrieves."""
        test_id = str(uuid.uuid4())
        set_tenant_id(test_id)
        self.assertEqual(get_tenant_id(), test_id)

    def test_set_get_actor_id(self):
        """set_actor_id stores and get_actor_id retrieves."""
        set_actor_id("user@example.com")
        self.assertEqual(get_actor_id(), "user@example.com")

    def test_log_context_manager(self):
        """LogContext sets and restores context variables."""
        test_cid = str(uuid.uuid4())
        test_tid = str(uuid.uuid4())

        self.assertIsNone(get_correlation_id())

        with LogContext(correlation_id=test_cid, tenant_id=test_tid, actor_id="actor"):
            self.assertEqual(get_correlation_id(), test_cid)
            self.assertEqual(get_tenant_id(), test_tid)
            self.assertEqual(get_actor_id(), "actor")

        # Restored after exit
        self.assertIsNone(get_correlation_id())
        self.assertIsNone(get_tenant_id())
        self.assertIsNone(get_actor_id())


# =============================================================================
# LOG FILTER TESTS (LOG-001 §5.4) — SOC 2 CC7.1
# =============================================================================


class CorrelationFilterTest(SimpleTestCase):
    """LOG-001 §5.4: CorrelationFilter adds correlation_id to records."""

    def setUp(self):
        correlation_id_var.set(None)

    def test_adds_correlation_id_from_context(self):
        """Filter adds correlation_id from context variable."""
        test_id = str(uuid.uuid4())
        set_correlation_id(test_id)

        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        f = CorrelationFilter()
        f.filter(record)

        self.assertEqual(record.correlation_id, test_id)

    def test_preserves_existing_correlation_id(self):
        """Filter does not overwrite existing correlation_id on record."""
        existing_id = "existing-id"
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        record.correlation_id = existing_id

        f = CorrelationFilter()
        f.filter(record)

        self.assertEqual(record.correlation_id, existing_id)

    def test_generates_uuid_when_no_context(self):
        """Filter generates UUID when no context correlation_id."""
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        f = CorrelationFilter()
        f.filter(record)

        # Should be a valid UUID string
        self.assertIsNotNone(record.correlation_id)
        uuid.UUID(record.correlation_id)  # Validates format

    def test_always_returns_true(self):
        """Filter always returns True (never filters out records)."""
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        f = CorrelationFilter()
        self.assertTrue(f.filter(record))


class TenantFilterTest(SimpleTestCase):
    """LOG-001 §5.4: TenantFilter adds tenant_id to records."""

    def setUp(self):
        tenant_id_var.set(None)

    def test_adds_tenant_id_from_context(self):
        """Filter adds tenant_id from context variable."""
        test_id = str(uuid.uuid4())
        set_tenant_id(test_id)

        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        f = TenantFilter()
        f.filter(record)

        self.assertEqual(record.tenant_id, test_id)

    def test_preserves_existing_tenant_id(self):
        """Filter does not overwrite existing tenant_id on record."""
        existing_id = "existing-tenant"
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        record.tenant_id = existing_id

        f = TenantFilter()
        f.filter(record)

        self.assertEqual(record.tenant_id, existing_id)


class ActorFilterTest(SimpleTestCase):
    """LOG-001 §5.4: ActorFilter adds actor_id to records."""

    def setUp(self):
        actor_id_var.set(None)

    def test_adds_actor_id_from_context(self):
        """Filter adds actor_id from context variable."""
        set_actor_id("admin@svend.ai")

        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        f = ActorFilter()
        f.filter(record)

        self.assertEqual(record.actor_id, "admin@svend.ai")

    def test_preserves_existing_actor_id(self):
        """Filter does not overwrite existing actor_id on record."""
        existing = "existing-actor"
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        record.actor_id = existing

        f = ActorFilter()
        f.filter(record)

        self.assertEqual(record.actor_id, existing)


# =============================================================================
# JSON FORMATTER TESTS (LOG-001 §5.1) — SOC 2 CC7.1
# =============================================================================


class JsonFormatterTest(SimpleTestCase):
    """LOG-001 §5.1: JsonFormatter produces valid structured JSON."""

    def test_formats_valid_json(self):
        """Output is valid JSON with required fields."""
        record = logging.LogRecord(
            "syn.test", logging.INFO, "", 0, "test message", (), None
        )
        record.correlation_id = str(uuid.uuid4())
        record.tenant_id = None

        formatter = JsonFormatter()
        output = formatter.format(record)

        data = json.loads(output)
        self.assertIn("timestamp", data)
        self.assertIn("level", data)
        self.assertIn("logger", data)
        self.assertIn("message", data)
        self.assertEqual(data["level"], "INFO")
        self.assertEqual(data["logger"], "syn.test")
        self.assertEqual(data["message"], "test message")

    def test_includes_correlation_id(self):
        """JSON output includes correlation_id field."""
        test_cid = str(uuid.uuid4())
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        record.correlation_id = test_cid
        record.tenant_id = None

        formatter = JsonFormatter()
        data = json.loads(formatter.format(record))

        self.assertEqual(data["correlation_id"], test_cid)

    def test_includes_exception(self):
        """JSON output includes exception when present."""
        try:
            raise ValueError("test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord("test", logging.ERROR, "", 0, "error", (), exc_info)
        record.correlation_id = None
        record.tenant_id = None

        formatter = JsonFormatter()
        data = json.loads(formatter.format(record))

        self.assertIn("exception", data)
        self.assertIn("ValueError", data["exception"])


# =============================================================================
# CONFIGURE DJANGO LOGGING TESTS (LOG-002 §5) — SOC 2 CC7.1
# =============================================================================


class ConfigureDjangoLoggingTest(SimpleTestCase):
    """LOG-002 §5: configure_django_logging returns valid LOGGING dict."""

    def test_returns_valid_dict(self):
        """Returns dict with version, handlers, loggers keys."""
        config = configure_django_logging()
        self.assertIn("version", config)
        self.assertEqual(config["version"], 1)
        self.assertIn("handlers", config)
        self.assertIn("loggers", config)
        self.assertIn("filters", config)

    def test_includes_synara_handler(self):
        """Configuration includes synara handler."""
        config = configure_django_logging()
        self.assertIn("synara", config["handlers"])
        self.assertEqual(
            config["handlers"]["synara"]["class"],
            "syn.log.handlers.SynaraLogHandler",
        )

    def test_custom_loggers(self):
        """Custom loggers list is respected."""
        config = configure_django_logging(loggers=["myapp", "otherapp"])
        self.assertIn("myapp", config["loggers"])
        self.assertIn("otherapp", config["loggers"])
        self.assertNotIn("syn", config["loggers"])


# =============================================================================
# SYNARA LOG HANDLER TESTS (LOG-002 §4.2) — SOC 2 CC7.1, AU-3
# =============================================================================


class SynaraLogHandlerTest(SimpleTestCase):
    """LOG-002 §4.2: SynaraLogHandler emits to DB via LogEntry."""

    def test_level_map_complete(self):
        """LEVEL_MAP covers all Python log levels."""
        expected = {
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        }
        self.assertEqual(set(SynaraLogHandler.LEVEL_MAP.keys()), expected)

    def test_build_entry_data_fields(self):
        """_build_entry_data produces dict with required fields."""
        handler = SynaraLogHandler.__new__(SynaraLogHandler)
        handler.include_context = True

        record = logging.LogRecord(
            "syn.test", logging.WARNING, "/path", 42, "test msg", (), None
        )
        record.correlation_id = str(uuid.uuid4())
        record.tenant_id = None
        record.actor_id = None

        data = handler._build_entry_data(record)

        self.assertIn("timestamp", data)
        self.assertEqual(data["level"], "WARNING")
        self.assertEqual(data["logger"], "syn.test")
        self.assertEqual(data["message"], "test msg")
        self.assertIn("correlation_id", data)
        self.assertIn("metadata", data)
        self.assertIn("hostname", data["metadata"])
        self.assertIn("process_id", data["metadata"])

    def test_build_entry_data_captures_exception(self):
        """_build_entry_data captures exception info."""
        handler = SynaraLogHandler.__new__(SynaraLogHandler)
        handler.include_context = True

        try:
            raise RuntimeError("boom")
        except RuntimeError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord("test", logging.ERROR, "", 0, "error", (), exc_info)
        record.correlation_id = str(uuid.uuid4())
        record.tenant_id = None
        record.actor_id = None

        data = handler._build_entry_data(record)

        self.assertIsNotNone(data["exception"])
        self.assertEqual(data["exception"]["type"], "RuntimeError")
        self.assertEqual(data["exception"]["message"], "boom")

    def test_invalid_correlation_id_generates_new(self):
        """Invalid correlation_id string generates a new UUID."""
        handler = SynaraLogHandler.__new__(SynaraLogHandler)
        handler.include_context = True

        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        record.correlation_id = "not-a-uuid"
        record.tenant_id = None
        record.actor_id = None

        data = handler._build_entry_data(record)

        # Should be a valid UUID (newly generated)
        self.assertIsInstance(data["correlation_id"], uuid.UUID)

    def test_batch_mode_buffers(self):
        """In batch mode, emit buffers instead of writing immediately."""
        handler = SynaraLogHandler.__new__(SynaraLogHandler)
        handler.batch_size = 10
        handler._buffer = []
        handler._buffer_lock = __import__("threading").Lock()
        handler.include_context = True
        handler.level = logging.DEBUG
        handler.filters = []

        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        record.correlation_id = str(uuid.uuid4())
        record.tenant_id = None
        record.actor_id = None

        # Mock _write_entry and _flush_buffer to avoid DB
        with mock.patch.object(handler, "_write_entry"), mock.patch.object(
            handler, "_flush_buffer"
        ):
            handler.emit(record)

        self.assertEqual(len(handler._buffer), 1)


# =============================================================================
# CORRELATION MIDDLEWARE TESTS (LOG-001 §5.3) — SOC 2 CC7.1
# =============================================================================


class CorrelationMiddlewareTest(SimpleTestCase):
    """LOG-001 §5.3: CorrelationMiddleware propagates correlation IDs."""

    def setUp(self):
        self.factory = RequestFactory()
        correlation_id_var.set(None)
        tenant_id_var.set(None)
        actor_id_var.set(None)

    def _make_middleware(self, response_status=200):
        """Create middleware with mock get_response."""
        from django.http import HttpResponse

        def get_response(request):
            return HttpResponse(status=response_status)

        return CorrelationMiddleware(get_response)

    def test_generates_correlation_id_when_absent(self):
        """Generates UUID correlation_id when no header provided."""
        middleware = self._make_middleware()
        request = self.factory.get("/api/test/")

        response = middleware(request)

        # Response should have correlation ID header
        self.assertIn(HEADER_CORRELATION_ID, response)
        # Should be a valid UUID
        uuid.UUID(response[HEADER_CORRELATION_ID])

    def test_extracts_from_x_correlation_id(self):
        """Extracts correlation_id from X-Correlation-Id header."""
        test_id = str(uuid.uuid4())
        middleware = self._make_middleware()
        request = self.factory.get("/api/test/", HTTP_X_CORRELATION_ID=test_id)

        response = middleware(request)

        self.assertEqual(response[HEADER_CORRELATION_ID], test_id)

    def test_extracts_from_syn_request_id(self):
        """Extracts correlation_id from Syn-Request-Id header."""
        test_id = str(uuid.uuid4())
        middleware = self._make_middleware()
        request = self.factory.get("/api/test/", HTTP_SYN_REQUEST_ID=test_id)

        response = middleware(request)

        self.assertEqual(response[HEADER_CORRELATION_ID], test_id)

    def test_extracts_from_traceparent(self):
        """Extracts trace-id from W3C traceparent header."""
        trace_id = "abcdef1234567890abcdef1234567890"
        traceparent = f"00-{trace_id}-0123456789abcdef-01"
        middleware = self._make_middleware()
        request = self.factory.get("/api/test/", HTTP_TRACEPARENT=traceparent)

        response = middleware(request)

        self.assertEqual(response[HEADER_CORRELATION_ID], trace_id)

    def test_sets_both_response_headers(self):
        """Both X-Correlation-Id and Syn-Request-Id set on response."""
        middleware = self._make_middleware()
        request = self.factory.get("/api/test/")

        response = middleware(request)

        self.assertIn(HEADER_CORRELATION_ID, response)
        self.assertIn(HEADER_SYN_REQUEST_ID, response)
        self.assertEqual(
            response[HEADER_CORRELATION_ID], response[HEADER_SYN_REQUEST_ID]
        )

    def test_stores_on_request_object(self):
        """Middleware stores correlation_id on request object."""
        stored_ids = {}

        def capture_response(request):
            from django.http import HttpResponse

            stored_ids["correlation_id"] = request.correlation_id
            stored_ids["syn_correlation_id"] = request.syn_correlation_id
            return HttpResponse()

        middleware = CorrelationMiddleware(capture_response)
        request = self.factory.get("/api/test/")

        middleware(request)

        self.assertIn("correlation_id", stored_ids)
        self.assertEqual(stored_ids["correlation_id"], stored_ids["syn_correlation_id"])

    def test_clears_context_after_request(self):
        """Context variables cleared after request completes."""
        middleware = self._make_middleware()
        request = self.factory.get("/api/test/")

        middleware(request)

        self.assertIsNone(get_correlation_id())
        self.assertIsNone(get_tenant_id())
        self.assertIsNone(get_actor_id())


# =============================================================================
# REQUEST LOGGING MIDDLEWARE TESTS (LOG-001 §6) — SOC 2 CC7.1
# =============================================================================


class RequestLoggingMiddlewareTest(SimpleTestCase):
    """LOG-001 §6: RequestLoggingMiddleware logs request/response details."""

    def setUp(self):
        self.factory = RequestFactory()
        correlation_id_var.set(None)

    def _make_middleware(self, response_status=200):
        from django.http import HttpResponse

        def get_response(request):
            return HttpResponse(status=response_status)

        return RequestLoggingMiddleware(get_response)

    def test_excludes_health_path(self):
        """Health check paths are excluded from logging."""
        middleware = self._make_middleware()
        self.assertTrue(middleware._should_exclude("/health/"))
        self.assertTrue(middleware._should_exclude("/ready/"))
        self.assertTrue(middleware._should_exclude("/static/foo.js"))
        self.assertTrue(middleware._should_exclude("/favicon.ico"))

    def test_does_not_exclude_api_path(self):
        """API paths are not excluded."""
        middleware = self._make_middleware()
        self.assertFalse(middleware._should_exclude("/api/dsw/"))

    def test_extracts_client_ip_forwarded(self):
        """Extracts IP from X-Forwarded-For header."""
        middleware = self._make_middleware()
        request = self.factory.get(
            "/api/test/", HTTP_X_FORWARDED_FOR="1.2.3.4, 5.6.7.8"
        )
        ip = middleware._get_client_ip(request)
        self.assertEqual(ip, "1.2.3.4")

    def test_extracts_client_ip_real_ip(self):
        """Extracts IP from X-Real-IP header."""
        middleware = self._make_middleware()
        request = self.factory.get("/api/test/", HTTP_X_REAL_IP="10.0.0.1")
        ip = middleware._get_client_ip(request)
        self.assertEqual(ip, "10.0.0.1")

    def test_user_display_anonymous(self):
        """Anonymous request shows 'anonymous'."""
        middleware = self._make_middleware()
        request = self.factory.get("/api/test/")
        # RequestFactory doesn't set request.user by default
        display = middleware._get_user_display(request)
        self.assertEqual(display, "anonymous")

    def test_logs_on_500(self):
        """500 responses logged at error level."""
        middleware = self._make_middleware(response_status=500)
        request = self.factory.get("/api/test/")

        with mock.patch("syn.log.middleware.logger") as mock_logger:
            middleware(request)
            mock_logger.error.assert_called()

    def test_logs_on_400(self):
        """400 responses logged at warning level."""
        middleware = self._make_middleware(response_status=400)
        request = self.factory.get("/api/test/")

        with mock.patch("syn.log.middleware.logger") as mock_logger:
            middleware(request)
            mock_logger.warning.assert_called()


# =============================================================================
# PERFORMANCE MIDDLEWARE TESTS (SLA-001 §6) — SOC 2 CC7.1
# =============================================================================


class PerformanceMiddlewareTest(SimpleTestCase):
    """SLA-001 §6: PerformanceMiddleware buckets HTTP telemetry."""

    def setUp(self):
        self.factory = RequestFactory()

    def _make_middleware(self, response_status=200):
        from django.http import HttpResponse

        def get_response(request):
            return HttpResponse(status=response_status)

        return PerformanceMiddleware(get_response)

    def test_excludes_health_paths(self):
        """Health/static paths are excluded from metrics."""
        middleware = self._make_middleware()
        request = self.factory.get("/health/")

        # Should pass through without buffering
        middleware(request)
        self.assertEqual(len(middleware._buffer), 0)

    def test_normalizes_uuid_path(self):
        """UUIDs in paths replaced with <id>."""
        middleware = self._make_middleware()
        test_uuid = str(uuid.uuid4())
        normalized = middleware._normalize_path(f"/api/changes/{test_uuid}/")
        self.assertEqual(normalized, "/api/changes/<id>/")

    def test_normalizes_int_id_path(self):
        """Numeric IDs in paths replaced with <id>."""
        middleware = self._make_middleware()
        normalized = middleware._normalize_path("/api/problems/42/evidence/")
        self.assertEqual(normalized, "/api/problems/<id>/evidence/")

    def test_buffers_metrics(self):
        """Metrics buffered on API request."""
        middleware = self._make_middleware()
        # Prevent auto-flush by setting last_flush to now
        import time

        middleware._last_flush = time.monotonic()

        request = self.factory.get("/api/test/")
        middleware(request)

        self.assertEqual(len(middleware._buffer), 1)
        # Buffer entry structure: (bucket_start, method, path_pattern, status_class, duration_ms)
        entry = middleware._buffer[0]
        self.assertEqual(entry[1], "GET")  # method
        self.assertEqual(entry[3], "2xx")  # status_class

    def test_status_class_mapping(self):
        """Status codes correctly classified."""
        middleware = self._make_middleware(response_status=404)
        import time

        middleware._last_flush = time.monotonic()

        request = self.factory.get("/api/test/")
        middleware(request)

        entry = middleware._buffer[0]
        self.assertEqual(entry[3], "4xx")

    def test_adds_trailing_slash(self):
        """Paths without trailing slash get one added."""
        middleware = self._make_middleware()
        normalized = middleware._normalize_path("/api/test")
        self.assertTrue(normalized.endswith("/"))


# =============================================================================
# AUDIT LOGGING MIDDLEWARE TESTS (AUD-001 §5) — SOC 2 CC7.2
# =============================================================================


class AuditLoggingMiddlewareTest(SimpleTestCase):
    """AUD-001 §5: AuditLoggingMiddleware creates audit entries."""

    def setUp(self):
        self.factory = RequestFactory()

    def _make_middleware(self, response_status=200):
        from django.http import HttpResponse

        def get_response(request):
            return HttpResponse(status=response_status)

        return AuditLoggingMiddleware(get_response)

    def test_audits_post_api(self):
        """POST to /api/ path is flagged for audit."""
        middleware = self._make_middleware()
        self.assertTrue(middleware._should_audit(self.factory.post("/api/dsw/")))

    def test_skips_get_request(self):
        """GET requests are not audited by default."""
        middleware = self._make_middleware()
        self.assertFalse(middleware._should_audit(self.factory.get("/api/dsw/")))

    def test_skips_non_api_path(self):
        """Non-API paths are not audited."""
        middleware = self._make_middleware()
        self.assertFalse(middleware._should_audit(self.factory.post("/login/")))

    def test_path_to_event_basic(self):
        """_path_to_event converts path to dotted event name."""
        middleware = self._make_middleware()
        event = middleware._path_to_event("/api/dsw/analyze/")
        self.assertEqual(event, "api.dsw.analyze")

    def test_path_to_event_strips_uuids(self):
        """_path_to_event removes UUIDs from path."""
        middleware = self._make_middleware()
        test_uuid = str(uuid.uuid4())
        event = middleware._path_to_event(f"/api/changes/{test_uuid}/")
        self.assertNotIn(test_uuid, event)

    def test_skips_audit_on_500(self):
        """500 responses are not audited."""
        middleware = self._make_middleware(response_status=500)
        request = self.factory.post("/api/test/")

        with mock.patch("syn.audit.generate_entry") as mock_gen:
            middleware(request)
            mock_gen.assert_not_called()


# =============================================================================
# LOG CONSTANTS TESTS (LOG-001 §5.2) — SOC 2 CC7.1
# =============================================================================


class LogConstantsTest(SimpleTestCase):
    """LOG-001 §5.2: Log constants are properly defined."""

    def test_log_level_choices_complete(self):
        """All 5 standard log levels defined."""
        levels = {choice[0] for choice in LOG_LEVEL_CHOICES}
        self.assertEqual(levels, {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

    def test_log_level_numeric_ordering(self):
        """Numeric levels increase in severity."""
        self.assertLess(LOG_LEVEL_NUMERIC["DEBUG"], LOG_LEVEL_NUMERIC["INFO"])
        self.assertLess(LOG_LEVEL_NUMERIC["INFO"], LOG_LEVEL_NUMERIC["WARNING"])
        self.assertLess(LOG_LEVEL_NUMERIC["WARNING"], LOG_LEVEL_NUMERIC["ERROR"])
        self.assertLess(LOG_LEVEL_NUMERIC["ERROR"], LOG_LEVEL_NUMERIC["CRITICAL"])

    def test_log_source_choices_defined(self):
        """LOG_SOURCE_CHOICES has expected sources."""
        sources = {choice[0] for choice in LOG_SOURCE_CHOICES}
        self.assertIn("application", sources)
        self.assertIn("middleware", sources)
        self.assertIn("handler", sources)


# =============================================================================
# LOG STREAM MODEL TESTS (LOG-002 §4.3) — SOC 2 CC7.1
# =============================================================================


class LogStreamModelTest(TestCase):
    """LOG-002 §4.3: LogStream model for log grouping."""

    def test_create_stream(self):
        """Create a log stream with required fields."""
        stream = LogStream.objects.create(
            name="test-application",
            retention_days=90,
            min_level="INFO",
        )
        self.assertIsNotNone(stream.id)
        self.assertEqual(stream.name, "test-application")
        self.assertTrue(stream.is_active)

    def test_unique_name_constraint(self):
        """Stream names must be unique."""
        LogStream.objects.create(name="unique-stream")
        from django.db import IntegrityError

        with self.assertRaises(IntegrityError):
            LogStream.objects.create(name="unique-stream")

    def test_min_level_numeric(self):
        """min_level_numeric returns correct numeric value."""
        stream = LogStream(min_level="ERROR")
        self.assertEqual(stream.min_level_numeric, 40)

    def test_default_retention_90_days(self):
        """Default retention is 90 days."""
        stream = LogStream()
        self.assertEqual(stream.retention_days, 90)

    def test_str_representation(self):
        """__str__ includes name, min_level, retention."""
        stream = LogStream(name="app", min_level="WARNING", retention_days=30)
        s = str(stream)
        self.assertIn("app", s)
        self.assertIn("WARNING", s)
        self.assertIn("30", s)


# =============================================================================
# LOG ENTRY MODEL TESTS (LOG-002 §4.2) — SOC 2 CC7.1, AU-3
# =============================================================================


class LogEntryModelTest(TestCase):
    """LOG-002 §4.2: LogEntry stores structured log records."""

    def setUp(self):
        self.stream = LogStream.objects.create(
            name="test-entry-stream",
            min_level="DEBUG",
        )

    def test_create_entry(self):
        """Create a log entry with required fields."""
        entry = LogEntry.objects.create(
            stream=self.stream,
            level="INFO",
            logger="syn.test",
            message="Test log message",
            correlation_id=uuid.uuid4(),
        )
        self.assertIsNotNone(entry.id)
        self.assertEqual(entry.level, "INFO")

    def test_level_numeric_auto_computed(self):
        """level_numeric is auto-computed from level on save."""
        entry = LogEntry.objects.create(
            stream=self.stream,
            level="ERROR",
            logger="syn.test",
            message="Error message",
            correlation_id=uuid.uuid4(),
        )
        self.assertEqual(entry.level_numeric, 40)

    def test_is_error_property(self):
        """is_error returns True for ERROR and CRITICAL."""
        entry = LogEntry(level="ERROR")
        self.assertTrue(entry.is_error)

        entry2 = LogEntry(level="CRITICAL")
        self.assertTrue(entry2.is_error)

        entry3 = LogEntry(level="WARNING")
        self.assertFalse(entry3.is_error)

    def test_ordering_most_recent_first(self):
        """Default ordering is -timestamp."""
        self.assertEqual(LogEntry._meta.ordering, ["-timestamp"])

    def test_uuid_primary_key(self):
        """Primary key is UUID."""
        entry = LogEntry.objects.create(
            stream=self.stream,
            level="INFO",
            logger="test",
            message="msg",
            correlation_id=uuid.uuid4(),
        )
        self.assertIsInstance(entry.pk, uuid.UUID)

    def test_exception_field_stores_dict(self):
        """Exception field stores structured exception data."""
        exc_data = {
            "type": "ValueError",
            "message": "invalid input",
            "traceback": ["line 1", "line 2"],
        }
        entry = LogEntry.objects.create(
            stream=self.stream,
            level="ERROR",
            logger="test",
            message="Error occurred",
            correlation_id=uuid.uuid4(),
            exception=exc_data,
        )
        entry.refresh_from_db()
        self.assertEqual(entry.exception["type"], "ValueError")


# =============================================================================
# LOG ALERT MODEL TESTS (LOG-002 §4.4) — SOC 2 CC7.1
# =============================================================================


class LogAlertModelTest(TestCase):
    """LOG-002 §4.4: LogAlert threshold-based alerting."""

    def setUp(self):
        self.stream = LogStream.objects.create(
            name="test-alert-stream",
            min_level="INFO",
        )

    def test_create_alert(self):
        """Create alert with threshold configuration."""
        alert = LogAlert.objects.create(
            name="High Error Rate",
            stream=self.stream,
            level="ERROR",
            threshold_count=10,
            threshold_window_minutes=5,
        )
        self.assertIsNotNone(alert.id)
        self.assertTrue(alert.is_enabled)

    def test_not_in_cooldown_initially(self):
        """Alert not in cooldown when never triggered."""
        alert = LogAlert(
            name="Test",
            stream=self.stream,
            cooldown_minutes=15,
        )
        self.assertFalse(alert.is_in_cooldown())

    def test_in_cooldown_after_trigger(self):
        """Alert in cooldown immediately after trigger."""
        alert = LogAlert.objects.create(
            name="Cooldown Test",
            stream=self.stream,
            cooldown_minutes=15,
            last_triggered_at=timezone.now(),
        )
        self.assertTrue(alert.is_in_cooldown())

    def test_cooldown_expires(self):
        """Alert not in cooldown after cooldown period passes."""
        alert = LogAlert.objects.create(
            name="Expired Cooldown",
            stream=self.stream,
            cooldown_minutes=15,
            last_triggered_at=timezone.now() - timedelta(minutes=20),
        )
        self.assertFalse(alert.is_in_cooldown())


# =============================================================================
# LOG METRIC MODEL TESTS (LOG-002 §4.5) — SOC 2 CC7.1
# =============================================================================


class LogMetricModelTest(TestCase):
    """LOG-002 §4.5: LogMetric aggregated metrics."""

    def setUp(self):
        self.stream = LogStream.objects.create(
            name="test-metric-stream",
            min_level="INFO",
        )

    def test_create_metric(self):
        """Create metric bucket."""
        metric = LogMetric.objects.create(
            stream=self.stream,
            bucket_start=timezone.now(),
            level="INFO",
            count=100,
            error_count=5,
        )
        self.assertIsNotNone(metric.id)

    def test_bucket_end_computed(self):
        """bucket_end is bucket_start + duration."""
        now = timezone.now()
        metric = LogMetric(
            bucket_start=now,
            bucket_duration_minutes=5,
        )
        expected = now + timedelta(minutes=5)
        self.assertEqual(metric.bucket_end, expected)

    def test_error_rate(self):
        """error_rate = error_count / count."""
        metric = LogMetric(count=100, error_count=5)
        self.assertAlmostEqual(metric.error_rate, 0.05)

    def test_error_rate_zero_count(self):
        """error_rate returns 0.0 when count is 0."""
        metric = LogMetric(count=0, error_count=0)
        self.assertEqual(metric.error_rate, 0.0)


# =============================================================================
# REQUEST METRIC MODEL TESTS (SLA-001 §6) — SOC 2 CC7.1
# =============================================================================


class RequestMetricModelTest(TestCase):
    """SLA-001 §6: RequestMetric HTTP telemetry aggregates."""

    def test_create_metric(self):
        """Create request metric bucket."""
        metric = RequestMetric.objects.create(
            bucket_start=timezone.now(),
            method="GET",
            path_pattern="/api/test/",
            status_class="2xx",
            request_count=50,
            total_duration_ms=2500.0,
            min_duration_ms=10.0,
            max_duration_ms=200.0,
            duration_samples=[10.0, 50.0, 100.0, 200.0],
        )
        self.assertIsNotNone(metric.id)

    def test_avg_duration(self):
        """avg_duration_ms = total / count."""
        metric = RequestMetric(request_count=100, total_duration_ms=5000.0)
        self.assertAlmostEqual(metric.avg_duration_ms, 50.0)

    def test_avg_duration_zero_count(self):
        """avg_duration_ms returns 0.0 when count is 0."""
        metric = RequestMetric(request_count=0, total_duration_ms=0.0)
        self.assertEqual(metric.avg_duration_ms, 0.0)

    def test_percentile_computation(self):
        """percentile() returns correct value."""
        metric = RequestMetric(duration_samples=[10.0, 20.0, 30.0, 40.0, 50.0])
        p50 = metric.percentile(50)
        self.assertAlmostEqual(p50, 30.0)

    def test_percentile_empty_samples(self):
        """percentile() returns None for empty samples."""
        metric = RequestMetric(duration_samples=[])
        self.assertIsNone(metric.percentile(50))

    def test_percentile_single_sample(self):
        """percentile() with one sample returns that sample."""
        metric = RequestMetric(duration_samples=[42.0])
        self.assertAlmostEqual(metric.percentile(99), 42.0)


# =============================================================================
# EVENT CATALOG TESTS (EVT-001 §5) — SOC 2 CC4.1
# =============================================================================


class EventCatalogTest(SimpleTestCase):
    """EVT-001 §5: LOG_EVENTS catalog structure and completeness."""

    def test_all_events_have_required_keys(self):
        """Every event has description, category, severity, payload_schema."""
        for name, event in LOG_EVENTS.items():
            with self.subTest(event=name):
                self.assertIn("description", event)
                self.assertIn("category", event)
                self.assertIn("severity", event)
                self.assertIn("payload_schema", event)

    def test_event_names_dotted_convention(self):
        """Event names follow log.category.action pattern."""
        import re

        pattern = re.compile(r"^log\.[a-z]+\.[a-z_]+$")
        for name in LOG_EVENTS:
            with self.subTest(event=name):
                self.assertRegex(
                    name,
                    pattern,
                    f"Event name '{name}' doesn't match dotted convention",
                )

    def test_valid_severity_values(self):
        """Severity values are info, warning, or error."""
        valid = {"info", "warning", "error"}
        for name, event in LOG_EVENTS.items():
            with self.subTest(event=name):
                self.assertIn(event["severity"], valid)

    def test_minimum_event_count(self):
        """At least 20 events defined in catalog."""
        self.assertGreaterEqual(len(LOG_EVENTS), 20)

    def test_expected_categories_present(self):
        """Key categories exist: entry, stream, alert, metric, retention, error."""
        categories = {event["category"] for event in LOG_EVENTS.values()}
        for expected in ("entry", "stream", "alert", "metric", "retention", "error"):
            with self.subTest(category=expected):
                self.assertIn(expected, categories)


# =============================================================================
# EMIT LOG EVENT TESTS (EVT-001 §6) — SOC 2 CC4.1
# =============================================================================


class EmitLogEventTest(SimpleTestCase):
    """EVT-001 §6: emit_log_event validates and emits."""

    def test_raises_on_unknown_event(self):
        """ValueError raised for unknown event name."""
        with self.assertRaises(ValueError):
            emit_log_event("log.nonexistent.event", {})

    def test_accepts_valid_event(self):
        """Valid event name accepted without error."""
        # Cortex import will fail, falls back to local logger — no error raised
        emit_log_event("log.entry.created", {"log_id": str(uuid.uuid4())})


# =============================================================================
# PAYLOAD BUILDER TESTS (EVT-001 §6.2) — SOC 2 CC7.1
# =============================================================================


class PayloadBuilderTest(SimpleTestCase):
    """EVT-001 §6.2: Payload builders produce valid event data."""

    def test_entry_created_payload(self):
        """build_entry_created_payload has required fields."""
        payload = build_entry_created_payload(
            log_id=uuid.uuid4(),
            stream_name="application",
            level="INFO",
            logger="syn.test",
            correlation_id=uuid.uuid4(),
        )
        self.assertIn("log_id", payload)
        self.assertIn("stream_name", payload)
        self.assertIn("level", payload)
        self.assertEqual(payload["stream_name"], "application")

    def test_stream_created_payload(self):
        """build_stream_created_payload has required fields."""
        payload = build_stream_created_payload(
            stream_id=uuid.uuid4(),
            name="security",
            retention_days=365,
            min_level="WARNING",
        )
        self.assertEqual(payload["name"], "security")
        self.assertEqual(payload["retention_days"], 365)

    def test_alert_configured_payload(self):
        """build_alert_configured_payload has required fields."""
        payload = build_alert_configured_payload(
            alert_id=uuid.uuid4(),
            name="High Error Rate",
            stream_name="application",
            level="ERROR",
            threshold_count=10,
            threshold_window_minutes=5,
        )
        self.assertEqual(payload["threshold_count"], 10)
        self.assertEqual(payload["threshold_window_minutes"], 5)

    def test_alert_triggered_payload(self):
        """build_alert_triggered_payload has required fields."""
        payload = build_alert_triggered_payload(
            alert_id=uuid.uuid4(),
            alert_name="Test Alert",
            stream_name="app",
            level="ERROR",
            count=15,
            threshold=10,
            window_minutes=5,
        )
        self.assertEqual(payload["count"], 15)
        self.assertIn("alert_name", payload)

    def test_metrics_computed_payload(self):
        """build_metrics_computed_payload has required fields."""
        payload = build_metrics_computed_payload(
            metric_id=uuid.uuid4(),
            stream_name="app",
            bucket_start=timezone.now(),
            bucket_duration_minutes=5,
            count=100,
            error_count=5,
        )
        self.assertEqual(payload["count"], 100)
        self.assertEqual(payload["error_count"], 5)

    def test_siem_forward_started_payload(self):
        """build_siem_forward_started_payload has required fields."""
        payload = build_siem_forward_started_payload(
            stream_name="security",
            siem_target="splunk",
            batch_size=500,
        )
        self.assertEqual(payload["siem_target"], "splunk")
        self.assertEqual(payload["batch_size"], 500)

    def test_governance_alert_payload(self):
        """build_governance_log_alert_payload has required fields."""
        payload = build_governance_log_alert_payload(
            alert_type="error_rate_high",
            message="Error rate exceeded threshold",
            severity="warning",
            recommended_action="Investigate error logs",
        )
        self.assertEqual(payload["alert_type"], "error_rate_high")
        self.assertIn("recommended_action", payload)


# =============================================================================
# GET SYNARA LOGGER TESTS (LOG-001 §5.5) — SOC 2 CC7.1
# =============================================================================


class GetSynaraLoggerTest(SimpleTestCase):
    """LOG-001 §5.5: get_synara_logger factory."""

    def test_returns_logger(self):
        """Returns a configured logging.Logger instance."""
        logger = get_synara_logger("syn.test.factory")
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "syn.test.factory")

    def test_adds_synara_handler(self):
        """Logger has SynaraLogHandler attached."""
        # Use unique name to avoid handler accumulation
        name = f"syn.test.handler.{uuid.uuid4().hex[:8]}"
        logger = get_synara_logger(name)
        has_synara = any(isinstance(h, SynaraLogHandler) for h in logger.handlers)
        self.assertTrue(has_synara)

    def test_idempotent_handler_addition(self):
        """Calling twice does not add duplicate handlers."""
        name = f"syn.test.idempotent.{uuid.uuid4().hex[:8]}"
        get_synara_logger(name)
        get_synara_logger(name)

        logger = logging.getLogger(name)
        synara_handlers = [
            h for h in logger.handlers if isinstance(h, SynaraLogHandler)
        ]
        self.assertEqual(len(synara_handlers), 1)
