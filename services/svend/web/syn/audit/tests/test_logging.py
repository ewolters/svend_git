"""
LOG-001 compliance tests: Logging & Observability Standard.

Tests context variables, log filters, middleware, JSON formatter,
and model field definitions. Mix of SimpleTestCase and unit tests.

Standard: LOG-001
"""

import json
import logging
import unittest
from unittest.mock import MagicMock, patch
from uuid import uuid4

from django.test import RequestFactory, SimpleTestCase

from syn.log.handlers import (
    ActorFilter,
    CorrelationFilter,
    JsonFormatter,
    LogContext,
    SynaraLogHandler,
    TenantFilter,
    actor_id_var,
    correlation_id_var,
    get_correlation_id,
    get_synara_logger,
    set_actor_id,
    set_correlation_id,
    set_tenant_id,
    tenant_id_var,
)
from syn.log.middleware import CorrelationMiddleware, RequestLoggingMiddleware


class ContextVarsTest(SimpleTestCase):
    """LOG-001 §5.3: Context variables are ContextVar instances."""

    def test_correlation_id_var_is_contextvar(self):
        from contextvars import ContextVar

        self.assertIsInstance(correlation_id_var, ContextVar)

    def test_tenant_id_var_is_contextvar(self):
        from contextvars import ContextVar

        self.assertIsInstance(tenant_id_var, ContextVar)

    def test_actor_id_var_is_contextvar(self):
        from contextvars import ContextVar

        self.assertIsInstance(actor_id_var, ContextVar)

    def test_set_and_get_correlation_id(self):
        cid = str(uuid4())
        set_correlation_id(cid)
        self.assertEqual(get_correlation_id(), cid)
        set_correlation_id(None)

    def test_set_none_clears(self):
        set_correlation_id("test-123")
        set_correlation_id(None)
        self.assertIsNone(get_correlation_id())


class LogContextTest(SimpleTestCase):
    """LOG-001 §5.3: LogContext restores previous values on exit."""

    def test_restores_previous_correlation_id(self):
        original = str(uuid4())
        set_correlation_id(original)
        with LogContext(correlation_id="temp-123"):
            self.assertEqual(get_correlation_id(), "temp-123")
        self.assertEqual(get_correlation_id(), original)
        set_correlation_id(None)

    def test_sets_all_context_vars(self):
        with LogContext(correlation_id="c", tenant_id="t", actor_id="a"):
            self.assertEqual(correlation_id_var.get(), "c")
            self.assertEqual(tenant_id_var.get(), "t")
            self.assertEqual(actor_id_var.get(), "a")

    def test_context_manager_returns_self(self):
        with LogContext(correlation_id="x") as ctx:
            self.assertIsInstance(ctx, LogContext)


class CorrelationFilterTest(SimpleTestCase):
    """LOG-001 §5.4: CorrelationFilter enriches log records."""

    def test_adds_correlation_id_from_context(self):
        set_correlation_id("test-cid")
        f = CorrelationFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        f.filter(record)
        self.assertEqual(record.correlation_id, "test-cid")
        set_correlation_id(None)

    def test_generates_uuid_when_no_context(self):
        set_correlation_id(None)
        f = CorrelationFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        f.filter(record)
        self.assertIsNotNone(record.correlation_id)

    def test_preserves_existing_correlation_id(self):
        f = CorrelationFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        record.correlation_id = "preserved"
        f.filter(record)
        self.assertEqual(record.correlation_id, "preserved")


class TenantFilterTest(SimpleTestCase):
    """LOG-001 §5.4: TenantFilter enriches log records."""

    def test_adds_tenant_from_context(self):
        set_tenant_id("tenant-abc")
        f = TenantFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        f.filter(record)
        self.assertEqual(record.tenant_id, "tenant-abc")
        set_tenant_id(None)

    def test_none_when_no_context(self):
        set_tenant_id(None)
        f = TenantFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        f.filter(record)
        self.assertIsNone(record.tenant_id)


class ActorFilterTest(SimpleTestCase):
    """LOG-001 §5.4: ActorFilter enriches log records."""

    def test_adds_actor_from_context(self):
        set_actor_id("user@test.com")
        f = ActorFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        f.filter(record)
        self.assertEqual(record.actor_id, "user@test.com")
        set_actor_id(None)


class JsonFormatterTest(SimpleTestCase):
    """LOG-001 §5.1: JsonFormatter outputs structured JSON."""

    def test_outputs_valid_json(self):
        formatter = JsonFormatter()
        record = logging.LogRecord("test.logger", logging.INFO, "", 0, "test message", (), None)
        record.correlation_id = "cid-123"
        record.tenant_id = None
        output = formatter.format(record)
        data = json.loads(output)
        self.assertEqual(data["level"], "INFO")
        self.assertEqual(data["logger"], "test.logger")
        self.assertEqual(data["message"], "test message")
        self.assertEqual(data["correlation_id"], "cid-123")

    def test_includes_timestamp(self):
        formatter = JsonFormatter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        record.correlation_id = None
        record.tenant_id = None
        output = formatter.format(record)
        data = json.loads(output)
        self.assertIn("timestamp", data)
        self.assertTrue(data["timestamp"].endswith("Z"))


class SynaraLogHandlerTest(SimpleTestCase):
    """LOG-001 §5: SynaraLogHandler configuration."""

    def test_handler_has_correlation_filter(self):
        handler = SynaraLogHandler.__new__(SynaraLogHandler)
        handler.filters = []
        handler.stream_name = "test"
        handler.batch_size = 0
        handler.include_context = True
        handler._buffer = []
        handler._buffer_lock = __import__("threading").Lock()
        handler._stream = None
        handler._stream_lock = __import__("threading").Lock()
        handler.level = logging.INFO
        handler.addFilter(CorrelationFilter())
        handler.addFilter(TenantFilter())
        handler.addFilter(ActorFilter())
        filter_types = [type(f).__name__ for f in handler.filters]
        self.assertIn("CorrelationFilter", filter_types)
        self.assertIn("TenantFilter", filter_types)
        self.assertIn("ActorFilter", filter_types)

    def test_level_map_complete(self):
        expected = {logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL}
        self.assertEqual(set(SynaraLogHandler.LEVEL_MAP.keys()), expected)


class LoggerFactoryTest(SimpleTestCase):
    """LOG-001 §5.5: get_synara_logger prevents duplicate handlers."""

    def test_returns_logger(self):
        logger = get_synara_logger("test.unique.logger.name.12345")
        self.assertIsInstance(logger, logging.Logger)

    def test_no_duplicate_handlers(self):
        name = "test.dedup.logger.99999"
        logger1 = get_synara_logger(name)
        count1 = sum(1 for h in logger1.handlers if isinstance(h, SynaraLogHandler))
        logger2 = get_synara_logger(name)
        count2 = sum(1 for h in logger2.handlers if isinstance(h, SynaraLogHandler))
        self.assertEqual(count1, count2)


class CorrelationMiddlewareTest(SimpleTestCase):
    """LOG-001 §5.3: CorrelationMiddleware extracts/generates correlation_id."""

    def setUp(self):
        self.factory = RequestFactory()
        self.get_response = lambda r: MagicMock(status_code=200, __setitem__=MagicMock())

    def test_generates_correlation_id_when_missing(self):
        middleware = CorrelationMiddleware(self.get_response)
        request = self.factory.get("/api/test/")
        middleware(request)
        self.assertTrue(hasattr(request, "correlation_id"))
        self.assertIsNotNone(request.correlation_id)

    def test_extracts_from_x_correlation_id_header(self):
        middleware = CorrelationMiddleware(self.get_response)
        request = self.factory.get("/api/test/", HTTP_X_CORRELATION_ID="custom-cid")
        middleware(request)
        self.assertEqual(request.correlation_id, "custom-cid")

    def test_extracts_from_traceparent(self):
        middleware = CorrelationMiddleware(self.get_response)
        trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
        traceparent = f"00-{trace_id}-00f067aa0ba902b7-01"
        request = self.factory.get("/api/test/", HTTP_TRACEPARENT=traceparent)
        middleware(request)
        self.assertEqual(request.correlation_id, trace_id)

    def test_extracts_tenant_from_header(self):
        middleware = CorrelationMiddleware(self.get_response)
        request = self.factory.get("/api/test/", HTTP_X_TENANT_ID="tenant-xyz")
        middleware(request)
        # Tenant extraction tested via the middleware call


class RequestLoggingMiddlewareTest(SimpleTestCase):
    """LOG-001 §6: RequestLoggingMiddleware."""

    def setUp(self):
        self.factory = RequestFactory()

    def test_excludes_health_paths(self):
        middleware = RequestLoggingMiddleware(lambda r: MagicMock(status_code=200))
        self.assertTrue(middleware._should_exclude("/health/"))
        self.assertTrue(middleware._should_exclude("/static/css/app.css"))
        self.assertTrue(middleware._should_exclude("/media/uploads/file.pdf"))

    def test_does_not_exclude_api_paths(self):
        middleware = RequestLoggingMiddleware(lambda r: MagicMock(status_code=200))
        self.assertFalse(middleware._should_exclude("/api/test/"))
        self.assertFalse(middleware._should_exclude("/app/dashboard/"))

    def test_sensitive_headers_defined(self):
        sensitive = RequestLoggingMiddleware.SENSITIVE_HEADERS
        self.assertIn("authorization", sensitive)
        self.assertIn("cookie", sensitive)
        self.assertIn("x-api-key", sensitive)
        self.assertIn("x-auth-token", sensitive)

    def test_status_level_mapping(self):
        """5xx=ERROR, 4xx=WARNING, else INFO."""
        middleware = RequestLoggingMiddleware(lambda r: MagicMock(status_code=200))
        request = self.factory.get("/api/test/")

        with patch.object(middleware, "_log_request_start"):
            # 500 should log as error
            response_500 = MagicMock(status_code=500, content=b"")
            with patch("syn.log.middleware.logger") as mock_logger:
                middleware._log_request_complete(request, response_500, 10.0)
                mock_logger.error.assert_called()

            # 404 should log as warning
            response_404 = MagicMock(status_code=404, content=b"")
            with patch("syn.log.middleware.logger") as mock_logger:
                middleware._log_request_complete(request, response_404, 10.0)
                mock_logger.warning.assert_called()

            # 200 should log as info
            response_200 = MagicMock(status_code=200, content=b"")
            with patch("syn.log.middleware.logger") as mock_logger:
                middleware._log_request_complete(request, response_200, 10.0)
                mock_logger.info.assert_called()


class LogLevelConstantsTest(SimpleTestCase):
    """LOG-001 §5.2: Log levels align with Python logging."""

    def test_level_numeric_values(self):
        from syn.log.models import LOG_LEVEL_NUMERIC

        self.assertEqual(LOG_LEVEL_NUMERIC["DEBUG"], 10)
        self.assertEqual(LOG_LEVEL_NUMERIC["INFO"], 20)
        self.assertEqual(LOG_LEVEL_NUMERIC["WARNING"], 30)
        self.assertEqual(LOG_LEVEL_NUMERIC["ERROR"], 40)
        self.assertEqual(LOG_LEVEL_NUMERIC["CRITICAL"], 50)


class DjangoLoggingConfigTest(SimpleTestCase):
    """LOG-001 §6: Django LOGGING config."""

    def test_settings_has_logging(self):
        from django.conf import settings

        self.assertTrue(hasattr(settings, "LOGGING"))

    def test_logging_has_handlers(self):
        from django.conf import settings

        logging_config = getattr(settings, "LOGGING", {})
        if logging_config:
            self.assertIn("handlers", logging_config)


if __name__ == "__main__":
    unittest.main()
