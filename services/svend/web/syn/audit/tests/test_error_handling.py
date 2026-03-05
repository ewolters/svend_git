"""
ERR-001 compliance tests: Error Handling Standard.

Tests the Synara error hierarchy, retry patterns, circuit breaker,
bulkhead, and error envelope. Pure unit tests — no database required.

Standard: ERR-001
"""

import logging
import threading
import time
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch
from uuid import UUID

from syn.err.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ConflictError,
    DatabaseError,
    DependencyError,
    NotFoundError,
    RateLimitError,
    SynaraError,
    SystemError,
    TimeoutError,
    ValidationError,
    wrap_exception,
)
from syn.err.retry import (
    Bulkhead,
    BulkheadFullError,
    CircuitBreaker,
    CircuitBreakerOpenError,
    ExponentialBackoff,
    retry,
)
from syn.err.types import (
    CATEGORY_STATUS_CODES,
    ERROR_REGISTRY,
    RETRYABLE_CATEGORIES,
    CircuitBreakerConfig,
    CircuitBreakerState,
    ErrorCategory,
    ErrorContext,
    ErrorEnvelope,
    ErrorSeverity,
    RetryConfig,
    RetryStrategy,
    SystemLayer,
)


class ErrorHierarchyTest(unittest.TestCase):
    """ERR-001 §5: All exceptions inherit from SynaraError."""

    SUBCLASSES = [
        ValidationError,
        AuthenticationError,
        AuthorizationError,
        NotFoundError,
        ConflictError,
        RateLimitError,
        DependencyError,
        DatabaseError,
        TimeoutError,
        SystemError,
    ]

    def test_all_subclasses_inherit_from_synara_error(self):
        for cls in self.SUBCLASSES:
            self.assertTrue(
                issubclass(cls, SynaraError),
                f"{cls.__name__} does not inherit from SynaraError",
            )

    def test_synara_error_inherits_from_exception(self):
        self.assertTrue(issubclass(SynaraError, Exception))

    def test_circuit_breaker_open_error_inherits(self):
        self.assertTrue(issubclass(CircuitBreakerOpenError, SynaraError))

    def test_bulkhead_full_error_inherits(self):
        self.assertTrue(issubclass(BulkheadFullError, SynaraError))


class CorrelationIdTest(unittest.TestCase):
    """ERR-001 §6: correlation_id auto-generated if not provided."""

    def test_auto_generated_when_not_provided(self):
        err = SynaraError("test error")
        self.assertIsInstance(err.correlation_id, UUID)

    def test_explicit_correlation_id_preserved(self):
        from uuid import uuid4

        cid = uuid4()
        err = SynaraError("test error", correlation_id=cid)
        self.assertEqual(err.correlation_id, cid)

    def test_each_error_gets_unique_id(self):
        e1 = SynaraError("error 1")
        e2 = SynaraError("error 2")
        self.assertNotEqual(e1.correlation_id, e2.correlation_id)


class HttpStatusMappingTest(unittest.TestCase):
    """ERR-001 §3.1: All subclasses map to correct HTTP status codes."""

    EXPECTED_MAPPINGS = {
        ValidationError: 400,
        AuthenticationError: 401,
        AuthorizationError: 403,
        NotFoundError: 404,
        ConflictError: 409,
        RateLimitError: 429,
        DependencyError: 503,
        DatabaseError: 500,
        TimeoutError: 504,
        SystemError: 500,
    }

    def test_each_subclass_has_correct_http_status(self):
        for cls, expected_status in self.EXPECTED_MAPPINGS.items():
            err = cls("test")
            self.assertEqual(
                err.http_status,
                expected_status,
                f"{cls.__name__}.http_status = {err.http_status}, expected {expected_status}",
            )

    def test_category_status_codes_complete(self):
        for cat in ErrorCategory:
            self.assertIn(
                cat,
                CATEGORY_STATUS_CODES,
                f"ErrorCategory.{cat.name} missing from CATEGORY_STATUS_CODES",
            )


class ErrorAutoLoggingTest(unittest.TestCase):
    """ERR-001 §5.1: SynaraError auto-logs at severity-appropriate level."""

    def test_error_severity_logs_at_error(self):
        with patch("syn.err.exceptions.error_logger") as mock_logger:
            SynaraError("test", severity=ErrorSeverity.ERROR)
            mock_logger.error.assert_called_once()

    def test_warning_severity_logs_at_warning(self):
        with patch("syn.err.exceptions.error_logger") as mock_logger:
            SynaraError("test", severity=ErrorSeverity.WARNING)
            mock_logger.warning.assert_called_once()

    def test_critical_severity_logs_at_critical(self):
        with patch("syn.err.exceptions.error_logger") as mock_logger:
            SynaraError("test", severity=ErrorSeverity.CRITICAL)
            mock_logger.critical.assert_called_once()

    def test_critical_also_logs_to_audit(self):
        with patch("syn.err.exceptions.audit_logger") as mock_audit:
            SynaraError("test", severity=ErrorSeverity.CRITICAL)
            mock_audit.critical.assert_called_once()


class ErrorRegistryTest(unittest.TestCase):
    """ERR-001 §4: Error registry contains registered codes with metadata."""

    EXPECTED_CODES = [
        "bad_request",
        "validation_error",
        "unauthorized",
        "forbidden",
        "not_found",
        "conflict",
        "rate_limit_exceeded",
        "internal_error",
        "service_unavailable",
        "gateway_timeout",
        "database_error",
        "dependency_error",
    ]

    def test_all_expected_codes_registered(self):
        for code in self.EXPECTED_CODES:
            self.assertIn(code, ERROR_REGISTRY, f"Error code '{code}' not in registry")

    def test_registry_entries_have_required_fields(self):
        for code, entry in ERROR_REGISTRY.items():
            self.assertTrue(entry.code, f"{code}: missing code")
            self.assertIsInstance(entry.category, ErrorCategory)
            self.assertIsInstance(entry.http_status, int)
            self.assertTrue(entry.message_template)
            self.assertIsInstance(entry.retryable, bool)
            self.assertTrue(entry.doc_url)

    def test_retryable_codes_match_categories(self):
        for code, entry in ERROR_REGISTRY.items():
            if entry.category in RETRYABLE_CATEGORIES:
                self.assertTrue(
                    entry.retryable,
                    f"{code}: category {entry.category} is retryable but entry.retryable=False",
                )


class ErrorEnvelopeTest(unittest.TestCase):
    """ERR-001 §3: ErrorEnvelope API response format."""

    def test_to_dict_structure(self):
        envelope = ErrorEnvelope(
            code="test_error",
            message="Test message",
            retryable=False,
            request_id="req-123",
        )
        d = envelope.to_dict()
        self.assertIn("error", d)
        self.assertEqual(d["error"]["code"], "test_error")
        self.assertEqual(d["error"]["message"], "Test message")
        self.assertEqual(d["error"]["retryable"], False)
        self.assertEqual(d["error"]["request_id"], "req-123")

    def test_synara_error_to_envelope(self):
        err = SynaraError("Something went wrong", code="test_code")
        envelope = err.to_envelope(request_id="req-456")
        self.assertEqual(envelope.code, "test_code")
        self.assertEqual(envelope.message, "Something went wrong")
        self.assertEqual(envelope.request_id, "req-456")


class RetryDecoratorTest(unittest.TestCase):
    """ERR-001 §7.1: Retry decorator respects max_attempts."""

    def test_retry_respects_max_attempts(self):
        call_count = 0

        @retry(max_attempts=3, base_delay_ms=1)
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "ok"

        result = flaky()
        self.assertEqual(result, "ok")
        self.assertEqual(call_count, 3)

    def test_retry_raises_after_max_attempts(self):
        @retry(max_attempts=2, base_delay_ms=1)
        def always_fails():
            raise ConnectionError("permanent")

        with self.assertRaises(ConnectionError):
            always_fails()

    def test_no_retry_on_non_retryable(self):
        call_count = 0

        @retry(max_attempts=3, base_delay_ms=1)
        def validation_error():
            nonlocal call_count
            call_count += 1
            raise ValidationError("bad input")

        with self.assertRaises(ValidationError):
            validation_error()
        self.assertEqual(call_count, 1)


class ExponentialBackoffTest(unittest.TestCase):
    """ERR-001 §7.1: Exponential backoff calculator."""

    def test_exponential_increases(self):
        config = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay_ms=100,
            jitter_factor=0,
        )
        backoff = ExponentialBackoff(config)
        d0 = backoff.get_delay(0)
        d1 = backoff.get_delay(1)
        d2 = backoff.get_delay(2)
        self.assertEqual(d0, 100)
        self.assertEqual(d1, 200)
        self.assertEqual(d2, 400)

    def test_max_delay_cap(self):
        config = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay_ms=1000,
            max_delay_ms=5000,
            jitter_factor=0,
        )
        backoff = ExponentialBackoff(config)
        d10 = backoff.get_delay(10)
        self.assertLessEqual(d10, 5000)

    def test_constant_strategy(self):
        config = RetryConfig(
            strategy=RetryStrategy.CONSTANT,
            base_delay_ms=500,
            jitter_factor=0,
        )
        backoff = ExponentialBackoff(config)
        self.assertEqual(backoff.get_delay(0), 500)
        self.assertEqual(backoff.get_delay(5), 500)

    def test_linear_strategy(self):
        config = RetryConfig(
            strategy=RetryStrategy.LINEAR,
            base_delay_ms=100,
            jitter_factor=0,
        )
        backoff = ExponentialBackoff(config)
        self.assertEqual(backoff.get_delay(0), 100)
        self.assertEqual(backoff.get_delay(1), 200)
        self.assertEqual(backoff.get_delay(2), 300)


class CircuitBreakerTest(unittest.TestCase):
    """ERR-001 §7.3: Circuit breaker state transitions."""

    def test_starts_closed(self):
        cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=3))
        self.assertEqual(cb.state, CircuitBreakerState.CLOSED)

    def test_opens_after_failure_threshold(self):
        cb = CircuitBreaker(
            "test",
            CircuitBreakerConfig(failure_threshold=3, monitored_categories=frozenset(ErrorCategory)),
        )
        for i in range(3):
            try:
                with cb.call():
                    raise SynaraError("fail", category=ErrorCategory.DEPENDENCY)
            except SynaraError:
                pass
        self.assertEqual(cb._state, CircuitBreakerState.OPEN)

    def test_open_rejects_requests(self):
        cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=1))
        cb.force_open()
        with self.assertRaises(CircuitBreakerOpenError):
            with cb.call():
                pass

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker(
            "test",
            CircuitBreakerConfig(failure_threshold=1, recovery_timeout_ms=1),
        )
        cb.force_open()
        cb._metrics.last_failure_time = datetime.utcnow() - timedelta(seconds=1)
        self.assertEqual(cb.state, CircuitBreakerState.HALF_OPEN)

    def test_closes_after_success_in_half_open(self):
        cb = CircuitBreaker(
            "test",
            CircuitBreakerConfig(
                failure_threshold=1,
                success_threshold=1,
                recovery_timeout_ms=1,
            ),
        )
        cb.force_open()
        cb._metrics.last_failure_time = datetime.utcnow() - timedelta(seconds=1)
        _ = cb.state  # trigger transition to HALF_OPEN
        with cb.call():
            pass  # success
        self.assertEqual(cb._state, CircuitBreakerState.CLOSED)

    def test_reset_returns_to_closed(self):
        cb = CircuitBreaker("test")
        cb.force_open()
        cb.reset()
        self.assertEqual(cb._state, CircuitBreakerState.CLOSED)


class BulkheadTest(unittest.TestCase):
    """ERR-001 §7.2: Bulkhead limits concurrent calls."""

    def test_allows_within_limit(self):
        bh = Bulkhead("test", max_concurrent=2)
        with bh.acquire():
            self.assertEqual(bh.active_calls, 1)
        self.assertEqual(bh.active_calls, 0)

    def test_raises_when_full(self):
        bh = Bulkhead("test", max_concurrent=1, max_wait_ms=10)
        with bh.acquire():
            with self.assertRaises(BulkheadFullError):
                with bh.acquire():
                    pass

    def test_bulkhead_full_error_status_429(self):
        err = BulkheadFullError("full", bulkhead_name="test", max_concurrent=5)
        self.assertEqual(err.http_status, 429)

    def test_available_slots_tracks_correctly(self):
        bh = Bulkhead("test", max_concurrent=3)
        self.assertEqual(bh.available_slots, 3)
        with bh.acquire():
            self.assertEqual(bh.available_slots, 2)


class WrapExceptionTest(unittest.TestCase):
    """ERR-001 §5: wrap_exception maps stdlib exceptions."""

    def test_value_error_to_validation(self):
        err = wrap_exception(ValueError("bad"))
        self.assertIsInstance(err, ValidationError)

    def test_permission_error_to_authorization(self):
        err = wrap_exception(PermissionError("denied"))
        self.assertIsInstance(err, AuthorizationError)

    def test_file_not_found_to_not_found(self):
        err = wrap_exception(FileNotFoundError("gone"))
        self.assertIsInstance(err, NotFoundError)

    def test_connection_error_to_dependency(self):
        err = wrap_exception(ConnectionError("down"))
        self.assertIsInstance(err, DependencyError)

    def test_unknown_to_system_error(self):
        err = wrap_exception(RuntimeError("wat"))
        self.assertIsInstance(err, SystemError)

    def test_preserves_cause(self):
        original = ValueError("original")
        err = wrap_exception(original)
        self.assertIs(err.cause, original)


if __name__ == "__main__":
    unittest.main()
