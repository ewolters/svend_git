"""
ERR-001/002 compliance tests: Error infrastructure lock-in.

Tests verify the SynaraError exception hierarchy, circuit breaker state machine,
exponential backoff/retry, bulkhead concurrency limiting, and error envelope
serialization. Each test class maps to SOC 2 controls.

Compliance: ERR-001 (Error Handling), ERR-002 (Error Envelope), SOC 2 CC7.2, CC9.1
"""

import threading
import time
from unittest import mock
from uuid import UUID, uuid4

from django.test import SimpleTestCase

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
    create_error_from_code,
    wrap_exception,
)
from syn.err.retry import (
    Bulkhead,
    BulkheadFullError,
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitBreakerRegistry,
    ExponentialBackoff,
    retry,
    with_circuit_breaker,
)
from syn.err.types import (
    CATEGORY_STATUS_CODES,
    ERROR_REGISTRY,
    RETRYABLE_CATEGORIES,
    SEVERITY_LEVELS,
    CircuitBreakerConfig,
    CircuitBreakerState,
    ErrorCategory,
    ErrorContext,
    ErrorDetail,
    ErrorEnvelope,
    ErrorRegistryEntry,
    ErrorSeverity,
    RecoveryMode,
    RetryConfig,
    RetryStrategy,
    SystemLayer,
)

# =============================================================================
# EXCEPTION HIERARCHY TESTS (ERR-001 §5, SOC 2 CC7.2)
# =============================================================================


class ExceptionHierarchyTest(SimpleTestCase):
    """ERR-001 §5: SynaraError hierarchy contracts."""

    # ----- subclass instantiation + attributes -----

    def test_all_subclasses_are_synara_errors(self):
        """All 10 subclasses inherit from SynaraError."""
        subclasses = [
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
        for cls in subclasses:
            self.assertTrue(
                issubclass(cls, SynaraError),
                f"{cls.__name__} does not inherit from SynaraError",
            )

    def test_validation_error_attributes(self):
        """ValidationError stores field/expected/received."""
        err = ValidationError(
            "Invalid ID format",
            field="capa_id",
            expected="UUID v4",
            received="not-a-uuid",
        )
        self.assertEqual(err.field, "capa_id")
        self.assertEqual(err.expected, "UUID v4")
        self.assertEqual(err.received, "not-a-uuid")
        self.assertEqual(err.http_status, 400)
        self.assertFalse(err.retryable)

    def test_authentication_error_attributes(self):
        """AuthenticationError stores auth_method."""
        err = AuthenticationError("JWT expired", auth_method="jwt")
        self.assertEqual(err.auth_method, "jwt")
        self.assertEqual(err.http_status, 401)
        self.assertFalse(err.retryable)

    def test_authorization_error_attributes(self):
        """AuthorizationError stores required_permission."""
        err = AuthorizationError(
            "Forbidden",
            required_permission="admin:write",
            resource="CAPA",
        )
        self.assertEqual(err.required_permission, "admin:write")
        self.assertEqual(err.resource, "CAPA")
        self.assertEqual(err.http_status, 403)

    def test_not_found_error_attributes(self):
        """NotFoundError stores resource_type/resource_id."""
        err = NotFoundError(
            "CAPA not found",
            resource_type="CAPA",
            resource_id="abc-123",
        )
        self.assertEqual(err.resource_type, "CAPA")
        self.assertEqual(err.resource_id, "abc-123")
        self.assertEqual(err.http_status, 404)

    def test_conflict_error_attributes(self):
        """ConflictError stores current_state/requested_action."""
        err = ConflictError(
            "Already closed",
            current_state="CLOSED",
            requested_action="close",
        )
        self.assertEqual(err.current_state, "CLOSED")
        self.assertEqual(err.requested_action, "close")
        self.assertEqual(err.http_status, 409)

    def test_rate_limit_error_attributes(self):
        """RateLimitError stores limit/window/retry_after."""
        err = RateLimitError(
            "Rate exceeded",
            limit=100,
            window_seconds=60,
            retry_after_seconds=30,
        )
        self.assertEqual(err.limit, 100)
        self.assertEqual(err.window_seconds, 60)
        self.assertEqual(err.retry_after_seconds, 30)
        self.assertEqual(err.http_status, 429)
        self.assertTrue(err.retryable)

    def test_dependency_error_attributes(self):
        """DependencyError stores service_name/endpoint."""
        err = DependencyError(
            "Service down",
            service_name="doc-svc",
            endpoint="/api/v1/docs",
            status_code=503,
        )
        self.assertEqual(err.service_name, "doc-svc")
        self.assertEqual(err.endpoint, "/api/v1/docs")
        self.assertEqual(err.service_status_code, 503)
        self.assertEqual(err.http_status, 503)
        self.assertTrue(err.retryable)

    def test_database_error_attributes(self):
        """DatabaseError stores operation/table/constraint."""
        err = DatabaseError(
            "Insert failed",
            operation="INSERT",
            table="capa",
            constraint="unique_capa_id",
        )
        self.assertEqual(err.operation, "INSERT")
        self.assertEqual(err.table, "capa")
        self.assertEqual(err.constraint, "unique_capa_id")
        self.assertEqual(err.http_status, 500)

    def test_timeout_error_attributes(self):
        """TimeoutError stores operation/timeout_ms/elapsed_ms."""
        err = TimeoutError(
            "Analysis timed out",
            operation="analyze_rca",
            timeout_ms=30000,
            elapsed_ms=30500,
        )
        self.assertEqual(err.operation, "analyze_rca")
        self.assertEqual(err.timeout_ms, 30000)
        self.assertEqual(err.elapsed_ms, 30500)
        self.assertEqual(err.http_status, 504)
        self.assertTrue(err.retryable)

    def test_system_error_attributes(self):
        """SystemError stores component."""
        err = SystemError("Unexpected state", component="CognitionEngine")
        self.assertEqual(err.component, "CognitionEngine")
        self.assertEqual(err.http_status, 500)

    # ----- HTTP status codes -----

    def test_http_status_codes_match_categories(self):
        """Each subclass maps to the correct HTTP status via its category."""
        expectations = [
            (ValidationError("x"), 400),
            (AuthenticationError("x"), 401),
            (AuthorizationError("x"), 403),
            (NotFoundError("x"), 404),
            (ConflictError("x"), 409),
            (RateLimitError("x"), 429),
            (DependencyError("x"), 503),
            (DatabaseError("x"), 500),
            (TimeoutError("x"), 504),
            (SystemError("x"), 500),
        ]
        for err, expected_status in expectations:
            self.assertEqual(
                err.http_status,
                expected_status,
                f"{err.__class__.__name__} expected {expected_status}, got {err.http_status}",
            )

    # ----- retryable derivation -----

    def test_retryable_derived_from_category(self):
        """Only RATE_LIMIT, DEPENDENCY, TIMEOUT categories are retryable."""
        retryable_classes = [RateLimitError, DependencyError, TimeoutError]
        non_retryable_classes = [
            ValidationError,
            AuthenticationError,
            AuthorizationError,
            NotFoundError,
            ConflictError,
            DatabaseError,
            SystemError,
        ]
        for cls in retryable_classes:
            self.assertTrue(cls("x").retryable, f"{cls.__name__} should be retryable")
        for cls in non_retryable_classes:
            # DatabaseError category is DATABASE — check it's not in RETRYABLE_CATEGORIES
            err = cls("x")
            self.assertFalse(err.retryable, f"{cls.__name__} should NOT be retryable")

    # ----- correlation ID -----

    def test_correlation_id_auto_generated(self):
        """Correlation ID is auto-generated UUID if not provided."""
        err = SynaraError("test")
        self.assertIsInstance(err.correlation_id, UUID)

    def test_correlation_id_respected(self):
        """Explicit correlation_id is preserved."""
        cid = uuid4()
        err = SynaraError("test", correlation_id=cid)
        self.assertEqual(err.correlation_id, cid)

    # ----- to_dict -----

    def test_to_dict_required_keys(self):
        """to_dict() includes all required keys."""
        err = ValidationError("Bad input", field="name")
        d = err.to_dict()
        required = {
            "error_type",
            "code",
            "message",
            "category",
            "severity",
            "correlation_id",
            "timestamp",
            "retryable",
            "http_status",
            "doc_url",
        }
        for key in required:
            self.assertIn(key, d, f"to_dict() missing key: {key}")

    def test_to_dict_includes_details(self):
        """to_dict() includes field-level details for ValidationError."""
        err = ValidationError("Bad", field="email", received="not-email")
        d = err.to_dict()
        self.assertIn("details", d)
        self.assertEqual(len(d["details"]), 1)
        self.assertEqual(d["details"][0]["field"], "email")

    def test_to_dict_includes_cause(self):
        """to_dict() includes cause when set."""
        original = ValueError("original error")
        err = SynaraError("wrapped", cause=original)
        d = err.to_dict()
        self.assertIn("cause", d)
        self.assertEqual(d["cause"]["type"], "ValueError")

    # ----- to_envelope -----

    def test_to_envelope_structure(self):
        """to_envelope() produces valid ErrorEnvelope."""
        err = NotFoundError("CAPA not found", resource_type="CAPA")
        envelope = err.to_envelope(request_id="req-123")
        self.assertIsInstance(envelope, ErrorEnvelope)
        self.assertEqual(envelope.code, "not_found")
        self.assertEqual(envelope.request_id, "req-123")
        self.assertFalse(envelope.retryable)

    def test_to_envelope_to_dict_has_error_key(self):
        """ErrorEnvelope.to_dict() wraps under 'error' key per ERR-002."""
        err = SystemError("Crash")
        d = err.to_envelope().to_dict()
        self.assertIn("error", d)
        for key in ("code", "message", "retryable", "request_id"):
            self.assertIn(key, d["error"])


class WrapExceptionTest(SimpleTestCase):
    """ERR-001 §5: wrap_exception() maps stdlib → SynaraError."""

    def test_value_error_maps_to_validation(self):
        """ValueError → ValidationError."""
        err = wrap_exception(ValueError("bad value"))
        self.assertIsInstance(err, ValidationError)
        self.assertEqual(err.cause.__class__, ValueError)

    def test_permission_error_maps_to_authorization(self):
        """PermissionError → AuthorizationError."""
        err = wrap_exception(PermissionError("denied"))
        self.assertIsInstance(err, AuthorizationError)

    def test_file_not_found_maps_to_not_found(self):
        """FileNotFoundError → NotFoundError."""
        err = wrap_exception(FileNotFoundError("missing"))
        self.assertIsInstance(err, NotFoundError)

    def test_connection_error_maps_to_dependency(self):
        """ConnectionError → DependencyError."""
        err = wrap_exception(ConnectionError("refused"))
        self.assertIsInstance(err, DependencyError)

    def test_unknown_maps_to_system_error(self):
        """Unknown exception type → SystemError."""
        err = wrap_exception(RuntimeError("kaboom"))
        self.assertIsInstance(err, SystemError)

    def test_custom_message_overrides(self):
        """Custom message overrides original exception message."""
        err = wrap_exception(ValueError("original"), message="custom msg")
        self.assertEqual(err.message, "custom msg")

    def test_correlation_id_preserved(self):
        """Correlation ID is passed through to wrapped error."""
        cid = uuid4()
        err = wrap_exception(ValueError("x"), correlation_id=cid)
        self.assertEqual(err.correlation_id, cid)


class CreateErrorFromCodeTest(SimpleTestCase):
    """ERR-002 §4: create_error_from_code() from registry."""

    def test_known_code_creates_error(self):
        """Known registry code creates SynaraError."""
        err = create_error_from_code("not_found")
        self.assertIsInstance(err, SynaraError)
        self.assertEqual(err.code, "not_found")

    def test_unknown_code_raises_value_error(self):
        """Unknown code raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            create_error_from_code("nonexistent_code")
        self.assertIn("Unknown error code", str(ctx.exception))

    def test_custom_message_overrides_template(self):
        """Custom message replaces default template."""
        err = create_error_from_code("not_found", message="Custom 404")
        self.assertEqual(err.message, "Custom 404")

    def test_correlation_id_passed_through(self):
        """Correlation ID is set on created error."""
        cid = uuid4()
        err = create_error_from_code("internal_error", correlation_id=cid)
        self.assertEqual(err.correlation_id, cid)


# =============================================================================
# ERROR TYPES TESTS (ERR-001 §3, SOC 2 CC7.2)
# =============================================================================


class ErrorTypesTest(SimpleTestCase):
    """ERR-001 §3: Type definitions and enums."""

    def test_severity_has_five_levels(self):
        """ErrorSeverity has DEBUG, INFO, WARNING, ERROR, CRITICAL."""
        expected = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        actual = {s.value for s in ErrorSeverity}
        self.assertEqual(actual, expected)

    def test_severity_levels_ordered(self):
        """Severity numeric levels increase: DEBUG < INFO < WARNING < ERROR < CRITICAL."""
        self.assertLess(SEVERITY_LEVELS[ErrorSeverity.DEBUG], SEVERITY_LEVELS[ErrorSeverity.INFO])
        self.assertLess(SEVERITY_LEVELS[ErrorSeverity.INFO], SEVERITY_LEVELS[ErrorSeverity.WARNING])
        self.assertLess(SEVERITY_LEVELS[ErrorSeverity.WARNING], SEVERITY_LEVELS[ErrorSeverity.ERROR])
        self.assertLess(
            SEVERITY_LEVELS[ErrorSeverity.ERROR],
            SEVERITY_LEVELS[ErrorSeverity.CRITICAL],
        )

    def test_category_has_ten_values(self):
        """ErrorCategory has 10 values."""
        self.assertEqual(len(ErrorCategory), 10)

    def test_every_category_has_status_code(self):
        """Every ErrorCategory maps to an HTTP status code."""
        for cat in ErrorCategory:
            self.assertIn(
                cat,
                CATEGORY_STATUS_CODES,
                f"{cat.value} missing from CATEGORY_STATUS_CODES",
            )

    def test_retryable_categories_subset(self):
        """RETRYABLE_CATEGORIES is a subset of ErrorCategory."""
        for cat in RETRYABLE_CATEGORIES:
            self.assertIn(cat, ErrorCategory)

    def test_circuit_breaker_state_values(self):
        """CircuitBreakerState has CLOSED, OPEN, HALF_OPEN."""
        expected = {"CLOSED", "OPEN", "HALF_OPEN"}
        actual = {s.value for s in CircuitBreakerState}
        self.assertEqual(actual, expected)

    def test_retry_strategy_values(self):
        """RetryStrategy has 5 strategies."""
        expected = {"NONE", "IMMEDIATE", "LINEAR", "EXPONENTIAL", "CONSTANT"}
        actual = {s.value for s in RetryStrategy}
        self.assertEqual(actual, expected)

    def test_recovery_mode_values(self):
        """RecoveryMode has FAIL_FAST, FALLBACK, QUEUE, SKIP."""
        expected = {"FAIL_FAST", "FALLBACK", "QUEUE", "SKIP"}
        actual = {s.value for s in RecoveryMode}
        self.assertEqual(actual, expected)

    def test_system_layer_values(self):
        """SystemLayer has 7 layers."""
        self.assertEqual(len(SystemLayer), 7)
        self.assertIn("API", {layer.value for layer in SystemLayer})

    def test_error_context_to_dict(self):
        """ErrorContext.to_dict() serializes all fields."""
        ctx = ErrorContext(
            correlation_id=uuid4(),
            operation="test_op",
            layer=SystemLayer.API,
        )
        d = ctx.to_dict()
        self.assertIn("correlation_id", d)
        self.assertIn("operation", d)
        self.assertEqual(d["layer"], "API")


class ErrorRegistryTest(SimpleTestCase):
    """ERR-002 §4: Error registry entries."""

    def test_registry_has_entries(self):
        """ERROR_REGISTRY contains known error codes."""
        self.assertGreaterEqual(len(ERROR_REGISTRY), 10)

    def test_all_entries_have_required_fields(self):
        """Every registry entry has code, category, http_status, message_template."""
        for code, entry in ERROR_REGISTRY.items():
            self.assertIsInstance(entry, ErrorRegistryEntry)
            self.assertEqual(entry.code, code)
            self.assertIsInstance(entry.category, ErrorCategory)
            self.assertIsInstance(entry.http_status, int)
            self.assertTrue(len(entry.message_template) > 0)
            self.assertTrue(len(entry.doc_url) > 0)

    def test_retryable_entries_are_intentional(self):
        """Registry retryable=True entries are from transient-error categories."""
        # Registry retryable is broader than RETRYABLE_CATEGORIES (which only
        # governs exception.retryable). Registry marks database_error as retryable
        # because DB errors can be transient even though the exception category
        # isn't in the strict retryable set.
        transient_categories = RETRYABLE_CATEGORIES | {
            ErrorCategory.SYSTEM,
            ErrorCategory.DATABASE,
        }
        for code, entry in ERROR_REGISTRY.items():
            if entry.retryable:
                self.assertIn(
                    entry.category,
                    transient_categories,
                    f"{code} is retryable but category {entry.category} seems non-transient",
                )

    def test_known_codes_exist(self):
        """Critical codes exist: bad_request, unauthorized, not_found, internal_error."""
        for code in ("bad_request", "unauthorized", "not_found", "internal_error"):
            self.assertIn(code, ERROR_REGISTRY)


# =============================================================================
# ERROR ENVELOPE TESTS (ERR-002 §3, SOC 2 CC7.2)
# =============================================================================


class ErrorEnvelopeTest(SimpleTestCase):
    """ERR-002 §3: Canonical error envelope serialization."""

    def test_to_dict_required_fields(self):
        """Envelope to_dict() has error.code, error.message, error.retryable, error.request_id."""
        envelope = ErrorEnvelope(
            code="not_found",
            message="Resource not found",
            retryable=False,
            request_id="req-abc",
        )
        d = envelope.to_dict()
        self.assertIn("error", d)
        for key in ("code", "message", "retryable", "request_id"):
            self.assertIn(key, d["error"])

    def test_details_included_when_present(self):
        """Validation details are serialized in envelope."""
        details = [ErrorDetail(field="email", code="invalid", message="bad email")]
        envelope = ErrorEnvelope(
            code="validation_error",
            message="Validation failed",
            retryable=False,
            request_id="req-1",
            details=details,
        )
        d = envelope.to_dict()
        self.assertIn("details", d["error"])
        self.assertEqual(len(d["error"]["details"]), 1)
        self.assertEqual(d["error"]["details"][0]["field"], "email")

    def test_doc_included_when_present(self):
        """Doc URL appears in envelope when set."""
        envelope = ErrorEnvelope(
            code="test",
            message="t",
            retryable=False,
            request_id="r",
            doc="https://docs.synara.io/errors#test",
        )
        d = envelope.to_dict()
        self.assertEqual(d["error"]["doc"], "https://docs.synara.io/errors#test")

    def test_correlation_included_when_present(self):
        """Correlation ID appears in envelope when set."""
        envelope = ErrorEnvelope(
            code="test",
            message="t",
            retryable=False,
            request_id="r",
            correlation="corr-123",
        )
        d = envelope.to_dict()
        self.assertEqual(d["error"]["correlation"], "corr-123")

    def test_locale_omitted_when_default(self):
        """Default locale (en-US) is not serialized."""
        envelope = ErrorEnvelope(
            code="test",
            message="t",
            retryable=False,
            request_id="r",
        )
        d = envelope.to_dict()
        self.assertNotIn("locale", d["error"])


# =============================================================================
# EXPONENTIAL BACKOFF TESTS (ERR-001 §7.1, SOC 2 CC9.1)
# =============================================================================


class ExponentialBackoffTest(SimpleTestCase):
    """ERR-001 §7.1: Backoff delay calculation."""

    def test_exponential_doubles_each_attempt(self):
        """Exponential strategy roughly doubles delay per attempt (ignoring jitter)."""
        config = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay_ms=1000,
            jitter_factor=0,
            max_delay_ms=60000,
        )
        backoff = ExponentialBackoff(config)
        self.assertEqual(backoff.get_delay(0), 1000)  # 1000 * 2^0
        self.assertEqual(backoff.get_delay(1), 2000)  # 1000 * 2^1
        self.assertEqual(backoff.get_delay(2), 4000)  # 1000 * 2^2
        self.assertEqual(backoff.get_delay(3), 8000)  # 1000 * 2^3

    def test_linear_increases_linearly(self):
        """Linear strategy: delay = base * (attempt + 1)."""
        config = RetryConfig(
            strategy=RetryStrategy.LINEAR,
            base_delay_ms=500,
            jitter_factor=0,
            max_delay_ms=60000,
        )
        backoff = ExponentialBackoff(config)
        self.assertEqual(backoff.get_delay(0), 500)
        self.assertEqual(backoff.get_delay(1), 1000)
        self.assertEqual(backoff.get_delay(2), 1500)

    def test_constant_same_every_attempt(self):
        """Constant strategy: same delay every attempt."""
        config = RetryConfig(
            strategy=RetryStrategy.CONSTANT,
            base_delay_ms=2000,
            jitter_factor=0,
            max_delay_ms=60000,
        )
        backoff = ExponentialBackoff(config)
        self.assertEqual(backoff.get_delay(0), 2000)
        self.assertEqual(backoff.get_delay(5), 2000)

    def test_none_strategy_returns_zero(self):
        """NONE strategy: zero delay."""
        config = RetryConfig(strategy=RetryStrategy.NONE)
        backoff = ExponentialBackoff(config)
        self.assertEqual(backoff.get_delay(0), 0)

    def test_immediate_strategy_returns_zero(self):
        """IMMEDIATE strategy: zero delay."""
        config = RetryConfig(strategy=RetryStrategy.IMMEDIATE)
        backoff = ExponentialBackoff(config)
        self.assertEqual(backoff.get_delay(0), 0)

    def test_max_delay_cap(self):
        """Delay is capped at max_delay_ms."""
        config = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay_ms=1000,
            max_delay_ms=5000,
            jitter_factor=0,
        )
        backoff = ExponentialBackoff(config)
        # 1000 * 2^10 = 1024000, but capped at 5000
        self.assertEqual(backoff.get_delay(10), 5000)

    def test_jitter_adds_randomness(self):
        """Jitter adds variability to delay."""
        config = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay_ms=1000,
            jitter_factor=0.5,
            max_delay_ms=60000,
        )
        backoff = ExponentialBackoff(config)
        # With jitter_factor=0.5, delay at attempt 0 is:
        # 1000 * (1 + uniform(0, 0.5)) → between 1000 and 1500
        delay = backoff.get_delay(0)
        self.assertGreaterEqual(delay, 1000)
        self.assertLessEqual(delay, 1500)

    def test_should_retry_respects_max_attempts(self):
        """should_retry returns False when attempt >= max_attempts."""
        config = RetryConfig(max_attempts=3)
        backoff = ExponentialBackoff(config)
        err = DependencyError("down")  # retryable
        self.assertTrue(backoff.should_retry(0, err))
        self.assertTrue(backoff.should_retry(2, err))
        self.assertFalse(backoff.should_retry(3, err))

    def test_should_retry_checks_retryable(self):
        """should_retry returns False for non-retryable SynaraError."""
        config = RetryConfig(max_attempts=5)
        backoff = ExponentialBackoff(config)
        retryable = DependencyError("down")
        non_retryable = ValidationError("bad input")
        self.assertTrue(backoff.should_retry(0, retryable))
        self.assertFalse(backoff.should_retry(0, non_retryable))

    def test_should_retry_stdlib_transient(self):
        """should_retry returns True for ConnectionError/OSError."""
        config = RetryConfig(max_attempts=3)
        backoff = ExponentialBackoff(config)
        self.assertTrue(backoff.should_retry(0, ConnectionError("refused")))
        self.assertTrue(backoff.should_retry(0, OSError("io error")))


# =============================================================================
# RETRY DECORATOR TESTS (ERR-001 §7.1, SOC 2 CC9.1)
# =============================================================================


class RetryDecoratorTest(SimpleTestCase):
    """ERR-001 §7.1: retry() decorator behavior."""

    @mock.patch("time.sleep")
    def test_retries_on_retryable_error(self, mock_sleep):
        """Decorator retries on retryable exception."""
        call_count = 0

        @retry(max_attempts=3, base_delay_ms=100)
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise DependencyError("temp failure")
            return "ok"

        result = flaky()
        self.assertEqual(result, "ok")
        self.assertEqual(call_count, 3)
        # sleep called twice (before attempt 2 and 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @mock.patch("time.sleep")
    def test_no_retry_on_non_retryable(self, mock_sleep):
        """Decorator does not retry non-retryable errors."""

        @retry(max_attempts=3, base_delay_ms=100)
        def bad_input():
            raise ValidationError("invalid")

        with self.assertRaises(ValidationError):
            bad_input()
        mock_sleep.assert_not_called()

    @mock.patch("time.sleep")
    def test_exhausted_retries_raises(self, mock_sleep):
        """After max_attempts, the last error is raised."""

        @retry(max_attempts=2, base_delay_ms=10)
        def always_fails():
            raise DependencyError("still down")

        with self.assertRaises(DependencyError):
            always_fails()

    @mock.patch("time.sleep")
    def test_on_retry_callback(self, mock_sleep):
        """on_retry callback is called before each retry."""
        attempts_seen = []

        def record(attempt, error):
            attempts_seen.append(attempt)

        call_count = 0

        @retry(max_attempts=3, base_delay_ms=10, on_retry=record)
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise DependencyError("fail")
            return "done"

        flaky()
        self.assertEqual(attempts_seen, [0, 1])

    @mock.patch("time.sleep")
    def test_retryable_exceptions_override(self, mock_sleep):
        """retryable_exceptions allows retrying on specified exception types."""
        call_count = 0

        @retry(
            max_attempts=3,
            base_delay_ms=10,
            retryable_exceptions={RuntimeError},
        )
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("temp")
            return "ok"

        result = flaky()
        self.assertEqual(result, "ok")
        self.assertEqual(call_count, 2)


# =============================================================================
# CIRCUIT BREAKER TESTS (ERR-001 §7.3, SOC 2 CC9.1)
# =============================================================================


class CircuitBreakerTest(SimpleTestCase):
    """ERR-001 §7.3: Circuit breaker state machine."""

    def _make_cb(self, name=None, failure_threshold=3, recovery_timeout_ms=100):
        """Create a circuit breaker with short thresholds for testing."""
        return CircuitBreaker(
            name=name or f"test-cb-{uuid4().hex[:8]}",
            config=CircuitBreakerConfig(
                failure_threshold=failure_threshold,
                success_threshold=2,
                recovery_timeout_ms=recovery_timeout_ms,
                half_open_max_requests=2,
            ),
        )

    def test_initial_state_closed(self):
        """Circuit breaker starts in CLOSED state."""
        cb = self._make_cb()
        self.assertEqual(cb.state, CircuitBreakerState.CLOSED)

    def test_success_keeps_closed(self):
        """Successful calls keep circuit CLOSED."""
        cb = self._make_cb()
        with cb.call():
            pass
        self.assertEqual(cb.state, CircuitBreakerState.CLOSED)

    def test_failure_threshold_opens_circuit(self):
        """Reaching failure_threshold consecutive failures opens circuit."""
        cb = self._make_cb(failure_threshold=3)
        for _ in range(3):
            try:
                with cb.call():
                    raise DependencyError("down")
            except DependencyError:
                pass
        self.assertEqual(cb._state, CircuitBreakerState.OPEN)

    def test_open_circuit_rejects_calls(self):
        """OPEN circuit raises CircuitBreakerOpenError."""
        cb = self._make_cb(failure_threshold=2)
        # Trip the breaker
        for _ in range(2):
            try:
                with cb.call():
                    raise DependencyError("down")
            except DependencyError:
                pass

        with self.assertRaises(CircuitBreakerOpenError) as ctx:
            with cb.call():
                pass
        self.assertIn(cb.name, str(ctx.exception))

    def test_recovery_timeout_transitions_to_half_open(self):
        """After recovery_timeout, OPEN transitions to HALF_OPEN."""
        cb = self._make_cb(failure_threshold=2, recovery_timeout_ms=50)
        # Trip
        for _ in range(2):
            try:
                with cb.call():
                    raise DependencyError("down")
            except DependencyError:
                pass
        self.assertEqual(cb._state, CircuitBreakerState.OPEN)

        # Wait for recovery timeout
        time.sleep(0.1)
        # Accessing .state triggers timeout check
        self.assertEqual(cb.state, CircuitBreakerState.HALF_OPEN)

    def test_half_open_success_closes_circuit(self):
        """success_threshold successes in HALF_OPEN → CLOSED."""
        cb = self._make_cb(failure_threshold=2, recovery_timeout_ms=10)
        # Trip
        for _ in range(2):
            try:
                with cb.call():
                    raise DependencyError("down")
            except DependencyError:
                pass
        time.sleep(0.05)
        # Now HALF_OPEN — need 2 successes (success_threshold=2)
        with cb.call():
            pass
        with cb.call():
            pass
        self.assertEqual(cb.state, CircuitBreakerState.CLOSED)

    def test_half_open_failure_reopens_circuit(self):
        """Any failure in HALF_OPEN immediately re-opens circuit."""
        cb = self._make_cb(failure_threshold=2, recovery_timeout_ms=10)
        # Trip
        for _ in range(2):
            try:
                with cb.call():
                    raise DependencyError("down")
            except DependencyError:
                pass
        time.sleep(0.05)
        self.assertEqual(cb.state, CircuitBreakerState.HALF_OPEN)
        # Fail in half-open
        try:
            with cb.call():
                raise DependencyError("still down")
        except DependencyError:
            pass
        self.assertEqual(cb._state, CircuitBreakerState.OPEN)

    def test_force_open(self):
        """force_open() immediately opens circuit."""
        cb = self._make_cb()
        cb.force_open()
        self.assertEqual(cb._state, CircuitBreakerState.OPEN)

    def test_reset(self):
        """reset() returns circuit to CLOSED."""
        cb = self._make_cb(failure_threshold=2)
        cb.force_open()
        cb.reset()
        self.assertEqual(cb.state, CircuitBreakerState.CLOSED)

    def test_metrics_tracked(self):
        """Metrics count total, successful, failed, rejected calls."""
        cb = self._make_cb(failure_threshold=3)
        # 2 successes
        with cb.call():
            pass
        with cb.call():
            pass
        # 3 failures → opens
        for _ in range(3):
            try:
                with cb.call():
                    raise DependencyError("down")
            except DependencyError:
                pass
        # 1 rejected
        try:
            with cb.call():
                pass
        except CircuitBreakerOpenError:
            pass

        metrics = cb.metrics
        self.assertEqual(metrics.successful_calls, 2)
        self.assertEqual(metrics.failed_calls, 3)
        self.assertEqual(metrics.rejected_calls, 1)
        self.assertEqual(metrics.total_calls, 6)  # 2+3+1

    def test_non_monitored_errors_not_counted(self):
        """Errors from non-monitored categories don't trip the breaker."""
        cb = self._make_cb(failure_threshold=2)
        # ValidationError is VALIDATION category — not in monitored_categories
        for _ in range(5):
            try:
                with cb.call():
                    raise ValidationError("bad input")
            except ValidationError:
                pass
        # Circuit should still be CLOSED — validation errors not monitored
        self.assertEqual(cb._state, CircuitBreakerState.CLOSED)


class CircuitBreakerOpenErrorTest(SimpleTestCase):
    """CircuitBreakerOpenError is a proper SynaraError."""

    def test_is_synara_error(self):
        """CircuitBreakerOpenError inherits SynaraError."""
        err = CircuitBreakerOpenError(
            "open",
            circuit_breaker_name="test",
            state=CircuitBreakerState.OPEN,
        )
        self.assertIsInstance(err, SynaraError)
        self.assertEqual(err.code, "circuit_breaker_open")
        self.assertEqual(err.category, ErrorCategory.DEPENDENCY)


class CircuitBreakerRegistryTest(SimpleTestCase):
    """ERR-001 §7.3: CircuitBreakerRegistry singleton."""

    def test_get_or_create(self):
        """get_or_create returns same instance for same name."""
        reg = CircuitBreakerRegistry()
        name = f"test-reg-{uuid4().hex[:8]}"
        cb1 = reg.get_or_create(name)
        cb2 = reg.get_or_create(name)
        self.assertIs(cb1, cb2)

    def test_get_nonexistent_returns_none(self):
        """get() returns None for unknown name."""
        reg = CircuitBreakerRegistry()
        self.assertIsNone(reg.get(f"no-such-{uuid4().hex[:8]}"))

    def test_all_returns_dict(self):
        """all() returns dict of all registered breakers."""
        reg = CircuitBreakerRegistry()
        name = f"test-all-{uuid4().hex[:8]}"
        reg.get_or_create(name)
        all_breakers = reg.all()
        self.assertIn(name, all_breakers)

    def test_reset_all(self):
        """reset_all() closes all registered breakers."""
        reg = CircuitBreakerRegistry()
        name = f"test-reset-{uuid4().hex[:8]}"
        cb = reg.get_or_create(name)
        cb.force_open()
        self.assertEqual(cb._state, CircuitBreakerState.OPEN)
        reg.reset_all()
        self.assertEqual(cb.state, CircuitBreakerState.CLOSED)


# =============================================================================
# CIRCUIT BREAKER DECORATOR TESTS (ERR-001 §7.3)
# =============================================================================


class WithCircuitBreakerDecoratorTest(SimpleTestCase):
    """ERR-001 §7.3: with_circuit_breaker decorator."""

    def test_fallback_on_open(self):
        """FALLBACK recovery mode invokes fallback when circuit is open."""
        cb_name = f"test-fallback-{uuid4().hex[:8]}"

        @with_circuit_breaker(
            cb_name,
            config=CircuitBreakerConfig(failure_threshold=1),
            fallback=lambda: "cached",
            recovery_mode=RecoveryMode.FALLBACK,
        )
        def call_service():
            raise DependencyError("down")

        # First call trips the breaker
        try:
            call_service()
        except DependencyError:
            pass

        # Second call gets fallback
        result = call_service()
        self.assertEqual(result, "cached")

    def test_skip_on_open(self):
        """SKIP recovery mode returns None when circuit is open."""
        cb_name = f"test-skip-{uuid4().hex[:8]}"

        @with_circuit_breaker(
            cb_name,
            config=CircuitBreakerConfig(failure_threshold=1),
            recovery_mode=RecoveryMode.SKIP,
        )
        def call_service():
            raise DependencyError("down")

        # Trip
        try:
            call_service()
        except DependencyError:
            pass

        result = call_service()
        self.assertIsNone(result)

    def test_fail_fast_raises(self):
        """FAIL_FAST (default) raises CircuitBreakerOpenError."""
        cb_name = f"test-failfast-{uuid4().hex[:8]}"

        @with_circuit_breaker(
            cb_name,
            config=CircuitBreakerConfig(failure_threshold=1),
        )
        def call_service():
            raise DependencyError("down")

        try:
            call_service()
        except DependencyError:
            pass

        with self.assertRaises(CircuitBreakerOpenError):
            call_service()


# =============================================================================
# BULKHEAD TESTS (ERR-001 §7.2, SOC 2 CC9.1)
# =============================================================================


class BulkheadTest(SimpleTestCase):
    """ERR-001 §7.2: Bulkhead concurrency limiting."""

    def test_acquire_and_release(self):
        """Bulkhead slot is acquired and released."""
        bh = Bulkhead("test-bh", max_concurrent=2, max_wait_ms=100)
        self.assertEqual(bh.available_slots, 2)
        with bh.acquire():
            self.assertEqual(bh.active_calls, 1)
            self.assertEqual(bh.available_slots, 1)
        self.assertEqual(bh.active_calls, 0)
        self.assertEqual(bh.available_slots, 2)

    def test_full_raises_error(self):
        """BulkheadFullError raised when all slots taken."""
        bh = Bulkhead("test-full", max_concurrent=1, max_wait_ms=50)
        with bh.acquire():
            with self.assertRaises(BulkheadFullError):
                with bh.acquire():
                    pass

    def test_bulkhead_full_error_is_synara_error(self):
        """BulkheadFullError inherits SynaraError."""
        err = BulkheadFullError(
            "full",
            bulkhead_name="test",
            max_concurrent=5,
        )
        self.assertIsInstance(err, SynaraError)
        self.assertEqual(err.code, "bulkhead_full")
        self.assertEqual(err.category, ErrorCategory.RATE_LIMIT)

    def test_concurrent_access(self):
        """Multiple threads can use bulkhead concurrently up to limit."""
        bh = Bulkhead("test-concurrent", max_concurrent=3, max_wait_ms=500)
        results = []
        barrier = threading.Barrier(3)

        def worker():
            with bh.acquire():
                barrier.wait(timeout=2)
                results.append(threading.current_thread().name)

        threads = [threading.Thread(target=worker, name=f"w-{i}") for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertEqual(len(results), 3)

    def test_release_on_exception(self):
        """Bulkhead slot is released even if body raises."""
        bh = Bulkhead("test-exc", max_concurrent=1, max_wait_ms=100)
        try:
            with bh.acquire():
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        # Slot should be released
        self.assertEqual(bh.active_calls, 0)
        self.assertEqual(bh.available_slots, 1)


# =============================================================================
# DEFAULT CONFIG TESTS
# =============================================================================


class DefaultConfigTest(SimpleTestCase):
    """Verify default configs have sane values."""

    def test_default_retry_config(self):
        """DEFAULT_RETRY_CONFIG has expected defaults."""
        from syn.err.types import DEFAULT_RETRY_CONFIG

        self.assertEqual(DEFAULT_RETRY_CONFIG.strategy, RetryStrategy.EXPONENTIAL)
        self.assertEqual(DEFAULT_RETRY_CONFIG.max_attempts, 3)
        self.assertGreater(DEFAULT_RETRY_CONFIG.base_delay_ms, 0)
        self.assertGreater(DEFAULT_RETRY_CONFIG.max_delay_ms, DEFAULT_RETRY_CONFIG.base_delay_ms)

    def test_default_circuit_breaker_config(self):
        """DEFAULT_CIRCUIT_BREAKER_CONFIG has expected defaults."""
        from syn.err.types import DEFAULT_CIRCUIT_BREAKER_CONFIG

        self.assertGreater(DEFAULT_CIRCUIT_BREAKER_CONFIG.failure_threshold, 0)
        self.assertGreater(DEFAULT_CIRCUIT_BREAKER_CONFIG.success_threshold, 0)
        self.assertGreater(DEFAULT_CIRCUIT_BREAKER_CONFIG.recovery_timeout_ms, 0)
