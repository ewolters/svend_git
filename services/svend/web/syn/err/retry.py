"""
Synara Retry Utilities (ERR-001 §7)
===================================

Retry strategies and circuit breaker implementation for resilient
error recovery.

Standard:     ERR-001 §7
Compliance:   ISO 27001 A.17.1, SOC 2 CC9.1
Version:      1.0.0

Components
----------
- ExponentialBackoff: Exponential backoff with jitter
- CircuitBreaker: Circuit breaker pattern implementation
- retry decorator: Decorator for automatic retry with backoff
- with_circuit_breaker decorator: Decorator for circuit breaker protection
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import (
    TypeVar,
)

from .exceptions import SynaraError, TimeoutError
from .types import (
    DEFAULT_CIRCUIT_BREAKER_CONFIG,
    DEFAULT_RETRY_CONFIG,
    CircuitBreakerConfig,
    CircuitBreakerState,
    ErrorCategory,
    RecoveryMode,
    RetryConfig,
    RetryStrategy,
)

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("syn.audit")

T = TypeVar("T")


# =============================================================================
# EXPONENTIAL BACKOFF (ERR-001 §7.1)
# =============================================================================


class ExponentialBackoff:
    """
    Exponential backoff calculator with jitter per ERR-001 §7.1.

    Implements exponential backoff with optional jitter to prevent
    thundering herd effects during recovery.

    Formula: delay = min(base_delay * (2 ** attempt) * (1 + jitter), max_delay)

    Attributes:
        config: RetryConfig with backoff parameters

    Example:
        >>> backoff = ExponentialBackoff(RetryConfig(base_delay_ms=1000))
        >>> for attempt in range(3):
        ...     delay = backoff.get_delay(attempt)
        ...     print(f"Attempt {attempt}: wait {delay}ms")
    """

    def __init__(self, config: RetryConfig | None = None):
        self.config = config or DEFAULT_RETRY_CONFIG

    def get_delay(self, attempt: int) -> int:
        """
        Calculate delay for given attempt number.

        Args:
            attempt: Zero-based attempt number

        Returns:
            Delay in milliseconds
        """
        if self.config.strategy == RetryStrategy.NONE:
            return 0
        elif self.config.strategy == RetryStrategy.IMMEDIATE:
            return 0
        elif self.config.strategy == RetryStrategy.CONSTANT:
            delay = self.config.base_delay_ms
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay_ms * (attempt + 1)
        else:  # EXPONENTIAL (default)
            delay = self.config.base_delay_ms * (2**attempt)

        # Apply jitter
        if self.config.jitter_factor > 0:
            jitter = random.uniform(0, self.config.jitter_factor)
            delay = int(delay * (1 + jitter))

        # Cap at max delay
        return min(delay, self.config.max_delay_ms)

    def should_retry(self, attempt: int, error: Exception) -> bool:
        """
        Determine if retry should be attempted.

        Args:
            attempt: Current attempt number (0-based)
            error: The exception that occurred

        Returns:
            True if retry should be attempted
        """
        # Check max attempts
        if attempt >= self.config.max_attempts:
            return False

        # Check if error is retryable
        if isinstance(error, SynaraError):
            return error.retryable

        # Check category for SynaraError subclasses
        if hasattr(error, "category"):
            return error.category in self.config.retryable_categories

        # Default: retry on common transient errors
        return isinstance(error, (ConnectionError, TimeoutError, OSError))


# =============================================================================
# CIRCUIT BREAKER (ERR-001 §7.3)
# =============================================================================


@dataclass
class CircuitBreakerMetrics:
    """Metrics tracked by circuit breaker."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: datetime | None = None
    last_success_time: datetime | None = None
    last_state_change: datetime | None = None


class CircuitBreaker:
    """
    Circuit breaker implementation per ERR-001 §7.3.

    Protects against cascading failures by failing fast when a
    dependency is unavailable.

    State Machine:
        CLOSED → OPEN (on failure_threshold consecutive failures)
        OPEN → HALF_OPEN (after recovery_timeout)
        HALF_OPEN → CLOSED (on success_threshold successes)
        HALF_OPEN → OPEN (on any failure)

    Thread-safe implementation using locks.

    Attributes:
        name: Identifier for this circuit breaker
        config: CircuitBreakerConfig with thresholds
        state: Current circuit state

    Example:
        >>> cb = CircuitBreaker("document-service")
        >>> try:
        ...     with cb.call():
        ...         response = call_document_service()
        ... except CircuitBreakerOpenError:
        ...     # Use fallback
        ...     response = get_cached_document()
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ):
        self.name = name
        self.config = config or DEFAULT_CIRCUIT_BREAKER_CONFIG
        self._state = CircuitBreakerState.CLOSED
        self._metrics = CircuitBreakerMetrics()
        self._lock = threading.RLock()
        self._half_open_calls = 0

        logger.info(
            f"CircuitBreaker initialized: {name}",
            extra={
                "circuit_breaker": name,
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout_ms": self.config.recovery_timeout_ms,
            },
        )

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit state, checking for recovery timeout."""
        with self._lock:
            if self._state == CircuitBreakerState.OPEN:
                # Check if recovery timeout has elapsed
                if self._metrics.last_failure_time:
                    elapsed = datetime.utcnow() - self._metrics.last_failure_time
                    timeout = timedelta(milliseconds=self.config.recovery_timeout_ms)
                    if elapsed >= timeout:
                        self._transition_to(CircuitBreakerState.HALF_OPEN)
            return self._state

    @property
    def metrics(self) -> CircuitBreakerMetrics:
        """Get current metrics snapshot."""
        with self._lock:
            return CircuitBreakerMetrics(
                total_calls=self._metrics.total_calls,
                successful_calls=self._metrics.successful_calls,
                failed_calls=self._metrics.failed_calls,
                rejected_calls=self._metrics.rejected_calls,
                consecutive_failures=self._metrics.consecutive_failures,
                consecutive_successes=self._metrics.consecutive_successes,
                last_failure_time=self._metrics.last_failure_time,
                last_success_time=self._metrics.last_success_time,
                last_state_change=self._metrics.last_state_change,
            )

    def _transition_to(self, new_state: CircuitBreakerState) -> None:
        """Transition to new state with logging."""
        old_state = self._state
        self._state = new_state
        self._metrics.last_state_change = datetime.utcnow()

        if new_state == CircuitBreakerState.HALF_OPEN:
            self._half_open_calls = 0

        logger.info(
            f"CircuitBreaker state change: {self.name} {old_state.value} -> {new_state.value}",
            extra={
                "circuit_breaker": self.name,
                "old_state": old_state.value,
                "new_state": new_state.value,
                "consecutive_failures": self._metrics.consecutive_failures,
            },
        )

        # Emit audit event for state changes
        audit_logger.info(
            f"Circuit breaker {self.name} changed to {new_state.value}",
            extra={
                "event_type": f"circuit_breaker.{new_state.value.lower()}",
                "circuit_breaker": self.name,
                "old_state": old_state.value,
                "metrics": {
                    "total_calls": self._metrics.total_calls,
                    "failed_calls": self._metrics.failed_calls,
                    "consecutive_failures": self._metrics.consecutive_failures,
                },
            },
        )

    def _record_success(self) -> None:
        """Record successful call."""
        with self._lock:
            self._metrics.successful_calls += 1
            self._metrics.consecutive_successes += 1
            self._metrics.consecutive_failures = 0
            self._metrics.last_success_time = datetime.utcnow()

            if self._state == CircuitBreakerState.HALF_OPEN:
                if self._metrics.consecutive_successes >= self.config.success_threshold:
                    self._transition_to(CircuitBreakerState.CLOSED)

    def _record_failure(self, error: Exception) -> None:
        """Record failed call."""
        with self._lock:
            self._metrics.failed_calls += 1
            self._metrics.consecutive_failures += 1
            self._metrics.consecutive_successes = 0
            self._metrics.last_failure_time = datetime.utcnow()

            # Check if we should open the circuit
            if self._state == CircuitBreakerState.CLOSED:
                if self._metrics.consecutive_failures >= self.config.failure_threshold:
                    self._transition_to(CircuitBreakerState.OPEN)
            elif self._state == CircuitBreakerState.HALF_OPEN:
                # Any failure in half-open immediately opens circuit
                self._transition_to(CircuitBreakerState.OPEN)

    def _allow_request(self) -> bool:
        """Check if request should be allowed."""
        with self._lock:
            current_state = self.state  # This may trigger OPEN -> HALF_OPEN

            if current_state == CircuitBreakerState.CLOSED:
                return True
            elif current_state == CircuitBreakerState.OPEN:
                self._metrics.rejected_calls += 1
                return False
            else:  # HALF_OPEN
                # Allow limited requests in half-open
                if self._half_open_calls < self.config.half_open_max_requests:
                    self._half_open_calls += 1
                    return True
                self._metrics.rejected_calls += 1
                return False

    @contextmanager
    def call(self):
        """
        Context manager for circuit-breaker-protected calls.

        Usage:
            with circuit_breaker.call():
                result = external_service.call()

        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        self._metrics.total_calls += 1

        if not self._allow_request():
            raise CircuitBreakerOpenError(
                f"Circuit breaker {self.name} is open",
                circuit_breaker_name=self.name,
                state=self.state,
                metrics=self.metrics,
            )

        try:
            yield
            self._record_success()
        except Exception as e:
            # Only record failure for monitored error categories
            if self._should_record_failure(e):
                self._record_failure(e)
            raise

    def _should_record_failure(self, error: Exception) -> bool:
        """Check if error should be recorded as failure."""
        if isinstance(error, SynaraError):
            return error.category in self.config.monitored_categories
        # Record connection and timeout errors
        return isinstance(error, (ConnectionError, TimeoutError, OSError))

    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        with self._lock:
            self._transition_to(CircuitBreakerState.CLOSED)
            self._metrics.consecutive_failures = 0
            self._metrics.consecutive_successes = 0

    def force_open(self) -> None:
        """Manually force circuit breaker to open state."""
        with self._lock:
            self._transition_to(CircuitBreakerState.OPEN)


class CircuitBreakerOpenError(SynaraError):
    """Exception raised when circuit breaker is open."""

    default_category = ErrorCategory.DEPENDENCY
    default_code = "circuit_breaker_open"

    def __init__(
        self,
        message: str,
        *,
        circuit_breaker_name: str,
        state: CircuitBreakerState,
        metrics: CircuitBreakerMetrics | None = None,
        **kwargs,
    ):
        self.circuit_breaker_name = circuit_breaker_name
        self.circuit_state = state
        self.circuit_metrics = metrics

        extra = kwargs.pop("extra", {})
        extra.update(
            {
                "circuit_breaker_name": circuit_breaker_name,
                "circuit_state": state.value,
            }
        )
        if metrics:
            extra["circuit_metrics"] = {
                "total_calls": metrics.total_calls,
                "failed_calls": metrics.failed_calls,
                "rejected_calls": metrics.rejected_calls,
            }

        super().__init__(message, extra=extra, **kwargs)


# =============================================================================
# CIRCUIT BREAKER REGISTRY
# =============================================================================


class CircuitBreakerRegistry:
    """
    Registry for managing circuit breakers.

    Provides centralized access to circuit breakers by name.
    Thread-safe singleton pattern.
    """

    _instance: CircuitBreakerRegistry | None = None
    _lock = threading.Lock()

    def __new__(cls) -> CircuitBreakerRegistry:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._breakers: dict[str, CircuitBreaker] = {}
                cls._instance._breaker_lock = threading.RLock()
            return cls._instance

    def get_or_create(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """Get existing circuit breaker or create new one."""
        with self._breaker_lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]

    def get(self, name: str) -> CircuitBreaker | None:
        """Get circuit breaker by name."""
        with self._breaker_lock:
            return self._breakers.get(name)

    def all(self) -> dict[str, CircuitBreaker]:
        """Get all registered circuit breakers."""
        with self._breaker_lock:
            return dict(self._breakers)

    def reset_all(self) -> None:
        """Reset all circuit breakers to closed state."""
        with self._breaker_lock:
            for breaker in self._breakers.values():
                breaker.reset()


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()


# =============================================================================
# RETRY DECORATOR (ERR-001 §7.1)
# =============================================================================


def retry(
    config: RetryConfig | None = None,
    *,
    max_attempts: int | None = None,
    base_delay_ms: int | None = None,
    retryable_exceptions: set[type[Exception]] | None = None,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> Callable:
    """
    Decorator for automatic retry with exponential backoff.

    Args:
        config: RetryConfig (overrides individual params)
        max_attempts: Maximum retry attempts
        base_delay_ms: Base delay between retries
        retryable_exceptions: Exception types to retry
        on_retry: Callback called before each retry

    Example:
        @retry(max_attempts=3, base_delay_ms=1000)
        def call_external_service():
            return requests.get(url)

        @retry(config=RetryConfig(strategy=RetryStrategy.LINEAR))
        async def async_call():
            return await client.fetch()
    """
    if config is None:
        config = RetryConfig(
            max_attempts=max_attempts or DEFAULT_RETRY_CONFIG.max_attempts,
            base_delay_ms=base_delay_ms or DEFAULT_RETRY_CONFIG.base_delay_ms,
        )

    backoff = ExponentialBackoff(config)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            last_error: Exception | None = None

            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e

                    # Check if we should retry
                    should_retry = backoff.should_retry(attempt, e)
                    if retryable_exceptions:
                        should_retry = should_retry or isinstance(
                            e, tuple(retryable_exceptions)
                        )

                    if not should_retry or attempt >= config.max_attempts - 1:
                        raise

                    # Calculate delay and wait
                    delay_ms = backoff.get_delay(attempt)

                    logger.warning(
                        f"Retry attempt {attempt + 1}/{config.max_attempts} after {delay_ms}ms: {e}",
                        extra={
                            "attempt": attempt + 1,
                            "max_attempts": config.max_attempts,
                            "delay_ms": delay_ms,
                            "error": str(e),
                        },
                    )

                    if on_retry:
                        on_retry(attempt, e)

                    time.sleep(delay_ms / 1000)

            # Should not reach here, but just in case
            if last_error:
                raise last_error
            raise RuntimeError("Unexpected retry loop exit")

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            last_error: Exception | None = None

            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e

                    should_retry = backoff.should_retry(attempt, e)
                    if retryable_exceptions:
                        should_retry = should_retry or isinstance(
                            e, tuple(retryable_exceptions)
                        )

                    if not should_retry or attempt >= config.max_attempts - 1:
                        raise

                    delay_ms = backoff.get_delay(attempt)

                    logger.warning(
                        f"Retry attempt {attempt + 1}/{config.max_attempts} after {delay_ms}ms: {e}",
                        extra={
                            "attempt": attempt + 1,
                            "max_attempts": config.max_attempts,
                            "delay_ms": delay_ms,
                            "error": str(e),
                        },
                    )

                    if on_retry:
                        on_retry(attempt, e)

                    await asyncio.sleep(delay_ms / 1000)

            if last_error:
                raise last_error
            raise RuntimeError("Unexpected retry loop exit")

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# =============================================================================
# CIRCUIT BREAKER DECORATOR (ERR-001 §7.3)
# =============================================================================


def with_circuit_breaker(
    name: str,
    config: CircuitBreakerConfig | None = None,
    *,
    fallback: Callable[..., T] | None = None,
    recovery_mode: RecoveryMode = RecoveryMode.FAIL_FAST,
) -> Callable:
    """
    Decorator for circuit-breaker-protected calls.

    Args:
        name: Circuit breaker name (shared across calls)
        config: CircuitBreakerConfig
        fallback: Fallback function when circuit is open
        recovery_mode: How to handle circuit open state

    Example:
        @with_circuit_breaker("document-service")
        def get_document(doc_id: str):
            return document_service.get(doc_id)

        @with_circuit_breaker(
            "cache-service",
            fallback=lambda key: None,
            recovery_mode=RecoveryMode.FALLBACK
        )
        def get_cached_value(key: str):
            return cache.get(key)
    """
    breaker = circuit_breaker_registry.get_or_create(name, config)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            try:
                with breaker.call():
                    return func(*args, **kwargs)
            except CircuitBreakerOpenError:
                if recovery_mode == RecoveryMode.FALLBACK and fallback:
                    logger.info(
                        f"Using fallback for {name}",
                        extra={"circuit_breaker": name},
                    )
                    return fallback(*args, **kwargs)
                elif recovery_mode == RecoveryMode.SKIP:
                    logger.info(
                        f"Skipping call to {name}",
                        extra={"circuit_breaker": name},
                    )
                    return None  # type: ignore
                raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            try:
                with breaker.call():
                    return await func(*args, **kwargs)
            except CircuitBreakerOpenError:
                if recovery_mode == RecoveryMode.FALLBACK and fallback:
                    logger.info(
                        f"Using fallback for {name}",
                        extra={"circuit_breaker": name},
                    )
                    if asyncio.iscoroutinefunction(fallback):
                        return await fallback(*args, **kwargs)
                    return fallback(*args, **kwargs)
                elif recovery_mode == RecoveryMode.SKIP:
                    logger.info(
                        f"Skipping call to {name}",
                        extra={"circuit_breaker": name},
                    )
                    return None  # type: ignore
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# =============================================================================
# BULKHEAD PATTERN (ERR-001 §7.2)
# =============================================================================


class Bulkhead:
    """
    Bulkhead pattern implementation per ERR-001 §7.2.

    Limits concurrent calls to prevent resource exhaustion.
    Thread-safe using semaphores.

    Attributes:
        name: Bulkhead identifier
        max_concurrent: Maximum concurrent calls allowed
        max_wait_ms: Maximum time to wait for slot

    Example:
        bulkhead = Bulkhead("external-api", max_concurrent=10)
        with bulkhead.acquire():
            response = call_external_api()
    """

    def __init__(
        self,
        name: str,
        max_concurrent: int = 10,
        max_wait_ms: int = 5000,
    ):
        self.name = name
        self.max_concurrent = max_concurrent
        self.max_wait_ms = max_wait_ms
        self._semaphore = threading.Semaphore(max_concurrent)
        self._async_semaphore: asyncio.Semaphore | None = None
        self._active_calls = 0
        self._rejected_calls = 0
        self._lock = threading.Lock()

    @contextmanager
    def acquire(self):
        """Acquire a bulkhead slot (blocking)."""
        acquired = self._semaphore.acquire(timeout=self.max_wait_ms / 1000)
        if not acquired:
            with self._lock:
                self._rejected_calls += 1
            raise BulkheadFullError(
                f"Bulkhead {self.name} is full",
                bulkhead_name=self.name,
                max_concurrent=self.max_concurrent,
            )

        with self._lock:
            self._active_calls += 1

        try:
            yield
        finally:
            with self._lock:
                self._active_calls -= 1
            self._semaphore.release()

    @property
    def active_calls(self) -> int:
        """Get number of active calls."""
        with self._lock:
            return self._active_calls

    @property
    def available_slots(self) -> int:
        """Get number of available slots."""
        with self._lock:
            return self.max_concurrent - self._active_calls


class BulkheadFullError(SynaraError):
    """Exception raised when bulkhead is full."""

    default_category = ErrorCategory.RATE_LIMIT
    default_code = "bulkhead_full"

    def __init__(
        self,
        message: str,
        *,
        bulkhead_name: str,
        max_concurrent: int,
        **kwargs,
    ):
        self.bulkhead_name = bulkhead_name
        self.bulkhead_max_concurrent = max_concurrent

        extra = kwargs.pop("extra", {})
        extra.update(
            {
                "bulkhead_name": bulkhead_name,
                "max_concurrent": max_concurrent,
            }
        )

        super().__init__(message, extra=extra, **kwargs)
