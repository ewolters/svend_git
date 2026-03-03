"""
Python logging handler for LOG-001/002 compliant structured logging.

Standard: LOG-001 §5, LOG-002 §4.2
Compliance: NIST SP 800-53 AU-2, AU-3, AU-9 / ISO 27001 A.12.4.1

This module provides:
- SynaraLogHandler: Writes logs to LogEntry model
- CorrelationFilter: Adds correlation_id to log records
- TenantFilter: Adds tenant_id to log records
- get_synara_logger: Factory for configured loggers

Usage:
    from syn.log.handlers import SynaraLogHandler, get_synara_logger

    # Configure in Django settings
    LOGGING = {
        'handlers': {
            'synara': {
                'class': 'syn.log.handlers.SynaraLogHandler',
                'stream_name': 'application',
            }
        }
    }

    # Or use directly
    logger = get_synara_logger('syn.mymodule')
    logger.info('Operation completed', extra={'user_id': 'user-123'})
"""

import logging
import socket
import threading
import traceback
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Any

# Context variables for correlation tracking (thread-safe)
correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)
tenant_id_var: ContextVar[str | None] = ContextVar("tenant_id", default=None)
actor_id_var: ContextVar[str | None] = ContextVar("actor_id", default=None)


# =============================================================================
# Context Management (LOG-001 §5.3)
# =============================================================================


def set_correlation_id(correlation_id: str | uuid.UUID | None) -> None:
    """
    Set the correlation ID for the current context.

    Standard: LOG-001 §5.3, CTG-001 §5
    """
    correlation_id_var.set(str(correlation_id) if correlation_id else None)


def get_correlation_id() -> str | None:
    """Get the correlation ID from the current context."""
    return correlation_id_var.get()


def set_tenant_id(tenant_id: str | uuid.UUID | None) -> None:
    """
    Set the tenant ID for the current context.

    Standard: LOG-001 §5.3, SEC-001 §5.2
    """
    tenant_id_var.set(str(tenant_id) if tenant_id else None)


def get_tenant_id() -> str | None:
    """Get the tenant ID from the current context."""
    return tenant_id_var.get()


def set_actor_id(actor_id: str | None) -> None:
    """Set the actor ID for the current context."""
    actor_id_var.set(actor_id)


def get_actor_id() -> str | None:
    """Get the actor ID from the current context."""
    return actor_id_var.get()


class LogContext:
    """
    Context manager for setting log context variables.

    Usage:
        with LogContext(correlation_id='abc', tenant_id='tenant-1'):
            logger.info('This log will have correlation and tenant info')
    """

    def __init__(
        self,
        correlation_id: str | uuid.UUID | None = None,
        tenant_id: str | uuid.UUID | None = None,
        actor_id: str | None = None,
    ):
        self.correlation_id = correlation_id
        self.tenant_id = tenant_id
        self.actor_id = actor_id
        self._tokens: list = []

    def __enter__(self):
        if self.correlation_id:
            token = correlation_id_var.set(str(self.correlation_id))
            self._tokens.append((correlation_id_var, token))
        if self.tenant_id:
            token = tenant_id_var.set(str(self.tenant_id))
            self._tokens.append((tenant_id_var, token))
        if self.actor_id:
            token = actor_id_var.set(self.actor_id)
            self._tokens.append((actor_id_var, token))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for var, token in reversed(self._tokens):
            var.reset(token)
        return False


# =============================================================================
# Log Filters (LOG-001 §5.4)
# =============================================================================


class CorrelationFilter(logging.Filter):
    """
    Adds correlation_id to log records from context.

    Standard: LOG-001 §5.4, CTG-001 §5
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Add correlation_id from context if not already set
        if not hasattr(record, "correlation_id") or record.correlation_id is None:
            record.correlation_id = get_correlation_id() or str(uuid.uuid4())
        return True


class TenantFilter(logging.Filter):
    """
    Adds tenant_id to log records from context.

    Standard: LOG-001 §5.4, SEC-001 §5.2
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Add tenant_id from context if not already set
        if not hasattr(record, "tenant_id") or record.tenant_id is None:
            record.tenant_id = get_tenant_id()
        return True


class ActorFilter(logging.Filter):
    """Adds actor_id to log records from context."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "actor_id") or record.actor_id is None:
            record.actor_id = get_actor_id()
        return True


# =============================================================================
# Synara Log Handler (LOG-002 §4.2)
# =============================================================================


class SynaraLogHandler(logging.Handler):
    """
    Python logging handler that persists logs to LogEntry model.

    Standard: LOG-002 §4.2
    Compliance: NIST SP 800-53 AU-2, AU-3, AU-9 / ISO 27001 A.12.4.1

    Features:
    - Writes to LogEntry model with full LOG-001 fields
    - Automatic stream lookup/creation
    - Exception capture with traceback
    - Context enrichment (hostname, process, thread)
    - Batched writes for performance (optional)
    - Graceful degradation on database errors

    Args:
        stream_name: Log stream name (default: 'application')
        level: Minimum log level (default: INFO)
        batch_size: Number of logs to batch before write (0 = immediate)
        include_context: Include extra context in log entries

    Django LOGGING configuration:
        LOGGING = {
            'version': 1,
            'handlers': {
                'synara': {
                    'class': 'syn.log.handlers.SynaraLogHandler',
                    'level': 'INFO',
                    'stream_name': 'application',
                }
            },
            'loggers': {
                'syn': {
                    'handlers': ['synara'],
                    'level': 'INFO',
                }
            }
        }
    """

    # Level mapping from Python to LOG-001
    LEVEL_MAP = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: "WARNING",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "CRITICAL",
    }

    def __init__(
        self,
        stream_name: str = "application",
        level: int = logging.INFO,
        batch_size: int = 0,
        include_context: bool = True,
    ):
        super().__init__(level)
        self.stream_name = stream_name
        self.batch_size = batch_size
        self.include_context = include_context

        # Batch buffer
        self._buffer: list = []
        self._buffer_lock = threading.Lock()

        # Stream cache
        self._stream = None
        self._stream_lock = threading.Lock()

        # Add filters
        self.addFilter(CorrelationFilter())
        self.addFilter(TenantFilter())
        self.addFilter(ActorFilter())

    def _get_stream(self):
        """Get or create the log stream (cached)."""
        if self._stream is not None:
            return self._stream

        with self._stream_lock:
            if self._stream is not None:
                return self._stream

            try:
                from syn.log.models import LogStream

                self._stream, _ = LogStream.objects.get_or_create(
                    name=self.stream_name,
                    defaults={
                        "retention_days": 90,
                        "min_level": "INFO",
                        "is_active": True,
                    },
                )
                return self._stream
            except Exception as e:
                # Log to stderr if database unavailable
                import sys

                print(f"[SynaraLogHandler] Failed to get stream: {e}", file=sys.stderr)
                return None

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record to the database.

        Standard: LOG-002 §4.2
        """
        try:
            # Build log entry data
            entry_data = self._build_entry_data(record)

            if self.batch_size > 0:
                # Batched mode
                with self._buffer_lock:
                    self._buffer.append(entry_data)
                    if len(self._buffer) >= self.batch_size:
                        self._flush_buffer()
            else:
                # Immediate mode
                self._write_entry(entry_data)

        except Exception as e:
            # Don't raise exceptions from logging handler
            # Log to stderr for visibility (can't use logger here)
            import sys
            print(f"[LOG-002] Handler emit error: {type(e).__name__}: {e}", file=sys.stderr)
            self.handleError(record)

    def _build_entry_data(self, record: logging.LogRecord) -> dict[str, Any]:
        """Build entry data from log record."""
        # Get level
        level = self.LEVEL_MAP.get(record.levelno, "INFO")

        # Get correlation_id (added by filter)
        correlation_id = getattr(record, "correlation_id", None)
        if correlation_id:
            try:
                correlation_id = uuid.UUID(correlation_id)
            except (ValueError, TypeError):
                correlation_id = uuid.uuid4()
        else:
            correlation_id = uuid.uuid4()

        # Get tenant_id (added by filter)
        tenant_id = getattr(record, "tenant_id", None)
        if tenant_id:
            try:
                tenant_id = uuid.UUID(tenant_id)
            except (ValueError, TypeError):
                tenant_id = None

        # Build context
        context = {}
        if self.include_context:
            # Standard context fields
            for attr in ["user_id", "request_id", "operation", "resource_type", "resource_id"]:
                if hasattr(record, attr):
                    context[attr] = getattr(record, attr)

            # Actor from filter
            actor_id = getattr(record, "actor_id", None)
            if actor_id:
                context["actor_id"] = actor_id

        # Build metadata
        metadata = {
            "hostname": socket.gethostname(),
            "process_id": record.process,
            "thread_id": record.thread,
            "pathname": record.pathname,
            "lineno": record.lineno,
            "funcName": record.funcName,
        }

        # Build exception info
        exception = None
        if record.exc_info:
            exc_type, exc_value, exc_tb = record.exc_info
            if exc_type is not None:
                exception = {
                    "type": exc_type.__name__,
                    "message": str(exc_value),
                    "traceback": traceback.format_exception(exc_type, exc_value, exc_tb),
                }

        return {
            "timestamp": datetime.fromtimestamp(record.created),
            "level": level,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id,
            "tenant_id": tenant_id,
            "context": context,
            "metadata": metadata,
            "exception": exception,
        }

    def _write_entry(self, entry_data: dict[str, Any]) -> None:
        """Write a single log entry to database."""
        stream = self._get_stream()
        if stream is None:
            return

        try:
            from syn.log.models import LogEntry

            LogEntry.objects.create(stream=stream, **entry_data)

        except Exception as e:
            import sys

            print(f"[SynaraLogHandler] Failed to write log: {e}", file=sys.stderr)

    def _flush_buffer(self) -> None:
        """Flush buffered entries to database."""
        stream = self._get_stream()
        if stream is None:
            return

        entries_to_write = []
        with self._buffer_lock:
            entries_to_write = self._buffer.copy()
            self._buffer.clear()

        if not entries_to_write:
            return

        try:
            from syn.log.models import LogEntry

            # Bulk create for performance
            log_entries = [LogEntry(stream=stream, **data) for data in entries_to_write]
            LogEntry.objects.bulk_create(log_entries)

        except Exception as e:
            import sys

            print(f"[SynaraLogHandler] Failed to flush buffer: {e}", file=sys.stderr)

    def flush(self) -> None:
        """Flush any buffered entries."""
        if self.batch_size > 0:
            self._flush_buffer()

    def close(self) -> None:
        """Close handler and flush remaining entries."""
        self.flush()
        super().close()


# =============================================================================
# Logger Factory (LOG-001 §5.5)
# =============================================================================


def get_synara_logger(
    name: str,
    stream_name: str = "application",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Get a logger configured with SynaraLogHandler.

    Standard: LOG-001 §5.5

    Args:
        name: Logger name (usually __name__)
        stream_name: Log stream to write to
        level: Minimum log level

    Returns:
        Configured logger instance

    Usage:
        logger = get_synara_logger(__name__)
        logger.info('Processing request', extra={'user_id': 'u-123'})
    """
    logger = logging.getLogger(name)

    # Check if handler already added
    has_synara_handler = any(isinstance(h, SynaraLogHandler) for h in logger.handlers)

    if not has_synara_handler:
        handler = SynaraLogHandler(stream_name=stream_name, level=level)
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger


# =============================================================================
# Django Integration (LOG-002 §5)
# =============================================================================


def configure_django_logging(
    stream_name: str = "application",
    level: str = "INFO",
    loggers: list[str] | None = None,
) -> dict[str, Any]:
    """
    Generate Django LOGGING configuration for Synara logging.

    Standard: LOG-002 §5

    Args:
        stream_name: Log stream name
        level: Minimum log level
        loggers: List of logger names to configure (default: ['syn', 'django'])

    Returns:
        Django LOGGING dictionary

    Usage:
        from syn.log.handlers import configure_django_logging
        LOGGING = configure_django_logging()
    """
    if loggers is None:
        loggers = ["syn", "django"]

    logger_config = {
        name: {
            "handlers": ["synara", "console"],
            "level": level,
            "propagate": False,
        }
        for name in loggers
    }

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "correlation": {
                "()": "syn.log.handlers.CorrelationFilter",
            },
            "tenant": {
                "()": "syn.log.handlers.TenantFilter",
            },
        },
        "formatters": {
            "verbose": {
                "format": "[{asctime}] {levelname} {name} [{correlation_id}] {message}",
                "style": "{",
            },
            "json": {
                "()": "syn.log.handlers.JsonFormatter",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "verbose",
                "filters": ["correlation", "tenant"],
            },
            "synara": {
                "class": "syn.log.handlers.SynaraLogHandler",
                "level": level,
                "stream_name": stream_name,
            },
        },
        "loggers": logger_config,
    }


class JsonFormatter(logging.Formatter):
    """
    JSON formatter for structured log output.

    Standard: LOG-001 §5.1
    """

    def format(self, record: logging.LogRecord) -> str:
        import json

        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", None),
            "tenant_id": getattr(record, "tenant_id", None),
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)
