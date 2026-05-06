"""
ExecutionContext (IO-001 §9)
============================

Execution context for I/O operations providing tenant isolation,
correlation tracking, and request metadata.

Standard:     IO-001 §9 (Execution Context)
Compliance:   SEC-001 §5.2 (Tenant Isolation), CTG-001 §5 (Correlation)
Location:     syn/io/context.py
Version:      1.0.0
"""

import logging
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from syn.io.types import IdempotencyStrategy

logger = logging.getLogger(__name__)


# =============================================================================
# CONTEXT VARIABLES (Thread-Safe Storage)
# =============================================================================

_execution_context_var: ContextVar[Optional["ExecutionContext"]] = ContextVar("execution_context", default=None)


# =============================================================================
# EXECUTION CONTEXT (IO-001 §9)
# =============================================================================


@dataclass
class ExecutionContext:
    """
    Execution context for I/O operations.

    Provides:
    - Tenant isolation (SEC-001 §5.2)
    - Correlation tracking (CTG-001 §5)
    - Request metadata
    - Secrets access
    - Idempotency support

    Standard: IO-001 §9 (Execution Context)

    Usage:
        # Create context for a request
        ctx = ExecutionContext(
            tenant_id="tenant-uuid",
            correlation_id="corr-uuid",
        )

        # Use context manager
        with ctx:
            # All operations in this scope have access to context
            current = ExecutionContext.current()

        # Or set manually
        ctx.set_current()
        try:
            # ... operations
        finally:
            ctx.clear_current()
    """

    # Required fields (IO-001 §9.1)
    tenant_id: str
    """Tenant UUID for multi-tenant isolation (SEC-001 §5.2)"""

    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Request chain tracking ID (CTG-001 §5)"""

    # Optional fields
    parent_correlation_id: Optional[str] = None
    """Upstream lineage - parent request correlation ID"""

    request_id: Optional[str] = None
    """HTTP request ID (Syn-Request-Id header)"""

    idempotency_key: Optional[str] = None
    """Idempotency key for POST operations"""

    idempotency_strategy: IdempotencyStrategy = IdempotencyStrategy.CORRELATION_ID
    """Strategy for idempotency enforcement"""

    task_id: Optional[str] = None
    """Task ID if executing within a scheduled task (TASK-001)"""

    user_id: Optional[str] = None
    """Authenticated user ID"""

    actor: Optional[str] = None
    """Actor identifier (email or system name) for audit"""

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional execution metadata"""

    secrets: Dict[str, str] = field(default_factory=dict)
    """Tenant-specific credentials (encrypted, do not log)"""

    # Timing
    deadline: Optional[datetime] = None
    """Request deadline (from Syn-Deadline header)"""

    created_at: datetime = field(default_factory=datetime.utcnow)
    """When this context was created"""

    # Tracing
    trace_id: Optional[str] = None
    """W3C traceparent trace ID"""

    span_id: Optional[str] = None
    """W3C traceparent span ID"""

    trace_flags: Optional[str] = None
    """W3C traceparent flags"""

    def __enter__(self) -> "ExecutionContext":
        """Enter context manager scope."""
        self.set_current()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit context manager scope."""
        self.clear_current()
        return False

    def set_current(self) -> None:
        """Set this context as the current execution context."""
        _execution_context_var.set(self)

    @staticmethod
    def clear_current() -> None:
        """Clear the current execution context."""
        _execution_context_var.set(None)

    @staticmethod
    def current() -> Optional["ExecutionContext"]:
        """
        Get the current execution context.

        Returns:
            ExecutionContext or None if not in a context scope
        """
        return _execution_context_var.get()

    @staticmethod
    def require() -> "ExecutionContext":
        """
        Get the current execution context, raising if not set.

        Returns:
            ExecutionContext

        Raises:
            RuntimeError: If no context is set

        Standard: IO-001 §9.2 (Tenant Isolation enforcement)
        """
        ctx = _execution_context_var.get()
        if ctx is None:
            raise RuntimeError(
                "ExecutionContext required but not set. Ensure request is processed through IO middleware."
            )
        return ctx

    def child(
        self,
        correlation_id: Optional[str] = None,
        **overrides,
    ) -> "ExecutionContext":
        """
        Create a child context for nested operations.

        The child inherits tenant_id and sets parent_correlation_id
        to this context's correlation_id.

        Args:
            correlation_id: New correlation ID (auto-generated if not provided)
            **overrides: Field overrides

        Returns:
            New ExecutionContext with parent linkage
        """
        return ExecutionContext(
            tenant_id=self.tenant_id,
            correlation_id=correlation_id or str(uuid.uuid4()),
            parent_correlation_id=self.correlation_id,
            request_id=overrides.get("request_id", self.request_id),
            user_id=overrides.get("user_id", self.user_id),
            actor=overrides.get("actor", self.actor),
            metadata={**self.metadata, **overrides.get("metadata", {})},
            secrets=self.secrets,  # Inherit secrets
            trace_id=self.trace_id,
            span_id=overrides.get("span_id"),
            trace_flags=self.trace_flags,
            **{
                k: v for k, v in overrides.items() if k not in ["request_id", "user_id", "actor", "metadata", "span_id"]
            },
        )

    def to_dict(self, include_secrets: bool = False) -> Dict[str, Any]:
        """
        Convert context to dictionary for serialization.

        Args:
            include_secrets: Whether to include secrets (default: False)

        Returns:
            Dictionary representation (safe for logging if include_secrets=False)
        """
        result = {
            "tenant_id": self.tenant_id,
            "correlation_id": self.correlation_id,
            "parent_correlation_id": self.parent_correlation_id,
            "request_id": self.request_id,
            "idempotency_key": self.idempotency_key,
            "task_id": self.task_id,
            "user_id": self.user_id,
            "actor": self.actor,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "created_at": self.created_at.isoformat(),
        }
        if include_secrets:
            result["secrets"] = self.secrets
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    def to_log_dict(self) -> Dict[str, Any]:
        """
        Convert context to dictionary for logging (excludes sensitive data).

        Standard: LOG-001 §5 (Correlation Tracking)
        """
        return {
            "tenant_id": self.tenant_id,
            "correlation_id": self.correlation_id,
            "request_id": self.request_id,
            "user_id": self.user_id,
            "task_id": self.task_id,
            "trace_id": self.trace_id,
        }

    @classmethod
    def from_request(cls, request) -> "ExecutionContext":
        """
        Create ExecutionContext from Django request.

        Extracts context from request attributes set by middleware:
        - request.tenant_id (from TenantIsolationMiddleware)
        - request.syn_request_id (from SynRequestIdMiddleware)
        - request.correlation_id (from CorrelationMiddleware)
        - request.user (from authentication)

        Args:
            request: Django HttpRequest

        Returns:
            ExecutionContext populated from request

        Standard: IO-001 §9, API-002 §8
        """
        # Extract tenant_id
        tenant_id = None
        if hasattr(request, "tenant_id"):
            tenant_id = str(request.tenant_id)
        elif hasattr(request, "tenant") and request.tenant:
            tenant_id = str(request.tenant.id)
        elif hasattr(request, "user") and request.user.is_authenticated:
            if hasattr(request.user, "tenant_id"):
                tenant_id = str(request.user.tenant_id)

        if not tenant_id:
            # Fallback for unauthenticated requests (with warning per SYS-200 INV-001)
            logger.debug(f"[IO CONTEXT] No tenant_id on request to {request.path}, using 'system' fallback")
            tenant_id = "system"

        # Extract correlation_id
        correlation_id = (
            getattr(request, "correlation_id", None)
            or getattr(request, "syn_correlation_id", None)
            or str(uuid.uuid4())
        )

        # Extract request_id
        request_id = getattr(request, "syn_request_id", None) or getattr(request, "request_id", None)

        # Extract idempotency key
        idempotency_key = request.headers.get("Idempotency-Key")

        # Extract user info
        user_id = None
        actor = None
        if hasattr(request, "user") and request.user.is_authenticated:
            user_id = str(request.user.pk)
            actor = getattr(request.user, "email", None) or str(request.user)

        # Extract trace context (prefer middleware-parsed values)
        trace_id = getattr(request, "syn_trace_id", None)
        span_id = getattr(request, "syn_span_id", None)
        trace_flags = getattr(request, "syn_trace_flags", None)

        # Fallback: parse traceparent header directly if not set by middleware
        if not trace_id:
            traceparent = request.headers.get("traceparent")
            if traceparent:
                parts = traceparent.split("-")
                if len(parts) >= 4:
                    trace_id = parts[1]
                    span_id = parts[2]
                    trace_flags = parts[3]

        # Extract deadline
        deadline = None
        deadline_header = request.headers.get("Syn-Deadline")
        if deadline_header:
            try:
                from datetime import timezone

                # Try RFC3339 format
                deadline = datetime.fromisoformat(deadline_header.replace("Z", "+00:00"))
            except ValueError:
                try:
                    # Try epoch milliseconds
                    deadline = datetime.fromtimestamp(int(deadline_header) / 1000, tz=timezone.utc)
                except (ValueError, TypeError):
                    pass

        return cls(
            tenant_id=tenant_id,
            correlation_id=correlation_id,
            request_id=request_id,
            idempotency_key=idempotency_key,
            user_id=user_id,
            actor=actor,
            trace_id=trace_id,
            span_id=span_id,
            trace_flags=trace_flags,
            deadline=deadline,
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_current_tenant_id() -> Optional[str]:
    """Get tenant_id from current execution context."""
    ctx = ExecutionContext.current()
    return ctx.tenant_id if ctx else None


def get_current_correlation_id() -> Optional[str]:
    """Get correlation_id from current execution context."""
    ctx = ExecutionContext.current()
    return ctx.correlation_id if ctx else None


def get_current_user_id() -> Optional[str]:
    """Get user_id from current execution context."""
    ctx = ExecutionContext.current()
    return ctx.user_id if ctx else None


def require_tenant_id() -> str:
    """
    Get tenant_id, raising if not in context.

    Standard: SEC-001 §5.2 (Tenant Isolation)
    """
    ctx = ExecutionContext.require()
    return ctx.tenant_id
