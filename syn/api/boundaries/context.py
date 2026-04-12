"""
Boundary Context per SRX-001 §3.3 and §5.3.

Standard:     SRX-001 §3.3 (BoundaryContext), §5.3 (Schema)
Compliance:   SEC-001 §5.2 (Tenant Isolation), CTG-001 §5 (Correlation)
Version:      1.0.0
Last Updated: 2025-12-03

Provides:
- BoundaryContext dataclass for serializer boundary operations
- Context propagation for tenant isolation and correlation tracking
- Request metadata capture for audit trails

References:
- SRX-001: Serializer Reflex Boundary Standard
- SEC-001: Security Architecture (tenant isolation)
- CTG-001: Causal Trace Graph (correlation tracking)
- AUD-001: Audit Logging
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from django.utils import timezone


@dataclass
class BoundaryContext:
    """
    Context for serializer boundary operations per SRX-001 §3.3.

    BoundaryContext ensures that all IO, governance, auditing, and events
    share the same lineage identifiers and tenant context.

    Attributes:
        tenant_id: Tenant scope for multi-tenant isolation (required)
        correlation_id: Event chain correlation ID for CTG linkage (required)
        source: Source identifier (e.g., 'api.v1.users.create') (required)
        user_id: Authenticated user performing the action (optional)
        request_ip: Source IP address for audit (optional)
        user_agent: Client user agent for audit (optional)
        metadata: Additional key-value pairs for custom context (optional)
        timestamp: When the boundary operation was initiated

    Example:
        >>> ctx = BoundaryContext(
        ...     tenant_id=UUID("123e4567-e89b-12d3-a456-426614174000"),
        ...     correlation_id=UUID("987fcdeb-51a2-3bc4-d567-890123456789"),
        ...     source="api.v1.users.create",
        ...     user_id=UUID("111e2222-e89b-12d3-a456-426614174000"),
        ...     request_ip="192.168.1.100",
        ... )

    Compliance:
    - SEC-001 §5.2.1: tenant_id enforces row-level tenant isolation
    - CTG-001 §5: correlation_id enables event chain reconstruction
    - AUD-001 §4.1: All required audit fields captured
    """

    # Required fields per SRX-001 §5.3
    tenant_id: UUID
    source: str

    # Correlation ID - required, auto-generated if not provided
    correlation_id: UUID = field(default_factory=uuid4)

    # Optional fields
    user_id: UUID | None = None
    request_ip: str | None = None
    user_agent: str | None = None
    metadata: dict[str, Any] | None = field(default_factory=dict)

    # Timestamp for tracking
    timestamp: datetime = field(default_factory=timezone.now)

    # Parent correlation for event chaining
    parent_correlation_id: UUID | None = None

    def __post_init__(self):
        """Validate and normalize context after initialization."""
        # Ensure tenant_id is a UUID
        if isinstance(self.tenant_id, str):
            self.tenant_id = UUID(self.tenant_id)

        # Ensure correlation_id is a UUID
        if isinstance(self.correlation_id, str):
            self.correlation_id = UUID(self.correlation_id)

        # Ensure user_id is a UUID if provided
        if self.user_id and isinstance(self.user_id, str):
            self.user_id = UUID(self.user_id)

        # Ensure parent_correlation_id is a UUID if provided
        if self.parent_correlation_id and isinstance(self.parent_correlation_id, str):
            self.parent_correlation_id = UUID(self.parent_correlation_id)

        # Initialize metadata if None
        if self.metadata is None:
            self.metadata = {}

    @classmethod
    def from_request(
        cls,
        request,
        source: str,
        tenant_id: UUID | None = None,
        correlation_id: UUID | None = None,
    ) -> BoundaryContext:
        """
        Create BoundaryContext from a Django REST Framework request.

        Extracts tenant_id, user_id, and correlation_id from request
        headers and authentication context.

        Args:
            request: DRF request object
            source: Source identifier (e.g., 'api.v1.users.create')
            tenant_id: Override tenant_id (uses request context if not provided)
            correlation_id: Override correlation_id (uses X-Correlation-ID if not provided)

        Returns:
            BoundaryContext instance

        Example:
            >>> ctx = BoundaryContext.from_request(request, "api.v1.users.create")
        """
        # Extract tenant_id
        if tenant_id is None:
            # Try request attributes first (set by middleware)
            tenant_id = getattr(request, "tenant_id", None)
            # Fall back to header
            if tenant_id is None:
                tenant_header = request.META.get("HTTP_X_TENANT_ID")
                if tenant_header:
                    tenant_id = UUID(tenant_header)

        if tenant_id is None:
            raise ValueError("tenant_id is required but not found in request")

        # Extract correlation_id from header or generate
        if correlation_id is None:
            corr_header = request.META.get("HTTP_X_CORRELATION_ID")
            if corr_header:
                correlation_id = UUID(corr_header)
            else:
                correlation_id = uuid4()

        # Extract user_id from authenticated user
        user_id = None
        if hasattr(request, "user") and request.user.is_authenticated:
            user_id = getattr(request.user, "id", None)
            if user_id:
                user_id = UUID(str(user_id)) if not isinstance(user_id, UUID) else user_id

        # Extract request metadata
        request_ip = cls._get_client_ip(request)
        user_agent = request.META.get("HTTP_USER_AGENT", "")

        return cls(
            tenant_id=tenant_id,
            correlation_id=correlation_id,
            source=source,
            user_id=user_id,
            request_ip=request_ip,
            user_agent=user_agent,
        )

    @staticmethod
    def _get_client_ip(request) -> str | None:
        """Extract client IP from request, handling proxies."""
        x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
        if x_forwarded_for:
            return x_forwarded_for.split(",")[0].strip()
        return request.META.get("REMOTE_ADDR")

    def to_dict(self) -> dict[str, Any]:
        """
        Convert context to dictionary for serialization.

        Returns:
            Dictionary representation suitable for JSON/event payloads
        """
        return {
            "tenant_id": str(self.tenant_id),
            "correlation_id": str(self.correlation_id),
            "source": self.source,
            "user_id": str(self.user_id) if self.user_id else None,
            "request_ip": self.request_ip,
            "user_agent": self.user_agent,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "parent_correlation_id": (str(self.parent_correlation_id) if self.parent_correlation_id else None),
        }

    def to_audit_payload(self) -> dict[str, Any]:
        """
        Convert context to AUD-001 compliant audit payload.

        Returns:
            Dictionary with AUD-001 required fields
        """
        return {
            "tenant_id": str(self.tenant_id),
            "correlation_id": str(self.correlation_id),
            "actor_type": "user" if self.user_id else "system",
            "actor_id": str(self.user_id) if self.user_id else "system",
            "actor_ip": self.request_ip,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_event_payload(self) -> dict[str, Any]:
        """
        Convert context to EVT-001 compliant event payload.

        Returns:
            Dictionary with EVT-001 required fields
        """
        payload = {
            "tenant_id": str(self.tenant_id),
            "correlation_id": str(self.correlation_id),
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.user_id:
            payload["user_id"] = str(self.user_id)
        if self.parent_correlation_id:
            payload["parent_correlation_id"] = str(self.parent_correlation_id)
        return payload

    def with_metadata(self, **kwargs) -> BoundaryContext:
        """
        Create a new context with additional metadata.

        Args:
            **kwargs: Key-value pairs to add to metadata

        Returns:
            New BoundaryContext with merged metadata
        """
        new_metadata = {**self.metadata, **kwargs}
        return BoundaryContext(
            tenant_id=self.tenant_id,
            correlation_id=self.correlation_id,
            source=self.source,
            user_id=self.user_id,
            request_ip=self.request_ip,
            user_agent=self.user_agent,
            metadata=new_metadata,
            timestamp=self.timestamp,
            parent_correlation_id=self.parent_correlation_id,
        )

    def child_context(self, new_source: str) -> BoundaryContext:
        """
        Create a child context for nested operations.

        The child inherits tenant and user context but gets a new
        correlation_id linked to the parent.

        Args:
            new_source: Source identifier for the child operation

        Returns:
            New BoundaryContext with parent linkage
        """
        return BoundaryContext(
            tenant_id=self.tenant_id,
            correlation_id=uuid4(),
            source=new_source,
            user_id=self.user_id,
            request_ip=self.request_ip,
            user_agent=self.user_agent,
            metadata=self.metadata.copy() if self.metadata else {},
            parent_correlation_id=self.correlation_id,
        )
