"""
IO Schemas (IO-001 §5)
======================

Base Pydantic V2 models for input/output validation at I/O boundaries.

Standard:     IO-001 §5 (Pydantic V2 Contract Model)
Compliance:   NIST SP 800-53 SI-10 (Information Input Validation)
Location:     syn/io/schemas.py
Version:      1.0.0

Usage:
    from syn.io import InputSchema, OutputSchema

    class CreateOrderInput(InputSchema):
        customer_id: str
        items: list[OrderItem]

        class Config:
            contract_version = "1.0.0"
            failure_domain = FailureDomain.FATAL

    class OrderOutput(OutputSchema):
        id: str
        status: str
        created_at: datetime
"""

import hashlib
from datetime import datetime
from typing import Any, ClassVar, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from syn.io.types import FailureDomain, IdempotencyStrategy

# =============================================================================
# INPUT SCHEMA (IO-001 §5.1)
# =============================================================================


class InputSchema(BaseModel):
    """
    Base schema for inbound data validation.

    All API request bodies and event payloads should inherit from this class.
    Provides:
    - Strict validation (no extra fields by default)
    - JSON Schema export
    - Failure domain classification
    - Idempotency support

    Standard: IO-001 §5.1 (InputSchema Requirements)

    Usage:
        class CreateUserInput(InputSchema):
            email: str
            name: str
            role: str = "user"

            class Config:
                contract_version = "1.0.0"
    """

    model_config = ConfigDict(
        # Strict mode: reject extra fields
        extra="forbid",
        # Validate on assignment
        validate_assignment=True,
        # Use enum values in serialization
        use_enum_values=True,
        # Enable JSON Schema generation
        json_schema_extra={
            "title": "InputSchema",
            "description": "Base input schema per IO-001 §5.1",
        },
    )

    # Contract metadata (set in subclasses)
    _contract_version: ClassVar[str] = "1.0.0"
    _failure_domain: ClassVar[FailureDomain] = FailureDomain.FATAL
    _idempotency_strategy: ClassVar[IdempotencyStrategy] = IdempotencyStrategy.CORRELATION_ID

    @classmethod
    def contract_version(cls) -> str:
        """Get contract version for this schema."""
        return cls._contract_version

    @classmethod
    def failure_domain(cls) -> FailureDomain:
        """Get failure domain for validation errors."""
        return cls._failure_domain

    @classmethod
    def idempotency_strategy(cls) -> IdempotencyStrategy:
        """Get idempotency strategy for this input."""
        return cls._idempotency_strategy

    def payload_hash(self) -> str:
        """
        Compute SHA256 hash of payload for idempotency.

        Standard: IO-001 §7.1 (Idempotency Strategies - payload_hash)
        """
        payload_json = self.model_dump_json(exclude_none=True)
        return hashlib.sha256(payload_json.encode()).hexdigest()


# =============================================================================
# OUTPUT SCHEMA (IO-001 §5.2)
# =============================================================================


class OutputSchema(BaseModel):
    """
    Base schema for outbound data validation.

    All API responses and event outputs should inherit from this class.
    Provides:
    - JSON Schema export
    - Consistent serialization
    - Envelope wrapping support

    Standard: IO-001 §5.2 (OutputSchema Requirements)

    Usage:
        class UserOutput(OutputSchema):
            id: str
            email: str
            name: str
            created_at: datetime
    """

    model_config = ConfigDict(
        # Allow extra fields in output (forward compatibility)
        extra="ignore",
        # Use enum values
        use_enum_values=True,
        # Serialize datetime as ISO8601
        ser_json_timedelta="iso8601",
        # JSON Schema metadata
        json_schema_extra={
            "title": "OutputSchema",
            "description": "Base output schema per IO-001 §5.2",
        },
    )

    # Contract metadata
    _contract_version: ClassVar[str] = "1.0.0"

    @classmethod
    def contract_version(cls) -> str:
        """Get contract version for this schema."""
        return cls._contract_version


# =============================================================================
# PAGINATION SCHEMAS (API-002 §7)
# =============================================================================


T = TypeVar("T", bound=OutputSchema)


class PaginatedOutput(OutputSchema, Generic[T]):
    """
    Paginated list response per API-002 §7.

    Standard: API-002 §7.1 (Pagination Response Shape)

    Usage:
        class UserListOutput(PaginatedOutput[UserOutput]):
            pass

        return UserListOutput(
            data=[user1, user2],
            next_cursor="abc123",
            total_estimate=100,
        )
    """

    data: List[T]
    """List of items in current page"""

    next_cursor: Optional[str] = None
    """Cursor for next page (None if last page)"""

    total_estimate: Optional[int] = None
    """Estimated total count (may be approximate)"""


class CursorPaginationInput(InputSchema):
    """
    Cursor pagination input per API-002 §7.

    Standard: API-002 §7.1 (Pagination Parameters)
    """

    cursor: Optional[str] = None
    """Cursor for page position (None for first page)"""

    limit: int = Field(default=50, ge=1, le=200)
    """Number of items per page (1-200, default 50)"""

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, v: int) -> int:
        if v < 1:
            raise ValueError("limit must be at least 1")
        if v > 200:
            raise ValueError("limit cannot exceed 200")
        return v


# =============================================================================
# ERROR SCHEMAS (ERR-002 Integration)
# =============================================================================


class ErrorDetail(OutputSchema):
    """
    Individual error detail per ERR-002.

    Standard: ERR-002 §4 (Error Envelope)
    """

    field: Optional[str] = None
    """Field that caused the error (for validation errors)"""

    issue: str
    """Human-readable issue description"""

    code: Optional[str] = None
    """Machine-readable error code"""


class ErrorOutput(OutputSchema):
    """
    Error response envelope per ERR-002.

    Standard: ERR-002 §4 (Canonical Error Envelope)
    """

    code: str
    """Machine-readable error code (e.g., 'validation.failed')"""

    message: str
    """Human-readable error message"""

    retryable: bool = False
    """Whether the client should retry this request"""

    request_id: Optional[str] = None
    """Request ID for correlation (Syn-Request-Id)"""

    correlation_id: Optional[str] = None
    """Correlation ID for tracing"""

    details: List[ErrorDetail] = Field(default_factory=list)
    """Additional error details"""

    doc: Optional[str] = None
    """Link to error documentation"""


# =============================================================================
# COMMON INPUT SCHEMAS
# =============================================================================


class TenantScopedInput(InputSchema):
    """
    Input that includes tenant scoping.

    Standard: SEC-001 §5.2 (Tenant Isolation)
    """

    tenant_id: Optional[str] = None
    """Tenant UUID (optional - usually from context)"""


class CorrelatedInput(InputSchema):
    """
    Input with correlation tracking.

    Standard: CTG-001 §5 (Correlation Tracking)
    """

    correlation_id: Optional[str] = None
    """Correlation ID for request tracing"""


class IdempotentInput(InputSchema):
    """
    Input with idempotency support.

    Standard: IO-001 §7 (Idempotency)
    """

    idempotency_key: Optional[str] = None
    """Idempotency key (ULID format)"""

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            import re

            if not re.match(r"^[0-9A-HJKMNP-TV-Z]{26}$", v):
                raise ValueError("idempotency_key must be ULID format (26 characters)")
        return v


# =============================================================================
# COMMON OUTPUT SCHEMAS
# =============================================================================


class TimestampedOutput(OutputSchema):
    """
    Output with standard timestamps.

    Standard: API-002 §10 (Field Conventions)
    """

    created_at: datetime
    """When the resource was created (RFC3339 UTC)"""

    updated_at: Optional[datetime] = None
    """When the resource was last updated (RFC3339 UTC)"""


class IdentifiedOutput(OutputSchema):
    """
    Output with standard identifier.

    Standard: API-002 §6 (Resource Naming)
    """

    id: str
    """Resource identifier (ULID recommended)"""


class ResourceOutput(IdentifiedOutput, TimestampedOutput):
    """
    Standard resource output combining ID and timestamps.

    Standard: API-002 §6, §10
    """

    pass


# =============================================================================
# VALIDATION RESULT (IO-001 §6)
# =============================================================================


class ValidationResult(OutputSchema):
    """
    Validation result from I/O validation gate.

    Standard: IO-001 §6 (Validation Gate)
    """

    valid: bool
    """Whether validation passed"""

    layer: int
    """Validation layer (1=structural, 2=governance, 7=schema)"""

    errors: List[ErrorDetail] = Field(default_factory=list)
    """Validation errors"""

    warnings: List[ErrorDetail] = Field(default_factory=list)
    """Validation warnings"""

    confidence: Optional[float] = None
    """Validation confidence score (0.0-1.0) per COG-001"""


# =============================================================================
# SCHEMA UTILITIES
# =============================================================================


def export_json_schema(schema_class: type[BaseModel]) -> Dict[str, Any]:
    """
    Export Pydantic model as JSON Schema.

    Standard: IO-001 §5.3 (Schema Exportability)

    Args:
        schema_class: Pydantic model class

    Returns:
        JSON Schema dictionary
    """
    return schema_class.model_json_schema()


def validate_input(schema_class: type[InputSchema], data: Dict[str, Any]) -> InputSchema:
    """
    Validate input data against schema.

    Standard: IO-001 §6.2 (Inbound I/O Validation)

    Args:
        schema_class: InputSchema subclass
        data: Input data dictionary

    Returns:
        Validated InputSchema instance

    Raises:
        ValidationError: If validation fails
    """
    return schema_class.model_validate(data)


def validate_output(schema_class: type[OutputSchema], data: Dict[str, Any]) -> OutputSchema:
    """
    Validate output data against schema.

    Standard: IO-001 §6.3 (Outbound I/O Validation)

    Args:
        schema_class: OutputSchema subclass
        data: Output data dictionary

    Returns:
        Validated OutputSchema instance

    Raises:
        ValidationError: If validation fails
    """
    return schema_class.model_validate(data)
