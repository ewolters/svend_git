"""
Synara IO Module (IO-001/002)
=============================

Input/Output Contract framework for validating data at I/O boundaries.

Standard:     IO-001 (Contract Standard), IO-002 (Behavior Standard)
Compliance:   NIST SP 800-53 SI-10, ISO 27001 A.14.2.5
Location:     syn/io/
Version:      1.0.0

Features:
---------
- ExecutionContext: Request context with tenant/correlation/idempotency
- InputSchema: Pydantic base for input validation
- OutputSchema: Pydantic base for output validation
- Type definitions for failure domains, idempotency strategies

Usage:
------
    from syn.io import (
        ExecutionContext,
        InputSchema,
        OutputSchema,
        FailureDomain,
        IdempotencyStrategy,
    )

    # Create execution context from request
    ctx = ExecutionContext.from_request(request)

    # Define input schema
    class CreateOrderInput(InputSchema):
        customer_id: str
        items: list[str]

    # Define output schema
    class OrderOutput(OutputSchema):
        id: str
        status: str
        created_at: datetime
"""

from syn.io.context import (
    ExecutionContext,
    get_current_correlation_id,
    get_current_tenant_id,
    get_current_user_id,
    require_tenant_id,
)

# =============================================================================
# Schemas (IO-001 §5)
# =============================================================================
from syn.io.schemas import (
    CorrelatedInput,
    CursorPaginationInput,
    # Error schemas
    ErrorDetail,
    ErrorOutput,
    IdempotentInput,
    IdentifiedOutput,
    # Base schemas
    InputSchema,
    OutputSchema,
    # Pagination
    PaginatedOutput,
    ResourceOutput,
    # Common patterns
    TenantScopedInput,
    TimestampedOutput,
    # Validation
    ValidationResult,
    # Utilities
    export_json_schema,
    validate_input,
    validate_output,
)
from syn.io.types import (
    NON_RETRYABLE_STATUS_CODES,
    # Status code helpers
    RETRYABLE_STATUS_CODES,
    STATUS_CODE_TO_FAILURE_DOMAIN,
    # Enums
    FailureDomain,
    IdempotencyDefaults,
    IdempotencyStrategy,
    # Header constants
    IOHeaders,
    PaginationDefaults,
    RateLimitDefaults,
    RetryDefaults,
    StreamingDefaults,
    # Timeout/Retry/Rate limit defaults
    TimeoutDefaults,
    ValidationLayer,
    get_failure_domain,
    is_retryable,
)

__version__ = "1.0.0"
__standard__ = "IO-002"
default_app_config = "syn.io.apps.IoConfig"

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Version
    "__version__",
    "__standard__",
    # Types
    "FailureDomain",
    "IdempotencyStrategy",
    "ValidationLayer",
    "IOHeaders",
    "TimeoutDefaults",
    "RetryDefaults",
    "RateLimitDefaults",
    "PaginationDefaults",
    "IdempotencyDefaults",
    "StreamingDefaults",
    "RETRYABLE_STATUS_CODES",
    "NON_RETRYABLE_STATUS_CODES",
    "STATUS_CODE_TO_FAILURE_DOMAIN",
    "get_failure_domain",
    "is_retryable",
    # Context
    "ExecutionContext",
    "get_current_tenant_id",
    "get_current_correlation_id",
    "get_current_user_id",
    "require_tenant_id",
    # Schemas - Base
    "InputSchema",
    "OutputSchema",
    # Schemas - Pagination
    "PaginatedOutput",
    "CursorPaginationInput",
    # Schemas - Errors
    "ErrorDetail",
    "ErrorOutput",
    # Schemas - Common
    "TenantScopedInput",
    "CorrelatedInput",
    "IdempotentInput",
    "TimestampedOutput",
    "IdentifiedOutput",
    "ResourceOutput",
    # Validation
    "ValidationResult",
    "export_json_schema",
    "validate_input",
    "validate_output",
]
