"""
Synara API Boundaries Module (SRX-001)
======================================

Serializer Reflex Boundary implementation per SRX-001.

Standard:     SRX-001 (Serializer Reflex Boundary Standard)
Compliance:   IO-001, SER-001, CGS-1001, AUD-001, EVT-001
Location:     syn/api/boundaries/
Version:      1.0.0

Features:
---------
- SerializerBoundary: Wraps DRF serializers with IO-001 contracts
- BoundaryContext: Request context for tenant/correlation/user
- IOContractRegistry: Registry for IO-001 contracts
- Boundary-specific exceptions

Usage:
------
    from syn.api.boundaries import (
        SerializerBoundary,
        BoundaryContext,
        IOContractRegistry,
        io_contract,
    )

    # Create boundary context from request
    ctx = BoundaryContext.from_request(request, "api.v1.users.create")

    # Create and execute boundary
    boundary = SerializerBoundary.for_create(
        serializer_cls=UserSerializer,
        io_contract_code="IO-USER-001",
        context=ctx,
    )
    result = boundary.execute(request.data)

    # Register a contract
    @io_contract("IO-USER-001", version="1.0.0")
    class CreateUserInput(InputSchema):
        email: str
        name: str

Architecture:
-------------
The boundary provides a canonical abstraction between:
- External API layer (DRF serializers)
- Internal SBL-001 pipeline (primitives, reflexes)

Flow: API Request -> SerializerBoundary -> Validation -> Governance -> Persist -> Events -> Audit

References:
-----------
- SRX-001: Serializer Reflex Boundary Standard
- IO-001: Input/Output Contract Standard
- SER-001: Serializer Patterns
- CGS-1001: Cognitive Governance Systems
- AUD-001: Audit Logging
- EVT-001: Event Taxonomy
"""

__version__ = "1.0.0"
__standard__ = "SRX-001"

# =============================================================================
# Context (SRX-001 §3.3)
# =============================================================================

# =============================================================================
# Boundary (SRX-001 §5)
# =============================================================================
from syn.api.boundaries.base import SerializerBoundary
from syn.api.boundaries.context import BoundaryContext

# =============================================================================
# Exceptions (SRX-001 §7)
# =============================================================================
from syn.api.boundaries.exceptions import (
    BoundaryConfigurationError,
    # Base
    BoundaryError,
    # Validation
    BoundaryValidationError,
    # Governance
    GovernanceBlockedError,
    GovernanceEscalatedError,
    IOContractViolationError,
    # Persistence
    PersistenceError,
)

# =============================================================================
# Registry (SRX-001 §6, IO-001 §8)
# =============================================================================
from syn.api.boundaries.registry import (
    IOContract,
    IOContractRegistry,
    io_contract,
)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Version
    "__version__",
    "__standard__",
    # Context
    "BoundaryContext",
    # Boundary
    "SerializerBoundary",
    # Registry
    "IOContract",
    "IOContractRegistry",
    "io_contract",
    # Exceptions - Base
    "BoundaryError",
    "BoundaryConfigurationError",
    # Exceptions - Validation
    "BoundaryValidationError",
    "IOContractViolationError",
    # Exceptions - Governance
    "GovernanceBlockedError",
    "GovernanceEscalatedError",
    # Exceptions - Persistence
    "PersistenceError",
]
