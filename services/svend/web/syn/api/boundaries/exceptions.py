"""
Boundary Exceptions per SRX-001.

Standard:     SRX-001 §5.1 (Exception Types)
Compliance:   ERR-001 (Error Handling), API-001 (Error Responses)
Version:      1.0.0
Last Updated: 2025-12-03

Provides:
- BoundaryError base exception
- ValidationError for SER-001/IO-001 failures
- GovernanceBlockedError for CGS-1001 BLOCK decisions
- GovernanceEscalatedError for CGS-1001 ESCALATE decisions
- IOContractViolationError for IO-001 contract failures
- PersistenceError for database operation failures

References:
- SRX-001: Serializer Reflex Boundary Standard
- ERR-001: Error Handling and Recovery Standard
- CGS-1001: Cognitive Governance Systems
- IO-001: Input/Output Contract Standard
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID


class BoundaryError(Exception):
    """
    Base exception for all boundary operations.

    Provides structured error information for audit trails
    and API error responses.

    Attributes:
        code: Error code (e.g., 'SRX-VAL-001')
        message: Human-readable error message
        correlation_id: Correlation ID for tracing
        details: Additional error context
    """

    def __init__(
        self,
        code: str,
        message: str,
        correlation_id: Optional[UUID] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.code = code
        self.message = message
        self.correlation_id = correlation_id
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        result = {
            "error_code": self.code,
            "message": self.message,
            "details": self.details,
        }
        if self.correlation_id:
            result["correlation_id"] = str(self.correlation_id)
        return result

    def to_audit_payload(self) -> Dict[str, Any]:
        """Convert exception to AUD-001 audit payload."""
        return {
            "error_code": self.code,
            "error_message": self.message,
            "error_details": self.details,
            "correlation_id": str(self.correlation_id) if self.correlation_id else None,
        }


class BoundaryValidationError(BoundaryError):
    """
    Validation error from serializer or IO contract validation.

    Raised when:
    - DRF serializer validation fails (SER-001)
    - IO-001 contract validation fails
    - Custom boundary validators fail

    Attributes:
        field_errors: Per-field validation errors
        non_field_errors: Non-field-specific errors
    """

    def __init__(
        self,
        message: str = "Validation failed",
        field_errors: Optional[Dict[str, List[str]]] = None,
        non_field_errors: Optional[List[str]] = None,
        correlation_id: Optional[UUID] = None,
        contract_code: Optional[str] = None,
    ):
        self.field_errors = field_errors or {}
        self.non_field_errors = non_field_errors or []
        self.contract_code = contract_code

        details = {
            "field_errors": self.field_errors,
            "non_field_errors": self.non_field_errors,
        }
        if contract_code:
            details["contract_code"] = contract_code

        super().__init__(
            code="SRX-VAL-001",
            message=message,
            correlation_id=correlation_id,
            details=details,
        )

    @classmethod
    def from_serializer_errors(
        cls,
        errors: Dict[str, Any],
        correlation_id: Optional[UUID] = None,
    ) -> "BoundaryValidationError":
        """
        Create validation error from DRF serializer errors.

        Args:
            errors: Serializer.errors dictionary
            correlation_id: Correlation ID for tracing

        Returns:
            BoundaryValidationError instance
        """
        field_errors = {}
        non_field_errors = []

        for key, value in errors.items():
            if key == "non_field_errors":
                non_field_errors = [str(e) for e in value]
            else:
                if isinstance(value, list):
                    field_errors[key] = [str(e) for e in value]
                else:
                    field_errors[key] = [str(value)]

        return cls(
            message="Serializer validation failed",
            field_errors=field_errors,
            non_field_errors=non_field_errors,
            correlation_id=correlation_id,
        )


class IOContractViolationError(BoundaryError):
    """
    IO-001 contract validation failure.

    Raised when validated data does not conform to the
    declared IO-001 contract schema.

    Attributes:
        contract_code: IO-001 contract identifier (e.g., 'IO-USER-001')
        contract_version: Contract version
        violations: List of specific violations
    """

    def __init__(
        self,
        contract_code: str,
        violations: List[str],
        contract_version: Optional[str] = None,
        correlation_id: Optional[UUID] = None,
    ):
        self.contract_code = contract_code
        self.contract_version = contract_version
        self.violations = violations

        details = {
            "contract_code": contract_code,
            "violations": violations,
        }
        if contract_version:
            details["contract_version"] = contract_version

        super().__init__(
            code="SRX-IO-001",
            message=f"IO contract '{contract_code}' validation failed",
            correlation_id=correlation_id,
            details=details,
        )


class GovernanceBlockedError(BoundaryError):
    """
    Governance BLOCK decision per CGS-1001.

    Raised when governance preflight evaluation returns BLOCK,
    preventing the operation from proceeding.

    Attributes:
        rule_id: Governance rule that blocked the operation
        reason: Human-readable reason for blocking
        confidence: Confidence score of the decision
    """

    def __init__(
        self,
        reason: str,
        rule_id: Optional[str] = None,
        confidence: Optional[float] = None,
        correlation_id: Optional[UUID] = None,
    ):
        self.rule_id = rule_id
        self.reason = reason
        self.confidence = confidence

        details = {
            "judgment": "BLOCK",
            "reason": reason,
        }
        if rule_id:
            details["rule_id"] = rule_id
        if confidence is not None:
            details["confidence"] = confidence

        super().__init__(
            code="SRX-GOV-001",
            message=f"Operation blocked by governance: {reason}",
            correlation_id=correlation_id,
            details=details,
        )


class GovernanceEscalatedError(BoundaryError):
    """
    Governance ESCALATE decision per CGS-1001.

    Raised when governance preflight evaluation returns ESCALATE,
    requiring human approval before proceeding.

    Attributes:
        rule_id: Governance rule that triggered escalation
        reason: Reason for escalation
        escalation_id: ID of the created escalation request
        approvers: List of required approvers
    """

    def __init__(
        self,
        reason: str,
        rule_id: Optional[str] = None,
        escalation_id: Optional[str] = None,
        approvers: Optional[List[str]] = None,
        correlation_id: Optional[UUID] = None,
    ):
        self.rule_id = rule_id
        self.reason = reason
        self.escalation_id = escalation_id
        self.approvers = approvers or []

        details = {
            "judgment": "ESCALATE",
            "reason": reason,
        }
        if rule_id:
            details["rule_id"] = rule_id
        if escalation_id:
            details["escalation_id"] = escalation_id
        if approvers:
            details["approvers"] = approvers

        super().__init__(
            code="SRX-GOV-002",
            message=f"Operation escalated for approval: {reason}",
            correlation_id=correlation_id,
            details=details,
        )


class PersistenceError(BoundaryError):
    """
    Database operation failure.

    Raised when the persistence layer fails during
    create, update, or delete operations.

    Attributes:
        operation: Operation that failed (create, update, delete)
        model: Model class name
        original_error: Original database exception
    """

    def __init__(
        self,
        operation: str,
        model: str,
        original_error: Optional[Exception] = None,
        correlation_id: Optional[UUID] = None,
    ):
        self.operation = operation
        self.model = model
        self.original_error = original_error

        details = {
            "operation": operation,
            "model": model,
        }
        if original_error:
            details["original_error"] = str(original_error)

        super().__init__(
            code="SRX-DB-001",
            message=f"Failed to {operation} {model}",
            correlation_id=correlation_id,
            details=details,
        )


class BoundaryConfigurationError(BoundaryError):
    """
    Boundary configuration or setup error.

    Raised when the boundary is misconfigured, such as
    missing IO contract or invalid serializer class.
    """

    def __init__(
        self,
        message: str,
        correlation_id: Optional[UUID] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            code="SRX-CFG-001",
            message=message,
            correlation_id=correlation_id,
            details=details or {},
        )
