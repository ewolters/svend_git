"""
Serializer Boundary Base Class per SRX-001 §5.

Standard:     SRX-001 §5 (Boundary Contract)
Compliance:   IO-001, SER-001, CGS-1001, AUD-001, EVT-001
Version:      1.0.0
Last Updated: 2025-12-03

Provides:
- SerializerBoundary base class for wrapping DRF serializers
- Validation pipeline: SER-001 -> IO-001 -> CGS-1001
- Event emission for io.* events
- Audit trail integration
- Governance preflight checks

References:
- SRX-001: Serializer Reflex Boundary Standard
- SER-001: Serializer Patterns
- IO-001: Input/Output Contract
- CGS-1001: Cognitive Governance Systems
- AUD-001: Audit Logging
- EVT-001: Event Taxonomy
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, TypeVar

from django.db import transaction
from django.utils import timezone
from rest_framework import serializers

from syn.api.boundaries.context import BoundaryContext
from syn.api.boundaries.exceptions import (
    BoundaryConfigurationError,
    BoundaryValidationError,
    GovernanceBlockedError,
    GovernanceEscalatedError,
    IOContractViolationError,
    PersistenceError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SerializerBoundary:
    """
    Canonical boundary between API serializers and SBL-001 pipeline.

    Implements SRX-001: Serializer Reflex Boundary Standard.

    SerializerBoundary wraps (not subclasses) DRF serializers and orchestrates:
    - Structural validation via the serializer (SER-001)
    - IO-001 contract validation
    - Governance preflight checks (CGS-1001)
    - Event emission into Cortex (EVT-001)
    - Audit logging (AUD-001)
    - Correlation propagation and tenant isolation (SEC-001)

    Usage:
        >>> from syn.api.boundaries import SerializerBoundary, BoundaryContext
        >>> from myapp.serializers import UserSerializer
        >>>
        >>> ctx = BoundaryContext(
        ...     tenant_id=UUID("..."),
        ...     source="api.v1.users.create",
        ... )
        >>> boundary = SerializerBoundary(
        ...     serializer_cls=UserSerializer,
        ...     io_contract_code="IO-USER-001",
        ...     operation="create",
        ...     context=ctx,
        ... )
        >>> result = boundary.execute({"email": "test@example.com", "name": "Test"})

    Attributes:
        serializer_cls: DRF serializer class (not instance)
        io_contract_code: IO-001 contract identifier
        operation: Operation name (create, update, delete)
        context: BoundaryContext with tenant, correlation, user info
        instance: Model instance for update/delete operations
        validated_data: Data after successful validation
        persisted: Whether data was persisted
        result: Serialized result after persistence
    """

    # Operation types
    OPERATION_CREATE = "create"
    OPERATION_UPDATE = "update"
    OPERATION_DELETE = "delete"
    OPERATION_BULK_CREATE = "bulk_create"
    OPERATION_BULK_UPDATE = "bulk_update"
    OPERATION_BULK_DELETE = "bulk_delete"

    # Governance ruleset for boundary operations
    GOVERNANCE_RULESET = "io.boundary"

    def __init__(
        self,
        serializer_cls: type[serializers.Serializer],
        io_contract_code: str,
        operation: str,
        *,
        context: BoundaryContext,
        instance: Any | None = None,
        partial: bool = False,
        many: bool = False,
    ):
        """
        Initialize serializer boundary.

        Args:
            serializer_cls: DRF serializer class (not instance)
            io_contract_code: IO-001 contract identifier (e.g., "IO-USER-001")
            operation: Operation name (e.g., "create", "update", "delete")
            context: BoundaryContext with tenant, correlation, user info
            instance: Model instance for update/delete operations
            partial: Whether this is a partial update
            many: Whether handling multiple objects

        Raises:
            BoundaryConfigurationError: If configuration is invalid
        """
        # Validate serializer_cls is a class, not instance
        if not isinstance(serializer_cls, type):
            raise BoundaryConfigurationError(
                f"serializer_cls must be a class, not {type(serializer_cls).__name__}",
                correlation_id=context.correlation_id,
            )

        if not issubclass(serializer_cls, serializers.Serializer):
            raise BoundaryConfigurationError(
                "serializer_cls must be a Serializer subclass",
                correlation_id=context.correlation_id,
            )

        self.serializer_cls = serializer_cls
        self.io_contract_code = io_contract_code
        self.operation = operation
        self.context = context
        self.instance = instance
        self.partial = partial
        self.many = many

        # State tracking
        self._serializer: serializers.Serializer | None = None
        self._validated_data: dict[str, Any] | None = None
        self._persisted: bool = False
        self._result: dict[str, Any] | None = None
        self._start_time: datetime = timezone.now()

        # Event tracking
        self._events_emitted: list[str] = []

    @property
    def validated_data(self) -> dict[str, Any] | None:
        """Return validated data after validation."""
        return self._validated_data

    @property
    def persisted(self) -> bool:
        """Return whether data was persisted."""
        return self._persisted

    @property
    def result(self) -> dict[str, Any] | None:
        """Return serialized result after persistence."""
        return self._result

    @property
    def model_name(self) -> str:
        """Get model name from serializer Meta."""
        if hasattr(self.serializer_cls, "Meta") and hasattr(self.serializer_cls.Meta, "model"):
            return self.serializer_cls.Meta.model.__name__
        return self.serializer_cls.__name__.replace("Serializer", "")

    # =========================================================================
    # Core Boundary Operations (SRX-001 §5.1)
    # =========================================================================

    def validate(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Perform SER-001 structural validation and IO-001 contract validation.

        Validation order per SRX-001 §6.2:
        1. DRF serializer structural validation (SER-001)
        2. IO-001 contract validation (schema-level)

        Emits:
        - io.validated (on success)
        - io.validation.failed (on failure, EVT-001 compliant)

        Args:
            data: Input data to validate

        Returns:
            Validated data dictionary

        Raises:
            BoundaryValidationError: If validation fails
        """
        logger.debug(f"[BOUNDARY] Validating {self.operation} on {self.model_name} (contract: {self.io_contract_code})")

        try:
            # Step 1: DRF serializer validation (SER-001)
            self._serializer = self._create_serializer(data)
            if not self._serializer.is_valid():
                error = BoundaryValidationError.from_serializer_errors(
                    self._serializer.errors,
                    correlation_id=self.context.correlation_id,
                )
                self._emit_validation_failed(error)
                raise error

            self._validated_data = self._serializer.validated_data

            # Step 2: IO-001 contract validation
            self._validate_io_contract(self._validated_data)

            # Emit success event
            self._emit_validated()

            logger.debug(f"[BOUNDARY] Validation successful for {self.model_name}")
            return self._validated_data

        except BoundaryValidationError:
            raise
        except IOContractViolationError:
            raise
        except Exception as e:
            logger.error(f"[BOUNDARY] Unexpected validation error: {e}", exc_info=True)
            error = BoundaryValidationError(
                message=f"Validation error: {str(e)}",
                correlation_id=self.context.correlation_id,
            )
            self._emit_validation_failed(error)
            raise error

    def execute(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Full boundary execution: validate -> governance -> persist -> emit -> audit.

        Execution flow per SRX-001 §4.3:
        1. Validate data (SER-001 + IO-001)
        2. Emit io.governance.preflight event (EVT-001)
        3. Evaluate governance rules (CGS-1001)
        4. If ALLOW: Persist and emit io.persisted.*
        5. If BLOCK: Emit io.governance.blocked, raise error (EVT-001)
        6. If ESCALATE: Emit io.governance.escalated, raise error (EVT-001)
        7. Write audit log

        Args:
            data: Input data to process

        Returns:
            Serialized representation of the persisted object

        Raises:
            BoundaryValidationError: If validation fails
            IOContractViolationError: If IO contract check fails
            GovernanceBlockedError: If governance blocks the operation
            GovernanceEscalatedError: If governance escalates
            PersistenceError: If database operation fails
        """
        logger.info(f"[BOUNDARY] Executing {self.operation} on {self.model_name} (tenant: {self.context.tenant_id})")

        # Step 1: Validate
        validated_data = self.validate(data)

        # Step 2: Governance preflight
        judgment = self._execute_governance_preflight(validated_data)

        # Step 3: Handle judgment
        if judgment["decision"] == "BLOCK":
            self._handle_governance_block(judgment)
            # _handle_governance_block raises GovernanceBlockedError

        elif judgment["decision"] == "ESCALATE":
            self._handle_governance_escalate(judgment)
            # _handle_governance_escalate raises GovernanceEscalatedError

        # Step 4: ALLOW - Proceed with persistence
        try:
            with transaction.atomic():
                result = self._persist(validated_data)
                self._persisted = True
                self._result = result

        except Exception as e:
            logger.error(f"[BOUNDARY] Persistence error: {e}", exc_info=True)
            raise PersistenceError(
                operation=self.operation,
                model=self.model_name,
                original_error=e,
                correlation_id=self.context.correlation_id,
            )

        # Step 5: Emit persistence event
        self._emit_persisted(result)

        # Step 6: Write audit log
        self._write_audit_log("success", result)

        logger.info(f"[BOUNDARY] Successfully executed {self.operation} on {self.model_name}")

        return result

    # =========================================================================
    # Serializer Management
    # =========================================================================

    def _create_serializer(self, data: dict[str, Any]) -> serializers.Serializer:
        """Create serializer instance with proper context."""
        kwargs = {
            "data": data,
            "context": self._get_serializer_context(),
        }

        if self.instance is not None:
            kwargs["instance"] = self.instance

        if self.partial:
            kwargs["partial"] = True

        if self.many:
            kwargs["many"] = True

        return self.serializer_cls(**kwargs)

    def _get_serializer_context(self) -> dict[str, Any]:
        """Build serializer context with boundary information."""
        return {
            "tenant_id": self.context.tenant_id,
            "correlation_id": self.context.correlation_id,
            "user_id": self.context.user_id,
            "boundary_context": self.context,
            "operation": self.operation,
        }

    # =========================================================================
    # IO-001 Contract Validation (SRX-001 §6)
    # =========================================================================

    def _validate_io_contract(self, validated_data: dict[str, Any]) -> None:
        """
        Validate data against IO-001 contract.

        This method loads the IO contract by code and validates
        the data against its schema.

        Raises:
            IOContractViolationError: If contract validation fails
        """
        try:
            from syn.api.boundaries.registry import IOContractRegistry

            contract = IOContractRegistry.get(self.io_contract_code)
            if contract is None:
                # Contract not found - log warning but continue
                # This allows gradual migration
                logger.warning(
                    f"[BOUNDARY] IO contract '{self.io_contract_code}' not found, skipping contract validation"
                )
                return

            # Validate against contract
            violations = contract.validate(validated_data)
            if violations:
                raise IOContractViolationError(
                    contract_code=self.io_contract_code,
                    violations=violations,
                    contract_version=getattr(contract, "version", None),
                    correlation_id=self.context.correlation_id,
                )

        except IOContractViolationError:
            raise
        except ImportError:
            # Registry not available - skip contract validation
            logger.debug("[BOUNDARY] IOContractRegistry not available, skipping")
        except Exception as e:
            logger.warning(f"[BOUNDARY] IO contract validation error: {e}")
            # Don't fail on contract validation infrastructure issues

    # =========================================================================
    # Governance Integration (SRX-001 §8)
    # =========================================================================

    def _execute_governance_preflight(self, validated_data: dict[str, Any]) -> dict[str, Any]:
        """
        Execute governance preflight checks per CGS-1001.

        Emits io.governance_preflight event before evaluation.

        Returns:
            Judgment dictionary with 'decision' (ALLOW/BLOCK/ESCALATE)
            and optional 'reason', 'rule_id', 'confidence'
        """
        # Prepare governance context
        gov_context = {
            "tenant_id": str(self.context.tenant_id),
            "correlation_id": str(self.context.correlation_id),
            "user_id": str(self.context.user_id) if self.context.user_id else None,
            "action": f"io.{self.operation}",
            "model": self.model_name,
            "contract": self.io_contract_code,
            "payload": validated_data,
        }

        # Emit preflight event
        # EVT-001: domain.entity.action naming pattern
        self._emit_event("io.governance.preflight", gov_context)

        # Evaluate governance rules
        try:
            from syn.governance import evaluate_rules

            judgment = evaluate_rules(
                ruleset=self.GOVERNANCE_RULESET,
                context=gov_context,
                tenant_id=self.context.tenant_id,
            )

            return {
                "decision": judgment.get("decision", "ALLOW"),
                "reason": judgment.get("reason"),
                "rule_id": judgment.get("rule_id"),
                "confidence": judgment.get("confidence"),
            }

        except ImportError:
            # Governance not available - default to ALLOW
            logger.debug("[BOUNDARY] Governance engine not available, defaulting to ALLOW")
            return {"decision": "ALLOW"}
        except Exception as e:
            logger.warning(f"[BOUNDARY] Governance evaluation error: {e}")
            # Default to ALLOW on errors (fail-open for now)
            return {"decision": "ALLOW"}

    def _handle_governance_block(self, judgment: dict[str, Any]) -> None:
        """Handle BLOCK governance decision."""
        # Emit blocked event
        # EVT-001: domain.entity.action naming pattern
        self._emit_event(
            "io.governance.blocked",
            {
                **self.context.to_event_payload(),
                "model": self.model_name,
                "operation": self.operation,
                "contract": self.io_contract_code,
                "judgment": judgment,
            },
        )

        # Write audit log
        self._write_audit_log("blocked", None, judgment)

        raise GovernanceBlockedError(
            reason=judgment.get("reason", "Operation blocked by governance"),
            rule_id=judgment.get("rule_id"),
            confidence=judgment.get("confidence"),
            correlation_id=self.context.correlation_id,
        )

    def _handle_governance_escalate(self, judgment: dict[str, Any]) -> None:
        """Handle ESCALATE governance decision."""
        # Emit escalated event
        # EVT-001: domain.entity.action naming pattern
        self._emit_event(
            "io.governance.escalated",
            {
                **self.context.to_event_payload(),
                "model": self.model_name,
                "operation": self.operation,
                "contract": self.io_contract_code,
                "judgment": judgment,
            },
        )

        # Write audit log
        self._write_audit_log("escalated", None, judgment)

        raise GovernanceEscalatedError(
            reason=judgment.get("reason", "Operation requires approval"),
            rule_id=judgment.get("rule_id"),
            escalation_id=judgment.get("escalation_id"),
            approvers=judgment.get("approvers"),
            correlation_id=self.context.correlation_id,
        )

    # =========================================================================
    # Persistence (SRX-001 §4.3)
    # =========================================================================

    def _persist(self, validated_data: dict[str, Any]) -> dict[str, Any]:
        """
        Persist validated data through serializer.

        Returns:
            Serialized representation of the persisted object
        """
        if self._serializer is None:
            raise BoundaryConfigurationError(
                "Serializer not initialized - call validate() first",
                correlation_id=self.context.correlation_id,
            )

        # Add tenant_id to validated_data if model supports it
        if hasattr(self.serializer_cls, "Meta") and hasattr(self.serializer_cls.Meta, "model"):
            model = self.serializer_cls.Meta.model
            if hasattr(model, "tenant_id"):
                self._serializer.validated_data["tenant_id"] = str(self.context.tenant_id)

        # Perform save
        instance = self._serializer.save()

        # Return serialized representation
        output_serializer = self.serializer_cls(
            instance,
            context=self._get_serializer_context(),
        )
        return output_serializer.data

    # =========================================================================
    # Event Emission (SRX-001 §7)
    # =========================================================================

    def _emit_event(self, event_name: str, payload: dict[str, Any]) -> None:
        """Emit an event through the event system."""
        try:
            from syn.engine import Cortex

            # Ensure required fields
            payload["correlation_id"] = str(self.context.correlation_id)
            payload["tenant_id"] = str(self.context.tenant_id)
            payload["timestamp"] = timezone.now().isoformat()

            Cortex.publish(event_name, payload)
            self._events_emitted.append(event_name)

            logger.debug(f"[BOUNDARY] Emitted event: {event_name}")

        except ImportError:
            logger.debug(f"[BOUNDARY] Event system not available, skipping: {event_name}")
        except Exception as e:
            logger.warning(f"[BOUNDARY] Event emission error: {e}")

    def _emit_validated(self) -> None:
        """Emit io.validated event."""
        self._emit_event(
            "io.validated",
            {
                **self.context.to_event_payload(),
                "contract": self.io_contract_code,
                "model": self.model_name,
                "operation": self.operation,
            },
        )

    def _emit_validation_failed(self, error: BoundaryValidationError) -> None:
        """Emit io.validation.failed event (EVT-001 domain.entity.action pattern)."""
        self._emit_event(
            "io.validation.failed",
            {
                **self.context.to_event_payload(),
                "contract": self.io_contract_code,
                "model": self.model_name,
                "operation": self.operation,
                "errors": error.to_dict(),
            },
        )

    def _emit_persisted(self, result: dict[str, Any]) -> None:
        """Emit io.persisted.* event based on operation."""
        event_name = f"io.persisted.{self.operation}"

        # Get primary key from result
        primary_key = result.get("id") or result.get("pk") or result.get("uuid")

        self._emit_event(
            event_name,
            {
                **self.context.to_event_payload(),
                "contract": self.io_contract_code,
                "model": self.model_name,
                "operation": self.operation,
                "primary_key": str(primary_key) if primary_key else None,
            },
        )

    # =========================================================================
    # Audit Logging (SRX-001 §8.3)
    # =========================================================================

    def _write_audit_log(
        self,
        outcome: str,
        result: dict[str, Any] | None,
        judgment: dict[str, Any] | None = None,
    ) -> None:
        """Write AUD-001 compliant audit entry."""
        try:
            from syn.audit.models import SysLogEntry

            # Get primary key if available
            resource_id = None
            if result:
                resource_id = result.get("id") or result.get("pk") or result.get("uuid")

            # Build audit payload
            payload = {
                "operation": self.operation,
                "contract": self.io_contract_code,
                "model": self.model_name,
                "resource_id": str(resource_id) if resource_id else None,
                "outcome": outcome,
                "validation_outcome": "success" if self._validated_data else "failure",
            }

            if judgment:
                payload["governance_judgment"] = judgment.get("decision")
                payload["governance_reason"] = judgment.get("reason")
                payload["governance_rule_id"] = judgment.get("rule_id")

            # Calculate duration
            duration_ms = (timezone.now() - self._start_time).total_seconds() * 1000
            payload["duration_ms"] = duration_ms

            # Create audit entry
            SysLogEntry.objects.create(
                tenant_id=(str(self.context.tenant_id) if self.context.tenant_id else None),
                correlation_id=self.context.correlation_id,
                event_name=f"io.boundary.{self.operation}",
                actor=str(self.context.user_id) if self.context.user_id else "system",
                payload={
                    **payload,
                    "boundary_action": f"Boundary {self.operation} on {self.model_name}",
                },
            )

            logger.debug(f"[BOUNDARY] Wrote audit log for {self.operation}")

        except ImportError:
            logger.debug("[BOUNDARY] Audit system not available")
        except Exception as e:
            logger.warning(f"[BOUNDARY] Audit logging error: {e}")

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def for_create(
        cls,
        serializer_cls: type[serializers.Serializer],
        io_contract_code: str,
        context: BoundaryContext,
    ) -> SerializerBoundary:
        """Create a boundary for create operations."""
        return cls(
            serializer_cls=serializer_cls,
            io_contract_code=io_contract_code,
            operation=cls.OPERATION_CREATE,
            context=context,
        )

    @classmethod
    def for_update(
        cls,
        serializer_cls: type[serializers.Serializer],
        io_contract_code: str,
        context: BoundaryContext,
        instance: Any,
        partial: bool = False,
    ) -> SerializerBoundary:
        """Create a boundary for update operations."""
        return cls(
            serializer_cls=serializer_cls,
            io_contract_code=io_contract_code,
            operation=cls.OPERATION_UPDATE,
            context=context,
            instance=instance,
            partial=partial,
        )

    @classmethod
    def for_delete(
        cls,
        serializer_cls: type[serializers.Serializer],
        io_contract_code: str,
        context: BoundaryContext,
        instance: Any,
    ) -> SerializerBoundary:
        """Create a boundary for delete operations."""
        return cls(
            serializer_cls=serializer_cls,
            io_contract_code=io_contract_code,
            operation=cls.OPERATION_DELETE,
            context=context,
            instance=instance,
        )
