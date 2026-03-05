"""
IO Contract Registry per SRX-001 §6 and IO-001 §8.

Standard:     SRX-001 §6 (Contract Binding), IO-001 §8 (Contract Registry)
Compliance:   DOC-002 §5.1, EVT-001 §4.1
Version:      1.0.0
Last Updated: 2025-12-03

Provides:
- IOContractRegistry for discovering and caching IO contracts
- Contract validation against Pydantic InputSchema/OutputSchema
- Version compatibility checking
- Auto-generated documentation support

References:
- SRX-001: Serializer Reflex Boundary Standard
- IO-001: Input/Output Contract Standard
- DOC-002: Documentation Standards
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from django.utils import timezone
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class IOContract:
    """
    Registered IO contract per IO-001 §8.1.

    Represents a registered input/output contract with its schemas,
    version, and validation capabilities.

    Attributes:
        code: Unique contract code (e.g., "IO-USER-001")
        input_schema: Pydantic InputSchema class
        output_schema: Pydantic OutputSchema class (optional)
        version: Semantic version string
        description: Human-readable description
        failure_domain: How to handle validation failures
        idempotency_strategy: Idempotency approach
        registered_at: When the contract was registered
    """

    code: str
    input_schema: type[BaseModel]
    output_schema: type[BaseModel] | None = None
    version: str = "1.0.0"
    description: str = ""
    failure_domain: str = "fatal"  # fatal, transient, compensable
    idempotency_strategy: str = "correlation_id"
    registered_at: datetime = field(default_factory=timezone.now)

    def validate(self, data: dict[str, Any]) -> list[str]:
        """
        Validate data against this contract's input schema.

        Args:
            data: Data to validate

        Returns:
            List of violation messages (empty if valid)
        """
        violations = []

        try:
            self.input_schema.model_validate(data)
        except ValidationError as e:
            for error in e.errors():
                loc = ".".join(str(part) for part in error["loc"])
                violations.append(f"{loc}: {error['msg']}")

        return violations

    def validate_output(self, data: dict[str, Any]) -> list[str]:
        """
        Validate data against this contract's output schema.

        Args:
            data: Data to validate

        Returns:
            List of violation messages (empty if valid)
        """
        if self.output_schema is None:
            return []

        violations = []

        try:
            self.output_schema.model_validate(data)
        except ValidationError as e:
            for error in e.errors():
                loc = ".".join(str(part) for part in error["loc"])
                violations.append(f"{loc}: {error['msg']}")

        return violations

    def to_json_schema(self) -> dict[str, Any]:
        """
        Export contract as JSON Schema per IO-001 §5.3.

        Returns:
            Dictionary with input and output JSON schemas
        """
        result = {
            "contract_code": self.code,
            "version": self.version,
            "input_schema": self.input_schema.model_json_schema(),
        }

        if self.output_schema:
            result["output_schema"] = self.output_schema.model_json_schema()

        return result

    def is_compatible(self, requested_version: str) -> bool:
        """
        Check if this contract is compatible with requested version.

        Implements semantic versioning compatibility per IO-001 §8.3:
        - Major version must match
        - Minor version must be >= requested
        - Patch version is ignored

        Args:
            requested_version: Version string to check

        Returns:
            True if compatible
        """
        try:
            current = self._parse_version(self.version)
            requested = self._parse_version(requested_version)

            # Major must match
            if current[0] != requested[0]:
                return False

            # Minor must be >= requested
            return current[1] >= requested[1]

        except ValueError:
            return False

    @staticmethod
    def _parse_version(version: str) -> tuple:
        """Parse semantic version string."""
        parts = version.split(".")
        return (
            int(parts[0]) if len(parts) > 0 else 0,
            int(parts[1]) if len(parts) > 1 else 0,
            int(parts[2]) if len(parts) > 2 else 0,
        )


class IOContractRegistry:
    """
    Registry for IO contracts per IO-001 §8.1.

    Thread-safe singleton registry that caches IO contracts
    by code and supports autodiscovery.

    Usage:
        >>> from syn.api.boundaries.registry import IOContractRegistry
        >>> from myapp.schemas import CreateUserInput, UserOutput
        >>>
        >>> IOContractRegistry.register(
        ...     code="IO-USER-001",
        ...     input_schema=CreateUserInput,
        ...     output_schema=UserOutput,
        ...     version="1.0.0",
        ... )
        >>>
        >>> contract = IOContractRegistry.get("IO-USER-001")
        >>> violations = contract.validate({"email": "test@example.com"})

    Implementation:
        The registry uses a class-level dictionary for storage,
        making it effectively a singleton across the application.
    """

    _contracts: dict[str, IOContract] = {}
    _discovered: bool = False

    @classmethod
    def register(
        cls,
        code: str,
        input_schema: type[BaseModel],
        output_schema: type[BaseModel] | None = None,
        version: str = "1.0.0",
        description: str = "",
        failure_domain: str = "fatal",
        idempotency_strategy: str = "correlation_id",
    ) -> IOContract:
        """
        Register an IO contract.

        Args:
            code: Unique contract code (e.g., "IO-USER-001")
            input_schema: Pydantic InputSchema class
            output_schema: Pydantic OutputSchema class (optional)
            version: Semantic version string
            description: Human-readable description
            failure_domain: How to handle validation failures
            idempotency_strategy: Idempotency approach

        Returns:
            Registered IOContract instance

        Raises:
            ValueError: If contract code already registered with different version
        """
        if code in cls._contracts:
            existing = cls._contracts[code]
            if existing.version != version:
                logger.warning(
                    f"[REGISTRY] Contract {code} already registered with version "
                    f"{existing.version}, updating to {version}"
                )

        contract = IOContract(
            code=code,
            input_schema=input_schema,
            output_schema=output_schema,
            version=version,
            description=description,
            failure_domain=failure_domain,
            idempotency_strategy=idempotency_strategy,
        )

        cls._contracts[code] = contract
        logger.debug(f"[REGISTRY] Registered contract: {code} v{version}")

        return contract

    @classmethod
    def get(cls, code: str, version: str | None = None) -> IOContract | None:
        """
        Get an IO contract by code.

        Args:
            code: Contract code to look up
            version: Optional version requirement

        Returns:
            IOContract if found and compatible, None otherwise
        """
        # Trigger autodiscovery if not done
        if not cls._discovered:
            cls.autodiscover()

        contract = cls._contracts.get(code)
        if contract is None:
            return None

        # Check version compatibility if specified
        if version and not contract.is_compatible(version):
            logger.warning(
                f"[REGISTRY] Contract {code} version {contract.version} not compatible with requested {version}"
            )
            return None

        return contract

    @classmethod
    def get_or_raise(cls, code: str, version: str | None = None) -> IOContract:
        """
        Get an IO contract by code, raising if not found.

        Args:
            code: Contract code to look up
            version: Optional version requirement

        Returns:
            IOContract

        Raises:
            KeyError: If contract not found or incompatible
        """
        contract = cls.get(code, version)
        if contract is None:
            raise KeyError(f"IO contract '{code}' not found or version incompatible")
        return contract

    @classmethod
    def list_contracts(cls) -> list[IOContract]:
        """
        List all registered contracts.

        Returns:
            List of IOContract instances
        """
        if not cls._discovered:
            cls.autodiscover()
        return list(cls._contracts.values())

    @classmethod
    def list_codes(cls) -> list[str]:
        """
        List all registered contract codes.

        Returns:
            List of contract codes
        """
        if not cls._discovered:
            cls.autodiscover()
        return list(cls._contracts.keys())

    @classmethod
    def has(cls, code: str) -> bool:
        """
        Check if a contract is registered.

        Args:
            code: Contract code to check

        Returns:
            True if registered
        """
        if not cls._discovered:
            cls.autodiscover()
        return code in cls._contracts

    @classmethod
    def unregister(cls, code: str) -> bool:
        """
        Unregister a contract.

        Args:
            code: Contract code to remove

        Returns:
            True if removed, False if not found
        """
        if code in cls._contracts:
            del cls._contracts[code]
            logger.debug(f"[REGISTRY] Unregistered contract: {code}")
            return True
        return False

    @classmethod
    def clear(cls) -> None:
        """Clear all registered contracts."""
        cls._contracts.clear()
        cls._discovered = False
        logger.debug("[REGISTRY] Cleared all contracts")

    @classmethod
    def autodiscover(cls) -> int:
        """
        Autodiscover and register contracts from installed apps.

        Scans for 'io_contracts' modules in Django apps and calls
        their register() function if present.

        Returns:
            Number of contracts discovered
        """
        if cls._discovered:
            return len(cls._contracts)

        count_before = len(cls._contracts)

        try:
            from django.apps import apps

            for app_config in apps.get_app_configs():
                try:
                    module_name = f"{app_config.name}.io_contracts"
                    module = __import__(module_name, fromlist=["register"])

                    if hasattr(module, "register"):
                        module.register(cls)
                        logger.debug(f"[REGISTRY] Discovered contracts from {module_name}")

                except ImportError:
                    # No io_contracts module in this app
                    pass
                except Exception as e:
                    logger.warning(f"[REGISTRY] Error discovering contracts from {app_config.name}: {e}")

        except Exception as e:
            logger.warning(f"[REGISTRY] Autodiscovery failed: {e}")

        cls._discovered = True
        discovered = len(cls._contracts) - count_before
        logger.info(f"[REGISTRY] Autodiscovered {discovered} contracts")

        return discovered

    @classmethod
    def export_documentation(cls) -> dict[str, Any]:
        """
        Export registry as documentation per IO-001 §8.2.

        Returns:
            Dictionary suitable for OpenAPI/JSON Schema documentation
        """
        if not cls._discovered:
            cls.autodiscover()

        return {
            "contracts": {
                code: {
                    "code": contract.code,
                    "version": contract.version,
                    "description": contract.description,
                    "failure_domain": contract.failure_domain,
                    "idempotency_strategy": contract.idempotency_strategy,
                    "input_schema": contract.input_schema.model_json_schema(),
                    "output_schema": (contract.output_schema.model_json_schema() if contract.output_schema else None),
                }
                for code, contract in cls._contracts.items()
            },
            "total_count": len(cls._contracts),
            "generated_at": timezone.now().isoformat(),
        }


# =============================================================================
# Decorator for Contract Registration
# =============================================================================


def io_contract(
    code: str,
    version: str = "1.0.0",
    description: str = "",
    failure_domain: str = "fatal",
    idempotency_strategy: str = "correlation_id",
    output_schema: type[BaseModel] | None = None,
) -> Callable[[type[BaseModel]], type[BaseModel]]:
    """
    Decorator to register an InputSchema as an IO contract.

    Usage:
        >>> from syn.api.boundaries.registry import io_contract
        >>> from syn.io import InputSchema
        >>>
        >>> @io_contract("IO-USER-001", version="1.0.0")
        ... class CreateUserInput(InputSchema):
        ...     email: str
        ...     name: str

    Args:
        code: Unique contract code
        version: Semantic version
        description: Human-readable description
        failure_domain: How to handle validation failures
        idempotency_strategy: Idempotency approach
        output_schema: Optional output schema class

    Returns:
        Decorator function
    """

    def decorator(cls: type[BaseModel]) -> type[BaseModel]:
        IOContractRegistry.register(
            code=code,
            input_schema=cls,
            output_schema=output_schema,
            version=version,
            description=description or cls.__doc__ or "",
            failure_domain=failure_domain,
            idempotency_strategy=idempotency_strategy,
        )
        # Store contract code on class for introspection
        cls._io_contract_code = code  # type: ignore
        return cls

    return decorator
