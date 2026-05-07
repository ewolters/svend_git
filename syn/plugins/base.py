"""
Plugin ABC — the execution contract for all SVEND tools.

Every plugin declares:
- name: unique identifier (used in registry, Job records, canvas bindings)
- input_schema: Pydantic model for what it accepts
- execute(): computation -> list of PluginOutput

Design:
- Sync only (Django is sync, plugins are user-triggered)
- No retry logic (user re-runs on failure)
- No telemetry infra (Job IS the telemetry)
- Schema introspection via get_metadata() for canvas UI generation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel


@dataclass
class PluginOutput:
    """A named output from a plugin execution.

    Attributes:
        key: Output identifier (e.g. "cpk", "histogram", "summary").
        output_type: One of metric, chart, table, text, dataset.
        value: The payload — float for metrics, dict for everything else.
        provenance: observed | calculated | simulated | projected.
        measure_slug: If this output should write to PCL, which measure.
    """

    key: str
    output_type: str  # metric, chart, table, text, dataset
    value: Any
    provenance: str = "calculated"
    measure_slug: Optional[str] = None


class Plugin(ABC):
    """Abstract base for all SVEND plugins.

    Subclasses MUST define:
        - name (str): unique plugin identifier
        - description (str): human-readable purpose
        - input_schema (Pydantic BaseModel subclass): what the plugin accepts
        - execute(): the computation

    Optional:
        - version (str): defaults to "1.0.0"
        - output_schema: for documentation/introspection only
    """

    name: str = None
    version: str = "1.0.0"
    description: str = None
    input_schema: Type[BaseModel] = None
    output_schema: Optional[Type[BaseModel]] = None

    def __init__(self):
        if not self.name:
            raise TypeError(f"{self.__class__.__name__} must define 'name'")
        if not self.description:
            raise TypeError(f"{self.__class__.__name__} must define 'description'")

    @abstractmethod
    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        """Run the plugin computation.

        Args:
            validated_input: Dict from input_schema.model_dump() — already validated.
            context: Execution context (job_id, actor, tenant_id).

        Returns:
            List of PluginOutput objects.
        """
        ...

    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """Introspectable metadata for registry, canvas UI, documentation."""
        return {
            "name": cls.name,
            "version": cls.version,
            "description": cls.description,
            "input_schema": cls.input_schema.model_json_schema() if cls.input_schema else None,
            "output_schema": cls.output_schema.model_json_schema() if cls.output_schema else None,
        }

    def __repr__(self):
        return f"<Plugin: {self.name} v{self.version}>"
