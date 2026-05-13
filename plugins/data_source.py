"""Data Source Plugin — entry point for all flowcharts.

Takes raw measurement data + spec limits from the user and outputs them
as separate typed ports for downstream devices to consume.

No computation — this is a structuring device. It exists so the flowchart
has a named entry point with typed output ports that the renderer can draw
and the engine can route.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator

from syn.plugins.base import Plugin, PluginOutput


class DataSourceInput(BaseModel):
    """Input schema for data source device."""

    measurements: List[float]
    lsl: Optional[float] = None
    usl: Optional[float] = None
    problem_statement: str = ""

    @field_validator("measurements")
    @classmethod
    def measurements_not_empty(cls, v):
        if len(v) < 1:
            raise ValueError("Need at least 1 data point")
        return v

    @field_validator("usl")
    @classmethod
    def usl_gt_lsl(cls, v, info):
        lsl = info.data.get("lsl")
        if v is not None and lsl is not None and v <= lsl:
            raise ValueError("USL must be greater than LSL")
        return v


class DataSourcePlugin(Plugin):
    """Flowchart entry point — structures user data into typed output ports."""

    name = "data_source"
    version = "1.0.0"
    description = "Data entry point — outputs measurements, spec limits, and problem statement"
    input_schema = DataSourceInput

    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        outputs = []

        # Data port — measurements as list (downstream capability/control chart expects this)
        outputs.append(PluginOutput("measurements", "dataset", validated_input["measurements"]))

        # Spec ports — individual values for downstream routing
        if validated_input.get("usl") is not None:
            outputs.append(PluginOutput("usl", "metric", validated_input["usl"]))

        if validated_input.get("lsl") is not None:
            outputs.append(PluginOutput("lsl", "metric", validated_input["lsl"]))

        # Problem statement — for fishbone/RCA devices
        if validated_input.get("problem_statement"):
            outputs.append(PluginOutput("problem_statement", "text", {"text": validated_input["problem_statement"]}))

        return outputs
