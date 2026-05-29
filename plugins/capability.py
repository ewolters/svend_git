"""Capability Study Plugin — wraps existing SPC capability engine.

Inputs: raw data + spec limits.
Outputs: Cpk/Ppk metrics (PCL-writable), charts, summary text.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator

from syn.plugins.base import Plugin, PluginOutput


class CapabilityInput(BaseModel):
    """Input schema for capability study."""

    data: List[float]
    usl: Optional[float] = None
    lsl: Optional[float] = None
    target: Optional[float] = None
    measurement: str = "measurement"

    @field_validator("data")
    @classmethod
    def data_not_empty(cls, v):
        if len(v) < 2:
            raise ValueError("Need at least 2 data points")
        return v

    @field_validator("usl")
    @classmethod
    def usl_gt_lsl(cls, v, info):
        lsl = info.data.get("lsl")
        if v is not None and lsl is not None and v <= lsl:
            raise ValueError("USL must be greater than LSL")
        return v


class CapabilityStudyPlugin(Plugin):
    """Process capability analysis: Cp, Cpk, Pp, Ppk with histogram and Q-Q plot."""

    name = "capability_study"
    version = "1.0.0"
    description = "Process capability analysis with Cp, Cpk, Pp, Ppk indices"
    input_schema = CapabilityInput

    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        import pandas as pd

        from agents_api.analysis.spc.capability import run_capability

        # Build DataFrame in format expected by run_capability
        measurement = validated_input.get("measurement", "measurement")
        df = pd.DataFrame({measurement: validated_input["data"]})

        config = {
            "measurement": measurement,
            "usl": validated_input.get("usl"),
            "lsl": validated_input.get("lsl"),
            "target": validated_input.get("target"),
        }

        result = run_capability(df, config)

        # Convert to PluginOutputs
        outputs: List[PluginOutput] = []
        stats = result.get("statistics", {})

        # Metric outputs (PCL-writable)
        if stats.get("cp") is not None:
            outputs.append(PluginOutput("cp", "metric", stats["cp"], measure_slug="cp"))
        if stats.get("cpk") is not None:
            outputs.append(PluginOutput("cpk", "metric", stats["cpk"], measure_slug="cpk"))
        if stats.get("pp") is not None:
            outputs.append(PluginOutput("pp", "metric", stats["pp"], measure_slug="pp"))
        if stats.get("ppk") is not None:
            outputs.append(PluginOutput("ppk", "metric", stats["ppk"], measure_slug="ppk"))
        if stats.get("sigma_level") is not None:
            outputs.append(PluginOutput("sigma_level", "metric", stats["sigma_level"]))
        if stats.get("yield_pct") is not None:
            outputs.append(PluginOutput("yield_pct", "metric", stats["yield_pct"]))
        if stats.get("ppm_total") is not None:
            outputs.append(PluginOutput("ppm_total", "metric", stats["ppm_total"]))

        # Chart outputs
        for plot in result.get("plots", []):
            key = plot["title"].lower().replace(" ", "_").replace("(", "").replace(")", "")
            outputs.append(PluginOutput(key, "chart", plot))

        # Text summary
        if result.get("summary"):
            outputs.append(PluginOutput("summary", "text", {"text": result["summary"]}))

        # Narrative (if generated)
        if result.get("narrative"):
            outputs.append(PluginOutput("narrative", "text", result["narrative"]))

        return outputs
