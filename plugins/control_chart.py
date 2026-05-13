"""Control Chart Plugin — wraps forgespc I-MR/Xbar-R chart engine.

Inputs: raw data + optional spec limits + chart config.
Outputs: control chart, statistics (mean, UCL, LCL), violations list, narrative.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator

from syn.plugins.base import Plugin, PluginOutput


class ControlChartInput(BaseModel):
    data: List[float]
    usl: Optional[float] = None
    lsl: Optional[float] = None
    subgroup_size: int = 1

    @field_validator("data")
    @classmethod
    def data_not_empty(cls, v):
        if len(v) < 5:
            raise ValueError("Need at least 5 data points for control chart")
        return v


class ControlChartPlugin(Plugin):
    name = "control_chart"
    version = "1.0.0"
    description = "SPC control chart (I-MR) with Western Electric rules"
    input_schema = ControlChartInput

    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        from forgespc.charts import individuals_moving_range_chart

        result = individuals_moving_range_chart(
            data=validated_input["data"],
            usl=validated_input.get("usl"),
            lsl=validated_input.get("lsl"),
        )

        outputs = []

        # Metrics — ControlLimits uses .cl (not .center_line)
        if hasattr(result, "limits") and result.limits:
            outputs.append(PluginOutput("mean", "metric", result.limits.cl))
            outputs.append(PluginOutput("ucl", "metric", result.limits.ucl))
            outputs.append(PluginOutput("lcl", "metric", result.limits.lcl))

        if hasattr(result, "out_of_control") and result.out_of_control:
            outputs.append(PluginOutput("ooc_count", "metric", len(result.out_of_control)))
        else:
            outputs.append(PluginOutput("ooc_count", "metric", 0))

        # Chart
        outputs.append(PluginOutput("chart", "chart", result.to_dict()))

        # Summary text
        outputs.append(
            PluginOutput(
                "summary",
                "text",
                {
                    "in_control": result.in_control,
                    "ooc_points": [str(v) for v in (result.out_of_control or [])],
                    "run_violations": [str(v) for v in (result.run_violations or [])],
                },
            )
        )

        return outputs
