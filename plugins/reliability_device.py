"""Reliability Plugin — MTBF, availability, failure rate analysis."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator

from syn.plugins.base import Plugin, PluginOutput


class ReliabilityInput(BaseModel):
    failure_times: List[float]
    repair_times: Optional[List[float]] = None
    total_operating_time: Optional[float] = None
    confidence: float = 0.90

    @field_validator("failure_times")
    @classmethod
    def not_empty(cls, v):
        if len(v) < 1:
            raise ValueError("Need at least 1 failure time")
        return v


class ReliabilityPlugin(Plugin):
    name = "reliability"
    version = "1.0.0"
    description = "Reliability analysis — MTBF, availability, failure rate"
    input_schema = ReliabilityInput

    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        from forgerel import mtbf as mtbf_module

        result = mtbf_module.mtbf_analysis(
            failure_times=validated_input["failure_times"],
            repair_times=validated_input.get("repair_times"),
            total_operating_time=validated_input.get("total_operating_time"),
            confidence=validated_input.get("confidence", 0.90),
        )

        result_dict = result.__dict__ if hasattr(result, "__dict__") else vars(result)
        clean = {}
        for k, v in result_dict.items():
            if hasattr(v, "item"):
                clean[k] = v.item()
            elif hasattr(v, "tolist"):
                clean[k] = v.tolist()
            else:
                clean[k] = v

        outputs = [
            PluginOutput("mtbf", "metric", clean.get("mtbf", 0.0)),
            PluginOutput("availability", "metric", clean.get("availability", 0.0)),
            PluginOutput("failure_rate", "metric", clean.get("failure_rate", 0.0)),
        ]

        if "mttr" in clean and clean["mttr"] is not None:
            outputs.append(PluginOutput("mttr", "metric", clean["mttr"]))

        outputs.append(PluginOutput("result", "text", clean))

        return outputs
