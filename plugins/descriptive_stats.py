"""Descriptive Statistics Plugin — summary statistics for any dataset."""

from typing import Any, Dict, List

from pydantic import BaseModel, field_validator

from syn.plugins.base import Plugin, PluginOutput


class DescriptiveStatsInput(BaseModel):
    data: List[float]

    @field_validator("data")
    @classmethod
    def enough_data(cls, v):
        if len(v) < 2:
            raise ValueError("Need at least 2 data points")
        return v


class DescriptiveStatsPlugin(Plugin):
    name = "descriptive_stats"
    version = "1.0.0"
    description = "Summary statistics — mean, std, median, quartiles, shape"
    input_schema = DescriptiveStatsInput

    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        from forgestat.exploratory.univariate import describe

        result = describe(validated_input["data"])
        rd = result.__dict__ if hasattr(result, "__dict__") else vars(result)
        clean = {}
        for k, v in rd.items():
            if hasattr(v, "item"):
                clean[k] = v.item()
            elif hasattr(v, "tolist"):
                clean[k] = v.tolist()
            else:
                clean[k] = v

        outputs = [
            PluginOutput("mean", "metric", clean.get("mean", 0.0)),
            PluginOutput("std", "metric", clean.get("std", 0.0)),
            PluginOutput("median", "metric", clean.get("median", 0.0)),
            PluginOutput("n", "metric", float(clean.get("n", 0))),
        ]

        outputs.append(PluginOutput("result", "text", clean))
        return outputs
