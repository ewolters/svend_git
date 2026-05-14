"""Savings Analysis Plugin — lean financial analysis.

Wraps forgefpa savings calculator (8 methods), box score, and throughput accounting.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator

from syn.plugins.base import Plugin, PluginOutput


class SavingsInput(BaseModel):
    analysis_type: str  # savings, box_score, throughput, target_cost
    # For savings:
    method: str = "waste_pct"
    baseline: float = 0.0
    actual: float = 0.0
    volume: float = 1.0
    cost_per_unit: float = 1.0
    # For box_score:
    box_score_data: Optional[Dict[str, Any]] = None
    # For throughput:
    revenue: float = 0.0
    truly_variable_costs: float = 0.0
    inventory: float = 0.0
    operating_expense: float = 0.0
    # For target_cost:
    selling_price: float = 0.0
    target_margin_pct: float = 0.0
    current_cost: float = 0.0

    @field_validator("analysis_type")
    @classmethod
    def valid_type(cls, v):
        valid = {"savings", "box_score", "throughput", "target_cost"}
        if v not in valid:
            raise ValueError(f"analysis_type must be one of: {', '.join(sorted(valid))}")
        return v


class SavingsPlugin(Plugin):
    name = "savings_analysis"
    version = "1.0.0"
    description = "Lean financial analysis — savings, box score, throughput accounting, target cost"
    input_schema = SavingsInput

    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        atype = validated_input["analysis_type"]

        if atype == "savings":
            from forgefpa import calculate_savings

            result = calculate_savings(
                method=validated_input["method"],
                baseline=validated_input["baseline"],
                actual=validated_input["actual"],
                volume=validated_input.get("volume", 1.0),
                cost_per_unit=validated_input.get("cost_per_unit", 1.0),
            )
            return [
                PluginOutput("savings", "metric", result.savings),
                PluginOutput("improvement_pct", "metric", result.improvement_pct),
                PluginOutput(
                    "result",
                    "text",
                    {
                        "method": result.method,
                        "savings": result.savings,
                        "improvement_pct": result.improvement_pct,
                        "details": result.details,
                    },
                ),
            ]

        elif atype == "throughput":
            from forgefpa import throughput_accounting

            result = throughput_accounting(
                revenue=validated_input["revenue"],
                truly_variable_costs=validated_input["truly_variable_costs"],
                inventory=validated_input["inventory"],
                operating_expense=validated_input["operating_expense"],
            )
            return [
                PluginOutput("throughput", "metric", result.throughput),
                PluginOutput("net_profit", "metric", result.net_profit),
                PluginOutput("roi", "metric", result.roi),
                PluginOutput("result", "text", result.to_dict()),
            ]

        elif atype == "target_cost":
            from forgefpa import target_cost

            result = target_cost(
                selling_price=validated_input["selling_price"],
                target_margin_pct=validated_input["target_margin_pct"],
                current_cost=validated_input["current_cost"],
            )
            return [
                PluginOutput("target_cost", "metric", result.target_cost),
                PluginOutput("cost_gap", "metric", result.cost_gap),
                PluginOutput("gap_pct", "metric", result.gap_pct),
                PluginOutput("result", "text", result.to_dict()),
            ]

        elif atype == "box_score":
            from forgefpa import build_box_score

            data = validated_input.get("box_score_data") or {}
            result = build_box_score(**data)
            return [
                PluginOutput(
                    "result",
                    "text",
                    {
                        "operational": result.operational.__dict__,
                        "capacity": result.capacity.__dict__,
                        "financial": result.financial.__dict__,
                    },
                ),
            ]

        raise ValueError(f"Unknown analysis_type: {atype}")
