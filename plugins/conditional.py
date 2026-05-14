"""Conditional (Diamond) Plugin — route based on a predicate.

Evaluates: value <operator> threshold
Outputs: result (bool), pass_value (value if true, else None),
         fail_value (value if false, else None), summary (text).
"""

import operator as op
from typing import Any, Dict, List

from pydantic import BaseModel, field_validator

from syn.plugins.base import Plugin, PluginOutput

OPERATORS = {
    ">": op.gt,
    ">=": op.ge,
    "<": op.lt,
    "<=": op.le,
    "==": op.eq,
    "!=": op.ne,
}


class ConditionalInput(BaseModel):
    value: float
    operator: str
    threshold: float
    label: str = ""

    @field_validator("operator")
    @classmethod
    def operator_valid(cls, v):
        if v not in OPERATORS:
            raise ValueError(f"Invalid operator '{v}'. Must be one of: {', '.join(OPERATORS)}")
        return v


class ConditionalPlugin(Plugin):
    """Decision node — evaluates predicate, routes to pass or fail branch."""

    name = "conditional"
    version = "1.0.0"
    description = "Conditional gate — routes value to pass/fail branch based on threshold"
    input_schema = ConditionalInput

    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        value = validated_input["value"]
        threshold = validated_input["threshold"]
        op_str = validated_input["operator"]
        label = validated_input.get("label", "")
        op_fn = OPERATORS[op_str]

        passed = op_fn(value, threshold)

        return [
            PluginOutput("result", "metric", passed),
            PluginOutput("pass_value", "metric", value if passed else None),
            PluginOutput("fail_value", "metric", value if not passed else None),
            PluginOutput(
                "summary",
                "text",
                {
                    "text": f"{'PASS' if passed else 'FAIL'}: {value} {op_str} {threshold}",
                    "label": label,
                    "passed": passed,
                    "value": value,
                    "operator": op_str,
                    "threshold": threshold,
                },
            ),
        ]
