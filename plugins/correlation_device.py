"""Correlation Plugin — variable relationship analysis."""

from typing import Any, Dict, List

from pydantic import BaseModel, field_validator

from syn.plugins.base import Plugin, PluginOutput


class CorrelationInput(BaseModel):
    data: Dict[str, List[float]]  # {"var_a": [1,2,3], "var_b": [4,5,6]}
    method: str = "pearson"  # pearson, spearman, kendall

    @field_validator("data")
    @classmethod
    def at_least_two_vars(cls, v):
        if len(v) < 2:
            raise ValueError("Need at least 2 variables")
        lengths = [len(vals) for vals in v.values()]
        if len(set(lengths)) > 1:
            raise ValueError("All variables must have the same length")
        if lengths[0] < 3:
            raise ValueError("Need at least 3 observations per variable")
        return v

    @field_validator("method")
    @classmethod
    def valid_method(cls, v):
        if v not in {"pearson", "spearman", "kendall"}:
            raise ValueError("method must be pearson, spearman, or kendall")
        return v


def _clean_value(v: Any) -> Any:
    """Convert numpy scalars/arrays and complex objects to JSON-serializable Python types."""
    if hasattr(v, "item"):  # numpy scalar
        return v.item()
    if hasattr(v, "tolist"):  # numpy array
        return v.tolist()
    if isinstance(v, dict):
        return {k: _clean_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean_value(i) for i in v]
    if hasattr(v, "__dataclass_fields__") or (hasattr(v, "__dict__") and not isinstance(v, type)):
        try:
            return {k: _clean_value(val) for k, val in vars(v).items()}
        except TypeError:
            return str(v)
    return v


def _result_to_clean_dict(result: Any) -> Dict[str, Any]:
    """Extract and clean all fields from a forgestat result object."""
    raw = result.__dict__ if hasattr(result, "__dict__") else vars(result)
    return {k: _clean_value(v) for k, v in raw.items()}


class CorrelationPlugin(Plugin):
    name = "correlation"
    version = "1.0.0"
    description = "Correlation analysis — Pearson, Spearman, or Kendall"
    input_schema = CorrelationInput

    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        from forgestat.parametric.correlation import correlation

        result = correlation(
            data=validated_input["data"],
            method=validated_input.get("method", "pearson"),
        )

        clean = _result_to_clean_dict(result)

        outputs = []

        # Extract pairwise correlations
        pairs = clean.get("pairs", [])
        if pairs:
            first = pairs[0] if isinstance(pairs[0], dict) else pairs[0].__dict__
            r_val = first.get("r", 0)
            p_val = first.get("p_value", 0)
            if hasattr(r_val, "item"):
                r_val = r_val.item()
            if hasattr(p_val, "item"):
                p_val = p_val.item()
            outputs.append(PluginOutput("r", "metric", r_val))
            outputs.append(PluginOutput("p_value", "metric", p_val))

        outputs.append(PluginOutput("result", "text", clean))
        return outputs
