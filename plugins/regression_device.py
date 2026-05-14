"""Regression Plugin — linear and polynomial regression."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator

from syn.plugins.base import Plugin, PluginOutput


class RegressionInput(BaseModel):
    x: List[List[float]]  # feature matrix (each inner list is a row)
    y: List[float]
    feature_names: Optional[List[str]] = None
    degree: int = 1  # 1=linear, 2+=polynomial (single feature only)
    alpha: float = 0.05

    @field_validator("y")
    @classmethod
    def y_not_empty(cls, v):
        if len(v) < 3:
            raise ValueError("Need at least 3 observations")
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


class RegressionPlugin(Plugin):
    name = "regression"
    version = "1.0.0"
    description = "Linear and polynomial regression with diagnostics"
    input_schema = RegressionInput

    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        x = validated_input["x"]
        y = validated_input["y"]
        alpha = validated_input.get("alpha", 0.05)
        degree = validated_input.get("degree", 1)

        if degree > 1:
            # Polynomial — single feature, extract from x
            from forgestat.regression.linear import polynomial

            x_flat = [row[0] for row in x]
            result = polynomial(x_flat, y, degree=degree, alpha=alpha)
        else:
            from forgestat.regression.linear import ols

            result = ols(x, y, feature_names=validated_input.get("feature_names"), alpha=alpha)

        clean = _result_to_clean_dict(result)

        outputs = [
            PluginOutput("r_squared", "metric", clean.get("r_squared", 0.0)),
            PluginOutput("adj_r_squared", "metric", clean.get("adj_r_squared", 0.0)),
        ]

        if "f_p_value" in clean:
            outputs.append(PluginOutput("f_p_value", "metric", clean["f_p_value"]))

        outputs.append(PluginOutput("result", "text", clean))

        return outputs
