"""Hypothesis Test Plugin — t-tests, ANOVA, chi-square.

Dispatches to the correct forgestat function based on test_type config.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator

from syn.plugins.base import Plugin, PluginOutput


class HypothesisTestInput(BaseModel):
    test_type: str  # one_sample_t, two_sample_t, paired_t, one_way_anova, chi_square
    data: List[float]
    data2: Optional[List[float]] = None  # for two_sample/paired
    groups: Optional[Dict[str, List[float]]] = None  # for anova
    observed: Optional[List[List[float]]] = None  # for chi_square
    mu: float = 0.0  # for one_sample_t
    alpha: float = 0.05

    @field_validator("test_type")
    @classmethod
    def valid_test_type(cls, v):
        valid = {"one_sample_t", "two_sample_t", "paired_t", "one_way_anova", "chi_square"}
        if v not in valid:
            raise ValueError(f"test_type must be one of: {', '.join(sorted(valid))}")
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
    # Dataclasses / named objects (e.g. AssumptionCheck) — convert to repr string
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


class HypothesisTestPlugin(Plugin):
    name = "hypothesis_test"
    version = "1.0.0"
    description = "Hypothesis testing — t-tests, ANOVA, chi-square"
    input_schema = HypothesisTestInput

    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        test_type = validated_input["test_type"]
        alpha = validated_input.get("alpha", 0.05)

        if test_type == "one_sample_t":
            from forgestat.parametric.ttest import one_sample

            result = one_sample(validated_input["data"], mu=validated_input.get("mu", 0.0), alpha=alpha)
        elif test_type == "two_sample_t":
            from forgestat.parametric.ttest import two_sample

            result = two_sample(validated_input["data"], validated_input["data2"], alpha=alpha)
        elif test_type == "paired_t":
            from forgestat.parametric.ttest import paired

            result = paired(validated_input["data"], validated_input["data2"], alpha=alpha)
        elif test_type == "one_way_anova":
            from forgestat.parametric.anova import one_way_from_dict

            result = one_way_from_dict(validated_input["groups"], alpha=alpha)
        elif test_type == "chi_square":
            from forgestat.parametric.chi_square import chi_square_independence

            result = chi_square_independence(validated_input["observed"], alpha=alpha)
        else:
            raise ValueError(f"Unknown test_type: {test_type}")

        clean = _result_to_clean_dict(result)

        outputs = []

        # Key metrics
        if "p_value" in clean:
            outputs.append(PluginOutput("p_value", "metric", clean["p_value"]))
        if "statistic" in clean:
            outputs.append(PluginOutput("statistic", "metric", clean["statistic"]))
        if "effect_size" in clean and clean["effect_size"] is not None:
            outputs.append(PluginOutput("effect_size", "metric", clean["effect_size"]))

        # Full result as text
        outputs.append(PluginOutput("result", "text", {"test_type": test_type, **clean}))

        return outputs
