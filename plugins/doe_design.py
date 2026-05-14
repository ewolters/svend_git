"""DOE Design Plugin — generate experimental designs.

Creates factorial, response surface, and screening designs from factor definitions.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, field_validator

from syn.plugins.base import Plugin, PluginOutput


class FactorDef(BaseModel):
    name: str
    low: float
    high: float


class DOEDesignInput(BaseModel):
    design_type: str  # full_factorial, fractional_factorial, central_composite, box_behnken, plackett_burman, definitive_screening
    factors: List[FactorDef]
    center_points: int = 0
    resolution: int = 3  # for fractional_factorial
    alpha_type: str = "rotatable"  # for CCD: rotatable, face, spherical

    @field_validator("design_type")
    @classmethod
    def valid_design(cls, v):
        valid = {
            "full_factorial",
            "fractional_factorial",
            "central_composite",
            "box_behnken",
            "plackett_burman",
            "definitive_screening",
        }
        if v not in valid:
            raise ValueError(f"design_type must be one of: {', '.join(sorted(valid))}")
        return v

    @field_validator("factors")
    @classmethod
    def at_least_two(cls, v):
        if len(v) < 2:
            raise ValueError("Need at least 2 factors for DOE")
        return v


class DOEDesignPlugin(Plugin):
    name = "doe_design"
    version = "1.0.0"
    description = "DOE design generator — factorial, response surface, screening"
    input_schema = DOEDesignInput

    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        from forgedoe.core.types import Factor

        factors = [Factor(name=f["name"], low=f["low"], high=f["high"]) for f in validated_input["factors"]]
        design_type = validated_input["design_type"]
        center_points = validated_input.get("center_points", 0)

        if design_type == "full_factorial":
            from forgedoe.designs.factorial import full_factorial

            result = full_factorial(factors, center_points=center_points, randomize=False)
        elif design_type == "fractional_factorial":
            from forgedoe.designs.factorial import fractional_factorial

            result = fractional_factorial(factors, resolution=validated_input.get("resolution", 3), randomize=False)
        elif design_type == "central_composite":
            from forgedoe.designs.response_surface import central_composite_design

            result = central_composite_design(
                factors,
                alpha=validated_input.get("alpha_type", "rotatable"),
                center_points=max(center_points, 3),
                randomize=False,
            )
        elif design_type == "box_behnken":
            from forgedoe.designs.response_surface import box_behnken_design

            result = box_behnken_design(factors, center_points=max(center_points, 3), randomize=False)
        elif design_type == "plackett_burman":
            from forgedoe.designs.factorial import plackett_burman

            result = plackett_burman(factors, randomize=False)
        elif design_type == "definitive_screening":
            from forgedoe.designs.screening import definitive_screening_design

            result = definitive_screening_design(factors, randomize=False)
        else:
            raise ValueError(f"Unknown design_type: {design_type}")

        # Convert matrix to serializable format
        matrix = result.matrix
        if hasattr(matrix, "tolist"):
            matrix = matrix.tolist()
        elif isinstance(matrix, list) and matrix and hasattr(matrix[0], "tolist"):
            matrix = [row.tolist() for row in matrix]

        factor_names = [f.name for f in result.factors]
        n_runs = len(matrix)

        return [
            PluginOutput("n_runs", "metric", float(n_runs)),
            PluginOutput(
                "design_matrix",
                "text",
                {
                    "design_type": design_type,
                    "factors": factor_names,
                    "n_runs": n_runs,
                    "matrix": matrix,
                    "is_coded": result.is_coded,
                },
            ),
        ]
