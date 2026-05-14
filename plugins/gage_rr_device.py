"""Gage R&R Device Plugin — Measurement System Analysis.

Wraps forgestat.msa.gage_rr.crossed_gage_rr for ANOVA-based variance
decomposition of measurement system error (repeatability, reproducibility).
"""

from typing import Any, Dict, List

from pydantic import BaseModel, field_validator

from syn.plugins.base import Plugin, PluginOutput


class GageRRInput(BaseModel):
    """Input for a crossed Gage R&R study.

    Provide flat parallel lists: measurements[i] was taken by operators[i]
    on parts[i]. The design must be balanced (every operator measures every
    part the same number of times).
    """

    measurements: List[float]
    parts: List[Any]  # part labels — int or str
    operators: List[Any]  # operator labels — int or str

    @field_validator("measurements")
    @classmethod
    def min_measurements(cls, v):
        if len(v) < 4:
            raise ValueError("Need at least 4 measurements (2 operators × 2 parts × 1 replicate)")
        return v

    @field_validator("parts", "operators")
    @classmethod
    def labels_match_measurements(cls, v, info):
        # Cross-field length check is done in model_validator; here we just ensure non-empty
        if len(v) == 0:
            raise ValueError("labels list cannot be empty")
        return v

    def model_post_init(self, __context: Any) -> None:
        if len(self.measurements) != len(self.parts):
            raise ValueError(
                f"measurements length ({len(self.measurements)}) must equal parts length ({len(self.parts)})"
            )
        if len(self.measurements) != len(self.operators):
            raise ValueError(
                f"measurements length ({len(self.measurements)}) must equal operators length ({len(self.operators)})"
            )
        # Require at least 2 unique parts for a meaningful MSA
        if len(set(self.parts)) < 2:
            raise ValueError("Gage R&R requires at least 2 distinct parts")


class GageRRPlugin(Plugin):
    name = "gage_rr"
    version = "1.0.0"
    description = "Gage R&R measurement system analysis — crossed design (ANOVA)"
    input_schema = GageRRInput

    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        from forgestat.msa.gage_rr import crossed_gage_rr

        result = crossed_gage_rr(
            measurements=validated_input["measurements"],
            parts=validated_input["parts"],
            operators=validated_input["operators"],
        )

        outputs = [
            PluginOutput(
                key="repeatability",
                output_type="metric",
                value=float(result.var_repeatability),
                provenance="calculated",
                measure_slug="msa.repeatability",
            ),
            PluginOutput(
                key="reproducibility",
                output_type="metric",
                value=float(result.var_reproducibility),
                provenance="calculated",
                measure_slug="msa.reproducibility",
            ),
            PluginOutput(
                key="grr_pct",
                output_type="metric",
                value=float(result.pct_gage_rr),
                provenance="calculated",
                measure_slug="msa.grr_pct",
            ),
            PluginOutput(
                key="part_variation_pct",
                output_type="metric",
                value=float(result.pct_part),
                provenance="calculated",
                measure_slug="msa.part_variation_pct",
            ),
            PluginOutput(
                key="ndc",
                output_type="metric",
                value=int(result.ndc),
                provenance="calculated",
                measure_slug="msa.ndc",
            ),
            PluginOutput(
                key="result",
                output_type="text",
                value={
                    "design": result.design,
                    "n_operators": result.n_operators,
                    "n_parts": result.n_parts,
                    "n_replicates": result.n_replicates,
                    "var_repeatability": float(result.var_repeatability),
                    "var_reproducibility": float(result.var_reproducibility),
                    "var_operator": float(result.var_operator),
                    "var_interaction": float(result.var_interaction),
                    "var_part": float(result.var_part),
                    "var_gage_rr": float(result.var_gage_rr),
                    "var_total": float(result.var_total),
                    "pct_repeatability": float(result.pct_repeatability),
                    "pct_reproducibility": float(result.pct_reproducibility),
                    "pct_gage_rr": float(result.pct_gage_rr),
                    "pct_part": float(result.pct_part),
                    "ndc": int(result.ndc),
                    "anova_table": result.anova_table,
                },
                provenance="calculated",
            ),
        ]

        return outputs
