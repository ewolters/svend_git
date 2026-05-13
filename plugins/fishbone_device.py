"""Fishbone (Ishikawa) Plugin — cause-and-effect analysis device.

Inputs: problem statement text.
Outputs: structured cause list + diagram spec (for ForgeViz rendering).

Uses the 6M categories standard in manufacturing:
Man, Machine, Method, Material, Measurement, Mother Nature (Environment).
"""

from typing import Any, Dict, List

from pydantic import BaseModel, field_validator

from syn.plugins.base import Plugin, PluginOutput

CATEGORIES_6M = ["Man", "Machine", "Method", "Material", "Measurement", "Environment"]


class FishboneInput(BaseModel):
    problem_statement: Any  # str or {"text": str} from upstream routing
    categories: List[str] = CATEGORIES_6M

    @field_validator("problem_statement")
    @classmethod
    def normalize_statement(cls, v):
        # Upstream text ports arrive as {"text": "..."} via engine routing
        if isinstance(v, dict) and "text" in v:
            v = v["text"]
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Problem statement cannot be empty")
        return v.strip()


class FishbonePlugin(Plugin):
    """Cause-and-effect (Ishikawa) diagram — structures problem into 6M categories."""

    name = "fishbone"
    version = "1.0.0"
    description = "Fishbone diagram — 6M cause-and-effect analysis"
    input_schema = FishboneInput

    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        problem = validated_input["problem_statement"]
        categories = validated_input.get("categories", CATEGORIES_6M)

        # Structure: each category gets an empty cause list for the user to populate.
        # In template mode, these are filled in interactively. In build mode,
        # upstream devices or manual entry populate them.
        causes = {cat: [] for cat in categories}

        # Diagram spec — ForgeViz-compatible chart structure
        diagram = {
            "chart_type": "fishbone",
            "problem_statement": problem,
            "categories": categories,
            "causes": causes,
        }

        return [
            PluginOutput("causes", "text", {"problem": problem, "categories": causes}),
            PluginOutput("diagram", "chart", diagram),
        ]
