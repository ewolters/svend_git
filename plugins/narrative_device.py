"""Narrative Plugin — auto-generate analysis summaries.

Takes any analysis result dict and produces structured narrative text.
This is a meta-device: it consumes output from other devices.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, field_validator

from syn.plugins.base import Plugin, PluginOutput


class NarrativeInput(BaseModel):
    analysis_type: str  # e.g. "capability", "control_chart", "hypothesis_test"
    analysis_id: str = ""  # optional sub-type
    statistics: Dict[str, Any]  # the result dict from upstream device
    summary: str = ""  # optional user-provided context

    @field_validator("statistics")
    @classmethod
    def not_empty(cls, v):
        if not v:
            raise ValueError("statistics dict cannot be empty")
        return v


class NarrativePlugin(Plugin):
    name = "narrative"
    version = "1.0.0"
    description = "Auto-generate analysis narrative from result statistics"
    input_schema = NarrativeInput

    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        from forgenarr import narrate

        result = narrate(
            analysis_type=validated_input["analysis_type"],
            analysis_id=validated_input.get("analysis_id", ""),
            statistics=validated_input["statistics"],
            summary=validated_input.get("summary", ""),
        )

        return [
            PluginOutput("verdict", "text", {"text": result.get("verdict", "")}),
            PluginOutput("body", "text", {"text": result.get("body", "")}),
            PluginOutput("next_steps", "text", {"text": result.get("next_steps", "")}),
            PluginOutput("narrative", "text", result),
        ]
