"""Text Input Plugin — user-authored prose with dynamic semantic subtype.

The user types free text and picks a subtype (problem_statement, observation,
hypothesis, goal, note, etc.). The output port name and semantic type are
set dynamically based on that choice.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, field_validator

from syn.plugins.base import Plugin, PluginOutput


class TextInputInput(BaseModel):
    content: str
    subtype: str = "note"

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Content cannot be empty")
        return v.strip()

    @field_validator("subtype")
    @classmethod
    def subtype_valid(cls, v):
        if not v.strip():
            raise ValueError("Subtype cannot be empty")
        return v.strip().lower().replace(" ", "_")


class TextInputPlugin(Plugin):
    """Free text entry — user picks the semantic subtype, port label follows."""

    name = "text_input"
    version = "1.0.0"
    description = "Text entry with dynamic semantic type (problem statement, observation, goal, etc.)"
    input_schema = TextInputInput

    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        subtype = validated_input["subtype"]
        content = validated_input["content"]

        return [
            PluginOutput(
                key=subtype,
                output_type="text",
                value={"text": content},
            ),
        ]
