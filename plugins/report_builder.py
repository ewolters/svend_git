"""Report Builder Plugin — aggregation device that collects outputs into a report.

Inputs: charts (list), metrics (list), text (list), lists (list).
Outputs: assembled report document.

This is a multi-port device — each input accepts multiple connections.
The engine routes all connected values to the same input key. Since
engine.py overwrites input_data[tgt_port] per connection, multi-port
aggregation is handled here: inputs arrive as last-wins for now,
but the template definition declares ports as multi=True so the
renderer knows to allow multiple cables.

TODO: Engine needs multi-port support (collect into list instead of overwrite).
For now, we accept whatever arrives and wrap scalars into lists.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from syn.plugins.base import Plugin, PluginOutput


class ReportBuilderInput(BaseModel):
    title: str = "Report"
    # These arrive as whatever the last connection routed — scalar or list.
    # We normalize to lists in execute().
    charts: Optional[Union[Dict, List]] = None
    metrics: Optional[Union[float, Dict, List]] = None
    text: Optional[Union[str, Dict, List]] = None
    lists: Optional[Union[Dict, List]] = None


class ReportBuilderPlugin(Plugin):
    """Aggregation device — collects charts, metrics, text into a structured report."""

    name = "report_builder"
    version = "1.0.0"
    description = "Report builder — assembles charts, metrics, and text into a document"
    input_schema = ReportBuilderInput

    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        title = validated_input.get("title", "Report")

        # Normalize all inputs to lists
        def to_list(val):
            if val is None:
                return []
            if isinstance(val, list):
                return val
            return [val]

        charts = to_list(validated_input.get("charts"))
        metrics = to_list(validated_input.get("metrics"))
        texts = to_list(validated_input.get("text"))
        lists = to_list(validated_input.get("lists"))

        # Assemble report structure
        report = {
            "title": title,
            "sections": [],
        }

        if metrics:
            report["sections"].append(
                {
                    "type": "metrics",
                    "label": "Key Metrics",
                    "items": metrics,
                }
            )

        if charts:
            report["sections"].append(
                {
                    "type": "charts",
                    "label": "Charts & Visualizations",
                    "items": charts,
                }
            )

        if texts:
            report["sections"].append(
                {
                    "type": "text",
                    "label": "Analysis Summary",
                    "items": texts,
                }
            )

        if lists:
            report["sections"].append(
                {
                    "type": "lists",
                    "label": "Findings",
                    "items": lists,
                }
            )

        return [
            PluginOutput("report", "text", report),
        ]
