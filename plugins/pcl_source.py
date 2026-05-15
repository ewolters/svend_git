"""PCL Source Plugin — pull measures from Process Characteristics Library.

Reads one or more measures by slug. Two output modes:

1. Latest value (default): outputs a single metric per slug.
   Good for: conditional gates, contract thresholds, single-value routing.

2. History mode (history_count > 0): outputs a list of floats per slug.
   Good for: control charts, process behavior analysis — anything needing a series.

This is how non-data-entry flowcharts (monitoring, VSM) get input from PCL.
"""

import logging
from typing import Any, Dict, List

from pydantic import BaseModel, field_validator

from syn.plugins.base import Plugin, PluginOutput

logger = logging.getLogger(__name__)


class PCLSourceInput(BaseModel):
    measure_slugs: List[str]
    history_count: int = 0  # 0 = latest only, >0 = pull N most recent datapoints

    @field_validator("measure_slugs")
    @classmethod
    def slugs_not_empty(cls, v):
        if len(v) < 1:
            raise ValueError("Need at least one measure slug")
        return v


class PCLSourcePlugin(Plugin):
    """Reads measures from PCL — the process state entry point for flowcharts."""

    name = "pcl_source"
    version = "1.0.0"
    description = "PCL source — pull live process measures into a flowchart"
    input_schema = PCLSourceInput

    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        from pcl.models import Datapoint, Measure

        history_count = validated_input.get("history_count", 0)
        outputs = []

        for slug in validated_input["measure_slugs"]:
            try:
                measure = Measure.objects.get(slug=slug, is_deleted=False)
            except Measure.DoesNotExist:
                logger.debug("[PCL_SOURCE] Measure '%s' not found", slug)
                if history_count > 0:
                    outputs.append(PluginOutput(f"{slug}_series", "data", []))
                else:
                    outputs.append(PluginOutput(slug, "metric", None, provenance="observed"))
                continue

            if history_count > 0:
                # Pull historical series for control charts / process behavior
                datapoints = Datapoint.objects.filter(measure=measure).order_by("-created_at")[:history_count]
                # Reverse to chronological order
                values = [dp.value for dp in reversed(datapoints) if dp.value is not None]
                outputs.append(
                    PluginOutput(
                        key=f"{slug}_series",
                        output_type="data",
                        value=values,
                        measure_slug=slug,
                    )
                )
                # Also output latest as a single metric (for conditional gates)
                if values:
                    outputs.append(
                        PluginOutput(
                            key=slug,
                            output_type="metric",
                            value=values[-1],
                            provenance="observed",
                            measure_slug=slug,
                        )
                    )
            else:
                # Single latest value
                latest = Datapoint.objects.filter(measure=measure).order_by("-created_at").first()
                value = latest.value if latest else None
                provenance = getattr(latest, "provenance", "observed") if latest else "observed"
                outputs.append(
                    PluginOutput(
                        key=slug,
                        output_type="metric",
                        value=value,
                        provenance=provenance,
                        measure_slug=slug,
                    )
                )

        return outputs
