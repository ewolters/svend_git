"""PCL Source Plugin — pull measures from Process Characteristics Library.

Reads one or more measures by slug and outputs their current values
as metric ports. This is how non-data-entry flowcharts (VSM, monitoring)
get their input — by reading the process state from PCL.
"""

import logging
from typing import Any, Dict, List

from pydantic import BaseModel, field_validator

from syn.plugins.base import Plugin, PluginOutput

logger = logging.getLogger(__name__)


class PCLSourceInput(BaseModel):
    measure_slugs: List[str]

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

        outputs = []
        for slug in validated_input["measure_slugs"]:
            try:
                measure = Measure.objects.get(slug=slug, is_deleted=False)
            except Measure.DoesNotExist:
                logger.debug("[PCL_SOURCE] Measure '%s' not found", slug)
                outputs.append(PluginOutput(slug, "metric", None, provenance="observed"))
                continue

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
