"""VSM Plugin — wraps forgevsm value stream analysis engine.

Inputs: process steps with cycle times, setup times, batch sizes, WIP, uptime.
Outputs: lead time, process time, PCE, takt, bottleneck, waste analysis.
"""

from typing import Any, Dict, List

from pydantic import BaseModel

from syn.plugins.base import Plugin, PluginOutput


class ProcessStepInput(BaseModel):
    """Input schema for a VSM process step.

    Matches forgevsm.ProcessStep fields. See forgevsm for full documentation.
    """

    name: str
    cycle_time: float
    changeover_time: float = 0.0
    batch_size: int = 0
    uptime: float = 1.0
    operators: int = 1
    shifts: int = 1
    scrap_rate: float = 0.0


# Fields that map directly from ProcessStepInput → forgevsm.ProcessStep
_STEP_FIELDS = {"name", "cycle_time", "changeover_time", "batch_size", "uptime", "operators", "shifts", "scrap_rate"}


class VSMInput(BaseModel):
    steps: List[ProcessStepInput]
    demand_rate: float = 100.0
    available_seconds: float = 28800.0  # 8-hour shift


class VSMPlugin(Plugin):
    name = "vsm_analysis"
    version = "1.0.0"
    description = "Value stream map analysis — lead time, PCE, bottleneck, waste"
    input_schema = VSMInput

    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        from forgevsm import ProcessStep, analyze_vsm

        raw_steps = validated_input["steps"]
        steps = []
        for s in raw_steps:
            d = s if isinstance(s, dict) else s.dict()
            # Only pass fields that forgevsm.ProcessStep accepts
            steps.append(ProcessStep(**{k: v for k, v in d.items() if k in _STEP_FIELDS}))

        result = analyze_vsm(
            steps=steps,
            demand_rate=validated_input.get("demand_rate", 100.0),
            available_seconds=validated_input.get("available_seconds", 28800.0),
        )

        outputs = [
            PluginOutput("lead_time", "metric", result.lead_time_days, measure_slug="lead_time"),
            PluginOutput("process_time", "metric", result.process_time_sec),
            PluginOutput("pce", "metric", result.pce, measure_slug="pce"),
            PluginOutput("total_wip", "metric", result.total_wip),
            PluginOutput("takt_time", "metric", result.takt_time),
        ]

        # Bottleneck
        if result.bottleneck_name:
            outputs.append(
                PluginOutput(
                    "bottleneck",
                    "text",
                    {
                        "name": result.bottleneck_name,
                        "cycle_time": result.bottleneck_ct,
                    },
                )
            )

        # Summary
        outputs.append(
            PluginOutput(
                "summary",
                "text",
                {
                    "text": f"VSM: {len(steps)} steps, lead time {result.lead_time_days:.1f} days, PCE {result.pce:.2f}%",
                },
            )
        )

        return outputs
