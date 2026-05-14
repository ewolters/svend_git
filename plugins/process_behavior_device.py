"""Process Behavior Plugin — change detection and belief charts.

Uses forgepbs BeliefChart in batch mode. Accepts a data array,
processes all points, returns changepoints and alert summary.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, field_validator

from syn.plugins.base import Plugin, PluginOutput


class ProcessBehaviorInput(BaseModel):
    data: List[float]
    hazard_rate: float = 0.005  # 1/200 default
    signal_threshold: float = 0.5
    caution_threshold: float = 0.2

    @field_validator("data")
    @classmethod
    def enough_data(cls, v):
        if len(v) < 10:
            raise ValueError("Need at least 10 observations for process behavior analysis")
        return v


class ProcessBehaviorPlugin(Plugin):
    name = "process_behavior"
    version = "1.0.0"
    description = "Process behavior — BOCPD change detection with belief chart"
    input_schema = ProcessBehaviorInput

    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        from forgepbs.charts.belief import BeliefChart

        chart = BeliefChart(
            hazard_rate=validated_input.get("hazard_rate", 0.005),
            signal_threshold=validated_input.get("signal_threshold", 0.5),
            caution_threshold=validated_input.get("caution_threshold", 0.2),
        )

        points = chart.process_batch(validated_input["data"])
        changepoints = chart.changepoints

        # Count alerts
        signals = sum(1 for p in points if p.alert_level == "signal")
        cautions = sum(1 for p in points if p.alert_level == "caution")

        # Build series for charting
        shift_probs = [p.shift_probability for p in points]
        values = [p.value for p in points]

        cp_list = []
        for cp in changepoints:
            cp_dict = cp.__dict__ if hasattr(cp, "__dict__") else {"index": getattr(cp, "index", None)}
            # Clean numpy scalars
            clean_cp = {}
            for k, v in cp_dict.items():
                if hasattr(v, "item"):
                    clean_cp[k] = v.item()
                elif hasattr(v, "tolist"):
                    clean_cp[k] = v.tolist()
                else:
                    clean_cp[k] = v
            cp_list.append(clean_cp)

        return [
            PluginOutput("n_signals", "metric", float(signals)),
            PluginOutput("n_cautions", "metric", float(cautions)),
            PluginOutput("n_changepoints", "metric", float(len(cp_list))),
            PluginOutput(
                "chart_data",
                "chart",
                {
                    "chart_type": "belief_chart",
                    "values": values,
                    "shift_probabilities": shift_probs,
                    "changepoints": cp_list,
                },
            ),
            PluginOutput(
                "result",
                "text",
                {
                    "n_observations": len(points),
                    "n_signals": signals,
                    "n_cautions": cautions,
                    "n_changepoints": len(cp_list),
                    "changepoints": cp_list,
                },
            ),
        ]
