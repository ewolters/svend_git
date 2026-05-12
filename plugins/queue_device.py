"""Queueing Theory Plugin — wraps forgequeue M/M/1 and M/M/c models.

Inputs: arrival rate, service rate, server count.
Outputs: wait time, queue length, utilization.
"""

from typing import Any, Dict, List

from pydantic import BaseModel

from syn.plugins.base import Plugin, PluginOutput


class QueueInput(BaseModel):
    arrival_rate: float
    service_rate: float
    servers: int = 1


class QueuePlugin(Plugin):
    name = "queue_analysis"
    version = "1.0.0"
    description = "Queueing theory — M/M/1 and M/M/c wait time, utilization"
    input_schema = QueueInput

    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        lam = validated_input["arrival_rate"]
        mu = validated_input["service_rate"]
        c = validated_input.get("servers", 1)

        # M/M/1 for single server
        if c == 1:
            rho = lam / mu
            if rho >= 1:
                return [PluginOutput("error", "text", {"text": "System unstable: arrival >= service rate"})]
            lq = rho**2 / (1 - rho)
            wq = lq / lam
            w = wq + 1 / mu
            ls = lam * w
        else:
            # M/M/c approximation
            rho = lam / (c * mu)
            if rho >= 1:
                return [PluginOutput("error", "text", {"text": "System unstable"})]
            # Simplified M/M/c
            lq = (rho ** (c + 1)) / (1 - rho) * (1 / c)
            wq = lq / lam
            w = wq + 1 / mu
            ls = lam * w

        return [
            PluginOutput("utilization", "metric", round(rho, 4)),
            PluginOutput("avg_wait_time", "metric", round(wq, 4)),
            PluginOutput("avg_system_time", "metric", round(w, 4)),
            PluginOutput("avg_queue_length", "metric", round(lq, 4)),
            PluginOutput("avg_system_length", "metric", round(ls, 4)),
            PluginOutput(
                "summary",
                "text",
                {
                    "servers": c,
                    "utilization": f"{rho * 100:.1f}%",
                    "avg_wait": f"{wq:.2f} time units",
                },
            ),
        ]
