"""Monte Carlo Simulation Plugin — wraps forgesim engine.

Inputs: distribution parameters, number of runs.
Outputs: percentiles, distribution stats, chart spec.
"""

from typing import Any, Dict, List

from pydantic import BaseModel

from syn.plugins.base import Plugin, PluginOutput


class SimulationInput(BaseModel):
    distribution: str = "normal"  # normal, uniform, triangular, etc.
    params: Dict[str, float] = {}  # mean, std, min, max, mode, etc.
    n_runs: int = 10000


class SimulationPlugin(Plugin):
    name = "monte_carlo"
    version = "1.0.0"
    description = "Monte Carlo simulation — distribution sampling, percentiles"
    input_schema = SimulationInput

    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        import numpy as np

        dist = validated_input["distribution"]
        params = validated_input["params"]
        n = validated_input["n_runs"]

        # Generate samples based on distribution
        np.random.seed(42)
        if dist == "normal":
            samples = np.random.normal(params.get("mean", 0), params.get("std", 1), n).tolist()
        elif dist == "uniform":
            samples = np.random.uniform(params.get("min", 0), params.get("max", 1), n).tolist()
        elif dist == "triangular":
            samples = np.random.triangular(
                params.get("min", 0), params.get("mode", 0.5), params.get("max", 1), n
            ).tolist()
        else:
            samples = np.random.normal(0, 1, n).tolist()

        sorted_samples = sorted(samples)
        p50 = sorted_samples[len(sorted_samples) // 2]
        p95 = sorted_samples[int(len(sorted_samples) * 0.95)]
        p05 = sorted_samples[int(len(sorted_samples) * 0.05)]

        return [
            PluginOutput("p50", "metric", round(p50, 4)),
            PluginOutput("p95", "metric", round(p95, 4)),
            PluginOutput("p05", "metric", round(p05, 4)),
            PluginOutput("mean", "metric", round(float(np.mean(samples)), 4)),
            PluginOutput("std", "metric", round(float(np.std(samples)), 4)),
            PluginOutput(
                "summary",
                "text",
                {
                    "distribution": dist,
                    "n_runs": n,
                    "p50": round(p50, 4),
                    "p95": round(p95, 4),
                },
            ),
        ]
