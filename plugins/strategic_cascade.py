"""Strategic Cascade Plugin — Hoshin objective decomposition.

Takes a strategic objective and incoming improvement contracts (from VSM,
FMEA, capability, etc.) and structures them as Hoshin breakthrough projects.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, field_validator

from syn.plugins.base import Plugin, PluginOutput


class StrategicCascadeInput(BaseModel):
    objective: str
    target_metric: str = ""
    target_value: float = 0.0
    baseline_value: float = 0.0
    timeline_months: int = 12
    # Contracts arrive from contract_router's strategic output
    contracts: Optional[Union[Dict, List]] = None

    @field_validator("objective", mode="before")
    @classmethod
    def unwrap_objective(cls, v):
        """Unwrap text PluginOutput dicts — upstream text devices emit {"text": "..."}."""
        if isinstance(v, dict) and "text" in v:
            v = v["text"]
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Strategic objective cannot be empty")
        return v.strip()


class StrategicCascadePlugin(Plugin):
    name = "strategic_cascade"
    version = "1.0.0"
    description = "Hoshin strategic cascade — objectives to breakthrough projects from contracts"
    input_schema = StrategicCascadeInput

    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        objective = validated_input["objective"]
        target_value = validated_input.get("target_value", 0)
        baseline_value = validated_input.get("baseline_value", 0)
        timeline = validated_input.get("timeline_months", 12)

        # Normalize contracts
        raw = validated_input.get("contracts")
        contracts = []
        if isinstance(raw, list):
            contracts = raw
        elif isinstance(raw, dict):
            # Could be a single contract or a router output with "projects" key
            if "projects" in raw:
                contracts = raw["projects"] if isinstance(raw["projects"], list) else [raw["projects"]]
            else:
                contracts = [raw]

        # Build projects from contracts
        projects = []
        for c in contracts:
            if not isinstance(c, dict):
                continue
            projects.append(
                {
                    "name": c.get("problem", "Improvement Project"),
                    "source": c.get("source_type", "unknown"),
                    "metric_baseline": c.get("metric_baseline", 0),
                    "metric_target": c.get("metric_target", 0),
                    "savings_estimate": c.get("savings_estimate", 0),
                    "action_items": c.get("action_items", []),
                    "timeline_months": c.get("timeline_months", timeline),
                    "owner_role": c.get("owner_role", ""),
                }
            )

        gap = target_value - baseline_value
        gap_pct = (gap / baseline_value * 100) if baseline_value else 0

        cascade = {
            "strategic_objective": objective,
            "target_metric": validated_input.get("target_metric", ""),
            "baseline": baseline_value,
            "target": target_value,
            "gap": gap,
            "gap_pct": gap_pct,
            "timeline_months": timeline,
            "breakthrough_projects": projects,
            "n_projects": len(projects),
        }

        total_savings = sum(p.get("savings_estimate", 0) for p in projects)

        return [
            PluginOutput(
                "cascade",
                "text",
                {
                    "text": f"Hoshin: {objective} — {len(projects)} breakthrough projects",
                    "cascade": cascade,
                },
            ),
            PluginOutput(
                "projects",
                "text",
                {
                    "text": "\n".join(p["name"] for p in projects) if projects else "No projects",
                    "projects": projects,
                },
            ),
            PluginOutput("total_savings", "metric", total_savings),
            PluginOutput("project_count", "metric", float(len(projects))),
        ]
