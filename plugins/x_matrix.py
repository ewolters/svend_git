"""X-Matrix Plugin — Hoshin cross-reference matrix.

The X-matrix shows:
  South: Strategic objectives
  West: Breakthrough projects
  North: KPIs / targets
  East: Process metrics / owners

This device aggregates cascade outputs into the X-matrix structure.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from syn.plugins.base import Plugin, PluginOutput


class XMatrixInput(BaseModel):
    title: str = "Hoshin X-Matrix"
    # Multi-port: receives cascade outputs from 1+ strategic_cascade devices
    cascades: Optional[Union[Dict, List]] = None
    # Optional direct KPI list
    kpis: Optional[List[str]] = None


class XMatrixPlugin(Plugin):
    name = "x_matrix"
    version = "1.0.0"
    description = "Hoshin X-Matrix — cross-reference objectives, projects, KPIs"
    input_schema = XMatrixInput

    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        title = validated_input.get("title", "Hoshin X-Matrix")

        # Normalize cascades
        raw = validated_input.get("cascades")
        cascades = []
        if isinstance(raw, list):
            cascades = raw
        elif isinstance(raw, dict):
            cascades = [raw]

        objectives = []
        all_projects = []

        for c in cascades:
            if not isinstance(c, dict):
                continue
            # Could be direct cascade dict or wrapped in "cascade" key
            cascade = c.get("cascade", c)
            obj = cascade.get("strategic_objective", "")
            if obj:
                objectives.append(obj)
            for p in cascade.get("breakthrough_projects", []):
                all_projects.append(p)

        kpis = validated_input.get("kpis") or []
        # Auto-generate KPIs from project metrics if none provided
        if not kpis:
            for p in all_projects:
                metric = p.get("metric_target")
                if isinstance(metric, (int, float)):
                    name = p.get("name", "")[:30]
                    kpis.append(f"{name}: target={metric}")

        total_savings = sum(p.get("savings_estimate", 0) for p in all_projects if isinstance(p, dict))

        x_matrix = {
            "title": title,
            "south_objectives": objectives,
            "west_projects": [p.get("name", str(p)[:50]) if isinstance(p, dict) else str(p) for p in all_projects],
            "north_kpis": kpis,
            "east_owners": list({p.get("owner_role", "TBD") for p in all_projects if isinstance(p, dict)}),
            "n_objectives": len(objectives),
            "n_projects": len(all_projects),
            "n_kpis": len(kpis),
            "total_portfolio_savings": total_savings,
        }

        return [
            PluginOutput(
                "x_matrix",
                "text",
                {
                    "text": f"X-Matrix: {len(objectives)} objectives, {len(all_projects)} projects, {len(kpis)} KPIs",
                    "x_matrix": x_matrix,
                },
            ),
            PluginOutput("total_portfolio_savings", "metric", total_savings),
            PluginOutput("project_count", "metric", float(len(all_projects))),
            PluginOutput("objective_count", "metric", float(len(objectives))),
        ]
