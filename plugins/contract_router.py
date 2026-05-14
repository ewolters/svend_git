"""Contract Router Plugin — route contracts by priority.

Strategic contracts → Hoshin breakthrough projects
Tactical contracts → standard project tracker
Quick wins → Kanban / action item list

The router classifies and outputs. It does NOT create records.
A bus subscriber or view layer creates the actual records.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from syn.plugins.base import Plugin, PluginOutput


class ContractRouterInput(BaseModel):
    contracts: Optional[Union[Dict, List]] = None  # multi-port: list of contracts
    contract: Optional[Any] = None  # single contract input


class ContractRouterPlugin(Plugin):
    name = "contract_router"
    version = "1.0.0"
    description = "Route improvement contracts to Hoshin, project tracker, or Kanban by priority"
    input_schema = ContractRouterInput

    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        # Normalize inputs to list
        all_contracts = []

        raw = validated_input.get("contracts")
        if raw:
            if isinstance(raw, list):
                all_contracts.extend(raw)
            else:
                all_contracts.append(raw)

        single = validated_input.get("contract")
        if single:
            if isinstance(single, dict) and "contract" in single:
                all_contracts.append(single["contract"])
            elif isinstance(single, dict):
                all_contracts.append(single)

        # Route by priority
        strategic = []
        tactical = []
        quick_wins = []

        for c in all_contracts:
            if not isinstance(c, dict):
                continue
            priority = c.get("priority", "tactical")
            if priority == "strategic":
                strategic.append(c)
            elif priority == "quick_win":
                quick_wins.append(c)
            else:
                tactical.append(c)

        total_savings = sum(c.get("savings_estimate", 0) for c in all_contracts if isinstance(c, dict))

        return [
            PluginOutput(
                "strategic",
                "text",
                {
                    "text": f"{len(strategic)} strategic projects for Hoshin",
                    "projects": strategic,
                },
            ),
            PluginOutput(
                "tactical",
                "text",
                {
                    "text": f"{len(tactical)} tactical projects",
                    "projects": tactical,
                },
            ),
            PluginOutput(
                "quick_wins",
                "text",
                {
                    "text": f"{len(quick_wins)} quick wins",
                    "projects": quick_wins,
                },
            ),
            PluginOutput("total_savings", "metric", total_savings),
            PluginOutput("strategic_count", "metric", float(len(strategic))),
            PluginOutput("tactical_count", "metric", float(len(tactical))),
            PluginOutput("quick_win_count", "metric", float(len(quick_wins))),
        ]
