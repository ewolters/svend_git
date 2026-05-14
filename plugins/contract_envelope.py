"""Contract Envelope Plugin — universal improvement contract.

Any methodology that finds a problem and quantifies a fix produces a contract:
- VSM diff → contract (reduce lead time)
- FMEA high-RPN → contract (mitigate failure mode)
- Capability Cpk < 1.33 → contract (bring process into spec)
- A3 root cause → contract (implement countermeasure)

The contract doesn't know its destination. A router decides where it goes.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator

from syn.plugins.base import Plugin, PluginOutput


class ContractEnvelopeInput(BaseModel):
    problem: Any = ""
    metric_name: str = ""
    metric_baseline: float = 0.0
    metric_target: float = 0.0
    savings_estimate: float = 0.0
    priority: str = "tactical"  # strategic | tactical | quick_win
    timeline_months: int = 6
    owner_role: str = ""
    source_type: str = "manual"  # vsm, fmea, capability, a3, rca, manual
    action_items: Optional[List[str]] = None
    evidence: Optional[Dict] = None

    @field_validator("problem")
    @classmethod
    def normalize_problem(cls, v):
        if isinstance(v, dict) and "text" in v:
            return v["text"]
        return str(v) if v else ""

    @field_validator("priority")
    @classmethod
    def valid_priority(cls, v):
        valid = {"strategic", "tactical", "quick_win"}
        if v not in valid:
            raise ValueError(f"priority must be one of: {', '.join(sorted(valid))}")
        return v


class ContractEnvelopePlugin(Plugin):
    name = "contract_envelope"
    version = "1.0.0"
    description = "Package analysis results into a standardized improvement contract"
    input_schema = ContractEnvelopeInput

    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        # Normalize problem — may arrive as raw dict from upstream text port
        problem = validated_input.get("problem", "")
        if isinstance(problem, dict) and "text" in problem:
            problem = problem["text"]
        elif not isinstance(problem, str):
            problem = str(problem) if problem else ""

        baseline = validated_input["metric_baseline"]
        target = validated_input["metric_target"]
        gap = baseline - target
        gap_pct = (gap / baseline * 100) if baseline else 0

        metric_name = validated_input.get("metric_name", "")
        savings = validated_input.get("savings_estimate", 0.0)
        priority = validated_input.get("priority", "tactical")

        contract = {
            "type": "improvement_contract",
            "version": "1.0",
            "problem": problem,
            "metric_name": metric_name,
            "metric_baseline": baseline,
            "metric_target": target,
            "gap": gap,
            "gap_pct": gap_pct,
            "savings_estimate": savings,
            "priority": priority,
            "timeline_months": validated_input.get("timeline_months", 6),
            "owner_role": validated_input.get("owner_role", ""),
            "source_type": validated_input.get("source_type", "manual"),
            "action_items": validated_input.get("action_items")
            or [
                f"Close gap: {metric_name} from {baseline} to {target}",
            ],
            "evidence": validated_input.get("evidence"),
        }

        return [
            PluginOutput(
                "contract",
                "text",
                {
                    "text": f"Contract: {problem[:60]} [{priority}]",
                    "contract": contract,
                },
            ),
            PluginOutput(
                "action_items",
                "text",
                {
                    "text": "\n".join(contract["action_items"]),
                },
            ),
            PluginOutput("savings_estimate", "metric", savings),
            PluginOutput("gap_pct", "metric", gap_pct),
        ]
