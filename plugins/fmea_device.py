"""FMEA Plugin — wraps forgefmea RPN scoring engine.

Inputs: list of FMEA rows with S/O/D scores.
Outputs: RPN values, action priority, risk summary.
"""

from typing import Any, Dict, List

from pydantic import BaseModel

from syn.plugins.base import Plugin, PluginOutput


class FMEARowInput(BaseModel):
    process_step: str
    failure_mode: str
    severity: int
    occurrence: int
    detection: int
    effect: str = ""
    cause: str = ""


class FMEAInput(BaseModel):
    rows: List[FMEARowInput]


class FMEAPlugin(Plugin):
    name = "fmea_analysis"
    version = "1.0.0"
    description = "FMEA risk analysis — RPN scoring, action priority, risk summary"
    input_schema = FMEAInput

    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        from forgefmea import FMEARow, compute_action_priority, compute_rpn, rpn_summary

        rows = []
        rpn_values = []
        for r in validated_input["rows"]:
            row = r if isinstance(r, dict) else r
            s, o, d = row["severity"], row["occurrence"], row["detection"]
            rpn = compute_rpn(s, o, d)
            ap = compute_action_priority(s, o, d)
            rpn_values.append(
                {
                    "process_step": row["process_step"],
                    "failure_mode": row["failure_mode"],
                    "rpn": rpn,
                    "action_priority": ap,
                }
            )
            rows.append(
                FMEARow(
                    process_step=row["process_step"],
                    failure_mode=row["failure_mode"],
                    effect=row.get("effect", ""),
                    cause=row.get("cause", ""),
                    severity=s,
                    occurrence=o,
                    detection=d,
                )
            )

        summary = rpn_summary(rows)
        max_rpn = max(r["rpn"] for r in rpn_values) if rpn_values else 0
        high_risk = [r for r in rpn_values if r["action_priority"] == "H"]

        return [
            PluginOutput("rpn_table", "text", {"rows": rpn_values}),
            PluginOutput("max_rpn", "metric", max_rpn),
            PluginOutput("high_risk_count", "metric", len(high_risk)),
            PluginOutput(
                "summary",
                "text",
                {
                    "total_rows": len(rpn_values),
                    "max_rpn": max_rpn,
                    "high_risk_count": len(high_risk),
                    "mean_rpn": summary.mean_rpn if hasattr(summary, "mean_rpn") else 0,
                },
            ),
        ]
