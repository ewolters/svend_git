"""Triage Plugin — wraps forgetriage data cleaning engine.

Inputs: columnar data dict.
Outputs: cleaned data, column profiles, outlier flags, normality checks.
"""

from typing import Any, Dict, List

from pydantic import BaseModel

from syn.plugins.base import Plugin, PluginOutput


class TriageInput(BaseModel):
    data: Dict[str, List]  # column_name -> values
    outlier_method: str = "iqr"
    normality_test: str = "shapiro"


class TriagePlugin(Plugin):
    name = "triage"
    version = "1.0.0"
    description = "Data cleaning and profiling — outlier detection, normality testing"
    input_schema = TriageInput

    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        from forgetriage import triage

        report = triage(
            data=validated_input["data"],
            outlier_method=validated_input.get("outlier_method", "iqr"),
            normality_test=validated_input.get("normality_test", "shapiro"),
        )

        outputs = []

        # Column profiles
        for profile in report.columns:
            outputs.append(
                PluginOutput(
                    f"profile_{profile.name}",
                    "text",
                    {
                        "column": profile.name,
                        "dtype": profile.dtype,
                        "count": profile.count,
                        "missing": profile.missing,
                        "mean": profile.mean,
                        "std": profile.std,
                    },
                )
            )

        # Outlier summary
        outlier_flags = {}
        for col_profile in report.columns:
            if col_profile.outliers and col_profile.outliers.count > 0:
                outlier_flags[col_profile.name] = col_profile.outliers.count
        outputs.append(PluginOutput("outlier_flags", "text", outlier_flags))

        # Normality results
        normality = {}
        for col_profile in report.columns:
            if col_profile.normality:
                normality[col_profile.name] = {
                    "is_normal": col_profile.normality.is_normal,
                    "p_value": col_profile.normality.p_value,
                }
        outputs.append(PluginOutput("normality", "text", normality))

        # Summary
        outputs.append(
            PluginOutput(
                "summary",
                "text",
                {
                    "total_columns": len(report.columns),
                    "total_rows": report.columns[0].count if report.columns else 0,
                    "columns_with_outliers": len(outlier_flags),
                },
            )
        )

        return outputs
