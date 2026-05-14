"""Tests for StrategicCascadePlugin — Hoshin objective decomposition."""

from django.test import TestCase
from pydantic import ValidationError

from syn.plugins.registry import PluginRegistry


class TestStrategicCascadePlugin(TestCase):
    def setUp(self):
        from plugins.strategic_cascade import StrategicCascadePlugin

        self.registry = PluginRegistry()
        self.registry.register(StrategicCascadePlugin)
        self.plugin = self.registry.get("strategic_cascade")

    def test_cascade_with_contracts(self):
        outputs = self.plugin.execute(
            {
                "objective": "Reduce manufacturing lead time 40%",
                "target_metric": "lead_time_days",
                "baseline_value": 85,
                "target_value": 51,
                "contracts": [
                    {
                        "problem": "Press changeover too slow",
                        "source_type": "vsm",
                        "savings_estimate": 120000,
                        "action_items": ["Implement SMED"],
                    },
                    {
                        "problem": "WIP between stations",
                        "source_type": "vsm",
                        "savings_estimate": 80000,
                        "action_items": ["Install pull system"],
                    },
                ],
            },
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["project_count"].value == 2
        assert by_key["total_savings"].value == 200000
        cascade = by_key["cascade"].value["cascade"]
        assert cascade["strategic_objective"] == "Reduce manufacturing lead time 40%"
        assert len(cascade["breakthrough_projects"]) == 2

    def test_cascade_from_router_output(self):
        """Router outputs contracts wrapped in {"projects": [...]}"""
        outputs = self.plugin.execute(
            {
                "objective": "Improve quality",
                "contracts": {
                    "projects": [
                        {"problem": "High defect rate", "savings_estimate": 50000},
                    ]
                },
            },
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["project_count"].value == 1

    def test_empty_objective_raises(self):
        from plugins.strategic_cascade import StrategicCascadeInput

        with self.assertRaises(ValidationError):
            StrategicCascadeInput(objective="")

    def test_no_contracts(self):
        outputs = self.plugin.execute(
            {"objective": "Reduce cost 20%"},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["project_count"].value == 0
