"""Tests for XMatrixPlugin — Hoshin cross-reference matrix."""

from django.test import TestCase

from syn.plugins.registry import PluginRegistry


class TestXMatrixPlugin(TestCase):
    def setUp(self):
        from plugins.x_matrix import XMatrixPlugin

        self.registry = PluginRegistry()
        self.registry.register(XMatrixPlugin)
        self.plugin = self.registry.get("x_matrix")

    def test_x_matrix_from_cascades(self):
        outputs = self.plugin.execute(
            {
                "title": "FY2026 Hoshin",
                "cascades": [
                    {
                        "cascade": {
                            "strategic_objective": "Reduce lead time 40%",
                            "breakthrough_projects": [
                                {"name": "SMED implementation", "savings_estimate": 120000, "owner_role": "VS Manager"},
                                {"name": "Pull system", "savings_estimate": 80000, "owner_role": "VS Manager"},
                            ],
                        }
                    },
                    {
                        "cascade": {
                            "strategic_objective": "Improve quality to 99.5% FPY",
                            "breakthrough_projects": [
                                {"name": "SPC deployment", "savings_estimate": 60000, "owner_role": "Quality Manager"},
                            ],
                        }
                    },
                ],
            },
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["objective_count"].value == 2
        assert by_key["project_count"].value == 3
        assert by_key["total_portfolio_savings"].value == 260000
        xm = by_key["x_matrix"].value["x_matrix"]
        assert "Reduce lead time" in xm["south_objectives"][0]
        assert len(xm["west_projects"]) == 3

    def test_single_cascade(self):
        outputs = self.plugin.execute(
            {
                "cascades": {
                    "cascade": {
                        "strategic_objective": "Test",
                        "breakthrough_projects": [{"name": "P1", "savings_estimate": 10}],
                    }
                }
            },
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["project_count"].value == 1

    def test_empty_input(self):
        outputs = self.plugin.execute(
            {"title": "Empty"},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["project_count"].value == 0
        assert by_key["total_portfolio_savings"].value == 0
