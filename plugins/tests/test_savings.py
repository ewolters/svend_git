from django.test import TestCase

from syn.plugins.registry import PluginRegistry


class TestSavingsPlugin(TestCase):
    def setUp(self):
        from plugins.savings_device import SavingsPlugin

        self.registry = PluginRegistry()
        self.registry.register(SavingsPlugin)
        self.plugin = self.registry.get("savings_analysis")

    def test_waste_savings(self):
        outputs = self.plugin.execute(
            {
                "analysis_type": "savings",
                "method": "waste_pct",
                "baseline": 10.0,
                "actual": 5.0,
                "volume": 1000,
                "cost_per_unit": 2.0,
            },
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert "savings" in by_key
        assert by_key["savings"].value > 0

    def test_throughput_accounting(self):
        outputs = self.plugin.execute(
            {
                "analysis_type": "throughput",
                "revenue": 100000,
                "truly_variable_costs": 40000,
                "inventory": 20000,
                "operating_expense": 30000,
            },
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["throughput"].value == 60000.0
        assert by_key["net_profit"].value == 30000.0

    def test_target_cost(self):
        outputs = self.plugin.execute(
            {"analysis_type": "target_cost", "selling_price": 100, "target_margin_pct": 20, "current_cost": 90},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["target_cost"].value == 80.0
        assert by_key["cost_gap"].value == 10.0

    def test_invalid_type(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            self.plugin.input_schema(analysis_type="bogus")
