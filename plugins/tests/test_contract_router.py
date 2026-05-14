from django.test import TestCase

from syn.plugins.registry import PluginRegistry


class TestContractRouterPlugin(TestCase):
    def setUp(self):
        from plugins.contract_router import ContractRouterPlugin

        self.registry = PluginRegistry()
        self.registry.register(ContractRouterPlugin)
        self.plugin = self.registry.get("contract_router")

    def test_routes_by_priority(self):
        outputs = self.plugin.execute(
            {
                "contracts": [
                    {"priority": "strategic", "problem": "Lead time", "savings_estimate": 200000},
                    {"priority": "tactical", "problem": "Defect rate", "savings_estimate": 50000},
                    {"priority": "quick_win", "problem": "5S", "savings_estimate": 5000},
                    {"priority": "strategic", "problem": "WIP reduction", "savings_estimate": 150000},
                ]
            },
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["strategic_count"].value == 2
        assert by_key["tactical_count"].value == 1
        assert by_key["quick_win_count"].value == 1
        assert by_key["total_savings"].value == 405000

    def test_single_contract_input(self):
        outputs = self.plugin.execute(
            {"contract": {"contract": {"priority": "strategic", "problem": "X", "savings_estimate": 100}}},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["strategic_count"].value == 1

    def test_empty_input(self):
        outputs = self.plugin.execute(
            {},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["total_savings"].value == 0
        assert by_key["strategic_count"].value == 0
