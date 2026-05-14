from django.test import TestCase

from syn.plugins.registry import PluginRegistry


class TestContractEnvelopePlugin(TestCase):
    def setUp(self):
        from plugins.contract_envelope import ContractEnvelopePlugin

        self.registry = PluginRegistry()
        self.registry.register(ContractEnvelopePlugin)
        self.plugin = self.registry.get("contract_envelope")

    def test_vsm_contract(self):
        outputs = self.plugin.execute(
            {
                "problem": "Excessive lead time in press operations",
                "metric_name": "lead_time_days",
                "metric_baseline": 85.0,
                "metric_target": 55.0,
                "savings_estimate": 240000,
                "priority": "strategic",
                "source_type": "vsm",
                "timeline_months": 12,
                "owner_role": "Value Stream Manager",
            },
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["savings_estimate"].value == 240000
        assert by_key["gap_pct"].value > 0
        contract = by_key["contract"].value["contract"]
        assert contract["priority"] == "strategic"
        assert contract["source_type"] == "vsm"
        assert contract["gap"] == 30.0

    def test_capability_contract(self):
        outputs = self.plugin.execute(
            {
                "problem": "Cpk below threshold",
                "metric_name": "cpk",
                "metric_baseline": 0.9,
                "metric_target": 1.33,
                "savings_estimate": 50000,
                "priority": "tactical",
                "source_type": "capability",
            },
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        # Negative gap means baseline < target (improvement needed upward)
        assert by_key["gap_pct"].value < 0

    def test_text_normalization(self):
        outputs = self.plugin.execute(
            {
                "problem": {"text": "From upstream text port"},
                "metric_name": "defect_rate",
                "metric_baseline": 5.0,
                "metric_target": 1.0,
            },
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert "From upstream" in by_key["contract"].value["contract"]["problem"]

    def test_default_action_items(self):
        outputs = self.plugin.execute(
            {"problem": "Test", "metric_name": "x", "metric_baseline": 10, "metric_target": 5},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert "action_items" in by_key
        assert len(by_key["action_items"].value["text"]) > 0

    def test_invalid_priority(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            self.plugin.input_schema(problem="x", priority="urgent")
