from django.test import TestCase

from syn.plugins.registry import PluginRegistry


class TestHypothesisTestPlugin(TestCase):
    def setUp(self):
        from plugins.hypothesis_test import HypothesisTestPlugin

        self.registry = PluginRegistry()
        self.registry.register(HypothesisTestPlugin)
        self.plugin = self.registry.get("hypothesis_test")

    def test_one_sample_t(self):
        outputs = self.plugin.execute(
            {"test_type": "one_sample_t", "data": [50.1, 49.8, 50.3, 49.9, 50.0, 50.2, 49.7, 50.1], "mu": 50.0},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert "p_value" in by_key
        assert "statistic" in by_key
        assert isinstance(by_key["p_value"].value, float)

    def test_two_sample_t(self):
        outputs = self.plugin.execute(
            {"test_type": "two_sample_t", "data": [10, 11, 12, 13, 14], "data2": [20, 21, 22, 23, 24]},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["p_value"].value < 0.05  # clearly different groups

    def test_one_way_anova(self):
        outputs = self.plugin.execute(
            {"test_type": "one_way_anova", "data": [], "groups": {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert "p_value" in by_key

    def test_invalid_test_type(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            self.plugin.input_schema(test_type="bogus", data=[1, 2, 3])
