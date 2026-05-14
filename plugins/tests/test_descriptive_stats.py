from django.test import TestCase

from syn.plugins.registry import PluginRegistry


class TestDescriptiveStatsPlugin(TestCase):
    def setUp(self):
        from plugins.descriptive_stats import DescriptiveStatsPlugin

        self.registry = PluginRegistry()
        self.registry.register(DescriptiveStatsPlugin)
        self.plugin = self.registry.get("descriptive_stats")

    def test_basic_stats(self):
        outputs = self.plugin.execute(
            {"data": [10, 20, 30, 40, 50]},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["mean"].value == 30.0
        assert by_key["median"].value == 30.0
        assert by_key["n"].value == 5.0

    def test_includes_std(self):
        outputs = self.plugin.execute(
            {"data": [1, 1, 1, 1, 100]},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["std"].value > 0

    def test_too_few_raises(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            self.plugin.input_schema(data=[1])
