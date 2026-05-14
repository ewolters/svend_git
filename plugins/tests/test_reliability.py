from django.test import TestCase

from syn.plugins.registry import PluginRegistry


class TestReliabilityPlugin(TestCase):
    def setUp(self):
        from plugins.reliability_device import ReliabilityPlugin

        self.registry = PluginRegistry()
        self.registry.register(ReliabilityPlugin)
        self.plugin = self.registry.get("reliability")

    def test_basic_mtbf(self):
        outputs = self.plugin.execute(
            {"failure_times": [100, 250, 400, 600, 850]},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert "mtbf" in by_key
        assert by_key["mtbf"].value > 0

    def test_with_repair_times(self):
        outputs = self.plugin.execute(
            {"failure_times": [100, 300, 500], "repair_times": [5, 8, 3]},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert "availability" in by_key
        assert "mttr" in by_key

    def test_empty_raises(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            self.plugin.input_schema(failure_times=[])
