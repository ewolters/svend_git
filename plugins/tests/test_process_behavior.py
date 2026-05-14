from django.test import TestCase

from syn.plugins.registry import PluginRegistry


class TestProcessBehaviorPlugin(TestCase):
    def setUp(self):
        from plugins.process_behavior_device import ProcessBehaviorPlugin

        self.registry = PluginRegistry()
        self.registry.register(ProcessBehaviorPlugin)
        self.plugin = self.registry.get("process_behavior")

    def test_stable_process(self):
        import random

        random.seed(42)
        data = [50.0 + random.gauss(0, 0.5) for _ in range(50)]
        outputs = self.plugin.execute(
            {"data": data},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert "n_signals" in by_key
        assert "chart_data" in by_key

    def test_shift_detection(self):
        # Stable at 50, then shift to 55
        data = [50.0 + (i * 0.01) for i in range(30)] + [55.0 + (i * 0.01) for i in range(30)]
        outputs = self.plugin.execute(
            {"data": data},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        # Should detect at least one changepoint or signal
        assert by_key["n_signals"].value > 0 or by_key["n_changepoints"].value > 0

    def test_too_few_observations(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            self.plugin.input_schema(data=[1, 2, 3])
