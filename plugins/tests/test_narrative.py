from django.test import TestCase

from syn.plugins.registry import PluginRegistry


class TestNarrativePlugin(TestCase):
    def setUp(self):
        from plugins.narrative_device import NarrativePlugin

        self.registry = PluginRegistry()
        self.registry.register(NarrativePlugin)
        self.plugin = self.registry.get("narrative")

    def test_basic_narrative(self):
        outputs = self.plugin.execute(
            {
                "analysis_type": "capability",
                "statistics": {"cpk": 1.45, "cp": 1.5, "p_value": 0.23, "n": 50},
            },
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert "verdict" in by_key
        assert "body" in by_key
        assert "narrative" in by_key

    def test_empty_stats_raises(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            self.plugin.input_schema(analysis_type="test", statistics={})
