from django.test import TestCase

from syn.plugins.registry import PluginRegistry


class TestTextInputPlugin(TestCase):
    def setUp(self):
        from plugins.text_input import TextInputPlugin

        self.registry = PluginRegistry()
        self.registry.register(TextInputPlugin)
        self.plugin = self.registry.get("text_input")

    def test_basic_output(self):
        outputs = self.plugin.execute(
            {"content": "Part surface finish exceeds tolerance", "subtype": "problem_statement"},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        assert len(outputs) == 1
        assert outputs[0].key == "problem_statement"
        assert outputs[0].output_type == "text"
        assert outputs[0].value == {"text": "Part surface finish exceeds tolerance"}

    def test_observation_subtype(self):
        outputs = self.plugin.execute(
            {"content": "Cycle time increasing since Monday", "subtype": "observation"},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        assert outputs[0].key == "observation"
        assert outputs[0].value["text"] == "Cycle time increasing since Monday"

    def test_custom_subtype(self):
        outputs = self.plugin.execute(
            {"content": "Goal: reduce lead time 30%", "subtype": "goal"},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        assert outputs[0].key == "goal"

    def test_empty_content_raises(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            self.plugin.input_schema(content="", subtype="problem_statement")

    def test_default_subtype(self):
        outputs = self.plugin.execute(
            {"content": "Some text", "subtype": "note"},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        assert outputs[0].key == "note"
