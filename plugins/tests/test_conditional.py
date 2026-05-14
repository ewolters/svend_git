from django.test import TestCase

from syn.plugins.registry import PluginRegistry


class TestConditionalPlugin(TestCase):
    def setUp(self):
        from plugins.conditional import ConditionalPlugin

        self.registry = PluginRegistry()
        self.registry.register(ConditionalPlugin)
        self.plugin = self.registry.get("conditional")

    def test_pass_when_above_threshold(self):
        outputs = self.plugin.execute(
            {"value": 1.5, "operator": ">=", "threshold": 1.33},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["result"].value is True
        assert by_key["pass_value"].value == 1.5
        assert by_key["fail_value"].value is None

    def test_fail_when_below_threshold(self):
        outputs = self.plugin.execute(
            {"value": 0.9, "operator": ">=", "threshold": 1.33},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["result"].value is False
        assert by_key["pass_value"].value is None
        assert by_key["fail_value"].value == 0.9

    def test_less_than_operator(self):
        outputs = self.plugin.execute(
            {"value": 5.0, "operator": "<", "threshold": 10.0},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["result"].value is True

    def test_equals_operator(self):
        outputs = self.plugin.execute(
            {"value": 3.0, "operator": "==", "threshold": 3.0},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["result"].value is True

    def test_invalid_operator_raises(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            self.plugin.input_schema(value=1.0, operator="LIKE", threshold=2.0)

    def test_label_in_output(self):
        outputs = self.plugin.execute(
            {"value": 1.5, "operator": ">=", "threshold": 1.33, "label": "Cpk check"},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["summary"].value["label"] == "Cpk check"
