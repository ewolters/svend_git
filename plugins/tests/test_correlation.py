from django.test import TestCase

from syn.plugins.registry import PluginRegistry


class TestCorrelationPlugin(TestCase):
    def setUp(self):
        from plugins.correlation_device import CorrelationPlugin

        self.registry = PluginRegistry()
        self.registry.register(CorrelationPlugin)
        self.plugin = self.registry.get("correlation")

    def test_strong_positive(self):
        outputs = self.plugin.execute(
            {"data": {"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]}},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["r"].value > 0.99

    def test_spearman(self):
        outputs = self.plugin.execute(
            {
                "data": {"x": [1, 2, 3, 4, 5], "y": [1, 4, 9, 16, 25]},
                "method": "spearman",
            },
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["r"].value > 0.9

    def test_too_few_vars_raises(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            self.plugin.input_schema(data={"x": [1, 2, 3]})

    def test_mismatched_lengths_raises(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            self.plugin.input_schema(
                data={"x": [1, 2, 3], "y": [1, 2]}  # Different lengths
            )

    def test_too_few_observations_raises(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            self.plugin.input_schema(
                data={"x": [1, 2], "y": [3, 4]}  # Only 2 observations
            )

    def test_invalid_method_raises(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            self.plugin.input_schema(data={"x": [1, 2, 3], "y": [4, 5, 6]}, method="invalid")
