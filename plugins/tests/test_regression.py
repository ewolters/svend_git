from django.test import TestCase

from syn.plugins.registry import PluginRegistry


class TestRegressionPlugin(TestCase):
    def setUp(self):
        from plugins.regression_device import RegressionPlugin

        self.registry = PluginRegistry()
        self.registry.register(RegressionPlugin)
        self.plugin = self.registry.get("regression")

    def test_linear_regression(self):
        x = [[1], [2], [3], [4], [5]]
        y = [2.1, 3.9, 6.2, 7.8, 10.1]
        outputs = self.plugin.execute(
            {"x": x, "y": y},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["r_squared"].value > 0.95  # strong linear relationship

    def test_polynomial_regression(self):
        x = [[1], [2], [3], [4], [5]]
        y = [1, 4, 9, 16, 25]
        outputs = self.plugin.execute(
            {"x": x, "y": y, "degree": 2},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["r_squared"].value > 0.99  # perfect quadratic

    def test_too_few_observations(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            self.plugin.input_schema(x=[[1]], y=[1])
