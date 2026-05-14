from django.test import TestCase

from syn.plugins.registry import PluginRegistry


class TestDOEDesignPlugin(TestCase):
    def setUp(self):
        from plugins.doe_design import DOEDesignPlugin

        self.registry = PluginRegistry()
        self.registry.register(DOEDesignPlugin)
        self.plugin = self.registry.get("doe_design")

    def test_full_factorial(self):
        outputs = self.plugin.execute(
            {
                "design_type": "full_factorial",
                "factors": [
                    {"name": "Temperature", "low": 150, "high": 200},
                    {"name": "Pressure", "low": 50, "high": 100},
                ],
            },
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["n_runs"].value == 4.0  # 2^2
        assert len(by_key["design_matrix"].value["matrix"]) == 4

    def test_fractional_factorial(self):
        outputs = self.plugin.execute(
            {
                "design_type": "fractional_factorial",
                "factors": [
                    {"name": "A", "low": -1, "high": 1},
                    {"name": "B", "low": -1, "high": 1},
                    {"name": "C", "low": -1, "high": 1},
                    {"name": "D", "low": -1, "high": 1},
                ],
                "resolution": 3,
            },
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["n_runs"].value < 16  # fractional = fewer than full

    def test_central_composite(self):
        outputs = self.plugin.execute(
            {
                "design_type": "central_composite",
                "factors": [
                    {"name": "Temp", "low": 150, "high": 200},
                    {"name": "Time", "low": 10, "high": 30},
                ],
                "center_points": 5,
            },
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["n_runs"].value > 4  # factorial + axial + center

    def test_invalid_design_type(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            self.plugin.input_schema(
                design_type="bogus", factors=[{"name": "A", "low": 0, "high": 1}, {"name": "B", "low": 0, "high": 1}]
            )

    def test_single_factor_rejected(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            self.plugin.input_schema(design_type="full_factorial", factors=[{"name": "A", "low": 0, "high": 1}])
