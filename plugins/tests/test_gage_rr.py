"""Tests for GageRRPlugin — crossed Gage R&R measurement system analysis."""

from django.test import TestCase

from syn.plugins.registry import PluginRegistry


def _build_crossed_data(n_ops=3, n_parts=10, n_reps=2, base=10.0, op_bias=0.05, part_spread=0.5, noise=0.02):
    """Generate realistic balanced crossed Gage R&R data.

    Each part has a true value, each operator has a small bias, and
    within-cell variation models repeatability error.
    """
    import random

    random.seed(42)

    part_true = {p: base + part_spread * (p - (n_parts / 2)) for p in range(1, n_parts + 1)}
    op_bias_val = {o: op_bias * (o - 1) for o in range(1, n_ops + 1)}

    measurements, parts, operators = [], [], []
    for p in range(1, n_parts + 1):
        for o in range(1, n_ops + 1):
            for _ in range(n_reps):
                val = part_true[p] + op_bias_val[o] + random.gauss(0, noise)
                measurements.append(round(val, 4))
                parts.append(p)
                operators.append(o)

    return measurements, parts, operators


class TestGageRRPlugin(TestCase):
    def setUp(self):
        from plugins.gage_rr_device import GageRRPlugin

        self.registry = PluginRegistry()
        self.registry.register(GageRRPlugin)
        self.plugin = self.registry.get("gage_rr")

    def test_crossed_gage_rr(self):
        """3 operators × 10 parts × 2 replicates — full study."""
        measurements, parts, operators = _build_crossed_data(n_ops=3, n_parts=10, n_reps=2)

        outputs = self.plugin.execute(
            {"measurements": measurements, "parts": parts, "operators": operators},
            {"job_id": "test", "actor": "test@svend.ai"},
        )

        by_key = {o.key: o for o in outputs}

        # All expected output keys are present
        self.assertIn("repeatability", by_key)
        self.assertIn("reproducibility", by_key)
        self.assertIn("grr_pct", by_key)
        self.assertIn("part_variation_pct", by_key)
        self.assertIn("ndc", by_key)
        self.assertIn("result", by_key)

        # Variances are non-negative floats
        self.assertGreaterEqual(by_key["repeatability"].value, 0.0)
        self.assertGreaterEqual(by_key["reproducibility"].value, 0.0)

        # Percentages sum close to 100 (GRR% + Part% ≈ 100)
        grr = by_key["grr_pct"].value
        pv = by_key["part_variation_pct"].value
        self.assertAlmostEqual(grr + pv, 100.0, delta=1.0)

        # Result text output is a dict with anova_table
        result_val = by_key["result"].value
        self.assertIsInstance(result_val, dict)
        self.assertIn("anova_table", result_val)
        self.assertEqual(result_val["design"], "crossed")
        self.assertEqual(result_val["n_operators"], 3)
        self.assertEqual(result_val["n_parts"], 10)
        self.assertEqual(result_val["n_replicates"], 2)

    def test_ndc_output(self):
        """NDC (number of distinct categories) is present and >= 1."""
        measurements, parts, operators = _build_crossed_data(n_ops=2, n_parts=8, n_reps=3)

        outputs = self.plugin.execute(
            {"measurements": measurements, "parts": parts, "operators": operators},
            {"job_id": "test", "actor": "test@svend.ai"},
        )

        by_key = {o.key: o for o in outputs}
        self.assertIn("ndc", by_key)
        ndc = by_key["ndc"].value
        self.assertIsInstance(ndc, int)
        self.assertGreaterEqual(ndc, 1)

    def test_too_few_parts(self):
        """Validation error when only 1 distinct part is provided."""
        from pydantic import ValidationError

        # All measurements on the same part
        with self.assertRaises(ValidationError):
            self.plugin.input_schema(
                measurements=[10.0, 10.1, 10.05, 9.98],
                parts=[1, 1, 1, 1],
                operators=[1, 2, 1, 2],
            )
