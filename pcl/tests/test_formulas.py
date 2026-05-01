"""Formula evaluation tests — safety guards and arithmetic."""

import pytest

from pcl.formulas import evaluate_pcl_formula, extract_slugs


class TestExtractSlugs:
    def test_single_slug(self):
        assert extract_slugs("[press-a-ct]") == ["press-a-ct"]

    def test_multiple_slugs(self):
        assert extract_slugs("[avail] * [perf] * [qual]") == ["avail", "perf", "qual"]

    def test_no_slugs(self):
        assert extract_slugs("42 + 3") == []

    def test_duplicate_slugs_deduped(self):
        assert extract_slugs("[x] + [x]") == ["x"]


class TestEvaluatePclFormula:
    def test_simple_arithmetic(self):
        result = evaluate_pcl_formula("[a] + [b]", {"a": 10.0, "b": 20.0})
        assert result == 30.0

    def test_oee_formula(self):
        result = evaluate_pcl_formula(
            "[avail] * [perf] * [qual]",
            {"avail": 0.90, "perf": 0.85, "qual": 0.95},
        )
        assert result == pytest.approx(0.72675, abs=0.001)

    def test_takt_formula(self):
        result = evaluate_pcl_formula(
            "[available_time] / [demand]",
            {"available_time": 28800.0, "demand": 480.0},
        )
        assert result == 60.0

    def test_safe_functions(self):
        result = evaluate_pcl_formula("max([a], [b])", {"a": 10.0, "b": 20.0})
        assert result == 20.0

    def test_sqrt_function(self):
        result = evaluate_pcl_formula("sqrt([x])", {"x": 16.0})
        assert result == 4.0

    def test_hyphenated_slugs(self):
        result = evaluate_pcl_formula("[press-a-ct] + 1", {"press-a-ct": 44.0})
        assert result == 45.0

    def test_unknown_slug_raises(self):
        with pytest.raises(ValueError, match="Unknown variable"):
            evaluate_pcl_formula("[missing]", {})

    def test_division_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            evaluate_pcl_formula("[a] / [b]", {"a": 10.0, "b": 0.0})

    def test_rejects_imports(self):
        with pytest.raises(ValueError):
            evaluate_pcl_formula("__import__('os')", {})

    def test_rejects_attribute_access(self):
        with pytest.raises(ValueError):
            evaluate_pcl_formula("[a].__class__", {"a": 1.0})

    def test_max_length_guard(self):
        with pytest.raises(ValueError, match="too long"):
            evaluate_pcl_formula("[a] + " * 200, {"a": 1.0})

    def test_max_depth_guard(self):
        formula = "[a]"
        for _ in range(25):
            formula = f"({formula} + [a])"
        with pytest.raises(ValueError, match="too deep|too complex"):
            evaluate_pcl_formula(formula, {"a": 1.0})

    def test_negation(self):
        result = evaluate_pcl_formula("-[a]", {"a": 5.0})
        assert result == -5.0

    def test_constants_in_formula(self):
        result = evaluate_pcl_formula("[a] * 100", {"a": 0.85})
        assert result == pytest.approx(85.0, abs=0.01)
