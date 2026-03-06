"""Golden file tests — verify analysis outputs match known-correct reference values.

Each golden file in agents_api/tests/golden/ contains:
- case_id: links to CalibrationCase
- analysis_type, analysis_id: dispatch info
- config: analysis configuration
- expected: dict of {result_path: {value, tolerance}} pairs

Standard: CAL-001 §6 (Statistical Correctness Verification)
Compliance: SOC 2 CC4.1, ISO/IEC 17025:2017
"""

import json
from pathlib import Path

import pandas as pd
from django.test import TestCase

from agents_api.calibration import get_reference_pool

GOLDEN_DIR = Path(__file__).parent / "golden"


def _extract_nested(d, key):
    """Extract value from nested dict using dot notation and bracket indexing.

    Supports: "statistics.pairs[0].meandiff", "statistics.p_value", etc.
    """
    import re

    parts = key.split(".")
    current = d
    for part in parts:
        if current is None:
            return None
        # Handle bracket indexing: e.g. "pairs[0]"
        bracket = re.match(r"^(\w+)\[(\d+)\]$", part)
        if bracket:
            name, idx = bracket.group(1), int(bracket.group(2))
            if isinstance(current, dict):
                current = current.get(name)
            else:
                return None
            if isinstance(current, list) and idx < len(current):
                current = current[idx]
            else:
                return None
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _run_analysis(analysis_type, analysis_id, config, data_dict):
    """Run an analysis and return the result dict."""
    df = pd.DataFrame(data_dict) if data_dict else pd.DataFrame()

    if analysis_type == "stats":
        from agents_api.dsw.stats import run_statistical_analysis

        return run_statistical_analysis(df, analysis_id, config)
    elif analysis_type == "bayesian":
        from agents_api.dsw.bayesian import run_bayesian_analysis

        return run_bayesian_analysis(df, analysis_id, config)
    elif analysis_type == "spc":
        from agents_api.dsw.spc import run_spc_analysis

        return run_spc_analysis(df, analysis_id, config)
    elif analysis_type == "reliability":
        from agents_api.dsw.reliability import run_reliability_analysis

        return run_reliability_analysis(df, analysis_id, config)
    elif analysis_type == "ml":
        from agents_api.dsw.ml import run_ml_analysis

        return run_ml_analysis(df, analysis_id, config, user=None)
    elif analysis_type == "simulation":
        from agents_api.dsw.simulation import run_simulation

        return run_simulation(df, analysis_id, config, user=None)
    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}")


class GoldenFileTest(TestCase):
    """Test each golden file against its calibration case data."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Build lookup from case_id to CalibrationCase
        cls.cases = {c.case_id: c for c in get_reference_pool()}

    def _check_golden(self, golden_path):
        """Verify one golden file."""
        with open(golden_path) as f:
            golden = json.load(f)

        case_id = golden["case_id"]
        case = self.cases.get(case_id)
        self.assertIsNotNone(case, f"No calibration case for {case_id}")

        # Run the analysis using case data
        result = _run_analysis(case.analysis_type, case.analysis_id, case.config, case.data)
        self.assertIsNotNone(result, f"{case_id}: analysis returned None")

        # Check each expected value
        expected = golden.get("expected", {})
        failures = []

        for key, spec in expected.items():
            # Handle "_contains" assertions: check substring in result field
            if key.endswith("_contains"):
                field = key.rsplit("_contains", 1)[0]
                actual_str = str(result.get(field, ""))
                if spec not in actual_str:
                    failures.append(f"  {key}: '{spec}' not found in {field}")
                continue

            actual = _extract_nested(result, key)
            if actual is None:
                failures.append(f"  {key}: not found in result")
                continue

            # Handle boolean expected values
            if isinstance(spec.get("value"), bool):
                if actual != spec["value"]:
                    failures.append(f"  {key}: expected {spec['value']}, got {actual}")
                continue

            try:
                actual_f = float(actual)
                expected_f = float(spec["value"])
                tolerance = float(spec["tolerance"])
                deviation = abs(actual_f - expected_f)
                if deviation > tolerance:
                    failures.append(f"  {key}: |{actual_f:.6f} - {expected_f}| = {deviation:.6f} > {tolerance}")
            except (TypeError, ValueError):
                failures.append(f"  {key}: cannot compare {actual} vs {spec['value']}")

        if failures:
            self.fail(f"Golden file {golden_path.name} ({case_id}) failed:\n" + "\n".join(failures))


def _make_golden_test(golden_path):
    """Factory: create a test method for one golden file."""

    def test_method(self):
        self._check_golden(golden_path)

    test_method.__doc__ = f"Golden: {golden_path.stem}"
    return test_method


# Dynamically generate test methods for each golden file
for _gf in sorted(GOLDEN_DIR.glob("*.json")):
    _test_name = f"test_golden_{_gf.stem}"
    setattr(GoldenFileTest, _test_name, _make_golden_test(_gf))


class GoldenFileInventoryTest(TestCase):
    """Verify golden file inventory meets CAL-001 requirements."""

    def test_golden_directory_exists(self):
        self.assertTrue(GOLDEN_DIR.exists(), "Golden file directory missing")

    def test_minimum_golden_files(self):
        """CAL-001 §6.3: Priority 1 requires 42+ golden files."""
        count = len(list(GOLDEN_DIR.glob("*.json")))
        self.assertGreaterEqual(count, 42, f"Only {count} golden files (need 42+)")

    def test_golden_files_have_case_ids(self):
        """Every golden file must reference a valid calibration case."""
        cases = {c.case_id for c in get_reference_pool()}
        for gf in GOLDEN_DIR.glob("*.json"):
            with open(gf) as f:
                data = json.load(f)
            self.assertIn("case_id", data, f"{gf.name} missing case_id")
            self.assertIn(data["case_id"], cases, f"{gf.name} references unknown case {data['case_id']}")

    def test_priority_1_coverage(self):
        """Check Priority 1 analysis types have golden files."""
        priority_1 = {
            "ttest",
            "ttest2",
            "paired_t",
            "anova",
            "correlation",
            "chi2",
            "fisher_exact",
            "mann_whitney",
            "kruskal",
            "wilcoxon",
            "regression",
            "logistic",
            "stepwise",
            "imr",
            "xbar_r",
            "xbar_s",
            "p_chart",
            "cusum",
            "ewma",
            "capability",
            "bayes_ttest",
            "bayes_ab",
            "bayes_regression",
            "weibull",
            "kaplan_meier",
        }
        covered = set()
        for gf in GOLDEN_DIR.glob("*.json"):
            with open(gf) as f:
                data = json.load(f)
            covered.add(data.get("analysis_id", ""))

        missing = priority_1 - covered
        self.assertEqual(missing, set(), f"Priority 1 analyses without golden files: {missing}")
