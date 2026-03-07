"""Coverage tests for PBS (Process Belief System) analyses — CAL-001 §6 / TST-001 §10.

Standard: CAL-001 §6 (Statistical Correctness Verification)
Compliance: SOC 2 CC4.1
<!-- test: agents_api.tests.test_pbs_coverage -->
"""

import numpy as np
import pandas as pd
from django.test import TestCase

# Shared test data
NORMAL_60 = list(np.random.RandomState(42).normal(50, 5, 60))


def _run(analysis_id, config, data_dict):
    """Run PBS analysis — no exception masking (TST-001 §11.6)."""
    from agents_api.pbs_engine import run_pbs

    df = pd.DataFrame(data_dict)
    return run_pbs(df, analysis_id, config)


def _check_schema(tc, r):
    """Verify output schema per CAL-001 §6 / TST-001 §10.6."""
    aid = tc._testMethodName
    tc.assertIsInstance(r, dict, f"{aid} did not return a dict")
    tc.assertIn("plots", r, f"{aid} missing 'plots' key")
    tc.assertIsInstance(r["plots"], list, f"{aid} plots is not a list")
    tc.assertIn("summary", r, f"{aid} missing 'summary' key")


class PBSCoverageTest(TestCase):
    """PBS analysis IDs — all require column, USL, LSL config."""

    def test_pbs_full(self):
        r = _run(
            "pbs_full",
            {"column": "y", "USL": 65, "LSL": 35},
            {"y": NORMAL_60},
        )
        _check_schema(self, r)

    def test_pbs_belief(self):
        r = _run(
            "pbs_belief",
            {"column": "y", "USL": 65, "LSL": 35},
            {"y": NORMAL_60},
        )
        _check_schema(self, r)

    def test_pbs_edetector(self):
        r = _run(
            "pbs_edetector",
            {"column": "y", "USL": 65, "LSL": 35},
            {"y": NORMAL_60},
        )
        _check_schema(self, r)

    def test_pbs_evidence(self):
        r = _run(
            "pbs_evidence",
            {"column": "y", "USL": 65, "LSL": 35},
            {"y": NORMAL_60},
        )
        _check_schema(self, r)

    def test_pbs_predictive(self):
        r = _run(
            "pbs_predictive",
            {"column": "y", "USL": 65, "LSL": 35},
            {"y": NORMAL_60},
        )
        _check_schema(self, r)

    def test_pbs_adaptive(self):
        r = _run(
            "pbs_adaptive",
            {"column": "y", "USL": 65, "LSL": 35},
            {"y": NORMAL_60},
        )
        _check_schema(self, r)

    def test_pbs_cpk(self):
        r = _run(
            "pbs_cpk",
            {"column": "y", "USL": 65, "LSL": 35},
            {"y": NORMAL_60},
        )
        _check_schema(self, r)

    def test_pbs_cpk_traj(self):
        r = _run(
            "pbs_cpk_traj",
            {"column": "y", "USL": 65, "LSL": 35},
            {"y": NORMAL_60},
        )
        _check_schema(self, r)

    def test_pbs_health(self):
        r = _run(
            "pbs_health",
            {"column": "y", "USL": 65, "LSL": 35},
            {"y": NORMAL_60},
        )
        _check_schema(self, r)
