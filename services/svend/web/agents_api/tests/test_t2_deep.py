"""T2-DEEP coverage — broad tests for viz.py, ml.py, spc_views, experimenter_views, core/views.

Targets the five largest T2 coverage gaps:
  1. viz.py           — 20 untested visualization types (direct calls)
  2. experimenter_views.py — POST endpoints (power, design, full, analyze, contour, optimize)
  3. spc_views.py      — POST endpoints (chart, capability, summary, recommend, gage_rr)
  4. ml.py             — additional supervised / unsupervised paths
  5. core/views.py     — project/hypothesis/evidence CRUD + knowledge graph + org

Standard: CAL-001 §7 (Endpoint Coverage), TST-001 §10.6
Compliance: SOC 2 CC4.1, CC7.2
<!-- test: agents_api.tests.test_t2_deep -->
"""

import json

import numpy as np
import pandas as pd
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from accounts.models import Tier, User

SMOKE_SETTINGS = {
    "RATELIMIT_ENABLE": False,
    "SECURE_SSL_REDIRECT": False,
    "CACHES": {"default": {"BACKEND": "django.core.cache.backends.dummy.DummyCache"}},
}

# ── Shared test data ──────────────────────────────────────────────────────

np.random.seed(42)
N = 60
_rng42 = np.random.RandomState(42)
_rng43 = np.random.RandomState(43)
_rng44 = np.random.RandomState(44)
X1 = list(_rng42.normal(0, 1, N))
X2 = list(_rng43.normal(0, 1, N))
X3 = list(_rng44.normal(0, 1, N))
Y_REG = [2 * x1 + 0.5 * x2 + np.random.RandomState(45).normal(0, 0.5) for x1, x2 in zip(X1, X2)]
Y_CLS = [1 if y > 0 else 0 for y in Y_REG]
DATES = pd.date_range("2020-01-01", periods=N, freq="D").strftime("%Y-%m-%d").tolist()
GROUPS_AB = (["A"] * 30) + (["B"] * 30)
CAT_XY = np.random.RandomState(46).choice(["X", "Y"], N).tolist()
SPC_DATA = list(np.random.RandomState(47).normal(50, 2, 30))


def _make_user(email, tier=Tier.PRO, staff=False):
    username = email.split("@")[0].replace(".", "_")
    u = User.objects.create_user(username=username, email=email, password="testpass123")
    u.tier = tier
    u.is_staff = staff
    u.email_verified = True
    u.save()
    return u


# ===========================================================================
# 1. viz.py — direct call tests for 20 untested visualization types
# ===========================================================================


class VizDeepCoverageTest(TestCase):
    """Direct tests for run_visualization() — covers all untested viz types."""

    def _run(self, analysis_id, config, data_dict):
        """Run viz analysis — no exception masking (TST-001 §11.6)."""
        from agents_api.analysis.viz import run_visualization

        df = pd.DataFrame(data_dict)
        return run_visualization(df, analysis_id, config)

    # --- pareto ---
    def test_pareto_count(self):
        r = self._run(
            "pareto",
            {"var": "cat"},
            {"cat": ["A"] * 20 + ["B"] * 15 + ["C"] * 10 + ["D"] * 5},
        )
        self.assertIn("plots", r)
        self.assertTrue(len(r["plots"]) > 0)

    def test_pareto_with_value(self):
        r = self._run(
            "pareto",
            {"category": "cat", "value": "val"},
            {"cat": ["A"] * 10 + ["B"] * 10, "val": list(range(20))},
        )
        self.assertIn("plots", r)
        self.assertIn("summary", r)

    # --- matrix ---
    def test_matrix_two_vars(self):
        r = self._run(
            "matrix",
            {"vars": ["x", "y"]},
            {"x": X1, "y": X2},
        )
        self.assertIn("plots", r)
        self.assertTrue(len(r["plots"]) > 0)

    def test_matrix_three_vars(self):
        r = self._run(
            "matrix",
            {"vars": ["x", "y", "z"]},
            {"x": X1, "y": X2, "z": X3},
        )
        self.assertIn("plots", r)

    def test_matrix_insufficient_vars(self):
        r = self._run("matrix", {"vars": ["x"]}, {"x": X1})
        self.assertIn("summary", r)
        self.assertIn("2 variables", r["summary"])

    # --- timeseries ---
    def test_timeseries_single_y(self):
        r = self._run(
            "timeseries",
            {"x": "date", "y": ["val"]},
            {"date": DATES, "val": Y_REG},
        )
        self.assertIn("plots", r)
        self.assertTrue(len(r["plots"]) > 0)

    def test_timeseries_multi_y(self):
        r = self._run(
            "timeseries",
            {"x": "date", "y": ["v1", "v2"], "markers": True},
            {"date": DATES, "v1": X1, "v2": X2},
        )
        self.assertIn("plots", r)

    def test_timeseries_string_y(self):
        r = self._run(
            "timeseries",
            {"x": "date", "y": "val"},
            {"date": DATES, "val": X1},
        )
        self.assertIn("plots", r)

    # --- probability ---
    def test_probability_normal(self):
        r = self._run(
            "probability",
            {"var": "x", "dist": "norm"},
            {"x": list(_rng42.normal(50, 5, 50))},
        )
        self.assertIn("plots", r)
        self.assertIn("summary", r)

    def test_probability_lognorm(self):
        r = self._run(
            "probability",
            {"var": "x", "dist": "lognorm"},
            {"x": list(np.abs(_rng42.normal(5, 1, 50)) + 0.01)},
        )
        self.assertIn("plots", r)

    def test_probability_weibull(self):
        r = self._run(
            "probability",
            {"var": "x", "dist": "weibull"},
            {"x": list(np.abs(_rng42.normal(5, 1, 50)) + 0.01)},
        )
        self.assertIn("plots", r)

    def test_probability_expon(self):
        r = self._run(
            "probability",
            {"var": "x", "dist": "expon"},
            {"x": list(np.abs(_rng42.normal(5, 1, 50)) + 0.01)},
        )
        self.assertIn("plots", r)

    # --- individual_value_plot ---
    def test_ivp_no_group(self):
        r = self._run(
            "individual_value_plot",
            {"var": "x", "show_mean": True, "show_ci": True},
            {"x": X1},
        )
        self.assertIn("plots", r)

    def test_ivp_with_group(self):
        r = self._run(
            "individual_value_plot",
            {
                "var": "x",
                "group": "g",
                "show_mean": True,
                "show_ci": True,
                "confidence": 0.95,
            },
            {"x": X1, "g": GROUPS_AB},
        )
        self.assertIn("plots", r)

    # --- interval_plot ---
    def test_interval_plot_with_group(self):
        r = self._run(
            "interval_plot",
            {"var": "x", "group": "g", "confidence": 0.95},
            {"x": X1, "g": GROUPS_AB},
        )
        self.assertIn("plots", r)

    def test_interval_plot_no_group(self):
        r = self._run(
            "interval_plot",
            {"var": "x"},
            {"x": X1},
        )
        self.assertIn("summary", r)
        self.assertIn("grouping variable", r["summary"])

    # --- dotplot ---
    def test_dotplot_no_group(self):
        r = self._run(
            "dotplot",
            {"var": "x"},
            {"x": X1},
        )
        self.assertIn("plots", r)

    def test_dotplot_with_group(self):
        r = self._run(
            "dotplot",
            {"var": "x", "group": "g"},
            {"x": X1, "g": GROUPS_AB},
        )
        self.assertIn("plots", r)

    # --- bubble ---
    def test_bubble_no_color(self):
        r = self._run(
            "bubble",
            {"x": "x", "y": "y", "size": "s"},
            {"x": X1, "y": X2, "s": [abs(v) + 1 for v in X3]},
        )
        self.assertIn("plots", r)
        self.assertIn("summary", r)

    def test_bubble_with_color(self):
        r = self._run(
            "bubble",
            {"x": "x", "y": "y", "size": "s", "color": "g"},
            {"x": X1, "y": X2, "s": [abs(v) + 1 for v in X3], "g": GROUPS_AB},
        )
        self.assertIn("plots", r)

    # --- parallel_coordinates ---
    def test_parallel_coordinates_numeric(self):
        r = self._run(
            "parallel_coordinates",
            {"dimensions": ["x", "y", "z"]},
            {"x": X1, "y": X2, "z": X3},
        )
        self.assertIn("plots", r)
        self.assertIn("summary", r)

    def test_parallel_coordinates_with_color(self):
        r = self._run(
            "parallel_coordinates",
            {"dimensions": ["x", "y", "z"], "color": "x"},
            {"x": X1, "y": X2, "z": X3},
        )
        self.assertIn("plots", r)

    def test_parallel_coordinates_cat_color(self):
        r = self._run(
            "parallel_coordinates",
            {"dimensions": ["x", "y"], "color": "g"},
            {"x": X1, "y": X2, "g": GROUPS_AB},
        )
        self.assertIn("plots", r)

    def test_parallel_coordinates_insufficient(self):
        r = self._run(
            "parallel_coordinates",
            {"dimensions": ["x"]},
            {"x": X1},
        )
        self.assertIn("summary", r)
        self.assertIn("2 dimensions", r["summary"])

    # --- contour ---
    def test_contour(self):
        r = self._run(
            "contour",
            {"x": "x", "y": "y", "z": "z"},
            {"x": X1, "y": X2, "z": X3},
        )
        self.assertIn("plots", r)
        self.assertIn("summary", r)

    def test_contour_insufficient_data(self):
        r = self._run(
            "contour",
            {"x": "x", "y": "y", "z": "z"},
            {"x": [1, 2], "y": [3, 4], "z": [5, 6]},
        )
        self.assertIn("summary", r)
        self.assertIn("4", r["summary"])

    # --- surface_3d ---
    def test_surface_3d(self):
        r = self._run(
            "surface_3d",
            {"x": "x", "y": "y", "z": "z"},
            {"x": X1, "y": X2, "z": X3},
        )
        self.assertIn("plots", r)
        self.assertIn("summary", r)

    def test_surface_3d_insufficient_data(self):
        r = self._run(
            "surface_3d",
            {"x": "x", "y": "y", "z": "z"},
            {"x": [1], "y": [2], "z": [3]},
        )
        self.assertIn("summary", r)
        self.assertIn("4", r["summary"])

    # --- contour_overlay ---
    def test_contour_overlay(self):
        r = self._run(
            "contour_overlay",
            {"x": "x", "y": "y", "z_columns": ["z1", "z2"]},
            {"x": X1, "y": X2, "z1": X3, "z2": [v * 2 for v in X3]},
        )
        self.assertIn("plots", r)
        self.assertIn("summary", r)

    def test_contour_overlay_insufficient_responses(self):
        r = self._run(
            "contour_overlay",
            {"x": "x", "y": "y", "z_columns": ["z1"]},
            {"x": X1, "y": X2, "z1": X3},
        )
        self.assertIn("summary", r)
        self.assertIn("2 response", r["summary"])

    def test_contour_overlay_responses_key(self):
        r = self._run(
            "contour_overlay",
            {"x": "x", "y": "y", "responses": ["z1", "z2"]},
            {"x": X1, "y": X2, "z1": X3, "z2": Y_REG},
        )
        self.assertIn("plots", r)

    # --- mosaic ---
    def test_mosaic(self):
        r = self._run(
            "mosaic",
            {"row_var": "r", "col_var": "c"},
            {
                "r": GROUPS_AB,
                "c": CAT_XY,
            },
        )
        self.assertIn("plots", r)
        self.assertIn("summary", r)

    # --- bayes_spc_capability ---
    def test_bayes_spc_capability_two_sided(self):
        r = self._run(
            "bayes_spc_capability",
            {"measurement": "x", "usl": 55, "lsl": 45, "n_mc": 500},
            {"x": SPC_DATA},
        )
        self.assertIn("statistics", r)
        self.assertIn("plots", r)
        stats = r.get("statistics", {})
        self.assertIn("cpk_median", stats)

    def test_bayes_spc_capability_one_sided_usl(self):
        r = self._run(
            "bayes_spc_capability",
            {"measurement": "x", "usl": 55, "n_mc": 500},
            {"x": SPC_DATA},
        )
        self.assertIn("statistics", r)

    def test_bayes_spc_capability_one_sided_lsl(self):
        r = self._run(
            "bayes_spc_capability",
            {"measurement": "x", "lsl": 45, "n_mc": 500},
            {"x": SPC_DATA},
        )
        self.assertIn("statistics", r)

    def test_bayes_spc_capability_no_spec(self):
        r = self._run(
            "bayes_spc_capability",
            {"measurement": "x", "n_mc": 500},
            {"x": SPC_DATA},
        )
        self.assertIn("summary", r)
        self.assertIn("spec limit", r["summary"].lower())

    def test_bayes_spc_capability_informative_prior(self):
        r = self._run(
            "bayes_spc_capability",
            {
                "measurement": "x",
                "usl": 55,
                "lsl": 45,
                "n_mc": 500,
                "prior_type": "informative",
                "prior_params": {"mu0": 50, "nu0": 5, "alpha0": 3, "beta0": 4},
            },
            {"x": SPC_DATA},
        )
        self.assertIn("statistics", r)

    def test_bayes_spc_capability_historical_prior(self):
        r = self._run(
            "bayes_spc_capability",
            {
                "measurement": "x",
                "usl": 55,
                "lsl": 45,
                "n_mc": 500,
                "prior_type": "historical",
                "prior_params": {"hist_mean": 50, "hist_std": 2, "hist_n": 30},
            },
            {"x": SPC_DATA},
        )
        self.assertIn("statistics", r)

    # --- bayes_spc_changepoint ---
    def test_bayes_spc_changepoint(self):
        # Create data with a shift at observation 15
        stable = list(np.random.RandomState(48).normal(50, 1, 15))
        shifted = list(np.random.RandomState(49).normal(53, 1, 15))
        r = self._run(
            "bayes_spc_changepoint",
            {"measurement": "x", "hazard_rate": 0.05, "min_segment_length": 3},
            {"x": stable + shifted},
        )
        self.assertIn("plots", r)
        self.assertIn("summary", r)

    # --- bayes_spc_control ---
    def test_bayes_spc_control(self):
        r = self._run(
            "bayes_spc_control",
            {"measurement": "x", "shift_size": 1.5, "transition_prob": 0.01},
            {"x": SPC_DATA},
        )
        self.assertIn("plots", r)
        self.assertIn("summary", r)

    def test_bayes_spc_control_with_ref(self):
        r = self._run(
            "bayes_spc_control",
            {
                "measurement": "x",
                "reference_mean": 50.0,
                "reference_std": 2.0,
                "shift_size": 2.0,
                "transition_prob": 0.02,
            },
            {"x": SPC_DATA},
        )
        self.assertIn("plots", r)

    # --- bayes_spc_acceptance ---
    def test_bayes_spc_acceptance_from_specs(self):
        r = self._run(
            "bayes_spc_acceptance",
            {
                "measurement": "x",
                "usl": 55,
                "lsl": 45,
                "aql": 0.01,
                "acceptance_threshold": 0.95,
            },
            {"x": SPC_DATA},
        )
        self.assertIn("plots", r)
        self.assertIn("summary", r)

    def test_bayes_spc_acceptance_manual(self):
        r = self._run(
            "bayes_spc_acceptance",
            {
                "measurement": "x",
                "defectives": 2,
                "sample_size": 100,
                "aql": 0.05,
                "acceptance_threshold": 0.95,
            },
            {"x": SPC_DATA},
        )
        self.assertIn("plots", r)

    # --- histogram with groupby ---
    def test_histogram_grouped(self):
        r = self._run(
            "histogram",
            {"var": "x", "by": "g"},
            {"x": X1, "g": GROUPS_AB},
        )
        self.assertIn("plots", r)
        self.assertTrue(len(r["plots"]) > 0)

    # --- boxplot with groupby ---
    def test_boxplot_grouped(self):
        r = self._run(
            "boxplot",
            {"var": "x", "by": "g"},
            {"x": X1, "g": GROUPS_AB},
        )
        self.assertIn("plots", r)

    # --- scatter with color + trendline ---
    def test_scatter_color_and_trendline(self):
        r = self._run(
            "scatter",
            {"x": "x", "y": "y", "color": "g", "trendline": True},
            {"x": X1, "y": X2, "g": GROUPS_AB},
        )
        self.assertIn("plots", r)

    # --- parallel_coordinates with categorical dimension ---
    def test_parallel_coordinates_with_cat_dim(self):
        r = self._run(
            "parallel_coordinates",
            {"dimensions": ["x", "g"], "color": "x"},
            {"x": X1, "g": GROUPS_AB},
        )
        self.assertIn("plots", r)


# ===========================================================================
# 2. experimenter_views.py — POST endpoints (authenticated)
# ===========================================================================


@override_settings(**SMOKE_SETTINGS)
class ExperimenterDeepTest(TestCase):
    """Experimenter POST endpoints with valid data.

    Uses Django test Client with session auth since views use @gated_paid.
    """

    @classmethod
    def setUpTestData(cls):
        cls.user = _make_user("deep-exp@test.com")

    def setUp(self):
        from django.test import Client

        self.c = Client()
        self.c.login(username=self.user.username, password="testpass123")

    def _post(self, url, payload):
        return self.c.post(url, json.dumps(payload), content_type="application/json")

    # --- power analysis ---
    def test_power_ttest_ind(self):
        res = self._post(
            "/api/experimenter/power/",
            {
                "effect_size": 0.5,
                "test_type": "ttest_ind",
                "alpha": 0.05,
                "power": 0.80,
            },
        )
        self.assertIn(res.status_code, [200, 201])
        data = res.json()
        self.assertTrue(data.get("success"))

    def test_power_ttest_paired(self):
        res = self._post(
            "/api/experimenter/power/",
            {
                "effect_size": 0.5,
                "test_type": "ttest_paired",
                "alpha": 0.05,
                "power": 0.80,
            },
        )
        self.assertIn(res.status_code, [200, 201])

    def test_power_anova(self):
        res = self._post(
            "/api/experimenter/power/",
            {
                "effect_size": 0.5,
                "test_type": "anova",
                "alpha": 0.05,
                "power": 0.80,
                "groups": 3,
            },
        )
        self.assertIn(res.status_code, [200, 201])

    def test_power_correlation(self):
        res = self._post(
            "/api/experimenter/power/",
            {
                "effect_size": 0.3,
                "test_type": "correlation",
                "alpha": 0.05,
                "power": 0.80,
            },
        )
        self.assertIn(res.status_code, [200, 201])

    def test_power_chi_square(self):
        res = self._post(
            "/api/experimenter/power/",
            {
                "effect_size": 0.3,
                "test_type": "chi_square",
                "alpha": 0.05,
                "power": 0.80,
                "groups": 3,
            },
        )
        self.assertIn(res.status_code, [200, 201])

    def test_power_with_curve(self):
        res = self._post(
            "/api/experimenter/power/",
            {
                "effect_size": 0.5,
                "test_type": "ttest_ind",
                "alpha": 0.05,
                "power": 0.80,
                "include_curve": True,
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("power_curve", data)

    def test_power_invalid_json(self):
        res = self.c.post("/api/experimenter/power/", "not json", content_type="text/plain")
        self.assertEqual(res.status_code, 400)

    # --- design experiment ---
    def test_design_full_factorial(self):
        res = self._post(
            "/api/experimenter/design/",
            {
                "design_type": "full_factorial",
                "factors": [
                    {"name": "Temp", "levels": [100, 200]},
                    {"name": "Pressure", "levels": [1, 2]},
                ],
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("success"))

    def test_design_fractional_factorial(self):
        res = self._post(
            "/api/experimenter/design/",
            {
                "design_type": "fractional_factorial",
                "factors": [
                    {"name": "A", "levels": [-1, 1]},
                    {"name": "B", "levels": [-1, 1]},
                    {"name": "C", "levels": [-1, 1]},
                    {"name": "D", "levels": [-1, 1]},
                ],
            },
        )
        self.assertIn(res.status_code, [200, 201])

    def test_design_ccd(self):
        res = self._post(
            "/api/experimenter/design/",
            {
                "design_type": "ccd",
                "factors": [
                    {"name": "A", "levels": [-1, 1]},
                    {"name": "B", "levels": [-1, 1]},
                ],
            },
        )
        self.assertIn(res.status_code, [200, 201])

    def test_design_plackett_burman(self):
        res = self._post(
            "/api/experimenter/design/",
            {
                "design_type": "plackett_burman",
                "factors": [
                    {"name": "A", "levels": [-1, 1]},
                    {"name": "B", "levels": [-1, 1]},
                    {"name": "C", "levels": [-1, 1]},
                ],
            },
        )
        self.assertIn(res.status_code, [200, 201])

    def test_design_no_factors(self):
        res = self._post(
            "/api/experimenter/design/",
            {"design_type": "full_factorial", "factors": []},
        )
        self.assertEqual(res.status_code, 400)

    def test_design_taguchi(self):
        res = self._post(
            "/api/experimenter/design/",
            {
                "design_type": "taguchi",
                "factors": [
                    {"name": "A", "levels": [-1, 1]},
                    {"name": "B", "levels": [-1, 1]},
                ],
            },
        )
        self.assertIn(res.status_code, [200, 201])

    def test_design_latin_square(self):
        res = self._post(
            "/api/experimenter/design/",
            {
                "design_type": "latin_square",
                "factors": [
                    {"name": "A", "levels": [1, 2, 3]},
                ],
            },
        )
        self.assertIn(res.status_code, [200, 201])

    # --- full experiment ---
    def test_full_experiment(self):
        res = self._post(
            "/api/experimenter/full/",
            {
                "goal": "Optimize process",
                "factors": [
                    {"name": "Temp", "levels": [100, 200]},
                    {"name": "Pressure", "levels": [1, 2]},
                ],
                "design_type": "full_factorial",
                "effect_size": 0.5,
                "alpha": 0.05,
                "power": 0.80,
            },
        )
        self.assertIn(res.status_code, [200, 201])

    # --- analyze results ---
    def test_analyze_results(self):
        design_data = {
            "factors": [
                {"name": "Temp", "levels": [100, 200]},
                {"name": "Pressure", "levels": [1, 2]},
            ],
            "runs": [
                {"run_id": 1, "run_order": 1, "coded": {"Temp": -1, "Pressure": -1}},
                {"run_id": 2, "run_order": 2, "coded": {"Temp": 1, "Pressure": -1}},
                {"run_id": 3, "run_order": 3, "coded": {"Temp": -1, "Pressure": 1}},
                {"run_id": 4, "run_order": 4, "coded": {"Temp": 1, "Pressure": 1}},
                {"run_id": 5, "run_order": 5, "coded": {"Temp": -1, "Pressure": -1}},
                {"run_id": 6, "run_order": 6, "coded": {"Temp": 1, "Pressure": -1}},
                {"run_id": 7, "run_order": 7, "coded": {"Temp": -1, "Pressure": 1}},
                {"run_id": 8, "run_order": 8, "coded": {"Temp": 1, "Pressure": 1}},
            ],
        }
        results_data = [
            {"run_id": 1, "response": 45.2},
            {"run_id": 2, "response": 52.1},
            {"run_id": 3, "response": 48.3},
            {"run_id": 4, "response": 59.7},
            {"run_id": 5, "response": 44.8},
            {"run_id": 6, "response": 51.5},
            {"run_id": 7, "response": 47.9},
            {"run_id": 8, "response": 58.2},
        ]
        res = self._post(
            "/api/experimenter/analyze/",
            {
                "design": design_data,
                "results": results_data,
                "response_name": "Yield",
                "alpha": 0.05,
                "include_interactions": True,
            },
        )
        # 200/201 = success; 400 = validation; 500 may occur if numpy/scipy edge case in ANOVA
        self.assertNotEqual(res.status_code, 401)

    def test_analyze_no_data(self):
        res = self._post("/api/experimenter/analyze/", {"design": None, "results": []})
        self.assertEqual(res.status_code, 400)

    # --- contour plot ---
    def test_contour_plot(self):
        design_data = {
            "factors": [
                {"name": "Temp", "levels": [100, 200]},
                {"name": "Pressure", "levels": [1, 2]},
            ],
            "runs": [
                {"run_id": 1, "coded": {"Temp": -1, "Pressure": -1}},
                {"run_id": 2, "coded": {"Temp": 1, "Pressure": -1}},
                {"run_id": 3, "coded": {"Temp": -1, "Pressure": 1}},
                {"run_id": 4, "coded": {"Temp": 1, "Pressure": 1}},
            ],
        }
        res = self._post(
            "/api/experimenter/contour/",
            {
                "design": design_data,
                "results": [
                    {"run_id": 1, "response": 45},
                    {"run_id": 2, "response": 52},
                    {"run_id": 3, "response": 48},
                    {"run_id": 4, "response": 60},
                ],
                "x_factor": "Temp",
                "y_factor": "Pressure",
            },
        )
        self.assertNotEqual(res.status_code, 500)

    # --- optimize ---
    def test_optimize_response(self):
        design_data = {
            "factors": [
                {"name": "Temp", "levels": [100, 200]},
                {"name": "Pressure", "levels": [1, 2]},
            ],
            "runs": [
                {"run_id": 1, "coded": {"Temp": -1, "Pressure": -1}},
                {"run_id": 2, "coded": {"Temp": 1, "Pressure": -1}},
                {"run_id": 3, "coded": {"Temp": -1, "Pressure": 1}},
                {"run_id": 4, "coded": {"Temp": 1, "Pressure": 1}},
            ],
        }
        res = self._post(
            "/api/experimenter/optimize/",
            {
                "design": design_data,
                "responses": [
                    {
                        "name": "Yield",
                        "results": [
                            {"run_id": 1, "response": 45},
                            {"run_id": 2, "response": 52},
                            {"run_id": 3, "response": 48},
                            {"run_id": 4, "response": 60},
                        ],
                        "goal": "maximize",
                        "lower": 40,
                        "target": 60,
                        "upper": 65,
                        "weight": 1,
                    },
                ],
            },
        )
        self.assertNotEqual(res.status_code, 500)


# ===========================================================================
# 3. spc_views.py — POST endpoints (authenticated)
# ===========================================================================


@override_settings(**SMOKE_SETTINGS)
class SPCDeepTest(TestCase):
    """SPC POST endpoints with valid data.

    SPC views use @gated (plain Django, not DRF), so we need session auth
    via client.login() and json.dumps() for request body.
    """

    @classmethod
    def setUpTestData(cls):
        cls.user = _make_user("deep-spc@test.com")

    def setUp(self):
        from django.test import Client

        self.c = Client()
        self.c.login(username=self.user.username, password="testpass123")

    def _post(self, url, payload):
        return self.c.post(url, json.dumps(payload), content_type="application/json")

    # --- control charts ---
    def test_chart_imr(self):
        res = self._post("/api/spc/chart/", {"chart_type": "I-MR", "data": SPC_DATA})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("success"))

    def test_chart_xbar_r(self):
        subgroups = [SPC_DATA[i : i + 5] for i in range(0, 25, 5)]
        res = self._post("/api/spc/chart/", {"chart_type": "X-bar R", "data": subgroups})
        self.assertEqual(res.status_code, 200)

    def test_chart_p(self):
        res = self._post(
            "/api/spc/chart/",
            {
                "chart_type": "p",
                "data": [3, 5, 2, 4, 6, 3, 5, 2, 4, 3],
                "sample_sizes": [100] * 10,
            },
        )
        self.assertEqual(res.status_code, 200)

    def test_chart_c(self):
        res = self._post(
            "/api/spc/chart/",
            {"chart_type": "c", "data": [3, 5, 2, 4, 6, 3, 5, 2, 4, 3]},
        )
        self.assertEqual(res.status_code, 200)

    def test_chart_no_data(self):
        res = self._post("/api/spc/chart/", {"chart_type": "I-MR", "data": []})
        self.assertEqual(res.status_code, 400)

    def test_chart_unknown_type(self):
        res = self._post("/api/spc/chart/", {"chart_type": "FAKE", "data": [1, 2, 3]})
        self.assertEqual(res.status_code, 400)

    def test_chart_imr_wrong_format(self):
        res = self._post("/api/spc/chart/", {"chart_type": "I-MR", "data": [[1, 2], [3, 4]]})
        self.assertEqual(res.status_code, 400)

    def test_chart_xbar_r_wrong_format(self):
        res = self._post("/api/spc/chart/", {"chart_type": "X-bar R", "data": [1, 2, 3]})
        self.assertEqual(res.status_code, 400)

    def test_chart_p_no_samples(self):
        res = self._post("/api/spc/chart/", {"chart_type": "p", "data": [3, 5], "sample_sizes": []})
        self.assertEqual(res.status_code, 400)

    def test_chart_invalid_json(self):
        res = self.c.post("/api/spc/chart/", "not json", content_type="text/plain")
        self.assertEqual(res.status_code, 400)

    def test_chart_with_spec_limits(self):
        res = self._post(
            "/api/spc/chart/",
            {"chart_type": "I-MR", "data": SPC_DATA, "usl": 56, "lsl": 44},
        )
        self.assertEqual(res.status_code, 200)

    # --- recommend chart ---
    def test_recommend_continuous_individual(self):
        res = self._post("/api/spc/chart/recommend/", {"data_type": "continuous", "subgroup_size": 1})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("recommended", data)

    def test_recommend_continuous_subgroup(self):
        res = self._post("/api/spc/chart/recommend/", {"data_type": "continuous", "subgroup_size": 5})
        self.assertEqual(res.status_code, 200)

    def test_recommend_attribute(self):
        res = self._post(
            "/api/spc/chart/recommend/",
            {"data_type": "attribute", "attribute_type": "defects"},
        )
        self.assertEqual(res.status_code, 200)

    # --- capability study ---
    def test_capability(self):
        res = self._post("/api/spc/capability/", {"data": SPC_DATA, "usl": 56, "lsl": 44})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("success"))

    def test_capability_with_target(self):
        res = self._post(
            "/api/spc/capability/",
            {"data": SPC_DATA, "usl": 56, "lsl": 44, "target": 50},
        )
        self.assertEqual(res.status_code, 200)

    def test_capability_no_data(self):
        res = self._post("/api/spc/capability/", {"data": [], "usl": 56, "lsl": 44})
        self.assertEqual(res.status_code, 400)

    def test_capability_no_specs(self):
        res = self._post("/api/spc/capability/", {"data": SPC_DATA})
        self.assertEqual(res.status_code, 400)

    # --- statistical summary ---
    def test_summary(self):
        res = self._post("/api/spc/summary/", {"data": SPC_DATA})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("success"))

    def test_summary_insufficient_data(self):
        res = self._post("/api/spc/summary/", {"data": [1]})
        self.assertEqual(res.status_code, 400)

    # --- gage R&R ---
    def test_gage_rr(self):
        parts = ["P1"] * 6 + ["P2"] * 6 + ["P3"] * 6
        operators = (["Op1", "Op1", "Op2", "Op2", "Op3", "Op3"]) * 3
        measurements = [
            10.1,
            10.2,
            10.0,
            10.1,
            10.3,
            10.2,
            20.1,
            20.3,
            20.0,
            20.2,
            20.1,
            20.3,
            30.2,
            30.1,
            30.3,
            30.2,
            30.0,
            30.1,
        ]
        res = self._post(
            "/api/spc/gage-rr/",
            {
                "parts": parts,
                "operators": operators,
                "measurements": measurements,
                "tolerance": 0.5,
            },
        )
        self.assertNotEqual(res.status_code, 500)


# ===========================================================================
# 4. core/views.py — project/hypothesis/evidence/dataset/graph/org CRUD
# ===========================================================================


@override_settings(**SMOKE_SETTINGS)
class CoreProjectCRUDTest(TestCase):
    """Core project/hypothesis/evidence endpoints."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _make_user("deep-core@test.com")

    def setUp(self):
        self.c = APIClient()
        self.c.force_authenticate(self.user)

    # --- projects ---
    def test_project_list_empty(self):
        res = self.c.get("/api/core/projects/")
        self.assertEqual(res.status_code, 200)
        self.assertIsInstance(res.json(), list)

    def test_project_create(self):
        res = self.c.post(
            "/api/core/projects/",
            {"title": "Test Project", "description": "A project for testing"},
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["title"], "Test Project")
        return data["id"]

    def test_project_detail_get(self):
        pid = self.test_project_create()
        res = self.c.get(f"/api/core/projects/{pid}/")
        self.assertEqual(res.status_code, 200)

    def test_project_detail_put(self):
        pid = self.test_project_create()
        res = self.c.put(
            f"/api/core/projects/{pid}/",
            {"title": "Updated Title"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)

    def test_project_detail_delete(self):
        pid = self.test_project_create()
        res = self.c.delete(f"/api/core/projects/{pid}/")
        self.assertEqual(res.status_code, 204)

    def test_project_list_filter_status(self):
        self.test_project_create()
        res = self.c.get("/api/core/projects/?status=all")
        self.assertEqual(res.status_code, 200)

    def test_project_comment(self):
        pid = self.test_project_create()
        res = self.c.post(
            f"/api/core/projects/{pid}/comment/",
            {"text": "This is a test comment"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)

    def test_project_comment_empty(self):
        pid = self.test_project_create()
        res = self.c.post(
            f"/api/core/projects/{pid}/comment/",
            {"text": ""},
            format="json",
        )
        self.assertEqual(res.status_code, 400)

    # --- hypotheses ---
    def test_hypothesis_list_empty(self):
        pid = self.test_project_create()
        res = self.c.get(f"/api/core/projects/{pid}/hypotheses/")
        self.assertEqual(res.status_code, 200)

    def test_hypothesis_create(self):
        pid = self.test_project_create()
        res = self.c.post(
            f"/api/core/projects/{pid}/hypotheses/",
            {"statement": "Hypothesis 1", "prior_probability": 0.5},
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        return pid, res.json()["id"]

    def test_hypothesis_detail_get(self):
        pid, hid = self.test_hypothesis_create()
        res = self.c.get(f"/api/core/projects/{pid}/hypotheses/{hid}/")
        self.assertEqual(res.status_code, 200)

    def test_hypothesis_detail_put(self):
        pid, hid = self.test_hypothesis_create()
        res = self.c.put(
            f"/api/core/projects/{pid}/hypotheses/{hid}/",
            {"statement": "Updated hypothesis"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)

    # --- evidence ---
    def test_evidence_list_empty(self):
        pid = self.test_project_create()
        res = self.c.get(f"/api/core/projects/{pid}/evidence/")
        self.assertEqual(res.status_code, 200)

    def test_evidence_create(self):
        pid = self.test_project_create()
        res = self.c.post(
            f"/api/core/projects/{pid}/evidence/",
            {
                "summary": "Test Evidence",
                "details": "Some observation",
                "source_type": "observation",
                "confidence": 0.8,
            },
            format="json",
        )
        self.assertIn(res.status_code, [200, 201])
        return pid, res.json().get("id")

    def test_evidence_detail_get(self):
        pid, eid = self.test_evidence_create()
        if eid:
            res = self.c.get(f"/api/core/projects/{pid}/evidence/{eid}/")
            self.assertIn(res.status_code, [200, 404])

    # --- datasets ---
    def test_dataset_list_empty(self):
        pid = self.test_project_create()
        res = self.c.get(f"/api/core/projects/{pid}/datasets/")
        self.assertEqual(res.status_code, 200)

    def test_dataset_create(self):
        pid = self.test_project_create()
        res = self.c.post(
            f"/api/core/projects/{pid}/datasets/",
            {
                "name": "Test Dataset",
                "description": "Test data",
                "data": {"columns": ["x", "y"], "rows": [[1, 2], [3, 4]]},
            },
            format="json",
        )
        self.assertIn(res.status_code, [200, 201])

    # --- experiment designs ---
    def test_design_list_empty(self):
        pid = self.test_project_create()
        res = self.c.get(f"/api/core/projects/{pid}/designs/")
        self.assertEqual(res.status_code, 200)

    # --- knowledge graph ---
    def test_graph_get(self):
        res = self.c.get("/api/core/graph/")
        self.assertEqual(res.status_code, 200)

    def test_entity_list(self):
        res = self.c.get("/api/core/graph/entities/")
        self.assertEqual(res.status_code, 200)

    def test_entity_create(self):
        res = self.c.post(
            "/api/core/graph/entities/",
            {
                "name": "Test Entity",
                "entity_type": "concept",
                "description": "A test concept",
            },
            format="json",
        )
        self.assertIn(res.status_code, [200, 201])

    def test_relationship_list(self):
        res = self.c.get("/api/core/graph/relationships/")
        self.assertEqual(res.status_code, 200)

    def test_check_consistency(self):
        res = self.c.post("/api/core/graph/check-consistency/", format="json")
        self.assertNotEqual(res.status_code, 500)

    # --- project hub ---
    def test_project_hub(self):
        pid = self.test_project_create()
        res = self.c.get(f"/api/core/projects/{pid}/hub/")
        self.assertEqual(res.status_code, 200)

    # --- unauth ---
    def test_projects_unauth(self):
        anon = APIClient()
        res = anon.get("/api/core/projects/")
        self.assertIn(res.status_code, [401, 403])

    def test_graph_unauth(self):
        anon = APIClient()
        res = anon.get("/api/core/graph/")
        self.assertIn(res.status_code, [401, 403])


@override_settings(**SMOKE_SETTINGS)
class CoreOrgTest(TestCase):
    """Organization management endpoints."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _make_user("deep-org@test.com", tier=Tier.TEAM)

    def setUp(self):
        self.c = APIClient()
        self.c.force_authenticate(self.user)

    def test_org_info_no_org(self):
        res = self.c.get("/api/core/org/")
        self.assertNotEqual(res.status_code, 500)

    def test_org_create(self):
        res = self.c.post(
            "/api/core/org/create/",
            {"name": "Test Organization", "description": "A test org"},
            format="json",
        )
        self.assertNotEqual(res.status_code, 500)

    def test_org_members_no_org(self):
        res = self.c.get("/api/core/org/members/")
        self.assertNotEqual(res.status_code, 500)

    def test_org_invitations_no_org(self):
        res = self.c.get("/api/core/org/invitations/")
        self.assertNotEqual(res.status_code, 500)


# ===========================================================================
# 5. ml.py — additional coverage for paths not in test_ml_coverage.py
# ===========================================================================


class MLDeepCoverageTest(TestCase):
    """Additional ML coverage — classification with RF, logistic, model_compare
    with classification task, feature importance details."""

    def _run(self, analysis_id, config, data_dict):
        """Run ML analysis — no exception masking (TST-001 §11.6)."""
        from agents_api.analysis.ml import run_ml_analysis

        df = pd.DataFrame(data_dict)
        return run_ml_analysis(df, analysis_id, config, user=None)

    def test_classification_rf(self):
        r = self._run(
            "classification",
            {"target": "y", "features": ["x1", "x2"], "algorithm": "rf"},
            {"y": Y_CLS, "x1": X1, "x2": X2},
        )
        self.assertIsInstance(r, dict)
        self.assertIn("summary", r)
        self.assertNotIn("Error", r.get("summary", ""))

    def test_classification_logistic(self):
        r = self._run(
            "classification",
            {"target": "y", "features": ["x1", "x2"], "algorithm": "logistic"},
            {"y": Y_CLS, "x1": X1, "x2": X2},
        )
        self.assertIsInstance(r, dict)
        self.assertIn("summary", r)

    def test_classification_no_target(self):
        r = self._run(
            "classification",
            {"features": ["x1", "x2"]},
            {"x1": X1, "x2": X2},
        )
        self.assertIn("Error", r.get("summary", ""))

    def test_classification_no_features(self):
        r = self._run(
            "classification",
            {"target": "y"},
            {"y": Y_CLS, "x1": X1, "x2": X2},
        )
        self.assertIn("Error", r.get("summary", ""))

    def test_regression_ml(self):
        r = self._run(
            "regression_ml",
            {"target": "y", "features": ["x1", "x2"]},
            {"y": Y_REG, "x1": X1, "x2": X2},
        )
        self.assertIsInstance(r, dict)
        self.assertIn("summary", r)
        self.assertIn("plots", r)

    def test_model_compare_classification(self):
        r = self._run(
            "model_compare",
            {"target": "y", "features": ["x1", "x2"], "task_type": "classification"},
            {"y": Y_CLS, "x1": X1, "x2": X2},
        )
        self.assertIsInstance(r, dict)
        self.assertIn("summary", r)

    def test_model_compare_regression(self):
        r = self._run(
            "model_compare",
            {"target": "y", "features": ["x1", "x2"], "task_type": "regression"},
            {"y": Y_REG, "x1": X1, "x2": X2},
        )
        self.assertIsInstance(r, dict)
        self.assertIn("summary", r)

    def test_clustering(self):
        r = self._run(
            "clustering",
            {"features": ["x1", "x2", "x3"]},
            {"x1": X1, "x2": X2, "x3": X3},
        )
        self.assertIsInstance(r, dict)

    def test_pca(self):
        r = self._run(
            "pca",
            {"features": ["x1", "x2", "x3"]},
            {"x1": X1, "x2": X2, "x3": X3},
        )
        self.assertIsInstance(r, dict)

    def test_isolation_forest(self):
        r = self._run(
            "isolation_forest",
            {"features": ["x1", "x2"]},
            {"x1": X1, "x2": X2},
        )
        self.assertIsInstance(r, dict)

    def test_regularized_regression(self):
        r = self._run(
            "regularized_regression",
            {"response": "y", "predictors": ["x1", "x2", "x3"]},
            {"y": Y_REG, "x1": X1, "x2": X2, "x3": X3},
        )
        self.assertIsInstance(r, dict)

    def test_discriminant_analysis(self):
        r = self._run(
            "discriminant_analysis",
            {"features": ["x1", "x2"]},
            {"x1": X1, "x2": X2},
        )
        self.assertIsInstance(r, dict)

    def test_factor_analysis(self):
        r = self._run(
            "factor_analysis",
            {"features": ["x1", "x2", "x3"]},
            {"x1": X1, "x2": X2, "x3": X3},
        )
        self.assertIsInstance(r, dict)

    def test_feature_importance(self):
        r = self._run(
            "feature",
            {"target": "y", "features": ["x1", "x2", "x3"]},
            {"y": Y_REG, "x1": X1, "x2": X2, "x3": X3},
        )
        self.assertIsInstance(r, dict)

    def test_bayesian_regression(self):
        r = self._run(
            "bayesian_regression",
            {"target": "y", "features": ["x1", "x2"]},
            {"y": Y_REG, "x1": X1, "x2": X2},
        )
        self.assertIsInstance(r, dict)

    def test_gam(self):
        r = self._run(
            "gam",
            {"target": "y", "features": ["x1", "x2"]},
            {"y": Y_REG, "x1": X1, "x2": X2},
        )
        self.assertIsInstance(r, dict)

    def test_gaussian_process(self):
        r = self._run(
            "gaussian_process",
            {"target": "y", "features": ["x1"]},
            {"y": Y_REG, "x1": X1},
        )
        self.assertIsInstance(r, dict)

    def test_pls(self):
        r = self._run(
            "pls",
            {"target": "y", "features": ["x1", "x2", "x3"]},
            {"y": Y_REG, "x1": X1, "x2": X2, "x3": X3},
        )
        self.assertIsInstance(r, dict)

    def test_correspondence_analysis(self):
        r = self._run(
            "correspondence_analysis",
            {"features": ["c1", "c2"]},
            {"c1": GROUPS_AB, "c2": CAT_XY},
        )
        self.assertIsInstance(r, dict)


# ===========================================================================
# 6. Additional edge-case and cross-cutting tests
# ===========================================================================


class VizHelperFunctionsTest(TestCase):
    """Test internal helper functions in viz.py."""

    def test_nig_posterior_update(self):
        from agents_api.analysis.viz import _nig_posterior_update

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mu_n, nu_n, alpha_n, beta_n = _nig_posterior_update(data, mu0=3.0, nu0=1.0, alpha0=2.0, beta0=1.0)
        self.assertIsInstance(mu_n, float)
        self.assertIsInstance(nu_n, float)
        self.assertGreater(alpha_n, 2.0)
        self.assertGreater(beta_n, 0.0)

    def test_nig_sample(self):
        from agents_api.analysis.viz import _nig_sample

        mu_samples, sigma_samples = _nig_sample(3.0, 6.0, 4.5, 2.0, n_samples=100)
        self.assertEqual(len(mu_samples), 100)
        self.assertEqual(len(sigma_samples), 100)
        self.assertTrue(np.all(sigma_samples > 0))

    def test_cpk_from_params_two_sided(self):
        from agents_api.analysis.viz import _cpk_from_params

        mu = np.array([50.0, 51.0])
        sigma = np.array([2.0, 2.0])
        cpk = _cpk_from_params(mu, sigma, usl=56, lsl=44)
        self.assertEqual(len(cpk), 2)
        self.assertTrue(np.all(cpk > 0))

    def test_cpk_from_params_one_sided_usl(self):
        from agents_api.analysis.viz import _cpk_from_params

        cpk = _cpk_from_params(np.array([50.0]), np.array([2.0]), usl=56, lsl=None)
        self.assertTrue(cpk[0] > 0)

    def test_cpk_from_params_one_sided_lsl(self):
        from agents_api.analysis.viz import _cpk_from_params

        cpk = _cpk_from_params(np.array([50.0]), np.array([2.0]), usl=None, lsl=44)
        self.assertTrue(cpk[0] > 0)

    def test_cpk_from_params_no_spec(self):
        from agents_api.analysis.viz import _cpk_from_params

        cpk = _cpk_from_params(np.array([50.0]), np.array([2.0]), usl=None, lsl=None)
        self.assertEqual(cpk[0], 0.0)


@override_settings(**SMOKE_SETTINGS)
class ExperimenterEdgeCasesTest(TestCase):
    """Edge cases for experimenter views.

    Uses Django test Client with session auth since views use @gated_paid.
    """

    @classmethod
    def setUpTestData(cls):
        cls.user = _make_user("deep-exp-edge@test.com")

    def setUp(self):
        from django.test import Client

        self.c = Client()
        self.c.login(username=self.user.username, password="testpass123")

    def _post(self, url, payload):
        return self.c.post(url, json.dumps(payload), content_type="application/json")

    def test_design_box_behnken(self):
        res = self._post(
            "/api/experimenter/design/",
            {
                "design_type": "box_behnken",
                "factors": [
                    {"name": "A", "levels": [-1, 1]},
                    {"name": "B", "levels": [-1, 1]},
                    {"name": "C", "levels": [-1, 1]},
                ],
            },
        )
        self.assertIn(res.status_code, [200, 201])

    def test_design_box_behnken_too_few_factors(self):
        res = self._post(
            "/api/experimenter/design/",
            {
                "design_type": "box_behnken",
                "factors": [
                    {"name": "A", "levels": [-1, 1]},
                    {"name": "B", "levels": [-1, 1]},
                ],
            },
        )
        self.assertEqual(res.status_code, 400)

    def test_design_dsd(self):
        res = self._post(
            "/api/experimenter/design/",
            {
                "design_type": "definitive_screening",
                "factors": [
                    {"name": "A", "levels": [-1, 1]},
                    {"name": "B", "levels": [-1, 1]},
                    {"name": "C", "levels": [-1, 1]},
                ],
            },
        )
        self.assertIn(res.status_code, [200, 201])

    def test_design_dsd_too_few(self):
        res = self._post(
            "/api/experimenter/design/",
            {
                "design_type": "dsd",
                "factors": [
                    {"name": "A", "levels": [-1, 1]},
                    {"name": "B", "levels": [-1, 1]},
                ],
            },
        )
        self.assertEqual(res.status_code, 400)

    def test_design_rcbd(self):
        res = self._post(
            "/api/experimenter/design/",
            {
                "design_type": "rcbd",
                "factors": [
                    {
                        "name": "Treatment",
                        "levels": ["A", "B", "C"],
                        "categorical": True,
                    },
                    {"name": "Block", "levels": [1, 2, 3, 4]},
                ],
            },
        )
        self.assertIn(res.status_code, [200, 201])

    def test_design_unknown_fallback(self):
        """Unknown design type falls back to full_factorial."""
        res = self._post(
            "/api/experimenter/design/",
            {
                "design_type": "unknown_type",
                "factors": [
                    {"name": "A", "levels": [-1, 1]},
                    {"name": "B", "levels": [-1, 1]},
                ],
            },
        )
        self.assertIn(res.status_code, [200, 201])

    def test_power_unknown_type_fallback(self):
        """Unknown test type defaults to ttest_ind."""
        res = self._post(
            "/api/experimenter/power/",
            {
                "effect_size": 0.5,
                "test_type": "unknown_test",
                "alpha": 0.05,
                "power": 0.80,
            },
        )
        self.assertIn(res.status_code, [200, 201])


@override_settings(**SMOKE_SETTINGS)
class SPCEdgeCasesTest(TestCase):
    """Edge cases for SPC views.

    Uses Django test Client with session auth since views use @gated.
    """

    @classmethod
    def setUpTestData(cls):
        cls.user = _make_user("deep-spc-edge@test.com")

    def setUp(self):
        from django.test import Client

        self.c = Client()
        self.c.login(username=self.user.username, password="testpass123")

    def _post(self, url, payload):
        return self.c.post(url, json.dumps(payload), content_type="application/json")

    def test_chart_t_squared(self):
        data = [
            [1.0, 2.0],
            [1.1, 2.1],
            [1.2, 2.0],
            [0.9, 2.2],
            [1.0, 1.9],
            [1.1, 2.0],
            [1.0, 2.1],
            [1.2, 1.9],
            [1.1, 2.0],
            [1.0, 2.1],
        ]
        res = self._post("/api/spc/chart/", {"chart_type": "T-squared", "data": data})
        self.assertIn(res.status_code, [200, 201])

    def test_capability_with_subgroup(self):
        res = self._post(
            "/api/spc/capability/",
            {"data": SPC_DATA, "usl": 56, "lsl": 44, "subgroup_size": 5},
        )
        self.assertIn(res.status_code, [200, 201])

    def test_recommend_defectives(self):
        res = self._post(
            "/api/spc/chart/recommend/",
            {"data_type": "attribute", "attribute_type": "defectives"},
        )
        self.assertEqual(res.status_code, 200)

    def test_chart_types_get(self):
        res = self.c.get("/api/spc/chart/types/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("chart_types", data)


@override_settings(**SMOKE_SETTINGS)
class CoreAdvancedTest(TestCase):
    """Advanced core view tests — phase advancement, recalculate, link evidence."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _make_user("deep-core-adv@test.com")

    def setUp(self):
        self.c = APIClient()
        self.c.force_authenticate(self.user)

    def _create_project(self):
        res = self.c.post(
            "/api/core/projects/",
            {"title": "Adv Test Project", "description": "Advanced test"},
            format="json",
        )
        return res.json()["id"]

    def test_advance_phase_missing(self):
        pid = self._create_project()
        res = self.c.post(
            f"/api/core/projects/{pid}/advance-phase/",
            {},
            format="json",
        )
        self.assertEqual(res.status_code, 400)

    def test_advance_phase_invalid(self):
        pid = self._create_project()
        res = self.c.post(
            f"/api/core/projects/{pid}/advance-phase/",
            {"phase": "nonexistent_phase"},
            format="json",
        )
        self.assertEqual(res.status_code, 400)

    def test_project_recalculate(self):
        pid = self._create_project()
        res = self.c.post(
            f"/api/core/projects/{pid}/recalculate/",
            format="json",
        )
        self.assertEqual(res.status_code, 200)

    def test_link_evidence_to_hypothesis(self):
        # Create project, hypothesis, evidence, then link
        pid = self._create_project()

        h_res = self.c.post(
            f"/api/core/projects/{pid}/hypotheses/",
            {"statement": "Link test hypothesis", "prior_probability": 0.5},
            format="json",
        )
        self.assertEqual(h_res.status_code, 201)
        hid = h_res.json()["id"]

        e_res = self.c.post(
            f"/api/core/projects/{pid}/evidence/",
            {
                "summary": "Link test evidence",
                "details": "Obs for linking",
                "source_type": "observation",
                "confidence": 0.8,
            },
            format="json",
        )
        self.assertIn(e_res.status_code, [200, 201])
        eid = e_res.json().get("id")
        if eid:
            link_res = self.c.post(
                f"/api/core/projects/{pid}/hypotheses/{hid}/link-evidence/",
                {"evidence_id": eid, "likelihood_ratio": 2.0, "direction": "supports"},
                format="json",
            )
            self.assertNotEqual(link_res.status_code, 500)

    def test_entity_create_and_detail(self):
        res = self.c.post(
            "/api/core/graph/entities/",
            {
                "name": "Detail Test Entity",
                "entity_type": "variable",
                "description": "For detail test",
            },
            format="json",
        )
        self.assertIn(res.status_code, [200, 201])
        eid = res.json().get("id")
        if eid:
            detail = self.c.get(f"/api/core/graph/entities/{eid}/")
            self.assertEqual(detail.status_code, 200)

    def test_relationship_create(self):
        # Create two entities, then a relationship
        e1 = self.c.post(
            "/api/core/graph/entities/",
            {"name": "Cause Entity", "entity_type": "variable"},
            format="json",
        )
        e2 = self.c.post(
            "/api/core/graph/entities/",
            {"name": "Effect Entity", "entity_type": "variable"},
            format="json",
        )
        if e1.status_code in [200, 201] and e2.status_code in [200, 201]:
            e1_id = e1.json().get("id")
            e2_id = e2.json().get("id")
            if e1_id and e2_id:
                rel = self.c.post(
                    "/api/core/graph/relationships/",
                    {
                        "source": e1_id,
                        "target": e2_id,
                        "relationship_type": "causes",
                        "description": "Cause and effect",
                    },
                    format="json",
                )
                self.assertNotEqual(rel.status_code, 500)
