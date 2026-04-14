"""VIS-001 compliance tests — chart visualization consistency.

Validates that all DSW chart output conforms to VIS-001:
- Trace colors from SVEND_COLORS or semantic palette (§4.1)
- Legends not inside chart body (§5.2)
- Transparent backgrounds (§6.1)
- Standard dimensions and font (§7.1, §8.1)
- Pipeline coverage — all plots pass through apply_chart_defaults (§11.1)

Standard: VIS-001 §4-11
Compliance: SOC 2 CC4.1
<!-- test: agents_api.tests.test_vis_compliance -->
"""

import re

import numpy as np
import pandas as pd
from django.test import TestCase

from agents_api.analysis.common import (
    COLOR_BAD,
    COLOR_GOLD,
    COLOR_GOOD,
    COLOR_INFO,
    COLOR_NEUTRAL,
    COLOR_REFERENCE,
    COLOR_WARNING,
    SVEND_COLORS,
)

# ── Allowed color set ────────────────────────────────────────────────────

ALLOWED_HEX = {c.lower() for c in SVEND_COLORS} | {
    c.lower()
    for c in [
        COLOR_GOOD,
        COLOR_BAD,
        COLOR_WARNING,
        COLOR_INFO,
        COLOR_NEUTRAL,
        COLOR_REFERENCE,
        COLOR_GOLD,
    ]
}

# rgba() pattern: rgba(R, G, B, A)
_RGBA_RE = re.compile(r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)")


def _to_base_hex(color_str):
    """Extract base hex (#rrggbb) from a color string. Returns None if not a color."""
    if not color_str or not isinstance(color_str, str):
        return None
    s = color_str.strip().lower()
    if s.startswith("#") and len(s) == 7:
        return s
    m = _RGBA_RE.match(s)
    if m:
        r, g, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"#{r:02x}{g:02x}{b:02x}"
    return None


def _extract_colors(trace):
    """Extract all color strings from a Plotly trace dict."""
    colors = []
    if not isinstance(trace, dict):
        return colors

    # marker.color
    marker = trace.get("marker", {})
    if isinstance(marker, dict):
        mc = marker.get("color")
        if isinstance(mc, str):
            colors.append(mc)
        # marker.line.color
        ml = marker.get("line", {})
        if isinstance(ml, dict) and isinstance(ml.get("color"), str):
            colors.append(ml["color"])

    # line.color
    line = trace.get("line", {})
    if isinstance(line, dict) and isinstance(line.get("color"), str):
        colors.append(line["color"])

    # fillcolor
    fc = trace.get("fillcolor")
    if isinstance(fc, str):
        colors.append(fc)

    return colors


def _is_color_allowed(color_str):
    """Check if a color is in the allowed palette (hex or rgba of allowed hex)."""
    base = _to_base_hex(color_str)
    if base is None:
        return True  # not a parseable color, skip
    return base in ALLOWED_HEX


def _is_legend_inside_chart(legend):
    """Check if legend is positioned inside the chart body (VIS-001 §5.2)."""
    if not isinstance(legend, dict):
        return False
    y = legend.get("y")
    if y is None:
        return False
    orientation = legend.get("orientation", "v")
    # Inside chart = y between 0 and 1 AND not horizontal orientation
    if orientation == "h":
        return False  # horizontal legends above/below are ok
    return 0 < y < 1


def _validate_plot(tc, plot_dict, context=""):
    """Validate a single plot dict against VIS-001. Asserts on FAIL rules."""
    if not isinstance(plot_dict, dict):
        return

    layout = plot_dict.get("layout", {})
    if not isinstance(layout, dict):
        layout = {}

    # FAIL: non-transparent background (VIS-001 §6.1)
    for key in ("paper_bgcolor", "plot_bgcolor"):
        val = layout.get(key)
        if val is not None:
            tc.assertIn(
                val,
                ("rgba(0,0,0,0)", "transparent"),
                f"{context}: {key}='{val}' — must be transparent (VIS-001 §6.1)",
            )

    # FAIL: trace colors not in palette (VIS-001 §4.1)
    for i, trace in enumerate(plot_dict.get("data", [])):
        for color in _extract_colors(trace):
            tc.assertTrue(
                _is_color_allowed(color),
                f"{context}: trace[{i}] uses off-palette color '{color}' (VIS-001 §4.1)",
            )

    # FAIL: legend inside chart (VIS-001 §5.2)
    legend = layout.get("legend", {})
    tc.assertFalse(
        _is_legend_inside_chart(legend),
        f"{context}: legend inside chart body at y={legend.get('y')} (VIS-001 §5.2)",
    )


def _validate_all_plots(tc, result, context=""):
    """Validate all plots in an analysis result dict."""
    if not isinstance(result, dict):
        return
    plots = result.get("plots", [])
    for i, plot in enumerate(plots):
        _validate_plot(tc, plot, context=f"{context} plot[{i}]")


# ── Shared test data ─────────────────────────────────────────────────────

RS = np.random.RandomState(42)
N = 50
NORMAL_50 = list(RS.normal(50, 5, N))
NORMAL_50B = list(np.random.RandomState(43).normal(52, 5, N))
GROUPS_50 = ["A"] * 25 + ["B"] * 25


def _pipeline(result, analysis_type, analysis_id):
    """Apply the standardize_output pipeline (VIS-001 §11.1)."""
    from agents_api.analysis.standardize import standardize_output

    return standardize_output(result, analysis_type, analysis_id)


def _run_stats(analysis_id, config, data_dict):
    """Run statistical analysis through full pipeline (TST-001 §11.6)."""
    from agents_api.analysis.stats import run_statistical_analysis

    df = pd.DataFrame(data_dict)
    result = run_statistical_analysis(df, analysis_id, config)
    return _pipeline(result, "stats", analysis_id)


def _run_spc(analysis_id, config, data_dict):
    """Run SPC analysis through full pipeline."""
    from agents_api.analysis.spc import run_spc_analysis

    df = pd.DataFrame(data_dict)
    result = run_spc_analysis(df, analysis_id, config)
    return _pipeline(result, "spc", analysis_id)


def _run_bayesian(analysis_id, config, data_dict):
    """Run Bayesian analysis through full pipeline."""
    from agents_api.analysis.bayesian import run_bayesian_analysis

    df = pd.DataFrame(data_dict)
    result = run_bayesian_analysis(df, analysis_id, config)
    return _pipeline(result, "bayesian", analysis_id)


def _run_ml(analysis_id, config, data_dict):
    """Run ML analysis through full pipeline."""
    from agents_api.analysis.ml import run_ml_analysis

    df = pd.DataFrame(data_dict)
    result = run_ml_analysis(df, analysis_id, config, user=None)
    return _pipeline(result, "ml", analysis_id)


def _run_viz(analysis_id, config, data_dict):
    """Run visualization analysis through full pipeline."""
    from agents_api.analysis.viz import run_visualization

    df = pd.DataFrame(data_dict)
    result = run_visualization(df, analysis_id, config)
    return _pipeline(result, "viz", analysis_id)


def _run_sim(analysis_id, config, data_dict=None):
    """Run simulation analysis through full pipeline."""
    from agents_api.analysis.simulation import run_simulation

    df = pd.DataFrame(data_dict) if data_dict else pd.DataFrame()
    result = run_simulation(df, analysis_id, config, user=None)
    return _pipeline(result, "simulation", analysis_id)


# ── Color Palette Tests (VIS-001 §4.1) ──────────────────────────────────


class ColorPaletteTest(TestCase):
    """VIS-001 §4.1: All trace colors from SVEND_COLORS or semantic palette."""

    def test_stats_ttest(self):
        r = _run_stats("ttest", {"var1": "x", "mu": 50}, {"x": NORMAL_50})
        _validate_all_plots(self, r, "ttest")

    def test_stats_ttest2(self):
        r = _run_stats(
            "ttest2",
            {"var1": "x", "var2": "y"},
            {"x": NORMAL_50, "y": NORMAL_50B},
        )
        _validate_all_plots(self, r, "ttest2")

    def test_stats_correlation(self):
        r = _run_stats(
            "correlation",
            {"var1": "x", "var2": "y"},
            {"x": NORMAL_50, "y": NORMAL_50B},
        )
        _validate_all_plots(self, r, "correlation")

    def test_stats_anova(self):
        r = _run_stats(
            "anova",
            {"var": "x", "group_var": "g"},
            {
                "x": NORMAL_50 + NORMAL_50B + list(RS.normal(48, 5, 50)),
                "g": ["A"] * 50 + ["B"] * 50 + ["C"] * 50,
            },
        )
        _validate_all_plots(self, r, "anova")

    def test_stats_mann_whitney(self):
        r = _run_stats(
            "mann_whitney",
            {"var": "x", "group_var": "g"},
            {"x": NORMAL_50 + NORMAL_50B, "g": GROUPS_50 + GROUPS_50},
        )
        _validate_all_plots(self, r, "mann_whitney")

    def test_stats_regression(self):
        x = list(RS.normal(0, 1, N))
        y = [2 * xi + RS.normal(0, 0.5) for xi in x]
        r = _run_stats(
            "regression",
            {"response": "y", "predictors": ["x"]},
            {"x": x, "y": y},
        )
        _validate_all_plots(self, r, "regression")

    def test_stats_tukey_hsd(self):
        r = _run_stats(
            "tukey_hsd",
            {"var": "x", "group_var": "g"},
            {
                "x": NORMAL_50 + NORMAL_50B + list(RS.normal(48, 5, 50)),
                "g": ["A"] * 50 + ["B"] * 50 + ["C"] * 50,
            },
        )
        _validate_all_plots(self, r, "tukey_hsd")

    def test_stats_capability_sixpack(self):
        r = _run_stats(
            "capability_sixpack",
            {"var": "x", "lsl": 35, "usl": 65, "subgroup_size": 5},
            {"x": NORMAL_50},
        )
        _validate_all_plots(self, r, "capability_sixpack")

    def test_stats_descriptive(self):
        r = _run_stats("descriptive", {"var": "x"}, {"x": NORMAL_50})
        _validate_all_plots(self, r, "descriptive")

    def test_stats_gage_rr(self):
        parts = [f"P{i}" for i in range(10)] * 6
        operators = (["Op1"] * 10 + ["Op2"] * 10 + ["Op3"] * 10) * 2
        measurements = list(RS.normal(50, 2, 60))
        r = _run_stats(
            "gage_rr",
            {"part": "part", "operator": "operator", "measurement": "meas"},
            {"part": parts, "operator": operators, "meas": measurements},
        )
        _validate_all_plots(self, r, "gage_rr")

    def test_spc_i_mr(self):
        r = _run_spc("i_mr", {}, {"x": NORMAL_50})
        _validate_all_plots(self, r, "spc_i_mr")

    def test_spc_xbar_r(self):
        r = _run_spc("xbar_r", {"subgroup_size": 5}, {"x": NORMAL_50})
        _validate_all_plots(self, r, "spc_xbar_r")

    def test_spc_p_chart(self):
        r = _run_spc(
            "p_chart",
            {"defectives": "d", "sample_size": "n"},
            {"d": list(RS.binomial(100, 0.05, 30)), "n": [100] * 30},
        )
        _validate_all_plots(self, r, "spc_p_chart")

    def test_bayesian_ttest(self):
        r = _run_bayesian(
            "bayes_ttest",
            {"var1": "x", "var2": "y"},
            {"x": NORMAL_50, "y": NORMAL_50B},
        )
        _validate_all_plots(self, r, "bayes_ttest")

    def test_bayesian_ewma(self):
        r = _run_bayesian(
            "bayes_ewma",
            {"measurement": "x"},
            {"x": NORMAL_50},
        )
        _validate_all_plots(self, r, "bayes_ewma")

    def test_ml_clustering(self):
        r = _run_ml(
            "clustering",
            {"features": ["x", "y"], "n_clusters": 3},
            {"x": NORMAL_50, "y": NORMAL_50B},
        )
        _validate_all_plots(self, r, "ml_clustering")

    def test_viz_histogram(self):
        r = _run_viz("histogram", {"var": "x"}, {"x": NORMAL_50})
        _validate_all_plots(self, r, "viz_histogram")

    def test_viz_scatter(self):
        r = _run_viz(
            "scatter",
            {"x": "x", "y": "y"},
            {"x": NORMAL_50, "y": NORMAL_50B},
        )
        _validate_all_plots(self, r, "viz_scatter")

    def test_simulation_monte_carlo(self):
        r = _run_sim(
            "monte_carlo",
            {
                "expression": "x + y",
                "variables": {
                    "x": {"distribution": "normal", "mean": 10, "std": 1},
                    "y": {"distribution": "normal", "mean": 5, "std": 0.5},
                },
                "n_simulations": 500,
            },
        )
        _validate_all_plots(self, r, "sim_monte_carlo")

    def test_normalization(self):
        """VIS-001 §4.2: Off-palette colors are normalized to palette equivalents."""
        from agents_api.analysis.chart_defaults import _normalize_color

        # Known off-palette → palette mappings
        self.assertEqual(_normalize_color("#d63031"), COLOR_BAD)
        self.assertEqual(_normalize_color("#e85747"), COLOR_BAD)
        self.assertEqual(_normalize_color("#00b894"), COLOR_GOOD)
        self.assertEqual(_normalize_color("#4a90d9"), "#5b8bd6")
        # Palette colors pass through unchanged
        self.assertEqual(_normalize_color("#4a9f6e"), "#4a9f6e")
        self.assertEqual(_normalize_color(COLOR_BAD), COLOR_BAD)
        # None/empty pass through
        self.assertIsNone(_normalize_color(None))
        self.assertEqual(_normalize_color(""), "")


# ── Legend Position Tests (VIS-001 §5) ───────────────────────────────────


class LegendPositionTest(TestCase):
    """VIS-001 §5: Legend placement — horizontal below chart, not inside."""

    def test_no_legend_inside_chart(self):
        """VIS-001 §5.2: Legend must not be inside chart body."""
        # Run analyses that historically had inside-chart legends
        r = _run_spc("i_mr", {}, {"x": NORMAL_50})
        _validate_all_plots(self, r, "spc_i_mr legend")

    def test_spc_xbar_r_legend(self):
        r = _run_spc("xbar_r", {"subgroup_size": 5}, {"x": NORMAL_50})
        _validate_all_plots(self, r, "spc_xbar_r legend")

    def test_legend_helper(self):
        """Verify _is_legend_inside_chart detection logic."""
        # Inside chart — FAIL
        self.assertTrue(_is_legend_inside_chart({"x": 0.98, "y": 0.98, "xanchor": "right"}))
        self.assertTrue(_is_legend_inside_chart({"x": 1, "y": 0.5}))
        # Below chart — OK
        self.assertFalse(_is_legend_inside_chart({"y": -0.25, "orientation": "h"}))
        # Above chart horizontal — OK
        self.assertFalse(_is_legend_inside_chart({"y": 1.15, "orientation": "h", "x": 0.5}))
        # Empty/None — OK
        self.assertFalse(_is_legend_inside_chart({}))
        self.assertFalse(_is_legend_inside_chart(None))


# ── Background Tests (VIS-001 §6) ───────────────────────────────────────


class BackgroundTest(TestCase):
    """VIS-001 §6.1: Transparent backgrounds for theme compatibility."""

    def test_stats_transparent(self):
        r = _run_stats("ttest", {"var1": "x", "mu": 50}, {"x": NORMAL_50})
        _validate_all_plots(self, r, "ttest bg")

    def test_spc_transparent(self):
        r = _run_spc("i_mr", {}, {"x": NORMAL_50})
        _validate_all_plots(self, r, "spc bg")

    def test_ml_transparent(self):
        r = _run_ml(
            "clustering",
            {"features": ["x", "y"], "n_clusters": 3},
            {"x": NORMAL_50, "y": NORMAL_50B},
        )
        _validate_all_plots(self, r, "ml bg")


# ── Dimensions Tests (VIS-001 §7) ───────────────────────────────────────


class DimensionsTest(TestCase):
    """VIS-001 §7.1: Standard chart height and margins."""

    def test_standard_height(self):
        r = _run_stats("ttest", {"var1": "x", "mu": 50}, {"x": NORMAL_50})
        for i, plot in enumerate(r.get("plots", [])):
            layout = plot.get("layout", {})
            h = layout.get("height")
            if h is not None:
                self.assertIn(h, (300, 350), f"plot[{i}] height={h}, expected 300 or 350")


# ── Typography Tests (VIS-001 §8) ───────────────────────────────────────


class TypographyTest(TestCase):
    """VIS-001 §8.1: Standard font family."""

    def test_font_family(self):
        r = _run_stats("ttest", {"var1": "x", "mu": 50}, {"x": NORMAL_50})
        for i, plot in enumerate(r.get("plots", [])):
            layout = plot.get("layout", {})
            font = layout.get("font", {})
            if font:
                family = font.get("family", "")
                self.assertIn("Inter", family, f"plot[{i}] font={family}")


# ── Post-Processing Pipeline Tests (VIS-001 §11) ────────────────────────


class PostProcessingTest(TestCase):
    """VIS-001 §11.1: All chart output passes through apply_chart_defaults."""

    def test_apply_chart_defaults_sets_bgcolor(self):
        """Verify apply_chart_defaults enforces transparent background."""
        from agents_api.analysis.chart_defaults import apply_chart_defaults

        plot = {
            "data": [{"type": "scatter", "x": [1, 2], "y": [3, 4]}],
            "layout": {},
        }
        apply_chart_defaults(plot)
        self.assertEqual(plot["layout"]["paper_bgcolor"], "rgba(0,0,0,0)")
        self.assertEqual(plot["layout"]["plot_bgcolor"], "rgba(0,0,0,0)")

    def test_apply_chart_defaults_sets_legend(self):
        """Verify apply_chart_defaults sets legend when missing."""
        from agents_api.analysis.chart_defaults import apply_chart_defaults

        plot = {"data": [{"type": "scatter", "x": [1], "y": [1]}], "layout": {}}
        apply_chart_defaults(plot)
        legend = plot["layout"]["legend"]
        self.assertEqual(legend["orientation"], "h")
        self.assertLess(legend["y"], 0)  # below chart

    def test_apply_chart_defaults_fixes_inside_legend(self):
        """Verify apply_chart_defaults replaces inside-chart legends."""
        from agents_api.analysis.chart_defaults import apply_chart_defaults

        plot = {
            "data": [{"type": "scatter", "x": [1], "y": [1]}],
            "layout": {"legend": {"x": 0.98, "y": 0.98, "xanchor": "right"}},
        }
        apply_chart_defaults(plot)
        legend = plot["layout"]["legend"]
        # Should have been replaced with standard position
        self.assertFalse(_is_legend_inside_chart(legend))

    def test_apply_chart_defaults_normalizes_colors(self):
        """Verify apply_chart_defaults normalizes off-palette trace colors."""
        from agents_api.analysis.chart_defaults import apply_chart_defaults

        plot = {
            "data": [
                {"type": "scatter", "x": [1], "y": [1], "marker": {"color": "#d63031"}},
                {"type": "scatter", "x": [1], "y": [1], "line": {"color": "#00b894"}},
            ],
            "layout": {},
        }
        apply_chart_defaults(plot)
        self.assertEqual(plot["data"][0]["marker"]["color"], COLOR_BAD)
        self.assertEqual(plot["data"][1]["line"]["color"], COLOR_GOOD)

    def test_standardize_output_applies_defaults(self):
        """Verify standardize_output() calls apply_chart_defaults on plots."""
        from agents_api.analysis.standardize import standardize_output

        result = {
            "summary": "Test",
            "plots": [
                {
                    "data": [{"type": "bar", "x": ["A"], "y": [1]}],
                    "layout": {},
                }
            ],
        }
        out = standardize_output(result, "stats", "test")
        layout = out["plots"][0]["layout"]
        self.assertEqual(layout["paper_bgcolor"], "rgba(0,0,0,0)")


# ── Color Utility Tests ──────────────────────────────────────────────────


class ColorUtilTest(TestCase):
    """Tests for color validation helper functions."""

    def test_to_base_hex_from_hex(self):
        self.assertEqual(_to_base_hex("#4a9f6e"), "#4a9f6e")
        self.assertEqual(_to_base_hex("#D06060"), "#d06060")

    def test_to_base_hex_from_rgba(self):
        self.assertEqual(_to_base_hex("rgba(74, 159, 110, 0.4)"), "#4a9f6e")
        self.assertEqual(_to_base_hex("rgba(208,96,96,0.5)"), "#d06060")

    def test_to_base_hex_none(self):
        self.assertIsNone(_to_base_hex(None))
        self.assertIsNone(_to_base_hex(""))
        self.assertIsNone(_to_base_hex("red"))

    def test_is_color_allowed_palette(self):
        for c in SVEND_COLORS:
            self.assertTrue(_is_color_allowed(c), f"{c} should be allowed")
        self.assertTrue(_is_color_allowed(COLOR_BAD))
        self.assertTrue(_is_color_allowed(COLOR_GOOD))

    def test_is_color_allowed_rgba_of_palette(self):
        self.assertTrue(_is_color_allowed("rgba(74, 159, 110, 0.4)"))  # rgba of #4a9f6e

    def test_is_color_allowed_rejects_off_palette(self):
        self.assertFalse(_is_color_allowed("#d63031"))
        self.assertFalse(_is_color_allowed("#00b894"))

    def test_extract_colors(self):
        trace = {
            "type": "scatter",
            "marker": {"color": "#4a9f6e", "line": {"color": "#d06060"}},
            "line": {"color": "#4a9faf"},
            "fillcolor": "rgba(74, 159, 110, 0.3)",
        }
        colors = _extract_colors(trace)
        self.assertEqual(len(colors), 4)
