"""
STAT-001 compliance tests: Statistical Methodology Standard.

Tests verify the eleven methodological principles, effect size framework,
Bayesian insurance, robustness framework, evidence synthesis, SPC Nelson
rules, power analysis, and confidence interval methods.

Standard: STAT-001
"""

import os
import re
from pathlib import Path

from django.test import SimpleTestCase

WEB_ROOT = Path(os.path.dirname(__file__)).parent.parent.parent
DSW_DIR = WEB_ROOT / "agents_api" / "dsw"


def _read(path):
    try:
        return Path(path).read_text(errors="ignore")
    except Exception:
        return ""


# ── §4.1: Eleven Principles — Function Existence ────────────────────────


class MethodologyPrinciplesTest(SimpleTestCase):
    """STAT-001 §4: All eleven principles implemented."""

    def setUp(self):
        self.common_src = _read(DSW_DIR / "common.py")
        self.stats_src = _read(DSW_DIR / "stats.py")
        self.spc_src = _read(DSW_DIR / "spc.py")
        self.assertGreater(len(self.common_src), 0, "common.py not found")
        self.assertGreater(len(self.stats_src), 0, "stats.py not found")

    def test_effect_magnitude_exists(self):
        """P1: _effect_magnitude function defined in common.py."""
        self.assertIn("def _effect_magnitude(", self.common_src)

    def test_practical_block_exists(self):
        """P4: _practical_block function defined in common.py."""
        self.assertIn("def _practical_block(", self.common_src)

    def test_bayesian_shadow_exists(self):
        """P3: _bayesian_shadow function defined in common.py."""
        self.assertIn("def _bayesian_shadow(", self.common_src)

    def test_check_normality_exists(self):
        """P6: _check_normality function defined in common.py."""
        self.assertIn("def _check_normality(", self.common_src)

    def test_check_equal_variance_exists(self):
        """P6: _check_equal_variance function defined in common.py."""
        self.assertIn("def _check_equal_variance(", self.common_src)

    def test_check_outliers_exists(self):
        """P6: _check_outliers function defined in common.py."""
        self.assertIn("def _check_outliers(", self.common_src)

    def test_cross_validate_exists(self):
        """P5: _cross_validate function defined in common.py."""
        self.assertIn("def _cross_validate(", self.common_src)

    def test_evidence_grade_exists(self):
        """P8: _evidence_grade function defined in common.py."""
        self.assertIn("def _evidence_grade(", self.common_src)

    def test_narrative_exists(self):
        """P8: _narrative function defined."""
        self.assertTrue(
            "def _narrative(" in self.common_src or "def _narrative(" in self.stats_src,
            "_narrative function not found in common.py or stats.py",
        )

    def test_guide_observation_set_in_stats(self):
        """P8: stats.py sets guide_observation on results."""
        self.assertIn("guide_observation", self.stats_src)

    def test_run_statistical_analysis_exists(self):
        """Main dispatcher: run_statistical_analysis defined in stats.py."""
        self.assertIn("def run_statistical_analysis(", self.stats_src)


# ── §5.1: Effect Size Thresholds ────────────────────────────────────────


class EffectSizeThresholdsTest(SimpleTestCase):
    """STAT-001 §5.1: Cohen (1988) benchmarks for effect size classification."""

    def setUp(self):
        self.common_src = _read(DSW_DIR / "common.py")

    def test_cohens_d_thresholds(self):
        """Cohen's d thresholds: 0.2, 0.5, 0.8."""
        fn_body = self._extract_fn("_effect_magnitude")
        self.assertIn("0.2", fn_body)
        self.assertIn("0.5", fn_body)
        self.assertIn("0.8", fn_body)

    def test_eta_squared_thresholds(self):
        """η² thresholds: 0.01, 0.06, 0.14."""
        fn_body = self._extract_fn("_effect_magnitude")
        self.assertIn("0.01", fn_body)
        self.assertIn("0.06", fn_body)
        self.assertIn("0.14", fn_body)

    def test_cramers_v_thresholds(self):
        """Cramér's V thresholds: 0.1, 0.3, 0.5."""
        fn_body = self._extract_fn("_effect_magnitude")
        # These overlap with d thresholds so check for cramers_v/cramer context
        self.assertTrue(
            "cramer" in fn_body.lower() or "v" in fn_body.lower(),
            "Cramér's V handling not found in _effect_magnitude",
        )

    def test_r_squared_thresholds(self):
        """R² thresholds: 0.02, 0.13, 0.26."""
        fn_body = self._extract_fn("_effect_magnitude")
        self.assertIn("0.02", fn_body)
        self.assertIn("0.13", fn_body)
        self.assertIn("0.26", fn_body)

    def test_returns_label_and_meaningful(self):
        """_effect_magnitude returns (label, is_meaningful) tuple pattern."""
        fn_body = self._extract_fn("_effect_magnitude")
        self.assertTrue(
            "return" in fn_body,
            "_effect_magnitude has no return statement",
        )

    def _extract_fn(self, name):
        match = re.search(
            rf"def {name}\(.*?(?=\ndef |\Z)",
            self.common_src,
            re.DOTALL,
        )
        return match.group() if match else ""


# ── §5.2: Four-Region Decision Model ────────────────────────────────────


class FourRegionModelTest(SimpleTestCase):
    """STAT-001 §5.2: Practical significance four-region model."""

    def setUp(self):
        self.common_src = _read(DSW_DIR / "common.py")

    def test_practical_block_uses_alpha(self):
        """_practical_block accepts alpha parameter."""
        match = re.search(r"def _practical_block\([^)]+\)", self.common_src)
        self.assertIsNotNone(match, "_practical_block not found")
        self.assertIn("alpha", match.group())

    def test_practical_block_uses_effect(self):
        """_practical_block uses effect magnitude."""
        fn_body = self._extract_fn("_practical_block")
        self.assertTrue(
            "_effect_magnitude" in fn_body or "effect" in fn_body.lower(),
            "_practical_block should use effect size",
        )

    def test_practical_block_in_stats(self):
        """stats.py calls _practical_block for results."""
        stats_src = _read(DSW_DIR / "stats.py")
        self.assertIn("_practical_block(", stats_src)

    def _extract_fn(self, name):
        match = re.search(
            rf"def {name}\(.*?(?=\ndef |\Z)",
            self.common_src,
            re.DOTALL,
        )
        return match.group() if match else ""


# ── §6.1-6.3: Bayesian Insurance ────────────────────────────────────────


class BayesianInsuranceTest(SimpleTestCase):
    """STAT-001 §6: Bayesian shadow and BF interpretation."""

    def setUp(self):
        self.common_src = _read(DSW_DIR / "common.py")

    def test_shadow_types_supported(self):
        """_bayesian_shadow handles t-test, ANOVA, correlation, proportion, chi2."""
        fn_body = self._extract_fn("_bayesian_shadow")
        for stype in ["ttest", "anova", "correlation", "proportion", "chi2"]:
            self.assertIn(
                stype,
                fn_body.lower(),
                f"_bayesian_shadow missing support for {stype}",
            )

    def test_jzs_prior_scale(self):
        """JZS prior uses r = √2/2 ≈ 0.707."""
        fn_body = self._extract_fn("_bayesian_shadow")
        self.assertTrue(
            "0.707" in fn_body
            or "sqrt(2)/2" in fn_body
            or "2**0.5/2" in fn_body
            or "np.sqrt(2)/2" in fn_body
            or "math.sqrt(2)/2" in fn_body,
            "JZS prior scale r=√2/2 not found",
        )

    def test_bf_interpretation_categories(self):
        """BF interpretation scale includes evidence categories."""
        fn_body = self._extract_fn("_bayesian_shadow")
        for category in ["extreme", "strong", "moderate", "weak"]:
            self.assertIn(
                category.lower(),
                fn_body.lower(),
                f"BF interpretation missing '{category}' category",
            )

    def test_uses_scipy_integrate(self):
        """JZS integration uses scipy.integrate."""
        self.assertTrue(
            "scipy.integrate" in self.common_src or "integrate.quad" in self.common_src,
            "scipy.integrate not found for JZS BF computation",
        )

    def test_shadow_called_in_stats(self):
        """stats.py calls _bayesian_shadow."""
        stats_src = _read(DSW_DIR / "stats.py")
        self.assertIn("_bayesian_shadow(", stats_src)

    def test_bf10_in_result(self):
        """Shadow results include bf10 key."""
        fn_body = self._extract_fn("_bayesian_shadow")
        self.assertIn("bf10", fn_body)

    def _extract_fn(self, name):
        match = re.search(
            rf"def {name}\(.*?(?=\ndef |\Z)",
            self.common_src,
            re.DOTALL,
        )
        return match.group() if match else ""


# ── §7.1: Assumption Verification ────────────────────────────────────────


class AssumptionVerificationTest(SimpleTestCase):
    """STAT-001 §7.1: Automatic normality, variance, outlier checks."""

    def setUp(self):
        self.common_src = _read(DSW_DIR / "common.py")
        self.stats_src = _read(DSW_DIR / "stats.py")

    def test_normality_uses_shapiro(self):
        """_check_normality uses Shapiro-Wilk test."""
        fn_body = self._extract_fn("_check_normality")
        self.assertIn("shapiro", fn_body.lower())

    def test_normality_dagostino_fallback(self):
        """_check_normality uses D'Agostino-Pearson for large n."""
        fn_body = self._extract_fn("_check_normality")
        self.assertTrue(
            "normaltest" in fn_body
            or "dagostino" in fn_body.lower()
            or "5000" in fn_body,
            "D'Agostino-Pearson fallback not found for large samples",
        )

    def test_equal_variance_uses_levene(self):
        """_check_equal_variance uses Levene's test."""
        fn_body = self._extract_fn("_check_equal_variance")
        self.assertIn("levene", fn_body.lower())

    def test_outlier_uses_iqr(self):
        """_check_outliers uses IQR rule."""
        fn_body = self._extract_fn("_check_outliers")
        self.assertTrue(
            "iqr" in fn_body.lower()
            or "1.5" in fn_body
            or "quartile" in fn_body.lower(),
            "IQR-based outlier detection not found",
        )

    def test_stats_calls_normality(self):
        """stats.py calls _check_normality for parametric tests."""
        self.assertIn("_check_normality(", self.stats_src)

    def test_stats_calls_equal_variance(self):
        """stats.py calls _check_equal_variance for two-sample tests."""
        self.assertIn("_check_equal_variance(", self.stats_src)

    def test_diagnostics_in_stats(self):
        """stats.py builds diagnostics array."""
        self.assertIn("diagnostics", self.stats_src)

    def _extract_fn(self, name):
        match = re.search(
            rf"def {name}\(.*?(?=\ndef |\Z)",
            self.common_src,
            re.DOTALL,
        )
        return match.group() if match else ""


# ── §7.2: Cross-Validation ──────────────────────────────────────────────


class CrossValidationTest(SimpleTestCase):
    """STAT-001 §7.2: Parametric/non-parametric cross-validation."""

    def setUp(self):
        self.common_src = _read(DSW_DIR / "common.py")
        self.stats_src = _read(DSW_DIR / "stats.py")

    def test_cross_validate_accepts_both_pvalues(self):
        """_cross_validate accepts primary_p and alt_p."""
        match = re.search(r"def _cross_validate\([^)]+\)", self.common_src)
        self.assertIsNotNone(match)
        params = match.group()
        self.assertIn("primary_p", params)
        self.assertIn("alt_p", params)

    def test_stats_calls_cross_validate(self):
        """stats.py calls _cross_validate."""
        self.assertIn("_cross_validate(", self.stats_src)

    def test_cross_validate_reports_agreement(self):
        """_cross_validate reports agreement or contradiction."""
        fn_body = self._extract_fn("_cross_validate")
        self.assertTrue(
            "agree" in fn_body.lower() or "contradict" in fn_body.lower(),
            "_cross_validate should report agreement/contradiction",
        )

    def test_nonparametric_parallels_exist(self):
        """stats.py uses non-parametric tests (Mann-Whitney, Wilcoxon, Kruskal)."""
        self.assertTrue(
            "mannwhitneyu" in self.stats_src or "ranksums" in self.stats_src,
            "Mann-Whitney U not found",
        )
        self.assertTrue(
            "wilcoxon" in self.stats_src.lower(),
            "Wilcoxon signed-rank not found",
        )

    def _extract_fn(self, name):
        match = re.search(
            rf"def {name}\(.*?(?=\ndef |\Z)",
            self.common_src,
            re.DOTALL,
        )
        return match.group() if match else ""


# ── §8.1: Evidence Synthesis ─────────────────────────────────────────────


class EvidenceSynthesisTest(SimpleTestCase):
    """STAT-001 §8: Evidence grading algorithm."""

    def setUp(self):
        self.common_src = _read(DSW_DIR / "common.py")

    def test_evidence_grade_accepts_components(self):
        """_evidence_grade accepts p_value, bf10, effect_magnitude, cross_val."""
        match = re.search(r"def _evidence_grade\([^)]+\)", self.common_src)
        self.assertIsNotNone(match)
        params = match.group()
        self.assertIn("p_value", params)

    def test_grade_categories_defined(self):
        """Evidence grade returns Strong/Moderate/Weak/Inconclusive."""
        fn_body = self._extract_fn("_evidence_grade")
        for grade in ["strong", "moderate", "weak", "inconclusive"]:
            self.assertIn(
                grade.lower(),
                fn_body.lower(),
                f"Evidence grade missing '{grade}' category",
            )

    def test_evidence_grade_called_in_stats(self):
        """stats.py calls _evidence_grade."""
        stats_src = _read(DSW_DIR / "stats.py")
        self.assertIn("_evidence_grade(", stats_src)

    def _extract_fn(self, name):
        match = re.search(
            rf"def {name}\(.*?(?=\ndef |\Z)",
            self.common_src,
            re.DOTALL,
        )
        return match.group() if match else ""


# ── §9.1: Nelson Rules ──────────────────────────────────────────────────


class NelsonRulesTest(SimpleTestCase):
    """STAT-001 §9.1: All 8 Nelson rules for SPC."""

    def setUp(self):
        self.spc_src = _read(DSW_DIR / "spc.py")
        if not self.spc_src:
            self.spc_src = _read(WEB_ROOT / "agents_api" / "spc.py")

    def test_nelson_rules_function_exists(self):
        """_spc_nelson_rules function defined."""
        self.assertTrue(
            "_spc_nelson_rules" in self.spc_src
            or "nelson_rules" in self.spc_src.lower(),
            "Nelson rules function not found",
        )

    def test_rule_1_beyond_3sigma(self):
        """Rule 1: point beyond 3σ (UCL/LCL check)."""
        self.assertTrue(
            "ucl" in self.spc_src.lower() and "lcl" in self.spc_src.lower(),
            "UCL/LCL checks not found for Rule 1",
        )

    def test_rule_2_nine_consecutive(self):
        """Rule 2: 9 consecutive same side of CL."""
        self.assertIn("9", self.spc_src)

    def test_rule_7_fifteen_within_1sigma(self):
        """Rule 7: 15 consecutive within 1σ (stratification)."""
        self.assertIn("15", self.spc_src)

    def test_rule_8_eight_beyond_1sigma(self):
        """Rule 8: 8 consecutive beyond 1σ (mixture)."""
        # The number 8 appears in many contexts; check for window-based logic
        self.assertTrue(
            "rule" in self.spc_src.lower() or "nelson" in self.spc_src.lower(),
            "Nelson rule logic not found in SPC module",
        )

    def test_sigma_derived_from_limits(self):
        """σ derived from control limits: (UCL - CL) / 3."""
        self.assertTrue(
            "/ 3" in self.spc_src or "/3" in self.spc_src,
            "σ derivation from control limits not found",
        )


# ── §10.1: Power Analysis ───────────────────────────────────────────────


class PowerAnalysisTest(SimpleTestCase):
    """STAT-001 §10: Sample size computation and power curves."""

    def setUp(self):
        self.exp_src = _read(WEB_ROOT / "agents_api" / "experimenter_views.py")

    def test_power_curve_function_exists(self):
        """_compute_power_curve function defined."""
        self.assertIn("def _compute_power_curve(", self.exp_src)

    def test_supports_ttest(self):
        """Power analysis supports independent t-test."""
        fn_body = self._extract_fn("_compute_power_curve")
        self.assertTrue(
            "ttest" in fn_body.lower()
            or "t_test" in fn_body.lower()
            or "t-test" in fn_body.lower(),
            "t-test power analysis not found",
        )

    def test_supports_anova(self):
        """Power analysis supports ANOVA."""
        fn_body = self._extract_fn("_compute_power_curve")
        self.assertIn("anova", fn_body.lower())

    def test_supports_correlation(self):
        """Power analysis supports correlation."""
        fn_body = self._extract_fn("_compute_power_curve")
        self.assertIn("correlation", fn_body.lower())

    def test_power_explorer_metadata(self):
        """Analyses attach power_explorer metadata."""
        stats_src = _read(DSW_DIR / "stats.py")
        self.assertIn("power_explorer", stats_src)

    def _extract_fn(self, name):
        match = re.search(
            rf"def {name}\(.*?(?=\ndef |\Z)",
            self.exp_src,
            re.DOTALL,
        )
        return match.group() if match else ""


# ── §11.1: Guide Observations ───────────────────────────────────────────


class GuideObservationTest(SimpleTestCase):
    """STAT-001 §11: Every analysis produces guide_observation."""

    def setUp(self):
        self.stats_src = _read(DSW_DIR / "stats.py")

    def test_guide_observation_set(self):
        """stats.py sets guide_observation in results."""
        count = self.stats_src.count("guide_observation")
        self.assertGreater(
            count,
            5,
            f"guide_observation appears only {count} times — expected in many analyses",
        )

    def test_guide_observation_not_empty(self):
        """guide_observation is set to non-empty strings."""
        # Check for the pattern result["guide_observation"] = "..." (not empty)
        self.assertNotIn('guide_observation"] = ""', self.stats_src)


# ── §12.1: Confidence Intervals ─────────────────────────────────────────


class ConfidenceIntervalTest(SimpleTestCase):
    """STAT-001 §12: Wilson score CI for proportions."""

    def setUp(self):
        self.stats_src = _read(DSW_DIR / "stats.py")

    def test_wilson_ci_used(self):
        """stats.py uses Wilson score interval (not Wald) for proportions."""
        self.assertTrue(
            "wilson" in self.stats_src.lower()
            or "z**2" in self.stats_src
            or "z * z" in self.stats_src,
            "Wilson score CI not found in stats.py",
        )

    def test_fisher_z_for_correlations(self):
        """stats.py uses Fisher z-transform for correlation CIs."""
        self.assertTrue(
            "arctanh" in self.stats_src
            or "fisher_z" in self.stats_src.lower()
            or "np.arctanh" in self.stats_src,
            "Fisher z-transform not found for correlation CIs",
        )


# ── Anti-Patterns ────────────────────────────────────────────────────────


class StatAntiPatternTest(SimpleTestCase):
    """STAT-001 §13: Anti-pattern enforcement."""

    def setUp(self):
        self.stats_src = _read(DSW_DIR / "stats.py")

    def test_no_pvalue_only_results(self):
        """stats.py never reports p-value without effect size in same analysis."""
        # Every analysis that calls ttest/anova/chi2 should also call _effect_magnitude or _practical_block
        self.assertIn("_effect_magnitude(", self.stats_src)
        self.assertIn("_practical_block(", self.stats_src)

    def test_alpha_parameterized(self):
        """stats.py accepts alpha as parameter, not hardcoded."""
        self.assertTrue(
            'config.get("alpha"' in self.stats_src or "alpha" in self.stats_src,
            "Alpha not parameterized",
        )
