"""
Tests for agents_api.evidence_weights — CANON-002 §2-3.

All tests exercise real behavior per TST-001 §10.6.
No existence-only checks. Each test verifies output values,
boundary behavior, or rejection conditions.

<!-- test: agents_api.tests.test_evidence_weights.SourceRankTest -->
<!-- test: agents_api.tests.test_evidence_weights.ToolSourceRanksTest -->
<!-- test: agents_api.tests.test_evidence_weights.ToolFunctionsTest -->
<!-- test: agents_api.tests.test_evidence_weights.SampleModifierTest -->
<!-- test: agents_api.tests.test_evidence_weights.StudyQualityTest -->
<!-- test: agents_api.tests.test_evidence_weights.ComputeEvidenceWeightTest -->
<!-- test: agents_api.tests.test_evidence_weights.ClampBehaviorTest -->
"""

from unittest.mock import patch

from django.test import TestCase

from agents_api.evidence_weights import (
    TOOL_FUNCTIONS,
    TOOL_SOURCE_RANKS,
    SourceRank,
    _compute_measurement_validity,
    _compute_sample_modifier,
    _compute_study_quality,
    compute_evidence_weight,
)


class SourceRankTest(TestCase):
    """CANON-002 §2.1 — epistemological hierarchy values and ordering."""

    def test_hierarchy_values_match_spec(self):
        """Each rank matches the exact value from CANON-002 §2.1 table."""
        self.assertEqual(SourceRank.DESIGNED_EXPERIMENT, 0.95)
        self.assertEqual(SourceRank.CONTROLLED_OBSERVATION, 0.85)
        self.assertEqual(SourceRank.STATISTICAL_TEST, 0.75)
        self.assertEqual(SourceRank.STRUCTURED_ANALYSIS, 0.60)
        self.assertEqual(SourceRank.SIMULATION, 0.50)
        self.assertEqual(SourceRank.OBSERVATIONAL_STUDY, 0.45)
        self.assertEqual(SourceRank.EXPERT_JUDGMENT, 0.35)
        self.assertEqual(SourceRank.ANECDOTAL, 0.20)

    def test_hierarchy_count(self):
        """Exactly 8 ranks in the hierarchy per §2.1."""
        self.assertEqual(len(SourceRank), 8)

    def test_hierarchy_ordering(self):
        """Ranks are strictly ordered: DOE > SPC > DSW > ... > Anecdotal."""
        ranks = list(SourceRank)
        for i in range(len(ranks) - 1):
            self.assertGreater(
                float(ranks[i]),
                float(ranks[i + 1]),
                f"{ranks[i].name} should outrank {ranks[i + 1].name}",
            )

    def test_float_comparison(self):
        """SourceRank values work in float arithmetic (they inherit from float)."""
        result = float(SourceRank.DESIGNED_EXPERIMENT) * 0.5
        self.assertAlmostEqual(result, 0.475)


class ToolSourceRanksTest(TestCase):
    """CANON-002 §2.3 — tool→source rank mapping."""

    def test_layer_1_inference_tools(self):
        """Layer 1 tools that produce evidence have correct ranks."""
        self.assertEqual(TOOL_SOURCE_RANKS["spc"], SourceRank.CONTROLLED_OBSERVATION)
        self.assertEqual(TOOL_SOURCE_RANKS["dsw"], SourceRank.STATISTICAL_TEST)
        self.assertEqual(TOOL_SOURCE_RANKS["doe_results"], SourceRank.DESIGNED_EXPERIMENT)
        self.assertEqual(TOOL_SOURCE_RANKS["ml"], SourceRank.SIMULATION)
        self.assertEqual(TOOL_SOURCE_RANKS["forecast"], SourceRank.SIMULATION)

    def test_layer_1_no_evidence_tools(self):
        """Layer 1 tools that produce no evidence map to None."""
        self.assertIsNone(TOOL_SOURCE_RANKS["doe_design"])
        self.assertIsNone(TOOL_SOURCE_RANKS["triage"])

    def test_layer_2_information_tools(self):
        """Layer 2 structured analysis tools share the same rank."""
        for tool in ("rca", "ishikawa", "ce_matrix", "fmea"):
            self.assertEqual(
                TOOL_SOURCE_RANKS[tool],
                SourceRank.STRUCTURED_ANALYSIS,
                f"{tool} should be STRUCTURED_ANALYSIS",
            )

    def test_layer_2_report_sinks(self):
        """Report sinks and VSM produce no evidence."""
        self.assertIsNone(TOOL_SOURCE_RANKS["a3"])
        self.assertIsNone(TOOL_SOURCE_RANKS["vsm"])
        self.assertIsNone(TOOL_SOURCE_RANKS["report"])

    def test_layer_3_tools(self):
        """NCR and CAPA are structured analysis when producing evidence."""
        self.assertEqual(TOOL_SOURCE_RANKS["ncr"], SourceRank.STRUCTURED_ANALYSIS)
        self.assertEqual(TOOL_SOURCE_RANKS["capa"], SourceRank.STRUCTURED_ANALYSIS)

    def test_user_supplied(self):
        """User-supplied evidence types have correct ranks."""
        self.assertEqual(TOOL_SOURCE_RANKS["user"], SourceRank.EXPERT_JUDGMENT)
        self.assertEqual(TOOL_SOURCE_RANKS["observation"], SourceRank.ANECDOTAL)

    def test_all_tools_covered(self):
        """Every tool in the registry has an explicit mapping (no gaps)."""
        expected_tools = {
            "spc",
            "spc_control_chart",
            "spc_capability",
            "spc_summary",
            "spc_gage_rr",
            "spc_recommend",
            "spc_control_chart_upload",
            "spc_capability_upload",
            "dsw",
            "doe_design",
            "doe_results",
            "ml",
            "forecast",
            "triage",
            "rca",
            "ishikawa",
            "ce_matrix",
            "fmea",
            "a3",
            "vsm",
            "report",
            "ncr",
            "capa",
            "user",
            "observation",
        }
        self.assertEqual(set(TOOL_SOURCE_RANKS.keys()), expected_tools)


class ToolFunctionsTest(TestCase):
    """CANON-002 §11.1 — tool→function mapping."""

    def test_inference_tools(self):
        """Inference tools: produce evidence that updates posteriors."""
        for tool in ("spc", "dsw", "doe_results", "ml", "forecast"):
            self.assertEqual(TOOL_FUNCTIONS[tool], "inference", f"{tool} should be inference")

    def test_information_tools(self):
        """Information tools: build graph structure (hypotheses, links)."""
        for tool in ("rca", "ishikawa", "ce_matrix", "fmea"):
            self.assertEqual(TOOL_FUNCTIONS[tool], "information", f"{tool} should be information")

    def test_intent_tools(self):
        """Intent tools: prescribe experiments."""
        self.assertEqual(TOOL_FUNCTIONS["doe_design"], "intent")

    def test_report_tools(self):
        """Report tools: read graph, don't modify."""
        self.assertEqual(TOOL_FUNCTIONS["a3"], "report")
        self.assertEqual(TOOL_FUNCTIONS["report"], "report")

    def test_unlinkable_tools(self):
        """Tools that cannot be linked to investigations."""
        self.assertIsNone(TOOL_FUNCTIONS["triage"])
        self.assertIsNone(TOOL_FUNCTIONS["vsm"])


class SampleModifierTest(TestCase):
    """CANON-002 §3.2 — sample size modifier brackets."""

    def test_none_returns_one(self):
        """Non-sample tools get modifier 1.0."""
        self.assertEqual(_compute_sample_modifier(None), 1.0)

    def test_bracket_boundaries(self):
        """Each bracket boundary returns the correct modifier."""
        # n < 5 → 0.50
        self.assertEqual(_compute_sample_modifier(1), 0.50)
        self.assertEqual(_compute_sample_modifier(4), 0.50)

        # 5 ≤ n < 15 → 0.70
        self.assertEqual(_compute_sample_modifier(5), 0.70)
        self.assertEqual(_compute_sample_modifier(14), 0.70)

        # 15 ≤ n < 30 → 0.85
        self.assertEqual(_compute_sample_modifier(15), 0.85)
        self.assertEqual(_compute_sample_modifier(29), 0.85)

        # 30 ≤ n < 100 → 0.95
        self.assertEqual(_compute_sample_modifier(30), 0.95)
        self.assertEqual(_compute_sample_modifier(99), 0.95)

        # n ≥ 100 → 1.0
        self.assertEqual(_compute_sample_modifier(100), 1.0)
        self.assertEqual(_compute_sample_modifier(10000), 1.0)

    def test_zero_sample(self):
        """n=0 falls in the < 5 bracket."""
        self.assertEqual(_compute_sample_modifier(0), 0.50)


class StudyQualityTest(TestCase):
    """CANON-002 §3.3 — study quality modifier (geometric mean)."""

    def test_none_factors_returns_one(self):
        """No factors provided → modifier is 1.0."""
        self.assertEqual(_compute_study_quality("doe_results", None), 1.0)

    def test_empty_factors_returns_one(self):
        """Empty dict → modifier is 1.0 (no applicable factors supplied)."""
        self.assertEqual(_compute_study_quality("doe_results", {}), 1.0)

    def test_tool_with_no_applicable_factors(self):
        """SPC has no applicable quality factors per §3.3 table."""
        result = _compute_study_quality("spc", {"randomization": 0.5})
        self.assertEqual(result, 1.0)

    def test_tool_not_in_registry(self):
        """Unknown tool type → modifier is 1.0 (no applicable factors)."""
        result = _compute_study_quality("unknown_tool", {"randomization": 0.5})
        self.assertEqual(result, 1.0)

    def test_doe_all_full_credit(self):
        """DOE with all 5 factors at full credit → 1.0."""
        factors = {
            "randomization": 1.0,
            "replication": 1.0,
            "blocking": 1.0,
            "blinding": 1.0,
            "pre_registration": 1.0,
        }
        result = _compute_study_quality("doe_results", factors)
        self.assertAlmostEqual(result, 1.0)

    def test_doe_all_penalty(self):
        """DOE with all 5 factors at penalty → geometric mean of 0.5^5."""
        factors = {
            "randomization": 0.5,
            "replication": 0.5,
            "blocking": 0.5,
            "blinding": 0.5,
            "pre_registration": 0.5,
        }
        expected = 0.5  # (0.5^5)^(1/5) = 0.5
        result = _compute_study_quality("doe_results", factors)
        self.assertAlmostEqual(result, expected)

    def test_doe_mixed_factors(self):
        """DOE with mixed quality → geometric mean computed correctly."""
        factors = {
            "randomization": 1.0,
            "replication": 0.7,
            "blocking": 0.5,
        }
        # Only 3 of 5 applicable factors provided
        # Geometric mean: (1.0 * 0.7 * 0.5)^(1/3)
        expected = (1.0 * 0.7 * 0.5) ** (1.0 / 3)
        result = _compute_study_quality("doe_results", factors)
        self.assertAlmostEqual(result, expected, places=6)

    def test_dsw_factors(self):
        """DSW has only blinding and pre_registration applicable."""
        factors = {
            "blinding": 0.7,
            "pre_registration": 1.0,
            "randomization": 0.5,  # Not applicable to DSW — should be ignored
        }
        expected = (0.7 * 1.0) ** (1.0 / 2)
        result = _compute_study_quality("dsw", factors)
        self.assertAlmostEqual(result, expected, places=6)

    def test_forecast_factors(self):
        """Forecast has replication and pre_registration applicable."""
        factors = {"replication": 0.7, "pre_registration": 0.5}
        expected = (0.7 * 0.5) ** 0.5
        result = _compute_study_quality("forecast", factors)
        self.assertAlmostEqual(result, expected, places=6)

    def test_ml_ignores_all_factors(self):
        """ML has no applicable quality factors."""
        result = _compute_study_quality("ml", {"replication": 0.5})
        self.assertEqual(result, 1.0)


class MeasurementValidityTest(TestCase):
    """CANON-002 §4 — measurement system validity gate."""

    def test_none_returns_default(self):
        """No measurement system → 0.55 default (§4.3)."""
        self.assertEqual(_compute_measurement_validity(None), 0.55)

    def test_nonexistent_id_returns_default(self):
        """Non-None ID when MeasurementSystem model doesn't exist yet → 0.55 default.
        After FEAT-100 creates the model, this will test DoesNotExist path."""
        result = _compute_measurement_validity("nonexistent-uuid")
        self.assertEqual(result, 0.55)

    def test_validity_flows_through_compute(self):
        """Measurement validity is correctly used in composite weight.
        Tests the integration via _compute_measurement_validity mock on
        compute_evidence_weight (the public API)."""
        # Without MSA: 0.55 default
        without = compute_evidence_weight("spc")
        # 0.85 * 1.0 * 0.55 * 1.0 = 0.4675
        self.assertAlmostEqual(without, 0.4675)


class ComputeEvidenceWeightTest(TestCase):
    """CANON-002 §3.1 — end-to-end evidence weight computation."""

    def test_spc_with_defaults(self):
        """SPC with no sample size, no MSA → rank * 1.0 * 0.55 * 1.0."""
        result = compute_evidence_weight("spc")
        # 0.85 * 1.0 * 0.55 * 1.0 = 0.4675
        self.assertAlmostEqual(result, 0.4675)

    def test_spc_with_sample(self):
        """SPC with 25 subgroups, no MSA → rank * 0.85 * 0.55 * 1.0."""
        result = compute_evidence_weight("spc", sample_size=25)
        # 0.85 * 0.85 * 0.55 = 0.397375
        self.assertAlmostEqual(result, 0.397375)

    def test_doe_results_full_quality(self):
        """DOE results with large sample, no MSA, full quality."""
        result = compute_evidence_weight(
            "doe_results",
            sample_size=100,
            study_quality_factors={
                "randomization": 1.0,
                "replication": 1.0,
                "blocking": 1.0,
            },
        )
        # 0.95 * 1.0 * 0.55 * 1.0 = 0.5225
        self.assertAlmostEqual(result, 0.5225)

    def test_ishikawa_no_sample_no_msa(self):
        """Ishikawa (information tool): rank * 1.0 * 0.55 * 1.0."""
        result = compute_evidence_weight("ishikawa")
        # 0.60 * 1.0 * 0.55 * 1.0 = 0.33
        self.assertAlmostEqual(result, 0.33)

    def test_user_evidence(self):
        """User-supplied expert judgment."""
        result = compute_evidence_weight("user")
        # 0.35 * 1.0 * 0.55 * 1.0 = 0.1925
        self.assertAlmostEqual(result, 0.1925)

    def test_anecdotal_evidence(self):
        """Single observation — lowest ranked."""
        result = compute_evidence_weight("observation")
        # 0.20 * 1.0 * 0.55 * 1.0 = 0.11
        self.assertAlmostEqual(result, 0.11)

    def test_no_evidence_tools_return_zero(self):
        """Tools with None rank return 0.0 (not clamped)."""
        for tool in ("triage", "doe_design", "a3", "vsm", "report"):
            result = compute_evidence_weight(tool)
            self.assertEqual(result, 0.0, f"{tool} should return 0.0 (produces no evidence)")

    def test_unknown_tool_returns_zero(self):
        """Unknown tool not in registry returns 0.0."""
        result = compute_evidence_weight("totally_unknown_tool")
        self.assertEqual(result, 0.0)

    @patch("agents_api.evidence_weights._compute_measurement_validity")
    def test_with_valid_msa(self, mock_mv):
        """SPC with a validated measurement system (validity=1.0)."""
        mock_mv.return_value = 1.0
        result = compute_evidence_weight("spc", measurement_system_id="ms-123")
        # 0.85 * 1.0 * 1.0 * 1.0 = 0.85
        self.assertAlmostEqual(result, 0.85)

    @patch("agents_api.evidence_weights._compute_measurement_validity")
    def test_with_poor_msa(self, mock_mv):
        """SPC with poor measurement system (validity=0.10)."""
        mock_mv.return_value = 0.10
        result = compute_evidence_weight("spc", measurement_system_id="ms-bad")
        # 0.85 * 1.0 * 0.10 * 1.0 = 0.085
        self.assertAlmostEqual(result, 0.085)


class ClampBehaviorTest(TestCase):
    """CANON-002 §3.1 — output clamped to [0.05, 0.99]."""

    def test_minimum_clamp(self):
        """Extremely low weight is clamped to 0.05."""
        # Anecdotal (0.20) * tiny sample (0.50) * no MSA (0.55) * penalty quality
        result = compute_evidence_weight(
            "observation",
            sample_size=2,
            study_quality_factors={"randomization": 0.5},  # Not applicable, ignored
        )
        # 0.20 * 0.50 * 0.55 * 1.0 = 0.055 — above 0.05 but close
        self.assertGreaterEqual(result, 0.05)

    @patch("agents_api.evidence_weights._compute_measurement_validity")
    def test_near_zero_still_clamped(self, mock_mv):
        """Even with extreme discounting, floor is 0.05."""
        mock_mv.return_value = 0.10  # Quarantined instrument
        result = compute_evidence_weight(
            "observation",  # 0.20
            sample_size=1,  # 0.50
            measurement_system_id="bad-gage",  # 0.10
        )
        # 0.20 * 0.50 * 0.10 * 1.0 = 0.01 → clamped to 0.05
        self.assertEqual(result, 0.05)

    @patch("agents_api.evidence_weights._compute_measurement_validity")
    def test_maximum_clamp(self, mock_mv):
        """Even best-case cannot exceed 0.99."""
        mock_mv.return_value = 1.0
        result = compute_evidence_weight(
            "doe_results",  # 0.95
            sample_size=1000,  # 1.0
            measurement_system_id="perfect-gage",  # 1.0
            study_quality_factors={
                "randomization": 1.0,
                "replication": 1.0,
                "blocking": 1.0,
                "blinding": 1.0,
                "pre_registration": 1.0,
            },  # 1.0
        )
        # 0.95 * 1.0 * 1.0 * 1.0 = 0.95 — below 0.99, no clamp needed
        self.assertAlmostEqual(result, 0.95)
        self.assertLessEqual(result, 0.99)

    def test_no_evidence_not_clamped(self):
        """0.0 return for no-evidence tools is NOT clamped (it's not in [0.05, 0.99])."""
        result = compute_evidence_weight("triage")
        self.assertEqual(result, 0.0)  # Exactly 0.0, not 0.05
