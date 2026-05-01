"""Confidence computation tests."""

from pcl.confidence import CONFIDENCE_WEIGHTS, compute_confidence


class TestComputeConfidence:
    def test_manual_single_observation(self):
        c = compute_confidence("manual", 1)
        # base=0.50, log2(1)/log2(30) = 0/~4.9 = 0
        assert c == 0.0

    def test_manual_few_observations(self):
        c = compute_confidence("manual", 3)
        assert 0.0 < c < 0.5

    def test_manual_at_plateau(self):
        c = compute_confidence("manual", 30)
        assert abs(c - 0.50) < 0.01

    def test_doe_at_plateau(self):
        c = compute_confidence("doe", 8)
        assert abs(c - 0.90) < 0.01

    def test_automated_high(self):
        c = compute_confidence("automated", 10)
        assert abs(c - 0.95) < 0.01

    def test_estimate_always_low(self):
        c = compute_confidence("estimate", 1)
        assert c == 0.0
        c30 = compute_confidence("estimate", 30)
        assert abs(c30 - 0.25) < 0.01

    def test_doe_beats_manual_at_same_n(self):
        c_doe = compute_confidence("doe", 8)
        c_manual = compute_confidence("manual", 8)
        assert c_doe > c_manual

    def test_caps_at_one(self):
        c = compute_confidence("automated", 10000)
        assert c <= 1.0

    def test_unknown_source_uses_default(self):
        c = compute_confidence("unknown_source", 10)
        assert 0.0 < c <= 1.0

    def test_zero_observation_treated_as_one(self):
        c = compute_confidence("manual", 0)
        assert c == compute_confidence("manual", 1)

    def test_all_sources_defined(self):
        """Every defined source has base and plateau."""
        for source, params in CONFIDENCE_WEIGHTS.items():
            assert "base" in params
            assert "plateau" in params
            c = compute_confidence(source, params["plateau"])
            assert abs(c - params["base"]) < 0.01, f"{source} at plateau should equal base"
