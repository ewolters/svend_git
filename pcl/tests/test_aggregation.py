"""Bayesian weighted aggregation tests."""

import pytest

from pcl.aggregation import compute_aggregate, update_aggregate_incremental


class TestComputeAggregate:
    def test_single_datapoint(self):
        """First datapoint IS the aggregate."""
        result = compute_aggregate(
            values=[45.0],
            confidences=[0.50],
            timestamps_age_days=[0.0],
        )
        assert result["value"] == 45.0
        assert result["n"] == 1

    def test_two_equal_confidence(self):
        """Equal confidence = simple mean."""
        result = compute_aggregate(
            values=[40.0, 50.0],
            confidences=[0.50, 0.50],
            timestamps_age_days=[0.0, 0.0],
        )
        assert result["value"] == pytest.approx(45.0, abs=0.01)

    def test_high_confidence_dominates(self):
        """DOE at 54 should dominate over manual at 45."""
        result = compute_aggregate(
            values=[45.0, 54.0],
            confidences=[0.30, 0.90],
            timestamps_age_days=[0.0, 0.0],
        )
        assert result["value"] > 50.0

    def test_fat_finger_suppressed(self):
        """Outlier with low confidence barely moves aggregate."""
        result = compute_aggregate(
            values=[54.0, 53.8, 54.2, 45.0],
            confidences=[0.90, 0.85, 0.85, 0.30],
            timestamps_age_days=[3.0, 2.0, 1.0, 0.0],
        )
        assert result["value"] > 52.0

    def test_recency_decay(self):
        """Old data decays when decay_enabled."""
        result = compute_aggregate(
            values=[50.0, 55.0],
            confidences=[0.90, 0.90],
            timestamps_age_days=[365.0, 0.0],
            decay_enabled=True,
            decay_halflife_days=90.0,
        )
        assert result["value"] > 53.0

    def test_no_decay_by_default(self):
        """Without decay, old data has full weight."""
        result = compute_aggregate(
            values=[50.0, 55.0],
            confidences=[0.90, 0.90],
            timestamps_age_days=[365.0, 0.0],
            decay_enabled=False,
        )
        assert result["value"] == pytest.approx(52.5, abs=0.5)

    def test_shift_detection(self):
        """Sustained high-confidence deviation = shift detected."""
        # 8 old points at ~50, then 4 new points at ~70 — clear shift
        values = [50.0, 50.5, 49.8, 50.2, 50.1, 49.7, 50.3, 49.9, 70.0, 70.3, 69.8, 70.1]
        confidences = [0.85] * 12
        ages = [12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 3.0, 2.0, 1.0, 0.0]
        result = compute_aggregate(
            values=values,
            confidences=confidences,
            timestamps_age_days=ages,
            decay_enabled=False,
        )
        assert result["shift_detected"] is True

    def test_empty_returns_none(self):
        result = compute_aggregate(values=[], confidences=[], timestamps_age_days=[])
        assert result["value"] is None
        assert result["n"] == 0

    def test_all_zero_confidence(self):
        """Edge case: all zero confidence should not crash."""
        result = compute_aggregate(
            values=[10.0, 20.0],
            confidences=[0.0, 0.0],
            timestamps_age_days=[0.0, 0.0],
        )
        assert result["n"] == 2


class TestUpdateAggregateIncremental:
    def test_first_datapoint(self):
        result = update_aggregate_incremental(
            cached_value=None,
            cached_variance=None,
            cached_confidence=None,
            cached_n=0,
            cached_effective_n=0.0,
            new_value=45.0,
            new_confidence=0.50,
        )
        assert result["value"] == 45.0
        assert result["n"] == 1

    def test_second_datapoint_consistent(self):
        """Second consistent value moves aggregate slightly."""
        result = update_aggregate_incremental(
            cached_value=45.0,
            cached_variance=0.0,
            cached_confidence=0.50,
            cached_n=1,
            cached_effective_n=0.50,
            new_value=46.0,
            new_confidence=0.50,
        )
        assert 45.0 < result["value"] < 46.0
        assert result["n"] == 2

    def test_outlier_suppressed_incrementally(self):
        """Outlier against established aggregate gets low weight."""
        # Simulate stable process at ~50 with std=1
        result = update_aggregate_incremental(
            cached_value=50.0,
            cached_variance=1.0,
            cached_confidence=0.85,
            cached_n=10,
            cached_effective_n=8.0,
            new_value=100.0,
            new_confidence=0.30,
        )
        # Should barely move from 50
        assert result["value"] < 55.0

    def test_incremental_matches_direction(self):
        """Higher confidence new value should pull aggregate toward it."""
        result = update_aggregate_incremental(
            cached_value=50.0,
            cached_variance=4.0,
            cached_confidence=0.50,
            cached_n=3,
            cached_effective_n=1.5,
            new_value=55.0,
            new_confidence=0.90,
        )
        assert result["value"] > 50.0
