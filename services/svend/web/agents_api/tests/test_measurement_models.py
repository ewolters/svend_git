"""
Tests for core.models.measurement — CANON-002 §4, §12.2.

All tests exercise real behavior per TST-001 §10.6.
Tests use DB fixtures (model creation) to verify validity computation,
auto-quarantine, and default behavior.

<!-- test: agents_api.tests.test_measurement_models.MeasurementSystemTest -->
<!-- test: agents_api.tests.test_measurement_models.GageStudyVariableTest -->
<!-- test: agents_api.tests.test_measurement_models.GageStudyAttributeTest -->
<!-- test: agents_api.tests.test_measurement_models.AutoQuarantineTest -->
<!-- test: agents_api.tests.test_measurement_models.EvidenceWeightIntegrationTest -->
"""

from django.test import TestCase
from django.utils import timezone

from agents_api.evidence_weights import _compute_measurement_validity
from core.models import GageStudy, MeasurementSystem


def _make_user(email="test@example.com"):
    from django.contrib.auth import get_user_model

    User = get_user_model()
    return User.objects.create_user(username=email, email=email, password="testpass123")


def _make_measurement_system(user, name="Test Gage", system_type="variable"):
    return MeasurementSystem.objects.create(
        name=name,
        system_type=system_type,
        owner=user,
    )


def _make_gage_study(
    ms, study_type="grr_crossed", grr_percent=None, kappa=None, completed=True
):
    return GageStudy.objects.create(
        measurement_system=ms,
        study_type=study_type,
        grr_percent=grr_percent,
        kappa=kappa,
        completed_at=timezone.now() if completed else None,
    )


class MeasurementSystemTest(TestCase):
    """CANON-002 §12.2 — MeasurementSystem model behavior."""

    def setUp(self):
        self.user = _make_user()

    def test_default_validity_no_studies(self):
        """No GageStudy exists → current_validity returns 0.55 (§4.3)."""
        ms = _make_measurement_system(self.user)
        self.assertEqual(ms.current_validity, 0.55)

    def test_default_status_is_active(self):
        """New measurement systems default to active status."""
        ms = _make_measurement_system(self.user)
        self.assertEqual(ms.status, MeasurementSystem.Status.ACTIVE)

    def test_validity_from_completed_study(self):
        """current_validity uses the most recent completed study."""
        ms = _make_measurement_system(self.user)
        _make_gage_study(ms, grr_percent=8.0)  # Valid → 1.0
        self.assertEqual(ms.current_validity, 1.0)

    def test_validity_ignores_incomplete_study(self):
        """Studies without completed_at are not considered."""
        ms = _make_measurement_system(self.user)
        _make_gage_study(ms, grr_percent=8.0, completed=False)
        self.assertEqual(ms.current_validity, 0.55)

    def test_validity_uses_most_recent(self):
        """When multiple studies exist, the most recent completed one wins."""
        ms = _make_measurement_system(self.user)
        _make_gage_study(ms, grr_percent=8.0)  # First: valid → 1.0
        import time

        time.sleep(0.01)  # Ensure different timestamps
        _make_gage_study(ms, grr_percent=25.0)  # Second: poor → 0.50
        self.assertEqual(ms.current_validity, 0.50)

    def test_str_representation(self):
        """String representation includes name, type, and status."""
        ms = _make_measurement_system(self.user, name="Keyence IM-8000")
        self.assertIn("Keyence IM-8000", str(ms))
        self.assertIn("Variable", str(ms))


class GageStudyVariableTest(TestCase):
    """CANON-002 §4.1 — variable measurement system validity (%GRR thresholds)."""

    def setUp(self):
        self.user = _make_user("variable@test.com")
        self.ms = _make_measurement_system(self.user)

    def test_grr_valid(self):
        """≤ 10% → 1.0 (valid)."""
        study = _make_gage_study(self.ms, grr_percent=10.0)
        self.assertEqual(study.measurement_validity, 1.0)

    def test_grr_at_boundary_valid(self):
        """Exactly 10% is still valid."""
        study = _make_gage_study(self.ms, grr_percent=10.0)
        self.assertEqual(study.measurement_validity, 1.0)

    def test_grr_marginal(self):
        """10-20% → 0.80 (marginal)."""
        study = _make_gage_study(self.ms, grr_percent=15.0)
        self.assertEqual(study.measurement_validity, 0.80)

    def test_grr_at_boundary_marginal(self):
        """Exactly 20% is still marginal."""
        study = _make_gage_study(self.ms, grr_percent=20.0)
        self.assertEqual(study.measurement_validity, 0.80)

    def test_grr_poor(self):
        """20-30% → 0.50 (poor)."""
        study = _make_gage_study(self.ms, grr_percent=25.0)
        self.assertEqual(study.measurement_validity, 0.50)

    def test_grr_at_boundary_poor(self):
        """Exactly 30% is still poor."""
        study = _make_gage_study(self.ms, grr_percent=30.0)
        self.assertEqual(study.measurement_validity, 0.50)

    def test_grr_invalid(self):
        """> 30% → 0.10 (invalid)."""
        study = _make_gage_study(self.ms, grr_percent=35.0)
        self.assertEqual(study.measurement_validity, 0.10)

    def test_grr_none(self):
        """No %GRR result → 0.55 default."""
        study = _make_gage_study(self.ms, grr_percent=None)
        self.assertEqual(study.measurement_validity, 0.55)

    def test_grr_extreme(self):
        """%GRR = 100% → still 0.10 (not lower)."""
        study = _make_gage_study(self.ms, grr_percent=100.0)
        self.assertEqual(study.measurement_validity, 0.10)

    def test_grr_excellent(self):
        """%GRR = 5% → 1.0."""
        study = _make_gage_study(self.ms, grr_percent=5.0)
        self.assertEqual(study.measurement_validity, 1.0)


class GageStudyAttributeTest(TestCase):
    """CANON-002 §4.4 — attribute measurement system validity (Kappa thresholds)."""

    def setUp(self):
        self.user = _make_user("attribute@test.com")
        self.ms = _make_measurement_system(self.user, system_type="attribute")

    def test_kappa_valid(self):
        """≥ 0.90 → 1.0 (valid)."""
        study = _make_gage_study(self.ms, study_type="attribute_agreement", kappa=0.95)
        self.assertEqual(study.measurement_validity, 1.0)

    def test_kappa_at_boundary_valid(self):
        """Exactly 0.90 is valid."""
        study = _make_gage_study(self.ms, study_type="attribute_agreement", kappa=0.90)
        self.assertEqual(study.measurement_validity, 1.0)

    def test_kappa_marginal(self):
        """0.75-0.90 → 0.80 (marginal)."""
        study = _make_gage_study(self.ms, study_type="attribute_agreement", kappa=0.80)
        self.assertEqual(study.measurement_validity, 0.80)

    def test_kappa_poor(self):
        """0.50-0.75 → 0.50 (poor)."""
        study = _make_gage_study(self.ms, study_type="attribute_agreement", kappa=0.60)
        self.assertEqual(study.measurement_validity, 0.50)

    def test_kappa_invalid(self):
        """< 0.50 → 0.10 (invalid)."""
        study = _make_gage_study(self.ms, study_type="attribute_agreement", kappa=0.40)
        self.assertEqual(study.measurement_validity, 0.10)

    def test_kappa_none(self):
        """No Kappa result → 0.55 default."""
        study = _make_gage_study(self.ms, study_type="attribute_agreement", kappa=None)
        self.assertEqual(study.measurement_validity, 0.55)


class AutoQuarantineTest(TestCase):
    """CANON-002 §12.2 — auto-quarantine on failed GRR."""

    def setUp(self):
        self.user = _make_user("quarantine@test.com")

    def test_grr_over_30_quarantines(self):
        """Completed study with %GRR > 30% auto-sets status to quarantined."""
        ms = _make_measurement_system(self.user)
        self.assertEqual(ms.status, MeasurementSystem.Status.ACTIVE)

        _make_gage_study(ms, grr_percent=35.0)
        ms.refresh_from_db()
        self.assertEqual(ms.status, MeasurementSystem.Status.QUARANTINED)

    def test_grr_under_30_no_quarantine(self):
        """Completed study with %GRR ≤ 30% does not quarantine."""
        ms = _make_measurement_system(self.user)
        _make_gage_study(ms, grr_percent=25.0)
        ms.refresh_from_db()
        self.assertEqual(ms.status, MeasurementSystem.Status.ACTIVE)

    def test_incomplete_study_no_quarantine(self):
        """Incomplete study does not trigger quarantine even with bad %GRR."""
        ms = _make_measurement_system(self.user)
        _make_gage_study(ms, grr_percent=50.0, completed=False)
        ms.refresh_from_db()
        self.assertEqual(ms.status, MeasurementSystem.Status.ACTIVE)

    def test_kappa_under_50_quarantines(self):
        """Attribute study with Kappa < 0.50 auto-quarantines."""
        ms = _make_measurement_system(self.user, system_type="attribute")
        _make_gage_study(ms, study_type="attribute_agreement", kappa=0.40)
        ms.refresh_from_db()
        self.assertEqual(ms.status, MeasurementSystem.Status.QUARANTINED)

    def test_kappa_above_50_no_quarantine(self):
        """Attribute study with Kappa ≥ 0.50 does not quarantine."""
        ms = _make_measurement_system(self.user, system_type="attribute")
        _make_gage_study(ms, study_type="attribute_agreement", kappa=0.75)
        ms.refresh_from_db()
        self.assertEqual(ms.status, MeasurementSystem.Status.ACTIVE)

    def test_already_quarantined_stays_quarantined(self):
        """If already quarantined, a second bad study doesn't error."""
        ms = _make_measurement_system(self.user)
        _make_gage_study(ms, grr_percent=35.0)
        ms.refresh_from_db()
        self.assertEqual(ms.status, MeasurementSystem.Status.QUARANTINED)

        # Second bad study — should not error
        _make_gage_study(ms, grr_percent=40.0)
        ms.refresh_from_db()
        self.assertEqual(ms.status, MeasurementSystem.Status.QUARANTINED)


class EvidenceWeightIntegrationTest(TestCase):
    """CANON-002 §3.1 + §4 — _compute_measurement_validity with real DB."""

    def setUp(self):
        self.user = _make_user("integration@test.com")

    def test_real_system_valid(self):
        """_compute_measurement_validity resolves a real MeasurementSystem."""
        ms = _make_measurement_system(self.user)
        _make_gage_study(ms, grr_percent=8.0)

        result = _compute_measurement_validity(str(ms.id))
        self.assertEqual(result, 1.0)

    def test_real_system_poor(self):
        """Real system with poor GRR returns 0.50."""
        ms = _make_measurement_system(self.user)
        _make_gage_study(ms, grr_percent=25.0)

        result = _compute_measurement_validity(str(ms.id))
        self.assertEqual(result, 0.50)

    def test_nonexistent_id(self):
        """Non-existent UUID returns 0.55 default."""
        import uuid

        result = _compute_measurement_validity(str(uuid.uuid4()))
        self.assertEqual(result, 0.55)
