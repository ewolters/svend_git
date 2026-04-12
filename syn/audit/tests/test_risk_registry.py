"""
FEAT-090: Persistent risk registry tests.

Verifies RiskEntry model, RPN auto-computation, risk_level property,
compliance check logic, and check registration.

Standard: RISK-001
"""

from django.test import SimpleTestCase, TestCase

from syn.audit.compliance import ALL_CHECKS, WEEKDAY_ROTATION

# ── Model tests ────────────────────────────────────────────────────────


class RiskEntryModelTest(TestCase):
    """RISK-001 §3: RiskEntry model and scoring."""

    def _create_entry(self, **kwargs):
        from syn.audit.models import RiskEntry

        defaults = {
            "title": "Test risk",
            "description": "Test description",
            "category": "security",
            "likelihood": 3,
            "severity": 3,
            "detectability": 3,
            "owner": "test_owner",
        }
        defaults.update(kwargs)
        return RiskEntry.objects.create(**defaults)

    def test_rpn_auto_computed(self):
        """RPN is auto-computed as L×S×D on save."""
        entry = self._create_entry(likelihood=4, severity=5, detectability=3)
        self.assertEqual(entry.rpn, 60)

    def test_risk_level_high(self):
        """RPN > 60 → high risk."""
        entry = self._create_entry(likelihood=5, severity=5, detectability=3)
        self.assertEqual(entry.rpn, 75)
        self.assertEqual(entry.risk_level, "high")

    def test_risk_level_medium(self):
        """20 < RPN ≤ 60 → medium risk."""
        entry = self._create_entry(likelihood=3, severity=3, detectability=3)
        self.assertEqual(entry.rpn, 27)
        self.assertEqual(entry.risk_level, "medium")

    def test_risk_level_low(self):
        """RPN ≤ 20 → low risk."""
        entry = self._create_entry(likelihood=2, severity=2, detectability=2)
        self.assertEqual(entry.rpn, 8)
        self.assertEqual(entry.risk_level, "low")

    def test_source_cr_linkage(self):
        """RiskEntry can link to a ChangeRequest."""
        from syn.audit.models import ChangeRequest

        cr = ChangeRequest.objects.create(
            title="Test change request for risk",
            description="Testing risk entry CR linkage",
            change_type="feature",
            author="test",
        )
        entry = self._create_entry(source_cr=cr)
        self.assertEqual(entry.source_cr_id, cr.id)
        self.assertIn(entry, cr.risk_entries.all())

    def test_status_choices_valid(self):
        """All status choices are accepted."""
        from syn.audit.models import RiskEntry

        for status, _ in RiskEntry.STATUS_CHOICES:
            entry = self._create_entry(status=status, title=f"Risk {status}")
            self.assertEqual(entry.status, status)

    def test_category_choices_valid(self):
        """All category choices are accepted."""
        from syn.audit.models import RiskEntry

        for category, _ in RiskEntry.CATEGORY_CHOICES:
            entry = self._create_entry(category=category, title=f"Risk {category}")
            self.assertEqual(entry.category, category)


# ── Compliance check tests ─────────────────────────────────────────────


class RiskRegistryCheckTest(TestCase):
    """RISK-001 §5: risk_registry compliance check logic."""

    def _create_entry(self, **kwargs):
        from syn.audit.models import RiskEntry

        defaults = {
            "title": "Test risk",
            "description": "Test",
            "category": "security",
            "likelihood": 3,
            "severity": 3,
            "detectability": 3,
            "owner": "test",
        }
        defaults.update(kwargs)
        return RiskEntry.objects.create(**defaults)

    def test_empty_registry_passes(self):
        """Empty risk registry → pass."""
        from syn.audit.compliance import check_risk_registry

        result = check_risk_registry()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["details"]["total_risks"], 0)

    def test_low_risk_open_passes(self):
        """Open low-risk items → pass."""
        self._create_entry(likelihood=2, severity=2, detectability=2, status="identified")
        from syn.audit.compliance import check_risk_registry

        result = check_risk_registry()
        self.assertEqual(result["status"], "pass")

    def test_high_risk_with_mitigation_passes(self):
        """High-risk item with mitigation plan in mitigating status → pass."""
        self._create_entry(
            likelihood=5,
            severity=5,
            detectability=3,
            status="mitigating",
            mitigation_plan="Deploy WAF and rotate credentials",
        )
        from syn.audit.compliance import check_risk_registry

        result = check_risk_registry()
        self.assertEqual(result["status"], "pass")

    def test_high_risk_no_mitigation_fails(self):
        """High-risk item without mitigation plan → fail."""
        self._create_entry(
            likelihood=5,
            severity=5,
            detectability=3,
            status="identified",
            mitigation_plan="",
        )
        from syn.audit.compliance import check_risk_registry

        result = check_risk_registry()
        self.assertEqual(result["status"], "fail")
        self.assertEqual(len(result["details"]["high_without_mitigation"]), 1)

    def test_high_risk_identified_with_plan_warns(self):
        """High-risk item with plan but still in 'identified' status → warning."""
        self._create_entry(
            likelihood=5,
            severity=5,
            detectability=3,
            status="identified",
            mitigation_plan="Plan exists but status not updated",
        )
        from syn.audit.compliance import check_risk_registry

        result = check_risk_registry()
        self.assertEqual(result["status"], "warning")
        self.assertEqual(len(result["details"]["high_not_mitigating"]), 1)


# ── Registration tests ─────────────────────────────────────────────────


class RiskRegistryRegistrationTest(SimpleTestCase):
    """RISK-001 §5: Check is properly registered."""

    def test_check_registered(self):
        """risk_registry check is in ALL_CHECKS."""
        self.assertIn("risk_registry", ALL_CHECKS)

    def test_check_has_soc2_controls(self):
        """risk_registry check declares SOC 2 controls."""
        fn, _cat = ALL_CHECKS["risk_registry"]
        controls = getattr(fn, "soc2_controls", [])
        self.assertIn("CC3.2", controls)
        self.assertIn("CC9.1", controls)

    def test_check_in_rotation(self):
        """risk_registry is in Thursday rotation."""
        thursday = WEEKDAY_ROTATION[3]
        self.assertIn("risk_registry", thursday)
