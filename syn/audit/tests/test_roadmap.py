"""
RDM-001 compliance tests: Product Roadmap Standard.

Tests verify roadmap model structure, compliance check logic,
and quarter format validation.

Compliance: RDM-001 (Product Roadmap), SOC 2 CC9.1
"""

from django.test import SimpleTestCase, TestCase
from django.utils import timezone


class QuarterFormatTest(SimpleTestCase):
    """RDM-001 §2.3: Quarter format validation."""

    def test_valid_quarters(self):
        pattern = r"^Q[1-4]-\d{4}$"
        for q in ["Q1-2026", "Q2-2026", "Q3-2026", "Q4-2026"]:
            self.assertRegex(q, pattern)

    def test_invalid_quarters(self):
        pattern = r"^Q[1-4]-\d{4}$"
        for q in ["Q5-2026", "Q0-2026", "2026-Q1", "Q1-26", ""]:
            self.assertNotRegex(q, pattern)


class RoadmapModelTest(TestCase):
    """RDM-001 §2: RoadmapItem model structure."""

    def test_model_exists(self):
        from api.models import RoadmapItem

        self.assertTrue(hasattr(RoadmapItem, "_meta"))

    def test_area_choices_count(self):
        from api.models import RoadmapItem

        self.assertEqual(len(RoadmapItem.Area.choices), 8)

    def test_status_choices(self):
        from api.models import RoadmapItem

        values = [c[0] for c in RoadmapItem.Status.choices]
        for expected in ["planned", "in_progress", "shipped", "deferred", "cancelled"]:
            self.assertIn(expected, values)

    def test_db_table(self):
        from api.models import RoadmapItem

        self.assertEqual(RoadmapItem._meta.db_table, "roadmap_items")


class ShippedTrackingTest(TestCase):
    """RDM-001 §3.1: Shipped items must have shipped_at."""

    def test_shipped_requires_shipped_at(self):
        from api.models import RoadmapItem
        from syn.audit.compliance import check_roadmap

        item = RoadmapItem.objects.create(
            title="Test shipped",
            area="dsw",
            quarter="Q2-2026",
            status="shipped",
        )
        try:
            result = check_roadmap()
            self.assertTrue(
                any("shipped_at" in i for i in result["details"]["issues"]),
                "Shipped item without shipped_at should be flagged",
            )
        finally:
            item.delete()


class ComplianceCheckTest(TestCase):
    """RDM-001 §3.2, §4, §6: Compliance check logic."""

    def test_stale_items_flagged(self):
        from api.models import RoadmapItem
        from syn.audit.compliance import check_roadmap

        item = RoadmapItem.objects.create(
            title="Stale item",
            area="dsw",
            quarter="Q1-2020",
            status="planned",
        )
        try:
            result = check_roadmap()
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any("past quarters" in i for i in result["details"]["issues"]))
        finally:
            item.delete()

    def test_empty_upcoming_warning(self):
        from api.models import RoadmapItem
        from syn.audit.compliance import check_roadmap

        # Delete any items for the upcoming quarter
        now = timezone.now()
        q_num = (now.month - 1) // 3 + 1
        if q_num == 4:
            next_q = f"Q1-{now.year + 1}"
        else:
            next_q = f"Q{q_num + 1}-{now.year}"
        RoadmapItem.objects.filter(quarter=next_q).delete()
        result = check_roadmap()
        self.assertTrue(any("upcoming quarter" in i for i in result["details"]["issues"]))

    def test_clean_roadmap_passes(self):
        from api.models import RoadmapItem
        from syn.audit.compliance import check_roadmap

        now = timezone.now()
        q_num = (now.month - 1) // 3 + 1
        current_q = f"Q{q_num}-{now.year}"
        if q_num == 4:
            next_q = f"Q1-{now.year + 1}"
        else:
            next_q = f"Q{q_num + 1}-{now.year}"
        # Clear stale items and create clean state
        RoadmapItem.objects.filter(status="planned", quarter__lt=current_q).delete()
        RoadmapItem.objects.filter(status="shipped", shipped_at__isnull=True).delete()
        items = [
            RoadmapItem.objects.create(
                title="Current",
                area="dsw",
                quarter=current_q,
                status="in_progress",
            ),
            RoadmapItem.objects.create(
                title="Next",
                area="platform",
                quarter=next_q,
                status="planned",
            ),
        ]
        try:
            result = check_roadmap()
            self.assertEqual(result["status"], "pass")
        finally:
            for item in items:
                item.delete()

    def test_check_registered(self):
        from syn.audit.compliance import ALL_CHECKS

        self.assertIn("roadmap", ALL_CHECKS)

    def test_check_returns_valid_structure(self):
        from syn.audit.compliance import ALL_CHECKS

        entry = ALL_CHECKS["roadmap"]
        fn = entry[0] if isinstance(entry, tuple) else entry
        result = fn()
        for key in ["status", "details", "soc2_controls"]:
            self.assertIn(key, result)
        self.assertIn(result["status"], ("pass", "fail", "warning"))
