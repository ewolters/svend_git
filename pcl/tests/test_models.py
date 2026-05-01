"""PCL model tests — Measure, Datapoint, MeasureTarget."""

from datetime import date

from django.db import IntegrityError
from django.test import TestCase

from conftest import SECURE_OFF, make_membership, make_tenant, make_user
from pcl.models import Datapoint, Measure, MeasureTarget


@SECURE_OFF
class MeasureModelTest(TestCase):
    def setUp(self):
        self.user = make_user("pcl@test.com", tier="team")
        self.tenant = make_tenant("PCL Org", slug="pcl-org", plan="team")
        make_membership(self.tenant, self.user)

    def test_create_raw_measure(self):
        m = Measure.objects.create(
            tenant_id=self.tenant.id,
            name="Press A Cycle Time",
            slug="press-a-ct",
            definition="Cycle time for press A stamping operation",
            unit="sec",
            measure_type="process",
            value_type="continuous",
            created_by=self.user.email,
        )
        assert m.id is not None
        assert m.slug == "press-a-ct"
        assert m.formula is None
        assert m.is_deleted is False

    def test_create_calculated_measure(self):
        for slug, name in [("avail", "Availability"), ("perf", "Performance"), ("qual", "Quality")]:
            Measure.objects.create(
                tenant_id=self.tenant.id,
                name=name,
                slug=slug,
                unit="%",
                measure_type="process",
                value_type="proportion",
                created_by=self.user.email,
            )
        oee = Measure.objects.create(
            tenant_id=self.tenant.id,
            name="OEE",
            slug="oee",
            unit="%",
            measure_type="process",
            value_type="proportion",
            formula="[avail] * [perf] * [qual]",
            created_by=self.user.email,
        )
        assert oee.formula is not None
        assert oee.is_calculated

    def test_slug_unique_per_tenant(self):
        Measure.objects.create(
            tenant_id=self.tenant.id,
            name="CT",
            slug="ct",
            unit="sec",
            measure_type="process",
            value_type="continuous",
            created_by=self.user.email,
        )
        with self.assertRaises(IntegrityError):
            Measure.objects.create(
                tenant_id=self.tenant.id,
                name="CT Duplicate",
                slug="ct",
                unit="sec",
                measure_type="process",
                value_type="continuous",
                created_by=self.user.email,
            )

    def test_range_alarm_fields(self):
        m = Measure.objects.create(
            tenant_id=self.tenant.id,
            name="Temp",
            slug="temp",
            unit="C",
            measure_type="process",
            value_type="continuous",
            range_min=18.0,
            range_max=25.0,
            created_by=self.user.email,
        )
        assert m.range_min == 18.0
        assert m.range_max == 25.0

    def test_soft_delete(self):
        m = Measure.objects.create(
            tenant_id=self.tenant.id,
            name="Deletable",
            slug="del-me",
            unit="x",
            measure_type="process",
            value_type="continuous",
            created_by=self.user.email,
        )
        m.delete()
        assert m.is_deleted
        assert Measure.objects.filter(slug="del-me").count() == 0
        assert Measure.all_objects.filter(slug="del-me").count() == 1

    def test_to_dict(self):
        m = Measure.objects.create(
            tenant_id=self.tenant.id,
            name="Dict Test",
            slug="dict-test",
            unit="kg",
            measure_type="material",
            value_type="continuous",
            created_by=self.user.email,
        )
        d = m.to_dict()
        assert d["slug"] == "dict-test"
        assert d["is_calculated"] is False
        assert "id" in d
        assert "tenant_id" in d


@SECURE_OFF
class DatapointModelTest(TestCase):
    def setUp(self):
        self.user = make_user("dp@test.com", tier="team")
        self.tenant = make_tenant("DP Org", slug="dp-org", plan="team")
        make_membership(self.tenant, self.user)
        self.measure = Measure.objects.create(
            tenant_id=self.tenant.id,
            name="CT",
            slug="ct",
            unit="sec",
            measure_type="process",
            value_type="continuous",
            created_by=self.user.email,
        )

    def test_create_datapoint(self):
        dp = Datapoint.objects.create(
            measure=self.measure,
            value=45.0,
            source_type="manual",
            observation_count=1,
            actor=self.user.email,
            tenant_id=self.tenant.id,
        )
        assert dp.id is not None
        assert dp.value == 45.0
        assert dp.confidence > 0 or dp.confidence == 0.0  # manual n=1 → 0.0

    def test_datapoint_immutable(self):
        dp = Datapoint.objects.create(
            measure=self.measure,
            value=45.0,
            source_type="manual",
            observation_count=1,
            actor=self.user.email,
            tenant_id=self.tenant.id,
        )
        dp.value = 99.0
        with self.assertRaises(ValueError):
            dp.save()

    def test_datapoint_cannot_delete(self):
        dp = Datapoint.objects.create(
            measure=self.measure,
            value=45.0,
            source_type="manual",
            observation_count=1,
            actor=self.user.email,
            tenant_id=self.tenant.id,
        )
        with self.assertRaises(PermissionError):
            dp.delete()

    def test_confidence_auto_computed(self):
        dp_manual = Datapoint.objects.create(
            measure=self.measure,
            value=45.0,
            source_type="manual",
            observation_count=5,
            actor=self.user.email,
            tenant_id=self.tenant.id,
        )
        dp_doe = Datapoint.objects.create(
            measure=self.measure,
            value=44.7,
            source_type="doe",
            observation_count=30,
            actor=self.user.email,
            tenant_id=self.tenant.id,
        )
        assert dp_doe.confidence > dp_manual.confidence

    def test_timestamp_auto_set(self):
        dp = Datapoint.objects.create(
            measure=self.measure,
            value=45.0,
            source_type="manual",
            observation_count=1,
            actor=self.user.email,
            tenant_id=self.tenant.id,
        )
        assert dp.timestamp is not None

    def test_to_dict(self):
        dp = Datapoint.objects.create(
            measure=self.measure,
            value=42.0,
            source_type="doe",
            observation_count=10,
            actor=self.user.email,
            tenant_id=self.tenant.id,
        )
        d = dp.to_dict()
        assert d["value"] == 42.0
        assert d["source_type"] == "doe"
        assert "confidence" in d


@SECURE_OFF
class MeasureTargetModelTest(TestCase):
    def setUp(self):
        self.user = make_user("mt@test.com", tier="team")
        self.tenant = make_tenant("MT Org", slug="mt-org", plan="team")
        make_membership(self.tenant, self.user)
        self.measure = Measure.objects.create(
            tenant_id=self.tenant.id,
            name="CT",
            slug="ct-target",
            unit="sec",
            measure_type="process",
            value_type="continuous",
            created_by=self.user.email,
        )

    def test_create_target(self):
        t = MeasureTarget.objects.create(
            tenant_id=self.tenant.id,
            measure=self.measure,
            target_value=38.0,
            target_date=date(2026, 9, 1),
            source="hoshin",
            created_by=self.user.email,
        )
        assert t.target_value == 38.0
        assert t.source == "hoshin"

    def test_target_is_mutable(self):
        t = MeasureTarget.objects.create(
            tenant_id=self.tenant.id,
            measure=self.measure,
            target_value=38.0,
            source="manual",
            created_by=self.user.email,
        )
        t.target_value = 35.0
        t.save()
        t.refresh_from_db()
        assert t.target_value == 35.0

    def test_to_dict(self):
        t = MeasureTarget.objects.create(
            tenant_id=self.tenant.id,
            measure=self.measure,
            target_value=40.0,
            source="calculator",
            created_by=self.user.email,
        )
        d = t.to_dict()
        assert d["target_value"] == 40.0
        assert d["source"] == "calculator"
        assert "measure_id" in d
