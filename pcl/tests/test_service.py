"""PCL service layer integration tests."""

from datetime import date, timedelta

from django.test import TestCase
from django.utils import timezone

from conftest import SECURE_OFF, make_membership, make_tenant, make_user
from pcl import service
from pcl.models import Measure


@SECURE_OFF
class ReadWriteTest(TestCase):
    def setUp(self):
        self.user = make_user("svc@test.com", tier="team")
        self.tenant = make_tenant("Svc Org", slug="svc-org", plan="team")
        make_membership(self.tenant, self.user)
        self.measure = Measure.objects.create(
            tenant_id=self.tenant.id,
            name="Cycle Time",
            slug="ct",
            unit="sec",
            measure_type="process",
            value_type="continuous",
            created_by=self.user.email,
        )

    def test_read_empty_returns_none(self):
        assert service.read("ct", tenant_id=self.tenant.id) is None

    def test_read_nonexistent_returns_none(self):
        assert service.read("nonexistent", tenant_id=self.tenant.id) is None

    def test_write_and_read(self):
        service.write(
            measure_slug="ct",
            value=45.0,
            source_type="manual",
            actor=self.user.email,
            tenant_id=self.tenant.id,
            observation_count=1,
        )
        val = service.read("ct", tenant_id=self.tenant.id)
        assert val == 45.0

    def test_write_updates_aggregate(self):
        service.write("ct", 45.0, "manual", self.user.email, self.tenant.id, observation_count=5)
        service.write("ct", 44.0, "doe", self.user.email, self.tenant.id, observation_count=30)

        self.measure.refresh_from_db()
        assert self.measure.cached_n == 2
        assert self.measure.cached_value is not None
        # DOE with higher confidence should pull aggregate toward 44
        assert self.measure.cached_value < 45.0

    def test_write_to_calculated_raises(self):
        Measure.objects.create(
            tenant_id=self.tenant.id,
            name="OEE",
            slug="oee",
            unit="%",
            measure_type="process",
            value_type="proportion",
            formula="[ct] * 2",
            created_by=self.user.email,
        )
        with self.assertRaises(ValueError, msg="Cannot write to calculated"):
            service.write("oee", 0.85, "manual", self.user.email, self.tenant.id)

    def test_range_alarm(self):
        self.measure.range_max = 50.0
        self.measure.save(update_fields=["range_max"])

        result = service.write("ct", 55.0, "manual", self.user.email, self.tenant.id)
        assert "alarm" in result
        assert result["alarm"]["type"] == "above_max"


@SECURE_OFF
class ReadWithMetaTest(TestCase):
    def setUp(self):
        self.user = make_user("meta@test.com", tier="team")
        self.tenant = make_tenant("Meta Org", slug="meta-org", plan="team")
        make_membership(self.tenant, self.user)
        self.measure = Measure.objects.create(
            tenant_id=self.tenant.id,
            name="CT",
            slug="ct-meta",
            unit="sec",
            measure_type="process",
            value_type="continuous",
            created_by=self.user.email,
        )

    def test_meta_empty(self):
        meta = service.read_with_meta("ct-meta", tenant_id=self.tenant.id)
        assert meta["value"] is None
        assert meta["n"] == 0

    def test_meta_after_write(self):
        service.write("ct-meta", 45.0, "doe", self.user.email, self.tenant.id, observation_count=10)
        meta = service.read_with_meta("ct-meta", tenant_id=self.tenant.id)
        assert meta["value"] == 45.0
        assert meta["confidence"] > 0
        assert meta["n"] == 1
        assert meta["unit"] == "sec"
        assert meta["is_calculated"] is False

    def test_meta_nonexistent(self):
        meta = service.read_with_meta("nope", tenant_id=self.tenant.id)
        assert meta["value"] is None
        assert "error" in meta


@SECURE_OFF
class FormulaResolutionTest(TestCase):
    def setUp(self):
        self.user = make_user("formula@test.com", tier="team")
        self.tenant = make_tenant("Formula Org", slug="formula-org", plan="team")
        make_membership(self.tenant, self.user)

        # Create component measures
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

        # Calculated measure
        Measure.objects.create(
            tenant_id=self.tenant.id,
            name="OEE",
            slug="oee",
            unit="%",
            measure_type="process",
            value_type="proportion",
            formula="[avail] * [perf] * [qual]",
            created_by=self.user.email,
        )

    def test_formula_resolves(self):
        service.write("avail", 0.90, "manual", self.user.email, self.tenant.id, observation_count=10)
        service.write("perf", 0.85, "manual", self.user.email, self.tenant.id, observation_count=10)
        service.write("qual", 0.95, "manual", self.user.email, self.tenant.id, observation_count=10)

        val = service.read("oee", tenant_id=self.tenant.id)
        assert val is not None
        assert abs(val - 0.72675) < 0.01

    def test_formula_missing_component_returns_none(self):
        service.write("avail", 0.90, "manual", self.user.email, self.tenant.id, observation_count=10)
        # perf and qual have no data
        val = service.read("oee", tenant_id=self.tenant.id)
        assert val is None

    def test_formula_meta_weakest_link(self):
        service.write("avail", 0.90, "doe", self.user.email, self.tenant.id, observation_count=30)
        service.write("perf", 0.85, "manual", self.user.email, self.tenant.id, observation_count=2)
        service.write("qual", 0.95, "doe", self.user.email, self.tenant.id, observation_count=30)

        meta = service.read_with_meta("oee", tenant_id=self.tenant.id)
        assert meta["is_calculated"] is True
        # Confidence should be weakest component (manual n=2)
        perf_meta = service.read_with_meta("perf", tenant_id=self.tenant.id)
        assert meta["confidence"] == perf_meta["confidence"]


@SECURE_OFF
class SetTargetTest(TestCase):
    def setUp(self):
        self.user = make_user("target@test.com", tier="team")
        self.tenant = make_tenant("Target Org", slug="target-org", plan="team")
        make_membership(self.tenant, self.user)
        Measure.objects.create(
            tenant_id=self.tenant.id,
            name="CT",
            slug="ct-tgt",
            unit="sec",
            measure_type="process",
            value_type="continuous",
            created_by=self.user.email,
        )

    def test_set_target(self):
        result = service.set_target(
            "ct-tgt",
            target_value=38.0,
            source="hoshin",
            actor=self.user.email,
            tenant_id=self.tenant.id,
            target_date=date(2026, 9, 1),
        )
        assert result["target_value"] == 38.0
        assert result["source"] == "hoshin"

    def test_update_target(self):
        service.set_target("ct-tgt", 38.0, "hoshin", self.user.email, self.tenant.id)
        service.set_target("ct-tgt", 35.0, "hoshin", self.user.email, self.tenant.id)

        from pcl.models import MeasureTarget

        targets = MeasureTarget.objects.filter(measure__slug="ct-tgt")
        assert targets.count() == 1
        assert targets.first().target_value == 35.0

    def test_set_target_nonexistent_measure(self):
        with self.assertRaises(ValueError):
            service.set_target("nope", 42.0, "manual", self.user.email, self.tenant.id)


@SECURE_OFF
class ProvenanceAggregationTest(TestCase):
    def setUp(self):
        self.user = make_user("prov@test.com", tier="team")
        self.tenant = make_tenant("Prov Org", slug="prov-org", plan="team")
        make_membership(self.tenant, self.user)
        self.measure = Measure.objects.create(
            tenant_id=self.tenant.id,
            name="CT",
            slug="ct-prov",
            unit="sec",
            measure_type="process",
            value_type="continuous",
            created_by=self.user.email,
        )

    def test_observed_updates_cache(self):
        service.write(
            "ct-prov",
            45.0,
            "manual",
            self.user.email,
            self.tenant.id,
            observation_count=5,
            provenance="observed",
        )
        self.measure.refresh_from_db()
        assert self.measure.cached_value == 45.0
        assert self.measure.cached_n == 1

    def test_calculated_updates_cache(self):
        service.write(
            "ct-prov",
            1.33,
            "workbench",
            self.user.email,
            self.tenant.id,
            observation_count=1,
            provenance="calculated",
        )
        self.measure.refresh_from_db()
        assert self.measure.cached_value == 1.33
        assert self.measure.cached_n == 1

    def test_simulated_skips_cache(self):
        service.write(
            "ct-prov",
            45.0,
            "manual",
            self.user.email,
            self.tenant.id,
            observation_count=5,
            provenance="observed",
        )
        service.write(
            "ct-prov",
            999.0,
            "workbench",
            self.user.email,
            self.tenant.id,
            observation_count=1,
            provenance="simulated",
        )
        self.measure.refresh_from_db()
        assert self.measure.cached_value == 45.0
        assert self.measure.cached_n == 1

    def test_projected_skips_cache(self):
        service.write(
            "ct-prov",
            45.0,
            "manual",
            self.user.email,
            self.tenant.id,
            observation_count=5,
            provenance="observed",
        )
        service.write(
            "ct-prov",
            30.0,
            "workbench",
            self.user.email,
            self.tenant.id,
            observation_count=1,
            provenance="projected",
        )
        self.measure.refresh_from_db()
        assert self.measure.cached_value == 45.0
        assert self.measure.cached_n == 1

    def test_simulated_datapoint_still_stored(self):
        result = service.write(
            "ct-prov",
            999.0,
            "workbench",
            self.user.email,
            self.tenant.id,
            observation_count=1,
            provenance="simulated",
        )
        assert result["value"] == 999.0
        assert result["provenance"] == "simulated"
        from pcl.models import Datapoint

        assert Datapoint.objects.filter(measure=self.measure).count() == 1

    def test_default_provenance_is_observed(self):
        service.write(
            "ct-prov",
            45.0,
            "manual",
            self.user.email,
            self.tenant.id,
            observation_count=5,
        )
        self.measure.refresh_from_db()
        assert self.measure.cached_n == 1  # observed updates cache


@SECURE_OFF
class HistoricalReadTest(TestCase):
    def setUp(self):
        self.user = make_user("hist@test.com", tier="team")
        self.tenant = make_tenant("Hist Org", slug="hist-org", plan="team")
        make_membership(self.tenant, self.user)
        self.measure = Measure.objects.create(
            tenant_id=self.tenant.id,
            name="CT",
            slug="ct-hist",
            unit="sec",
            measure_type="process",
            value_type="continuous",
            created_by=self.user.email,
        )

    def test_historical_read(self):
        now = timezone.now()
        t1 = now - timedelta(days=10)
        t2 = now - timedelta(days=5)

        service.write("ct-hist", 50.0, "manual", self.user.email, self.tenant.id, timestamp=t1, observation_count=5)
        service.write("ct-hist", 45.0, "doe", self.user.email, self.tenant.id, timestamp=t2, observation_count=30)

        # Historical read at t1+1 day should only see the first datapoint
        val = service.read("ct-hist", tenant_id=self.tenant.id, at=t1 + timedelta(days=1))
        assert val == 50.0

    def test_historical_read_empty(self):
        long_ago = timezone.now() - timedelta(days=365)
        val = service.read("ct-hist", tenant_id=self.tenant.id, at=long_ago)
        assert val is None


@SECURE_OFF
class JobIntegrationTest(TestCase):
    """End-to-end: Job creates outputs, bound output writes to PCL."""

    def setUp(self):
        self.user = make_user("integ@test.com", tier="team")
        self.tenant = make_tenant("Integ Org", slug="integ-org", plan="team")
        make_membership(self.tenant, self.user)
        self.measure = Measure.objects.create(
            tenant_id=self.tenant.id,
            name="Bore Cpk",
            slug="bore-cpk",
            unit="",
            measure_type="product",
            value_type="continuous",
            created_by=self.user.email,
        )

    def test_job_output_writes_to_pcl(self):
        from job.models import Job, JobOutput

        # 1. Create job (silent, no modals)
        job = Job.objects.create(
            tenant_id=self.tenant.id,
            canvas_id=None,
            status="completed",
            inputs={"data": [10.1, 10.2, 9.9], "usl": 10.5, "lsl": 9.5},
            outputs_summary={"cpk": 1.33},
            actor=self.user.email,
            created_by=self.user.email,
        )

        # 2. Create job output (always recorded)
        out = JobOutput.objects.create(
            job=job,
            output_key="cpk",
            output_type="metric",
            value_numeric=1.33,
            value_json={"cpk": 1.33, "cpl": 1.45, "cpu": 1.21},
            provenance="calculated",
            measure_slug="bore-cpk",
            actor=self.user.email,
            tenant_id=self.tenant.id,
        )

        # 3. Explicit write to PCL (canvas binding would do this)
        result = service.write(
            measure_slug="bore-cpk",
            value=out.value_numeric,
            source_type="workbench",
            actor=self.user.email,
            tenant_id=self.tenant.id,
            provenance="calculated",
            source_job_id=job.id,
        )

        # 4. Verify PCL has the value
        assert result["provenance"] == "calculated"
        val = service.read("bore-cpk", tenant_id=self.tenant.id)
        assert val == 1.33

        # 5. Verify Datapoint links back to Job
        from pcl.models import Datapoint

        dp = Datapoint.objects.get(measure=self.measure)
        assert dp.source_job_id == job.id
        assert dp.provenance == "calculated"

    def test_scratch_job_simulated_skips_cache(self):
        from job.models import Job, JobOutput

        # Write a real observed value
        service.write(
            "bore-cpk",
            1.33,
            "manual",
            self.user.email,
            self.tenant.id,
            provenance="observed",
            observation_count=5,
        )

        # Scratch job with simulated output
        job = Job.objects.create(
            tenant_id=self.tenant.id,
            status="completed",
            inputs={"data": [10.1, 10.2, 9.9], "usl": 11.0, "lsl": 9.0},
            actor=self.user.email,
            is_scratch=True,
            created_by=self.user.email,
        )
        JobOutput.objects.create(
            job=job,
            output_key="cpk",
            output_type="metric",
            value_numeric=2.50,
            value_json={"cpk": 2.50},
            provenance="simulated",
            measure_slug="bore-cpk",
            actor=self.user.email,
            tenant_id=self.tenant.id,
        )

        # Write simulated value to PCL
        service.write(
            "bore-cpk",
            2.50,
            "workbench",
            self.user.email,
            self.tenant.id,
            provenance="simulated",
            source_job_id=job.id,
        )

        # Cache should still show the observed value
        self.measure.refresh_from_db()
        assert self.measure.cached_value == 1.33
        assert self.measure.cached_n == 1

        # But both datapoints exist
        from pcl.models import Datapoint

        assert Datapoint.objects.filter(measure=self.measure).count() == 2


@SECURE_OFF
class EnsureAndWriteTest(TestCase):
    """Test ensure_and_write — auto-creates measures on first write."""

    def setUp(self):
        self.user = make_user("ensure@test.com", tier="team")
        self.tenant = make_tenant("Ensure Org", slug="ensure-org", plan="team")
        make_membership(self.tenant, self.user)

    def test_auto_creates_measure(self):
        result = service.ensure_and_write(
            slug="wb/spc/diameter/cpk",
            value=1.45,
            source_type="workbench",
            actor=self.user.email,
            tenant_id=self.tenant.id,
            unit="",
            notes="spc/capability",
        )
        assert result["value"] == 1.45
        m = Measure.objects.get(slug="wb/spc/diameter/cpk", tenant_id=self.tenant.id)
        assert m.cached_value == 1.45
        assert m.measure_type == "process"

    def test_reuses_existing_measure(self):
        Measure.objects.create(
            tenant_id=self.tenant.id,
            name="Existing",
            slug="wb/spc/diameter/cpk",
            unit="",
            measure_type="product",
            value_type="continuous",
            created_by=self.user.email,
        )
        service.ensure_and_write(
            slug="wb/spc/diameter/cpk",
            value=1.45,
            source_type="workbench",
            actor=self.user.email,
            tenant_id=self.tenant.id,
        )
        assert Measure.objects.filter(slug="wb/spc/diameter/cpk", tenant_id=self.tenant.id).count() == 1
        m = Measure.objects.get(slug="wb/spc/diameter/cpk", tenant_id=self.tenant.id)
        assert m.measure_type == "product"  # didn't overwrite

    def test_auto_name_from_slug(self):
        service.ensure_and_write(
            slug="wb/stats/yield/p-value",
            value=0.03,
            source_type="workbench",
            actor=self.user.email,
            tenant_id=self.tenant.id,
        )
        m = Measure.objects.get(slug="wb/stats/yield/p-value", tenant_id=self.tenant.id)
        assert "Wb" in m.name  # auto-generated from slug


@SECURE_OFF
class ChainPCLWritebackTest(TestCase):
    """Test that analysis chain writes statistics to PCL."""

    def setUp(self):
        self.user = make_user("chain@test.com", tier="team")
        self.tenant = make_tenant("Chain Org", slug="chain-org", plan="team")
        make_membership(self.tenant, self.user)

    def test_assemble_writes_cpk_to_pcl(self):
        from analysis.chain import assemble

        raw = {
            "statistics": {"cpk": 1.33, "ppk": 1.21, "sigma_level": 4.0},
            "charts": [],
            "_config": {"measurement": "diameter"},
        }
        result = assemble(raw, "spc", "capability")
        assert result["statistics"]["cpk"] == 1.33

        # Verify PCL has the values (no tenant — chain doesn't know tenant)
        m = Measure.objects.filter(slug="wb/spc/diameter/cpk").first()
        assert m is not None
        assert m.cached_value == 1.33

        m2 = Measure.objects.filter(slug="wb/spc/diameter/ppk").first()
        assert m2 is not None
        assert m2.cached_value == 1.21

    def test_assemble_ignores_non_numeric(self):
        from analysis.chain import assemble

        raw = {
            "statistics": {"cpk": 1.33, "method": "xbar_r", "details": {"foo": "bar"}},
            "charts": [],
            "_config": {"measurement": "width"},
        }
        assemble(raw, "spc", "capability")
        assert not Measure.objects.filter(slug="wb/spc/width/method").exists()
        assert not Measure.objects.filter(slug="wb/spc/width/details").exists()

    def test_assemble_skips_nan(self):
        from analysis.chain import assemble

        raw = {
            "statistics": {"cpk": float("nan"), "ppk": 1.21},
            "charts": [],
            "_config": {"measurement": "x"},
        }
        assemble(raw, "spc", "capability")
        assert not Measure.objects.filter(slug="wb/spc/x/cpk").exists()
        assert Measure.objects.filter(slug="wb/spc/x/ppk").exists()
