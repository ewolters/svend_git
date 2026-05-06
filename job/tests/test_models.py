"""Job model tests — Job, JobOutput."""

import uuid

from django.test import TestCase
from django.utils import timezone

from conftest import SECURE_OFF, make_membership, make_tenant, make_user
from job.models import Job, JobOutput


@SECURE_OFF
class JobModelTest(TestCase):
    def setUp(self):
        self.user = make_user("job@test.com", tier="team")
        self.tenant = make_tenant("Job Org", slug="job-org", plan="team")
        make_membership(self.tenant, self.user)

    def test_create_job(self):
        job = Job.objects.create(
            tenant_id=self.tenant.id,
            canvas_id=uuid.uuid4(),
            status="pending",
            inputs={"data": [1, 2, 3], "usl": 10.0, "lsl": 0.0},
            actor=self.user.email,
            created_by=self.user.email,
        )
        assert job.id is not None
        assert job.status == "pending"
        assert job.is_scratch is False

    def test_job_without_canvas(self):
        job = Job.objects.create(
            tenant_id=self.tenant.id,
            canvas_id=None,
            status="completed",
            inputs={"data": [1, 2, 3]},
            actor=self.user.email,
            created_by=self.user.email,
        )
        assert job.canvas_id is None

    def test_job_lifecycle(self):
        now = timezone.now()
        job = Job.objects.create(
            tenant_id=self.tenant.id,
            status="pending",
            inputs={"data": [1, 2, 3]},
            actor=self.user.email,
            created_by=self.user.email,
        )
        job.status = "running"
        job.started_at = now
        job.save(update_fields=["status", "started_at", "updated_at"])

        job.status = "completed"
        job.completed_at = timezone.now()
        job.duration_ms = 1500
        job.outputs_summary = {"cpk": 1.33, "output_count": 2}
        job.save(
            update_fields=[
                "status",
                "completed_at",
                "duration_ms",
                "outputs_summary",
                "updated_at",
            ]
        )
        job.refresh_from_db()
        assert job.status == "completed"
        assert job.duration_ms == 1500

    def test_scratch_job(self):
        job = Job.objects.create(
            tenant_id=self.tenant.id,
            status="completed",
            inputs={},
            actor=self.user.email,
            is_scratch=True,
            created_by=self.user.email,
        )
        assert job.is_scratch is True

    def test_soft_delete(self):
        job = Job.objects.create(
            tenant_id=self.tenant.id,
            status="completed",
            inputs={},
            actor=self.user.email,
            created_by=self.user.email,
        )
        job.delete()
        assert job.is_deleted
        assert Job.objects.filter(id=job.id).count() == 0
        assert Job.all_objects.filter(id=job.id).count() == 1

    def test_to_dict(self):
        job = Job.objects.create(
            tenant_id=self.tenant.id,
            status="completed",
            inputs={"x": 1},
            actor=self.user.email,
            created_by=self.user.email,
        )
        d = job.to_dict()
        assert d["status"] == "completed"
        assert d["inputs"] == {"x": 1}
        assert "id" in d
        assert "actor" in d


@SECURE_OFF
class JobOutputModelTest(TestCase):
    def setUp(self):
        self.user = make_user("jout@test.com", tier="team")
        self.tenant = make_tenant("JOut Org", slug="jout-org", plan="team")
        make_membership(self.tenant, self.user)
        self.job = Job.objects.create(
            tenant_id=self.tenant.id,
            status="completed",
            inputs={"data": [1, 2, 3]},
            actor=self.user.email,
            created_by=self.user.email,
        )

    def test_create_metric_output(self):
        out = JobOutput.objects.create(
            job=self.job,
            output_key="cpk",
            output_type="metric",
            value_numeric=1.33,
            value_json={"cpk": 1.33, "cpl": 1.45, "cpu": 1.21},
            provenance="calculated",
            measure_slug="bore-cpk",
            actor=self.user.email,
            tenant_id=self.tenant.id,
        )
        assert out.id is not None
        assert out.value_numeric == 1.33
        assert out.provenance == "calculated"
        assert out.measure_slug == "bore-cpk"

    def test_create_chart_output(self):
        out = JobOutput.objects.create(
            job=self.job,
            output_key="histogram",
            output_type="chart",
            value_json={"chart_type": "histogram", "bins": 20},
            provenance="calculated",
            actor=self.user.email,
            tenant_id=self.tenant.id,
        )
        assert out.output_type == "chart"
        assert out.value_numeric is None
        assert out.measure_slug is None

    def test_output_immutable(self):
        out = JobOutput.objects.create(
            job=self.job,
            output_key="cpk",
            output_type="metric",
            value_numeric=1.33,
            value_json={},
            provenance="calculated",
            actor=self.user.email,
            tenant_id=self.tenant.id,
        )
        out.value_numeric = 2.0
        with self.assertRaises(ValueError):
            out.save()

    def test_output_cannot_delete(self):
        out = JobOutput.objects.create(
            job=self.job,
            output_key="cpk",
            output_type="metric",
            value_numeric=1.33,
            value_json={},
            provenance="calculated",
            actor=self.user.email,
            tenant_id=self.tenant.id,
        )
        with self.assertRaises(PermissionError):
            out.delete()

    def test_multiple_outputs_per_job(self):
        for key, otype in [("cpk", "metric"), ("histogram", "chart"), ("summary", "text")]:
            JobOutput.objects.create(
                job=self.job,
                output_key=key,
                output_type=otype,
                value_json={},
                provenance="calculated",
                actor=self.user.email,
                tenant_id=self.tenant.id,
            )
        assert self.job.outputs.count() == 3

    def test_to_dict(self):
        out = JobOutput.objects.create(
            job=self.job,
            output_key="cpk",
            output_type="metric",
            value_numeric=1.33,
            value_json={"cpk": 1.33},
            provenance="calculated",
            measure_slug="bore-cpk",
            actor=self.user.email,
            tenant_id=self.tenant.id,
        )
        d = out.to_dict()
        assert d["output_key"] == "cpk"
        assert d["value_numeric"] == 1.33
        assert d["provenance"] == "calculated"
        assert d["measure_slug"] == "bore-cpk"
