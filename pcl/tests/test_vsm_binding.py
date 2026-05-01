"""VSM ↔ PCL binding tests — bind, unbind, step context resolution."""

import json

from django.test import TestCase

from agents_api.models import ValueStreamMap
from conftest import SECURE_OFF, make_membership, make_tenant, make_user
from pcl import service
from pcl.models import Measure


@SECURE_OFF
class VSMBindingTest(TestCase):
    def setUp(self):
        self.user = make_user("vsm-bind@test.com", tier="team")
        self.tenant = make_tenant("VSM Bind Org", slug="vsm-bind-org", plan="team")
        make_membership(self.tenant, self.user)
        self.client.force_login(self.user)

        self.vsm = ValueStreamMap.objects.create(
            owner=self.user,
            tenant=self.tenant,
            name="Test VSM",
            process_steps=[
                {
                    "id": "step-1",
                    "name": "Press A",
                    "cycle_time": 45.0,
                    "changeover_time": 1800.0,
                    "uptime": 95.0,
                    "x": 100,
                    "y": 200,
                }
            ],
        )

        self.ct_measure = Measure.objects.create(
            tenant_id=self.tenant.id,
            name="Press A CT",
            slug="press-a-ct",
            unit="sec",
            measure_type="process",
            value_type="continuous",
            created_by=self.user.email,
        )
        service.write(
            measure_slug="press-a-ct",
            value=54.0,
            source_type="doe",
            actor=self.user.email,
            tenant_id=self.tenant.id,
            observation_count=30,
        )

    def test_bind_step_to_measure(self):
        resp = self.client.post(
            f"/api/vsm/{self.vsm.id}/bind-pcl/step-1/",
            json.dumps({"field": "cycle_time", "measure_slug": "press-a-ct"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        self.vsm.refresh_from_db()
        step = self.vsm.process_steps[0]
        assert step["pcl_bindings"]["cycle_time"] == "press-a-ct"

    def test_unbind_step_snapshots_value(self):
        self.vsm.process_steps[0]["pcl_bindings"] = {"cycle_time": "press-a-ct"}
        self.vsm.save()

        resp = self.client.post(
            f"/api/vsm/{self.vsm.id}/unbind-pcl/step-1/",
            json.dumps({"field": "cycle_time"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        self.vsm.refresh_from_db()
        step = self.vsm.process_steps[0]
        bindings = step.get("pcl_bindings", {})
        assert "cycle_time" not in bindings
        # Inline value should be snapshot of PCL value (54.0), not original 45.0
        assert step["cycle_time"] == 54.0

    def test_bind_missing_fields(self):
        resp = self.client.post(
            f"/api/vsm/{self.vsm.id}/bind-pcl/step-1/",
            json.dumps({"field": "cycle_time"}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_bind_nonexistent_step(self):
        resp = self.client.post(
            f"/api/vsm/{self.vsm.id}/bind-pcl/step-999/",
            json.dumps({"field": "cycle_time", "measure_slug": "press-a-ct"}),
            content_type="application/json",
        )
        assert resp.status_code == 404

    def test_parse_step_context_resolves_binding(self):
        """When a step has pcl_bindings, _parse_step_context returns PCL value."""
        self.vsm.process_steps[0]["pcl_bindings"] = {"cycle_time": "press-a-ct"}
        self.vsm.save()

        from vsm.views import _parse_step_context

        ctx = _parse_step_context(self.vsm.process_steps[0], self.vsm)
        # PCL value is 54.0 (from DOE), not inline 45.0
        assert ctx["ct"] == 54.0

    def test_parse_step_context_no_binding_uses_inline(self):
        """Without binding, _parse_step_context uses inline value."""
        from vsm.views import _parse_step_context

        ctx = _parse_step_context(self.vsm.process_steps[0], self.vsm)
        assert ctx["ct"] == 45.0

    def test_parse_step_context_binding_empty_pcl_falls_back(self):
        """Bound to a measure with no data falls back to inline."""
        Measure.objects.create(
            tenant_id=self.tenant.id,
            name="Empty",
            slug="empty-measure",
            unit="sec",
            measure_type="process",
            value_type="continuous",
            created_by=self.user.email,
        )
        self.vsm.process_steps[0]["pcl_bindings"] = {"cycle_time": "empty-measure"}
        self.vsm.save()

        from vsm.views import _parse_step_context

        ctx = _parse_step_context(self.vsm.process_steps[0], self.vsm)
        # No PCL data → falls back to inline 45.0
        assert ctx["ct"] == 45.0
