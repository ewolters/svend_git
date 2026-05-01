"""PCL API view tests."""

import json

from django.test import TestCase

from conftest import SECURE_OFF, make_membership, make_tenant, make_user
from pcl import service
from pcl.models import Measure


@SECURE_OFF
class MeasureListTest(TestCase):
    def setUp(self):
        self.user = make_user("vlist@test.com", tier="team")
        self.tenant = make_tenant("VList Org", slug="vlist-org", plan="team")
        make_membership(self.tenant, self.user)
        self.client.force_login(self.user)
        Measure.objects.create(
            tenant_id=self.tenant.id,
            name="Cycle Time",
            slug="ct",
            unit="sec",
            measure_type="process",
            value_type="continuous",
            created_by=self.user.email,
        )

    def test_list_measures(self):
        resp = self.client.get("/api/pcl/measures/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["measures"][0]["slug"] == "ct"

    def test_list_filter_by_type(self):
        resp = self.client.get("/api/pcl/measures/?measure_type=process")
        assert resp.status_code == 200
        assert resp.json()["count"] == 1

        resp = self.client.get("/api/pcl/measures/?measure_type=material")
        assert resp.json()["count"] == 0

    def test_list_search(self):
        resp = self.client.get("/api/pcl/measures/?q=cycle")
        assert resp.json()["count"] == 1

    def test_unauthenticated(self):
        self.client.logout()
        resp = self.client.get("/api/pcl/measures/")
        assert resp.status_code in (401, 403)


@SECURE_OFF
class MeasureCreateTest(TestCase):
    def setUp(self):
        self.user = make_user("vcreate@test.com", tier="team")
        self.tenant = make_tenant("VCreate Org", slug="vcreate-org", plan="team")
        make_membership(self.tenant, self.user)
        self.client.force_login(self.user)

    def test_create_measure(self):
        resp = self.client.post(
            "/api/pcl/measures/create/",
            json.dumps(
                {
                    "name": "Press A Cycle Time",
                    "slug": "press-a-ct",
                    "unit": "sec",
                    "measure_type": "process",
                    "value_type": "continuous",
                }
            ),
            content_type="application/json",
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["slug"] == "press-a-ct"
        assert Measure.objects.filter(slug="press-a-ct").exists()

    def test_create_duplicate_slug(self):
        Measure.objects.create(
            tenant_id=self.tenant.id,
            name="Existing",
            slug="dup",
            measure_type="process",
            value_type="continuous",
            created_by=self.user.email,
        )
        resp = self.client.post(
            "/api/pcl/measures/create/",
            json.dumps({"name": "Dup", "slug": "dup", "measure_type": "process", "value_type": "continuous"}),
            content_type="application/json",
        )
        assert resp.status_code == 409

    def test_create_missing_fields(self):
        resp = self.client.post(
            "/api/pcl/measures/create/",
            json.dumps({"name": "No Slug"}),
            content_type="application/json",
        )
        assert resp.status_code == 400


@SECURE_OFF
class DatapointWriteTest(TestCase):
    def setUp(self):
        self.user = make_user("vwrite@test.com", tier="team")
        self.tenant = make_tenant("VWrite Org", slug="vwrite-org", plan="team")
        make_membership(self.tenant, self.user)
        self.client.force_login(self.user)
        Measure.objects.create(
            tenant_id=self.tenant.id,
            name="CT",
            slug="ct-write",
            unit="sec",
            measure_type="process",
            value_type="continuous",
            created_by=self.user.email,
        )

    def test_write_manual_with_confirmation(self):
        resp = self.client.post(
            "/api/pcl/write/",
            json.dumps(
                {
                    "measure_slug": "ct-write",
                    "value": 45.0,
                    "confirm_value": 45.0,
                    "source_type": "manual",
                    "observation_count": 1,
                }
            ),
            content_type="application/json",
        )
        assert resp.status_code == 201

    def test_write_manual_without_confirmation(self):
        resp = self.client.post(
            "/api/pcl/write/",
            json.dumps(
                {
                    "measure_slug": "ct-write",
                    "value": 45.0,
                    "source_type": "manual",
                }
            ),
            content_type="application/json",
        )
        assert resp.status_code == 400
        body = resp.json()
        # ErrorEnvelopeMiddleware wraps as {"error": {"message": "..."}}
        err_msg = body.get("error", {}).get("message", "")
        assert "confirm_value" in err_msg

    def test_write_automated_no_confirmation_needed(self):
        resp = self.client.post(
            "/api/pcl/write/",
            json.dumps(
                {
                    "measure_slug": "ct-write",
                    "value": 44.7,
                    "source_type": "doe",
                    "observation_count": 30,
                }
            ),
            content_type="application/json",
        )
        assert resp.status_code == 201

    def test_write_nonexistent_measure(self):
        resp = self.client.post(
            "/api/pcl/write/",
            json.dumps({"measure_slug": "nope", "value": 1.0, "source_type": "doe"}),
            content_type="application/json",
        )
        assert resp.status_code == 400


@SECURE_OFF
class MeasureReadTest(TestCase):
    def setUp(self):
        self.user = make_user("vread@test.com", tier="team")
        self.tenant = make_tenant("VRead Org", slug="vread-org", plan="team")
        make_membership(self.tenant, self.user)
        self.client.force_login(self.user)
        Measure.objects.create(
            tenant_id=self.tenant.id,
            name="CT",
            slug="ct-read",
            unit="sec",
            measure_type="process",
            value_type="continuous",
            created_by=self.user.email,
        )
        service.write("ct-read", 45.0, "doe", self.user.email, self.tenant.id, observation_count=10)

    def test_read_value(self):
        resp = self.client.get("/api/pcl/read/ct-read/")
        assert resp.status_code == 200
        assert resp.json()["value"] == 45.0

    def test_read_with_meta(self):
        resp = self.client.get("/api/pcl/read/ct-read/?meta=true")
        assert resp.status_code == 200
        data = resp.json()
        assert data["value"] == 45.0
        assert "confidence" in data
        assert data["unit"] == "sec"


@SECURE_OFF
class TargetSetTest(TestCase):
    def setUp(self):
        self.user = make_user("vtarget@test.com", tier="team")
        self.tenant = make_tenant("VTarget Org", slug="vtarget-org", plan="team")
        make_membership(self.tenant, self.user)
        self.client.force_login(self.user)
        Measure.objects.create(
            tenant_id=self.tenant.id,
            name="CT",
            slug="ct-tgt-v",
            unit="sec",
            measure_type="process",
            value_type="continuous",
            created_by=self.user.email,
        )

    def test_set_target(self):
        resp = self.client.post(
            "/api/pcl/target/",
            json.dumps(
                {
                    "measure_slug": "ct-tgt-v",
                    "target_value": 38.0,
                    "source": "hoshin",
                }
            ),
            content_type="application/json",
        )
        assert resp.status_code == 201
        assert resp.json()["target_value"] == 38.0


@SECURE_OFF
class MeasureSearchTest(TestCase):
    def setUp(self):
        self.user = make_user("vsearch@test.com", tier="team")
        self.tenant = make_tenant("VSearch Org", slug="vsearch-org", plan="team")
        make_membership(self.tenant, self.user)
        self.client.force_login(self.user)
        Measure.objects.create(
            tenant_id=self.tenant.id,
            name="Press A Cycle Time",
            slug="press-a-ct",
            unit="sec",
            measure_type="process",
            value_type="continuous",
            created_by=self.user.email,
        )

    def test_search_by_name(self):
        resp = self.client.get("/api/pcl/measures/search/?q=press")
        assert resp.status_code == 200
        assert resp.json()["count"] == 1

    def test_search_by_slug(self):
        resp = self.client.get("/api/pcl/measures/search/?q=press-a")
        assert resp.status_code == 200
        assert resp.json()["count"] == 1

    def test_search_too_short(self):
        resp = self.client.get("/api/pcl/measures/search/?q=p")
        assert resp.status_code == 400
