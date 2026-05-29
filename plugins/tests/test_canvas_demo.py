"""Tests for canvas demo endpoint."""

import json
import numpy as np
from django.test import TestCase

from conftest import SECURE_OFF, make_user
from job.models import Job
from syn.plugins.registry import get_registry
from plugins.capability import CapabilityStudyPlugin


@SECURE_OFF
class TestCanvasDemoEndpoint(TestCase):
    def setUp(self):
        self.user = make_user("canvas@test.com")
        self.client.login(username="canvas", password="testpass123!")
        reg = get_registry()
        if not reg.has("capability_study"):
            reg.register(CapabilityStudyPlugin)

        np.random.seed(42)
        self.good_data = np.random.normal(50, 2, 100).tolist()

    def test_successful_run(self):
        resp = self.client.post(
            "/api/demo/canvas/run/",
            data=json.dumps({
                "plugin_name": "capability_study",
                "input_data": {
                    "data": self.good_data,
                    "usl": 56.0,
                    "lsl": 44.0,
                },
            }),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "completed")
        self.assertIn("job_id", body)
        self.assertIsInstance(body["duration_ms"], int)
        output_keys = [o["key"] for o in body["outputs"]]
        self.assertIn("cpk", output_keys)
        self.assertIn("ppk", output_keys)
        self.assertIn("summary", output_keys)
        chart_outputs = [o for o in body["outputs"] if o["type"] == "chart"]
        self.assertGreaterEqual(len(chart_outputs), 1)

    def test_job_created_in_db(self):
        initial_count = Job.objects.count()
        self.client.post(
            "/api/demo/canvas/run/",
            data=json.dumps({
                "plugin_name": "capability_study",
                "input_data": {
                    "data": self.good_data,
                    "usl": 56.0,
                    "lsl": 44.0,
                },
            }),
            content_type="application/json",
        )
        self.assertEqual(Job.objects.count(), initial_count + 1)
        job = Job.objects.order_by("-created_at").first()
        self.assertEqual(job.plugin_name, "capability_study")
        self.assertEqual(job.status, "completed")
        self.assertTrue(job.outputs.filter(output_key="cpk").exists())

    def test_scratch_flag(self):
        self.client.post(
            "/api/demo/canvas/run/",
            data=json.dumps({
                "plugin_name": "capability_study",
                "input_data": {
                    "data": self.good_data,
                    "usl": 56.0,
                    "lsl": 44.0,
                },
                "is_scratch": True,
            }),
            content_type="application/json",
        )
        job = Job.objects.order_by("-created_at").first()
        self.assertTrue(job.is_scratch)

    def test_validation_error(self):
        resp = self.client.post(
            "/api/demo/canvas/run/",
            data=json.dumps({
                "plugin_name": "capability_study",
                "input_data": {"data": [1.0]},
            }),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        body = resp.json()
        self.assertIn("error", body)

    def test_unknown_plugin(self):
        resp = self.client.post(
            "/api/demo/canvas/run/",
            data=json.dumps({
                "plugin_name": "nonexistent",
                "input_data": {},
            }),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_requires_auth(self):
        self.client.logout()
        resp = self.client.post(
            "/api/demo/canvas/run/",
            data=json.dumps({
                "plugin_name": "capability_study",
                "input_data": {"data": self.good_data, "usl": 56.0, "lsl": 44.0},
            }),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 401)
