"""SPC view gap tests — upload, analyze_uploaded, gage_rr cache paths.

Covers upload_data(), analyze_uploaded(), and gage_rr() cache-based paths
that had zero or minimal coverage in existing smoke tests.

Standard: CAL-001 §7 (Endpoint Coverage), TST-001 §10.6
Compliance: SOC 2 CC4.1, CC7.2
<!-- test: agents_api.tests.test_spc_views_gaps -->
"""

import json

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from accounts.models import Tier, User
from agents_api import spc, spc_views

SMOKE_SETTINGS = {"RATELIMIT_ENABLE": False, "SECURE_SSL_REDIRECT": False}

_PASSWORD = "testpass123"


def _err_msg(resp):
    """Extract error message from ErrorEnvelopeMiddleware response."""
    data = resp.json()
    if isinstance(data.get("error"), dict):
        return data["error"].get("message", "")
    if "message" in data:
        return data["message"]
    return data.get("error", "")


def _make_user(email, tier=Tier.PRO):
    username = email.split("@")[0].replace(".", "_")
    u = User.objects.create_user(username=username, email=email, password=_PASSWORD)
    u.tier = tier
    u.email_verified = True
    u.save()
    return u


def _make_parsed_dataset(filename="test.csv", values=None):
    """Build a minimal ParsedDataset for cache seeding."""
    if values is None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
    return spc.ParsedDataset(
        filename=filename,
        row_count=len(values),
        columns=[
            spc.ColumnInfo(
                name="measurement",
                dtype="numeric",
                sample_values=values[:5],
                null_count=0,
                unique_count=len(set(values)),
                min_val=min(values),
                max_val=max(values),
                mean_val=sum(values) / len(values),
            ),
        ],
        data={"measurement": values},
        errors=[],
    )


def _make_gage_rr_dataset():
    """Build a ParsedDataset suitable for gage R&R (part/operator/measurement)."""
    # 3 parts x 2 operators x 2 replicates = 12 rows
    parts = ["P1", "P1", "P1", "P1", "P2", "P2", "P2", "P2", "P3", "P3", "P3", "P3"]
    operators = ["A", "A", "B", "B", "A", "A", "B", "B", "A", "A", "B", "B"]
    measurements = [1.0, 1.1, 1.05, 1.08, 2.0, 2.05, 2.02, 2.01, 3.0, 3.1, 3.05, 3.02]

    return spc.ParsedDataset(
        filename="gage_rr.csv",
        row_count=len(parts),
        columns=[
            spc.ColumnInfo(
                name="Part",
                dtype="text",
                sample_values=parts[:5],
                null_count=0,
                unique_count=3,
            ),
            spc.ColumnInfo(
                name="Operator",
                dtype="text",
                sample_values=operators[:5],
                null_count=0,
                unique_count=2,
            ),
            spc.ColumnInfo(
                name="Measurement",
                dtype="numeric",
                sample_values=measurements[:5],
                null_count=0,
                unique_count=len(set(measurements)),
                min_val=min(measurements),
                max_val=max(measurements),
                mean_val=sum(measurements) / len(measurements),
            ),
        ],
        data={
            "Part": parts,
            "Operator": operators,
            "Measurement": measurements,
        },
        errors=[],
    )


# ---------------------------------------------------------------------------
# Upload endpoint
# ---------------------------------------------------------------------------


@override_settings(**SMOKE_SETTINGS)
class SPCUploadTest(TestCase):
    """Tests for POST /api/spc/upload/ — file upload and parsing."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _make_user("spc-upload-gap@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=_PASSWORD)

    def tearDown(self):
        spc_views._parsed_data_cache.clear()

    def test_upload_unauth_returns_401(self):
        self.client.logout()
        res = self.client.post("/api/spc/upload/", {})
        self.assertIn(res.status_code, [401, 403])

    def test_upload_valid_csv_returns_columns(self):
        csv_content = b"measurement,batch\n1.0,A\n2.0,A\n3.0,B\n4.0,B\n5.0,B\n"
        uploaded = SimpleUploadedFile("data.csv", csv_content, content_type="text/csv")

        res = self.client.post("/api/spc/upload/", {"file": uploaded})
        self.assertEqual(res.status_code, 200)

        body = res.json()
        self.assertTrue(body.get("success"))
        self.assertEqual(body["filename"], "data.csv")
        self.assertEqual(body["row_count"], 5)
        self.assertIn("columns", body)
        self.assertIn("cache_key", body)
        # Verify cache was populated
        self.assertIn(body["cache_key"], spc_views._parsed_data_cache)

    def test_upload_no_file_returns_400(self):
        res = self.client.post("/api/spc/upload/", {})
        self.assertEqual(res.status_code, 400)

    def test_upload_unsupported_type_returns_400(self):
        uploaded = SimpleUploadedFile(
            "data.json", b'{"a": 1}', content_type="application/json"
        )

        res = self.client.post("/api/spc/upload/", {"file": uploaded})
        self.assertEqual(res.status_code, 400)
        self.assertIn("Unsupported", _err_msg(res))


# ---------------------------------------------------------------------------
# Analyze uploaded endpoint
# ---------------------------------------------------------------------------


@override_settings(**SMOKE_SETTINGS)
class SPCAnalyzeUploadedTest(TestCase):
    """Tests for POST /api/spc/analyze/ — analyze data from cache."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _make_user("spc-analyze-gap@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=_PASSWORD)

    def tearDown(self):
        spc_views._parsed_data_cache.clear()

    def _post(self, data):
        return self.client.post(
            "/api/spc/analyze/",
            json.dumps(data),
            content_type="application/json",
        )

    def test_nonexistent_cache_key_returns_400(self):
        res = self._post(
            {
                "cache_key": f"{self.user.id}:missing.csv",
                "analysis_type": "control_chart",
                "value_column": "measurement",
            }
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("not found", _err_msg(res).lower())

    def test_idor_wrong_user_returns_403(self):
        """Cache key prefixed with a different user ID should be rejected."""
        other_key = "999:secret.csv"
        spc_views._parsed_data_cache[other_key] = _make_parsed_dataset("secret.csv")

        res = self._post(
            {
                "cache_key": other_key,
                "analysis_type": "control_chart",
                "value_column": "measurement",
            }
        )
        self.assertEqual(res.status_code, 403)

    def test_valid_cache_key_control_chart(self):
        """Pre-seeded cache entry for the correct user should return 200."""
        values = [10.0, 10.2, 9.8, 10.1, 9.9, 10.3, 10.0, 9.7, 10.1, 10.0]
        parsed = _make_parsed_dataset("good.csv", values)
        cache_key = f"{self.user.id}:good.csv"
        spc_views._parsed_data_cache[cache_key] = parsed

        res = self._post(
            {
                "cache_key": cache_key,
                "analysis_type": "control_chart",
                "value_column": "measurement",
                "chart_type": "I-MR",
            }
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertTrue(body.get("success"))
        self.assertEqual(body.get("analysis_type"), "control_chart")

    def test_missing_value_column_returns_400(self):
        cache_key = f"{self.user.id}:data.csv"
        spc_views._parsed_data_cache[cache_key] = _make_parsed_dataset()

        res = self._post(
            {
                "cache_key": cache_key,
                "analysis_type": "control_chart",
            }
        )
        self.assertEqual(res.status_code, 400)


# ---------------------------------------------------------------------------
# Gage R&R — cache-based path and problem evidence saving
# ---------------------------------------------------------------------------


@override_settings(**SMOKE_SETTINGS)
class SPCGageRRCacheTest(TestCase):
    """Tests for POST /api/spc/gage-rr/ — cache path and evidence saving."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _make_user("spc-grr-gap@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=_PASSWORD)

    def tearDown(self):
        spc_views._parsed_data_cache.clear()

    def _post(self, data):
        return self.client.post(
            "/api/spc/gage-rr/",
            json.dumps(data),
            content_type="application/json",
        )

    def test_cache_based_gage_rr(self):
        """Gage R&R from cached upload data with column mapping."""
        cache_key = f"{self.user.id}:gage_rr.csv"
        spc_views._parsed_data_cache[cache_key] = _make_gage_rr_dataset()

        res = self._post(
            {
                "cache_key": cache_key,
                "part_column": "Part",
                "operator_column": "Operator",
                "measurement_column": "Measurement",
                "tolerance": 5.0,
            }
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertTrue(body.get("success"))
        self.assertIn("gage_rr", body)
        grr = body["gage_rr"]
        self.assertIn("grr_percent", grr)
        self.assertIn("assessment", grr)
        self.assertIn("ndc", grr)

    def test_cache_gage_rr_missing_columns_returns_400(self):
        """Missing column mapping parameters should return 400."""
        cache_key = f"{self.user.id}:gage_rr.csv"
        spc_views._parsed_data_cache[cache_key] = _make_gage_rr_dataset()

        res = self._post(
            {
                "cache_key": cache_key,
                "part_column": "Part",
                # operator_column and measurement_column omitted
            }
        )
        self.assertEqual(res.status_code, 400)

    def test_inline_gage_rr_no_problem(self):
        """Inline gage R&R without problem_id should still succeed."""
        res = self._post(
            {
                "parts": [
                    "P1",
                    "P1",
                    "P1",
                    "P1",
                    "P2",
                    "P2",
                    "P2",
                    "P2",
                    "P3",
                    "P3",
                    "P3",
                    "P3",
                ],
                "operators": [
                    "A",
                    "A",
                    "B",
                    "B",
                    "A",
                    "A",
                    "B",
                    "B",
                    "A",
                    "A",
                    "B",
                    "B",
                ],
                "measurements": [
                    1.0,
                    1.1,
                    1.05,
                    1.08,
                    2.0,
                    2.05,
                    2.02,
                    2.01,
                    3.0,
                    3.1,
                    3.05,
                    3.02,
                ],
            }
        )
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json().get("success"))
