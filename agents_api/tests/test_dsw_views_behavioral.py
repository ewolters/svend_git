"""Behavioral tests for DSW view functions with zero prior coverage.

Covers: _read_csv_safe, _load_dataset, explain_selection, hypothesis_timeline.

Standard: CAL-001 §7 (Endpoint Coverage), TST-001 §10.6
Compliance: SOC 2 CC4.1, CC7.2
<!-- test: agents_api.tests.test_dsw_views_behavioral -->
"""

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from django.test import SimpleTestCase, TestCase, override_settings

from accounts.models import Tier, User

SMOKE_SETTINGS = {"RATELIMIT_ENABLE": False, "SECURE_SSL_REDIRECT": False}


def _make_user(email, tier=Tier.PRO, staff=False):
    username = email.split("@")[0].replace(".", "_")
    u = User.objects.create_user(username=username, email=email, password="testpass123")
    u.tier = tier
    u.is_staff = staff
    u.email_verified = True
    u.save()
    return u


# ---------------------------------------------------------------------------
# _read_csv_safe -- encoding fallback
# ---------------------------------------------------------------------------


class ReadCsvSafeTest(SimpleTestCase):
    """Unit tests for _read_csv_safe encoding fallback logic."""

    def _read(self, file_or_path):
        from dsw.views import _read_csv_safe

        return _read_csv_safe(file_or_path)

    def test_utf8_file_object(self):
        """UTF-8 encoded BytesIO reads correctly."""
        csv_bytes = "name,value\nalice,1\nbob,2\n".encode("utf-8")
        df = self._read(io.BytesIO(csv_bytes))
        self.assertEqual(len(df), 2)
        self.assertIn("name", df.columns)
        self.assertEqual(df.iloc[0]["name"], "alice")

    def test_latin1_fallback_file_object(self):
        """Latin-1 file with special chars falls back and reads."""
        csv_bytes = "name,city\nRene,Montr\xe9al\nPe\xf1a,Le\xf3n\n".encode("latin-1")
        df = self._read(io.BytesIO(csv_bytes))
        self.assertEqual(len(df), 2)
        self.assertIn("Montr\xe9al", df["city"].values)

    def test_string_path(self):
        """String path (file on disk) reads correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("x,y\n1,2\n3,4\n")
            f.flush()
            path = f.name
        df = self._read(path)
        self.assertEqual(len(df), 2)
        self.assertListEqual(list(df.columns), ["x", "y"])


# ---------------------------------------------------------------------------
# _load_dataset -- priority resolution
# ---------------------------------------------------------------------------


@override_settings(**SMOKE_SETTINGS)
class LoadDatasetTest(TestCase):
    """Tests for _load_dataset data source priority."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _make_user("dsw-load@test.com")

    def _load(self, data_id):
        from dsw.views import _load_dataset

        return _load_dataset(self.user, data_id)

    @patch("dsw.views._read_csv_safe")
    def test_media_root_found(self, mock_read):
        """Source 1: MEDIA_ROOT/analysis_data/{user.id}/{data_id}.csv exists."""
        expected_df = pd.DataFrame({"a": [1, 2]})
        mock_read.return_value = expected_df

        with patch.object(Path, "exists", return_value=True):
            df = self._load("data_abc123")

        self.assertIsNotNone(df)
        mock_read.assert_called_once()

    @patch("dsw.views._read_csv_safe")
    def test_tmp_fallback(self, mock_read):
        """Source 2: MEDIA_ROOT miss, /tmp/svend_analysis/ found."""
        expected_df = pd.DataFrame({"b": [3, 4]})
        mock_read.return_value = expected_df

        call_count = 0

        def exists_side_effect(self_path):
            nonlocal call_count
            call_count += 1
            # First call (MEDIA_ROOT) -> False, second call (tmp) -> True
            if call_count == 1:
                return False
            return True

        with patch.object(Path, "exists", exists_side_effect):
            df = self._load("data_abc123")

        self.assertIsNotNone(df)
        mock_read.assert_called_once()

    def test_triage_result_fallback(self):
        """Source 3: Both file paths miss, TriageResult DB lookup succeeds."""
        # Use a non-data_ prefixed ID so it skips file-based lookups entirely
        mock_triage = MagicMock()
        mock_triage.cleaned_csv = "c\n5\n"

        with patch("triage.models.TriageResult.objects") as mock_qs:
            mock_qs.get.return_value = mock_triage
            df = self._load("triage_abc123")

        self.assertIsNotNone(df)
        self.assertEqual(len(df), 1)

    def test_all_miss_returns_none(self):
        """All sources miss -> returns None."""
        with patch.object(Path, "exists", return_value=False):
            with patch("triage.models.TriageResult.objects") as mock_qs:
                mock_qs.get.side_effect = Exception("not found")
                df = self._load("data_abc123")

        self.assertIsNone(df)


# ---------------------------------------------------------------------------
# explain_selection -- POST /api/dsw/explain-selection/
# ---------------------------------------------------------------------------


@override_settings(**SMOKE_SETTINGS)
class ExplainSelectionTest(TestCase):
    """Behavioral tests for the explain_selection endpoint."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _make_user("dsw-explain@test.com", tier=Tier.PRO)

    def setUp(self):
        self.client.login(username="dsw-explain", password="testpass123")
        self.url = "/api/dsw/explain-selection/"

    def _post(self, data, logged_in=True):
        if not logged_in:
            self.client.logout()
        return self.client.post(
            self.url,
            json.dumps(data),
            content_type="application/json",
        )

    def test_unauth_returns_401(self):
        resp = self._post({"data_id": "x", "indices": [1]}, logged_in=False)
        self.assertEqual(resp.status_code, 401)

    def test_missing_data_id_returns_400(self):
        resp = self._post({"indices": [1, 2, 3]})
        self.assertEqual(resp.status_code, 400)

    def test_missing_indices_returns_400(self):
        resp = self._post({"data_id": "data_test"})
        self.assertEqual(resp.status_code, 400)

    def test_too_many_indices_returns_400(self):
        resp = self._post({"data_id": "data_test", "indices": list(range(101))})
        self.assertEqual(resp.status_code, 400)

    @patch("dsw.views._load_dataset", return_value=None)
    def test_nonexistent_dataset_returns_404(self, _mock_load):
        resp = self._post({"data_id": "data_missing", "indices": [0, 1]})
        self.assertEqual(resp.status_code, 404)

    @patch("dsw.views._load_dataset")
    def test_valid_request_returns_explanation(self, mock_load):
        mock_load.return_value = pd.DataFrame(
            {
                "machine": ["A", "B", "A", "B", "A"],
                "output": [10, 20, 15, 25, 12],
            }
        )

        with patch("agents_api.llm_manager.LLMManager.chat") as mock_chat:  # llm_manager still in agents_api
            mock_chat.return_value = "Selected points are all from machine A."
            resp = self._post(
                {
                    "data_id": "data_test",
                    "indices": [0, 2, 4],
                    "analysis_context": "production review",
                }
            )

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("explanation", body)
        self.assertEqual(body["explanation"], "Selected points are all from machine A.")


# ---------------------------------------------------------------------------
# hypothesis_timeline -- GET /api/dsw/hypothesis-timeline/
# ---------------------------------------------------------------------------


@override_settings(**SMOKE_SETTINGS)
class HypothesisTimelineTest(TestCase):
    """Behavioral tests for the hypothesis_timeline endpoint."""

    @classmethod
    def setUpTestData(cls):
        from core.models import Evidence, Hypothesis, Project

        cls.user = _make_user("dsw-timeline@test.com")
        cls.other_user = _make_user("dsw-timeline-other@test.com")

        cls.project = Project.objects.create(
            user=cls.user,
            title="Timeline Test Project",
        )

        cls.evidence = Evidence.objects.create(
            project=cls.project,
            summary="Test evidence for timeline",
            source_type="analysis",
            p_value=0.03,
        )

        cls.hypothesis = Hypothesis.objects.create(
            project=cls.project,
            statement="Machine A causes higher defects",
            prior_probability=0.5,
            current_probability=0.75,
            status="active",
            probability_history=[
                {
                    "probability": 0.75,
                    "previous": 0.5,
                    "timestamp": "2026-01-15T10:00:00Z",
                    "strength": "moderate",
                    "likelihood_ratio": 3.0,
                    "evidence_id": str(cls.evidence.id),
                },
            ],
        )

        # Hypothesis owned by a different user's project
        cls.other_project = Project.objects.create(
            user=cls.other_user,
            title="Other Project",
        )
        cls.other_hypothesis = Hypothesis.objects.create(
            project=cls.other_project,
            statement="Other user hypothesis",
            prior_probability=0.5,
            current_probability=0.5,
            status="active",
        )

    def setUp(self):
        self.client.login(username="dsw-timeline", password="testpass123")
        self.url = "/api/dsw/hypothesis-timeline/"

    def test_unauth_returns_401(self):
        self.client.logout()
        resp = self.client.get(
            self.url,
            {
                "project_id": str(self.project.id),
                "hypothesis_id": str(self.hypothesis.id),
            },
        )
        self.assertEqual(resp.status_code, 401)

    def test_missing_project_id_returns_400(self):
        resp = self.client.get(self.url, {"hypothesis_id": str(self.hypothesis.id)})
        self.assertEqual(resp.status_code, 400)

    def test_missing_hypothesis_id_returns_400(self):
        resp = self.client.get(self.url, {"project_id": str(self.project.id)})
        self.assertEqual(resp.status_code, 400)

    def test_wrong_owner_returns_404(self):
        resp = self.client.get(
            self.url,
            {
                "project_id": str(self.other_project.id),
                "hypothesis_id": str(self.other_hypothesis.id),
            },
        )
        self.assertEqual(resp.status_code, 404)

    def test_valid_returns_timeline(self):
        resp = self.client.get(
            self.url,
            {
                "project_id": str(self.project.id),
                "hypothesis_id": str(self.hypothesis.id),
            },
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()

        # Top-level structure
        self.assertIn("hypothesis", body)
        self.assertIn("timeline", body)
        self.assertIn("confirmation_threshold", body)
        self.assertIn("rejection_threshold", body)

        # Hypothesis detail
        h = body["hypothesis"]
        self.assertEqual(h["id"], str(self.hypothesis.id))
        self.assertEqual(h["prior"], 0.5)
        self.assertEqual(h["current"], 0.75)
        self.assertEqual(h["status"], "active")

        # Timeline entries
        self.assertEqual(len(body["timeline"]), 1)
        point = body["timeline"][0]
        self.assertEqual(point["probability"], 0.75)
        self.assertEqual(point["previous"], 0.5)
        self.assertIn("evidence", point)
        self.assertEqual(point["evidence"]["summary"], "Test evidence for timeline")
