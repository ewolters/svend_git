"""Functional scenario tests for internal CRUD and management endpoints.

Covers: blog, whitepapers, experiments, automation rules, automation log,
autopilot, roadmap, plan documents, features, changes, incidents,
calibration, and risk registry.
"""

import uuid
from datetime import date
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

User = get_user_model()

SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)


def _make_user(email, tier="free", password="testpass123!", **kwargs):
    username = kwargs.pop("username", email.split("@")[0])
    user = User.objects.create_user(username=username, email=email, password=password, **kwargs)
    user.tier = tier
    user.save(update_fields=["tier"])
    return user


def _make_staff(email="admin@test.com"):
    return _make_user(email, is_staff=True)


# =========================================================================
# Blog Management
# =========================================================================


@SECURE_OFF
class BlogManagementTest(TestCase):
    """Scenario: full blog post lifecycle via internal API."""

    def setUp(self):
        self.staff = _make_staff()
        self.client = APIClient()
        self.client.force_authenticate(user=self.staff)

    def test_non_staff_denied(self):
        """Non-staff users get 403 on all blog endpoints."""
        regular = _make_user("regular@test.com")
        c = APIClient()
        c.force_authenticate(user=regular)
        res = c.get("/api/internal/blog/")
        self.assertEqual(res.status_code, 403)

    def test_unauthenticated_denied(self):
        """Unauthenticated requests get 403."""
        c = APIClient()
        res = c.get("/api/internal/blog/")
        self.assertEqual(res.status_code, 403)

    def test_blog_lifecycle_scenario(self):
        """Create -> list -> get -> publish -> verify published -> delete -> verify gone."""
        # Step 1: Create a blog post
        res = self.client.post(
            "/api/internal/blog/save/",
            {"title": "Test Blog Post", "body": "# Hello\nThis is a test.", "meta_description": "A test post"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        post_id = data["id"]
        self.assertEqual(data["status"], "draft")
        self.assertEqual(data["slug"], "test-blog-post")

        # Step 2: Verify it appears in the list
        res = self.client.get("/api/internal/blog/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["counts"]["total"], 1)
        self.assertEqual(data["counts"]["draft"], 1)
        self.assertEqual(len(data["posts"]), 1)
        self.assertEqual(data["posts"][0]["id"], post_id)

        # Step 3: Get full post
        res = self.client.get(f"/api/internal/blog/{post_id}/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["title"], "Test Blog Post")
        self.assertEqual(data["body"], "# Hello\nThis is a test.")
        self.assertEqual(data["status"], "draft")

        # Step 4: Publish the post
        res = self.client.post(
            f"/api/internal/blog/{post_id}/publish/",
            {"action": "publish"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "published")

        # Step 5: Verify published in list
        res = self.client.get("/api/internal/blog/")
        data = res.json()
        self.assertEqual(data["counts"]["published"], 1)
        self.assertEqual(data["counts"]["draft"], 0)

        # Step 6: Delete the post
        res = self.client.delete(f"/api/internal/blog/{post_id}/delete/")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["deleted"])

        # Step 7: Verify gone from list
        res = self.client.get("/api/internal/blog/")
        self.assertEqual(res.json()["counts"]["total"], 0)

    def test_blog_update_existing(self):
        """Update an existing blog post via save with id."""
        # Create
        res = self.client.post(
            "/api/internal/blog/save/",
            {"title": "Original Title", "body": "Original body"},
            format="json",
        )
        post_id = res.json()["id"]

        # Update
        res = self.client.post(
            "/api/internal/blog/save/",
            {"id": post_id, "title": "Updated Title", "body": "Updated body"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)

        # Verify updated
        res = self.client.get(f"/api/internal/blog/{post_id}/")
        self.assertEqual(res.json()["title"], "Updated Title")
        self.assertEqual(res.json()["body"], "Updated body")

    def test_blog_schedule(self):
        """Schedule a blog post for future publication."""
        res = self.client.post(
            "/api/internal/blog/save/",
            {"title": "Scheduled Post", "body": "Content"},
            format="json",
        )
        post_id = res.json()["id"]

        future = "2030-06-15T12:00:00Z"
        res = self.client.post(
            f"/api/internal/blog/{post_id}/publish/",
            {"action": "schedule", "scheduled_at": future},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "scheduled")
        self.assertIsNotNone(res.json()["scheduled_at"])

    def test_blog_unpublish(self):
        """Unpublish a published blog post (revert to draft)."""
        res = self.client.post(
            "/api/internal/blog/save/",
            {"title": "To Unpublish", "body": "Content"},
            format="json",
        )
        post_id = res.json()["id"]

        self.client.post(f"/api/internal/blog/{post_id}/publish/", {"action": "publish"}, format="json")

        res = self.client.post(
            f"/api/internal/blog/{post_id}/publish/",
            {"action": "unpublish"},
            format="json",
        )
        self.assertEqual(res.json()["status"], "draft")

    def test_blog_get_nonexistent(self):
        """Getting a nonexistent blog post returns 404."""
        fake_id = uuid.uuid4()
        res = self.client.get(f"/api/internal/blog/{fake_id}/")
        self.assertEqual(res.status_code, 404)

    def test_blog_analytics(self):
        """Blog analytics endpoint returns expected structure."""
        res = self.client.get("/api/internal/blog/analytics/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("daily_views", data)
        self.assertIn("top_posts", data)
        self.assertIn("referrers", data)
        self.assertIn("totals", data)
        self.assertIn("views", data["totals"])
        self.assertIn("unique_visitors", data["totals"])


# =========================================================================
# Whitepaper Management
# =========================================================================


@SECURE_OFF
class WhitepaperManagementTest(TestCase):
    """Scenario: full whitepaper lifecycle via internal API."""

    def setUp(self):
        self.staff = _make_staff()
        self.client = APIClient()
        self.client.force_authenticate(user=self.staff)

    def test_non_staff_denied(self):
        regular = _make_user("regular2@test.com")
        c = APIClient()
        c.force_authenticate(user=regular)
        res = c.get("/api/internal/whitepapers/")
        self.assertEqual(res.status_code, 403)

    def test_whitepaper_lifecycle_scenario(self):
        """Create -> list -> get -> publish -> delete -> verify gone."""
        # Create
        res = self.client.post(
            "/api/internal/whitepapers/save/",
            {
                "title": "SPC Best Practices",
                "body": "# SPC\nLong form content here.",
                "description": "A whitepaper about SPC.",
                "topic": "SPC",
                "meta_description": "SPC best practices guide",
                "gated": True,
            },
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        paper_id = data["id"]
        self.assertEqual(data["status"], "draft")

        # List
        res = self.client.get("/api/internal/whitepapers/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["counts"]["total"], 1)
        self.assertEqual(data["counts"]["draft"], 1)

        # Get
        res = self.client.get(f"/api/internal/whitepapers/{paper_id}/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["title"], "SPC Best Practices")
        self.assertTrue(res.json()["gated"])

        # Publish
        res = self.client.post(
            f"/api/internal/whitepapers/{paper_id}/publish/",
            {"action": "publish"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "published")

        # Unpublish
        res = self.client.post(
            f"/api/internal/whitepapers/{paper_id}/publish/",
            {"action": "unpublish"},
            format="json",
        )
        self.assertEqual(res.json()["status"], "draft")

        # Delete
        res = self.client.delete(f"/api/internal/whitepapers/{paper_id}/delete/")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["ok"])

        # Verify gone
        res = self.client.get("/api/internal/whitepapers/")
        self.assertEqual(res.json()["counts"]["total"], 0)


# =========================================================================
# Experiment Management
# =========================================================================


@SECURE_OFF
class ExperimentManagementTest(TestCase):
    """Scenario: create experiment, list, conclude with winner."""

    def setUp(self):
        self.staff = _make_staff()
        self.client = APIClient()
        self.client.force_authenticate(user=self.staff)

    def test_non_staff_denied(self):
        regular = _make_user("regular3@test.com")
        c = APIClient()
        c.force_authenticate(user=regular)
        res = c.get("/api/internal/experiments/")
        self.assertEqual(res.status_code, 403)

    def test_experiment_create_and_list(self):
        """Create an experiment via POST, then list it via GET."""
        res = self.client.post(
            "/api/internal/experiments/",
            {
                "name": "Onboarding Flow Test",
                "hypothesis": "Simplified onboarding increases conversion",
                "experiment_type": "onboarding_flow",
                "metric": "conversion",
                "variants": [{"name": "A", "weight": 50}, {"name": "B", "weight": 50}],
                "status": "running",
                "target": "new_users",
                "min_sample_size": 200,
            },
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["id"])  # has UUID
        self.assertEqual(data["status"], "running")

        # List
        res = self.client.get("/api/internal/experiments/")
        self.assertEqual(res.status_code, 200)
        experiments = res.json()["experiments"]
        self.assertEqual(len(experiments), 1)
        self.assertEqual(experiments[0]["name"], "Onboarding Flow Test")
        self.assertEqual(experiments[0]["status"], "running")

    def test_experiment_conclude(self):
        """Conclude an experiment with a chosen winner."""
        res = self.client.post(
            "/api/internal/experiments/",
            {
                "name": "Email Subject Test",
                "hypothesis": "Shorter subjects get more opens",
                "experiment_type": "email_subject",
                "metric": "engagement",
                "variants": [{"name": "short", "weight": 50}, {"name": "long", "weight": 50}],
                "status": "running",
            },
            format="json",
        )
        exp_id = res.json()["id"]

        res = self.client.post(
            f"/api/internal/experiments/{exp_id}/conclude/",
            {"winner": "short"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "concluded")
        self.assertEqual(res.json()["winner"], "short")

    def test_experiment_conclude_nonexistent(self):
        """Concluding a nonexistent experiment returns 404."""
        fake_id = uuid.uuid4()
        res = self.client.post(
            f"/api/internal/experiments/{fake_id}/conclude/",
            {"winner": "A"},
            format="json",
        )
        self.assertEqual(res.status_code, 404)


# =========================================================================
# Automation Rules
# =========================================================================


@SECURE_OFF
class AutomationRulesTest(TestCase):
    """Scenario: create rule, list, toggle on/off."""

    def setUp(self):
        self.staff = _make_staff()
        self.client = APIClient()
        self.client.force_authenticate(user=self.staff)

    def test_non_staff_denied(self):
        regular = _make_user("regular4@test.com")
        c = APIClient()
        c.force_authenticate(user=regular)
        res = c.get("/api/internal/automation/rules/")
        self.assertEqual(res.status_code, 403)

    def test_create_list_toggle_scenario(self):
        """Create a rule -> list -> toggle off -> verify inactive -> toggle on."""
        # Create
        res = self.client.post(
            "/api/internal/automation/rules/",
            {
                "name": "Inactive Nudge",
                "description": "Send email after 7 days inactive",
                "trigger": "inactive_days",
                "trigger_config": {"days": 7},
                "action": "send_email",
                "action_config": {"template": "inactive_nudge"},
                "cooldown_hours": 72,
            },
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        rule_id = res.json()["id"]

        # List
        res = self.client.get("/api/internal/automation/rules/")
        self.assertEqual(res.status_code, 200)
        rules = res.json()["rules"]
        self.assertEqual(len(rules), 1)
        self.assertTrue(rules[0]["is_active"])

        # Toggle off
        res = self.client.post(f"/api/internal/automation/rules/{rule_id}/toggle/", format="json")
        self.assertEqual(res.status_code, 200)
        self.assertFalse(res.json()["is_active"])

        # Toggle back on
        res = self.client.post(f"/api/internal/automation/rules/{rule_id}/toggle/", format="json")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["is_active"])


# =========================================================================
# Automation Log
# =========================================================================


@SECURE_OFF
class AutomationLogTest(TestCase):
    """Test automation log listing."""

    def setUp(self):
        self.staff = _make_staff()
        self.client = APIClient()
        self.client.force_authenticate(user=self.staff)

    def test_non_staff_denied(self):
        regular = _make_user("regular5@test.com")
        c = APIClient()
        c.force_authenticate(user=regular)
        res = c.get("/api/internal/automation/log/")
        self.assertEqual(res.status_code, 403)

    def test_automation_log_empty(self):
        """Empty log returns empty list."""
        res = self.client.get("/api/internal/automation/log/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["log"], [])

    def test_automation_log_with_entries(self):
        """Log entries appear when created."""
        from api.models import AutomationLog, AutomationRule

        rule = AutomationRule.objects.create(
            name="Test Rule for Log",
            trigger="inactive_days",
            trigger_config={"days": 7},
            action="send_email",
            action_config={},
        )
        AutomationLog.objects.create(
            rule=rule,
            user=self.staff,
            action_taken="Sent inactive nudge email",
            result="success",
        )

        res = self.client.get("/api/internal/automation/log/")
        self.assertEqual(res.status_code, 200)
        log = res.json()["log"]
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["rule"], "Test Rule for Log")
        self.assertEqual(log[0]["result"], "success")

    def test_automation_log_filter_by_rule(self):
        """Log can be filtered by rule_id."""
        from api.models import AutomationLog, AutomationRule

        rule1 = AutomationRule.objects.create(
            name="Rule One Filter",
            trigger="inactive_days",
            trigger_config={},
            action="send_email",
            action_config={},
        )
        rule2 = AutomationRule.objects.create(
            name="Rule Two Filter",
            trigger="churn_signal",
            trigger_config={},
            action="internal_alert",
            action_config={},
        )
        AutomationLog.objects.create(rule=rule1, user=self.staff, action_taken="Action 1", result="success")
        AutomationLog.objects.create(rule=rule2, user=self.staff, action_taken="Action 2", result="success")

        res = self.client.get(f"/api/internal/automation/log/?rule_id={rule1.id}")
        self.assertEqual(len(res.json()["log"]), 1)
        self.assertEqual(res.json()["log"][0]["rule"], "Rule One Filter")


# =========================================================================
# Autopilot
# =========================================================================


@SECURE_OFF
class AutopilotTest(TestCase):
    """Test autopilot reports listing and approval."""

    def setUp(self):
        self.staff = _make_staff()
        self.client = APIClient()
        self.client.force_authenticate(user=self.staff)

    def test_non_staff_denied(self):
        regular = _make_user("regular6@test.com")
        c = APIClient()
        c.force_authenticate(user=regular)
        res = c.get("/api/internal/autopilot/")
        self.assertEqual(res.status_code, 403)

    def test_autopilot_list_empty(self):
        """Empty autopilot reports list."""
        res = self.client.get("/api/internal/autopilot/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["reports"], [])

    def test_autopilot_list_with_report(self):
        """Reports appear in list."""
        from api.models import AutopilotReport

        AutopilotReport.objects.create(
            data_snapshot={"users": {"total": 100}},
            insights=["Signup rate increasing"],
            recommendations=[{"type": "blog", "title": "Write about SPC", "config": {"title": "SPC Guide"}}],
            alerts=[],
            status="pending_review",
        )

        res = self.client.get("/api/internal/autopilot/")
        self.assertEqual(res.status_code, 200)
        reports = res.json()["reports"]
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]["status"], "pending_review")

    def test_autopilot_approve_blog_recommendation(self):
        """Approving a blog recommendation creates a draft blog post."""
        from api.models import AutopilotReport, BlogPost

        report = AutopilotReport.objects.create(
            data_snapshot={},
            insights=[],
            recommendations=[
                {
                    "type": "blog",
                    "title": "Write about SPC",
                    "config": {"title": "SPC Control Charts Guide", "target_keyword": "SPC"},
                }
            ],
            alerts=[],
        )

        res = self.client.post(
            f"/api/internal/autopilot/{report.id}/approve/",
            {"index": 0},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["action"], "blog_draft_created")

        # Verify a BlogPost was created
        self.assertTrue(BlogPost.objects.filter(title="SPC Control Charts Guide").exists())

    def test_autopilot_approve_invalid_index(self):
        """Approving an out-of-range recommendation returns 400."""
        from api.models import AutopilotReport

        report = AutopilotReport.objects.create(
            data_snapshot={},
            insights=[],
            recommendations=[{"type": "blog", "title": "X", "config": {"title": "Y"}}],
            alerts=[],
        )

        res = self.client.post(
            f"/api/internal/autopilot/{report.id}/approve/",
            {"index": 5},
            format="json",
        )
        self.assertEqual(res.status_code, 400)

    @patch("syn.sched.scheduler.schedule_task")
    def test_autopilot_run(self, mock_schedule):
        """Trigger a manual autopilot run."""
        res = self.client.post("/api/internal/autopilot/run/", format="json")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "scheduled")
        mock_schedule.assert_called_once()


# =========================================================================
# Roadmap Management
# =========================================================================


@SECURE_OFF
class RoadmapManagementTest(TestCase):
    """Scenario: create, get, list, update, delete roadmap items."""

    def setUp(self):
        self.staff = _make_staff()
        self.client = APIClient()
        self.client.force_authenticate(user=self.staff)

    def test_non_staff_denied(self):
        regular = _make_user("regular7@test.com")
        c = APIClient()
        c.force_authenticate(user=regular)
        res = c.get("/api/internal/roadmap/")
        self.assertEqual(res.status_code, 403)

    def test_roadmap_lifecycle_scenario(self):
        """Create -> list -> get -> update status to shipped -> delete."""
        # Create
        res = self.client.post(
            "/api/internal/roadmap/save/",
            {
                "title": "DOE Module v2",
                "description": "Enhanced DOE with CCD support",
                "area": "dsw",
                "quarter": "Q2-2026",
                "status": "planned",
                "tier": "professional",
                "is_public": True,
            },
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        item_id = res.json()["id"]

        # List
        res = self.client.get("/api/internal/roadmap/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["items"]), 1)
        self.assertIn("Q2-2026", data["quarters"])
        self.assertEqual(data["stats"]["total"], 1)

        # Get
        res = self.client.get(f"/api/internal/roadmap/{item_id}/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["title"], "DOE Module v2")
        self.assertEqual(res.json()["area"], "dsw")

        # Update to shipped
        res = self.client.post(
            "/api/internal/roadmap/save/",
            {"id": item_id, "status": "shipped"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)

        # Verify shipped_at was set
        res = self.client.get(f"/api/internal/roadmap/{item_id}/")
        self.assertEqual(res.json()["status"], "shipped")
        self.assertIsNotNone(res.json()["shipped_at"])

        # Delete
        res = self.client.post(f"/api/internal/roadmap/{item_id}/delete/", format="json")
        self.assertEqual(res.status_code, 200)

        # Verify gone
        res = self.client.get("/api/internal/roadmap/")
        self.assertEqual(len(res.json()["items"]), 0)

    def test_roadmap_invalid_quarter_format(self):
        """Invalid quarter format is rejected."""
        res = self.client.post(
            "/api/internal/roadmap/save/",
            {"title": "Bad Quarter", "quarter": "2026-Q2", "area": "dsw"},
            format="json",
        )
        self.assertEqual(res.status_code, 400)


# =========================================================================
# Plan Documents
# =========================================================================


@SECURE_OFF
class PlanDocumentTest(TestCase):
    """Scenario: create, get, list, update, delete plan documents."""

    def setUp(self):
        self.staff = _make_staff()
        self.client = APIClient()
        self.client.force_authenticate(user=self.staff)

    def test_non_staff_denied(self):
        regular = _make_user("regular8@test.com")
        c = APIClient()
        c.force_authenticate(user=regular)
        res = c.get("/api/internal/plans/")
        self.assertEqual(res.status_code, 403)

    def test_plan_lifecycle_scenario(self):
        """Create -> list -> get -> update -> delete."""
        # Create
        res = self.client.post(
            "/api/internal/plans/save/",
            {
                "title": "Migration Plan: PostgreSQL 16",
                "body": "## Overview\nMigrate to PG 16 for performance.",
                "status": "draft",
                "category": "plan",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        plan_id = res.json()["id"]

        # List
        res = self.client.get("/api/internal/plans/")
        self.assertEqual(res.status_code, 200)
        plans = res.json()["plans"]
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0]["category"], "plan")

        # Get
        res = self.client.get(f"/api/internal/plans/{plan_id}/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["title"], "Migration Plan: PostgreSQL 16")
        self.assertIn("PG 16", res.json()["body"])

        # Update to active
        res = self.client.post(
            "/api/internal/plans/save/",
            {"id": plan_id, "status": "active"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)

        res = self.client.get(f"/api/internal/plans/{plan_id}/")
        self.assertEqual(res.json()["status"], "active")

        # Delete
        res = self.client.post(f"/api/internal/plans/{plan_id}/delete/", format="json")
        self.assertEqual(res.status_code, 200)

        # Verify gone
        res = self.client.get("/api/internal/plans/")
        self.assertEqual(len(res.json()["plans"]), 0)

    def test_plan_filter_by_category(self):
        """Filter plans by category."""
        self.client.post(
            "/api/internal/plans/save/",
            {"title": "Spec Document AAAAAA", "category": "spec"},
            format="json",
        )
        self.client.post(
            "/api/internal/plans/save/",
            {"title": "RFC Document AAAAAA", "category": "rfc"},
            format="json",
        )

        res = self.client.get("/api/internal/plans/?category=spec")
        self.assertEqual(len(res.json()["plans"]), 1)
        self.assertEqual(res.json()["plans"][0]["category"], "spec")


# =========================================================================
# Feature Planning
# =========================================================================


@SECURE_OFF
class FeaturePlanningTest(TestCase):
    """Scenario: list features, get detail, update status, save, add note."""

    def setUp(self):
        self.staff = _make_staff()
        self.client = APIClient()
        self.client.force_authenticate(user=self.staff)
        # Create an initiative and feature directly
        from api.models import Feature, Initiative

        self.initiative = Initiative.objects.create(
            title="Core Platform Build",
            status="active",
            target_quarter="Q1-2026",
        )
        self.feature = Feature.objects.create(
            title="CAPA Report Generator",
            description="Generate CAPA reports from RCA sessions.",
            initiative=self.initiative,
            status="backlog",
            priority="high",
        )

    def test_non_staff_denied(self):
        regular = _make_user("regular9@test.com")
        c = APIClient()
        c.force_authenticate(user=regular)
        res = c.get("/api/internal/features/")
        self.assertEqual(res.status_code, 403)

    def test_features_list(self):
        """List features returns initiatives and features."""
        res = self.client.get("/api/internal/features/?all=1")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("initiatives", data)
        self.assertIn("features", data)
        self.assertIn("stats", data)
        self.assertGreaterEqual(len(data["features"]), 1)

    def test_feature_get_by_uuid(self):
        """Get a feature by UUID."""
        res = self.client.get(f"/api/internal/features/{self.feature.id}/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["title"], "CAPA Report Generator")
        self.assertEqual(data["status"], "backlog")
        self.assertEqual(data["priority"], "high")
        self.assertIn("tasks", data)

    def test_feature_update_status(self):
        """Update feature status to in_progress."""
        res = self.client.post(
            f"/api/internal/features/{self.feature.id}/status/",
            {"status": "in_progress"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["old_status"], "backlog")
        self.assertEqual(res.json()["new_status"], "in_progress")

        # Verify started_at was set
        self.feature.refresh_from_db()
        self.assertIsNotNone(self.feature.started_at)

    def test_feature_update_invalid_status(self):
        """Invalid status is rejected."""
        res = self.client.post(
            f"/api/internal/features/{self.feature.id}/status/",
            {"status": "nonexistent"},
            format="json",
        )
        self.assertEqual(res.status_code, 400)

    def test_feature_save_description(self):
        """Update feature fields via save endpoint."""
        res = self.client.post(
            f"/api/internal/features/{self.feature.id}/save/",
            {"description": "Updated description.", "acceptance_criteria": "Must pass all tests."},
            format="json",
        )
        self.assertEqual(res.status_code, 200)

        self.feature.refresh_from_db()
        self.assertEqual(self.feature.description, "Updated description.")
        self.assertEqual(self.feature.acceptance_criteria, "Must pass all tests.")

    def test_feature_add_note(self):
        """Add a note to a feature."""
        res = self.client.post(
            f"/api/internal/features/{self.feature.id}/note/",
            {"text": "Needs architecture review", "user": True},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        entry = res.json()["entry"]
        self.assertIn("$ Needs architecture review", entry)

        self.feature.refresh_from_db()
        self.assertIn("$ Needs architecture review", self.feature.notes)

    def test_feature_add_empty_note_rejected(self):
        """Empty note text is rejected."""
        res = self.client.post(
            f"/api/internal/features/{self.feature.id}/note/",
            {"text": "", "user": False},
            format="json",
        )
        self.assertEqual(res.status_code, 400)


# =========================================================================
# Change Management
# =========================================================================


@SECURE_OFF
class ChangeManagementEndpointTest(TestCase):
    """Scenario: create change request, list, get detail, transition."""

    def setUp(self):
        self.staff = _make_staff()
        self.client = APIClient()
        self.client.force_authenticate(user=self.staff)

    def test_non_staff_denied(self):
        regular = _make_user("regular10@test.com")
        c = APIClient()
        c.force_authenticate(user=regular)
        res = c.get("/api/internal/changes/")
        self.assertEqual(res.status_code, 403)

    def test_change_lifecycle_scenario(self):
        """Create -> list -> get detail -> transition to submitted."""
        # Create
        res = self.client.post(
            "/api/internal/changes/create/",
            {
                "title": "Add CAPA report generation feature",
                "description": "Implement CAPA report generation from existing RCA sessions",
                "change_type": "feature",
                "risk_level": "medium",
                "priority": "high",
                "justification": "Customer request for compliance reporting",
                "affected_files": ["agents_api/rca_views.py", "agents_api/models.py"],
            },
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertTrue(data["ok"])
        change_id = data["id"]

        # List
        res = self.client.get("/api/internal/changes/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertGreaterEqual(data["total"], 1)
        self.assertGreaterEqual(len(data["changes"]), 1)
        self.assertIn("stats", data)

        # Get detail
        res = self.client.get(f"/api/internal/changes/{change_id}/")
        self.assertEqual(res.status_code, 200)
        detail = res.json()
        self.assertEqual(detail["change"]["title"], "Add CAPA report generation feature")
        self.assertEqual(detail["change"]["status"], "draft")
        self.assertGreaterEqual(len(detail["logs"]), 1)  # Initial log entry

        # Transition to submitted
        res = self.client.post(
            f"/api/internal/changes/{change_id}/transition/",
            {"action": "submitted", "message": "Ready for review"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["ok"])
        self.assertEqual(res.json()["status"], "submitted")

    def test_change_list_filter_by_type(self):
        """Filter changes by type."""
        self.client.post(
            "/api/internal/changes/create/",
            {
                "title": "Documentation update for testing standards",
                "description": "Update TST-001 with new test patterns and conventions",
                "change_type": "documentation",
            },
            format="json",
        )

        res = self.client.get("/api/internal/changes/?type=documentation")
        self.assertEqual(res.status_code, 200)
        changes = res.json()["changes"]
        for c in changes:
            self.assertEqual(c["change_type"], "documentation")

    def test_change_detail_nonexistent(self):
        """Getting a nonexistent change returns 404."""
        fake_id = uuid.uuid4()
        res = self.client.get(f"/api/internal/changes/{fake_id}/")
        self.assertEqual(res.status_code, 404)

    def test_change_transition_nonexistent(self):
        """Transitioning a nonexistent change returns 404."""
        fake_id = uuid.uuid4()
        res = self.client.post(
            f"/api/internal/changes/{fake_id}/transition/",
            {"action": "submitted"},
            format="json",
        )
        self.assertEqual(res.status_code, 404)


# =========================================================================
# Incident Management
# =========================================================================


@SECURE_OFF
class IncidentManagementEndpointTest(TestCase):
    """Scenario: create incident, list, get detail, transition through lifecycle."""

    def setUp(self):
        self.staff = _make_staff()
        self.client = APIClient()
        self.client.force_authenticate(user=self.staff)

    def test_non_staff_denied(self):
        regular = _make_user("regular11@test.com")
        c = APIClient()
        c.force_authenticate(user=regular)
        res = c.get("/api/internal/incidents/")
        self.assertEqual(res.status_code, 403)

    def test_incident_lifecycle_scenario(self):
        """Create -> list -> detail -> acknowledge -> investigate -> resolve."""
        # Create
        res = self.client.post(
            "/api/internal/incidents/create/",
            {
                "title": "Database connection pool exhausted",
                "description": "PG connections maxed out causing 503 errors for 15 minutes",
                "severity": "high",
                "category": "degradation",
                "assigned_to": "admin@test.com",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertTrue(data["ok"])
        inc_id = data["id"]

        # List
        res = self.client.get("/api/internal/incidents/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertGreaterEqual(data["total"], 1)
        self.assertIn("stats", data)

        # Detail
        res = self.client.get(f"/api/internal/incidents/{inc_id}/")
        self.assertEqual(res.status_code, 200)
        detail = res.json()
        self.assertEqual(detail["incident"]["title"], "Database connection pool exhausted")
        self.assertEqual(detail["incident"]["status"], "detected")
        self.assertGreaterEqual(len(detail["logs"]), 1)

        # Acknowledge
        res = self.client.post(
            f"/api/internal/incidents/{inc_id}/transition/",
            {"action": "acknowledged", "message": "Looking into it"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "acknowledged")

        # Investigate
        res = self.client.post(
            f"/api/internal/incidents/{inc_id}/transition/",
            {"action": "investigating", "message": "Checking PG logs"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "investigating")

        # Resolve (requires resolution_summary)
        res = self.client.post(
            f"/api/internal/incidents/{inc_id}/transition/",
            {
                "action": "resolved",
                "message": "Fixed",
                "resolution_summary": "Increased connection pool from 20 to 50 and added pgbouncer.",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "resolved")

    def test_incident_resolve_without_summary_rejected(self):
        """Resolving without resolution_summary returns 400."""
        res = self.client.post(
            "/api/internal/incidents/create/",
            {
                "title": "Minor logging failure on SPC endpoint",
                "description": "SPC endpoint logging fails silently on some requests",
                "severity": "low",
                "category": "other",
            },
            format="json",
        )
        inc_id = res.json()["id"]

        res = self.client.post(
            f"/api/internal/incidents/{inc_id}/transition/",
            {"action": "resolved", "message": "Fixed"},
            format="json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("resolution_summary", res.json()["error"]["message"])

    def test_incident_create_short_title_rejected(self):
        """Incident with title < 5 chars is rejected."""
        res = self.client.post(
            "/api/internal/incidents/create/",
            {"title": "Bad", "description": "Short title incident", "severity": "low"},
            format="json",
        )
        self.assertEqual(res.status_code, 400)

    def test_incident_detail_nonexistent(self):
        """Getting a nonexistent incident returns 404."""
        fake_id = uuid.uuid4()
        res = self.client.get(f"/api/internal/incidents/{fake_id}/")
        self.assertEqual(res.status_code, 404)


# =========================================================================
# Calibration
# =========================================================================


@SECURE_OFF
class CalibrationEndpointTest(TestCase):
    """Test calibration data retrieval and run action."""

    def setUp(self):
        self.staff = _make_staff()
        self.client = APIClient()
        self.client.force_authenticate(user=self.staff)

    def test_non_staff_denied(self):
        regular = _make_user("regular12@test.com")
        c = APIClient()
        c.force_authenticate(user=regular)
        res = c.get("/api/internal/calibration/")
        self.assertEqual(res.status_code, 403)

    def test_calibration_get_empty(self):
        """Calibration endpoint with no reports returns empty structure."""
        res = self.client.get("/api/internal/calibration/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["reports"], [])
        self.assertEqual(data["trend"], [])

    def test_calibration_get_with_report(self):
        """Calibration data includes reports and stats."""
        from syn.audit.models import CalibrationReport

        CalibrationReport.objects.create(
            date=date(2026, 3, 1),
            overall_coverage=42.5,
            tier1_coverage=80.0,
            tier2_coverage=55.0,
            tier3_coverage=30.0,
            tier4_coverage=20.0,
            calibration_pass_rate=95.0,
            calibration_cases_run=100,
            calibration_cases_passed=95,
            golden_file_count=15,
            complexity_violations=2,
            ratchet_baseline=40.0,
        )

        res = self.client.get("/api/internal/calibration/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["reports"]), 1)
        self.assertEqual(data["stats"]["overall_coverage"], 42.5)
        self.assertEqual(data["stats"]["ratchet_baseline"], 40.0)

    @patch("django.core.management.call_command")
    def test_calibration_run_measure_coverage(self, mock_cmd):
        """Run measure_coverage action."""
        res = self.client.post(
            "/api/internal/calibration/run/",
            {"action": "measure_coverage"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["ok"])
        mock_cmd.assert_called_once()

    def test_calibration_run_unknown_action(self):
        """Unknown calibration action returns 400."""
        res = self.client.post(
            "/api/internal/calibration/run/",
            {"action": "unknown_action"},
            format="json",
        )
        self.assertEqual(res.status_code, 400)


# =========================================================================
# Risk Registry
# =========================================================================


@SECURE_OFF
class RiskRegistryEndpointTest(TestCase):
    """Test risk registry listing and filtering."""

    def setUp(self):
        self.staff = _make_staff()
        self.client = APIClient()
        self.client.force_authenticate(user=self.staff)

    def test_non_staff_denied(self):
        regular = _make_user("regular13@test.com")
        c = APIClient()
        c.force_authenticate(user=regular)
        res = c.get("/api/internal/risks/")
        self.assertEqual(res.status_code, 403)

    def test_risk_registry_empty(self):
        """Empty risk registry returns empty entries and stats."""
        res = self.client.get("/api/internal/risks/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["entries"], [])
        self.assertEqual(data["stats"]["total"], 0)

    def test_risk_registry_with_entries(self):
        """Risk entries appear with computed RPN and risk_level."""
        from syn.audit.models import RiskEntry

        RiskEntry.objects.create(
            title="SQL injection in legacy endpoint",
            description="Parameterized queries not used in one legacy path",
            category="security",
            likelihood=3,
            severity=5,
            detectability=2,
            status="identified",
            owner="security_team",
        )
        RiskEntry.objects.create(
            title="Backup script timeout",
            description="Nightly backup occasionally times out on large tables",
            category="availability",
            likelihood=2,
            severity=3,
            detectability=1,
            status="mitigating",
            mitigation_plan="Increase timeout and add streaming backup",
            owner="ops_team",
        )

        res = self.client.get("/api/internal/risks/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["stats"]["total"], 2)
        self.assertEqual(data["stats"]["open"], 2)

        # Check RPN is computed correctly (L * S * D)
        sql_entry = next(e for e in data["entries"] if "SQL" in e["title"])
        self.assertEqual(sql_entry["rpn"], 3 * 5 * 2)  # 30
        self.assertEqual(sql_entry["risk_level"], "medium")

    def test_risk_registry_filter_by_status(self):
        """Filter risks by status."""
        from syn.audit.models import RiskEntry

        RiskEntry.objects.create(
            title="Open risk filter test AAA",
            description="Test risk for filtering by status",
            category="operational",
            likelihood=1,
            severity=1,
            detectability=1,
            status="identified",
            owner="test",
        )
        RiskEntry.objects.create(
            title="Closed risk filter test AAA",
            description="Test closed risk for filtering by status",
            category="operational",
            likelihood=1,
            severity=1,
            detectability=1,
            status="closed",
            owner="test",
        )

        res = self.client.get("/api/internal/risks/?status=identified")
        self.assertEqual(res.status_code, 200)
        entries = res.json()["entries"]
        self.assertTrue(all(e["status"] == "identified" for e in entries))

    def test_risk_registry_filter_by_category(self):
        """Filter risks by category."""
        from syn.audit.models import RiskEntry

        RiskEntry.objects.create(
            title="Security risk category test BB",
            description="Test risk for category filtering",
            category="security",
            likelihood=2,
            severity=2,
            detectability=2,
            status="identified",
            owner="test",
        )
        RiskEntry.objects.create(
            title="Availability risk category test BB",
            description="Test risk for category filtering",
            category="availability",
            likelihood=2,
            severity=2,
            detectability=2,
            status="identified",
            owner="test",
        )

        res = self.client.get("/api/internal/risks/?category=security")
        self.assertEqual(res.status_code, 200)
        entries = res.json()["entries"]
        self.assertTrue(all(e["category"] == "security" for e in entries))

    def test_risk_high_rpn_count(self):
        """Stats correctly count high-RPN risks (RPN > 60)."""
        from syn.audit.models import RiskEntry

        # High RPN: 4 * 5 * 4 = 80
        RiskEntry.objects.create(
            title="High RPN risk stat test",
            description="This has a high risk priority number",
            category="integrity",
            likelihood=4,
            severity=5,
            detectability=4,
            status="identified",
            owner="test",
        )
        # Low RPN: 1 * 1 * 1 = 1
        RiskEntry.objects.create(
            title="Low RPN risk stat test",
            description="This has a low risk priority number",
            category="operational",
            likelihood=1,
            severity=1,
            detectability=1,
            status="mitigated",
            owner="test",
        )

        res = self.client.get("/api/internal/risks/")
        data = res.json()
        self.assertEqual(data["stats"]["high_rpn"], 1)
        self.assertEqual(data["stats"]["mitigated"], 1)
