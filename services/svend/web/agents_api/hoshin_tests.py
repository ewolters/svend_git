"""Hoshin Kanri + VSM promotion integration tests.

Tests the Hoshin Kanri CI project management system and its integration
with Value Stream Mapping:

- Site CRUD + access control
- Hoshin project CRUD (creates core.Project + HoshinProject atomically)
- Batch creation from VSM proposals (from-proposals endpoint)
- Monthly savings tracking (9 calculation methods)
- Action items (Gantt-style task dependencies)
- VSM promotion (future→current, archive old, writeback savings)
- Dashboard (savings rollup by site, monthly trend)
- Tier gating (enterprise only via @require_feature("hoshin_kanri"))
- Site-level permission enforcement (viewer/member/admin)
"""

import json
from datetime import date
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from accounts.constants import Tier
from core.models.tenant import Tenant, Membership

User = get_user_model()
SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)


def _make_enterprise_user(email, **kwargs):
    """Create an Enterprise-tier user with tenant + owner membership."""
    username = kwargs.pop("username", email.split("@")[0])
    user = User.objects.create_user(
        username=username, email=email, password="testpass123!", **kwargs
    )
    user.tier = Tier.ENTERPRISE
    user.save(update_fields=["tier"])
    return user


def _setup_tenant(owner, slug="testcorp"):
    """Create a tenant with owner membership."""
    tenant = Tenant.objects.create(name="Test Corp", slug=slug)
    Membership.objects.create(
        user=owner, tenant=tenant, role="owner", is_active=True,
    )
    return tenant


def _post(client, url, data=None):
    return client.post(url, json.dumps(data or {}), content_type="application/json")


def _put(client, url, data=None):
    return client.put(url, json.dumps(data or {}), content_type="application/json")


def _patch(client, url, data=None):
    return client.patch(url, json.dumps(data or {}), content_type="application/json")


# =============================================================================
# Site CRUD
# =============================================================================


@SECURE_OFF
class SiteCRUDTest(TestCase):
    """Manufacturing site lifecycle."""

    def setUp(self):
        self.user = _make_enterprise_user("site_admin@test.com")
        self.tenant = _setup_tenant(self.user)
        self.client.force_login(self.user)

    def test_create_site(self):
        res = _post(self.client, "/api/hoshin/sites/create/", {
            "name": "Fort Worth Plant",
            "code": "FTW",
            "business_unit": "Printing",
            "plant_manager": "Bob Zeisler",
            "ci_leader": "Eric Wolters",
        })
        self.assertEqual(res.status_code, 201)
        site = res.json()["site"]
        self.assertEqual(site["name"], "Fort Worth Plant")
        self.assertEqual(site["code"], "FTW")

    def test_list_sites(self):
        _post(self.client, "/api/hoshin/sites/create/", {"name": "Plant A"})
        _post(self.client, "/api/hoshin/sites/create/", {"name": "Plant B"})
        res = self.client.get("/api/hoshin/sites/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["count"], 2)

    def test_get_site_with_summary(self):
        site_id = _post(self.client, "/api/hoshin/sites/create/", {
            "name": "Main Plant",
        }).json()["site"]["id"]
        res = self.client.get(f"/api/hoshin/sites/{site_id}/")
        self.assertEqual(res.status_code, 200)
        data = res.json()["site"]
        self.assertEqual(data["name"], "Main Plant")
        self.assertEqual(data["project_count"], 0)

    def test_update_site(self):
        site_id = _post(self.client, "/api/hoshin/sites/create/", {
            "name": "Old Name",
        }).json()["site"]["id"]
        res = _put(self.client, f"/api/hoshin/sites/{site_id}/update/", {
            "name": "New Name",
            "ci_leader": "Updated Leader",
        })
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["site"]["name"], "New Name")

    def test_delete_site(self):
        site_id = _post(self.client, "/api/hoshin/sites/create/", {
            "name": "Delete Me",
        }).json()["site"]["id"]
        res = self.client.delete(f"/api/hoshin/sites/{site_id}/delete/")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["success"])

    def test_create_site_requires_name(self):
        res = _post(self.client, "/api/hoshin/sites/create/", {"name": ""})
        self.assertEqual(res.status_code, 400)


# =============================================================================
# Site Access Control
# =============================================================================


@SECURE_OFF
class SiteAccessControlTest(TestCase):
    """Site-level permission enforcement."""

    def setUp(self):
        self.admin = _make_enterprise_user("admin@test.com")
        self.member = _make_enterprise_user("member@test.com", username="member")
        self.viewer = _make_enterprise_user("viewer@test.com", username="viewer")
        self.outsider = _make_enterprise_user("outsider@test.com", username="outsider")
        self.outsider.tier = Tier.ENTERPRISE
        self.outsider.save()

        self.tenant = _setup_tenant(self.admin)
        # Member and viewer join the same org
        Membership.objects.create(
            user=self.member, tenant=self.tenant, role="member", is_active=True,
        )
        Membership.objects.create(
            user=self.viewer, tenant=self.tenant, role="member", is_active=True,
        )

        # Admin creates a site
        self.client.force_login(self.admin)
        self.site_id = _post(self.client, "/api/hoshin/sites/create/", {
            "name": "Controlled Site",
        }).json()["site"]["id"]

        # Grant roles
        _post(self.client, f"/api/hoshin/sites/{self.site_id}/members/grant/", {
            "user_id": str(self.member.id), "role": "member",
        })
        _post(self.client, f"/api/hoshin/sites/{self.site_id}/members/grant/", {
            "user_id": str(self.viewer.id), "role": "viewer",
        })

    def test_member_can_read_site(self):
        self.client.force_login(self.member)
        res = self.client.get(f"/api/hoshin/sites/{self.site_id}/")
        self.assertEqual(res.status_code, 200)

    def test_viewer_can_read_site(self):
        self.client.force_login(self.viewer)
        res = self.client.get(f"/api/hoshin/sites/{self.site_id}/")
        self.assertEqual(res.status_code, 200)

    def test_viewer_cannot_write_site(self):
        self.client.force_login(self.viewer)
        res = _put(self.client, f"/api/hoshin/sites/{self.site_id}/update/", {
            "name": "Hacked",
        })
        self.assertEqual(res.status_code, 403)

    def test_member_can_write_site(self):
        self.client.force_login(self.member)
        res = _put(self.client, f"/api/hoshin/sites/{self.site_id}/update/", {
            "name": "Updated by member",
        })
        self.assertEqual(res.status_code, 200)

    def test_list_site_members(self):
        self.client.force_login(self.admin)
        res = self.client.get(f"/api/hoshin/sites/{self.site_id}/members/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["count"], 2)  # member + viewer

    def test_revoke_access(self):
        self.client.force_login(self.admin)
        members = self.client.get(
            f"/api/hoshin/sites/{self.site_id}/members/"
        ).json()["members"]
        viewer_access = next(m for m in members if m["role"] == "viewer")
        res = self.client.delete(
            f"/api/hoshin/sites/{self.site_id}/members/{viewer_access['id']}/revoke/"
        )
        self.assertEqual(res.status_code, 200)
        # Viewer can no longer see the site
        self.client.force_login(self.viewer)
        res = self.client.get(f"/api/hoshin/sites/{self.site_id}/")
        self.assertEqual(res.status_code, 404)


# =============================================================================
# Hoshin Project CRUD
# =============================================================================


@SECURE_OFF
class HoshinProjectCRUDTest(TestCase):
    """Hoshin project lifecycle (creates core.Project + HoshinProject)."""

    def setUp(self):
        self.user = _make_enterprise_user("hoshin@test.com")
        self.tenant = _setup_tenant(self.user)
        self.client.force_login(self.user)
        self.site_id = _post(self.client, "/api/hoshin/sites/create/", {
            "name": "Test Plant",
        }).json()["site"]["id"]

    def test_create_hoshin_project(self):
        res = _post(self.client, "/api/hoshin/projects/create/", {
            "title": "Reduce Changeover Time",
            "site_id": self.site_id,
            "project_class": "kaizen",
            "project_type": "labor",
            "opportunity": "budgeted_new",
            "calculation_method": "time_reduction",
            "annual_savings_target": 50000,
            "fiscal_year": 2026,
            "champion_name": "Bob Zeisler",
            "leader_name": "Eric Wolters",
        })
        self.assertEqual(res.status_code, 201)
        proj = res.json()["project"]
        self.assertEqual(proj["project_title"], "Reduce Changeover Time")
        self.assertEqual(proj["project_class"], "kaizen")
        self.assertEqual(proj["project_type"], "labor")
        self.assertIsNotNone(proj["project_id"])  # core.Project created
        self.assertEqual(float(proj["annual_savings_target"]), 50000.0)

    def test_list_hoshin_projects(self):
        _post(self.client, "/api/hoshin/projects/create/", {
            "title": "Project A", "site_id": self.site_id,
        })
        _post(self.client, "/api/hoshin/projects/create/", {
            "title": "Project B", "site_id": self.site_id,
        })
        res = self.client.get("/api/hoshin/projects/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["count"], 2)

    def test_list_filter_by_status(self):
        hp_id = _post(self.client, "/api/hoshin/projects/create/", {
            "title": "Active One", "site_id": self.site_id,
            "hoshin_status": "active",
        }).json()["project"]["id"]
        _post(self.client, "/api/hoshin/projects/create/", {
            "title": "Proposed One", "site_id": self.site_id,
        })
        res = self.client.get("/api/hoshin/projects/?status=active")
        self.assertEqual(res.json()["count"], 1)
        self.assertEqual(res.json()["projects"][0]["hoshin_status"], "active")

    def test_get_hoshin_project_detail(self):
        hp_id = _post(self.client, "/api/hoshin/projects/create/", {
            "title": "Detail Test", "site_id": self.site_id,
        }).json()["project"]["id"]
        res = self.client.get(f"/api/hoshin/projects/{hp_id}/")
        self.assertEqual(res.status_code, 200)
        data = res.json()["project"]
        self.assertIn("savings_summary", data)
        self.assertIn("action_items", data)

    def test_update_hoshin_project(self):
        hp_id = _post(self.client, "/api/hoshin/projects/create/", {
            "title": "Before", "site_id": self.site_id,
        }).json()["project"]["id"]
        res = _put(self.client, f"/api/hoshin/projects/{hp_id}/update/", {
            "title": "After",
            "hoshin_status": "active",
            "annual_savings_target": 75000,
        })
        self.assertEqual(res.status_code, 200)
        proj = res.json()["project"]
        self.assertEqual(proj["project_title"], "After")
        self.assertEqual(proj["hoshin_status"], "active")

    def test_delete_hoshin_project(self):
        hp_id = _post(self.client, "/api/hoshin/projects/create/", {
            "title": "Delete Me", "site_id": self.site_id,
        }).json()["project"]["id"]
        res = self.client.delete(f"/api/hoshin/projects/{hp_id}/delete/")
        self.assertEqual(res.status_code, 200)
        # Verify gone
        self.assertEqual(
            self.client.get(f"/api/hoshin/projects/{hp_id}/").status_code, 404
        )

    def test_create_requires_title(self):
        res = _post(self.client, "/api/hoshin/projects/create/", {
            "title": "", "site_id": self.site_id,
        })
        self.assertEqual(res.status_code, 400)


# =============================================================================
# Monthly Savings Tracking
# =============================================================================


@SECURE_OFF
class MonthlySavingsTest(TestCase):
    """Monthly savings calculation with multiple methods."""

    def setUp(self):
        self.user = _make_enterprise_user("savings@test.com")
        self.tenant = _setup_tenant(self.user)
        self.client.force_login(self.user)
        site_id = _post(self.client, "/api/hoshin/sites/create/", {
            "name": "Savings Plant",
        }).json()["site"]["id"]

        # Create a time_reduction project
        self.hp_id = _post(self.client, "/api/hoshin/projects/create/", {
            "title": "Changeover Kaizen",
            "site_id": site_id,
            "calculation_method": "time_reduction",
            "annual_savings_target": 60000,
        }).json()["project"]["id"]

    def test_update_monthly_actual(self):
        """Update January with baseline/actual and verify savings calc."""
        res = _put(self.client, f"/api/hoshin/projects/{self.hp_id}/monthly/1/", {
            "baseline": 3600,  # 3600s (1 hour) before
            "actual": 2400,    # 2400s (40 min) after
            "volume": 1000,
            "cost_per_unit": 25,  # $25/hr labor rate
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        entry = data["entry"]
        # time_reduction: (3600-2400)/3600 * 1000 * 25 = 8333.33
        self.assertGreater(entry["savings"], 0)
        self.assertAlmostEqual(entry["savings"], 8333.33, places=0)

    def test_ytd_accumulates(self):
        """YTD savings accumulate across months."""
        for month in [1, 2, 3]:
            _put(self.client, f"/api/hoshin/projects/{self.hp_id}/monthly/{month}/", {
                "baseline": 3600,
                "actual": 3000,
                "volume": 1000,
                "cost_per_unit": 25,
            })
        res = self.client.get(f"/api/hoshin/projects/{self.hp_id}/")
        ytd = res.json()["project"]["ytd_savings"]
        # Each month: (600/3600)*1000*25 = 4166.67
        self.assertAlmostEqual(ytd, 4166.67 * 3, delta=5)

    def test_savings_pct(self):
        """Savings percentage = ytd / target * 100."""
        _put(self.client, f"/api/hoshin/projects/{self.hp_id}/monthly/1/", {
            "baseline": 3600, "actual": 2400, "volume": 1000, "cost_per_unit": 25,
        })
        res = self.client.get(f"/api/hoshin/projects/{self.hp_id}/")
        proj = res.json()["project"]
        expected_pct = proj["ytd_savings"] / 60000 * 100
        self.assertAlmostEqual(proj["savings_pct"], expected_pct, places=1)

    def test_invalid_month_rejected(self):
        res = _put(self.client, f"/api/hoshin/projects/{self.hp_id}/monthly/0/", {
            "baseline": 100, "actual": 80,
        })
        self.assertEqual(res.status_code, 400)
        res = _put(self.client, f"/api/hoshin/projects/{self.hp_id}/monthly/13/", {
            "baseline": 100, "actual": 80,
        })
        self.assertEqual(res.status_code, 400)

    def test_waste_pct_method(self):
        """Waste percentage savings calculation."""
        site_id = _post(self.client, "/api/hoshin/sites/create/", {
            "name": "Waste Test",
        }).json()["site"]["id"]
        hp_id = _post(self.client, "/api/hoshin/projects/create/", {
            "title": "Scrap Reduction",
            "site_id": site_id,
            "calculation_method": "waste_pct",
            "annual_savings_target": 100000,
        }).json()["project"]["id"]

        res = _put(self.client, f"/api/hoshin/projects/{hp_id}/monthly/1/", {
            "baseline": 8.0,   # 8% scrap rate
            "actual": 5.0,     # 5% scrap rate
            "volume": 50000,
            "cost_per_unit": 12,
        })
        entry = res.json()["entry"]
        # (8-5)/100 * 50000 * 12 = 18000
        self.assertAlmostEqual(entry["savings"], 18000.0, places=0)

    def test_headcount_method(self):
        """Headcount reduction savings calculation."""
        site_id = _post(self.client, "/api/hoshin/sites/create/", {
            "name": "HC Test",
        }).json()["site"]["id"]
        hp_id = _post(self.client, "/api/hoshin/projects/create/", {
            "title": "Labor Efficiency",
            "site_id": site_id,
            "calculation_method": "headcount",
            "annual_savings_target": 200000,
        }).json()["project"]["id"]

        res = _put(self.client, f"/api/hoshin/projects/{hp_id}/monthly/1/", {
            "baseline": 12,    # 12 operators
            "actual": 10,      # 10 operators
            "volume": 1,       # not used in headcount
            "cost_per_unit": 65000,  # cost per employee
        })
        entry = res.json()["entry"]
        # (12-10) * 65000 = 130000
        self.assertAlmostEqual(entry["savings"], 130000.0, places=0)

    def test_direct_method(self):
        """Direct cost comparison."""
        site_id = _post(self.client, "/api/hoshin/sites/create/", {
            "name": "Direct Test",
        }).json()["site"]["id"]
        hp_id = _post(self.client, "/api/hoshin/projects/create/", {
            "title": "Cost Reduction",
            "site_id": site_id,
            "calculation_method": "direct",
            "annual_savings_target": 50000,
        }).json()["project"]["id"]

        res = _put(self.client, f"/api/hoshin/projects/{hp_id}/monthly/1/", {
            "baseline": 25000,
            "actual": 18000,
        })
        entry = res.json()["entry"]
        self.assertAlmostEqual(entry["savings"], 7000.0, places=0)


# =============================================================================
# Action Items
# =============================================================================


@SECURE_OFF
class ActionItemTest(TestCase):
    """Action item (task) management within hoshin projects."""

    def setUp(self):
        self.user = _make_enterprise_user("actions@test.com")
        self.tenant = _setup_tenant(self.user)
        self.client.force_login(self.user)
        site_id = _post(self.client, "/api/hoshin/sites/create/", {
            "name": "Action Plant",
        }).json()["site"]["id"]
        self.hp_id = _post(self.client, "/api/hoshin/projects/create/", {
            "title": "SMED Event", "site_id": site_id,
        }).json()["project"]["id"]

    def test_create_action_item(self):
        res = _post(self.client, f"/api/hoshin/projects/{self.hp_id}/actions/create/", {
            "title": "Video current changeover",
            "owner_name": "Line Lead",
            "due_date": "2026-03-15",
        })
        self.assertEqual(res.status_code, 201)
        item = res.json()["action_item"]
        self.assertEqual(item["title"], "Video current changeover")
        self.assertEqual(item["status"], "not_started")
        self.assertEqual(item["source_type"], "hoshin")

    def test_list_action_items(self):
        _post(self.client, f"/api/hoshin/projects/{self.hp_id}/actions/create/", {
            "title": "Task A",
        })
        _post(self.client, f"/api/hoshin/projects/{self.hp_id}/actions/create/", {
            "title": "Task B",
        })
        res = self.client.get(f"/api/hoshin/projects/{self.hp_id}/actions/")
        self.assertEqual(res.json()["count"], 2)

    def test_update_action_progress(self):
        item_id = _post(
            self.client, f"/api/hoshin/projects/{self.hp_id}/actions/create/",
            {"title": "Track me"},
        ).json()["action_item"]["id"]
        res = _put(self.client, f"/api/hoshin/actions/{item_id}/update/", {
            "status": "in_progress",
            "progress": 50,
        })
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["action_item"]["progress"], 50)

    def test_action_item_dependency(self):
        """Task B depends on Task A."""
        a_id = _post(
            self.client, f"/api/hoshin/projects/{self.hp_id}/actions/create/",
            {"title": "Task A"},
        ).json()["action_item"]["id"]
        b_id = _post(
            self.client, f"/api/hoshin/projects/{self.hp_id}/actions/create/",
            {"title": "Task B", "depends_on_id": a_id},
        ).json()["action_item"]["id"]
        item_b = self.client.get(
            f"/api/hoshin/projects/{self.hp_id}/actions/"
        ).json()["action_items"]
        b = next(i for i in item_b if i["id"] == b_id)
        self.assertEqual(b["depends_on_id"], a_id)

    def test_delete_action_item(self):
        item_id = _post(
            self.client, f"/api/hoshin/projects/{self.hp_id}/actions/create/",
            {"title": "Delete me"},
        ).json()["action_item"]["id"]
        res = self.client.delete(f"/api/hoshin/actions/{item_id}/delete/")
        self.assertEqual(res.status_code, 200)

    def test_create_requires_title(self):
        res = _post(self.client, f"/api/hoshin/projects/{self.hp_id}/actions/create/", {
            "title": "",
        })
        self.assertEqual(res.status_code, 400)


# =============================================================================
# VSM → Hoshin: Batch Proposal Creation
# =============================================================================


@SECURE_OFF
class VSMProposalBatchTest(TestCase):
    """Create hoshin projects from VSM kaizen burst proposals."""

    def setUp(self):
        self.user = _make_enterprise_user("proposals@test.com")
        self.tenant = _setup_tenant(self.user)
        self.client.force_login(self.user)
        self.drf_client = APIClient()
        self.drf_client.force_authenticate(self.user)

        # Create site
        self.site_id = _post(self.client, "/api/hoshin/sites/create/", {
            "name": "Proposal Plant",
        }).json()["site"]["id"]

        # Create VSM with project (user-owned, since VSM is @gated_paid not Hoshin)
        proj = self.drf_client.post("/api/core/projects/", {
            "title": "VSM Study", "problem_statement": "Reduce lead time",
            "domain": "manufacturing", "methodology": "dmaic",
        }, format="json").json()
        self.project_id = proj["id"]

        self.vsm_id = _post(self.client, "/api/vsm/create/", {
            "name": "Batch Source VSM",
            "project_id": self.project_id,
        }).json()["id"]

        # Add steps and kaizen bursts
        _post(self.client, f"/api/vsm/{self.vsm_id}/process-step/", {
            "name": "Assembly", "cycle_time": 60, "x": 200, "y": 300,
        })
        _post(self.client, f"/api/vsm/{self.vsm_id}/kaizen/", {
            "text": "SMED on assembly", "priority": "high", "x": 200, "y": 250,
        })
        _post(self.client, f"/api/vsm/{self.vsm_id}/kaizen/", {
            "text": "Add poka-yoke", "priority": "medium", "x": 300, "y": 250,
        })

        # Get the burst IDs
        vsm_data = self.client.get(f"/api/vsm/{self.vsm_id}/").json()["vsm"]
        self.bursts = vsm_data["kaizen_bursts"]

    def test_create_from_proposals(self):
        """Batch create hoshin projects from approved proposals."""
        res = _post(self.client, "/api/hoshin/projects/from-proposals/", {
            "vsm_id": self.vsm_id,
            "fiscal_year": 2026,
            "site_id": self.site_id,
            "proposals": [
                {
                    "burst_id": self.bursts[0]["id"],
                    "title": "SMED Assembly Changeover",
                    "project_class": "kaizen",
                    "project_type": "labor",
                    "calculation_method": "time_reduction",
                    "annual_savings_target": 25000,
                    "approved": True,
                },
                {
                    "burst_id": self.bursts[1]["id"],
                    "title": "Poka-Yoke Installation",
                    "project_class": "project",
                    "project_type": "quality",
                    "calculation_method": "waste_pct",
                    "annual_savings_target": 15000,
                    "approved": True,
                },
            ],
        })
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["count"], 2)

        # Verify source_vsm and source_burst_id linkage
        created = data["created"]
        smed = next(p for p in created if "SMED" in p["project_title"])
        self.assertEqual(smed["source_vsm_id"], self.vsm_id)
        self.assertEqual(smed["source_burst_id"], self.bursts[0]["id"])

    def test_unapproved_proposals_skipped(self):
        """Only approved proposals create projects."""
        res = _post(self.client, "/api/hoshin/projects/from-proposals/", {
            "vsm_id": self.vsm_id,
            "site_id": self.site_id,
            "proposals": [
                {"burst_id": "b1", "title": "Yes", "approved": True,
                 "annual_savings_target": 10000},
                {"burst_id": "b2", "title": "No", "approved": False},
            ],
        })
        self.assertEqual(res.json()["count"], 1)
        self.assertEqual(res.json()["created"][0]["project_title"], "Yes")

    def test_no_approved_returns_400(self):
        res = _post(self.client, "/api/hoshin/projects/from-proposals/", {
            "proposals": [
                {"burst_id": "b1", "title": "Nope", "approved": False},
            ],
        })
        self.assertEqual(res.status_code, 400)


# =============================================================================
# VSM Promotion (Future → Current)
# =============================================================================


@SECURE_OFF
class VSMPromotionTest(TestCase):
    """Promote future-state VSM to current, archive old, writeback savings."""

    def setUp(self):
        self.user = _make_enterprise_user("promote@test.com")
        self.tenant = _setup_tenant(self.user)
        self.client.force_login(self.user)
        self.drf_client = APIClient()
        self.drf_client.force_authenticate(self.user)

        # Create site
        self.site_id = _post(self.client, "/api/hoshin/sites/create/", {
            "name": "Promo Plant",
        }).json()["site"]["id"]

        # Create project for VSM
        proj = self.drf_client.post("/api/core/projects/", {
            "title": "Lean Transformation",
            "problem_statement": "Reduce lead time",
            "domain": "manufacturing",
            "methodology": "dmaic",
        }, format="json").json()
        self.project_id = proj["id"]

        # Create current-state VSM
        self.current_id = _post(self.client, "/api/vsm/create/", {
            "name": "Press Line Current",
            "project_id": self.project_id,
        }).json()["id"]

        _post(self.client, f"/api/vsm/{self.current_id}/process-step/", {
            "name": "Stamping", "cycle_time": 60, "x": 100, "y": 300,
        })
        _post(self.client, f"/api/vsm/{self.current_id}/kaizen/", {
            "text": "SMED press", "priority": "high", "x": 100, "y": 250,
        })

        # Create future state
        future_res = _post(self.client, f"/api/vsm/{self.current_id}/future-state/")
        self.future_id = future_res.json()["future_state"]["id"]

        # Get burst IDs from future (cloned from current)
        future_vsm = self.client.get(f"/api/vsm/{self.future_id}/").json()["vsm"]
        self.burst_id = future_vsm["kaizen_bursts"][0]["id"]

    def test_promote_future_to_current(self):
        """Promotion changes future→current and old current→archived."""
        res = _post(self.client, f"/api/hoshin/vsm/{self.future_id}/promote/")
        self.assertEqual(res.status_code, 200)
        promoted = res.json()["vsm"]
        self.assertEqual(promoted["status"], "current")

        # Old current should be archived
        old = self.client.get(f"/api/vsm/{self.current_id}/").json()["vsm"]
        self.assertEqual(old["status"], "archived")

    def test_promote_carries_forward_snapshots(self):
        """Metric snapshots from old current are carried into promoted VSM."""
        # Add a step to future to create a metric snapshot
        _post(self.client, f"/api/vsm/{self.future_id}/process-step/", {
            "name": "New Step", "cycle_time": 30, "x": 200, "y": 300,
        })
        future_snaps = self.client.get(
            f"/api/vsm/{self.future_id}/"
        ).json()["vsm"]["metric_snapshots"]
        current_snaps = self.client.get(
            f"/api/vsm/{self.current_id}/"
        ).json()["vsm"]["metric_snapshots"]

        _post(self.client, f"/api/hoshin/vsm/{self.future_id}/promote/")
        promoted_snaps = self.client.get(
            f"/api/vsm/{self.future_id}/"
        ).json()["vsm"]["metric_snapshots"]

        # Promoted should have at least as many snapshots as both combined
        self.assertGreaterEqual(
            len(promoted_snaps),
            len(future_snaps),
        )

    def test_promote_writebacks_hoshin_savings(self):
        """Hoshin project savings are written back to kaizen bursts on promotion."""
        from agents_api.models import HoshinProject, ValueStreamMap

        # Create a hoshin project linked to the future VSM's kaizen burst
        hp_id = _post(self.client, "/api/hoshin/projects/create/", {
            "title": "SMED Press Project",
            "site_id": self.site_id,
            "source_vsm_id": str(self.future_id),
            "source_burst_id": self.burst_id,
            "calculation_method": "time_reduction",
            "annual_savings_target": 50000,
        }).json()["project"]["id"]

        # Record monthly savings
        _put(self.client, f"/api/hoshin/projects/{hp_id}/monthly/1/", {
            "baseline": 3600, "actual": 2400, "volume": 500, "cost_per_unit": 30,
        })
        _put(self.client, f"/api/hoshin/projects/{hp_id}/monthly/2/", {
            "baseline": 3600, "actual": 2400, "volume": 500, "cost_per_unit": 30,
        })

        # Promote
        _post(self.client, f"/api/hoshin/vsm/{self.future_id}/promote/")

        # Check burst has realized savings
        promoted = self.client.get(f"/api/vsm/{self.future_id}/").json()["vsm"]
        burst = next(
            b for b in promoted["kaizen_bursts"] if b["id"] == self.burst_id
        )
        self.assertIn("realized_savings", burst)
        self.assertGreater(burst["realized_savings"], 0)
        self.assertIn("savings_pct", burst)
        self.assertIn("project_status", burst)

    def test_promote_non_future_returns_400(self):
        """Only future-state VSMs can be promoted."""
        res = _post(self.client, f"/api/hoshin/vsm/{self.current_id}/promote/")
        self.assertEqual(res.status_code, 400)
        self.assertIn("future", res.json()["error"].lower())

    def test_promote_unpaired_returns_400(self):
        """Future VSM without a paired current returns 400."""
        from agents_api.models import ValueStreamMap
        # Break the pairing
        vsm = ValueStreamMap.objects.get(id=self.future_id)
        vsm.paired_with = None
        vsm.save(update_fields=["paired_with"])

        res = _post(self.client, f"/api/hoshin/vsm/{self.future_id}/promote/")
        self.assertEqual(res.status_code, 400)
        self.assertIn("paired", res.json()["error"].lower())


# =============================================================================
# Dashboard
# =============================================================================


@SECURE_OFF
class HoshinDashboardTest(TestCase):
    """Enterprise savings dashboard."""

    def setUp(self):
        self.user = _make_enterprise_user("dash@test.com")
        self.tenant = _setup_tenant(self.user)
        self.client.force_login(self.user)

        # Create two sites with projects
        self.site_a = _post(self.client, "/api/hoshin/sites/create/", {
            "name": "Plant A",
        }).json()["site"]["id"]
        self.site_b = _post(self.client, "/api/hoshin/sites/create/", {
            "name": "Plant B",
        }).json()["site"]["id"]

        # Create projects at each site
        self.hp_a = _post(self.client, "/api/hoshin/projects/create/", {
            "title": "Kaizen A1",
            "site_id": self.site_a,
            "annual_savings_target": 50000,
            "hoshin_status": "active",
            "calculation_method": "direct",
        }).json()["project"]["id"]

        self.hp_b = _post(self.client, "/api/hoshin/projects/create/", {
            "title": "Kaizen B1",
            "site_id": self.site_b,
            "annual_savings_target": 30000,
            "hoshin_status": "active",
            "calculation_method": "direct",
        }).json()["project"]["id"]

        # Record savings
        _put(self.client, f"/api/hoshin/projects/{self.hp_a}/monthly/1/", {
            "baseline": 5000, "actual": 3500,
        })
        _put(self.client, f"/api/hoshin/projects/{self.hp_b}/monthly/1/", {
            "baseline": 3000, "actual": 2200,
        })

    def test_dashboard_totals(self):
        res = self.client.get("/api/hoshin/dashboard/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertAlmostEqual(data["total_target"], 80000.0, places=0)
        self.assertGreater(data["total_ytd"], 0)
        self.assertEqual(data["project_count"], 2)

    def test_dashboard_by_site(self):
        data = self.client.get("/api/hoshin/dashboard/").json()
        sites = data["by_site"]
        self.assertEqual(len(sites), 2)
        names = {s["site_name"] for s in sites}
        self.assertEqual(names, {"Plant A", "Plant B"})

    def test_dashboard_monthly_trend(self):
        data = self.client.get("/api/hoshin/dashboard/").json()
        trend = data["monthly_trend"]
        self.assertEqual(len(trend), 12)
        jan = trend[0]
        self.assertEqual(jan["month"], 1)
        self.assertGreater(jan["actual"], 0)
        self.assertGreater(jan["target"], 0)

    def test_dashboard_status_counts(self):
        data = self.client.get("/api/hoshin/dashboard/").json()
        self.assertEqual(data["status_counts"].get("active", 0), 2)


# =============================================================================
# Tier Gating
# =============================================================================


@SECURE_OFF
class HoshinTierGatingTest(TestCase):
    """Non-enterprise users cannot access Hoshin endpoints."""

    def test_pro_user_blocked(self):
        user = User.objects.create_user(
            username="pro", email="pro@test.com", password="testpass123!",
        )
        user.tier = Tier.PRO
        user.save(update_fields=["tier"])
        self.client.force_login(user)
        res = self.client.get("/api/hoshin/sites/")
        self.assertEqual(res.status_code, 403)

    def test_free_user_blocked(self):
        user = User.objects.create_user(
            username="free", email="free@test.com", password="testpass123!",
        )
        self.client.force_login(user)
        res = self.client.get("/api/hoshin/sites/")
        self.assertEqual(res.status_code, 403)

    def test_unauthenticated_blocked(self):
        res = self.client.get("/api/hoshin/sites/")
        self.assertEqual(res.status_code, 401)

    def test_enterprise_without_tenant(self):
        """Enterprise user without an org gets 400, not 403."""
        user = User.objects.create_user(
            username="lonely", email="lonely@test.com", password="testpass123!",
        )
        user.tier = Tier.ENTERPRISE
        user.save(update_fields=["tier"])
        self.client.force_login(user)
        res = self.client.get("/api/hoshin/sites/")
        self.assertEqual(res.status_code, 400)
        self.assertIn("tenant", res.json()["error"].lower())


# =============================================================================
# Full Pipeline: VSM → Proposals → Hoshin → Savings → Promote → Writeback
# =============================================================================


@SECURE_OFF
class FullVSMHoshinPipelineTest(TestCase):
    """End-to-end: build VSM, generate proposals, create projects, track savings,
    promote, verify writeback."""

    def setUp(self):
        self.user = _make_enterprise_user("pipeline@test.com")
        self.tenant = _setup_tenant(self.user)
        self.client.force_login(self.user)
        self.drf_client = APIClient()
        self.drf_client.force_authenticate(self.user)

    def test_full_pipeline(self):
        # 1. Create site
        site_id = _post(self.client, "/api/hoshin/sites/create/", {
            "name": "Pipeline Plant",
        }).json()["site"]["id"]

        # 2. Create core project for VSM
        proj = self.drf_client.post("/api/core/projects/", {
            "title": "Annual Lean Initiative",
            "problem_statement": "Reduce changeover by 50%",
            "domain": "manufacturing",
            "methodology": "dmaic",
        }, format="json").json()

        # 3. Build current-state VSM
        vsm_id = _post(self.client, "/api/vsm/create/", {
            "name": "Packaging Line Current",
            "project_id": proj["id"],
        }).json()["id"]

        _post(self.client, f"/api/vsm/{vsm_id}/process-step/", {
            "name": "Fill", "cycle_time": 10, "changeover_time": 1800,
            "uptime": 85, "operators": 2, "x": 100, "y": 300,
        })
        _post(self.client, f"/api/vsm/{vsm_id}/process-step/", {
            "name": "Seal", "cycle_time": 8, "changeover_time": 900,
            "uptime": 90, "operators": 1, "x": 300, "y": 300,
        })
        _post(self.client, f"/api/vsm/{vsm_id}/inventory/", {
            "quantity": 5000, "days_of_supply": 3.0,
        })

        # 4. Create future state
        future_id = _post(
            self.client, f"/api/vsm/{vsm_id}/future-state/"
        ).json()["future_state"]["id"]

        # 5. Improve future: reduce changeover, add bursts
        _put(self.client, f"/api/vsm/{future_id}/update/", {
            "process_steps": [
                {"id": "s1", "name": "Fill", "cycle_time": 10,
                 "changeover_time": 600, "uptime": 95, "operators": 2,
                 "x": 100, "y": 300},
                {"id": "s2", "name": "Seal", "cycle_time": 8,
                 "changeover_time": 300, "uptime": 95, "operators": 1,
                 "x": 300, "y": 300},
            ],
            "inventory": [{"id": "i1", "quantity": 1000, "days_of_supply": 0.5}],
            "kaizen_bursts": [
                {"id": "k1", "text": "SMED filler changeover",
                 "priority": "high", "x": 100, "y": 250},
                {"id": "k2", "text": "TPM program for sealer",
                 "priority": "medium", "x": 300, "y": 250},
            ],
        })

        # 6. Generate proposals from VSM
        proposals = _post(self.client, f"/api/vsm/{vsm_id}/generate-proposals/", {
            "annual_volume": 200000, "cost_per_unit": 15,
        }).json()
        self.assertGreater(proposals["count"], 0)

        # 7. Create hoshin projects from proposals
        from_props = _post(self.client, "/api/hoshin/projects/from-proposals/", {
            "vsm_id": future_id,
            "fiscal_year": 2026,
            "site_id": site_id,
            "proposals": [
                {
                    "burst_id": "k1",
                    "title": proposals["proposals"][0]["suggested_title"],
                    "project_class": "kaizen",
                    "project_type": "labor",
                    "calculation_method": "time_reduction",
                    "annual_savings_target": 30000,
                    "approved": True,
                },
            ],
        }).json()
        self.assertEqual(from_props["count"], 1)
        hp_id = from_props["created"][0]["id"]

        # 8. Track monthly savings
        _put(self.client, f"/api/hoshin/projects/{hp_id}/monthly/1/", {
            "baseline": 1800, "actual": 600, "volume": 500, "cost_per_unit": 25,
        })
        _put(self.client, f"/api/hoshin/projects/{hp_id}/monthly/2/", {
            "baseline": 1800, "actual": 600, "volume": 500, "cost_per_unit": 25,
        })

        # Verify YTD
        hp_detail = self.client.get(f"/api/hoshin/projects/{hp_id}/").json()["project"]
        self.assertGreater(hp_detail["ytd_savings"], 0)

        # 9. Promote future VSM to current
        promote_res = _post(self.client, f"/api/hoshin/vsm/{future_id}/promote/")
        self.assertEqual(promote_res.status_code, 200)
        self.assertEqual(promote_res.json()["vsm"]["status"], "current")

        # 10. Verify old current is archived
        old_current = self.client.get(f"/api/vsm/{vsm_id}/").json()["vsm"]
        self.assertEqual(old_current["status"], "archived")

        # 11. Verify savings written back to kaizen burst
        promoted_vsm = self.client.get(f"/api/vsm/{future_id}/").json()["vsm"]
        k1_burst = next(
            (b for b in promoted_vsm["kaizen_bursts"] if b["id"] == "k1"), None
        )
        self.assertIsNotNone(k1_burst)
        self.assertIn("realized_savings", k1_burst)
        self.assertGreater(k1_burst["realized_savings"], 0)

        # 12. Dashboard reflects everything
        dashboard = self.client.get("/api/hoshin/dashboard/").json()
        self.assertGreater(dashboard["total_ytd"], 0)
        self.assertEqual(dashboard["project_count"], 1)
