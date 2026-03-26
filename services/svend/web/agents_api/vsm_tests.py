"""VSM (Value Stream Mapping) integration tests.

Tests the full VSM surface:
- CRUD lifecycle (create, list, get, update, delete)
- Process steps, inventory, kaizen bursts
- Metric calculations (lead time, process time, PCE)
- Bottleneck detection (standalone + work centers)
- Future state creation and current/future comparison
- Project linking and hub integration
- User isolation
- Auto-kaizen deduplication
- Metric snapshot history
- Tier gating (@gated_paid)
- Generate proposals (enterprise-only, @require_feature)
"""

import json

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from accounts.constants import Tier

User = get_user_model()
SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)


def _make_user(email, tier=Tier.PRO, **kwargs):
    username = kwargs.pop("username", email.split("@")[0])
    user = User.objects.create_user(
        username=username, email=email, password="testpass123!", **kwargs
    )
    user.tier = tier
    user.save(update_fields=["tier"])
    return user


def _post(client, url, data=None):
    return client.post(url, json.dumps(data or {}), content_type="application/json")


def _put(client, url, data=None):
    return client.put(url, json.dumps(data or {}), content_type="application/json")


def _patch(client, url, data=None):
    return client.patch(url, json.dumps(data or {}), content_type="application/json")


# =============================================================================
# VSM CRUD Lifecycle
# =============================================================================


@SECURE_OFF
class VSMCRUDTest(TestCase):
    """Full create/read/update/delete lifecycle."""

    def setUp(self):
        self.user = _make_user("vsm@test.com")
        self.client.force_login(self.user)

    def test_create_vsm_defaults(self):
        res = _post(self.client, "/api/vsm/create/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("id", data)
        vsm = data["vsm"]
        self.assertEqual(vsm["name"], "Untitled VSM")
        self.assertEqual(vsm["status"], "current")
        self.assertIsNone(vsm["project_id"])

    def test_create_vsm_with_fields(self):
        res = _post(
            self.client,
            "/api/vsm/create/",
            {
                "name": "Paint Line VSM",
                "product_family": "Automotive Parts",
                "customer_name": "OEM Corp",
                "customer_demand": "460 units/day",
                "supplier_name": "Steel Co",
                "supply_frequency": "Weekly",
            },
        )
        self.assertEqual(res.status_code, 200)
        vsm = res.json()["vsm"]
        self.assertEqual(vsm["name"], "Paint Line VSM")
        self.assertEqual(vsm["product_family"], "Automotive Parts")
        self.assertEqual(vsm["customer_name"], "OEM Corp")
        self.assertEqual(vsm["customer_demand"], "460 units/day")
        self.assertEqual(vsm["supplier_name"], "Steel Co")
        self.assertEqual(vsm["supply_frequency"], "Weekly")

    def test_list_vsm(self):
        _post(self.client, "/api/vsm/create/", {"name": "VSM A"})
        _post(self.client, "/api/vsm/create/", {"name": "VSM B"})
        res = self.client.get("/api/vsm/")
        self.assertEqual(res.status_code, 200)
        maps = res.json()["maps"]
        self.assertEqual(len(maps), 2)
        names = {m["name"] for m in maps}
        self.assertEqual(names, {"VSM A", "VSM B"})

    def test_get_vsm(self):
        vsm_id = _post(self.client, "/api/vsm/create/", {"name": "Detail Test"}).json()[
            "id"
        ]
        res = self.client.get(f"/api/vsm/{vsm_id}/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["vsm"]["name"], "Detail Test")
        self.assertIn("bottleneck", data)

    def test_update_vsm_name(self):
        vsm_id = _post(self.client, "/api/vsm/create/").json()["id"]
        res = _put(self.client, f"/api/vsm/{vsm_id}/update/", {"name": "Renamed"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["vsm"]["name"], "Renamed")

    def test_update_vsm_patch(self):
        vsm_id = _post(self.client, "/api/vsm/create/").json()["id"]
        res = _patch(
            self.client,
            f"/api/vsm/{vsm_id}/update/",
            {
                "product_family": "Electronics",
            },
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["vsm"]["product_family"], "Electronics")

    def test_delete_vsm(self):
        vsm_id = _post(self.client, "/api/vsm/create/").json()["id"]
        res = self.client.delete(f"/api/vsm/{vsm_id}/delete/")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["success"])
        # Verify gone
        self.assertEqual(self.client.get(f"/api/vsm/{vsm_id}/").status_code, 404)

    def test_get_nonexistent_returns_404(self):
        import uuid

        res = self.client.get(f"/api/vsm/{uuid.uuid4()}/")
        self.assertEqual(res.status_code, 404)

    def test_list_filter_by_status(self):
        _post(self.client, "/api/vsm/create/", {"name": "Current"})
        # Create a future-state by cloning
        _post(self.client, "/api/vsm/create/", {"name": "Base"}).json()["id"]
        _post(
            self.client, "/api/vsm/create/future-state/"
        )  # won't work, need to use actual endpoint
        # Instead, create and update status directly
        future_id = _post(self.client, "/api/vsm/create/", {"name": "Future"}).json()[
            "id"
        ]
        _put(self.client, f"/api/vsm/{future_id}/update/", {"status": "future"})

        res = self.client.get("/api/vsm/?status=current")
        current_maps = res.json()["maps"]
        self.assertTrue(all(m["status"] == "current" for m in current_maps))

        res = self.client.get("/api/vsm/?status=future")
        future_maps = res.json()["maps"]
        self.assertEqual(len(future_maps), 1)
        self.assertEqual(future_maps[0]["name"], "Future")


# =============================================================================
# Process Steps and Metrics
# =============================================================================


@SECURE_OFF
class VSMProcessStepTest(TestCase):
    """Process step creation and metric calculation."""

    def setUp(self):
        self.user = _make_user("steps@test.com")
        self.client.force_login(self.user)
        self.vsm_id = _post(
            self.client,
            "/api/vsm/create/",
            {
                "name": "Assembly Line",
            },
        ).json()["id"]

    def test_add_process_step(self):
        res = _post(
            self.client,
            f"/api/vsm/{self.vsm_id}/process-step/",
            {
                "name": "Welding",
                "x": 200,
                "y": 300,
                "cycle_time": 45,
                "changeover_time": 600,
                "uptime": 95,
                "operators": 2,
                "shifts": 2,
                "batch_size": 10,
            },
        )
        self.assertEqual(res.status_code, 200)
        step = res.json()["step"]
        self.assertEqual(step["name"], "Welding")
        self.assertEqual(step["cycle_time"], 45)
        self.assertIn("id", step)

    def test_metrics_update_on_step_add(self):
        """Adding a process step recalculates total_process_time."""
        _post(
            self.client,
            f"/api/vsm/{self.vsm_id}/process-step/",
            {
                "name": "Cut",
                "cycle_time": 30,
            },
        )
        _post(
            self.client,
            f"/api/vsm/{self.vsm_id}/process-step/",
            {
                "name": "Weld",
                "cycle_time": 45,
            },
        )
        res = self.client.get(f"/api/vsm/{self.vsm_id}/")
        vsm = res.json()["vsm"]
        # 30 + 45 = 75 seconds total process time
        self.assertAlmostEqual(vsm["total_process_time"], 75.0, places=1)
        # Lead time = process time in days (75/86400)
        self.assertGreater(vsm["total_lead_time"], 0)

    def test_multiple_steps_accumulate(self):
        """Three steps should sum correctly."""
        for name, ct in [("A", 10), ("B", 20), ("C", 30)]:
            _post(
                self.client,
                f"/api/vsm/{self.vsm_id}/process-step/",
                {
                    "name": name,
                    "cycle_time": ct,
                },
            )
        vsm = self.client.get(f"/api/vsm/{self.vsm_id}/").json()["vsm"]
        self.assertAlmostEqual(vsm["total_process_time"], 60.0, places=1)


# =============================================================================
# Inventory and Lead Time
# =============================================================================


@SECURE_OFF
class VSMInventoryTest(TestCase):
    """Inventory triangles and lead time calculation."""

    def setUp(self):
        self.user = _make_user("inv@test.com")
        self.client.force_login(self.user)
        self.vsm_id = _post(self.client, "/api/vsm/create/").json()["id"]
        # Add a step first to have a reference
        step_res = _post(
            self.client,
            f"/api/vsm/{self.vsm_id}/process-step/",
            {
                "name": "Assembly",
                "cycle_time": 60,
            },
        )
        self.step_id = step_res.json()["step"]["id"]

    def test_add_inventory(self):
        res = _post(
            self.client,
            f"/api/vsm/{self.vsm_id}/inventory/",
            {
                "before_step_id": self.step_id,
                "quantity": 500,
                "days_of_supply": 2.5,
                "x": 150,
                "y": 350,
            },
        )
        self.assertEqual(res.status_code, 200)
        inv = res.json()["inventory"]
        self.assertEqual(inv["quantity"], 500)
        self.assertEqual(inv["days_of_supply"], 2.5)
        self.assertIn("id", inv)

    def test_inventory_affects_lead_time(self):
        """Adding inventory increases total lead time (days of supply adds wait time)."""
        vsm_before = self.client.get(f"/api/vsm/{self.vsm_id}/").json()["vsm"]
        lead_before = vsm_before["total_lead_time"]

        _post(
            self.client,
            f"/api/vsm/{self.vsm_id}/inventory/",
            {
                "before_step_id": self.step_id,
                "quantity": 1000,
                "days_of_supply": 5.0,
            },
        )
        vsm_after = self.client.get(f"/api/vsm/{self.vsm_id}/").json()["vsm"]
        lead_after = vsm_after["total_lead_time"]

        # Lead time should increase by ~5.0 days
        self.assertAlmostEqual(lead_after - lead_before, 5.0, places=2)

    def test_pce_calculation(self):
        """PCE = (process time in days / total lead time) * 100."""
        # Add significant inventory to make PCE meaningful
        _post(
            self.client,
            f"/api/vsm/{self.vsm_id}/inventory/",
            {
                "before_step_id": self.step_id,
                "quantity": 500,
                "days_of_supply": 10.0,
            },
        )
        vsm = self.client.get(f"/api/vsm/{self.vsm_id}/").json()["vsm"]
        # process_time = 60s = 60/86400 days ≈ 0.000694 days
        # total_lead_time = 10.0 + 0.000694 ≈ 10.000694 days
        # PCE = 0.000694 / 10.000694 * 100 ≈ 0.007%
        self.assertGreater(vsm["pce"], 0)
        self.assertLess(vsm["pce"], 1)  # Typical lean finding: very low PCE


# =============================================================================
# Kaizen Bursts
# =============================================================================


@SECURE_OFF
class VSMKaizenBurstTest(TestCase):
    """Kaizen burst (improvement opportunity) management."""

    def setUp(self):
        self.user = _make_user("kaizen@test.com")
        self.client.force_login(self.user)
        self.vsm_id = _post(self.client, "/api/vsm/create/").json()["id"]

    def test_add_kaizen_burst(self):
        res = _post(
            self.client,
            f"/api/vsm/{self.vsm_id}/kaizen/",
            {
                "x": 300,
                "y": 200,
                "text": "Reduce changeover time",
                "priority": "high",
            },
        )
        self.assertEqual(res.status_code, 200)
        burst = res.json()["burst"]
        self.assertEqual(burst["text"], "Reduce changeover time")
        self.assertEqual(burst["priority"], "high")

    def test_multiple_bursts(self):
        for text in ["Reduce setup", "Add poka-yoke", "Eliminate rework"]:
            _post(self.client, f"/api/vsm/{self.vsm_id}/kaizen/", {"text": text})
        vsm = self.client.get(f"/api/vsm/{self.vsm_id}/").json()["vsm"]
        self.assertEqual(len(vsm["kaizen_bursts"]), 3)

    def test_auto_kaizen_dedup(self):
        """Auto-kaizen via update should deduplicate by text."""
        # Add a process step for positioning
        _post(
            self.client,
            f"/api/vsm/{self.vsm_id}/process-step/",
            {
                "name": "Paint",
                "cycle_time": 30,
                "x": 200,
                "y": 300,
            },
        )

        # Add via auto_kaizen
        _put(
            self.client,
            f"/api/vsm/{self.vsm_id}/update/",
            {
                "auto_kaizen": {
                    "text": "Reduce paint drying time",
                    "near_step": "Paint",
                    "priority": "high",
                }
            },
        )
        vsm = self.client.get(f"/api/vsm/{self.vsm_id}/").json()["vsm"]
        self.assertEqual(len(vsm["kaizen_bursts"]), 1)
        self.assertEqual(vsm["kaizen_bursts"][0]["text"], "Reduce paint drying time")

        # Add same text again — should NOT duplicate
        _put(
            self.client,
            f"/api/vsm/{self.vsm_id}/update/",
            {
                "auto_kaizen": {
                    "text": "Reduce paint drying time",
                    "near_step": "Paint",
                    "priority": "medium",
                }
            },
        )
        vsm = self.client.get(f"/api/vsm/{self.vsm_id}/").json()["vsm"]
        self.assertEqual(len(vsm["kaizen_bursts"]), 1)


# =============================================================================
# Bottleneck Detection
# =============================================================================


@SECURE_OFF
class VSMBottleneckTest(TestCase):
    """Bottleneck identification with takt time comparison."""

    def setUp(self):
        self.user = _make_user("bottleneck@test.com")
        self.client.force_login(self.user)
        self.vsm_id = _post(
            self.client,
            "/api/vsm/create/",
            {
                "name": "Bottleneck Test",
            },
        ).json()["id"]

    def test_bottleneck_identified(self):
        """Step with highest cycle time is the bottleneck."""
        _post(
            self.client,
            f"/api/vsm/{self.vsm_id}/process-step/",
            {
                "name": "Fast",
                "cycle_time": 10,
            },
        )
        _post(
            self.client,
            f"/api/vsm/{self.vsm_id}/process-step/",
            {
                "name": "Slow",
                "cycle_time": 60,
            },
        )
        _post(
            self.client,
            f"/api/vsm/{self.vsm_id}/process-step/",
            {
                "name": "Medium",
                "cycle_time": 30,
            },
        )

        res = self.client.get(f"/api/vsm/{self.vsm_id}/")
        data = res.json()
        bn = data["bottleneck"]
        self.assertIsNotNone(bn)
        self.assertEqual(bn["bottleneck_step_name"], "Slow")
        self.assertEqual(bn["bottleneck_ct"], 60)
        # Throughput = 3600/60 = 60 units/hr
        self.assertAlmostEqual(bn["theoretical_throughput"], 60.0, places=0)

    def test_no_bottleneck_with_no_steps(self):
        """Empty VSM has no bottleneck."""
        res = self.client.get(f"/api/vsm/{self.vsm_id}/")
        self.assertIsNone(res.json()["bottleneck"])

    def test_takt_flags_on_steps(self):
        """Steps exceeding takt time get flagged."""
        # Set takt time to 40s
        _put(self.client, f"/api/vsm/{self.vsm_id}/update/", {"takt_time": 40})
        _post(
            self.client,
            f"/api/vsm/{self.vsm_id}/process-step/",
            {
                "name": "Under Takt",
                "cycle_time": 30,
            },
        )
        _post(
            self.client,
            f"/api/vsm/{self.vsm_id}/process-step/",
            {
                "name": "Over Takt",
                "cycle_time": 50,
            },
        )

        vsm = self.client.get(f"/api/vsm/{self.vsm_id}/").json()["vsm"]
        steps = {s["name"]: s for s in vsm["process_steps"]}
        self.assertFalse(steps["Under Takt"]["flags"]["exceeds_takt"])
        self.assertTrue(steps["Over Takt"]["flags"]["exceeds_takt"])
        self.assertAlmostEqual(
            steps["Over Takt"]["flags"]["takt_ratio"], 1.25, places=2
        )

    def test_takt_time_validation(self):
        """Takt time must be positive."""
        res = _put(self.client, f"/api/vsm/{self.vsm_id}/update/", {"takt_time": -5})
        self.assertEqual(res.status_code, 400)
        self.assertIn("positive", res.json()["error"].lower())

        res = _put(self.client, f"/api/vsm/{self.vsm_id}/update/", {"takt_time": 0})
        self.assertEqual(res.status_code, 400)

    def test_work_center_parallel_effective_ct(self):
        """Parallel machines in a work center reduce effective cycle time."""
        # Two machines in same work center, each 60s CT
        # Effective CT = 1 / (1/60 + 1/60) = 30s
        _put(
            self.client,
            f"/api/vsm/{self.vsm_id}/update/",
            {
                "process_steps": [
                    {
                        "id": "m1",
                        "name": "Machine 1",
                        "cycle_time": 60,
                        "x": 100,
                        "y": 300,
                        "work_center_id": "wc1",
                    },
                    {
                        "id": "m2",
                        "name": "Machine 2",
                        "cycle_time": 60,
                        "x": 200,
                        "y": 300,
                        "work_center_id": "wc1",
                    },
                    {
                        "id": "finish",
                        "name": "Finishing",
                        "cycle_time": 45,
                        "x": 300,
                        "y": 300,
                    },
                ],
                "work_centers": [{"id": "wc1", "name": "Machining Center"}],
            },
        )

        vsm = self.client.get(f"/api/vsm/{self.vsm_id}/").json()["vsm"]
        # Total process time: 30 (effective WC) + 45 (standalone) = 75
        self.assertAlmostEqual(vsm["total_process_time"], 75.0, places=1)

        # Bottleneck should be Finishing (45s > 30s effective)
        bn = self.client.get(f"/api/vsm/{self.vsm_id}/").json()["bottleneck"]
        self.assertEqual(bn["bottleneck_step_name"], "Finishing")


# =============================================================================
# Future State and Comparison
# =============================================================================


@SECURE_OFF
class VSMFutureStateTest(TestCase):
    """Future state creation, pairing, and comparison."""

    def setUp(self):
        self.user = _make_user("future@test.com")
        self.client.force_login(self.user)
        # Create current-state VSM with a project
        self.drf_client = APIClient()
        self.drf_client.force_authenticate(self.user)
        proj = self.drf_client.post(
            "/api/core/projects/",
            {
                "title": "CI Project",
                "problem_statement": "Reduce lead time",
                "domain": "manufacturing",
                "methodology": "dmaic",
            },
            format="json",
        ).json()
        self.project_id = proj["id"]

        # Create VSM linked to project with steps + inventory
        self.vsm_id = _post(
            self.client,
            "/api/vsm/create/",
            {
                "name": "Current State",
                "project_id": self.project_id,
            },
        ).json()["id"]

        _post(
            self.client,
            f"/api/vsm/{self.vsm_id}/process-step/",
            {
                "name": "Cut",
                "cycle_time": 30,
                "changeover_time": 300,
                "uptime": 90,
                "operators": 1,
                "x": 100,
                "y": 300,
            },
        )
        _post(
            self.client,
            f"/api/vsm/{self.vsm_id}/process-step/",
            {
                "name": "Weld",
                "cycle_time": 60,
                "changeover_time": 600,
                "uptime": 85,
                "operators": 2,
                "x": 300,
                "y": 300,
            },
        )
        _post(
            self.client,
            f"/api/vsm/{self.vsm_id}/inventory/",
            {
                "before_step_id": "weld_step",
                "quantity": 200,
                "days_of_supply": 3.0,
            },
        )
        _post(
            self.client,
            f"/api/vsm/{self.vsm_id}/kaizen/",
            {
                "text": "Reduce weld changeover",
                "priority": "high",
                "x": 300,
                "y": 250,
            },
        )

    def test_create_future_state(self):
        """Future state clones current and pairs them."""
        res = _post(self.client, f"/api/vsm/{self.vsm_id}/future-state/")
        self.assertEqual(res.status_code, 200)
        future = res.json()["future_state"]
        self.assertEqual(future["status"], "future")
        self.assertIn("Future State", future["name"])
        self.assertEqual(future["paired_with_id"], self.vsm_id)
        self.assertEqual(future["project_id"], self.project_id)
        # Steps, inventory, and kaizen bursts should be cloned
        self.assertEqual(len(future["process_steps"]), 2)
        self.assertEqual(len(future["inventory"]), 1)
        self.assertEqual(len(future["kaizen_bursts"]), 1)

    def test_compare_current_vs_future(self):
        """Comparison shows metric deltas between current and future."""
        # Create future state
        future_res = _post(self.client, f"/api/vsm/{self.vsm_id}/future-state/")
        future_id = future_res.json()["future_state"]["id"]

        # Improve future state: reduce weld cycle time and remove inventory
        _put(
            self.client,
            f"/api/vsm/{future_id}/update/",
            {
                "process_steps": [
                    {"id": "s1", "name": "Cut", "cycle_time": 30, "x": 100, "y": 300},
                    {"id": "s2", "name": "Weld", "cycle_time": 40, "x": 300, "y": 300},
                ],
                "inventory": [],  # Eliminated WIP
            },
        )

        # Compare from current perspective
        res = self.client.get(f"/api/vsm/{self.vsm_id}/compare/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIsNotNone(data["future"])
        comp = data["comparison"]
        self.assertIsNotNone(comp)
        # Lead time should improve (less inventory + faster cycle)
        self.assertGreater(comp["lead_time"]["improvement"], 0)
        # Inventory reduced
        self.assertGreater(
            comp["inventory_reduction"]["current_count"],
            comp["inventory_reduction"]["future_count"],
        )

    def test_compare_no_future_returns_null(self):
        """Comparison with no future state returns null comparison."""
        res = self.client.get(f"/api/vsm/{self.vsm_id}/compare/")
        data = res.json()
        self.assertIsNone(data["future"])
        self.assertIsNone(data["comparison"])


# =============================================================================
# Project Integration
# =============================================================================


@SECURE_OFF
class VSMProjectIntegrationTest(TestCase):
    """VSM ↔ Project linkage and hub visibility."""

    def setUp(self):
        self.user = _make_user("projvsm@test.com")
        self.client.force_login(self.user)
        self.drf_client = APIClient()
        self.drf_client.force_authenticate(self.user)
        proj = self.drf_client.post(
            "/api/core/projects/",
            {
                "title": "VSM Project",
                "problem_statement": "Improve flow",
                "domain": "manufacturing",
                "methodology": "dmaic",
            },
            format="json",
        ).json()
        self.project_id = proj["id"]

    def test_create_vsm_linked_to_project(self):
        res = _post(
            self.client,
            "/api/vsm/create/",
            {
                "name": "Linked VSM",
                "project_id": self.project_id,
            },
        )
        vsm = res.json()["vsm"]
        self.assertEqual(vsm["project_id"], self.project_id)

    def test_list_filter_by_project(self):
        _post(
            self.client,
            "/api/vsm/create/",
            {
                "name": "In Project",
                "project_id": self.project_id,
            },
        )
        _post(self.client, "/api/vsm/create/", {"name": "Standalone"})

        res = self.client.get(f"/api/vsm/?project_id={self.project_id}")
        maps = res.json()["maps"]
        self.assertEqual(len(maps), 1)
        self.assertEqual(maps[0]["name"], "In Project")

    def test_project_hub_shows_vsm(self):
        """VSM linked to project appears in the project hub."""
        _post(
            self.client,
            "/api/vsm/create/",
            {
                "name": "Hub VSM",
                "project_id": self.project_id,
            },
        )
        res = self.drf_client.get(f"/api/core/projects/{self.project_id}/hub/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["counts"]["vsm_maps"], 1)
        self.assertEqual(len(data["tools"]["vsm_maps"]), 1)
        self.assertEqual(data["tools"]["vsm_maps"][0]["name"], "Hub VSM")

    def test_update_project_link(self):
        """Can link and unlink VSM from project via update."""
        vsm_id = _post(self.client, "/api/vsm/create/", {"name": "Floater"}).json()[
            "id"
        ]
        # Link
        _put(
            self.client,
            f"/api/vsm/{vsm_id}/update/",
            {
                "project_id": self.project_id,
            },
        )
        vsm = self.client.get(f"/api/vsm/{vsm_id}/").json()["vsm"]
        self.assertEqual(vsm["project_id"], self.project_id)
        # Unlink
        _put(self.client, f"/api/vsm/{vsm_id}/update/", {"project_id": None})
        vsm = self.client.get(f"/api/vsm/{vsm_id}/").json()["vsm"]
        self.assertIsNone(vsm["project_id"])

    def test_create_vsm_invalid_project(self):
        """Linking to nonexistent project returns 404."""
        import uuid

        res = _post(
            self.client,
            "/api/vsm/create/",
            {
                "name": "Bad Link",
                "project_id": str(uuid.uuid4()),
            },
        )
        self.assertEqual(res.status_code, 404)


# =============================================================================
# User Isolation
# =============================================================================


@SECURE_OFF
class VSMIsolationTest(TestCase):
    """VSM ownership and cross-user isolation."""

    def setUp(self):
        self.user_a = _make_user("alice_vsm@test.com")
        self.user_b = _make_user("bob_vsm@test.com")

    def test_user_cannot_see_others_vsm(self):
        """User B cannot list or access User A's VSMs."""
        self.client.force_login(self.user_a)
        vsm_id = _post(self.client, "/api/vsm/create/", {"name": "Alice's VSM"}).json()[
            "id"
        ]

        self.client.force_login(self.user_b)
        # List should be empty for Bob
        maps = self.client.get("/api/vsm/").json()["maps"]
        self.assertEqual(len(maps), 0)
        # Direct access returns 404
        self.assertEqual(self.client.get(f"/api/vsm/{vsm_id}/").status_code, 404)

    def test_user_cannot_update_others_vsm(self):
        self.client.force_login(self.user_a)
        vsm_id = _post(self.client, "/api/vsm/create/", {"name": "Alice's VSM"}).json()[
            "id"
        ]

        self.client.force_login(self.user_b)
        res = _put(self.client, f"/api/vsm/{vsm_id}/update/", {"name": "Hacked"})
        self.assertEqual(res.status_code, 404)

    def test_user_cannot_delete_others_vsm(self):
        self.client.force_login(self.user_a)
        vsm_id = _post(self.client, "/api/vsm/create/", {"name": "Alice's VSM"}).json()[
            "id"
        ]

        self.client.force_login(self.user_b)
        res = self.client.delete(f"/api/vsm/{vsm_id}/delete/")
        self.assertEqual(res.status_code, 404)
        # Verify still exists for Alice
        self.client.force_login(self.user_a)
        self.assertEqual(self.client.get(f"/api/vsm/{vsm_id}/").status_code, 200)


# =============================================================================
# Metric Snapshots
# =============================================================================


@SECURE_OFF
class VSMMetricSnapshotTest(TestCase):
    """Metric snapshot history tracking."""

    def setUp(self):
        self.user = _make_user("snapshot@test.com")
        self.client.force_login(self.user)
        self.vsm_id = _post(self.client, "/api/vsm/create/").json()["id"]

    def test_snapshot_recorded_on_step_add(self):
        """Adding a step records a metric snapshot."""
        _post(
            self.client,
            f"/api/vsm/{self.vsm_id}/process-step/",
            {
                "name": "Step 1",
                "cycle_time": 30,
            },
        )
        vsm = self.client.get(f"/api/vsm/{self.vsm_id}/").json()["vsm"]
        self.assertGreaterEqual(len(vsm["metric_snapshots"]), 1)
        snap = vsm["metric_snapshots"][-1]
        self.assertIn("timestamp", snap)
        self.assertAlmostEqual(snap["process_time"], 30.0, places=1)

    def test_snapshot_not_duplicated_if_unchanged(self):
        """Updating without metric change doesn't add a snapshot."""
        _post(
            self.client,
            f"/api/vsm/{self.vsm_id}/process-step/",
            {
                "name": "Step 1",
                "cycle_time": 30,
            },
        )
        snap_count_1 = len(
            self.client.get(f"/api/vsm/{self.vsm_id}/").json()["vsm"][
                "metric_snapshots"
            ]
        )
        # Update only name — no metric change
        _put(self.client, f"/api/vsm/{self.vsm_id}/update/", {"name": "Renamed"})
        snap_count_2 = len(
            self.client.get(f"/api/vsm/{self.vsm_id}/").json()["vsm"][
                "metric_snapshots"
            ]
        )
        self.assertEqual(snap_count_1, snap_count_2)

    def test_snapshot_added_on_metric_change(self):
        """Adding inventory changes lead time and should add a snapshot."""
        _post(
            self.client,
            f"/api/vsm/{self.vsm_id}/process-step/",
            {
                "name": "Step 1",
                "cycle_time": 30,
            },
        )
        snap_count_1 = len(
            self.client.get(f"/api/vsm/{self.vsm_id}/").json()["vsm"][
                "metric_snapshots"
            ]
        )
        _post(
            self.client,
            f"/api/vsm/{self.vsm_id}/inventory/",
            {
                "before_step_id": "s1",
                "quantity": 100,
                "days_of_supply": 5.0,
            },
        )
        snap_count_2 = len(
            self.client.get(f"/api/vsm/{self.vsm_id}/").json()["vsm"][
                "metric_snapshots"
            ]
        )
        self.assertGreater(snap_count_2, snap_count_1)


# =============================================================================
# Tier Gating
# =============================================================================


@SECURE_OFF
class VSMTierGatingTest(TestCase):
    """Free users cannot access VSM endpoints."""

    def setUp(self):
        self.free_user = _make_user("free_vsm@test.com", tier=Tier.FREE)
        self.paid_user = _make_user("paid_vsm@test.com", tier=Tier.PRO)

    def test_free_user_blocked_from_list(self):
        self.client.force_login(self.free_user)
        res = self.client.get("/api/vsm/")
        self.assertEqual(res.status_code, 403)
        self.assertIn("Upgrade", res.json()["error"])

    def test_free_user_blocked_from_create(self):
        self.client.force_login(self.free_user)
        res = _post(self.client, "/api/vsm/create/", {"name": "Blocked"})
        self.assertEqual(res.status_code, 403)

    def test_paid_user_can_create(self):
        self.client.force_login(self.paid_user)
        res = _post(self.client, "/api/vsm/create/", {"name": "Allowed"})
        self.assertEqual(res.status_code, 200)

    def test_unauthenticated_blocked(self):
        res = self.client.get("/api/vsm/")
        self.assertEqual(res.status_code, 401)


# =============================================================================
# Generate Proposals (Enterprise-only Hoshin integration)
# =============================================================================


@SECURE_OFF
class VSMGenerateProposalsTest(TestCase):
    """Proposal generation from kaizen bursts (enterprise only)."""

    def setUp(self):
        self.user = _make_user("hoshin@test.com", tier=Tier.ENTERPRISE)
        self.client.force_login(self.user)
        self.drf_client = APIClient()
        self.drf_client.force_authenticate(self.user)

        # Create project
        proj = self.drf_client.post(
            "/api/core/projects/",
            {
                "title": "Hoshin Project",
                "problem_statement": "Reduce lead time",
                "domain": "manufacturing",
                "methodology": "dmaic",
            },
            format="json",
        ).json()
        self.project_id = proj["id"]

        # Create current-state VSM with steps
        self.vsm_id = _post(
            self.client,
            "/api/vsm/create/",
            {
                "name": "Current State",
                "project_id": self.project_id,
            },
        ).json()["id"]

        _post(
            self.client,
            f"/api/vsm/{self.vsm_id}/process-step/",
            {
                "name": "Assembly",
                "cycle_time": 60,
                "changeover_time": 600,
                "uptime": 85,
                "operators": 3,
                "x": 200,
                "y": 300,
            },
        )
        _post(
            self.client,
            f"/api/vsm/{self.vsm_id}/process-step/",
            {
                "name": "Inspection",
                "cycle_time": 30,
                "changeover_time": 0,
                "uptime": 100,
                "operators": 1,
                "x": 400,
                "y": 300,
            },
        )

    def _create_future_with_improvements(self):
        """Create future state and improve it."""
        future = _post(self.client, f"/api/vsm/{self.vsm_id}/future-state/").json()[
            "future_state"
        ]
        future_id = future["id"]

        # Improve: reduce Assembly CT from 60→40, operators 3→2, add kaizen
        _put(
            self.client,
            f"/api/vsm/{future_id}/update/",
            {
                "process_steps": [
                    {
                        "id": "s1",
                        "name": "Assembly",
                        "cycle_time": 40,
                        "changeover_time": 300,
                        "uptime": 95,
                        "operators": 2,
                        "x": 200,
                        "y": 300,
                    },
                    {
                        "id": "s2",
                        "name": "Inspection",
                        "cycle_time": 30,
                        "changeover_time": 0,
                        "uptime": 100,
                        "operators": 1,
                        "x": 400,
                        "y": 300,
                    },
                ],
                "kaizen_bursts": [
                    {
                        "id": "k1",
                        "text": "SMED changeover reduction",
                        "priority": "high",
                        "x": 200,
                        "y": 250,
                    },
                    {
                        "id": "k2",
                        "text": "Automate inspection",
                        "priority": "medium",
                        "x": 400,
                        "y": 250,
                    },
                ],
            },
        )
        return future_id

    def test_generate_proposals(self):
        """Proposals generated from current→future deltas."""
        self._create_future_with_improvements()
        res = _post(
            self.client,
            f"/api/vsm/{self.vsm_id}/generate-proposals/",
            {
                "annual_volume": 100000,
                "cost_per_unit": 50.0,
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["vsm_id"], self.vsm_id)
        self.assertGreater(data["count"], 0)
        proposals = data["proposals"]
        # At least one proposal should match Assembly improvement
        assembly_prop = next(
            (p for p in proposals if p["process_step"] == "Assembly"), None
        )
        self.assertIsNotNone(assembly_prop)
        self.assertTrue(assembly_prop["has_current_match"])
        self.assertGreater(assembly_prop["estimated_annual_savings"], 0)
        self.assertIn("suggested_title", assembly_prop)

    def test_proposals_include_confidence_intervals(self):
        """Monte Carlo produces confidence intervals."""
        self._create_future_with_improvements()
        data = _post(
            self.client,
            f"/api/vsm/{self.vsm_id}/generate-proposals/",
            {
                "annual_volume": 100000,
                "cost_per_unit": 50.0,
            },
        ).json()
        prop = next(p for p in data["proposals"] if p["has_current_match"])
        # Monte Carlo fields
        self.assertIn("median_savings", prop)
        self.assertIn("lower_5", prop)
        self.assertIn("upper_95", prop)
        self.assertIn("p_positive", prop)
        # p_positive should be between 0 and 1
        self.assertGreaterEqual(prop["p_positive"], 0)
        self.assertLessEqual(prop["p_positive"], 1)

    def test_no_future_state_returns_400(self):
        """Proposals require a future state."""
        res = _post(self.client, f"/api/vsm/{self.vsm_id}/generate-proposals/")
        self.assertEqual(res.status_code, 400)
        self.assertIn("future state", res.json()["error"].lower())

    def test_no_kaizen_bursts_returns_400(self):
        """Future state with no bursts returns 400."""
        future = _post(self.client, f"/api/vsm/{self.vsm_id}/future-state/").json()[
            "future_state"
        ]
        # Clear kaizen bursts from future
        _put(self.client, f"/api/vsm/{future['id']}/update/", {"kaizen_bursts": []})
        res = _post(self.client, f"/api/vsm/{self.vsm_id}/generate-proposals/")
        self.assertEqual(res.status_code, 400)
        self.assertIn("kaizen", res.json()["error"].lower())

    def test_non_enterprise_blocked(self):
        """PRO users cannot generate proposals (hoshin_kanri is enterprise only)."""
        pro_user = _make_user("pro_vsm@test.com", tier=Tier.PRO)
        self.client.force_login(pro_user)
        # PRO user needs their own VSM
        vsm_id = _post(self.client, "/api/vsm/create/").json()["id"]
        res = _post(self.client, f"/api/vsm/{vsm_id}/generate-proposals/")
        self.assertEqual(res.status_code, 403)


# =============================================================================
# Full Lifecycle: Current → Future → Compare → Proposals
# =============================================================================


@SECURE_OFF
class VSMFullLifecycleTest(TestCase):
    """End-to-end: build current state, clone future, improve, compare, propose."""

    def setUp(self):
        self.user = _make_user("lifecycle@test.com", tier=Tier.ENTERPRISE)
        self.client.force_login(self.user)
        self.drf_client = APIClient()
        self.drf_client.force_authenticate(self.user)

    def test_full_lifecycle(self):
        """Current → add steps/inv → future → improve → compare → proposals."""
        # 1. Create project
        proj = self.drf_client.post(
            "/api/core/projects/",
            {
                "title": "Lean Transformation",
                "problem_statement": "Reduce lead time from 15 to 5 days",
                "domain": "manufacturing",
                "methodology": "dmaic",
            },
            format="json",
        ).json()

        # 2. Create current-state VSM
        vsm_id = _post(
            self.client,
            "/api/vsm/create/",
            {
                "name": "Press Line Current",
                "project_id": proj["id"],
                "product_family": "Automotive Panels",
                "customer_name": "OEM Corp",
                "customer_demand": "460 units/day",
            },
        ).json()["id"]

        # 3. Add process steps
        _post(
            self.client,
            f"/api/vsm/{vsm_id}/process-step/",
            {
                "name": "Stamping",
                "cycle_time": 1,
                "changeover_time": 3600,
                "uptime": 80,
                "operators": 1,
                "x": 100,
                "y": 300,
            },
        )
        _post(
            self.client,
            f"/api/vsm/{vsm_id}/process-step/",
            {
                "name": "Welding",
                "cycle_time": 39,
                "changeover_time": 600,
                "uptime": 90,
                "operators": 2,
                "x": 300,
                "y": 300,
            },
        )
        _post(
            self.client,
            f"/api/vsm/{vsm_id}/process-step/",
            {
                "name": "Assembly",
                "cycle_time": 62,
                "changeover_time": 0,
                "uptime": 100,
                "operators": 3,
                "x": 500,
                "y": 300,
            },
        )

        # 4. Add inventory buffers
        _post(
            self.client,
            f"/api/vsm/{vsm_id}/inventory/",
            {
                "quantity": 4600,
                "days_of_supply": 5.0,
            },
        )
        _post(
            self.client,
            f"/api/vsm/{vsm_id}/inventory/",
            {
                "quantity": 2400,
                "days_of_supply": 3.0,
            },
        )

        # 5. Verify current state metrics
        current = self.client.get(f"/api/vsm/{vsm_id}/").json()["vsm"]
        self.assertAlmostEqual(current["total_process_time"], 102.0, places=0)
        self.assertGreater(
            current["total_lead_time"], 8.0
        )  # 5+3 days inventory + process

        # 6. Create future state
        future_id = _post(self.client, f"/api/vsm/{vsm_id}/future-state/").json()[
            "future_state"
        ]["id"]

        # 7. Improve future: reduce changeover, cut inventory, add kaizen
        _put(
            self.client,
            f"/api/vsm/{future_id}/update/",
            {
                "process_steps": [
                    {
                        "id": "s1",
                        "name": "Stamping",
                        "cycle_time": 1,
                        "changeover_time": 600,
                        "uptime": 95,
                        "operators": 1,
                        "x": 100,
                        "y": 300,
                    },
                    {
                        "id": "s2",
                        "name": "Welding",
                        "cycle_time": 39,
                        "changeover_time": 300,
                        "uptime": 95,
                        "operators": 2,
                        "x": 300,
                        "y": 300,
                    },
                    {
                        "id": "s3",
                        "name": "Assembly",
                        "cycle_time": 55,
                        "changeover_time": 0,
                        "uptime": 100,
                        "operators": 2,
                        "x": 500,
                        "y": 300,
                    },
                ],
                "inventory": [
                    {"id": "i1", "quantity": 920, "days_of_supply": 1.0},
                ],
                "kaizen_bursts": [
                    {
                        "id": "k1",
                        "text": "SMED on stamping press",
                        "priority": "high",
                        "x": 100,
                        "y": 250,
                    },
                    {
                        "id": "k2",
                        "text": "Reduce WIP with kanban",
                        "priority": "high",
                        "x": 200,
                        "y": 250,
                    },
                    {
                        "id": "k3",
                        "text": "Cross-train assembly operators",
                        "priority": "medium",
                        "x": 500,
                        "y": 250,
                    },
                ],
            },
        )

        # 8. Compare
        comp = self.client.get(f"/api/vsm/{vsm_id}/compare/").json()
        self.assertIsNotNone(comp["comparison"])
        self.assertGreater(comp["comparison"]["lead_time"]["improvement"], 0)

        # 9. Generate proposals
        proposals = _post(
            self.client,
            f"/api/vsm/{vsm_id}/generate-proposals/",
            {
                "annual_volume": 168000,
                "cost_per_unit": 45.0,
            },
        ).json()
        self.assertEqual(proposals["count"], 3)
        self.assertTrue(all("suggested_title" in p for p in proposals["proposals"]))
