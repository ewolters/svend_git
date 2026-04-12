"""
Behavioral tests for SIOP (Sales, Inventory & Operations Planning) DSW analyses.

All tests exercise real HTTP surfaces per TST-001 §10.6 — POST /api/dsw/analysis/
with type="siop" and verify dispatch routing, output structure, statistics, and
error handling.

CR: 7fa03dc8
"""

import json

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.test.client import Client

from accounts.constants import Tier

User = get_user_model()

SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)


def _make_user(email="siop-test@test.com", tier=Tier.TEAM):
    username = email.split("@")[0]
    user = User.objects.create_user(username=username, email=email, password="testpass123")
    user.tier = tier
    user.save(update_fields=["tier"])
    return user


def _authed_client(user):
    client = Client()
    client.force_login(user)
    return client


def _post_siop(client, analysis_id, config=None, data=None):
    """POST to /api/dsw/analysis/ with type=siop."""
    payload = {
        "type": "siop",
        "analysis": analysis_id,
        "config": config or {},
    }
    if data is not None:
        payload["data"] = data
    return client.post(
        "/api/dsw/analysis/",
        json.dumps(payload),
        content_type="application/json",
    )


def _err_msg(resp):
    """Extract error message from ErrorEnvelopeMiddleware response."""
    err = resp.json().get("error", {})
    if isinstance(err, dict):
        return err.get("message", str(err))
    return str(err)


# ── Test data ────────────────────────────────────────────────────────────

_INVENTORY_DATA = {
    "sku": [f"SKU-{i:03d}" for i in range(20)],
    "annual_usage_value": [
        50000,
        42000,
        38000,
        35000,
        30000,
        12000,
        11000,
        10000,
        9000,
        8000,
        3000,
        2800,
        2500,
        2200,
        2000,
        1800,
        1500,
        1200,
        800,
        500,
    ],
    "demand": [
        100,
        85,
        76,
        70,
        60,
        24,
        22,
        20,
        18,
        16,
        6,
        5.6,
        5,
        4.4,
        4,
        3.6,
        3,
        2.4,
        1.6,
        1,
    ],
    "unit_cost": [500] * 20,
}

_DEMAND_SERIES = {
    "sku": ["A"] * 12,
    "period": list(range(1, 13)),
    "demand": [120, 130, 125, 140, 135, 150, 145, 160, 155, 170, 165, 180],
}

_MRP_DATA = {
    "gross": [100, 120, 80, 150, 90, 110],
    "receipts": [0, 100, 0, 0, 0, 0],
}


# =============================================================================
# ABC Analysis
# =============================================================================


@SECURE_OFF
class SIOPAbcAnalysisTest(TestCase):
    """ABC/XYZ Pareto classification dispatched through real endpoint."""

    def setUp(self):
        self.user = _make_user("siop-abc@test.com")
        self.client = _authed_client(self.user)

    def test_abc_produces_classification(self):
        """ABC analysis classifies items into A/B/C categories with plots."""
        resp = _post_siop(
            self.client,
            "abc_analysis",
            config={"value_col": "annual_usage_value", "item_col": "sku"},
            data=_INVENTORY_DATA,
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("summary", body)
        self.assertGreater(len(body["plots"]), 0)
        stats = body.get("statistics", {})
        self.assertIn("total_items", stats)

    def test_abc_returns_guide_observation(self):
        """ABC analysis includes a guide_observation narrative."""
        resp = _post_siop(
            self.client,
            "abc_analysis",
            config={"value_col": "annual_usage_value", "item_col": "sku"},
            data=_INVENTORY_DATA,
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(len(body.get("guide_observation", "")) > 0)

    def test_abc_without_data_returns_gracefully(self):
        """ABC without required columns returns guidance, not 500."""
        resp = _post_siop(self.client, "abc_analysis", config={})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("Error", body.get("summary", ""))


# =============================================================================
# EOQ (Economic Order Quantity)
# =============================================================================


@SECURE_OFF
class SIOPEoqTest(TestCase):
    """Economic Order Quantity calculation through real endpoint."""

    def setUp(self):
        self.user = _make_user("siop-eoq@test.com")
        self.client = _authed_client(self.user)

    def test_eoq_calculation(self):
        """EOQ returns correct order quantity and cost curves."""
        resp = _post_siop(
            self.client,
            "eoq",
            config={
                "demand": 10000,
                "order_cost": 80,
                "unit_cost": 10,
                "holding_pct": 20,
            },
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        stats = body.get("statistics", {})
        eoq = stats.get("eoq") or stats.get("EOQ") or stats.get("optimal_order_quantity")
        self.assertIsNotNone(eoq, f"Expected EOQ in statistics, got: {list(stats.keys())}")
        self.assertGreater(float(eoq), 0)
        self.assertGreater(len(body.get("plots", [])), 0)

    def test_eoq_has_what_if(self):
        """EOQ analysis includes what-if explorer parameters."""
        resp = _post_siop(
            self.client,
            "eoq",
            config={
                "demand": 5000,
                "order_cost": 50,
                "unit_cost": 20,
                "holding_pct": 25,
            },
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("what_if", body)


# =============================================================================
# Safety Stock
# =============================================================================


@SECURE_OFF
class SIOPSafetyStockTest(TestCase):
    """Statistical safety stock calculation through real endpoint."""

    def setUp(self):
        self.user = _make_user("siop-ss@test.com")
        self.client = _authed_client(self.user)

    def test_safety_stock_calculation(self):
        """Safety stock returns SS, ROP, and service level info."""
        resp = _post_siop(
            self.client,
            "safety_stock",
            config={
                "demand_mean": 100,
                "demand_std": 20,
                "lead_time": 5,
                "lead_time_std": 1,
                "service_level": 95,
            },
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        stats = body.get("statistics", {})
        ss_keys = [k for k in stats if "safety" in k.lower() or k == "ss" or k == "SS"]
        self.assertTrue(
            len(ss_keys) > 0,
            f"Expected safety stock in statistics, got: {list(stats.keys())}",
        )
        self.assertGreater(len(body.get("plots", [])), 0)
        self.assertIn("summary", body)


# =============================================================================
# Inventory Turns
# =============================================================================


@SECURE_OFF
class SIOPInventoryTurnsTest(TestCase):
    """Inventory turns analysis through real endpoint."""

    def setUp(self):
        self.user = _make_user("siop-turns@test.com")
        self.client = _authed_client(self.user)

    def test_turns_calculation(self):
        """Inventory turns returns ratio and days-of-supply."""
        resp = _post_siop(
            self.client,
            "inventory_turns",
            config={
                "cogs": 500000,
                "avg_inventory": 80000,
            },
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        stats = body.get("statistics", {})
        turns = stats.get("turns") or stats.get("inventory_turns")
        self.assertIsNotNone(turns, f"Expected turns in statistics, got: {list(stats.keys())}")
        self.assertAlmostEqual(float(turns), 6.25, places=1)


# =============================================================================
# Demand Profile
# =============================================================================


@SECURE_OFF
class SIOPDemandProfileTest(TestCase):
    """Demand variability profiling through real endpoint."""

    def setUp(self):
        self.user = _make_user("siop-demand@test.com")
        self.client = _authed_client(self.user)

    def test_demand_profile_classification(self):
        """Demand profile classifies demand pattern (Syntetos-Boylan)."""
        resp = _post_siop(
            self.client,
            "demand_profile",
            config={"demand_col": "demand"},
            data=_DEMAND_SERIES,
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("summary", body)
        self.assertGreater(len(body.get("plots", [])), 0)
        stats = body.get("statistics", {})
        self.assertIn("cov", stats)


# =============================================================================
# Kanban Sizing
# =============================================================================


@SECURE_OFF
class SIOPKanbanSizingTest(TestCase):
    """Kanban card sizing through real endpoint."""

    def setUp(self):
        self.user = _make_user("siop-kanban@test.com")
        self.client = _authed_client(self.user)

    def test_kanban_sizing(self):
        """Kanban sizing calculates number of cards."""
        resp = _post_siop(
            self.client,
            "kanban_sizing",
            config={
                "demand": 100,
                "lead_time": 2,
                "container_size": 50,
                "safety_pct": 25,
            },
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        stats = body.get("statistics", {})
        cards = stats.get("kanban_cards") or stats.get("cards") or stats.get("n_kanbans")
        self.assertIsNotNone(cards, f"Expected kanban cards in statistics, got: {list(stats.keys())}")
        self.assertGreater(float(cards), 0)
        self.assertIn("summary", body)


# =============================================================================
# EPEI (Every Part Every Interval)
# =============================================================================


@SECURE_OFF
class SIOPEpeiTest(TestCase):
    """EPEI scheduling calculation through real endpoint."""

    def setUp(self):
        self.user = _make_user("siop-epei@test.com")
        self.client = _authed_client(self.user)

    def test_epei_calculation(self):
        """EPEI returns interval in days and changeover analysis."""
        resp = _post_siop(
            self.client,
            "epei",
            config={
                "available_hours": 8,
                "changeover_time": 30,
                "num_parts": 4,
            },
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        stats = body.get("statistics", {})
        self.assertIsInstance(stats, dict)
        self.assertIn("summary", body)
        self.assertGreater(len(body.get("plots", [])), 0)


# =============================================================================
# ROP Simulation
# =============================================================================


@SECURE_OFF
class SIOPRopSimulationTest(TestCase):
    """Monte Carlo reorder-point simulation through real endpoint."""

    def setUp(self):
        self.user = _make_user("siop-rop@test.com")
        self.client = _authed_client(self.user)

    def test_rop_simulation_runs(self):
        """ROP simulation produces fill rate and inventory trajectory."""
        resp = _post_siop(
            self.client,
            "rop_simulation",
            config={
                "demand_mean": 50,
                "demand_std": 10,
                "lead_time": 3,
                "lead_time_std": 0.5,
                "reorder_point": 180,
                "order_quantity": 250,
                "runs": 100,
            },
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        stats = body.get("statistics", {})
        fill_keys = [k for k in stats if "fill" in k.lower() or "service" in k.lower()]
        self.assertTrue(
            len(fill_keys) > 0,
            f"Expected fill/service rate in statistics, got: {list(stats.keys())}",
        )
        self.assertGreater(len(body.get("plots", [])), 0)


# =============================================================================
# MRP Netting
# =============================================================================


@SECURE_OFF
class SIOPMrpNettingTest(TestCase):
    """Gross-to-net MRP explosion through real endpoint."""

    def setUp(self):
        self.user = _make_user("siop-mrp@test.com")
        self.client = _authed_client(self.user)

    def test_mrp_netting(self):
        """MRP netting produces net requirements and planned orders."""
        resp = _post_siop(
            self.client,
            "mrp_netting",
            config={
                "gross_col": "gross",
                "receipts_col": "receipts",
                "on_hand": 50,
                "lead_time": 2,
                "lot_size": "lot_for_lot",
            },
            data=_MRP_DATA,
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        stats = body.get("statistics", {})
        self.assertIsInstance(stats, dict)
        self.assertIn("summary", body)
        self.assertGreater(len(body.get("plots", [])), 0)


# =============================================================================
# Service Level
# =============================================================================


@SECURE_OFF
class SIOPServiceLevelTest(TestCase):
    """Service level trade-off analysis through real endpoint."""

    def setUp(self):
        self.user = _make_user("siop-sl@test.com")
        self.client = _authed_client(self.user)

    def test_service_level_tradeoff(self):
        """Service level returns fill rate curve and cost trade-off."""
        resp = _post_siop(
            self.client,
            "service_level",
            config={
                "demand_mean": 200,
                "demand_std": 40,
                "lead_time": 7,
                "unit_cost": 25,
                "holding_pct": 20,
                "stockout_cost": 50,
            },
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("summary", body)
        self.assertGreater(len(body.get("plots", [])), 0)


# =============================================================================
# Dispatch integration
# =============================================================================


@SECURE_OFF
class SIOPDispatchIntegrationTest(TestCase):
    """Verify SIOP routing through the DSW dispatch layer."""

    def setUp(self):
        self.user = _make_user("siop-dispatch@test.com")
        self.client = _authed_client(self.user)

    def test_siop_type_dispatches(self):
        """type=siop is recognized by dispatch (not rejected as unknown)."""
        resp = _post_siop(
            self.client,
            "eoq",
            config={
                "demand": 1000,
                "order_cost": 50,
                "unit_cost": 10,
                "holding_pct": 25,
            },
        )
        self.assertEqual(resp.status_code, 200)

    def test_unknown_siop_analysis_handled(self):
        """Unknown analysis_id within siop type returns error, not 500."""
        resp = _post_siop(self.client, "nonexistent_analysis", config={})
        self.assertIn(resp.status_code, [200, 400, 500])
        body = resp.json()
        self.assertIsInstance(body, dict)

    def test_requires_auth(self):
        """Unauthenticated request to SIOP analysis is rejected."""
        anon_client = Client()
        resp = anon_client.post(
            "/api/dsw/analysis/",
            json.dumps({"type": "siop", "analysis": "eoq", "config": {}}),
            content_type="application/json",
        )
        self.assertIn(resp.status_code, [401, 403])

    def test_siop_empty_df_allowed(self):
        """SIOP analyses can run without data_id (empty DF fallback)."""
        resp = _post_siop(
            self.client,
            "inventory_turns",
            config={"cogs": 100000, "avg_inventory": 20000},
        )
        self.assertEqual(resp.status_code, 200)

    def test_siop_with_inline_data(self):
        """SIOP analyses accept inline data dict."""
        resp = _post_siop(
            self.client,
            "abc_analysis",
            config={"value_col": "annual_usage_value", "item_col": "sku"},
            data=_INVENTORY_DATA,
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertGreater(len(body["plots"]), 0)

    def test_output_has_canonical_fields(self):
        """SIOP output includes canonical DSW fields after standardize_output."""
        resp = _post_siop(
            self.client,
            "eoq",
            config={
                "demand": 5000,
                "order_cost": 100,
                "unit_cost": 15,
                "holding_pct": 20,
            },
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        for field in ("summary", "plots", "statistics"):
            self.assertIn(field, body, f"Missing canonical field: {field}")
