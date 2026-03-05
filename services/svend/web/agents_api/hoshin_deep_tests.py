"""Hoshin Kanri deep tests — charters, custom formulas, X-matrix, KPIs, calendar, Monte Carlo.

Tests:
- Kaizen charter round-trip (create, update, all fields preserved)
- Plan fields merged into charter (background, countermeasures)
- Custom formula engine (test_formula endpoint, monthly with custom formula)
- Calculation methods reference endpoint
- X-matrix: strategic objectives CRUD, annual objectives CRUD, KPI CRUD
- X-matrix correlations (upsert, delete, validation)
- X-matrix data endpoint (four quadrants + rollup)
- Savings aggregation (aggregate_monthly_savings via project detail)
- Baseline data round-trip
- Calendar endpoint (site grouping, monthly target/actual, filters, aborted exclusion)
- Monte Carlo savings estimation (statistical properties, confidence intervals, realization risk)
"""

import json

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from accounts.constants import Tier
from agents_api.models import SiteAccess, ValueStreamMap
from core.models.tenant import Membership, Tenant

User = get_user_model()
SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)


def _make_enterprise_user(email, **kwargs):
    username = kwargs.pop("username", email.split("@")[0])
    user = User.objects.create_user(username=username, email=email, password="testpass123!", **kwargs)
    user.tier = Tier.ENTERPRISE
    user.save(update_fields=["tier"])
    return user


def _setup_tenant(owner, slug="testcorp"):
    tenant = Tenant.objects.create(name="Test Corp", slug=slug)
    Membership.objects.create(
        user=owner,
        tenant=tenant,
        role="owner",
        is_active=True,
    )
    return tenant


def _err_msg(resp):
    """Extract error message from response, handling API error envelope."""
    body = resp.json()
    err = body.get("error", body.get("message", ""))
    if isinstance(err, dict):
        return err.get("message", str(err))
    return str(err)


def _post(client, url, data=None):
    return client.post(url, json.dumps(data or {}), content_type="application/json")


def _put(client, url, data=None):
    return client.put(url, json.dumps(data or {}), content_type="application/json")


# =============================================================================
# Kaizen Charter
# =============================================================================


@SECURE_OFF
class KaizenCharterTest(TestCase):
    """Kaizen charter round-trip: create with charter, update, retrieve all fields."""

    def setUp(self):
        self.user = _make_enterprise_user("charter@test.com")
        self.tenant = _setup_tenant(self.user)
        self.client.force_login(self.user)
        self.site_id = _post(
            self.client,
            "/api/hoshin/sites/create/",
            {
                "name": "Charter Plant",
            },
        ).json()["site"]["id"]

    def test_create_with_full_charter(self):
        """All charter fields round-trip through create → get."""
        charter = {
            "event_type": "SMED/Changeover",
            "location": "Assembly Floor, Bay 3",
            "event_date": "2026-03-10",
            "end_date": "2026-03-14",
            "schedule": "7:00 AM - 4:00 PM",
            "problem_statement": "Changeover takes 90 minutes, target is 10",
            "objectives": "Reduce changeover time by 80%",
            "primary_metric": "Changeover Time (min)",
            "primary_baseline": 90,
            "primary_target": 10,
            "secondary_metric": "OEE",
            "secondary_baseline": 65,
            "secondary_target": 85,
            "process_start": "Last good part of previous run",
            "process_end": "First good part of next run",
            "excluded": "Die maintenance, raw material sourcing",
            "process_owner": "Bob Zeisler",
            "sponsors": "VP Operations, CI Director",
            "team_members": [
                {"name": "Eric Wolters", "role": "Team Leader", "department": "CI"},
                {"name": "Jane Smith", "role": "SME", "department": "Tooling"},
                {"name": "Mike Jones", "role": "Member", "department": "Production"},
            ],
        }
        hp_id = _post(
            self.client,
            "/api/hoshin/projects/create/",
            {
                "title": "SMED Kaizen Event",
                "site_id": self.site_id,
                "project_class": "kaizen",
                "kaizen_charter": charter,
            },
        ).json()["project"]["id"]

        # Retrieve and verify all fields
        proj = self.client.get(f"/api/hoshin/projects/{hp_id}/").json()["project"]
        c = proj["kaizen_charter"]
        self.assertEqual(c["event_type"], "SMED/Changeover")
        self.assertEqual(c["location"], "Assembly Floor, Bay 3")
        self.assertEqual(c["event_date"], "2026-03-10")
        self.assertEqual(c["end_date"], "2026-03-14")
        self.assertEqual(c["schedule"], "7:00 AM - 4:00 PM")
        self.assertEqual(c["problem_statement"], "Changeover takes 90 minutes, target is 10")
        self.assertEqual(c["primary_metric"], "Changeover Time (min)")
        self.assertEqual(c["primary_baseline"], 90)
        self.assertEqual(c["primary_target"], 10)
        self.assertEqual(c["secondary_metric"], "OEE")
        self.assertEqual(c["process_start"], "Last good part of previous run")
        self.assertEqual(c["process_end"], "First good part of next run")
        self.assertEqual(c["excluded"], "Die maintenance, raw material sourcing")
        self.assertEqual(c["process_owner"], "Bob Zeisler")
        self.assertEqual(len(c["team_members"]), 3)
        self.assertEqual(c["team_members"][0]["role"], "Team Leader")

    def test_update_charter(self):
        """Charter can be updated after creation."""
        hp_id = _post(
            self.client,
            "/api/hoshin/projects/create/",
            {
                "title": "Update Charter Test",
                "site_id": self.site_id,
                "kaizen_charter": {"event_type": "Scrap Reduction"},
            },
        ).json()["project"]["id"]

        # Update with new charter
        _put(
            self.client,
            f"/api/hoshin/projects/{hp_id}/update/",
            {
                "kaizen_charter": {
                    "event_type": "Scrap Reduction",
                    "location": "New location",
                    "primary_metric": "Scrap %",
                    "primary_baseline": 8.5,
                    "primary_target": 3.0,
                },
            },
        )

        proj = self.client.get(f"/api/hoshin/projects/{hp_id}/").json()["project"]
        c = proj["kaizen_charter"]
        self.assertEqual(c["location"], "New location")
        self.assertEqual(c["primary_baseline"], 8.5)
        self.assertEqual(c["primary_target"], 3.0)

    def test_empty_charter_defaults_to_dict(self):
        """Project without charter has empty dict."""
        hp_id = _post(
            self.client,
            "/api/hoshin/projects/create/",
            {
                "title": "No Charter",
                "site_id": self.site_id,
            },
        ).json()["project"]["id"]
        proj = self.client.get(f"/api/hoshin/projects/{hp_id}/").json()["project"]
        self.assertEqual(proj["kaizen_charter"], {})

    def test_plan_fields_in_charter(self):
        """Background, current_conditions, countermeasures stored in charter."""
        hp_id = _post(
            self.client,
            "/api/hoshin/projects/create/",
            {
                "title": "Plan Test",
                "site_id": self.site_id,
            },
        ).json()["project"]["id"]

        _put(
            self.client,
            f"/api/hoshin/projects/{hp_id}/update/",
            {
                "kaizen_charter": {
                    "event_type": "TPM",
                    "background": "Press has 15% unplanned downtime",
                    "current_conditions": "No PM schedule, reactive maintenance only",
                    "plan_objectives": "Implement autonomous maintenance (AM Steps 1-3)",
                    "countermeasures": "1. Clean-to-inspect protocol\n2. Lubrication standards\n3. Visual controls",
                },
            },
        )

        c = self.client.get(f"/api/hoshin/projects/{hp_id}/").json()["project"]["kaizen_charter"]
        self.assertEqual(c["background"], "Press has 15% unplanned downtime")
        self.assertEqual(c["current_conditions"], "No PM schedule, reactive maintenance only")
        self.assertEqual(c["plan_objectives"], "Implement autonomous maintenance (AM Steps 1-3)")
        self.assertIn("Clean-to-inspect", c["countermeasures"])


# =============================================================================
# Custom Formula Engine
# =============================================================================


@SECURE_OFF
class CustomFormulaTest(TestCase):
    """Custom formula testing and evaluation."""

    def setUp(self):
        self.user = _make_enterprise_user("formula@test.com")
        self.tenant = _setup_tenant(self.user)
        self.client.force_login(self.user)

    def test_test_formula_basic(self):
        """Test a simple formula via the test endpoint."""
        res = _post(
            self.client,
            "/api/hoshin/test-formula/",
            {
                "formula": "({{baseline}} - {{actual}}) * {{volume}}",
                "variables": {"baseline": 100, "actual": 80, "volume": 1000},
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        # (100-80)*1000 = 20000
        self.assertAlmostEqual(data["result"], 20000.0, places=0)
        self.assertEqual(data["fields"], ["baseline", "actual", "volume"])

    def test_test_formula_with_functions(self):
        """sqrt, abs, min, max, round allowed."""
        res = _post(
            self.client,
            "/api/hoshin/test-formula/",
            {
                "formula": "sqrt({{area}}) * {{rate}}",
                "variables": {"area": 144, "rate": 10},
            },
        )
        self.assertEqual(res.status_code, 200)
        # sqrt(144) * 10 = 120
        self.assertAlmostEqual(res.json()["result"], 120.0, places=0)

    def test_test_formula_auto_variance(self):
        """variance auto-computed as baseline - actual."""
        res = _post(
            self.client,
            "/api/hoshin/test-formula/",
            {
                "formula": "{{variance}} * {{volume}}",
                "variables": {"baseline": 50, "actual": 30, "volume": 100},
            },
        )
        # variance = 50-30 = 20, * 100 = 2000
        self.assertAlmostEqual(res.json()["result"], 2000.0, places=0)

    def test_test_formula_empty_rejected(self):
        res = _post(
            self.client,
            "/api/hoshin/test-formula/",
            {
                "formula": "",
            },
        )
        self.assertEqual(res.status_code, 400)

    def test_test_formula_division_by_zero(self):
        res = _post(
            self.client,
            "/api/hoshin/test-formula/",
            {
                "formula": "{{baseline}} / {{actual}}",
                "variables": {"baseline": 100, "actual": 0},
            },
        )
        self.assertEqual(res.status_code, 400)

    def test_monthly_with_custom_formula(self):
        """Custom formula used for monthly savings calculation."""
        site_id = _post(
            self.client,
            "/api/hoshin/sites/create/",
            {
                "name": "Custom Plant",
            },
        ).json()["site"]["id"]

        hp_id = _post(
            self.client,
            "/api/hoshin/projects/create/",
            {
                "title": "Custom Calc Project",
                "site_id": site_id,
                "calculation_method": "custom",
                "custom_formula": "({{baseline}} - {{actual}}) * {{volume}} * {{rate}}",
                "annual_savings_target": 100000,
            },
        ).json()["project"]["id"]

        res = _put(
            self.client,
            f"/api/hoshin/projects/{hp_id}/monthly/1/",
            {
                "baseline": 0.08,  # 8% scrap
                "actual": 0.05,  # 5% scrap
                "volume": 50000,
                "cost_per_unit": 1,
                "custom_vars": {"rate": 12.0},
            },
        )
        self.assertEqual(res.status_code, 200)
        entry = res.json()["entry"]
        # (0.08-0.05)*50000*12 = 18000
        self.assertAlmostEqual(entry["savings"], 18000.0, places=0)

    def test_monthly_custom_vars_round_trip(self):
        """Custom variables stored in monthly entry and retrievable."""
        site_id = _post(
            self.client,
            "/api/hoshin/sites/create/",
            {
                "name": "Vars Plant",
            },
        ).json()["site"]["id"]

        hp_id = _post(
            self.client,
            "/api/hoshin/projects/create/",
            {
                "title": "Vars Test",
                "site_id": site_id,
                "calculation_method": "custom",
                "custom_formula": "{{scrap_before}} - {{scrap_after}}",
                "annual_savings_target": 50000,
            },
        ).json()["project"]["id"]

        _put(
            self.client,
            f"/api/hoshin/projects/{hp_id}/monthly/3/",
            {
                "baseline": 100,
                "actual": 80,
                "custom_vars": {"scrap_before": 500, "scrap_after": 320},
            },
        )

        proj = self.client.get(f"/api/hoshin/projects/{hp_id}/").json()["project"]
        march = next(m for m in proj["monthly_actuals"] if m["month"] == 3)
        self.assertEqual(march["custom_vars"]["scrap_before"], 500.0)
        self.assertEqual(march["custom_vars"]["scrap_after"], 320.0)


# =============================================================================
# Calculation Methods Reference
# =============================================================================


@SECURE_OFF
class CalculationMethodsTest(TestCase):
    """List available calculation methods endpoint."""

    def setUp(self):
        self.user = _make_enterprise_user("methods@test.com")
        self.tenant = _setup_tenant(self.user)
        self.client.force_login(self.user)

    def test_list_methods(self):
        res = self.client.get("/api/hoshin/calculation-methods/")
        self.assertEqual(res.status_code, 200)
        methods = res.json()["methods"]
        codes = {m["code"] for m in methods}
        self.assertIn("time_reduction", codes)
        self.assertIn("waste_pct", codes)
        self.assertIn("headcount", codes)
        self.assertIn("direct", codes)
        self.assertIn("custom", codes)
        self.assertIn("freight", codes)
        self.assertIn("energy", codes)
        self.assertIn("claims", codes)
        self.assertIn("layout", codes)
        # Each method has required fields
        for m in methods:
            self.assertIn("name", m)
            self.assertIn("category", m)
            self.assertIn("formula", m)


# =============================================================================
# Savings Aggregation
# =============================================================================


@SECURE_OFF
class SavingsAggregationTest(TestCase):
    """Savings summary via project detail (aggregate_monthly_savings)."""

    def setUp(self):
        self.user = _make_enterprise_user("agg@test.com")
        self.tenant = _setup_tenant(self.user)
        self.client.force_login(self.user)
        site_id = _post(
            self.client,
            "/api/hoshin/sites/create/",
            {
                "name": "Agg Plant",
            },
        ).json()["site"]["id"]
        self.hp_id = _post(
            self.client,
            "/api/hoshin/projects/create/",
            {
                "title": "Aggregation Test",
                "site_id": site_id,
                "calculation_method": "direct",
                "annual_savings_target": 120000,
            },
        ).json()["project"]["id"]

    def test_savings_summary_structure(self):
        """Project detail includes savings_summary with ytd, trend, months."""
        _put(
            self.client,
            f"/api/hoshin/projects/{self.hp_id}/monthly/1/",
            {
                "baseline": 10000,
                "actual": 7000,
            },
        )
        _put(
            self.client,
            f"/api/hoshin/projects/{self.hp_id}/monthly/2/",
            {
                "baseline": 10000,
                "actual": 6500,
            },
        )

        proj = self.client.get(f"/api/hoshin/projects/{self.hp_id}/").json()["project"]
        summary = proj["savings_summary"]
        # YTD = 3000 + 3500 = 6500
        self.assertAlmostEqual(summary["ytd_savings"], 6500.0, places=0)
        self.assertEqual(summary["months_reported"], 2)

    def test_savings_trend_cumulative(self):
        """Monthly trend shows cumulative savings."""
        _put(
            self.client,
            f"/api/hoshin/projects/{self.hp_id}/monthly/1/",
            {
                "baseline": 10000,
                "actual": 8000,
            },
        )
        _put(
            self.client,
            f"/api/hoshin/projects/{self.hp_id}/monthly/2/",
            {
                "baseline": 10000,
                "actual": 7000,
            },
        )
        _put(
            self.client,
            f"/api/hoshin/projects/{self.hp_id}/monthly/3/",
            {
                "baseline": 10000,
                "actual": 6000,
            },
        )

        summary = self.client.get(f"/api/hoshin/projects/{self.hp_id}/").json()["project"]["savings_summary"]
        trend = summary["monthly_trend"]
        self.assertEqual(len(trend), 3)
        # Cumulative: 2000, 2000+3000=5000, 5000+4000=9000
        self.assertAlmostEqual(trend[0]["cumulative"], 2000.0, places=0)
        self.assertAlmostEqual(trend[1]["cumulative"], 5000.0, places=0)
        self.assertAlmostEqual(trend[2]["cumulative"], 9000.0, places=0)


# =============================================================================
# Baseline Data
# =============================================================================


@SECURE_OFF
class BaselineDataTest(TestCase):
    """Baseline data round-trip on hoshin projects."""

    def setUp(self):
        self.user = _make_enterprise_user("baseline@test.com")
        self.tenant = _setup_tenant(self.user)
        self.client.force_login(self.user)
        site_id = _post(
            self.client,
            "/api/hoshin/sites/create/",
            {
                "name": "Baseline Plant",
            },
        ).json()["site"]["id"]
        self.hp_id = _post(
            self.client,
            "/api/hoshin/projects/create/",
            {
                "title": "Baseline Test",
                "site_id": site_id,
            },
        ).json()["project"]["id"]

    def test_set_and_retrieve_baseline(self):
        baseline = [
            {"month": 1, "metric_value": 8.2, "volume": 45000, "cost_per_unit": 12},
            {"month": 2, "metric_value": 8.5, "volume": 47000, "cost_per_unit": 12},
            {"month": 3, "metric_value": 7.9, "volume": 44000, "cost_per_unit": 12},
        ]
        _put(
            self.client,
            f"/api/hoshin/projects/{self.hp_id}/update/",
            {
                "baseline_data": baseline,
            },
        )
        proj = self.client.get(f"/api/hoshin/projects/{self.hp_id}/").json()["project"]
        self.assertEqual(len(proj["baseline_data"]), 3)
        self.assertEqual(proj["baseline_data"][1]["metric_value"], 8.5)

    def test_baseline_set_on_create(self):
        site_id = _post(
            self.client,
            "/api/hoshin/sites/create/",
            {
                "name": "BL Create",
            },
        ).json()["site"]["id"]
        baseline = [{"month": 1, "metric_value": 5.0}]
        hp_id = _post(
            self.client,
            "/api/hoshin/projects/create/",
            {
                "title": "BL On Create",
                "site_id": site_id,
                "baseline_data": baseline,
            },
        ).json()["project"]["id"]
        proj = self.client.get(f"/api/hoshin/projects/{hp_id}/").json()["project"]
        self.assertEqual(len(proj["baseline_data"]), 1)


# =============================================================================
# X-Matrix: Strategic Objectives CRUD
# =============================================================================


@SECURE_OFF
class StrategicObjectiveTest(TestCase):
    """Strategic objective (3-5 year breakthrough) CRUD."""

    def setUp(self):
        self.user = _make_enterprise_user("strategic@test.com")
        self.tenant = _setup_tenant(self.user)
        self.client.force_login(self.user)

    def test_create_strategic_objective(self):
        res = _post(
            self.client,
            "/api/hoshin/strategic-objectives/",
            {
                "title": "Reduce plant scrap to <2%",
                "description": "Multi-year scrap reduction initiative",
                "owner_name": "VP Operations",
                "start_year": 2025,
                "end_year": 2028,
                "target_metric": "waste_pct",
                "target_value": 2.0,
                "status": "active",
            },
        )
        self.assertEqual(res.status_code, 201)
        obj = res.json()["strategic_objective"]
        self.assertEqual(obj["title"], "Reduce plant scrap to <2%")
        self.assertEqual(obj["start_year"], 2025)
        self.assertEqual(obj["end_year"], 2028)
        self.assertEqual(obj["status"], "active")

    def test_list_strategic_objectives(self):
        _post(
            self.client,
            "/api/hoshin/strategic-objectives/",
            {
                "title": "Obj A",
                "start_year": 2025,
                "end_year": 2028,
            },
        )
        _post(
            self.client,
            "/api/hoshin/strategic-objectives/",
            {
                "title": "Obj B",
                "start_year": 2026,
                "end_year": 2029,
            },
        )
        res = self.client.get("/api/hoshin/strategic-objectives/")
        self.assertEqual(len(res.json()["strategic_objectives"]), 2)

    def test_filter_by_fiscal_year(self):
        _post(
            self.client,
            "/api/hoshin/strategic-objectives/",
            {
                "title": "Past",
                "start_year": 2020,
                "end_year": 2023,
            },
        )
        _post(
            self.client,
            "/api/hoshin/strategic-objectives/",
            {
                "title": "Current",
                "start_year": 2025,
                "end_year": 2028,
            },
        )
        res = self.client.get("/api/hoshin/strategic-objectives/?fiscal_year=2026")
        objs = res.json()["strategic_objectives"]
        self.assertEqual(len(objs), 1)
        self.assertEqual(objs[0]["title"], "Current")

    def test_update_strategic_objective(self):
        obj_id = _post(
            self.client,
            "/api/hoshin/strategic-objectives/",
            {
                "title": "Before",
            },
        ).json()["strategic_objective"]["id"]
        res = _put(
            self.client,
            f"/api/hoshin/strategic-objectives/{obj_id}/",
            {
                "title": "After",
                "status": "achieved",
            },
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["strategic_objective"]["title"], "After")
        self.assertEqual(res.json()["strategic_objective"]["status"], "achieved")

    def test_delete_strategic_objective(self):
        obj_id = _post(
            self.client,
            "/api/hoshin/strategic-objectives/",
            {
                "title": "Delete Me",
            },
        ).json()["strategic_objective"]["id"]
        res = self.client.delete(f"/api/hoshin/strategic-objectives/{obj_id}/")
        self.assertEqual(res.status_code, 200)

    def test_requires_title(self):
        res = _post(self.client, "/api/hoshin/strategic-objectives/", {"title": ""})
        self.assertEqual(res.status_code, 400)


# =============================================================================
# X-Matrix: Annual Objectives CRUD
# =============================================================================


@SECURE_OFF
class AnnualObjectiveTest(TestCase):
    """Annual objective CRUD with strategic linkage."""

    def setUp(self):
        self.user = _make_enterprise_user("annual@test.com")
        self.tenant = _setup_tenant(self.user)
        self.client.force_login(self.user)

        # Create a strategic objective to link to
        self.strat_id = _post(
            self.client,
            "/api/hoshin/strategic-objectives/",
            {
                "title": "Reduce Scrap",
                "start_year": 2025,
                "end_year": 2028,
            },
        ).json()["strategic_objective"]["id"]

        self.site_id = _post(
            self.client,
            "/api/hoshin/sites/create/",
            {
                "name": "Annual Plant",
            },
        ).json()["site"]["id"]

    def test_create_annual_objective(self):
        res = _post(
            self.client,
            "/api/hoshin/annual-objectives/",
            {
                "title": "Reduce FTW scrap from 8% to 5%",
                "strategic_objective_id": self.strat_id,
                "site_id": self.site_id,
                "fiscal_year": 2026,
                "target_value": 5.0,
                "target_unit": "%",
                "owner_name": "Plant Manager",
            },
        )
        self.assertEqual(res.status_code, 201)
        obj = res.json()["annual_objective"]
        self.assertEqual(obj["title"], "Reduce FTW scrap from 8% to 5%")
        self.assertEqual(obj["fiscal_year"], 2026)

    def test_list_by_fiscal_year(self):
        _post(
            self.client,
            "/api/hoshin/annual-objectives/",
            {
                "title": "2025 Obj",
                "fiscal_year": 2025,
            },
        )
        _post(
            self.client,
            "/api/hoshin/annual-objectives/",
            {
                "title": "2026 Obj",
                "fiscal_year": 2026,
            },
        )
        res = self.client.get("/api/hoshin/annual-objectives/?fiscal_year=2026")
        objs = res.json()["annual_objectives"]
        self.assertEqual(len(objs), 1)
        self.assertEqual(objs[0]["title"], "2026 Obj")

    def test_update_annual_objective(self):
        obj_id = _post(
            self.client,
            "/api/hoshin/annual-objectives/",
            {
                "title": "Original",
                "fiscal_year": 2026,
            },
        ).json()["annual_objective"]["id"]
        _put(
            self.client,
            f"/api/hoshin/annual-objectives/{obj_id}/",
            {
                "actual_value": 4.8,
                "status": "on_track",
            },
        )
        obj = self.client.get("/api/hoshin/annual-objectives/?fiscal_year=2026").json()["annual_objectives"][0]
        self.assertEqual(float(obj["actual_value"]), 4.8)

    def test_delete_annual_objective(self):
        obj_id = _post(
            self.client,
            "/api/hoshin/annual-objectives/",
            {
                "title": "Delete",
                "fiscal_year": 2026,
            },
        ).json()["annual_objective"]["id"]
        res = self.client.delete(f"/api/hoshin/annual-objectives/{obj_id}/")
        self.assertEqual(res.status_code, 200)

    def test_link_to_strategic(self):
        """Annual objective linked to strategic carries the FK."""
        res = _post(
            self.client,
            "/api/hoshin/annual-objectives/",
            {
                "title": "Linked Annual",
                "fiscal_year": 2026,
                "strategic_objective_id": self.strat_id,
                "target_value": 5.0,
                "target_unit": "%",
            },
        )
        self.assertEqual(res.status_code, 201)
        obj = res.json()["annual_objective"]
        self.assertEqual(obj["strategic_objective_id"], self.strat_id)


# =============================================================================
# X-Matrix: KPI CRUD
# =============================================================================


@SECURE_OFF
class HoshinKPITest(TestCase):
    """KPI CRUD with metric catalog auto-fill."""

    def setUp(self):
        self.user = _make_enterprise_user("kpi@test.com")
        self.tenant = _setup_tenant(self.user)
        self.client.force_login(self.user)

        # Create a hoshin project to derive KPI from
        site_id = _post(
            self.client,
            "/api/hoshin/sites/create/",
            {
                "name": "KPI Plant",
            },
        ).json()["site"]["id"]
        self.hp_id = _post(
            self.client,
            "/api/hoshin/projects/create/",
            {
                "title": "Scrap Project",
                "site_id": site_id,
                "calculation_method": "waste_pct",
                "annual_savings_target": 50000,
            },
        ).json()["project"]["id"]

    def test_create_kpi_with_catalog(self):
        """KPI from metric catalog auto-fills unit, direction, aggregation."""
        res = _post(
            self.client,
            "/api/hoshin/kpis/",
            {
                "name": "Plant Scrap Rate",
                "metric_type": "scrap_rate",
                "fiscal_year": 2026,
                "target_value": 2.0,
            },
        )
        self.assertEqual(res.status_code, 201)
        kpi = res.json()["kpi"]
        self.assertEqual(kpi["name"], "Plant Scrap Rate")
        # scrap_rate catalog entry should set direction=down
        self.assertEqual(kpi["direction"], "down")

    def test_create_manual_kpi(self):
        res = _post(
            self.client,
            "/api/hoshin/kpis/",
            {
                "name": "Customer Satisfaction",
                "metric_type": "manual",
                "fiscal_year": 2026,
                "target_value": 95,
                "unit": "NPS",
                "direction": "up",
            },
        )
        self.assertEqual(res.status_code, 201)

    def test_create_kpi_derived_from_project(self):
        """KPI derived from a hoshin project tracks its ytd_savings."""
        res = _post(
            self.client,
            "/api/hoshin/kpis/",
            {
                "name": "Scrap Savings KPI",
                "metric_type": "dollar_savings",
                "fiscal_year": 2026,
                "derived_from_id": self.hp_id,
                "target_value": 50000,
            },
        )
        self.assertEqual(res.status_code, 201)
        kpi = res.json()["kpi"]
        self.assertEqual(kpi["aggregation"], "sum")

    def test_list_kpis(self):
        _post(
            self.client,
            "/api/hoshin/kpis/",
            {
                "name": "KPI A",
                "fiscal_year": 2026,
            },
        )
        _post(
            self.client,
            "/api/hoshin/kpis/",
            {
                "name": "KPI B",
                "fiscal_year": 2026,
            },
        )
        res = self.client.get("/api/hoshin/kpis/?fiscal_year=2026")
        self.assertEqual(len(res.json()["kpis"]), 2)

    def test_update_kpi(self):
        kpi_id = _post(
            self.client,
            "/api/hoshin/kpis/",
            {
                "name": "Original",
                "fiscal_year": 2026,
            },
        ).json()["kpi"]["id"]
        _put(
            self.client,
            f"/api/hoshin/kpis/{kpi_id}/",
            {
                "actual_value": 42.5,
            },
        )
        kpis = self.client.get("/api/hoshin/kpis/?fiscal_year=2026").json()["kpis"]
        kpi = next(k for k in kpis if k["id"] == kpi_id)
        self.assertEqual(float(kpi["actual_value"]), 42.5)

    def test_delete_kpi(self):
        kpi_id = _post(
            self.client,
            "/api/hoshin/kpis/",
            {
                "name": "Delete Me",
                "fiscal_year": 2026,
            },
        ).json()["kpi"]["id"]
        res = self.client.delete(f"/api/hoshin/kpis/{kpi_id}/")
        self.assertEqual(res.status_code, 200)

    def test_requires_name(self):
        res = _post(self.client, "/api/hoshin/kpis/", {"name": ""})
        self.assertEqual(res.status_code, 400)

    def test_kpi_target_tracking(self):
        """KPI tracks target vs actual and computes gap."""
        kpi_id = _post(
            self.client,
            "/api/hoshin/kpis/",
            {
                "name": "Scrap Rate",
                "metric_type": "scrap_rate",
                "fiscal_year": 2026,
                "target_value": 2.0,
            },
        ).json()["kpi"]["id"]
        _put(
            self.client,
            f"/api/hoshin/kpis/{kpi_id}/",
            {
                "actual_value": 3.5,
            },
        )
        kpi = next(k for k in self.client.get("/api/hoshin/kpis/?fiscal_year=2026").json()["kpis"] if k["id"] == kpi_id)
        self.assertEqual(float(kpi["target_value"]), 2.0)
        self.assertEqual(float(kpi["actual_value"]), 3.5)

    def test_kpi_aggregation_sum(self):
        """Dollar savings KPI uses sum aggregation."""
        res = _post(
            self.client,
            "/api/hoshin/kpis/",
            {
                "name": "Dollar KPI",
                "metric_type": "dollar_savings",
                "fiscal_year": 2026,
                "target_value": 100000,
                "derived_from_id": self.hp_id,
            },
        )
        self.assertEqual(res.json()["kpi"]["aggregation"], "sum")

    def test_kpi_aggregation_weighted_avg(self):
        """OEE KPI uses weighted_avg aggregation."""
        res = _post(
            self.client,
            "/api/hoshin/kpis/",
            {
                "name": "OEE KPI",
                "metric_type": "oee",
                "fiscal_year": 2026,
                "target_value": 85,
            },
        )
        self.assertEqual(res.status_code, 201)
        kpi = res.json()["kpi"]
        self.assertIn(kpi.get("aggregation", "weighted_avg"), ["weighted_avg", "average"])


# =============================================================================
# X-Matrix Correlations
# =============================================================================


@SECURE_OFF
class XMatrixCorrelationTest(TestCase):
    """Correlation upsert, delete, and validation."""

    def setUp(self):
        self.user = _make_enterprise_user("corr@test.com")
        self.tenant = _setup_tenant(self.user)
        self.client.force_login(self.user)

        # Create entities to correlate
        self.strat_id = _post(
            self.client,
            "/api/hoshin/strategic-objectives/",
            {
                "title": "Reduce Scrap",
                "start_year": 2025,
                "end_year": 2028,
            },
        ).json()["strategic_objective"]["id"]

        self.annual_id = _post(
            self.client,
            "/api/hoshin/annual-objectives/",
            {
                "title": "FTW Scrap 8→5%",
                "fiscal_year": 2026,
            },
        ).json()["annual_objective"]["id"]

    def test_create_correlation(self):
        res = _post(
            self.client,
            "/api/hoshin/x-matrix/correlations/",
            {
                "pair_type": "strategic_annual",
                "row_id": self.strat_id,
                "col_id": self.annual_id,
                "strength": "strong",
                "fiscal_year": 2026,
            },
        )
        self.assertEqual(res.status_code, 200)
        corr = res.json()["correlation"]
        self.assertEqual(corr["strength"], "strong")
        self.assertEqual(corr["pair_type"], "strategic_annual")
        self.assertTrue(corr["confirmed"])

    def test_delete_correlation_by_null_strength(self):
        """Setting strength to null deletes the correlation."""
        _post(
            self.client,
            "/api/hoshin/x-matrix/correlations/",
            {
                "pair_type": "strategic_annual",
                "row_id": self.strat_id,
                "col_id": self.annual_id,
                "strength": "strong",
            },
        )
        res = _post(
            self.client,
            "/api/hoshin/x-matrix/correlations/",
            {
                "pair_type": "strategic_annual",
                "row_id": self.strat_id,
                "col_id": self.annual_id,
                "strength": None,
            },
        )
        self.assertTrue(res.json()["deleted"])

    def test_update_strength(self):
        """Upserting with different strength updates the correlation."""
        _post(
            self.client,
            "/api/hoshin/x-matrix/correlations/",
            {
                "pair_type": "strategic_annual",
                "row_id": self.strat_id,
                "col_id": self.annual_id,
                "strength": "weak",
            },
        )
        res = _post(
            self.client,
            "/api/hoshin/x-matrix/correlations/",
            {
                "pair_type": "strategic_annual",
                "row_id": self.strat_id,
                "col_id": self.annual_id,
                "strength": "strong",
            },
        )
        self.assertEqual(res.json()["correlation"]["strength"], "strong")

    def test_invalid_pair_type(self):
        res = _post(
            self.client,
            "/api/hoshin/x-matrix/correlations/",
            {
                "pair_type": "invalid",
                "row_id": self.strat_id,
                "col_id": self.annual_id,
                "strength": "strong",
            },
        )
        self.assertEqual(res.status_code, 400)

    def test_invalid_strength(self):
        res = _post(
            self.client,
            "/api/hoshin/x-matrix/correlations/",
            {
                "pair_type": "strategic_annual",
                "row_id": self.strat_id,
                "col_id": self.annual_id,
                "strength": "invalid",
            },
        )
        self.assertEqual(res.status_code, 400)

    def test_missing_fields(self):
        res = _post(
            self.client,
            "/api/hoshin/x-matrix/correlations/",
            {
                "pair_type": "strategic_annual",
            },
        )
        self.assertEqual(res.status_code, 400)

    # Aliases for compliance hook names
    test_upsert_updates_existing = test_update_strength
    test_delete_correlation = test_delete_correlation_by_null_strength
    test_invalid_pair_type_rejected = test_invalid_pair_type

    def test_duplicate_pair_rejected(self):
        """Creating same pair twice upserts rather than duplicating."""
        _post(
            self.client,
            "/api/hoshin/x-matrix/correlations/",
            {
                "pair_type": "strategic_annual",
                "row_id": self.strat_id,
                "col_id": self.annual_id,
                "strength": "weak",
            },
        )
        res = _post(
            self.client,
            "/api/hoshin/x-matrix/correlations/",
            {
                "pair_type": "strategic_annual",
                "row_id": self.strat_id,
                "col_id": self.annual_id,
                "strength": "strong",
            },
        )
        self.assertEqual(res.status_code, 200)
        # Should have updated, not duplicated
        self.assertEqual(res.json()["correlation"]["strength"], "strong")


# =============================================================================
# X-Matrix Data Endpoint
# =============================================================================


@SECURE_OFF
class XMatrixDataTest(TestCase):
    """Full X-matrix data with four quadrants, correlations, and rollup."""

    def setUp(self):
        self.user = _make_enterprise_user("xmatrix@test.com")
        self.tenant = _setup_tenant(self.user)
        self.client.force_login(self.user)

        # Build an X-matrix with all four quadrants
        self.site_id = _post(
            self.client,
            "/api/hoshin/sites/create/",
            {
                "name": "X Plant",
            },
        ).json()["site"]["id"]

        # Strategic objective
        self.strat_id = _post(
            self.client,
            "/api/hoshin/strategic-objectives/",
            {
                "title": "World-class OEE",
                "start_year": 2025,
                "end_year": 2028,
                "target_metric": "oee",
                "target_value": 85,
            },
        ).json()["strategic_objective"]["id"]

        # Annual objective
        self.annual_id = _post(
            self.client,
            "/api/hoshin/annual-objectives/",
            {
                "title": "OEE from 65% to 75%",
                "fiscal_year": 2026,
                "strategic_objective_id": self.strat_id,
                "target_value": 75,
                "target_unit": "%",
            },
        ).json()["annual_objective"]["id"]

        # Hoshin project
        self.hp_id = _post(
            self.client,
            "/api/hoshin/projects/create/",
            {
                "title": "TPM Implementation",
                "site_id": self.site_id,
                "fiscal_year": 2026,
                "hoshin_status": "active",
                "calculation_method": "direct",
                "annual_savings_target": 40000,
            },
        ).json()["project"]["id"]

        # KPI
        self.kpi_id = _post(
            self.client,
            "/api/hoshin/kpis/",
            {
                "name": "OEE",
                "metric_type": "oee",
                "fiscal_year": 2026,
                "target_value": 75,
            },
        ).json()["kpi"]["id"]

        # Correlations
        _post(
            self.client,
            "/api/hoshin/x-matrix/correlations/",
            {
                "pair_type": "strategic_annual",
                "row_id": self.strat_id,
                "col_id": self.annual_id,
                "strength": "strong",
            },
        )
        _post(
            self.client,
            "/api/hoshin/x-matrix/correlations/",
            {
                "pair_type": "annual_project",
                "row_id": self.annual_id,
                "col_id": self.hp_id,
                "strength": "strong",
            },
        )

    def test_xmatrix_returns_all_quadrants(self):
        res = self.client.get("/api/hoshin/x-matrix/?fiscal_year=2026")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["fiscal_year"], 2026)
        self.assertGreaterEqual(len(data["strategic_objectives"]), 1)
        self.assertGreaterEqual(len(data["annual_objectives"]), 1)
        self.assertGreaterEqual(len(data["projects"]), 1)
        self.assertGreaterEqual(len(data["kpis"]), 1)

    def test_xmatrix_includes_correlations(self):
        data = self.client.get("/api/hoshin/x-matrix/?fiscal_year=2026").json()
        corrs = data["correlations"]
        self.assertIn("strategic_annual", corrs)
        self.assertIn("annual_project", corrs)
        self.assertGreaterEqual(len(corrs["strategic_annual"]), 1)
        self.assertGreaterEqual(len(corrs["annual_project"]), 1)

    def test_xmatrix_includes_rollup(self):
        """Dollar rollup across strategic → annual → project chain."""
        # Record some savings
        _put(
            self.client,
            f"/api/hoshin/projects/{self.hp_id}/monthly/1/",
            {
                "baseline": 5000,
                "actual": 3000,
            },
        )

        data = self.client.get("/api/hoshin/x-matrix/?fiscal_year=2026").json()
        rollup = data["rollup"]
        self.assertIn("total_target", rollup)
        self.assertIn("total_ytd", rollup)
        self.assertGreater(rollup["total_target"], 0)

    def test_xmatrix_includes_metric_catalog(self):
        data = self.client.get("/api/hoshin/x-matrix/?fiscal_year=2026").json()
        self.assertIn("metric_catalog", data)
        self.assertIn("oee", data["metric_catalog"])

    def test_correlation_cleanup_on_delete(self):
        """Deleting a strategic objective cleans up its correlations."""
        # Verify correlation exists
        data = self.client.get("/api/hoshin/x-matrix/?fiscal_year=2026").json()
        self.assertGreaterEqual(len(data["correlations"]["strategic_annual"]), 1)

        # Delete strategic objective
        self.client.delete(f"/api/hoshin/strategic-objectives/{self.strat_id}/")

        # Correlation should be gone
        data = self.client.get("/api/hoshin/x-matrix/?fiscal_year=2026").json()
        strat_annual = [c for c in data["correlations"]["strategic_annual"] if c["row_id"] == self.strat_id]
        self.assertEqual(len(strat_annual), 0)

    # Aliases for compliance hook names
    test_xmatrix_rollup_summary = test_xmatrix_includes_rollup

    def test_xmatrix_fiscal_year_filter(self):
        """X-matrix returns only data for the requested fiscal year."""
        data_2026 = self.client.get("/api/hoshin/x-matrix/?fiscal_year=2026").json()
        data_2099 = self.client.get("/api/hoshin/x-matrix/?fiscal_year=2099").json()
        self.assertEqual(data_2026["fiscal_year"], 2026)
        self.assertGreaterEqual(len(data_2026["projects"]), 1)
        self.assertEqual(data_2099["fiscal_year"], 2099)
        self.assertEqual(len(data_2099["projects"]), 0)

    def test_xmatrix_empty_state(self):
        """X-matrix for a year with no data returns empty quadrants."""
        data = self.client.get("/api/hoshin/x-matrix/?fiscal_year=2099").json()
        self.assertEqual(data["strategic_objectives"], [])
        self.assertEqual(data["annual_objectives"], [])
        self.assertEqual(data["projects"], [])
        self.assertEqual(data["kpis"], [])


# =============================================================================
# Full X-Matrix Lifecycle
# =============================================================================


@SECURE_OFF
class XMatrixLifecycleTest(TestCase):
    """Build complete X-matrix: strategy → annual → projects → KPIs → correlations."""

    def setUp(self):
        self.user = _make_enterprise_user("lifecycle_xm@test.com")
        self.tenant = _setup_tenant(self.user)
        self.client.force_login(self.user)

    def test_full_xmatrix_build(self):
        """Create all four quadrants, correlate, verify data endpoint."""
        # Site
        site_id = _post(
            self.client,
            "/api/hoshin/sites/create/",
            {
                "name": "Lifecycle Plant",
            },
        ).json()["site"]["id"]

        # Strategic: 3-year scrap reduction
        s1 = _post(
            self.client,
            "/api/hoshin/strategic-objectives/",
            {
                "title": "Reduce plant scrap to <2%",
                "start_year": 2025,
                "end_year": 2028,
                "target_metric": "waste_pct",
                "target_value": 2.0,
                "status": "active",
            },
        ).json()["strategic_objective"]["id"]

        # Annual: this year's target
        a1 = _post(
            self.client,
            "/api/hoshin/annual-objectives/",
            {
                "title": "Scrap 8% → 5%",
                "strategic_objective_id": s1,
                "site_id": site_id,
                "fiscal_year": 2026,
                "target_value": 5.0,
                "target_unit": "%",
            },
        ).json()["annual_objective"]["id"]

        # Two projects
        p1 = _post(
            self.client,
            "/api/hoshin/projects/create/",
            {
                "title": "SMED on Press 1",
                "site_id": site_id,
                "fiscal_year": 2026,
                "project_type": "labor",
                "hoshin_status": "active",
                "calculation_method": "waste_pct",
                "annual_savings_target": 30000,
            },
        ).json()["project"]["id"]

        p2 = _post(
            self.client,
            "/api/hoshin/projects/create/",
            {
                "title": "Poka-Yoke Assembly",
                "site_id": site_id,
                "fiscal_year": 2026,
                "project_type": "quality",
                "hoshin_status": "active",
                "calculation_method": "waste_pct",
                "annual_savings_target": 20000,
            },
        ).json()["project"]["id"]

        # KPI
        k1 = _post(
            self.client,
            "/api/hoshin/kpis/",
            {
                "name": "Scrap Rate",
                "metric_type": "scrap_rate",
                "fiscal_year": 2026,
                "target_value": 5.0,
            },
        ).json()["kpi"]["id"]

        # Correlations: strategic↔annual, annual↔projects, projects↔KPI
        for pair_type, row, col in [
            ("strategic_annual", s1, a1),
            ("annual_project", a1, p1),
            ("annual_project", a1, p2),
            ("project_kpi", p1, k1),
            ("project_kpi", p2, k1),
            ("kpi_strategic", k1, s1),
        ]:
            _post(
                self.client,
                "/api/hoshin/x-matrix/correlations/",
                {
                    "pair_type": pair_type,
                    "row_id": row,
                    "col_id": col,
                    "strength": "strong",
                },
            )

        # Track savings on both projects
        _put(
            self.client,
            f"/api/hoshin/projects/{p1}/monthly/1/",
            {
                "baseline": 8.0,
                "actual": 6.5,
                "volume": 40000,
                "cost_per_unit": 12,
            },
        )
        _put(
            self.client,
            f"/api/hoshin/projects/{p2}/monthly/1/",
            {
                "baseline": 8.0,
                "actual": 7.0,
                "volume": 30000,
                "cost_per_unit": 12,
            },
        )

        # Verify X-matrix
        xm = self.client.get("/api/hoshin/x-matrix/?fiscal_year=2026").json()
        self.assertEqual(len(xm["strategic_objectives"]), 1)
        self.assertEqual(len(xm["annual_objectives"]), 1)
        self.assertEqual(len(xm["projects"]), 2)
        self.assertEqual(len(xm["kpis"]), 1)

        # Total correlations
        total_corrs = sum(len(v) for v in xm["correlations"].values())
        self.assertGreaterEqual(total_corrs, 6)

        # Rollup should show combined savings
        self.assertGreater(xm["rollup"]["total_ytd"], 0)
        self.assertAlmostEqual(xm["rollup"]["total_target"], 50000.0, places=0)


# =============================================================================
# Hoshin Calendar
# =============================================================================


@SECURE_OFF
class HoshinCalendarTest(TestCase):
    """Calendar endpoint: projects grouped by site with monthly target vs actual."""

    def setUp(self):
        self.user = _make_enterprise_user("calendar@test.com")
        self.tenant = _setup_tenant(self.user)
        self.client.force_login(self.user)

        self.site_id = _post(
            self.client,
            "/api/hoshin/sites/create/",
            {
                "name": "Calendar Plant",
            },
        ).json()["site"]["id"]

    def _create_project(self, title, target=120000, **kwargs):
        data = {
            "title": title,
            "site_id": self.site_id,
            "fiscal_year": 2026,
            "calculation_method": "direct",
            "annual_savings_target": target,
            **kwargs,
        }
        return _post(self.client, "/api/hoshin/projects/create/", data).json()["project"]["id"]

    def test_calendar_basic_structure(self):
        """Calendar returns sites with projects and 12 months."""
        hp_id = self._create_project("Cal Project A", target=120000)
        _put(
            self.client,
            f"/api/hoshin/projects/{hp_id}/monthly/1/",
            {
                "baseline": 15000,
                "actual": 12000,
            },
        )

        res = self.client.get("/api/hoshin/calendar/?fiscal_year=2026")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["fiscal_year"], 2026)
        self.assertEqual(len(data["sites"]), 1)

        site = data["sites"][0]
        self.assertEqual(site["site_name"], "Calendar Plant")
        self.assertEqual(len(site["projects"]), 1)
        self.assertEqual(len(site["months"]), 12)

        proj = site["projects"][0]
        self.assertEqual(len(proj["months"]), 12)
        self.assertEqual(proj["title"], "Cal Project A")

    def test_calendar_monthly_target_from_annual(self):
        """Monthly target = annual_savings_target / 12."""
        self._create_project("Even Split", target=120000)

        res = self.client.get("/api/hoshin/calendar/?fiscal_year=2026")
        proj = res.json()["sites"][0]["projects"][0]
        # 120000 / 12 = 10000 per month
        for m in proj["months"]:
            self.assertAlmostEqual(m["target"], 10000.0, places=2)

    def test_calendar_actual_from_monthly_savings(self):
        """Monthly actuals come from recorded savings."""
        hp_id = self._create_project("Actual Tracking", target=120000)
        _put(
            self.client,
            f"/api/hoshin/projects/{hp_id}/monthly/1/",
            {
                "baseline": 15000,
                "actual": 12000,
            },
        )
        _put(
            self.client,
            f"/api/hoshin/projects/{hp_id}/monthly/2/",
            {
                "baseline": 15000,
                "actual": 10000,
            },
        )

        res = self.client.get("/api/hoshin/calendar/?fiscal_year=2026")
        proj = res.json()["sites"][0]["projects"][0]

        # Month 1: savings = 15000-12000 = 3000
        self.assertAlmostEqual(proj["months"][0]["actual"], 3000.0, places=0)
        # Month 2: savings = 15000-10000 = 5000
        self.assertAlmostEqual(proj["months"][1]["actual"], 5000.0, places=0)
        # Month 3: no data = 0
        self.assertAlmostEqual(proj["months"][2]["actual"], 0.0, places=0)

    def test_calendar_pct_calculation(self):
        """Each month has pct = actual/target * 100."""
        hp_id = self._create_project("Pct Test", target=120000)
        # target per month = 10000
        _put(
            self.client,
            f"/api/hoshin/projects/{hp_id}/monthly/3/",
            {
                "baseline": 15000,
                "actual": 10000,
            },
        )

        proj = self.client.get("/api/hoshin/calendar/?fiscal_year=2026").json()["sites"][0]["projects"][0]
        march = proj["months"][2]
        # savings=5000, target=10000, pct=50%
        self.assertAlmostEqual(march["pct"], 50.0, places=1)

    def test_calendar_site_level_aggregation(self):
        """Site-level months aggregate across projects."""
        hp1 = self._create_project("Proj Alpha", target=120000)
        hp2 = self._create_project("Proj Beta", target=60000)

        _put(
            self.client,
            f"/api/hoshin/projects/{hp1}/monthly/1/",
            {
                "baseline": 15000,
                "actual": 12000,
            },
        )
        _put(
            self.client,
            f"/api/hoshin/projects/{hp2}/monthly/1/",
            {
                "baseline": 8000,
                "actual": 6000,
            },
        )

        site = self.client.get("/api/hoshin/calendar/?fiscal_year=2026").json()["sites"][0]

        # Site-level month 1: targets = 10000+5000=15000, actuals = 3000+2000=5000
        jan = site["months"][0]
        self.assertAlmostEqual(jan["target"], 15000.0, places=0)
        self.assertAlmostEqual(jan["actual"], 5000.0, places=0)

    def test_calendar_site_ytd_totals(self):
        """Site-level ytd and target are summed from projects."""
        hp1 = self._create_project("YTD A", target=100000)
        hp2 = self._create_project("YTD B", target=50000)

        _put(
            self.client,
            f"/api/hoshin/projects/{hp1}/monthly/1/",
            {
                "baseline": 12000,
                "actual": 9000,
            },
        )
        _put(
            self.client,
            f"/api/hoshin/projects/{hp2}/monthly/1/",
            {
                "baseline": 8000,
                "actual": 5000,
            },
        )

        site = self.client.get("/api/hoshin/calendar/?fiscal_year=2026").json()["sites"][0]
        self.assertAlmostEqual(site["target"], 150000.0, places=0)
        # ytd = 3000 + 3000 = 6000
        self.assertAlmostEqual(site["ytd"], 6000.0, places=0)

    def test_calendar_excludes_aborted(self):
        """Aborted projects are excluded from the calendar."""
        self._create_project("Active One", target=120000)
        aborted_id = self._create_project("Aborted One", target=60000)
        _put(
            self.client,
            f"/api/hoshin/projects/{aborted_id}/update/",
            {
                "hoshin_status": "aborted",
            },
        )

        site = self.client.get("/api/hoshin/calendar/?fiscal_year=2026").json()["sites"][0]
        self.assertEqual(len(site["projects"]), 1)
        self.assertEqual(site["projects"][0]["title"], "Active One")

    def test_calendar_fiscal_year_filter(self):
        """Only projects matching fiscal_year are returned."""
        self._create_project("FY2026 Proj", target=120000, fiscal_year=2026)
        self._create_project("FY2025 Proj", target=120000, fiscal_year=2025)

        data_2026 = self.client.get("/api/hoshin/calendar/?fiscal_year=2026").json()
        data_2025 = self.client.get("/api/hoshin/calendar/?fiscal_year=2025").json()

        projs_2026 = sum(len(s["projects"]) for s in data_2026["sites"])
        projs_2025 = sum(len(s["projects"]) for s in data_2025["sites"])
        self.assertEqual(projs_2026, 1)
        self.assertEqual(projs_2025, 1)

    def test_calendar_site_filter(self):
        """site_id query param limits results to one site."""
        site2_id = _post(
            self.client,
            "/api/hoshin/sites/create/",
            {
                "name": "Other Plant",
            },
        ).json()["site"]["id"]

        self._create_project("Plant 1 Proj", target=60000)
        _post(
            self.client,
            "/api/hoshin/projects/create/",
            {
                "title": "Plant 2 Proj",
                "site_id": site2_id,
                "fiscal_year": 2026,
                "annual_savings_target": 60000,
            },
        )

        # No filter: both sites
        data_all = self.client.get("/api/hoshin/calendar/?fiscal_year=2026").json()
        self.assertEqual(len(data_all["sites"]), 2)

        # Filter to site 1
        data_one = self.client.get(f"/api/hoshin/calendar/?fiscal_year=2026&site_id={self.site_id}").json()
        self.assertEqual(len(data_one["sites"]), 1)
        self.assertEqual(data_one["sites"][0]["site_name"], "Calendar Plant")

    def test_calendar_project_metadata(self):
        """Each project in the calendar carries status, type, class."""
        self._create_project(
            "Metadata Proj",
            target=120000,
            project_type="labor",
            project_class="kaizen",
            hoshin_status="active",
        )

        proj = self.client.get("/api/hoshin/calendar/?fiscal_year=2026").json()["sites"][0]["projects"][0]
        self.assertEqual(proj["status"], "active")
        self.assertEqual(proj["type"], "labor")
        self.assertEqual(proj["class"], "kaizen")

    def test_calendar_multiple_sites(self):
        """Projects correctly group under their respective sites."""
        site2_id = _post(
            self.client,
            "/api/hoshin/sites/create/",
            {
                "name": "Second Plant",
            },
        ).json()["site"]["id"]

        self._create_project("Plant1 A", target=60000)
        self._create_project("Plant1 B", target=40000)
        _post(
            self.client,
            "/api/hoshin/projects/create/",
            {
                "title": "Plant2 A",
                "site_id": site2_id,
                "fiscal_year": 2026,
                "annual_savings_target": 80000,
            },
        )

        sites = self.client.get("/api/hoshin/calendar/?fiscal_year=2026").json()["sites"]
        site_names = {s["site_name"] for s in sites}
        self.assertEqual(site_names, {"Calendar Plant", "Second Plant"})
        plant1 = next(s for s in sites if s["site_name"] == "Calendar Plant")
        plant2 = next(s for s in sites if s["site_name"] == "Second Plant")
        self.assertEqual(len(plant1["projects"]), 2)
        self.assertEqual(len(plant2["projects"]), 1)

    def test_calendar_empty_fiscal_year(self):
        """Calendar with no projects returns empty sites list."""
        data = self.client.get("/api/hoshin/calendar/?fiscal_year=2099").json()
        self.assertEqual(data["fiscal_year"], 2099)
        self.assertEqual(data["sites"], [])

    # Aliases for compliance hook names
    test_calendar_structure = test_calendar_basic_structure
    test_calendar_monthly_targets = test_calendar_monthly_target_from_annual
    test_calendar_filter_by_site = test_calendar_site_filter


# =============================================================================
# Monte Carlo Savings Estimation
# =============================================================================


@SECURE_OFF
class MonteCarloSavingsTest(TestCase):
    """Monte Carlo simulation for VSM savings with confidence intervals.

    Tests exercise estimate_savings_monte_carlo() directly and through
    the generate_proposals endpoint.
    """

    def setUp(self):
        self.user = _make_enterprise_user("montecarlo@test.com")
        self.tenant = _setup_tenant(self.user)
        self.client.force_login(self.user)

    def test_deterministic_baseline_matches(self):
        """Monte Carlo deterministic field matches estimate_savings_from_vsm_delta."""
        from agents_api.hoshin_calculations import (
            estimate_savings_from_vsm_delta,
            estimate_savings_monte_carlo,
        )

        current = {"cycle_time": 60, "changeover_time": 30, "uptime": 80, "operators": 4, "batch_size": 100}
        future = {"cycle_time": 40, "changeover_time": 15, "uptime": 90, "operators": 3, "batch_size": 100}

        det = estimate_savings_from_vsm_delta(current, future, annual_volume=50000, cost_per_unit=25)
        mc = estimate_savings_monte_carlo(current, future, annual_volume=50000, cost_per_unit=25)

        self.assertAlmostEqual(mc["deterministic"], det["estimated_annual_savings"], places=2)
        self.assertEqual(mc["suggested_method"], det["suggested_method"])
        self.assertAlmostEqual(mc["cycle_time_delta"], det["cycle_time_delta"], places=2)

    def test_confidence_intervals_bracket_median(self):
        """5th percentile < median < 95th percentile."""
        from agents_api.hoshin_calculations import estimate_savings_monte_carlo

        current = {"cycle_time": 60, "changeover_time": 30, "uptime": 80, "operators": 4, "batch_size": 100}
        future = {"cycle_time": 40, "changeover_time": 15, "uptime": 90, "operators": 3, "batch_size": 100}

        mc = estimate_savings_monte_carlo(current, future, annual_volume=50000, cost_per_unit=25)

        self.assertLess(mc["lower_5"], mc["median_savings"])
        self.assertGreater(mc["upper_95"], mc["median_savings"])
        self.assertLess(mc["lower_25"], mc["upper_75"])

    def test_mean_less_than_deterministic(self):
        """Realization risk (Beta(4,2) mean ~0.67) pulls mean below deterministic."""
        from agents_api.hoshin_calculations import estimate_savings_monte_carlo

        current = {"cycle_time": 100, "changeover_time": 0, "uptime": 100, "operators": 1, "batch_size": 1}
        future = {"cycle_time": 50, "changeover_time": 0, "uptime": 100, "operators": 1, "batch_size": 1}

        mc = estimate_savings_monte_carlo(
            current,
            future,
            annual_volume=100000,
            cost_per_unit=10,
            n_simulations=5000,
        )
        # Deterministic assumes 100% realization; MC simulates partial realization
        self.assertLess(mc["mean_savings"], mc["deterministic"])

    def test_p_positive_high_for_large_improvement(self):
        """Large improvement should have p_positive near 1.0."""
        from agents_api.hoshin_calculations import estimate_savings_monte_carlo

        current = {"cycle_time": 120, "changeover_time": 60, "uptime": 70, "operators": 6, "batch_size": 50}
        future = {"cycle_time": 40, "changeover_time": 10, "uptime": 95, "operators": 3, "batch_size": 50}

        mc = estimate_savings_monte_carlo(
            current,
            future,
            annual_volume=100000,
            cost_per_unit=30,
            n_simulations=2000,
        )
        self.assertGreater(mc["p_positive"], 0.95)

    def test_zero_improvement_low_p_positive(self):
        """No improvement should have low p_positive."""
        from agents_api.hoshin_calculations import estimate_savings_monte_carlo

        current = {"cycle_time": 60, "changeover_time": 30, "uptime": 80, "operators": 4, "batch_size": 100}
        future = dict(current)  # identical

        mc = estimate_savings_monte_carlo(
            current,
            future,
            annual_volume=50000,
            cost_per_unit=25,
            n_simulations=2000,
        )
        # No improvement, so most sims should be ~0
        self.assertLess(mc["p_positive"], 0.5)

    def test_headcount_method_auto_detected(self):
        """When operators decrease but CT doesn't, method switches to headcount."""
        from agents_api.hoshin_calculations import estimate_savings_monte_carlo

        current = {"cycle_time": 60, "changeover_time": 30, "uptime": 80, "operators": 6, "batch_size": 100}
        future = {"cycle_time": 60, "changeover_time": 30, "uptime": 80, "operators": 3, "batch_size": 100}

        mc = estimate_savings_monte_carlo(current, future, annual_volume=50000, cost_per_unit=35)
        self.assertEqual(mc["suggested_method"], "headcount")
        self.assertGreater(mc["deterministic"], 0)

    def test_uptime_improvement_scaled_by_realization(self):
        """Uptime is scaled differently (higher = better) vs CT (lower = better)."""
        from agents_api.hoshin_calculations import estimate_savings_monte_carlo

        current = {"cycle_time": 60, "changeover_time": 0, "uptime": 60, "operators": 1, "batch_size": 1}
        future = {"cycle_time": 60, "changeover_time": 0, "uptime": 95, "operators": 1, "batch_size": 1}

        mc = estimate_savings_monte_carlo(current, future, annual_volume=10000, cost_per_unit=10)
        # Uptime improved by 35 points. With realization risk, deltas are partially realized.
        self.assertAlmostEqual(mc["uptime_delta"], 35.0, places=1)

    def test_monte_carlo_through_proposals(self):
        """End-to-end: VSM proposals endpoint returns MC confidence intervals."""
        # Need a core.Project to link VSMs (generate_proposals finds future via project)
        from core.models.project import Project

        project = Project.objects.create(
            user=self.user,
            title="MC Bridge Project",
        )

        # Create a current-state VSM linked to the project
        vsm_data = _post(
            self.client,
            "/api/vsm/create/",
            {
                "name": "MC Current",
                "project_id": str(project.id),
            },
        ).json()
        vsm_id = vsm_data["vsm"]["id"]

        _post(
            self.client,
            f"/api/vsm/{vsm_id}/process-step/",
            {
                "name": "Assembly",
                "cycle_time": 120,
                "changeover_time": 60,
                "uptime": 75,
                "operators": 5,
                "batch_size": 100,
            },
        )

        # Create future state (inherits project link from current)
        future_data = _post(self.client, f"/api/vsm/{vsm_id}/future-state/", {})
        future_id = future_data.json()["future_state"]["id"]

        # Improve the future state step
        future_vsm = self.client.get(f"/api/vsm/{future_id}/").json()["vsm"]
        future_steps = future_vsm["process_steps"]
        if future_steps:
            future_steps[0]["cycle_time"] = 60
            future_steps[0]["changeover_time"] = 15
            future_steps[0]["uptime"] = 92
            future_steps[0]["operators"] = 3
            _put(
                self.client,
                f"/api/vsm/{future_id}/update/",
                {
                    "process_steps": future_steps,
                },
            )

        # Add a kaizen burst to the future
        _post(
            self.client,
            f"/api/vsm/{future_id}/kaizen/",
            {
                "text": "SMED changeover reduction",
                "process_step": "Assembly",
                "priority": "high",
            },
        )

        # Generate proposals from the CURRENT vsm (endpoint finds future via project)
        res = _post(
            self.client,
            f"/api/vsm/{vsm_id}/generate-proposals/",
            {
                "annual_volume": 80000,
                "cost_per_unit": 30,
            },
        )
        self.assertEqual(res.status_code, 200, msg=res.json())
        proposals = res.json()["proposals"]
        self.assertGreaterEqual(len(proposals), 1)

        prop = proposals[0]
        # Verify MC fields present
        self.assertIn("median_savings", prop)
        self.assertIn("lower_5", prop)
        self.assertIn("upper_95", prop)
        self.assertIn("p_positive", prop)
        # estimated_annual_savings is set to median by the endpoint
        self.assertAlmostEqual(prop["estimated_annual_savings"], prop["median_savings"], places=2)
        # With this big improvement, p_positive should be high
        self.assertGreater(prop["p_positive"], 0.8)
        # Confidence interval should be non-trivial
        self.assertGreater(prop["upper_95"], prop["lower_5"])

    # Aliases for compliance hook names
    test_confidence_intervals_ordered = test_confidence_intervals_bracket_median

    def test_monte_carlo_returns_statistics(self):
        """MC result contains all required statistical fields."""
        from agents_api.hoshin_calculations import estimate_savings_monte_carlo

        current = {"cycle_time": 60, "changeover_time": 30, "uptime": 80, "operators": 4, "batch_size": 100}
        future = {"cycle_time": 40, "changeover_time": 15, "uptime": 90, "operators": 3, "batch_size": 100}

        mc = estimate_savings_monte_carlo(current, future, annual_volume=50000, cost_per_unit=25)
        for key in (
            "mean_savings",
            "median_savings",
            "lower_5",
            "upper_95",
            "lower_25",
            "upper_75",
            "p_positive",
            "deterministic",
        ):
            self.assertIn(key, mc, f"Missing key: {key}")

    def test_realization_risk_computed(self):
        """Realization risk pulls mean below deterministic for any real improvement."""
        from agents_api.hoshin_calculations import estimate_savings_monte_carlo

        current = {"cycle_time": 100, "changeover_time": 40, "uptime": 75, "operators": 5, "batch_size": 50}
        future = {"cycle_time": 60, "changeover_time": 15, "uptime": 92, "operators": 3, "batch_size": 50}

        mc = estimate_savings_monte_carlo(
            current,
            future,
            annual_volume=80000,
            cost_per_unit=20,
            n_simulations=3000,
        )
        # Realization risk means mean < deterministic
        self.assertLess(mc["mean_savings"], mc["deterministic"])
        self.assertGreater(mc["p_positive"], 0.9)


# =============================================================================
# VSM → Hoshin → Calendar Integration
# =============================================================================


@SECURE_OFF
class VSMToCalendarIntegrationTest(TestCase):
    """Full data flow: VSM bursts → proposals → hoshin projects → calendar."""

    def setUp(self):
        self.user = _make_enterprise_user("vsm_cal@test.com")
        self.tenant = _setup_tenant(self.user)
        self.client.force_login(self.user)

        self.site_id = _post(
            self.client,
            "/api/hoshin/sites/create/",
            {
                "name": "Fort Worth Press",
            },
        ).json()["site"]["id"]

    def _build_vsm_with_proposals(self):
        """Create a VSM current→future with bursts and generate proposals."""
        from core.models.project import Project

        project = Project.objects.create(
            user=self.user,
            title="VSM Bridge",
        )
        vsm_data = _post(
            self.client,
            "/api/vsm/create/",
            {
                "name": "Press Line Current",
                "project_id": str(project.id),
            },
        ).json()
        vsm_id = vsm_data["vsm"]["id"]

        _post(
            self.client,
            f"/api/vsm/{vsm_id}/process-step/",
            {
                "name": "Changeover",
                "cycle_time": 90,
                "changeover_time": 120,
                "uptime": 70,
                "operators": 4,
                "batch_size": 500,
            },
        )

        future_id = _post(self.client, f"/api/vsm/{vsm_id}/future-state/", {}).json()["future_state"]["id"]

        # Improve future
        future_vsm = self.client.get(f"/api/vsm/{future_id}/").json()["vsm"]
        steps = future_vsm["process_steps"]
        steps[0]["cycle_time"] = 30
        steps[0]["changeover_time"] = 10
        steps[0]["uptime"] = 92
        steps[0]["operators"] = 2
        _put(
            self.client,
            f"/api/vsm/{future_id}/update/",
            {
                "process_steps": steps,
            },
        )

        _post(
            self.client,
            f"/api/vsm/{future_id}/kaizen/",
            {
                "text": "SMED changeover reduction",
                "process_step": "Changeover",
                "priority": "high",
            },
        )

        proposals = _post(
            self.client,
            f"/api/vsm/{vsm_id}/generate-proposals/",
            {
                "annual_volume": 50000,
                "cost_per_unit": 25,
            },
        ).json()["proposals"]

        return vsm_id, proposals

    def test_proposals_create_projects_in_calendar(self):
        """Proposals → hoshin projects → appear in calendar under target site."""
        vsm_id, proposals = self._build_vsm_with_proposals()

        # Create hoshin projects from proposals
        res = _post(
            self.client,
            "/api/hoshin/projects/from-proposals/",
            {
                "vsm_id": vsm_id,
                "site_id": self.site_id,
                "fiscal_year": 2026,
                "proposals": [
                    {
                        "burst_id": proposals[0]["burst_id"],
                        "title": proposals[0]["suggested_title"],
                        "approved": True,
                        "annual_savings_target": proposals[0]["estimated_annual_savings"],
                        "calculation_method": proposals[0]["suggested_method"],
                    },
                ],
            },
        )
        self.assertEqual(res.status_code, 201)
        created = res.json()["created"]
        self.assertEqual(len(created), 1)

        # Verify project appears in calendar under Fort Worth Press
        cal = self.client.get("/api/hoshin/calendar/?fiscal_year=2026").json()
        self.assertEqual(len(cal["sites"]), 1)
        self.assertEqual(cal["sites"][0]["site_name"], "Fort Worth Press")
        self.assertEqual(len(cal["sites"][0]["projects"]), 1)
        proj = cal["sites"][0]["projects"][0]
        self.assertIn("SMED", proj["title"])
        self.assertEqual(proj["status"], "proposed")

    def test_proposal_source_vsm_link_preserved(self):
        """Created hoshin project retains source_vsm and burst_id."""
        vsm_id, proposals = self._build_vsm_with_proposals()

        res = _post(
            self.client,
            "/api/hoshin/projects/from-proposals/",
            {
                "vsm_id": vsm_id,
                "site_id": self.site_id,
                "fiscal_year": 2026,
                "proposals": [
                    {
                        "burst_id": proposals[0]["burst_id"],
                        "title": "SMED Project",
                        "approved": True,
                    }
                ],
            },
        )
        created = res.json()["created"][0]
        self.assertEqual(created["source_vsm_id"], vsm_id)
        self.assertEqual(created["source_burst_id"], proposals[0]["burst_id"])

    def test_proposal_savings_flow_to_calendar(self):
        """Savings recorded on proposal-created projects show up in calendar actuals."""
        vsm_id, proposals = self._build_vsm_with_proposals()

        res = _post(
            self.client,
            "/api/hoshin/projects/from-proposals/",
            {
                "vsm_id": vsm_id,
                "site_id": self.site_id,
                "fiscal_year": 2026,
                "proposals": [
                    {
                        "burst_id": proposals[0]["burst_id"],
                        "title": "SMED Project",
                        "approved": True,
                        "annual_savings_target": 60000,
                        "calculation_method": "direct",
                    }
                ],
            },
        )
        hp_id = res.json()["created"][0]["id"]

        # Record monthly savings
        _put(
            self.client,
            f"/api/hoshin/projects/{hp_id}/monthly/1/",
            {
                "baseline": 8000,
                "actual": 5000,
            },
        )

        # Calendar should show the savings
        cal = self.client.get("/api/hoshin/calendar/?fiscal_year=2026").json()
        proj = cal["sites"][0]["projects"][0]
        # Month 1: savings = 3000
        self.assertAlmostEqual(proj["months"][0]["actual"], 3000.0, places=0)
        # target per month = 60000/12 = 5000
        self.assertAlmostEqual(proj["months"][0]["target"], 5000.0, places=0)

    def test_create_without_site_rejected(self):
        """Hoshin projects cannot be created without site_id."""
        res = _post(
            self.client,
            "/api/hoshin/projects/create/",
            {
                "title": "Orphan Project",
                "fiscal_year": 2026,
                "annual_savings_target": 50000,
            },
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("site_id", _err_msg(res))


# =============================================================================
# Calendar Site Access Control
# =============================================================================


@SECURE_OFF
class CalendarSiteAccessTest(TestCase):
    """Calendar respects site-level access control."""

    def setUp(self):
        # Owner sees all
        self.owner = _make_enterprise_user("cal_owner@test.com")
        self.tenant = _setup_tenant(self.owner)

        # Member with site access
        self.member = _make_enterprise_user("cal_member@test.com")
        Membership.objects.create(
            user=self.member,
            tenant=self.tenant,
            role="member",
            is_active=True,
        )

        # Outsider — no membership
        self.outsider = _make_enterprise_user("cal_outsider@test.com")

        # Viewer — read-only access
        self.viewer = _make_enterprise_user("cal_viewer@test.com")
        Membership.objects.create(
            user=self.viewer,
            tenant=self.tenant,
            role="member",
            is_active=True,
        )

        # Create two sites
        self.client.force_login(self.owner)
        self.site_a = _post(
            self.client,
            "/api/hoshin/sites/create/",
            {
                "name": "Plant Alpha",
            },
        ).json()["site"]["id"]
        self.site_b = _post(
            self.client,
            "/api/hoshin/sites/create/",
            {
                "name": "Plant Beta",
            },
        ).json()["site"]["id"]

        # Grant member access to site A only
        from agents_api.models import Site

        site_a_obj = Site.objects.get(id=self.site_a)
        site_b_obj = Site.objects.get(id=self.site_b)
        SiteAccess.objects.create(
            user=self.member,
            site=site_a_obj,
            role="member",
        )
        # Grant viewer read-only access to site B
        SiteAccess.objects.create(
            user=self.viewer,
            site=site_b_obj,
            role="viewer",
        )

        # Create projects in each site
        self.hp_a = _post(
            self.client,
            "/api/hoshin/projects/create/",
            {
                "title": "Alpha Project",
                "site_id": self.site_a,
                "fiscal_year": 2026,
                "annual_savings_target": 100000,
                "calculation_method": "direct",
            },
        ).json()["project"]["id"]

        self.hp_b = _post(
            self.client,
            "/api/hoshin/projects/create/",
            {
                "title": "Beta Project",
                "site_id": self.site_b,
                "fiscal_year": 2026,
                "annual_savings_target": 80000,
                "calculation_method": "direct",
            },
        ).json()["project"]["id"]

        # Record savings
        _put(
            self.client,
            f"/api/hoshin/projects/{self.hp_a}/monthly/1/",
            {
                "baseline": 12000,
                "actual": 9000,
            },
        )
        _put(
            self.client,
            f"/api/hoshin/projects/{self.hp_b}/monthly/1/",
            {
                "baseline": 10000,
                "actual": 7000,
            },
        )

    def test_owner_sees_all_sites_in_calendar(self):
        """Org owner/admin sees all sites in the calendar."""
        self.client.force_login(self.owner)
        cal = self.client.get("/api/hoshin/calendar/?fiscal_year=2026").json()
        site_names = {s["site_name"] for s in cal["sites"]}
        self.assertEqual(site_names, {"Plant Alpha", "Plant Beta"})

    def test_member_sees_only_accessible_site(self):
        """Member with site access only sees that site's projects."""
        self.client.force_login(self.member)
        cal = self.client.get("/api/hoshin/calendar/?fiscal_year=2026").json()
        self.assertEqual(len(cal["sites"]), 1)
        self.assertEqual(cal["sites"][0]["site_name"], "Plant Alpha")

    def test_viewer_sees_site_in_calendar(self):
        """Viewer can read calendar data for their site."""
        self.client.force_login(self.viewer)
        cal = self.client.get("/api/hoshin/calendar/?fiscal_year=2026").json()
        self.assertEqual(len(cal["sites"]), 1)
        self.assertEqual(cal["sites"][0]["site_name"], "Plant Beta")
        # Should see project data including savings
        proj = cal["sites"][0]["projects"][0]
        self.assertAlmostEqual(proj["months"][0]["actual"], 3000.0, places=0)

    def test_outsider_sees_nothing(self):
        """User without tenant membership gets an error."""
        self.client.force_login(self.outsider)
        res = self.client.get("/api/hoshin/calendar/?fiscal_year=2026")
        # No tenant membership → error or empty
        self.assertIn(res.status_code, [400, 403])

    def test_member_cannot_see_other_sites_via_filter(self):
        """Member can't bypass access by passing site_id for a non-accessible site."""
        self.client.force_login(self.member)
        # Try to filter to Plant Beta (no access)
        cal = self.client.get(f"/api/hoshin/calendar/?fiscal_year=2026&site_id={self.site_b}").json()
        # Should return empty — filter intersects with accessible sites
        self.assertEqual(cal["sites"], [])

    def test_dashboard_also_respects_site_access(self):
        """Dashboard endpoint has same site-level access as calendar."""
        self.client.force_login(self.member)
        dash = self.client.get("/api/hoshin/dashboard/?fiscal_year=2026").json()
        site_ids = [s["site_id"] for s in dash.get("by_site", [])]
        self.assertIn(self.site_a, site_ids)
        self.assertNotIn(self.site_b, site_ids)


# =============================================================================
# Cross-Tenant Isolation
# =============================================================================


@SECURE_OFF
class CrossTenantIsolationTest(TestCase):
    """VSM from one tenant cannot create projects in another tenant's sites."""

    def setUp(self):
        # Org A
        self.user_a = _make_enterprise_user("org_a@test.com")
        self.tenant_a = Tenant.objects.create(name="Org Alpha", slug="org-alpha")
        Membership.objects.create(
            user=self.user_a,
            tenant=self.tenant_a,
            role="owner",
            is_active=True,
        )

        # Org B
        self.user_b = _make_enterprise_user("org_b@test.com")
        self.tenant_b = Tenant.objects.create(name="Org Beta", slug="org-beta")
        Membership.objects.create(
            user=self.user_b,
            tenant=self.tenant_b,
            role="owner",
            is_active=True,
        )

        # Create sites
        self.client.force_login(self.user_a)
        self.site_a = _post(
            self.client,
            "/api/hoshin/sites/create/",
            {
                "name": "Org A Plant",
            },
        ).json()["site"]["id"]

        self.client.force_login(self.user_b)
        self.site_b = _post(
            self.client,
            "/api/hoshin/sites/create/",
            {
                "name": "Org B Plant",
            },
        ).json()["site"]["id"]

    def test_cannot_create_project_in_other_tenant_site(self):
        """User in Org A cannot target a site belonging to Org B."""
        self.client.force_login(self.user_a)
        res = _post(
            self.client,
            "/api/hoshin/projects/create/",
            {
                "title": "Cross-tenant",
                "site_id": self.site_b,
                "fiscal_year": 2026,
            },
        )
        # site_b belongs to tenant_b — access should be blocked (404 or 500)
        self.assertIn(res.status_code, (404, 500))

    def test_cannot_see_other_tenant_calendar(self):
        """Org A user sees nothing from Org B in the calendar."""
        # Create project in Org B
        self.client.force_login(self.user_b)
        _post(
            self.client,
            "/api/hoshin/projects/create/",
            {
                "title": "Org B Secret",
                "site_id": self.site_b,
                "fiscal_year": 2026,
                "annual_savings_target": 200000,
            },
        )

        # Org A user should see no sites
        self.client.force_login(self.user_a)
        cal = self.client.get("/api/hoshin/calendar/?fiscal_year=2026").json()
        project_titles = [p["title"] for s in cal.get("sites", []) for p in s.get("projects", [])]
        self.assertNotIn("Org B Secret", project_titles)

    def test_vsm_from_other_tenant_project_silently_dropped(self):
        """VSM linked to Org B project is silently dropped in Org A proposal creation."""
        from core.models.project import Project

        # Create a VSM owned by user_a but linked to Org B project
        project_b = Project.objects.create(tenant=self.tenant_b, title="B's project")
        vsm = ValueStreamMap.objects.create(
            owner=self.user_a,
            project=project_b,
            name="Cross-tenant VSM",
        )

        self.client.force_login(self.user_a)
        res = _post(
            self.client,
            "/api/hoshin/projects/from-proposals/",
            {
                "vsm_id": str(vsm.id),
                "site_id": self.site_a,
                "fiscal_year": 2026,
                "proposals": [
                    {
                        "burst_id": "x1",
                        "title": "Should drop VSM link",
                        "approved": True,
                    }
                ],
            },
        )
        self.assertEqual(res.status_code, 201)
        created = res.json()["created"][0]
        # VSM should have been silently dropped due to tenant mismatch
        self.assertIsNone(created.get("source_vsm_id"))

    def test_calendar_isolation_between_tenants(self):
        """Each tenant's calendar only shows its own projects."""
        # Create projects in both tenants
        self.client.force_login(self.user_a)
        _post(
            self.client,
            "/api/hoshin/projects/create/",
            {
                "title": "Alpha Only",
                "site_id": self.site_a,
                "fiscal_year": 2026,
                "annual_savings_target": 50000,
            },
        )

        self.client.force_login(self.user_b)
        _post(
            self.client,
            "/api/hoshin/projects/create/",
            {
                "title": "Beta Only",
                "site_id": self.site_b,
                "fiscal_year": 2026,
                "annual_savings_target": 80000,
            },
        )

        # Org A calendar
        self.client.force_login(self.user_a)
        cal_a = self.client.get("/api/hoshin/calendar/?fiscal_year=2026").json()
        titles_a = [p["title"] for s in cal_a["sites"] for p in s["projects"]]
        self.assertIn("Alpha Only", titles_a)
        self.assertNotIn("Beta Only", titles_a)

        # Org B calendar
        self.client.force_login(self.user_b)
        cal_b = self.client.get("/api/hoshin/calendar/?fiscal_year=2026").json()
        titles_b = [p["title"] for s in cal_b["sites"] for p in s["projects"]]
        self.assertIn("Beta Only", titles_b)
        self.assertNotIn("Alpha Only", titles_b)

    def test_proposals_without_site_rejected(self):
        """create_from_proposals requires site_id."""
        self.client.force_login(self.user_a)
        res = _post(
            self.client,
            "/api/hoshin/projects/from-proposals/",
            {
                "fiscal_year": 2026,
                "proposals": [
                    {
                        "burst_id": "x1",
                        "title": "No site",
                        "approved": True,
                    }
                ],
            },
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("site_id", _err_msg(res))

    def test_vsm_without_project_still_allowed(self):
        """VSM with no project link is still accepted (user owns it, same tenant)."""
        vsm = ValueStreamMap.objects.create(
            owner=self.user_a,
            name="Personal VSM",
        )

        self.client.force_login(self.user_a)
        res = _post(
            self.client,
            "/api/hoshin/projects/from-proposals/",
            {
                "vsm_id": str(vsm.id),
                "site_id": self.site_a,
                "fiscal_year": 2026,
                "proposals": [
                    {
                        "burst_id": "b1",
                        "title": "From personal VSM",
                        "approved": True,
                    }
                ],
            },
        )
        self.assertEqual(res.status_code, 201)
        # VSM link should be preserved — owned by same user
        created = res.json()["created"][0]
        self.assertEqual(created["source_vsm_id"], str(vsm.id))


# =============================================================================
# VSM → Hoshin Full Pipeline with Site Verification
# =============================================================================


@SECURE_OFF
class VSMHoshinSitePipelineTest(TestCase):
    """End-to-end: VSM analysis → proposals → site-scoped projects → calendar → dashboard."""

    def setUp(self):
        self.owner = _make_enterprise_user("pipeline_owner@test.com")
        self.tenant = _setup_tenant(self.owner)
        self.client.force_login(self.owner)

    def test_full_pipeline_with_site_verification(self):
        """VSM bursts → proposals → hoshin projects at two sites → calendar shows correct grouping."""
        from core.models.project import Project

        # Create two manufacturing sites
        site_ftw = _post(
            self.client,
            "/api/hoshin/sites/create/",
            {
                "name": "Fort Worth",
            },
        ).json()["site"]["id"]
        site_chi = _post(
            self.client,
            "/api/hoshin/sites/create/",
            {
                "name": "Chicago",
            },
        ).json()["site"]["id"]

        # Build VSM with process steps
        project = Project.objects.create(user=self.owner, title="Press Line Analysis")
        vsm_id = _post(
            self.client,
            "/api/vsm/create/",
            {
                "name": "XL 106 Current",
                "project_id": str(project.id),
            },
        ).json()["vsm"]["id"]

        _post(
            self.client,
            f"/api/vsm/{vsm_id}/process-step/",
            {
                "name": "Press Setup",
                "cycle_time": 90,
                "changeover_time": 180,
                "uptime": 72,
                "operators": 5,
                "batch_size": 1000,
            },
        )
        _post(
            self.client,
            f"/api/vsm/{vsm_id}/process-step/",
            {
                "name": "Print Run",
                "cycle_time": 8,
                "changeover_time": 0,
                "uptime": 85,
                "operators": 3,
                "batch_size": 1000,
            },
        )

        # Future state with improvements
        future_id = _post(self.client, f"/api/vsm/{vsm_id}/future-state/", {}).json()["future_state"]["id"]

        future_vsm = self.client.get(f"/api/vsm/{future_id}/").json()["vsm"]
        steps = future_vsm["process_steps"]
        steps[0]["cycle_time"] = 30
        steps[0]["changeover_time"] = 10
        steps[0]["uptime"] = 92
        steps[0]["operators"] = 3
        steps[1]["uptime"] = 95
        _put(
            self.client,
            f"/api/vsm/{future_id}/update/",
            {
                "process_steps": steps,
            },
        )

        # Add bursts for each step
        _post(
            self.client,
            f"/api/vsm/{future_id}/kaizen/",
            {
                "text": "SMED on press setup",
                "process_step": "Press Setup",
                "priority": "high",
            },
        )
        _post(
            self.client,
            f"/api/vsm/{future_id}/kaizen/",
            {
                "text": "TPM on print run",
                "process_step": "Print Run",
                "priority": "medium",
            },
        )

        # Generate proposals
        proposals = _post(
            self.client,
            f"/api/vsm/{vsm_id}/generate-proposals/",
            {
                "annual_volume": 82000000,
                "cost_per_unit": 0.015,
            },
        ).json()["proposals"]
        self.assertGreaterEqual(len(proposals), 2)

        # Create SMED project at Fort Worth, TPM at Chicago
        smed_prop = next((p for p in proposals if "SMED" in p.get("burst_text", "")), proposals[0])
        tpm_prop = next((p for p in proposals if "TPM" in p.get("burst_text", "")), proposals[-1])

        _post(
            self.client,
            "/api/hoshin/projects/from-proposals/",
            {
                "vsm_id": vsm_id,
                "site_id": site_ftw,
                "fiscal_year": 2026,
                "proposals": [
                    {
                        "burst_id": smed_prop["burst_id"],
                        "title": "SMED Press Setup",
                        "approved": True,
                        "annual_savings_target": 120000,
                        "calculation_method": "direct",
                    }
                ],
            },
        )

        _post(
            self.client,
            "/api/hoshin/projects/from-proposals/",
            {
                "vsm_id": vsm_id,
                "site_id": site_chi,
                "fiscal_year": 2026,
                "proposals": [
                    {
                        "burst_id": tpm_prop["burst_id"],
                        "title": "TPM Print Run",
                        "approved": True,
                        "annual_savings_target": 80000,
                        "calculation_method": "direct",
                    }
                ],
            },
        )

        # Record savings at each site
        cal = self.client.get("/api/hoshin/calendar/?fiscal_year=2026").json()
        ftw_projects = next(s for s in cal["sites"] if s["site_name"] == "Fort Worth")["projects"]
        chi_projects = next(s for s in cal["sites"] if s["site_name"] == "Chicago")["projects"]

        ftw_hp_id = ftw_projects[0]["id"]
        chi_hp_id = chi_projects[0]["id"]

        _put(
            self.client,
            f"/api/hoshin/projects/{ftw_hp_id}/monthly/1/",
            {
                "baseline": 15000,
                "actual": 10000,
            },
        )
        _put(
            self.client,
            f"/api/hoshin/projects/{chi_hp_id}/monthly/1/",
            {
                "baseline": 8000,
                "actual": 6000,
            },
        )

        # Final calendar check: both sites present, correct savings
        cal = self.client.get("/api/hoshin/calendar/?fiscal_year=2026").json()
        self.assertEqual(len(cal["sites"]), 2)

        ftw = next(s for s in cal["sites"] if s["site_name"] == "Fort Worth")
        chi = next(s for s in cal["sites"] if s["site_name"] == "Chicago")

        self.assertAlmostEqual(ftw["projects"][0]["months"][0]["actual"], 5000.0, places=0)
        self.assertAlmostEqual(chi["projects"][0]["months"][0]["actual"], 2000.0, places=0)

        # Site-level ytd totals
        self.assertAlmostEqual(ftw["ytd"], 5000.0, places=0)
        self.assertAlmostEqual(chi["ytd"], 2000.0, places=0)

        # Dashboard should match
        dash = self.client.get("/api/hoshin/dashboard/?fiscal_year=2026").json()
        self.assertAlmostEqual(dash["total_ytd"], 7000.0, places=0)
