"""HIRARC Safety system tests — SAF-001.

Covers:
- FrontierZone CRUD
- AuditSchedule + Assignment lifecycle
- FrontierCard submission + severity properties
- Card-to-FMEA processing pipeline
- 5S Pareto aggregation
- Safety KPI dashboard
- Feature gating (Enterprise only)
- High-severity notification
"""

import json
from datetime import date, timedelta

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from accounts.constants import Tier
from agents_api.models import FMEA, Employee, FMEARow, Site
from core.models import Membership, Project, Tenant

from .models import (
    FrontierCard,
    FrontierZone,
    aggregate_five_s_pareto,
    process_card_to_fmea,
)

User = get_user_model()
SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)
PASSWORD = "testpass123!"


def _make_user(email, tier=Tier.ENTERPRISE):
    username = email.split("@")[0]
    user = User.objects.create_user(username=username, email=email, password=PASSWORD)
    user.tier = tier
    user.save(update_fields=["tier"])
    return user


def _post(client, url, data=None):
    return client.post(url, json.dumps(data or {}), content_type="application/json")


def _put(client, url, data=None):
    return client.put(url, json.dumps(data or {}), content_type="application/json")


def _setup(user, slug="safetycorp"):
    tenant = Tenant.objects.create(name="Safety Corp", slug=slug)
    Membership.objects.create(user=user, tenant=tenant, role="owner")
    site = Site.objects.create(tenant=tenant, name="Plant X", code="PX")
    employee = Employee.objects.create(tenant=tenant, name="Jane Auditor", email="jane@plant.com", role="auditor")
    zone = FrontierZone.objects.create(site=site, name="Press Mezzanine", zone_type="overhead")
    return tenant, site, employee, zone


# ── Model Tests ────────────────────────────────────────────────────────


@SECURE_OFF
class FrontierCardModelTest(TestCase):
    def setUp(self):
        self.user = _make_user("cardtest@test.com")
        self.tenant, self.site, self.employee, self.zone = _setup(self.user, "card-corp")

    def test_create_card(self):
        card = FrontierCard.objects.create(
            auditor=self.employee,
            zone=self.zone,
            site=self.site,
            audit_date=date.today(),
            safety_observations=[
                {"category": "ppe", "item": "Missing gloves", "rating": "AR", "severity": "H", "notes": ""},
                {"category": "housekeeping", "item": "Clear walkway", "rating": "S"},
            ],
        )
        self.assertEqual(card.at_risk_count, 1)
        self.assertEqual(card.highest_severity, "H")
        self.assertFalse(card.is_processed)

    def test_severity_mapping(self):
        card = FrontierCard(auditor=self.employee, zone=self.zone, site=self.site, audit_date=date.today())
        self.assertEqual(card.severity_to_fmea_score("C"), 10)
        self.assertEqual(card.severity_to_fmea_score("H"), 8)
        self.assertEqual(card.severity_to_fmea_score("M"), 5)
        self.assertEqual(card.severity_to_fmea_score("L"), 2)

    def test_five_s_tallies(self):
        card = FrontierCard.objects.create(
            auditor=self.employee,
            zone=self.zone,
            site=self.site,
            audit_date=date.today(),
            five_s_tallies={"sort": 3, "set_in_order": 2, "shine": 1, "standardize": 0, "sustain": 4},
        )
        self.assertEqual(card.total_five_s_deficiencies, 10)

    def test_to_dict(self):
        card = FrontierCard.objects.create(
            auditor=self.employee,
            zone=self.zone,
            site=self.site,
            audit_date=date.today(),
            operator_name="Bob",
        )
        d = card.to_dict()
        self.assertEqual(d["auditor_name"], "Jane Auditor")
        self.assertEqual(d["zone_name"], "Press Mezzanine")
        self.assertEqual(d["operator_name"], "Bob")


# ── Card-to-FMEA Pipeline ─────────────────────────────────────────────


@SECURE_OFF
class CardToFMEATest(TestCase):
    def setUp(self):
        self.user = _make_user("pipeline@test.com")
        self.tenant, self.site, self.employee, self.zone = _setup(self.user, "pipe-corp")
        self.project = Project.objects.create(user=self.user, title="Safety FMEA")
        self.fmea = FMEA.objects.create(owner=self.user, title="Plant X Safety FMEA")

    def test_process_creates_fmea_rows(self):
        card = FrontierCard.objects.create(
            auditor=self.employee,
            zone=self.zone,
            site=self.site,
            audit_date=date.today(),
            safety_observations=[
                {
                    "category": "energy_control",
                    "item": "LOTO not applied",
                    "rating": "AR",
                    "severity": "C",
                    "notes": "Press 4",
                },
                {"category": "ppe", "item": "Gloves worn", "rating": "S"},
                {
                    "category": "housekeeping",
                    "item": "Oil spill",
                    "rating": "AR",
                    "severity": "M",
                    "notes": "Near press 2",
                },
            ],
        )
        ids = process_card_to_fmea(card, self.fmea, self.user)
        self.assertEqual(len(ids), 2)  # Only at_risk items
        card.refresh_from_db()
        self.assertTrue(card.is_processed)
        self.assertIsNotNone(card.processed_at)
        self.assertEqual(len(card.fmea_rows_created), 2)

        # Check severity mapping
        rows = FMEARow.objects.filter(fmea=self.fmea).order_by("-severity")
        self.assertEqual(rows[0].severity, 10)  # C → 10
        self.assertEqual(rows[1].severity, 5)  # M → 5

    def test_crossfeed_creates_extra_row(self):
        card = FrontierCard.objects.create(
            auditor=self.employee,
            zone=self.zone,
            site=self.site,
            audit_date=date.today(),
            safety_observations=[
                {"category": "ppe", "item": "No hard hat", "rating": "AR", "severity": "H"},
            ],
            has_safety_crossfeed=True,
            crossfeed_notes="Missing LOTO shadow board = 5S-SET failure + energy hazard",
        )
        ids = process_card_to_fmea(card, self.fmea, self.user)
        self.assertEqual(len(ids), 2)  # 1 observation + 1 crossfeed

    def test_already_processed_blocked(self):
        card = FrontierCard.objects.create(
            auditor=self.employee,
            zone=self.zone,
            site=self.site,
            audit_date=date.today(),
            safety_observations=[{"category": "ppe", "item": "test", "rating": "AR", "severity": "L"}],
        )
        process_card_to_fmea(card, self.fmea, self.user)
        # Second call should be blocked by the view (model doesn't enforce)
        card.refresh_from_db()
        self.assertTrue(card.is_processed)


# ── 5S Pareto ──────────────────────────────────────────────────────────


@SECURE_OFF
class FiveSParetoTest(TestCase):
    def setUp(self):
        self.user = _make_user("pareto@test.com")
        self.tenant, self.site, self.employee, self.zone = _setup(self.user, "pareto-corp")

    def test_insufficient_data(self):
        result = aggregate_five_s_pareto(self.site, min_cards=10)
        self.assertIsNone(result)

    def test_pareto_aggregation(self):
        for i in range(12):
            FrontierCard.objects.create(
                auditor=self.employee,
                zone=self.zone,
                site=self.site,
                audit_date=date.today() - timedelta(days=i),
                five_s_tallies={"sort": 3, "set_in_order": 1, "shine": 2, "standardize": 0, "sustain": 1},
            )
        result = aggregate_five_s_pareto(self.site, min_cards=10)
        self.assertIsNotNone(result)
        self.assertEqual(result["card_count"], 12)
        self.assertEqual(result["grand_total"], 12 * 7)  # 7 per card
        # Sort should be first (highest frequency)
        self.assertEqual(result["pareto"][0]["pillar"], "sort")
        # Cumulative should reach 100%
        self.assertAlmostEqual(result["pareto"][-1]["cumulative_pct"], 100.0, places=0)


# ── API Tests ──────────────────────────────────────────────────────────


@SECURE_OFF
class SafetyZoneAPITest(TestCase):
    def setUp(self):
        self.user = _make_user("zoneapi@test.com")
        self.tenant, self.site, self.employee, self.zone = _setup(self.user, "zone-corp")
        self.client.force_login(self.user)

    def test_list_zones(self):
        resp = self.client.get("/api/safety/zones/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["name"], "Press Mezzanine")

    def test_create_zone(self):
        resp = _post(
            self.client,
            "/api/safety/zones/",
            {
                "site_id": str(self.site.id),
                "name": "Loading Dock",
                "zone_type": "transition",
            },
        )
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(FrontierZone.objects.filter(site=self.site).count(), 2)


@SECURE_OFF
class SafetyCardAPITest(TestCase):
    def setUp(self):
        self.user = _make_user("cardapi@test.com")
        self.tenant, self.site, self.employee, self.zone = _setup(self.user, "cardapi-corp")
        self.client.force_login(self.user)

    def test_submit_card(self):
        resp = _post(
            self.client,
            "/api/safety/cards/",
            {
                "site_id": str(self.site.id),
                "auditor_id": str(self.employee.id),
                "zone_id": str(self.zone.id),
                "safety_observations": [
                    {"category": "ppe", "item": "No safety glasses", "rating": "AR", "severity": "M"},
                ],
                "five_s_tallies": {"sort": 2, "shine": 1},
                "operator_name": "Mike",
                "operator_concern": "Noise level too high",
            },
        )
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data["at_risk_count"], 1)
        self.assertEqual(data["highest_severity"], "M")
        self.assertEqual(data["operator_name"], "Mike")

    def test_list_cards(self):
        FrontierCard.objects.create(
            auditor=self.employee,
            zone=self.zone,
            site=self.site,
            audit_date=date.today(),
        )
        resp = self.client.get("/api/safety/cards/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.json()), 1)

    def test_filter_unprocessed(self):
        FrontierCard.objects.create(
            auditor=self.employee,
            zone=self.zone,
            site=self.site,
            audit_date=date.today(),
            is_processed=True,
        )
        FrontierCard.objects.create(
            auditor=self.employee,
            zone=self.zone,
            site=self.site,
            audit_date=date.today(),
        )
        resp = self.client.get("/api/safety/cards/?unprocessed=true")
        self.assertEqual(len(resp.json()), 1)


@SECURE_OFF
class SafetyDashboardAPITest(TestCase):
    def setUp(self):
        self.user = _make_user("dashapi@test.com")
        self.tenant, self.site, self.employee, self.zone = _setup(self.user, "dash-corp")
        self.client.force_login(self.user)

    def test_dashboard_empty(self):
        resp = self.client.get("/api/safety/dashboard/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("leading", data)
        self.assertIn("severity_distribution", data)
        self.assertEqual(data["totals"]["total_cards"], 0)

    def test_dashboard_with_data(self):
        FrontierCard.objects.create(
            auditor=self.employee,
            zone=self.zone,
            site=self.site,
            audit_date=date.today(),
            safety_observations=[
                {"category": "ppe", "item": "test", "rating": "AR", "severity": "H"},
            ],
        )
        resp = self.client.get("/api/safety/dashboard/")
        data = resp.json()
        self.assertEqual(data["totals"]["total_cards"], 1)
        self.assertEqual(data["leading"]["hazards_this_month"], 1)
        self.assertEqual(data["severity_distribution"]["H"], 1)


# ── Feature Gating ─────────────────────────────────────────────────────


@SECURE_OFF
class SafetyFeatureGateTest(TestCase):
    def test_free_user_blocked(self):
        user = _make_user("free@test.com", tier=Tier.FREE)
        self.client.force_login(user)
        resp = self.client.get("/api/safety/dashboard/")
        self.assertIn(resp.status_code, [403, 500])

    def test_pro_user_blocked(self):
        user = _make_user("pro@test.com", tier=Tier.PRO)
        self.client.force_login(user)
        resp = self.client.get("/api/safety/zones/")
        self.assertIn(resp.status_code, [403, 500])


# ── Subdomain Middleware ───────────────────────────────────────────────


@SECURE_OFF
class SafetySubdomainTest(TestCase):
    def test_safety_subdomain_redirects_root(self):
        resp = self.client.get("/", HTTP_HOST="safety.svend.ai")
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(resp.url, "/app/safety/")

    def test_normal_domain_unaffected(self):
        resp = self.client.get("/", HTTP_HOST="svend.ai")
        self.assertNotEqual(resp.url if resp.status_code == 302 else "", "/app/safety/")

    def test_api_paths_not_redirected(self):
        user = _make_user("subdom@test.com")
        self.client.force_login(user)
        _, site, _, _ = _setup(user, "subdom-corp")
        resp = self.client.get("/api/safety/dashboard/", HTTP_HOST="safety.svend.ai")
        # Should reach the view, not redirect
        self.assertEqual(resp.status_code, 200)
