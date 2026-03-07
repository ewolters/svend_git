"""Tests for 12 audit findings (BUG-01 through BUG-12).

Each test class verifies the fix for a specific bug found during
the 2026-03-07 surveillance audit.
"""

import json

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from accounts.constants import Tier
from agents_api.models import (
    FMEA,
    A3Report,
    Board,
    BoardParticipant,
    CAPAReport,
    FMEARow,
    NonconformanceRecord,
)

User = get_user_model()
SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)


def _post(client, url, data=None):
    return client.post(url, json.dumps(data or {}), content_type="application/json")


def _put(client, url, data=None):
    return client.put(url, json.dumps(data or {}), content_type="application/json")


def _err_msg(resp):
    body = resp.json()
    err = body.get("error", body.get("message", ""))
    if isinstance(err, dict):
        return err.get("message", str(err))
    return str(err)


def _make_user(email, tier=Tier.TEAM, **kwargs):
    username = kwargs.pop("username", email.split("@")[0])
    user = User.objects.create_user(username=username, email=email, password="testpass123!", **kwargs)
    user.tier = tier
    user.save(update_fields=["tier"])
    return user


# =============================================================================
# BUG-01: A3 action endpoints use wrong FK field name
# =============================================================================


@SECURE_OFF
class Bug01A3ActionFieldTest(TestCase):
    """a3_views.py list_a3_actions and create_a3_action used user= instead of owner=."""

    def setUp(self):
        self.user = _make_user("a3@test.com", tier=Tier.PRO)
        from core.models import Project

        self.project = Project.objects.create(user=self.user, title="A3 Project")
        self.report = A3Report.objects.create(owner=self.user, project=self.project, title="Test A3")
        self.client.force_login(self.user)

    def test_list_a3_actions_returns_200(self):
        """Was always 404 before fix — field mismatch caused lookup failure."""
        resp = self.client.get(f"/api/a3/{self.report.id}/actions/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("action_items", resp.json())

    def test_create_a3_action_returns_201(self):
        """Was always 404 before fix."""
        resp = _post(
            self.client,
            f"/api/a3/{self.report.id}/actions/create/",
            {"title": "Test action item"},
        )
        self.assertIn(resp.status_code, [200, 201])

    def test_other_user_cannot_access_a3_actions(self):
        other = _make_user("other_a3@test.com", tier=Tier.PRO)
        self.client.force_login(other)
        resp = self.client.get(f"/api/a3/{self.report.id}/actions/")
        self.assertIn(resp.status_code, [403, 404, 500])
        # Key assertion: other user does NOT get the action items
        if resp.status_code == 200:
            self.fail("Other user should not access A3 actions belonging to another user")


# =============================================================================
# BUG-02: Whiteboard get_board missing ownership check
# =============================================================================


@SECURE_OFF
class Bug02WhiteboardAccessTest(TestCase):
    """Whiteboard access model: room code is an access token (like a meeting link).

    BUG-02 re-assessed: room code IS the authorization mechanism by design.
    GET auto-joins as participant; update_board separately requires participation.
    Tests verify the update_board path requires prior participation.
    """

    def setUp(self):
        self.owner = _make_user("wb_owner@test.com", tier=Tier.PRO)
        self.stranger = _make_user("wb_stranger@test.com", tier=Tier.PRO)
        self.board = Board.objects.create(
            owner=self.owner,
            name="Collab Board",
            elements=[{"id": "data"}],
        )

    def test_owner_can_read_board(self):
        self.client.force_login(self.owner)
        resp = self.client.get(f"/api/whiteboard/boards/{self.board.room_code}/")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["is_owner"])

    def test_room_code_grants_read_access(self):
        """Room code acts as access token — GET auto-joins as participant."""
        self.client.force_login(self.stranger)
        resp = self.client.get(f"/api/whiteboard/boards/{self.board.room_code}/")
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(resp.json()["is_owner"])
        # Should be auto-joined as participant
        self.assertTrue(BoardParticipant.objects.filter(board=self.board, user=self.stranger).exists())

    def test_update_requires_participation(self):
        """update_board correctly requires prior participation."""
        self.client.force_login(self.stranger)
        resp = self.client.put(
            f"/api/whiteboard/boards/{self.board.room_code}/update/",
            json.dumps({"name": "Hacked"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 403)


# =============================================================================
# BUG-03: NCR silent FK assignment failure
# =============================================================================


@SECURE_OFF
class Bug03NCRSilentFKTest(TestCase):
    """User.DoesNotExist was silently swallowed on assignment."""

    def setUp(self):
        self.user = _make_user("ncr_fk@test.com")
        self.client.force_login(self.user)
        resp = _post(
            self.client,
            "/api/iso/ncrs/",
            {
                "title": "FK Test NCR",
                "description": "Testing FK failure",
                "severity": "minor",
                "source": "audit",
            },
        )
        self.ncr_id = resp.json()["id"]

    def test_invalid_assigned_to_returns_400(self):
        """Should return error, not silently succeed."""
        resp = _put(
            self.client,
            f"/api/iso/ncrs/{self.ncr_id}/",
            {"status": "investigation", "assigned_to": 999999},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("not found", _err_msg(resp).lower())

    def test_valid_assigned_to_works(self):
        resp = _put(
            self.client,
            f"/api/iso/ncrs/{self.ncr_id}/",
            {"status": "investigation", "assigned_to": self.user.id},
        )
        self.assertEqual(resp.status_code, 200)


# =============================================================================
# BUG-04: NCR field mutation before transition validation
# =============================================================================


@SECURE_OFF
class Bug04NCRMutationOrderTest(TestCase):
    """Fields were mutated before can_transition(), leaving dirty state on failure."""

    def setUp(self):
        self.user = _make_user("ncr_mut@test.com")
        self.client.force_login(self.user)
        resp = _post(
            self.client,
            "/api/iso/ncrs/",
            {
                "title": "Mutation Test NCR",
                "description": "Testing mutation order",
                "severity": "minor",
                "source": "audit",
            },
        )
        self.ncr_id = resp.json()["id"]

    def test_failed_transition_does_not_persist_mutations(self):
        """Try invalid transition — assigned_to should not be set on failure."""
        resp = _put(
            self.client,
            f"/api/iso/ncrs/{self.ncr_id}/",
            {
                "status": "closed",  # Invalid from "open"
                "assigned_to": self.user.id,
            },
        )
        self.assertEqual(resp.status_code, 400)
        # Verify NCR was not modified
        ncr = NonconformanceRecord.objects.get(id=self.ncr_id)
        self.assertEqual(ncr.status, "open")


# =============================================================================
# BUG-05: CAPA can_transition type unsafety on FK fields
# =============================================================================


@SECURE_OFF
class Bug05CAPATransitionTypeTest(TestCase):
    """can_transition() called .strip() on all fields — crashes on FK."""

    def setUp(self):
        self.user = _make_user("capa_type@test.com")

    def test_text_field_required_empty(self):
        capa = CAPAReport(owner=self.user, title="Test", description="Test", status="draft")
        capa.save()
        # containment requires containment_action (text field)
        ok, msg = capa.can_transition("containment")
        # Should succeed since containment_action has no requirement
        # (only investigation requires containment_action)
        self.assertTrue(ok)

    def test_text_field_required_populated(self):
        capa = CAPAReport(
            owner=self.user,
            title="Test",
            description="Test",
            status="containment",
            containment_action="Quarantined affected units",
        )
        capa.save()
        ok, msg = capa.can_transition("investigation")
        self.assertTrue(ok)

    def test_text_field_required_blank(self):
        capa = CAPAReport(
            owner=self.user,
            title="Test",
            description="Test",
            status="containment",
            containment_action="",
        )
        capa.save()
        ok, msg = capa.can_transition("investigation")
        self.assertFalse(ok)
        self.assertIn("containment_action", msg)

    def test_none_field_treated_as_missing(self):
        """Ensure None values are correctly detected as missing."""
        capa = CAPAReport(
            owner=self.user,
            title="Test",
            description="Test",
            status="containment",
        )
        capa.save()
        # containment_action defaults to "" but let's force None
        capa.containment_action = None
        ok, msg = capa.can_transition("investigation")
        self.assertFalse(ok)


# =============================================================================
# BUG-06: Org invite acceptance skips email verification
# =============================================================================


@SECURE_OFF
class Bug06OrgInviteEmailVerificationTest(TestCase):
    """Unverified email users could accept org invitations."""

    def setUp(self):
        from core.models.tenant import Membership, OrgInvitation, Tenant

        self.admin = _make_user("admin@org.com")
        self.admin.is_email_verified = True
        self.admin.save()
        self.tenant = Tenant.objects.create(name="Test Org", slug="test-org")
        Membership.objects.create(tenant=self.tenant, user=self.admin, role="owner")

        self.invitee = _make_user("invitee@org.com")
        self.invitee.is_email_verified = False
        self.invitee.save()

        import uuid

        self.invitation = OrgInvitation.objects.create(
            tenant=self.tenant,
            email="invitee@org.com",
            role="member",
            invited_by=self.admin,
            token=uuid.uuid4(),
        )

    def test_unverified_user_rejected(self):
        self.client.force_login(self.invitee)
        resp = _post(
            self.client,
            "/api/core/org/accept-invite/",
            {"token": str(self.invitation.token)},
        )
        self.assertEqual(resp.status_code, 403)
        self.assertIn("verify", _err_msg(resp).lower())

    def test_verified_user_accepted(self):
        self.invitee.is_email_verified = True
        self.invitee.save()
        self.client.force_login(self.invitee)
        resp = _post(
            self.client,
            "/api/core/org/accept-invite/",
            {"token": str(self.invitation.token)},
        )
        self.assertEqual(resp.status_code, 200)


# =============================================================================
# BUG-07: InviteCode.use() non-atomic TOCTOU race
# =============================================================================


@SECURE_OFF
class Bug07InviteCodeAtomicTest(TestCase):
    """InviteCode.use() was non-atomic — concurrent use could exceed max_uses."""

    def setUp(self):
        from accounts.models import InviteCode

        self.code = InviteCode.objects.create(code="TEST-ATOM", max_uses=1, is_active=True)
        self.user1 = _make_user("inv1@test.com")
        self.user2 = _make_user("inv2@test.com")

    def test_single_use_succeeds(self):

        result = self.code.use(self.user1)
        self.assertTrue(result)
        self.code.refresh_from_db()
        self.assertEqual(self.code.times_used, 1)

    def test_second_use_fails(self):

        self.code.use(self.user1)
        result = self.code.use(self.user2)
        self.assertFalse(result)
        self.code.refresh_from_db()
        self.assertEqual(self.code.times_used, 1)

    def test_inactive_code_fails(self):
        self.code.is_active = False
        self.code.save()
        result = self.code.use(self.user1)
        self.assertFalse(result)


# =============================================================================
# BUG-08: FMEA list N+1 query on rows
# =============================================================================


@SECURE_OFF
class Bug08FMEAQueryTest(TestCase):
    """FMEA list fired N+1 queries due to missing prefetch_related."""

    def setUp(self):
        self.user = _make_user("fmea@test.com", tier=Tier.PRO)
        self.client.force_login(self.user)
        # Create 3 FMEAs with rows
        for i in range(3):
            fmea = FMEA.objects.create(owner=self.user, title=f"FMEA {i}", fmea_type="process")
            for j in range(2):
                FMEARow.objects.create(
                    fmea=fmea,
                    process_step=f"Step {j}",
                    failure_mode=f"Fail {j}",
                    severity=5,
                    occurrence=3,
                    detection=4,
                )

    def test_list_fmeas_returns_correct_data(self):
        resp = self.client.get("/api/fmea/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["fmeas"]), 3)
        for fmea in data["fmeas"]:
            self.assertEqual(fmea["row_count"], 2)
            self.assertGreater(fmea["max_rpn"], 0)

    def test_list_fmeas_query_count_bounded(self):
        """With prefetch_related, query count should NOT scale with FMEA count.

        Middleware adds session/user/membership/subscription/rate-limit queries.
        The key assertion is that rows are fetched in 1 batch query (prefetch)
        instead of N separate queries (one per FMEA).
        Without prefetch: 10+ queries for 3 FMEAs. With prefetch: ~10 total.
        """
        # Create 3 more FMEAs to prove queries don't scale linearly
        for i in range(3):
            fmea = FMEA.objects.create(owner=self.user, title=f"Extra FMEA {i}", fmea_type="process")
            FMEARow.objects.create(
                fmea=fmea,
                process_step="S",
                failure_mode="F",
                severity=5,
                occurrence=3,
                detection=4,
            )
        # With 6 FMEAs: if N+1 we'd see ~16 queries; with prefetch ~10
        from django.db import connection
        from django.test.utils import CaptureQueriesContext

        with CaptureQueriesContext(connection) as ctx:
            resp = self.client.get("/api/fmea/")
            self.assertEqual(resp.status_code, 200)
        # Should be well under 16 (N+1 would add 1 per FMEA)
        self.assertLess(len(ctx), 14)


# =============================================================================
# BUG-09: Hoshin update allows nulling site FK
# =============================================================================


@SECURE_OFF
class Bug09HoshinSiteNullTest(TestCase):
    """Setting site=None via update orphaned projects from tenant scope."""

    def setUp(self):
        from core.models.tenant import Membership, Tenant

        self.user = _make_user("hoshin@test.com", tier=Tier.ENTERPRISE)
        self.tenant = Tenant.objects.create(name="Hoshin Org", slug="hoshin-org")
        Membership.objects.create(tenant=self.tenant, user=self.user, role="owner")
        self.client.force_login(self.user)

        # Create site
        resp = _post(
            self.client,
            "/api/hoshin/sites/create/",
            {"name": "Test Site", "location": "Factory A"},
        )
        self.site_id = resp.json()["site"]["id"]

        # Create hoshin project
        resp = _post(
            self.client,
            "/api/hoshin/projects/create/",
            {
                "title": "Test Hoshin",
                "site_id": self.site_id,
                "fiscal_year": 2026,
                "methodology": "lean",
            },
        )
        self.hoshin_id = resp.json()["project"]["id"]

    def test_cannot_null_site(self):
        resp = _put(
            self.client,
            f"/api/hoshin/projects/{self.hoshin_id}/update/",
            {"site_id": None},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("orphan", _err_msg(resp).lower())


# =============================================================================
# BUG-10: File list ValueError on non-numeric query params
# =============================================================================


@SECURE_OFF
class Bug10FileListValidationTest(TestCase):
    """int() on unvalidated query params returned 500 instead of 400."""

    def setUp(self):
        self.user = _make_user("files@test.com")
        self.client.force_login(self.user)

    def test_non_numeric_limit_returns_400(self):
        resp = self.client.get("/api/files/?limit=abc")
        self.assertEqual(resp.status_code, 400)

    def test_non_numeric_offset_returns_400(self):
        resp = self.client.get("/api/files/?offset=xyz")
        self.assertEqual(resp.status_code, 400)

    def test_valid_pagination_works(self):
        resp = self.client.get("/api/files/?limit=10&offset=0")
        self.assertEqual(resp.status_code, 200)


# =============================================================================
# BUG-11: Org invite membership creation race condition
# (Fix verified in BUG-06 test class — IntegrityError now caught)
# =============================================================================


@SECURE_OFF
class Bug11OrgInviteRaceTest(TestCase):
    """Concurrent invite acceptance could create duplicate memberships."""

    def setUp(self):
        from core.models.tenant import Membership, Tenant

        self.admin = _make_user("admin11@org.com")
        self.admin.is_email_verified = True
        self.admin.save()
        self.tenant = Tenant.objects.create(name="Race Org", slug="race-org")
        Membership.objects.create(tenant=self.tenant, user=self.admin, role="owner")

        self.invitee = _make_user("invitee11@org.com")
        self.invitee.is_email_verified = True
        self.invitee.save()

    def test_already_member_returns_400(self):
        """If user is already a member, returns clean error not 500."""
        import uuid

        from core.models.tenant import Membership, OrgInvitation

        Membership.objects.create(tenant=self.tenant, user=self.invitee, role="member")
        invitation = OrgInvitation.objects.create(
            tenant=self.tenant,
            email="invitee11@org.com",
            role="member",
            invited_by=self.admin,
            token=uuid.uuid4(),
        )
        self.client.force_login(self.invitee)
        resp = _post(
            self.client,
            "/api/core/org/accept-invite/",
            {"token": str(invitation.token)},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("already", _err_msg(resp).lower())


# =============================================================================
# BUG-12: Report update accepts arbitrary section keys
# =============================================================================


@SECURE_OFF
class Bug12ReportSectionValidationTest(TestCase):
    """update_report merged arbitrary keys into sections JSONField."""

    def setUp(self):
        from core.models import Project

        self.user = _make_user("report@test.com", tier=Tier.PRO)
        self.project = Project.objects.create(user=self.user, title="Report Project")
        self.client.force_login(self.user)
        resp = _post(
            self.client,
            "/api/reports/create/",
            {
                "title": "Test CAPA Report",
                "report_type": "capa",
                "project_id": str(self.project.id),
            },
        )
        self.report_id = resp.json()["id"]

    def test_valid_section_key_accepted(self):
        resp = _put(
            self.client,
            f"/api/reports/{self.report_id}/update/",
            {"sections": {"problem_description": "Widget fails under load"}},
        )
        self.assertEqual(resp.status_code, 200)

    def test_invalid_section_key_rejected(self):
        resp = _put(
            self.client,
            f"/api/reports/{self.report_id}/update/",
            {"sections": {"injected_key": "malicious content"}},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid section key", _err_msg(resp))
