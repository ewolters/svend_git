"""Comprehensive tests for organization management.

Covers: org creation, info, invitations, member management,
role changes, tier gating, and edge cases.
"""

import uuid
from datetime import timedelta

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.utils import timezone
from rest_framework.test import APIClient

# Production has SECURE_SSL_REDIRECT=True — disable in tests so the test
# client's plain-HTTP requests don't get 301'd to HTTPS.
SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)

from accounts.constants import Tier
from core.models import Membership, OrgInvitation, Tenant

User = get_user_model()


def _make_user(email, tier=Tier.FREE, **kwargs):
    """Create a user with a given tier."""
    username = email.split("@")[0]
    user = User.objects.create_user(username=username, email=email, password="testpass123", **kwargs)
    user.tier = tier
    user.save(update_fields=["tier"])
    return user


def _make_tenant(name="Test Org", slug="test-org", plan=Tenant.Plan.TEAM, **kwargs):
    """Create a tenant."""
    return Tenant.objects.create(name=name, slug=slug, plan=plan, **kwargs)


def _make_membership(tenant, user, role=Membership.Role.MEMBER, **kwargs):
    """Create an active membership."""
    return Membership.objects.create(tenant=tenant, user=user, role=role, joined_at=timezone.now(), **kwargs)


# =========================================================================
# Org Info
# =========================================================================


@SECURE_OFF
class OrgInfoTest(TestCase):
    """Tests for GET /api/core/org/."""

    def setUp(self):
        self.client = APIClient()

    def test_unauthenticated_returns_forbidden(self):
        res = self.client.get("/api/core/org/")
        self.assertIn(res.status_code, [401, 403])

    def test_free_user_no_org(self):
        user = _make_user("free@example.com", Tier.FREE)
        self.client.force_authenticate(user)
        res = self.client.get("/api/core/org/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertFalse(data["has_org"])
        self.assertFalse(data["can_create_org"])

    def test_pro_user_no_org(self):
        user = _make_user("pro@example.com", Tier.PRO)
        self.client.force_authenticate(user)
        res = self.client.get("/api/core/org/")
        data = res.json()
        self.assertFalse(data["has_org"])
        self.assertFalse(data["can_create_org"])

    def test_team_user_no_org_can_create(self):
        user = _make_user("team@example.com", Tier.TEAM)
        self.client.force_authenticate(user)
        res = self.client.get("/api/core/org/")
        data = res.json()
        self.assertFalse(data["has_org"])
        self.assertTrue(data["can_create_org"])

    def test_enterprise_user_no_org_can_create(self):
        user = _make_user("ent@example.com", Tier.ENTERPRISE)
        self.client.force_authenticate(user)
        res = self.client.get("/api/core/org/")
        data = res.json()
        self.assertFalse(data["has_org"])
        self.assertTrue(data["can_create_org"])

    def test_user_with_org_returns_org_info(self):
        user = _make_user("member@example.com", Tier.TEAM)
        tenant = _make_tenant()
        _make_membership(tenant, user, Membership.Role.OWNER)
        self.client.force_authenticate(user)

        res = self.client.get("/api/core/org/")
        data = res.json()
        self.assertTrue(data["has_org"])
        self.assertEqual(data["org"]["name"], "Test Org")
        self.assertEqual(data["org"]["slug"], "test-org")
        self.assertEqual(data["membership"]["role"], "owner")
        self.assertTrue(data["membership"]["can_admin"])

    def test_inactive_membership_treated_as_no_org(self):
        user = _make_user("inactive@example.com", Tier.TEAM)
        tenant = _make_tenant()
        m = _make_membership(tenant, user)
        m.is_active = False
        m.save()
        self.client.force_authenticate(user)

        res = self.client.get("/api/core/org/")
        data = res.json()
        self.assertFalse(data["has_org"])
        self.assertTrue(data["can_create_org"])


# =========================================================================
# Org Creation
# =========================================================================


@SECURE_OFF
class OrgCreateTest(TestCase):
    """Tests for POST /api/core/org/create/."""

    def setUp(self):
        self.client = APIClient()

    def test_unauthenticated_returns_forbidden(self):
        res = self.client.post("/api/core/org/create/", {"name": "X", "slug": "x1"})
        self.assertIn(res.status_code, [401, 403])

    def test_free_user_blocked(self):
        user = _make_user("free@example.com", Tier.FREE)
        self.client.force_authenticate(user)
        res = self.client.post(
            "/api/core/org/create/",
            {"name": "My Org", "slug": "my-org"},
            format="json",
        )
        self.assertEqual(res.status_code, 403)
        self.assertIn("Team plan required", res.json()["error"]["message"])

    def test_pro_user_blocked(self):
        user = _make_user("pro@example.com", Tier.PRO)
        self.client.force_authenticate(user)
        res = self.client.post(
            "/api/core/org/create/",
            {"name": "My Org", "slug": "my-org"},
            format="json",
        )
        self.assertEqual(res.status_code, 403)

    def test_founder_user_blocked(self):
        user = _make_user("founder@example.com", Tier.FOUNDER)
        self.client.force_authenticate(user)
        res = self.client.post(
            "/api/core/org/create/",
            {"name": "My Org", "slug": "my-org"},
            format="json",
        )
        self.assertEqual(res.status_code, 403)

    def test_team_user_can_create(self):
        user = _make_user("team@example.com", Tier.TEAM)
        self.client.force_authenticate(user)
        res = self.client.post(
            "/api/core/org/create/",
            {"name": "Team Org", "slug": "team-org"},
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["org"]["name"], "Team Org")
        self.assertEqual(data["org"]["slug"], "team-org")
        self.assertEqual(data["org"]["plan"], "team")

        # Verify tenant and membership created
        tenant = Tenant.objects.get(slug="team-org")
        self.assertEqual(tenant.name, "Team Org")
        membership = Membership.objects.get(tenant=tenant, user=user)
        self.assertEqual(membership.role, Membership.Role.OWNER)
        self.assertTrue(membership.is_active)
        self.assertIsNotNone(membership.joined_at)

    def test_enterprise_user_can_create(self):
        user = _make_user("ent@example.com", Tier.ENTERPRISE)
        self.client.force_authenticate(user)
        res = self.client.post(
            "/api/core/org/create/",
            {"name": "Ent Corp", "slug": "ent-corp"},
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["org"]["plan"], "enterprise")

    def test_duplicate_slug_rejected(self):
        _make_tenant(name="Existing", slug="taken-slug")
        user = _make_user("team@example.com", Tier.TEAM)
        self.client.force_authenticate(user)
        res = self.client.post(
            "/api/core/org/create/",
            {"name": "New Org", "slug": "taken-slug"},
            format="json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("slug is already taken", res.json()["error"]["message"])

    def test_already_in_org_blocked(self):
        user = _make_user("team@example.com", Tier.TEAM)
        tenant = _make_tenant()
        _make_membership(tenant, user, Membership.Role.MEMBER)
        self.client.force_authenticate(user)

        res = self.client.post(
            "/api/core/org/create/",
            {"name": "Another", "slug": "another"},
            format="json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("already belong", res.json()["error"]["message"])

    def test_missing_name_rejected(self):
        user = _make_user("team@example.com", Tier.TEAM)
        self.client.force_authenticate(user)
        res = self.client.post(
            "/api/core/org/create/",
            {"name": "", "slug": "my-org"},
            format="json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("name is required", res.json()["error"]["message"])

    def test_missing_slug_rejected(self):
        user = _make_user("team@example.com", Tier.TEAM)
        self.client.force_authenticate(user)
        res = self.client.post(
            "/api/core/org/create/",
            {"name": "My Org", "slug": ""},
            format="json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("slug is required", res.json()["error"]["message"])

    def test_invalid_slug_format_rejected(self):
        user = _make_user("team@example.com", Tier.TEAM)
        self.client.force_authenticate(user)

        # Note: view lowercases input, so "CAPS" becomes valid "caps".
        # Only test truly invalid patterns.
        for bad_slug in ["has spaces", "has@special", "-starts-with-dash"]:
            res = self.client.post(
                "/api/core/org/create/",
                {"name": "Org", "slug": bad_slug},
                format="json",
            )
            self.assertIn(res.status_code, [400], msg=f"Expected 400 for slug '{bad_slug}'")

    def test_slug_too_short_rejected(self):
        user = _make_user("team@example.com", Tier.TEAM)
        self.client.force_authenticate(user)
        res = self.client.post(
            "/api/core/org/create/",
            {"name": "Org", "slug": "x"},
            format="json",
        )
        self.assertEqual(res.status_code, 400)

    def test_name_too_long_rejected(self):
        user = _make_user("team@example.com", Tier.TEAM)
        self.client.force_authenticate(user)
        res = self.client.post(
            "/api/core/org/create/",
            {"name": "X" * 256, "slug": "valid-slug"},
            format="json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("255 characters", res.json()["error"]["message"])


# =========================================================================
# Org Members
# =========================================================================


@SECURE_OFF
class OrgMembersTest(TestCase):
    """Tests for GET /api/core/org/members/."""

    def setUp(self):
        self.client = APIClient()
        self.owner = _make_user("owner@example.com", Tier.TEAM)
        self.tenant = _make_tenant()
        _make_membership(self.tenant, self.owner, Membership.Role.OWNER)

    def test_unauthenticated_returns_forbidden(self):
        res = self.client.get("/api/core/org/members/")
        self.assertIn(res.status_code, [401, 403])

    def test_non_member_returns_403(self):
        outsider = _make_user("outsider@example.com", Tier.TEAM)
        self.client.force_authenticate(outsider)
        res = self.client.get("/api/core/org/members/")
        self.assertEqual(res.status_code, 403)

    def test_viewer_returns_403(self):
        viewer = _make_user("viewer@example.com", Tier.TEAM)
        _make_membership(self.tenant, viewer, Membership.Role.VIEWER)
        self.client.force_authenticate(viewer)
        res = self.client.get("/api/core/org/members/")
        self.assertEqual(res.status_code, 403)

    def test_member_role_returns_403(self):
        member = _make_user("member@example.com", Tier.TEAM)
        _make_membership(self.tenant, member, Membership.Role.MEMBER)
        self.client.force_authenticate(member)
        res = self.client.get("/api/core/org/members/")
        self.assertEqual(res.status_code, 403)

    def test_admin_can_list_members(self):
        admin = _make_user("admin@example.com", Tier.TEAM)
        _make_membership(self.tenant, admin, Membership.Role.ADMIN)
        self.client.force_authenticate(admin)

        res = self.client.get("/api/core/org/members/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["members"]), 2)  # owner + admin

    def test_owner_can_list_members(self):
        self.client.force_authenticate(self.owner)
        res = self.client.get("/api/core/org/members/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertGreaterEqual(len(data["members"]), 1)
        self.assertEqual(data["max_members"], self.tenant.max_members)

    def test_inactive_members_excluded(self):
        inactive = _make_user("gone@example.com", Tier.TEAM)
        m = _make_membership(self.tenant, inactive)
        m.is_active = False
        m.save()

        self.client.force_authenticate(self.owner)
        res = self.client.get("/api/core/org/members/")
        emails = [m["email"] for m in res.json()["members"]]
        self.assertNotIn("gone@example.com", emails)


# =========================================================================
# Org Invite
# =========================================================================


@SECURE_OFF
class OrgInviteTest(TestCase):
    """Tests for POST /api/core/org/invite/."""

    def setUp(self):
        self.client = APIClient()
        self.owner = _make_user("owner@example.com", Tier.TEAM)
        self.tenant = _make_tenant()
        _make_membership(self.tenant, self.owner, Membership.Role.OWNER)

    def test_unauthenticated_returns_forbidden(self):
        res = self.client.post("/api/core/org/invite/", {"email": "x@example.com"})
        self.assertIn(res.status_code, [401, 403])

    def test_non_admin_cannot_invite(self):
        member = _make_user("member@example.com", Tier.TEAM)
        _make_membership(self.tenant, member, Membership.Role.MEMBER)
        self.client.force_authenticate(member)
        res = self.client.post(
            "/api/core/org/invite/",
            {"email": "new@example.com"},
            format="json",
        )
        self.assertEqual(res.status_code, 403)

    def test_owner_can_invite(self):
        self.client.force_authenticate(self.owner)
        res = self.client.post(
            "/api/core/org/invite/",
            {"email": "new@example.com", "role": "member"},
            format="json",
        )
        # May fail with 402 (Stripe not configured in test) or 201
        # Check that the logic path is correct
        if res.status_code == 201:
            data = res.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["invitation"]["email"], "new@example.com")
            self.assertEqual(data["invitation"]["role"], "member")
            # Verify invitation created
            inv = OrgInvitation.objects.get(email="new@example.com")
            self.assertEqual(inv.status, OrgInvitation.Status.PENDING)
            self.assertEqual(inv.tenant, self.tenant)
        else:
            # Stripe not configured — 402 is expected
            self.assertEqual(res.status_code, 402)

    def test_admin_can_invite_as_member(self):
        admin = _make_user("admin@example.com", Tier.TEAM)
        _make_membership(self.tenant, admin, Membership.Role.ADMIN)
        self.client.force_authenticate(admin)
        res = self.client.post(
            "/api/core/org/invite/",
            {"email": "new@example.com", "role": "member"},
            format="json",
        )
        # Either 201 or 402 (Stripe)
        self.assertIn(res.status_code, [201, 402])

    def test_admin_cannot_invite_as_owner(self):
        admin = _make_user("admin@example.com", Tier.TEAM)
        _make_membership(self.tenant, admin, Membership.Role.ADMIN)
        self.client.force_authenticate(admin)
        res = self.client.post(
            "/api/core/org/invite/",
            {"email": "new@example.com", "role": "owner"},
            format="json",
        )
        self.assertEqual(res.status_code, 403)
        self.assertIn("Only owners", res.json()["error"]["message"])

    def test_admin_cannot_invite_as_admin(self):
        admin = _make_user("admin@example.com", Tier.TEAM)
        _make_membership(self.tenant, admin, Membership.Role.ADMIN)
        self.client.force_authenticate(admin)
        res = self.client.post(
            "/api/core/org/invite/",
            {"email": "new@example.com", "role": "admin"},
            format="json",
        )
        self.assertEqual(res.status_code, 403)

    def test_invite_existing_member_rejected(self):
        existing = _make_user("existing@example.com", Tier.TEAM)
        _make_membership(self.tenant, existing)
        self.client.force_authenticate(self.owner)

        res = self.client.post(
            "/api/core/org/invite/",
            {"email": "existing@example.com"},
            format="json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("already a member", res.json()["error"]["message"])

    def test_duplicate_pending_invite_rejected(self):
        self.client.force_authenticate(self.owner)
        OrgInvitation.objects.create(
            tenant=self.tenant,
            email="dup@example.com",
            invited_by=self.owner,
        )
        res = self.client.post(
            "/api/core/org/invite/",
            {"email": "dup@example.com"},
            format="json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("already pending", res.json()["error"]["message"])

    def test_empty_email_rejected(self):
        self.client.force_authenticate(self.owner)
        res = self.client.post(
            "/api/core/org/invite/",
            {"email": ""},
            format="json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("Email is required", res.json()["error"]["message"])

    def test_invalid_role_rejected(self):
        self.client.force_authenticate(self.owner)
        res = self.client.post(
            "/api/core/org/invite/",
            {"email": "x@example.com", "role": "superadmin"},
            format="json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("Invalid role", res.json()["error"]["message"])


# =========================================================================
# Accept Invitation
# =========================================================================


@SECURE_OFF
class AcceptInvitationTest(TestCase):
    """Tests for POST /api/core/org/accept-invite/."""

    def setUp(self):
        self.client = APIClient()
        self.owner = _make_user("owner@example.com", Tier.TEAM)
        self.tenant = _make_tenant()
        _make_membership(self.tenant, self.owner, Membership.Role.OWNER)
        self.invitation = OrgInvitation.objects.create(
            tenant=self.tenant,
            email="invitee@example.com",
            role=Membership.Role.MEMBER,
            invited_by=self.owner,
        )

    def test_unauthenticated_returns_forbidden(self):
        res = self.client.post("/api/core/org/accept-invite/", {"token": str(self.invitation.token)})
        self.assertIn(res.status_code, [401, 403])

    def test_accept_valid_invitation(self):
        invitee = _make_user("invitee@example.com", Tier.TEAM)
        self.client.force_authenticate(invitee)
        res = self.client.post(
            "/api/core/org/accept-invite/",
            {"token": str(self.invitation.token)},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["role"], "member")

        # Verify membership created
        m = Membership.objects.get(tenant=self.tenant, user=invitee)
        self.assertEqual(m.role, Membership.Role.MEMBER)
        self.assertTrue(m.is_active)

        # Verify invitation marked accepted
        self.invitation.refresh_from_db()
        self.assertEqual(self.invitation.status, OrgInvitation.Status.ACCEPTED)

    def test_wrong_email_rejected(self):
        wrong_user = _make_user("wrong@example.com", Tier.TEAM)
        self.client.force_authenticate(wrong_user)
        res = self.client.post(
            "/api/core/org/accept-invite/",
            {"token": str(self.invitation.token)},
            format="json",
        )
        self.assertEqual(res.status_code, 403)
        self.assertIn("different email", res.json()["error"]["message"])

    def test_expired_invitation_rejected(self):
        self.invitation.expires_at = timezone.now() - timedelta(days=1)
        self.invitation.save()

        invitee = _make_user("invitee@example.com", Tier.TEAM)
        self.client.force_authenticate(invitee)
        res = self.client.post(
            "/api/core/org/accept-invite/",
            {"token": str(self.invitation.token)},
            format="json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("expired", res.json()["error"]["message"])

    def test_cancelled_invitation_rejected(self):
        self.invitation.status = OrgInvitation.Status.CANCELLED
        self.invitation.save()

        invitee = _make_user("invitee@example.com", Tier.TEAM)
        self.client.force_authenticate(invitee)
        res = self.client.post(
            "/api/core/org/accept-invite/",
            {"token": str(self.invitation.token)},
            format="json",
        )
        self.assertEqual(res.status_code, 400)

    def test_invalid_token_returns_404(self):
        invitee = _make_user("invitee@example.com", Tier.TEAM)
        self.client.force_authenticate(invitee)
        res = self.client.post(
            "/api/core/org/accept-invite/",
            {"token": str(uuid.uuid4())},
            format="json",
        )
        self.assertEqual(res.status_code, 404)

    def test_missing_token_returns_400(self):
        invitee = _make_user("invitee@example.com", Tier.TEAM)
        self.client.force_authenticate(invitee)
        res = self.client.post(
            "/api/core/org/accept-invite/",
            {},
            format="json",
        )
        self.assertEqual(res.status_code, 400)

    def test_already_member_returns_400(self):
        invitee = _make_user("invitee@example.com", Tier.TEAM)
        _make_membership(self.tenant, invitee)
        self.client.force_authenticate(invitee)
        res = self.client.post(
            "/api/core/org/accept-invite/",
            {"token": str(self.invitation.token)},
            format="json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("already a member", res.json()["error"]["message"])

    def test_accept_preserves_invited_role(self):
        """Invitation with admin role should create admin membership."""
        admin_invite = OrgInvitation.objects.create(
            tenant=self.tenant,
            email="newadmin@example.com",
            role=Membership.Role.ADMIN,
            invited_by=self.owner,
        )
        invitee = _make_user("newadmin@example.com", Tier.TEAM)
        self.client.force_authenticate(invitee)
        res = self.client.post(
            "/api/core/org/accept-invite/",
            {"token": str(admin_invite.token)},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        m = Membership.objects.get(tenant=self.tenant, user=invitee)
        self.assertEqual(m.role, Membership.Role.ADMIN)


# =========================================================================
# Role Changes
# =========================================================================


@SECURE_OFF
class OrgChangeRoleTest(TestCase):
    """Tests for PUT /api/core/org/members/<id>/role/."""

    def setUp(self):
        self.client = APIClient()
        self.owner = _make_user("owner@example.com", Tier.TEAM)
        self.tenant = _make_tenant()
        self.owner_membership = _make_membership(
            self.tenant,
            self.owner,
            Membership.Role.OWNER,
        )
        self.member_user = _make_user("member@example.com", Tier.TEAM)
        self.member_membership = _make_membership(
            self.tenant,
            self.member_user,
            Membership.Role.MEMBER,
        )

    def test_owner_can_promote_to_admin(self):
        self.client.force_authenticate(self.owner)
        res = self.client.put(
            f"/api/core/org/members/{self.member_membership.id}/role/",
            {"role": "admin"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.member_membership.refresh_from_db()
        self.assertEqual(self.member_membership.role, Membership.Role.ADMIN)

    def test_owner_can_promote_to_owner(self):
        self.client.force_authenticate(self.owner)
        res = self.client.put(
            f"/api/core/org/members/{self.member_membership.id}/role/",
            {"role": "owner"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.member_membership.refresh_from_db()
        self.assertEqual(self.member_membership.role, Membership.Role.OWNER)

    def test_admin_cannot_promote_to_owner(self):
        admin = _make_user("admin@example.com", Tier.TEAM)
        _make_membership(self.tenant, admin, Membership.Role.ADMIN)
        self.client.force_authenticate(admin)

        res = self.client.put(
            f"/api/core/org/members/{self.member_membership.id}/role/",
            {"role": "owner"},
            format="json",
        )
        self.assertEqual(res.status_code, 403)
        self.assertIn("Only owners", res.json()["error"]["message"])

    def test_admin_cannot_promote_to_admin(self):
        admin = _make_user("admin@example.com", Tier.TEAM)
        _make_membership(self.tenant, admin, Membership.Role.ADMIN)
        self.client.force_authenticate(admin)

        res = self.client.put(
            f"/api/core/org/members/{self.member_membership.id}/role/",
            {"role": "admin"},
            format="json",
        )
        self.assertEqual(res.status_code, 403)

    def test_admin_can_demote_to_viewer(self):
        admin = _make_user("admin@example.com", Tier.TEAM)
        _make_membership(self.tenant, admin, Membership.Role.ADMIN)
        self.client.force_authenticate(admin)

        res = self.client.put(
            f"/api/core/org/members/{self.member_membership.id}/role/",
            {"role": "viewer"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.member_membership.refresh_from_db()
        self.assertEqual(self.member_membership.role, Membership.Role.VIEWER)

    def test_last_owner_cannot_demote_self(self):
        self.client.force_authenticate(self.owner)
        res = self.client.put(
            f"/api/core/org/members/{self.owner_membership.id}/role/",
            {"role": "admin"},
            format="json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("only owner", res.json()["error"]["message"])

    def test_owner_can_demote_self_if_another_owner_exists(self):
        second_owner = _make_user("owner2@example.com", Tier.TEAM)
        _make_membership(self.tenant, second_owner, Membership.Role.OWNER)

        self.client.force_authenticate(self.owner)
        res = self.client.put(
            f"/api/core/org/members/{self.owner_membership.id}/role/",
            {"role": "admin"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.owner_membership.refresh_from_db()
        self.assertEqual(self.owner_membership.role, Membership.Role.ADMIN)

    def test_invalid_role_rejected(self):
        self.client.force_authenticate(self.owner)
        res = self.client.put(
            f"/api/core/org/members/{self.member_membership.id}/role/",
            {"role": "superadmin"},
            format="json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("Invalid role", res.json()["error"]["message"])

    def test_nonexistent_member_returns_404(self):
        self.client.force_authenticate(self.owner)
        fake_id = uuid.uuid4()
        res = self.client.put(
            f"/api/core/org/members/{fake_id}/role/",
            {"role": "admin"},
            format="json",
        )
        self.assertEqual(res.status_code, 404)


# =========================================================================
# Remove Member
# =========================================================================


@SECURE_OFF
class OrgRemoveMemberTest(TestCase):
    """Tests for DELETE /api/core/org/members/<id>/remove/."""

    def setUp(self):
        self.client = APIClient()
        self.owner = _make_user("owner@example.com", Tier.TEAM)
        self.tenant = _make_tenant()
        self.owner_membership = _make_membership(
            self.tenant,
            self.owner,
            Membership.Role.OWNER,
        )

    def test_owner_can_remove_member(self):
        member = _make_user("member@example.com", Tier.TEAM)
        m = _make_membership(self.tenant, member)
        self.client.force_authenticate(self.owner)

        res = self.client.delete(f"/api/core/org/members/{m.id}/remove/")
        self.assertEqual(res.status_code, 200)
        m.refresh_from_db()
        self.assertFalse(m.is_active)

    def test_cannot_remove_self(self):
        self.client.force_authenticate(self.owner)
        res = self.client.delete(f"/api/core/org/members/{self.owner_membership.id}/remove/")
        self.assertEqual(res.status_code, 400)
        self.assertIn("Cannot remove yourself", res.json()["error"]["message"])

    def test_admin_cannot_remove_owner(self):
        admin = _make_user("admin@example.com", Tier.TEAM)
        _make_membership(self.tenant, admin, Membership.Role.ADMIN)
        self.client.force_authenticate(admin)

        res = self.client.delete(f"/api/core/org/members/{self.owner_membership.id}/remove/")
        self.assertEqual(res.status_code, 403)
        self.assertIn("Only owners", res.json()["error"]["message"])

    def test_owner_can_remove_another_owner(self):
        owner2 = _make_user("owner2@example.com", Tier.TEAM)
        m2 = _make_membership(self.tenant, owner2, Membership.Role.OWNER)
        self.client.force_authenticate(self.owner)

        res = self.client.delete(f"/api/core/org/members/{m2.id}/remove/")
        self.assertEqual(res.status_code, 200)
        m2.refresh_from_db()
        self.assertFalse(m2.is_active)

    def test_nonexistent_member_returns_404(self):
        self.client.force_authenticate(self.owner)
        res = self.client.delete(f"/api/core/org/members/{uuid.uuid4()}/remove/")
        self.assertEqual(res.status_code, 404)

    def test_non_admin_cannot_remove(self):
        member = _make_user("member@example.com", Tier.TEAM)
        _make_membership(self.tenant, member, Membership.Role.MEMBER)
        target = _make_user("target@example.com", Tier.TEAM)
        target_m = _make_membership(self.tenant, target, Membership.Role.VIEWER)

        self.client.force_authenticate(member)
        res = self.client.delete(f"/api/core/org/members/{target_m.id}/remove/")
        self.assertEqual(res.status_code, 403)


# =========================================================================
# Cancel Invitation
# =========================================================================


@SECURE_OFF
class OrgCancelInvitationTest(TestCase):
    """Tests for POST /api/core/org/invitations/<id>/cancel/."""

    def setUp(self):
        self.client = APIClient()
        self.owner = _make_user("owner@example.com", Tier.TEAM)
        self.tenant = _make_tenant()
        _make_membership(self.tenant, self.owner, Membership.Role.OWNER)
        self.invitation = OrgInvitation.objects.create(
            tenant=self.tenant,
            email="pending@example.com",
            invited_by=self.owner,
        )

    def test_owner_can_cancel(self):
        self.client.force_authenticate(self.owner)
        res = self.client.post(f"/api/core/org/invitations/{self.invitation.id}/cancel/")
        self.assertEqual(res.status_code, 200)
        self.invitation.refresh_from_db()
        self.assertEqual(self.invitation.status, OrgInvitation.Status.CANCELLED)

    def test_non_admin_cannot_cancel(self):
        member = _make_user("member@example.com", Tier.TEAM)
        _make_membership(self.tenant, member, Membership.Role.MEMBER)
        self.client.force_authenticate(member)

        res = self.client.post(f"/api/core/org/invitations/{self.invitation.id}/cancel/")
        self.assertEqual(res.status_code, 403)

    def test_cancel_nonexistent_returns_404(self):
        self.client.force_authenticate(self.owner)
        res = self.client.post(f"/api/core/org/invitations/{uuid.uuid4()}/cancel/")
        self.assertEqual(res.status_code, 404)

    def test_cancel_already_accepted_returns_404(self):
        self.invitation.status = OrgInvitation.Status.ACCEPTED
        self.invitation.save()
        self.client.force_authenticate(self.owner)

        res = self.client.post(f"/api/core/org/invitations/{self.invitation.id}/cancel/")
        self.assertEqual(res.status_code, 404)


# =========================================================================
# List Invitations
# =========================================================================


@SECURE_OFF
class OrgInvitationsListTest(TestCase):
    """Tests for GET /api/core/org/invitations/."""

    def setUp(self):
        self.client = APIClient()
        self.owner = _make_user("owner@example.com", Tier.TEAM)
        self.tenant = _make_tenant()
        _make_membership(self.tenant, self.owner, Membership.Role.OWNER)

    def test_owner_can_list(self):
        OrgInvitation.objects.create(
            tenant=self.tenant,
            email="a@example.com",
            invited_by=self.owner,
        )
        OrgInvitation.objects.create(
            tenant=self.tenant,
            email="b@example.com",
            invited_by=self.owner,
        )
        self.client.force_authenticate(self.owner)
        res = self.client.get("/api/core/org/invitations/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()["invitations"]), 2)

    def test_non_admin_cannot_list(self):
        member = _make_user("member@example.com", Tier.TEAM)
        _make_membership(self.tenant, member, Membership.Role.MEMBER)
        self.client.force_authenticate(member)
        res = self.client.get("/api/core/org/invitations/")
        self.assertEqual(res.status_code, 403)


# =========================================================================
# Full Lifecycle Integration Test
# =========================================================================


@SECURE_OFF
class OrgLifecycleTest(TestCase):
    """End-to-end test: create org -> invite -> accept -> manage -> remove."""

    def setUp(self):
        self.client = APIClient()

    def test_full_lifecycle(self):
        # 1. Team user creates org
        owner = _make_user("owner@example.com", Tier.TEAM)
        self.client.force_authenticate(owner)

        res = self.client.get("/api/core/org/")
        self.assertFalse(res.json()["has_org"])
        self.assertTrue(res.json()["can_create_org"])

        res = self.client.post(
            "/api/core/org/create/",
            {"name": "Lifecycle Corp", "slug": "lifecycle-corp"},
            format="json",
        )
        self.assertEqual(res.status_code, 201)

        # 2. Verify org shows in info
        res = self.client.get("/api/core/org/")
        data = res.json()
        self.assertTrue(data["has_org"])
        self.assertEqual(data["org"]["name"], "Lifecycle Corp")
        self.assertTrue(data["membership"]["can_admin"])

        # 3. Owner can list members (just themselves)
        res = self.client.get("/api/core/org/members/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()["members"]), 1)
        self.assertTrue(res.json()["members"][0]["is_current_user"])

        # 4. Create invitation manually (bypassing Stripe)
        tenant = Tenant.objects.get(slug="lifecycle-corp")
        invitation = OrgInvitation.objects.create(
            tenant=tenant,
            email="colleague@example.com",
            role=Membership.Role.MEMBER,
            invited_by=owner,
        )

        # 5. Colleague accepts
        colleague = _make_user("colleague@example.com", Tier.TEAM)
        self.client.force_authenticate(colleague)
        res = self.client.post(
            "/api/core/org/accept-invite/",
            {"token": str(invitation.token)},
            format="json",
        )
        self.assertEqual(res.status_code, 200)

        # 6. Colleague sees org info
        res = self.client.get("/api/core/org/")
        self.assertTrue(res.json()["has_org"])
        self.assertEqual(res.json()["membership"]["role"], "member")
        self.assertFalse(res.json()["membership"]["can_admin"])

        # 7. Owner promotes colleague to admin
        self.client.force_authenticate(owner)
        colleague_membership = Membership.objects.get(
            tenant=tenant,
            user=colleague,
        )
        res = self.client.put(
            f"/api/core/org/members/{colleague_membership.id}/role/",
            {"role": "admin"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)

        # 8. Colleague now has admin access
        self.client.force_authenticate(colleague)
        res = self.client.get("/api/core/org/")
        self.assertTrue(res.json()["membership"]["can_admin"])

        # 9. Colleague (admin) can list members
        res = self.client.get("/api/core/org/members/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()["members"]), 2)

        # 10. Owner removes colleague
        self.client.force_authenticate(owner)
        res = self.client.delete(f"/api/core/org/members/{colleague_membership.id}/remove/")
        self.assertEqual(res.status_code, 200)

        # 11. Colleague no longer in org
        self.client.force_authenticate(colleague)
        res = self.client.get("/api/core/org/")
        self.assertFalse(res.json()["has_org"])


@SECURE_OFF
class TierGatingIntegrationTest(TestCase):
    """Verify tier gating works across all org endpoints."""

    def setUp(self):
        self.client = APIClient()

    def test_each_tier_create_access(self):
        """Only team and enterprise can create orgs."""
        tiers_blocked = [Tier.FREE, Tier.FOUNDER, Tier.PRO]
        tiers_allowed = [Tier.TEAM, Tier.ENTERPRISE]

        for tier in tiers_blocked:
            user = _make_user(f"{tier}@example.com", tier)
            self.client.force_authenticate(user)
            res = self.client.post(
                "/api/core/org/create/",
                {"name": f"{tier} Org", "slug": f"{tier}-org"},
                format="json",
            )
            self.assertEqual(
                res.status_code,
                403,
                msg=f"Tier {tier} should be blocked from creating org",
            )

        for tier in tiers_allowed:
            user = _make_user(f"{tier}@example.com", tier)
            self.client.force_authenticate(user)
            res = self.client.post(
                "/api/core/org/create/",
                {"name": f"{tier} Org", "slug": f"{tier}-org"},
                format="json",
            )
            self.assertEqual(
                res.status_code,
                201,
                msg=f"Tier {tier} should be allowed to create org",
            )

    def test_can_create_org_flag_per_tier(self):
        """org_info returns correct can_create_org for each tier."""
        expected = {
            Tier.FREE: False,
            Tier.FOUNDER: False,
            Tier.PRO: False,
            Tier.TEAM: True,
            Tier.ENTERPRISE: True,
        }
        for tier, should_allow in expected.items():
            user = _make_user(f"{tier}-info@example.com", tier)
            self.client.force_authenticate(user)
            res = self.client.get("/api/core/org/")
            self.assertEqual(
                res.json()["can_create_org"],
                should_allow,
                msg=f"Tier {tier}: can_create_org should be {should_allow}",
            )
