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

from accounts.constants import Tier
from core.models import Membership, OrgInvitation, Tenant

# Production has SECURE_SSL_REDIRECT=True — disable in tests so the test
# client's plain-HTTP requests don't get 301'd to HTTPS.
SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)

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
        invitee.is_email_verified = True
        invitee.save(update_fields=["is_email_verified"])
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
        wrong_user.is_email_verified = True
        wrong_user.save(update_fields=["is_email_verified"])
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
        invitee.is_email_verified = True
        invitee.save(update_fields=["is_email_verified"])
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
        invitee.is_email_verified = True
        invitee.save(update_fields=["is_email_verified"])
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
        colleague.is_email_verified = True
        colleague.save(update_fields=["is_email_verified"])
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


# =========================================================================
# Site Management (ORG-001 §8.2)
# =========================================================================


@SECURE_OFF
class OrgSitesListTest(TestCase):
    """Tests for GET /api/core/org/sites/."""

    def setUp(self):
        self.client = APIClient()
        self.tenant = _make_tenant(plan=Tenant.Plan.TEAM)
        self.admin = _make_user("site-admin@example.com", Tier.TEAM)
        _make_membership(self.tenant, self.admin, Membership.Role.ADMIN)
        self.member = _make_user("site-member@example.com", Tier.TEAM)
        _make_membership(self.tenant, self.member, Membership.Role.MEMBER)

    def test_unauthenticated_returns_forbidden(self):
        res = self.client.get("/api/core/org/sites/")
        self.assertIn(res.status_code, [401, 403])

    def test_member_can_list_sites(self):
        """Any org member can list sites (read-only)."""
        from agents_api.models import Site

        Site.objects.create(tenant=self.tenant, name="Plant A", code="PLT-A")
        Site.objects.create(tenant=self.tenant, name="Plant B", code="PLT-B")

        self.client.force_authenticate(self.member)
        res = self.client.get("/api/core/org/sites/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["sites"]), 2)
        self.assertFalse(data["can_manage"])  # member can't manage
        self.assertEqual(data["max_sites"], 1)  # team plan

    def test_admin_can_manage_flag(self):
        self.client.force_authenticate(self.admin)
        res = self.client.get("/api/core/org/sites/")
        self.assertTrue(res.json()["can_manage"])

    def test_enterprise_unlimited_sites(self):
        ent_tenant = _make_tenant(name="Ent Org", slug="ent-org", plan=Tenant.Plan.ENTERPRISE)
        user = _make_user("ent-sites@example.com", Tier.ENTERPRISE)
        _make_membership(ent_tenant, user, Membership.Role.OWNER)
        self.client.force_authenticate(user)
        res = self.client.get("/api/core/org/sites/")
        self.assertIsNone(res.json()["max_sites"])

    def test_no_org_returns_error(self):
        user = _make_user("lonely@example.com", Tier.TEAM)
        self.client.force_authenticate(user)
        res = self.client.get("/api/core/org/sites/")
        self.assertEqual(res.status_code, 400)


@SECURE_OFF
class OrgCreateSiteTest(TestCase):
    """Tests for POST /api/core/org/sites/create/."""

    def setUp(self):
        self.client = APIClient()
        self.tenant = _make_tenant(plan=Tenant.Plan.TEAM)
        self.admin = _make_user("create-site-admin@example.com", Tier.TEAM)
        _make_membership(self.tenant, self.admin, Membership.Role.ADMIN)
        self.member = _make_user("create-site-member@example.com", Tier.TEAM)
        _make_membership(self.tenant, self.member, Membership.Role.MEMBER)

    def test_admin_creates_site(self):
        self.client.force_authenticate(self.admin)
        res = self.client.post(
            "/api/core/org/sites/create/",
            {"name": "New Plant", "code": "NP-01", "business_unit": "Operations"},
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        site = res.json()["site"]
        self.assertEqual(site["name"], "New Plant")
        self.assertEqual(site["code"], "NP-01")
        self.assertEqual(site["business_unit"], "Operations")

    def test_member_cannot_create_site(self):
        self.client.force_authenticate(self.member)
        res = self.client.post(
            "/api/core/org/sites/create/",
            {"name": "Blocked Plant"},
            format="json",
        )
        self.assertEqual(res.status_code, 403)

    def test_name_required(self):
        self.client.force_authenticate(self.admin)
        res = self.client.post("/api/core/org/sites/create/", {"code": "X"}, format="json")
        self.assertEqual(res.status_code, 400)

    def test_team_one_site_limit(self):
        """Team plan can only have 1 site."""
        from agents_api.models import Site

        Site.objects.create(tenant=self.tenant, name="Existing Plant")
        self.client.force_authenticate(self.admin)
        res = self.client.post(
            "/api/core/org/sites/create/",
            {"name": "Second Plant"},
            format="json",
        )
        self.assertEqual(res.status_code, 403)

    def test_enterprise_multiple_sites(self):
        """Enterprise plan has no site limit."""
        from agents_api.models import Site

        ent_tenant = _make_tenant(name="Enterprise Org", slug="enterprise-org", plan=Tenant.Plan.ENTERPRISE)
        user = _make_user("ent-create@example.com", Tier.ENTERPRISE)
        _make_membership(ent_tenant, user, Membership.Role.ADMIN)

        Site.objects.create(tenant=ent_tenant, name="Plant Alpha")
        self.client.force_authenticate(user)
        res = self.client.post(
            "/api/core/org/sites/create/",
            {"name": "Plant Beta"},
            format="json",
        )
        self.assertEqual(res.status_code, 201)


@SECURE_OFF
class OrgUpdateSiteTest(TestCase):
    """Tests for PUT/PATCH /api/core/org/sites/<uuid>/."""

    def setUp(self):
        from agents_api.models import Site

        self.client = APIClient()
        self.tenant = _make_tenant(plan=Tenant.Plan.TEAM)
        self.admin = _make_user("update-site-admin@example.com", Tier.TEAM)
        _make_membership(self.tenant, self.admin, Membership.Role.ADMIN)
        self.site = Site.objects.create(tenant=self.tenant, name="Old Name", code="OLD")

    def test_admin_updates_site(self):
        self.client.force_authenticate(self.admin)
        res = self.client.put(
            f"/api/core/org/sites/{self.site.id}/",
            {"name": "New Name", "code": "NEW"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["site"]["name"], "New Name")
        self.assertEqual(res.json()["site"]["code"], "NEW")

    def test_partial_update(self):
        self.client.force_authenticate(self.admin)
        res = self.client.patch(
            f"/api/core/org/sites/{self.site.id}/",
            {"plant_manager": "John Doe"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["site"]["plant_manager"], "John Doe")
        self.assertEqual(res.json()["site"]["name"], "Old Name")  # unchanged

    def test_cannot_update_other_tenant_site(self):
        other_tenant = _make_tenant(name="Other", slug="other", plan=Tenant.Plan.TEAM)
        from agents_api.models import Site

        other_site = Site.objects.create(tenant=other_tenant, name="Other Site")
        self.client.force_authenticate(self.admin)
        res = self.client.put(
            f"/api/core/org/sites/{other_site.id}/",
            {"name": "Hacked"},
            format="json",
        )
        self.assertEqual(res.status_code, 404)

    def test_member_cannot_update(self):
        member = _make_user("update-site-member@example.com", Tier.TEAM)
        _make_membership(self.tenant, member, Membership.Role.MEMBER)
        self.client.force_authenticate(member)
        res = self.client.put(
            f"/api/core/org/sites/{self.site.id}/",
            {"name": "Nope"},
            format="json",
        )
        self.assertEqual(res.status_code, 403)


@SECURE_OFF
class OrgDeleteSiteTest(TestCase):
    """Tests for DELETE /api/core/org/sites/<uuid>/delete/."""

    def setUp(self):
        from agents_api.models import Site

        self.client = APIClient()
        self.tenant = _make_tenant(plan=Tenant.Plan.TEAM)
        self.admin = _make_user("delete-site-admin@example.com", Tier.TEAM)
        _make_membership(self.tenant, self.admin, Membership.Role.ADMIN)
        self.site = Site.objects.create(tenant=self.tenant, name="Doomed Site")

    def test_hard_delete_no_projects(self):
        self.client.force_authenticate(self.admin)
        res = self.client.delete(f"/api/core/org/sites/{self.site.id}/delete/")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["deleted"])

        from agents_api.models import Site

        self.assertFalse(Site.objects.filter(id=self.site.id).exists())

    def test_soft_delete_with_projects(self):
        """Sites with hoshin projects are deactivated, not deleted."""
        from agents_api.models import HoshinProject
        from core.models import Project

        core_proj = Project.objects.create(title="CI Project", tenant=self.tenant)
        HoshinProject.objects.create(project=core_proj, site=self.site)
        self.client.force_authenticate(self.admin)
        res = self.client.delete(f"/api/core/org/sites/{self.site.id}/delete/")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["deactivated"])
        self.assertEqual(res.json()["project_count"], 1)

        self.site.refresh_from_db()
        self.assertFalse(self.site.is_active)

    def test_delete_nonexistent_site(self):
        self.client.force_authenticate(self.admin)
        fake_id = uuid.uuid4()
        res = self.client.delete(f"/api/core/org/sites/{fake_id}/delete/")
        self.assertEqual(res.status_code, 404)


# =========================================================================
# Employee Management (ORG-001 §7)
# =========================================================================


@SECURE_OFF
class OrgEmployeesListTest(TestCase):
    """Tests for GET /api/core/org/employees/."""

    def setUp(self):
        from agents_api.models import Employee, Site

        self.client = APIClient()
        self.tenant = _make_tenant(plan=Tenant.Plan.TEAM)
        self.admin = _make_user("emp-admin@example.com", Tier.TEAM)
        _make_membership(self.tenant, self.admin, Membership.Role.ADMIN)
        self.site = Site.objects.create(tenant=self.tenant, name="Main Plant")
        Employee.objects.create(tenant=self.tenant, name="Alice", email="alice@plant.com", site=self.site)
        Employee.objects.create(tenant=self.tenant, name="Bob", email="bob@plant.com", is_active=False)

    def test_list_all_employees(self):
        self.client.force_authenticate(self.admin)
        res = self.client.get("/api/core/org/employees/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["count"], 2)

    def test_filter_by_site(self):
        self.client.force_authenticate(self.admin)
        res = self.client.get(f"/api/core/org/employees/?site={self.site.id}")
        self.assertEqual(res.json()["count"], 1)
        self.assertEqual(res.json()["employees"][0]["name"], "Alice")

    def test_filter_active_only(self):
        self.client.force_authenticate(self.admin)
        res = self.client.get("/api/core/org/employees/?active_only=true")
        self.assertEqual(res.json()["count"], 1)
        self.assertEqual(res.json()["employees"][0]["name"], "Alice")

    def test_member_can_list(self):
        member = _make_user("emp-member@example.com", Tier.TEAM)
        _make_membership(self.tenant, member, Membership.Role.MEMBER)
        self.client.force_authenticate(member)
        res = self.client.get("/api/core/org/employees/")
        self.assertEqual(res.status_code, 200)
        self.assertFalse(res.json()["can_manage"])


@SECURE_OFF
class OrgCreateEmployeeTest(TestCase):
    """Tests for POST /api/core/org/employees/create/."""

    def setUp(self):
        self.client = APIClient()
        self.tenant = _make_tenant(plan=Tenant.Plan.TEAM)
        self.admin = _make_user("create-emp-admin@example.com", Tier.TEAM)
        _make_membership(self.tenant, self.admin, Membership.Role.ADMIN)

    def test_create_employee(self):
        self.client.force_authenticate(self.admin)
        res = self.client.post(
            "/api/core/org/employees/create/",
            {
                "name": "Charlie",
                "email": "charlie@plant.com",
                "role": "Operator",
                "department": "Assembly",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        emp = res.json()["employee"]
        self.assertEqual(emp["name"], "Charlie")
        self.assertEqual(emp["role"], "Operator")

    def test_create_employee_with_site(self):
        from agents_api.models import Site

        site = Site.objects.create(tenant=self.tenant, name="Plant X")
        self.client.force_authenticate(self.admin)
        res = self.client.post(
            "/api/core/org/employees/create/",
            {"name": "Diana", "email": "diana@plant.com", "site_id": str(site.id)},
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        self.assertEqual(res.json()["employee"]["site_id"], str(site.id))

    def test_name_required(self):
        self.client.force_authenticate(self.admin)
        res = self.client.post(
            "/api/core/org/employees/create/",
            {"email": "no-name@plant.com"},
            format="json",
        )
        self.assertEqual(res.status_code, 400)

    def test_email_required(self):
        self.client.force_authenticate(self.admin)
        res = self.client.post(
            "/api/core/org/employees/create/",
            {"name": "No Email"},
            format="json",
        )
        self.assertEqual(res.status_code, 400)

    def test_duplicate_email_rejected(self):
        from agents_api.models import Employee

        Employee.objects.create(tenant=self.tenant, name="Existing", email="dupe@plant.com")
        self.client.force_authenticate(self.admin)
        res = self.client.post(
            "/api/core/org/employees/create/",
            {"name": "New Person", "email": "dupe@plant.com"},
            format="json",
        )
        self.assertEqual(res.status_code, 409)

    def test_member_cannot_create(self):
        member = _make_user("emp-blocked@example.com", Tier.TEAM)
        _make_membership(self.tenant, member, Membership.Role.MEMBER)
        self.client.force_authenticate(member)
        res = self.client.post(
            "/api/core/org/employees/create/",
            {"name": "Blocked", "email": "blocked@plant.com"},
            format="json",
        )
        self.assertEqual(res.status_code, 403)

    def test_invalid_site_rejected(self):
        self.client.force_authenticate(self.admin)
        res = self.client.post(
            "/api/core/org/employees/create/",
            {"name": "Eve", "email": "eve@plant.com", "site_id": str(uuid.uuid4())},
            format="json",
        )
        self.assertEqual(res.status_code, 404)


@SECURE_OFF
class OrgUpdateEmployeeTest(TestCase):
    """Tests for PUT/PATCH /api/core/org/employees/<uuid>/."""

    def setUp(self):
        from agents_api.models import Employee

        self.client = APIClient()
        self.tenant = _make_tenant(plan=Tenant.Plan.TEAM)
        self.admin = _make_user("update-emp-admin@example.com", Tier.TEAM)
        _make_membership(self.tenant, self.admin, Membership.Role.ADMIN)
        self.emp = Employee.objects.create(
            tenant=self.tenant,
            name="Old Name",
            email="update-me@plant.com",
            role="Operator",
        )

    def test_update_employee(self):
        self.client.force_authenticate(self.admin)
        res = self.client.put(
            f"/api/core/org/employees/{self.emp.id}/",
            {"name": "New Name", "role": "Supervisor"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["employee"]["name"], "New Name")
        self.assertEqual(res.json()["employee"]["role"], "Supervisor")

    def test_reassign_site(self):
        from agents_api.models import Site

        site = Site.objects.create(tenant=self.tenant, name="New Site")
        self.client.force_authenticate(self.admin)
        res = self.client.patch(
            f"/api/core/org/employees/{self.emp.id}/",
            {"site_id": str(site.id)},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["employee"]["site_id"], str(site.id))

    def test_clear_site(self):
        from agents_api.models import Site

        site = Site.objects.create(tenant=self.tenant, name="Temp Site")
        self.emp.site = site
        self.emp.save()
        self.client.force_authenticate(self.admin)
        res = self.client.patch(
            f"/api/core/org/employees/{self.emp.id}/",
            {"site_id": None},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertIsNone(res.json()["employee"]["site_id"])

    def test_cannot_update_other_tenant_employee(self):
        from agents_api.models import Employee

        other_tenant = _make_tenant(name="Other", slug="other-emp", plan=Tenant.Plan.TEAM)
        other_emp = Employee.objects.create(tenant=other_tenant, name="Other", email="other@plant.com")
        self.client.force_authenticate(self.admin)
        res = self.client.put(
            f"/api/core/org/employees/{other_emp.id}/",
            {"name": "Hacked"},
            format="json",
        )
        self.assertEqual(res.status_code, 404)


@SECURE_OFF
class OrgDeleteEmployeeTest(TestCase):
    """Tests for DELETE /api/core/org/employees/<uuid>/delete/."""

    def setUp(self):
        from agents_api.models import Employee

        self.client = APIClient()
        self.tenant = _make_tenant(plan=Tenant.Plan.TEAM)
        self.admin = _make_user("delete-emp-admin@example.com", Tier.TEAM)
        _make_membership(self.tenant, self.admin, Membership.Role.ADMIN)
        self.emp = Employee.objects.create(tenant=self.tenant, name="Deletable", email="delete-me@plant.com")

    def test_delete_employee(self):
        self.client.force_authenticate(self.admin)
        res = self.client.delete(f"/api/core/org/employees/{self.emp.id}/delete/")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["success"])

        from agents_api.models import Employee

        self.assertFalse(Employee.objects.filter(id=self.emp.id).exists())

    def test_delete_blocked_with_active_commitments(self):
        """Employee with active ResourceCommitments cannot be deleted."""
        from datetime import date

        from agents_api.models import HoshinProject, ResourceCommitment, Site
        from core.models import Project

        site = Site.objects.create(tenant=self.tenant, name="Commitment Site")
        core_proj = Project.objects.create(title="HP", tenant=self.tenant)
        hoshin = HoshinProject.objects.create(project=core_proj, site=site)
        ResourceCommitment.objects.create(
            employee=self.emp,
            project=hoshin,
            role="team_member",
            status="active",
            start_date=date(2026, 1, 1),
            end_date=date(2026, 12, 31),
        )
        self.client.force_authenticate(self.admin)
        res = self.client.delete(f"/api/core/org/employees/{self.emp.id}/delete/")
        self.assertEqual(res.status_code, 409)
        err = res.json().get("error", {})
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        self.assertIn("commitment", msg.lower())

    def test_delete_nonexistent_employee(self):
        self.client.force_authenticate(self.admin)
        fake_id = uuid.uuid4()
        res = self.client.delete(f"/api/core/org/employees/{fake_id}/delete/")
        self.assertEqual(res.status_code, 404)
