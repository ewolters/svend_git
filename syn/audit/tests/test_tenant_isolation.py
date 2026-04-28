"""Tenant isolation and privilege escalation tests.

⚠ COMPLIANCE-CRITICAL: These tests are the proof that multi-tenant data
isolation works. They are referenced by the tenant_isolation compliance
check (daily critical) and directly support SOC 2 CC6.3 (role-based access).

If any of these tests fail, it means a user can access data they should not
be able to see. This is a security incident, not a test flake.

DO NOT:
  - Skip or @expectedFailure any test in this file (TST-001 §11.6)
  - Weaken assertions to make tests pass
  - Remove tests without a ChangeRequest (CHG-001)

Standard: SEC-001 §5 (Tenant Isolation)
Compliance: SOC 2 CC6.3 (Role-Based Access)

<!-- assert: Tenant isolation tests verify cross-user and cross-tenant boundaries | check=sec-tenant-isolation -->
<!-- impl: syn/audit/tests/test_tenant_isolation.py -->
<!-- test: syn.audit.tests.test_tenant_isolation.CrossUserProjectTest.test_user_cannot_access_other_users_project -->
<!-- test: syn.audit.tests.test_tenant_isolation.CrossTenantSiteTest.test_user_cannot_link_other_tenants_site -->
<!-- test: syn.audit.tests.test_tenant_isolation.ViewerWriteTest.test_viewer_cannot_create_fmea -->
"""

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from accounts.constants import Tier
from conftest import make_membership, make_tenant, make_user
from core.models.tenant import Membership

User = get_user_model()

SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False, RATELIMIT_ENABLE=False)


def _verified_user(email, tier=Tier.PRO, **kwargs):
    """Create a user who passes all auth gates (email verified, paid tier)."""
    user = make_user(email, tier=tier, is_email_verified=True, **kwargs)
    return user


# ---------------------------------------------------------------------------
# Cross-user project access (individual users, no tenant)
# ---------------------------------------------------------------------------


@SECURE_OFF
class CrossUserProjectTest(TestCase):
    """Verify users cannot access each other's personal projects."""

    def setUp(self):
        self.alice = _verified_user("alice@test.com", tier=Tier.PRO)
        self.bob = _verified_user("bob@test.com", tier=Tier.PRO)

        self.alice_client = APIClient()
        self.alice_client.force_authenticate(self.alice)
        self.bob_client = APIClient()
        self.bob_client.force_authenticate(self.bob)

        # Alice creates a project
        res = self.alice_client.post(
            "/api/core/projects/",
            {"title": "Alice's Secret Project", "description": "Private data"},
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        self.alice_project_id = res.json()["id"]

    def test_user_cannot_access_other_users_project(self):
        """Bob cannot GET Alice's project."""
        res = self.bob_client.get(f"/api/core/projects/{self.alice_project_id}/")
        self.assertIn(res.status_code, [403, 404])

    def test_user_cannot_update_other_users_project(self):
        """Bob cannot PUT Alice's project."""
        res = self.bob_client.put(
            f"/api/core/projects/{self.alice_project_id}/",
            {"title": "Hacked by Bob"},
            format="json",
        )
        self.assertIn(res.status_code, [403, 404])

    def test_user_cannot_delete_other_users_project(self):
        """Bob cannot DELETE Alice's project."""
        res = self.bob_client.delete(f"/api/core/projects/{self.alice_project_id}/")
        self.assertIn(res.status_code, [403, 404])

    def test_project_list_only_shows_own(self):
        """Bob's project list does not include Alice's project."""
        res = self.bob_client.get("/api/core/projects/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        # Response may be a list or dict with 'projects' key
        if isinstance(data, list):
            project_ids = [p["id"] for p in data]
        else:
            project_ids = [p["id"] for p in data.get("projects", [])]
        self.assertNotIn(self.alice_project_id, project_ids)


# ---------------------------------------------------------------------------
# Cross-tenant site isolation
# ---------------------------------------------------------------------------


@SECURE_OFF
class CrossTenantSiteTest(TestCase):
    """Verify users cannot link sites from other tenants."""

    def setUp(self):
        from agents_api.models import Site

        # Tenant A with user
        self.tenant_a = make_tenant(name="Org A", slug="org-a")
        self.user_a = _verified_user("user-a@test.com", tier=Tier.TEAM)
        make_membership(self.tenant_a, self.user_a, role=Membership.Role.ADMIN)
        self.site_a = Site.objects.create(tenant=self.tenant_a, name="Plant Alpha", code="ALPHA")

        # Tenant B with user
        self.tenant_b = make_tenant(name="Org B", slug="org-b")
        self.user_b = _verified_user("user-b@test.com", tier=Tier.TEAM)
        make_membership(self.tenant_b, self.user_b, role=Membership.Role.ADMIN)
        self.site_b = Site.objects.create(tenant=self.tenant_b, name="Plant Beta", code="BETA")

        self.client_a = APIClient()
        self.client_a.force_authenticate(self.user_a)
        self.client_b = APIClient()
        self.client_b.force_authenticate(self.user_b)

    def test_user_cannot_link_other_tenants_site(self):
        """User A cannot create an FMEA linked to Tenant B's site.

        Tests the resolve_site() helper directly since the HTTP endpoint
        has auth gates that complicate end-to-end testing.
        """
        from qms_core.permissions import resolve_site

        site, err = resolve_site(self.user_a, str(self.site_b.id))
        self.assertIsNotNone(err, "resolve_site should reject cross-tenant site")
        self.assertIsNone(site)

    def test_user_can_link_own_tenants_site(self):
        """User A CAN link their own tenant's site."""
        from qms_core.permissions import resolve_site

        site, err = resolve_site(self.user_a, str(self.site_a.id))
        self.assertIsNone(err)
        self.assertIsNotNone(site)
        self.assertEqual(site.id, self.site_a.id)

    def test_resolve_site_returns_none_for_wrong_tenant(self):
        """resolve_site helper rejects cross-tenant site IDs."""
        from qms_core.permissions import resolve_site

        site, err = resolve_site(self.user_a, str(self.site_b.id))
        # Should return error (site not found in their tenant)
        self.assertIsNotNone(err)
        self.assertIsNone(site)

    def test_resolve_site_works_for_own_tenant(self):
        """resolve_site helper accepts own tenant's site ID."""
        from qms_core.permissions import resolve_site

        site, err = resolve_site(self.user_a, str(self.site_a.id))
        self.assertIsNone(err)
        self.assertEqual(site.id, self.site_a.id)

    def test_resolve_site_returns_none_for_individual_user(self):
        """Individual users (no tenant) get None for any site ID."""
        from qms_core.permissions import resolve_site

        individual = _verified_user("solo@test.com", tier=Tier.PRO)
        site, err = resolve_site(individual, str(self.site_a.id))
        self.assertIsNone(err)
        self.assertIsNone(site)


# ---------------------------------------------------------------------------
# resolve_project tenant awareness
# ---------------------------------------------------------------------------


@SECURE_OFF
class ResolveProjectTest(TestCase):
    """Verify resolve_project includes tenant projects and excludes others."""

    def setUp(self):
        from core.models import Project

        # Tenant with two members
        self.tenant = make_tenant(name="Shared Org", slug="shared-org")
        self.alice = _verified_user("rp-alice@test.com", tier=Tier.TEAM)
        self.bob = _verified_user("rp-bob@test.com", tier=Tier.TEAM)
        self.outsider = _verified_user("rp-outsider@test.com", tier=Tier.PRO)
        make_membership(self.tenant, self.alice, role=Membership.Role.ADMIN)
        make_membership(self.tenant, self.bob, role=Membership.Role.MEMBER)

        # Tenant project (no user, owned by tenant)
        self.tenant_project = Project.objects.create(tenant=self.tenant, title="Shared Project")
        # Alice's personal project
        self.alice_personal = Project.objects.create(user=self.alice, title="Alice Personal")

    def test_tenant_member_can_resolve_tenant_project(self):
        """Bob (tenant member) can resolve the shared tenant project."""
        from qms_core.permissions import resolve_project

        project, err = resolve_project(self.bob, str(self.tenant_project.id))
        self.assertIsNone(err)
        self.assertIsNotNone(project)
        self.assertEqual(project.id, self.tenant_project.id)

    def test_outsider_cannot_resolve_tenant_project(self):
        """Outsider cannot resolve tenant project."""
        from qms_core.permissions import resolve_project

        project, err = resolve_project(self.outsider, str(self.tenant_project.id))
        self.assertIsNotNone(err)

    def test_outsider_cannot_resolve_alice_personal(self):
        """Outsider cannot resolve Alice's personal project."""
        from qms_core.permissions import resolve_project

        project, err = resolve_project(self.outsider, str(self.alice_personal.id))
        self.assertIsNotNone(err)

    def test_alice_can_resolve_own_personal(self):
        """Alice can resolve her own personal project."""
        from qms_core.permissions import resolve_project

        project, err = resolve_project(self.alice, str(self.alice_personal.id))
        self.assertIsNone(err)
        self.assertIsNotNone(project)


# ---------------------------------------------------------------------------
# VIEWER role write restrictions
# ---------------------------------------------------------------------------


@SECURE_OFF
class ViewerWriteTest(TestCase):
    """Verify VIEWER role members cannot create/modify QMS resources."""

    def setUp(self):
        from agents_api.models import Site

        self.tenant = make_tenant(name="Viewer Org", slug="viewer-org")
        self.admin = _verified_user("vw-admin@test.com", tier=Tier.TEAM)
        self.viewer = _verified_user("vw-viewer@test.com", tier=Tier.TEAM)
        make_membership(self.tenant, self.admin, role=Membership.Role.ADMIN)
        make_membership(self.tenant, self.viewer, role=Membership.Role.VIEWER)

        self.site = Site.objects.create(tenant=self.tenant, name="Viewer Plant", code="VP")

        self.viewer_client = APIClient()
        self.viewer_client.force_authenticate(self.viewer)

    def test_viewer_cannot_create_fmea(self):
        """VIEWER cannot create FMEA at a site they have no write access to."""
        from qms_core.permissions import check_site_write

        can_write = check_site_write(self.viewer, self.site, self.tenant)
        self.assertFalse(can_write)

    def test_viewer_cannot_edit_qms_record(self):
        """VIEWER cannot edit a QMS record they don't own."""
        # Create a record as admin, verify viewer can't edit
        from agents_api.models import FMEA
        from qms_core.permissions import qms_can_edit

        fmea = FMEA.objects.create(title="Admin FMEA", site=self.site, created_by=self.admin)
        can_edit = qms_can_edit(self.viewer, fmea, self.tenant)
        self.assertFalse(can_edit)

    def test_admin_can_edit_any_record(self):
        """Admin CAN edit any record in their tenant."""
        from agents_api.models import FMEA
        from qms_core.permissions import qms_can_edit

        fmea = FMEA.objects.create(title="Viewer FMEA", site=self.site, created_by=self.viewer)
        can_edit = qms_can_edit(self.admin, fmea, self.tenant)
        self.assertTrue(can_edit)


# ---------------------------------------------------------------------------
# UUID guessing (direct ID access without ownership)
# ---------------------------------------------------------------------------


@SECURE_OFF
class UUIDGuessingTest(TestCase):
    """Verify that knowing a UUID is not sufficient to access a resource."""

    def setUp(self):
        self.alice = _verified_user("uuid-alice@test.com", tier=Tier.PRO)
        self.bob = _verified_user("uuid-bob@test.com", tier=Tier.PRO)

        self.alice_client = APIClient()
        self.alice_client.force_authenticate(self.alice)
        self.bob_client = APIClient()
        self.bob_client.force_authenticate(self.bob)

    def test_cannot_access_fmea_by_uuid(self):
        """Bob cannot GET Alice's FMEA by UUID."""
        from agents_api.models import FMEA

        fmea = FMEA.objects.create(title="Alice FMEA", owner=self.alice)
        res = self.bob_client.get(f"/api/fmea/{fmea.id}/")
        # 401 (auth gate), 403 (forbidden), or 404 (not found) all acceptable
        self.assertIn(res.status_code, [401, 403, 404])

    def test_cannot_access_rca_by_uuid(self):
        """Bob cannot GET Alice's RCA session by UUID."""
        from agents_api.models import RCASession

        rca = RCASession.objects.create(title="Alice RCA", event="Test", owner=self.alice)
        res = self.bob_client.get(f"/api/rca/{rca.id}/")
        self.assertIn(res.status_code, [401, 403, 404])

    def test_cannot_access_a3_by_uuid(self):
        """Bob cannot GET Alice's A3 report by UUID."""
        from agents_api.models import A3Report
        from core.models import Project

        project = Project.objects.create(user=self.alice, title="Alice A3 Project")
        a3 = A3Report.objects.create(title="Alice A3", owner=self.alice, project=project)
        res = self.bob_client.get(f"/api/a3/{a3.id}/")
        self.assertIn(res.status_code, [401, 403, 404])
