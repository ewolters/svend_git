"""Centralized test fixtures for all Svend test files.

Replaces 46 duplicate _make_user() helpers across the codebase with a single
canonical implementation. All test files can import these directly or use them
as pytest fixtures.

Standard: TST-001 §6 (Fixture Patterns)
Compliance: SOC 2 CC4.1 (Monitoring Activities)

Usage in Django TestCase (existing pattern — no migration required):

    from conftest import make_user, make_tenant, make_membership, SECURE_OFF

    @SECURE_OFF
    class MyTest(TestCase):
        def setUp(self):
            self.user = make_user("test@example.com", tier=Tier.PRO)

Usage as pytest fixtures (new tests):

    def test_something(user, pro_user, team_user, api_client):
        api_client.force_authenticate(pro_user)
        res = api_client.get("/api/endpoint/")

⚠ COMPLIANCE NOTE: This module is loaded by the compliance runner during
test_execution checks. Changes here affect ALL tests across the platform.
Verify with: python manage.py run_compliance --check=test_execution
"""

import pytest
from django.test import override_settings

# ---------------------------------------------------------------------------
# Production SSL redirect breaks test client HTTP requests.
# TST-001 §7.2 — standard decorator for all API test classes.
# ---------------------------------------------------------------------------
SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)


# ---------------------------------------------------------------------------
# Canonical helper functions (callable from Django TestCase setUp)
# ---------------------------------------------------------------------------


def make_user(email, tier=None, password="testpass123!", **kwargs):
    """Create a user with a given tier.

    This is the canonical implementation — TST-001 §6.1. All 46 prior copies
    of _make_user() across the codebase delegate to this pattern.

    Args:
        email: User email (username derived from local part).
        tier: Tier enum value (FREE, FOUNDER, PRO, TEAM, ENTERPRISE). Defaults to FREE.
        password: Plain-text password for test auth.
        **kwargs: Additional User fields (is_staff, is_email_verified, etc.).
    """
    from django.contrib.auth import get_user_model

    from accounts.constants import Tier

    User = get_user_model()
    if tier is None:
        tier = Tier.FREE
    username = kwargs.pop("username", email.split("@")[0])
    user = User.objects.create_user(username=username, email=email, password=password, **kwargs)
    user.tier = tier
    user.save(update_fields=["tier"])
    return user


def make_tenant(name="Test Org", slug="test-org", plan=None, **kwargs):
    """Create a tenant (organization).

    Args:
        name: Organization display name.
        slug: URL-safe slug (unique).
        plan: Tenant.Plan value (defaults to TEAM).
        **kwargs: Additional Tenant fields (max_members, settings, etc.).
    """
    from core.models.tenant import Tenant

    if plan is None:
        plan = Tenant.Plan.TEAM
    return Tenant.objects.create(name=name, slug=slug, plan=plan, **kwargs)


def make_membership(tenant, user, role=None, **kwargs):
    """Create an active membership linking a user to a tenant.

    Args:
        tenant: Tenant instance.
        user: User instance.
        role: Membership.Role value (defaults to MEMBER).
        **kwargs: Additional Membership fields.
    """
    from django.utils import timezone

    from core.models.tenant import Membership

    if role is None:
        role = Membership.Role.MEMBER
    return Membership.objects.create(tenant=tenant, user=user, role=role, joined_at=timezone.now(), **kwargs)


# ---------------------------------------------------------------------------
# pytest fixtures (for new tests using pytest style)
# ---------------------------------------------------------------------------


@pytest.fixture
def api_client():
    """DRF APIClient instance — TST-001 §7.1."""
    from rest_framework.test import APIClient

    return APIClient()


@pytest.fixture
def user(db):
    """Free-tier user for basic auth tests."""
    return make_user("fixture@test.com")


@pytest.fixture
def pro_user(db):
    """Pro-tier user for feature-gated tests."""
    from accounts.constants import Tier

    return make_user("pro@test.com", tier=Tier.PRO)


@pytest.fixture
def team_user(db):
    """Team-tier user for collaboration tests."""
    from accounts.constants import Tier

    return make_user("team@test.com", tier=Tier.TEAM)


@pytest.fixture
def enterprise_user(db):
    """Enterprise-tier user for full-access tests."""
    from accounts.constants import Tier

    return make_user("enterprise@test.com", tier=Tier.ENTERPRISE)


@pytest.fixture
def staff_user(db):
    """Staff user for internal dashboard tests."""
    from accounts.constants import Tier

    return make_user("staff@test.com", tier=Tier.PRO, is_staff=True)


@pytest.fixture
def tenant(db):
    """Team-plan tenant for multi-tenancy tests."""
    return make_tenant()


@pytest.fixture
def membership(db, tenant, team_user):
    """Active membership linking team_user to tenant."""
    return make_membership(tenant, team_user)
