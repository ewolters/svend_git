"""Tests for ResourceCommitment notification wiring.

Covers:
- Path A: user-linked employee gets in-app notification
- Path B: non-user employee gets ActionTokens + email scheduled
- ActionToken HTML confirm/decline flow
- Response notification to requester
- Failure isolation
"""

import json
from datetime import date, timedelta
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.utils import timezone

from accounts.constants import Tier
from agents_api.models import (
    ActionToken,
    Employee,
    HoshinProject,
    ResourceCommitment,
    Site,
)
from core.models import Membership, Project, Tenant
from notifications.models import Notification

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


def _setup_hoshin(user, slug="testcorp"):
    """Create tenant, site, core project, and hoshin project."""
    tenant = Tenant.objects.create(name="Test Corp", slug=slug)
    Membership.objects.create(user=user, tenant=tenant, role="owner")
    site = Site.objects.create(tenant=tenant, name="Plant A", code="PA")
    core_proj = Project.objects.create(user=user, title="OEE Improvement")
    hoshin = HoshinProject.objects.create(
        project=core_proj,
        site=site,
        hoshin_status="active",
        fiscal_year=2026,
    )
    return tenant, site, hoshin


# ── Path A: User-linked employee ────────────────────────────────────────


@SECURE_OFF
class UserLinkedCommitmentNotificationTest(TestCase):
    def setUp(self):
        self.coordinator = _make_user("coordinator@test.com")
        self.facilitator_user = _make_user("facilitator@test.com")
        self.tenant, self.site, self.hoshin = _setup_hoshin(self.coordinator, "corp-a")
        self.employee = Employee.objects.create(
            tenant=self.tenant,
            name="Jane Facilitator",
            email="facilitator@test.com",
            role="facilitator",
            user_link=self.facilitator_user,
        )

    def test_user_gets_notification_on_commitment_creation(self):
        from agents_api.commitment_notifications import notify_commitment_requested

        commitment = ResourceCommitment.objects.create(
            employee=self.employee,
            project=self.hoshin,
            role="facilitator",
            start_date=date.today(),
            end_date=date.today() + timedelta(days=5),
            requested_by=self.coordinator,
        )
        notify_commitment_requested(commitment)

        notif = Notification.objects.filter(
            recipient=self.facilitator_user,
            notification_type="commitment_requested",
        ).first()
        self.assertIsNotNone(notif)
        self.assertIn("OEE Improvement", notif.title)
        self.assertEqual(notif.entity_type, "hoshin_project")

    def test_requester_notified_on_confirm(self):
        from agents_api.commitment_notifications import notify_commitment_response

        commitment = ResourceCommitment.objects.create(
            employee=self.employee,
            project=self.hoshin,
            role="team_member",
            start_date=date.today(),
            end_date=date.today() + timedelta(days=3),
            requested_by=self.coordinator,
        )
        commitment.status = "confirmed"
        commitment.save()
        notify_commitment_response(commitment, old_status="requested")

        notif = Notification.objects.filter(
            recipient=self.coordinator,
            notification_type="commitment_confirmed",
        ).first()
        self.assertIsNotNone(notif)
        self.assertIn("confirmed", notif.title)
        self.assertIn("Jane Facilitator", notif.title)

    def test_requester_notified_on_decline(self):
        from agents_api.commitment_notifications import notify_commitment_response

        commitment = ResourceCommitment.objects.create(
            employee=self.employee,
            project=self.hoshin,
            role="team_member",
            start_date=date.today(),
            end_date=date.today() + timedelta(days=3),
            requested_by=self.coordinator,
        )
        commitment.status = "declined"
        commitment.save()
        notify_commitment_response(commitment, old_status="requested")

        notif = Notification.objects.filter(
            recipient=self.coordinator,
            notification_type="commitment_declined",
        ).first()
        self.assertIsNotNone(notif)
        self.assertIn("declined", notif.title)


# ── Path B: Non-user employee ──────────────────────────────────────────


@SECURE_OFF
class NonUserCommitmentNotificationTest(TestCase):
    def setUp(self):
        self.coordinator = _make_user("coord2@test.com")
        self.tenant, self.site, self.hoshin = _setup_hoshin(self.coordinator, "corp-b")
        self.employee = Employee.objects.create(
            tenant=self.tenant,
            name="Bob FloorWorker",
            email="bob@plant.com",
            role="team_member",
            user_link=None,
        )

    @patch("syn.sched.scheduler.schedule_task")
    def test_non_user_gets_action_tokens_and_email_scheduled(self, mock_schedule):
        from agents_api.commitment_notifications import notify_commitment_requested

        commitment = ResourceCommitment.objects.create(
            employee=self.employee,
            project=self.hoshin,
            role="team_member",
            start_date=date.today(),
            end_date=date.today() + timedelta(days=5),
            requested_by=self.coordinator,
        )
        notify_commitment_requested(commitment)

        # Two ActionTokens created
        tokens = ActionToken.objects.filter(employee=self.employee)
        self.assertEqual(tokens.count(), 2)
        action_types = set(tokens.values_list("action_type", flat=True))
        self.assertEqual(action_types, {"confirm_availability", "decline"})

        # Both scoped to this commitment
        for tok in tokens:
            self.assertEqual(tok.scoped_to["commitment_id"], str(commitment.id))

        # Email task scheduled
        mock_schedule.assert_called_once()

    @patch("syn.sched.scheduler.schedule_task")
    def test_no_notification_in_db_for_non_user(self, mock_schedule):
        """Non-user path should NOT create a Notification record (no User to attach to)."""
        from agents_api.commitment_notifications import notify_commitment_requested

        commitment = ResourceCommitment.objects.create(
            employee=self.employee,
            project=self.hoshin,
            role="team_member",
            start_date=date.today(),
            end_date=date.today() + timedelta(days=5),
            requested_by=self.coordinator,
        )
        notify_commitment_requested(commitment)
        self.assertEqual(Notification.objects.count(), 0)


# ── ActionToken HTML View ───────────────────────────────────────────────


@SECURE_OFF
class ActionTokenCommitmentViewTest(TestCase):
    def setUp(self):
        self.coordinator = _make_user("coord3@test.com")
        self.tenant, self.site, self.hoshin = _setup_hoshin(self.coordinator, "corp-c")
        self.employee = Employee.objects.create(
            tenant=self.tenant,
            name="Carol Engineer",
            email="carol@plant.com",
            user_link=None,
        )
        self.commitment = ResourceCommitment.objects.create(
            employee=self.employee,
            project=self.hoshin,
            role="subject_expert",
            start_date=date.today(),
            end_date=date.today() + timedelta(days=5),
            requested_by=self.coordinator,
        )
        self.confirm_tok = ActionToken.objects.create(
            employee=self.employee,
            action_type="confirm_availability",
            scoped_to={"commitment_id": str(self.commitment.id)},
        )
        self.decline_tok = ActionToken.objects.create(
            employee=self.employee,
            action_type="decline",
            scoped_to={"commitment_id": str(self.commitment.id)},
        )

    def test_get_confirm_returns_html(self):
        resp = self.client.get(f"/action/{self.confirm_tok.token}/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp["Content-Type"], "text/html")
        content = resp.content.decode()
        self.assertIn("Confirm", content)
        self.assertIn("OEE Improvement", content)
        self.assertIn("Subject Expert", content)

    def test_get_decline_returns_html(self):
        resp = self.client.get(f"/action/{self.decline_tok.token}/")
        self.assertEqual(resp.status_code, 200)
        content = resp.content.decode()
        self.assertIn("Decline", content)

    def test_post_confirm_transitions_commitment(self):
        resp = self.client.post(f"/action/{self.confirm_tok.token}/")
        self.assertEqual(resp.status_code, 200)
        content = resp.content.decode()
        self.assertIn("Confirmed", content)
        self.commitment.refresh_from_db()
        self.assertEqual(self.commitment.status, "confirmed")
        self.confirm_tok.refresh_from_db()
        self.assertIsNotNone(self.confirm_tok.used_at)

    def test_post_decline_transitions_commitment(self):
        resp = self.client.post(f"/action/{self.decline_tok.token}/")
        self.assertEqual(resp.status_code, 200)
        self.commitment.refresh_from_db()
        self.assertEqual(self.commitment.status, "declined")

    def test_post_confirm_notifies_requester(self):
        self.client.post(f"/action/{self.confirm_tok.token}/")
        notif = Notification.objects.filter(
            recipient=self.coordinator,
            notification_type="commitment_confirmed",
        ).first()
        self.assertIsNotNone(notif)
        self.assertIn("Carol Engineer", notif.title)

    def test_expired_token_returns_410_html(self):
        self.confirm_tok.expires_at = timezone.now() - timedelta(hours=1)
        self.confirm_tok.save(update_fields=["expires_at"])
        resp = self.client.get(f"/action/{self.confirm_tok.token}/")
        self.assertEqual(resp.status_code, 410)
        self.assertEqual(resp["Content-Type"], "text/html")
        self.assertIn("expired", resp.content.decode())

    def test_used_token_returns_410_html(self):
        self.confirm_tok.use()
        resp = self.client.get(f"/action/{self.confirm_tok.token}/")
        self.assertEqual(resp.status_code, 410)
        self.assertIn("already been used", resp.content.decode())

    def test_already_actioned_returns_409(self):
        self.commitment.status = "confirmed"
        self.commitment.save()
        resp = self.client.post(f"/action/{self.decline_tok.token}/")
        self.assertEqual(resp.status_code, 409)
        self.assertIn("already been actioned", resp.content.decode())


# ── Failure Isolation ───────────────────────────────────────────────────


@SECURE_OFF
class CommitmentNotificationFailureTest(TestCase):
    def setUp(self):
        self.coordinator = _make_user("coord4@test.com")
        self.tenant, self.site, self.hoshin = _setup_hoshin(self.coordinator, "corp-d")
        self.employee = Employee.objects.create(
            tenant=self.tenant,
            name="Dave",
            email="dave@test.com",
            user_link=None,
        )

    @patch("syn.sched.scheduler.schedule_task", side_effect=Exception("Scheduler down"))
    def test_notification_failure_does_not_block(self, mock_schedule):
        """notify_commitment_requested never raises even if scheduler fails."""
        from agents_api.commitment_notifications import notify_commitment_requested

        commitment = ResourceCommitment.objects.create(
            employee=self.employee,
            project=self.hoshin,
            role="team_member",
            start_date=date.today(),
            end_date=date.today() + timedelta(days=3),
            requested_by=self.coordinator,
        )
        # Should not raise
        notify_commitment_requested(commitment)
        # Commitment still exists
        self.assertTrue(ResourceCommitment.objects.filter(id=commitment.id).exists())
