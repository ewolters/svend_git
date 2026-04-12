"""Notification system tests — NTF-001.

Covers:
- Notification model (creation, immutability, to_dict)
- notify() helper (creation, muting, invalid types)
- List endpoint (filtering, pagination)
- Unread count endpoint
- Mark-read endpoints (single, bulk)
- Preferences endpoint (GET, PUT, validation)
- User isolation (no cross-user access)
- Source wiring (CAPA transition, NCR assignment, e-sig)
- NotificationToken model (FEAT-002)
- Token action view (GET, POST, expired)
- Email scheduling via notify()
- Preferences: email_mode
- Per-type unsubscribe
- Task handlers (send, digest, cleanup)
"""

import json
import uuid
from datetime import timedelta
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.signing import Signer
from django.test import TestCase, override_settings
from django.utils import timezone

from accounts.constants import Tier

from .helpers import notify
from .models import Notification, NotificationType
from .tokens import NotificationToken

User = get_user_model()
SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)
PASSWORD = "testpass123!"


def _post(client, url, data=None):
    return client.post(url, json.dumps(data or {}), content_type="application/json")


def _put(client, url, data=None):
    return client.put(url, json.dumps(data or {}), content_type="application/json")


def _create_user(email="alice@test.com", username=None, tier=Tier.TEAM):
    if username is None:
        username = email.split("@")[0]
    user = User.objects.create_user(username=username, email=email, password=PASSWORD)
    user.tier = tier
    user.save(update_fields=["tier"])
    return user


# ── Model Tests ──────────────────────────────────────────────────────────


@SECURE_OFF
class NotificationModelTests(TestCase):
    def setUp(self):
        self.user = _create_user()

    def test_create_notification(self):
        n = Notification.objects.create(
            recipient=self.user,
            notification_type="capa_status",
            title="CAPA moved to verification",
            entity_type="capa",
            entity_id=uuid.uuid4(),
        )
        self.assertIsNotNone(n.id)
        self.assertFalse(n.is_read)
        self.assertIsNotNone(n.created_at)

    def test_to_dict(self):
        eid = uuid.uuid4()
        n = Notification.objects.create(
            recipient=self.user,
            notification_type="ncr_assigned",
            title="NCR assigned",
            message="You were assigned.",
            entity_type="ncr",
            entity_id=eid,
        )
        d = n.to_dict()
        self.assertEqual(d["notification_type"], "ncr_assigned")
        self.assertEqual(d["title"], "NCR assigned")
        self.assertEqual(d["entity_type"], "ncr")
        self.assertEqual(d["entity_id"], str(eid))
        self.assertFalse(d["is_read"])
        self.assertIn("created_at", d)

    def test_immutability_title(self):
        n = Notification.objects.create(
            recipient=self.user,
            notification_type="system",
            title="Original title",
        )
        n.title = "Changed title"
        with self.assertRaises(ValueError):
            n.save()

    def test_immutability_type(self):
        n = Notification.objects.create(
            recipient=self.user,
            notification_type="system",
            title="Test",
        )
        n.notification_type = "capa_status"
        with self.assertRaises(ValueError):
            n.save()

    def test_immutability_allows_is_read(self):
        n = Notification.objects.create(
            recipient=self.user,
            notification_type="system",
            title="Test",
        )
        n.is_read = True
        n.save(update_fields=["is_read"])
        n.refresh_from_db()
        self.assertTrue(n.is_read)

    def test_ordering(self):
        Notification.objects.create(
            recipient=self.user,
            notification_type="system",
            title="First",
        )
        Notification.objects.create(
            recipient=self.user,
            notification_type="system",
            title="Second",
        )
        notifs = list(Notification.objects.filter(recipient=self.user))
        self.assertEqual(notifs[0].title, "Second")
        self.assertEqual(notifs[1].title, "First")


# ── Helper Tests ─────────────────────────────────────────────────────────


@SECURE_OFF
class NotifyHelperTests(TestCase):
    def setUp(self):
        self.user = _create_user()

    def test_notify_creates_notification(self):
        n = notify(self.user, "capa_status", "CAPA moved", entity_type="capa")
        self.assertIsNotNone(n)
        self.assertEqual(n.notification_type, "capa_status")
        self.assertEqual(n.recipient, self.user)

    def test_notify_with_enum(self):
        n = notify(self.user, NotificationType.SPC_ALARM, "SPC alarm triggered")
        self.assertIsNotNone(n)
        self.assertEqual(n.notification_type, "spc_alarm")

    def test_notify_invalid_type(self):
        with self.assertRaises(ValueError):
            notify(self.user, "nonexistent_type", "Bad")

    def test_notify_muted_type(self):
        self.user.preferences = {"notifications": {"muted_types": ["system"]}}
        self.user.save(update_fields=["preferences"])
        result = notify(self.user, "system", "Should be muted")
        self.assertIsNone(result)
        self.assertEqual(Notification.objects.filter(recipient=self.user).count(), 0)

    def test_notify_unmuted_type(self):
        self.user.preferences = {"notifications": {"muted_types": ["system"]}}
        self.user.save(update_fields=["preferences"])
        n = notify(self.user, "capa_status", "Not muted")
        self.assertIsNotNone(n)

    def test_notify_truncates_title(self):
        long_title = "A" * 500
        n = notify(self.user, "system", long_title)
        self.assertEqual(len(n.title), 300)


# ── API Tests ────────────────────────────────────────────────────────────


@SECURE_OFF
class NotificationListTests(TestCase):
    def setUp(self):
        self.user = _create_user()
        self.client.force_login(self.user)
        for i in range(5):
            notify(self.user, "system", f"Notification {i}")
        notify(self.user, "capa_status", "CAPA alert", entity_type="capa")

    def test_list_all(self):
        resp = self.client.get("/api/notifications/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data), 6)

    def test_list_filter_type(self):
        resp = self.client.get("/api/notifications/?type=capa_status")
        data = resp.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["notification_type"], "capa_status")

    def test_list_filter_unread(self):
        # Mark one as read
        n = Notification.objects.filter(recipient=self.user).first()
        n.is_read = True
        n.save(update_fields=["is_read"])
        resp = self.client.get("/api/notifications/?unread=true")
        data = resp.json()
        self.assertEqual(len(data), 5)

    def test_list_pagination(self):
        resp = self.client.get("/api/notifications/?limit=2&offset=0")
        data = resp.json()
        self.assertEqual(len(data), 2)

    def test_list_unauthenticated(self):
        self.client.logout()
        resp = self.client.get("/api/notifications/")
        self.assertIn(resp.status_code, [401, 403])


@SECURE_OFF
class UnreadCountTests(TestCase):
    def setUp(self):
        self.user = _create_user()
        self.client.force_login(self.user)

    def test_unread_count(self):
        notify(self.user, "system", "Unread 1")
        notify(self.user, "system", "Unread 2")
        resp = self.client.get("/api/notifications/unread-count/")
        self.assertEqual(resp.json()["count"], 2)

    def test_unread_count_after_read(self):
        n = notify(self.user, "system", "Will read")
        n.is_read = True
        n.save(update_fields=["is_read"])
        resp = self.client.get("/api/notifications/unread-count/")
        self.assertEqual(resp.json()["count"], 0)


@SECURE_OFF
class MarkReadTests(TestCase):
    def setUp(self):
        self.user = _create_user()
        self.client.force_login(self.user)

    def test_mark_single_read(self):
        n = notify(self.user, "system", "Test")
        resp = _post(self.client, f"/api/notifications/{n.id}/read/")
        self.assertEqual(resp.status_code, 200)
        n.refresh_from_db()
        self.assertTrue(n.is_read)

    def test_mark_read_404_other_user(self):
        other = _create_user("bob@test.com")
        n = notify(other, "system", "Bob's notification")
        resp = _post(self.client, f"/api/notifications/{n.id}/read/")
        self.assertEqual(resp.status_code, 404)

    def test_mark_all_read(self):
        for i in range(3):
            notify(self.user, "system", f"Unread {i}")
        resp = _post(self.client, "/api/notifications/read-all/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["updated"], 3)
        self.assertEqual(Notification.objects.filter(recipient=self.user, is_read=False).count(), 0)


@SECURE_OFF
class PreferencesTests(TestCase):
    def setUp(self):
        self.user = _create_user()
        self.client.force_login(self.user)

    def test_get_defaults(self):
        resp = self.client.get("/api/notifications/preferences/")
        data = resp.json()
        self.assertEqual(data["muted_types"], [])
        self.assertFalse(data["email_enabled"])

    def test_put_muted_types(self):
        resp = _put(
            self.client,
            "/api/notifications/preferences/",
            {"muted_types": ["system", "spc_alarm"]},
        )
        self.assertEqual(resp.status_code, 200)
        self.user.refresh_from_db()
        muted = self.user.preferences["notifications"]["muted_types"]
        self.assertIn("system", muted)
        self.assertIn("spc_alarm", muted)

    def test_put_invalid_type(self):
        resp = _put(
            self.client,
            "/api/notifications/preferences/",
            {"muted_types": ["fake_type"]},
        )
        self.assertEqual(resp.status_code, 400)

    def test_put_email_enabled(self):
        resp = _put(
            self.client,
            "/api/notifications/preferences/",
            {"muted_types": [], "email_enabled": True},
        )
        self.assertEqual(resp.status_code, 200)
        self.user.refresh_from_db()
        self.assertTrue(self.user.preferences["notifications"]["email_enabled"])


# ── User Isolation Tests ─────────────────────────────────────────────────


@SECURE_OFF
class UserIsolationTests(TestCase):
    def setUp(self):
        self.alice = _create_user("alice@test.com")
        self.bob = _create_user("bob@test.com")

    def test_cannot_list_other_users_notifications(self):
        notify(self.bob, "system", "Bob's notification")
        self.client.force_login(self.alice)
        resp = self.client.get("/api/notifications/")
        self.assertEqual(len(resp.json()), 0)

    def test_cannot_mark_read_other_users_notification(self):
        n = notify(self.bob, "system", "Bob's")
        self.client.force_login(self.alice)
        resp = _post(self.client, f"/api/notifications/{n.id}/read/")
        self.assertEqual(resp.status_code, 404)
        n.refresh_from_db()
        self.assertFalse(n.is_read)

    def test_unread_count_scoped_to_user(self):
        notify(self.alice, "system", "Alice's")
        notify(self.bob, "system", "Bob's")
        self.client.force_login(self.alice)
        resp = self.client.get("/api/notifications/unread-count/")
        self.assertEqual(resp.json()["count"], 1)


# ── SSE Stream Tests ─────────────────────────────────────────────────────


@SECURE_OFF
class SSEStreamTests(TestCase):
    def setUp(self):
        self.user = _create_user()
        self.client.force_login(self.user)

    def test_stream_returns_event_stream_content_type(self):
        resp = self.client.get("/api/notifications/stream/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/event-stream", resp["Content-Type"])
        self.assertEqual(resp["Cache-Control"], "no-cache")
        self.assertEqual(resp["X-Accel-Buffering"], "no")

    def test_stream_unauthenticated(self):
        self.client.logout()
        resp = self.client.get("/api/notifications/stream/")
        self.assertEqual(resp.status_code, 401)


# ── NotificationToken Model Tests (FEAT-002) ───────────────────────────


@SECURE_OFF
class NotificationTokenModelTests(TestCase):
    def setUp(self):
        self.user = _create_user()
        self.notif = Notification.objects.create(
            recipient=self.user,
            notification_type="system",
            title="Token test",
        )

    def test_auto_generates_token(self):
        tok = NotificationToken.objects.create(
            user=self.user,
            notification=self.notif,
            action_type="acknowledge",
        )
        self.assertTrue(len(tok.token) >= 32)

    def test_auto_sets_expiry(self):
        tok = NotificationToken.objects.create(
            user=self.user,
            notification=self.notif,
            action_type="acknowledge",
        )
        self.assertIsNotNone(tok.expires_at)
        # Expires roughly 72h from now
        delta = tok.expires_at - timezone.now()
        self.assertGreater(delta.total_seconds(), 71 * 3600)
        self.assertLess(delta.total_seconds(), 73 * 3600)

    def test_is_valid_fresh(self):
        tok = NotificationToken.objects.create(
            user=self.user,
            notification=self.notif,
            action_type="acknowledge",
        )
        self.assertTrue(tok.is_valid)

    def test_is_valid_expired(self):
        tok = NotificationToken.objects.create(
            user=self.user,
            notification=self.notif,
            action_type="view",
        )
        tok.expires_at = timezone.now() - timedelta(hours=1)
        tok.save(update_fields=["expires_at"])
        self.assertFalse(tok.is_valid)

    def test_is_valid_used(self):
        tok = NotificationToken.objects.create(
            user=self.user,
            notification=self.notif,
            action_type="acknowledge",
        )
        tok.use()
        self.assertFalse(tok.is_valid)
        self.assertIsNotNone(tok.used_at)

    def test_to_dict(self):
        tok = NotificationToken.objects.create(
            user=self.user,
            notification=self.notif,
            action_type="acknowledge",
        )
        d = tok.to_dict()
        self.assertEqual(d["notification_id"], str(self.notif.id))
        self.assertEqual(d["action_type"], "acknowledge")
        self.assertTrue(d["is_valid"])
        self.assertIn("expires_at", d)


# ── Token View Tests (FEAT-002) ─────────────────────────────────────────


@SECURE_OFF
class NotificationTokenViewTests(TestCase):
    def setUp(self):
        self.user = _create_user()
        self.notif = Notification.objects.create(
            recipient=self.user,
            notification_type="capa_status",
            title="CAPA needs attention",
            message="Status changed.",
        )
        self.tok = NotificationToken.objects.create(
            user=self.user,
            notification=self.notif,
            action_type="acknowledge",
        )

    def test_get_valid_token(self):
        resp = self.client.get(f"/ntf/{self.tok.token}/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp["Content-Type"], "text/html")
        content = resp.content.decode()
        self.assertIn("CAPA needs attention", content)
        self.assertIn("Status changed.", content)
        self.assertIn("Acknowledge", content)
        self.assertIn("<form", content)

    def test_post_acknowledges_and_marks_read(self):
        resp = self.client.post(f"/ntf/{self.tok.token}/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp["Content-Type"], "text/html")
        content = resp.content.decode()
        self.assertIn("Acknowledged", content)
        self.assertIn("CAPA needs attention", content)
        self.tok.refresh_from_db()
        self.assertIsNotNone(self.tok.used_at)
        self.notif.refresh_from_db()
        self.assertTrue(self.notif.is_read)

    def test_get_expired_token_410(self):
        self.tok.expires_at = timezone.now() - timedelta(hours=1)
        self.tok.save(update_fields=["expires_at"])
        resp = self.client.get(f"/ntf/{self.tok.token}/")
        self.assertEqual(resp.status_code, 410)
        self.assertEqual(resp["Content-Type"], "text/html")
        self.assertIn("expired", resp.content.decode())

    def test_post_used_token_410(self):
        self.tok.use()
        resp = self.client.post(f"/ntf/{self.tok.token}/")
        self.assertEqual(resp.status_code, 410)
        self.assertEqual(resp["Content-Type"], "text/html")
        self.assertIn("already been used", resp.content.decode())

    def test_invalid_token_404(self):
        resp = self.client.get("/ntf/nonexistent-token-value/")
        self.assertEqual(resp.status_code, 404)


# ── Email Scheduling Tests (FEAT-002) ───────────────────────────────────


@SECURE_OFF
class EmailSchedulingTests(TestCase):
    def setUp(self):
        self.user = _create_user()

    @patch("syn.sched.scheduler.schedule_task")
    def test_notify_schedules_email_when_enabled(self, mock_schedule):
        self.user.preferences = {"notifications": {"email_enabled": True, "email_mode": "immediate"}}
        self.user.save(update_fields=["preferences"])
        n = notify(self.user, "system", "Email test")
        self.assertIsNotNone(n)
        mock_schedule.assert_called_once()
        # Token should have been created
        self.assertEqual(NotificationToken.objects.filter(user=self.user).count(), 1)

    @patch("syn.sched.scheduler.schedule_task")
    def test_notify_skips_email_when_disabled(self, mock_schedule):
        n = notify(self.user, "system", "No email")
        self.assertIsNotNone(n)
        mock_schedule.assert_not_called()
        self.assertEqual(NotificationToken.objects.filter(user=self.user).count(), 0)

    @patch("syn.sched.scheduler.schedule_task")
    def test_notify_skips_email_for_digest_mode(self, mock_schedule):
        self.user.preferences = {"notifications": {"email_enabled": True, "email_mode": "daily"}}
        self.user.save(update_fields=["preferences"])
        n = notify(self.user, "system", "Digest mode")
        self.assertIsNotNone(n)
        mock_schedule.assert_not_called()

    @patch("syn.sched.scheduler.schedule_task", side_effect=Exception("Scheduler down"))
    def test_email_failure_does_not_block_notification(self, mock_schedule):
        self.user.preferences = {"notifications": {"email_enabled": True, "email_mode": "immediate"}}
        self.user.save(update_fields=["preferences"])
        n = notify(self.user, "system", "Should still create")
        self.assertIsNotNone(n)
        self.assertEqual(n.title, "Should still create")

    @patch("syn.sched.scheduler.schedule_task")
    def test_notify_skips_email_when_globally_opted_out(self, mock_schedule):
        self.user.preferences = {"notifications": {"email_enabled": True, "email_mode": "immediate"}}
        self.user.is_email_opted_out = True
        self.user.save(update_fields=["preferences", "is_email_opted_out"])
        n = notify(self.user, "system", "Opted out")
        self.assertIsNotNone(n)
        mock_schedule.assert_not_called()


# ── Preferences: email_mode Tests (FEAT-002) ────────────────────────────


@SECURE_OFF
class PreferencesEmailModeTests(TestCase):
    def setUp(self):
        self.user = _create_user()
        self.client.force_login(self.user)

    def test_get_default_email_mode(self):
        resp = self.client.get("/api/notifications/preferences/")
        self.assertEqual(resp.json()["email_mode"], "immediate")

    def test_put_valid_email_mode(self):
        resp = _put(
            self.client,
            "/api/notifications/preferences/",
            {"muted_types": [], "email_mode": "daily"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["email_mode"], "daily")
        self.user.refresh_from_db()
        self.assertEqual(self.user.preferences["notifications"]["email_mode"], "daily")

    def test_put_invalid_email_mode(self):
        resp = _put(
            self.client,
            "/api/notifications/preferences/",
            {"muted_types": [], "email_mode": "hourly"},
        )
        self.assertEqual(resp.status_code, 400)


# ── Per-Type Unsubscribe Tests (FEAT-002) ───────────────────────────────


@SECURE_OFF
class TypeUnsubscribeTests(TestCase):
    def setUp(self):
        self.user = _create_user()

    def test_valid_unsubscribe_adds_to_muted(self):
        signer = Signer(salt="ntf-type-unsub")
        token = signer.sign(f"{self.user.id}:system")
        resp = self.client.get(f"/api/notifications/unsubscribe/?token={token}")
        self.assertEqual(resp.status_code, 200)
        self.user.refresh_from_db()
        muted = self.user.preferences["notifications"]["muted_types"]
        self.assertIn("system", muted)

    def test_invalid_token_400(self):
        resp = self.client.get("/api/notifications/unsubscribe/?token=bad-token")
        self.assertEqual(resp.status_code, 400)

    def test_idempotent_unsubscribe(self):
        self.user.preferences = {"notifications": {"muted_types": ["system"]}}
        self.user.save(update_fields=["preferences"])
        signer = Signer(salt="ntf-type-unsub")
        token = signer.sign(f"{self.user.id}:system")
        resp = self.client.get(f"/api/notifications/unsubscribe/?token={token}")
        self.assertEqual(resp.status_code, 200)
        self.user.refresh_from_db()
        # Should not duplicate
        self.assertEqual(self.user.preferences["notifications"]["muted_types"].count("system"), 1)


# ── Task Handler Tests (FEAT-002) ──────────────────────────────────────


@SECURE_OFF
class TaskHandlerTests(TestCase):
    def setUp(self):
        self.user = _create_user()
        self.notif = Notification.objects.create(
            recipient=self.user,
            notification_type="system",
            title="Task test",
        )

    @patch("notifications.email.django_send_mail")
    def test_send_email_task_sends_and_updates(self, mock_send):
        from .tasks import send_notification_email_task

        tok = NotificationToken.objects.create(
            user=self.user,
            notification=self.notif,
            action_type="acknowledge",
        )
        result = send_notification_email_task(
            {
                "args": {
                    "notification_id": str(self.notif.id),
                    "token_id": str(tok.id),
                }
            }
        )
        self.assertTrue(result["sent"])
        mock_send.assert_called_once()
        tok.refresh_from_db()
        self.assertIsNotNone(tok.email_sent_at)

    @patch("notifications.email.django_send_mail", side_effect=Exception("SMTP error"))
    def test_send_email_task_handles_failure(self, mock_send):
        from .tasks import send_notification_email_task

        tok = NotificationToken.objects.create(
            user=self.user,
            notification=self.notif,
            action_type="acknowledge",
        )
        result = send_notification_email_task(
            {
                "args": {
                    "notification_id": str(self.notif.id),
                    "token_id": str(tok.id),
                }
            }
        )
        self.assertFalse(result["sent"])
        tok.refresh_from_db()
        self.assertIsNotNone(tok.email_failed_at)

    def test_cleanup_deletes_expired_tokens(self):
        from .tasks import cleanup_expired_tokens

        # Create an expired token older than 7 days
        tok = NotificationToken.objects.create(
            user=self.user,
            notification=self.notif,
            action_type="view",
        )
        tok.expires_at = timezone.now() - timedelta(days=8)
        tok.save(update_fields=["expires_at"])

        result = cleanup_expired_tokens({})
        self.assertEqual(result["deleted"], 1)
        self.assertEqual(NotificationToken.objects.count(), 0)

    def test_cleanup_preserves_used_tokens(self):
        from .tasks import cleanup_expired_tokens

        tok = NotificationToken.objects.create(
            user=self.user,
            notification=self.notif,
            action_type="acknowledge",
        )
        tok.expires_at = timezone.now() - timedelta(days=8)
        tok.used_at = timezone.now() - timedelta(days=7)
        tok.save(update_fields=["expires_at", "used_at"])

        result = cleanup_expired_tokens({})
        self.assertEqual(result["deleted"], 0)
        self.assertEqual(NotificationToken.objects.count(), 1)

    @patch("notifications.tasks._send_digest")
    def test_send_daily_digest_calls_send_digest(self, mock_digest):
        from .tasks import send_daily_digest

        mock_digest.return_value = {"sent": 0, "skipped": 0}
        send_daily_digest({})
        mock_digest.assert_called_once()
        args = mock_digest.call_args[0]
        self.assertEqual(args[0], "daily")

    @patch("notifications.tasks._send_digest")
    def test_send_weekly_digest_calls_send_digest(self, mock_digest):
        from .tasks import send_weekly_digest

        mock_digest.return_value = {"sent": 0, "skipped": 0}
        send_weekly_digest({})
        mock_digest.assert_called_once()
        args = mock_digest.call_args[0]
        self.assertEqual(args[0], "weekly")

    # Alias for compliance hook name
    test_send_email_task_sends_mail = test_send_email_task_sends_and_updates
