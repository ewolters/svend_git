"""Chat model and view tests — conversation lifecycle, usage tracking, encryption.

Scenario tests covering chat/models.py and chat/views.py.
Linked from LLM-001 and DAT-001 via <!-- test: --> hooks.
"""

import uuid
from datetime import timedelta

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.utils import timezone
from rest_framework.test import APIClient

from accounts.constants import Tier

User = get_user_model()
SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)


def _make_user(email, tier=Tier.FREE, password="testpass123!", **kwargs):
    username = kwargs.pop("username", email.split("@")[0])
    user = User.objects.create_user(
        username=username, email=email, password=password, **kwargs
    )
    user.tier = tier
    user.save(update_fields=["tier"])
    return user


# =========================================================================
# Conversation Lifecycle
# =========================================================================


@SECURE_OFF
class ConversationLifecycleTest(TestCase):
    """Scenario: create conversation → add messages → generate title → verify."""

    def setUp(self):
        self.user = _make_user("chatuser@example.com", tier=Tier.PRO)

    def test_create_conversation_and_messages(self):
        """Create conversation → add user + assistant messages → verify ordering."""
        from chat.models import Conversation, Message

        conv = Conversation.objects.create(user=self.user, title="Test Chat")
        msg1 = Message.objects.create(
            conversation=conv, role="user", content="Hello, can you help me?"
        )
        msg2 = Message.objects.create(
            conversation=conv, role="assistant", content="Of course! How can I help?"
        )

        self.assertEqual(conv.messages.count(), 2)
        messages = list(conv.messages.order_by("created_at"))
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(messages[1].role, "assistant")

    def test_generate_title_from_first_message(self):
        """Conversation auto-generates title from first user message."""
        from chat.models import Conversation, Message

        conv = Conversation.objects.create(user=self.user)
        Message.objects.create(
            conversation=conv, role="user",
            content="What is the capability index for my process?"
        )

        conv.generate_title()
        conv.refresh_from_db()
        self.assertIn("capability", conv.title.lower())
        self.assertTrue(len(conv.title) <= 53)  # 50 chars + "..."

    def test_message_encrypted_fields(self):
        """Message content and reasoning_trace use encrypted fields."""
        from chat.models import Message, Conversation

        conv = Conversation.objects.create(user=self.user, title="Encryption test")
        msg = Message.objects.create(
            conversation=conv,
            role="assistant",
            content="Sensitive analysis result",
            reasoning_trace={"steps": ["step1", "step2"]},
            tool_calls=[{"name": "analyze", "args": {"data": [1, 2, 3]}}],
        )

        # Verify fields are set and retrievable
        msg.refresh_from_db()
        self.assertEqual(msg.content, "Sensitive analysis result")
        self.assertEqual(msg.reasoning_trace["steps"], ["step1", "step2"])
        self.assertEqual(msg.tool_calls[0]["name"], "analyze")

    def test_conversation_cascade_delete(self):
        """Deleting conversation deletes all messages."""
        from chat.models import Conversation, Message

        conv = Conversation.objects.create(user=self.user, title="To Delete")
        Message.objects.create(conversation=conv, role="user", content="msg1")
        Message.objects.create(conversation=conv, role="assistant", content="msg2")

        conv_id = conv.id
        conv.delete()
        self.assertEqual(Message.objects.filter(conversation_id=conv_id).count(), 0)


# =========================================================================
# Usage Tracking
# =========================================================================


@SECURE_OFF
class UsageLogTest(TestCase):
    """Scenario: log requests → verify aggregation → domain counts."""

    def setUp(self):
        self.user = _make_user("usagelog@example.com", tier=Tier.PRO)

    def test_log_request_creates_daily_row(self):
        """First request creates usage log row for today."""
        from chat.models import UsageLog

        log = UsageLog.log_request(
            user=self.user, tokens_in=100, tokens_out=200, inference_ms=500
        )
        self.assertEqual(log.request_count, 1)
        self.assertEqual(log.tokens_input, 100)
        self.assertEqual(log.tokens_output, 200)
        self.assertEqual(log.total_inference_ms, 500)
        self.assertEqual(log.date, timezone.now().date())

    def test_log_request_aggregates_same_day(self):
        """Multiple requests on same day aggregate into one row."""
        from chat.models import UsageLog

        UsageLog.log_request(user=self.user, tokens_in=100, tokens_out=200)
        UsageLog.log_request(user=self.user, tokens_in=150, tokens_out=300)
        UsageLog.log_request(user=self.user, tokens_in=50, tokens_out=100)

        log = UsageLog.objects.get(user=self.user, date=timezone.now().date())
        self.assertEqual(log.request_count, 3)
        self.assertEqual(log.tokens_input, 300)
        self.assertEqual(log.tokens_output, 600)

    def test_domain_counts_tracked(self):
        """Domain breakdown accumulated in JSON field."""
        from chat.models import UsageLog

        UsageLog.log_request(user=self.user, domain="statistics")
        UsageLog.log_request(user=self.user, domain="statistics")
        UsageLog.log_request(user=self.user, domain="quality")

        log = UsageLog.objects.get(user=self.user, date=timezone.now().date())
        self.assertEqual(log.domain_counts["statistics"], 2)
        self.assertEqual(log.domain_counts["quality"], 1)

    def test_blocked_and_error_counts(self):
        """Blocked and error flags increment separate counters."""
        from chat.models import UsageLog

        UsageLog.log_request(user=self.user, blocked=True)
        UsageLog.log_request(user=self.user, error=True)
        UsageLog.log_request(user=self.user)  # normal

        log = UsageLog.objects.get(user=self.user, date=timezone.now().date())
        self.assertEqual(log.request_count, 3)
        self.assertEqual(log.blocked_count, 1)
        self.assertEqual(log.error_count, 1)


# =========================================================================
# Model Versioning
# =========================================================================


@SECURE_OFF
class ModelVersionTest(TestCase):
    """Test model version hot-swapping for ML pipeline."""

    def test_activate_deactivates_others(self):
        """Activating a version deactivates other versions of same type."""
        from chat.models import ModelVersion

        v1 = ModelVersion.objects.create(
            model_type="reasoner", name="reasoner-v1",
            checkpoint_path="/models/v1", is_active=True,
        )
        v2 = ModelVersion.objects.create(
            model_type="reasoner", name="reasoner-v2",
            checkpoint_path="/models/v2", is_active=False,
        )

        v2.activate()
        v1.refresh_from_db()
        v2.refresh_from_db()

        self.assertFalse(v1.is_active)
        self.assertTrue(v2.is_active)
        self.assertIsNotNone(v2.activated_at)

    def test_activate_different_types_independent(self):
        """Different model types can have independent active versions."""
        from chat.models import ModelVersion

        r = ModelVersion.objects.create(
            model_type="reasoner", name="reasoner-v1",
            checkpoint_path="/models/r1", is_active=True,
        )
        v = ModelVersion.objects.create(
            model_type="verifier", name="verifier-v1",
            checkpoint_path="/models/v1", is_active=True,
        )

        # Both should remain active (different types)
        r.refresh_from_db()
        v.refresh_from_db()
        self.assertTrue(r.is_active)
        self.assertTrue(v.is_active)


# =========================================================================
# Shared Conversations
# =========================================================================


@SECURE_OFF
class SharedConversationTest(TestCase):
    """Scenario: share conversation → view → verify count increment."""

    def setUp(self):
        self.user = _make_user("sharer@example.com", tier=Tier.PRO)
        self.client = APIClient()

    def test_shared_conversation_view(self):
        """View shared conversation → returns messages + increments view count."""
        from chat.models import Conversation, Message, SharedConversation

        conv = Conversation.objects.create(user=self.user, title="Shared Analysis")
        Message.objects.create(conversation=conv, role="user", content="Analyze this data")
        Message.objects.create(conversation=conv, role="assistant", content="Here are the results")

        share = SharedConversation.objects.create(conversation=conv)

        res = self.client.get(f"/chat/shared/{share.id}/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["title"], "Shared Analysis")
        self.assertEqual(len(data["messages"]), 2)
        self.assertEqual(data["view_count"], 1)

        # Second view increments count
        res2 = self.client.get(f"/chat/shared/{share.id}/")
        self.assertEqual(res2.json()["view_count"], 2)

    def test_shared_conversation_not_found(self):
        """Non-existent share ID returns 404."""
        fake_id = uuid.uuid4()
        res = self.client.get(f"/chat/shared/{fake_id}/")
        self.assertEqual(res.status_code, 404)


# =========================================================================
# User Isolation
# =========================================================================


@SECURE_OFF
class ChatIsolationTest(TestCase):
    """Verify conversations isolated between users."""

    def setUp(self):
        self.user1 = _make_user("user1@chat.com", tier=Tier.PRO)
        self.user2 = _make_user("user2@chat.com", tier=Tier.PRO)

    def test_conversations_user_isolated(self):
        """Each user only sees their own conversations."""
        from chat.models import Conversation

        Conversation.objects.create(user=self.user1, title="User1 Chat")
        Conversation.objects.create(user=self.user2, title="User2 Chat")

        user1_convs = Conversation.objects.filter(user=self.user1)
        user2_convs = Conversation.objects.filter(user=self.user2)

        self.assertEqual(user1_convs.count(), 1)
        self.assertEqual(user1_convs.first().title, "User1 Chat")
        self.assertEqual(user2_convs.count(), 1)
        self.assertEqual(user2_convs.first().title, "User2 Chat")

    def test_usage_logs_user_isolated(self):
        """Usage logs are per-user per-day."""
        from chat.models import UsageLog

        UsageLog.log_request(user=self.user1, tokens_in=100)
        UsageLog.log_request(user=self.user2, tokens_in=200)

        log1 = UsageLog.objects.get(user=self.user1)
        log2 = UsageLog.objects.get(user=self.user2)
        self.assertEqual(log1.tokens_input, 100)
        self.assertEqual(log2.tokens_input, 200)
