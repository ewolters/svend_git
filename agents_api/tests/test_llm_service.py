"""
LLMService behavioral tests.

Standard:     LLM-001 §3 (LLM Integration)
Compliance:   TST-001 §4

<!-- test: agents_api.tests.test_llm_service.LLMServiceChatTests -->
<!-- test: agents_api.tests.test_llm_service.LLMServiceTemperatureTests -->
<!-- test: agents_api.tests.test_llm_service.LLMServiceModelTests -->
<!-- test: agents_api.tests.test_llm_service.LLMResultTests -->

CR: 2096abb9
"""

from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from accounts.constants import Tier
from agents_api.llm_service import LLMResult, llm_service

User = get_user_model()

SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)


def _user(email="llm@test.com", tier=Tier.PRO):
    username = email.split("@")[0]
    u = User.objects.create_user(username=username, email=email, password="testpass123!")
    u.tier = tier
    u.save(update_fields=["tier"])
    return u


def _success_response(content="Analysis complete", model="claude-sonnet-4-20250514"):
    return {
        "content": content,
        "model": model,
        "usage": {"input_tokens": 100, "output_tokens": 50},
        "stop_reason": "end_turn",
    }


def _rate_limited_response():
    return {
        "error": "Daily limit of 50 requests reached. Resets at midnight.",
        "rate_limited": True,
        "rate_limit": {"remaining": 0, "limit": 50},
    }


# ---------------------------------------------------------------------------
# Chat behavior
# ---------------------------------------------------------------------------
@SECURE_OFF
class LLMServiceChatTests(TestCase):
    """Core chat() call paths: success, rate-limited, total failure."""

    def setUp(self):
        self.user = _user()

    @patch("agents_api.llm_service.LLMManager")
    def test_successful_call(self, mock_manager):
        mock_manager.chat.return_value = _success_response()
        mock_manager.get_model_for_user.return_value = "claude-sonnet-4-20250514"

        result = llm_service.chat(self.user, "test prompt")

        self.assertTrue(result.success)
        self.assertEqual(result.content, "Analysis complete")
        self.assertEqual(result.model, "claude-sonnet-4-20250514")
        self.assertFalse(result.rate_limited)
        self.assertEqual(result.error, "")
        self.assertEqual(result.input_tokens, 100)
        self.assertEqual(result.output_tokens, 50)

    @patch("agents_api.llm_service.LLMManager")
    def test_rate_limited_response(self, mock_manager):
        mock_manager.chat.return_value = _rate_limited_response()
        mock_manager.get_model_for_user.return_value = "claude-sonnet-4-20250514"

        result = llm_service.chat(self.user, "test prompt")

        self.assertFalse(result.success)
        self.assertTrue(result.rate_limited)
        self.assertIn("limit", result.error.lower())
        self.assertEqual(result.input_tokens, 0)

    @patch("agents_api.llm_service.LLMManager")
    def test_none_response_total_failure(self, mock_manager):
        mock_manager.chat.return_value = None
        mock_manager.get_model_for_user.return_value = "claude-sonnet-4-20250514"

        result = llm_service.chat(self.user, "test prompt")

        self.assertFalse(result.success)
        self.assertFalse(result.rate_limited)
        self.assertEqual(result.content, "")
        self.assertNotEqual(result.error, "")

    @patch("agents_api.llm_service.LLMManager")
    def test_prompt_wrapped_as_user_message(self, mock_manager):
        mock_manager.chat.return_value = _success_response()
        mock_manager.get_model_for_user.return_value = "claude-sonnet-4-20250514"

        llm_service.chat(self.user, "What is SPC?")

        args, kwargs = mock_manager.chat.call_args
        messages_arg = args[1]
        self.assertEqual(len(messages_arg), 1)
        self.assertEqual(messages_arg[0]["role"], "user")
        self.assertEqual(messages_arg[0]["content"], "What is SPC?")

    @patch("agents_api.llm_service.LLMManager")
    def test_multi_turn_messages_passthrough(self, mock_manager):
        mock_manager.chat.return_value = _success_response()
        mock_manager.get_model_for_user.return_value = "claude-sonnet-4-20250514"

        multi = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Explain FMEA"},
        ]
        llm_service.chat(self.user, messages=multi)

        args, kwargs = mock_manager.chat.call_args
        messages_arg = args[1]
        self.assertEqual(messages_arg, multi)
        self.assertEqual(len(messages_arg), 3)

    @patch("agents_api.llm_service.LLMManager")
    def test_system_prompt_forwarded(self, mock_manager):
        mock_manager.chat.return_value = _success_response()
        mock_manager.get_model_for_user.return_value = "claude-sonnet-4-20250514"

        llm_service.chat(self.user, "test", system="You are a quality engineer.")

        _, kwargs = mock_manager.chat.call_args
        self.assertEqual(kwargs["system"], "You are a quality engineer.")

    @patch("agents_api.llm_service.LLMManager")
    def test_exception_returns_failure_result(self, mock_manager):
        mock_manager.chat.side_effect = RuntimeError("connection timeout")
        mock_manager.get_model_for_user.return_value = "claude-sonnet-4-20250514"

        result = llm_service.chat(self.user, "test")

        self.assertFalse(result.success)
        self.assertIn("connection timeout", result.error)
        self.assertIsInstance(result, LLMResult)


# ---------------------------------------------------------------------------
# Temperature resolution
# ---------------------------------------------------------------------------
@SECURE_OFF
class LLMServiceTemperatureTests(TestCase):
    """Context-based temperature defaults and explicit overrides."""

    def setUp(self):
        self.user = _user(email="temp@test.com")

    @patch("agents_api.llm_service.LLMManager")
    def test_analysis_context_uses_low_temperature(self, mock_manager):
        mock_manager.chat.return_value = _success_response()
        mock_manager.get_model_for_user.return_value = "claude-sonnet-4-20250514"

        llm_service.chat(self.user, "test", context="analysis")

        _, kwargs = mock_manager.chat.call_args
        self.assertAlmostEqual(kwargs["temperature"], 0.3)

    @patch("agents_api.llm_service.LLMManager")
    def test_critique_context_temperature(self, mock_manager):
        mock_manager.chat.return_value = _success_response()
        mock_manager.get_model_for_user.return_value = "claude-sonnet-4-20250514"

        llm_service.chat(self.user, "test", context="critique")

        _, kwargs = mock_manager.chat.call_args
        self.assertAlmostEqual(kwargs["temperature"], 0.5)

    @patch("agents_api.llm_service.LLMManager")
    def test_chat_context_temperature(self, mock_manager):
        mock_manager.chat.return_value = _success_response()
        mock_manager.get_model_for_user.return_value = "claude-sonnet-4-20250514"

        llm_service.chat(self.user, "test", context="chat")

        _, kwargs = mock_manager.chat.call_args
        self.assertAlmostEqual(kwargs["temperature"], 0.7)

    @patch("agents_api.llm_service.LLMManager")
    def test_generation_context_temperature(self, mock_manager):
        mock_manager.chat.return_value = _success_response()
        mock_manager.get_model_for_user.return_value = "claude-sonnet-4-20250514"

        llm_service.chat(self.user, "test", context="generation")

        _, kwargs = mock_manager.chat.call_args
        self.assertAlmostEqual(kwargs["temperature"], 0.7)

    @patch("agents_api.llm_service.LLMManager")
    def test_explicit_temperature_overrides_context(self, mock_manager):
        mock_manager.chat.return_value = _success_response()
        mock_manager.get_model_for_user.return_value = "claude-sonnet-4-20250514"

        llm_service.chat(self.user, "test", context="analysis", temperature=0.9)

        _, kwargs = mock_manager.chat.call_args
        self.assertAlmostEqual(kwargs["temperature"], 0.9)

    @patch("agents_api.llm_service.LLMManager")
    def test_unknown_context_falls_back_to_default(self, mock_manager):
        mock_manager.chat.return_value = _success_response()
        mock_manager.get_model_for_user.return_value = "claude-sonnet-4-20250514"

        llm_service.chat(self.user, "test", context="unknown_context")

        _, kwargs = mock_manager.chat.call_args
        self.assertAlmostEqual(kwargs["temperature"], 0.7)


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
@SECURE_OFF
class LLMServiceModelTests(TestCase):
    """Tier-based model selection delegation."""

    @patch("agents_api.llm_service.LLMManager")
    def test_delegates_to_llm_manager(self, mock_manager):
        mock_manager.get_model_for_user.return_value = "claude-sonnet-4-20250514"
        user = _user(email="del@test.com", tier=Tier.PRO)

        model = llm_service.get_model_for_user(user)

        mock_manager.get_model_for_user.assert_called_once_with(user)
        self.assertEqual(model, "claude-sonnet-4-20250514")

    @patch("agents_api.llm_service.LLMManager")
    def test_pro_tier_gets_sonnet(self, mock_manager):
        mock_manager.get_model_for_user.return_value = "claude-sonnet-4-20250514"
        user = _user(email="pro@test.com", tier=Tier.PRO)
        model = llm_service.get_model_for_user(user)
        self.assertIn("sonnet", model)

    @patch("agents_api.llm_service.LLMManager")
    def test_free_tier_gets_haiku(self, mock_manager):
        mock_manager.get_model_for_user.return_value = "claude-3-5-haiku-20241022"
        user = _user(email="free@test.com", tier=Tier.FREE)
        model = llm_service.get_model_for_user(user)
        self.assertIn("haiku", model)

    @patch("agents_api.llm_service.LLMManager")
    def test_enterprise_tier_gets_opus(self, mock_manager):
        mock_manager.get_model_for_user.return_value = "claude-opus-4-20250514"
        user = _user(email="ent@test.com", tier=Tier.ENTERPRISE)
        model = llm_service.get_model_for_user(user)
        self.assertIn("opus", model)


# ---------------------------------------------------------------------------
# LLMResult dataclass
# ---------------------------------------------------------------------------
@SECURE_OFF
class LLMResultTests(TestCase):
    """LLMResult field access and state representation."""

    def test_success_state_fields(self):
        r = LLMResult(
            content="Hello",
            model="claude-sonnet-4-20250514",
            success=True,
            rate_limited=False,
            error="",
            input_tokens=10,
            output_tokens=5,
        )
        self.assertTrue(r.success)
        self.assertEqual(r.content, "Hello")
        self.assertEqual(r.error, "")
        self.assertEqual(r.input_tokens, 10)
        self.assertEqual(r.output_tokens, 5)

    def test_failure_state_fields(self):
        r = LLMResult(
            content="",
            model="claude-3-5-haiku-20241022",
            success=False,
            rate_limited=False,
            error="LLM request failed",
            input_tokens=0,
            output_tokens=0,
        )
        self.assertFalse(r.success)
        self.assertEqual(r.content, "")
        self.assertNotEqual(r.error, "")

    def test_rate_limited_state_fields(self):
        r = LLMResult(
            content="",
            model="claude-sonnet-4-20250514",
            success=False,
            rate_limited=True,
            error="Rate limit exceeded",
            input_tokens=0,
            output_tokens=0,
        )
        self.assertFalse(r.success)
        self.assertTrue(r.rate_limited)

    def test_all_fields_accessible(self):
        r = LLMResult(
            content="x",
            model="m",
            success=True,
            rate_limited=False,
            error="",
            input_tokens=1,
            output_tokens=2,
        )
        # Verify all 7 fields exist and are the expected types
        self.assertIsInstance(r.content, str)
        self.assertIsInstance(r.model, str)
        self.assertIsInstance(r.success, bool)
        self.assertIsInstance(r.rate_limited, bool)
        self.assertIsInstance(r.error, str)
        self.assertIsInstance(r.input_tokens, int)
        self.assertIsInstance(r.output_tokens, int)
