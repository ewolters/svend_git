"""
SVEND Flywheel - Continuous improvement loop.

Architecture:
    User Query → Synara (fast, free)
                    ↓
            [low confidence?] → Opus API (accurate, paid)
                    ↓
            Log everything → Training candidates
                    ↓
            Analyze patterns → New tools / priors

The flywheel turns failures into improvements.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum

from django.conf import settings

logger = logging.getLogger(__name__)


class EscalationReason(Enum):
    """Why we escalated to Opus."""
    LOW_CONFIDENCE = "low_confidence"
    SYNARA_FAILED = "synara_failed"
    VERIFICATION_FAILED = "verification_failed"
    COMPLEX_QUERY = "complex_query"
    USER_REQUESTED = "user_requested"


@dataclass
class FlywheelResult:
    """Result from flywheel processing."""
    # Source
    used_synara: bool = True
    used_opus: bool = False
    escalation_reason: Optional[EscalationReason] = None

    # Synara result
    synara_success: bool = False
    synara_confidence: float = 0.0
    synara_answer: Any = None
    synara_trace: List[Dict] = field(default_factory=list)

    # Opus result (if escalated)
    opus_response: Optional[str] = None
    opus_answer: Any = None

    # Final output
    final_answer: Any = None
    final_response: str = ""
    final_trace: List[Dict] = field(default_factory=list)

    # Timing
    synara_time_ms: int = 0
    opus_time_ms: int = 0
    total_time_ms: int = 0

    # Logging
    logged: bool = False
    training_candidate: bool = False


class OpusClient:
    """Client for Claude Opus API escalation."""

    def __init__(self):
        self.api_key = getattr(settings, 'ANTHROPIC_API_KEY', None)
        self._client = None

    def _get_client(self):
        """Lazy load Anthropic client."""
        if self._client is None and self.api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                logger.warning("anthropic package not installed")
            except Exception as e:
                logger.error(f"Failed to create Anthropic client: {e}")
        return self._client

    def is_available(self) -> bool:
        """Check if Opus is available."""
        return self.api_key is not None and self._get_client() is not None

    def solve(
        self,
        query: str,
        synara_trace: List[Dict] = None,
        domain: str = "",
    ) -> tuple:
        """
        Send query to Opus for solving.

        Returns (success, answer, response, time_ms)
        """
        client = self._get_client()
        if not client:
            return False, None, "Opus not available", 0

        start = time.time()

        # Build prompt with context
        system = """You are a precise STEM problem solver.
Give accurate, verified answers. Be concise.
Format: State the answer clearly, then briefly explain if needed."""

        # Include Synara's attempt if available
        user_content = f"Problem: {query}"
        if synara_trace:
            trace_summary = "\n".join([
                f"- {s.get('tool', '?')}: {s.get('expression', '?')} → {s.get('result', '?')}"
                for s in synara_trace[:3]
            ])
            user_content += f"\n\nPrevious attempt (may be wrong):\n{trace_summary}"
            user_content += "\n\nPlease verify and provide the correct answer."

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",  # Use Sonnet for cost, Opus for accuracy
                max_tokens=500,
                system=system,
                messages=[{"role": "user", "content": user_content}]
            )

            time_ms = int((time.time() - start) * 1000)
            answer_text = response.content[0].text

            # Try to extract just the answer
            answer = self._extract_answer(answer_text, domain)

            return True, answer, answer_text, time_ms

        except Exception as e:
            logger.error(f"Opus API error: {e}")
            time_ms = int((time.time() - start) * 1000)
            return False, None, str(e), time_ms

    def _extract_answer(self, response: str, domain: str) -> Any:
        """Extract the answer from Opus response."""
        import re

        # Look for common answer patterns
        patterns = [
            r'(?:answer|result|solution)(?:\s+is)?[:\s]+([^\n.]+)',
            r'=\s*([^\n.]+)',
            r'\*\*([^*]+)\*\*',  # Bold markdown
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Return first line as fallback
        return response.split('\n')[0].strip()


class FlywheelService:
    """
    The SVEND flywheel - continuous improvement through logging and escalation.

    Flow:
    1. Try Synara first (fast, free)
    2. If low confidence/failure → escalate to Opus
    3. Log everything for training
    4. Analyze patterns for new tools
    """

    # Thresholds for escalation
    CONFIDENCE_THRESHOLD = 0.7  # Below this → consider escalation
    AUTO_ESCALATE_THRESHOLD = 0.4  # Below this → auto-escalate

    def __init__(self):
        self.opus = OpusClient()

        # Stats
        self.stats = {
            'total_queries': 0,
            'synara_success': 0,
            'synara_low_confidence': 0,
            'opus_escalations': 0,
            'opus_success': 0,
            'training_candidates': 0,
        }

    def process(
        self,
        query: str,
        synara_result: Any,  # AlphaPipelineOutput
        user_id: str = None,
        allow_escalation: bool = True,
    ) -> FlywheelResult:
        """
        Process a query through the flywheel.

        Args:
            query: User's input
            synara_result: Result from Synara pipeline
            user_id: For logging
            allow_escalation: Whether to allow Opus escalation
        """
        self.stats['total_queries'] += 1
        result = FlywheelResult()
        start_time = time.time()

        # Extract Synara results
        result.synara_success = synara_result.synara_success
        result.synara_answer = synara_result.final_answer
        result.synara_trace = synara_result.reasoning_trace or []
        result.synara_confidence = self._get_synara_confidence(synara_result)
        result.synara_time_ms = getattr(synara_result, 'inference_time_ms', 0)

        # Decide if we need escalation
        needs_escalation = False
        escalation_reason = None

        if synara_result.blocked:
            # Don't escalate blocked queries
            result.final_response = synara_result.response
            result.total_time_ms = int((time.time() - start_time) * 1000)
            return result

        if not result.synara_success:
            needs_escalation = True
            escalation_reason = EscalationReason.SYNARA_FAILED
            self.stats['synara_low_confidence'] += 1

        elif result.synara_confidence < self.AUTO_ESCALATE_THRESHOLD:
            needs_escalation = True
            escalation_reason = EscalationReason.LOW_CONFIDENCE
            self.stats['synara_low_confidence'] += 1

        elif result.synara_confidence < self.CONFIDENCE_THRESHOLD:
            # Log as training candidate but don't escalate
            result.training_candidate = True

        else:
            self.stats['synara_success'] += 1

        # Escalate to Opus if needed
        if needs_escalation and allow_escalation and self.opus.is_available():
            result.used_opus = True
            result.escalation_reason = escalation_reason
            self.stats['opus_escalations'] += 1

            success, answer, response, time_ms = self.opus.solve(
                query=query,
                synara_trace=result.synara_trace,
                domain=synara_result.domain or "",
            )

            result.opus_response = response
            result.opus_answer = answer
            result.opus_time_ms = time_ms

            if success:
                self.stats['opus_success'] += 1
                result.final_answer = answer
                result.final_response = response
                result.training_candidate = True  # Log for training
            else:
                # Opus also failed - use Synara result anyway
                result.final_answer = result.synara_answer
                result.final_response = synara_result.response or "Could not solve."
        else:
            # Use Synara result
            result.final_answer = result.synara_answer
            result.final_response = synara_result.response
            result.final_trace = result.synara_trace

        result.total_time_ms = int((time.time() - start_time) * 1000)

        # Log if it's a training candidate
        if result.training_candidate:
            self._log_training_candidate(query, result, synara_result, user_id)
            self.stats['training_candidates'] += 1

        return result

    def _get_synara_confidence(self, synara_result) -> float:
        """Extract confidence from Synara result."""
        # Check trace for confidence
        trace = synara_result.reasoning_trace or []
        if trace:
            confidences = [s.get('confidence', 0.5) for s in trace if 'confidence' in s]
            if confidences:
                return min(confidences)  # Use minimum confidence

        # Check verification confidence
        if synara_result.verification_confidence:
            return synara_result.verification_confidence

        # Default based on success
        return 0.8 if synara_result.synara_success else 0.2

    def _log_training_candidate(
        self,
        query: str,
        result: FlywheelResult,
        synara_result: Any,
        user_id: str = None,
    ):
        """Log as training candidate in database."""
        try:
            from chat.models import TrainingCandidate, TraceLog

            # Determine candidate type
            if not result.synara_success:
                candidate_type = TrainingCandidate.CandidateType.ERROR
            elif result.synara_confidence < self.AUTO_ESCALATE_THRESHOLD:
                candidate_type = TrainingCandidate.CandidateType.LOW_CONFIDENCE
            elif result.used_opus:
                candidate_type = TrainingCandidate.CandidateType.VERIFICATION_FAILED
            else:
                candidate_type = TrainingCandidate.CandidateType.LOW_CONFIDENCE

            # Create training candidate
            candidate = TrainingCandidate.objects.create(
                candidate_type=candidate_type,
                input_text=query,
                domain=synara_result.domain or "",
                difficulty=synara_result.difficulty,
                reasoning_trace=result.synara_trace,
                model_response=synara_result.response or "",
                corrected_response=result.opus_response or "",  # Opus as correction
                verification_confidence=result.synara_confidence,
            )

            result.logged = True
            logger.info(f"Logged training candidate: {candidate.id}")

        except Exception as e:
            logger.error(f"Failed to log training candidate: {e}")

    def log_trace(
        self,
        query: str,
        synara_result: Any,
        flywheel_result: FlywheelResult,
        user_id: str = None,
        message_id: str = None,
    ):
        """Log full trace for diagnostics."""
        try:
            from chat.models import TraceLog, Message

            message = None
            if message_id:
                try:
                    message = Message.objects.get(id=message_id)
                except Message.DoesNotExist:
                    pass

            trace = TraceLog.objects.create(
                message=message,
                input_text=query,
                user_id=user_id,
                safety_passed=not synara_result.blocked,
                domain=synara_result.domain or "",
                difficulty=synara_result.difficulty,
                reasoning_trace=flywheel_result.synara_trace,
                tool_calls=synara_result.tool_calls,
                verified=synara_result.verified,
                verification_confidence=synara_result.verification_confidence,
                response=flywheel_result.final_response,
                gate_passed=flywheel_result.synara_success,
                fallback_used=flywheel_result.used_opus,
                total_time_ms=flywheel_result.total_time_ms,
            )

            logger.debug(f"Logged trace: {trace.id}")

        except Exception as e:
            logger.error(f"Failed to log trace: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get flywheel statistics."""
        total = max(1, self.stats['total_queries'])
        return {
            **self.stats,
            'synara_success_rate': self.stats['synara_success'] / total,
            'escalation_rate': self.stats['opus_escalations'] / total,
            'opus_success_rate': (
                self.stats['opus_success'] / max(1, self.stats['opus_escalations'])
            ),
        }

    def analyze_patterns(self) -> Dict[str, Any]:
        """
        Analyze logged failures to identify patterns for new tools.

        Returns suggestions for:
        - New tools to add
        - Priors to update
        - Domains needing attention
        """
        try:
            from chat.models import TrainingCandidate
            from django.db.models import Count

            # Get recent failures by domain
            domain_failures = TrainingCandidate.objects.filter(
                candidate_type__in=['error', 'low_confidence'],
            ).values('domain').annotate(
                count=Count('id')
            ).order_by('-count')[:10]

            # Get common error patterns
            error_patterns = TrainingCandidate.objects.filter(
                candidate_type='error',
            ).values('error_type').annotate(
                count=Count('id')
            ).order_by('-count')[:10]

            return {
                'domain_failures': list(domain_failures),
                'error_patterns': list(error_patterns),
                'suggestions': self._generate_suggestions(domain_failures, error_patterns),
            }

        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            return {'error': str(e)}

    def _generate_suggestions(self, domain_failures, error_patterns) -> List[str]:
        """Generate suggestions based on failure patterns."""
        suggestions = []

        for df in domain_failures[:3]:
            domain = df.get('domain', 'unknown')
            count = df.get('count', 0)
            if count > 10:
                suggestions.append(f"Consider adding more tools for '{domain}' domain ({count} failures)")

        return suggestions


# Singleton instance
_flywheel: Optional[FlywheelService] = None


def get_flywheel() -> FlywheelService:
    """Get the flywheel service singleton."""
    global _flywheel
    if _flywheel is None:
        _flywheel = FlywheelService()
    return _flywheel
