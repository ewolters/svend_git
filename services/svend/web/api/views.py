"""API views."""

import logging
import random
from django.views.decorators.csrf import csrf_exempt
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.throttling import AnonRateThrottle
from rest_framework.permissions import IsAuthenticated, AllowAny, IsAdminUser
from rest_framework.response import Response

from accounts.constants import (
    TIER_FEATURES, Tier, Industry, Role, ExperienceLevel, OrganizationSize,
)
from chat.models import Conversation, Message, SharedConversation, TraceLog, TrainingCandidate
from inference import process_query
from inference.flywheel import get_flywheel, FlywheelResult

from .serializers import (
    ConversationSerializer,
    ConversationListSerializer,
    MessageSerializer,
    ChatInputSerializer,
    ShareSerializer,
)

logger = logging.getLogger(__name__)

# Model mappings for enterprise users
ENTERPRISE_MODELS = {
    "opus": "claude-opus-4-20250514",
    "sonnet": "claude-sonnet-4-20250514",
    "haiku": "claude-3-5-haiku-20241022",
    "qwen": "qwen",  # Placeholder - would need separate integration
}


class EnterpriseModelResult:
    """Simple result container for enterprise model calls."""
    def __init__(self, response: str, inference_time_ms: int = 0):
        self.response = response
        self.inference_time_ms = inference_time_ms


def call_enterprise_model(query: str, model: str, conversation) -> EnterpriseModelResult:
    """Call a specific model for enterprise users."""
    import time
    from django.conf import settings

    start = time.time()

    # Get conversation history for context
    messages = []
    for msg in conversation.messages.order_by('created_at')[:20]:  # Last 20 messages
        messages.append({
            "role": "user" if msg.role == "user" else "assistant",
            "content": msg.content
        })

    # Add current query
    messages.append({"role": "user", "content": query})

    if model == "qwen":
        # Use local Qwen via cognition pipeline
        result = process_query(query, mode="auto")
        return EnterpriseModelResult(
            response=result.response,
            inference_time_ms=int(result.inference_time_ms)
        )

    # Call Anthropic API
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

        response = client.messages.create(
            model=ENTERPRISE_MODELS.get(model, "claude-sonnet-4-20250514"),
            max_tokens=4096,
            system="You are SVEND, a helpful AI assistant specializing in reasoning, problem-solving, and data analysis.",
            messages=messages
        )

        time_ms = int((time.time() - start) * 1000)
        return EnterpriseModelResult(
            response=response.content[0].text,
            inference_time_ms=time_ms
        )

    except Exception as e:
        logger.error(f"Enterprise model call failed: {e}")
        time_ms = int((time.time() - start) * 1000)
        return EnterpriseModelResult(
            response=f"Model call failed: {str(e)}",
            inference_time_ms=time_ms
        )


# Correctness gate thresholds
VERIFICATION_CONFIDENCE_THRESHOLD = 0.3  # Below this = low confidence
RANDOM_SAMPLE_RATE = 0.05  # 5% random sampling for training data


@api_view(["GET"])
@permission_classes([AllowAny])
def health(request):
    """Health check endpoint."""
    return Response({"status": "ok", "service": "svend"})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def conversations(request):
    """List user's conversations."""
    convos = Conversation.objects.filter(user=request.user)
    serializer = ConversationListSerializer(convos, many=True)
    return Response(serializer.data)


@api_view(["GET", "DELETE"])
@permission_classes([IsAuthenticated])
def conversation_detail(request, conversation_id):
    """Get or delete a specific conversation."""
    try:
        convo = Conversation.objects.get(id=conversation_id, user=request.user)
    except Conversation.DoesNotExist:
        return Response(
            {"error": "Conversation not found"},
            status=status.HTTP_404_NOT_FOUND,
        )

    if request.method == "DELETE":
        convo.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

    serializer = ConversationSerializer(convo)
    return Response(serializer.data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def chat(request):
    """Send a message and get a response.

    Includes:
    - Full trace logging for diagnostics
    - Fail-closed correctness gate
    - Training candidate collection
    """
    serializer = ChatInputSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    user = request.user
    message_text = serializer.validated_data["message"]
    conversation_id = serializer.validated_data.get("conversation_id")
    selected_model = serializer.validated_data.get("model", "default")

    # Model selection is enterprise-only
    if selected_model != "default" and user.tier != "enterprise":
        selected_model = "default"  # Silently reset for non-enterprise

    # Check email verification
    if user.email and not user.email_verified:
        return Response(
            {
                "error": "Please verify your email before using SVEND",
                "email": user.email,
                "action": "verify_email",
            },
            status=status.HTTP_403_FORBIDDEN,
        )

    # Check rate limit
    if not user.can_query():
        return Response(
            {
                "error": "Daily query limit reached",
                "limit": user.daily_limit,
                "tier": user.tier,
            },
            status=status.HTTP_429_TOO_MANY_REQUESTS,
        )

    # Get or create conversation
    if conversation_id:
        try:
            conversation = Conversation.objects.get(id=conversation_id, user=user)
        except Conversation.DoesNotExist:
            return Response(
                {"error": "Conversation not found"},
                status=status.HTTP_404_NOT_FOUND,
            )
    else:
        conversation = Conversation.objects.create(user=user)

    # Save user message
    user_message = Message.objects.create(
        conversation=conversation,
        role=Message.Role.USER,
        content=message_text,
    )

    # Initialize trace log
    trace_log = TraceLog(
        input_text=message_text,
        user_id=user.id,
    )

    # Process through pipeline with flywheel
    gate_passed = True
    gate_reason = ""
    fallback_used = False
    final_response = ""
    flywheel_result = None
    visualizations = []

    # Get mode preference from request (default: auto)
    mode = serializer.validated_data.get("mode", "auto")

    try:
        # Enterprise model selection - direct API call
        if selected_model != "default" and user.tier == "enterprise":
            result = call_enterprise_model(message_text, selected_model, conversation)
            trace_log.domain = f"enterprise:{selected_model}"
            trace_log.response = result.response
            trace_log.total_time_ms = result.inference_time_ms
            trace_log.gate_passed = True
            trace_log.gate_reason = f"Enterprise model: {selected_model}"
            gate_passed = True
            final_response = result.response

            # Save trace and response
            trace_log.message = user_message
            trace_log.save()

            assistant_message = Message.objects.create(
                conversation=conversation,
                role=Message.Role.ASSISTANT,
                content=final_response,
            )

            user.increment_queries()

            return Response({
                "response": final_response,
                "conversation_id": str(conversation.id),
                "message_id": str(assistant_message.id),
                "model": selected_model,
                "trace_id": str(trace_log.id) if trace_log.id else None,
            })

        # Get result from appropriate pipeline
        result = process_query(message_text, mode=mode)

        # Capture visualizations from coder mode
        if hasattr(result, 'visualizations') and result.visualizations:
            visualizations = result.visualizations

        # Handle cognition pipeline (all modes including EXECUTIVE)
        if result.pipeline_type == "cognition":
            trace_log.safety_passed = not result.blocked
            trace_log.domain = result.selected_mode or result.domain or "cognition"
            trace_log.difficulty = result.difficulty
            trace_log.reasoning_trace = result.reasoning_trace
            trace_log.tool_calls = result.tool_calls
            trace_log.verified = result.verified
            trace_log.verification_confidence = result.verification_confidence
            trace_log.response = result.response
            trace_log.total_time_ms = result.inference_time_ms

            if result.blocked:
                gate_passed = False
                gate_reason = f"Pipeline error: {result.block_reason}"
                final_response = "I couldn't process that request."
            elif result.success or result.response:
                gate_passed = True
                gate_reason = f"Cognition mode: {result.selected_mode}"
                final_response = result.response
            else:
                gate_passed = False
                gate_reason = "No response generated"
                final_response = result.response or "Could not generate response."

            trace_log.gate_passed = gate_passed
            trace_log.gate_reason = gate_reason

        # Legacy coder mode handling
        elif result.pipeline_type == "coder":
            trace_log.safety_passed = not result.blocked
            trace_log.domain = result.domain or "computation"
            trace_log.difficulty = result.difficulty
            trace_log.reasoning_trace = result.reasoning_trace
            trace_log.tool_calls = result.tool_calls
            trace_log.verified = result.verified
            trace_log.verification_confidence = result.verification_confidence
            trace_log.response = result.response
            trace_log.total_time_ms = result.inference_time_ms

            if result.blocked:
                gate_passed = False
                gate_reason = f"Coder error: {result.block_reason}"
                final_response = "I couldn't execute that code."
            elif result.verified:
                gate_passed = True
                gate_reason = "Code executed successfully"
                final_response = result.response
            else:
                gate_passed = False
                gate_reason = "Code execution failed"
                final_response = result.response or "Code execution failed."

            trace_log.gate_passed = gate_passed
            trace_log.gate_reason = gate_reason

        else:
            # Synara mode - process through flywheel (may escalate to Opus)
            flywheel = get_flywheel()
            flywheel_result = flywheel.process(
                query=message_text,
                synara_result=result,
                user_id=str(user.id),
                allow_escalation=False,  # Disabled until cost controls added
            )

            # Update trace log with results
            trace_log.safety_passed = not result.blocked
            trace_log.domain = result.domain or ""
            trace_log.difficulty = result.difficulty
            trace_log.reasoning_trace = flywheel_result.final_trace or result.reasoning_trace
            trace_log.tool_calls = result.tool_calls
            trace_log.verified = result.verified
            trace_log.verification_confidence = result.verification_confidence
            trace_log.response = flywheel_result.final_response
            trace_log.total_time_ms = flywheel_result.total_time_ms or result.inference_time_ms

            # === FAIL-CLOSED CORRECTNESS GATE ===
            if result.blocked:
                # Safety blocked - use block reason
                gate_passed = False
                gate_reason = f"Safety blocked: {result.block_reason}"
                final_response = "I can't help with that request."

            elif flywheel_result.used_opus:
                # Opus was used - flywheel handled it
                gate_passed = flywheel_result.synara_success or bool(flywheel_result.opus_response)
                gate_reason = f"Escalated to Opus: {flywheel_result.escalation_reason.value if flywheel_result.escalation_reason else 'unknown'}"
                fallback_used = True
                final_response = flywheel_result.final_response

            elif not flywheel_result.synara_success:
                # Synara failed and no Opus available
                gate_passed = False
                gate_reason = "Synara failed, no escalation available"
                fallback_used = True
                final_response = "Could not solve. Try rephrasing."

            elif flywheel_result.synara_confidence < VERIFICATION_CONFIDENCE_THRESHOLD:
                # Low confidence but Synara succeeded
                gate_passed = True  # Still pass but mark as low confidence
                gate_reason = f"Low confidence: {flywheel_result.synara_confidence:.2f}"
                final_response = flywheel_result.final_response

            elif not flywheel_result.final_response or len(flywheel_result.final_response.strip()) < 1:
                # Empty response
                gate_passed = False
                gate_reason = "Empty response"
                fallback_used = True
                final_response = "Could not generate response."

            else:
                # Gate passed - use flywheel response
                final_response = flywheel_result.final_response

            trace_log.gate_passed = gate_passed
            trace_log.gate_reason = gate_reason
            trace_log.fallback_used = fallback_used

    except Exception as e:
        # Pipeline error - fail closed
        logger.exception(f"Pipeline error for user {user.id}")
        trace_log.error_stage = "pipeline"
        trace_log.error_message = str(e)
        trace_log.gate_passed = False
        trace_log.gate_reason = f"Pipeline error: {type(e).__name__}"

        gate_passed = False
        fallback_used = True
        final_response = "I encountered an error processing your request. Please try again."

        # Create result-like object for the message
        class ErrorResult:
            response = final_response
            domain = ""
            difficulty = None
            verified = None
            verification_confidence = None
            blocked = False
            block_reason = ""
            reasoning_trace = None
            tool_calls = None
            inference_time_ms = None
            formatted_trace = None
        result = ErrorResult()

    # Save assistant response
    assistant_message = Message.objects.create(
        conversation=conversation,
        role=Message.Role.ASSISTANT,
        content=final_response,
        domain=result.domain or "",
        difficulty=result.difficulty,
        verified=result.verified,
        verification_confidence=result.verification_confidence,
        blocked=result.blocked,
        block_reason=result.block_reason or "",
        reasoning_trace=result.reasoning_trace,
        tool_calls=result.tool_calls,
        inference_time_ms=result.inference_time_ms,
    )

    # Save trace log linked to message
    trace_log.message = assistant_message
    trace_log.save()

    # === TRAINING CANDIDATE COLLECTION ===
    should_collect = False
    candidate_type = None

    if not gate_passed:
        if trace_log.error_stage:
            should_collect = True
            candidate_type = TrainingCandidate.CandidateType.ERROR
        elif result.verification_confidence is not None and result.verification_confidence < VERIFICATION_CONFIDENCE_THRESHOLD:
            should_collect = True
            candidate_type = TrainingCandidate.CandidateType.LOW_CONFIDENCE
        elif result.verified is False:
            should_collect = True
            candidate_type = TrainingCandidate.CandidateType.VERIFICATION_FAILED
    elif random.random() < RANDOM_SAMPLE_RATE:
        # Random sampling of successful responses for quality monitoring
        should_collect = True
        candidate_type = TrainingCandidate.CandidateType.RANDOM_SAMPLE

    if should_collect and candidate_type:
        TrainingCandidate.objects.create(
            trace_log=trace_log,
            candidate_type=candidate_type,
            input_text=message_text,
            domain=result.domain or "",
            difficulty=result.difficulty,
            reasoning_trace=result.reasoning_trace,
            model_response=result.response or "",
            verification_confidence=result.verification_confidence,
            error_type=trace_log.error_stage if trace_log.error_stage else "",
        )
        logger.info(f"Created training candidate: {candidate_type} for input: {message_text[:50]}...")

    # Update conversation title if first message
    if conversation.messages.count() == 2:
        conversation.generate_title()

    # Increment user query count
    user.increment_queries()

    # Build response with formatted trace if available
    response_data = {
        "conversation_id": str(conversation.id),
        "user_message": MessageSerializer(user_message).data,
        "assistant_message": MessageSerializer(assistant_message).data,
        "pipeline_type": result.pipeline_type if hasattr(result, 'pipeline_type') else "synara",
    }

    # Include cognition mode info
    if hasattr(result, 'selected_mode') and result.selected_mode:
        response_data["selected_mode"] = result.selected_mode
    if hasattr(result, 'mode_scores') and result.mode_scores:
        response_data["mode_scores"] = result.mode_scores
    if hasattr(result, 'confidence') and result.confidence:
        response_data["confidence"] = result.confidence

    # Include formatted trace for frontend rendering (KaTeX, visualizations)
    if hasattr(result, 'formatted_trace') and result.formatted_trace:
        response_data["formatted_trace"] = result.formatted_trace

    # Include code and visualizations from coder/executive mode
    if hasattr(result, 'code') and result.code:
        response_data["code"] = result.code
    if visualizations:
        response_data["visualizations"] = visualizations  # Base64 PNGs

    # Include execution outputs for executive mode
    if hasattr(result, 'execution_outputs') and result.execution_outputs:
        response_data["execution_outputs"] = result.execution_outputs
    if hasattr(result, 'execution_errors') and result.execution_errors:
        response_data["execution_errors"] = result.execution_errors
    if hasattr(result, 'tools_used') and result.tools_used:
        response_data["tools_used"] = result.tools_used

    # Include flywheel metadata
    if flywheel_result:
        response_data["flywheel"] = {
            "used_opus": flywheel_result.used_opus,
            "synara_confidence": flywheel_result.synara_confidence,
            "escalation_reason": flywheel_result.escalation_reason.value if flywheel_result.escalation_reason else None,
        }

    return Response(response_data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def share_conversation(request):
    """Create a shareable link for a conversation."""
    serializer = ShareSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    try:
        conversation = Conversation.objects.get(
            id=serializer.validated_data["conversation_id"],
            user=request.user,
        )
    except Conversation.DoesNotExist:
        return Response(
            {"error": "Conversation not found"},
            status=status.HTTP_404_NOT_FOUND,
        )

    # Create or get share
    share, created = SharedConversation.objects.get_or_create(
        conversation=conversation,
    )

    return Response({
        "share_id": str(share.id),
        "url": f"/chat/shared/{share.id}/",
    })


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def user_info(request):
    """Get current user info and usage."""
    user = request.user
    return Response({
        "email": user.email,
        "tier": user.tier,
        "queries_today": user.queries_today,
        "daily_limit": user.daily_limit,
        "subscription_active": hasattr(user, 'subscription') and user.subscription.is_active,
    })


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def flag_message(request, message_id):
    """Flag a message as problematic for training data collection."""
    try:
        message = Message.objects.get(id=message_id)
        # Verify user owns this conversation
        if message.conversation.user != request.user:
            return Response(
                {"error": "Not authorized"},
                status=status.HTTP_403_FORBIDDEN,
            )
    except Message.DoesNotExist:
        return Response(
            {"error": "Message not found"},
            status=status.HTTP_404_NOT_FOUND,
        )

    reason = request.data.get("reason", "")

    # Get or create trace log
    trace_log = getattr(message, 'trace_log', None)
    if not trace_log:
        # Create minimal trace log for older messages
        trace_log = TraceLog.objects.create(
            message=message,
            input_text=message.conversation.messages.filter(
                role=Message.Role.USER,
                created_at__lt=message.created_at
            ).last().content if message.role == Message.Role.ASSISTANT else message.content,
            user_id=request.user.id,
            response=message.content,
        )

    # Create training candidate
    candidate, created = TrainingCandidate.objects.get_or_create(
        trace_log=trace_log,
        candidate_type=TrainingCandidate.CandidateType.USER_FLAGGED,
        defaults={
            "input_text": trace_log.input_text,
            "domain": message.domain or "",
            "difficulty": message.difficulty,
            "reasoning_trace": message.reasoning_trace,
            "model_response": message.content,
            "verification_confidence": message.verification_confidence,
            "reviewer_notes": f"User flagged: {reason}",
        }
    )

    if not created:
        # Update existing with additional flag reason
        candidate.reviewer_notes += f"\nAdditional flag: {reason}"
        candidate.save()

    logger.info(f"Message {message_id} flagged by user {request.user.id}: {reason}")

    return Response({"status": "flagged", "candidate_id": str(candidate.id)})


@api_view(["GET"])
@permission_classes([IsAdminUser])
def trace_stats(request):
    """Get trace logging statistics (for monitoring dashboard). Admin only."""
    from django.db.models import Count, Avg
    from django.utils import timezone
    from datetime import timedelta

    # Last 24 hours
    since = timezone.now() - timedelta(hours=24)

    total = TraceLog.objects.filter(created_at__gte=since).count()
    gate_passed = TraceLog.objects.filter(created_at__gte=since, gate_passed=True).count()
    gate_failed = TraceLog.objects.filter(created_at__gte=since, gate_passed=False).count()
    errors = TraceLog.objects.filter(created_at__gte=since).exclude(error_stage="").count()

    avg_time = TraceLog.objects.filter(
        created_at__gte=since,
        total_time_ms__isnull=False
    ).aggregate(avg=Avg('total_time_ms'))['avg']

    # Domain breakdown
    domains = TraceLog.objects.filter(
        created_at__gte=since
    ).values('domain').annotate(count=Count('id')).order_by('-count')[:10]

    # Training candidates
    candidates = TrainingCandidate.objects.filter(created_at__gte=since)
    candidate_counts = {
        "total": candidates.count(),
        "pending": candidates.filter(status=TrainingCandidate.Status.PENDING).count(),
        "by_type": dict(
            candidates.values('candidate_type').annotate(count=Count('id')).values_list('candidate_type', 'count')
        ),
    }

    return Response({
        "period": "24h",
        "total_requests": total,
        "gate_passed": gate_passed,
        "gate_failed": gate_failed,
        "error_count": errors,
        "avg_inference_time_ms": round(avg_time, 1) if avg_time else None,
        "pass_rate": round(gate_passed / total * 100, 1) if total > 0 else 0,
        "domains": list(domains),
        "training_candidates": candidate_counts,
    })


@api_view(["GET"])
@permission_classes([IsAdminUser])
def flywheel_stats(request):
    """Get flywheel statistics (for monitoring). Admin only."""
    flywheel = get_flywheel()
    stats = flywheel.get_stats()

    # Add pattern analysis if requested
    if request.query_params.get('analyze') == 'true':
        stats['patterns'] = flywheel.analyze_patterns()

    return Response(stats)


class LoginRateThrottle(AnonRateThrottle):
    """Rate limit login attempts to prevent brute force."""
    rate = "5/minute"


@api_view(["POST"])
@permission_classes([AllowAny])
@throttle_classes([LoginRateThrottle])
def login(request):
    """Login and create session. Rate limited to 5 attempts/minute."""
    from django.contrib.auth import authenticate, login as auth_login

    username = request.data.get("username", "").strip()
    password = request.data.get("password", "")

    if not username or not password:
        return Response(
            {"error": "Username and password required"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    user = authenticate(request, username=username, password=password)

    if user is None:
        # Try email as username
        from django.contrib.auth import get_user_model
        User = get_user_model()
        try:
            user_by_email = User.objects.get(email=username)
            user = authenticate(request, username=user_by_email.username, password=password)
        except User.DoesNotExist:
            pass

    if user is None:
        logger.warning(f"Failed login attempt for: {username}")
        return Response(
            {"error": "Invalid credentials"},
            status=status.HTTP_401_UNAUTHORIZED,
        )

    if not user.is_active:
        return Response(
            {"error": "Account is disabled"},
            status=status.HTTP_403_FORBIDDEN,
        )

    auth_login(request, user)
    logger.info(f"User logged in: {user.username}")

    return Response({
        "status": "logged_in",
        "user": {
            "id": str(user.id),
            "username": user.username,
            "email": user.email,
            "tier": user.tier,
            "daily_limit": user.daily_limit,
        },
    })


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def logout(request):
    """Logout and destroy session."""
    from django.contrib.auth import logout as auth_logout

    username = request.user.username
    auth_logout(request)
    logger.info(f"User logged out: {username}")

    return Response({"status": "logged_out"})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def me(request):
    """Get current authenticated user details."""
    user = request.user
    return Response({
        "id": str(user.id),
        "username": user.username,
        "email": user.email,
        "email_verified": user.email_verified,
        "tier": user.tier,
        "display_name": user.display_name or user.username,
        "avatar_url": user.avatar_url,
        "bio": user.bio,
        "industry": user.industry,
        "role": user.role,
        "experience_level": user.experience_level,
        "organization_size": user.organization_size,
        "is_staff": user.is_staff,
        "queries_today": user.queries_today,
        "daily_limit": user.daily_limit,
        "total_queries": user.total_queries,
        "subscription_active": hasattr(user, 'subscription') and user.subscription.is_active,
        "preferences": user.preferences or {},
        "current_theme": user.current_theme,
        "features": TIER_FEATURES.get(user.tier, TIER_FEATURES[Tier.FREE]),
    })


@api_view(["PATCH"])
@permission_classes([IsAuthenticated])
def update_profile(request):
    """Update user profile."""
    user = request.user

    # Allowed fields to update
    allowed = [
        "display_name", "avatar_url", "bio", "preferences", "current_theme",
        "industry", "role", "experience_level", "organization_size",
    ]

    # Valid choices for constrained fields
    valid_choices = {
        "industry": [c[0] for c in Industry.choices] + [""],
        "role": [c[0] for c in Role.choices] + [""],
        "experience_level": [c[0] for c in ExperienceLevel.choices] + [""],
        "organization_size": [c[0] for c in OrganizationSize.choices] + [""],
    }

    for field in allowed:
        if field in request.data:
            value = request.data[field]
            if field in valid_choices and value not in valid_choices[field]:
                return Response(
                    {"error": f"Invalid value for {field}"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            setattr(user, field, value)

    user.save()

    return Response({
        "status": "updated",
        "user": {
            "display_name": user.display_name,
            "avatar_url": user.avatar_url,
            "bio": user.bio,
            "preferences": user.preferences,
            "current_theme": user.current_theme,
            "industry": user.industry,
            "role": user.role,
            "experience_level": user.experience_level,
            "organization_size": user.organization_size,
        },
    })


@api_view(["POST"])
@permission_classes([AllowAny])
def register(request):
    """Register a new user.

    Requires invite code when SVEND_REQUIRE_INVITE=true (default for alpha).
    """
    from django.contrib.auth import get_user_model
    from accounts.models import InviteCode
    from svend_config.config import get_settings

    settings = get_settings()
    User = get_user_model()

    username = request.data.get("username", "").strip()
    email = request.data.get("email", "").strip()
    password = request.data.get("password", "")
    invite_code = request.data.get("invite_code", "").strip().upper()
    plan = request.data.get("plan", "").strip().lower()  # founder, pro, team, enterprise

    # Paid plans bypass invite requirement (they're paying customers)
    paid_plans = ["founder", "pro", "team", "enterprise"]
    is_paid_signup = plan in paid_plans

    # Validation
    if not username or len(username) < 3:
        return Response(
            {"error": "Username must be at least 3 characters"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if not password or len(password) < 8:
        return Response(
            {"error": "Password must be at least 8 characters"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if User.objects.filter(username=username).exists():
        return Response(
            {"error": "Username already taken"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if email and User.objects.filter(email=email).exists():
        return Response(
            {"error": "Email already registered"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Check invite code if required (paid plans bypass this)
    invite = None
    if settings.require_invite and not is_paid_signup:
        if not invite_code:
            return Response(
                {"error": "Invite code required for alpha access"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            invite = InviteCode.objects.get(code=invite_code)
            if not invite.is_valid:
                return Response(
                    {"error": "Invite code is expired or fully used"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
        except InviteCode.DoesNotExist:
            return Response(
                {"error": "Invalid invite code"},
                status=status.HTTP_400_BAD_REQUEST,
            )

    # Create user
    user = User.objects.create_user(
        username=username,
        email=email,
        password=password,
        tier=User.Tier.FREE,  # New users start on free tier
    )

    # Mark invite as used
    if invite:
        invite.use(user)

    # Send verification email
    verification_sent = False
    if email:
        try:
            user.send_verification_email()
            verification_sent = True
            logger.info(f"Verification email sent to: {email}")
        except Exception as e:
            logger.error(f"Failed to send verification email: {e}")

    logger.info(f"New user registered: {username} (invite: {invite_code or 'none'})")

    return Response({
        "status": "registered",
        "username": username,
        "tier": user.tier,
        "email_verified": False,
        "verification_sent": verification_sent,
        "message": "Welcome to SVEND! Please check your email to verify your account.",
    }, status=status.HTTP_201_CREATED)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def change_password(request):
    """Change user password.

    Request body:
        current_password: str - Current password for verification
        new_password: str - New password (min 8 characters)
    """
    user = request.user

    current_password = request.data.get("current_password", "")
    new_password = request.data.get("new_password", "")

    if not current_password or not new_password:
        return Response(
            {"error": "Current password and new password required"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if not user.check_password(current_password):
        return Response(
            {"error": "Current password is incorrect"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if len(new_password) < 8:
        return Response(
            {"error": "New password must be at least 8 characters"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    user.set_password(new_password)
    user.save()

    # Re-authenticate to maintain session
    from django.contrib.auth import update_session_auth_hash
    update_session_auth_hash(request, user)

    logger.info(f"Password changed for user: {user.username}")

    return Response({"status": "password_changed"})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def send_verification_email(request):
    """Send or resend verification email to current user."""
    user = request.user

    if user.email_verified:
        return Response({"status": "already_verified"})

    if not user.email:
        return Response(
            {"error": "No email address on account"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        user.send_verification_email()
        logger.info(f"Verification email sent to: {user.email}")
        return Response({"status": "sent", "email": user.email})
    except Exception as e:
        logger.error(f"Failed to send verification email: {e}")
        return Response(
            {"error": "Failed to send email. Please try again."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["GET", "POST"])
@permission_classes([AllowAny])
def verify_email(request):
    """Verify email with token.

    GET /api/auth/verify-email/?token=xxx
    POST /api/auth/verify-email/ with {"token": "xxx"}
    """
    from accounts.models import User

    token = request.query_params.get("token") or request.data.get("token", "")

    if not token:
        return Response(
            {"error": "Verification token required"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        user = User.objects.get(email_verification_token=token)
        if user.verify_email(token):
            logger.info(f"Email verified for user: {user.username}")
            return Response({
                "status": "verified",
                "username": user.username,
                "message": "Email verified successfully! You can now use all features.",
            })
        else:
            return Response(
                {"error": "Invalid or expired token"},
                status=status.HTTP_400_BAD_REQUEST,
            )
    except User.DoesNotExist:
        return Response(
            {"error": "Invalid verification token"},
            status=status.HTTP_400_BAD_REQUEST,
        )


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def export_pdf(request):
    """Export content to PDF.

    Request body:
        content: str - The content to export (markdown, html, or plain text)
        format: str - Content format: 'markdown', 'html', or 'text' (default: 'markdown')
        title: str - Document title (optional)
        include_math: bool - Whether to render LaTeX math (default: True)
    """
    import subprocess
    import tempfile
    import os
    from django.http import HttpResponse

    content = request.data.get("content", "")
    content_format = request.data.get("format", "markdown")
    title = request.data.get("title", "Document")
    include_math = request.data.get("include_math", True)

    if not content:
        return Response(
            {"error": "No content provided"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Convert to HTML if needed
    if content_format == "markdown":
        try:
            import markdown as md_lib
            html_content = md_lib.markdown(
                content,
                extensions=['tables', 'fenced_code', 'toc']
            )
        except ImportError:
            # Fallback: basic HTML escaping
            import html
            html_content = f"<div>{html.escape(content).replace(chr(10), '<br>')}</div>"
    elif content_format == "text":
        import html
        html_content = f"<pre>{html.escape(content)}</pre>"
    else:
        html_content = content

    # Build full HTML document with Svend styling
    html_doc = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    {'<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">' if include_math else ''}
    <style>
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
            color: #1a1a1a;
        }}
        h1, h2, h3 {{ color: #2d5a3d; margin-top: 1.5em; }}
        h1 {{ font-size: 2rem; border-bottom: 2px solid #4a9f6e; padding-bottom: 0.5rem; }}
        h2 {{ font-size: 1.5rem; }}
        h3 {{ font-size: 1.25rem; }}
        code {{
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            background: #f5f5f5;
            padding: 0.2em 0.4em;
            border-radius: 3px;
            font-size: 0.9em;
        }}
        pre {{
            background: #1a1a1a;
            color: #e8efe8;
            padding: 1rem;
            border-radius: 6px;
            overflow-x: auto;
        }}
        pre code {{ background: none; padding: 0; color: inherit; }}
        blockquote {{
            border-left: 4px solid #4a9f6e;
            margin: 1em 0;
            padding-left: 1em;
            color: #555;
        }}
        table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
        th, td {{ border: 1px solid #ddd; padding: 0.5em; text-align: left; }}
        th {{ background: #4a9f6e; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        a {{ color: #4a9f6e; }}
        .katex-display {{ overflow-x: auto; padding: 0.5em 0; }}
        @media print {{
            body {{ padding: 0; }}
            pre {{ white-space: pre-wrap; word-wrap: break-word; }}
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    {html_content}
</body>
</html>"""

    # Try to generate PDF with wkhtmltopdf or weasyprint
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as html_file:
            html_file.write(html_doc)
            html_path = html_file.name

        pdf_path = html_path.replace('.html', '.pdf')

        # Try wkhtmltopdf first (faster, better CSS support)
        try:
            subprocess.run(
                ['wkhtmltopdf', '--quiet', '--enable-local-file-access',
                 '--margin-top', '20mm', '--margin-bottom', '20mm',
                 '--margin-left', '15mm', '--margin-right', '15mm',
                 html_path, pdf_path],
                check=True,
                capture_output=True,
                timeout=30
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to weasyprint
            try:
                from weasyprint import HTML
                HTML(filename=html_path).write_pdf(pdf_path)
            except ImportError:
                # Last resort: return HTML for browser printing
                os.unlink(html_path)
                response = HttpResponse(html_doc, content_type='text/html')
                response['Content-Disposition'] = f'inline; filename="{title}.html"'
                return response

        # Read and return PDF
        with open(pdf_path, 'rb') as pdf_file:
            pdf_content = pdf_file.read()

        # Cleanup
        os.unlink(html_path)
        os.unlink(pdf_path)

        response = HttpResponse(pdf_content, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{title}.pdf"'
        return response

    except Exception as e:
        logger.error(f"PDF export failed: {e}")
        return Response(
            {"error": f"PDF generation failed: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
