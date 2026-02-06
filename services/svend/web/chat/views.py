"""Chat views."""

from django.http import JsonResponse
from django.shortcuts import get_object_or_404

from .models import SharedConversation


def shared_conversation(request, share_id):
    """View a shared conversation."""
    share = get_object_or_404(SharedConversation, id=share_id)

    # Increment view count
    share.view_count += 1
    share.save(update_fields=["view_count"])

    # Return conversation data
    messages = [
        {
            "role": msg.role,
            "content": msg.content,
            "domain": msg.domain,
            "verified": msg.verified,
            "reasoning_trace": msg.reasoning_trace,
            "tool_calls": msg.tool_calls,
        }
        for msg in share.conversation.messages.all()
    ]

    return JsonResponse({
        "title": share.conversation.title,
        "messages": messages,
        "view_count": share.view_count,
    })
