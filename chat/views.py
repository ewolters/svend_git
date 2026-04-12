"""Chat views."""

from django.db.models import F
from django.http import JsonResponse
from django.shortcuts import get_object_or_404

from .models import SharedConversation


def shared_conversation(request, share_id):
    """View a shared conversation."""
    share = get_object_or_404(SharedConversation, id=share_id)

    # Atomic view count increment
    SharedConversation.objects.filter(pk=share.pk).update(view_count=F("view_count") + 1)
    share.refresh_from_db(fields=["view_count"])

    # Return conversation data (exclude internal fields: reasoning_trace, tool_calls)
    messages = [
        {
            "role": msg.role,
            "content": msg.content,
            "domain": msg.domain,
            "verified": msg.is_verified,
        }
        for msg in share.conversation.messages.all()
    ]

    return JsonResponse(
        {
            "title": share.conversation.title,
            "messages": messages,
            "view_count": share.view_count,
        }
    )
