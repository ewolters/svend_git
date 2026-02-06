"""API serializers."""

from rest_framework import serializers

from chat.models import Conversation, Message


class MessageSerializer(serializers.ModelSerializer):
    """Serializer for Message model."""

    class Meta:
        model = Message
        fields = [
            "id",
            "role",
            "content",
            "domain",
            "difficulty",
            "verified",
            "verification_confidence",
            "blocked",
            "block_reason",
            "reasoning_trace",
            "tool_calls",
            "inference_time_ms",
            "created_at",
        ]
        read_only_fields = fields


class ConversationSerializer(serializers.ModelSerializer):
    """Serializer for Conversation model."""

    messages = MessageSerializer(many=True, read_only=True)
    message_count = serializers.SerializerMethodField()

    class Meta:
        model = Conversation
        fields = [
            "id",
            "title",
            "messages",
            "message_count",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]

    def get_message_count(self, obj):
        return obj.messages.count()


class ConversationListSerializer(serializers.ModelSerializer):
    """Lighter serializer for conversation list."""

    message_count = serializers.SerializerMethodField()
    last_message = serializers.SerializerMethodField()

    class Meta:
        model = Conversation
        fields = [
            "id",
            "title",
            "message_count",
            "last_message",
            "updated_at",
        ]

    def get_message_count(self, obj):
        return obj.messages.count()

    def get_last_message(self, obj):
        last = obj.messages.last()
        if last:
            return {
                "role": last.role,
                "content": last.content[:100],
            }
        return None


class ChatInputSerializer(serializers.Serializer):
    """Serializer for chat input."""

    message = serializers.CharField(max_length=10000)
    conversation_id = serializers.UUIDField(required=False, allow_null=True)
    mode = serializers.ChoiceField(
        choices=["auto", "coder", "synara"],
        default="auto",
        required=False,
        help_text="Pipeline mode: auto (smart routing), coder (computation), synara (reasoning)"
    )
    model = serializers.ChoiceField(
        choices=["default", "opus", "sonnet", "haiku", "qwen"],
        default="default",
        required=False,
        help_text="Model selection (enterprise only): opus, sonnet, haiku, qwen"
    )


class ShareSerializer(serializers.Serializer):
    """Serializer for sharing a conversation."""

    conversation_id = serializers.UUIDField()
