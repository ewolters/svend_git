"""Privacy data export tasks (PRIV-001 §5, SOC 2 P1.8).

Async handlers for generating user data exports and cleaning up expired files.
Registered with syn.sched in svend_tasks.py.
"""

import json
import logging
import os
from datetime import timedelta

from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)

EXPORTS_DIR = os.path.join(settings.MEDIA_ROOT, "exports")
EXPORT_EXPIRY_DAYS = 7


def generate_export(payload, context=None):
    """Generate a JSON data export for a user (PRIV-001 §5.2).

    Collects all PII-bearing data for the user, writes a JSON file,
    and updates the DataExportRequest record.
    """
    from accounts.models import DataExportRequest

    export_id = (
        payload.get("export_id")
        if isinstance(payload, dict)
        else getattr(payload, "payload", {}).get("export_id")
    )
    if not export_id:
        logger.error("[privacy] generate_export called without export_id")
        return {"error": "missing export_id"}

    try:
        export_req = DataExportRequest.objects.select_related("user").get(id=export_id)
    except DataExportRequest.DoesNotExist:
        logger.error("[privacy] DataExportRequest %s not found", export_id)
        return {"error": "not_found"}

    if export_req.status != DataExportRequest.Status.PENDING:
        logger.warning(
            "[privacy] Export %s not pending (status=%s)", export_id, export_req.status
        )
        return {"error": "not_pending"}

    export_req.status = DataExportRequest.Status.PROCESSING
    export_req.processing_started_at = timezone.now()
    export_req.save(update_fields=["status", "processing_started_at"])

    user = export_req.user

    # Audit: export started
    _audit_log(user, "privacy.export.requested", {"export_id": str(export_id)})

    try:
        data = _collect_user_data(user, export_req)

        # Ensure exports directory exists
        os.makedirs(EXPORTS_DIR, exist_ok=True)

        file_path = os.path.join(EXPORTS_DIR, f"export-{user.id}-{export_id}.json")
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        file_size = os.path.getsize(file_path)
        now = timezone.now()

        export_req.status = DataExportRequest.Status.COMPLETED
        export_req.file_path = file_path
        export_req.file_size_bytes = file_size
        export_req.completed_at = now
        export_req.expires_at = now + timedelta(days=EXPORT_EXPIRY_DAYS)
        export_req.save(
            update_fields=[
                "status",
                "file_path",
                "file_size_bytes",
                "completed_at",
                "expires_at",
            ]
        )

        _audit_log(
            user,
            "privacy.export.completed",
            {
                "export_id": str(export_id),
                "file_size_bytes": file_size,
            },
        )

        logger.info("[privacy] Export %s completed (%d bytes)", export_id, file_size)
        return {"status": "completed", "file_size": file_size}

    except Exception as e:
        export_req.status = DataExportRequest.Status.FAILED
        export_req.error_message = str(e)[:1000]
        export_req.save(update_fields=["status", "error_message"])

        _audit_log(
            user,
            "privacy.export.failed",
            {
                "export_id": str(export_id),
                "error": str(e)[:200],
            },
        )

        logger.exception("[privacy] Export %s failed", export_id)
        return {"error": str(e)}


def cleanup_expired_exports(payload=None, context=None):
    """Delete expired export files and update status (PRIV-001 §5.4).

    Runs weekly via syn.sched. Finds completed exports past their
    expiry date, deletes the file, and marks them expired.
    """
    from accounts.models import DataExportRequest

    now = timezone.now()
    expired = DataExportRequest.objects.filter(
        status=DataExportRequest.Status.COMPLETED,
        expires_at__lt=now,
    )

    count = 0
    for export_req in expired:
        if export_req.file_path and os.path.exists(export_req.file_path):
            try:
                os.remove(export_req.file_path)
            except OSError:
                logger.warning("[privacy] Could not delete %s", export_req.file_path)

        export_req.status = DataExportRequest.Status.EXPIRED
        export_req.save(update_fields=["status"])

        _audit_log(
            export_req.user,
            "privacy.export.expired",
            {
                "export_id": str(export_req.id),
            },
        )
        count += 1

    logger.info("[privacy] Cleaned up %d expired exports", count)
    return {"expired_count": count}


def _collect_user_data(user, export_req):
    """Collect all PII-bearing data for a user into a dict."""
    data = {
        "export_metadata": {
            "export_id": str(export_req.id),
            "user_id": str(user.id),
            "generated_at": timezone.now().isoformat(),
            "format_version": "1.0",
        },
        "profile": _export_profile(user),
        "subscription": _export_subscription(user),
        "conversations": _export_conversations(user),
        "analysis_results": _export_dsw_results(user),
        "triage_results": _export_triage_results(user),
        "saved_models": _export_saved_models(user),
        "usage_summary": _export_usage_summary(user),
        "notifications": _export_notifications(user),
    }
    return data


def _export_profile(user):
    return {
        "username": user.username,
        "email": user.email,
        "display_name": user.display_name,
        "bio": user.bio,
        "avatar_url": user.avatar_url,
        "industry": user.industry,
        "role": user.role,
        "experience_level": user.experience_level,
        "organization_size": user.organization_size,
        "tier": user.tier,
        "preferences": user.preferences,
        "current_theme": user.current_theme,
        "date_joined": user.date_joined.isoformat() if user.date_joined else None,
        "last_active_at": (
            user.last_active_at.isoformat() if user.last_active_at else None
        ),
        "is_email_verified": user.is_email_verified,
        "onboarding_completed_at": (
            user.onboarding_completed_at.isoformat()
            if user.onboarding_completed_at
            else None
        ),
        "total_queries": user.total_queries,
    }


def _export_subscription(user):
    try:
        sub = user.subscription
        return {
            "status": sub.status,
            "current_period_start": (
                sub.current_period_start.isoformat()
                if sub.current_period_start
                else None
            ),
            "current_period_end": (
                sub.current_period_end.isoformat() if sub.current_period_end else None
            ),
            "created_at": sub.created_at.isoformat() if sub.created_at else None,
        }
    except Exception:
        return None


def _export_conversations(user):
    from chat.models import Conversation

    conversations = []
    for conv in Conversation.objects.filter(user=user).prefetch_related("messages"):
        messages = []
        for msg in conv.messages.all():
            messages.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat(),
                }
            )
        conversations.append(
            {
                "id": str(conv.id),
                "title": conv.title,
                "created_at": conv.created_at.isoformat(),
                "updated_at": conv.updated_at.isoformat(),
                "message_count": len(messages),
                "messages": messages,
            }
        )
    return conversations


def _export_dsw_results(user):
    from agents_api.models import DSWResult

    results = []
    for r in DSWResult.objects.filter(user=user):
        results.append(
            {
                "id": r.id,
                "result_type": r.result_type,
                "title": r.title,
                "data": r.data,
                "created_at": r.created_at.isoformat(),
            }
        )
    return results


def _export_triage_results(user):
    from agents_api.models import TriageResult

    results = []
    for r in TriageResult.objects.filter(user=user):
        results.append(
            {
                "id": r.id,
                "original_filename": r.original_filename,
                "report_markdown": r.report_markdown,
                "summary_json": r.summary_json,
                "created_at": r.created_at.isoformat(),
            }
        )
    return results


def _export_saved_models(user):
    from agents_api.models import SavedModel

    models_list = []
    for m in SavedModel.objects.filter(user=user):
        models_list.append(
            {
                "id": str(m.id),
                "name": m.name,
                "description": m.description,
                "model_type": m.model_type,
                "metrics": m.metrics,
                "feature_names": m.feature_names,
                "target_name": m.target_name,
                "created_at": m.created_at.isoformat(),
            }
        )
    return models_list


def _export_usage_summary(user):
    from chat.models import UsageLog

    logs = UsageLog.objects.filter(user=user).order_by("-date")[:90]
    return [
        {
            "date": str(log.date),
            "request_count": log.request_count,
            "tokens_input": log.tokens_input,
            "tokens_output": log.tokens_output,
        }
        for log in logs
    ]


def _export_notifications(user):
    from notifications.models import Notification

    notifications = []
    for n in Notification.objects.filter(recipient=user).order_by("-created_at")[:500]:
        notifications.append(
            {
                "id": str(n.id),
                "type": n.notification_type,
                "title": n.title,
                "message": n.message,
                "is_read": n.is_read,
                "created_at": n.created_at.isoformat(),
            }
        )
    return notifications


def _audit_log(user, event_name, payload):
    """Log a privacy event to the audit trail."""
    try:
        from syn.audit.utils import generate_entry

        generate_entry(
            tenant_id=str(user.id),
            actor=user.username or str(user.id),
            event_name=event_name,
            payload=payload,
        )
    except Exception:
        logger.warning("[privacy] Failed to write audit log for %s", event_name)
