"""Privacy data export API views (PRIV-001 §4.1, SOC 2 P1.8).

Provides self-service data export endpoints:
  POST /api/privacy/exports/     — request a new export
  GET  /api/privacy/exports/     — list user's exports
  GET  /api/privacy/exports/<id>/ — status detail or download
  DELETE /api/privacy/exports/<id>/ — cancel or delete
"""

import os
from datetime import timedelta

from django.http import FileResponse
from django.utils import timezone
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import DataExportRequest


@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def exports_collection(request):
    """Handle export list (GET) and creation (POST)."""
    if request.method == "POST":
        return _create_export(request)
    return _list_exports(request)


@api_view(["GET", "DELETE"])
@permission_classes([IsAuthenticated])
def export_resource(request, export_id):
    """Handle export detail/download (GET) and cancel (DELETE)."""
    if request.method == "DELETE":
        return _cancel_export(request, export_id)
    return _export_detail(request, export_id)


def _create_export(request):
    """Create a data export request (rate limited: 1 per 24h)."""
    user = request.user

    # Rate limit: 1 per 24 hours (exclude cancelled)
    recent = (
        DataExportRequest.objects.filter(
            user=user,
            created_at__gte=timezone.now() - timedelta(hours=24),
        )
        .exclude(
            status__in=[
                DataExportRequest.Status.CANCELLED,
                DataExportRequest.Status.EXPIRED,
            ],
        )
        .exists()
    )

    if recent:
        return Response(
            {"error": "You can request one export per 24 hours."},
            status=status.HTTP_429_TOO_MANY_REQUESTS,
        )

    export_req = DataExportRequest.objects.create(user=user)

    # Run export generation (synchronous — syn.sched integration is async fallback)
    from accounts.privacy_tasks import generate_export

    generate_export({"export_id": str(export_req.id)})

    export_req.refresh_from_db()
    return Response(_serialize_export(export_req), status=status.HTTP_201_CREATED)


def _list_exports(request):
    """List the authenticated user's export requests."""
    exports = DataExportRequest.objects.filter(user=request.user)[:20]
    return Response([_serialize_export(e) for e in exports])


def _export_detail(request, export_id):
    """Get export status or download the file."""
    try:
        export_req = DataExportRequest.objects.get(id=export_id, user=request.user)
    except DataExportRequest.DoesNotExist:
        return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)

    # Download mode
    if request.query_params.get("download") == "true":
        if export_req.status == DataExportRequest.Status.EXPIRED:
            return Response(
                {"error": "Export has expired"},
                status=status.HTTP_410_GONE,
            )
        if export_req.status != DataExportRequest.Status.COMPLETED:
            return Response(
                {"error": "Export not ready for download"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if not export_req.file_path or not os.path.exists(export_req.file_path):
            return Response(
                {"error": "Export file not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        # Mark as downloaded
        if not export_req.downloaded_at:
            export_req.downloaded_at = timezone.now()
            export_req.save(update_fields=["downloaded_at"])

            # Audit log
            try:
                from accounts.privacy_tasks import _audit_log

                _audit_log(
                    request.user,
                    "privacy.export.downloaded",
                    {
                        "export_id": str(export_req.id),
                    },
                )
            except Exception:
                pass

        date_str = export_req.created_at.strftime("%Y-%m-%d")
        filename = f"svend-data-export-{date_str}.json"
        response = FileResponse(
            open(export_req.file_path, "rb"),
            content_type="application/json",
        )
        response["Content-Disposition"] = f'attachment; filename="{filename}"'
        return response

    return Response(_serialize_export(export_req))


def _cancel_export(request, export_id):
    """Cancel a pending export or delete a completed one."""
    try:
        export_req = DataExportRequest.objects.get(id=export_id, user=request.user)
    except DataExportRequest.DoesNotExist:
        return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)

    if export_req.status in (
        DataExportRequest.Status.PENDING,
        DataExportRequest.Status.PROCESSING,
    ):
        export_req.status = DataExportRequest.Status.CANCELLED
        export_req.save(update_fields=["status"])
    elif export_req.status == DataExportRequest.Status.COMPLETED:
        # Delete file and expire
        if export_req.file_path and os.path.exists(export_req.file_path):
            os.remove(export_req.file_path)
        export_req.status = DataExportRequest.Status.EXPIRED
        export_req.save(update_fields=["status"])
    else:
        return Response(
            {"error": f"Cannot cancel export in {export_req.status} state"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        from accounts.privacy_tasks import _audit_log

        _audit_log(
            request.user,
            "privacy.export.cancelled",
            {
                "export_id": str(export_req.id),
            },
        )
    except Exception:
        pass

    return Response(_serialize_export(export_req))


def _serialize_export(export_req):
    """Serialize a DataExportRequest to dict."""
    return {
        "id": str(export_req.id),
        "status": export_req.status,
        "export_format": export_req.export_format,
        "file_size_bytes": export_req.file_size_bytes,
        "created_at": (export_req.created_at.isoformat() if export_req.created_at else None),
        "completed_at": (export_req.completed_at.isoformat() if export_req.completed_at else None),
        "expires_at": (export_req.expires_at.isoformat() if export_req.expires_at else None),
        "downloaded_at": (export_req.downloaded_at.isoformat() if export_req.downloaded_at else None),
    }
