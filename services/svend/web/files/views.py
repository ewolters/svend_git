"""File management API views."""

import logging
import mimetypes
from pathlib import Path

from django.http import FileResponse, Http404
from django.utils import timezone
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response

from .models import UserFile, UserQuota

logger = logging.getLogger(__name__)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_files(request):
    """List user's files with optional filtering."""
    user = request.user

    # Build queryset
    queryset = UserFile.objects.filter(user=user)

    # Filter by folder
    folder = request.query_params.get("folder")
    if folder:
        queryset = queryset.filter(folder=folder)

    # Filter by type
    file_type = request.query_params.get("type")
    if file_type:
        queryset = queryset.filter(file_type=file_type)

    # Search by name
    search = request.query_params.get("search")
    if search:
        queryset = queryset.filter(original_name__icontains=search)

    # Pagination
    limit = min(int(request.query_params.get("limit", 50)), 100)
    offset = int(request.query_params.get("offset", 0))

    total = queryset.count()
    files = queryset[offset:offset + limit]

    return Response({
        "total": total,
        "limit": limit,
        "offset": offset,
        "files": [
            {
                "id": str(f.id),
                "name": f.original_name,
                "type": f.file_type,
                "mime_type": f.mime_type,
                "size": f.size_bytes,
                "folder": f.folder,
                "url": f.url,
                "created_at": f.created_at.isoformat(),
                "is_public": f.is_public,
            }
            for f in files
        ],
    })


@api_view(["POST"])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FormParser])
def upload_file(request):
    """Upload a file.

    Security:
    - Files are stored in user-segregated directories
    - Quota is checked before upload
    - MIME type is validated
    """
    user = request.user

    if "file" not in request.FILES:
        return Response(
            {"error": "No file provided"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    uploaded = request.FILES["file"]
    folder = request.data.get("folder", "")
    description = request.data.get("description", "")

    # Get or create quota
    quota = UserQuota.get_or_create_for_user(user)

    # Check quota
    can_upload, error_msg = quota.can_upload(uploaded.size)
    if not can_upload:
        return Response(
            {"error": error_msg},
            status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )

    # Detect MIME type
    mime_type = uploaded.content_type or mimetypes.guess_type(uploaded.name)[0] or "application/octet-stream"

    # Security: Block dangerous file types
    dangerous_types = [
        "application/x-executable",
        "application/x-msdos-program",
        "application/x-msdownload",
    ]
    dangerous_extensions = [".exe", ".bat", ".cmd", ".sh", ".ps1", ".dll"]

    if mime_type in dangerous_types:
        return Response(
            {"error": "File type not allowed"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    ext = Path(uploaded.name).suffix.lower()
    if ext in dangerous_extensions:
        return Response(
            {"error": "File extension not allowed"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Create file record
    user_file = UserFile.objects.create(
        user=user,
        file=uploaded,
        original_name=uploaded.name,
        mime_type=mime_type,
        size_bytes=uploaded.size,
        folder=folder,
        description=description,
    )

    # Update quota
    quota.add_file(uploaded.size)

    # Compute checksum async (or skip for now)
    # user_file.compute_checksum()

    logger.info(f"File uploaded: {user.username}/{user_file.original_name} ({user_file.size_bytes} bytes)")

    return Response({
        "id": str(user_file.id),
        "name": user_file.original_name,
        "type": user_file.file_type,
        "mime_type": user_file.mime_type,
        "size": user_file.size_bytes,
        "url": user_file.url,
        "created_at": user_file.created_at.isoformat(),
    }, status=status.HTTP_201_CREATED)


@api_view(["GET", "DELETE", "PATCH"])
@permission_classes([IsAuthenticated])
def file_detail(request, file_id):
    """Get, update, or delete a file."""
    user = request.user

    try:
        user_file = UserFile.objects.get(id=file_id, user=user)
    except UserFile.DoesNotExist:
        return Response(
            {"error": "File not found"},
            status=status.HTTP_404_NOT_FOUND,
        )

    if request.method == "DELETE":
        # Update quota
        quota = UserQuota.get_or_create_for_user(user)
        quota.remove_file(user_file.size_bytes)

        # Delete file
        user_file.delete()

        logger.info(f"File deleted: {user.username}/{user_file.original_name}")
        return Response(status=status.HTTP_204_NO_CONTENT)

    elif request.method == "PATCH":
        # Update metadata
        if "folder" in request.data:
            user_file.folder = request.data["folder"]
        if "description" in request.data:
            user_file.description = request.data["description"]
        if "tags" in request.data:
            user_file.tags = request.data["tags"]
        if "is_public" in request.data:
            user_file.is_public = request.data["is_public"]

        user_file.save()

        return Response({
            "id": str(user_file.id),
            "name": user_file.original_name,
            "folder": user_file.folder,
            "description": user_file.description,
            "tags": user_file.tags,
            "is_public": user_file.is_public,
        })

    # GET - return file metadata
    user_file.accessed_at = timezone.now()
    user_file.save(update_fields=["accessed_at"])

    return Response({
        "id": str(user_file.id),
        "name": user_file.original_name,
        "type": user_file.file_type,
        "mime_type": user_file.mime_type,
        "size": user_file.size_bytes,
        "folder": user_file.folder,
        "description": user_file.description,
        "tags": user_file.tags,
        "url": user_file.url,
        "checksum": user_file.checksum,
        "is_public": user_file.is_public,
        "share_token": user_file.share_token if user_file.is_public else None,
        "created_at": user_file.created_at.isoformat(),
        "accessed_at": user_file.accessed_at.isoformat() if user_file.accessed_at else None,
    })


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def download_file(request, file_id):
    """Download a file.

    Security: Only file owner can download (unless shared).
    """
    user = request.user

    try:
        user_file = UserFile.objects.get(id=file_id, user=user)
    except UserFile.DoesNotExist:
        return Response(
            {"error": "File not found"},
            status=status.HTTP_404_NOT_FOUND,
        )

    if not user_file.file:
        return Response(
            {"error": "File data not available"},
            status=status.HTTP_404_NOT_FOUND,
        )

    # Update access time
    user_file.accessed_at = timezone.now()
    user_file.save(update_fields=["accessed_at"])

    response = FileResponse(
        user_file.file.open("rb"),
        content_type=user_file.mime_type or "application/octet-stream",
    )
    response["Content-Disposition"] = f'attachment; filename="{user_file.original_name}"'
    return response


@api_view(["GET"])
@permission_classes([AllowAny])
def shared_file(request, share_token):
    """Access a shared file by token."""
    try:
        user_file = UserFile.objects.get(share_token=share_token, is_public=True)
    except UserFile.DoesNotExist:
        raise Http404("File not found")

    # Update access time
    user_file.accessed_at = timezone.now()
    user_file.save(update_fields=["accessed_at"])

    # Return file or metadata based on query param
    if request.query_params.get("download") == "true":
        response = FileResponse(
            user_file.file.open("rb"),
            content_type=user_file.mime_type or "application/octet-stream",
        )
        response["Content-Disposition"] = f'attachment; filename="{user_file.original_name}"'
        return response

    return Response({
        "id": str(user_file.id),
        "name": user_file.original_name,
        "type": user_file.file_type,
        "mime_type": user_file.mime_type,
        "size": user_file.size_bytes,
        "created_at": user_file.created_at.isoformat(),
    })


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def create_share_link(request, file_id):
    """Create a share link for a file."""
    user = request.user

    try:
        user_file = UserFile.objects.get(id=file_id, user=user)
    except UserFile.DoesNotExist:
        return Response(
            {"error": "File not found"},
            status=status.HTTP_404_NOT_FOUND,
        )

    # Generate share token
    token = user_file.generate_share_token()
    user_file.is_public = True
    user_file.save(update_fields=["is_public"])

    return Response({
        "share_token": token,
        "url": f"/api/files/shared/{token}/",
    })


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def revoke_share_link(request, file_id):
    """Revoke a share link."""
    user = request.user

    try:
        user_file = UserFile.objects.get(id=file_id, user=user)
    except UserFile.DoesNotExist:
        return Response(
            {"error": "File not found"},
            status=status.HTTP_404_NOT_FOUND,
        )

    user_file.share_token = ""
    user_file.is_public = False
    user_file.save(update_fields=["share_token", "is_public"])

    return Response({"status": "share_revoked"})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def storage_quota(request):
    """Get user's storage quota and usage."""
    user = request.user
    quota = UserQuota.get_or_create_for_user(user)

    return Response({
        "quota_bytes": quota.quota_bytes,
        "used_bytes": quota.used_bytes,
        "remaining_bytes": quota.remaining_bytes,
        "usage_percent": round(quota.usage_percent, 1),
        "file_count": quota.file_count,
        "max_files": quota.max_files,
        "max_file_size_bytes": quota.max_file_size_bytes,
        "tier": user.tier,
    })


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_folders(request):
    """List user's folders."""
    user = request.user

    folders = (
        UserFile.objects.filter(user=user)
        .exclude(folder="")
        .values_list("folder", flat=True)
        .distinct()
    )

    return Response({
        "folders": sorted(set(folders)),
    })
