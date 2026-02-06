"""User file storage models."""

import hashlib
import os
import uuid
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.db import models


def user_file_path(instance, filename):
    """Generate user-segregated file path.

    Structure: files/<user_id>/<year>/<month>/<uuid>_<filename>
    This ensures complete isolation between users.
    """
    from django.utils import timezone

    now = timezone.now()
    ext = Path(filename).suffix.lower()
    safe_name = Path(filename).stem[:50]  # Limit filename length

    # Generate unique filename
    unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}{ext}"

    return f"files/{instance.user.id}/{now.year}/{now.month:02d}/{unique_name}"


class UserFile(models.Model):
    """File uploaded by a user.

    Files are stored in user-segregated directories.
    Access is controlled at the view level.
    """

    class FileType(models.TextChoices):
        IMAGE = "image", "Image"
        DOCUMENT = "document", "Document"
        DATA = "data", "Data File"
        CODE = "code", "Code"
        OTHER = "other", "Other"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="files",
    )

    # File metadata
    file = models.FileField(upload_to=user_file_path)
    original_name = models.CharField(max_length=255)
    file_type = models.CharField(
        max_length=20,
        choices=FileType.choices,
        default=FileType.OTHER,
    )
    mime_type = models.CharField(max_length=100, blank=True)
    size_bytes = models.BigIntegerField(default=0)
    checksum = models.CharField(max_length=64, blank=True)  # SHA-256

    # Organization
    folder = models.CharField(max_length=255, blank=True, db_index=True)
    description = models.TextField(blank=True)
    tags = models.JSONField(default=list, blank=True)

    # Sharing
    is_public = models.BooleanField(default=False)
    share_token = models.CharField(max_length=32, blank=True, db_index=True)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    accessed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "user_files"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "folder"]),
            models.Index(fields=["user", "file_type"]),
            models.Index(fields=["user", "created_at"]),
        ]

    def __str__(self):
        return f"{self.user.username}/{self.original_name}"

    def save(self, *args, **kwargs):
        # Compute file type from mime type
        if self.mime_type and not self.file_type:
            self.file_type = self._detect_file_type(self.mime_type)

        super().save(*args, **kwargs)

    def _detect_file_type(self, mime_type: str) -> str:
        """Detect file type from MIME type."""
        if mime_type.startswith("image/"):
            return self.FileType.IMAGE
        elif mime_type.startswith("text/"):
            if "python" in mime_type or "javascript" in mime_type:
                return self.FileType.CODE
            return self.FileType.DOCUMENT
        elif mime_type in ("application/pdf", "application/msword"):
            return self.FileType.DOCUMENT
        elif mime_type in ("application/json", "text/csv", "application/xml"):
            return self.FileType.DATA
        elif "python" in mime_type or "javascript" in mime_type:
            return self.FileType.CODE
        return self.FileType.OTHER

    def generate_share_token(self) -> str:
        """Generate a unique share token."""
        import secrets
        self.share_token = secrets.token_urlsafe(24)
        self.save(update_fields=["share_token"])
        return self.share_token

    def compute_checksum(self) -> str:
        """Compute SHA-256 checksum of file."""
        if not self.file:
            return ""

        sha256 = hashlib.sha256()
        self.file.seek(0)
        for chunk in iter(lambda: self.file.read(8192), b""):
            sha256.update(chunk)
        self.file.seek(0)

        self.checksum = sha256.hexdigest()
        self.save(update_fields=["checksum"])
        return self.checksum

    def delete(self, *args, **kwargs):
        """Delete file from storage when model is deleted."""
        if self.file:
            storage = self.file.storage
            if storage.exists(self.file.name):
                storage.delete(self.file.name)
        super().delete(*args, **kwargs)

    @property
    def url(self) -> str:
        """Get file URL."""
        return self.file.url if self.file else ""

    @property
    def path(self) -> str:
        """Get file path."""
        return self.file.path if self.file else ""


class UserQuota(models.Model):
    """Storage quota for a user."""

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="storage_quota",
    )

    # Quota limits (in bytes)
    quota_bytes = models.BigIntegerField(default=100 * 1024 * 1024)  # 100 MB default
    used_bytes = models.BigIntegerField(default=0)

    # File count limits
    max_files = models.IntegerField(default=1000)
    file_count = models.IntegerField(default=0)

    # Per-file limits
    max_file_size_bytes = models.BigIntegerField(default=10 * 1024 * 1024)  # 10 MB

    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "user_quotas"

    def __str__(self):
        return f"{self.user.username}: {self.used_bytes}/{self.quota_bytes}"

    @property
    def usage_percent(self) -> float:
        """Get quota usage percentage."""
        if self.quota_bytes == 0:
            return 100.0
        return (self.used_bytes / self.quota_bytes) * 100

    @property
    def remaining_bytes(self) -> int:
        """Get remaining storage."""
        return max(0, self.quota_bytes - self.used_bytes)

    def can_upload(self, file_size: int) -> tuple[bool, str]:
        """Check if user can upload a file of given size."""
        if file_size > self.max_file_size_bytes:
            return False, f"File too large. Maximum size: {self.max_file_size_bytes // 1024 // 1024} MB"

        if self.file_count >= self.max_files:
            return False, f"File limit reached. Maximum files: {self.max_files}"

        if self.used_bytes + file_size > self.quota_bytes:
            return False, f"Storage quota exceeded. Remaining: {self.remaining_bytes // 1024 // 1024} MB"

        return True, ""

    def add_file(self, size_bytes: int):
        """Track new file upload."""
        self.used_bytes += size_bytes
        self.file_count += 1
        self.save(update_fields=["used_bytes", "file_count", "updated_at"])

    def remove_file(self, size_bytes: int):
        """Track file deletion."""
        self.used_bytes = max(0, self.used_bytes - size_bytes)
        self.file_count = max(0, self.file_count - 1)
        self.save(update_fields=["used_bytes", "file_count", "updated_at"])

    def recalculate(self):
        """Recalculate usage from actual files."""
        from django.db.models import Sum, Count

        stats = UserFile.objects.filter(user=self.user).aggregate(
            total_size=Sum("size_bytes"),
            total_count=Count("id"),
        )

        self.used_bytes = stats["total_size"] or 0
        self.file_count = stats["total_count"] or 0
        self.save(update_fields=["used_bytes", "file_count", "updated_at"])

    @classmethod
    def get_or_create_for_user(cls, user) -> "UserQuota":
        """Get or create quota for user with tier-based limits."""
        quota, created = cls.objects.get_or_create(user=user)

        if created:
            # Set limits based on tier
            tier_limits = {
                "free": (100 * 1024 * 1024, 100, 5 * 1024 * 1024),      # 100 MB, 100 files, 5 MB/file
                "beta": (500 * 1024 * 1024, 500, 10 * 1024 * 1024),     # 500 MB, 500 files, 10 MB/file
                "pro": (5 * 1024 * 1024 * 1024, 5000, 100 * 1024 * 1024),  # 5 GB, 5000 files, 100 MB/file
            }

            limits = tier_limits.get(user.tier, tier_limits["free"])
            quota.quota_bytes = limits[0]
            quota.max_files = limits[1]
            quota.max_file_size_bytes = limits[2]
            quota.save()

        return quota
