"""
Management command for rotating encryption keys.

Usage:
    python manage.py rotate_keys --old-version 1 --new-version 2
    python manage.py rotate_keys --rotate-secrets  # Rotate DEKs only

This command supports two types of rotation:

1. KEK Rotation (--old-version, --new-version):
   Rotates the Key Encryption Key by re-encrypting all DEKs
   with a new KEK version.

2. Secret Rotation (--rotate-secrets):
   Rotates individual secrets by generating new DEKs for secrets
   that are due for rotation based on their schedule.

Compliance: ISO 27001 A.10.1.2 (Key Management)
"""

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

from syn.core.secrets import (
    SecretStore,
    rotate_all_keys,
    rotate_secret,
)


class Command(BaseCommand):
    help = "Rotate encryption keys for secrets"

    def add_arguments(self, parser):
        parser.add_argument("--old-version", type=int, help="Old KEK version to rotate from")
        parser.add_argument("--new-version", type=int, help="New KEK version to rotate to")
        parser.add_argument(
            "--rotate-secrets", action="store_true", help="Rotate secrets that are due for rotation (based on schedule)"
        )
        parser.add_argument("--tenant", type=str, help="Only rotate secrets for specific tenant")
        parser.add_argument("--force", action="store_true", help="Force rotation even if not due")
        parser.add_argument("--dry-run", action="store_true", help="Show what would be rotated without making changes")

    def handle(self, *args, **options):
        """Execute the key rotation command."""
        old_version = options.get("old_version")
        new_version = options.get("new_version")
        rotate_secrets_flag = options.get("rotate_secrets")
        tenant_id = options.get("tenant")
        force = options.get("force")
        dry_run = options.get("dry_run")

        # Validate arguments
        if not rotate_secrets_flag and (not old_version or not new_version):
            raise CommandError("Must specify either --rotate-secrets or both " "--old-version and --new-version")

        if rotate_secrets_flag and (old_version or new_version):
            raise CommandError("Cannot specify both --rotate-secrets and KEK version arguments")

        # Perform KEK rotation
        if old_version and new_version:
            self.rotate_kek(old_version, new_version, dry_run)

        # Perform secret rotation
        if rotate_secrets_flag:
            self.rotate_secrets(tenant_id, force, dry_run)

    def rotate_kek(self, old_version: int, new_version: int, dry_run: bool):
        """
        Rotate Key Encryption Key.

        Args:
            old_version: Old KEK version
            new_version: New KEK version
            dry_run: If True, don't make changes
        """
        self.stdout.write(self.style.WARNING(f"Rotating KEK from version {old_version} to {new_version}..."))

        # Count secrets to rotate
        secrets = SecretStore.objects.filter(kek_version=old_version)
        count = secrets.count()

        if count == 0:
            self.stdout.write(self.style.SUCCESS(f"No secrets found with KEK version {old_version}"))
            return

        self.stdout.write(f"Found {count} secrets to rotate")

        if dry_run:
            self.stdout.write(self.style.WARNING("[DRY RUN] Would rotate the following secrets:"))
            for secret in secrets[:10]:  # Show first 10
                self.stdout.write(f"  - {secret.name} ({secret.tenant_id})")
            if count > 10:
                self.stdout.write(f"  ... and {count - 10} more")
            return

        # Confirm before proceeding
        if not self._confirm(f"Rotate {count} secrets?"):
            self.stdout.write(self.style.WARNING("Aborted"))
            return

        # Perform rotation
        try:
            rotated_count = rotate_all_keys(old_version, new_version)
            self.stdout.write(self.style.SUCCESS(f"Successfully rotated {rotated_count} secrets"))
        except Exception as e:
            raise CommandError(f"Key rotation failed: {e}")

    def rotate_secrets(self, tenant_id: str, force: bool, dry_run: bool):
        """
        Rotate individual secrets based on rotation schedule.

        Args:
            tenant_id: Optional tenant filter
            force: Force rotation even if not due
            dry_run: If True, don't make changes
        """
        self.stdout.write(self.style.WARNING("Rotating secrets based on rotation schedule..."))

        # Build query
        query = SecretStore.objects.all()
        if tenant_id:
            query = query.filter(tenant_id=tenant_id)

        # Find secrets that need rotation
        secrets_to_rotate = []
        for secret in query:
            if force or secret.needs_rotation():
                secrets_to_rotate.append(secret)

        if not secrets_to_rotate:
            self.stdout.write(self.style.SUCCESS("No secrets need rotation at this time"))
            return

        count = len(secrets_to_rotate)
        self.stdout.write(f"Found {count} secrets to rotate")

        if dry_run:
            self.stdout.write(self.style.WARNING("[DRY RUN] Would rotate the following secrets:"))
            for secret in secrets_to_rotate[:10]:
                days_since = (timezone.now() - (secret.last_rotated_at or secret.created_at)).days
                self.stdout.write(
                    f"  - {secret.name} ({secret.tenant_id}) - "
                    f"{days_since} days old (schedule: {secret.rotation_schedule_days} days)"
                )
            if count > 10:
                self.stdout.write(f"  ... and {count - 10} more")
            return

        # Confirm before proceeding
        if not force and not self._confirm(f"Rotate {count} secrets?"):
            self.stdout.write(self.style.WARNING("Aborted"))
            return

        # Perform rotation
        rotated = 0
        failed = 0

        for secret in secrets_to_rotate:
            try:
                rotate_secret(secret.name, secret.tenant_id)
                rotated += 1
                self.stdout.write(self.style.SUCCESS(f"✓ Rotated: {secret.name} ({secret.tenant_id})"))
            except Exception as e:
                failed += 1
                self.stdout.write(self.style.ERROR(f"✗ Failed: {secret.name} ({secret.tenant_id}) - {e}"))

        # Summary
        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS(f"Rotated: {rotated}"))
        if failed > 0:
            self.stdout.write(self.style.ERROR(f"Failed: {failed}"))

    def _confirm(self, message: str) -> bool:
        """
        Ask user for confirmation.

        Args:
            message: Confirmation message

        Returns:
            True if user confirms
        """
        response = input(f"{message} [y/N]: ")
        return response.lower() in ["y", "yes"]
