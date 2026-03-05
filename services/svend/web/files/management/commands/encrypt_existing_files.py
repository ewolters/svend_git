"""One-time management command to encrypt existing unencrypted files on disk.

Usage:
    python manage.py encrypt_existing_files --dry-run  # Preview
    python manage.py encrypt_existing_files             # Encrypt
"""

import os

from django.core.management.base import BaseCommand

from core.encryption import encrypt_bytes
from files.models import UserFile


class Command(BaseCommand):
    help = "Encrypt existing unencrypted files on disk using Fernet"

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be encrypted without making changes",
        )

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        files = UserFile.objects.all()
        total = files.count()
        encrypted = 0
        skipped = 0
        errors = 0

        self.stdout.write(f"Processing {total} files{'  (DRY RUN)' if dry_run else ''}...")

        for uf in files:
            try:
                if not uf.file or not os.path.exists(uf.file.path):
                    skipped += 1
                    continue

                with open(uf.file.path, "rb") as f:
                    raw = f.read()

                # Check if already encrypted (Fernet tokens start with 'gAAAAA')
                if raw.startswith(b"gAAAAA"):
                    skipped += 1
                    continue

                if dry_run:
                    self.stdout.write(f"  Would encrypt: {uf.original_name} ({len(raw)} bytes)")
                    encrypted += 1
                    continue

                ciphertext = encrypt_bytes(raw)
                with open(uf.file.path, "wb") as f:
                    f.write(ciphertext)

                encrypted += 1

            except Exception as e:
                errors += 1
                self.stderr.write(f"  Error processing {uf.original_name}: {e}")

        self.stdout.write(f"\nDone: {encrypted} encrypted, {skipped} skipped, {errors} errors")
