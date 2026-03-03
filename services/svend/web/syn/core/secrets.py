"""
Encrypted secret storage with AES-256 envelope encryption.

This module provides secure storage for sensitive data using Fernet (AES-256)
with envelope encryption pattern. Supports optional AWS KMS for enhanced security.

Features:
- AES-256 symmetric encryption via Fernet
- Envelope encryption (data encrypted with DEK, DEK encrypted with KEK)
- Tenant isolation for multi-tenant security
- Optional AWS KMS integration for enterprise deployments
- Automatic key rotation with zero-downtime
- Audit trail integration

Compliance: ISO 27001 A.10.1 (Cryptographic Controls)

Architecture:
    KEK (Key Encryption Key) -> Encrypts -> DEK (Data Encryption Key)
    DEK -> Encrypts -> Secret Value

    The KEK is stored in environment variable or AWS KMS.
    The DEK is unique per secret and stored encrypted in the database.
"""

import base64
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models, transaction
from django.utils import timezone

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────
# SECRET STORE MODEL
# ───────────────────────────────────────────────


class SecretStore(models.Model):
    """
    Encrypted secret storage with tenant isolation.

    Stores sensitive data (API keys, passwords, tokens) encrypted at rest
    using AES-256 via Fernet. Each secret has its own Data Encryption Key (DEK)
    which is itself encrypted by a Key Encryption Key (KEK).

    Compliance:
    - ISO 27001 A.10.1: Cryptographic controls for data protection
    - ISO 27001 A.9.4: Secret information management
    - SOC 2 CC6.1: Encryption of sensitive data
    """

    # Unique identifier for the secret
    name = models.CharField(
        max_length=255, db_index=True, help_text="Unique name for this secret (e.g., 'stripe_api_key')"
    )

    # Encrypted value (base64 encoded)
    value_encrypted = models.TextField(help_text="Encrypted secret value (base64 encoded)")

    # Encrypted Data Encryption Key (base64 encoded)
    dek_encrypted = models.TextField(help_text="Encrypted Data Encryption Key used to encrypt this secret")

    # Key version for rotation support
    kek_version = models.IntegerField(default=1, help_text="Version of the Key Encryption Key used")

    # Tenant isolation
    tenant_id = models.UUIDField(
        db_index=True, null=True, blank=True, help_text="Tenant identifier for multi-tenant isolation (SEC-001 §5.2)"
    )

    # Metadata
    created_at = models.DateTimeField(default=timezone.now, db_index=True, help_text="When this secret was created")

    updated_at = models.DateTimeField(auto_now=True, help_text="When this secret was last updated")

    created_by = models.CharField(max_length=255, blank=True, help_text="User or service that created this secret")

    last_rotated_at = models.DateTimeField(null=True, blank=True, help_text="When this secret was last rotated")

    rotation_schedule_days = models.IntegerField(default=90, help_text="How often to rotate this secret (in days)")

    # Secret metadata (non-sensitive)
    metadata = models.JSONField(default=dict, blank=True, help_text="Non-sensitive metadata about this secret")

    class Meta:
        app_label = "core"
        db_table = "core_secret_store"
        unique_together = [("name", "tenant_id")]
        indexes = [
            models.Index(fields=["tenant_id", "name"], name="secret_tenant_name"),
            models.Index(fields=["tenant_id", "created_at"], name="secret_tenant_time"),
            models.Index(fields=["last_rotated_at"], name="secret_rotation"),
        ]
        ordering = ["tenant_id", "name"]
        verbose_name = "Secret"
        verbose_name_plural = "Secrets"

    def __str__(self):
        return f"{self.name} ({self.tenant_id})"

    def clean(self):
        """Validate secret data."""
        super().clean()
        if not self.name or not self.name.strip():
            raise ValidationError("Secret name cannot be empty")
        if not self.tenant_id:
            raise ValidationError("Tenant ID is required")

    def needs_rotation(self) -> bool:
        """
        Check if this secret needs rotation based on schedule.

        Returns:
            bool: True if secret should be rotated
        """
        if not self.last_rotated_at:
            # Never rotated, use created_at
            reference_date = self.created_at
        else:
            reference_date = self.last_rotated_at

        rotation_due = reference_date + timedelta(days=self.rotation_schedule_days)
        return timezone.now() >= rotation_due


# ───────────────────────────────────────────────
# ENCRYPTION MANAGER
# ───────────────────────────────────────────────


class SecretEncryptionManager:
    """
    Manages encryption and decryption of secrets using envelope encryption.

    Envelope Encryption Pattern:
    1. Generate random DEK (Data Encryption Key) for each secret
    2. Encrypt secret value with DEK using Fernet (AES-256)
    3. Encrypt DEK with KEK (Key Encryption Key)
    4. Store encrypted DEK alongside encrypted value

    The KEK comes from:
    - Environment variable SECRET_ENCRYPTION_KEY (default)
    - AWS KMS (if configured)
    """

    def __init__(self):
        self.kek_version = 1
        self._kek_cache = {}
        self._kms_client = None

    def get_kek(self, version: int = 1) -> bytes:
        """
        Get the Key Encryption Key for a specific version.

        Args:
            version: KEK version number

        Returns:
            32-byte KEK

        Raises:
            ValueError: If KEK not found or invalid
        """
        if version in self._kek_cache:
            return self._kek_cache[version]

        # Try environment variable
        env_key = f"SECRET_ENCRYPTION_KEY_V{version}"
        kek_b64 = os.environ.get(env_key)

        if not kek_b64:
            # Fall back to default key
            kek_b64 = os.environ.get("SECRET_ENCRYPTION_KEY")

        if not kek_b64:
            raise ValueError(
                f"No encryption key found. Set {env_key} or SECRET_ENCRYPTION_KEY "
                "environment variable. Generate with: python -c "
                "'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
            )

        try:
            kek = base64.urlsafe_b64decode(kek_b64)
            if len(kek) != 32:
                raise ValueError("Encryption key must be 32 bytes")

            self._kek_cache[version] = kek
            return kek

        except Exception as e:
            raise ValueError(f"Invalid encryption key: {e}")

    def _get_kms_client(self):
        """Get AWS KMS client if configured."""
        if self._kms_client is not None:
            return self._kms_client

        try:
            import boto3

            kms_key_id = getattr(settings, "AWS_KMS_KEY_ID", None)
            if kms_key_id:
                self._kms_client = boto3.client("kms")
                logger.info("AWS KMS client initialized")
            else:
                self._kms_client = False  # Explicitly disabled
        except ImportError:
            logger.debug("boto3 not installed, KMS support unavailable")
            self._kms_client = False
        except Exception as e:
            logger.warning(f"Failed to initialize KMS client: {e}")
            self._kms_client = False

        return self._kms_client

    def generate_dek(self) -> bytes:
        """
        Generate a new Data Encryption Key.

        Returns:
            32-byte DEK suitable for Fernet
        """
        return Fernet.generate_key()

    def encrypt_dek(self, dek: bytes, version: int = 1) -> str:
        """
        Encrypt a Data Encryption Key with the Key Encryption Key.

        Args:
            dek: Data Encryption Key to encrypt
            version: KEK version to use

        Returns:
            Base64-encoded encrypted DEK
        """
        # Try KMS first if configured
        kms_client = self._get_kms_client()
        if kms_client:
            try:
                kms_key_id = settings.AWS_KMS_KEY_ID
                response = kms_client.encrypt(KeyId=kms_key_id, Plaintext=dek)
                encrypted_dek = base64.b64encode(response["CiphertextBlob"]).decode()
                logger.debug("Encrypted DEK with AWS KMS")
                return encrypted_dek
            except Exception as e:
                logger.warning(f"KMS encryption failed, falling back to Fernet: {e}")

        # Use Fernet with KEK
        kek = self.get_kek(version)
        fernet = Fernet(base64.urlsafe_b64encode(kek))
        encrypted_dek = fernet.encrypt(dek).decode()
        return encrypted_dek

    def decrypt_dek(self, encrypted_dek: str, version: int = 1) -> bytes:
        """
        Decrypt a Data Encryption Key.

        Args:
            encrypted_dek: Base64-encoded encrypted DEK
            version: KEK version used for encryption

        Returns:
            Decrypted DEK

        Raises:
            ValueError: If decryption fails
        """
        # Try KMS first if configured
        kms_client = self._get_kms_client()
        if kms_client:
            try:
                ciphertext_blob = base64.b64decode(encrypted_dek)
                response = kms_client.decrypt(CiphertextBlob=ciphertext_blob)
                dek = response["Plaintext"]
                logger.debug("Decrypted DEK with AWS KMS")
                return dek
            except Exception as e:
                logger.warning(f"KMS decryption failed, falling back to Fernet: {e}")

        # Use Fernet with KEK
        try:
            kek = self.get_kek(version)
            fernet = Fernet(base64.urlsafe_b64encode(kek))
            dek = fernet.decrypt(encrypted_dek.encode())
            return dek
        except InvalidToken:
            raise ValueError("Failed to decrypt DEK: invalid key or corrupted data")

    def encrypt_value(self, plaintext: str, dek: bytes) -> str:
        """
        Encrypt a secret value with a Data Encryption Key.

        Args:
            plaintext: Secret value to encrypt
            dek: Data Encryption Key

        Returns:
            Base64-encoded encrypted value
        """
        fernet = Fernet(dek)
        encrypted = fernet.encrypt(plaintext.encode())
        return encrypted.decode()

    def decrypt_value(self, encrypted: str, dek: bytes) -> str:
        """
        Decrypt a secret value with a Data Encryption Key.

        Args:
            encrypted: Base64-encoded encrypted value
            dek: Data Encryption Key

        Returns:
            Decrypted plaintext value

        Raises:
            ValueError: If decryption fails
        """
        try:
            fernet = Fernet(dek)
            plaintext = fernet.decrypt(encrypted.encode())
            return plaintext.decode()
        except InvalidToken:
            raise ValueError("Failed to decrypt value: invalid key or corrupted data")


# Singleton instance
_encryption_manager = SecretEncryptionManager()


# ───────────────────────────────────────────────
# PUBLIC API
# ───────────────────────────────────────────────


def set_secret(
    name: str,
    value: str,
    tenant_id: str,
    created_by: str = "",
    rotation_days: int = 90,
    metadata: Optional[Dict[str, Any]] = None,
) -> SecretStore:
    """
    Store an encrypted secret.

    Args:
        name: Unique name for the secret
        value: Secret value to encrypt
        tenant_id: Tenant identifier
        created_by: User or service creating the secret
        rotation_days: Days between automatic rotations
        metadata: Optional non-sensitive metadata

    Returns:
        SecretStore instance

    Example:
        >>> set_secret(
        ...     name="stripe_api_key",
        ...     value="sk_live_...",
        ...     tenant_id="acme_corp",
        ...     created_by="admin@acme.com"
        ... )
    """
    with transaction.atomic():
        # Generate new DEK for this secret
        dek = _encryption_manager.generate_dek()

        # Encrypt the secret value with DEK
        value_encrypted = _encryption_manager.encrypt_value(value, dek)

        # Encrypt the DEK with KEK
        dek_encrypted = _encryption_manager.encrypt_dek(dek)

        # Create or update the secret
        secret, created = SecretStore.objects.update_or_create(
            name=name,
            tenant_id=tenant_id,
            defaults={
                "value_encrypted": value_encrypted,
                "dek_encrypted": dek_encrypted,
                "kek_version": _encryption_manager.kek_version,
                "created_by": created_by,
                "rotation_schedule_days": rotation_days,
                "metadata": metadata or {},
            },
        )

        action = "created" if created else "updated"
        logger.info(
            f"Secret {action}",
            extra={"secret_name": name, "tenant_id": tenant_id, "created_by": created_by, "action": action},
        )

        # Log to audit trail if available
        try:
            from syn.audit.models import SysLogEntry

            SysLogEntry.objects.create(
                actor=created_by or "system",
                event_name=f"secret.{action}",
                payload={
                    "secret_name": name,
                    "tenant_id": str(tenant_id),
                    "kek_version": _encryption_manager.kek_version,
                },
                tenant_id=tenant_id,
            )
        except ImportError:
            pass  # Audit module not available

        return secret


def get_secret(name: str, tenant_id: str) -> str:
    """
    Retrieve and decrypt a secret.

    Args:
        name: Secret name
        tenant_id: Tenant identifier

    Returns:
        Decrypted secret value

    Raises:
        SecretStore.DoesNotExist: If secret not found
        ValueError: If decryption fails

    Example:
        >>> api_key = get_secret("stripe_api_key", "acme_corp")
        >>> print(api_key)  # sk_live_...
    """
    try:
        secret = SecretStore.objects.get(name=name, tenant_id=tenant_id)
    except SecretStore.DoesNotExist:
        logger.error(f"Secret not found: {name}", extra={"secret_name": name, "tenant_id": tenant_id})
        raise

    # Decrypt DEK with KEK
    dek = _encryption_manager.decrypt_dek(secret.dek_encrypted, version=secret.kek_version)

    # Decrypt value with DEK
    plaintext = _encryption_manager.decrypt_value(secret.value_encrypted, dek)

    logger.debug(f"Secret retrieved: {name}", extra={"secret_name": name, "tenant_id": tenant_id})

    return plaintext


def rotate_secret(name: str, tenant_id: str, new_value: Optional[str] = None) -> SecretStore:
    """
    Rotate a secret by re-encrypting with a new DEK.

    Args:
        name: Secret name
        tenant_id: Tenant identifier
        new_value: Optional new value (if None, re-encrypts existing value)

    Returns:
        Updated SecretStore instance

    Example:
        >>> # Re-encrypt with new DEK (same value)
        >>> rotate_secret("stripe_api_key", "acme_corp")

        >>> # Update value and re-encrypt
        >>> rotate_secret("stripe_api_key", "acme_corp", "sk_live_new...")
    """
    with transaction.atomic():
        secret = SecretStore.objects.select_for_update().get(name=name, tenant_id=tenant_id)

        # Get current value if not providing new one
        if new_value is None:
            current_value = get_secret(name, tenant_id)
            new_value = current_value

        # Generate new DEK
        new_dek = _encryption_manager.generate_dek()

        # Encrypt value with new DEK
        value_encrypted = _encryption_manager.encrypt_value(new_value, new_dek)

        # Encrypt new DEK with current KEK
        dek_encrypted = _encryption_manager.encrypt_dek(new_dek)

        # Update secret
        secret.value_encrypted = value_encrypted
        secret.dek_encrypted = dek_encrypted
        secret.kek_version = _encryption_manager.kek_version
        secret.last_rotated_at = timezone.now()
        secret.save()

        logger.info(f"Secret rotated: {name}", extra={"secret_name": name, "tenant_id": tenant_id})

        # Log to audit trail
        try:
            from syn.audit.models import SysLogEntry

            SysLogEntry.objects.create(
                actor="system",
                event_name="secret.rotated",
                payload={
                    "secret_name": name,
                    "tenant_id": str(tenant_id),
                    "kek_version": secret.kek_version,
                },
                tenant_id=tenant_id,
            )
        except ImportError:
            pass

        return secret


def delete_secret(name: str, tenant_id: str) -> None:
    """
    Delete a secret.

    Args:
        name: Secret name
        tenant_id: Tenant identifier

    Raises:
        SecretStore.DoesNotExist: If secret not found
    """
    secret = SecretStore.objects.get(name=name, tenant_id=tenant_id)
    secret.delete()

    logger.info(f"Secret deleted: {name}", extra={"secret_name": name, "tenant_id": tenant_id})

    # Log to audit trail
    try:
        from syn.audit.models import SysLogEntry

        SysLogEntry.objects.create(
            actor="system",
            event_name="secret.deleted",
            payload={
                "secret_name": name,
                "tenant_id": str(tenant_id),
            },
            tenant_id=tenant_id,
        )
    except ImportError:
        pass


def list_secrets(tenant_id: str) -> models.QuerySet:
    """
    List all secrets for a tenant (without decrypting values).

    Args:
        tenant_id: Tenant identifier

    Returns:
        QuerySet of SecretStore instances
    """
    return SecretStore.objects.filter(tenant_id=tenant_id)


def rotate_all_keys(old_kek_version: int = 1, new_kek_version: int = 2) -> int:
    """
    Rotate encryption keys for all secrets.

    Re-encrypts all DEKs with a new KEK version. This is used when
    rotating the master encryption key.

    Args:
        old_kek_version: Current KEK version
        new_kek_version: New KEK version

    Returns:
        Number of secrets rotated

    Note:
        This operation is safe to run multiple times and supports
        zero-downtime key rotation.
    """
    rotated_count = 0

    # Update manager to use new version
    _encryption_manager.kek_version = new_kek_version

    secrets = SecretStore.objects.filter(kek_version=old_kek_version)

    for secret in secrets:
        try:
            with transaction.atomic():
                # Decrypt DEK with old KEK
                dek = _encryption_manager.decrypt_dek(secret.dek_encrypted, version=old_kek_version)

                # Re-encrypt DEK with new KEK
                new_dek_encrypted = _encryption_manager.encrypt_dek(dek, version=new_kek_version)

                # Update secret
                secret.dek_encrypted = new_dek_encrypted
                secret.kek_version = new_kek_version
                secret.save()

                rotated_count += 1

                logger.info(
                    f"Rotated key for secret: {secret.name}",
                    extra={
                        "secret_name": secret.name,
                        "tenant_id": secret.tenant_id,
                        "old_version": old_kek_version,
                        "new_version": new_kek_version,
                    },
                )

        except Exception as e:
            logger.error(
                f"Failed to rotate key for secret {secret.name}: {e}",
                extra={"secret_name": secret.name, "tenant_id": secret.tenant_id},
                exc_info=True,
            )

    logger.info(f"Key rotation complete: {rotated_count} secrets rotated")
    return rotated_count
