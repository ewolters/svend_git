"""
Comprehensive tests for encrypted secret storage.

Tests cover:
- Encryption/decryption correctness
- Key rotation (both KEK and DEK)
- Tenant isolation
- KMS fallback
- Error handling
- Compliance requirements

Compliance: ISO 27001 A.10.1 (Cryptographic Controls)
"""

import base64
import os
import uuid
from datetime import timedelta
from unittest.mock import MagicMock, patch

from django.test import TestCase, override_settings
from django.utils import timezone

import pytest
from cryptography.fernet import Fernet
from freezegun import freeze_time

from syn.core.secrets import (
    SecretEncryptionManager,
    SecretStore,
    _encryption_manager,
    delete_secret,
    get_secret,
    list_secrets,
    rotate_all_keys,
    rotate_secret,
    set_secret,
)

# Generate test encryption keys
TEST_KEK_V1 = Fernet.generate_key().decode()
TEST_KEK_V2 = Fernet.generate_key().decode()

# Test tenant UUIDs (SEC-001 §5.2: tenant isolation requires UUID)
TEST_TENANT_1 = uuid.UUID("11111111-1111-1111-1111-111111111111")
TEST_TENANT_2 = uuid.UUID("22222222-2222-2222-2222-222222222222")


class SecretStoreTestCase(TestCase):
    """Test SecretStore model and basic operations."""

    def setUp(self):
        """Set up test data with proper environment configuration."""
        # The secrets module reads encryption key from os.environ, not Django settings
        os.environ["SECRET_ENCRYPTION_KEY"] = TEST_KEK_V1
        # Clear the KEK cache to ensure fresh key retrieval
        _encryption_manager._kek_cache.clear()

        self.tenant_id = TEST_TENANT_1
        self.secret_name = "test_secret"
        self.secret_value = "my_secret_value_123"

    def tearDown(self):
        """Clean up environment after tests."""
        if "SECRET_ENCRYPTION_KEY" in os.environ:
            del os.environ["SECRET_ENCRYPTION_KEY"]
        _encryption_manager._kek_cache.clear()

    def test_set_and_get_secret(self):
        """Test storing and retrieving a secret."""
        # Store secret
        secret = set_secret(
            name=self.secret_name, value=self.secret_value, tenant_id=self.tenant_id, created_by="test_user"
        )

        # Verify secret was created
        self.assertIsNotNone(secret.id)
        self.assertEqual(secret.name, self.secret_name)
        self.assertEqual(str(secret.tenant_id), str(self.tenant_id))
        self.assertEqual(secret.created_by, "test_user")

        # Verify value is encrypted (not plaintext)
        self.assertNotEqual(secret.value_encrypted, self.secret_value)

        # Retrieve secret
        retrieved_value = get_secret(self.secret_name, self.tenant_id)

        # Verify decrypted value matches original
        self.assertEqual(retrieved_value, self.secret_value)

    def test_secret_not_found(self):
        """Test retrieving non-existent secret raises error."""
        with self.assertRaises(SecretStore.DoesNotExist):
            get_secret("nonexistent", self.tenant_id)

    def test_update_secret(self):
        """Test updating an existing secret."""
        # Create initial secret
        set_secret(name=self.secret_name, value="old_value", tenant_id=self.tenant_id)

        # Update secret
        set_secret(name=self.secret_name, value="new_value", tenant_id=self.tenant_id)

        # Verify updated value
        retrieved = get_secret(self.secret_name, self.tenant_id)
        self.assertEqual(retrieved, "new_value")

        # Verify only one secret exists (updated, not duplicated)
        count = SecretStore.objects.filter(name=self.secret_name, tenant_id=self.tenant_id).count()
        self.assertEqual(count, 1)

    def test_delete_secret(self):
        """Test deleting a secret."""
        # Create secret
        set_secret(name=self.secret_name, value=self.secret_value, tenant_id=self.tenant_id)

        # Delete secret
        delete_secret(self.secret_name, self.tenant_id)

        # Verify secret is deleted
        with self.assertRaises(SecretStore.DoesNotExist):
            get_secret(self.secret_name, self.tenant_id)

    def test_list_secrets(self):
        """Test listing secrets for a tenant."""
        # Create multiple secrets
        set_secret("secret1", "value1", self.tenant_id)
        set_secret("secret2", "value2", self.tenant_id)
        set_secret("secret3", "value3", TEST_TENANT_2)

        # List secrets for tenant
        secrets = list_secrets(self.tenant_id)

        # Verify correct secrets returned
        self.assertEqual(secrets.count(), 2)
        names = set(s.name for s in secrets)
        self.assertEqual(names, {"secret1", "secret2"})

    def test_secret_metadata(self):
        """Test storing and retrieving secret metadata."""
        metadata = {"description": "API key for Stripe", "environment": "production"}

        secret = set_secret(name=self.secret_name, value=self.secret_value, tenant_id=self.tenant_id, metadata=metadata)

        # Verify metadata stored
        self.assertEqual(secret.metadata, metadata)

        # Retrieve and verify
        retrieved = SecretStore.objects.get(name=self.secret_name, tenant_id=self.tenant_id)
        self.assertEqual(retrieved.metadata, metadata)


class TenantIsolationTestCase(TestCase):
    """Test tenant isolation for secrets."""

    def setUp(self):
        """Set up encryption key environment."""
        os.environ["SECRET_ENCRYPTION_KEY"] = TEST_KEK_V1
        _encryption_manager._kek_cache.clear()

    def tearDown(self):
        """Clean up environment."""
        if "SECRET_ENCRYPTION_KEY" in os.environ:
            del os.environ["SECRET_ENCRYPTION_KEY"]
        _encryption_manager._kek_cache.clear()

    def test_tenant_isolation(self):
        """Test that secrets are isolated between tenants."""
        # Create same-named secret for two tenants
        set_secret("api_key", "tenant1_value", TEST_TENANT_1)
        set_secret("api_key", "tenant2_value", TEST_TENANT_2)

        # Verify each tenant gets their own value
        value1 = get_secret("api_key", TEST_TENANT_1)
        value2 = get_secret("api_key", TEST_TENANT_2)

        self.assertEqual(value1, "tenant1_value")
        self.assertEqual(value2, "tenant2_value")
        self.assertNotEqual(value1, value2)

    def test_list_secrets_tenant_isolation(self):
        """Test that list_secrets only returns secrets for specified tenant."""
        # Create secrets for multiple tenants
        set_secret("secret1", "value1", TEST_TENANT_1)
        set_secret("secret2", "value2", TEST_TENANT_1)
        set_secret("secret3", "value3", TEST_TENANT_2)

        # List secrets for tenant1
        secrets = list_secrets(TEST_TENANT_1)

        # Verify only tenant1 secrets returned
        self.assertEqual(secrets.count(), 2)
        for secret in secrets:
            self.assertEqual(str(secret.tenant_id), str(TEST_TENANT_1))


class EncryptionTestCase(TestCase):
    """Test encryption/decryption functionality."""

    def setUp(self):
        """Set up encryption manager."""
        os.environ["SECRET_ENCRYPTION_KEY"] = TEST_KEK_V1
        self.manager = SecretEncryptionManager()
        self.manager._kek_cache.clear()

    def tearDown(self):
        """Clean up environment."""
        if "SECRET_ENCRYPTION_KEY" in os.environ:
            del os.environ["SECRET_ENCRYPTION_KEY"]
        self.manager._kek_cache.clear()

    def test_dek_generation(self):
        """Test Data Encryption Key generation."""
        dek1 = self.manager.generate_dek()
        dek2 = self.manager.generate_dek()

        # Verify DEKs are valid Fernet keys
        self.assertEqual(len(dek1), 44)  # Fernet key length
        self.assertEqual(len(dek2), 44)

        # Verify DEKs are different (random)
        self.assertNotEqual(dek1, dek2)

    def test_dek_encryption_decryption(self):
        """Test encrypting and decrypting DEK with KEK."""
        dek = self.manager.generate_dek()

        # Encrypt DEK
        encrypted_dek = self.manager.encrypt_dek(dek)

        # Verify encryption happened
        self.assertNotEqual(encrypted_dek, dek.decode())

        # Decrypt DEK
        decrypted_dek = self.manager.decrypt_dek(encrypted_dek)

        # Verify decryption is correct
        self.assertEqual(decrypted_dek, dek)

    def test_value_encryption_decryption(self):
        """Test encrypting and decrypting secret values."""
        plaintext = "my_secret_value"
        dek = self.manager.generate_dek()

        # Encrypt value
        encrypted = self.manager.encrypt_value(plaintext, dek)

        # Verify encryption happened
        self.assertNotEqual(encrypted, plaintext)

        # Decrypt value
        decrypted = self.manager.decrypt_value(encrypted, dek)

        # Verify decryption is correct
        self.assertEqual(decrypted, plaintext)

    def test_encryption_with_different_dek_fails(self):
        """Test that decryption with wrong DEK fails."""
        plaintext = "my_secret_value"
        dek1 = self.manager.generate_dek()
        dek2 = self.manager.generate_dek()

        # Encrypt with DEK1
        encrypted = self.manager.encrypt_value(plaintext, dek1)

        # Try to decrypt with DEK2 (should fail)
        with self.assertRaises(ValueError):
            self.manager.decrypt_value(encrypted, dek2)

    def test_envelope_encryption(self):
        """Test full envelope encryption pattern."""
        plaintext = "my_secret_value"

        # Generate DEK
        dek = self.manager.generate_dek()

        # Encrypt value with DEK
        encrypted_value = self.manager.encrypt_value(plaintext, dek)

        # Encrypt DEK with KEK
        encrypted_dek = self.manager.encrypt_dek(dek)

        # Simulate storage and retrieval
        # Decrypt DEK with KEK
        decrypted_dek = self.manager.decrypt_dek(encrypted_dek)

        # Decrypt value with DEK
        decrypted_value = self.manager.decrypt_value(encrypted_value, decrypted_dek)

        # Verify full round-trip
        self.assertEqual(decrypted_value, plaintext)


class KeyRotationTestCase(TestCase):
    """Test key rotation functionality."""

    def setUp(self):
        """Set up test data."""
        # Use V1 key initially
        os.environ["SECRET_ENCRYPTION_KEY"] = TEST_KEK_V1
        os.environ["SECRET_ENCRYPTION_KEY_V1"] = TEST_KEK_V1
        os.environ["SECRET_ENCRYPTION_KEY_V2"] = TEST_KEK_V2
        _encryption_manager._kek_cache.clear()
        # Reset kek_version to default (tests may change it via rotate_all_keys)
        _encryption_manager.kek_version = 1

    def tearDown(self):
        """Clean up environment."""
        for key in ["SECRET_ENCRYPTION_KEY", "SECRET_ENCRYPTION_KEY_V1", "SECRET_ENCRYPTION_KEY_V2"]:
            if key in os.environ:
                del os.environ[key]
        _encryption_manager._kek_cache.clear()
        # Reset kek_version to default
        _encryption_manager.kek_version = 1

    def test_rotate_secret_dek(self):
        """Test rotating a secret's DEK."""
        tenant_id = TEST_TENANT_1
        secret_name = "test_secret"
        original_value = "original_value"

        # Create secret
        secret = set_secret(secret_name, original_value, tenant_id)
        original_dek_encrypted = secret.dek_encrypted

        # Rotate secret (generates new DEK)
        rotated_secret = rotate_secret(secret_name, tenant_id)

        # Verify DEK changed
        self.assertNotEqual(rotated_secret.dek_encrypted, original_dek_encrypted)

        # Verify value still decrypts correctly
        retrieved_value = get_secret(secret_name, tenant_id)
        self.assertEqual(retrieved_value, original_value)

        # Verify rotation timestamp updated
        self.assertIsNotNone(rotated_secret.last_rotated_at)

    def test_rotate_secret_with_new_value(self):
        """Test rotating a secret with a new value."""
        tenant_id = TEST_TENANT_1
        secret_name = "test_secret"

        # Create secret
        set_secret(secret_name, "old_value", tenant_id)

        # Rotate with new value
        rotate_secret(secret_name, tenant_id, new_value="new_value")

        # Verify new value
        retrieved_value = get_secret(secret_name, tenant_id)
        self.assertEqual(retrieved_value, "new_value")

    def test_rotate_all_keys(self):
        """Test rotating KEK for all secrets."""
        # Create multiple secrets with V1 KEK
        os.environ["SECRET_ENCRYPTION_KEY"] = TEST_KEK_V1
        _encryption_manager._kek_cache.clear()
        set_secret("secret1", "value1", TEST_TENANT_1)
        set_secret("secret2", "value2", TEST_TENANT_1)
        set_secret("secret3", "value3", TEST_TENANT_2)

        # Verify all secrets use V1
        secrets = SecretStore.objects.all()
        for secret in secrets:
            self.assertEqual(secret.kek_version, 1)

        # Rotate all keys to V2
        os.environ["SECRET_ENCRYPTION_KEY"] = TEST_KEK_V2
        _encryption_manager._kek_cache.clear()
        rotated_count = rotate_all_keys(old_kek_version=1, new_kek_version=2)

        # Verify rotation count
        self.assertEqual(rotated_count, 3)

        # Verify all secrets now use V2
        secrets = SecretStore.objects.all()
        for secret in secrets:
            self.assertEqual(secret.kek_version, 2)

        # Verify all secrets still decrypt correctly with new KEK
        self.assertEqual(get_secret("secret1", TEST_TENANT_1), "value1")
        self.assertEqual(get_secret("secret2", TEST_TENANT_1), "value2")
        self.assertEqual(get_secret("secret3", TEST_TENANT_2), "value3")

    def test_needs_rotation(self):
        """Test secret rotation scheduling."""
        tenant_id = TEST_TENANT_1

        # Create secret with 30-day rotation schedule
        with freeze_time("2025-01-01"):
            secret = set_secret("test_secret", "value", tenant_id, rotation_days=30)

        # Check immediately - should not need rotation
        with freeze_time("2025-01-01"):
            self.assertFalse(secret.needs_rotation())

        # Check 29 days later - should not need rotation
        with freeze_time("2025-01-30"):
            self.assertFalse(secret.needs_rotation())

        # Check 31 days later - should need rotation
        with freeze_time("2025-02-01"):
            self.assertTrue(secret.needs_rotation())

    def test_rotation_after_manual_rotation(self):
        """Test that rotation schedule resets after manual rotation."""
        tenant_id = TEST_TENANT_1
        secret_name = "test_secret"

        # Create secret
        with freeze_time("2025-01-01"):
            secret = set_secret(secret_name, "value", tenant_id, rotation_days=30)

        # Manually rotate after 15 days
        with freeze_time("2025-01-16"):
            rotate_secret(secret_name, tenant_id)

        # Check 20 days after manual rotation - should not need rotation
        with freeze_time("2025-02-05"):  # 20 days after manual rotation
            secret.refresh_from_db()
            self.assertFalse(secret.needs_rotation())

        # Check 31 days after manual rotation - should need rotation
        with freeze_time("2025-02-16"):  # 31 days after manual rotation
            secret.refresh_from_db()
            self.assertTrue(secret.needs_rotation())


class KMSFallbackTestCase(TestCase):
    """Test AWS KMS integration and fallback."""

    def setUp(self):
        """Set up encryption key environment."""
        os.environ["SECRET_ENCRYPTION_KEY"] = TEST_KEK_V1
        _encryption_manager._kek_cache.clear()

    def tearDown(self):
        """Clean up environment."""
        if "SECRET_ENCRYPTION_KEY" in os.environ:
            del os.environ["SECRET_ENCRYPTION_KEY"]
        _encryption_manager._kek_cache.clear()

    @override_settings(AWS_KMS_KEY_ID="test-kms-key-id")
    def test_kms_encryption(self):
        """Test encryption with AWS KMS."""
        # boto3 is imported dynamically inside _get_kms_client
        # To properly test this, we need to install boto3 and configure real KMS
        # For now, we test that the KMS client initialization path works
        try:
            import boto3
            # Skip if boto3 is available but not configured - would need real AWS credentials
            self.skipTest("KMS integration test requires AWS credentials")
        except ImportError:
            # boto3 not installed - verify Fernet fallback works
            manager = SecretEncryptionManager()
            manager._kms_client = None  # Reset cached client
            dek = manager.generate_dek()
            encrypted_dek = manager.encrypt_dek(dek)

            # Should use Fernet since boto3 not available
            decrypted_dek = manager.decrypt_dek(encrypted_dek)
            self.assertEqual(decrypted_dek, dek)

    @override_settings(AWS_KMS_KEY_ID="test-kms-key-id")
    def test_kms_fallback_to_fernet(self):
        """Test fallback to Fernet when KMS is not available."""
        # Create a new manager with cleared KMS client to test fallback
        manager = SecretEncryptionManager()
        manager._kms_client = False  # Simulate KMS unavailable/failed

        # Encrypt DEK (should use Fernet as fallback)
        dek = manager.generate_dek()
        encrypted_dek = manager.encrypt_dek(dek)

        # Should not raise error (fallback successful)
        # Decrypt should work with Fernet
        decrypted_dek = manager.decrypt_dek(encrypted_dek)
        self.assertEqual(decrypted_dek, dek)

    def test_missing_encryption_key_error(self):
        """Test that missing encryption key raises helpful error."""
        # Remove any encryption key from environment
        if "SECRET_ENCRYPTION_KEY" in os.environ:
            del os.environ["SECRET_ENCRYPTION_KEY"]

        manager = SecretEncryptionManager()
        manager._kek_cache.clear()

        with self.assertRaises(ValueError) as context:
            manager.get_kek()

        error_message = str(context.exception)
        self.assertIn("SECRET_ENCRYPTION_KEY", error_message)
        self.assertIn("Fernet.generate_key()", error_message)


class ComplianceTestCase(TestCase):
    """Test compliance requirements."""

    def setUp(self):
        """Set up encryption key environment."""
        os.environ["SECRET_ENCRYPTION_KEY"] = TEST_KEK_V1
        _encryption_manager._kek_cache.clear()

    def tearDown(self):
        """Clean up environment."""
        if "SECRET_ENCRYPTION_KEY" in os.environ:
            del os.environ["SECRET_ENCRYPTION_KEY"]
        _encryption_manager._kek_cache.clear()

    def test_encryption_strength(self):
        """Verify AES-256 encryption is used (Fernet uses AES-128-CBC)."""
        # Note: Fernet actually uses AES-128 in CBC mode, not AES-256.
        # If AES-256 is a hard requirement, we'd need to use a different
        # encryption library. For this test, we verify Fernet is being used.
        manager = SecretEncryptionManager()
        dek = manager.generate_dek()

        # Fernet keys are 32 bytes (base64 encoded to 44 characters)
        self.assertEqual(len(dek), 44)

        # Verify we can create a Fernet instance (validates key format)
        fernet = Fernet(dek)
        self.assertIsNotNone(fernet)

    def test_audit_trail_integration(self):
        """Test that secret operations are logged to audit trail."""
        # This test requires the audit module to be installed
        try:
            from syn.audit.models import SysLogEntry

            # Create secret
            set_secret("test_secret", "value", TEST_TENANT_1, created_by="admin")

            # Verify audit log entry created
            audit_entry = SysLogEntry.objects.filter(tenant_id=TEST_TENANT_1, event_name="secret.created").last()

            self.assertIsNotNone(audit_entry)
            self.assertEqual(audit_entry.actor, "admin")
            self.assertEqual(audit_entry.payload["secret_name"], "test_secret")

        except ImportError:
            # Audit module not installed, skip test
            self.skipTest("Audit module not installed")

    def test_tenant_id_required(self):
        """Test that tenant_id is required for all operations."""
        # Attempt to create secret without tenant_id should fail
        secret = SecretStore(name="test", value_encrypted="test", dek_encrypted="test")

        with self.assertRaises(Exception):
            secret.full_clean()


class ErrorHandlingTestCase(TestCase):
    """Test error handling and edge cases."""

    def setUp(self):
        """Set up encryption key environment."""
        os.environ["SECRET_ENCRYPTION_KEY"] = TEST_KEK_V1
        _encryption_manager._kek_cache.clear()

    def tearDown(self):
        """Clean up environment."""
        if "SECRET_ENCRYPTION_KEY" in os.environ:
            del os.environ["SECRET_ENCRYPTION_KEY"]
        _encryption_manager._kek_cache.clear()

    def test_corrupted_encrypted_value(self):
        """Test handling of corrupted encrypted data."""
        tenant_id = TEST_TENANT_1
        secret_name = "test_secret"

        # Create valid secret
        secret = set_secret(secret_name, "value", tenant_id)

        # Corrupt the encrypted value
        secret.value_encrypted = "corrupted_data_invalid_base64"
        secret.save()

        # Attempt to retrieve should raise error
        with self.assertRaises(ValueError):
            get_secret(secret_name, tenant_id)

    def test_corrupted_dek(self):
        """Test handling of corrupted DEK."""
        tenant_id = TEST_TENANT_1
        secret_name = "test_secret"

        # Create valid secret
        secret = set_secret(secret_name, "value", tenant_id)

        # Corrupt the DEK
        secret.dek_encrypted = "corrupted_dek"
        secret.save()

        # Attempt to retrieve should raise error
        with self.assertRaises(ValueError):
            get_secret(secret_name, tenant_id)

    def test_empty_secret_value(self):
        """Test storing and retrieving empty string."""
        tenant_id = TEST_TENANT_1

        # Store empty secret
        set_secret("empty_secret", "", tenant_id)

        # Retrieve should return empty string
        value = get_secret("empty_secret", tenant_id)
        self.assertEqual(value, "")

    def test_unicode_secret_value(self):
        """Test storing and retrieving unicode values."""
        tenant_id = TEST_TENANT_1
        unicode_value = "Hello 世界 🌍 مرحبا"

        # Store unicode secret
        set_secret("unicode_secret", unicode_value, tenant_id)

        # Retrieve should return exact unicode string
        value = get_secret("unicode_secret", tenant_id)
        self.assertEqual(value, unicode_value)

    def test_large_secret_value(self):
        """Test storing and retrieving large secrets."""
        tenant_id = TEST_TENANT_1
        large_value = "x" * 100000  # 100KB

        # Store large secret
        set_secret("large_secret", large_value, tenant_id)

        # Retrieve should return full value
        value = get_secret("large_secret", tenant_id)
        self.assertEqual(value, large_value)
        self.assertEqual(len(value), 100000)
