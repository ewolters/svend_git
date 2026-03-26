"""Field-level encryption for sensitive data at rest.

Uses Fernet (AES-128-CBC + HMAC-SHA256) from the cryptography library.
Custom Django model fields encrypt on save, decrypt on read.
"""

import hashlib
import json
import logging

from cryptography.fernet import Fernet, InvalidToken
from django.conf import settings
from django.db import models

logger = logging.getLogger(__name__)

_fernet_instance = None


def _get_fernet() -> Fernet:
    """Get or create a cached Fernet instance from the configured key."""
    global _fernet_instance
    if _fernet_instance is None:
        key = getattr(settings, "FIELD_ENCRYPTION_KEY", "")
        if not key:
            raise ValueError(
                "FIELD_ENCRYPTION_KEY is not set. "
                "Ensure SVEND_FIELD_ENCRYPTION_KEY is exported in the environment."
            )
        _fernet_instance = Fernet(key.encode() if isinstance(key, str) else key)
    return _fernet_instance


def encrypt_str(plaintext: str) -> str:
    """Encrypt a string, returning a base64-encoded Fernet token."""
    if not plaintext:
        return plaintext
    f = _get_fernet()
    return f.encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt_str(ciphertext: str) -> str:
    """Decrypt a Fernet token back to a string.

    Graceful fallback: if decryption fails (pre-migration plaintext data),
    returns the input as-is.
    """
    if not ciphertext:
        return ciphertext
    f = _get_fernet()
    try:
        return f.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except (InvalidToken, Exception):
        # Pre-migration data stored as plaintext — return as-is
        return ciphertext


def encrypt_bytes(data: bytes) -> bytes:
    """Encrypt raw bytes, returning a Fernet token (bytes)."""
    if not data:
        return data
    f = _get_fernet()
    return f.encrypt(data)


def decrypt_bytes(data: bytes) -> bytes:
    """Decrypt raw bytes. Falls back to returning input on failure."""
    if not data:
        return data
    f = _get_fernet()
    try:
        return f.decrypt(data)
    except (InvalidToken, Exception):
        return data


def hash_token(token: str) -> str:
    """One-way SHA-256 hash for verification tokens and lookups."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Custom Django model fields
# ---------------------------------------------------------------------------


class EncryptedTextField(models.TextField):
    """TextField that encrypts data at rest using Fernet.

    Stores ciphertext in the database. Decrypts transparently on read.
    Gracefully handles pre-migration plaintext rows.
    """

    def get_prep_value(self, value):
        """Encrypt before saving to database."""
        value = super().get_prep_value(value)
        if value is None:
            return value
        return encrypt_str(str(value))

    def from_db_value(self, value, expression, connection):
        """Decrypt after reading from database."""
        if value is None:
            return value
        return decrypt_str(value)


class EncryptedCharField(models.CharField):
    """CharField that encrypts data at rest.

    Default max_length=500 to accommodate Fernet overhead (~1.4x expansion).
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("max_length", 500)
        super().__init__(*args, **kwargs)

    def get_prep_value(self, value):
        value = super().get_prep_value(value)
        if value is None or value == "":
            return value
        return encrypt_str(str(value))

    def from_db_value(self, value, expression, connection):
        if value is None or value == "":
            return value
        return decrypt_str(value)


class EncryptedJSONField(models.TextField):
    """Stores JSON data as an encrypted text blob.

    Serializes Python objects to JSON, encrypts the string, stores as TEXT.
    On read, decrypts and deserializes back to Python objects.
    """

    def get_prep_value(self, value):
        if value is None:
            return value
        json_str = json.dumps(value, default=str)
        return encrypt_str(json_str)

    def from_db_value(self, value, expression, connection):
        if value is None:
            return value
        decrypted = decrypt_str(value)
        try:
            return json.loads(decrypted)
        except (json.JSONDecodeError, TypeError):
            # If it's already a Python object (shouldn't happen, but safety)
            return decrypted

    def value_to_string(self, obj):
        """Serialize for dumpdata/loaddata."""
        value = self.value_from_object(obj)
        return json.dumps(value, default=str) if value is not None else ""
