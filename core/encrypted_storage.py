"""Encrypted file storage backend.

Wraps Django's FileSystemStorage to encrypt files at rest on disk.
Uses Fernet symmetric encryption from core.encryption.
"""

from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage

from core.encryption import decrypt_bytes, encrypt_bytes


class EncryptedFileSystemStorage(FileSystemStorage):
    """FileSystemStorage that encrypts file contents on disk.

    - _save(): reads uploaded content, encrypts, writes ciphertext to disk.
    - _open(): reads ciphertext from disk, decrypts, returns plaintext file.

    Graceful fallback: if decryption fails (pre-migration unencrypted file),
    returns the raw bytes as-is.
    """

    def _save(self, name, content):
        """Encrypt content before writing to disk."""
        raw = content.read()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        encrypted = encrypt_bytes(raw)
        return super()._save(name, ContentFile(encrypted))

    def _open(self, name, mode="rb"):
        """Decrypt content after reading from disk."""
        f = super()._open(name, mode)
        raw = f.read()
        f.close()
        decrypted = decrypt_bytes(raw)
        return ContentFile(decrypted, name=name)
